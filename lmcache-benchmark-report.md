# LMCache KV-offload on Nemotron-3.5-Lightning (DGX Spark / GB10) — Benchmark Report

> **Scope: the NVFP4 variant only.** The block size 2128, the ~326,000 KV
> tokens/GiB and the 17.85 GiB of weights in §10.3 are all
> `…-A3B-NVFP4` measurements. The BF16 variant (`nemotron/bf16/`) does not use
> LMCache and has none of these figures measured — see `nemotron/bf16/README.md`.
>
> Paths were reorganised on 2026-08-31; `nemotron-deployment.yaml` is now
> `nemotron/nvfp4/deployment.yaml` and `lmcache-deployment.yaml` is now
> `nemotron/common/lmcache.yaml`.

**Date:** 2026-08-27
**Cluster:** DGX Spark (GB10, arm64, single GPU, 128 GiB unified memory), node `spark-45f7`, namespace `vllm`
**Engine:** `vllm/vllm-openai:v0.27.1`, model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (hybrid Mamba2 + attention, NVFP4)
**Cache:** standalone LMCache 0.5.3 MP server (`nemotron/common/lmcache.yaml`), `LMCacheMPConnector`, L1 pinned DRAM + L2 `fs` adapter on NVMe (`/var/lib/lmcache-l2`)
**Bench tool:** `lmcache bench engine` (LMCache 0.5.4 client), workload `long-doc-qa`

---

## 1. Executive summary

Earlier benchmarks (`multi-round-chat`, and `long-doc-qa` with a small working set) showed **no measurable difference** with LMCache enabled. This report explains why, documents a node crash caused by an over-aggressive config, and presents a corrected run that isolates LMCache's contribution.

**Result (corrected run):** when the reused-prefix working set is **1.6× larger than the GPU KV-cache pool**, so that vLLM's own prefix cache cannot hold it:

| metric | LMCache **on** (A₂) | LMCache **off** (B) | LMCache advantage |
|---|---:|---:|---:|
| Benchmark duration | **1 286 s** | 2 974 s | **2.31× faster** |
| Input throughput | **12 183 tok/s** | 5 269 tok/s | **2.31×** |
| Mean TTFT | **782 ms** | 3 524 ms | **4.51× faster** |
| P50 TTFT | **829 ms** | 2 861 ms | 3.45× |
| P99 TTFT | **1 111 ms** | 4 969 ms | 4.47× |
| Mean end-to-end latency | **2.63 s** | 6.08 s | 2.31× |
| Mean decode speed | 70.8 tok/s | 58.4 tok/s | 1.21× |

**Conclusion:** LMCache offload is highly effective on this deployment **when — and only when — reused prefixes are evicted from vLLM's GPU KV cache before they are reused.** In the regime where the working set fits the GPU pool, vLLM's built-in prefix cache already serves all reuse for free and LMCache adds nothing.

---

## 2. Why the earlier runs showed no difference

### 2.1 `multi-round-chat` (166 vs 164 requests, ~identical)

| | without LMCache | with LMCache |
|---|---:|---:|
| Mean TTFT | 17 806 ms | 18 090 ms |
| Mean decode | 13.08 tok/s | 13.02 tok/s |
| Input throughput | 20 890 tok/s | 20 863 tok/s |

Causes:
1. **QPS-saturated.** `mrc_qps 5.0` for 60 s → ~300 requests, only ~165 completed in ~97 s. TTFT was dominated by ~20 s of queue wait, not prefill — so shaving prefill time was invisible.
2. **Shared prefix (2 000 tokens) < one LMCache chunk (2 128).** Nothing cross-session to cache.
3. Both runs hit the identical engine throughput ceiling — the benchmark measured the engine's max sustained rate, which is set by decode, not prefill.

### 2.2 `long-doc-qa`, small working set (3 runs, all ~identical: 1 015–1 035 ms TTFT, ~52 tok/s decode)

Config: `kv_cache_volume 10`, `document_length 10000`, defaults → 91 docs, 182 requests.

Causes:
1. **The bench tool runs a `warmup()` phase** that sends every document once *before timing starts*. This repopulates vLLM's own prefix cache at the start of **every** run — so "clear the GPU prefix cache" was undone before the first measured request. All 182 timed requests were vLLM-local prefix hits in every scenario.
2. **`GPU KV cache usage: 0.4 %`** — the ~10 GiB working set was tiny next to the ~37.7 GiB GPU KV pool. Nothing was ever evicted, so vLLM's local cache held every document for the whole run and LMCache had no eviction gap to fill during the timed phase.
3. LMCache *was* working — `External prefix cache hit rate: 55–68 %` in the vLLM logs — but its contribution was confined to the **unmeasured warmup phase** (after a vLLM restart, warmup pulled document KV from L2 instead of recomputing).

**The determining factor is eviction.** LMCache only helps a request whose prefix (a) is reused and (b) is no longer resident in vLLM's GPU KV cache.

---

## 3. The node crash (first corrected attempt)

The first attempt to build an eviction-forcing config **hard-crashed the node** (down 20:51:22 → 20:54:21, 3-minute reboot; k8s control plane went down with it).

### 3.1 Config that crashed

```
vLLM:   --gpu-memory-utilization 0.50        (unchanged from production)
LMCache: --l1-size-gb 8
bench:  kv_cache_volume 50, document_length 24000, query_per_document 3,
        shuffle_policy tile, num_inflight_requests 3, tokens_per_gb_kvcache 91000
```

### 3.2 Root cause: unified-memory exhaustion → NVIDIA driver allocation failure → hard lockup

Memory budget on the 128 GiB unified pool at crash time:

| consumer | GiB |
|---|---:|
| vLLM (`--gpu-memory-utilization 0.50`): 17.85 weights + 37.69 KV pool + ~8 overhead | ~64 |
| LMCache L1 pinned DRAM (`--l1-size-gb 8`) | 8 |
| LMCache transfer buffers (4 workers, CUDA-IPC) + NVMe write-back dirty pages (24+ GiB of KV chunks flushing to `/kv-l2` during warmup) | ~15–25 |
| OS + k8s + containerd + bench client | ~20 |
| **total** | **> 121 / 128** |

- Even with vLLM **idle**, `free` showed 84 GiB used / 36 GiB available under the production config — almost no headroom.
- The kernel on this node cannot recover from a hard lockup: `watchdog: NMI not fully supported`, `watchdog: Hard watchdog permanently disabled`. A GPU driver `NV_ERR_NO_MEMORY` in this state panics the box rather than OOM-killing a process.
- The crash hit ~5.5 min into the `A₁` warmup; L2 had already written 24 GB / 1 065 chunk files.

### 3.3 Contributing config errors

- **`tokens_per_gb_kvcache: 91000` was wrong.** The vLLM startup log reports **3 260 912 tokens in the 10 GiB capped pool ≈ 326 000 tokens/GiB** (≈ 12.29 M tokens / 37.69 GiB under the production config). The bench sized its document set off the wrong number.
- **`shuffle_policy: tile`** with 3 in-flight → bursty back-to-back full-prefill of all documents, all flushing to L2 simultaneously.

---

## 4. Corrected ("safe") configuration

The key change: **cap the vLLM KV pool explicitly** so a modest working set overflows it *without* consuming enough memory to threaten the box.

### 4.1 vLLM (`nemotron/nvfp4/deployment.yaml`, temporary — reverted after the run)

```diff
- --gpu-memory-utilization 0.50
+ --gpu-memory-utilization 0.35
+ --kv-cache-memory-bytes 10737418240      # hard-cap KV pool at 10 GiB (was 37.7 GiB)
```

Effect: vLLM total footprint ~64 GiB → **~34 GiB**. Measured KV pool: **3 260 912 tokens (10.0 GiB, 326 091 tok/GiB)**.

> Note: the vLLM flag is `--kv-cache-memory-bytes` (arg-utils field `kv_cache_memory_bytes`), **not** `--kv-cache-memory`. It accepts a raw byte count.

### 4.2 LMCache (`nemotron/common/lmcache.yaml`, temporary — reverted)

```diff
- --l1-size-gb 8
+ --l1-size-gb 4
- --max-workers 4
+ --max-workers 2
```

### 4.3 Benchmark (`nemotron/nvfp4/bench/bench_config.json`)

```json
{
  "workload": "long-doc-qa",
  "model": "nemotron-3.5-lightning",
  "tokens_per_gb_kvcache": 326000,
  "seed": 42,
  "kv_cache_volume": 16.0,
  "ignore_eos": true,
  "ldqa_document_length": 16000,
  "ldqa_query_per_document": 3,
  "ldqa_shuffle_policy": "random",
  "ldqa_num_inflight_requests": 2,
  "ldqa_max_output_length": 128
}
```

Derived: `num_documents = int(16 × 326000 / 16000) = 326` → **978 timed requests** (326 docs × 3 queries), each **16 023 input / 128 output tokens**. Warmup sends 326 documents once (5.2 M tokens, untimed).

**Working set = 326 × 16 000 = 5 216 000 tokens ≈ 16.0 GiB = 1.60× the 10 GiB GPU KV pool.**

### 4.4 Safety guard

A background watchdog polled `MemAvailable` every 5 s and would kill the bench + scale LMCache to 0 + abort if it dropped below 15 GiB. **It never fired** — host memory stayed flat at **~57 GiB used / ~67 GiB available for the entire ~4-hour run.**

---

## 5. Methodology

Three phases, driven by `scratchpad/ab_bench2.sh`:

| phase | setup | purpose |
|---|---|---|
| **A₁** `lmcache_cold` | wipe L2, restart both pods, LMCache on | populate L2 (throwaway) |
| **A₂** `lmcache_warm` | restart **vLLM only** (GPU cache cold, L2 warm) | **the LMCache result** |
| **B** `no_lmcache` | `kubectl apply` a copy with `--kv-transfer-config` stripped, restart vLLM | **the baseline** (prefix caching only, same 10 GiB pool) |

`seed: 42` + deterministic synthetic documents → A₁, A₂, B all generate the identical 978-request sequence. `ignore_eos: true` fixes output length at 128 tokens for a reproducible decode phase.

After B, the script restored the original engine manifest, LMCache manifest, and `bench_config.json` from backups and redeployed.

---

## 6. Full results

### 6.1 A₁ — LMCache on, L2 cold

```
Successful requests:        978          Benchmark duration (s):   1290.50
Total input tokens:    15670816          Total output tokens:        125184
Input throughput (tok/s):  12143.20      Output throughput (tok/s):   97.00
Mean TTFT (ms):  780.18   P50: 827.68   P90: 1007.63   P99: 1103.86   max: 2920
Mean decode (tok/s):  69.42             P99 decode: 78.62
End-to-end latency (s): mean 2.64  p50 2.61  p99 2.92
LMCache: lookup_requested 19.42M tok, lookup_hit 14.54M tok  (74.9% — includes cold first pass)
```

### 6.2 A₂ — LMCache on, L2 warm (vLLM restarted)

```
Successful requests:        978          Benchmark duration (s):   1286.24
Total input tokens:    15670816          Total output tokens:        125184
Input throughput (tok/s):  12183.43      Output throughput (tok/s):   97.33
Mean TTFT (ms):  781.72   P50: 828.86   P90: 1016.51   P99: 1110.66   max: 2850
Mean decode (tok/s):  70.81             P99 decode: 79.10
End-to-end latency (s): mean 2.63  p50 2.60  p99 2.89
LMCache (this run): lookup_requested 19.42M tok, lookup_hit 19.42M tok  (100%)
vLLM logs during run: Prefix cache hit rate ~22%  /  External prefix cache hit rate ~91%
```

### 6.3 B — LMCache off (prefix caching only, same 10 GiB pool)

```
Successful requests:        978          Benchmark duration (s):   2974.25
Total input tokens:    15670816          Total output tokens:        125184
Input throughput (tok/s):   5268.83      Output throughput (tok/s):   42.09
Mean TTFT (ms): 3524.37   P50: 2860.83  P90: 4808.40  P99: 4968.71   max: 7840
Mean decode (tok/s):  58.40             P99 decode: 86.39
End-to-end latency (s): mean 6.08  p50 6.46  p99 6.67
vLLM logs during run: Prefix cache hit rate → 0% at start, climbing; no External line (LMCache disabled)
```

### 6.4 A₂ vs B

| metric | A₂ (on) | B (off) | ratio |
|---|---:|---:|---:|
| Duration | 1 286 s | 2 974 s | **2.31×** |
| Input throughput | 12 183 tok/s | 5 269 tok/s | **2.31×** |
| Output throughput | 97.3 tok/s | 42.1 tok/s | 2.31× |
| Mean TTFT | 782 ms | 3 524 ms | **4.51×** |
| P50 TTFT | 829 ms | 2 861 ms | 3.45× |
| P99 TTFT | 1 111 ms | 4 969 ms | 4.47× |
| Max TTFT | 2 850 ms | 7 840 ms | 2.75× |
| Mean e2e latency | 2.63 s | 6.08 s | 2.31× |
| Mean decode | 70.8 tok/s | 58.4 tok/s | 1.21× |

### 6.5 A₁ vs A₂ — cold vs warm L2 made no difference

`1290 s / 780 ms` vs `1286 s / 782 ms`. In A₁ the warmup phase and the first query-round populate L2, so by the time the bulk of the timed requests run, L2 is already warm. **The throwaway A₁ run is not necessary** — future A/B runs can skip straight to "restart vLLM, run with LMCache, then run without."

---

## 7. Analysis

### 7.1 What LMCache is doing in A₂

- **vLLM local prefix cache hit rate ~22 %.** The 16 GiB reused-prefix working set does not fit the 10 GiB pool, so at any moment ~40 % of document KV has been evicted. With `random` ordering, most of a document's 2nd and 3rd queries arrive after it has been pushed out.
- **LMCache external hit rate ~91 %, token hit rate 100 %.** Every evicted prefix that gets reused is served from L2 (NVMe read → GPU) instead of being recomputed.
- A 16 000-token prefix reload from the `fs` L2 adapter costs a few hundred ms; a 16 000-token **recompute** on this model costs ~2–3 s of prefill and contends with decode for compute. That difference, multiplied across ~600 evicted-and-reused requests, is the 1 688 s gap in benchmark duration.

### 7.2 Why B (no LMCache) is ~2.3× slower everywhere

Without offload, every one of those ~600 evicted prefixes is a full 16 k-token prefill. Prefill competes with decode on the single GB10 GPU, so:
- input throughput halves (5.3 k vs 12.2 k tok/s),
- decode slows (58 vs 71 tok/s) because the GPU is busy re-prefilling,
- TTFT balloons 4.5× (mean 3.5 s) — each request waits behind other requests' recompute work.

### 7.3 Decode speed

LMCache does not touch decode, but A₂ still shows 71 vs 58 tok/s. This is a **second-order** effect: in B the GPU spends far more time on prefill, starving in-flight decode. It is not a direct LMCache benefit.

### 7.4 Why the memory cap did not hurt engine quality

The 10 GiB pool = 3.26 M tokens = ~12× concurrency for full 262 k-token requests, or ~200 concurrent 16 k-token contexts. Far more than `--max-num-seqs 32` needs. The cap only removed prefix-cache *headroom*, which is exactly what we wanted to test.

---

## 8. When does LMCache help on this deployment?

| scenario | LMCache benefit | why |
|---|---|---|
| Reused prefixes larger than the GPU KV pool (long shared docs/system-prompts, many distinct users, RAG over a large corpus) | **Large** (this report: 2–4×) | vLLM evicts prefixes before reuse; LMCache serves them from L2 instead of recompute |
| Reused prefixes fit the GPU pool, single engine, steady traffic | **~None** | vLLM's own prefix cache serves all reuse for free; LMCache lookups are redundant |
| Cross-restart / rolling-deploy KV reuse (warm a new engine from L2) | **Moderate** | new engine's local cache is empty; L2 survives the restart (see A₁≈A₂, and §2.2) |
| Multiple engine replicas sharing one LMCache | **Moderate–large** | KV computed by one replica is reused by another |
| QPS-saturated engine (queue wait ≫ prefill time) | **Marginal** | TTFT is dominated by scheduling delay, not prefill |
| Short prompts / short shared prefixes (< chunk size 2128) | **None** | nothing to cache at chunk granularity |

The production `--gpu-memory-utilization 0.50` gives a **37.7 GiB / 12.29 M-token** pool. LMCache earns its keep for this model when the *set of concurrently-reused prefixes* exceeds roughly that size.

---

## 9. Recommendations

1. **Keep LMCache enabled** for workloads with large shared context (RAG, long-document QA, multi-tenant with big system prompts). It is a clear win there and harmless (a redundant lookup) when the pool already covers reuse.
2. **Keep `--enable-prefix-caching` on** — it is the fast path (zero-copy local hits); LMCache only fills its misses. The two stack.
3. **Do not benchmark LMCache with a working set that fits the GPU pool** — the result is always "no difference," and it is not informative.
4. **Memory safety for future large-scale tests on this box:** never run an eviction-forcing benchmark without capping the vLLM KV pool (`--kv-cache-memory-bytes`) or lowering `--gpu-memory-utilization`. Idle headroom under the production config is only ~36 GiB, and a GPU driver OOM crashes the whole node.
5. ~~**Fix `tokens_per_gb_kvcache` in `nemotron/nvfp4/bench/bench_config.json`**~~ — DONE 2026-08-30: it is now 326 000. The real value for this model is **~326 000** (was 91 000). Or drop the key and let the bench auto-resolve from `https://lmcache.krishb.in/status`.
6. **A₁ (cold-L2 warm-up run) can be skipped** in future A/B runs; A₁ ≈ A₂.
7. If OpenCode / interactive sessions crash the box again, the mitigation is the same class of change: cap the KV pool and/or lower `--l1-size-gb`, not `--block-size` (which is pinned to 2128 for this hybrid model).

---

## 10. Appendix

### 10.1 Files

| path | role |
|---|---|
| `bench_ab_results.txt` | raw driver log + all three result blocks + LMCache metric snapshots |
| `nemotron/nvfp4/deployment.yaml` | vLLM engine (restored to production config: `--gpu-memory-utilization 0.50`, `--kv-transfer-config` present, no `--kv-cache-memory-bytes`) |
| `nemotron/common/lmcache.yaml` | LMCache MP server (restored: `--l1-size-gb 8`, `--max-workers 4`) |
| `nemotron/nvfp4/bench/bench_config.json` | bench config (moved from the repo root 2026-08-30, then under `nemotron/` 2026-08-31) |
| `scratchpad/ab_bench2.sh` | the A/B driver (capped configs, memory guard, auto-restore) |
| `scratchpad/raw2_*.txt`, `scratchpad/csv2_*.csv` | per-phase raw bench output + per-request CSVs |
| `scratchpad/*.orig` | pre-run backups of the three config files |

### 10.2 Post-run cluster state

- Node `spark-45f7`: `Ready`.
- `vllm-nemotron` (now `vllm-nemotron-nvfp4`) + `lmcache` pods: `Running`, production config, LMCache connected.
- Host memory: ~45 GiB used / ~79 GiB available (settling as the engine finishes warm-up).
- L2 cache dir `/var/lib/lmcache-l2`: contains chunks from the benchmark documents (synthetic "hi"-token docs; harmless, LRU-evicted over time, or `rm -rf /var/lib/lmcache-l2/*` on the node to clear).

### 10.3 Key measured constants for this model

| quantity | value |
|---|---|
| KV cache tokens per GiB | ~326 000 |
| KV pool @ `--gpu-memory-utilization 0.50` | 37.69 GiB ≈ 12.29 M tokens |
| Model weights | 17.85 GiB |
| Unified attention/mamba block size (no DSpark) | 2 128 tokens |
| LMCache server `--chunk-size` (must equal block size) | 2 128 |
| vLLM `--max-num-batched-tokens` (required range for hybrid + LMCache) | [2 128, 4 256) → set to 2 192 |
| 16 000-token prefix: L2 reload | few hundred ms |
| 16 000-token prefix: recompute (prefill) | ~2–3 s, contends with decode |
