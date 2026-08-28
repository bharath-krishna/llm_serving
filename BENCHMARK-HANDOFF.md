# vLLM concurrent-user benchmark — handoff

**Date:** 2026-08-28
**Node:** `spark-45f7` — NVIDIA DGX Spark (GB10, arm64, single iGPU, 128 GB unified memory)
**Goal:** measure how many concurrent users the vLLM deployment of NemotronH can serve
before latency SLOs break.

**Status: BLOCKED. The benchmark host hard-crashes under GPU load — 4+ times in 24 h,
independent of memory pressure. Believed to be a GPU/SoC firmware/driver fault, not a
configuration problem. The benchmark needs to be re-run on different hardware, or this
unit needs a firmware/driver fix + NVIDIA support case first.**

---

## 1. TL;DR for the next agent

- vLLM serves `nemotron-3.5-lightning` (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4,
  hybrid Mamba2+attention) on a single GB10.
- We built a clean concurrency sweep (`bench/conc_sweep.sh`) using `vllm bench serve`,
  driven from a separate pod, with a host-side memory guard.
- **Results we DID get (before the crash):** concurrency 1, 4, 8 — see §5.
  - Single user: **2.8 s TTFT** on a 16k-token prompt (prefill-bound), 12 ms/token decode.
  - 4 users: 171 tok/s aggregate, TTFT p99 1.1 s, 22 ms/token.
  - 8 users: 228 tok/s aggregate, TTFT p99 1.3 s, 33 ms/token, e2e p50 20.7 s.
  - Throughput is already flattening by 8 users; TPOT rising linearly.
- **Concurrency 16 hard-crashed the node ~40 s in.** Memory was flat at ~67 GB
  available the entire time; the memory guard never fired. Same for the two earlier
  crashes today.
- The scratchpad holding the live run logs was on tmpfs and was wiped by the reboot.
  Everything recoverable is in this doc and in `bench/`.

**Recommended next step:** do not keep hammering this box. Either (a) move the model +
sweep to a cloud GB200/H100 or a second Spark, or (b) get this unit onto a newer
DGX Spark BSP / NVIDIA driver and open a support case with the crash signature in §3,
then re-run.

---

## 2. Environment / constraints (carry these to the new machine)

| thing | value |
|---|---|
| GPU | NVIDIA GB10 (Grace-Blackwell, sm_121), unified 128 GB LPDDR5X |
| kernel | `6.17.0-1026-nvidia` (aarch64), `watchdog: Hard watchdog permanently disabled` |
| NVIDIA driver | Open Kernel Module `580.173.02`, CUDA 13 runtime |
| container image | `vllm/vllm-openai:v0.27.1` (arm64; bundles LMCache 0.5.3) |
| model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, served as `nemotron-3.5-lightning` |
| model arch | NemotronH hybrid: Mamba2 SSM layers + periodic attention layers, MoE (A3B active) |
| quant | NVFP4 weights, run through Marlin (no native FP4 on sm_121) |
| k8s | single-node, namespace `vllm`, haproxy ingress, in-cluster registry `registry.krishb.in` |

**Hard model/serving constraints (do not "fix" these without understanding why):**

- **Unified block size = 2128 tokens** (from engine log: *"Setting attention block size to
  2128 tokens to ensure that attention page size is >= mamba page size"*). Would be 2192
  if DSpark speculative decoding were on — it is **off** because DSpark + LMCache are
  mutually exclusive on this model (DSpark's 1024-token sliding window is not a multiple
  of the block size).
- **`--max-num-batched-tokens` must be in `[2128, 4256)`** when LMCache is attached to a
  Mamba-hybrid model. It is currently **2192**. This is the single biggest throughput
  limiter (see §6).
- LMCache for this hybrid model **requires** `LMCacheMPConnector` (not `LMCacheConnectorV1`)
  and node-local CUDA-IPC / `/dev/shm` transport — no remote `lm://` server is possible.
  Both pods run `hostIPC: true` with **no** `/dev/shm` emptyDir.
- KV cache sizing for this model (measured): **~326,000 tokens per GiB** of KV pool.
  A 10 GiB pool ⇒ 3,260,912 tokens (confirmed in engine log).
- Model weights resident: ~17.85 GiB. Engine process total with a 10 GiB KV pool: ~31 GiB
  GPU / ~31 GiB RSS.

---

## 3. The crash — full evidence

### Boot history (this is `journalctl --list-boots`, PDT)

```
 -9  Thu 08-27 07:26 → 08:36   (70 min)
 -8  Thu 08-27 18:15 → 20:51
 -7  Thu 08-27 20:54 → Fri 00:08
 -6  Fri 00:15:31 → 00:22:15   (7 min)
 -5  Fri 00:23:13 → 00:32:35   (9 min)
 -4  Fri 00:33:25 → 00:34:07   (42 SECONDS)
 -3  Fri 00:35:06 → 00:49:41
 -2  Fri 06:29:18 → 06:57:25   (28 min — crash #1 today, during opencode use)
 -1  Fri 07:04:18 → 08:12:44   (68 min — crash #3, during THIS benchmark, at concurrency 16)
  0  Fri 09:02:49 → ...        (current)
```

### Signature (identical every time)

- Journal / kernel log **ends mid-line** on a routine entry (kubelet or containerd), then
  the machine is simply gone. `/var/log/kern.log` has **nothing** in the seconds before
  the reset.
- **No** `NVRM: Xid`, **no** kernel panic / oops, **no** `oom-kill`, **no** thermal event,
  **no** MCE — anywhere, in any boot.
- Reboot takes ~3 min and also bounces the k8s control plane.
- `watchdog: Hard watchdog permanently disabled` on this kernel — so the SoC cannot even
  produce a clean panic/kdump; it just resets.

### Crash #3 (this benchmark) specifics

- Started sweep 07:58 PDT. Completed concurrency **1, 4, 8** cleanly.
- Concurrency **16** began 08:12:02 PDT; node died **08:12:44 PDT** (~42 s in — right as
  16 concurrent ~16k-token prefills hit the GPU together).
- Host memory the entire run: **~67.5 GB available, dead flat** (146 guard samples, min
  67,257 MB, floor was 13,000 MB — never close). vLLM KV cap (10 GiB) held perfectly.
- `ollama` was **stopped** (`systemctl disable --now ollama`) for this run — ruled out.
- The desktop GDM/Xorg session was still on the GPU (`Xorg` 18 MiB, `gnome-shell` 6 MiB) —
  trivial, but ideally kill it too (`systemctl set-default multi-user.target`).

### Interpretation

Memory-independent, load-triggered, unlogged instant reset on a GB10, recurring, with a
42-second boot-to-crash in the history — this is a **hardware / firmware / GPU-driver
fault**, not a k8s or vLLM resource issue. The earlier hypothesis (unified-memory
exhaustion → driver OOM) was disproved by this run: caps applied, memory flat, still
crashed. Sustained/bursty GPU compute on this unit wedges the memory fabric or the GPU
MMU and the CPU can't recover.

### Contributing environment problems worth cleaning up regardless

- `ollama.service` autostarts and grabs the full 121.7 GiB as "VRAM" with
  `OLLAMA_MAX_LOADED_MODELS=0`. Keep it disabled while vLLM owns the GPU.
- Desktop session on the GPU (see above).
- ~9 pods in permanent CrashLoopBackOff — `dify-release-redis-*` at **37,000+ restarts**,
  `wandb-controller` at 37k, `banking/*` on ImagePullBackOff. Constant churn; delete them.
- **Security:** `sshd` is exposed with password auth and taking root brute-force attempts
  from the internet (`Failed password for root from 91.92.47.123` seen seconds before the
  crash — coincidental, but the exposure is real). Lock SSH down (key-only, no root,
  firewall) independent of the GPU issue.

---

## 4. Current deployment (the CAPPED config the benchmark ran against)

Full manifests are in the repo root: `nemotron-deployment.yaml`, `lmcache-deployment.yaml`.
These already contain the 2026-08-28 memory caps. Key lines:

### `nemotron-deployment.yaml` — Deployment `vllm-nemotron`, ns `vllm`

```
image: vllm/vllm-openai:v0.27.1
runtimeClassName: nvidia
hostIPC: true
strategy: { type: Recreate }

args (vllm serve):
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
  --served-model-name nemotron-3.5-lightning
  --max-num-seqs 32
  --max-model-len 262144
  --enable-prefix-caching
  --async-scheduling
  --max-num-batched-tokens 2192          # forced into [2128,4256) by LMCache+Mamba
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16
  --gpu-memory-utilization 0.35          # IGNORED for sizing once the next line is set
  --kv-cache-memory-bytes 10737418240    # 10 GiB KV pool = 3,260,912 tokens
  --mamba-cache-mode align
  --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5
  --reasoning-parser nemotron_v3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --host 0.0.0.0 --port 8000
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
     "kv_connector_extra_config":{"lmcache.mp.host":"tcp://lmcache.vllm.svc.cluster.local",
     "lmcache.mp.port":5555}}'

resources:
  requests: { nvidia.com/gpu: 1, memory: 32Gi }
  limits:   { nvidia.com/gpu: 1, memory: 48Gi }   # cgroup backstop (host-side runaway only)

volumes: hostPath /home/bharath/.cache/huggingface -> /root/.cache/huggingface
Service vllm-nemotron :8000 ; Ingress hosts nemotron.local, nemo35-lightning.krishb.in
```

### `lmcache-deployment.yaml` — Deployment `lmcache`, ns `vllm`

```
image: vllm/vllm-openai:v0.27.1 ; runtimeClassName: nvidia ; hostIPC: true
pod annotation: cdi.k8s.io/lmcache: "nvidia.com/gpu=0"   # CDI GPU inject, no alloc count

command: python3 -m lmcache.v1.multiprocess.http_server
  --http-host 0.0.0.0 --http-port 8080 --host 0.0.0.0 --port 5555
  --l1-size-gb 4                      # pinned DRAM hot tier (was 8, lowered 08-28)
  --eviction-policy LRU --max-workers 4
  --chunk-size 2128                   # MUST equal vLLM block size
  --separate-object-groups
  --supported-transfer-mode lmcache_driven
  --l2-adapter '{"type":"fs","base_path":"/kv-l2"}'   # NVMe hostPath /var/lib/lmcache-l2

resources: requests {cpu 4, memory 12Gi} ; limits {memory 16Gi}
Services: lmcache (headless, zmq 5555 + http 8080), lmcache-http (ClusterIP 80->8080)
Ingress: lmcache.krishb.in -> lmcache-http
```

**To restore the ORIGINAL uncapped production config** (do NOT, it crashes harder):
`--gpu-memory-utilization 0.50`, no `--kv-cache-memory-bytes`, lmcache `--l1-size-gb 8`,
no pod `memory` limits.

---

## 5. Results collected (concurrency 1, 4, 8)

Workload per request: prompt 8k–24k tokens (mean 16k, `--random-range-ratio 0.5`),
600 output tokens (`--ignore-eos`), plus a shared 2,128-token cached prefix.
`--request-rate inf` so exactly N requests are always in flight. 48 prompts per level.

| concurrency | dur (s) | req/s | **out tok/s** | total tok/s | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 | e2e p50 | e2e p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1**  | 500.4 | 0.10 | 59.2  | 1851 | 2796 ms | 3899 ms | 4184 ms | 12.3 ms | 12.6 ms | 37 ms  | 10.5 s | 14.2 s |
| **4**  | 173.4 | 0.28 | 170.8 | 5341 | 477 ms  | 629 ms  | 1110 ms | 22.3 ms | 23.5 ms | 66 ms  | 14.2 s | 20.3 s |
| **8**  | 130.0 | 0.37 | 227.9 | 7126 | 486 ms  | 877 ms  | 1271 ms | 33.4 ms | 34.9 ms | 133 ms | 20.7 s | 29.7 s |
| **16** | — | — | — | — | — | — | — | — | — | — | — | **NODE CRASH ~40 s in** |

Raw `vllm bench serve` blocks:

```
==== concurrency 1 ====
Successful requests: 48    Benchmark duration (s): 500.44
Total input tokens: 896633    Total generated tokens: 29620
Output token throughput (tok/s): 59.19    Total token throughput (tok/s): 1850.86
Peak concurrent requests: 2.00
Mean TTFT: 2842.06 ms   P50: 2796.45   P90: 3898.84   P99: 4183.76
Mean TPOT: 12.31 ms     P50: 12.29     P90: 12.45     P99: 12.62
Mean ITL: 13.32 ms      P50: 12.34     P90: 13.43     P99: 37.49
Mean E2EL: 10425.73 ms  P50: 10513.12  P90: 12911.11  P99: 14198.30

==== concurrency 4 ====
Successful requests: 48    Benchmark duration (s): 173.42
Output token throughput (tok/s): 170.80    Total token throughput (tok/s): 5341.15
Peak concurrent requests: 6.00
Mean TTFT: 488.74 ms    P50: 477.01    P90: 628.85    P99: 1109.91
Mean TPOT: 22.27 ms     P50: 22.34     P90: 22.98     P99: 23.52
Mean ITL: 24.48 ms      P50: 21.54     P90: 25.03     P99: 66.01
Mean E2EL: 14230.76 ms  P50: 14187.62  P90: 18440.35  P99: 20276.41

==== concurrency 8 ====
Successful requests: 48    Benchmark duration (s): 129.99
Output token throughput (tok/s): 227.87    Total token throughput (tok/s): 7125.76
Peak concurrent requests: 11.00
Mean TTFT: 560.99 ms    P50: 485.78    P90: 877.02    P99: 1271.04
Mean TPOT: 32.90 ms     P50: 33.40     P90: 34.45     P99: 34.86
Mean ITL: 35.68 ms      P50: 31.12     P90: 41.84     P99: 132.77
Mean E2EL: 20781.69 ms  P50: 20745.58  P90: 27219.96  P99: 29742.35
```

---

## 6. Preliminary analysis (incomplete — no data past N=8)

**The single-user 2.8 s TTFT is real** (mean ≈ median across all 48 requests, not a warmup
artifact). It is pure prefill of a ~16k-token prompt at `--max-num-batched-tokens 2192`
(~7–8 chunked-prefill steps at ~380 ms each). At N≥4 the scheduler interleaves new
prefills with running decodes, so *observed* per-request TTFT drops to ~0.5 s while the
prefill work is amortised — but total prefill capacity is fixed.

**Throughput is saturating early:**
- N=1 → N=4: output tok/s 59 → 171 (2.9×, near-linear)
- N=4 → N=8: 171 → 228 (only +33%)
- The knee is around **4–8 concurrent long-context users** for this config.

**TPOT climbs linearly with batch size** (12 → 22 → 33 ms) because every decode step runs
the whole batch. At N=8, e2e for a 600-token completion is already ~21 s (p50).

**Two ceilings in the current config:**
1. `--max-num-seqs 32` — max 32 requests decoding at once; extras queue.
2. `--max-num-batched-tokens 2192` — the real limiter. All concurrent prefills share a
   2,192-token/step budget. This is pinned into `[2128, 4256)` *only because LMCache is
   attached*. Without LMCache you could set it to 8k–16k and prefill several users per
   step, roughly 4–8× the prefill throughput.

**LMCache is barely helping this workload.** During the earlier opencode session the
engine logged `Prefix cache hit rate: 96%` (vLLM local) vs `External prefix cache hit
rate: 2%` (LMCache). vLLM's own GPU prefix cache serves nearly all reuse for a small
number of users; LMCache only pays off under heavy multi-user KV eviction. Given it also
forces the tiny `max-num-batched-tokens`, **strongly consider running WITHOUT LMCache**
for the multi-user test and comparing.

---

## 7. What to run on the new machine

### 7a. Bring up the model + bench pod

```bash
kubectl apply -f nemotron-deployment.yaml      # or your equivalent
kubectl apply -f lmcache-deployment.yaml       # optional — see §6, consider skipping
kubectl -n vllm rollout status deploy/vllm-nemotron --timeout=600s

kubectl apply -f bench/bench-pod.yaml
kubectl -n vllm wait --for=condition=Ready pod/vllm-bench --timeout=120s
```

### 7b. Run the sweep

```bash
# from the k8s host (or anywhere kubectl works; the memguard only matters on the host)
sudo systemctl disable --now ollama          # if present
bash bench/run_sweep.sh                       # ~30–50 min; writes bench/bench_results/
```

Or manually inside the pod:
```bash
kubectl -n vllm cp bench/conc_sweep.sh vllm-bench:/root/conc_sweep.sh
kubectl -n vllm exec -it vllm-bench -- bash -lc 'chmod +x /root/conc_sweep.sh && /root/conc_sweep.sh'
kubectl -n vllm cp vllm-bench:/results ./bench_results
```

### 7c. Recommended additional runs (once the box is stable)

1. **Same sweep, LMCache OFF, `--max-num-batched-tokens 16384`** — strip
   `--kv-transfer-config` from the vLLM args and scale `deploy/lmcache` to 0. This is the
   apples-to-apples "does LMCache help or hurt multi-user" test and should massively
   improve prefill/TTFT.
2. **Raise `--max-num-seqs` to 64–128** and re-sweep to 128 concurrency.
3. **Short-chat profile** for comparison: `--dataset-name sharegpt --sharegpt-output-len
   300`, sweep concurrency to 128.
4. Push `--kv-cache-memory-bytes` up (20–40 GiB) once you trust the hardware — with more
   KV pool, more long contexts fit and `--max-num-seqs` becomes the binding limit.

### 7d. Reading the output

`bench_results/pod-results/summary.csv`, one row per concurrency level. **Capacity = the
highest concurrency where `ttft_p99_ms` and `tpot_p50_ms` are still under your SLO** and
`out_tok_per_s` is still rising. Suggested SLOs for a coding agent: TTFT p99 < 5000 ms,
TPOT p50 < 80 ms. Above the knee, throughput plateaus and latency grows linearly.

---

## 8. Files

| path | what |
|---|---|
| `BENCHMARK-HANDOFF.md` | this doc |
| `bench/bench-pod.yaml` | load-generator pod (same image, no GPU, HF cache mounted RO) |
| `bench/conc_sweep.sh` | the sweep — runs inside the bench pod |
| `bench/run_sweep.sh` | host orchestrator: memory guard + launch + collect |
| `nemotron-deployment.yaml` | vLLM engine (repo root) — has the 08-28 caps |
| `lmcache-deployment.yaml` | LMCache MP server (repo root) — has the 08-28 caps |
| `lmcache-benchmark-report.md` | earlier A/B (LMCache on vs off) full report + the first crash write-up |
| `bench_config.json` | config for the OTHER tool (`lmcache bench engine`), unrelated to this sweep |
| `bench_config.UNSAFE-crashed-the-node.json` | quarantined config that caused crash #? on 08-27 |

Live run artifacts (guard.log, sweep.out, conc-16.log) were on tmpfs and lost in the
reboot. conc-1/4/8 raw numbers are transcribed in §5.

---

## 9. Open questions for whoever picks this up

1. Is the GB10 lockup reproducible on a second Spark, or specific to this unit? (If
   unit-specific → RMA. If model-wide → NVIDIA driver/BSP bug, needs a support case with
   the §3 signature.)
2. Does a newer DGX Spark BSP / NVIDIA driver (> 580.173.02) fix it?
3. With LMCache removed and `max-num-batched-tokens` raised, where does the real
   concurrency knee land?
4. Power/thermal: the crashes have no thermal log entry, but rule it out with
   `nvidia-smi -q -l 1` logging + a lower power cap during the next attempt.
