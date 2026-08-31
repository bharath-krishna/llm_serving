# nemotron BF16 variant

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, served at
`https://nemo35-lightning.krishb.in` — the same URL and the same `vllm-auth` API
key as the NVFP4 variant. Only one runs at a time (one GPU); `../switch.sh bf16`.

| | value | vs NVFP4 |
|---|---|---|
| weights | 58.93 GiB | 17.86 GiB |
| checkpoint load | ~334 s (measured 329.9 s) | ~103 s |
| KV dtype | bf16 (`auto`; checkpoint ships no KV scales) | `fp8_e4m3` |
| KV bytes/token (engine) | 6,144 | 3,497 |
| **attention block size** | **1072** (measured) | 2128 |
| `--max-num-batched-tokens` | 1072 | 2192 |
| `--max-model-len` | 131072 | 131072 |
| GPU KV pool | 1 GiB (174,762 tokens) | 512 MiB |
| LMCache L1 | 6 GiB (216,946 tokens) | 4 GiB |
| LMCache B/token | 29,696 (measured) | 11,799 |
| engine footprint | ~64 GiB | ~22.5 GiB |
| model names reported | `nemotron-3.5-lightning-bf16`, aliases `nemotron-3.5-lightning` + repo ID | `nemotron-3.5-lightning` + repo ID |

The alias means a client hardcoding `nemotron-3.5-lightning` keeps working across
a switch. `/v1/models` still leads with the `-bf16` name so the API stays honest
about which variant answered.

## Block size — the number everything else hangs off

Measured 2026-08-31T07:47:33Z from this engine's own log:

```
interface.py:911  Setting attention block size to 1072 tokens
                  to ensure that attention page size is >= mamba page size
```

It is not a free choice. vLLM sizes the attention page so it is at least one
mamba page, per layer:

```
block = ceil16( mamba_state_elems_per_layer × sizeof(mamba_ssm_cache_dtype)
                ─────────────────────────────────────────────────────────── )
                     2 × num_kv_heads × head_dim × sizeof(kv_cache_dtype)
```

For this model — 542,720 mamba state elems/layer (18,432 conv + 524,288 ssm),
`num_kv_heads` 2, `head_dim` 128:

| kv dtype | ssm dtype | attn B/token/layer | raw | block |
|---|---|---|---|---|
| `fp8_e4m3` | float16 | 512 | 2120.0 | 2128 ← NVFP4 |
| `bf16` | float16 | 1024 | 1060.0 | **1072** ← this variant |
| `bf16` | float32 | 1024 | 2120.0 | 2128 |

Two consequences worth internalising:

* **`--mamba-ssm-cache-dtype float16` is as load-bearing as the KV dtype.** Drop
  it and this variant's block becomes 2128 — silently changing what
  `--chunk-size` and `--max-num-batched-tokens` must be.
* **The LMCache server's `--chunk-size` must equal it**, which is why there is a
  server per variant (`./lmcache.yaml` at 1072, `../nvfp4/lmcache.yaml` at 2128)
  rather than one shared one. `validate_mamba_step_alignment`
  (`lmcache_mp_connector.py:145`) checks it on connect and fails the engine at
  startup on a mismatch — loudly, not silently.

This also settles a contradiction that stood in this repo: 2128 is NVFP4's block
size, 2176 was simply wrong, and **2192 was never a block size** — it is NVFP4's
`--max-num-batched-tokens`, which must lie in `[2128, 4256)`.

## Context length and how to change it

`--max-model-len 131072`, matching NVFP4. Raised from 65536 on 2026-08-31 after
opencode hit `maximum context length is 65536 ... 33,537 input + 32,000 output`.

The binding constraint is that the GPU KV pool must hold **one full-length
request** — vLLM refuses to start otherwise, and its error names the exact
ceiling (`estimated maximum model length is N`). Attention KV costs:

```
2 (K+V) × num_kv_heads(2) × head_dim(128) × sizeof(bf16) × attn_layers(6)
  = 6,144 B/token
```

Verified against the running engine at the old setting: 81 blocks × 1072 tokens
× 6,144 = 508.8 MiB, matching the 512 MiB pool then configured. NVFP4 pays half
(fp8_e4m3 KV, 1 byte/elem) — which is the entire reason this variant needed a
bigger pool to reach the same context length.

**Mamba state does not grow with context.** It is sized by `--max-num-seqs` (32),
not by `max-model-len`, so raising context is purely an attention-KV cost. That
is what makes 131072 affordable here at all.

| max-model-len | min pool | suggested pool | L1 needed¹ | extra memory |
|---|---|---|---|---|
| 65,536 | 389 MiB | 512 MiB | 2.4 GiB | — |
| 98,304 | 578 MiB | 768 MiB | 3.6 GiB | +0.25 GiB |
| **131,072** ← current | **773 MiB** | **1 GiB** | **4.8 GiB → use 6** | **+2.5 GiB** |
| 262,144 | 1.5 GiB | 2 GiB | 9.7 GiB → use 10 | +7.5 GiB |

¹ min pool = `ceil(len / 1072) blocks × 1072 × 6,144`. **L1 must hold more
*tokens* than the pool it backs**, and at LMCache's 29,696 B/token that is what
actually costs memory — not the pool. The two always move together: raising
`--kv-cache-memory-bytes` without raising `--l1-size-gb` silently inverts the
tiering the wrong way and the external hit rate collapses. Both live in the same
pair of files (`deployment.yaml`, `lmcache.yaml`).

`--kv-cache-dtype fp8` would halve the per-token cost and buy all of this for
free, but this checkpoint ships no KV scales, so that is an accuracy trade rather
than a win.

Changing either value restarts both the LMCache server and the engine (~6 min) —
use `../switch.sh apply`, never a bare `kubectl apply`.

## LMCache

**On.** `deployment.yaml` carries `--kv-transfer-config`, so this engine
registers an instance ID with `lmcache-bf16` (`./lmcache.yaml`) at startup and
offloads KV to it. `../switch.sh` scales that server together with this engine —
server up first, or `REGISTER_KV_CACHE` fails.

The GPU KV pool is 1 GiB, deliberately **smaller in tokens** than the 6 GiB L1. That
inversion is the whole point: LMCache is queried only on a *local* prefix-cache
miss, so a pool large enough that nothing ever evicts means a local miss is
always genuinely-new content LMCache cannot have either. Measured on NVFP4 with
a 10 GiB pool against a 4 GiB L1: GPU KV usage 2.4%, external hit rate 6.6%.

L1 is 6 GiB, and that is a **derived floor** rather than a comfort setting: it
must hold more tokens than the pool it backs. Against the 1 GiB pool (174,762
tokens), 4 GiB would give only 144,631 — 0.83×, the wrong way round. 6 GiB gives
216,946 = 1.24×. The same mistake was made once already at the old 512 MiB pool,
where a 2 GiB L1 measured 72,315 tokens against 86,832 and had to be raised.
See the sizing table above: pool and L1 always move together.

L2 lives at `/kv-l2/bf16`, separate from NVFP4's `/kv-l2/nvfp4`, so neither has
to be cleared on a switch and a warm tier survives being switched away from and
back. (When the servers were split on 2026-08-31, the old shared server's 2,415
NVFP4 chunks — 56 GB, written at the matching `--chunk-size 2128` — were moved
from the `/kv-l2` root into `/kv-l2/nvfp4` rather than discarded, so NVFP4's cold
tier is still warm.)

Confirm the engine actually registered rather than assuming:

```bash
curl -s https://lmcache.krishb.in/status | jq .registered_gpu_ids
```

Dead registrations are never reaped (`--worker-reap-timeout-seconds 0`, because
the vLLM-side heartbeat never runs in lmcache 0.5.3), so that list accumulates
across restarts of the *same* variant. Scaling the server down — which
`../switch.sh` does — is what clears it.

## Benchmarking

`bench/` mirrors `../nvfp4/bench/` with BF16 endpoints, pod names and ConfigMap.
Results land under `/var/lib/vllm-bench-results/bf16/` so they can never be
confused with NVFP4 runs.

```bash
bench/run.sh                            # smoke, 15 requests
bench/run.sh bench_config_sweep.json    # 27 requests, one sweep level
bench/sweep.sh "1 2 4 8"                # concurrency sweep
```

### Measured constants

All read from the registered engine on 2026-08-31 (`lmcache /status` →
`cache_context_meta`), not derived:

| constant | value | note |
|---|---|---|
| GPU KV pool | 163 blocks × 1072 = **174,736 tokens** | 1 GiB → 6,144 B/token engine-side |
| `tokens_per_gb_kvcache` | **174,736** | pool capacity, used to size working sets |
| LMCache `cache_size_per_token` | **29,696 B** | offload volume — **2.52×** NVFP4's 11,799 |
| L1 (6 GiB) | **216,946 tokens** | 1.24× the pool ✓ |

Before the 2026-08-31 context raise these were 86,832 / 173,664 / 144,631 (512 MiB
pool, 4 GiB L1) — the smoke result below was taken at those settings.

The 2.52× is the one that catches people out. It is *not* the 1.96× the KV-dtype
ratio predicts, because LMCache stores the mamba recurrent state as well as
attention KV — 29 layers in the reported layout (23 mamba + 6 attention). An
earlier derivation guessed 23,126 and was 22% low. **Use 173,664 for pool
capacity and 29,696 for offload volume; mixing them up sizes a run ~5× wrong.**

### Smoke results (2026-08-31)

`bench/run.sh bench_config_smoke.json`, both runs 0 failures:

| | @ 65536 (512 MiB pool, 4 GiB L1) | @ 131072 (1 GiB pool, 6 GiB L1) |
|---|---|---|
| requests | 21 | 24 |
| duration | 62.3 s | 72.8 s |
| input tokens | 336,455 | 384,520 |
| mean TTFT | 785.7 ms | 1,123 ms (P99 7,539) |
| mean decode | 26.8 tok/s | 27.3 tok/s |
| L1 after | 98 obj / 2.905 GiB of 4 (72.6%) | 37 obj / 1.097 GiB of 6 (18.3%) |
| L2 after | 119 files / 3.6 GB | 299 files / 8.9 GB |

Decode is unchanged; TTFT rose with the larger working set. L2 grows even at 18%
L1 because LMCache writes *through* to L2, not only on eviction.

**Two cautions from these runs:**

- The same `kv_cache_volume` (0.553) produced 7 documents at
  `tokens_per_gb_kvcache` 173,664 and 8 at 174,736 — neither `int()` nor `ceil()`
  of the quoted formula. `long_doc_qa.py` resolves differently than the comments
  claim. **Read request counts off the run, not off the arithmetic.**
- At 65536 the working set was 1.11× the pool and evicted mildly, so the run
  exercised LMCache *loads*. At 131072 it is 0.73× the pool and nothing evicts —
  the run still proves stores, spills and the full path, but loads now only
  happen across runs.

### Still guarded

`bench_config.json` and `bench_config_mrc.json` — now for engineering reasons
rather than missing measurements. Each config's `_comment` carries the arithmetic:

| config | offload | vs 6 GiB L1 (budget 4.83 GB) | why |
|---|---|---|---|
| `bench_config.json` | ≥174,736 tok × 29,696 = ≥5.19 GB | 1.07× budget | see below — now structurally unreachable |
| `bench_config_mrc.json` | 312,080 tok × 29,696 = 9.27 GB | 1.44× L1 | needs users + duration re-derived first |

**Raising max-model-len made the eviction benchmark harder, not easier.** The
pool doubled (86,832 → 174,736 tokens) but L1 rose only 1.5× in tokens
(144,631 → 216,946), because the pool is charged 6,144 B/token and L1 is charged
29,696 — LMCache stores the mamba state too. The tiers do not scale together. To
force eviction the working set must exceed 174,736 tokens, which is 5.19 GB of
offload before *any* oversubscription — already past the 4.83 GB budget.

Options, if the eviction regime is wanted: shrink the pool back toward 512 MiB
(accepting lower context), raise `--l1-size-gb` to ~12 (another 6 GiB of shared
memory), or characterise this variant's eviction behaviour at the short-context
setting — which is the honest default, since the NVFP4 numbers it would be
compared against were themselves taken at a 512 MiB pool.

The MRC config on NVFP4 is 3.68 GB against a 4 GiB L1 — safe there, not here,
entirely because of the 2.52× bytes/token. And NVFP4's 978-request heavy run
translates to 40.9 GB here: **there is no configuration in which this variant
runs that benchmark on this box.**

Also re-check `PREFIX_LEN=1072` in `bench/conc_sweep*.sh` — it models one shared
system prompt as exactly one cacheable page. A wrong value there does not fail,
it just quietly measures less prefix reuse than intended.

> The concurrency thresholds in `bench/sweep-pod.yaml` (knee at 8, collapse at
> 12, node crash at 16) were measured against NVFP4 and are very likely
> optimistic here. Run `../tools/start-telemetry.sh` on the node before anything
> past a smoke config.
