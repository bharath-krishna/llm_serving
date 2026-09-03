# nemotron NVFP4 variant

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, served at
`https://nemo35-lightning.krishb.in`. This is the default variant and the one
every measured constant in this repo was taken against — `BENCHMARK-HANDOFF.md`,
`lmcache-benchmark-report.md` and `kv-cache-sizing-calculator.xlsx` are all
NVFP4.

Committed at `replicas: 1`. `../switch.sh nvfp4` to make it live.

## Why the numbers are what they are

The checkpoint ships KV scales, so vLLM selects `kv_cache_dtype=fp8_e4m3` —
1 byte/elem, 3,497 B/token, and an attention page holding twice as many tokens as
BF16's. That single fact cascades into `--max-model-len 131072`, the 2128 block
size, and `--max-num-batched-tokens 2192`.

The GPU KV pool is capped **deliberately small** (512 MiB) relative to LMCache's
4 GiB L1. This is the tiering inversion: at a 10 GiB pool the GPU tier held
3,260,912 tokens while L1 held 364,011 — the backing tier was 9× *smaller* than
what it backed, so nothing ever evicted (GPU KV usage 2.4%) and the external hit
rate sat at 6.6%. LMCache is only queried on a local miss, and with no eviction a
local miss is content nobody has ever seen. `deployment.yaml` carries the full
note and the revert values.

## LMCache

This variant carries `--kv-transfer-config`, so it registers an instance ID with
its own MP server (`./lmcache.yaml`, Deployment `lmcache-nvfp4`) at startup and
offloads KV to it. `../switch.sh` scales that server together with this engine —
server up first, or `REGISTER_KV_CACHE` fails.

`--chunk-size 2128` must equal this engine's unified block size, and
`--max-num-batched-tokens 2192` must sit in `[2128, 4256)`.
`validate_mamba_step_alignment` checks this on connect and fails the engine
outright on a mismatch.

2128 is not arbitrary — vLLM sizes the attention page to cover one mamba page
per layer, so with `fp8_e4m3` KV (1 byte/elem) and a float16 SSM cache it lands
on `ceil16(542720×2 / (2×2×128×1))` = `ceil16(2120)` = 2128. The BF16 variant's
KV is 2 bytes/elem, so its block is 1072 and it needs its own server; the full
derivation is in `../bf16/README.md`.

Note that **2192 is not a block size** — it is this engine's
`--max-num-batched-tokens`. `AGENTS.md` conflated the two for a while, which is
where the repo's spurious "block size 2192" came from.

Confirm the engine actually connected, rather than assuming:

```bash
curl -s https://lmcache.krishb.in/status | jq .registered_gpu_ids
```

Dead registrations are never reaped (`--worker-reap-timeout-seconds 0`, because
the vLLM-side heartbeat never runs in lmcache 0.5.3), so that list accumulates
across restarts. It is not a liveness signal on its own.

## Benchmarks

```bash
bench/run.sh                              # smoke, 15 requests, nothing evicts
bench/run.sh bench_config_sweep.json      # 27 requests, one sweep level
bench/run.sh bench_config_mrc.json        # multi-round chat, 8 sessions
bench/run.sh bench_config.json            # HEAVY, 978 requests -- prompts first
bench/sweep.sh "1 2 4 8 12"               # concurrency sweep, one pod, all levels
```

Results land in `/var/lib/vllm-bench-results/nvfp4/` on the node — real disk, not
a tmpfs, because session 1 of the sweep lost everything to a tmpfs when the node
hard-reset.

`bench/run.sh` preflights `long-doc-permutator` configs against an offload budget
derived from L1 (4 GiB → 3.0 GB). `N=5 C=20000 P=20` is 23.83 GB and hard-reset
the node on 2026-08-30; the budget exists to make that unrepresentable by
accident. `BUDGET_GB=<n>` overrides it deliberately.

The knee is at **8 concurrent** (227.9 out tok/s); 12 collapses to 125.2 with
TTFT p99 30.1 s; **16 hard-reset the node** 102 s in. `bench/sweep.sh` defaults to
`1 2 4 8 12` and makes 16 opt-in.

## Contents

| | |
|---|---|
| `deployment.yaml` | Deployment + Service `vllm-nemotron-nvfp4` |
| `lmcache.yaml` | Deployment `lmcache-nvfp4` — LMCache MP server, `--chunk-size 2128`, L1 4 GiB |
| `bench/` | 5 live configs + 1 quarantined, 2 pod specs, 5 drivers |
| `reference/nemotron_run.sh` | the raw uncapped `vllm serve` argv — reference only, NOT the deployed caps |
| `results/` | committed results from the A/B and sweep sessions |
| `evidence/crash-evidence-2026-08-28/` | crash #4: 10 Hz GPU telemetry through the reset, fsync'd level markers, netconsole silence |

`bench/bench_config.UNSAFE-crashed-the-node.json` is quarantined on purpose: it
points at the *public* URLs, uses a wrong `tokens_per_gb_kvcache` (91000), and
caused the 2026-08-27 crash. It is excluded from the ConfigMap the drivers build.
