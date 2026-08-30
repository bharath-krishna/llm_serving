# vLLM concurrent-user benchmark — handoff

**Date:** 2026-08-28 (updated after instrumented run #2, 09:27–09:45 PDT)
**Node:** `spark-45f7` — NVIDIA DGX Spark (GB10, arm64, single iGPU, 128 GB unified memory)
**Goal:** measure how many concurrent users the vLLM deployment of NemotronH can serve
before latency SLOs break.

**Status: capacity question ANSWERED (knee = 8 users). Crash still not root-caused, but
now properly characterised: it is an instant hardware-level reset with verified-zero
kernel output. Session 2 added real instrumentation and one new data point (N=12).
Paused by user; next step is the clock-lock experiment in §10.**

---

## 1. TL;DR for the next agent

- vLLM serves `nemotron-3.5-lightning` (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4,
  hybrid Mamba2+attention) on a single GB10.
- Concurrency sweep via `vllm bench serve` from a separate pod (`bench-nemotron/conc_sweep.sh`,
  resume variant `bench-nemotron/conc_sweep_resume.sh`).
- **Results: concurrency 1, 4, 8, 12 — see §5.**
  - Throughput peaks at **N=8 (228 out tok/s)** and then **COLLAPSES to 125 tok/s at N=12**
    — it does not merely flatten.
  - At N=12 TTFT p99 is **30.1 s** and e2e p50 is **58.4 s**: unusable for a coding agent.
  - **Capacity against a TTFT-p99 < 5 s SLO is 8 concurrent long-context users.**
- **Concurrency 16 crashes the node, reproducibly** — twice now (~42 s in on 08-28 07:58
  run, **102 s in** on the instrumented 09:39 run).
- Levels 16/24/32/48/64 were never collected and are **past the usable knee anyway** —
  there is little benchmark value in chasing them on this config (see §6).

**Recommended next step:** run the clock-lock experiment (§10). It is ~5 minutes and is the
single test that discriminates the leading hypothesis. Then, separately, do the
LMCache-off re-sweep (§7c #1) — that is the change likely to fix the N=12 collapse.

---

## 2. Environment / constraints (carry these to a new machine)

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
| host net | **WiFi only** (`wlP9s9`, mt7925e). `enP7s7` (Realtek 5GbE) is DOWN / NO-CARRIER |

**Hard model/serving constraints (do not "fix" these without understanding why):**

- **Unified block size = 2128 tokens** (from engine log: *"Setting attention block size to
  2128 tokens to ensure that attention page size is >= mamba page size"*). Would be 2192
  if DSpark speculative decoding were on — it is **off** because DSpark + LMCache are
  mutually exclusive on this model (DSpark's 1024-token sliding window is not a multiple
  of the block size).
- **`--max-num-batched-tokens` must be in `[2128, 4256)`** when LMCache is attached to a
  Mamba-hybrid model. It is currently **2192**. This is the single biggest throughput
  limiter and the prime suspect for the N=12 collapse (see §6).
- LMCache for this hybrid model **requires** `LMCacheMPConnector` (not `LMCacheConnectorV1`)
  and node-local CUDA-IPC / `/dev/shm` transport — no remote `lm://` server is possible.
  Both pods run `hostIPC: true` with **no** `/dev/shm` emptyDir.
- KV cache sizing for this model (measured): **~326,000 tokens per GiB** of KV pool.
  A 10 GiB pool ⇒ 3,260,912 tokens (confirmed in engine log).
- Model weights resident: ~17.85 GiB. Engine process total with a 10 GiB KV pool: ~31 GiB
  GPU / ~31 GiB RSS.

---

## 3. The crash — evidence from sessions 1 and 2

### Boot history (`journalctl --list-boots`, PDT)

```
 -7  Fri 00:15:31 → 00:22:15   (7 min)
 -6  Fri 00:23:13 → 00:32:35   (9 min)
 -5  Fri 00:33:25 → 00:34:07   (42 SECONDS — cold, idle. See "why this matters" below)
 -4  Fri 00:35:06 → 00:49:41
 -3  Fri 00:52:36 → 01:29:04
 -2  Fri 06:29:18 → 06:57:25   (28 min — crash during opencode use)
 -1  Fri 07:04:18 → 08:12:44   (68 min — crash #3, sweep session 1, at concurrency 16)
  0  Fri 09:02:49 → 09:41:24   (39 min — crash #4, sweep session 2, at concurrency 16)
```

### Signature (identical every time)

- Journal / kernel log **ends mid-line** on a routine entry, then the machine is gone.
- **No** `NVRM: Xid`, **no** kernel panic / oops, **no** `oom-kill`, **no** thermal event,
  **no** MCE — anywhere, in any boot.
- Reboot takes ~3 min and also bounces the k8s control plane.

### Why there was never any evidence: the box is instrumentation-blind

This was the key discovery of session 2. The absence of error records is **not** evidence
of no hardware fault — **there is no mechanism on this unit capable of recording one:**

| channel | state | consequence |
|---|---|---|
| ACPI `HEST` / `BERT` | **absent** from `/sys/firmware/acpi/tables/` | no APEI/GHES; firmware cannot report hardware errors to the kernel, and no boot error record survives a reset |
| EDAC | no `mc0` instance, no edac modules loaded | no DRAM ECC reporting at all |
| GPU ECC / retired pages / power limit | all `N/A` in `nvidia-smi -q` | GB10 exposes no ECC or power-cap telemetry |
| Hard watchdog | `permanently disabled` on this kernel | SoC cannot produce a clean panic |
| kdump | `crashkernel=1G-:0M` (from `/etc/default/grub.d/kdump-tools.cfg`) | 0 MB reserved — kdump disabled |
| Tegra CBB (SoC fabric errors) | `initcall_blacklist=tegra234_cbb_init` (from **NVIDIA's own** `/etc/default/grub.d/nvidia-spark-initcall-bl.cfg`), **and** the module is absent from the modules tree despite `CONFIG_SOC_TEGRA_CBB=m` | interconnect / bus-timeout / illegal-access faults are silently unreportable. Cannot be re-enabled without a custom kernel. |

The CBB blacklist is shipped by NVIDIA's DGX Spark BSP, so this blindness is
vendor-default, not a local misconfiguration.

### Ruled OUT by direct measurement (session 2)

- **Memory pressure.** `MemAvailable` flat at **67.1 GB** for the entire run, right up to
  the last synced sample. PSI memory `avg10 = 0.00` throughout. The 10 GiB KV cap held.
- **DRAM / MCE / extlog errors.** `ras-mc-ctl --summary`: *"No Memory errors. No Extlog
  errors. No MCE errors."*
- **PCIe faults.** rasdaemon shows 329 corrected AER events, but every one is `RxErr` on
  `0000:00:00.0` and `0002:00:00.0` — **empty NVIDIA root ports** with nothing enumerated
  behind them (benign link-training noise). The **GPU (`000f:01:00.0`) and NVMe have zero
  AER errors**, and there are **no uncorrectable or fatal AER errors on any device**.
- **Kernel-side fault paths.** `/sys/fs/pstore` is empty — no panic was ever recorded.
- **ollama.** Disabled (`systemctl disable --now ollama`) for both sweep runs.

### The decisive new evidence: netconsole silence

Session 2 armed netconsole (kernel printk shipped over UDP to a laptop, so it is
*transmitted* rather than written — it survives a reset that discards the page cache).

- netconsole was **verified working**: manual `echo ... > /dev/kmsg` test messages arrived
  at the collector (`bench-nemotron/crash-evidence-2026-08-28/netconsole.log`).
- At the crash it delivered **zero bytes**.

The kernel printed **nothing at all** before the reset. That positively excludes kernel
panic, OOM kill, GPU driver Xid, and soft/hard lockup — all of which print first. The CPU
never got the opportunity to emit a single character. This is an **instant hardware-level
reset** (power delivery, PMIC, or SoC-level), not a software fault path.

### Crash #4 telemetry (the instrumented one)

Sampled at 1 Hz (system) and 10 Hz (GPU), every sample written `O_SYNC` so the page cache
could not swallow it. Level 16 began 09:39:42.175; **last telemetry sample 09:41:24.549
PDT ⇒ 102.4 s in**, with 8 of 96 requests complete.

Final seconds (10 Hz):

```
09:41:13  gpu=77C  p=50.9W  sm=2463 MHz  throttle=0x0
09:41:16  gpu=82C  p=79.5W  sm=2385 MHz  throttle=0x0
09:41:18  gpu=78C  p=50.8W  sm=2463 MHz  throttle=0x0
09:41:20  gpu=83C  p=78.8W  sm=2385 MHz  throttle=0x0
09:41:22  gpu=83C  p=76.5W  sm=2359 MHz  throttle=0x20
09:41:23  gpu=85C  p=73.6W  sm=2405 MHz  throttle=0x20
09:41:24  gpu=80C  p=74.1W  sm=2366 MHz  throttle=0x0   ← last sample, then reset
```

Two features stand out:

1. **Power sawtooths 50 ↔ 80 W on a ~3 s cycle** — a ~60% swing, driven by chunked prefill
   alternating with decode. At N=16 more prefills coincide, so the transients get larger.
2. **Chronic thermal throttling.** Across 5,530 samples, 465 (**8.4%**) carry a nonzero
   `clocks_event_reasons`, starting 09:34:50 (during level 12):

   | flag | meaning | samples |
   |---|---|---|
   | `0x20` | SwThermalSlowdown | 410 |
   | `0x48` | **HwThermalSlowdown + HwSlowdown** | 25 |
   | `0x68` | HwThermal + SwThermal + HwSlowdown | 20 |
   | `0x04` | SwPowerCap | 10 |

   45 samples carry the **hardware** thermal-slowdown bit — the silicon's own emergency net.

Thermal envelope: GPU peaks **85 °C**; host SoC zones peak **96 °C** against a **104 °C**
critical trip. NVMe 60 °C.

**Why temperature is not a sufficient explanation:** at the instant of death the GPU was
*cooling* (85 → 80 °C) with throttle cleared, temps plateaued rather than ran away, and
boot `-5` died **42 seconds after boot** while cold and idle. Heat is a real stress factor
that erodes margin; it does not by itself explain the resets.

### Interpretation

Memory-independent, load-triggered, reproducible-at-N=16, with **zero kernel output on a
verified-working netconsole** — this is a hardware / firmware fault, not a k8s or vLLM
resource issue. Leading hypothesis: **transient power-delivery collapse** under the
repetitive 50↔80 W prefill/decode sawtooth, with sustained ~95 °C SoC temperature reducing
margin. GB10 exposes no power limit at all (`Current/Min/Max Power Limit: N/A`), so the
only lever available is clock capping — see §10.

### Contributing environment problems worth cleaning up regardless

- Desktop session still on the GPU (`systemctl set-default multi-user.target` to drop it).
- SoC at 96 °C under load — improve airflow / ambient.
- ~9 pods in permanent CrashLoopBackOff — `dify-release-redis-*` and `wandb-controller` at
  37,000+ restarts, `banking/*` on ImagePullBackOff. Constant containerd churn adds noise
  to crash correlation.
- **Security:** `sshd` exposed with password auth, taking root brute-force attempts from
  the internet. Lock it down (key-only, no root, firewall) independent of the GPU issue.

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

## 5. Results collected (concurrency 1, 4, 8, 12)

Workload per request: prompt 8k–24k tokens (mean 16k, `--random-range-ratio 0.5`),
600 output tokens (`--ignore-eos`), plus a shared 2,128-token cached prefix.
`--request-rate inf` so exactly N requests are always in flight.
N=1/4/8 used 48 prompts; N=12 used 72.

| concurrency | dur (s) | req/s | **out tok/s** | total tok/s | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 | e2e p50 | e2e p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1**  | 500.4 | 0.096 | 59.2  | 1851 | 2796 ms | 3899 ms | 4184 ms | 12.3 ms | 12.6 ms | 37 ms  | 10.5 s | 14.2 s |
| **4**  | 173.4 | 0.277 | 170.8 | 5341 | 477 ms  | 629 ms  | 1110 ms | 22.3 ms | 23.5 ms | 66 ms  | 14.2 s | 20.3 s |
| **8**  | 130.0 | 0.369 | **227.9** | 7126 | 486 ms | 877 ms | 1271 ms | 33.4 ms | 34.9 ms | 133 ms | 20.7 s | 29.7 s |
| **12** | 346.0 | 0.208 | **125.2** ↓ | 3996 | 3607 ms | 13085 ms | **30121 ms** | 87.6 ms | 111.0 ms | 451 ms | **58.4 s** | 95.5 s |
| **16** | — | — | — | — | — | — | — | — | — | — | — | **NODE CRASH 102 s in** |

Raw `vllm bench serve` blocks for 1/4/8 (session 1, transcribed — the tmpfs logs were lost
in that reboot):

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

N=12 raw JSON/log are preserved at `bench-nemotron/crash-evidence-2026-08-28/conc-12.{json,log}`.

---

## 6. Analysis

**Capacity answer: 8 concurrent long-context users.** Against a coding-agent SLO of
TTFT p99 < 5 s and TPOT p50 < 80 ms, N=8 passes (1.27 s / 33 ms) and N=12 fails badly
(30.1 s / 87.6 ms).

**Throughput does not merely flatten past the knee — it collapses.**
- N=1 → N=4: 59 → 171 out tok/s (2.9×, near-linear)
- N=4 → N=8: 171 → 228 (+33%) — knee
- N=8 → N=12: 228 → **125 (−45%)** — collapse

A 45% *drop* in aggregate throughput while adding load means the scheduler is thrashing,
not saturating. KV capacity is not the cause: 12 requests × ≤24k tokens ≈ 288k tokens
against a 3,260,912-token pool. The cause is almost certainly prefill starvation —
12 concurrent ~16k-token prefills sharing a **2,192-token/step** budget, interleaved with
decode steps that must run the whole batch. TTFT p90 jumps 877 ms → 13.1 s across that
step, which is the fingerprint of prefill queueing.

**The single-user 2.8 s TTFT is real** (mean ≈ median across all 48 requests, not a warmup
artifact) — pure prefill of a ~16k prompt at 2192 tokens/step (~7–8 chunked steps).

**Two ceilings in the current config:**
1. `--max-num-seqs 32` — max 32 requests decoding at once.
2. `--max-num-batched-tokens 2192` — **the real limiter**, and pinned into `[2128, 4256)`
   *only because LMCache is attached*. Without LMCache this can be 8k–16k, prefilling
   several users per step: roughly 4–8× the prefill throughput.

**LMCache is barely helping this workload.** During an earlier opencode session the engine
logged `Prefix cache hit rate: 96%` (vLLM local) vs `External prefix cache hit rate: 2%`
(LMCache). vLLM's own GPU prefix cache serves nearly all reuse at low user counts; LMCache
only pays off under heavy multi-user KV eviction. Since it also forces the tiny
`max-num-batched-tokens` that produces the N=12 collapse, **the LMCache-off comparison
(§7c #1) is now the highest-value benchmark change.**

---

## 7. What to run

### 7a. Bring up the model + bench pod

```bash
kubectl apply -f nemotron-deployment.yaml
kubectl apply -f lmcache-deployment.yaml       # optional — see §6, consider skipping
kubectl -n vllm rollout status deploy/vllm-nemotron --timeout=600s

kubectl apply -f bench-nemotron/bench-pod.yaml
kubectl -n vllm wait --for=condition=Ready pod/vllm-bench --timeout=120s
```

### 7b. Run the sweep (with monitoring — see §8)

```bash
kubectl -n vllm cp bench-nemotron/conc_sweep_resume.sh vllm-bench:/root/conc_sweep_resume.sh
kubectl -n vllm exec vllm-bench -- chmod +x /root/conc_sweep_resume.sh
kubectl -n vllm exec vllm-bench -- bash -lc '/root/conc_sweep_resume.sh'
```

Results land on a **hostPath** (`/var/lib/vllm-bench-results`), so they survive a reset —
unlike session 1, whose tmpfs logs were lost. `MARKERS.txt` is written with
`dd oflag=dsync` before each level, so a crash cannot erase which level was running.

### 7c. Recommended additional runs

1. **Same sweep, LMCache OFF, `--max-num-batched-tokens 16384`** — strip
   `--kv-transfer-config` from the vLLM args and scale `deploy/lmcache` to 0. This is the
   apples-to-apples "does LMCache help or hurt multi-user" test and is the most likely fix
   for the N=12 collapse. **Highest-value remaining benchmark work.**
2. **Raise `--max-num-seqs` to 64–128** and re-sweep — only meaningful after #1.
3. **Short-chat profile** for comparison: `--dataset-name sharegpt --sharegpt-output-len
   300`, sweep concurrency to 128.
4. Push `--kv-cache-memory-bytes` up (20–40 GiB) once the hardware is trusted.

### 7d. Reading the output

`/var/lib/vllm-bench-results/summary.csv`, one row per concurrency level. **Capacity = the
highest concurrency where `ttft_p99_ms` and `tpot_p50_ms` are still under SLO** and
`out_tok_per_s` is still rising. Suggested SLOs for a coding agent: TTFT p99 < 5000 ms,
TPOT p50 < 80 ms.

---

## 8. Monitoring / forensics tooling (built in session 2)

All under `bench-nemotron/`. Everything writes `O_SYNC`, because an instant SoC reset discards the
page cache — that is why session 1's logs "ended mid-line" with nothing useful.

| file | what |
|---|---|
| `bench-nemotron/spark-forensics-setup.sh` | **run as root on the node.** Dumps pstore + rasdaemon DB + the tail of every crashed boot; arms netconsole at a collector IP; sets journald `SyncIntervalSec=1s`; installs `crashmon.service`. Idempotent. Edit `MAC_IP` before use. |
| `bench-nemotron/crashmon.py` | 1 Hz system sampler → `sysmon.csv`: GPU temp/util/clock/power/throttle, MemAvailable, Committed_AS, Dirty, **PSI cpu/mem/io**, loadavg, NVMe temp, all 7 thermal zones. Installed as `crashmon.service` (auto-restarts after a crash-reboot). |
| `bench-nemotron/gpumon.py` | 10 Hz GPU sampler → `gpu10hz.csv`. One long-lived `nvidia-smi -lms 100`. Needed to see the 3 s power sawtooth that 1 Hz averages away. **Not** a service — restart manually via `start-telemetry.sh` after a reboot. |
| `bench-nemotron/start-telemetry.sh` | launcher for `gpumon.py`. Kept as a file specifically so the ssh command line never contains the `pkill` pattern (an inline `pkill -f gpumon.py` matches — and kills — its own shell). |
| `bench-nemotron/conc_sweep_resume.sh` | sweep starting at N=12, with `dd oflag=dsync` level markers. |

**Collector side (netconsole):** `nc -u -l 6666 > netconsole.log` on the machine named in
`MAC_IP`. Verify it works before trusting silence:
`sudo bash -c "echo NETCONSOLE-TEST > /dev/kmsg"` should appear in the log.
netconsole did attach over WiFi (mt7925e) despite mac80211's usual lack of netpoll support.
Plugging `enP7s7` (5GbE) in would make it more reliable.

**Currently left running on the node:** `crashmon.service` (1 Hz, cheap). To stop it:
`sudo systemctl disable --now crashmon`. `gpumon` is stopped; restart with
`ssh spark 'bash /home/bharath/start-telemetry.sh'` before the next run.

---

## 9. Files

| path | what |
|---|---|
| `BENCHMARK-HANDOFF.md` | this doc |
| `bench-nemotron/bench-pod.yaml` | load-generator pod (same image, no GPU, HF cache mounted RO) |
| `bench-nemotron/conc_sweep.sh` | original full sweep (levels 1…64) |
| `bench-nemotron/conc_sweep_resume.sh` | resume sweep from N=12, with synced markers |
| `bench-nemotron/run_sweep.sh` | host orchestrator: memory guard + launch + collect |
| `bench-nemotron/crashmon.py`, `bench-nemotron/gpumon.py`, `bench-nemotron/start-telemetry.sh`, `bench-nemotron/spark-forensics-setup.sh` | §8 tooling |
| `bench-nemotron/crash-evidence-2026-08-28/` | **crash #4 evidence** — see below |
| `nemotron-deployment.yaml` | vLLM engine (repo root) — has the 08-28 caps |
| `lmcache-deployment.yaml` | LMCache MP server (repo root) — has the 08-28 caps |
| `lmcache-benchmark-report.md` | earlier A/B (LMCache on vs off) full report + first crash write-up |
| `bench-nemotron/bench_config.json` | `lmcache bench engine` — HEAVY long-doc-qa, 978 requests, forces KV eviction |
| `bench-nemotron/bench_config_smoke.json` | `lmcache bench engine` — 15 requests, nothing evicts; plumbing check |
| `bench-nemotron/bench_config_sweep.json` | `lmcache bench engine` — sweep base, 27 requests per level |
| `bench-nemotron/bench_config_mrc.json` | `lmcache bench engine` — multi-round-chat, 8 concurrent sessions |
| `bench-nemotron/bench_config.UNSAFE-crashed-the-node.json` | quarantined config that caused an 08-27 crash |
| `bench-nemotron/run.sh`, `bench-nemotron/sweep.sh` | drivers for the four configs above (ConfigMap + pod + follow logs) |
| `bench-nemotron/sweep-pod.yaml` | sequential concurrency sweep, one pod, fsync'd level markers |
| `bench-nemotron/nemotron_run.sh` | the raw uncapped `vllm serve` argv (reference only — NOT the deployed caps) |

`bench-nemotron/crash-evidence-2026-08-28/` contains: `sysmon.csv` (1 Hz, both boots),
`gpu10hz.csv` (10 Hz, 5,530 samples through the crash), `MARKERS.txt` (level timings),
`bench-summary.csv`, `conc-12.{json,log}`, `netconsole.log` (the verified-working,
crash-silent capture), `ras-summary.txt`.

---

## 10. Open questions & the next experiment

**Do this first — the clock-lock experiment (~5 min).** It is the one test that
discriminates the leading hypothesis, and GB10 exposes no power limit, so clocks are the
only lever:

```bash
ssh spark 'bash /home/bharath/start-telemetry.sh'        # re-arm 10 Hz sampler
ssh -t spark 'sudo nvidia-smi -lgc 0,1800'               # cap SM clock (was 2411–2470)
# then run ONLY level 16 (edit LEVELS=(16) in conc_sweep_resume.sh)
ssh -t spark 'sudo nvidia-smi -rgc'                      # restore afterwards
```

- **Survives N=16 capped** ⇒ confirms transient power / thermal margin. Remedies: keep a
  clock cap in production, improve cooling, and open an NVIDIA case citing §3.
- **Still crashes capped** ⇒ power transient is ruled out; the fault is elsewhere in the
  SoC/firmware and the case for RMA / BSP update strengthens.

Remaining questions:

1. Is the GB10 lockup reproducible on a second Spark, or specific to this unit? (unit-specific
   → RMA; model-wide → NVIDIA driver/BSP bug, needs a support case with the §3 signature.)
2. Does a newer DGX Spark BSP / NVIDIA driver (> 580.173.02) fix it? Ask NVIDIA specifically
   why `tegra234_cbb` is blacklisted **and** absent from the modules tree — with it, a
   fabric error would be reportable.
3. With LMCache removed and `max-num-batched-tokens` raised to 16384, where does the real
   concurrency knee land, and does the N=12 collapse disappear? (§7c #1)
4. Does the crash follow *load transients* or *sustained load*? A ramped-arrival run
   (`--request-rate 2` instead of `inf`) at N=16 would separate these: same steady-state
   load, far gentler transients.
