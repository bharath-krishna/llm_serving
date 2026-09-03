# nemotron

Two variants of `NVIDIA-Nemotron-3.5-Lightning-30B-A3B`, served from one URL on a
single-GPU DGX Spark (GB10) node.

```
common/     shared by both variants -- ingress, the live-variant Service, LMCache, auth
nvfp4/      the NVFP4 variant (default, known-good) + its benchmarks
bf16/       the BF16 variant + its benchmarks
tools/      node-side crash/GPU telemetry -- variant-agnostic
switch.sh   the supported way to change which variant is live
```

## The two variants

| | nvfp4 | bf16 |
|---|---|---|
| checkpoint | `…-A3B-NVFP4` | `…-A3B-BF16` |
| weights | 17.86 GiB | 58.93 GiB |
| load time | ~103 s | ~334 s |
| KV dtype | `fp8_e4m3` (checkpoint ships KV scales) | bf16 (`auto`; no scales) |
| KV bytes/token | 3,497 | ~6,860 |
| `--max-model-len` | 131072 | 131072 |
| GPU KV pool | 512 MiB | 1 GiB |
| LMCache L1 | 4 GiB | 6 GiB |
| engine footprint | ~22.5 GiB | ~64 GiB |
| LMCache | on (`lmcache-nvfp4`) | on (`lmcache-bf16`) |
| attention block size = LMCache `--chunk-size` | 2128 | 1072 |
| committed `replicas` | 1 | 0 |

Only one runs at a time: both request `nvidia.com/gpu: "1"` and the node has one
GPU, and their footprints could not coexist in the shared 128 GiB unified pool
anyway.

## One URL, one key, either variant

`https://nemo35-lightning.krishb.in` (and `nemotron.local`) reach whichever
variant is up, authenticated with the same `vllm-auth` API key
([common/SECRETS.md](common/SECRETS.md)).

That works without anything being edited at switch time. Both Deployments stamp
`serving: nemotron` on their pods; `common/service-active.yaml` selects on that
label alone, so its endpoints follow whichever pod is Running; and
`common/ingress.yaml` points at that Service permanently.

BF16 also answers to the plain `nemotron-3.5-lightning` model string as an alias,
so a client hardcoding the NVFP4 name keeps working across a switch. `/v1/models`
still leads with `nemotron-3.5-lightning-bf16` so the API stays honest about
which variant answered.

**Do not point the Ingress at a per-variant Service, and do not add a second
Ingress for these hosts.** haproxy-ingress binds a host to exactly one Ingress
and resolves ties to the *older* one, silently ignoring the newer duplicate. On
2026-08-31 the BF16 deployment shipped its own Ingress for
`nemo35-lightning.krishb.in`, the older one won, and it pointed at a Service with
zero endpoints — the public URL failed while BF16 was serving fine behind the
loser. The per-variant Services (`vllm-nemotron-nvfp4`, `vllm-nemotron-bf16`)
exist for benchmarks, which address a variant explicitly so a result can never be
misattributed.

## Switching

```bash
./switch.sh status     # what is live right now
./switch.sh bf16       # ~6 min: BF16 loads a 61.31 GiB checkpoint
./switch.sh nvfp4
```

It scales the outgoing engine to 0, waits for the pod to actually be *gone* (the
GPU is not released until the container exits, and the incoming pod stays Pending
until then), takes the outgoing LMCache server down, brings the incoming one up
*before* its engine (`REGISTER_KV_CACHE` runs at engine startup and fails with
nothing listening), then starts the engine and verifies through the public URL.

It never touches the Ingress — see the header of `switch.sh` for why.

## LMCache

**Both variants use LMCache, and each has its own server** —
`nvfp4/lmcache.yaml` and `bf16/lmcache.yaml`. They cannot share one, because the
server's `--chunk-size` must equal vLLM's unified block size and that is
dtype-derived:

```
block = ceil16( mamba_state_elems_per_layer × sizeof(mamba_ssm_cache_dtype)
                ─────────────────────────────────────────────────────────── )
                     2 × num_kv_heads × head_dim × sizeof(kv_cache_dtype)
```

| variant | kv dtype | ssm dtype | block = `--chunk-size` | `--max-num-batched-tokens` | L1 |
|---|---|---|---|---|---|
| nvfp4 | `fp8_e4m3` | float16 | 2128 | 2192 | 4 GiB |
| bf16 | bf16 | float16 | 1072 | 1072 | 6 GiB |

Both measured from the engine's own `Setting attention block size to N tokens`
line. `validate_mamba_step_alignment` checks `--max-num-batched-tokens` against
the server's chunk size on connect and **fails the engine at startup** on a
mismatch — so this is loud, not silent.

A shared server would have to be reconfigured and restarted on every switch, and
restarting it drops the live engine's GPU-context registration. One server per
variant, scaled with its engine by `switch.sh`, avoids that. Each also keeps its
own L2 directory (`/kv-l2/nvfp4`, `/kv-l2/bf16`), so neither needs clearing and a
warm cold-tier survives being switched away from and back.

**Exactly one server runs at a time.** Both carry the pod label
`serving: lmcache` that the stable `lmcache` / `lmcache-http` Services in
`common/lmcache-services.yaml` select on — which is what keeps the engine's
`--kv-transfer-config`, the UI's nginx upstream and every bench config's
`lmcache_url` constant across a switch. Two running servers would give the ZMQ
connector an arbitrary choice of endpoint with the wrong chunk size behind it;
`switch.sh` enforces the ordering and verifies the count.

`common/lmcache-ui.yaml` is the dashboard (`lmcache-ui.krishb.in`). It is shared:
it polls over HTTP through `lmcache-http`, so it follows the live server without
knowing anything about variants.

## Benchmarks

Each variant owns its bench assets under `<variant>/bench/`, with distinct pod
names, ConfigMaps and result directories so both can exist on the cluster at once:

| | nvfp4 | bf16 |
|---|---|---|
| ConfigMap | `vllm-bench-config-nvfp4` | `vllm-bench-config-bf16` |
| bench pod | `vllm-bench-nvfp4` | `vllm-bench-bf16` |
| sweep pod | `vllm-sweep-nvfp4` | `vllm-sweep-bf16` |
| results | `/var/lib/vllm-bench-results/nvfp4/` | `/var/lib/vllm-bench-results/bf16/` |

```bash
nvfp4/bench/run.sh                          # smoke, 15 requests
nvfp4/bench/run.sh bench_config_mrc.json    # multi-round chat
nvfp4/bench/sweep.sh "1 2 4 8 12"           # concurrency sweep
```

BF16's heavy configs are **guarded** — they size themselves from constants that
are still derivations rather than measurements. `bf16/README.md` says how to
unblock them.

> **This node hard-resets under sustained long-context load.** Four times so far,
> with no kernel output — the failure is a unified-memory exhaustion → NVIDIA
> driver allocation failure → SoC lockup, not a recoverable OOM. Run
> `tools/start-telemetry.sh` on the node before anything past a smoke config, and
> read `../BENCHMARK-HANDOFF.md` section 3 first.

## Applying from scratch

```bash
kubectl apply -f common/          # ingress, stable Services, lmcache UI
kubectl apply -f nvfp4/           # engine + lmcache-nvfp4  (replicas: 1)
kubectl apply -f bf16/            # engine + lmcache-bf16   (replicas: 0)
kubectl apply -f ../servicemonitors.yaml
```

Order does not matter; `bf16/` at `replicas: 0` means applying everything never
contends for the GPU. The `vllm-auth` Secret is created out of band
([common/SECRETS.md](common/SECRETS.md)).

## One-time migration from the pre-split layout

**Not yet applied to the cluster.** The old objects were named `vllm-nemotron`
(NVFP4) and `vllm-nemotron-bf16`; NVFP4 is now `vllm-nemotron-nvfp4`, so a plain
apply leaves the old Deployment and Service orphaned rather than replacing them.

Two things make the order matter:

* Both pod templates gain the `serving: nemotron` label. That is a template
  change, so the running pod is **recreated** — unavoidably paying the checkpoint
  reload (~334 s on BF16, ~103 s on NVFP4). There is no zero-downtime path.
* `common/ingress.yaml` re-points the existing `vllm-nemotron` Ingress at
  `vllm-nemotron-active`, which has no endpoints until a relabelled pod is
  Running. Apply it **after** the engines, not before, or the public URL serves
  503s for the whole reload window.

Assuming BF16 is the live variant (adjust if NVFP4 is):

```bash
# 1. engines first -- the OLD ingress still routes to vllm-nemotron-bf16
#    throughout, so the only outage is the reload itself
kubectl apply -f nvfp4/ -f bf16/
kubectl scale deploy/vllm-nemotron-bf16 -n vllm --replicas=1
kubectl rollout status deploy/vllm-nemotron-bf16 -n vllm --timeout=600s

# 2. now that a labelled pod is Running, hand the Ingress over
kubectl apply -f common/
kubectl get endpoints vllm-nemotron-active -n vllm      # must be non-empty
./switch.sh status

# 3. retire the orphans -- Deployment and Service only. Do NOT delete the
#    Ingress named vllm-nemotron: common/ingress.yaml reuses that name and
#    updates it in place, and deleting it would drop the host binding.
kubectl delete deploy/vllm-nemotron svc/vllm-nemotron -n vllm
kubectl delete pod vllm-bench -n vllm --ignore-not-found   # old bench pod name

# 4. monitoring
kubectl apply -f ../servicemonitors.yaml
```

Verify the public URL end to end before and after step 2:

```bash
KEY=$(kubectl get secret vllm-auth -n vllm -o jsonpath='{.data.api-key}' | base64 -d)
curl -fsS https://nemo35-lightning.krishb.in/v1/models -H "Authorization: Bearer $KEY"
```

## Related docs at the repo root

| | |
|---|---|
| `BENCHMARK-HANDOFF.md` | concurrency sweep, the four node crashes, forensics tooling. NVFP4. |
| `lmcache-benchmark-report.md` | LMCache on-vs-off A/B: 2.31× end-to-end when the working set oversubscribes the pool. NVFP4. |
| `kv-cache-sizing-calculator.xlsx` | the tier-sizing spreadsheet. Populated for NVFP4 only. |
| `AGENTS.md` | repo-level notes, incl. the unused Helm/`production-stack` path. |
