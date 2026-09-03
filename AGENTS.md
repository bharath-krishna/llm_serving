# AGENTS.md — llm-serving

## Quickstart

- **Run the app**: `python main.py` — prints "Hello from llm-serving!"
- **Install deps**: `uv sync` (this repo uses `uv` for Python 3.12 dependency management; `pyproject.toml` requires `>=3.12`, dependencies: `lmcache`, `openai`)
- **Lint/typecheck**: Check `.github` or repo root for CI scripts; no dedicated lint config found beyond standard Python tooling.

## What is actually deployed

The nemotron models are served by **hand-rolled manifests under `nemotron/`**, not
by Helm. Start at [`nemotron/README.md`](nemotron/README.md).

```
nemotron/common/   ingress, the live-variant Service, LMCache + its UI, auth notes
nemotron/nvfp4/    NVFP4 variant (default, replicas: 1) + benchmarks
nemotron/bf16/     BF16 variant (replicas: 0) + benchmarks
nemotron/switch.sh the supported way to change which variant is live
```

Two variants of the same model share one URL (`nemo35-lightning.krishb.in`) and
one API key. Only one runs at a time — single GPU. The Ingress is never edited to
switch: it points at a Service that selects on a label both variants carry.

`gemma4-deployment*.yaml` and `bench-gemma4/` at the repo root are a separate,
unrelated model and were not part of that reorganisation.

## Helm deployment (vLLM Production Stack) — ALTERNATE, NOT IN USE

`production-stack/` is a parallel Helm path that is **not what is running**, and
its constants contradict the deployed manifests in several places (LMCache port
6555 vs 5555, `--chunk-size` 2192 vs 2128, `--l1-size-gb` 16 vs 4, DSpark on vs
off, `maxModelLen` 1048576 vs 131072, an in-pod sidecar vs a separate
Deployment). Treat the numbers below as belonging to that path only. Workflow:

- **Deploy**: `helm upgrade --install vllm vllm/vllm-stack -n vllm -f production-stack/values.yaml --version 0.1.12 --post-renderer production-stack/post-render.sh`
- **Always pass `--post-renderer`** — a plain `helm upgrade` drops the LMCache MP sidecar. The post-renderer injects `lmcache-mp-server` as a native sidecar into the nemotron pod via `kubectl kustomize` + `patch-lmcache-sidecar.yaml`.
- **Swap models** (single GPU only): set `enabled: false` on the current model, `enabled: true` on the target, then re-run the helm upgrade with `--post-renderer`. If the old pod is stuck `Terminating`, force-delete it:
  ```
  kubectl -n vllm delete pod -l model=nemotron --force --grace-period=0
  ```

## LMCache MP constraints (nemotron only)

These are **hard constraints** — vLLM will refuse to start if they're wrong. Note
that the numeric ones are **per-variant**, because the block size is derived from
the KV dtype: the NVFP4 checkpoint ships KV scales so vLLM picks
`kv_cache_dtype=fp8_e4m3` (1 byte/elem), while BF16 has none and lands on 2, so
its attention page holds half as many tokens.

| | NVFP4 (deployed) | BF16 |
|---|---|---|
| `--chunk-size` (server) == `block_size` | 2128 | **unmeasured** |
| `--max-num-batched-tokens` in `[block, 2*block)` | 2192 | unmeasured (1072 is a derivation) |
| uses LMCache | yes | no |

The 2192 quoted below is the **DSpark** block size from the Helm path, not the
deployed one. Do not carry it into the hand-rolled manifests, and do not treat
any block size as fixed — read it off the engine log
(`grep -i "attention block size"`). Getting BF16's is step 1 of
[`nemotron/bf16/README.md`](nemotron/bf16/README.md).

Helm-path values:

- `--chunk-size` (server) == `block_size` == **2192** tokens
- `2192 <= --max-num-batched-tokens < 4384` (must be exactly 2192 for this config)
- `--mamba-cache-mode align` --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5
- `--mamba-backend flashinfer`
- `--kv-transfer-config {"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6555}}`
- LMCache sidecar runs in the same pod, sharing IPC namespace + `/dev/shm` — the post-renderer handles this; **do not skip `--post-renderer`**

If using `LMCacheConnectorV1` (the chart's default `lmcacheConfig`), vLLM aborts with:
```
ValueError: Failed to promote local KV cache specs to one unified type.
```
(LMCacheMPConnector is required for hybrid Mamba+attention models.)

## Model-specific notes

- **nemotron** — now two variants, NVFP4 and BF16; see
  [`nemotron/README.md`](nemotron/README.md) for the comparison table. In the
  Helm path (below) only NVFP4 exists, enabled by default, using
  `vllm/vllm-openai:v0.27.1` and sharing `/dev/shm` (32Gi) with an LMCache MP
  sidecar via `extraVolumes`/`extraVolumeMounts` (`dshm` emptyDir). The deployed
  manifests do the opposite — `hostIPC: true` with NO `dshm` emptyDir, and a
  separate LMCache Deployment rather than a sidecar.
- **gemma4** (Gemma-4-26B-A4B-it-Uncensored-NVFP4): disabled by default. Uses custom image `ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4`. No LMCache support. Has a `gemma4-patch` hostPath volume for monkeypatching `gemma4.py`.

## Shared HuggingFace cache

- `sharedPVCStorage` creates a `hostPath` PVC bound to `/home/bharath/.cache/huggingface`, mounted at `/data/shared-pvc-storage` in every pod, set as `HF_HOME`.
- `storageClass: "manual"` prevents the default StorageClass webhook from re-provisioning this PVC as NFS (which would break the hostPath PV).

## Router (arm64)

- Upstream `lmcache/lmstack-router` is amd64-only. This repo provides an arm64 rebuild via the `production-stack/router-arm64/Dockerfile`.
- Build: `docker build -f production-stack/router-arm64/Dockerfile -t registry.krishb.in/vllm/lmstack-router:0.1.12-arm64 .` then push.
- `INSTALL_OPTIONAL_DEP=default` is recommended — optional deps (semantic_cache/lmcache) pull torch/faiss/vllm with no arm64 wheels at pinned versions. Round-robin routing needs no optional deps.
- `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.12` is needed because the shallow git clone has no version tag.