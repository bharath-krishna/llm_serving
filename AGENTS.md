# AGENTS.md — llm-serving

## Quickstart

- **Run the app**: `python main.py` — prints "Hello from llm-serving!"
- **Install deps**: `uv sync` (this repo uses `uv` for Python 3.12 dependency management; `pyproject.toml` requires `>=3.12`, dependencies: `lmcache`, `openai`)
- **Lint/typecheck**: Check `.github` or repo root for CI scripts; no dedicated lint config found beyond standard Python tooling.

## Helm deployment (vLLM Production Stack)

This repo is a custom configuration for the **vLLM Production Stack** Helm chart. Key workflow:

- **Deploy**: `helm upgrade --install vllm vllm/vllm-stack -n vllm -f production-stack/values.yaml --version 0.1.12 --post-renderer production-stack/post-render.sh`
- **Always pass `--post-renderer`** — a plain `helm upgrade` drops the LMCache MP sidecar. The post-renderer injects `lmcache-mp-server` as a native sidecar into the nemotron pod via `kubectl kustomize` + `patch-lmcache-sidecar.yaml`.
- **Swap models** (single GPU only): set `enabled: false` on the current model, `enabled: true` on the target, then re-run the helm upgrade with `--post-renderer`. If the old pod is stuck `Terminating`, force-delete it:
  ```
  kubectl -n vllm delete pod -l model=nemotron --force --grace-period=0
  ```

## LMCache MP constraints (nemotron only)

These are **hard constraints** — vLLM will refuse to start if they're wrong:

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

- **nemotron** (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4): enabled by default. Uses `vllm/vllm-openai:v0.27.1` image. Shares `/dev/shm` (32Gi) with LMCache MP sidecar via `extraVolumes`/`extraVolumeMounts` (`dshm` emptyDir).
- **gemma4** (Gemma-4-26B-A4B-it-Uncensored-NVFP4): disabled by default. Uses custom image `ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4`. No LMCache support. Has a `gemma4-patch` hostPath volume for monkeypatching `gemma4.py`.

## Shared HuggingFace cache

- `sharedPVCStorage` creates a `hostPath` PVC bound to `/home/bharath/.cache/huggingface`, mounted at `/data/shared-pvc-storage` in every pod, set as `HF_HOME`.
- `storageClass: "manual"` prevents the default StorageClass webhook from re-provisioning this PVC as NFS (which would break the hostPath PV).

## Router (arm64)

- Upstream `lmcache/lmstack-router` is amd64-only. This repo provides an arm64 rebuild via the `production-stack/router-arm64/Dockerfile`.
- Build: `docker build -f production-stack/router-arm64/Dockerfile -t registry.krishb.in/vllm/lmstack-router:0.1.12-arm64 .` then push.
- `INSTALL_OPTIONAL_DEP=default` is recommended — optional deps (semantic_cache/lmcache) pull torch/faiss/vllm with no arm64 wheels at pinned versions. Round-robin routing needs no optional deps.
- `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.12` is needed because the shallow git clone has no version tag.