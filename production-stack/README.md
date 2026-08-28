# vLLM Production Stack deployment

Replaces the hand-rolled `../nemotron-deployment.yaml` and `../gemma4-deployment.yaml`
with the [vLLM Production Stack](https://docs.vllm.ai/projects/production-stack/en/latest/getting_started/quickstart.html)
Helm chart (`vllm/vllm-stack` **0.1.12**): a router (OpenAI-compatible front door +
metrics) plus one serving engine per model.

Deployed as helm release `vllm` in namespace `vllm`. **Status: nemotron live.**

## Layout

- `values.yaml` — the file you normally edit. Two `modelSpec` entries
  (`nemotron`, `gemma4`). Single-GPU box (GB10 / arm64), so only **one** model is
  `enabled: true` at a time.
- `router-arm64/Dockerfile` — arm64 rebuild of the router image (see below).
- `post-render.sh` + `kustomization.yaml` + `patch-lmcache-sidecar.yaml` — Helm
  post-renderer that injects the LMCache MP server sidecar into the nemotron pod
  (chart has no field for it). **Every `helm upgrade` must pass
  `--post-renderer production-stack/post-render.sh`.**

## What maps where

| Old manifest | Production stack |
| --- | --- |
| `vllm serve … <flags>` | `modelSpec.vllmConfig` (typed flags) + `vllmConfig.extraArgs` (everything else, incl. `--served-model-name` — the chart has no field for it) |
| `--enable-auto-tool-choice` / `--tool-call-parser` | `modelSpec.enableTool` / `modelSpec.toolCallParser` |
| `hostPath` HF cache at `/root/.cache/huggingface` | `sharedPvcStorage.hostPath` → hostPath PV, mounted at `/data/shared-pvc-storage`, set as `HF_HOME` |
| per-model `Ingress` (haproxy) | `routerSpec.ingress` → all 4 hostnames point at `vllm-router-service:80` |
| `gemma4.py` patch mount | `gemma4.extraVolumeMounts` (unchanged) |
| manual `--kv-transfer-config` `LMCacheMPConnector` + `hostIPC` | `vllmConfig.extraArgs` kv-transfer-config (`tcp://localhost:6555`) + LMCache MP server as a post-render sidecar — see "LMCache" below |

The router dispatches by the request body `model` field, **not** by Host header —
`nemo35-lightning.krishb.in` + `"model":"gemma4-26b-uncensored"` still routes to gemma4.

## arm64 router image

Upstream `lmcache/lmstack-router` (and the ghcr equivalent) ship **amd64 only** — no
arm64 manifest at any tag. Rebuilt from the production-stack repo:

```bash
git clone --depth 1 https://github.com/vllm-project/production-stack.git
cd production-stack
cp /path/to/this/repo/production-stack/router-arm64/Dockerfile docker/Dockerfile.arm64
docker build -f docker/Dockerfile.arm64 -t registry.krishb.in/vllm/lmstack-router:0.1.12-arm64 .
docker push registry.krishb.in/vllm/lmstack-router:0.1.12-arm64
```

- `INSTALL_OPTIONAL_DEP=default` (empty extras) — skips `semantic_cache` / `lmcache`,
  which pull torch / faiss / vllm with no arm64 wheels at the pinned versions.
  `routingLogic: roundrobin` needs none of them. (If you ever switch to `kvaware`
  routing you'll need to solve the `lmcache` extra for arm64 first.)
- `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.12` — the shallow clone has no router version
  tag, so setuptools_scm needs it pinned.
- Pulled via imagePullSecret `registry-krishb-creds` in the `vllm` namespace (copied
  from the `banking` namespace: `kubectl -n banking get secret registry-krishb-creds -o yaml | ... | kubectl -n vllm apply -f -`).

## LMCache (nemotron) — MP connector + in-pod sidecar

The chart's `lmcacheConfig.enabled` is **not usable** here: it emits
`--kv-transfer-config {"kv_connector":"LMCacheConnectorV1",...}`, and
`LMCacheConnectorV1` has no hybrid-model code — vLLM 0.27.1 then disables its
hybrid KV-cache manager and aborts:

```
ValueError: Failed to promote local KV cache specs to one unified type.
  vllm/v1/core/kv_cache_utils.py:_promote_local_kv_cache_specs
```

(the fp8 attention cache + float16 mamba-state cache have no common unified spec).

Instead, `values.yaml` passes a **`LMCacheMPConnector`** kv-transfer-config by
hand (in `vllmConfig.extraArgs`). That connector advertises hybrid support, so
vLLM keeps the hybrid manager on and stores each KV group separately. Its only
hybrid-capable transport is `lmcache_driven` = **CUDA IPC + POSIX `/dev/shm`**,
which is strictly node-local — there is **no** network/remote-server path for a
hybrid model (`engine_driven`/`lm://` reject hybrid KV groups). So:

- The LMCache MP server runs as a **native sidecar** (`lmcache-mp-server`) in the
  nemotron pod, sharing the pod IPC namespace + the `dshm` `/dev/shm` emptyDir +
  localhost. Injected by the Helm **post-renderer** (`post-render.sh` →
  `kubectl kustomize` → `patch-lmcache-sidecar.yaml`), because chart 0.1.12 has
  no field for a custom sidecar / hostIPC / pod annotations.
- Same image as the engine (`vllm/vllm-openai:v0.27.1`) ⇒ identical LMCache
  0.5.3 ⇒ MP wire protocol always matches. No separate image.
- **GPU for the sidecar:** the node's device plugin is CDI-only
  (`DEVICE_LIST_STRATEGY=cdi-*`), so `NVIDIA_VISIBLE_DEVICES` is ignored and a
  container without `nvidia.com/gpu` gets no GPU. The pod annotation
  `cdi.k8s.io/lmcache: nvidia.com/gpu=0` injects the GPU (device nodes + driver
  libs) into every container in the pod **without** consuming the allocatable
  count. The vLLM container gets a redundant injection of the same device;
  containerd 2.x dedupes it.
- **Mamba/LMCache hard constraints** (engine refuses to start otherwise):
  `--chunk-size` (server) == vLLM unified `block_size` == **2192** for this model
  (from `interface.py`: *"Setting attention block size to 2192 tokens ..."*), and
  `2192 <= --max-num-batched-tokens < 4384`. Server also needs
  `--separate-object-groups`; vLLM needs `--mamba-cache-mode align`
  `--enable-prefix-caching` (already set).

Caveats: GDN/Mamba cached generation is **not bit-exact** vs a cold run; Mamba
`align`-mode prefix caching is flagged experimental upstream; this exact stack
(GB10 + NemotronH + MP mode) is not on LMCache's validated list. `LMCACHE_LOG_LEVEL`
is `DEBUG` on the sidecar for bringup — drop to `INFO` in `patch-lmcache-sidecar.yaml`
once store/load activity is confirmed.

### Deploy / upgrade with the sidecar

```bash
helm upgrade --install vllm vllm/vllm-stack -n vllm \
  -f production-stack/values.yaml --version 0.1.12 \
  --post-renderer production-stack/post-render.sh
```

Always pass `--post-renderer` — a plain `helm upgrade` drops the sidecar.

## Chart 0.1.12 quirks worked around in values.yaml

- Bare ints > 1e6 render in scientific notation → `maxModelLen` is quoted.
- `shmSize` is only honored when `tensorParallelSize` is set, **and** the shared-cache
  volumeMount renders without its `volumeMounts:` header (invalid YAML) unless the
  modelSpec has `extraVolumeMounts` → `nemotron` defines an explicit `/dev/shm`
  emptyDir (`dshm`), covering both — and this same volume is the shared-memory
  channel to the LMCache sidecar (see LMCache section).
- No field for a custom sidecar, `hostIPC`, or pod annotations → the LMCache MP
  server is added by the `--post-renderer` (`post-render.sh` + `kustomization.yaml`
  + `patch-lmcache-sidecar.yaml`).
- `sharedPvcStorage` with an empty `storageClass` lets the default-StorageClass webhook
  rewrite the PVC to `nfs-client`, which then won't bind the hostPath PV
  (`storageClassName does not match`) → set `storageClass: "manual"` (a non-existent
  class; `volumeName` still forces static binding).
- `vllm-gemma4-engine-service` is created even while gemma4 is disabled (Service
  template isn't gated on `enabled`). Harmless — no endpoints.
- `requestCPU` was dropped 12 → 4; the node only had ~4.5 cores unreserved.

## Swap models (single GPU)

```bash
# edit values.yaml: nemotron enabled:false, gemma4 enabled:true
helm upgrade vllm vllm/vllm-stack -n vllm -f production-stack/values.yaml --version 0.1.12 \
  --post-renderer production-stack/post-render.sh
# the rollout deadlocks on 1 GPU (maxUnavailable:0) — delete the old pod to break it;
# if the old pod is stuck Terminating (slow SIGTERM during model load), force it and
# scale its ReplicaSet to 0:
kubectl -n vllm delete pod -l model=nemotron --force --grace-period=0
```

The post-renderer only patches the `vllm-nemotron-deployment-vllm` Deployment by name,
so it's a no-op when nemotron is disabled — still always pass `--post-renderer`.
For gemma4 (dense-ish hybrid, different image) the LMCache story is unverified; it runs
without a connector for now. Rebuild the router only if you change routing logic.

## Verify

```bash
kubectl -n vllm get pods,pv,pvc
# nemotron pod is 2/2: initContainers=[lmcache-mp-server (native sidecar)], containers=[vllm]
kubectl -n vllm logs deploy/vllm-nemotron-deployment-vllm -c lmcache-mp-server -f   # store/load activity
kubectl -n vllm port-forward svc/vllm-router-service 30080:80 &
curl -s localhost:30080/v1/models | jq
curl -s localhost:30080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"nemotron-3.5-lightning","messages":[{"role":"user","content":"2+2?"}],"max_tokens":50}' | jq
# external (haproxy NodePort 30080 on the node):
curl -s -H 'Host: nemo35-lightning.krishb.in' http://<node-ip>:30080/v1/models | jq
```

**Confirm LMCache is actually caching** (not just loaded): send a long shared prefix
twice; the sidecar log should show `STORE` on the first and `RETRIEVE` / lookup hits on
the second. To isolate LMCache from vLLM's own GPU prefix cache, launch with
`VLLM_SERVER_DEV_MODE=1` and `curl -XPOST .../reset_prefix_cache` between runs (omit
`reset_external=true`).

## Rollback

```bash
helm uninstall vllm -n vllm
kubectl delete pv vllm-shared-pvc-storage
kubectl apply -f nemotron-deployment.yaml
```

The shared-cache PV is a hostPath to `~/.cache/huggingface`, left untouched.
