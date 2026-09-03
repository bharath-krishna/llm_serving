#!/usr/bin/env bash
# Switch which nemotron variant serves https://nemo35-lightning.krishb.in.
#
#   ./switch.sh nvfp4     # the default / known-good variant
#   ./switch.sh bf16
#   ./switch.sh status    # what is live right now
#   ./switch.sh apply     # apply every manifest, then restore the live variant
#
# USE `apply`, NOT `kubectl apply -f`, TO ROLL OUT A CONFIG CHANGE. Each variant
# file carries a committed `replicas:` -- 1 for nvfp4, 0 for bf16 -- so that
# applying the whole tree from scratch never starts two engines on one GPU. The
# cost is that a plain `kubectl apply -f bf16/lmcache.yaml` to change one flag
# ALSO resets that Deployment to 0 and silently takes it down. `apply` below
# records what was live, applies everything, and puts the scale back.
#
# The node has ONE GPU and both Deployments request `nvidia.com/gpu: "1"`, so
# only one can be Running -- the other would sit Pending forever. This script is
# the supported way to trade them, because the order matters: the outgoing pod
# must be GONE (not merely scaled) before the incoming one can be scheduled.
#
# WHAT THIS SCRIPT DOES NOT DO
#
#   * It does not touch the Ingress. common/ingress.yaml points at the Service
#     `vllm-nemotron-active`, which selects on the pod label `serving: nemotron`
#     that both variants carry -- so the public URL follows whichever pod is
#     Running with no edit at all. Patching a backend by hand is what broke that
#     URL on 2026-08-31; there is deliberately nothing here to forget.
#
# WHAT THIS SCRIPT DOES MOVE, BESIDES THE ENGINE
#
#   * The LMCache server. There is one per variant (nvfp4/lmcache.yaml,
#     bf16/lmcache.yaml) because --chunk-size must equal the engine's unified
#     block size -- 2128 for NVFP4, 1072 for BF16, both measured. Exactly one may
#     run: they share the pod label `serving: lmcache` that common's stable
#     `lmcache` / `lmcache-http` Services select on, so two would give the ZMQ
#     connector an arbitrary choice of endpoint with the wrong chunk size behind
#     it. The order below matters -- the server must be up BEFORE its engine, or
#     the engine's REGISTER_KV_CACHE fails at startup.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
NS=vllm

usage() { echo "usage: $0 {nvfp4|bf16|status|apply}" >&2; exit 2; }

# Which variant is serving right now, by pod label rather than by replica count
# -- a Deployment scaled to 1 whose pod is still loading weights is not serving.
live_variant() {
  kubectl get pods -n "$NS" -l serving=nemotron --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.labels.variant}' 2>/dev/null || true
}
[ $# -eq 1 ] || usage

# BF16 loads a 61.31 GiB checkpoint in ~334 s; NVFP4's 17.86 GiB takes ~103 s.
# The rollout timeout has to cover the slower one plus scheduling.
deploy_of()  { case "$1" in nvfp4) echo vllm-nemotron-nvfp4 ;; bf16) echo vllm-nemotron-bf16 ;; esac; }
lmcache_of() { case "$1" in nvfp4) echo lmcache-nvfp4 ;; bf16) echo lmcache-bf16 ;; esac; }
timeout_of() { case "$1" in nvfp4) echo 420s ;; bf16) echo 900s ;; esac; }

status() {
  echo "=== engine pods ==="
  kubectl get pods -n "$NS" -l serving=nemotron \
    -o custom-columns='POD:.metadata.name,VARIANT:.metadata.labels.variant,STATUS:.status.phase,READY:.status.containerStatuses[0].ready' \
    2>/dev/null || echo "  (none)"
  echo
  echo "=== what the public URL resolves to ==="
  # Empty endpoints means no variant is up -- or the incoming one is still
  # loading weights, which on BF16 looks identical for ~5 minutes.
  kubectl get endpoints vllm-nemotron-active -n "$NS" \
    -o jsonpath='{range .subsets[*].addresses[*]}  {.ip}{"\t"}{.targetRef.name}{"\n"}{end}' 2>/dev/null \
    || echo "  (no endpoints -- nothing serving)"
  echo
  echo "=== lmcache servers (one per variant; exactly one should be up) ==="
  kubectl get pods -n "$NS" -l serving=lmcache \
    -o custom-columns='POD:.metadata.name,VARIANT:.metadata.labels.variant,STATUS:.status.phase,READY:.status.containerStatuses[0].ready' \
    2>/dev/null || echo "  (none)"
  live=$(kubectl get pods -n "$NS" -l serving=lmcache --field-selector=status.phase=Running \
           -o jsonpath='{.items[0].metadata.labels.variant}' 2>/dev/null || true)
  if [ -n "$live" ]; then
    echo "  registered engines (on lmcache-$live):"
    kubectl exec -n "$NS" "deploy/$(lmcache_of "$live")" -- \
      python3 -c "import urllib.request,json;print('   ',json.load(urllib.request.urlopen('http://localhost:8080/status')).get('registered_gpu_ids'))" \
      2>/dev/null || echo "    (unreachable)"
    echo "  NOTE: dead registrations are never reaped (worker-reap-timeout-seconds 0),"
    echo "  so this list accumulates across engine restarts of the SAME variant."
  fi
}

TARGET="$1"
case "$TARGET" in
  apply)
    # Record before applying: the apply is what destroys this information.
    LIVE=$(live_variant)
    if [ -z "$LIVE" ]; then
      LIVE=$(kubectl get deploy -n "$NS" -o jsonpath='{range .items[*]}{.metadata.labels.app}{" "}{.spec.replicas}{"\n"}{end}' 2>/dev/null \
             | awk '$1=="vllm-nemotron-bf16" && $2>0 {print "bf16"} $1=="vllm-nemotron-nvfp4" && $2>0 {print "nvfp4"}' | head -1)
    fi
    LIVE="${LIVE:-nvfp4}"
    echo "==> live variant before apply: $LIVE"
    kubectl apply -f "$HERE/common/" -f "$HERE/nvfp4/" -f "$HERE/bf16/"
    echo "==> restoring $LIVE"
    # "$HERE/switch.sh", not "$0": we cd'd to $HERE above, so a relative $0 like
    # ./nemotron/switch.sh no longer resolves from here.
    exec "$HERE/switch.sh" "$LIVE" ;;
  status) status; exit 0 ;;
  nvfp4)  OTHER=bf16  ;;
  bf16)   OTHER=nvfp4 ;;
  *)      usage ;;
esac

TARGET_DEPLOY=$(deploy_of "$TARGET")
OTHER_DEPLOY=$(deploy_of "$OTHER")
TARGET_LMCACHE=$(lmcache_of "$TARGET")
OTHER_LMCACHE=$(lmcache_of "$OTHER")

# Idempotent: re-running for the variant that is already up is a no-op that still
# verifies, rather than an unnecessary ~6 min reload.
if [ "$(kubectl get deploy "$TARGET_DEPLOY"  -n "$NS" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)" = "1" ] \
   && [ "$(kubectl get deploy "$TARGET_LMCACHE" -n "$NS" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)" = "1" ] \
   && [ "$(kubectl get deploy "$OTHER_DEPLOY"   -n "$NS" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)" = "0" ] \
   && [ "$(kubectl get deploy "$OTHER_LMCACHE"  -n "$NS" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)" = "0" ]; then
  echo "$TARGET is already live; verifying only."
else
  # Both engines go down, not just the outgoing one. The target may already be
  # running against a server that has since been restarted or scaled away -- its
  # GPU-context registration died with that server and an engine only registers
  # at startup, so it has to be recycled even though it looks healthy. Serving
  # /health 200 while silently doing no KV offload is the failure this prevents.
  echo "==> scaling down $OTHER_DEPLOY and $TARGET_DEPLOY"
  kubectl scale "deploy/$TARGET_DEPLOY" -n "$NS" --replicas=0
  kubectl wait --for=delete pod -l "app=$TARGET_DEPLOY" -n "$NS" --timeout=300s 2>/dev/null || true
  kubectl scale "deploy/$OTHER_DEPLOY" -n "$NS" --replicas=0
  # Wait for the pod to be GONE, not just for the Deployment to report 0. The
  # GPU is not released until the container actually exits, and the incoming pod
  # stays Pending until it is.
  kubectl wait --for=delete pod -l "app=$OTHER_DEPLOY" -n "$NS" --timeout=300s 2>/dev/null || true

  # The outgoing server goes down before the incoming one comes up: both carry
  # `serving: lmcache`, and two endpoints behind the stable Service would let the
  # ZMQ connector pick one arbitrarily -- with the wrong --chunk-size behind it.
  # This also frees its pinned-DRAM L1 before the new engine allocates weights,
  # and clears its stale GPU-context registrations (the reaper is disabled).
  echo "==> scaling down $OTHER_LMCACHE"
  kubectl scale "deploy/$OTHER_LMCACHE" -n "$NS" --replicas=0 2>/dev/null || true
  kubectl wait --for=delete pod -l "app=$OTHER_LMCACHE" -n "$NS" --timeout=120s 2>/dev/null || true

  # Server BEFORE engine: the engine's REGISTER_KV_CACHE runs at startup and
  # fails if there is nothing listening on tcp://lmcache:5555.
  echo "==> scaling up $TARGET_LMCACHE (--chunk-size must match the engine's block size)"
  kubectl scale "deploy/$TARGET_LMCACHE" -n "$NS" --replicas=1
  kubectl rollout status "deploy/$TARGET_LMCACHE" -n "$NS" --timeout=180s

  # The engine's startupProbe is what makes this wait mean something: without it
  # the container is Ready as soon as it starts, rollout status returns in
  # seconds, and the Service takes an endpoint while vLLM is still loading
  # weights -- which is exactly how the public URL ends up serving 503s.
  echo "==> scaling up $TARGET_DEPLOY (checkpoint load: nvfp4 ~103s, bf16 ~334s)"
  kubectl scale "deploy/$TARGET_DEPLOY" -n "$NS" --replicas=1
  kubectl rollout status "deploy/$TARGET_DEPLOY" -n "$NS" --timeout="$(timeout_of "$TARGET")"
fi

echo "==> verifying"
eps=$(kubectl get endpoints vllm-nemotron-active -n "$NS" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || true)
[ -n "$eps" ] || { echo "FAIL: vllm-nemotron-active has no endpoints" >&2; exit 1; }

# Exactly one lmcache pod, and it must be the target's. More than one means the
# stable Service has two endpoints with different chunk sizes behind them.
n=$(kubectl get pods -n "$NS" -l serving=lmcache --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "1" ] || { echo "FAIL: expected exactly 1 running lmcache pod, found $n" >&2; exit 1; }
v=$(kubectl get pods -n "$NS" -l serving=lmcache --field-selector=status.phase=Running \
      -o jsonpath='{.items[0].metadata.labels.variant}')
[ "$v" = "$TARGET" ] || { echo "FAIL: lmcache pod is variant '$v', expected '$TARGET'" >&2; exit 1; }

echo "--- lmcache registrations ---"
kubectl exec -n "$NS" "deploy/$TARGET_LMCACHE" -- python3 -c \
  "import urllib.request,json;print('   ',json.load(urllib.request.urlopen('http://localhost:8080/status')).get('registered_gpu_ids'))" \
  2>/dev/null || echo "    (unreachable)"

KEY=$(kubectl get secret vllm-auth -n "$NS" -o jsonpath='{.data.api-key}' | base64 -d)
echo "--- https://nemo35-lightning.krishb.in/v1/models ---"
curl -fsS https://nemo35-lightning.krishb.in/v1/models -H "Authorization: Bearer $KEY" \
  | python3 -c 'import json,sys; [print("   ", m["id"]) for m in json.load(sys.stdin)["data"]]'

echo
echo "$TARGET is live."
