#!/usr/bin/env bash
# Concurrency sweep driver. One pod runs every level sequentially so the engine
# is never restarted mid-sweep and the levels stay comparable to each other.
#
#   ./sweep.sh                 # default levels: 1 2 4 8 12
#   ./sweep.sh "1 2 4 8"       # explicit levels
#   ./sweep.sh "16"            # the level that crashed the node -- prompts first
#
# Levels above 32 are pointless: the deployment runs --max-num-seqs 32, so extra
# requests just queue in the client and measure nothing new.
set -euo pipefail
cd "$(dirname "$0")"
LEVELS="${1:-1 2 4 8 12}"
NS=vllm
POD=vllm-sweep

case " $LEVELS " in
  *" 16 "*|*" 24 "*|*" 32 "*)
    echo "WARNING: level >=16 is where the 2026-08-28 sweep hard-reset the node"
    echo "(102 s into N=16, BENCHMARK-HANDOFF.md section 5). Start telemetry first."
    read -r -p "type 'yes' to continue: " ok
    [ "$ok" = "yes" ] || { echo "aborted"; exit 1; }
    ;;
esac

kubectl create configmap vllm-bench-config -n "$NS" \
  --from-file=bench_config.json \
  --from-file=bench_config_smoke.json \
  --from-file=bench_config_mrc.json \
  --from-file=bench_config_sweep.json \
  --from-file=bench_config_permutator.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl delete pod "$POD" -n "$NS" --ignore-not-found --wait=true
sed "s|value: \"1 2 4 8 12\"|value: \"$LEVELS\"|" sweep-pod.yaml | kubectl apply -f -

kubectl wait --for=condition=Ready "pod/$POD" -n "$NS" --timeout=180s
kubectl logs -n "$NS" -f "$POD"

# `kubectl logs -f` returns when the log STREAM closes, which happens before the
# pod is marked terminal -- sampling .status.phase here reads "Running" and
# reports a spurious failure on a clean run. Poll for a terminal phase instead.
phase=""
for _ in $(seq 1 60); do
  phase=$(kubectl get pod "$POD" -n "$NS" -o jsonpath='{.status.phase}')
  case "$phase" in Succeeded|Failed) break ;; esac
  sleep 2
done
echo "=== pod phase: $phase ==="
# A sweep killed by a node reset leaves phase=Failed with exitCode 255 and a
# MARKERS.txt whose last line is a LEVEL-BEGIN with no matching END. That is the
# crash signature, not a bench failure -- check MARKERS.txt before re-running.
[ "$phase" = "Succeeded" ]
