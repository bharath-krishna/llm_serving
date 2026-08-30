#!/usr/bin/env bash
# Single `lmcache bench engine` run against the nemotron-3.5-lightning deployment.
#
#   ./run.sh                          # smoke  --  15 requests, nothing evicts
#   ./run.sh bench_config_sweep.json  # one level of the sweep shape (27 requests)
#   ./run.sh bench_config_mrc.json    # multi-round-chat, 8 concurrent sessions
#   ./run.sh bench_config.json        # HEAVY -- 978 requests, forces KV eviction
#
# The heavy config is the load profile that has hard-reset this node twice; it
# prompts before launching. Run ./sweep.sh for the concurrency sweep instead of
# calling this in a loop -- a loop restarts the client between levels but this
# script also re-creates the pod, and the extra teardown shows up in the numbers.
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-bench_config_smoke.json}"
NS=vllm
POD=vllm-bench

[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$CFG"

if [ "$CFG" = "bench_config.json" ]; then
  echo "WARNING: bench_config.json is the eviction-forcing 978-request run."
  echo "Sustained long-context load hard-reset this node on 2026-08-27 and"
  echo "2026-08-30T07:04:05Z. Start telemetry first (./start-telemetry.sh on the node)."
  read -r -p "type 'yes' to continue: " ok
  [ "$ok" = "yes" ] || { echo "aborted"; exit 1; }
fi

# Rebuild the ConfigMap every time: editing a config file changes nothing until
# it is re-created, and a stale ConfigMap silently benchmarks the OLD settings.
kubectl create configmap vllm-bench-config -n "$NS" \
  --from-file=bench_config.json \
  --from-file=bench_config_smoke.json \
  --from-file=bench_config_mrc.json \
  --from-file=bench_config_sweep.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl delete pod "$POD" -n "$NS" --ignore-not-found --wait=true
sed "s|value: bench_config_smoke.json|value: $CFG|" bench-pod.yaml | kubectl apply -f -

kubectl wait --for=condition=Ready "pod/$POD" -n "$NS" --timeout=180s
kubectl logs -n "$NS" -f "$POD"

# `kubectl logs -f` returns 0 even when the container failed, so read the real
# exit code back off the pod rather than trusting the pipeline.
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
[ "$phase" = "Succeeded" ]
