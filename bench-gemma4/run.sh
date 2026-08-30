#!/usr/bin/env bash
# Usage: ./run.sh [bench_config.json|bench_config_mrc.json]
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-bench_config.json}"
NS=vllm

kubectl create configmap gemma4-bench-config -n "$NS" \
  --from-file=bench_config.json --from-file=bench_config_mrc.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl delete pod gemma4-bench -n "$NS" --ignore-not-found --wait=true
sed "s|value: bench_config.json|value: $CFG|" bench-pod.yaml | kubectl apply -f -

kubectl wait --for=condition=Ready pod/gemma4-bench -n "$NS" --timeout=180s
kubectl logs -n "$NS" -f gemma4-bench
