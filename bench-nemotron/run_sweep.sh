#!/usr/bin/env bash
# Host-side orchestrator: memory guard + launch the in-pod concurrency sweep.
#
#   ./run_sweep.sh
#
# Prereqs:
#   - `kubectl apply -f bench-pod.yaml` and pod Ready
#   - conc_sweep.sh sitting next to this file
#   - vllm-nemotron running with the CAPPED config (see BENCHMARK-HANDOFF.md)
#   - ollama stopped:  sudo systemctl disable --now ollama
#
# NOTE (2026-08-28): on the DGX Spark node `spark-45f7` the sweep hard-crashes
# the machine at concurrency 16, with memory dead-flat at ~67 GB available and
# the memguard never tripping. Believed to be a GPU/SoC firmware fault, not a
# resource problem. See BENCHMARK-HANDOFF.md. The memory guard below is kept
# anyway; it does not and cannot stop that class of crash.
set -u
NS=vllm
GUARD_FLOOR_MB=13000
HERE="$(cd "$(dirname "$0")" && pwd)"
ABORT="$HERE/SWEEP_ABORT"
RESDIR="$HERE/bench_results"
rm -f "$ABORT"; mkdir -p "$RESDIR"

memguard() {
  while :; do
    [ -f "$ABORT" ] && return 0
    MA=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "${MA:-0}" -lt "$GUARD_FLOOR_MB" ]; then
      echo "[memguard] MemAvailable ${MA}MB < ${GUARD_FLOOR_MB}MB — ABORTING" | tee -a "$RESDIR/guard.log"
      touch "$ABORT"
      kubectl -n "$NS" exec vllm-bench -- pkill -9 -f "vllm bench serve" 2>/dev/null
      kubectl -n "$NS" scale deploy/vllm-nemotron --replicas=0 2>&1 | tee -a "$RESDIR/guard.log"
      return 1
    fi
    echo "$(date -u +%H:%M:%S) MemAvailable ${MA}MB" >> "$RESDIR/guard.log"
    sleep 5
  done
}
memguard &
GUARD_PID=$!

echo "[run] launching sweep in pod vllm-bench …"
kubectl -n "$NS" cp "$HERE/conc_sweep.sh" vllm-bench:/root/conc_sweep.sh
kubectl -n "$NS" exec vllm-bench -- chmod +x /root/conc_sweep.sh
kubectl -n "$NS" exec vllm-bench -- bash -lc '/root/conc_sweep.sh' 2>&1 | tee "$RESDIR/sweep.out"
RC=${PIPESTATUS[0]}

kill "$GUARD_PID" 2>/dev/null
kubectl -n "$NS" cp vllm-bench:/results "$RESDIR/pod-results" 2>/dev/null

if [ -f "$ABORT" ]; then
  echo "[run] sweep ABORTED by memory guard. vLLM scaled to 0 — bring it back with:"
  echo "      kubectl -n $NS scale deploy/vllm-nemotron --replicas=1"
  exit 2
fi
echo "[run] sweep finished rc=$RC. summary:"
column -s, -t "$RESDIR/pod-results/summary.csv" 2>/dev/null
