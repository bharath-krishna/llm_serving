#!/usr/bin/env bash
# Single `lmcache bench engine` run against the NVFP4 variant
# (Deployment/Service `vllm-nemotron-nvfp4`).
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
#
# The BF16 variant has its own copy of this script under ../../bf16/bench/ with
# its own pod, ConfigMap and constants. They are deliberately separate objects so
# both can exist on the cluster at once and a result is never misattributed.
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-bench_config_smoke.json}"
NS=vllm

# ─── variant identity ─── the only lines that differ from the BF16 copy ───
VARIANT=nvfp4
POD=vllm-bench-nvfp4
CM=vllm-bench-config-nvfp4
# LMCache's own measured cache_size_per_token for this model at world_size=1.
# It prints it every run ("-> 91002 tokens/GB"); this is that number, not an
# estimate. It is dtype-derived, so it does NOT carry to BF16.
BYTES_PER_TOKEN=11799
# --l1-size-gb from ../../common/lmcache.yaml. The offload budget is derived
# from it below rather than hardcoded, so the guard follows the tier it is
# guarding instead of drifting from it.
L1_SIZE_GB=4
# ────────────────────────────────────────────────────────────────────────────

[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$CFG"

# Offload-budget preflight. The binding limit on this box is NOT the GPU KV pool
# but LMCache's L1 pinned-DRAM tier: past it, chunks flush to the /kv-l2 NVMe
# tier continuously and the dirty pages + pinned CUDA-IPC buffers grow the shared
# 128 GiB unified pool until a driver allocation fails, which HARD-RESETS the
# node. N=5 C=20000 P=20 (23.83 GB) did exactly that on 2026-08-30.
#
# THE FIGURE BELOW COUNTS TOTAL PROMPT VOLUME, NOT UNIQUE KEYS, AND THAT IS
# DELIBERATE. On 2026-08-31 this guard refused a 128k-token x 2-permutation run
# at 7.60 GB and was overridden with BUDGET_GB=8, on the reasoning that
# lexicographic permutations share a long prefix so only ~5 GB of NEW keys would
# be stored. The node hard-reset ~7 minutes later. Two reasons that reasoning
# fails, both of which this guard is deliberately blind to in the safe direction:
#
#   * LMCache writes THROUGH to the L2 fs adapter, not only on L1 eviction, so
#     dedup reduces unique keys but NOT NVMe traffic. Measured on that run:
#     /kv-l2/bf16 grew 9.0 -> 15 GB in about 90 seconds.
#   * A shared prefix must be LOADED back (~86,000 tokens there) while the moved
#     suffix is stored. Concurrent bulk load+store is what multiplies the pinned
#     CUDA-IPC buffers -- the acute half of the mechanism. A higher hit rate makes
#     this workload MORE dangerous, not less.
#
# So: do not override BUDGET_GB on a dedup argument. Shrink the workload instead.
BUDGET_GB="${BUDGET_GB:-$(python3 -c "print(f'{$L1_SIZE_GB * 0.75:.1f}')")}"
python3 - "$CFG" "$BUDGET_GB" "$BYTES_PER_TOKEN" "$L1_SIZE_GB" <<'PREFLIGHT' || exit 1
import json, sys
cfg = json.load(open(sys.argv[1]))
budget, bytes_per_token, l1 = float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
if cfg.get("workload") != "long-doc-permutator":
    sys.exit(0)
prompt = cfg["ldp_system_prompt_length"] + cfg["ldp_num_contexts"] * cfg["ldp_context_length"]
total_gb = cfg["ldp_num_permutations"] * prompt * bytes_per_token / 1e9
print(f"  preflight: {prompt:,} tok/request, {total_gb:.2f} GB total offload "
      f"(budget {budget:.1f} GB, L1 is {l1:.1f} GB)")
if total_gb > budget:
    print(f"  REFUSING: {total_gb:.2f} GB exceeds the {budget:.1f} GB budget. Past the {l1:.1f} GB L1 "
          f"this sustains NVMe writeback and has hard-reset the node.\n"
          f"  Shrink ldp_context_length / ldp_num_permutations, or re-run with "
          f"BUDGET_GB=<n> to override deliberately.", file=sys.stderr)
    sys.exit(1)
PREFLIGHT

if [ "$CFG" = "bench_config.json" ]; then
  echo "WARNING: bench_config.json is the eviction-forcing 978-request run."
  echo "Sustained long-context load hard-reset this node on 2026-08-27 and"
  echo "2026-08-30T07:04:05Z. Start telemetry first (../../tools/start-telemetry.sh on the node)."
  read -r -p "type 'yes' to continue: " ok
  [ "$ok" = "yes" ] || { echo "aborted"; exit 1; }
fi

# Rebuild the ConfigMap every time: editing a config file changes nothing until
# it is re-created, and a stale ConfigMap silently benchmarks the OLD settings.
kubectl create configmap "$CM" -n "$NS" \
  --from-file=bench_config.json \
  --from-file=bench_config_smoke.json \
  --from-file=bench_config_mrc.json \
  --from-file=bench_config_sweep.json \
  --from-file=bench_config_permutator.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl delete pod "$POD" -n "$NS" --ignore-not-found --wait=true
# Select the config by substituting a placeholder that cannot be a valid config
# name, then VERIFY THE RENDERED SPEC before applying it.
#
# Both halves of this are load-bearing, and both were learned the hard way on
# 2026-08-31, when three consecutive runs silently executed bench_config.json --
# the heavy, GUARDED config -- no matter what was passed on the command line:
#
#   * The value line had drifted back to a real config name, so the sed matched
#     nothing and the pod ran whatever was pinned in the file. An earlier version
#     of this script sed'd for a real default name and had the same failure mode.
#   * The guard that was supposed to catch that grepped the WHOLE FILE for
#     __CONFIG_NAME__ -- and matched the explanatory comment sitting above the
#     value line. A guard satisfied by its own documentation is not a guard.
#   * Nothing checked the rendered output, so GUARDED_CONFIGS was validating the
#     command-line argument rather than the config the pod would actually run.
#
# Hence: anchor to the value line, and diff the render against what was asked for.
grep -qE '^[[:space:]]+value: __CONFIG_NAME__[[:space:]]*$' bench-pod.yaml || {
  echo "bench-pod.yaml has no 'value: __CONFIG_NAME__' line -- refusing to run, because" >&2
  echo "without it the pod silently executes whatever config is pinned in that file" >&2
  echo "rather than the one you asked for. Restore the placeholder." >&2
  exit 1
}
RENDERED=$(sed "s|__CONFIG_NAME__|$CFG|" bench-pod.yaml)
printf '%s\n' "$RENDERED" | grep -qE "^[[:space:]]+value: ${CFG//./\\.}[[:space:]]*$" || {
  echo "rendered pod spec does not select $CFG -- refusing." >&2
  exit 1
}
printf '%s\n' "$RENDERED" | kubectl apply -f -

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
echo "=== pod phase: $phase  (results under /var/lib/vllm-bench-results/$VARIANT/) ==="
[ "$phase" = "Succeeded" ]
