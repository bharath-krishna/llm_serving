#!/usr/bin/env bash
# Single `lmcache bench engine` run against the BF16 variant
# (Deployment/Service `vllm-nemotron-bf16`).
#
#   ./run.sh                          # smoke  --  15 requests, nothing evicts
#   ./run.sh bench_config_sweep.json  # one level of the sweep shape (27 requests)
#   ./run.sh bench_config_mrc.json    # multi-round-chat, 8 concurrent sessions
#   ./run.sh bench_config.json        # GUARDED -- see below
#
# TWO CONFIGS ARE REFUSED OUTRIGHT on this variant: bench_config.json and
# bench_config_permutator.json. Both size themselves from constants that have
# never been measured for BF16 -- tokens_per_gb_kvcache and LMCache's
# bytes/token -- and getting that arithmetic wrong is what hard-reset the node on
# 2026-08-30. Run the smoke config first, read the real numbers off the engine,
# fill them in, then remove the guard. ../README.md step 3 is the procedure.
#
# The heavy config is the load profile that has hard-reset this node twice; it
# also prompts before launching. Run ./sweep.sh for the concurrency sweep instead of
# calling this in a loop -- a loop restarts the client between levels but this
# script also re-creates the pod, and the extra teardown shows up in the numbers.
#
# The NVFP4 variant has its own copy of this script under ../../nvfp4/bench/ with
# its own pod, ConfigMap and constants. They are deliberately separate objects so
# both can exist on the cluster at once and a result is never misattributed.
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-bench_config_smoke.json}"
NS=vllm

# ─── variant identity ─── the only lines that differ from the BF16 copy ───
VARIANT=bf16
POD=vllm-bench-bf16
CM=vllm-bench-config-bf16
# MEASURED 2026-08-31 from the registered engine (lmcache /status ->
# cache_context_meta.cache_size_per_token). NVFP4's is 11,799, so this is 2.52x
# -- NOT the 1.96x the KV-dtype ratio alone predicts, because LMCache stores the
# mamba recurrent state as well as attention KV (29 layers in the reported
# layout: 23 mamba + 6 attention). An earlier derivation guessed 23,126 and was
# 22% low; do not re-derive this, read it.
BYTES_PER_TOKEN=29696
# --l1-size-gb from ../lmcache.yaml. 6 is a derived floor, not a comfort setting:
# L1 must hold more tokens than the 174,736-token GPU KV pool it backs or nothing
# it stores can ever be a hit. 6 GiB / 29,696 = 216,946 tokens = 1.24x the pool.
# Both moved together when max-model-len went 65536 -> 131072 on 2026-08-31; see
# the sizing table in ../README.md. The offload budget is derived from this rather
# than hardcoded, so the guard follows the tier it is guarding.
L1_SIZE_GB=6

# Configs whose OFFLOAD VOLUME exceeds what this L1 can absorb, which is the
# regime that sustains /kv-l2 NVMe writeback and has hard-reset this node.
# The constants above are now measured, so these are engineering limits rather
# than unknowns -- each config's _comment carries the arithmetic and what would
# have to change to unguard it.
#
#   bench_config.json  see its _comment -- raising max-model-len to 131072 made
#                      the eviction regime UNREACHABLE on this box, not merely
#                      expensive, so it is guarded for a new reason
#   bench_config_mrc   312,080 tok x 29,696 = 9.27 GB (1.44x the 6 GiB L1) --
#                      needs num_concurrent_users and mrc_duration re-derived
GUARDED_CONFIGS="bench_config.json bench_config_mrc.json"
# ────────────────────────────────────────────────────────────────────────────

[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$CFG"

case " $GUARDED_CONFIGS " in
  *" $CFG "*)
    cat >&2 <<EOF
REFUSING $CFG on the BF16 variant.

Its offload volume exceeds what the ${L1_SIZE_GB} GiB L1 can absorb at the measured
${BYTES_PER_TOKEN} B/token, which is the regime that sustains /kv-l2 NVMe writeback and
has hard-reset this node (BENCHMARK-HANDOFF.md section 3). This is a limit, not
a missing measurement -- see that config's _comment for the arithmetic and for
what would have to change.

  ./run.sh bench_config_smoke.json     # 18 requests, 2.85 GB offload, 66% of L1

To override deliberately: start ../../tools/start-telemetry.sh on the node, then
remove the config from GUARDED_CONFIGS in this script.
EOF
    exit 1 ;;
esac

# Offload-budget preflight. The binding limit on this box is NOT the GPU KV pool
# but LMCache's L1 pinned-DRAM tier: past it, chunks flush to the /kv-l2 NVMe
# tier continuously and the dirty pages + pinned CUDA-IPC buffers grow the shared
# 128 GiB unified pool until a driver allocation fails, which HARD-RESETS the
# node. N=5 C=20000 P=20 (23.83 GB) did exactly that on 2026-08-30. Refuse
# anything over 75% of L1 unless the caller explicitly overrides.
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
  echo "WARNING: bench_config.json is the eviction-forcing 24-request run."
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
# bench-pod.yaml carries the literal token __CONFIG_NAME__, which is not a valid
# config name. Substituting a placeholder rather than a real default is what
# keeps this honest: this sed used to look for whatever config happened to be
# pinned in the pod spec, so editing that default silently broke the
# substitution and every run executed the pinned config regardless of $CFG.
grep -q '__CONFIG_NAME__' bench-pod.yaml || {
  echo "bench-pod.yaml lost its __CONFIG_NAME__ placeholder -- refusing to run a" >&2
  echo "config other than the one you asked for. Restore the placeholder." >&2
  exit 1
}
sed "s|__CONFIG_NAME__|$CFG|" bench-pod.yaml | kubectl apply -f -

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
