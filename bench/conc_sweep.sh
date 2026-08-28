#!/usr/bin/env bash
# Concurrency sweep for vllm-nemotron — run INSIDE the bench pod (see bench-pod.yaml).
#
# Simulates N simultaneously-active coding-agent users (request-rate = inf, so
# exactly N requests are in flight at all times) and steps N up until latency
# falls apart.
#   --random-prefix-len 2128 models ONE shared system prompt that every user
#   sends (vLLM prefix-caches it after the first request), so this is close to
#   real agent traffic. The per-request body still varies 8k-24k tokens with no
#   reuse, so it stays a conservative estimate — real sessions reuse far more.
#
# Output: /results/summary.csv  (one row per level) + /results/conc-<N>.{json,log}
set -u
BASE_URL="http://vllm-nemotron.vllm.svc.cluster.local:8000"
MODEL="nemotron-3.5-lightning"
TOK="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
OUT=/results
mkdir -p "$OUT"

IN_LEN=16000          # mean prompt tokens (varies 8k-24k via range-ratio 0.5)
OUT_LEN=600           # generated tokens per request
LEVELS=(1 4 8 16 24 32 48 64)

echo "level,concurrency,n_prompts,duration_s,req_per_s,out_tok_per_s,total_tok_per_s,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,itl_p99_ms,e2e_p50_s,e2e_p99_s" > "$OUT/summary.csv"

# tiny warmup so the first timed level isn't paying cold graph-capture cost
curl -s "$BASE_URL/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1

for N in "${LEVELS[@]}"; do
  NP=$(( N * 6 )); [ "$NP" -lt 48 ] && NP=48; [ "$NP" -gt 240 ] && NP=240
  echo "=== concurrency $N  (num-prompts $NP) — $(date -u +%H:%M:%S) ==="
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$TOK" \
    --dataset-name random \
    --random-input-len "$IN_LEN" \
    --random-output-len "$OUT_LEN" \
    --random-range-ratio 0.5 \
    --random-prefix-len 2128 \
    --seed 42 \
    --num-prompts "$NP" \
    --max-concurrency "$N" \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "conc-$N.json" \
    2>&1 | tee "$OUT/conc-$N.log"

  python3 - "$N" "$NP" "$OUT/conc-$N.json" "$OUT/summary.csv" <<'PY'
import json,sys
N,NP,path,csv=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
d=json.load(open(path))
g=lambda k: d.get(k,"")
row=[N,N,NP,round(g("duration"),1),round(g("request_throughput"),3),
     round(g("output_throughput"),1),round(g("total_token_throughput"),1),
     round(g("p50_ttft_ms"),1),round(g("p90_ttft_ms"),1),round(g("p99_ttft_ms"),1),
     round(g("p50_tpot_ms"),1),round(g("p99_tpot_ms"),1),round(g("p99_itl_ms"),1),
     round(g("p50_e2el_ms")/1000,2),round(g("p99_e2el_ms")/1000,2)]
open(csv,"a").write(",".join(str(x) for x in row)+"\n")
print("  ->",row)
PY
done

echo "=== DONE — $(date -u +%H:%M:%S) ==="
column -s, -t "$OUT/summary.csv"
