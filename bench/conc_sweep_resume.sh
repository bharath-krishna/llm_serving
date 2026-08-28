#!/usr/bin/env bash
# Resume sweep — levels 1/4/8 already collected (see BENCHMARK-HANDOFF.md §5).
# 12 is new, inserted to localise the crash threshold between known-good 8 and
# known-crashing 16.
#
# Every level writes a MARKER with dd oflag=dsync before it starts, so an
# instant SoC reset cannot erase which level was running and when.
set -u
BASE_URL="http://vllm-nemotron.vllm.svc.cluster.local:8000"
MODEL="nemotron-3.5-lightning"
TOK="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
OUT=/results
mkdir -p "$OUT"

IN_LEN=16000
OUT_LEN=600
LEVELS=(12 16 24 32 48 64)

mark() { echo "$(date -u +%FT%T.%3NZ) $*" | dd of="$OUT/MARKERS.txt" oflag=append conv=notrunc,fsync status=none; }
touch "$OUT/MARKERS.txt"

[ -s "$OUT/summary.csv" ] || \
echo "level,concurrency,n_prompts,duration_s,req_per_s,out_tok_per_s,total_tok_per_s,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,itl_p99_ms,e2e_p50_s,e2e_p99_s" > "$OUT/summary.csv"

mark "SWEEP-START levels=${LEVELS[*]}"
curl -s "$BASE_URL/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1
mark "WARMUP-DONE"

for N in "${LEVELS[@]}"; do
  NP=$(( N * 6 )); [ "$NP" -lt 48 ] && NP=48; [ "$NP" -gt 240 ] && NP=240
  mark "LEVEL-BEGIN N=$N num_prompts=$NP"
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
  sync
  mark "LEVEL-END N=$N rc=$?"

  python3 - "$N" "$NP" "$OUT/conc-$N.json" "$OUT/summary.csv" <<'PY'
import json,sys,os
N,NP,path,csv=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
try: d=json.load(open(path))
except Exception as e: print("  !! no result json:",e); sys.exit(0)
g=lambda k: d.get(k,"")
row=[N,N,NP,round(g("duration"),1),round(g("request_throughput"),3),
     round(g("output_throughput"),1),round(g("total_token_throughput"),1),
     round(g("p50_ttft_ms"),1),round(g("p90_ttft_ms"),1),round(g("p99_ttft_ms"),1),
     round(g("p50_tpot_ms"),1),round(g("p99_tpot_ms"),1),round(g("p99_itl_ms"),1),
     round(g("p50_e2el_ms")/1000,2),round(g("p99_e2el_ms")/1000,2)]
fd=os.open(csv,os.O_WRONLY|os.O_APPEND|os.O_SYNC)
os.write(fd,(",".join(str(x) for x in row)+"\n").encode()); os.close(fd)
print("  ->",row)
PY
  sync
done
mark "SWEEP-DONE"
echo "=== DONE — $(date -u +%H:%M:%S) ==="
column -s, -t "$OUT/summary.csv"
