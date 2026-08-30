#!/usr/bin/env bash
# launcher kept in a file so the ssh command line never contains the pkill pattern
pkill -f "gpumon\.py" 2>/dev/null
sleep 1
cd /home/bharath
setsid nohup python3 /home/bharath/gpumon.py /home/bharath/bench-monitor \
  >/tmp/gpumon.out 2>&1 </dev/null &
sleep 4
echo "pid: $(pgrep -f 'gpumon\.py' | tr '\n' ' ')"
echo "stderr: $(cat /tmp/gpumon.out 2>&1 | head -5)"
echo "rows: $(wc -l < /home/bharath/bench-monitor/gpu10hz.csv 2>/dev/null || echo MISSING)"
tail -3 /home/bharath/bench-monitor/gpu10hz.csv 2>/dev/null
