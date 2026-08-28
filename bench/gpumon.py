#!/usr/bin/env python3
"""10 Hz GPU telemetry, O_SYNC'd. Catches power/clock transients that a 1 Hz
sampler averages away. One long-lived nvidia-smi -lms process, so it is cheap."""
import os, sys, subprocess, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/bharath/bench-monitor"
os.makedirs(OUT, exist_ok=True)
fd = os.open(os.path.join(OUT, "gpu10hz.csv"),
             os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_SYNC, 0o644)

Q = ("timestamp,temperature.gpu,utilization.gpu,utilization.memory,clocks.sm,"
     "clocks.gr,power.draw,pstate,clocks_event_reasons.active")
if os.lseek(fd, 0, os.SEEK_END) == 0:
    os.write(fd, (Q + "\n").encode())
os.write(fd, f"# start {time.strftime('%F %T %Z')}\n".encode())

p = subprocess.Popen(
    ["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits", "-lms", "100"],
    stdout=subprocess.PIPE, text=True, bufsize=1)
try:
    for line in p.stdout:
        line = line.strip()
        if line:
            os.write(fd, (line + "\n").encode())
finally:
    p.kill()
    os.close(fd)
