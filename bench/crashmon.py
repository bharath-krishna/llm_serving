#!/usr/bin/env python3
"""Crash forensics collector for spark-45f7 (DGX Spark GB10).

Every sample is written with O_SYNC so it reaches NVMe before the next one is
taken. An instant SoC reset loses the page cache -- that is why previous crash
logs ended mid-line with nothing useful in them.
"""
import os, sys, time, subprocess, signal

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/bharath/bench-monitor"
os.makedirs(OUT, exist_ok=True)

fd = os.open(os.path.join(OUT, "sysmon.csv"),
             os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_SYNC, 0o644)

ZONES = sorted(p for p in os.listdir("/sys/class/thermal") if p.startswith("thermal_zone"))
SMI_Q = ("timestamp,temperature.gpu,utilization.gpu,utilization.memory,"
         "clocks.sm,power.draw,pstate,clocks_event_reasons.active")

HDR = ("wall,mono,gpu_temp_c,gpu_util,mem_util,sm_mhz,power_w,pstate,throttle,"
       "mem_avail_kb,mem_free_kb,committed_kb,dirty_kb,writeback_kb,"
       "psi_cpu_avg10,psi_mem_avg10,psi_io_avg10,load1,nvme_temp_c,"
       + ",".join(f"tz{z[12:]}_c" for z in ZONES) + "\n")

def w(line):
    os.write(fd, line.encode())

def readf(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return default

def psi(path):
    """some avg10 -- fraction of time at least one task was stalled."""
    for ln in readf(path).splitlines():
        if ln.startswith("some"):
            for tok in ln.split():
                if tok.startswith("avg10="):
                    return tok[6:]
    return ""

def smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_Q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        f = [x.strip() for x in out.split(",")]
        return f[1:] if len(f) >= 8 else [""] * 7
    except Exception:
        return [""] * 7

def nvme_temp():
    for h in sorted(os.listdir("/sys/class/hwmon")):
        if readf(f"/sys/class/hwmon/{h}/name") == "nvme":
            v = readf(f"/sys/class/hwmon/{h}/temp1_input")
            return str(int(v) // 1000) if v.isdigit() else ""
    return ""

if os.lseek(fd, 0, os.SEEK_END) == 0:
    w(HDR)

w(f"# start {time.strftime('%F %T %Z')} boot_id={readf('/proc/sys/kernel/random/boot_id')}\n")

running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

while running:
    t0 = time.time()
    g = smi()
    mi = {}
    for ln in readf("/proc/meminfo").splitlines():
        k, _, v = ln.partition(":")
        mi[k] = v.strip().split()[0] if v.strip() else ""
    tz = [str(int(readf(f"/sys/class/thermal/{z}/temp", "0")) // 1000) for z in ZONES]
    row = [time.strftime("%F %T"), f"{time.monotonic():.1f}"] + g + [
        mi.get("MemAvailable", ""), mi.get("MemFree", ""), mi.get("Committed_AS", ""),
        mi.get("Dirty", ""), mi.get("Writeback", ""),
        psi("/proc/pressure/cpu"), psi("/proc/pressure/memory"), psi("/proc/pressure/io"),
        readf("/proc/loadavg").split()[0] if readf("/proc/loadavg") else "",
        nvme_temp()] + tz
    w(",".join(x.replace(",", ";") for x in row) + "\n")
    time.sleep(max(0.0, 1.0 - (time.time() - t0)))

w(f"# clean stop {time.strftime('%F %T %Z')}\n")
os.close(fd)
