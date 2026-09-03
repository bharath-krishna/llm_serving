#!/usr/bin/env bash
# Run as root on spark-45f7. Harvests evidence from past crashes, arms
# instrumentation for the next one. Idempotent -- safe to re-run.
set -u
OUT=/home/bharath/bench-monitor
MAC_IP=192.168.1.5          # collector (Bharath's laptop)
NETCON_PORT=6666
mkdir -p "$OUT"/{pstore,journal}

echo "### 1. pstore -- crash records that survive reboot"
ls -la /sys/fs/pstore/ 2>&1 | sed 's/^/    /'
if compgen -G "/sys/fs/pstore/*" >/dev/null; then
    cp -av /sys/fs/pstore/* "$OUT/pstore/" 2>&1 | sed 's/^/    /'
    echo "    ^^^ RECORDS FOUND AND SAVED"
else
    echo "    (empty -- no panic was ever recorded, consistent with silent reset)"
fi

echo
echo "### 2. rasdaemon persistent error DB"
DB=/var/lib/rasdaemon/ras-mc_event.db
if [ -f "$DB" ]; then
    ras-mc-ctl --summary  > "$OUT/ras-summary.txt" 2>&1
    ras-mc-ctl --errors   > "$OUT/ras-errors.txt"  2>&1
    sed 's/^/    /' "$OUT/ras-summary.txt"
    echo "    (full errors -> $OUT/ras-errors.txt, $(wc -l < "$OUT/ras-errors.txt") lines)"
else
    echo "    no DB at $DB"
fi

echo
echo "### 3. tail of every crashed boot (what the kernel printed last)"
for b in -1 -2 -3 -4 -5; do
    f="$OUT/journal/boot${b}.txt"
    journalctl -b "$b" -o short-precise --no-pager > "$f" 2>/dev/null || continue
    echo "  --- boot $b: last 6 lines before it died ---"
    tail -6 "$f" | sed 's/^/      /'
done
journalctl -k -b -1 --no-pager > "$OUT/journal/boot-1-kernel.txt" 2>/dev/null
echo "  (full journals -> $OUT/journal/)"

echo
echo "### 4. netconsole -> $MAC_IP:$NETCON_PORT"
modprobe netconsole 2>/dev/null
DEV=$(ip route get "$MAC_IP" 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
ip route get "$MAC_IP" >/dev/null 2>&1 && ping -c1 -W2 "$MAC_IP" >/dev/null 2>&1
TGTMAC=$(ip neigh show "$MAC_IP" | grep -oE '([0-9a-f]{2}:){5}[0-9a-f]{2}' | head -1)
SRCIP=$(ip -o -4 addr show "$DEV" | grep -oP 'inet \K[0-9.]+' | head -1)
echo "    dev=$DEV src=$SRCIP tgt=$MAC_IP tgtmac=${TGTMAC:-UNRESOLVED}"
if [ -n "${TGTMAC:-}" ] && [ -n "$DEV" ]; then
    D=/sys/kernel/config/netconsole/bench
    mountpoint -q /sys/kernel/config || mount -t configfs none /sys/kernel/config 2>/dev/null
    [ -d "$D" ] && { echo 0 > "$D/enabled" 2>/dev/null; rmdir "$D" 2>/dev/null; }
    if mkdir -p "$D" 2>/dev/null; then
        echo "$DEV"          > "$D/dev_name"
        echo "$SRCIP"        > "$D/local_ip"
        echo "$MAC_IP"       > "$D/remote_ip"
        echo "$NETCON_PORT"  > "$D/remote_port"
        echo "$TGTMAC"       > "$D/remote_mac"
        echo 1 > "$D/extended" 2>/dev/null
        if echo 1 > "$D/enabled" 2>&1; then
            echo "    NETCONSOLE ARMED (dynamic target 'bench')"
        else
            echo "    !! enable failed -- $DEV is WiFi (mt7925e); mac80211 has no netpoll."
            echo "    !! Plug ethernet into enP7s7 and re-run for a reliable kernel-log feed."
        fi
    else
        echo "    !! configfs netconsole unavailable"
    fi
else
    echo "    !! could not resolve target MAC -- is $MAC_IP reachable?"
fi

echo
echo "### 5. reduce log loss on reset (journald sync 1s, verbose printk)"
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/99-crash-forensics.conf <<'CONF'
[Journal]
Storage=persistent
SyncIntervalSec=1s
RateLimitIntervalSec=0
RateLimitBurst=0
CONF
systemctl restart systemd-journald && echo "    journald: SyncIntervalSec=1s, ratelimit off"
echo 8 > /proc/sys/kernel/printk_devkmsg 2>/dev/null
sysctl -qw kernel.printk="8 4 1 8" && echo "    printk verbosity raised"

echo
echo "### 6. GPU clock-lock capability (for the power-transient experiment)"
nvidia-smi -lgc 0,2000 >/dev/null 2>&1 \
  && { echo "    CLOCK LOCK SUPPORTED"; nvidia-smi -rgc >/dev/null 2>&1; echo "    (reset to default)"; } \
  || echo "    clock lock NOT supported on this GB10"

echo
echo "### 7. auto-restart the sampler after a crash-reboot"
cat > /etc/systemd/system/crashmon.service <<'CONF'
[Unit]
Description=DGX Spark crash forensics sampler
After=nvidia-persistenced.service
[Service]
ExecStart=/usr/bin/python3 /home/bharath/crashmon.py /home/bharath/bench-monitor
Restart=always
RestartSec=2
User=bharath
[Install]
WantedBy=multi-user.target
CONF
systemctl daemon-reload
systemctl enable --now crashmon.service && echo "    crashmon.service enabled+started (survives reboots)"

chown -R bharath:bharath "$OUT"
echo
echo "### DONE. Evidence in $OUT"
