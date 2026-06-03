#!/usr/bin/env bash
# Clear stuck read-aloud narration processes and report PulseAudio server health.
# Run as `bash cleanup_audio.sh` so this runner's argv (just the path) cannot be
# matched by the pkill -f patterns below.

echo "=== lingering audio procs ==="
ps -eo pid,etime,args | grep -E 'readaloud|/piper|/paplay' | grep -v grep

echo "=== killing strays ==="
pkill -9 -f readaloud; echo "readaloud rc=$?"
pkill -9 -f /piper;    echo "piper rc=$?"
pkill -9 -f /paplay;   echo "paplay rc=$?"

remaining=$(ps -eo args | grep -E 'readaloud|/piper|/paplay' | grep -v grep | wc -l)
echo "remaining audio strays: $remaining"

echo "=== PulseAudio server (12s timeout) ==="
if timeout 12 pactl info >/tmp/pactl-health.txt 2>&1; then
  grep -E 'Server String|Server Name|Default Sink' /tmp/pactl-health.txt
  echo "RESULT: server reachable"
else
  echo "RESULT: server UNREACHABLE"
  cat /tmp/pactl-health.txt
fi
