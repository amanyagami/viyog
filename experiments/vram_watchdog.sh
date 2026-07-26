#!/usr/bin/env bash
# Lightweight VRAM/OOM watchdog for the Wave-2 packed run (GPUs 4-7).
# Samples every 30s: per-GPU VRAM%, peak, and scans chain logs for OOM/CUDA errors.
# Writes a rolling status to $OUT and a FLAG file on first OOM detection.
set +e
OUT=/mnt/data1/asing725/viyog/logs/vram_watchdog.log
FLAG=/mnt/data1/asing725/viyog/logs/OOM_DETECTED.flag
MLOG=/mnt/data1/asing725/viyog/logs/matrix
GPUS="4 5 6 7"
declare -A PEAK
for g in $GPUS; do PEAK[$g]=0; done
N=${1:-1200}   # samples (1200 * 30s = 10h)
echo "# watchdog start; samples=$N interval=30s" > "$OUT"
for ((i=0; i<N; i++)); do
  TS=$(date '+%H:%M:%S')
  LINE="$TS"
  for g in $GPUS; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    tot=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$used" ] && used=0; [ -z "$tot" ] && tot=143771
    pct=$(( 100 * used / tot ))
    (( used > PEAK[$g] )) && PEAK[$g]=$used
    LINE="$LINE  g$g=${pct}%(${used}M)"
  done
  LINE="$LINE  peak:"
  for g in $GPUS; do LINE="$LINE g$g=${PEAK[$g]}M"; done
  echo "$LINE" >> "$OUT"
  # scan only RECENTLY-modified chain logs (avoid stale pre-relaunch matches)
  RECENT=$(find "$MLOG" -name '*.log' -mmin -4 2>/dev/null)
  HITS=""
  [ -n "$RECENT" ] && HITS=$(echo "$RECENT" | xargs grep -ilE "out of memory|illegal memory access|device-side assert" 2>/dev/null)
  if [ -n "$HITS" ]; then
    echo "$TS OOM/ERROR in: $HITS" >> "$OUT"
    echo "$TS $HITS" >> "$FLAG"
  fi
  sleep 30
done
echo "# watchdog done" >> "$OUT"
