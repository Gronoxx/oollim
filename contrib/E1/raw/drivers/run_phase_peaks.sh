#!/usr/bin/env bash
# Posterior analysis: separates the two phases of the peak with three readings per run.
#  (1) VmHWM in a pre-hook of block 0, before any attention
#  (2) VmRSS at the same moment
#  (3) VmHWM at the end, after the reset in (1), so the peak of the encoder blocks alone
set -uo pipefail
CLONE="<repo>"; cd "$CLONE"
AFTER="$CLONE/work/wt-after"
REC="$CLONE/work/E1/recordings"; OUT="$CLONE/work/E1/measurements-phases"
mkdir -p "$OUT"; PY="$CLONE/.venv/bin/python"

for enc in phoneticxeus xeus; do
  for sec in 15 30 45 60 90 120; do
    s=$(printf '%03d' "$sec"); tag="after-${enc}-${s}s"; out="$OUT/${tag}.json"
    [ -f "$out" ] && { echo "[skip ] $tag"; continue; }
    t0=$(date +%s)
    if systemd-run --user --scope --quiet -p MemoryMax=10G -p MemorySwapMax=0 \
         env OMP_NUM_THREADS=4 "$PY" "$AFTER/contrib/E1/measure.py" \
           --repo "$AFTER" --encoder "$enc" --audio "$REC/so-${s}s.wav" \
           --output "$out" --threads 4 --phase-peaks \
           > "$OUT/${tag}.stdout" 2>&1; then
      echo "[ok   ] $tag  $(( $(date +%s) - t0 ))s"
    else
      echo "[FAIL ] $tag"; tail -4 "$OUT/${tag}.stdout" | sed 's/^/         /'
    fi
  done
done
echo; echo "PHASES_DONE"
