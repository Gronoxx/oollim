#!/usr/bin/env bash
# Batch of E1 measurements. Each run in its own process, under a memory cap.
set -uo pipefail
CLONE="<repo>"
cd "$CLONE"
AFTER="$CLONE/work/wt-after"
BEFORE="$CLONE/work/wt-before"
REC="$CLONE/work/E1/recordings"
OUT="$CLONE/work/E1/measurements"
mkdir -p "$OUT"
PY="$CLONE/.venv/bin/python"

run() {
  local arm=$1 repo=$2 enc=$3 sec=$4
  local s=$(printf '%03d' "$sec")
  local tag="${arm}-${enc}-${s}s"
  local out="$OUT/${tag}.json"
  if [ -f "$out" ]; then echo "[skip ] $tag (already exists)"; return 0; fi
  local t0=$(date +%s)
  if systemd-run --user --scope --quiet -p MemoryMax=10G -p MemorySwapMax=0 \
       env OMP_NUM_THREADS=4 "$PY" "$AFTER/contrib/E1/measure.py" \
         --repo "$repo" --encoder "$enc" --audio "$REC/so-${s}s.wav" \
         --output "$out" --threads 4 > "$OUT/${tag}.stdout" 2>&1; then
    echo "[ok   ] $tag  $(( $(date +%s) - t0 ))s  $(tail -1 "$OUT/${tag}.stdout" | head -c 160)"
  else
    echo "[FAIL ] $tag  $(( $(date +%s) - t0 ))s"
    tail -4 "$OUT/${tag}.stdout" | sed 's/^/         /'
  fi
}

for enc in phoneticxeus xeus xlsr; do
  echo "########## $enc ##########"
  for sec in 15 30 45 60; do
    run before "$BEFORE" "$enc" "$sec"
    run after  "$AFTER"  "$enc" "$sec"
  done
  for sec in 90 120; do
    run after  "$AFTER"  "$enc" "$sec"
  done
done

echo
echo "########## SUMMARY ##########"
ls -1 "$OUT"/*.json 2>/dev/null | wc -l
echo "json files produced"
echo "BATCH_DONE"
