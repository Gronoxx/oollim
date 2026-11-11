#!/usr/bin/env bash
# Full release path: the before arm twice and the after arm, for the identity test.
set -uo pipefail
CLONE="<repo>"
cd "$CLONE"
AFTER="$CLONE/work/wt-after"; BEFORE="$CLONE/work/wt-before"
REC="$CLONE/work/E1/recordings"; OUT="$CLONE/work/E1/scores"
mkdir -p "$OUT"; PY="$CLONE/.venv/bin/python"

run() {
  local tag=$1 repo=$2 sec=$3
  local s=$(printf '%03d' "$sec")
  local pre="$OUT/${tag}-${s}s"
  if [ -f "${pre}.json" ]; then echo "[skip ] ${tag}-${s}s"; return 0; fi
  local t0=$(date +%s)
  if systemd-run --user --scope --quiet -p MemoryMax=10G -p MemorySwapMax=0 \
       env OMP_NUM_THREADS=4 "$PY" "$AFTER/contrib/E1/score.py" \
         --repo "$repo" --audio "$REC/so-${s}s.wav" --reference "$REC/so-${s}s.txt" \
         --output "$pre" --threads 4 > "${pre}.stdout" 2>&1; then
    echo "[ok   ] ${tag}-${s}s  $(( $(date +%s) - t0 ))s"
  else
    echo "[FAIL ] ${tag}-${s}s  $(( $(date +%s) - t0 ))s"; tail -5 "${pre}.stdout" | sed 's/^/         /'
  fi
}

for sec in 15 30 45 60; do
  run before1 "$BEFORE" "$sec"
  run before2 "$BEFORE" "$sec"
  run after   "$AFTER"  "$sec"
done
echo; echo "########## IDENTITY ##########"
for sec in 15 30 45 60; do
  s=$(printf '%03d' "$sec")
  for pair in "before1 after" "before1 before2"; do
    set -- $pair
    a="$OUT/$1-${s}s"; b="$OUT/$2-${s}s"
    if [ -f "${a}.json" ] && [ -f "${b}.json" ]; then
      echo "--- ${s}s: $1 vs $2 ---"
      "$PY" "$AFTER/contrib/E1/compare.py" identity "$a" "$b" 2>&1 | tail -30
    fi
  done
done
echo "IDENTITY_DONE"
