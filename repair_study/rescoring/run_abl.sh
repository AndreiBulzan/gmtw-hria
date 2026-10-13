#!/usr/bin/env bash
set -uo pipefail
PY="$HOME/envs/gmtw_rescore/bin/python"; R=../results; log=rescore_abl.log; : > "$log"
n=0; fail=0
for arm in abl-ro-only abl-no-a abl-no-b abl-no-c; do
  for s in 0 1 2; do
    for pair in "500:standard" "300:adversarial"; do
      IFS=: read -r sp split <<<"$pair"
      f="$R/rollama31-dpo_${arm}_s${s}/rollama31-dpo_${arm}_s${s}_${sp}_outputs.jsonl"
      if [ -f "$f" ]; then
        n=$((n+1))
        $PY -X utf8 code/rescore_saved_outputs.py --split "$split" --outputs "$f" \
          --out-dir "$OUTR/abl_${arm}_s${s}_${split}" --label "$arm seed$s $split" >>"$log" 2>&1 \
          || { echo "FAIL $arm s$s $split" | tee -a "$log"; fail=$((fail+1)); }
      else echo "MISSING $f" >>"$log"; fi
    done
  done
done
echo "ran $n, failures $fail"
