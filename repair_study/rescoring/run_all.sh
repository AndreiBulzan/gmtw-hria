#!/usr/bin/env bash
# Rescore every repair-experiment file through the supplied corrected evaluator.
# 8 checkpoints x (base + 3 seeds) x 2 splits = 64 runs. CPU only, no model calls.
set -uo pipefail
PY="$HOME/envs/gmtw_rescore/bin/python"
R=../results
OUTR=../results/rescored
mkdir -p "$OUTR"
log=rescore_all.log; : > "$log"

# display : baseline dir : repaired prefix
FAMS=(
 "RoLlama3.1-8B-DPO:RoLlama3.1-8b-Instruct-DPO:rollama31-dpo_4kv2"
 "RoGemma2-9B-DPO:RoGemma2-9b-Instruct-DPO:rogemma2-dpo_4kv2"
 "RoGemma2-9B-SFT:RoGemma2-9b-Instruct:rogemma2-sft_4kv2"
 "RoLlama3-8B-DPO:RoLlama3-8b-Instruct-DPO:rollama3-dpo_4kv2"
 "RoMistral-7B-DPO:RoMistral-7b-Instruct-DPO:romistral-dpo_4kv2"
 "RoMistral-7B-SFT:RoMistral-7b-Instruct:romistral-sft_4kv2"
 "RoLlama2-7B:RoLlama2-7b-Instruct-undated:rollama2-sft_4kv2"
 "RoGemma-7B-DPO:RoGemma-7b-Instruct-DPO:rogemma-dpo_4kv2"
)
n=0; fail=0
for e in "${FAMS[@]}"; do
  IFS=: read -r disp bdir pre <<<"$e"
  for pair in "500:standard" "300:adversarial"; do
    IFS=: read -r sp split <<<"$pair"
    # baseline
    f="$R/$bdir/${bdir}_${sp}_outputs.jsonl"
    if [ -f "$f" ]; then
      n=$((n+1))
      $PY -X utf8 code/rescore_saved_outputs.py --split "$split" --outputs "$f" \
         --out-dir "$OUTR/${pre}_base_${split}" --label "$disp base $split" >>"$log" 2>&1 \
         || { echo "FAIL base $disp $split" | tee -a "$log"; fail=$((fail+1)); }
    else echo "MISSING $f" | tee -a "$log"; fi
    # three seeds
    for s in 0 1 2; do
      f="$R/${pre}_s${s}/${pre}_s${s}_${sp}_outputs.jsonl"
      if [ -f "$f" ]; then
        n=$((n+1))
        $PY -X utf8 code/rescore_saved_outputs.py --split "$split" --outputs "$f" \
           --out-dir "$OUTR/${pre}_s${s}_${split}" --label "$disp seed$s $split" >>"$log" 2>&1 \
           || { echo "FAIL s$s $disp $split" | tee -a "$log"; fail=$((fail+1)); }
      else echo "MISSING $f" | tee -a "$log"; fi
    done
  done
done
echo "ran $n, failures $fail"
