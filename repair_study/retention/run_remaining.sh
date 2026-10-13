#!/usr/bin/env bash
# Everything still owed on the retention study, in one unattended pass.
#
#   datagen/retention/run_remaining.sh
#
# Three phases, in value order rather than convenience order: if the machine
# goes away halfway through, what survives should be the part that answers a
# question nothing else answers.
#
#   1. dose curve, GMTW-Ro side   merge checkpoint-{100,250,400} of the headline
#                                 seed and score the benchmark the repair was
#                                 for. Needs the TRAINING env (transformers 5.x,
#                                 vLLM).
#   2. dose curve, retention side MMLU and GSM8k at 5-shot on those same three
#                                 merges -- the cells the format control used,
#                                 so they line up with the base and full-dose
#                                 numbers already measured. Needs the EVAL env.
#   3. the three secondary pairs  RoGemma2-9B-DPO, RoGemma-7B-DPO,
#                                 RoMistral-7B-SFT, base and seed 0, at
#                                 LIMIT=2000. Confirmation that the damage is
#                                 not one checkpoint's quirk.
#
# The dose curve comes first because it is the only part that could change the
# recommendation. Every training run saved an adapter every 50 steps of its 500,
# so the curve between "no repair" and "8.28 points gone" is already on disk:
# if the repair saturates early while the damage keeps climbing, the recipe is
# simply over-dosed and the fix costs nothing. The secondary pairs, by contrast,
# tell us something three seeds already suggest.
#
# Two environments, which is the whole reason this is a script and not a paste:
# phase 1 merges adapters and serves them with vLLM, phase 2 and 3 need the
# 0.4.x-era harness and its transformers 4.53. Both are activated here rather
# than left to the operator.
#
#   KD_SETUP  hook for the training env  (default: /DATA/stephen/env.sh + envs/kd)
#   SETUP     hook for the eval env      (default: /DATA/stephen/rollm_env.sh)
#   BENCH     gmtw-hria checkout         (default: <repo>/../gmtw-hria)
#   ROLLM     OpenLLM-Ro harness         (default: /DATA/stephen/eval_rollm/...)
#   STEPS     dose points                (default: 100 250 400)
#   SKIP      phases to skip, e.g. "1 3" (default: none)
#
# Everything it calls is idempotent: finished rungs are skipped, and a merge
# that already exists is not redone. Re-running after a kill resumes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN="$(dirname "$HERE")"
REPO="$(dirname "$DATAGEN")"

BENCH="${BENCH:-$REPO/../gmtw-hria}"
ROLLM="${ROLLM:-/DATA/stephen/eval_rollm/lm-evaluation-harness}"
KD_SETUP="${KD_SETUP:-}"
SETUP="${SETUP:-/DATA/stephen/rollm_env.sh}"
STEPS="${STEPS:-100 250 400}"
SKIP="${SKIP:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE=OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23
RUN_SEED=rollama31-dpo_4kv2_s0
MERGED="$REPO/merged_models"

skipped () { case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
banner () { echo; echo "=============== $* ==============="; echo; }

# ---------------------------------------------------------------- phase 1
if skipped 1; then echo "phase 1 skipped"; else
banner "PHASE 1  dose curve, GMTW-Ro side"
(
  if [ -n "$KD_SETUP" ]; then . "$KD_SETUP"; else
    . /DATA/stephen/env.sh
    conda activate /DATA/stephen/envs/kd
  fi
  for C in $STEPS; do
    NAME="dose_rollama31_c$C"
    # merge_lora writes the weights first and the tokenizer last, so the
    # tokenizer is what says a merge finished. Testing for the directory is not
    # enough: a merge killed in between leaves 16 GB of weights that vLLM will
    # load and then fail to tokenise for.
    if [ -s "$MERGED/$NAME/tokenizer.json" ]; then
      echo "$NAME already merged"
    elif [ -s "$MERGED/$NAME/model.safetensors" ]; then
      echo "$NAME has weights but no tokenizer -- finishing the merge"
      python - "$BASE" "$MERGED/$NAME" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
print("tokenizer saved")
PY
    else
      python "$DATAGEN/scripts/merge_lora.py" "$BASE" \
        "$REPO/runs/$RUN_SEED/checkpoint-$C" "$MERGED/$NAME"
    fi
    if [ -s "$DATAGEN/results/$NAME/${NAME}_500.jsonl" ]; then
      echo "$NAME already scored on GMTW-Ro"
    else
      BENCH="$BENCH" "$DATAGEN/scripts/run_baseline.sh" "$MERGED/$NAME" "$NAME"
    fi
  done
)
fi

# ---------------------------------------------------------------- phase 2
if skipped 2; then echo "phase 2 skipped"; else
banner "PHASE 2  dose curve, retention side"
for C in $STEPS; do
  NAME="dose_rollama31_c$C"
  SETUP="$SETUP" ROLLM="$ROLLM" RUNGS=5 TASKS=_ro_mmlu,_ro_gsm8k TOKENIZER="$BASE" \
    "$HERE/run_rollm_eval.sh" "$MERGED/$NAME" "$NAME"
done
fi

# ---------------------------------------------------------------- phase 3
if skipped 3; then echo "phase 3 skipped"; else
banner "PHASE 3  the three secondary pairs, LIMIT=2000"
# base checkpoint, merged checkpoint, row name -- the tokenizer always comes
# from the base, because a merged checkpoint written by transformers 5.x
# declares a tokenizer class the eval env cannot resolve, and because it makes
# both arms of a pair tokenise identically.
# The fourth column is the batch size, and it is per family rather than global.
# Gemma carries a 256k vocabulary, so the logits tensor is ~8x Llama's -- at
# batch 64 a single forward asks for 38 GiB and the card refuses. The training
# configs already knew this (Gemma trains at 1x16 where Llama trains at 2x8);
# the eval side has to know it too. Both arms of a pair share one batch size, so
# nothing inside a comparison varies.
while read -r BASE_ID MERGE ROW BATCH; do
  [ -n "$BASE_ID" ] || continue
  SETUP="$SETUP" ROLLM="$ROLLM" LIMIT=2000 BATCH="$BATCH" \
    "$HERE/run_rollm_eval.sh" "$BASE_ID" "base_${ROW}_lim"
  SETUP="$SETUP" ROLLM="$ROLLM" LIMIT=2000 BATCH="$BATCH" TOKENIZER="$BASE_ID" \
    "$HERE/run_rollm_eval.sh" "$MERGED/$MERGE" "${MERGE}_lim"
done <<'PAIRS'
OpenLLM-Ro/RoGemma2-9b-Instruct-DPO-2025-04-23 rogemma2-dpo_4kv2_s0 rogemma2-dpo 16
OpenLLM-Ro/RoGemma-7b-Instruct-DPO rogemma-dpo_4kv2_s0 rogemma-dpo 16
OpenLLM-Ro/RoMistral-7b-Instruct-2025-04-23 romistral-sft_4kv2_s0 romistral-sft 64
PAIRS
fi

banner "ALL PHASES DONE"
echo "tables:  python datagen/retention/retention_tables.py \\"
echo "             --results datagen/retention/results --out datagen/retention/tables"
echo "the three merges under $MERGED/dose_rollama31_c* are ~15 GB each and are"
echo "deletable once phase 2 has scored them."
