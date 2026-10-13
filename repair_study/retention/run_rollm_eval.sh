#!/usr/bin/env bash
# Score one checkpoint on the benchmarks OpenLLM-Ro used to evaluate its own
# models -- the retention half of the repair claim.
#
#   datagen/retention/run_rollm_eval.sh <hf-model-id-or-path> <name>
#   datagen/retention/run_rollm_eval.sh OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23 base_rollama31-dpo
#   datagen/retention/run_rollm_eval.sh $WORKDIR/merged_models/rollama31-dpo_4kv2_s0 rollama31-dpo_4kv2_s0
#
# The review (3.3) says the repair demonstrates GMTW recovery, not a free
# repair: nothing outside the intervention task was measured. This measures it,
# on the suite the released checkpoints are themselves scored on, so a
# regression lands in the same units the model cards publish -- and so the base
# row can be checked against the published number rather than taken on trust.
#
# The published protocol is a few-shot ladder per task -- ARC 0/1/3/5/10/25,
# MMLU 0/1/3/5, Winogrande 0/1/3/5, HellaSwag 0/1/3/5/10, GSM8k 1/3/5,
# TruthfulQA 0 -- averaged. The harness ships exactly those ladders as groups
# (`ro_mmlu` and friends), but they cannot be called: in this version `include:`
# overrides the subtask's own `task:` key, so mmlu_fs0/1/3/5 all resolve to the
# same `_ro_mmlu` and the duplicate check rejects the group before any model
# loads. It fails on one group alone, so this is not about calling several.
#
# So the ladder is run explicitly: the base task with --num_fewshot per rung,
# batched by rung so a rung costs one model load rather than one per task. Six
# invocations per checkpoint, and the same numbers the group would have
# averaged -- each published task average is the plain mean of its rungs, which
# `published_baselines.py` asserts against the cards.
#
# One results_*.json per rung, under fs<K>/. Averaging is ours to do, which is
# also what makes the per-rung comparison against the published ladder possible.
#
# Prompts are plain completions, no chat template. That is how OpenLLM-Ro scored
# these checkpoints, and --apply_chat_template would make our rows incomparable
# with the cards. Keep it off unless both arms change together -- which is
# exactly what CHAT=1 is for, and only for the format control.
#
#   ROLLM   OpenLLM-Ro lm-evaluation-harness checkout
#   OUTDIR  where results land          (default: datagen/retention/results)
#   FORCE   1 to redo finished rungs     (default: skip them, so a kill resumes)
#   RUNGS   few-shot rungs to run        (default: 0 1 3 5 10 25)
#   TASKS   override the rung's task list, e.g. _ro_mmlu,_ro_gsm8k. Only for
#           diagnostics: a subset does not produce the published average.
#   CHAT    1 to pass --apply_chat_template. OFF for anything compared against
#           the cards -- OpenLLM-Ro scored raw completions -- and ON only for
#           the format control, where BOTH arms must carry it. Recorded in the
#           provenance file, and the tables key on it, so a chat row cannot be
#           averaged with a completion row by accident.
#   LIMIT   docs per task, for the reduced-coverage pass (default: all)
#   BATCH   --batch_size value          (default: 64, and not `auto`)
#   SEED    all four lm_eval seeds      (default: 1234, the harness default)
#   ATTN    attn_implementation passed through to from_pretrained. REQUIRED for
#           Gemma-2: that family applies attention and final logit softcapping
#           (attn 50.0, final 30.0 on both RoGemma2 checkpoints), and HF computes
#           it correctly only on the eager path -- SDPA silently skips it, so the
#           scores come out subtly wrong rather than loudly broken. train_lora.py
#           already carries the same rule for the training side. Empty for every
#           other family here: Gemma-1, Llama and Mistral configs have no
#           softcapping, so their existing rows are unaffected and this flag
#           leaves them byte-identical.
#   TOKENIZER  load the tokenizer from here instead of from the checkpoint.
#           Required for our merged checkpoints: they were written by the
#           training env's transformers 5.x, whose tokenizer_config.json says
#           `"tokenizer_class": "TokenizersBackend"`, and 4.53 answers
#           `ValueError: Tokenizer class TokenizersBackend does not exist`.
#           Point it at the checkpoint the LoRA was trained on -- LoRA does not
#           touch the tokenizer, and the merged tokenizer.json is byte-identical
#           to the base one, so this loads the same vocabulary and additionally
#           guarantees both arms of a pair tokenise identically.
#   SETUP   shell hook sourced first    (conda activate, HF_HOME, ...)
#
# A LIMIT run and a full run are not comparable. Both arms of a pair must be run
# the same way, and LIMIT is recorded in the provenance file for that reason.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # datagen/retention
DATAGEN="$(dirname "$HERE")"
REPO="$(dirname "$DATAGEN")"

[ -n "${SETUP:-}" ] && . "$SETUP"

ROLLM="${ROLLM:-$REPO/../lm-evaluation-harness}"
OUTDIR="${OUTDIR:-$HERE/results}"
RUNGS="${RUNGS:-0 1 3 5 10 25}"
TASKS_OVERRIDE="${TASKS:-}"
CHAT_FLAG=""
[ "${CHAT:-0}" = "1" ] && CHAT_FLAG="--apply_chat_template"

# Gemma-2 generation wraps forward in torch.compile(fullgraph=True) for its
# hybrid cache, and dynamo tracing is disabled in this environment, so compile
# finds no frames and raises instead of falling back to eager -- killing GSM8k
# on every Gemma-2 checkpoint. Disabling dynamo outright makes compile a no-op.
# Set for every run rather than the Gemma ones, so one flag does not vary across
# a table: it changes nothing for the models that never reach the compile path
# (Llama and Mistral use a dynamic cache), which is why the runs already
# finished stay valid.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

# rung -> the tasks whose published ladder includes it. Anything not listed for
# a rung is not part of that task's published average and must not be run there.
tasks_for_rung () {
  case "$1" in
    0)  echo "_ro_arc_challenge,_ro_mmlu,_ro_winogrande,_ro_hellaswag,_ro_truthfulqa" ;;
    1)  echo "_ro_arc_challenge,_ro_mmlu,_ro_winogrande,_ro_hellaswag,_ro_gsm8k" ;;
    3)  echo "_ro_arc_challenge,_ro_mmlu,_ro_winogrande,_ro_hellaswag,_ro_gsm8k" ;;
    5)  echo "_ro_arc_challenge,_ro_mmlu,_ro_winogrande,_ro_hellaswag,_ro_gsm8k" ;;
    10) echo "_ro_arc_challenge,_ro_hellaswag" ;;
    25) echo "_ro_arc_challenge" ;;
    *)  echo "" ;;
  esac
}
# NOT `auto`. Auto-detection sizes the batch by probing allocations, and after
# the log-likelihood phase the caching allocator is holding most of the card in
# its reserved pool -- so a 64 GiB probe fails against 12 GB of unreserved
# memory, the search backs off to 1, and generate_until crawls at one request
# at a time: GSM8k took 4.3 s/item, 1.6 h per rung, on a GPU sitting at 51%
# utilisation and 16 GB of 143 GB in use. Auto picked 64 for the log-likelihood
# phase, which is the number it should have kept, so 64 is what we pin.
BATCH="${BATCH:-64}"
SEED="${SEED:-1234}"

# The commit Andrei's checkout is pinned at, and the one these tasks were read
# from. A different commit is not fatal -- it is recorded, and reported here so
# it is a decision rather than a surprise.
PINNED=77e8ac48c750b661e2cf1f48085e4a4181406ed1

[ $# -eq 2 ] || { echo "usage: $0 <model> <name>" >&2; exit 1; }
MODEL="$1"; NAME="$2"
[ -d "$ROLLM" ] || { echo "harness checkout not found: $ROLLM" >&2; exit 1; }

python -c "import lm_eval" 2>/dev/null || {
  echo "lm_eval is not importable in this environment." >&2
  echo "It cannot go in the training env: that one runs transformers 5.x and" >&2
  echo "this harness is a 0.4.x fork. Build it its own env -- see" >&2
  echo "datagen/retention/README.md." >&2
  exit 1; }

HEAD_SHA="$(git -C "$ROLLM" rev-parse HEAD 2>/dev/null || echo unknown)"
[ "$HEAD_SHA" = "$PINNED" ] || \
  echo "NOTE: harness at $HEAD_SHA, expected $PINNED -- recorded, not blocking."

O="$OUTDIR/$NAME"; mkdir -p "$O"
echo "harness: $ROLLM ($HEAD_SHA)"
echo "model:   $MODEL"
echo "rungs:   $RUNGS"
echo "out:     $O"

{
  echo "date          $(date -Is)"
  echo "model         $MODEL"
  echo "name          $NAME"
  echo "harness_head  $HEAD_SHA"
  echo "harness_dirty $(git -C "$ROLLM" status --porcelain 2>/dev/null | wc -l) modified files"
  echo "rungs         $RUNGS"
  echo "limit         ${LIMIT:-none}"
  echo "batch_size    $BATCH"
  echo "seed          $SEED"
  echo "tokenizer      ${TOKENIZER:-from the checkpoint}"
  echo "attn_impl      ${ATTN:-default}"
  echo "chat_template  $([ "${CHAT:-0}" = "1" ] && echo on || echo off)"
  echo "dynamo         disabled (TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE)"
  echo "gpu           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "--- versions ---"
  python - <<'PY'
for m in ("torch", "transformers", "datasets", "accelerate", "lm_eval"):
    try:
        print(f"{m:14s} {__import__(m).__version__}")
    except Exception as e:
        print(f"{m:14s} unavailable ({e.__class__.__name__})")
PY
} > "$O/provenance_${NAME}.txt"

for K in $RUNGS; do
  TASKS="${TASKS_OVERRIDE:-$(tasks_for_rung "$K")}"
  [ -n "$TASKS" ] || { echo "no task takes $K shots in the published ladder, skipping" >&2; continue; }

  # lm_eval writes <output_path>/<sanitised model name>/results_<stamp>.json,
  # so one level down is the whole search.
  if compgen -G "$O/fs$K/*/results_*.json" > /dev/null && [ "${FORCE:-0}" != "1" ]; then
    echo "=================== $NAME / fs$K : already scored, skipping ==================="
    continue
  fi

  echo "=================== $NAME / fs$K : $TASKS ==================="
  set -x
  python -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16${TOKENIZER:+,tokenizer=$TOKENIZER}${ATTN:+,attn_implementation=$ATTN}" \
    --tasks "$TASKS" \
    --num_fewshot "$K" \
    $CHAT_FLAG \
    --batch_size "$BATCH" \
    --device cuda:0 \
    --seed "$SEED" \
    --output_path "$O/fs$K" \
    --log_samples \
    ${LIMIT:+--limit "$LIMIT"} \
    2>&1 | tee -a "$O/log_${NAME}.txt"
  set +x
done
echo "ROLLM EVAL DONE: $NAME -> $O"
