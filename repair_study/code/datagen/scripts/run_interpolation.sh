#!/usr/bin/env bash
# Interpolation sweep between the base checkpoint and the repaired one:
#
#     theta(lambda) = base + lambda * (repaired - base)
#
# lambda=0 is the untouched model, lambda=1 is the repair as shipped, and both
# ends are already measured at LIMIT=2000 (base_rollama31-dpo_lim and
# rollama31-dpo_4kv2_s0_lim). This fills in the middle.
#
# The question it answers. 3.3 measured the repair costing 8.35 points of
# academic average. Two recipe levers were tried and neither separated repair
# from damage: a fifth of the learning rate recovered 11% of the loss, and
# attention-only adaptation made it 12% worse. Those change how the update is
# computed. This changes nothing about training at all -- it asks whether the
# two effects separate along the straight line between the two models. If the
# repair saturates at a smaller lambda than the damage does, there is a
# mitigation available with no retraining and no replay corpus. If they rise
# together, repair and damage are one direction in weight space, which is a
# sharper statement of what the dose curve found in time.
#
# Both arms are scored for every lambda, always. A lambda that keeps the repair
# and keeps the damage has told us nothing, and so has one that loses both --
# the same rule the recipe variants were run under.
#
# DISK IS THE BINDING CONSTRAINT, not GPU. One merged 8B checkpoint is ~15 GB
# against ~46 GB free on a volume seven accounts share. So: one lambda at a
# time, scored on both benchmarks concurrently on two cards, then DELETED before
# the next merge. The sweep never holds two checkpoints. It refuses to start a
# merge that would take the volume below MIN_FREE_GB.
#
# Resumable: a lambda whose GMTW metrics and retention rungs are both present is
# skipped, so a kill or a full disk costs one lambda and not the sweep.
#
#   datagen/scripts/run_interpolation.sh [lambda ...]     (default: 0.25 0.5 0.75)
#
#   ADAPTER      LoRA to scale     (default: $HOME/adapters/rollama31-dpo_4kv2_s0/final)
#   BASE         base checkpoint   (default: the RoLlama3.1-8b DPO release)
#   GPU_GEN      card for vLLM generation           (default: 0)
#   GPU_RET      card for the lm_eval ladder        (default: 1)
#   MIN_FREE_GB  refuse to merge below this         (default: 20)
#   SETUP_KD     env hook for merge + vLLM          (default: $HOME/setup_kd.sh)
#   SETUP_ROLLM  env hook for lm_eval               (default: $HOME/rollm_env.sh)
#   KEEP         1 to keep merged checkpoints. Only with disk to spare.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # datagen/scripts
DATAGEN="$(dirname "$HERE")"
REPO="$(dirname "$DATAGEN")"

SETUP_KD="${SETUP_KD:-$HOME/setup_kd.sh}"
SETUP_ROLLM="${SETUP_ROLLM:-$HOME/rollm_env.sh}"
. "$SETUP_KD"

ADAPTER="${ADAPTER:-$HOME/adapters/rollama31-dpo_4kv2_s0/final}"
BASE="${BASE:-OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23}"
GPU_GEN="${GPU_GEN:-0}"
GPU_RET="${GPU_RET:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
BENCH="${BENCH:-$HOME/gmtw-hria}"
WORKDIR="${WORKDIR:-$HOME}"
OUTDIR="${OUTDIR:-$DATAGEN/results}"
RETDIR="${RETDIR:-$DATAGEN/retention/results}"

LAMBDAS=("$@")
[ ${#LAMBDAS[@]} -gt 0 ] || LAMBDAS=(0.25 0.5 0.75)

[ -d "$ADAPTER" ] || { echo "adapter not found: $ADAPTER" >&2; exit 1; }
[ -d "$BENCH" ]   || { echo "benchmark checkout not found: $BENCH" >&2; exit 1; }

free_gb () { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

echo "adapter:  $ADAPTER"
echo "base:     $BASE"
echo "lambdas:  ${LAMBDAS[*]}"
echo "free now: $(free_gb) GB"

for LAM in "${LAMBDAS[@]}"; do
  TAG="$(python3 -c "print(f'{round(float('$LAM')*100):03d}')")"
  NAME="rollama31-dpo_interp-l${TAG}_s0"
  MERGED="$WORKDIR/merged_models/$NAME"
  OUT="$OUTDIR/$NAME"
  RET="$RETDIR/${NAME}_lim"

  if [ -f "$OUT/${NAME}_500.jsonl" ] && [ -f "$OUT/${NAME}_300.jsonl" ] \
     && [ "$(ls -d "$RET"/fs*/ 2>/dev/null | wc -l)" -eq 6 ]; then
    echo "=================== $NAME : already complete, skipping ==================="
    continue
  fi

  AVAIL="$(free_gb)"
  if [ "$AVAIL" -lt "$MIN_FREE_GB" ]; then
    echo "REFUSING to merge $NAME: ${AVAIL} GB free, below MIN_FREE_GB=${MIN_FREE_GB}." >&2
    echo "Delete something under $WORKDIR/merged_models or \$HF_HOME and re-run." >&2
    exit 1
  fi

  echo "=================== $NAME : merge (lambda=$LAM, ${AVAIL} GB free) ==================="
  mkdir -p "$OUT"
  python "$DATAGEN/scripts/merge_lora.py" "$BASE" "$ADAPTER" "$MERGED" --scale "$LAM"
  python "$DATAGEN/scripts/align_stop_tokens.py" "$MERGED"

  echo "=================== $NAME : generate + retain (cards $GPU_GEN and $GPU_RET) ==================="
  # Both arms at once on separate cards. The merged checkpoint is read-only to
  # both, and neither writes where the other reads.
  (
    cd "$BENCH"
    CUDA_VISIBLE_DEVICES="$GPU_GEN" python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl \
      --model-path "$MERGED" --output "$OUT/${NAME}_500_outputs.jsonl"
    CUDA_VISIBLE_DEVICES="$GPU_GEN" python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl \
      --model-path "$MERGED" --output "$OUT/${NAME}_300_outputs.jsonl"
    python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl \
      "$OUT/${NAME}_500_outputs.jsonl" --save-metrics "$OUT/${NAME}_500.jsonl" | tail -12
    python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl \
      "$OUT/${NAME}_300_outputs.jsonl" --save-metrics "$OUT/${NAME}_300.jsonl" | tail -12
  ) > "$OUT/log_gmtw_${NAME}.txt" 2>&1 &
  GEN_PID=$!

  # LIMIT=2000 and TOKENIZER from the base, matching base_rollama31-dpo_lim and
  # rollama31-dpo_4kv2_s0_lim exactly -- those two are the ends of this curve and
  # a differently-run middle would not join to them.
  (
    CUDA_VISIBLE_DEVICES="$GPU_RET" SETUP="$SETUP_ROLLM" LIMIT=2000 TOKENIZER="$BASE" \
      "$DATAGEN/retention/run_rollm_eval.sh" "$MERGED" "${NAME}_lim"
  ) > "$WORKDIR/ret_${NAME}.log" 2>&1 &
  RET_PID=$!

  GEN_OK=0; RET_OK=0
  wait "$GEN_PID" || GEN_OK=$?
  wait "$RET_PID" || RET_OK=$?
  echo "  generation exit $GEN_OK  (log: $OUT/log_gmtw_${NAME}.txt)"
  echo "  retention  exit $RET_OK  (log: $WORKDIR/ret_${NAME}.log)"

  if [ "$GEN_OK" -ne 0 ] || [ "$RET_OK" -ne 0 ]; then
    echo "KEEPING $MERGED for diagnosis -- one arm failed." >&2
    exit 1
  fi

  if [ "${KEEP:-0}" != "1" ]; then
    echo "=================== $NAME : deleting merged checkpoint ==================="
    rm -rf "$MERGED"
  fi
  echo "  free after $NAME: $(free_gb) GB"
done

echo "INTERPOLATION SWEEP DONE: ${LAMBDAS[*]}"
