#!/usr/bin/env bash
# The academic cost of each corpus ablation -- the half of 3.2b that was never
# measured.
#
# 3.2b ablated four corpus ingredients and scored all four on GMTW-Ro, which
# answered "is this ingredient necessary for the REPAIR" (no ingredient is, and
# two removals improve end-to-end). It never asked the other question: does any
# one ingredient carry the 8.35 points of academic average the repair costs?
#
# That question is worth asking because it is the only remaining lever that
# needs no new training. Everything that changes the SIZE or LOCATION of the
# update sits on the free trade-off curve -- adapter scaling is the curve, and
# lr 2e-5 and attention-only both land on it. Corpus composition is a different
# axis, the adapters already exist, and nobody has looked.
#
# If one track carries most of the damage and is not needed for the repair,
# dropping it is a mitigation available today. If all four damage equally, that
# says the harm is generic to training on rigid structured output, which is
# itself worth knowing and is the case for attacking it with replay instead.
#
# RETENTION ONLY. Every one of these checkpoints already has its GMTW-Ro rows in
# datagen/results; the ladder is the missing half.
#
# Seed 0 only, deliberately. This is a SCREEN, not a result: a single seed read
# against a curve is what made lr 2e-5 look like a mitigation for a day. Anything
# that moves here gets seeds 1 and 2 before it is believed.
#
# Compare against rollama31-dpo_4kv2_s0_lim (-8.35) and base_rollama31-dpo_lim,
# which are the same model at the same LIMIT=2000 with the same base tokenizer.
# A differently-run row does not join to those two.
#
#   datagen/scripts/run_ablation_retention.sh [name ...]
#     default: abl-ro-only abl-no-a abl-no-b abl-no-c
#
#   GPU          card to use                      (default: 0)
#   MIN_FREE_GB  refuse to merge below this       (default: 18)
#   SETUP_KD     env hook for the merge           (default: $HOME/setup_kd.sh)
#   SETUP_ROLLM  env hook for lm_eval             (default: $HOME/rollm_env.sh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN="$(dirname "$HERE")"

SETUP_KD="${SETUP_KD:-$HOME/setup_kd.sh}"
SETUP_ROLLM="${SETUP_ROLLM:-$HOME/rollm_env.sh}"
. "$SETUP_KD"

BASE="${BASE:-OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23}"
GPU="${GPU:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-18}"
WORKDIR="${WORKDIR:-$HOME}"
RETDIR="${RETDIR:-$DATAGEN/retention/results}"

ABLATIONS=("$@")
[ ${#ABLATIONS[@]} -gt 0 ] || ABLATIONS=(abl-ro-only abl-no-a abl-no-b abl-no-c)

free_gb () { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }
echo "base:    $BASE"
echo "configs: ${ABLATIONS[*]}"
echo "card:    $GPU   free: $(free_gb) GB"

for ABL in "${ABLATIONS[@]}"; do
  NAME="rollama31-dpo_${ABL}_s0"
  RET="$RETDIR/${NAME}_lim"
  MERGED="$WORKDIR/merged_models/$NAME"

  if [ "$(ls -d "$RET"/fs*/ 2>/dev/null | wc -l)" -eq 6 ]; then
    echo "=================== $NAME : already scored, skipping ==================="
    continue
  fi

  # The ablation adapters live in two places: the three that were kept are under
  # adapters/, and no-b only ever existed as its training run. Both are the same
  # artefact -- the final/ the trainer wrote.
  ADAPTER=""
  for CAND in "$HOME/adapters/$NAME/final" "$WORKDIR/runs/$NAME/final"; do
    [ -d "$CAND" ] && { ADAPTER="$CAND"; break; }
  done
  [ -n "$ADAPTER" ] || { echo "no adapter found for $NAME" >&2; exit 1; }

  AVAIL="$(free_gb)"
  if [ "$AVAIL" -lt "$MIN_FREE_GB" ]; then
    echo "REFUSING to merge $NAME: ${AVAIL} GB free, below MIN_FREE_GB=${MIN_FREE_GB}." >&2
    exit 1
  fi

  echo "=================== $NAME : merge (${AVAIL} GB free) ==================="
  echo "  adapter: $ADAPTER"
  CUDA_VISIBLE_DEVICES="$GPU" python "$DATAGEN/scripts/merge_lora.py" "$BASE" "$ADAPTER" "$MERGED"
  CUDA_VISIBLE_DEVICES="$GPU" python "$DATAGEN/scripts/align_stop_tokens.py" "$MERGED"

  echo "=================== $NAME : retention ladder ==================="
  CUDA_VISIBLE_DEVICES="$GPU" SETUP="$SETUP_ROLLM" LIMIT=2000 TOKENIZER="$BASE" \
    bash "$DATAGEN/retention/run_rollm_eval.sh" "$MERGED" "${NAME}_lim"

  echo "=================== $NAME : deleting merged checkpoint ==================="
  rm -rf "$MERGED"
  echo "  free after $NAME: $(free_gb) GB"
done

echo "ABLATION RETENTION DONE: ${ABLATIONS[*]}"
