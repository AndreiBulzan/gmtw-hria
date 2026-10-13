#!/usr/bin/env bash
# One academic ladder for one checkpoint, base or repaired, at LIMIT=2000.
#
# Bracketing 3.3 across families: 3.3 is measured on RoLlama3.1-8B-DPO, which
# sits fourth of eight by how far the repair had to move it (dU +0.330). The
# interpolation curve says damage scales with the magnitude of the update, so
# dU is the principled ordering, and measuring its two ENDS -- RoGemma-7B-DPO
# (+0.497) and RoLlama2-7B (+0.147) -- speaks for the six families between them.
#
# BATCH IS PER FAMILY, not global -- the same asymmetry run_remaining.sh phase 3
# already encodes. Gemma carries a 256k vocabulary, so at batch 64 a single
# forward asks for a 38 GiB logits tensor and the card refuses. Both arms of a
# pair must share one value so nothing inside a comparison varies.
#
#   run_family_ladder.sh base     <hf-id>   <name> <gpu> <batch>
#   run_family_ladder.sh repaired <hf-base> <name> <gpu> <batch> <adapter-dir>
set -euo pipefail
MODE="$1"; HFBASE="$2"; NAME="$3"; GPU="$4"; BATCH="$5"; ADAPTER="${6:-}"
. "$HOME/setup_kd.sh"
cd "$HOME/kd-hria"
export CUDA_VISIBLE_DEVICES="$GPU"

FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
echo "########## $NAME ($MODE, card $GPU, batch $BATCH, ${FREE} GB free) ##########"

if [ "$MODE" = "repaired" ]; then
  [ -d "$ADAPTER" ] || { echo "no adapter: $ADAPTER" >&2; exit 1; }
  TARGET="$HOME/merged_models/$NAME"
  # Resumable: a merge that already produced a checkpoint is not repeated, so a
  # failure after the merge costs the ladder and not the 20 minutes before it.
  # The disk guard belongs INSIDE this branch -- an existing checkpoint needs no
  # room, and refusing to score it because the room a merge would have wanted is
  # gone is exactly backwards.
  if [ -f "$TARGET/config.json" ]; then
    echo "  merged checkpoint already present, skipping merge"
  else
    [ "$FREE" -ge 22 ] || { echo "only ${FREE} GB free, refusing to merge" >&2; exit 1; }
    python datagen/scripts/merge_lora.py "$HFBASE" "$ADAPTER" "$TARGET"
    python datagen/scripts/align_stop_tokens.py "$TARGET"
  fi
  TOK="$HFBASE"
else
  TARGET="$HFBASE"
  TOK=""          # base checkpoints load their own tokenizer
fi

# `env` rather than a bare prefix: bash only treats an assignment as an
# assignment when it is literal at parse time, so ${TOK:+TOKENIZER=...} expands
# into something bash tries to EXECUTE. env takes expanded assignments happily.
env SETUP="$HOME/rollm_env.sh" LIMIT=2000 BATCH="$BATCH" ${TOK:+TOKENIZER="$TOK"} \
  bash datagen/retention/run_rollm_eval.sh "$TARGET" "${NAME}_lim"

[ "$MODE" = "repaired" ] && rm -rf "$TARGET"
df -h / | tail -1
echo "LADDER DONE: $NAME"
