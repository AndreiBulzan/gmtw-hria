#!/usr/bin/env bash
# Baseline for a checkpoint not yet in the harness: generate + score, no
# training. Needed before any repaired number means anything -- a repair has
# nothing to sit against until its own starting point has been through the
# same scorer.
#
#   datagen/scripts/run_baseline.sh <hf-model-id-or-path> <name>
#   datagen/scripts/run_baseline.sh OpenLLM-Ro/RoGemma-7b-Instruct-DPO RoGemma-7b-Instruct-DPO
#
# Roots are derived from this script's location; override with BENCH, OUTDIR or
# SETUP exactly as in run_family.sh. Nothing is hardcoded to one machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN="$(dirname "$HERE")"
REPO="$(dirname "$DATAGEN")"

[ -n "${SETUP:-}" ] && . "$SETUP"

BENCH="${BENCH:-$REPO/../gmtw-hria}"
OUTDIR="${OUTDIR:-$DATAGEN/results}"

[ $# -eq 2 ] || { echo "usage: $0 <model> <name>" >&2; exit 1; }
MODEL="$1"; NAME="$2"
[ -d "$BENCH" ] || { echo "benchmark checkout not found: $BENCH" >&2; exit 1; }

O="$OUTDIR/$NAME"; mkdir -p "$O"
echo "bench: $BENCH"
echo "out:   $O"

cd "$BENCH"
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl \
  --model-path "$MODEL" --output "$O/${NAME}_500_outputs.jsonl"
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl \
  --model-path "$MODEL" --output "$O/${NAME}_300_outputs.jsonl"
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl \
  "$O/${NAME}_500_outputs.jsonl" --save-metrics "$O/${NAME}_500.jsonl" | tail -10
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl \
  "$O/${NAME}_300_outputs.jsonl" --save-metrics "$O/${NAME}_300.jsonl" | tail -10
echo "BASELINE DONE: $NAME"
