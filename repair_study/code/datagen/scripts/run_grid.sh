#!/usr/bin/env bash
# Score one checkpoint on the SAME grid the released checkpoints were scored on:
# five language x difficulty cells, generated and scored by the benchmark's own
# scripts with the flags gmtw-hria/run_all.sh uses.
#
#   datagen/scripts/run_grid.sh <hf-model-id-or-path> <name>
#   datagen/scripts/run_grid.sh $WORKDIR/merged_models/rollama31-dpo_4kv2_s0 rollama31-dpo_4kv2_s0
#
# run_family.sh produces Romanian only, which is all the repair tables needed.
# The manuscript's base rows are trilingual, so a repaired row cannot sit beside
# them until it has been through the same five cells. This does that and nothing
# else -- no training, no merging.
#
# Cells, taken from run_all.sh rather than reinvented:
#
#   easy_ro  data/gmtw_ro_v0.jsonl          --language ro   500
#   easy_en  data/gmtw_ro_v0.jsonl          --language en   500
#   easy_de  data/gmtw_ro_v0_DE_test.jsonl  --language de   500
#   hard_ro  data/gmtw_ro_hard.jsonl        --language ro   300
#   hard_en  data/gmtw_ro_hard.jsonl        --language en   300
#
# There is no hard_de: the benchmark ships no German adversarial split.
# gmtw_ro_v0_DE_test.jsonl is gmtw_ro_v0.jsonl plus a prompt_de field -- same 500
# instance ids, byte-identical prompt_ro -- so the easy cells are matched across
# all three languages.
#
# Roots are derived from this script's location and every one is overridable,
# exactly as in run_family.sh:
#
#   BENCH   the gmtw-hria checkout     (default: <repo>/../gmtw-hria)
#   OUTDIR  where records land         (default: datagen/results)
#   BATCH   vLLM batch size            (default: 64, the base grid's value)
#   FORCE   1 to redo finished cells   (default: skip them, so a killed run resumes)
#   SETUP   shell hook sourced first   (conda activate, HF_HOME, ...)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN="$(dirname "$HERE")"
REPO="$(dirname "$DATAGEN")"

[ -n "${SETUP:-}" ] && . "$SETUP"

BENCH="${BENCH:-$REPO/../gmtw-hria}"
OUTDIR="${OUTDIR:-$DATAGEN/results}"
BATCH="${BATCH:-64}"

[ $# -eq 2 ] || { echo "usage: $0 <model> <name>" >&2; exit 1; }
MODEL="$1"; NAME="$2"
[ -d "$BENCH" ] || { echo "benchmark checkout not found: $BENCH" >&2; exit 1; }

O="$OUTDIR/$NAME"; mkdir -p "$O"
echo "bench: $BENCH"
echo "model: $MODEL"
echo "out:   $O"

# What scored these numbers -- recorded, not asserted, because it varies by
# machine. Working copies of the harness have at times carried uncommitted
# scorer changes, and one of them (check_priority_slot_restriction) makes
# before-and-after scores non-comparable by its own docstring. A cell that does
# not record the commit AND the diff it was scored under cannot later be placed
# in a table with any other cell.
{
  echo "date            $(date -Is)"
  echo "model           $MODEL"
  echo "bench_head      $(git -C "$BENCH" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
  echo "bench_dirty     $(git -C "$BENCH" status --porcelain 2>/dev/null | wc -l) modified files"
  echo "bench_diff_sha  $(git -C "$BENCH" diff 2>/dev/null | sha256sum | cut -d' ' -f1)"
  echo "batch_size      $BATCH"
  echo "--- git diff --stat ---"
  git -C "$BENCH" diff --stat 2>/dev/null || true
} > "$O/provenance_${NAME}.txt"

cell () {  # <cell> <instances> <language>
  local cell="$1" inst="$2" lang="$3"
  local out="$O/${NAME}_${cell}_outputs.jsonl"
  local met="$O/${NAME}_${cell}.jsonl"

  if [ -s "$met" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $NAME / $cell : already scored, skipping (FORCE=1 to redo)"
    return
  fi

  echo "=================== $NAME / $cell : generate ==================="
  if [ -s "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  outputs present, reusing"
  else
    python scripts/run_vllm_batch.py "$inst" \
      --language "$lang" --model-path "$MODEL" \
      --output "$out" --batch-size "$BATCH"
  fi

  echo "=================== $NAME / $cell : score ==================="
  python scripts/evaluate_outputs.py "$inst" "$out" \
    --language "$lang" --save-metrics "$met" | tail -12
}

cd "$BENCH"
cell easy_ro data/gmtw_ro_v0.jsonl         ro
cell easy_en data/gmtw_ro_v0.jsonl         en
cell easy_de data/gmtw_ro_v0_DE_test.jsonl de
cell hard_ro data/gmtw_ro_hard.jsonl       ro
cell hard_en data/gmtw_ro_hard.jsonl       en
echo "GRID DONE: $NAME"
