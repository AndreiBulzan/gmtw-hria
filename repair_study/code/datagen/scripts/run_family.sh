#!/usr/bin/env bash
# Train one model family across seeds, end to end, on one GPU.
#
#   datagen/scripts/run_family.sh <config.yaml> <name-prefix> <seed> [seed...]
#   datagen/scripts/run_family.sh datagen/training/lora_rogemma2_9b_v2.yaml rogemma2-dpo 0 1 2
#
# Per seed: train -> merge -> align stop tokens -> generate both GMTW-Ro splits
# -> score both.
#
# NOTHING is hardcoded to one machine. Every root is derived from this script's
# own location, and each can be overridden:
#
#   BENCH   the gmtw-hria checkout        (default: <repo>/../gmtw-hria)
#   MIX     training corpus               (default: datagen/data/sft/train_mix_4k_v2.jsonl)
#   VAL     held-out slice                (default: <MIX minus .jsonl>.val.jsonl)
#   OUTDIR  where scored records land     (default: datagen/results)
#   WORKDIR adapters and merged weights   (default: <repo>, which gitignores both)
#   SETUP   optional shell hook sourced before anything (conda activate, HF_HOME,
#           ...). Machine-specific setup belongs here, not in this file.
#
# Strictly sequential, and deliberately so. Training and vLLM cannot share the
# card, and three concurrent trainings on one GPU do not finish sooner -- one
# run already saturates it, so parallel processes only time-slice while adding
# an OOM risk that can take all of them down together.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN="$(dirname "$HERE")"
REPO="$(dirname "$DATAGEN")"

[ -n "${SETUP:-}" ] && . "$SETUP"

BENCH="${BENCH:-$REPO/../gmtw-hria}"
MIX="${MIX:-$DATAGEN/data/sft/train_mix_4k_v2.jsonl}"
VAL="${VAL:-${MIX%.jsonl}.val.jsonl}"
OUTDIR="${OUTDIR:-$DATAGEN/results}"
WORKDIR="${WORKDIR:-$REPO}"

[ $# -ge 3 ] && [ -f "$1" ] || {
  echo "usage: $0 <config.yaml> <name-prefix> <seed> [seed...]" >&2; exit 1; }
CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; PREFIX="$2"; shift 2

for p in "$BENCH" "$MIX" "$VAL"; do
  [ -e "$p" ] || { echo "not found: $p" >&2; exit 1; }
done

echo "repo:   $REPO"
echo "bench:  $BENCH"
echo "config: $CONFIG"
echo "mix:    $MIX"
echo "out:    $OUTDIR"

for SEED in "$@"; do
  NAME="${PREFIX}_s${SEED}"
  OUT="$OUTDIR/$NAME"
  mkdir -p "$OUT"

  echo "=================== $NAME : train ==================="
  python "$DATAGEN/training/train_lora.py" --config "$CONFIG" \
    --data "$MIX" --val-data "$VAL" \
    --output "$WORKDIR/runs/$NAME" --run-name "$NAME" --seed "$SEED"

  # An adapter that trained nothing still merges and still serves; it just
  # reproduces the base model. Catch it here rather than in the scores.
  python - "$WORKDIR/runs/$NAME/final" <<'PY'
import sys, json, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "adapter_config.json").read_text())
print(f"  adapter targets {len(cfg.get('target_modules') or [])} module patterns")
PY

  echo "=================== $NAME : merge ==================="
  BASE=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['base_model'])")
  python "$DATAGEN/scripts/merge_lora.py" "$BASE" \
    "$WORKDIR/runs/$NAME/final" "$WORKDIR/merged_models/$NAME"

  # Align the merged model's stop tokens with what its chat template emits.
  # Gemma checkpoints ship eos_token_id=1 (<eos>) while their template ends a
  # turn with <end_of_turn>, so vLLM never stops: the model finishes its answer,
  # repeats it until max_tokens, and the last repetition is cut mid-JSON. That
  # looked exactly like a model unable to close a brace and was nearly written
  # up as a finding; on RoGemma2-DPO the one field was worth 55% -> 99.8% valid
  # JSON. No-op when already correct.
  python - "$WORKDIR/merged_models/$NAME" <<'EOSFIX'
import json, pathlib, sys
from transformers import AutoTokenizer
d = pathlib.Path(sys.argv[1]); gcf = d / "generation_config.json"
tok = AutoTokenizer.from_pretrained(str(d))
rendered = tok.apply_chat_template(
    [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    tokenize=False)
# The LAST token is often the newline after the turn marker (Gemma renders
# "...<end_of_turn>\n", where \n is id 108). Stopping on that would truncate
# every answer at its first line break -- worse than the bug being fixed. Take
# the last SPECIAL token, which is the real turn marker.
ids = tok.encode(rendered, add_special_tokens=False)
special = set(tok.all_special_ids) | set(
    tok.convert_tokens_to_ids(t) for t in getattr(tok, "additional_special_tokens", []))
marker = [i for i in ids if i in special]
if not marker:
    print("  stop token: no special token in the rendered template, leaving as-is")
    raise SystemExit(0)
last = marker[-1]
g = json.loads(gcf.read_text())
cur = g.get("eos_token_id")
allowed = cur if isinstance(cur, list) else [cur]
if last not in allowed:
    g["eos_token_id"] = sorted({*[a for a in allowed if a is not None], last})
    gcf.write_text(json.dumps(g, indent=2) + "\n")
    print("  stop token: %s -> %s (template ends with id %s)" % (cur, g["eos_token_id"], last))
else:
    print("  stop token: %s already covers the template id %s" % (cur, last))
EOSFIX

  echo "=================== $NAME : generate ==================="
  ( cd "$BENCH"
    python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl \
      --model-path "$WORKDIR/merged_models/$NAME" --output "$OUT/${NAME}_500_outputs.jsonl"
    python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl \
      --model-path "$WORKDIR/merged_models/$NAME" --output "$OUT/${NAME}_300_outputs.jsonl"

    echo "=================== $NAME : score ==================="
    python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl \
      "$OUT/${NAME}_500_outputs.jsonl" --save-metrics "$OUT/${NAME}_500.jsonl" | tail -12
    python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl \
      "$OUT/${NAME}_300_outputs.jsonl" --save-metrics "$OUT/${NAME}_300.jsonl" | tail -12 )
  echo "=================== $NAME : DONE ==================="
done
echo "ALL SEEDS DONE: $PREFIX"
