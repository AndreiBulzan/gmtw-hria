#!/usr/bin/env bash
# Regenerate one released checkpoint under both stopping conditions.
#
# The released RoGemma2 checkpoints ship eos_token_id=1 (<eos>, from
# pretraining) while their chat template ends a turn with <end_of_turn> (107).
# vLLM stops on the former, the model emits the latter, so generation runs past
# the answer into degenerate repetition until max_tokens and is cut off
# mid-JSON. Scored as-is that reads as a model unable to close a brace, and it
# inflates every repair delta measured against such a baseline.
#
# BOTH conditions are generated here rather than reusing the committed
# as-shipped outputs. Those were produced months ago on different hardware and
# a different vLLM; comparing a fresh aligned run against them would confound
# the stop token with environment drift. Generating the pair in one session
# isolates the stop token, and the fresh-vs-committed shipped comparison then
# measures the drift on its own.
#
#   regen_stop_token.sh <hf-model-id> <short-name> <cuda-device>
#
# Example:
#   regen_stop_token.sh OpenLLM-Ro/RoGemma2-9b-Instruct-DPO rogemma2-dpo 0
#
# Uses ONE GPU. Run two of these concurrently on different devices to cover
# both checkpoints at once; do not raise tensor-parallelism, a 9B model in
# bf16 is ~18.5GB and an H200 has 143GB.
set -euo pipefail

MODEL="${1:?usage: regen_stop_token.sh <hf-model-id> <short-name> <cuda-device>}"
NAME="${2:?missing short name}"
DEV="${3:?missing cuda device index}"

BENCH="${BENCH:-$HOME/gmtw-hria}"
PY="${PY:-$HOME/envs/kd2/bin/python}"
OUT="${OUT:-$HOME/regen}"
# 0.9 is the vLLM default and is what we want: after the weights load, the
# remainder is KV cache, and KV cache is concurrency. The as-shipped condition
# runs almost every sequence to the full 4096 tokens, which is precisely the
# case a small cache punishes.
GPU_FRAC="${GPU_FRAC:-0.9}"
# vLLM's compile path has Triton build a small CUDA helper with gcc, which needs
# Python.h. The system python3.12 has no headers and there is no root on this
# machine, so the interpreter is a standalone CPython fetched with uv, which
# ships its own -- see PY below. With headers present the compiled path works
# and is the faster one, so eager is off. Set EAGER=1 to fall back.
EAGER="${EAGER:-0}"
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
# NOT setting TORCHDYNAMO_DISABLE here, deliberately. The retention runner sets
# it because transformers wraps Gemma-2's forward in torch.compile(fullgraph)
# for its hybrid cache and dynamo tracing was unavailable there. This script
# never calls transformers.generate -- it is vLLM throughout -- and setting it
# breaks vLLM outright: compile is attempted, dynamo is off, and engine init
# dies with "aot_compile is not supported by the current configuration".
# vLLM routes sampling through FlashInfer, which JIT-builds a C++ extension on
# first use -- needing ninja and Python.h, neither of which is present and both
# of which need root to add. Decoding here is greedy (temperature 0), so the
# optimised sampler buys nothing: fall back to the native PyTorch path.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$DEV"

echo "=============================================================="
echo "model:  $MODEL"
echo "name:   $NAME"
echo "device: cuda:$DEV   (gpu_memory_utilization=$GPU_FRAC)"
echo "out:    $OUT"
echo "=============================================================="

for split in 500 300; do
  case "$split" in
    500) INST="$BENCH/data/gmtw_ro_v0.jsonl" ;;
    300) INST="$BENCH/data/gmtw_ro_hard.jsonl" ;;
  esac

  for cond in shipped aligned; do
    flag=""
    [ "$cond" = "aligned" ] && flag="--align-stop-token"
    stem="$OUT/${NAME}_${cond}_${split}"

    if [ -s "${stem}.jsonl" ]; then
      echo "--- $NAME $cond $split : already scored, skipping"
      continue
    fi

    echo "--- $NAME $cond $split : generate"
    ( cd "$BENCH" && "$PY" scripts/run_vllm_batch.py "$INST" \
        --model-path "$MODEL" \
        --output "${stem}_outputs.jsonl" \
        --gpu-memory-utilization "$GPU_FRAC" \
        $EAGER_FLAG $flag )

    echo "--- $NAME $cond $split : score"
    ( cd "$BENCH" && "$PY" scripts/evaluate_outputs.py "$INST" \
        "${stem}_outputs.jsonl" --save-metrics "${stem}.jsonl" | tail -10 )
  done
done

# Provenance, so a row without it cannot be placed beside another.
{
  echo "model         $MODEL"
  echo "device        $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEV" 2>/dev/null | head -1)"
  echo "gpu_mem_util  $GPU_FRAC"
  echo "enforce_eager $EAGER"
  echo "fi_sampler    $VLLM_USE_FLASHINFER_SAMPLER"
  echo "date          $(date -Is)"
  echo "--- versions ---"
  "$PY" - <<'PY'
import torch, transformers, vllm
print("torch        ", torch.__version__)
print("transformers ", transformers.__version__)
print("vllm         ", vllm.__version__)
PY
  echo "--- bench commit ---"
  git -C "$BENCH" rev-parse --short HEAD 2>/dev/null || echo "(not a git checkout)"
} > "$OUT/provenance_${NAME}.txt"

echo "=============================================================="
echo "DONE: $NAME  ->  $OUT"
cat "$OUT/provenance_${NAME}.txt"
