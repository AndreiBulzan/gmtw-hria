#!/usr/bin/env bash

echo "=== GMTW FULL PIPELINE (RO + EN) ==="

mkdir -p new_data/outputs_easy_ro
mkdir -p new_data/outputs_easy_en
mkdir -p new_data/outputs_hard_ro
mkdir -p new_data/outputs_hard_en

mkdir -p new_data/metrics_easy_ro
mkdir -p new_data/metrics_easy_en
mkdir -p new_data/metrics_hard_ro
mkdir -p new_data/metrics_hard_en


models=(
"google/gemma-2-9b-it"
"google/gemma-7b-it"
"mistralai/Mistral-7B-Instruct-v0.2"
"OpenLLM-Ro/RoLlama2-7b-Instruct"
"OpenLLM-Ro/RoMistral-7b-Instruct-DPO"
"OpenLLM-Ro/RoLlama3-8b-Instruct-DPO"
"OpenLLM-Ro/RoGemma2-9b-Instruct-DPO"
"OpenLLM-Ro/RoGemma-7b-Instruct-DPO"
"OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO"
"meta-llama/Meta-Llama-3-8B-Instruct"
"meta-llama/Llama-3.1-8B-Instruct"
"meta-llama/Llama-2-7b-hf"
)

generate=false

model_name () {
    basename "$1" | tr '[:upper:]' '[:lower:]'
}


echo ""
echo "=== GENERATING OUTPUTS ==="

if [ "$generate" = true ]; then
    for model in "${models[@]}"; do

        name=$(model_name "$model")

        echo ""
        echo "Running model: $model"

        ################################
        # EASY RO
        ################################
        python scripts/run_vllm_batch.py \
            data/gmtw_ro_v0.jsonl \
            --language ro \
            --model-path "$model" \
            --output new_data/outputs_easy_ro/ro_${name}.jsonl \
            --batch-size 64

        ################################
        # EASY EN
        ################################
        python scripts/run_vllm_batch.py \
            data/gmtw_ro_v0.jsonl \
            --language en \
            --model-path "$model" \
            --output new_data/outputs_easy_en/en_${name}.jsonl \
            --batch-size 64

        ################################
        # HARD RO
        ################################
        python scripts/run_vllm_batch.py \
            data/gmtw_ro_hard.jsonl \
            --language ro \
            --model-path "$model" \
            --output new_data/outputs_hard_ro/ro_${name}.jsonl \
            --batch-size 64

        ################################
        # HARD EN
        ################################
        python scripts/run_vllm_batch.py \
            data/gmtw_ro_hard.jsonl \
            --language en \
            --model-path "$model" \
            --output new_data/outputs_hard_en/en_${name}.jsonl \
            --batch-size 64

    done
fi

################################
# EVALUATION
################################

echo ""
echo "=== RUNNING EVALUATIONS ==="

for model in "${models[@]}"; do

    name=$(model_name "$model")

    echo ""
    echo "Evaluating model: $model"

    ################################
    # EASY RO
    ################################
    if [[ -f "new_data/outputs_easy_ro/ro_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            data/gmtw_ro_v0.jsonl \
            new_data/outputs_easy_ro/ro_${name}.jsonl \
            --language ro \
            --save-metrics new_data/metrics_easy_ro/ro_${name}.jsonl
    else
        echo "  [SKIP] no outputs for easy_ro: ro_${name}.jsonl"
    fi

    ################################
    # EASY EN
    ################################
    if [[ -f "new_data/outputs_easy_en/en_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            data/gmtw_ro_v0.jsonl \
            new_data/outputs_easy_en/en_${name}.jsonl \
            --language en \
            --save-metrics new_data/metrics_easy_en/en_${name}.jsonl
    else
        echo "  [SKIP] no outputs for easy_en: en_${name}.jsonl"
    fi

    ################################
    # HARD RO
    ################################
    if [[ -f "new_data/outputs_hard_ro/ro_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            data/gmtw_ro_hard.jsonl \
            new_data/outputs_hard_ro/ro_${name}.jsonl \
            --language ro \
            --save-metrics new_data/metrics_hard_ro/ro_${name}.jsonl
    else
        echo "  [SKIP] no outputs for hard_ro: ro_${name}.jsonl"
    fi

    ################################
    # HARD EN
    ################################
    if [[ -f "new_data/outputs_hard_en/en_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            data/gmtw_ro_hard.jsonl \
            new_data/outputs_hard_en/en_${name}.jsonl \
            --language en \
            --save-metrics new_data/metrics_hard_en/en_${name}.jsonl
    else
        echo "  [SKIP] no outputs for hard_en: en_${name}.jsonl"
    fi

done

echo ""
echo "======================================="
echo "=== ALL RUNS COMPLETED SUCCESSFULLY ==="
echo "======================================="