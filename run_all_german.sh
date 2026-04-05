#!/usr/bin/env bash

echo "=== GMTW TRILINGUAL EVALUATION (RO + EN + DE) ==="
echo "=== Using: data/gmtw_ro_v0_DE_test.jsonl       ==="

INSTANCES="data/gmtw_ro_v0_DE_test.jsonl"

mkdir -p new_data/outputs_de_test_ro
mkdir -p new_data/outputs_de_test_en
mkdir -p new_data/outputs_de_test_de

mkdir -p new_data/metrics_de_test_ro
mkdir -p new_data/metrics_de_test_en
mkdir -p new_data/metrics_de_test_de


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

generate=true

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
        # RO
        ################################
        python scripts/run_vllm_batch.py \
            "$INSTANCES" \
            --language ro \
            --model-path "$model" \
            --output new_data/outputs_de_test_ro/ro_${name}.jsonl \
            --batch-size 64

        ################################
        # EN
        ################################
        python scripts/run_vllm_batch.py \
            "$INSTANCES" \
            --language en \
            --model-path "$model" \
            --output new_data/outputs_de_test_en/en_${name}.jsonl \
            --batch-size 64

        ################################
        # DE
        ################################
        python scripts/run_vllm_batch.py \
            "$INSTANCES" \
            --language de \
            --model-path "$model" \
            --output new_data/outputs_de_test_de/de_${name}.jsonl \
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
    # RO
    ################################
    if [[ -f "new_data/outputs_de_test_ro/ro_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            "$INSTANCES" \
            new_data/outputs_de_test_ro/ro_${name}.jsonl \
            --language ro \
            --save-metrics new_data/metrics_de_test_ro/ro_${name}.jsonl
    else
        echo "  [SKIP] no outputs for ro: ro_${name}.jsonl"
    fi

    ################################
    # EN
    ################################
    if [[ -f "new_data/outputs_de_test_en/en_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            "$INSTANCES" \
            new_data/outputs_de_test_en/en_${name}.jsonl \
            --language en \
            --save-metrics new_data/metrics_de_test_en/en_${name}.jsonl
    else
        echo "  [SKIP] no outputs for en: en_${name}.jsonl"
    fi

    ################################
    # DE
    ################################
    if [[ -f "new_data/outputs_de_test_de/de_${name}.jsonl" ]]; then
        python scripts/evaluate_outputs.py \
            "$INSTANCES" \
            new_data/outputs_de_test_de/de_${name}.jsonl \
            --language de \
            --save-metrics new_data/metrics_de_test_de/de_${name}.jsonl
    else
        echo "  [SKIP] no outputs for de: de_${name}.jsonl"
    fi

done

echo ""
echo "======================================="
echo "=== ALL RUNS COMPLETED SUCCESSFULLY ==="
echo "======================================="

echo ""
echo "Results:"
echo "  Outputs:  new_data/outputs_de_test_{ro,en,de}/"
echo "  Metrics:  new_data/metrics_de_test_{ro,en,de}/"