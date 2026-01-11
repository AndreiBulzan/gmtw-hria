#!/bin/bash
# GMTW-Ro Evaluation Script (v3 - with correct chat templates)
# The run_vllm_batch.py script now auto-detects chat templates via tokenizer.apply_chat_template()

cd /DATA/andrei/gmtw-hria/gmtw-hria

echo "=== Re-running models with CORRECT chat templates (v3) ==="

# Models that need re-running (were using wrong Llama-3 template before):
# - Gemma models (need Gemma template)
# - Mistral models (need Mistral template)
# Note: RoLlama3.1 was already correct (Llama-3 template) but re-running for consistency

# --- Gemma-2-9B (base) ---
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl --model-path google/gemma-2-9b-it --output data/outputs_base_gemma2-9b_v3.jsonl --batch-size 64
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl --model-path google/gemma-2-9b-it --output data/outputs_hard_gemma2-9b_v3.jsonl --batch-size 64

# --- Gemma-7B (base) ---
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl --model-path google/gemma-7b-it --output data/outputs_base_gemma-7b_v3.jsonl --batch-size 64
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl --model-path google/gemma-7b-it --output data/outputs_hard_gemma-7b_v3.jsonl --batch-size 64

# --- RoGemma2-9B (Romanian-finetuned) ---
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl --model-path OpenLLM-Ro/RoGemma2-9b-Instruct-DPO --output data/outputs_base_rogemma2-9b_v3.jsonl --batch-size 64
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl --model-path OpenLLM-Ro/RoGemma2-9b-Instruct-DPO --output data/outputs_hard_rogemma2-9b_v3.jsonl --batch-size 64

# --- RoGemma-7B (Romanian-finetuned) ---
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl --model-path OpenLLM-Ro/RoGemma-7b-Instruct --output data/outputs_base_rogemma-7b_v3.jsonl --batch-size 64
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl --model-path OpenLLM-Ro/RoGemma-7b-Instruct --output data/outputs_hard_rogemma-7b_v3.jsonl --batch-size 64

# --- RoMistral-7B (Romanian-finetuned) ---
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl --model-path OpenLLM-Ro/RoMistral-7b-Instruct-DPO --output data/outputs_base_romistral-7b_v3.jsonl --batch-size 64
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl --model-path OpenLLM-Ro/RoMistral-7b-Instruct-DPO --output data/outputs_hard_romistral-7b_v3.jsonl --batch-size 64

# --- Mistral-7B (base - for comparison with RoMistral) ---
python scripts/run_vllm_batch.py data/gmtw_ro_v0.jsonl --model-path mistralai/Mistral-7B-Instruct-v0.2 --output data/outputs_base_mistral-7b_v3.jsonl --batch-size 64
python scripts/run_vllm_batch.py data/gmtw_ro_hard.jsonl --model-path mistralai/Mistral-7B-Instruct-v0.2 --output data/outputs_hard_mistral-7b_v3.jsonl --batch-size 64

echo ""
echo "=== Running evaluations ==="

# Evaluate all v3 outputs
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_base_gemma2-9b_v3.jsonl --save-metrics data/metrics_base_gemma2-9b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl data/outputs_hard_gemma2-9b_v3.jsonl --save-metrics data/metrics_hard_gemma2-9b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_base_gemma-7b_v3.jsonl --save-metrics data/metrics_base_gemma-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl data/outputs_hard_gemma-7b_v3.jsonl --save-metrics data/metrics_hard_gemma-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_base_rogemma2-9b_v3.jsonl --save-metrics data/metrics_base_rogemma2-9b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl data/outputs_hard_rogemma2-9b_v3.jsonl --save-metrics data/metrics_hard_rogemma2-9b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_base_rogemma-7b_v3.jsonl --save-metrics data/metrics_base_rogemma-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl data/outputs_hard_rogemma-7b_v3.jsonl --save-metrics data/metrics_hard_rogemma-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_base_romistral-7b_v3.jsonl --save-metrics data/metrics_base_romistral-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl data/outputs_hard_romistral-7b_v3.jsonl --save-metrics data/metrics_hard_romistral-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_base_mistral-7b_v3.jsonl --save-metrics data/metrics_base_mistral-7b_v3.jsonl
python scripts/evaluate_outputs.py data/gmtw_ro_hard.jsonl data/outputs_hard_mistral-7b_v3.jsonl --save-metrics data/metrics_hard_mistral-7b_v3.jsonl

echo ""
echo "=== ALL DONE ==="
echo "Results saved with _v3 suffix. Compare with _v2 to see template impact."
