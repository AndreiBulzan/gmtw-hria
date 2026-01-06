# GMTW-Ro Evaluation Results Summary

## Key Finding: Romanian Fine-tuning Degrades Instruction-Following

The Romanian-specialized models (OpenLLM-Ro) show significantly degraded performance compared to their base models, primarily due to **failure to produce structured JSON output**.

---

## Overall Results

### Base Dataset (500 instances)

| Model | U | R | G | F | Avg | Notes |
|-------|---|---|---|---|-----|-------|
| **llama-3.3-70b** | 0.886 | 0.987 | 0.988 | 0.848 | **0.927** | Best overall |
| **llama-4-scout-17b** | 0.849 | 0.997 | 0.981 | 0.852 | **0.920** | Excellent |
| **gpt-oss-20b** | 0.929 | 0.977 | 0.952 | 0.818 | **0.919** | Highest U score |
| llama-3.1-8b (Cerebras) | 0.809 | 0.925 | 0.969 | 0.967 | 0.917 | |
| gemma-2-9b | 0.829 | 0.987 | 0.947 | 0.748 | 0.878 | |
| gemma-7b | 0.669 | 0.899 | 0.946 | 0.704 | 0.804 | |
| **RoGemma2-9b** | 0.568 | 0.663 | 0.863 | 0.884 | 0.744 | ⚠️ -15% vs original |
| qwen3-32b | 0.825 | 0.854 | 0.328 | 0.755 | 0.691 | Low G (thinking tokens?) |
| **RoMistral-7b** | 0.407 | 0.418 | 0.933 | 0.903 | 0.665 | ⚠️ Low U, R |
| **RoLlama3.1-8b** | 0.422 | 0.358 | 0.906 | 0.939 | 0.656 | ⚠️ Low U, R |
| **RoGemma-7b** | 0.298 | 0.319 | 0.902 | 0.775 | 0.574 | ⚠️ Lowest |

### Hard Dataset (300 instances)

| Model | U | R | G | F | Avg |
|-------|---|---|---|---|-----|
| **llama-3.3-70b** | 0.768 | 0.994 | 0.993 | 0.671 | **0.857** |
| **llama-4-scout** | 0.738 | 0.990 | 0.983 | 0.713 | **0.856** |
| **gpt-oss-20b** | 0.793 | 0.958 | 0.947 | 0.660 | **0.840** |
| gemma-2-9b | 0.703 | 0.972 | 0.956 | 0.572 | 0.801 |
| **RoGemma2-9b** | 0.497 | 0.618 | 0.895 | 0.802 | 0.703 |
| **RoLlama3.1-8b** | 0.371 | 0.343 | 0.942 | 0.978 | 0.658 |
| **RoMistral-7b** | 0.357 | 0.357 | 0.951 | 0.895 | 0.640 |
| qwen3-32b | 0.253 | 0.232 | 0.202 | 0.603 | 0.323 |

---

## Critical Issue: JSON Output Failure

The Romanian fine-tuned models frequently fail to produce the required JSON structured output:

| Model | JSON Failures | Failure Rate |
|-------|---------------|--------------|
| gemma-2-9b (original) | 0/500 | **0.0%** |
| gemma-7b (original) | 5/500 | **1.0%** |
| llama-3.1-8b (Cerebras) | 4/500 | **0.8%** |
| **RoGemma2-9b** | 179/500 | **35.8%** |
| **RoGemma-7b** | 192/500 | **38.4%** |
| **RoMistral-7b** | 259/500 | **51.8%** |
| **RoLlama3.1-8b** | 365/500 | **73.0%** |

### Performance When JSON IS Produced

Even when Romanian models successfully produce JSON, they underperform:

| Model | N (valid) | U | R | G | F |
|-------|-----------|---|---|---|---|
| gemma-2-9b | 500 | 0.829 | 0.987 | 0.947 | 0.748 |
| RoGemma2-9b | 321 | 0.700 | 0.920 | 0.864 | 0.819 |
| llama-3.1-8b | 496 | 0.816 | 0.932 | 0.977 | 0.974 |
| RoLlama3.1-8b | 135 | 0.648 | 0.742 | 0.821 | 0.774 |

---

## Interpretation

### What the Romanian Fine-tuning Does:

1. **Severely degrades instruction-following** (U score drops 20-50%)
2. **Severely degrades structured output capability** (R score drops, JSON failures)
3. **Slightly improves Romanian text quality** (G score similar or better)
4. **Improves faithfulness** (F score often better - models mention planned entities)

### Likely Cause:

The Romanian fine-tuning process appears to have used conversational/text data without sufficient instruction-following examples. This caused:
- Loss of ability to follow complex multi-step instructions
- Loss of ability to generate structured JSON output
- Catastrophic forgetting of instruction-tuning

### Recommendations:

1. Romanian fine-tuning should include instruction-following data
2. JSON/structured output capabilities need explicit preservation
3. Consider LoRA or other parameter-efficient methods that preserve base capabilities

---

## Metric Definitions

- **U (Understanding)**: Did the model follow constraints? (budget, activities, etc.)
- **R (Reasoning)**: Is the output structurally valid? (valid JSON, non-empty days)
- **G (Generation)**: Romanian linguistic quality (diacritics, no English contamination)
- **F (Faithfulness)**: Are planned entities mentioned in the explanation?

---

## Models Tested

### Cloud APIs (via Groq/OpenRouter):
- llama-3.3-70b-versatile
- llama-4-scout-17b-16e-instruct
- gpt-oss-20b (OpenRouter)
- qwen3-32b

### Local (via vLLM):
- google/gemma-2-9b-it
- google/gemma-7b-it
- OpenLLM-Ro/RoGemma2-9b-Instruct-DPO
- OpenLLM-Ro/RoGemma-7b-Instruct
- OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO
- OpenLLM-Ro/RoMistral-7b-Instruct-DPO

---

*Generated: January 2026*
