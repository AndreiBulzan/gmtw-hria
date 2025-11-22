># GMTW-Ro Usage Guide

This guide provides step-by-step instructions for using the GMTW-Ro benchmark.

## Installation

```bash
cd rombench
pip install -e .
```

## Quick Start

### 1. Generate a Dataset

Generate a small test dataset:

```bash
python scripts/generate_gmtw_v0.py \
  --num-travel 20 \
  --num-schedule 20 \
  --num-fact 10 \
  --output data/gmtw_ro_v0/test_instances.jsonl
```

This creates 50 instances (20 travel, 20 schedule, 10 fact).

### 2. Run the Example

See how the benchmark works:

```bash
python examples/simple_example.py
```

This will:
- Generate a Travel World
- Create Romanian and English prompts
- Simulate a model output
- Evaluate it and show U/R/G/F scores

### 3. Evaluate Your Model

#### Step 3.1: Prepare Model Outputs

Your model outputs should be in JSONL format, with one output per instance:

```json
{"instance_id": "travel_000042", "output": "Explicația în română...\n\n{\"day1\": [...]}"}
{"instance_id": "schedule_000100", "output": "..."}
```

#### Step 3.2: Run Evaluation

```python
import json
from rombench.gmtw_ro import Instance, evaluate_instance

# Load instances
instances = {}
with open("data/gmtw_ro_v0/test_instances.jsonl") as f:
    for line in f:
        inst_data = json.loads(line)
        inst = Instance.from_dict(inst_data)
        instances[inst.instance_id] = inst

# Load model outputs
outputs = {}
with open("model_outputs.jsonl") as f:
    for line in f:
        data = json.loads(line)
        outputs[data["instance_id"]] = data["output"]

# Evaluate each
results = []
for inst_id, instance in instances.items():
    if inst_id in outputs:
        result = evaluate_instance(instance, outputs[inst_id])
        results.append(result.to_dict())

# Compute averages
avg_U = sum(r["U"] for r in results) / len(results)
avg_R = sum(r["R"] for r in results) / len(results)
avg_G = sum(r["G"] for r in results) / len(results)
avg_F = sum(r["F"] for r in results) / len(results)

print(f"Average Scores:")
print(f"  U: {avg_U:.3f}")
print(f"  R: {avg_R:.3f}")
print(f"  G: {avg_G:.3f}")
print(f"  F: {avg_F:.3f}")
```

### 4. Compute Foreign Language Penalty (Δ)

To compute the Δ metrics, you need to run your model on both Romanian and English prompts:

```python
# Run on Romanian prompts
ro_results = {}
for inst_id, instance in instances.items():
    # Get model output for Romanian prompt
    output_ro = your_model(instance.prompt_ro)
    result = evaluate_instance(instance, output_ro)
    ro_results[inst_id] = result

# Run on English prompts
en_results = {}
for inst_id, instance in instances.items():
    # Get model output for English prompt
    output_en = your_model(instance.prompt_en)
    result = evaluate_instance(instance, output_en)
    en_results[inst_id] = result

# Compute deltas
delta_U = sum(en_results[i].U - ro_results[i].U for i in instances.keys()) / len(instances)
delta_R = sum(en_results[i].R - ro_results[i].R for i in instances.keys()) / len(instances)
delta_F = sum(en_results[i].F - ro_results[i].F for i in instances.keys()) / len(instances)

print(f"Foreign Language Penalty:")
print(f"  ΔU: {delta_U:.3f}")
print(f"  ΔR: {delta_R:.3f}")
print(f"  ΔF: {delta_F:.3f}")
```

## Understanding the Metrics

### U - Understanding Score

Measures whether the model correctly understood the Romanian constraints.

- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 1.0 = All constraints satisfied
  - 0.5 = Half the constraints satisfied
  - 0.0 = No constraints satisfied

**Low U score indicates**: The model failed to parse or understand Romanian instructions.

### R - Reasoning Score

Measures whether the plan is logically valid within the world.

- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 1.0 = All structural goals satisfied (no logical errors)
  - 0.0 = Plan is logically invalid

**Low R score indicates**: The model has reasoning deficits (temporal conflicts, impossible plans).

### G - Generation Quality Score

Measures linguistic hygiene of the Romanian text.

- **Range**: 0.0 to 1.0
- **Components**:
  - Grammar correctness
  - Diacritic usage (ă, â, î, ș, ț)
  - Code-switching (mixing English)

**Low G score indicates**: Poor Romanian generation (grammar errors, missing diacritics, English words).

### F - Faithfulness Score

Measures consistency between the JSON plan and the Romanian explanation.

- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 1.0 = Perfect consistency (all planned entities mentioned, no hallucinations)
  - 0.0 = Complete mismatch

**Low F score indicates**: Hallucination or incomplete explanations.

### Δ - Foreign Language Penalty

Measures performance degradation from English to Romanian.

- **Calculation**: Δ_metric = metric_en - metric_ro
- **Interpretation**:
  - Δ = 0 → No penalty, equal performance
  - Δ > 0 → Model worse in Romanian
  - Δ < 0 → Model better in Romanian (rare)

**High Δ indicates**: The model's capabilities degrade significantly when operating in Romanian.

## Advanced Usage

### Generating Custom Worlds

```python
from rombench.gmtw_ro.worlds import TravelWorldGenerator

generator = TravelWorldGenerator()
world = generator.generate(
    world_id="custom_001",
    seed=123,
    difficulty="medium"
)

# Inspect the world
print(f"City: {world.payload['city']}")
print(f"Attractions: {len(world.canonical_entities)}")
print(f"Constraints: {[c.description_ro for c in world.constraints]}")
```

### Fine-Grained Analysis

```python
result = evaluate_instance(instance, output)

# Check which specific constraints failed
for constraint in result.U_details['constraints']:
    if not constraint['satisfied']:
        print(f"Failed: {constraint['description']}")

# Check faithfulness details
if result.F < 0.8:
    print(f"Missing entities: {result.F_details['missing']}")
    print(f"Hallucinated entities: {result.F_details['extra']}")
```

## Tips for Model Developers

1. **Start with Travel Worlds**: They're the most intuitive and have clear constraints.

2. **Monitor U vs R**: If U is high but R is low, your model understands Romanian but can't reason. If both are low, start with improving language understanding.

3. **Check F score**: Low F indicates hallucinations, a critical safety issue.

4. **Compute Δ metrics**: Quantify how much your Romanian fine-tuning helps (or hurts).

5. **Iterate**: Use GMTW-Ro during training to monitor improvements in real-time.

## Next Steps

- **Full Dataset**: Generate 500-1000 instances for robust evaluation
- **NLP Integration**: Integrate Romanian NLP tools (Stanza) for better G scores
- **Continuous Evaluation**: Run GMTW-Ro after each training epoch
- **RoWriteBench**: Explore the companion writing quality benchmark
