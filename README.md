# GMTW-Ro: Grounded Multilingual Task Worlds for Romanian LLM Evaluation

A deterministic, human-free benchmark for evaluating Romanian language model capabilities through grounded task worlds with dual-channel outputs.

Developed by the **Hub Român de Inteligență Artificială (HRIA)**.

## Overview

GMTW-Ro addresses the "Knowledge-Behavior Gap" in multilingual LLM evaluation by testing models on realistic, constraint-satisfaction tasks in Romanian. Unlike traditional benchmarks that rely on multiple-choice questions or LLM-as-a-judge approaches, GMTW-Ro uses:

- **Grounded Task Worlds**: Fully-specified environments (travel, scheduling, fact retrieval, meal planning)
- **Dual-Channel Architecture**: Models produce both structured output (JSON) and natural language explanations
- **Deterministic Evaluation**: No human raters, no LLM judges, completely reproducible
- **Decomposed Metrics**: Separate scores for Understanding (U), Reasoning (R), Generation (G), and Faithfulness (F)

## Task Worlds

| World | Task | Sample Constraints |
|-------|------|-------------------|
| **Travel** | Plan multi-day itinerary | Budget limits, outdoor activity caps, family-friendly |
| **Schedule** | Organize calendar | Keep high-priority, max per day, no overlaps |
| **Fact** | Answer from context | Context adherence (tests parametric vs. grounded knowledge) |
| **Recipe** | Plan daily menus | Vegetarian, gluten-free, calorie limits |

## Installation

```bash
cd rombench
pip install -e .
```

Required dependencies will be installed automatically: `ortools`, `rapidfuzz`, `json-repair`, `pydantic`, `requests`, and others.

## Quick Start

### 1. Generate Evaluation Questions

Create a dataset of 30 instances across all world types:

```bash
python scripts/generate_gmtw_v0.py \
  --num-travel 10 \
  --num-schedule 10 \
  --num-fact 5 \
  --num-recipe 5 \
  --output test.jsonl
```

For a full benchmark dataset (500+ instances):

```bash
python scripts/generate_gmtw_v0.py \
  --num-travel 150 \
  --num-schedule 150 \
  --num-fact 100 \
  --num-recipe 100 \
  --output data/gmtw_ro_v0/instances.jsonl
```

### 2. View Generated Questions

```bash
python scripts/view_instances.py test.jsonl
```

This displays each instance with:
- World type and difficulty
- Available entities (attractions, appointments, facts)
- Constraints to satisfy
- Full Romanian prompt

Press Enter to navigate through instances, 'q' to quit.

### 3. Test with Dummy Data

Generate synthetic outputs that satisfy most constraints (useful for testing the evaluator):

```bash
python scripts/create_dummy_outputs.py test.jsonl --output dummy_outputs.jsonl
```

Evaluate the dummy outputs:

```bash
python scripts/evaluate_outputs.py test.jsonl dummy_outputs.jsonl
```

Expected results: High scores (U≈0.83, R≈0.97, G≈1.0, F≈1.0) showing the evaluator works correctly.

## Running Models

### Using Groq API

#### Setup

Set your API key (only needed once per session):

```bash
# Windows
set GROQ_API_KEY=your_api_key_here

# Linux/Mac
export GROQ_API_KEY=your_api_key_here
```

Or pass it directly with `--api-key` flag.

#### Run Evaluation

```bash
# Full evaluation with 70B model
python scripts/run_groq_batch.py test.jsonl --output groq_70b.jsonl

# Faster 8B model
python scripts/run_groq_batch.py test.jsonl \
  --model llama-3.1-8b-instant \
  --output groq_8b.jsonl

# Test on just 5 instances first
python scripts/run_groq_batch.py test.jsonl \
  --model llama-3.3-70b-versatile \
  --output groq_test.jsonl \
  --max 5
```

Available models:
- `llama-3.3-70b-versatile` (default, best quality)
- `llama-3.1-8b-instant` (fastest, cheapest)
- `openai/gpt-oss-20b` (middle ground)
- `openai/gpt-oss-120b` (highest quality)

Options:
- `--delay`: Delay in seconds between API calls (default: 2.0). Adjust if you hit rate limits:
  ```bash
  python scripts/run_groq_batch.py test.jsonl --delay 5.0
  ```

### Using Custom Models

Create a JSONL file with model outputs in this format:

```json
{"instance_id": "travel_000000", "output": "Model's Romanian response with JSON here..."}
{"instance_id": "travel_000001", "output": "Another response..."}
```

The output should contain:
1. Romanian explanation (2-3 paragraphs)
2. JSON plan at the end

Example format:
```
Pentru călătoria de 3 zile în Cluj-Napoca, propun următorul plan:

[Explanation in Romanian...]

{
  "day1": ["Muzeul Național de Artă", "Cetățuia"],
  "day2": ["Grădina Botanică"]
}
```

## Evaluation

### Basic Evaluation

```bash
python scripts/evaluate_outputs.py test.jsonl groq_70b.jsonl
```

Output:
```
Loaded 25 instances
Loaded 25 outputs
✓ travel_000002: U=1.00 R=1.00 G=1.00 F=1.00
✗ travel_000000: U=1.00 R=1.00 G=1.00 F=0.67
...

============================================================
AVERAGE SCORES (25 instances)
============================================================
  U (Understanding): 0.840
  R (Reasoning):     0.920
  G (Generation):    0.992
  F (Faithfulness):  0.739
============================================================
```

### Save Detailed Metrics

```bash
python scripts/evaluate_outputs.py test.jsonl groq_70b.jsonl \
  --save-metrics detailed_metrics.jsonl
```

This creates a JSONL file with complete breakdowns for each instance.

## Visualizing Results

### View All Results

```bash
python scripts/debug_results.py test.jsonl groq_70b.jsonl
```

Shows for each instance:
- Actual prompt used
- Model output
- Scores (U/R/G/F)
- Detailed failure analysis (which constraints violated, which entities missing)

### View Only Failures

```bash
python scripts/debug_results.py test.jsonl groq_70b.jsonl --filter failed
```

### View Only Passes

```bash
python scripts/debug_results.py test.jsonl groq_70b.jsonl --filter passed
```

### Inspect Specific Outputs

```bash
# View first output
python scripts/peek_output.py groq_70b.jsonl 1

# View 10th output
python scripts/peek_output.py groq_70b.jsonl 10
```

## Comparing Multiple Models

### Run Multiple Models

```bash
# 70B model
python scripts/run_groq_batch.py test.jsonl \
  --model llama-3.3-70b-versatile \
  --output groq_70b.jsonl

# 8B model
python scripts/run_groq_batch.py test.jsonl \
  --model llama-3.1-8b-instant \
  --output groq_8b.jsonl

# GPT OSS 20B
python scripts/run_groq_batch.py test.jsonl \
  --model openai/gpt-oss-20b \
  --output groq_20b.jsonl
```

### Compare Side-by-Side

```bash
python scripts/compare_models.py test.jsonl groq_70b.jsonl groq_8b.jsonl groq_20b.jsonl
```

Output:
```
================================================================================
MODEL COMPARISON
================================================================================

Model                               U        R        G        F      Avg
--------------------------------------------------------------------------------
llama-3.3-70b-versatile          0.840    0.920    0.992    0.739    0.873
llama-3.1-8b-instant             0.720    0.780    0.950    0.620    0.768
openai/gpt-oss-20b               0.780    0.850    0.970    0.680    0.820
================================================================================

Per-Instance Breakdown:
--------------------------------------------------------------------------------

travel_000000 (travel)
  ✓ llama-3.3-70b      U=1.00 R=1.00 G=1.00 F=1.00
  ✗ llama-3.1-8b       U=0.50 R=0.67 G=1.00 F=0.89
  ✓ openai/gpt-oss     U=1.00 R=1.00 G=0.98 F=0.95
...
```

## Understanding the Metrics

### U - Understanding Score

Measures whether the model correctly understood the Romanian constraints.

- **1.0**: All constraints satisfied
- **0.5**: Half the constraints satisfied
- **0.0**: No constraints satisfied

**Low U indicates**: Model failed to parse or understand Romanian instructions.

### R - Reasoning Score

Measures whether the plan is logically valid within the world's rules.

- **1.0**: All structural goals satisfied (no logical errors)
- **0.0**: Plan is logically invalid

**Low R indicates**: Model has reasoning deficits (impossible plans, empty days, invalid entity IDs).

### G - Generation Quality Score

Measures linguistic quality of the Romanian text using a deterministic, lexicon-based analyzer.

Components:
- **G_dia (50%)**: Diacritic correctness — checks ~200 high-frequency words that must have diacritics (și, în, țară, fără...)
- **G_cs (30%)**: Code-switch detection — flags English contamination while excluding Romanian lookalikes
- **G_len (20%)**: Length adequacy — penalizes suspiciously short responses

**Low G indicates**: Missing diacritics, English contamination, or degenerate output.

### F - Faithfulness Score

Measures consistency between the JSON plan and the Romanian explanation.

- **1.0**: Perfect consistency (all planned entities mentioned in explanation)
- **0.0**: Complete mismatch

**Low F indicates**: The model's explanation doesn't match what it actually planned (hallucination or incomplete descriptions).

## Repository Structure

```
rombench/
├── rombench/
│   ├── gmtw_ro/                    # Core implementation
│   │   ├── worlds/                 # World generators
│   │   │   ├── base.py             # Data models
│   │   │   ├── travel.py           # Travel (6 cities, 37 attractions)
│   │   │   ├── schedule.py         # Schedule (calendar management)
│   │   │   ├── fact.py             # Fact (context adherence, misbelief traps)
│   │   │   ├── recipe.py           # Recipe (meal planning, dietary constraints)
│   │   │   ├── templates_ro.py     # Romanian prompts
│   │   │   └── templates_en.py     # English prompts
│   │   └── eval/                   # Evaluation tools
│   │       ├── parser.py           # Dual-channel parser
│   │       ├── faithfulness.py     # Deterministic F metric
│   │       ├── constraints.py      # Constraint checkers (20+ functions)
│   │       ├── metrics.py          # U/R/G/F metrics
│   │       └── scorer.py           # Main evaluator
│   └── nlp_ro/                     # Romanian NLP toolkit (84-word diacritic lexicon)
├── scripts/                        # CLI tools
├── data/                           # Generated datasets
├── examples/                       # Usage examples
└── tests/                          # Unit tests
```

## Example Workflow

Complete evaluation workflow:

```bash
# 1. Generate questions
python scripts/generate_gmtw_v0.py --num-travel 10 --num-schedule 10 --num-fact 5 --num-recipe 5 --output eval.jsonl

# 2. Preview questions
python scripts/view_instances.py eval.jsonl --max 3

# 3. Run model
python scripts/run_groq_batch.py eval.jsonl --output model_outputs.jsonl

# 4. Evaluate
python scripts/evaluate_outputs.py eval.jsonl model_outputs.jsonl

# 5. Debug failures
python scripts/debug_results.py eval.jsonl model_outputs.jsonl --filter failed

# 6. Save detailed metrics
python scripts/evaluate_outputs.py eval.jsonl model_outputs.jsonl --save-metrics metrics.jsonl
```

## Advanced Usage

### Running the Built-in Example

```bash
python examples/simple_example.py
```

This demonstrates the complete pipeline: world generation, prompt creation, simulated output, and evaluation.

### Batch Processing

For large-scale evaluation, process in batches:

```bash
# Generate large dataset (500 instances)
python scripts/generate_gmtw_v0.py \
  --num-travel 150 \
  --num-schedule 150 \
  --num-fact 100 \
  --num-recipe 100 \
  --output full_benchmark.jsonl

# Run model
python scripts/run_groq_batch.py full_benchmark.jsonl --output results.jsonl

# Evaluate
python scripts/evaluate_outputs.py full_benchmark.jsonl results.jsonl --save-metrics full_metrics.jsonl
```

## Cross-Lingual Evaluation (Delta Metric)

The benchmark supports computing the Foreign Language Penalty (Δ) to measure performance degradation when models process Romanian vs English prompts.

### How Delta Works

Run the same instances through a model twice:
1. With Romanian prompts (native benchmark)
2. With English prompts (translated entity names, diacritic-free)

Then compare:
```bash
python scripts/compute_delta.py instances.jsonl ro_outputs.jsonl en_outputs.jsonl
```

### English Prompt Features

- **Travel**: City names use ASCII (Brasov, Timisoara, Iasi), attraction names fully translated (Biserica Neagra → The Black Church)
- **Recipe**: Dish names fully translated (Oua jumari cu rosii → Scrambled Eggs with Tomatoes), JSON keys use English (day1_breakfast vs day1_mic_dejun)
- **Schedule**: Already supported via days_en/slots_en/name_en fields
- **Fact**: Questions in English, answers remain as-is (facts about Romania)

### Delta Interpretation

- **ΔU, ΔR, ΔF**: Compare these across languages (positive = worse in Romanian)
- **G metric**: Only meaningful for Romanian outputs (measures diacritic correctness)

## Future Work

### Core Functionality

- **Add constraint solver**: Integrate OR-Tools CP-SAT to verify all generated worlds are mathematically solvable
- **Expand world types**: Add budget/shopping world, logistics planning
- **Grammar checking**: Integrate LanguageTool for deeper G metric analysis (optional)

### Evaluation Improvements

- **Test suite**: Unit tests for all constraint checkers and metrics
- **CLI runner**: Unified command-line interface for batch evaluation
- **Aggregation tools**: Statistical analysis, confidence intervals, difficulty stratification
- **Leaderboard**: Public benchmark results for Romanian LLMs

### RoWriteBench

- **Implement creative writing benchmark**: Micro-essays, register-shift rewriting, summarization tasks
- **Deterministic stylometry**: Lexical diversity, discourse markers, readability metrics
- **Register detection**: Formal/informal tone analysis using lexicons

### Infrastructure

- **Dataset versioning**: Freeze benchmark versions for reproducible comparisons
- **Contamination resistance**: Procedural world generation to prevent training set leakage
- **Multi-language expansion**: Extend framework to other mid-resource languages

## Citation

```bibtex
@misc{gmtwro2025,
  title={GMTW-Ro: Grounded Multilingual Task Worlds for Romanian LLM Evaluation},
  author={GMTW-Ro Team, Hub Român de Inteligență Artificială (HRIA)},
  year={2025},
  url={https://github.com/AndreiBulzan/gmtw-hria}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue to discuss major changes before submitting pull requests.

Key areas for contribution:
- Additional world types and constraints
- Expanded diacritic lexicon coverage
- Test coverage
- Documentation improvements
- Bug reports and evaluation edge cases
