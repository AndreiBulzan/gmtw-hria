# GMTW-Ro: Grounded Multilingual Task Worlds for Romanian LLM Evaluation

A deterministic, human-free benchmark for evaluating Romanian language model capabilities through grounded task worlds with dual-channel outputs.

Developed by the **Hub Român de Inteligență Artificială (HRIA)**.

## Overview

GMTW-Ro addresses the "Knowledge-Behavior Gap" in multilingual LLM evaluation by testing models on realistic, constraint-satisfaction tasks in Romanian. Unlike traditional benchmarks that rely on multiple-choice questions or LLM-as-a-judge approaches, GMTW-Ro uses:

- **Grounded Task Worlds**: Fully-specified environments (travel, scheduling, fact retrieval, meal planning)
- **Dual-Channel Architecture**: Models produce both structured output (JSON) and natural language explanations
- **Deterministic Evaluation**: No human raters, no LLM judges, completely reproducible
- **Decomposed Metrics**: Separate scores for Understanding (U), Reasoning (R), Generation (G), and Faithfulness (F)

For detailed technical documentation, see [docs/GMTW_Ro_Technical_Specification.md](docs/GMTW_Ro_Technical_Specification.md).

## Task Worlds

| World | Task | Sample Constraints |
|-------|------|-------------------|
| **Travel** | Plan multi-day itinerary | Budget limits, outdoor activity caps, family-friendly |
| **Schedule** | Organize calendar | Keep high-priority, max per day, no overlaps |
| **Fact** | Answer from context | Context adherence (tests parametric vs. grounded knowledge) |
| **Recipe** | Plan daily menus | Vegetarian, gluten-free, calorie limits |

## Difficulty Levels

Each world supports three difficulty levels with progressively more constraints:

| Level | Travel | Schedule | Recipe | Fact |
|-------|--------|----------|--------|------|
| **Easy** | ~3 constraints | ~2 constraints | ~1 constraint | 0% misbelief traps |
| **Medium** | ~5-6 constraints | ~2 constraints | ~3-4 constraints | 40% misbelief traps |
| **Hard** | ~7-8 constraints | ~4-5 constraints | ~5-6 constraints | 80% misbelief traps |

**Hard mode adds:**
- **Travel**: Duration limits per day, type diversity requirements, type exclusions
- **Schedule**: No back-to-back appointments, total weekly limits, day restrictions
- **Recipe**: Prep time limits, lunch-must-be-heaviest, vegan requirements
- **Fact**: More facts to process, higher chance of context contradicting real-world knowledge

## Installation

```bash
cd rombench
pip install -e .
```

Required dependencies will be installed automatically: `json-repair`, `pydantic`, `requests`, and others.

---

## Quick Start: End-to-End Evaluation

### Step 1: Set Up API Key

Choose one provider:

**OpenRouter (recommended - free models available)**:
```bash
# Windows
set OPENROUTER_API_KEY=your_key_here

# Linux/Mac
export OPENROUTER_API_KEY=your_key_here
```
Get a free key at: https://openrouter.ai/keys

**Groq**:
```bash
# Windows
set GROQ_API_KEY=your_key_here

# Linux/Mac
export GROQ_API_KEY=your_key_here
```

### Step 2: Get the Dataset

**Option A**: Use the pre-generated v0 dataset (500 instances):
```bash
# Dataset included in data/gmtw_ro_v0.jsonl
```

**Option B**: Generate your own:
```bash
# Mixed difficulty (40% easy, 40% medium, 20% hard) - default
python scripts/generate_gmtw_v0.py --output data/instances.jsonl

# Hard mode only (most challenging)
python scripts/generate_gmtw_v0.py --difficulty hard --output data/instances_hard.jsonl

# Custom counts
python scripts/generate_gmtw_v0.py --num-travel 150 --num-schedule 150 --num-fact 100 --num-recipe 100 --output data/instances.jsonl
```

### Step 3: Run a Model

**Using OpenRouter (free models)**:
```bash
# Grok 4.1 Fast (default, recommended)
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --output data/outputs_grok.jsonl

# Or specify a different model
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --model google/gemma-3-12b-it:free --output data/outputs_gemma.jsonl

# Test on just 5 instances first
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --output data/test_outputs.jsonl --max 5
```

**Using Groq**:
```bash
python scripts/run_groq_batch.py data/gmtw_ro_v0.jsonl --model llama-3.3-70b-versatile --output data/outputs_llama70b.jsonl
```

### Step 4: Evaluate

```bash
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_grok.jsonl
```

Output:
```
============================================================
AVERAGE SCORES (500 instances)
============================================================
  U (Understanding): 0.847
  R (Reasoning):     0.923
  G (Generation):    0.891
  F (Faithfulness):  0.756
============================================================
```

### Step 5: Debug Failures

```bash
# View all results interactively
python scripts/debug_results.py data/gmtw_ro_v0.jsonl data/outputs_grok.jsonl

# View only failures
python scripts/debug_results.py data/gmtw_ro_v0.jsonl data/outputs_grok.jsonl --filter failed
```

---

## Using Custom Models / Your Own API

If you want to run a model not supported by the provided scripts, create a JSONL file with this format:

```json
{"instance_id": "travel_000000", "output": "Model's response here..."}
{"instance_id": "travel_000001", "output": "Another response..."}
```

### Required Output Format

**CRITICAL**: Models must produce output in this exact order:

1. **FIRST**: Romanian explanation (2-3 paragraphs)
2. **THEN**: JSON plan at the very end

```
Pentru călătoria de 3 zile în Cluj-Napoca, propun următorul plan:

Am ales să vizităm Grădina Botanică în prima zi deoarece...
[More explanation paragraphs...]

{
  "day1": ["Grădina Botanică", "Cetățuia"],
  "day2": ["Muzeul Național de Artă"],
  "day3": ["Parcul Central"]
}
```

**Models that put JSON before the explanation will be penalized** (format violation counts as a failed constraint in the U score).

---

## Available Models

### OpenRouter (Free Tier)

```bash
python scripts/run_openrouter_batch.py --list-models
```

| Model | Command |
|-------|---------|
| Grok 4.1 Fast (default) | `--model x-ai/grok-4.1-fast:free` |
| Gemma 3 12B | `--model google/gemma-3-12b-it:free` |
| Llama 3.1 8B | `--model meta-llama/llama-3.1-8b-instruct:free` |
| DeepSeek R1 | `--model deepseek/deepseek-r1:free` |

### Groq

| Model | Command |
|-------|---------|
| Llama 3.3 70B (best) | `--model llama-3.3-70b-versatile` |
| Llama 3.1 8B (fastest) | `--model llama-3.1-8b-instant` |

---

## Understanding the Metrics

### U - Understanding Score

Measures whether the model correctly understood and followed the Romanian constraints.

- **1.0**: All constraints satisfied + correct output format
- **< 1.0**: One or more constraints violated

**Includes**: Explicit constraints (e.g., "include a monument") + format compliance (JSON at end)

### R - Reasoning Score

Measures whether the plan is structurally valid.

- **1.0**: All structural goals satisfied
- **< 1.0**: Plan has logical errors (empty days, invalid entity references)

### G - Generation Quality Score

Measures linguistic quality of the Romanian text (deterministic, lexicon-based).

Components:
- **G_dia (50%)**: Diacritic correctness — checks 84 high-frequency words that must have diacritics (și, în, fără, când...)
- **G_cs (30%)**: Code-switch detection — flags English contamination
- **G_len (20%)**: Length adequacy — penalizes suspiciously short responses

### F - Faithfulness Score

Measures consistency between the JSON plan and the explanation.

- **1.0**: All planned entities mentioned in explanation
- **0.0**: Complete mismatch (often indicates format violation where explanation is empty)

---

## Detailed Workflows

### Save Detailed Metrics

```bash
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_grok.jsonl --save-metrics data/metrics_grok.jsonl
```

### Compare Multiple Models

```bash
# Run multiple models
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --model x-ai/grok-4.1-fast:free --output data/outputs_grok.jsonl
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --model google/gemma-3-12b-it:free --output data/outputs_gemma.jsonl

# Compare side-by-side
python scripts/compare_models.py data/gmtw_ro_v0.jsonl data/outputs_grok.jsonl data/outputs_gemma.jsonl
```

### Cross-Lingual Evaluation (Delta Metric)

Measure the "foreign language penalty" by comparing Romanian vs English performance:

```bash
# Run with Romanian prompts (default)
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --output data/outputs_ro.jsonl

# Run with English prompts
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --language en --output data/outputs_en.jsonl

# Compute Delta
python scripts/compute_delta.py data/gmtw_ro_v0.jsonl data/outputs_ro.jsonl data/outputs_en.jsonl
```

English prompts use fully translated entity names (e.g., "The Black Church" instead of "Biserica Neagră").

---

## Repository Structure

```
rombench/
├── data/
│   └── gmtw_ro_v0.jsonl          # Pre-generated 500-instance dataset
├── docs/
│   └── GMTW_Ro_Technical_Specification.md
├── rombench/
│   ├── gmtw_ro/
│   │   ├── worlds/               # World generators
│   │   │   ├── travel.py         # 6 cities, 37 attractions
│   │   │   ├── schedule.py       # Calendar management
│   │   │   ├── fact.py           # Context adherence, misbelief traps
│   │   │   ├── recipe.py         # 19 dishes, dietary constraints
│   │   │   ├── templates_ro.py   # Romanian prompts
│   │   │   └── templates_en.py   # English prompts
│   │   └── eval/                 # Evaluation tools
│   │       ├── parser.py         # Dual-channel parser
│   │       ├── constraints.py    # 20+ constraint checkers
│   │       ├── metrics.py        # U/R/G/F computation
│   │       └── faithfulness.py   # Deterministic F metric
│   └── nlp_ro/                   # Romanian NLP toolkit
│       ├── diacritics.py         # 84-word diacritic lexicon
│       └── codeswitch.py         # English contamination detector
├── scripts/
│   ├── generate_gmtw_v0.py       # Generate instances
│   ├── run_openrouter_batch.py   # Run via OpenRouter API
│   ├── run_groq_batch.py         # Run via Groq API
│   ├── evaluate_outputs.py       # Compute metrics
│   ├── debug_results.py          # Interactive result viewer
│   ├── compare_models.py         # Side-by-side comparison
│   └── compute_delta.py          # Cross-lingual analysis
└── commands.txt                  # Copy-paste commands for Windows
```

---

## Citation

```bibtex
@misc{gmtwro2025,
  title={GMTW-Ro: Grounded Multilingual Task Worlds for Romanian LLM Evaluation},
  author={Hub Român de Inteligență Artificială (HRIA)},
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
- Bug reports and evaluation edge cases
