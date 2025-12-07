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

## Installation

### Standard (recommended)
```bash
pip install -e .
```

### With Optional Enhancements
```bash
# Add grammar checking (LanguageTool)
pip install -e ".[grammar]"

# Add Stanza lemmatization
pip install -e ".[stanza]"

# Install everything
pip install -e ".[full]"
```

---

## Quick Start

### 1. Set Up API Key

```bash
# OpenRouter (free models available) - get key at https://openrouter.ai/keys
export OPENROUTER_API_KEY=your_key_here

# Or Groq
export GROQ_API_KEY=your_key_here
```

### 2. Run a Model

```bash
# Test on 5 instances first
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --output data/outputs.jsonl --max 5

# Full dataset (500 instances)
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --output data/outputs.jsonl
```

### 3. Evaluate

```bash
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl
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

### 4. Debug Failures

```bash
python scripts/debug_results.py data/gmtw_ro_v0.jsonl data/outputs.jsonl --filter failed
```

---

## Understanding the Metrics

### U - Understanding Score

Did the model follow the Romanian constraints?

- Checks explicit constraints (e.g., "include a monument", "max 2 outdoor activities")
- Checks format compliance (JSON must be at the end)

### R - Reasoning Score

Is the plan structurally valid?

- Valid entity references
- Non-empty days
- Logical consistency

### G - Generation Quality Score

Linguistic quality of the Romanian text (deterministic, lexicon-based).

**Standard mode:**
| Component | Weight | What it checks |
|-----------|--------|----------------|
| G_dia | 50% | Diacritic correctness (și, în, fără...) |
| G_cs | 30% | English contamination detection |
| G_len | 20% | Response length adequacy |

**With `--use-languagetool`:**
| Component | Weight | What it checks |
|-----------|--------|----------------|
| G_dia | 40% | Diacritic correctness |
| G_cs | 20% | English contamination |
| G_len | 15% | Response length |
| G_grammar | 25% | Grammar/spelling via LanguageTool |

### F - Faithfulness Score

Are all planned entities mentioned in the explanation?

- Checks that every entity in the JSON plan appears in the text
- Uses Romanian morphology to match inflected forms (muzeul → muzeu)

---

## Optional Enhancements

Two optional flags enable deeper analysis at the cost of speed:

### `--use-languagetool`

Adds **G_grammar** component to the G score using LanguageTool for grammar/spelling checking.

```bash
pip install language-tool-python
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl --use-languagetool
```

**What it catches:** Grammar errors, spelling mistakes, punctuation issues.

**Note:** Proper nouns (names, places) are automatically filtered to avoid false positives.

### `--use-stanza`

Uses **Stanza** (Stanford NLP) for Romanian lemmatization in the F score.

```bash
pip install stanza
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl --use-stanza
```

**What it does:** More accurate entity matching by reducing words to their base form.

| Without Stanza | With Stanza |
|----------------|-------------|
| Hand-written morphology rules | ML-based lemmatization |
| Generates ~15 forms per word | Compares lemma-to-lemma |
| Fast, no dependencies | Downloads ~100MB model |
| Good accuracy | Better accuracy |

**Example:** "muzeului" and "muzee" both lemmatize to "muzeu", enabling correct matching even with complex inflections.

### Full Enhanced Mode

```bash
pip install -e ".[full]"
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl --use-languagetool --use-stanza
```

---

## Standalone Grammar Checker

Check grammar quality of model outputs independently:

```bash
# View errors for specific instance
python scripts/check_grammar.py data/outputs.jsonl --instance travel_000001

# Show all instances with 5+ errors
python scripts/check_grammar.py data/outputs.jsonl --verbose --min-errors 5
```

---

## Detailed Workflows

### Generate Custom Dataset

```bash
# Mixed difficulty (default)
python scripts/generate_gmtw_v0.py --output data/instances.jsonl

# Hard mode only
python scripts/generate_gmtw_v0.py --difficulty hard --output data/instances_hard.jsonl

# Custom counts
python scripts/generate_gmtw_v0.py --num-travel 150 --num-schedule 150 --num-fact 100 --num-recipe 100
```

### Compare Multiple Models

```bash
python scripts/compare_models.py data/gmtw_ro_v0.jsonl data/outputs_model1.jsonl data/outputs_model2.jsonl
```

### Cross-Lingual Evaluation

```bash
# Romanian prompts (default)
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --output data/outputs_ro.jsonl

# English prompts
python scripts/run_openrouter_batch.py data/gmtw_ro_v0.jsonl --language en --output data/outputs_en.jsonl

# Compute delta (foreign language penalty)
python scripts/compute_delta.py data/gmtw_ro_v0.jsonl data/outputs_ro.jsonl data/outputs_en.jsonl
```

### Save Detailed Metrics

```bash
python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl --save-metrics data/metrics.jsonl
```

---

## Using Custom Models

Create a JSONL file with model outputs:

```json
{"instance_id": "travel_000000", "output": "Model's response here..."}
{"instance_id": "travel_000001", "output": "Another response..."}
```

**Required output format:**
1. Romanian explanation first (2-3 paragraphs)
2. JSON plan at the end

```
Pentru călătoria de 3 zile în Cluj-Napoca, propun următorul plan...

Am ales să vizităm Grădina Botanică în prima zi deoarece...

{
  "day1": ["Grădina Botanică", "Cetățuia"],
  "day2": ["Muzeul Național de Artă"],
  "day3": ["Parcul Central"]
}
```

---

## Repository Structure

```
rombench/
├── data/
│   └── gmtw_ro_v0.jsonl          # Pre-generated 500-instance dataset
├── rombench/
│   ├── gmtw_ro/
│   │   ├── worlds/               # World generators (travel, schedule, fact, recipe)
│   │   └── eval/                 # Evaluation (parser, constraints, metrics, faithfulness)
│   └── nlp_ro/                   # Romanian NLP toolkit (diacritics, codeswitch, grammar)
├── scripts/
│   ├── evaluate_outputs.py       # Main evaluation script
│   ├── check_grammar.py          # Standalone grammar checker
│   ├── run_openrouter_batch.py   # Run via OpenRouter API
│   ├── run_groq_batch.py         # Run via Groq API
│   ├── debug_results.py          # Interactive result viewer
│   └── compare_models.py         # Side-by-side comparison
└── docs/
    └── GMTW_Ro_Technical_Specification.md
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

Contributions welcome! Please open an issue to discuss major changes.

Key areas:
- Additional world types and constraints
- Expanded diacritic lexicon coverage
- Test coverage improvements
