# GMTW-Ro: Grounded Multilingual Task Worlds for Romanian LLM Evaluation

## Technical Specification Document

**Version:** 1.0
**Date:** November 2025
**Authors:** Hub Român de Inteligență Artificială (HRIA)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation: The Knowledge-Behavior Gap](#2-motivation-the-knowledge-behavior-gap)
3. [Design Principles](#3-design-principles)
4. [Architecture Overview](#4-architecture-overview)
5. [World Types](#5-world-types)
6. [Dual-Channel Output Format](#6-dual-channel-output-format)
7. [The Four Metrics: U, R, G, F](#7-the-four-metrics-u-r-g-f)
8. [Constraint System](#8-constraint-system)
9. [Evaluation Pipeline](#9-evaluation-pipeline)
10. [Cross-Lingual Evaluation (Delta Metric)](#10-cross-lingual-evaluation-delta-metric)
11. [Romanian NLP Toolkit](#11-romanian-nlp-toolkit)
12. [Instance Generation](#12-instance-generation)
13. [Limitations](#13-limitations)
14. [Conclusion](#14-conclusion)

---

## 1. Introduction

GMTW-Ro (Grounded Multilingual Task Worlds for Romanian) is a deterministic benchmark for evaluating Large Language Model (LLM) capabilities in Romanian. Unlike traditional benchmarks that rely on multiple-choice questions or LLM-as-a-judge approaches, GMTW-Ro uses fully-specified task environments ("worlds") where model outputs can be verified programmatically without human intervention.

The benchmark is designed to answer a specific question: **Can a model that performs well on Romanian text actually reason and plan in Romanian, or does it merely pattern-match surface features?**

### 1.1 Key Features

- **Deterministic evaluation**: No human raters, no LLM judges, completely reproducible results
- **Grounded tasks**: All tasks operate within fully-specified environments with verifiable constraints
- **Dual-channel outputs**: Models produce both structured data (JSON) and natural language explanations
- **Decomposed metrics**: Separate scores for Understanding, Reasoning, Generation quality, and Faithfulness
- **Cross-lingual comparison**: Built-in support for measuring the "foreign language penalty"

---

## 2. Motivation: The Knowledge-Behavior Gap

### 2.1 The Problem with Existing Benchmarks

Most multilingual benchmarks suffer from one or more of the following issues:

1. **Multiple-choice format**: Models can exploit statistical patterns without genuine understanding
2. **Translation artifacts**: Benchmarks translated from English may not reflect natural Romanian usage
3. **LLM-as-judge bias**: Using one LLM to evaluate another introduces circular dependencies
4. **Surface-level evaluation**: Testing vocabulary and grammar without testing reasoning ability

### 2.2 The Knowledge-Behavior Gap

A model may "know" Romanian (in the sense of generating fluent text) without being able to "behave" competently in Romanian (solve problems, follow complex instructions, plan). GMTW-Ro specifically targets this gap by requiring models to:

1. Parse Romanian instructions with embedded constraints
2. Reason about those constraints within a defined environment
3. Produce a valid plan that satisfies all requirements
4. Explain their reasoning in coherent Romanian

### 2.3 Why Grounded Tasks?

Grounded tasks provide several advantages:

- **Verifiability**: Given a world specification and a plan, correctness can be computed algorithmically
- **No ambiguity**: The world fully specifies what entities exist and what constraints apply
- **Contamination resistance**: Procedural generation creates unique instances unlikely to appear in training data
- **Difficulty calibration**: Constraints can be added or removed to adjust task difficulty

---

## 3. Design Principles

### 3.1 Determinism

Every aspect of GMTW-Ro evaluation is deterministic:

- World generation uses seeded random number generators for reproducibility
- Constraint checking uses exact algorithmic verification
- Language quality assessment uses lexicon-based rules, not ML models
- Faithfulness checking uses substring matching, not semantic similarity

### 3.2 Transparency

The benchmark does not use any "black box" components:

- All constraint checkers are simple, readable functions
- The Romanian NLP toolkit uses explicit word lists, not learned embeddings
- Scoring formulas are documented and can be audited

### 3.3 Strictness

The benchmark enforces strict adherence to instructions:

- Models must produce output in a specific format (explanation first, JSON last)
- JSON must use exact entity names from the provided list
- All stated constraints must be satisfied for full credit

### 3.4 Decomposition

Rather than producing a single score, GMTW-Ro decomposes performance into four orthogonal dimensions, allowing diagnosis of specific model weaknesses.

---

## 4. Architecture Overview

### 4.1 Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         GMTW-Ro                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   World     │    │   Prompt    │    │   Output    │         │
│  │ Generators  │───▶│  Templates  │───▶│   Parser    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                                      │                 │
│        │                                      ▼                 │
│        │            ┌─────────────────────────────────┐        │
│        │            │         Evaluator               │        │
│        │            │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│        │
│        └───────────▶│  │  U  │ │  R  │ │  G  │ │  F  ││        │
│                     │  └─────┘ └─────┘ └─────┘ └─────┘│        │
│                     └─────────────────────────────────┘        │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │  Romanian   │    │ Constraint  │                            │
│  │ NLP Toolkit │    │  Checkers   │                            │
│  └─────────────┘    └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. **World Generation**: A world generator creates a task instance with entities, constraints, and goals
2. **Prompt Creation**: Templates convert the world specification into Romanian (or English) prompts
3. **Model Inference**: The prompt is sent to an LLM, which produces a response
4. **Output Parsing**: The parser extracts the JSON plan and natural language explanation
5. **Evaluation**: Four metric functions compute U, R, G, and F scores
6. **Reporting**: Results are aggregated and presented

### 4.3 Data Structures

**Instance**: A single evaluation item containing:
- `instance_id`: Unique identifier (e.g., "travel_000042")
- `world`: The complete world specification
- `prompt_ro`: Romanian prompt text
- `prompt_en`: English prompt text (for Delta calculation)

**World**: A fully-specified task environment containing:
- `world_type`: Category (travel, schedule, fact, recipe)
- `payload`: Domain-specific data (cities, attractions, appointments, etc.)
- `constraints`: List of requirements the model must satisfy
- `goals`: Structural requirements for a valid plan
- `canonical_entities`: Dictionary of entities with IDs, names, and aliases

---

## 5. World Types

GMTW-Ro includes four distinct world types, each testing different aspects of language understanding and reasoning.

### 5.1 Travel World

**Task**: Plan a multi-day itinerary for a trip to a Romanian city.

**Environment**:
- 6 Romanian cities: Brașov, Cluj-Napoca, Sibiu, Timișoara, Iași, Constanța
- 37 total attractions across all cities
- Each attraction has properties: type (monument, museum, park, etc.), indoor/outdoor, family-friendly, cost

**Sample Constraints**:
- "Include at least one historic monument"
- "Maximum N outdoor activities per day"
- "Only include family-friendly activities"
- "Total budget must not exceed X lei"

**Output Format**:
```json
{
  "day1": ["Attraction A", "Attraction B"],
  "day2": ["Attraction C"],
  "day3": ["Attraction D", "Attraction E"]
}
```

**What It Tests**:
- Constraint satisfaction across multiple dimensions
- Planning over multiple time periods
- Entity selection from a closed set

### 5.2 Schedule World

**Task**: Organize appointments into a weekly calendar.

**Environment**:
- Days: Luni, Marți, Miercuri (Monday, Tuesday, Wednesday)
- Time slots: dimineață, după-amiază (morning, afternoon)
- Appointments with priorities (high/medium/low)

**Sample Constraints**:
- "Keep all high-priority appointments"
- "Maximum N appointments per day"
- "No overlapping appointments"

**Output Format**:
```json
{
  "Luni_dimineață": "Appointment A",
  "Luni_după-amiază": "Appointment B",
  "Marți_dimineață": null,
  ...
}
```

**What It Tests**:
- Scheduling and resource allocation
- Priority-based decision making
- Slot assignment without conflicts

### 5.3 Fact World

**Task**: Answer a question using only the provided context, ignoring prior knowledge.

**Environment**:
- A "database" of facts presented to the model
- A question that can be answered from the database
- Optionally: "misbelief traps" where the context contradicts common knowledge

**Sample Facts**:
```
Capital Romania: Sibiu  (deliberately wrong to test context adherence)
Romania EU: 2007
First King: Carol I
```

**Sample Constraint**:
- "Answer ONLY based on the given context, even if it contradicts your knowledge"

**Output Format**:
```json
{
  "answer": "Sibiu"
}
```

**What It Tests**:
- Context adherence vs. parametric knowledge
- Resistance to hallucination
- Following explicit instructions over implicit priors

**Misbelief Traps**: Some instances deliberately provide incorrect "facts" (e.g., "The capital of Romania is Sibiu"). A model that follows instructions should return the wrong answer from the context rather than the correct answer from its training data.

### 5.4 Recipe World

**Task**: Plan daily menus respecting dietary restrictions.

**Environment**:
- 19 Romanian dishes across three meal types (breakfast, lunch, dinner)
- Dish properties: vegetarian, vegan, gluten-free, lactose-free, calories

**Sample Constraints**:
- "All dishes must be vegetarian"
- "Avoid dishes containing gluten"
- "Daily calories must not exceed 1800 kcal"
- "Do not repeat the same dish"

**Output Format**:
```json
{
  "day1_mic_dejun": "Dish A",
  "day1_pranz": "Dish B",
  "day1_cina": "Dish C",
  "day2_mic_dejun": "Dish D",
  ...
}
```

**What It Tests**:
- Multi-constraint satisfaction
- Nutritional/dietary reasoning
- Variety and planning across days

---

## 6. Dual-Channel Output Format

### 6.1 Rationale

GMTW-Ro requires models to produce two distinct outputs:

1. **Natural Language Explanation**: A Romanian text explaining the plan and reasoning
2. **Structured JSON Plan**: The actual plan in machine-readable format

This dual-channel approach serves multiple purposes:

- The JSON provides unambiguous, verifiable output for constraint checking
- The explanation tests Romanian language generation quality
- Comparing the two tests faithfulness (does the explanation match the plan?)

### 6.2 Required Format

Models must produce output in this specific order:

```
[Romanian explanation - 2-3 paragraphs]

{
  "key1": "value1",
  "key2": "value2",
  ...
}
```

**Critical requirement**: The explanation must come FIRST, and the JSON must come LAST. This order is enforced because:

1. It tests instruction-following ability
2. It ensures the model commits to reasoning before producing the answer
3. It prevents models from generating JSON first and then post-hoc rationalizing

Models that produce JSON before the explanation receive a format violation penalty.

### 6.3 Parsing Strategy

The parser:
1. Searches for a JSON block (either in markdown code fences or raw braces)
2. Extracts everything BEFORE the JSON as the explanation
3. Attempts to parse the JSON (with repair fallback for minor syntax errors)
4. Returns both components for separate evaluation

---

## 7. The Four Metrics: U, R, G, F

### 7.1 Overview

| Metric | Name | Measures | Range |
|--------|------|----------|-------|
| **U** | Understanding | Did the model satisfy the stated constraints? | 0.0 - 1.0 |
| **R** | Reasoning | Is the plan structurally valid? | 0.0 - 1.0 |
| **G** | Generation | Is the Romanian text well-formed? | 0.0 - 1.0 |
| **F** | Faithfulness | Does the explanation match the JSON plan? | 0.0 - 1.0 |

### 7.2 U - Understanding Score

**Definition**: The proportion of instruction-level constraints that are satisfied.

**Formula**:
```
U = (satisfied constraints) / (total constraints + format penalty)
```

**What counts as a constraint**:
- Explicit requirements stated in the prompt (e.g., "include at least one monument")
- Format requirements (JSON at end, not at beginning)

**Example**:
- 3 constraints in prompt, 2 satisfied → U = 2/3 = 0.67
- If model also puts JSON first (format violation) → U = 2/4 = 0.50

**Interpretation**:
- U = 1.0: Model understood and followed all instructions
- U < 1.0: Model failed to satisfy one or more requirements
- Low U suggests the model struggled to parse or follow Romanian instructions

### 7.3 R - Reasoning Score

**Definition**: The proportion of structural goals that are satisfied.

**Structural goals** are implicit requirements for a valid plan:
- Each day must have at least one activity (non-empty)
- All referenced entities must exist in the world (valid IDs)
- Each meal slot must have a dish assigned

**Formula**:
```
R = (satisfied goals) / (total goals)
```

**Interpretation**:
- R = 1.0: The plan is structurally valid
- R < 1.0: The plan has logical errors (empty days, invalid references, etc.)
- Low R suggests reasoning deficits independent of language understanding

### 7.4 G - Generation Quality Score

**Definition**: A composite measure of Romanian text quality.

**Components**:

| Component | Weight | Measures |
|-----------|--------|----------|
| G_dia | 50% | Diacritic correctness |
| G_cs | 30% | Code-switch score (absence of English) |
| G_len | 20% | Length adequacy |

**Formula**:
```
G = 0.5 × G_dia + 0.3 × G_cs + 0.2 × G_len
```

**G_dia (Diacritic Score)**:

Romanian has five special characters: ă, â, î, ș, ț. Many words MUST have diacritics (e.g., "și" not "si", "în" not "in"). The diacritic analyzer checks ~84 high-frequency words that unambiguously require diacritics.

```
G_dia = (correct diacritics) / (correct + missing diacritics)
```

**G_cs (Code-Switch Score)**:

Measures the absence of English contamination in Romanian text. English words are detected using a curated list while excluding Romanian words that look English (e.g., "nu", "de", "care").

```
G_cs = 1.0 - (english_words / total_words)
```

**G_len (Length Score)**:

Penalizes suspiciously short responses that may indicate degenerate output.

```
G_len = min(1.0, word_count / 50)
```

**Interpretation**:
- G ≈ 1.0: High-quality Romanian text with proper diacritics
- G < 0.9: Some issues with diacritics or English contamination
- G < 0.5: Significant language quality problems

### 7.5 F - Faithfulness Score

**Definition**: The proportion of planned entities that are mentioned in the explanation.

**Rationale**: If a model's JSON plan includes "Biserica Neagră" but the explanation never mentions this attraction, there's a disconnect between what the model planned and what it described.

**Formula**:
```
F = (entities mentioned in explanation) / (entities in JSON plan)
```

**Matching Logic**:
- Entity names are normalized (lowercase, diacritics removed)
- Both Romanian and English names are accepted as matches
- Substring matching is used (simple and deterministic)

**Interpretation**:
- F = 1.0: Perfect consistency between plan and explanation
- F < 1.0: Some planned items not mentioned in explanation
- F = 0.0: Complete disconnect (often indicates format violation where explanation is empty)

---

## 8. Constraint System

### 8.1 Constraint Types

**Instruction Constraints** (affect U score):
- Explicit requirements stated in the prompt
- Must be satisfied for full Understanding credit

**Structural Goals** (affect R score):
- Implicit requirements for plan validity
- Must be satisfied for full Reasoning credit

### 8.2 Constraint Specification

Each constraint has:
- `id`: Unique identifier (e.g., "C_FAMILY_FRIENDLY")
- `type`: INSTRUCTION or STRUCTURAL
- `description_ro`: Romanian description shown to model
- `description_en`: English description for cross-lingual testing
- `check_fn`: Name of the verification function
- `params`: Parameters for the check function

### 8.3 Implemented Constraint Checkers

**Travel World**:
- `check_must_include_type`: Plan must include at least one entity of a given type
- `check_max_outdoor_per_day`: Maximum outdoor activities per day
- `check_all_family_friendly`: All activities must be family-friendly
- `check_budget_limit`: Total cost must not exceed budget
- `check_no_duplicates`: No repeated visits to the same place

**Schedule World**:
- `check_max_appointments_per_day`: Maximum appointments per day
- `check_keep_high_priority`: All high-priority appointments must be kept

**Fact World**:
- `check_answer_matches_context`: Answer must match the provided context

**Recipe World**:
- `check_all_vegetarian`: All dishes must be vegetarian
- `check_no_gluten`: No dishes containing gluten
- `check_no_lactose`: No dishes containing lactose
- `check_max_daily_calories`: Daily calorie limit
- `check_no_duplicates`: No repeated dishes

**General**:
- `check_days_non_empty`: Each day must have at least one item
- `check_valid_entity_ids`: All referenced entities must exist
- `check_all_meals_filled`: Each meal slot must have a dish

### 8.4 Entity Resolution

When checking constraints, entity references in the JSON plan must be resolved to canonical entities. The resolution process:

1. Check if reference is an entity ID (e.g., "A1")
2. Check if reference matches an entity name (case-insensitive)
3. Check if reference matches any entity alias
4. Both Romanian and English names are valid matches

---

## 9. Evaluation Pipeline

### 9.1 Input Files

**Instances File** (JSONL):
```json
{"instance_id": "travel_000001", "world": {...}, "prompt_ro": "...", "prompt_en": "..."}
{"instance_id": "travel_000002", "world": {...}, "prompt_ro": "...", "prompt_en": "..."}
```

**Outputs File** (JSONL):
```json
{"instance_id": "travel_000001", "output": "Model response text..."}
{"instance_id": "travel_000002", "output": "Model response text..."}
```

### 9.2 Evaluation Steps

For each instance:

1. **Parse output**: Extract JSON plan and explanation
2. **Compute U**: Check all instruction constraints
3. **Compute R**: Check all structural goals
4. **Compute G**: Analyze explanation text quality
5. **Compute F**: Check plan-explanation consistency
6. **Record results**: Store scores and detailed breakdowns

### 9.3 Output Format

**Summary Statistics**:
```
AVERAGE SCORES (500 instances)
============================================================
  U (Understanding): 0.847
  R (Reasoning):     0.923
  G (Generation):    0.891
  F (Faithfulness):  0.756
============================================================
```

**Detailed Metrics** (JSONL):
```json
{
  "instance_id": "travel_000001",
  "U": 0.67,
  "R": 1.00,
  "G": 0.86,
  "F": 0.80,
  "U_details": {"satisfied": 2, "total": 3, "constraints": [...]},
  "R_details": {"satisfied": 2, "total": 2, "goals": [...]},
  "G_details": {"G_dia": 0.85, "G_cs": 1.0, "G_len": 1.0, ...},
  "F_details": {"missing": ["A3"], "mentioned_count": 3, "total_count": 4}
}
```

---

## 10. Cross-Lingual Evaluation (Delta Metric)

### 10.1 Purpose

The Delta (Δ) metric quantifies the "foreign language penalty" - the performance degradation when a model processes Romanian instead of English.

### 10.2 Methodology

1. Run the same model on the same instances twice:
   - Once with Romanian prompts → scores (U_ro, R_ro, F_ro)
   - Once with English prompts → scores (U_en, R_en, F_en)

2. Compute Delta for each metric:
   ```
   ΔU = U_en - U_ro
   ΔR = R_en - R_ro
   ΔF = F_en - F_ro
   ```

3. Positive Δ indicates the model performs better in English (Romanian penalty)

### 10.3 English Prompt Design

To ensure valid comparison, English prompts:
- Use translated entity names (e.g., "The Black Church" instead of "Biserica Neagră")
- Use ASCII-friendly city names (e.g., "Brasov" instead of "Brașov")
- Use English JSON keys for Recipe world (e.g., "day1_breakfast" instead of "day1_mic_dejun")
- Maintain identical constraint semantics

### 10.4 Interpretation

| Delta Value | Interpretation |
|-------------|----------------|
| Δ ≈ 0 | Consistent performance across languages |
| Δ > 0.05 | Minor degradation in Romanian |
| Δ > 0.10 | Significant Romanian penalty |
| Δ > 0.20 | Severe capability gap in Romanian |

### 10.5 G Metric Exception

The G (Generation) metric is NOT compared across languages because:
- G measures Romanian-specific features (diacritics)
- English output naturally scores lower on Romanian metrics
- G is only meaningful for Romanian outputs

---

## 11. Romanian NLP Toolkit

### 11.1 Design Philosophy

The toolkit is entirely rule-based and deterministic:
- No machine learning models
- No external API calls
- No statistical inference

This ensures:
- Perfect reproducibility
- Fast execution
- Transparent behavior

### 11.2 Diacritic Analyzer

**Core Component**: A curated lexicon of ~84 words that MUST have diacritics.

**Lexicon Design**:
- Only includes words where the non-diacritic form is NEVER valid
- Example: "și" (and) - "si" is never correct in Romanian
- Excludes ambiguous cases: "sa" could be "să" (subjunctive) or "a sa" (possessive)

**Analysis Process**:
1. Tokenize text into words
2. Normalize to lowercase
3. For each word in the lexicon:
   - If word appears without diacritics → missing
   - If word appears with correct diacritics → correct
   - If word appears with wrong diacritics → missing (penalized)

### 11.3 Code-Switch Detector

**Purpose**: Identify English words in Romanian text.

**Implementation**:
- Curated list of common English words
- Exclusion list for Romanian words that look English (e.g., "nu", "de", "care", "a", "face")
- Simple word-level detection

**Limitations**:
- Does not detect English phrases
- May miss rare English words
- Intentionally conservative to avoid false positives

### 11.4 Length Analyzer

**Purpose**: Flag suspiciously short responses.

**Threshold**: 50 words minimum for full credit

**Rationale**: Very short responses often indicate:
- Degenerate output
- Failure to provide explanation
- Format misunderstanding

---

## 12. Instance Generation

### 12.1 Procedural Generation

All instances are procedurally generated with seeded randomness:

```python
generator = TravelWorldGenerator()
world = generator.generate(
    world_id="travel_000042",
    seed=42,
    difficulty="medium"
)
```

### 12.2 Difficulty Levels

| Level | Characteristics |
|-------|-----------------|
| Easy | Fewer constraints, more entities to choose from |
| Medium | 2-3 constraints, some require planning |
| Hard | Multiple interacting constraints, tight bounds |

### 12.3 Reproducibility

Given the same seed, instance generation produces identical results. This enables:
- Reproducible benchmarks
- Version-controlled datasets
- Fair comparisons across time

### 12.4 Contamination Resistance

Procedural generation helps resist training data contamination:
- Specific entity combinations are unlikely to appear in training data
- Constraint combinations create novel scenarios
- Seed-based generation can create unlimited unique instances

---

## 13. Limitations

### 13.1 Language Coverage

- Currently only supports Romanian
- English prompts are translations (not native English tasks)
- Cultural specificity to Romania (cities, dishes, etc.)

### 13.2 World Coverage

- Four world types may not cover all reasoning capabilities
- Tasks are relatively structured (constraint satisfaction)
- Does not test open-ended generation or creativity

### 13.3 G Metric Limitations

- Diacritic lexicon covers ~84 words (not exhaustive)
- Code-switch detection is word-level only
- No grammar or syntax checking

### 13.4 Constraint Solver

- No formal verification that generated instances are solvable
- Difficulty calibration is heuristic, not mathematical
- Some constraint combinations may be unsatisfiable

### 13.5 Faithfulness Metric

- Uses simple substring matching
- Does not detect negation ("I did NOT choose X")
- Does not weight by entity importance

---

## 14. Conclusion

GMTW-Ro provides a rigorous, deterministic framework for evaluating LLM capabilities in Romanian. By combining grounded task worlds, dual-channel outputs, and decomposed metrics, it enables fine-grained diagnosis of model strengths and weaknesses.

The benchmark's key contributions:

1. **Deterministic evaluation** eliminates human rater variability and LLM-as-judge circularity
2. **Grounded tasks** test genuine reasoning, not pattern matching
3. **Decomposed metrics** (U, R, G, F) enable targeted model improvement
4. **Cross-lingual support** quantifies the foreign language penalty
5. **Procedural generation** resists contamination and enables unlimited scaling

GMTW-Ro represents a step toward more rigorous, reproducible evaluation of multilingual language model capabilities.

---

## Appendix A: Metric Formulas Summary

| Metric | Formula |
|--------|---------|
| U | satisfied_constraints / (total_constraints + format_penalties) |
| R | satisfied_goals / total_goals |
| G | 0.5 × G_dia + 0.3 × G_cs + 0.2 × G_len |
| G_dia | correct_diacritics / (correct + missing) |
| G_cs | 1.0 - (english_words / total_words) |
| G_len | min(1.0, word_count / 50) |
| F | entities_mentioned / entities_planned |
| ΔX | X_english - X_romanian |

---

## Appendix B: World Type Summary

| World | Task | Entities | Key Constraints |
|-------|------|----------|-----------------|
| Travel | Multi-day itinerary | 37 attractions, 6 cities | Type inclusion, outdoor limits, budget, family-friendly |
| Schedule | Calendar organization | Appointments | Priority, max per day, no overlap |
| Fact | Context-based Q&A | Facts database | Context adherence, misbelief traps |
| Recipe | Menu planning | 19 dishes | Dietary restrictions, calories, variety |

---

## Appendix C: File Format Reference

**Instance JSONL**:
```json
{
  "instance_id": "string",
  "world": {
    "world_id": "string",
    "world_type": "travel|schedule|fact|recipe",
    "payload": {},
    "constraints": [],
    "goals": [],
    "canonical_entities": {}
  },
  "prompt_ro": "string",
  "prompt_en": "string"
}
```

**Output JSONL**:
```json
{
  "instance_id": "string",
  "output": "string",
  "model": "string (optional)",
  "language": "ro|en (optional)"
}
```

**Metrics JSONL**:
```json
{
  "instance_id": "string",
  "U": 0.0-1.0,
  "R": 0.0-1.0,
  "G": 0.0-1.0,
  "F": 0.0-1.0,
  "U_details": {},
  "R_details": {},
  "G_details": {},
  "F_details": {}
}
```
