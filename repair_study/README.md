# Structured-output repair and capability retention in Romanian-adapted LLMs

Artefacts for the repair and retention study: the training corpus, the adapter configurations, every per-instance score, the corrected evaluator and the tables built from them. Everything here is data and analysis code — no model weights.

The study fine-tunes eight Romanian-adapted checkpoints on a 4,000-example structured-output corpus with three training seeds each, then measures what the repair gains on grounded tasks and what it costs on a standard academic suite.

Built 2026-10-13T10:32:17.065328+00:00 from repo commit 80a64908a980

Every file is pinned by SHA-256 in `MANIFEST.json`.

## Reproducing the corrected scores

The corrected evaluator and the retained task cohorts are vendored under `rescoring/`, unmodified, with their own checksums. They need an exact environment, which `rescoring/NOTES.md` records:

```
conda create -y -p ./env python=3.13
./env/bin/python -m pip install -r rescoring/requirements.txt
./env/bin/python -X utf8 -m unittest discover -s rescoring/code -p "test_*.py"
```

Then, from `rescoring/`, `run_all.sh` and `run_abl.sh` rescore the saved generations in `scores/gmtw_ro/`, and `build_tables.py` and `paired.py` rebuild the corrected tables. No GPU and no model calls.

## What is here

- `corpus/` — 40 files, from `datagen/data/sft`
- `configs/` — 20 files, from `datagen/training`
- `tables/` — 39 files, from `datagen/results/tables`
- `eval/jsonschemabench/` — 9 files, from `ro-so/eval/jsonschemabench_holdout`
- `retention/` — 455 files, from `datagen/retention`
- `rescoring/` — 33 files, from `datagen/rescoring`
- `scores/english_schema/` — 38 files, from `datagen/results/holdout`
- `scores/english_schema_per_seed/` — 48 files, from `datagen/results/holdout_perseed`
- `scores/rescored/` — 176 files, from `datagen/results/rescored`
- `scores/gmtw_ro/` — 451 files, from `datagen/results`
- `code/datagen/analysis/` — 9 files, from `datagen/analysis`
- `code/datagen/core/` — 9 files, from `datagen/core`
- `code/datagen/scripts/` — 11 files, from `datagen/scripts`
- `code/datagen/tests/` — 3 files, from `datagen/tests`
- `code/ro-so/filters/` — 6 files, from `ro-so/filters`
- `code/ro-so/templates/` — 7 files, from `ro-so/templates`
- `requirements.txt/` — 1 files, from `datagen/requirements.txt`

## Not here, and why

- **Merged checkpoints.** Reconstructible from base + adapter; the merge is `datagen/scripts/merge_lora.py`.
- **Base models.** Upstream artifacts, pinned by dated revision in the training configs.
- **Adapter weights.** ~8.3 GB; pinned by SHA-256 above (39 entries under `adapter_sha256`).

## Regenerating each table

Run from the repository root with the code tree on the path. Nothing here needs a GPU or makes a model call.

```
export PYTHONPATH=code
python code/datagen/analysis/repair_tables.py \
    --analysis scores/gmtw_ro --split 500 --out tables
```

`code/` keeps the original package structure, because the scripts import one another: `holdout_overlap.py` reaches a `ro-so` sibling for `schema_fingerprint`, and `holdout_eval.py` imports `datagen.core`. Flattening it would ship scripts that cannot run.

Substitute the published paths for the placeholders below: `<results>` is `scores/gmtw_ro`, `<tables>` is `tables`, `<sft>` is `corpus`, `<english>` is `eval/jsonschemabench/test_split.jsonl`. The retention tables read `retention/results` and write `retention/tables`.

- `40_repair_main_*.csv` — `python code/datagen/analysis/repair_tables.py --analysis <results> --split <n>`
- `41_repair_deltas_*.csv` — `python code/datagen/analysis/repair_tables.py --analysis <results> --split <n>`
- `43_repair_per_seed_*.csv` — `python code/datagen/analysis/repair_tables.py --analysis <results> --split <n> --per-seed`
- `44_repair_multiplicity_*.csv` — `python code/datagen/analysis/repair_tables.py --analysis <results> --split <n> --multiplicity`
- `45_train_eval_separation.csv` — `python code/datagen/analysis/contamination_audit.py --corpus <sft> --eval <gmtw> --out <tables>`
- `46_corpus_ablations_*.csv` — `python code/datagen/analysis/repair_tables.py --analysis <results> --split <n> --ablations`
- `55_empty_plan_by_cause_*.csv` — `python code/datagen/analysis/empty_plan_audit.py --analysis <results> --split <n>`
- `56_faith_composite_v123_*.csv` — `python code/datagen/analysis/empty_plan_audit.py --analysis <results> --split <n>`
- `57_end_to_end_v2_v3_*.csv` — `python code/datagen/analysis/empty_plan_audit.py --analysis <results> --split <n>`
- `58_repair_deltas_v3_*.csv` — `python code/datagen/analysis/empty_plan_audit.py --analysis <results> --split <n>`
- `59_repair_paired_t_*.csv` — `python code/datagen/analysis/repair_tables.py --analysis <results> --split <n> --paired-t`
- `60_corpus_accounting.csv` — `python code/datagen/analysis/corpus_accounting.py --corpus <sft> --out <tables>`
- `61_ablation_replacement.csv` — `python code/datagen/analysis/corpus_accounting.py --corpus <sft> --out <tables>`
- `62_holdout_overlap.csv` — `python code/datagen/analysis/holdout_overlap.py --split <english> --mix <mix> --pool <pools> ...`
- `63_transfer_english_clean.csv` — `python code/datagen/analysis/holdout_overlap.py (same invocation)`
- `64_repair_v3_holm_*.csv` — `python code/datagen/analysis/empty_plan_audit.py --analysis <results> --split <n>`
- `65_scorer_audit.csv` — `python code/datagen/analysis/scorer_audit.py --metrics <gmtw new_data> --out <tables>`
- `66_transfer_per_seed.csv` — `python code/datagen/analysis/holdout_overlap.py (same invocation; reads scores/holdout_perseed)`

## Authoritative notes a reader needs

- `target_modules` in the training YAMLs is **dead config**. `datagen/training/train_lora.py` hardcodes the target pattern and passes it directly; the YAML key is never read. The regex in the trainer is authoritative.
- The English evaluation set is **1,804 schemas**, of which **1,210** reject every trivial answer and **1,209** remain after removing the single exact duplicate of a training schema. A file of 3,608 records was this split concatenated with itself and is not scored.
- That set is **in-distribution retention**, not a held-out source family: it is the upstream `test` split of the four trained sources, with zero instance-id overlap with training.
- GMTW-Ro per-instance records come from the authoritative run only (`metrics_de_test_{en,ro,de}`, n=500; `metrics_hard_ro`, n=300). The `metrics_easy_*` directories are a superseded run.
- Gemma-2 must be scored with `attn_implementation=eager`; the SDPA path silently skips logit softcapping.

