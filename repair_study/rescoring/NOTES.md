# Repair rescoring — vendored evaluator plus our drivers

## What is vendored, and must not be edited

Everything below arrived as `GMTW_rescoring_Stephen.zip` and is unmodified. Their
`SHA256SUMS.txt` still verifies in place (25 of 26 files; `THIRD_PARTY_NOTICE.txt`
is listed but was never shipped — worth asking them for).

    code/rescore.py                 the corrected evaluator
    code/rescore_saved_outputs.py   per-file runner for saved generations
    code/test_rescore.py            19 scorer tests
    code/test_saved_outputs.py      10 runner tests
    code/vendor/                    the two rombench modules they depend on
    cohorts/                        438 standard / 240 adversarial retained IDs,
                                    the excluded IDs, and the exclusion manifest
    inputs/                         the 500 standard and 300 adversarial worlds
    reference/                      their diagnostic results, for comparison
    README.txt, VALIDATION.txt, PROVENANCE.json, SHA256SUMS.txt, requirements.txt

Read their `README.txt` before changing anything. The runner hard-fails if the
bundled sources differ from the proposal or the installed package versions do not
match, which is deliberate.

## Environment

Lives OUTSIDE this repo, because it is 307 MB and 8,961 files:

    ~/envs/gmtw_rescore          python 3.13.16, numpy 2.4.6, json-repair 0.54.2

Rebuild if lost:

    ~/miniconda3/bin/conda create -y -p ~/envs/gmtw_rescore python=3.13
    ~/envs/gmtw_rescore/bin/python -m pip install -r requirements.txt
    ~/envs/gmtw_rescore/bin/python -X utf8 -m unittest discover -s code -p "test_*.py"

The local system python is 3.12 with numpy 1.26 and no `json_repair`, so it cannot
run the evaluator. There is no `python3-venv` on this machine, hence conda.

## Ours

    run_all.sh        64 runs: 8 checkpoints x (base + 3 seeds) x 2 splits
    run_abl.sh        24 runs: the 4 corpus ablations x 3 seeds x 2 splits
    build_tables.py   -> tables 67 (before/after, seed ranges), 68 (deltas), 70 (ablations)
    paired.py         -> table 69, paired CIs and Holm on T and J, both splits

Run from this directory, in this order. Per-run records land in
`../results/rescored/`, tables in `../results/tables/`.

## Before trusting a rerun

Their README requires confirming the repair runs used the same task definitions.
Checked once and recorded here: their `inputs/v0_DE_test.jsonl` differs from the
`data/gmtw_ro_v0.jsonl` our repair campaign scored ONLY by added German content —
prompts, constraint descriptions, question text and German entity aliases.
Constraints, check functions, params, expected answers and entity IDs/names are
identical on all 500; the adversarial worlds match exactly; and neither export
contains an ambiguous alias, so the added aliases cannot change a resolution
verdict. If either export is regenerated, redo that check.

Baselines are the repair experiment's own, through our vLLM pipeline. Do not
substitute a diagnostic baseline from the HuggingFace pipeline.
