GMTW repair rescoring

Exact scorer and task cohorts used for the October 3 article proposal.

What is included

code/rescore.py and code/test_rescore.py are unchanged copies of the scripts used for the proposal. The two required rombench modules are included under code/vendor, with the upstream license. You do not need to clone another repository or download model weights.

code/rescore_saved_outputs.py is a new file runner. It imports the original score() function without changing any evaluation rule. It accepts one saved-output file at a time, so it works with additional repair checkpoints and seeds. The main() function in the unchanged rescore.py expects the complete nine-checkpoint diagnostic archive; use the new runner for the repair files.

The original 500 standard and 300 adversarial task definitions are in inputs. v0_DE_test.jsonl contains the shared standard worlds and the Romanian, English and German prompts despite its filename. hard.jsonl contains the adversarial worlds.

The task ID lists

cohorts/standard_retained_ids.txt      438 IDs
cohorts/standard_excluded_ids.txt       62 IDs
cohorts/adversarial_retained_ids.txt   240 IDs
cohorts/adversarial_excluded_ids.txt    60 IDs

Each TXT contains one ID per line. cohorts/cohort_exclusions.csv is the exact saved manifest for all 800 tasks. It records the split, family, eligibility, duplicate source ID, exclusion reasons and task-content hash. cohorts/summary.json gives counts by reason and task family.

Standard exclusions involve 23 repeated worlds, 50 worlds with required timing information omitted from the prompt and five worlds with demonstrated incompatible type constraints. Reasons overlap, giving 62 excluded IDs in total. Adversarial exclusions involve 19 repeated worlds and 52 worlds with hidden timing information, again with overlap, giving 60 excluded IDs. These counts must not be added as if the reasons were disjoint.

Apply the same retained IDs to every baseline, repaired checkpoint, seed and language in the corresponding split. These cohorts address the known defects; they are not a proof that every remaining constraint or world is correct. Confirm that the repair runs used these same task definitions and prompts. Reusing an ID for a newly generated world would not make it the same task; use the bundled definitions and recorded hashes to check identity.

Setup and tests

Tested with Python 3.13.5, NumPy 2.4.6 and json-repair 0.54.2. Use a separate Python environment. From the extracted GMTW_rescoring_for_Stephen folder:

python -m pip install -r requirements.txt
python -X utf8 -m unittest discover -s code -p "test_*.py"

The UTF-8 flag also makes the unchanged tests read the Romanian data correctly on Windows. The requirements contain only the two packages needed for scoring; plotting dependencies from the article build are unnecessary. Scoring itself runs locally on CPU and makes no model or network calls.

Input format

One JSON object per line, with a string task ID and the full original response string. The default fields are instance_id and output. For example, the structure is:

{"instance_id": "travel_000000", "output": "the complete saved assistant response", "model": "checkpoint-name", "language": "ro", "seed": 0}

Use one file per checkpoint, language and training seed. The optional metadata fields are recorded, and mixed metadata or duplicate IDs are rejected. Exporting files without metadata requires you to keep that separation yourself. Preserve the original explanation, JSON, quotation marks and code fences in the response. Do not extract or repair the answer before passing it to the scorer.

Files can contain all 500/300 original records or the already filtered 438/240 records. Every retained ID must be present exactly once. Excluded records may be present and are ignored in the summary. A saved blank response is represented by an empty string. A missing record, null value or unknown ID stops the run rather than silently reducing the denominator. If the record uses different top-level keys, set --id-field and --output-field. Nested output formats need to be exported into this simple JSONL structure without changing the response text.

Example commands

Score a repair baseline on the standard split:

python -X utf8 code/rescore_saved_outputs.py --split standard --outputs raw/baseline_standard.jsonl --out-dir runs/baseline_standard

Score one repaired seed on the same tasks:

python -X utf8 code/rescore_saved_outputs.py --split standard --outputs raw/repaired_seed0_standard.jsonl --out-dir runs/repaired_seed0_standard

Score an adversarial file:

python -X utf8 code/rescore_saved_outputs.py --split adversarial --outputs raw/repaired_seed0_hard.jsonl --out-dir runs/repaired_seed0_hard

The paths above are examples for your own files. Repeat separately for the other seeds and checkpoints. An optional --label names the run. For fields called task_id and raw_response, append --id-field task_id --output-field raw_response. Each command writes into a new directory and refuses to overwrite an existing one. No edits to rescore.py are needed.

What the runner writes

per_instance.jsonl contains a score for each provided task, with eligible=true or false, the source line, parser stage and individual constraint outcomes. Use only eligible=true records for the primary 438/240-task comparison. The raw input file remains unchanged and is identified by its hash in summary.json.

summary.json contains T, raw_ok, normalized_ok, recovered, schema, joint, factual-QA accuracy and task-family summaries. Means and interval bounds are on a 0-to-1 scale. Multiply rates by 100 for presentation as percentages. Parser-stage counts refer to the retained cohort.

raw_ok is R0: the selected object parses as written. normalized_ok is Rq: raw parsing or declared quote normalization succeeds. recovered is Rr: these stages or general recovery succeed. The three are cumulative. schema is S. joint is J, requiring the expected schema and T=1 after declared recovery. R0 refers to the selected object, not to the entire response being JSON. J does not evaluate explanation semantics or response-language compliance.

T is corrected task compliance. This runner does not recompute legacy U, G, F or the old end-to-end proxy. Those columns should not be renamed as T/J or compared as though their definitions were unchanged.

Intervals use the same 5,000-resample, seed-20240617 task bootstrap as the diagnostic estimates. They describe a single saved run. For repair differences, match baseline and repaired records by instance_id within the retained cohort. Use the repair experiment's own baseline, not a diagnostic baseline from another inference pipeline. Keep each training seed separate. A paired bootstrap of task differences uses seed 42 in the original analysis. Task intervals do not capture between-training-seed uncertainty, and subtracting two interval endpoints is not a paired confidence interval. This file runner does not perform a combined seed analysis or choose a multiple-comparison family.

Reference and provenance

reference/diagnostic_summaries.json contains the already computed diagnostic results for comparison. These are reference diagnostics, not replacement repair baselines. On Romanian standard tasks, Llama-3.1-8B has T=0.9094368340943684 and J=0.6461187214611872 on 438 tasks.

PROVENANCE.json records source hashes, the upstream commit, package versions and exact cohort counts. The runner checks the copied sources and installed package versions before scoring. SHA256SUMS.txt covers the distributed files. The package retains the code and cohorts used in the proposal; any further evaluator change should receive a new version and be applied consistently to both baseline and repaired outputs.

The existing 19 scorer tests are included, along with 10 checks for the new file runner. VALIDATION.txt records the checks completed for this handoff. No repair outputs were available locally for this packaging step, so this handoff does not contain newly computed repair results.
