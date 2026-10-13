# Retention — does the repair cost anything outside GMTW-Ro?

This folder is the whole experiment: the runner, the published reference it is
checked against, the tables, and (once anything has run) the results.

```
run_rollm_eval.sh        one checkpoint through OpenLLM-Ro's own suite
published_baselines.py   their published scores, straight off the model cards
tables/                  50_ the averages, 51_ the few-shot ladder behind each
results/                 per-run lm_eval output + provenance (created on first run)
```

It is the one part of `datagen/` that does **not** run in the training
environment — see Setup below.

Review §3.3: *"There is no evaluation of general reasoning, factuality, Romanian
conversational quality, knowledge, safety, or out-of-domain instruction following
after repair. The study therefore demonstrates GMTW recovery, not a free repair."*

The answer has to be measured on a suite that (a) exists independently of us and
(b) already has a published row for every checkpoint we repaired. OpenLLM-Ro's
own evaluation is both: it is the suite their model cards report, run by their
own fork of lm-evaluation-harness, and the eight checkpoints under repair are
scored on it in public.

That gives the comparison three columns instead of two:

| column | what it is | what it is for |
|---|---|---|
| published | OpenLLM-Ro's number for the released checkpoint | validity check on our harness |
| base (ours) | that same checkpoint, re-run here | the paired reference |
| repaired (ours) | our merged checkpoint, same run | the retention answer |

**The base column is not optional.** A repaired row compared against a published
row would fold every harness, stack and hardware difference into the "cost of the
repair". Base and repaired must be run in the same environment, same seed, same
few-shot ladder, same `--limit` if any; `published` is then a check on whether
that environment reproduces OpenLLM-Ro's, not a substitute for it.

## The published numbers

`published_baselines.py` reads them out of the model cards' `model-index`
front matter and writes two tables:

```bash
python datagen/retention/published_baselines.py --out datagen/retention/tables
```

- `tables/50_published_openllm_ro.csv` — the averages
- `tables/51_published_openllm_ro_per_fewshot.csv` — the ladder behind each

| checkpoint | Academic avg | ARC | MMLU | Winogrande | HellaSwag | GSM8k | TruthfulQA |
|---|---|---|---|---|---|---|---|
| RoLlama3.1-8B-DPO | 53.76 | 51.09 | 56.22 | 66.77 | 59.38 | 31.54 | 57.56 |
| RoGemma2-9B-DPO | 59.79 | 55.66 | 64.00 | 73.16 | 64.26 | 37.80 | 63.86 |
| RoGemma2-9B-SFT | 54.39 | 50.24 | 62.00 | 70.38 | 52.25 | 40.51 | 50.97 |
| RoLlama3-8B-DPO | 55.86 | 52.26 | 55.35 | 66.62 | 59.93 | 43.95 | 57.06 |
| RoMistral-7B-DPO | 56.62 | 55.51 | 52.61 | 68.04 | 64.97 | 41.07 | 57.55 |
| RoMistral-7B-SFT | 54.40 | 52.86 | 52.33 | 68.57 | 63.50 | 38.15 | 51.01 |
| RoLlama2-7B | 45.51 | 45.70 | 40.36 | 63.26 | 60.25 | 18.02 | 45.48 |
| RoGemma-7B-DPO | 48.27 | 46.66 | 54.45 | 63.73 | 49.33 | 34.98 | 40.45 |

Each task average is the mean over a few-shot ladder — ARC 0/1/3/5/10/25, MMLU
0/1/3/5, Winogrande 0/1/3/5, HellaSwag 0/1/3/5/10, GSM8k 1/3/5, TruthfulQA 0 —
and the extractor asserts that every card still publishes exactly those ladders.
All eight averages reproduce as the plain mean of their own rungs, so the
published column and our runs are the same quantity.

**The ladder has to be run rung by rung.** The harness ships each ladder as a
group (`ro_mmlu` and friends), and those groups cannot be called: in this
version `include:` overrides the subtask's own `task:` key, so `mmlu_fs0/1/3/5`
all resolve to `_ro_mmlu` and the duplicate check rejects the group before a
model loads —

```
ValueError: Found 1 or more tasks ... that were members of more than 1 called
group: ['_ro_mmlu']. Offending groups: ['ro_mmlu']
```

It fails on one group alone, so this is not about calling several at once. The
runner therefore calls the base task with an explicit `--num_fewshot` per rung,
batched by rung so a rung costs one model load rather than one per task: six
invocations per checkpoint, one `results_*.json` per rung under `fs<K>/`,
averaged by us. Same arithmetic the group would have done, and it is what makes
the per-rung comparison against the published ladder possible.

RoMT-Bench and RoCulturaBench are also on the cards and are **out of scope**:
both are LLM-judged, and this project scores deterministically only.

## The anchor run reproduces the published numbers

`base_rollama31-dpo`, full protocol, all 23 rungs, run 2026-09-08 on an H200
against the released checkpoint — no LoRA, nothing of ours in it:

| task | ours | published | diff |
|---|---|---|---|
| ARC | 51.21 | 51.09 | +0.12 |
| MMLU | 56.22 | 56.22 | +0.00 |
| Winogrande | 66.75 | 66.77 | −0.02 |
| HellaSwag | 59.34 | 59.38 | −0.04 |
| GSM8k | 31.24 | 31.54 | −0.30 |
| TruthfulQA | 57.69 | 57.56 | +0.13 |
| **Academic average** | **53.74** | **53.76** | **−0.02** |

Every one of the 23 rungs lands within 1.13 points of its published counterpart
(worst: GSM8k 3-shot, 28.89 against 30.02), and the generative path checks out
as well as the log-likelihood one — GSM8k 5-shot 39.20 against 39.50. Different
GPU, newer torch, newer transformers, and the ladder still lands on their
numbers.

This is what licenses the `published` column: not "we used their harness" but a
reproduction of what they published with it. Anything a repaired checkpoint
does differently here is the repair, not the environment.

## The result: the repair costs 8.28 points

`rollama31-dpo_4kv2_s{0,1,2}` against `base_rollama31-dpo`, full protocol both
arms, three training seeds, 2026-09-08:

| task | base | repaired (mean of 3 seeds) | delta | seed range |
|---|---|---|---|---|
| MMLU | 56.22 | 45.00 | **−11.22** | 0.15 |
| GSM8k | 31.24 | 14.42 | **−16.82** | 1.70 |
| HellaSwag | 59.34 | 50.86 | −8.48 | 0.12 |
| ARC | 51.21 | 44.30 | −6.91 | 0.26 |
| Winogrande | 66.75 | 63.22 | −3.53 | 0.60 |
| TruthfulQA | 57.69 | 54.95 | −2.74 | 1.13 |
| **Academic average** | **53.74** | **45.46** | **−8.28** | |

Every task falls, and the seeds agree to within 1.7 points, so this is the
recipe rather than one unlucky run. §3.3 asked whether the repair is free. It is
not, and this is the size of the bill.

### It is not a harness artefact

The candidates were checked before the number was believed. Both arms load
byte-identical tokenizer files — the merged checkpoint's `tokenizer.json` is the
base one, and the runner points at the base tokenizer explicitly. Both ran the
same rungs at the same batch size and seed. The base arm reproduces
OpenLLM-Ro's published numbers to 0.02. And **MMLU is pure log-likelihood
ranking** — no generation, no output format, no stopping behaviour — and it
still loses 11 points.

### How it fails, where we can see it

GSM8k writes its answers out, so its samples show the mechanism:

| | accuracy | outputs that loop | emits the `####` answer marker |
|---|---|---|---|
| base | 39.20% | 32.5% | 69.4% |
| repaired s0 | 5.84% | 54.5% | 27.3% |

The repaired model stops following the answer format its own few-shot examples
demonstrate, and degenerates into repetition with token-level corruption —
`înțe-înțe-înțe`, `șurubzi`, `Ștefteșteună`. Not wrong arithmetic: broken text.

### The format control: it is the weights, not the format

These checkpoints were LoRA-tuned entirely on chat-formatted prose-then-JSON,
and the published protocol scores raw completions — so the measurement above is
out-of-format for them by construction, and the whole result could have been an
artefact of asking a chat-tuned model to complete text. It is not. Both arms
were re-run at 5-shot with `--apply_chat_template` and the same template
(the repaired arm loads the base tokenizer, so the prompt text is identical on
both sides):

| task | format | base | repaired s0 | gap |
|---|---|---|---|---|
| MMLU | completion | 56.31 | 42.91 | −13.40 |
| MMLU | **chat template** | 54.90 | 39.39 | **−15.51** |
| GSM8k | completion | 39.20 | 5.84 | −33.36 |
| GSM8k | **chat template** | 36.85 | 0.08 | **−36.77** |

**The gap does not close in the model's own format — it widens.** Putting the
template on costs the base a little (MMLU −1.41, GSM8k −2.35, the ordinary price
of switching a benchmark out of the format it was designed for) and costs the
repaired checkpoint more than twice as much (−3.52, −5.76). GSM8k in chat format
is 1 correct answer out of 1,319.

So the capability is not intact-but-unreachable. It is gone from the weights,
and the trained format does not recover it. The honest headline is catastrophic
forgetting, and the recipe is what has to change: replay data alongside the
structured-output corpus, a lower learning rate, fewer epochs, or a smaller
adapter — none of which this study has tested.

MMLU carries the argument. It is pure log-likelihood ranking over four fixed
options: no generation, no output format, no stopping behaviour, nothing that a
prompt convention could rescue. It loses 13–15 points either way.

## The dose curve: there is no gentler dose

Every training run saved an adapter every 50 steps of its 500, so the curve
between "no repair" and "8.28 points gone" was already on disk. Four of those
checkpoints were merged and put through both harnesses — GMTW-Ro for the repair,
MMLU and GSM8k at 5-shot for the damage:

| step | valid JSON % | U | F_v2 | MMLU | GSM8k |
|---|---|---|---|---|---|
| 0 (base) | 22.4 | 0.430 | 0.035 | 56.31 | 39.20 |
| **50** | **100.0** | **0.751** | **0.695** | **42.24** | **4.70** |
| 100 | 99.4 | 0.763 | 0.703 | 42.30 | 5.16 |
| 250 | 99.8 | 0.764 | 0.707 | 42.95 | 6.44 |
| 400 | 100.0 | 0.777 | 0.713 | 42.84 | 5.61 |
| 500 (full) | 100.0 | 0.760 | 0.678 | 42.70 | 5.13 |

**Everything happens in the first 50 steps** — 800 examples, a tenth of the
schedule. Valid JSON is already at 100.0% and MMLU has already lost 14 points.
From step 50 to step 500 neither axis moves: the repair does not deepen and the
damage does not worsen.

The hypothesis this was built to test — that the recipe is over-dosed, and
stopping early buys the repair without the bill — is dead. Repair and forgetting
are not two processes with different time constants that a well-placed early
stop could separate. At this learning rate they are the same event.

What that leaves is the recipe rather than its length: the learning rate itself
(1e-4 on an 8B LoRA, and at step 50 the cosine has barely begun to anneal),
replay data mixed into the corpus, or a smaller adapter. None of those can be
read off a checkpoint that already exists — they need training runs, and this
curve is the argument for why those runs are worth doing.

One caveat travels with every row: the schedule is cosine over 500 steps, so
checkpoint-50 is not the same object as a run *scheduled* for 50 steps — its
learning rate never annealed. It is a lower bound on what a short run achieves,
which is the conservative direction for the conclusion drawn here.

## Setup

The harness is OpenLLM-Ro's fork, pinned to the commit these tasks were read
from, and **it needs its own environment**: the training env runs transformers
5.x, this is a 0.4.x-era harness and will not import there. Nothing else in
`datagen/` shares that env.

```bash
git clone https://github.com/OpenLLM-Ro/lm-evaluation-harness \
    /DATA/stephen/eval_rollm/lm-evaluation-harness
cd /DATA/stephen/eval_rollm/lm-evaluation-harness
git checkout 77e8ac48c750b661e2cf1f48085e4a4181406ed1

source /DATA/stephen/env.sh                       # HOME, TMPDIR, HF_HOME, conda
conda create -y -p /DATA/stephen/envs/rollm python=3.11
conda activate /DATA/stephen/envs/rollm
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install "transformers==4.53.2"                # before -e, so it is not upgraded
pip install -e .
```

Then a hook the runner sources, so machine setup stays out of the script:

```bash
cat > /DATA/stephen/rollm_env.sh <<'EOF'
source /DATA/stephen/env.sh
conda activate /DATA/stephen/envs/rollm
export HF_DATASETS_CACHE=/DATA/data_cache
EOF
```

`HF_DATASETS_CACHE` points at the shared cache because `ro_mmlu` is already
there. None of the six datasets carries a loading script -- they are plain JSON
data repos -- so the datasets version is not delicate; the transformers pin is,
which is why it goes in before the editable install.

## Running

```bash
export SETUP=~/rollm_env.sh ROLLM=~/lm-evaluation-harness
datagen/retention/run_rollm_eval.sh OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23 base_rollama31-dpo
datagen/retention/run_rollm_eval.sh $WORKDIR/merged_models/rollama31-dpo_4kv2_s0 rollama31-dpo_4kv2_s0
```

Results land in `retention/results/<name>/fs<K>/`, with a `provenance_<name>.txt`
recording the harness commit, the stack versions, the seed, the batch size, the
rungs and the limit. A row without that file cannot be placed in a table with
another row. Finished rungs are skipped on a rerun, so a killed run resumes;
`FORCE=1` redoes them.

### Order

1. **`base_rollama31-dpo`, full protocol.** One run, and it gates everything
   else: if it does not land near 53.76 the environment is not reproducing
   OpenLLM-Ro's and the published column cannot be cited. It is the only full
   run in the plan -- its job is the published comparison, not a pairing.

2. **Everything else at `LIMIT=2000`**, base and repaired alike, so each pair is
   like-for-like. At that limit only MMLU and HellaSwag are truncated (the other
   four tasks hold fewer than 2,000 documents), which removes ~87% of the
   requests per rung and brings a checkpoint from ~8 h to ~2 h. Both arms see
   the same first 2,000 documents, so the delta -- the actual §3.3 answer --
   survives intact; what does not survive is the *level* on those two tasks
   being the published quantity. Say so wherever the numbers are reported.

   - `base_rollama31-dpo_lim`, `rollama31-dpo_4kv2_s{0,1,2}` — the headline
     family with its three seeds, which §3.4 already established as the unit of
     uncertainty for this study.
   - Three further pairs, base and seed 0 only, chosen for what each one puts at
     risk rather than for coverage of the table:
     **RoGemma2-9B-DPO** (the second documented lineage, a different
     architecture, and the checkpoint whose released `eos_token_id` makes its
     GMTW delta hardest to read), **RoGemma-7B-DPO** (the largest ΔU of the
     eight at +0.497 -- the most-changed model, so the most exposed to
     collateral damage), and **RoMistral-7B-SFT** (the SFT arm and the null
     case, ΔF_v2 +0.013 failing Holm -- the least-changed model, where a
     retention loss would be the hardest result to explain away).

Roughly a day of H200 after the anchor run. The four pairs left out --
RoLlama3-8B-DPO, RoMistral-7B-DPO, RoGemma2-9B-SFT, RoLlama2-7B -- are each a
same-family duplicate of a pair already in, and are the first thing to add back
if GPU time frees up.

### Two things that break the pairing

- **Chat template.** OpenLLM-Ro scored these checkpoints as plain completions.
  The runner keeps `--apply_chat_template` off for that reason. Turning it on for
  the repaired arm alone would compare a chat-formatted model to a completion-
  formatted one and call the difference retention.
- **Stop tokens.** `run_family.sh` aligns `eos_token_id` on merged checkpoints
  (Gemma-2 merges carry `[1, 107]`, the released checkpoint `1`). That asymmetry
  is invisible on the multiple-choice tasks, which score log-likelihoods, but it
  is live on the generative ones (GSM8k). If a Gemma-2 generative delta looks
  large, check that first.
