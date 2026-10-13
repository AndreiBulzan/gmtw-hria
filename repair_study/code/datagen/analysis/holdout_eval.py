#!/usr/bin/env python3
"""English schema-following retention, on unseen instances of the trained sources.

The review asks whether near-perfect JSON after the repair reflects restored
structured-output competence or adaptation to one benchmark's shape. This
answers it in a different language, on a different distribution, with a scorer
that shares nothing with GMTW-Ro.

READ THIS BEFORE DESCRIBING THE RESULT (corrected 2026-10-02):

This is RETENTION, not a source-level holdout, and an earlier version of this
docstring said the opposite. It claimed the split came from
fetch_schemas.py:HOLDOUT_SOURCES (Github_hard, Github_ultra, Kubernetes,
JsonSchemaStore, Snowplow, WashingtonPost) and that nothing in the corpus was
drawn from them. The file this runner is actually pointed at,
test_split.jsonl, contains none of those sources. It contains Github_medium
(586), Github_easy (577), Glaiveai2K (513) and Github_trivial (128) -- exactly
the four-source training pool. The source-level probe was never built.

What the set IS, and it is sound: upstream *test*-split instances of the four
trained sources. Training drew only from `train` and `val`, and instance-id
overlap is zero out of 1,804 (datagen/analysis/holdout_overlap.py). So these are
unseen instances in the same sources and complexity band -- a legitimate
retention measurement, and the primary JSONSchemaBench result. Describe it that
way. A claim about generalising to unseen schema FAMILIES needs the
HOLDOUT_SOURCES probe, which has not been run.

Overlap with training, measured rather than assumed: 1 of 1,804 schemas is an
exact duplicate (0.06%; dropped for the clean denominator of 1,209, which moves
every delta by at most 0.06 points), while 30.3% of records -- 40.1% of the
scored subset -- share a structural fingerprint with a training schema, mostly
Glaiveai2K function-call shapes. The twins are not filtered: schema validity has
no reference answer, so a memorised completion still has to validate. The rate
is reported instead.

The split is 1,804 schemas. A file of 3,608 was this split concatenated with
itself and is kept as test_split.jsonl.doubled_bak; do not score it.

Scoring is the training gate itself, `datagen.core.validation.run_gate`: parse,
load the schema at its own draft, validate. That is deliberate. The metric is
schema VALIDITY, not match-to-reference, so there is no answer to memorise --
a model that regurgitated a training completion verbatim would still have to
produce something that validates against a schema it has never seen.

A third of these schemas are permissive enough that a degenerate answer
satisfies them -- a bare `{}` validates against 24.9%, and some trivial value
against 32.9%. A raw pass rate therefore has a floor a model can reach by
answering `{}` every time, which makes it a poor number to compare two
checkpoints on. The runner reports the rate restricted to the schemas that
reject every trivial answer as well, and that is the one to quote: on those,
passing requires actually constructing an instance.

The gate is applied the way track_a.py applies it, with an extract_first_json
retry, and that is not a detail. `run_gate` alone calls json.loads on the whole
response, so a prose-then-JSON answer fails at the parse stage -- and every
repaired checkpoint here is trained to answer prose first. Scoring them without
the retry reports near-zero and reads as the repair having destroyed English
schema-following, which is the opposite of what happened.

  python datagen/analysis/holdout_eval.py \\
      --split ro-so/eval/jsonschemabench_holdout/test_split.jsonl \\
      --model-path merged_models/rollama31-dpo_4kv2_s0 \\
      --name rollama31-dpo_4kv2_s0 \\
      --out datagen/results/holdout

Writes <name>_pass.jsonl and <name>_fail.jsonl, matching the record shape the
earlier holdout run used so paper/scripts/extract_evidence.py keeps working.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from datagen.core.common import extract_first_json, load_templates
from datagen.core.validation import run_gate


TRIVIAL = ["{}", "[]", '""', "0", "null", "true", "[ ]"]


def trivially_satisfiable(records, cache: pathlib.Path) -> dict:
    """id -> True when some degenerate answer already validates.

    Computed once and cached: it is a property of the split, not of any model,
    and it takes a couple of minutes over 3,608 schemas."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = {}
    for r in records:
        hit = False
        for text in TRIVIAL:
            try:
                if run_gate(r["schema"], text).passed:
                    hit = True
                    break
            except Exception:
                pass
        out[r["id"]] = hit
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def build_prompts(records, templates, seed):
    """One instruction per schema, phrasing drawn with a fixed seed.

    Several phrasings rather than one, for the same reason the corpus has
    several: a single wording measures how well the model handles that wording.
    Seeded, so the prompt set is identical across every model compared.
    """
    rng = random.Random(seed)
    out = []
    for r in records:
        schema_text = json.dumps(r["schema"], ensure_ascii=False, indent=2)
        tpl = templates[rng.randrange(len(templates))]
        out.append(tpl.format(schema=schema_text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--name", required=True, help="output file stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--templates", default="templates/instructions_en.txt")
    ap.add_argument("--max", type=int, default=None, help="cap for a smoke run")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--prompt-seed", type=int, default=20240617,
                    help="picks the phrasing per schema; identical across models")
    ap.add_argument("--lora", action="append", default=None, metavar="NAME=PATH",
                    help="serve a LoRA adapter on top of --model-path instead of "
                         "a merged checkpoint, as NAME=PATH. Repeatable: several "
                         "adapters share one base load and one download, which is "
                         "what makes a per-seed sweep fit on a full disk. Each "
                         "adapter writes its own <NAME>_pass/_fail.jsonl.")
    ap.add_argument("--max-lora-rank", type=int, default=16,
                    help="must be >= the adapters' r (training used 16)")
    ap.add_argument("--trivial-cache", default=None,
                    help="where to cache which schemas accept a degenerate "
                         "answer (default: alongside the split)")
    args = ap.parse_args()

    split = pathlib.Path(args.split)
    records = [json.loads(l) for l in open(split, encoding="utf-8") if l.strip()]
    if args.max:
        records = records[:args.max]
    cache = pathlib.Path(args.trivial_cache) if args.trivial_cache \
        else split.with_name(split.stem + "_trivially_satisfiable.json")
    trivial = trivially_satisfiable(records, cache)
    n_hard = sum(1 for r in records if not trivial.get(r["id"], False))
    print(f"  non-gameable  : {n_hard} of {len(records)} schemas reject every "
          f"trivial answer", file=sys.stderr)
    templates = load_templates(args.templates)
    prompts = build_prompts(records, templates, args.prompt_seed)
    print(f"  schemas   : {len(records)}", file=sys.stderr)
    print(f"  phrasings : {len(templates)}", file=sys.stderr)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]

    # Greedy, matching the benchmark's own decoding so the two evaluations
    # differ in the task and not in how the model was sampled.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    if args.lora:
        from vllm.lora.request import LoRARequest
        pairs = []
        for spec in args.lora:
            if "=" not in spec:
                sys.exit(f"--lora wants NAME=PATH, got {spec!r}")
            nm, pth = spec.split("=", 1)
            if not pathlib.Path(pth).is_dir():
                sys.exit(f"--lora path is not a directory: {pth}")
            pairs.append((nm, pth))
        llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
                  enable_lora=True, max_loras=1, max_lora_rank=args.max_lora_rank)
        # One adapter at a time through the same loaded base. max_loras=1 keeps
        # the LoRA memory constant regardless of how many are swept, and the
        # adapters are 177 MB each so swapping is cheap next to a model load.
        runs = []
        for i, (nm, pth) in enumerate(pairs, start=1):
            print(f"  lora {i}/{len(pairs)}: {nm}", file=sys.stderr, flush=True)
            o = llm.generate(chats, sampling,
                             lora_request=LoRARequest(nm, i, pth))
            runs.append((nm, [x.outputs[0].text for x in o]))
    else:
        llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True)
        outs = llm.generate(chats, sampling)
        runs = [(args.name, [o.outputs[0].text for o in outs])]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_name, texts in runs:
      score_one(run_name, texts, records, prompts, trivial, out_dir)


def score_one(run_name, texts, records, prompts, trivial, out_dir) -> None:
    """Gate one run's generations and write its pass/fail files.

    Identical to the single-model path; factored out so a LoRA sweep scores
    every adapter exactly as a merged checkpoint would be scored.
    """
    n_pass, n_pass_hard, n_rescued = 0, 0, 0
    stages = {}
    with open(out_dir / f"{run_name}_pass.jsonl", "w", encoding="utf-8") as fp, \
         open(out_dir / f"{run_name}_fail.jsonl", "w", encoding="utf-8") as ff:
        for rec, prompt, text in zip(records, prompts, texts):
            gate = run_gate(rec["schema"], text)
            if not gate.passed and gate.stage == "parse":
                # Prose before the JSON block is the answer shape these models
                # were trained to produce; trimming to the first balanced object
                # is what track_a.py does and does not weaken the gate, since
                # the candidate still has to parse AND validate.
                candidate = extract_first_json(text)
                if candidate:
                    regate = run_gate(rec["schema"], candidate)
                    if regate.passed:
                        gate = regate
                        n_rescued += 1
            row = {"id": rec["id"], "instruction": prompt, "schema": rec["schema"],
                   "track": "holdout",
                   "meta": {"model": run_name, "stratum": rec.get("stratum"),
                            "source": rec["id"].split("/")[0],
                            "trivially_satisfiable": trivial.get(rec["id"], False)}}
            if gate.passed:
                n_pass += 1
                if not trivial.get(rec["id"], False):
                    n_pass_hard += 1
                row["output"] = gate.output
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                stages[gate.stage] = stages.get(gate.stage, 0) + 1
                row["output"] = text
                row["gate_stage"] = gate.stage
                row["validator_errors"] = gate.errors[:5]
                ff.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(records)
    n_hard = sum(1 for r in records if not trivial.get(r["id"], False))
    print(f"\n{run_name}: {n_pass}/{n} valid ({100 * n_pass / n:.1f}%)")
    print(f"  on non-gameable schemas: {n_pass_hard}/{n_hard} "
          f"({100 * n_pass_hard / max(n_hard, 1):.1f}%)   <-- quote this one")
    print(f"  of those, {n_rescued} needed the prose-then-JSON retry")
    if stages:
        print("  failures by stage: "
              + ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))


if __name__ == "__main__":
    main()
