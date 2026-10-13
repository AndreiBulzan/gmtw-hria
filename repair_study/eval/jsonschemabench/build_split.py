"""Measure structural overlap between a JSONSchemaBench eval set and the
training pool, and optionally emit a copy with the colliding schemas dropped.

filters/dedupe.py fingerprints STRUCTURE -- types and constraint kinds, with
names, titles, descriptions and concrete constraint values ignored -- so this
catches the same shape under different project names, which neither the id nor
the schema text would.

READ THIS BEFORE FILTERING ANYTHING (revised 2026-08-07):

Structural overlap with training is NOT contamination for this benchmark, and
an earlier version of this docstring implied it was. The metric here is schema
VALIDITY, not match-to-reference: there is no answer to memorize, and even
regurgitating a memorized training completion produces a valid instance.
Memorization and success are the same event, so the usual inflation mechanism
does not exist. Measured on the current test_split: 0.3% exact duplicates,
6.2% unique structural twins, and most of the remaining "collisions" are a
common shape being common (Glaiveai2K reads 89.1% any-collision but only 5.7%
unique twins, because function-call schemas repeat one shape endlessly).

So the default use of this tool is DESCRIPTIVE -- run it to report what an
eval set overlaps with, so a reader can calibrate. Do not filter a sound
retention set with it.

The two eval sets answer different questions, and both are legitimate:

  in-distribution retention -- upstream *test* splits of the four TRAINED
  sources. Unseen instances, same sources and complexity band. This is the
  primary JSONSchemaBench result and it is fine as measured; describe it
  precisely rather than calling it a holdout.

  source-level probe -- Github_hard/ultra, Kubernetes, JsonSchemaStore,
  Snowplow, WashingtonPost. Never trained on at any level. A SECOND question
  (does the repair hold on unseen schema families and complexity bands?), not
  a replacement. Note the fetch script's own caveat: these are also longer and
  more complex than anything in training, so a drop here mixes length
  generalization with source generalization -- only interpretable next to a
  same-family baseline on the same schemas.

The load-bearing rule from CLAUDE.md is the comparative one: never compare
these numbers against the public JSONSchemaBench leaderboard after training on
its schemas. That leaderboard scores constrained-decoding engines on the whole
benchmark; our numbers are a before/after on our own model and nothing else.

Usage:
  # 1. fetch the source-level holdout (eval-only path, never data/raw/)
  python -m pipelines.fetch_jsonschemabench \
      --sources "Github_hard,Github_ultra,Kubernetes,JsonSchemaStore,Snowplow,WashingtonPost" \
      --allow-holdout \
      --out eval/jsonschemabench_holdout/hard_ultra_raw.jsonl

  # 2. drop anything structurally present in the training pool
  python -m eval.jsonschemabench_holdout.build_split \
      --train-pool data/raw/jsonschemabench/train_pool.jsonl \
      --eval-in eval/jsonschemabench_holdout/hard_ultra_raw.jsonl \
      --out eval/jsonschemabench_holdout/source_holdout.jsonl

  # same treatment for the retention set, so both are reported on clean data
  python -m eval.jsonschemabench_holdout.build_split \
      --train-pool data/raw/jsonschemabench/train_pool.jsonl \
      --eval-in eval/jsonschemabench_holdout/test_split.jsonl \
      --out eval/jsonschemabench_holdout/test_split_deduped.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

from filters.dedupe import schema_fingerprint
from pipelines.common import read_jsonl


def source_of(record_id: str) -> str:
    return record_id.split("/")[0] if "/" in record_id else "?"


def fingerprint_pool(path: pathlib.Path) -> set[str]:
    prints = set()
    for rec in read_jsonl(path):
        schema = rec.get("schema", rec)
        try:
            prints.add(schema_fingerprint(schema))
        except Exception as e:            # malformed schema: can't fingerprint
            print(f"  warn: {rec.get('id', '?')}: {e}", file=sys.stderr)
    return prints


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-pool", required=True,
                    help="the pool Track A actually trained on")
    ap.add_argument("--eval-in", required=True, nargs="+")
    ap.add_argument("--out", required=True, help="refuses to overwrite")
    ap.add_argument("--dedupe-within", action="store_true",
                    help="also collapse structural duplicates inside the eval "
                         "set itself (keeps first occurrence). Off by default: "
                         "it changes N, so it is a reporting decision, not a "
                         "contamination one.")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out} — pick a fresh path")

    train_path = pathlib.Path(args.train_pool)
    if not train_path.exists():
        sys.exit(f"training pool not found: {train_path}\n"
                 f"This must be the pool the model was ACTUALLY trained on — "
                 f"regenerating it with a different seed or source list would "
                 f"make the dedupe meaningless.")

    train_prints = fingerprint_pool(train_path)
    print(f"training pool: {len(train_prints)} distinct structural fingerprints",
          file=sys.stderr)

    kept, dropped_train, dropped_within = [], [], []
    seen_within: set[str] = set()
    per_source_total: collections.Counter = collections.Counter()
    per_source_dropped: collections.Counter = collections.Counter()

    for path in args.eval_in:
        for rec in read_jsonl(path):
            schema = rec.get("schema", rec)
            src = source_of(rec.get("id", ""))
            per_source_total[src] += 1
            try:
                fp = schema_fingerprint(schema)
            except Exception as e:
                print(f"  warn: unfingerprintable, dropping {rec.get('id','?')}: {e}",
                      file=sys.stderr)
                continue
            if fp in train_prints:
                dropped_train.append(rec)
                per_source_dropped[src] += 1
                continue
            if args.dedupe_within:
                if fp in seen_within:
                    dropped_within.append(rec)
                    continue
                seen_within.add(fp)
            kept.append(rec)

    total = sum(per_source_total.values())
    print(f"\n{'source':<24}{'total':>8}{'collide':>9}{'rate':>8}", file=sys.stderr)
    for src in sorted(per_source_total):
        n, d = per_source_total[src], per_source_dropped[src]
        print(f"{src:<24}{n:>8}{d:>9}{100*d/n if n else 0:>7.1f}%", file=sys.stderr)
    print(f"{'TOTAL':<24}{total:>8}{len(dropped_train):>9}"
          f"{100*len(dropped_train)/total if total else 0:>7.1f}%", file=sys.stderr)
    if args.dedupe_within:
        print(f"\nalso collapsed {len(dropped_within)} structural duplicates "
              f"within the eval set", file=sys.stderr)
    print(f"\nkept {len(kept)} -> {out}", file=sys.stderr)

    if not kept:
        sys.exit("every eval schema collided with training — nothing to report on")

    with open(out, "w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "train_pool": str(train_path),
        "train_fingerprints": len(train_prints),
        "eval_inputs": list(args.eval_in),
        "total_in": total,
        "dropped_train_collision": len(dropped_train),
        "dropped_within_eval": len(dropped_within),
        "kept": len(kept),
        "per_source_total": dict(per_source_total),
        "per_source_dropped": dict(per_source_dropped),
        "dedupe_within": args.dedupe_within,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
