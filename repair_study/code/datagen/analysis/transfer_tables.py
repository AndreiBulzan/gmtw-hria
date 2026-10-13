#!/usr/bin/env python3
"""English schema-following on the held-out split: base against repaired.

`holdout_eval.py` runs one checkpoint and writes its pass/fail records. This
turns those records into the transfer table, which is the part the review asks
for: whether the repair generalises, or whether near-perfect JSON on GMTW-Ro is
adaptation to one benchmark's shape.

Four things differ from GMTW-Ro at once -- the task (produce an instance of a
raw JSON Schema, no worlds, no plans, no constraints), the language, the data
(JSONSchemaBench sources held out of training by design) and the scorer. Nothing
is memorisable: the metric is schema VALIDITY, not match-to-reference, so a
model regurgitating a training completion would still have to produce something
that validates against a schema it has never seen.

The rate reported is over the schemas that reject every trivial answer. A bare
`{}` validates against a quarter of this split, some trivial value against a
third, so a raw pass rate has a floor a model reaches by answering `{}` every
time -- useless for comparing two checkpoints. Which schemas those are is
recorded per record by the runner (`meta.trivially_satisfiable`), so this script
does not recompute it and cannot disagree with what was scored.

Standard library only, like repair_tables.py, so it runs wherever the records
are.

  python datagen/analysis/transfer_tables.py --results datagen/results/holdout
  python datagen/analysis/transfer_tables.py --results <dir> --out <csv-dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

# (display name, base file stem, repaired file stem). Ordered by base rate at
# the end rather than here.
MODELS = ["RoLlama3.1-8B-DPO", "RoGemma2-9B-DPO", "RoGemma2-9B-SFT",
          "RoLlama3-8B-DPO", "RoMistral-7B-DPO", "RoMistral-7B-SFT",
          "RoLlama2-7B", "RoGemma-7B-DPO"]

# Corpus ablations, all repaired from RoLlama3.1-8B-DPO seed 0.
ABLATIONS = [("full mix", "ours_RoLlama3.1-8B-DPO"),
             ("ro-only", "abl_ro-only"),
             ("no Track C", "abl_no-c"),
             ("no Track A", "abl_no-a")]


def score(results: pathlib.Path, stem: str):
    """(passed, total) restricted to schemas no trivial answer satisfies.

    Counts records, not distinct ids. Reading ids into a dict is what made an
    earlier version of this report rates above 100%: the split held every schema
    twice, so a deduplicated denominator sat under a doubled numerator."""
    n_hard = n_pass = 0
    for suffix, is_pass in (("_pass.jsonl", True), ("_fail.jsonl", False)):
        path = results / f"{stem}{suffix}"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["meta"].get("trivially_satisfiable"):
                    continue
                n_hard += 1
                n_pass += is_pass
    return n_pass, n_hard


def pct(pair):
    return 100.0 * pair[0] / pair[1] if pair and pair[1] else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="directory of <name>_pass.jsonl / <name>_fail.jsonl")
    ap.add_argument("--out", default=None, help="also write CSVs here")
    args = ap.parse_args()

    results = pathlib.Path(args.results)
    if not results.is_dir():
        sys.exit(f"not a directory: {results}")

    rows, missing = [], []
    for name in MODELS:
        b = score(results, f"base_{name}")
        o = score(results, f"ours_{name}")
        if b is None or o is None:
            missing.append(name)
            continue
        rows.append((name, b, o))
    if not rows:
        sys.exit(f"no paired base/ours results under {results}")
    rows.sort(key=lambda r: pct(r[1]))

    n_hard = rows[0][1][1]
    print("=" * 76)
    print(f"ENGLISH SCHEMA-FOLLOWING  --  held-out JSONSchemaBench sources, "
          f"{n_hard} schemas")
    print("=" * 76)
    print("%-20s %10s %10s %10s" % ("checkpoint", "base", "ours", "change"))
    print("-" * 76)
    csv_rows = []
    for name, b, o in rows:
        print("%-20s %9.1f%% %9.1f%% %+9.1f" % (name, pct(b), pct(o), pct(o) - pct(b)))
        csv_rows.append([name, b[1], b[0], round(pct(b), 2), o[0],
                         round(pct(o), 2), round(pct(o) - pct(b), 2)])
    deltas = [pct(o) - pct(b) for _, b, o in rows]
    print("-" * 76)
    print("  range %+.1f to %+.1f   mean %+.1f   improved on %d of %d" %
          (min(deltas), max(deltas), sum(deltas) / len(deltas),
           sum(1 for d in deltas if d > 0), len(deltas)))
    if missing:
        print(f"  missing (skipped): {', '.join(missing)}")

    abl_rows = []
    have = [(l, s) for l, s in ABLATIONS if score(results, s)]
    if len(have) > 1:
        ref = pct(score(results, have[0][1]))
        print()
        print("=" * 76)
        print("CORPUS ABLATIONS  --  same measurement, seed 0 of each mix")
        print("=" * 76)
        for label, stem in have:
            s = score(results, stem)
            print("  %-14s %5.1f%%  (%d/%d)%s" %
                  (label, pct(s), s[0], s[1],
                   "" if label == have[0][0] else "   %+.1f vs full" % (pct(s) - ref)))
            abl_rows.append([label, s[1], s[0], round(pct(s), 2),
                             round(pct(s) - ref, 2)])

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "47_transfer_english.csv", "w", encoding="utf-8",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["checkpoint", "n_non_gameable", "base_pass", "base_pct",
                        "ours_pass", "ours_pct", "delta_pts"])
            w.writerows(csv_rows)
        print(f"wrote {out / '47_transfer_english.csv'}", file=sys.stderr)
        if abl_rows:
            with open(out / "48_transfer_ablations.csv", "w", encoding="utf-8",
                      newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["mix", "n_non_gameable", "pass", "pct", "delta_vs_full"])
                w.writerows(abl_rows)
            print(f"wrote {out / '48_transfer_ablations.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
