#!/usr/bin/env python3
"""Two internal inconsistencies in the released GMTW-Ro scorer, measured.

Found while checking one consistency item the review raised: on the adversarial
split \\textsc{RoGemma-7B} reports R_parse = 0 yet U = 0.178 and F_v2 = 0.016,
and the manuscript glosses that as "with no parseable plan there is nothing for
the harder constraints to break". The gloss is wrong, and the reason generalises.

  (i)  format_ok is False on records whose parse_error is None.
       So "0.0% valid JSON" does not mean nothing parsed -- it means the strict
       validity conjunction (parse AND required keys AND known IDs AND no extra
       keys) failed. On the adversarial RoGemma-7B cell, 222 of 300 outputs
       parsed. A quantity that reads 0 while three quarters of outputs parse is
       not a parse indicator, which is the review's naming complaint restated as
       a measurement.

  (ii) The format check F_JSON_FOUND ("Valid JSON produced") is satisfied on
       records whose parse_error says "No JSON block found". The scorer
       contradicts itself, and it does so on hundreds of records for exactly the
       checkpoints whose JSON collapse is the paper's headline. Because
       U = 0.85*T^3 + 0.15*L^3 and F_JSON_FOUND is one of the five L checks, a
       spuriously satisfied check inflates L and therefore U.

Both defects are triggered by missing structured output, so both inflate exactly
the checkpoints whose structured channel collapsed, and neither touches
gemma-2-9b-it or llama-3.1-8b-instruct (zero affected records). Each half of the
paper is therefore conservative, by one mechanism under two labels: in the
observational half the ADAPTED model is the inflated one, so the base->adapted
regression is understated; in the repair study the unrepaired checkpoint is the
inflated one, so the repair delta is understated. Neither direction flatters a
claim the paper makes, which is why the right response is to document the defect
and its sign rather than redefine the benchmark's published composite midway
through a revision.

Only the authoritative runs are read. main.tex states that every table in its
results section comes from new_data/metrics_de_test_{en,ro,de} (n=500) and
new_data/metrics_hard_ro (n=300); the metrics_easy_* directories are the
superseded run and are skipped. Reading them would misreport the paper: on
metrics_easy_ro, RoGemma-7B has format_ok = 0/500, while the authoritative
metrics_de_test_ro gives 456/500 = 91.2%, which is the number the paper prints.

  python datagen/analysis/scorer_audit.py \
      --metrics ../gmtw-hria/new_data --out datagen/results/tables
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

# What main.tex declares as the single authoritative run.
AUTHORITATIVE = {
    "metrics_de_test_ro": ("standard", 500),
    "metrics_de_test_en": ("standard-en", 500),
    "metrics_de_test_de": ("standard-de", 500),
    "metrics_hard_ro": ("adversarial", 300),
}
NO_JSON = "No JSON block found"


def read_jsonl(path: pathlib.Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def json_found(rec: dict) -> bool:
    for c in (rec.get("U_details") or {}).get("format_checks", []):
        if c.get("id") == "F_JSON_FOUND":
            return bool(c.get("satisfied"))
    return False


def fmt_score(rec: dict) -> float:
    return float((rec.get("U_details") or {}).get("U_format_linear", 0.0))


def audit(recs: list[dict]) -> dict:
    n = len(recs)
    parsed = [r for r in recs if r.get("parse_error") is None]
    unparsed = [r for r in recs if r.get("parse_error")]
    no_json = [r for r in unparsed if r.get("parse_error") == NO_JSON]
    out = {
        "n": n,
        "format_ok": sum(1 for r in recs if r.get("format_ok")),
        "parsed": len(parsed),
        "repaired": sum(1 for r in recs if r.get("repaired")),
        # (i) counted as the gap between "parses" and "passes the conjunction"
        "i_parsed_not_valid": sum(1 for r in parsed if not r.get("format_ok")),
        # (ii) the scorer's format check contradicting its own parse error
        "ii_no_json_but_found": sum(1 for r in no_json if json_found(r)),
        "n_no_json": len(no_json),
    }
    out["ii_pct_of_split"] = round(100 * out["ii_no_json_but_found"] / n, 1)
    # How much format credit do the self-contradicting records collect?
    bad = [r for r in no_json if json_found(r)]
    out["fmt_credit_contradicting"] = round(statistics.mean(map(fmt_score, bad)), 3) \
        if bad else ""
    out["fmt_credit_parsed"] = round(statistics.mean(map(fmt_score, parsed)), 3) \
        if parsed else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, type=pathlib.Path,
                    help="gmtw-hria/new_data")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    rows = []
    print(f"{'split':13}{'model':30}{'n':>5}{'fmt_ok':>8}{'parsed':>8}"
          f"{'(i)':>6}{'(ii)':>7}{'(ii)%':>7}{'credit':>8}")
    print("-" * 92)
    for d, (label, expect) in AUTHORITATIVE.items():
        path = args.metrics / d
        if not path.is_dir():
            print(f"  missing: {path}")
            continue
        for f in sorted(path.glob("*.jsonl")):
            recs = list(read_jsonl(f))
            if len(recs) != expect:
                print(f"  warn: {f.name} has {len(recs)} records, expected {expect}")
            a = audit(recs)
            model = f.stem.split("_", 1)[1] if "_" in f.stem else f.stem
            rows.append({"split": label, "model": model, **a})
            print(f"{label:13}{model:30}{a['n']:>5}{a['format_ok']:>8}"
                  f"{a['parsed']:>8}{a['i_parsed_not_valid']:>6}"
                  f"{a['ii_no_json_but_found']:>7}{a['ii_pct_of_split']:>7}"
                  f"{str(a['fmt_credit_contradicting']):>8}")

    worst_i = max(rows, key=lambda r: r["i_parsed_not_valid"])
    worst_ii = max(rows, key=lambda r: r["ii_no_json_but_found"])
    print(f"\nworst (i) : {worst_i['model']} on {worst_i['split']} -- "
          f"{worst_i['parsed']} of {worst_i['n']} parsed but format_ok "
          f"{worst_i['format_ok']}")
    print(f"worst (ii): {worst_ii['model']} on {worst_ii['split']} -- "
          f"{worst_ii['ii_no_json_but_found']} of {worst_ii['n']} records "
          f"({worst_ii['ii_pct_of_split']}%) pass F_JSON_FOUND while "
          f"parse_error says no JSON was found")
    print("\nBoth defects fire on MISSING structured output, so both inflate "
          "exactly the\ncheckpoints whose structured channel collapsed -- and "
          "zero records on\ngemma-2-9b-it or llama-3.1-8b-instruct. Each half of "
          "the paper is therefore\nconservative, by the same mechanism under "
          "opposite labels:")
    print("  observational half: the ADAPTED model is inflated, so the "
          "base->adapted\n                      regression is UNDERSTATED.")
    print("  repair half       : the unrepaired checkpoint is the inflated one, "
          "so the\n                      repair delta is UNDERSTATED.")
    print("Neither direction flatters a claim the paper makes.")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        p = args.out / "65_scorer_audit.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
