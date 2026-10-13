#!/usr/bin/env python3
"""Repair-section tables, scored under the GMTW-Ro paper's own conventions.

Turns per-instance metric records (the `*_500.jsonl` files the harness writes)
into the three tables the repair section argues from. Standard library only, so
it runs anywhere the records are -- no VM, no model, no rombench -- which is the
reproducibility standard the paper sets for its own corrections.

Two conventions are taken from the paper rather than invented here:

  F_v2  An empty plan scores 0, not the 1 the released scorer awards. The
        released convention inflates exactly the checkpoints that fail hardest:
        a model emitting no plan has nothing to be unfaithful about. Verified
        against the paper's published numbers by the identity
        F_v2 == mean(F | non-empty) * n_nonempty / n, asserted at run time.

        "Empty" means `entities_total == 0` -- no entity resolved out of the
        plan -- which is the paper's definition and NOT the same as the
        scorer's `note == "Empty plan"`, which fires only when the plan carried
        no references at all. The two differ by a lot: 215 records against 63
        on the RoLlama2 baseline. Using the note is what made our numbers
        disagree with the paper's and produced an apparent pipeline
        discrepancy; on the definition below they agree to within ordinary
        serving variation (135 vs 143, 215 vs 230, 440 vs 443).

        Known consequence, and it is the paper's convention rather than ours:
        the `fact` family carries no plan entities by construction, so all 100
        of its instances count as empty for every model and score 0 here. That
        is a fixed offset applied equally to every row -- it moves levels by
        about 0.2 and leaves rankings and paired deltas essentially untouched.

  CIs   Percentile bootstrap, 5,000 resamples, seed 20240617 for single-split
        estimates and seed 42 for paired contrasts.

Our trained rows average three training seeds. The bootstrap resamples the 500
INSTANCES, using each instance's mean over seeds, so the interval is the
instance-level sampling variability the paper reports. Seed-to-seed variability
is a different quantity and is reported separately as `final_spread`.

  python datagen/analysis/repair_tables.py --analysis <analysis-dir>
  python datagen/analysis/repair_tables.py --analysis <dir> --out <csv-dir>

Lives beside the generator on purpose. `datagen/` is meant to be dropped into
gmtw-hria as `rombench/datagen/`, and a corpus that travels without the script
that scores what it produced is not reproducible by whoever receives it. Like
the rest of the folder this imports nothing from this repository -- here, not
even `jsonschema` -- so it runs wherever the metric records are.

Caveat that belongs with any use of these numbers: our baselines reproduce the
paper's U and G closely but NOT its empty-plan rate, and therefore not F_v2 or
the end-to-end criterion (we serve with vLLM, the paper with HuggingFace
transformers). The paired deltas are computed within our own harness on both
sides and are unaffected; the absolute end-to-end column is not comparable to
the paper's table.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
import sys

N_BOOT = 5000
SEED_SINGLE = 20240617      # paper: single-split tables
SEED_PAIRED = 42            # paper: paired contrasts

# (display name, baseline dir, trained-run prefix)
ROWS = [
    ("RoLlama3.1-8B-DPO", "RoLlama3.1-8b-Instruct-DPO",  "rollama31-dpo_4kv2"),
    ("RoGemma2-9B-DPO",   "RoGemma2-9b-Instruct-DPO",    "rogemma2-dpo_4kv2"),
    ("RoGemma2-9B-SFT",   "RoGemma2-9b-Instruct",        "rogemma2-sft_4kv2"),
    ("RoLlama3-8B-DPO",   "RoLlama3-8b-Instruct-DPO",    "rollama3-dpo_4kv2"),
    ("RoMistral-7B-DPO",  "RoMistral-7b-Instruct-DPO",   "romistral-dpo_4kv2"),
    ("RoMistral-7B-SFT",  "RoMistral-7b-Instruct",       "romistral-sft_4kv2"),
    # The undated alias, which is what the benchmark harness resolves and what
    # every published baseline was measured on. The dated 2024-10-09 release is
    # DIFFERENT weights -- the sole family where the two diverge.
    ("RoLlama2-7B",       "RoLlama2-7b-Instruct-undated", "rollama2-sft_4kv2"),
    ("RoGemma-7B-DPO",    "RoGemma-7b-Instruct-DPO",     "rogemma-dpo_4kv2"),
]
SEEDS = (0, 1, 2)

# Corpus-composition ablations: same model, same hyperparameters, same 4,000
# record budget, three seeds each. Only the mix differs, so a difference here is
# attributable to what was removed and to nothing else.
ABLATIONS = [
    ("full mix",   "rollama31-dpo_4kv2"),
    ("ro-only",    "rollama31-dpo_abl-ro-only"),
    ("no Track C", "rollama31-dpo_abl-no-c"),
    ("no Track A", "rollama31-dpo_abl-no-a"),
    ("no Track B", "rollama31-dpo_abl-no-b"),
]


def load(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def is_empty(r: dict) -> bool:
    """The degenerate output F was rewarding: a plan with no entities.

    `entities_total == 0`, matching the paper. Deliberately NOT the scorer's
    own `note == "Empty plan"`, which is a narrower condition -- it fires only
    when the plan held no references at all, and misses the case where refs
    were present but none resolved to an entity in the world."""
    ents = r.get("F_details", {}).get("entities_total")
    if ents is None:
        raise KeyError(
            "F_details.entities_total missing -- this metrics file predates the "
            "field and cannot be scored on the paper's empty-plan definition. "
            "Re-score it from its *_outputs.jsonl before using it.")
    return ents == 0


def f_v2(r: dict) -> float:
    return 0.0 if is_empty(r) else r["F"]


def t_raw(r: dict) -> float:
    """Constraint-satisfaction fraction before U's severity exponent.

    U is 0.85*T^3 + 0.15*L^3, so a change in U can come from constraints or
    from the format checks folded into L. Reporting T beside it separates the
    two, which is what the review asks for."""
    u = r.get("U_details", {})
    total = u.get("constraints_total")
    if not total:
        return None
    return u["constraints_satisfied"] / total


def all_constraints(r: dict) -> bool:
    u = r.get("U_details", {})
    total = u.get("constraints_total")
    return bool(total) and u.get("constraints_satisfied") == total


def end_to_end(r: dict) -> bool:
    """Parses, satisfies EVERY constraint, and faithfully explains a non-empty
    plan. Conjunctive on purpose: no single-axis degeneracy can inflate it."""
    return (bool(r.get("format_ok")) and all_constraints(r)
            and not is_empty(r) and r["F"] == 1.0)


def check_f_v2_identity(recs: list[dict], where: str) -> None:
    """F_v2 must equal mean(F | non-empty) * n_nonempty / n.

    This is the identity that reproduces the paper's published F_v2 values from
    its own F and empty counts, so it is the check that our correction is the
    same correction. It also catches a records file whose empty plans do not
    score exactly 1.0, in which case zeroing them is not the right operation."""
    ne = [r for r in recs if not is_empty(r)]
    empties = [r for r in recs if is_empty(r)]
    # Almost every empty plan scores F=1.0 -- with nothing planned, nothing can
    # be missing and nothing hallucinated. A handful legitimately score lower:
    # no plan resolved, yet the prose named an entity, so the hallucination
    # penalty bites. Those are still empty and v2 still zeroes them. The guard
    # therefore tolerates a few but fails if the fraction is large, which would
    # mean `entities_total` does not mean what this script assumes.
    # Observed range across our 104 metrics files: 0.0% to 18.6%. The gate sits
    # well above that -- it exists to catch a file where `entities_total` means
    # something other than "entities resolved out of the plan", not to police a
    # rate that is legitimately nonzero.
    if empties:
        odd = [r for r in empties if r["F"] != 1.0]
        if len(odd) > 0.40 * len(empties):
            sys.exit(f"{where}: {len(odd)} of {len(empties)} empty plans score "
                     f"F!=1.0 ({100*len(odd)/len(empties):.1f}%). The v1->v2 "
                     f"correction assumes an empty plan is vacuously faithful; "
                     f"check what entities_total means in this file.")
    direct = sum(f_v2(r) for r in recs) / len(recs)
    identity = (statistics.mean(r["F"] for r in ne) * len(ne) / len(recs)) if ne else 0.0
    if abs(direct - identity) > 1e-9:
        sys.exit(f"{where}: F_v2 identity failed ({direct} vs {identity})")


def per_instance(runs: list[list[dict]], fn) -> dict:
    """instance_id -> mean of fn over runs (1 run for baselines, 3 for ours)."""
    acc: dict = {}
    for recs in runs:
        for r in recs:
            acc.setdefault(r["instance_id"], []).append(fn(r))
    return {k: statistics.mean(v) for k, v in acc.items()}


def boot_ci(vals: list[float], seed: int, n_boot: int = N_BOOT):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def summarize(runs: list[list[dict]], seed: int = SEED_SINGLE) -> dict:
    ids = sorted(per_instance(runs, lambda r: 0.0))

    def vec(fn):
        d = per_instance(runs, fn)
        return [d[i] for i in ids]

    U, G, F = vec(lambda r: r["U"]), vec(lambda r: r["G"]), vec(f_v2)
    comp = [0.50 * u + 0.25 * g + 0.25 * f for u, g, f in zip(U, G, F)]
    e2e = vec(lambda r: 1.0 if end_to_end(r) else 0.0)
    out = {
        "n_seeds": len(runs), "n_instances": len(ids),
        "U": statistics.mean(U), "G": statistics.mean(G), "F_v2": statistics.mean(F),
        "F_v1": statistics.mean(vec(lambda r: r["F"])),
        "R_pct": 100 * statistics.mean(vec(lambda r: 1.0 if r.get("format_ok") else 0.0)),
        "empty_pct": 100 * statistics.mean(vec(lambda r: 1.0 if is_empty(r) else 0.0)),
        "composite": 100 * statistics.mean(comp),
        "e2e_pct": 100 * statistics.mean(e2e),
    }
    out["U_lo"], out["U_hi"] = boot_ci(U, seed)
    lo, hi = boot_ci(comp, seed)
    out["composite_lo"], out["composite_hi"] = 100 * lo, 100 * hi
    lo, hi = boot_ci(e2e, seed)
    out["e2e_lo"], out["e2e_hi"] = 100 * lo, 100 * hi
    if len(runs) > 1:
        finals = [100 * statistics.mean(
            0.50 * r["U"] + 0.25 * r["G"] + 0.25 * f_v2(r) for r in recs)
            for recs in runs]
        out["final_spread"] = max(finals) - min(finals)
    return out


def paired_delta(a_runs, b_runs, fn, seed: int = SEED_PAIRED):
    """mean(a - b) over matched instance_ids, with a paired bootstrap."""
    A, B = per_instance(a_runs, fn), per_instance(b_runs, fn)
    ids = sorted(set(A) & set(B))
    if not ids:
        sys.exit("no matched instance_ids between the two runs")
    d = [A[i] - B[i] for i in ids]
    return statistics.mean(d), boot_ci(d, seed), len(ids)


def collect(analysis: pathlib.Path, split: str):
    """(name -> {'base': runs, 'ours': runs}), skipping what is not present."""
    found = {}
    for name, base_dir, prefix in ROWS:
        entry = {}
        bp = analysis / base_dir / f"{base_dir}_{split}.jsonl"
        if bp.exists():
            recs = load(bp)
            check_f_v2_identity(recs, str(bp))
            entry["base"] = [recs]
        ours = []
        for s in SEEDS:
            p = analysis / f"{prefix}_s{s}" / f"{prefix}_s{s}_{split}.jsonl"
            if p.exists():
                recs = load(p)
                check_f_v2_identity(recs, str(p))
                ours.append(recs)
        if ours:
            entry["ours"] = ours
        if entry:
            found[name] = entry
    return found


def seed_stats(runs: list[list[dict]], fn) -> dict:
    """Per-seed values plus their spread, for a metric computed on one run."""
    vals = [statistics.mean([x for x in (fn(r) for r in recs) if x is not None])
            for recs in runs]
    return {"values": vals,
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "range": max(vals) - min(vals) if len(vals) > 1 else 0.0}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down over a family of tests.

    Returns name -> corrected p. Matches the procedure the rest of the paper
    applies to its own families rather than inventing a second convention."""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(ordered), {}, 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[name] = running
    return out


def boot_p(vals: list[float], seed: int, n_boot: int = N_BOOT) -> float:
    """Two-sided bootstrap p for 'the paired mean is zero'.

    The proportion of resampled means falling on the other side of zero,
    doubled. Floored at 1/n_boot: with 5,000 resamples this cannot resolve a
    p below 0.0002, and reporting 0 would overstate what was measured."""
    rng = random.Random(seed)
    n = len(vals)
    means = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    obs = statistics.mean(vals)
    tail = sum(1 for m in means if (m <= 0) if obs > 0) or \
           sum(1 for m in means if (m >= 0) if obs < 0)
    return max(1.0 / n_boot, min(1.0, 2.0 * tail / n_boot))


def paired_values(a_runs, b_runs, fn) -> list[float]:
    A, B = per_instance(a_runs, fn), per_instance(b_runs, fn)
    ids = sorted(set(A) & set(B))
    return [A[i] - B[i] for i in ids]


def write_csv(out_dir: pathlib.Path, name: str, header, rows) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {out_dir / name}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True,
                    help="directory of per-model metric dirs (no default on "
                         "purpose: pointing this at the wrong run is silent)")
    ap.add_argument("--split", default="500", choices=["500", "300"],
                    help="500 = standard, 300 = adversarial")
    ap.add_argument("--out", default=None, help="also write CSVs here")
    ap.add_argument("--per-seed", action="store_true",
                    help="also emit every training seed separately, with the "
                         "across-seed SD and range. The bootstrap intervals "
                         "elsewhere are over instances and say nothing about "
                         "seed-to-seed variability")
    ap.add_argument("--multiplicity", action="store_true",
                    help="Holm-Bonferroni across the eight repair contrasts, "
                         "with dU as the pre-specified primary endpoint")
    ap.add_argument("--ablations", action="store_true",
                    help="corpus-composition contrasts on RoLlama3.1-8B-DPO: "
                         "each single-variable mix against the full one, with "
                         "every seed shown because the differences are small "
                         "enough that a seed mean alone cannot carry them")
    ap.add_argument("--paired-t", action="store_true",
                    help="the paired table on T, the raw constraint-satisfaction "
                         "fraction, for every repaired checkpoint. T is a primary "
                         "outcome and the manuscript reports only the range of "
                         "its improvements, which cannot be inspected")
    args = ap.parse_args()

    analysis = pathlib.Path(args.analysis)
    if not analysis.is_dir():
        sys.exit(f"not a directory: {analysis}")
    data = collect(analysis, args.split)
    if not data:
        sys.exit(f"no metric files found under {analysis} for split {args.split}")

    summ = {(n, k): summarize(v) for n, v in data.items() for k, v in v.items()}

    print("=" * 104)
    print(f"MAIN  --  split {args.split}, F_v2, percentile bootstrap {N_BOOT} "
          f"resamples, seed {SEED_SINGLE}")
    print("=" * 104)
    print("%-18s %-6s %-22s %6s %6s %7s %-22s %7s" %
          ("Model", "stage", "U [95% CI]", "R%", "G", "F_v2", "Composite [95% CI]", "spread"))
    print("-" * 104)
    main_rows = []
    for name in data:
        for stage in ("base", "ours"):
            if (name, stage) not in summ:
                continue
            r = summ[(name, stage)]
            print("%-18s %-6s %.3f [%.3f,%.3f]   %6.1f %6.3f %7.3f  %6.2f [%5.2f,%5.2f] %7s" %
                  (name if stage == "base" else "", stage, r["U"], r["U_lo"], r["U_hi"],
                   r["R_pct"], r["G"], r["F_v2"], r["composite"],
                   r["composite_lo"], r["composite_hi"],
                   ("%.2f" % r["final_spread"]) if "final_spread" in r else "-"))
            main_rows.append([name, stage, r["n_seeds"], round(r["U"], 4),
                              round(r["U_lo"], 4), round(r["U_hi"], 4),
                              round(r["R_pct"], 2), round(r["G"], 4),
                              round(r["F_v1"], 4), round(r["F_v2"], 4),
                              round(r["empty_pct"], 2), round(r["composite"], 3),
                              round(r["composite_lo"], 3), round(r["composite_hi"], 3),
                              round(r["e2e_pct"], 2), round(r["e2e_lo"], 2),
                              round(r["e2e_hi"], 2),
                              round(r.get("final_spread", float("nan")), 3)])
        print()

    print("=" * 104)
    print("END-TO-END  --  parses AND every constraint satisfied AND F=1 on a non-empty plan")
    print("=" * 104)
    print("%-18s %-6s %8s %8s %8s  %-22s" %
          ("Model", "stage", "empty%", "F_v1", "F_v2", "end-to-end % [95% CI]"))
    print("-" * 104)
    for name in data:
        for stage in ("base", "ours"):
            if (name, stage) not in summ:
                continue
            r = summ[(name, stage)]
            print("%-18s %-6s %8.1f %8.3f %8.3f   %5.1f [%4.1f,%4.1f]" %
                  (name if stage == "base" else "", stage, r["empty_pct"],
                   r["F_v1"], r["F_v2"], r["e2e_pct"], r["e2e_lo"], r["e2e_hi"]))
        print()

    print("=" * 104)
    print(f"PAIRED DELTAS (ours - base), paired bootstrap over matched "
          f"instances, seed {SEED_PAIRED}")
    print("=" * 104)
    print("%-18s %-26s %-26s %-24s %5s" %
          ("Model", "dU [95% CI]", "dF_v2 [95% CI]", "d end-to-end pts", "n"))
    print("-" * 104)
    delta_rows = []
    for name, v in data.items():
        if "base" not in v or "ours" not in v:
            continue
        du, (ul, uh), n = paired_delta(v["ours"], v["base"], lambda r: r["U"])
        df, (fl, fh), _ = paired_delta(v["ours"], v["base"], f_v2)
        de, (el, eh), _ = paired_delta(
            v["ours"], v["base"], lambda r: 1.0 if end_to_end(r) else 0.0)
        print("%-18s %+.3f [%+.3f,%+.3f]     %+.3f [%+.3f,%+.3f]     %+5.1f [%+5.1f,%+5.1f] %5d" %
              (name, du, ul, uh, df, fl, fh, 100 * de, 100 * el, 100 * eh, n))
        delta_rows.append([name, n, round(du, 4), round(ul, 4), round(uh, 4),
                           round(df, 4), round(fl, 4), round(fh, 4),
                           round(100 * de, 2), round(100 * el, 2), round(100 * eh, 2)])

    if args.out:
        out = pathlib.Path(args.out)
        write_csv(out, f"40_repair_main_{args.split}.csv",
                  ["model", "stage", "n_seeds", "U", "U_lo", "U_hi", "R_pct", "G",
                   "F_v1", "F_v2", "empty_pct", "composite", "composite_lo",
                   "composite_hi", "e2e_pct", "e2e_lo", "e2e_hi", "final_spread"],
                  main_rows)
        write_csv(out, f"41_repair_deltas_{args.split}.csv",
                  ["model", "n_matched", "dU", "dU_lo", "dU_hi", "dF_v2",
                   "dF_v2_lo", "dF_v2_hi", "d_e2e_pts", "d_e2e_lo", "d_e2e_hi"],
                  delta_rows)

    if args.per_seed:
        print("=" * 104)
        print("PER SEED  --  instance bootstrap says nothing about seed variance; "
              "this is the other source")
        print("=" * 104)
        print("%-18s %-6s %8s %8s %8s   %8s %8s" %
              ("Model", "metric", "seed 0", "seed 1", "seed 2", "SD", "range"))
        print("-" * 104)
        seed_rows = []
        for name, v in data.items():
            if "ours" not in v or len(v["ours"]) < 2:
                continue
            # T first: the review asks for per-seed T, valid output and E2E by
            # name, and T is the raw constraint fraction the semantic claim
            # rests on. U is derived from it and was the only one reported.
            for label, fn in (("T", t_raw),
                              ("U", lambda r: r["U"]),
                              ("F_v2", f_v2),
                              ("G", lambda r: r["G"]),
                              ("valid%", lambda r: 100.0 if r.get("format_ok") else 0.0),
                              ("e2e%", lambda r: 100.0 if end_to_end(r) else 0.0)):
                s = seed_stats(v["ours"], fn)
                vals = s["values"] + [None] * (3 - len(s["values"]))
                print("%-18s %-6s %8s %8s %8s   %8.4f %8.4f" %
                      (name if label == "U" else "", label,
                       *["%8.4f" % x if x is not None else "-" for x in vals[:3]],
                       s["sd"], s["range"]))
                seed_rows.append([name, label] +
                                 [round(x, 5) if x is not None else "" for x in vals[:3]] +
                                 [round(s["mean"], 5), round(s["sd"], 5), round(s["range"], 5)])
            print()
        if args.out:
            write_csv(pathlib.Path(args.out), f"43_repair_per_seed_{args.split}.csv",
                      ["model", "metric", "seed0", "seed1", "seed2",
                       "mean", "sd", "range"], seed_rows)

    if args.multiplicity:
        fams = {"dU": lambda r: r["U"], "dF_v2": f_v2,
                "d_e2e": lambda r: 1.0 if end_to_end(r) else 0.0}
        raw, mrows = {}, []
        for metric, fn in fams.items():
            ps = {}
            for name, v in data.items():
                if "base" not in v or "ours" not in v:
                    continue
                d = paired_values(v["ours"], v["base"], fn)
                ps[name] = boot_p(d, SEED_PAIRED)
            raw[metric] = (ps, holm(ps))
        print("=" * 104)
        print("MULTIPLICITY  --  Holm-Bonferroni within each family of eight; "
              "dU is the primary endpoint")
        print("=" * 104)
        print("%-18s %-24s %-24s %s" % ("Model", "dU  p (Holm)",
                                        "dF_v2  p (Holm)", "d_e2e  p (Holm)"))
        print("-" * 104)
        for name in data:
            if "base" not in data[name] or "ours" not in data[name]:
                continue
            cells = []
            for metric in ("dU", "dF_v2", "d_e2e"):
                ps, hp = raw[metric]
                sig = "" if hp[name] < 0.05 else "  n.s."
                cells.append("%.4f (%.4f)%s" % (ps[name], hp[name], sig))
                mrows.append([name, metric, round(ps[name], 5), round(hp[name], 5),
                              int(hp[name] < 0.05)])
            print("%-18s %-24s %-24s %s" % (name, *cells))
        print()
        for metric in ("dU", "dF_v2", "d_e2e"):
            ps, hp = raw[metric]
            n_sig = sum(1 for k in hp if hp[k] < 0.05)
            print("  %-7s %d of %d survive Holm at alpha=0.05" % (metric, n_sig, len(hp)))
        if args.out:
            write_csv(pathlib.Path(args.out), f"44_repair_multiplicity_{args.split}.csv",
                      ["model", "metric", "p_raw", "p_holm", "survives"], mrows)

    if args.paired_t:
        # T, not U. U folds the format checks in through its 0.15*L^3 term, so a
        # U gain can come from constraints or from formatting; T is the constraint
        # fraction alone. The manuscript states only that T improved by +0.089 to
        # +0.373 and never prints the per-checkpoint numbers, so the claim it is
        # making about constraint following cannot be checked. This prints them.
        print("=" * 104)
        print(f"PAIRED T  --  raw constraint-satisfaction fraction, split {args.split}, "
              f"paired bootstrap {N_BOOT} resamples, seed {SEED_PAIRED}")
        print("=" * 104)
        print("%-20s %7s %7s %8s %-20s %5s %-26s %7s" %
              ("Model", "base", "ours", "delta", "95% CI", "n", "per seed", "range"))
        print("-" * 104)
        t_rows = []
        for name, v in data.items():
            if "base" not in v or "ours" not in v:
                continue
            b = statistics.mean(x for x in (t_raw(r) for r in v["base"][0])
                                if x is not None)
            per = [statistics.mean(x for x in (t_raw(r) for r in run)
                                   if x is not None) for run in v["ours"]]
            m = statistics.mean(per)
            d, (lo, hi), n = paired_delta(v["ours"], v["base"], t_raw)
            rng = max(per) - min(per) if len(per) > 1 else 0.0
            print("%-20s %7.3f %7.3f %+8.3f  [%+.3f,%+.3f]   %5d %-26s %7.3f" %
                  (name, b, m, d, lo, hi, n,
                   "|".join("s%d=%.3f" % (i, x) for i, x in enumerate(per)), rng))
            t_rows.append([name, args.split, len(per), round(b, 4), round(m, 4),
                           round(d, 4), round(lo, 4), round(hi, 4), n,
                           "|".join("s%d=%.4f" % (i, x) for i, x in enumerate(per)),
                           round(rng, 4)])
        pos = [r for r in t_rows if r[5] > 0]
        excl = [r for r in t_rows if r[6] > 0 or r[7] < 0]
        print(f"\n  improves on {len(pos)} of {len(t_rows)}; "
              f"{len(excl)} of {len(t_rows)} intervals exclude zero")
        print(f"  range of deltas {min(r[5] for r in t_rows):+.3f} to "
              f"{max(r[5] for r in t_rows):+.3f}  (the manuscript quotes only this)")
        if args.out:
            write_csv(pathlib.Path(args.out), f"59_repair_paired_t_{args.split}.csv",
                      ["model", "split", "n_seeds", "base_T", "ours_T", "delta",
                       "ci_lo", "ci_hi", "n_matched", "per_seed", "seed_range"],
                      t_rows)
        return

    if args.ablations:
        abl = {}
        for label, prefix in ABLATIONS:
            runs = []
            for s in SEEDS:
                q = analysis / f"{prefix}_s{s}" / f"{prefix}_s{s}_{args.split}.jsonl"
                if q.exists():
                    recs = load(q)
                    check_f_v2_identity(recs, str(q))
                    runs.append(recs)
            if runs:
                abl[label] = runs
        if not abl:
            sys.exit(f"no ablation runs found under {analysis}")
        if "full mix" not in abl:
            sys.exit("the full-mix run is the reference; nothing to compare against")

        METRICS = (("valid%", lambda r: 100.0 if r.get("format_ok") else 0.0),
                   ("U", lambda r: r["U"]),
                   ("T", t_raw),
                   ("G", lambda r: r["G"]),
                   ("F_v2", f_v2),
                   ("e2e%", lambda r: 100.0 if end_to_end(r) else 0.0))

        print("=" * 104)
        print(f"CORPUS ABLATIONS  --  RoLlama3.1-8B-DPO, split {args.split}, "
              f"every seed shown")
        print("=" * 104)
        print("%-12s %-7s %9s %9s %9s   %9s %8s %9s" %
              ("mix", "metric", "seed 0", "seed 1", "seed 2",
               "mean", "range", "vs full"))
        print("-" * 104)
        abl_rows = []
        ref = {label: seed_stats(abl["full mix"], fn)["mean"]
               for label, fn in METRICS}
        for label in abl:
            for mname, fn in METRICS:
                s = seed_stats(abl[label], fn)
                vals = s["values"] + [None] * (3 - len(s["values"]))
                delta = s["mean"] - ref[mname]
                # Non-overlapping per-seed ranges, not a difference of means:
                # with three seeds a mean gap smaller than the spread is not a
                # result, and this is the cheap check that says which is which.
                sep = ""
                if label != "full mix" and len(s["values"]) > 1 \
                        and len(abl["full mix"]) > 1:
                    f = seed_stats(abl["full mix"], fn)["values"]
                    if min(s["values"]) > max(f) or max(s["values"]) < min(f):
                        sep = " *"
                print("%-12s %-7s %9s %9s %9s   %9.4f %8.4f %+9.4f%s" %
                      (label if mname == "valid%" else "", mname,
                       *["%9.4f" % x if x is not None else "-" for x in vals[:3]],
                       s["mean"], s["range"], delta, sep))
                abl_rows.append([label, mname] +
                                [round(x, 5) if x is not None else ""
                                 for x in vals[:3]] +
                                [round(s["mean"], 5), round(s["sd"], 5),
                                 round(s["range"], 5), round(delta, 5),
                                 int(bool(sep))])
            print()
        print("  * per-seed ranges do not overlap the full mix's -- the only "
              "differences here that clear seed noise")
        if args.out:
            write_csv(pathlib.Path(args.out), f"46_corpus_ablations_{args.split}.csv",
                      ["mix", "metric", "seed0", "seed1", "seed2", "mean", "sd",
                       "range", "delta_vs_full", "seed_separated"], abl_rows)


if __name__ == "__main__":
    main()
