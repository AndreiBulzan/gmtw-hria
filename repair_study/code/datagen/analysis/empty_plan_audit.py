"""Audit the empty-plan rule by task family, and score the fact family correctly.

THE DEFECT. 3.9 adopted a corrected faithfulness scorer, F_v2, under which a plan
with no entities scores 0 instead of 1. That was right for the failure it was
written for: checkpoints that stop emitting plans were being awarded a perfect
faithfulness score for producing nothing.

It is wrong for one of the four task families. A `fact` world asks for
{"answer": "<string>"}; the answer is a value, not an entity ID, so the scorer
resolves no planned entity and records entities_total = 0 -- the same signature as
a missing plan. The benchmark holds exactly 100 fact items of 500 on the standard
split and 50 of 300 on the adversarial one, so the rule zeroes 20% and 17% of each
split regardless of whether those answers are right.

Measured: on the repaired RoLlama3.1 checkpoint the fact family is 100/100 empty
while travel, recipe and schedule sit at 0%, 0% and 1.3%. Those items are not
failing. They parse at 100%, satisfy their constraints, and average U = 0.909
against travel's 0.782 -- the strongest items in the split, scored zero.

THE v3 CONVENTION, and the reasoning for it. The released scorer gives a fact item
F = 1.0 because both faithfulness terms are vacuously satisfied; v2 gives it 0.
Both are wrong, and in opposite directions. F is UNDEFINED when there is nothing
to be faithful about: awarding 1.0 reinstates on 20% of the split exactly the
vacuous perfect score v2 existed to remove, and awarding 0 penalises a correct
answer. So:

  * F_v3 is defined only on the entity-plan families (travel, schedule, recipe).
    Within them an empty plan scores 0, as in v2.
  * Fact items carry no F. Their per-instance composite renormalises the
    surviving weights: (0.50 U + 0.25 G) / 0.75, rather than feeding a
    meaningless F into the 0.25 slot.
  * End-to-end success needs a faithfully explained plan, so on fact items it
    means parse plus every constraint satisfied.

This convention is Andrei's, from review3_empty_plan_audit.py; this module is the
kd-hria-side implementation of it so that one reviewable copy exists in version
control. The end-to-end rule was derived independently on both sides and agrees
to machine precision on all eight checkpoints; the faithfulness rule is his, and
it is the correct one -- an earlier version of this file gave fact items the
released F = 1.0 and was wrong for the reason above.

WHAT THE CORRECTION MOVES, AND WHAT IT DOES NOT. The levels shift a long way:
end-to-end by ~16 points. The DELTAS barely move, because fact items are empty
under the old rule in the base checkpoints too and cancel in the subtraction --
so the mitigation claim is unaffected and end-to-end deltas get slightly larger.
Both are printed so the distinction is visible rather than asserted.

DO NOT RESCALE BY n/n_entity_plan INSTEAD. That shortcut assumes fact items would
have succeeded at the rate of the rest. They succeed MORE often, so it is wrong by
2 to 14 points depending on the checkpoint -- unevenly enough to reorder them.
This recomputes per record.

Reads only the per-instance metric files already in results/: no GPU, no model,
no regeneration. A full pass over every checkpoint, seed and split is ~0.1 s.

  python -m datagen.analysis.empty_plan_audit --analysis datagen/results
  python -m datagen.analysis.empty_plan_audit --analysis datagen/results \
      --split 300 --out datagen/results/tables
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

from datagen.analysis.repair_tables import (ROWS, SEED_PAIRED, all_constraints,
                                            boot_ci, boot_p, f_v2, holm, is_empty)

# Families answered with an entity plan. Anything outside this set answers with a
# value, so faithfulness between an explanation and a plan is not defined on it.
ENTITY_FAMILIES = ("travel", "schedule", "recipe")
SEEDS = (0, 1, 2)


def family(rec: dict) -> str:
    """Task family from the instance id.

    The standard split names instances `travel_000000`, the adversarial split
    `fact_hard_000000`; the first segment gives the family on both.
    """
    return rec["instance_id"].split("_")[0]


def is_fact(rec: dict) -> bool:
    return family(rec) not in ENTITY_FAMILIES


def parses(rec: dict) -> bool:
    return bool(rec.get("format_ok"))


def f_v3(rec: dict) -> float | None:
    """None on fact items (F undefined there); otherwise the v2 rule."""
    return None if is_fact(rec) else f_v2(rec)


def comp_v1(rec: dict) -> float:
    return 0.50 * float(rec["U"]) + 0.25 * float(rec["G"]) + 0.25 * float(rec["F"])


def comp_v2(rec: dict) -> float:
    return 0.50 * float(rec["U"]) + 0.25 * float(rec["G"]) + 0.25 * f_v2(rec)


def comp_v3(rec: dict) -> float:
    u, g = float(rec["U"]), float(rec["G"])
    f = f_v3(rec)
    if f is None:
        return (0.50 * u + 0.25 * g) / 0.75
    return 0.50 * u + 0.25 * g + 0.25 * f


def e2e_v2(rec: dict) -> bool:
    """The criterion as the manuscript reports it: no fact item can satisfy it."""
    return (parses(rec) and all_constraints(rec)
            and not is_empty(rec) and float(rec["F"]) >= 0.999)


def e2e_v3(rec: dict) -> bool:
    if is_fact(rec):
        return parses(rec) and all_constraints(rec)
    return e2e_v2(rec)


def load(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def mean_f_v3(recs: list[dict]) -> float:
    vals = [x for x in (f_v3(r) for r in recs) if x is not None]
    return statistics.mean(vals)


def breakdown(recs: list[dict]) -> dict[str, int]:
    """Why each empty-plan record is empty -- the distinction the review asks for.

    fact  a valid domain answer with no entity plan, by design
    miss  an entity family that produced no parseable structured output
    void  an entity family that parsed but whose plan listed nothing
    """
    empty = [r for r in recs if is_empty(r)]
    fact = [r for r in empty if is_fact(r)]
    ent = [r for r in empty if not is_fact(r)]
    miss = sum(1 for r in ent if not parses(r))
    return {"empty": len(empty), "fact": len(fact),
            "fact_parses": sum(1 for r in fact if parses(r)),
            "fact_correct": sum(1 for r in fact if parses(r) and all_constraints(r)),
            "miss": miss, "void": len(ent) - miss}



def paired_v3(base: list[dict], per: list[list[dict]], fn, planful_only: bool):
    """Per-instance paired differences (ours - base), seeds averaged.

    planful_only drops the fact family. F_v3 is UNDEFINED there, so a paired
    difference on F_v3 exists only on the instances that call for a plan; using
    all 500 would silently reinstate a value v3 declines to assign. E2E_v3 is
    defined everywhere, so it keeps the full split. The two contrasts therefore
    have different denominators on purpose, and the table prints both.
    """
    def acc(runs):
        out: dict = {}
        for recs in runs:
            for r in recs:
                if planful_only and is_fact(r):
                    continue
                v = fn(r)
                if v is None:
                    continue
                out.setdefault(r["instance_id"], []).append(float(v))
        return {k: statistics.mean(v) for k, v in out.items()}

    A, B = acc(per), acc([base])
    ids = sorted(set(A) & set(B))
    return [A[i] - B[i] for i in ids]


def write_csv(out: pathlib.Path, name: str, header, rows) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {len(rows):4d} rows -> {out / name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True, type=pathlib.Path)
    ap.add_argument("--split", choices=["500", "300"], default="500")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    RES, split = args.analysis, args.split

    def seeds_of(pre):
        return [p for s in SEEDS
                if (p := RES / f"{pre}_s{s}" / f"{pre}_s{s}_{split}.jsonl").exists()]

    # ---------------------------------------------------------------- A
    print("=" * 78)
    print(f"A. EMPTY-PLAN RECORDS BY CAUSE  (split {split})")
    print("=" * 78)
    print("\n  fact = valid domain answer with no entity plan, by design")
    print("  miss = entity family, no parseable structured output")
    print("  void = entity family, parsed but plan listed nothing\n")
    print(f"  {'checkpoint':<24}{'arm':>5}{'empty':>7}{'fact':>6}{'f.ok':>6}"
          f"{'f.corr':>8}{'miss':>6}{'void':>6}")
    a_rows = []
    for disp, bdir, pre in ROWS:
        for arm, paths in (("base", [RES / bdir / f"{bdir}_{split}.jsonl"]),
                           ("ours", seeds_of(pre))):
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            b = breakdown(load(paths[0]))
            a_rows.append([disp, arm, split] + [b[k] for k in
                          ("empty", "fact", "fact_parses", "fact_correct", "miss", "void")])
            print(f"  {disp:<24}{arm:>5}{b['empty']:>7}{b['fact']:>6}"
                  f"{b['fact_parses']:>6}{b['fact_correct']:>8}{b['miss']:>6}{b['void']:>6}")
    print("\n  Every fact item is recorded as an empty plan on every checkpoint,")
    print("  including the ones that parse and answer correctly (f.corr).")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78)
    print(f"B. FAITHFULNESS AND COMPOSITE UNDER v1 / v2 / v3  (split {split})")
    print("=" * 78)
    print(f"\n  {'checkpoint':<24}{'arm':>5}{'F_v1':>7}{'F_v2':>7}{'F_v3':>7}"
          f"{'C_v1':>7}{'C_v2':>7}{'C_v3':>7}")
    b_rows = []
    for disp, bdir, pre in ROWS:
        for arm, paths in (("base", [RES / bdir / f"{bdir}_{split}.jsonl"]),
                           ("ours", seeds_of(pre))):
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            per = [load(p) for p in paths]
            f1 = statistics.mean(statistics.mean(float(r["F"]) for r in g) for g in per)
            f2 = statistics.mean(statistics.mean(f_v2(r) for r in g) for g in per)
            f3 = statistics.mean(mean_f_v3(g) for g in per)
            c1 = statistics.mean(statistics.mean(map(comp_v1, g)) for g in per)
            c2 = statistics.mean(statistics.mean(map(comp_v2, g)) for g in per)
            c3 = statistics.mean(statistics.mean(map(comp_v3, g)) for g in per)
            b_rows.append([disp, arm, split, len(per), round(f1, 4), round(f2, 4),
                           round(f3, 4), round(c1, 4), round(c2, 4), round(c3, 4)])
            print(f"  {disp:<24}{arm:>5}{f1:7.3f}{f2:7.3f}{f3:7.3f}"
                  f"{c1:7.3f}{c2:7.3f}{c3:7.3f}")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78)
    print(f"C. END-TO-END SUCCESS, v2 AS REPORTED vs v3  (split {split})")
    print("=" * 78)
    print(f"\n  no fact item can satisfy v2, so the v2 ceiling is "
          f"{100*len(ENTITY_FAMILIES)/4:.0f}% of the families, not 100%\n")
    print(f"  {'checkpoint':<24}{'arm':>5}{'E2E v2':>8}{'E2E v3':>8}{'shift':>8}")
    c_rows = []
    for disp, bdir, pre in ROWS:
        for arm, paths in (("base", [RES / bdir / f"{bdir}_{split}.jsonl"]),
                           ("ours", seeds_of(pre))):
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            per = [load(p) for p in paths]
            v2 = statistics.mean(100 * sum(map(e2e_v2, g)) / len(g) for g in per)
            v3 = statistics.mean(100 * sum(map(e2e_v3, g)) / len(g) for g in per)
            c_rows.append([disp, arm, split, len(per), round(v2, 2), round(v3, 2)])
            print(f"  {disp:<24}{arm:>5}{v2:8.1f}{v3:8.1f}{v3-v2:+8.1f}")

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 78)
    print(f"D. DELTAS -- WHAT THE MITIGATION CLAIM RESTS ON  (split {split})")
    print("=" * 78)
    print(f"\n  {'checkpoint':<24}{'dF_v2':>8}{'dF_v3':>8}{'dE2E v2':>9}{'dE2E v3':>9}")
    d_rows, dd = [], []
    for disp, bdir, pre in ROWS:
        bp = RES / bdir / f"{bdir}_{split}.jsonl"
        rps = seeds_of(pre)
        if not (bp.exists() and rps):
            continue
        B, per = load(bp), [load(p) for p in rps]
        dF2 = statistics.mean(statistics.mean(f_v2(r) for r in g) for g in per) \
            - statistics.mean(f_v2(r) for r in B)
        dF3 = statistics.mean(mean_f_v3(g) for g in per) - mean_f_v3(B)
        dE2 = statistics.mean(100*sum(map(e2e_v2, g))/len(g) for g in per) \
            - 100*sum(map(e2e_v2, B))/len(B)
        dE3 = statistics.mean(100*sum(map(e2e_v3, g))/len(g) for g in per) \
            - 100*sum(map(e2e_v3, B))/len(B)
        dd.append((dF2, dF3, dE2, dE3))
        d_rows.append([disp, split, len(per), round(dF2, 4), round(dF3, 4),
                       round(dE2, 2), round(dE3, 2)])
        print(f"  {disp:<24}{dF2:+8.3f}{dF3:+8.3f}{dE2:+9.1f}{dE3:+9.1f}")
    print(f"\n  dE2E as reported {min(x[2] for x in dd):+.1f}..{max(x[2] for x in dd):+.1f}"
          f"   under v3 {min(x[3] for x in dd):+.1f}..{max(x[3] for x in dd):+.1f}")
    print(f"  every v3 end-to-end delta still positive: {all(x[3] > 0 for x in dd)}")

    # ---------------------------------------------------------------- E
    print("\n" + "=" * 78)
    print(f"E. PAIRED BOOTSTRAP AND HOLM UNDER v3  (split {split})")
    print("=" * 78)
    print(f"\n  percentile bootstrap, {5000} resamples, seed {SEED_PAIRED}; "
          f"Holm over the eight")
    print("  repair contrasts within each metric, as the paper does for dU.")
    print("  dF_v3 is over planful instances only -- F_v3 is undefined on fact.\n")

    e_rows = []
    for metric, fn, planful, scale, label in (
            ("F_v3", f_v3, True, 1.0, "dF_v3"),
            ("E2E_v3", lambda r: 1.0 if e2e_v3(r) else 0.0, False, 100.0, "dE2E_v3 (pts)")):
        stats, raw_p = {}, {}
        for disp, bdir, pre in ROWS:
            bp = RES / bdir / f"{bdir}_{split}.jsonl"
            rps = seeds_of(pre)
            if not (bp.exists() and rps):
                continue
            vals = paired_v3(load(bp), [load(x) for x in rps], fn, planful)
            lo, hi = boot_ci(vals, SEED_PAIRED)
            stats[disp] = (scale * statistics.mean(vals), scale * lo, scale * hi,
                           len(vals))
            raw_p[disp] = boot_p(vals, SEED_PAIRED)
        adj = holm(raw_p)
        print(f"  {label}")
        print(f"  {'checkpoint':<24}{'n':>5}{'delta':>9}{'95% CI':>22}"
              f"{'raw p':>9}{'Holm p':>9}{'':>4}")
        n_sig = 0
        for disp, (d, lo, hi, n) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            excl = (lo > 0) or (hi < 0)
            sig = adj[disp] < 0.05 and excl
            n_sig += sig
            fmt = "6.3f" if scale == 1.0 else "6.1f"
            print(f"  {disp:<24}{n:>5}{d:>+9.{'3' if scale == 1.0 else '1'}f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]' if scale == 1.0 else f'[{lo:+.1f}, {hi:+.1f}]':>22}"
                  f"{raw_p[disp]:>9.4f}{adj[disp]:>9.4f}{'  *' if sig else '':>4}")
            e_rows.append([disp, split, metric, n, round(d, 4), round(lo, 4),
                           round(hi, 4), round(raw_p[disp], 4), round(adj[disp], 4),
                           int(sig)])
        print(f"  -> {n_sig}/{len(stats)} survive Holm with an interval excluding zero\n")

    if args.out:
        print("\nCSVs:")
        write_csv(args.out, f"55_empty_plan_by_cause_{split}.csv",
                  ["model", "arm", "split", "empty", "fact", "fact_parses",
                   "fact_correct", "miss", "void"], a_rows)
        write_csv(args.out, f"56_faith_composite_v123_{split}.csv",
                  ["model", "arm", "split", "n_seeds", "F_v1", "F_v2", "F_v3",
                   "comp_v1", "comp_v2", "comp_v3"], b_rows)
        write_csv(args.out, f"57_end_to_end_v2_v3_{split}.csv",
                  ["model", "arm", "split", "n_seeds", "e2e_v2", "e2e_v3"], c_rows)
        write_csv(args.out, f"58_repair_deltas_v3_{split}.csv",
                  ["model", "split", "n_seeds", "dF_v2", "dF_v3",
                   "dE2E_v2", "dE2E_v3"], d_rows)
        write_csv(args.out, f"64_repair_v3_holm_{split}.csv",
                  ["model", "split", "metric", "n", "delta", "ci_lo", "ci_hi",
                   "raw_p", "holm_p", "significant"], e_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
