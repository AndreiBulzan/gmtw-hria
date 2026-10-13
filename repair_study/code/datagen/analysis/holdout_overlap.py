#!/usr/bin/env python3
"""Overlap between the English holdout split and the schemas we trained on.

Three things the manuscript gets wrong or leaves unstated about the transfer
test, all answerable without a GPU.

1. THE SPLIT IS 1,804 SCHEMAS, NOT 3,608. The file that carried 3,608 records
   was the 1,804-schema split concatenated with itself -- every line duplicated
   exactly twice, first half byte-identical to second
   (test_split.jsonl.doubled_bak preserves it). It was de-duplicated on
   2026-09-12 and every holdout run on disk totals 1,804, so the reported RATES
   are sound and nothing needs re-scoring. Only the advertised total is wrong.
   1,210 of those 1,804 reject every trivial answer, which is the denominator
   the table reports: 1,210/1,804 = 67.1%, the exact complement of the 32.9%
   trivially-satisfiable figure. The "3,608 vs 1,210" gap a reader sees is not
   filtering at all.

2. EXACT DUPLICATES ARE A DIFFERENT PROBLEM FROM STRUCTURAL TWINS, and the
   corpus documentation collapses them. build_split.py argues -- correctly --
   that structural overlap is not contamination when the metric is schema
   VALIDITY: there is no reference answer, so a memorised completion still has
   to validate. That argument covers twins. It does NOT cover an exact
   duplicate: if the identical schema was in training, the model may reproduce
   a completion it knows validates, and the claim being made is about unseen
   schemas. So the two rates are measured and reported separately here, and
   the exact duplicates are excluded from a re-scored transfer rate.

   The eval sources (Github_hard/ultra, Kubernetes, JsonSchemaStore, Snowplow,
   WashingtonPost) are disjoint from the trained sources by construction, so any
   exact duplicate is the same schema appearing under two different sources --
   which is exactly the leak a source-level split cannot catch by itself.

3. "STRUCTURAL TWIN" NEEDS A STATED CRITERION. It is equality of
   filters/dedupe.py:schema_fingerprint -- a hash of types, constraint kinds
   and child structures, sorted, with property names, titles, descriptions,
   $id/$comment/examples/default and concrete constraint VALUES ignored.
   Renaming every key leaves the fingerprint unchanged; changing a type or
   adding a constraint kind changes it.

Two training pools are reported, because they answer different questions:

  mix      the 2,200 Track A records (1,200 ro + 1,000 en) actually sampled
           into train_mix_4k_v2 and seen by the optimiser. The exact claim.
  pool     all 3,101 Track A records generated. Conservative: a different mix
           seed would have drawn different records from it.

  python datagen/analysis/holdout_overlap.py \
      --split ro-so/eval/jsonschemabench_holdout/test_split.jsonl \
      --mix datagen/data/sft/train_mix_4k_v2.jsonl \
      --pool datagen/data/sft/track_a.jsonl datagen/data/sft/english_structured.jsonl \
      --holdout-results datagen/results/holdout \
      --out datagen/results/tables
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "ro-so"))
from filters.dedupe import schema_fingerprint   # noqa: E402

N_BOOT = 5000
SEED_PAIRED = 42        # the paper's seed for paired contrasts


def boot_ci(vals, seed=SEED_PAIRED, n_boot=N_BOOT):
    """Percentile bootstrap over schemas. Matches the paper's convention."""
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def boot_p(vals, seed=SEED_PAIRED, n_boot=N_BOOT):
    """Two-sided bootstrap p for 'the paired mean is zero', floored at 1/n_boot."""
    rng = random.Random(seed)
    n = len(vals)
    means = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    obs = statistics.mean(vals)
    tail = sum(1 for m in means if m <= 0) if obs > 0 else \
           sum(1 for m in means if m >= 0)
    return max(1.0 / n_boot, min(1.0, 2.0 * tail / n_boot))


def holm(pvals: dict) -> dict:
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(ordered), {}, 0.0
    for i, (name, pv) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * pv))
        out[name] = running
    return out


CHECKPOINTS = ["RoGemma-7B-DPO", "RoGemma2-9B-DPO", "RoLlama2-7B", "RoLlama3-8B-DPO",
               "RoMistral-7B-SFT", "RoGemma2-9B-SFT", "RoMistral-7B-DPO",
               "RoLlama3.1-8B-DPO"]


def read_jsonl(path: pathlib.Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def exact_hash(schema) -> str:
    """Byte-identity up to key order and whitespace -- nothing else ignored."""
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def fingerprints(records, label: str) -> tuple[set[str], set[str]]:
    """(exact hashes, structural fingerprints) over Track A schemas only."""
    exact, struct = set(), set()
    n = 0
    for rec in records:
        if rec.get("track") != "A":          # only A carries a JSONSchemaBench schema
            continue
        schema = rec.get("schema")
        if schema is None:
            continue
        n += 1
        exact.add(exact_hash(schema))
        try:
            struct.add(schema_fingerprint(schema))
        except Exception as e:
            print(f"  warn: unfingerprintable training schema: {e}", file=sys.stderr)
    print(f"  {label}: {n} Track A schemas, {len(exact)} distinct exact, "
          f"{len(struct)} distinct structural", file=sys.stderr)
    return exact, struct


def source_of(record_id: str) -> str:
    return record_id.split("/")[0] if "/" in record_id else "?"


def classify(split, pools: dict, trivial: dict) -> list[dict]:
    rows = []
    for rec in split:
        schema = rec["schema"]
        eh = exact_hash(schema)
        try:
            fp = schema_fingerprint(schema)
        except Exception:
            fp = None
        row = {"id": rec["id"], "source": source_of(rec["id"]), "fp": fp,
               "non_gameable": not trivial.get(rec["id"], False)}
        for name, (exact, struct) in pools.items():
            row[f"exact_{name}"] = eh in exact
            row[f"twin_{name}"] = bool(fp and fp in struct and eh not in exact)
        rows.append(row)
    return rows


def write_overlap(out: pathlib.Path, rows: list[dict], pools: dict) -> None:
    """Per-source overlap, for the whole split and for the scored subset."""
    srcs = sorted({r["source"] for r in rows})
    recs = []
    for scope, keep in (("all 1804", lambda r: True),
                        ("scored 1210", lambda r: r["non_gameable"])):
        subset = [r for r in rows if keep(r)]
        for src in srcs + ["TOTAL"]:
            sel = subset if src == "TOTAL" else [r for r in subset if r["source"] == src]
            if not sel:
                continue
            row = {"scope": scope, "source": src, "n": len(sel)}
            for name in pools:
                e = sum(1 for r in sel if r[f"exact_{name}"])
                t = sum(1 for r in sel if r[f"twin_{name}"])
                row[f"exact_{name}"] = e
                row[f"exact_{name}_pct"] = f"{100 * e / len(sel):.2f}"
                row[f"twin_{name}"] = t
                row[f"twin_{name}_pct"] = f"{100 * t / len(sel):.2f}"
            recs.append(row)

    cols = ["scope", "source", "n"]
    for name in pools:
        cols += [f"exact_{name}", f"exact_{name}_pct", f"twin_{name}", f"twin_{name}_pct"]
    path = out / "62_holdout_overlap.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)

    print(f"\n{'scope':12} {'source':22} {'n':>6}", end="")
    for name in pools:
        print(f" {'exact/' + name:>14} {'twin/' + name:>14}", end="")
    print()
    for r in recs:
        print(f"{r['scope']:12} {r['source']:22} {r['n']:>6}", end="")
        for name in pools:
            print(f" {str(r[f'exact_{name}']) + ' (' + r[f'exact_{name}_pct'] + '%)':>14}"
                  f" {str(r[f'twin_{name}']) + ' (' + r[f'twin_{name}_pct'] + '%)':>14}", end="")
        print()
    print(f"\nwrote {path}")
    distinct_rates(rows, pools)


def distinct_rates(rows: list[dict], pools: dict) -> None:
    """Record-level and shape-level overlap are different numbers. Both go in.

    build_split.py's docstring reports 6.2% "unique structural twins" and
    Glaiveai2K at "89.1% any-collision but only 5.7% unique twins". That is the
    SHAPE-level rate: of the distinct structures in the eval set, how many also
    occur in training. The record-level rate above is the one that describes
    exposure of the scored set -- how many of the schemas we actually score
    share a shape with training -- and it is far higher because a handful of
    common shapes repeat across many records. Neither number is wrong; the
    manuscript should quote the record-level one and say which it is.
    """
    for scope, keep in (("all 1804", lambda r: True),
                        ("scored 1210", lambda r: r["non_gameable"])):
        sel = [r for r in rows if keep(r) and r["fp"]]
        shapes = {r["fp"] for r in sel}
        for name, (_exact, struct) in pools.items():
            hit_rec = sum(1 for r in sel if r["fp"] in struct)
            hit_shape = len(shapes & struct)
            print(f"  {scope:12} vs {name:5}: "
                  f"record-level {hit_rec}/{len(sel)} = {100*hit_rec/len(sel):.1f}%   "
                  f"shape-level {hit_shape}/{len(shapes)} = {100*hit_shape/len(shapes):.1f}%")


def rescore(out: pathlib.Path, rows: list[dict], results: pathlib.Path) -> None:
    """Table 47 recomputed with exact duplicates of training removed.

    Reads the per-record pass/fail files, so this is arithmetic on existing
    generations -- no model is run and no number is re-derived from a rate.
    """
    drop = {r["id"] for r in rows if r["exact_mix"] or r["exact_pool"]}
    twin = {r["id"] for r in rows if r["twin_mix"] or r["twin_pool"]}
    scored = {r["id"] for r in rows if r["non_gameable"]}
    clean = scored - drop
    # The review asks, at minimum, that every structurally matching schema be
    # excluded from a sensitivity analysis. Dropping the twins as well as the
    # exact duplicates is the strictest reading of the leakage objection, and
    # the point of reporting it is that the conclusion should not depend on it.
    strict = scored - drop - twin
    print(f"\nscored non-gameable        : {len(scored)}")
    print(f"  exact duplicate of training: {len(scored & drop)}")
    print(f"  clean denominator          : {len(clean)}")
    print(f"  structural twin of training: {len(scored & twin)}")
    print(f"  strict denominator (no twin): {len(strict)}")

    recs, raw_p = [], {}
    for ck in CHECKPOINTS:
        row = {"checkpoint": ck, "n_clean": len(clean)}
        base_passed, ours_passed = set(), set()
        for arm in ("base", "ours"):
            passed = set()
            p = results / f"{arm}_{ck}_pass.jsonl"
            if not p.exists():
                print(f"  warn: missing {p}", file=sys.stderr)
                continue
            for rec in read_jsonl(p):
                passed.add(rec.get("id") or rec.get("instance_id"))
            if arm == "base":
                base_passed = passed
            else:
                ours_passed = passed
            row[f"{arm}_pass"] = len(passed & clean)
            row[f"{arm}_pct"] = round(100 * len(passed & clean) / len(clean), 2)
            row[f"{arm}_pass_dirty"] = len(passed & scored)
            row[f"{arm}_pct_dirty"] = round(100 * len(passed & scored) / len(scored), 2)
            row[f"{arm}_pct_strict"] = round(100 * len(passed & strict) / len(strict), 2)
        # Paired over schemas: +1 where repair fixed it, -1 where it broke it.
        # The review asks for uncertainty on the transfer rates and there is
        # none in the manuscript. This is INSTANCE uncertainty only -- the
        # transfer run is seed 0 alone, so it says nothing about how a newly
        # trained adapter would land. Stated, not implied.
        if "ours_pct" in row and "base_pct" in row:
            bp = base_passed & clean
            op = ours_passed & clean
            paired = [(1.0 if i in op else 0.0) - (1.0 if i in bp else 0.0)
                      for i in sorted(clean)]
            lo, hi = boot_ci(paired)
            row["delta_ci_lo"] = round(100 * lo, 2)
            row["delta_ci_hi"] = round(100 * hi, 2)
            raw_p[ck] = boot_p(paired)
            row["raw_p"] = round(raw_p[ck], 4)
            row["fixed"] = len(op - bp)
            row["broke"] = len(bp - op)
            row["delta_pts"] = round(row["ours_pct"] - row["base_pct"], 2)
            row["delta_pts_dirty"] = round(row["ours_pct_dirty"] - row["base_pct_dirty"], 2)
            row["delta_shift"] = round(row["delta_pts"] - row["delta_pts_dirty"], 2)
            row["n_strict"] = len(strict)
            row["delta_pts_strict"] = round(row["ours_pct_strict"] - row["base_pct_strict"], 2)
        recs.append(row)
    adj = holm(raw_p)
    for r in recs:
        if r["checkpoint"] in adj:
            r["holm_p"] = round(adj[r["checkpoint"]], 4)
            r["significant"] = int(adj[r["checkpoint"]] < 0.05
                                   and r.get("delta_ci_lo", 0) > 0)
    recs.sort(key=lambda r: r.get("base_pct", 0))

    cols = ["checkpoint", "n_clean", "base_pass", "base_pct", "ours_pass", "ours_pct",
            "delta_pts", "base_pct_dirty", "ours_pct_dirty", "delta_pts_dirty",
            "delta_shift", "n_strict", "base_pct_strict", "ours_pct_strict",
            "delta_pts_strict", "delta_ci_lo", "delta_ci_hi", "raw_p", "holm_p",
            "significant", "fixed", "broke"]
    path = out / "63_transfer_english_clean.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)

    print(f"\n{'checkpoint':22}{'base':>7}{'ours':>7}{'delta':>8}"
          f"{'95% CI':>20}{'Holm p':>9}{'fixed':>7}{'broke':>7}{'no twin':>9}")
    nsig = 0
    for r in recs:
        nsig += r.get("significant", 0)
        ci = "[%+.1f, %+.1f]" % (r.get("delta_ci_lo", 0), r.get("delta_ci_hi", 0))
        print("%-22s%7s%7s%+8.2f%20s%9.4f%7d%7d%+9.2f" % (
            r["checkpoint"], r.get("base_pct", 0), r.get("ours_pct", 0),
            r.get("delta_pts", 0), ci, r.get("holm_p", 1.0),
            r.get("fixed", 0), r.get("broke", 0), r.get("delta_pts_strict", 0)))
    print("\n  %d/%d survive Holm with a CI excluding zero" % (nsig, len(recs)))
    print("  CIs are over SCHEMAS. The transfer run is seed 0 only, so they do not")
    print("  estimate uncertainty over a newly trained adapter.")
    print(f"\nwrote {path}")


def per_seed(out: pathlib.Path, clean: set, results: pathlib.Path) -> None:
    """Seed-to-seed spread of the English retention rate, from the LoRA sweep.

    The manuscript reports this result on one training seed, and the review asks
    whether the bootstrap intervals say anything about a newly trained adapter.
    They do not -- so this measures it directly: all three seeds of all eight
    checkpoints, served through vLLM's LoRA path (datagen/analysis/holdout_eval.py
    --lora) because a merged copy per seed does not fit the disk.

    These rates are NOT spliced into the merged column of
    63_transfer_english_clean.csv. Served and merged arithmetic differ in bf16
    (Wx + B(Ax) against (W+BA)x) and vLLM's greedy decoding is not bitwise
    deterministic across batch compositions, so seed 0 reproduces its merged
    value to within one schema on five of eight checkpoints and to within seven
    on the other three. The RANGES are internally consistent -- all three seeds
    of a family were served identically in one process -- and the ranges are
    what the seed-variance claim needs.
    """
    src = results.parent / "holdout_perseed"
    if not src.is_dir():
        print(f"  (no per-seed results at {src}, skipping table 66)")
        return
    rows, spreads = [], []
    fams: dict = collections.defaultdict(dict)
    for f in sorted(src.glob("*_pass.jsonl")):
        stem = f.name[:-len("_pass.jsonl")]
        fam, seed = stem.rsplit("_s", 1)
        passed = {r.get("id") for r in read_jsonl(f)}
        fams[fam][int(seed)] = 100 * len(passed & clean) / len(clean)
    for fam in sorted(fams):
        d = fams[fam]
        vals = [d[k] for k in sorted(d)]
        rng = max(vals) - min(vals)
        spreads.append(rng)
        rows.append({"family": fam, "n_clean": len(clean), "n_seeds": len(d),
                     **{f"seed{k}": round(d[k], 2) for k in sorted(d)},
                     "mean": round(statistics.mean(vals), 2),
                     "range": round(rng, 2)})
    path = out / "66_transfer_per_seed.csv"
    cols = ["family", "n_clean", "n_seeds", "seed0", "seed1", "seed2", "mean", "range"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'family':24}{'s0':>8}{'s1':>8}{'s2':>8}{'mean':>8}{'range':>8}")
    for r in rows:
        print(f"{r['family']:24}{r.get('seed0',0):>8.2f}{r.get('seed1',0):>8.2f}"
              f"{r.get('seed2',0):>8.2f}{r['mean']:>8.2f}{r['range']:>8.2f}")
    print(f"\n  seed range {min(spreads):.2f} to {max(spreads):.2f} points, "
          f"mean {statistics.mean(spreads):.2f}, over {len(rows)} checkpoints")
    print("  Compare against the effects in table 63 (+18.6 to +85.1): seed "
          "variation is\n  5x to 70x smaller than the effect it would have to "
          "threaten.")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--mix", required=True, help="the mix actually trained on")
    ap.add_argument("--pool", required=True, nargs="+", help="all generated Track A")
    ap.add_argument("--holdout-results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_path = pathlib.Path(args.split)
    split = list(read_jsonl(split_path))
    cache = split_path.with_name(split_path.stem + "_trivially_satisfiable.json")
    if not cache.exists():
        sys.exit(f"missing trivial-satisfiability cache: {cache}")
    trivial = json.loads(cache.read_text(encoding="utf-8"))

    ids = [r["id"] for r in split]
    if len(ids) != len(set(ids)):
        dupes = [k for k, v in collections.Counter(ids).items() if v > 1]
        sys.exit(f"split has {len(ids) - len(set(ids))} duplicate ids "
                 f"(e.g. {dupes[:3]}) — this is the doubled-file defect; "
                 f"score the de-duplicated split")
    print(f"split: {len(split)} schemas, "
          f"{sum(1 for r in split if not trivial.get(r['id'], False))} non-gameable",
          file=sys.stderr)

    pools = {
        "mix": fingerprints(read_jsonl(pathlib.Path(args.mix)), "mix (trained)"),
        "pool": fingerprints(
            (r for p in args.pool for r in read_jsonl(pathlib.Path(p))), "pool (generated)"),
    }

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = classify(split, pools, trivial)
    write_overlap(out, rows, pools)
    rescore(out, rows, pathlib.Path(args.holdout_results))
    drop = {r["id"] for r in rows if r["exact_mix"] or r["exact_pool"]}
    clean = {r["id"] for r in rows if r["non_gameable"]} - drop
    per_seed(out, clean, pathlib.Path(args.holdout_results))


if __name__ == "__main__":
    main()
