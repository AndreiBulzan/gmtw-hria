#!/usr/bin/env python3
"""Corrected repair tables from the supplied evaluator's per-run summaries.

Replaces the legacy U / R_old / F_v3 / old-E2E columns with the corrected
outcomes the rest of the paper now uses: T, R0, Rq, Rr, S and J. Repaired rows
are the mean over three training seeds; the seed range is reported beside them so
instance and seed uncertainty stay separate.

Deltas are the repaired mean minus that checkpoint's OWN baseline, both scored by
the same evaluator on the same retained cohort.
"""
import csv, json, pathlib, statistics, sys

RUNS = pathlib.Path("../results/rescored")
OUT = pathlib.Path("../results/tables")
FAMS = [("RoLlama3.1-8B-DPO","rollama31-dpo_4kv2"),("RoGemma2-9B-DPO","rogemma2-dpo_4kv2"),
        ("RoGemma2-9B-SFT","rogemma2-sft_4kv2"),("RoLlama3-8B-DPO","rollama3-dpo_4kv2"),
        ("RoMistral-7B-DPO","romistral-dpo_4kv2"),("RoMistral-7B-SFT","romistral-sft_4kv2"),
        ("RoLlama2-7B","rollama2-sft_4kv2"),("RoGemma-7B-DPO","rogemma-dpo_4kv2")]
METRICS = ["T","raw_ok","normalized_ok","recovered","schema","joint"]

def summ(d):
    p = RUNS/d/"summary.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None

for split in ("standard","adversarial"):
    rows, deltas = [], []
    for disp, pre in FAMS:
        b = summ(f"{pre}_base_{split}")
        seeds = [summ(f"{pre}_s{s}_{split}") for s in (0,1,2)]
        seeds = [s for s in seeds if s]
        if not b or len(seeds) != 3:
            print(f"  skip {disp} {split}: base={bool(b)} seeds={len(seeds)}"); continue
        row = {"checkpoint": disp, "split": split, "n": b["n"], "n_seeds": len(seeds)}
        for m in METRICS:
            bv = b[m]["mean"]
            sv = [s[m]["mean"] for s in seeds]
            row[f"base_{m}"] = round(bv, 4)
            row[f"ours_{m}"] = round(statistics.mean(sv), 4)
            row[f"range_{m}"] = round(max(sv)-min(sv), 4)
        row["base_QA"] = round(b["QA"]["mean"], 4)
        row["ours_QA"] = round(statistics.mean(s["QA"]["mean"] for s in seeds), 4)
        rows.append(row)
        d = {"checkpoint": disp, "split": split, "n": b["n"]}
        for m in METRICS:
            d[f"d_{m}"] = round(statistics.mean(s[m]["mean"] for s in seeds) - b[m]["mean"], 4)
        d["d_QA"] = round(statistics.mean(s["QA"]["mean"] for s in seeds) - b["QA"]["mean"], 4)
        deltas.append(d)

    for name, data in ((f"67_repair_corrected_{split}.csv", rows),
                       (f"68_repair_corrected_deltas_{split}.csv", deltas)):
        if not data: continue
        with open(OUT/name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    if not rows: continue
    print(f"\n=== {split}: corrected repair outcomes (n={rows[0]['n']} retained tasks) ===")
    print(f"{'checkpoint':20}{'T base':>8}{'T ours':>8}{'dT':>8}"
          f"{'R0 base':>9}{'R0 ours':>9}{'J base':>8}{'J ours':>8}{'dJ':>8}{'seedR(J)':>9}")
    for r in rows:
        print(f"{r['checkpoint']:20}{r['base_T']:>8.3f}{r['ours_T']:>8.3f}"
              f"{r['ours_T']-r['base_T']:>+8.3f}{100*r['base_raw_ok']:>9.1f}"
              f"{100*r['ours_raw_ok']:>9.1f}{100*r['base_joint']:>8.1f}"
              f"{100*r['ours_joint']:>8.1f}{100*(r['ours_joint']-r['base_joint']):>+8.1f}"
              f"{100*r['range_joint']:>9.2f}")
    dT = [d["d_T"] for d in deltas]; dJ = [d["d_joint"] for d in deltas]
    print(f"  dT  {min(dT):+.3f} to {max(dT):+.3f}   all positive: {all(x>0 for x in dT)}")
    print(f"  dJ  {100*min(dJ):+.1f} to {100*max(dJ):+.1f} pts   all positive: {all(x>0 for x in dJ)}")
print(f"\nwrote {OUT}/67_* and {OUT}/68_*")
