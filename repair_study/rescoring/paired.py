#!/usr/bin/env python3
"""Paired repair effects under the corrected evaluator, with Holm correction.

Conventions taken from the handoff README: match baseline and repaired records by
instance_id inside the retained cohort, use the repair experiment's own baseline,
keep each training seed separate, paired task bootstrap with seed 42.

Seeds are averaged per instance before the paired bootstrap, which is what the
legacy analysis did; that makes these intervals task uncertainty conditional on
the three trained adapters, NOT uncertainty over a newly trained adapter. The
per-seed spread is reported separately in table 67.
"""
import csv, json, pathlib, random, statistics

RUNS = pathlib.Path("../results/rescored"); OUT = pathlib.Path("../results/tables")
N_BOOT, SEED = 5000, 42
FAMS = [("RoLlama3.1-8B-DPO","rollama31-dpo_4kv2"),("RoGemma2-9B-DPO","rogemma2-dpo_4kv2"),
        ("RoGemma2-9B-SFT","rogemma2-sft_4kv2"),("RoLlama3-8B-DPO","rollama3-dpo_4kv2"),
        ("RoMistral-7B-DPO","romistral-dpo_4kv2"),("RoMistral-7B-SFT","romistral-sft_4kv2"),
        ("RoLlama2-7B","rollama2-sft_4kv2"),("RoGemma-7B-DPO","rogemma-dpo_4kv2")]

def per_inst(d, field):
    p = RUNS/d/"per_instance.jsonl"
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("eligible"):
            continue
        v = r[field]
        out[r["instance_id"]] = float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
    return out

def boot(vals, seed=SEED, n=N_BOOT):
    rng = random.Random(seed); k = len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k))/k for _ in range(n))
    return means[int(0.025*n)], means[int(0.975*n)]

def boot_p(vals, seed=SEED, n=N_BOOT):
    rng = random.Random(seed); k = len(vals)
    means = [sum(vals[rng.randrange(k)] for _ in range(k))/k for _ in range(n)]
    obs = statistics.mean(vals)
    tail = sum(1 for m in means if m <= 0) if obs > 0 else sum(1 for m in means if m >= 0)
    return max(1.0/n, min(1.0, 2.0*tail/n))

def holm(p):
    o = sorted(p.items(), key=lambda kv: kv[1]); m = len(o); out = {}; run = 0.0
    for i, (k, v) in enumerate(o):
        run = max(run, min(1.0, (m-i)*v)); out[k] = run
    return out

for split in ("standard", "adversarial"):
    for field, label, scale in (("T", "T", 1.0), ("joint", "J", 100.0)):
        stats, praw = {}, {}
        for disp, pre in FAMS:
            b = per_inst(f"{pre}_base_{split}", field)
            seeds = [per_inst(f"{pre}_s{s}_{split}", field) for s in (0,1,2)]
            ids = sorted(set(b) & set.intersection(*(set(s) for s in seeds)))
            diffs = [statistics.mean(s[i] for s in seeds) - b[i] for i in ids]
            lo, hi = boot(diffs)
            stats[disp] = (statistics.mean(diffs)*scale, lo*scale, hi*scale, len(ids))
            praw[disp] = boot_p(diffs)
        adj = holm(praw)
        rows, nsig = [], 0
        for disp, (d, lo, hi, n) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            sig = int(adj[disp] < 0.05 and (lo > 0 or hi < 0)); nsig += sig
            rows.append({"checkpoint": disp, "split": split, "metric": label, "n": n,
                         "delta": round(d,4), "ci_lo": round(lo,4), "ci_hi": round(hi,4),
                         "raw_p": round(praw[disp],4), "holm_p": round(adj[disp],4),
                         "significant": sig})
        with open(OUT/f"69_repair_paired_{label}_{split}.csv","w",newline="",encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        unit = "" if label=="T" else " pts"
        print(f"\n=== {split}: paired Δ{label} under the corrected evaluator "
              f"(n={rows[0]['n']}) ===")
        print(f"{'checkpoint':20}{'Δ'+label:>9}{'95% CI':>22}{'Holm p':>9}")
        for r in rows:
            print(f"{r['checkpoint']:20}{r['delta']:>+9.3f}"
                  f"{f'[{r[chr(99)+chr(105)+chr(95)+chr(108)+chr(111)]:+.3f}, {r[chr(99)+chr(105)+chr(95)+chr(104)+chr(105)]:+.3f}]':>22}"
                  f"{r['holm_p']:>9.4f}{'  *' if r['significant'] else ''}")
        print(f"  -> {nsig}/8 survive Holm with an interval excluding zero")
print(f"\nwrote {OUT}/69_*")
