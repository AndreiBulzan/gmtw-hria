#!/usr/bin/env python3
"""Retention tables: what the repair costs on OpenLLM-Ro's own benchmarks.

Turns the per-rung `results_*.json` the harness writes into the three tables the
retention section argues from. Standard library only, like the rest of the
analysis code, so it runs wherever the results are.

  python datagen/retention/retention_tables.py --results datagen/retention/results \\
      --out datagen/retention/tables

Three tables, because three different questions are being asked:

  52_retention_per_rung   every rung against its published counterpart. This is
                          the environment check, and it is per-rung because the
                          cards publish per-rung -- an average agreeing while
                          its rungs disagree would be a coincidence, not a
                          reproduction.
  53_retention_averages   the published quantity: each task's mean over its
                          ladder, beside OpenLLM-Ro's number for base rows.
  54_retention_deltas     base vs repaired, per task AND the academic average,
                          with the seed spread. The §3.3 answer.

THE ACADEMIC AVERAGE IS DEFINED HERE, because the manuscript reported it without
saying which of two readings it used and the reviewer asked. A task's score is the
mean over exactly the rungs in its published ladder; the academic average is the
mean of the six task scores. So TASKS ARE WEIGHTED EQUALLY and shot counts are
not -- ARC contributes one sixth despite having six rungs, and TruthfulQA one
sixth despite having one. The alternative, weighting every task-and-shot
combination equally, would give ARC six times TruthfulQA's influence; that is not
what is reported anywhere in this repo.

The average row appears in 54_retention_deltas as task "ACADEMIC_AVG" and carries
the change in BOTH units: `delta` in score points and `delta_pct` as a percentage
of the base. The manuscript mixed them -- quoting a 12.4 figure that is a relative
percentage next to a seed_range column in score points -- so both are emitted and
named.

Which run is which is read from each run's `provenance_<name>.txt`, not from its
directory name: provenance records the checkpoint that was actually served, so a
mislabelled directory cannot silently become a row. The same file carries the
`limit` and whether the chat template was applied, and rows that differ in
either are not comparable -- the delta table pairs within one limit and one
prompt format, and the published column is attached only to completion-format
rows, because that is the format OpenLLM-Ro scored.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

TASK = {
    "_ro_arc_challenge": "ARC",
    "_ro_mmlu": "MMLU",
    "_ro_winogrande": "Winogrande",
    "_ro_hellaswag": "HellaSwag",
    "_ro_gsm8k": "GSM8k",
    "_ro_truthfulqa": "TruthfulQA",
}
ORDER = ["ARC", "MMLU", "Winogrande", "HellaSwag", "GSM8k", "TruthfulQA"]

# The published ladder, asserted against the cards by published_baselines.py.
# A task average is the mean over exactly these rungs; a run missing one of them
# does not have that task's published quantity and is reported as partial.
LADDER = {"ARC": (0, 1, 3, 5, 10, 25), "MMLU": (0, 1, 3, 5),
          "Winogrande": (0, 1, 3, 5), "HellaSwag": (0, 1, 3, 5, 10),
          "GSM8k": (1, 3, 5), "TruthfulQA": (0,)}

# Checkpoint we trained from -> display name, matching repair_tables.ROWS so the
# retention rows join to the repair rows by model.
BASE_TO_DISPLAY = {
    "OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23": "RoLlama3.1-8B-DPO",
    "OpenLLM-Ro/RoGemma2-9b-Instruct-DPO-2025-04-23": "RoGemma2-9B-DPO",
    "OpenLLM-Ro/RoGemma2-9b-Instruct-2025-04-23": "RoGemma2-9B-SFT",
    "OpenLLM-Ro/RoLlama3-8b-Instruct-DPO-2025-04-23": "RoLlama3-8B-DPO",
    "OpenLLM-Ro/RoMistral-7b-Instruct-DPO-2025-04-23": "RoMistral-7B-DPO",
    "OpenLLM-Ro/RoMistral-7b-Instruct-2025-04-23": "RoMistral-7B-SFT",
    "OpenLLM-Ro/RoLlama2-7b-Instruct": "RoLlama2-7B",
    "OpenLLM-Ro/RoGemma-7b-Instruct-DPO": "RoGemma-7B-DPO",
}
# Dose-curve runs: the intermediate adapters every training run saved, merged
# and evaluated to see where along training the repair lands and where the
# damage does. `dose_<key>_c<step>`; the step is carried through to the tables.
DOSE_TO_DISPLAY = {
    "dose_rollama31": "RoLlama3.1-8B-DPO",
}

# Merged-checkpoint prefix -> the same display name.
REPAIRED_TO_DISPLAY = {
    "rollama31-dpo_4kv2": "RoLlama3.1-8B-DPO",
    "rogemma2-dpo_4kv2": "RoGemma2-9B-DPO",
    "rogemma2-sft_4kv2": "RoGemma2-9B-SFT",
    "rollama3-dpo_4kv2": "RoLlama3-8B-DPO",
    "romistral-dpo_4kv2": "RoMistral-7B-DPO",
    "romistral-sft_4kv2": "RoMistral-7B-SFT",
    "rollama2-sft_4kv2": "RoLlama2-7B",
    "rogemma-dpo_4kv2": "RoGemma-7B-DPO",
}


def read_provenance(path: pathlib.Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("---"):
            break
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1].strip()
    return out


def classify(model: str) -> tuple[str, str, int | None, int | None]:
    """(display name, arm, seed, training step) for the checkpoint served."""
    if model in BASE_TO_DISPLAY:
        return BASE_TO_DISPLAY[model], "base", None, None
    name = model.rstrip("/").split("/")[-1]
    prefix, _, seed = name.rpartition("_s")
    if prefix in REPAIRED_TO_DISPLAY and seed.isdigit():
        return REPAIRED_TO_DISPLAY[prefix], "repaired", int(seed), None
    dose_prefix, sep, step = name.rpartition("_c")
    if sep and dose_prefix in DOSE_TO_DISPLAY and step.isdigit():
        return DOSE_TO_DISPLAY[dose_prefix], "dose", None, int(step)
    raise KeyError(model)


def load_runs(results: pathlib.Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        prov_files = list(run_dir.glob("provenance_*.txt"))
        if not prov_files:
            print(f"skipping {run_dir.name}: no provenance file", file=sys.stderr)
            continue
        prov = read_provenance(prov_files[0])
        try:
            display, arm, seed, step = classify(prov["model"])
        except KeyError as exc:
            # Skipped rather than fatal: an unregistered checkpoint should not
            # stop the tables the registered ones can still produce. It is
            # named loudly so it is not mistaken for having been included.
            print(f"skipping {run_dir.name}: unregistered checkpoint {exc.args[0]!r} "
                  f"-- add it to BASE_/REPAIRED_/DOSE_TO_DISPLAY to include it",
                  file=sys.stderr)
            continue
        limit = prov.get("limit", "none")
        # "off" for every run that predates the format control, which is what
        # those runs were.
        chat = prov.get("chat_template", "off")
        scores: dict[str, dict[int, tuple[float, str]]] = collections.defaultdict(dict)
        for res in sorted(run_dir.glob("fs*/*/results_*.json")):
            rung = int(res.parent.parent.name[2:])
            for task, metrics in json.load(open(res))["results"].items():
                keys = [k for k in metrics if k != "alias" and "stderr" not in k]
                if len(keys) != 1:
                    raise ValueError(f"{res}: {task} reports {keys}, expected one metric")
                scores[TASK[task]][rung] = (metrics[keys[0]] * 100, keys[0])
        runs.append({"run": run_dir.name, "model": display, "arm": arm, "seed": seed,
                     "step": step, "limit": limit, "chat": chat,
                     "checkpoint": prov["model"], "scores": scores})
    return runs


def published(tables: pathlib.Path):
    avg, rung = {}, {}
    with open(tables / "50_published_openllm_ro.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            avg[r["model"]] = r
    with open(tables / "51_published_openllm_ro_per_fewshot.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rung[(r["model"], r["task"], int(r["num_fewshot"]))] = float(r["published"])
    return avg, rung


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--tables", type=pathlib.Path,
                    help="where the published CSVs live (default: --out)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pub_avg, pub_rung = published(args.tables or args.out)

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no runs with provenance under {args.results}")

    per_rung, averages = [], []
    for run in runs:
        for task in ORDER:
            got = run["scores"].get(task)
            if not got:
                continue
            for rung in sorted(got):
                value, metric = got[rung]
                p = (pub_rung.get((run["model"], task, rung))
                     if run["arm"] == "base" and run["chat"] == "off" else None)
                per_rung.append({
                    "run": run["run"], "model": run["model"], "arm": run["arm"],
                    "seed": run["seed"], "step": run["step"], "limit": run["limit"],
                    "chat": run["chat"], "task": task,
                    "num_fewshot": rung, "metric": metric, "value": round(value, 2),
                    "published": p if p is not None else "",
                    "diff": round(value - p, 2) if p is not None else "",
                })
            complete = tuple(sorted(got)) == LADDER[task]
            mean = sum(v for v, _ in got.values()) / len(got)
            p = (float(pub_avg[run["model"]][task])
                 if run["arm"] == "base" and run["chat"] == "off" else None)
            averages.append({
                "run": run["run"], "model": run["model"], "arm": run["arm"],
                "seed": run["seed"], "step": run["step"], "limit": run["limit"],
                "chat": run["chat"], "task": task,
                "rungs": "|".join(str(r) for r in sorted(got)),
                "complete_ladder": complete,
                "average": round(mean, 2),
                "published": p if p is not None else "",
                "diff": round(mean - p, 2) if p is not None else "",
            })

    # Deltas: repaired minus base, per task, averaged over seeds, only where both
    # arms were measured under the same limit.
    by = collections.defaultdict(dict)
    for a in averages:
        by[(a["model"], a["limit"], a["chat"], a["task"])].setdefault(a["arm"], []).append(a)
    deltas = []
    for (model, limit, chat, task), arms in sorted(by.items()):
        if "base" not in arms or "repaired" not in arms:
            continue
        base = arms["base"][0]["average"]
        seeds = sorted((a["seed"], a["average"]) for a in arms["repaired"])
        vals = [v for _, v in seeds]
        mean = sum(vals) / len(vals)
        deltas.append({
            "model": model, "task": task, "limit": limit, "chat": chat,
            "base": base, "repaired_mean": round(mean, 2),
            "delta": round(mean - base, 2),
            "delta_pct": round(100 * (mean - base) / base, 2) if base else "",
            "n_seeds": len(vals),
            "per_seed": "|".join(f"s{s}={v:.2f}" for s, v in seeds),
            "seed_range": round(max(vals) - min(vals), 2) if len(vals) > 1 else "",
        })

    # The academic average: the mean of the six task scores, per seed, so the
    # spread is a spread of the quantity actually reported rather than of one
    # task. Emitted only where the full ladder is present for every task, since
    # an average over five tasks is a different number wearing the same name.
    per_model = collections.defaultdict(lambda: collections.defaultdict(dict))
    for a in averages:
        if a["complete_ladder"]:
            per_model[(a["model"], a["limit"], a["chat"])][a["arm"], a["seed"]][a["task"]] = a["average"]
    for (model, limit, chat), arms in sorted(per_model.items()):
        base_cells = [v for (arm, _), v in arms.items() if arm == "base"]
        rep_cells = sorted(((seed, v) for (arm, seed), v in arms.items()
                            if arm == "repaired"), key=lambda x: (x[0] is None, x[0]))
        if not base_cells or not rep_cells:
            continue
        if len(base_cells[0]) != len(ORDER) or any(len(v) != len(ORDER) for _, v in rep_cells):
            continue
        b = sum(base_cells[0].values()) / len(ORDER)
        seeds = [(s, sum(v.values()) / len(ORDER)) for s, v in rep_cells]
        vals = [v for _, v in seeds]
        m = sum(vals) / len(vals)
        deltas.append({
            "model": model, "task": "ACADEMIC_AVG", "limit": limit, "chat": chat,
            "base": round(b, 2), "repaired_mean": round(m, 2),
            "delta": round(m - b, 2),
            "delta_pct": round(100 * (m - b) / b, 2) if b else "",
            "n_seeds": len(vals),
            "per_seed": "|".join(f"s{s}={v:.2f}" for s, v in seeds),
            "seed_range": round(max(vals) - min(vals), 2) if len(vals) > 1 else "",
        })

    for name, rows in (("52_retention_per_rung", per_rung),
                       ("53_retention_averages", averages),
                       ("54_retention_deltas", deltas)):
        path = args.out / f"{name}.csv"
        if not rows:
            print(f"{name}: nothing to write yet")
            continue
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows):4d} rows -> {path}")

    for run in runs:
        partial = [a["task"] for a in averages
                   if a["run"] == run["run"] and not a["complete_ladder"]]
        note = f"  PARTIAL ladder: {', '.join(partial)}" if partial else ""
        step = f" step={run['step']}" if run["step"] is not None else ""
        print(f"  {run['run']:28s} {run['arm']:8s} seed={run['seed']}{step} "
              f"limit={run['limit']} chat={run['chat']}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
