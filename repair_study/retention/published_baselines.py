#!/usr/bin/env python3
"""OpenLLM-Ro's own published scores for the eight checkpoints we repaired.

The retention question -- does the repair cost anything outside GMTW-Ro -- is
asked against the suite OpenLLM-Ro scores its releases on, so the released
checkpoints already have a public row on it. That row is the reference our
own base run has to reproduce before any repaired row means anything: if we
cannot land near the published number for the base checkpoint, our harness is
not the one the numbers came from and no delta computed in it transfers.

Every number here is read out of the model card's `model-index` front matter,
which is where OpenLLM-Ro publishes them, and each card also carries the
per-few-shot values behind its own average. Both are emitted.

  python datagen/retention/published_baselines.py --out datagen/retention/tables

Standard library only, like everything under `analysis/` -- the front matter is
parsed by the small reader below rather than by pulling in PyYAML, because the
shape is fixed, narrow, and asserted on the way out.

The cards are fetched over the network (huggingface.co) and are third-party
content: this writes the numbers, not the cards.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import urllib.request

# (display name used by repair_tables.ROWS, the checkpoint we trained from)
MODELS = [
    ("RoLlama3.1-8B-DPO", "OpenLLM-Ro/RoLlama3.1-8b-Instruct-DPO-2025-04-23"),
    ("RoGemma2-9B-DPO",   "OpenLLM-Ro/RoGemma2-9b-Instruct-DPO-2025-04-23"),
    ("RoGemma2-9B-SFT",   "OpenLLM-Ro/RoGemma2-9b-Instruct-2025-04-23"),
    ("RoLlama3-8B-DPO",   "OpenLLM-Ro/RoLlama3-8b-Instruct-DPO-2025-04-23"),
    ("RoMistral-7B-DPO",  "OpenLLM-Ro/RoMistral-7b-Instruct-DPO-2025-04-23"),
    ("RoMistral-7B-SFT",  "OpenLLM-Ro/RoMistral-7b-Instruct-2025-04-23"),
    # Undated aliases, matching what was trained and what repair_tables uses.
    # Both resolve to a dated card, and the resolved name is recorded per row
    # because for RoLlama2 the undated alias and the 2024-10-09 release are
    # different weights.
    ("RoLlama2-7B",       "OpenLLM-Ro/RoLlama2-7b-Instruct"),
    ("RoGemma-7B-DPO",    "OpenLLM-Ro/RoGemma-7b-Instruct-DPO"),
]

# The six that make up the published "Romanian Academic Benchmarks" average, in
# the order the cards list them, each with the few-shot ladder its published
# average is taken over. The ladders are asserted against the cards rather than
# trusted: they are the same ladders the group YAMLs in the harness run, and if
# a card ever publishes a different one, our run would no longer be the
# published protocol. TruthfulQA is 0-shot only, so its card carries the average
# alone and no ladder block.
ACADEMIC = [
    ("ARC",         "OpenLLM-Ro/ro_arc_challenge", (0, 1, 3, 5, 10, 25)),
    ("MMLU",        "OpenLLM-Ro/ro_mmlu",          (0, 1, 3, 5)),
    ("Winogrande",  "OpenLLM-Ro/ro_winogrande",    (0, 1, 3, 5)),
    ("HellaSwag",   "OpenLLM-Ro/ro_hellaswag",     (0, 1, 3, 5, 10)),
    ("GSM8k",       "OpenLLM-Ro/ro_gsm8k",         (1, 3, 5)),
    ("TruthfulQA",  "OpenLLM-Ro/ro_truthfulqa",    (0,)),
]
JUDGED = [("RoMT-Bench", "RoMT-Bench"), ("RoCulturaBench", "RoCulturaBench")]
AVERAGE = ("Academic avg", "Romanian_Academic_Benchmarks")

CARD_URL = "https://huggingface.co/{}/raw/main/README.md"
SHOT = re.compile(r"^(\d+)-shot\s*$")


def fetch(hf_id: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(CARD_URL.format(hf_id), timeout=timeout) as fh:
        return fh.read().decode("utf-8")


def parse_card(text: str) -> tuple[str, dict[tuple[str, str], float]]:
    """(card name, {(dataset, metric name): value}) out of the front matter.

    An indentation-free reader on purpose: the only nesting that matters is
    "which dataset was named most recently", and the cards put every metric
    directly under the dataset it belongs to. Anything unrecognised is skipped
    rather than guessed at, and the caller asserts that what it needed arrived.
    """
    try:
        front = text.split("---", 2)[1]
    except IndexError:
        raise ValueError("no YAML front matter in card")

    card_name, dataset, metric = "", None, None
    in_dataset = in_metrics = False
    out: dict[tuple[str, str], float] = {}

    for raw in front.splitlines():
        line = raw.strip()
        if line.startswith("- name:") and not card_name and not in_metrics:
            card_name = line.split(":", 1)[1].strip()
        elif line == "dataset:":
            in_dataset, in_metrics = True, False
        elif line == "metrics:":
            in_dataset, in_metrics = False, True
        elif line in ("- task:", "results:"):
            in_dataset = in_metrics = False
        elif in_dataset and line.startswith("name:"):
            dataset = line.split(":", 1)[1].strip()
        elif in_metrics and line.startswith("- name:"):
            metric = line.split(":", 1)[1].strip()
        elif in_metrics and line.startswith("value:") and dataset and metric:
            try:
                out[(dataset, metric)] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass  # a non-numeric value is not a score
    return card_name, out


def average_of(values: dict[tuple[str, str], float], dataset: str) -> float | None:
    for key, val in values.items():
        if key[0] == dataset and key[1].startswith(("Average", "Score")):
            return val
    return None


def per_fewshot(values: dict[tuple[str, str], float], dataset: str) -> dict[int, float]:
    found = {}
    for (ds, metric), val in values.items():
        m = SHOT.match(metric)
        if ds == dataset and m:
            found[int(m.group(1))] = val
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="directory for the two CSVs")
    ap.add_argument("--cards", type=pathlib.Path,
                    help="read <hf-basename>.md from here instead of fetching")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, fewshot_rows, missing = [], [], []
    for display, hf_id in MODELS:
        if args.cards:
            text = (args.cards / f"{hf_id.split('/')[-1]}.md").read_text(encoding="utf-8")
        else:
            text = fetch(hf_id)
        card_name, values = parse_card(text)

        row = {"model": display, "requested": hf_id, "card": card_name}
        row[AVERAGE[0]] = average_of(values, AVERAGE[1])
        for label, dataset in [(l, d) for l, d, _ in ACADEMIC] + JUDGED:
            row[label] = average_of(values, dataset)
            if row[label] is None:
                missing.append(f"{display}: no average for {dataset}")
        for label, dataset, ladder in ACADEMIC:
            cells = per_fewshot(values, dataset)
            if not cells and len(ladder) == 1:
                # Single-setting task: the average IS the one cell.
                cells = {ladder[0]: row[label]}
            if tuple(sorted(cells)) != ladder:
                missing.append(f"{display}: {dataset} publishes shots "
                               f"{tuple(sorted(cells))}, expected {ladder}")
            for shots, val in sorted(cells.items()):
                fewshot_rows.append({"model": display, "task": label,
                                     "num_fewshot": shots, "published": val})
        rows.append(row)

    if missing:
        raise SystemExit("published cards do not match what we join on:\n  " +
                         "\n  ".join(missing))
    if not fewshot_rows:
        raise SystemExit("no per-few-shot values parsed -- the card layout changed")

    cols = ["model", "requested", "card", AVERAGE[0]] + \
           [c for c, _, _ in ACADEMIC] + [c for c, _ in JUDGED]
    main_csv = args.out / "50_published_openllm_ro.csv"
    with open(main_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    fs_csv = args.out / "51_published_openllm_ro_per_fewshot.csv"
    with open(fs_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "task", "num_fewshot", "published"])
        w.writeheader()
        w.writerows(fewshot_rows)

    print(f"{len(rows)} checkpoints -> {main_csv}")
    print(f"{len(fewshot_rows)} per-few-shot cells -> {fs_csv}")
    head = ["model", AVERAGE[0]] + [c for c, _, _ in ACADEMIC]
    print("\n" + "".join(f"{h:>13s}" for h in head))
    for r in rows:
        print("".join(f"{str(r[h]):>13s}" for h in head))
    for r in rows:
        if r["requested"].split("/")[-1] not in r["card"]:
            print(f"note: {r['requested']} resolves to the card for {r['card']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
