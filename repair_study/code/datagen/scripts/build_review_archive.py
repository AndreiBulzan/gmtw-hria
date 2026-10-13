#!/usr/bin/env python3
"""Package a review-time archive for the repair study (§5.10, Tables 21-26).

The review's §8 says plainly that promising a URL at camera-ready does not make
current results reproducible, and rates two of our protocols Insufficient. This
builds the thing it asks for: corpus, configurations, every per-instance score
record, the scripts, and adapter hashes, with a MANIFEST that pins each file by
SHA-256 and a README that maps every reported table to the one command that
regenerates it.

What goes in, and why each is load-bearing:

  corpus          the 4,000 training + 200 validation records and their mix
                  manifests. Without these the corpus-accounting table cannot be
                  checked and the 75/25 language split is an assertion.
  configs         every training YAML. Note that target_modules in the YAML is
                  dead config -- train_lora.py hardcodes the pattern -- so the
                  README records the regex as authoritative rather than letting
                  a reader believe the key.
  scores          per-instance records for all 8 checkpoints x (base + 3 seeds)
                  on both splits, plus the English retention runs (merged seed 0
                  and the 24-run per-seed sweep). Everything reported is
                  arithmetic over these; nothing needs a GPU to re-derive.
  eval split      the 1,804-schema English set and its trivial-satisfiability
                  cache, so the 1,210/1,209 denominators are checkable.
  scripts         the analysis and training code, and the tests.
  hashes          adapters are ~8.3 GB and live on the training host, so they are
                  pinned by SHA-256 rather than copied. The review explicitly
                  allows "adapters or hashes".

Deliberately NOT included: merged checkpoints (reconstructible from base +
adapter), and the base models themselves (upstream, pinned by revision).

  python datagen/scripts/build_review_archive.py --out /tmp/ro_so_review
  python datagen/scripts/build_review_archive.py --out /tmp/ro_so_review --tar
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

# (source, archive destination, required)
#
# Two rules shape this mapping. Data gets plain descriptive names, because a
# reader arrives at the repository with no context. Code keeps its original
# package structure under code/, because the analysis scripts import each other
# -- holdout_overlap.py reaches a `ro-so` sibling for schema_fingerprint, and
# holdout_eval.py imports datagen.core -- so flattening the code tree would ship
# scripts that cannot run and a README that claims they can.
CONTENTS = [
    # --- data a reader wants to find without reading anything first ---
    ("datagen/data/sft",                      "corpus",                True),
    ("datagen/training",                      "configs",               True),
    ("datagen/results/tables",                "tables",                True),
    ("ro-so/eval/jsonschemabench_holdout",    "eval/jsonschemabench",  True),
    ("datagen/retention",                     "retention",             True),
    ("datagen/rescoring",                     "rescoring",             False),
    # "holdout" is the wrong word and this study is what established that: the
    # set is the upstream test split of the four sources Track A trains on, so
    # it measures retention on unseen instances, not a held-out family.
    ("datagen/results/holdout",               "scores/english_schema", False),
    ("datagen/results/holdout_perseed",       "scores/english_schema_per_seed", False),
    ("datagen/results/rescored",              "scores/rescored",       False),
    # everything else under results/: the per-instance GMTW-Ro scores and the
    # raw generations, 8 checkpoints x (base + 3 seeds) x 2 splits.
    ("datagen/results",                       "scores/gmtw_ro",        True),
    # --- code, structure preserved so the imports resolve ---
    ("datagen/analysis",                      "code/datagen/analysis", True),
    ("datagen/core",                          "code/datagen/core",     True),
    ("datagen/scripts",                       "code/datagen/scripts",  True),
    ("datagen/tests",                         "code/datagen/tests",    True),
    ("ro-so/filters",                         "code/ro-so/filters",    True),
    ("ro-so/templates",                       "code/ro-so/templates",  True),
    # The analysis code imports jsonschema and others; shipping it without its
    # dependencies declared makes the regeneration instructions unfollowable.
    ("datagen/requirements.txt",              "requirements.txt",      False),
]

# Each reported table and the single command that regenerates it.
PROVENANCE = {
    "40_repair_main_*.csv":        "repair_tables.py --analysis <results> --split <n>",
    "41_repair_deltas_*.csv":      "repair_tables.py --analysis <results> --split <n>",
    "43_repair_per_seed_*.csv":    "repair_tables.py --analysis <results> --split <n> --per-seed",
    "44_repair_multiplicity_*.csv": "repair_tables.py --analysis <results> --split <n> --multiplicity",
    "45_train_eval_separation.csv": "contamination_audit.py --corpus <sft> --eval <gmtw> --out <tables>",
    "46_corpus_ablations_*.csv":   "repair_tables.py --analysis <results> --split <n> --ablations",
    "55_empty_plan_by_cause_*.csv": "empty_plan_audit.py --analysis <results> --split <n>",
    "56_faith_composite_v123_*.csv": "empty_plan_audit.py --analysis <results> --split <n>",
    "57_end_to_end_v2_v3_*.csv":  "empty_plan_audit.py --analysis <results> --split <n>",
    "58_repair_deltas_v3_*.csv":  "empty_plan_audit.py --analysis <results> --split <n>",
    "59_repair_paired_t_*.csv":   "repair_tables.py --analysis <results> --split <n> --paired-t",
    "60_corpus_accounting.csv":   "corpus_accounting.py --corpus <sft> --out <tables>",
    "61_ablation_replacement.csv": "corpus_accounting.py --corpus <sft> --out <tables>",
    "62_holdout_overlap.csv":     "holdout_overlap.py --split <english> --mix <mix> --pool <pools> ...",
    "63_transfer_english_clean.csv": "holdout_overlap.py (same invocation)",
    "64_repair_v3_holm_*.csv":    "empty_plan_audit.py --analysis <results> --split <n>",
    "65_scorer_audit.csv":        "scorer_audit.py --metrics <gmtw new_data> --out <tables>",
    "66_transfer_per_seed.csv":   "holdout_overlap.py (same invocation; reads scores/holdout_perseed)",
}


def sha256(path: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


# Subtrees of datagen/results that are archived under their own destination, so
# the catch-all entry must not copy them a second time.
ALREADY_PLACED = {"tables", "holdout", "holdout_perseed", "rescored"}


def copy_tree(src: pathlib.Path, dst: pathlib.Path, skip_top: set | None = None) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel_parts = p.relative_to(src).parts
        if skip_top and rel_parts and rel_parts[0] in skip_top:
            continue
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        n += 1
    return n


def adapter_hashes(host: str | None) -> dict:
    """SHA-256 per adapter on the training host. Hashes, not 8.3 GB of weights."""
    if not host:
        return {"note": "not collected (--adapter-host not given)"}
    cmd = ("for d in ~/adapters/*/final; do "
           "[ -f \"$d/adapter_model.safetensors\" ] && "
           "echo \"$(basename $(dirname $d)) $(sha256sum $d/adapter_model.safetensors "
           "| cut -d' ' -f1)\"; done")
    try:
        out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                             timeout=900)
    except Exception as e:                      # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:300]}
    got = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            got[parts[0]] = parts[1]
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="archive directory to create")
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument("--adapter-host", default=None,
                    help="ssh host holding ~/adapters, to collect SHA-256s")
    ap.add_argument("--adapter-hashes", default=None, type=pathlib.Path,
                    help="JSON file of {adapter: sha256} collected earlier, for "
                         "when the training host is unreachable. Written back to "
                         "adapter_sha256.json on every successful collection so a "
                         "later build never depends on that host being up.")
    ap.add_argument("--tar", action="store_true", help="also write <out>.tar.gz")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    out = pathlib.Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into a non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    copied = {}
    for src_rel, dst_rel, required in CONTENTS:
        src = repo / src_rel
        if not src.exists():
            if required:
                sys.exit(f"missing required content: {src}")
            print(f"  skip (absent): {src_rel}")
            continue
        if src.is_file():
            target = out / dst_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            n = 1
        else:
            skip = ALREADY_PLACED if src_rel == "datagen/results" else None
            n = copy_tree(src, out / dst_rel, skip)
        copied[src_rel] = {"dest": dst_rel, "files": n}
        print(f"  {src_rel:42} -> {dst_rel:26} {n:5} files")

    print("\ncollecting adapter hashes...")
    adapters = adapter_hashes(args.adapter_host)
    bad = ("error" in adapters) or ("note" in adapters)
    if bad and args.adapter_hashes and args.adapter_hashes.exists():
        adapters = json.loads(args.adapter_hashes.read_text(encoding="utf-8"))
        print(f"  using {len(adapters)} hashes from {args.adapter_hashes}")
    elif not bad:
        side = pathlib.Path("adapter_sha256.json")
        side.write_text(json.dumps(adapters, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        print(f"  cached {len(adapters)} hashes to {side} for later builds")
    print(f"  {len(adapters)} adapters" if "error" not in adapters
          and "note" not in adapters else f"  {adapters}")

    # Manifest: every file in the archive, pinned.
    files = [p for p in sorted(out.rglob("*")) if p.is_file()]
    manifest = {
        "built": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
                        or "unknown",
        "repo_dirty": bool(subprocess.run(["git", "-C", str(repo), "status",
                                           "--porcelain"], capture_output=True,
                                          text=True).stdout.strip()),
        "contents": copied,
        "adapter_sha256": adapters,
        "files": {str(p.relative_to(out)): {"bytes": p.stat().st_size,
                                            "sha256": sha256(p)} for p in files},
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = ["# Structured-output repair and capability retention in "
              "Romanian-adapted LLMs",
              "",
              "Artefacts for the repair and retention study: the training corpus, "
              "the adapter configurations, every per-instance score, the corrected "
              "evaluator and the tables built from them. Everything here is data "
              "and analysis code — no model weights.",
              "",
              "The study fine-tunes eight Romanian-adapted checkpoints on a "
              "4,000-example structured-output corpus with three training seeds "
              "each, then measures what the repair gains on grounded tasks and "
              "what it costs on a standard academic suite.",
              "",
              f"Built {manifest['built']} from repo commit "
              f"{manifest['repo_commit'][:12]}"
              + (" (WORKING TREE DIRTY)" if manifest["repo_dirty"] else ""),
              "",
              "Every file is pinned by SHA-256 in `MANIFEST.json`.",
              "",
              "## Reproducing the corrected scores",
              "",
              "The corrected evaluator and the retained task cohorts are vendored "
              "under `rescoring/`, unmodified, with their own checksums. They need "
              "an exact environment, which `rescoring/NOTES.md` records:",
              "",
              "```",
              "conda create -y -p ./env python=3.13",
              "./env/bin/python -m pip install -r rescoring/requirements.txt",
              "./env/bin/python -X utf8 -m unittest discover -s rescoring/code "
              "-p \"test_*.py\"",
              "```",
              "",
              "Then, from `rescoring/`, `run_all.sh` and `run_abl.sh` rescore the "
              "saved generations in `scores/gmtw_ro/`, and `build_tables.py` and "
              "`paired.py` rebuild the corrected tables. No GPU and no model calls.",
              "",
              "## What is here", ""]
    for src_rel, info in copied.items():
        readme.append(f"- `{info['dest']}/` — {info['files']} files, from "
                      f"`{src_rel}`")
    readme += ["",
               "## Not here, and why", "",
               "- **Merged checkpoints.** Reconstructible from base + adapter; "
               "the merge is `datagen/scripts/merge_lora.py`.",
               "- **Base models.** Upstream artifacts, pinned by dated revision "
               "in the training configs.",
               f"- **Adapter weights.** ~8.3 GB; pinned by SHA-256 above "
               f"({len(adapters) if isinstance(adapters, dict) and 'error' not in adapters and 'note' not in adapters else 0} "
               "entries under `adapter_sha256`).",
               "",
               "## Regenerating each table", "",
               "Run from the repository root with the code tree on the path. "
               "Nothing here needs a GPU or makes a model call.", "",
               "```",
               "export PYTHONPATH=code",
               "python code/datagen/analysis/repair_tables.py \\",
               "    --analysis scores/gmtw_ro --split 500 --out tables",
               "```", "",
               "`code/` keeps the original package structure, because the scripts "
               "import one another: `holdout_overlap.py` reaches a `ro-so` sibling "
               "for `schema_fingerprint`, and `holdout_eval.py` imports "
               "`datagen.core`. Flattening it would ship scripts that cannot run.",
               "",
               "Substitute the published paths for the placeholders below: "
               "`<results>` is `scores/gmtw_ro`, `<tables>` is `tables`, `<sft>` is "
               "`corpus`, `<english>` is "
               "`eval/jsonschemabench/test_split.jsonl`. The retention tables read "
               "`retention/results` and write `retention/tables`.", ""]
    for pat, cmd in PROVENANCE.items():
        readme.append(f"- `{pat}` — `python code/datagen/analysis/{cmd}`")
    readme += ["",
               "## Authoritative notes a reader needs", "",
               "- `target_modules` in the training YAMLs is **dead config**. "
               "`datagen/training/train_lora.py` hardcodes the target pattern "
               "and passes it directly; the YAML key is never read. The regex in "
               "the trainer is authoritative.",
               "- The English evaluation set is **1,804 schemas**, of which "
               "**1,210** reject every trivial answer and **1,209** remain after "
               "removing the single exact duplicate of a training schema. A file "
               "of 3,608 records was this split concatenated with itself and is "
               "not scored.",
               "- That set is **in-distribution retention**, not a held-out "
               "source family: it is the upstream `test` split of the four "
               "trained sources, with zero instance-id overlap with training.",
               "- GMTW-Ro per-instance records come from the authoritative run "
               "only (`metrics_de_test_{en,ro,de}`, n=500; `metrics_hard_ro`, "
               "n=300). The `metrics_easy_*` directories are a superseded run.",
               "- Gemma-2 must be scored with `attn_implementation=eager`; the "
               "SDPA path silently skips logit softcapping.",
               ""]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # The README tells a reader to build the scoring environment at ./env.
    # Without this, following those instructions stages ~9,000 library files
    # into the artefact repository -- which is exactly how this project lost an
    # afternoon once already.
    (out / ".gitignore").write_text(
        "\n".join(["# the scoring environment the README tells you to build here",
                    "env/", ".venv/", ".cenv/",
                    "",
                    "# python caches",
                    "__pycache__/", "*.pyc",
                    "",
                    "# NOTE: nothing else is ignored. The manifest pins every",
                    "# file copied here, so an ignore rule that hides one would",
                    "# make the manifest describe a repository that does not",
                    "# exist. The rescoring run logs are kept for that reason.",
                    "",
                    "# --- re-include what an enclosing repository might exclude ---",
                    "# This archive is designed to be committed inside another",
                    "# repository, whose own .gitignore applies to it. gmtw-hria,",
                    "# for one, ignores `results/` and `test*.jsonl` at its root,",
                    "# which silently drops the academic-ladder raw output and",
                    "# test_split.jsonl -- the evaluation set the English result is",
                    "# scored on. A deeper .gitignore takes precedence, so these",
                    "# negations restore them. A directory has to be re-included",
                    "# before anything inside it can be.",
                    "!results/", "!results/**", "!test_split.jsonl", ""])
        + "\n", encoding="utf-8")

    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"\n{len(files)} files, {total / 2**20:.0f} MiB -> {out}")
    print(f"  MANIFEST.json + README.md written")

    if args.tar:
        tar = out.with_suffix(".tar.gz")
        subprocess.run(["tar", "-czf", str(tar), "-C", str(out.parent),
                        out.name], check=True)
        print(f"  tarball: {tar}  ({tar.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
