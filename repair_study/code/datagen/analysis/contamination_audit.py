#!/usr/bin/env python3
"""Evidence that the repair corpus is separate from the evaluation set.

The review's sharpest methodological point is that the repair data resemble the
benchmark closely enough that near-perfect JSON afterwards might be benchmark
adaptation rather than restored competence, and that the paper never
demonstrates the two are disjoint. This measures the separation instead of
asserting it.

Four checks, all deterministic and none needing a GPU:

  domains    Track C generates worlds in its own domains. If any of them is a
             GMTW-Ro family, the constraint half of the corpus is training on
             the evaluation task rather than a cousin of it.

  entities   The identifiers a plan is scored against. Overlap here would be
             the most direct contamination available: the same entity in a
             training world and an evaluation world.

  ngrams     Verbatim text shared between training instructions and evaluation
             prompts. Catches template reuse that the structural checks miss --
             the two corpora are built by different generators, so a long
             shared span is evidence of a common ancestor rather than chance.

  schemas    Structural fingerprints of the Track A schemas against each other,
             for the exact-duplicate and structural-twin rates. Note what this
             is NOT: JSONSchemaBench scores schema VALIDITY, so there is no
             reference answer to memorise and regurgitating a memorised
             completion still has to validate. Overlap there is not the same
             kind of problem as overlap in a match-to-reference benchmark.

  python datagen/analysis/contamination_audit.py \\
      --corpus datagen/data/sft --eval ../gmtw-hria/data --out results/tables
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys
import unicodedata

NGRAM_N = 12          # long enough that a collision is not chance in Romanian
CORPUS_FILES = ["track_a.jsonl", "track_b_invoice.jsonl", "track_b_contract.jsonl",
                "track_b_anunt.jsonl", "track_c.jsonl", "english_structured.jsonl"]
EVAL_FILES = ["gmtw_ro_v0.jsonl", "gmtw_ro_hard.jsonl"]


def read_jsonl(path: pathlib.Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def norm(text: str) -> str:
    """NFC + case + whitespace, so a collision is not missed over spacing."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def ngrams(text: str, n: int = NGRAM_N):
    words = norm(text).split()
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def corpus_records(corpus: pathlib.Path):
    for name in CORPUS_FILES:
        p = corpus / name
        if p.exists():
            for r in read_jsonl(p):
                yield name, r


def eval_records(evaldir: pathlib.Path):
    for name in EVAL_FILES:
        p = evaldir / name
        if p.exists():
            for r in read_jsonl(p):
                yield name, r


def family(instance_id: str) -> str:
    return instance_id.rsplit("_", 1)[0]


def audit(corpus: pathlib.Path, evaldir: pathlib.Path) -> list[list]:
    rows = []

    # ---- domains -----------------------------------------------------------
    train_domains, train_seeds = set(), collections.Counter()
    for _, r in corpus_records(corpus):
        if r.get("track") == "C":
            train_domains.add(r["meta"]["domain"])
        s = r["meta"].get("seed")
        if s is not None:
            train_seeds[s] += 1
    eval_families = {family(r["instance_id"]) for _, r in eval_records(evaldir)}
    shared = sorted(train_domains & eval_families)
    rows.append(["domains", "train_domains", ",".join(sorted(train_domains))])
    rows.append(["domains", "eval_families", ",".join(sorted(eval_families))])
    rows.append(["domains", "shared", ",".join(shared) if shared else "none"])
    rows.append(["domains", "n_shared", len(shared)])

    # ---- entity identifiers ------------------------------------------------
    train_ents = set()
    for _, r in corpus_records(corpus):
        cat = r["meta"].get("catalogue")
        if isinstance(cat, dict):
            train_ents |= set(cat)
    eval_ents = set()
    for _, r in eval_records(evaldir):
        w = r.get("world") or {}
        ents = w.get("entities") or w.get("canonical_entities") or {}
        if isinstance(ents, dict):
            eval_ents |= set(ents)
        elif isinstance(ents, list):
            eval_ents |= {e.get("id") for e in ents if isinstance(e, dict) and e.get("id")}
    both = train_ents & eval_ents
    rows.append(["entities", "train_unique", len(train_ents)])
    rows.append(["entities", "eval_unique", len(eval_ents)])
    rows.append(["entities", "shared", len(both)])
    rows.append(["entities", "shared_examples",
                 ",".join(sorted(both)[:5]) if both else "none"])
    # Report the namespaces too. Zero overlap here follows from the domains
    # being disjoint -- each generator prefixes its ids by domain -- so this is
    # a check that would catch a mistake, not independent evidence on top of
    # the domain result. Saying so is more useful than a bare zero.
    rows.append(["entities", "train_prefixes",
                 ",".join(sorted({e[0] for e in train_ents})) or "none"])
    rows.append(["entities", "eval_prefixes",
                 ",".join(sorted({e[0] for e in eval_ents})) or "none"])

    # ---- verbatim n-gram overlap ------------------------------------------
    eval_ngrams = set()
    for _, r in eval_records(evaldir):
        for field in ("prompt", "instruction", "prompt_ro", "text"):
            if isinstance(r.get(field), str):
                eval_ngrams |= ngrams(r[field])
    hits, checked = 0, 0
    per_file = collections.Counter()
    for name, r in corpus_records(corpus):
        text = r.get("instruction", "")
        if not text:
            continue
        checked += 1
        if ngrams(text) & eval_ngrams:
            hits += 1
            per_file[name] += 1
    rows.append(["ngrams", "n", NGRAM_N])
    rows.append(["ngrams", "eval_ngrams", len(eval_ngrams)])
    rows.append(["ngrams", "train_records_checked", checked])
    rows.append(["ngrams", "train_records_with_overlap", hits])
    rows.append(["ngrams", "pct", round(100 * hits / max(checked, 1), 3)])
    rows.append(["ngrams", "by_file",
                 ";".join(f"{k}={v}" for k, v in per_file.most_common()) or "none"])

    # ---- schema fingerprints ----------------------------------------------
    fps = collections.Counter()
    for _, r in corpus_records(corpus):
        fp = r["meta"].get("schema_fingerprint")
        if fp:
            fps[fp] += 1
    rows.append(["schemas", "records_with_fingerprint", sum(fps.values())])
    rows.append(["schemas", "unique_structures", len(fps)])
    rows.append(["schemas", "structures_used_once", sum(1 for v in fps.values() if v == 1)])
    rows.append(["schemas", "largest_structural_group", max(fps.values()) if fps else 0])

    # ---- provenance --------------------------------------------------------
    rows.append(["provenance", "generator_seeds",
                 ",".join(str(s) for s in sorted(train_seeds))])
    for man in sorted(corpus.glob("*.manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        rows.append(["provenance", f"{man.name}:seed", m.get("seed")])
        rows.append(["provenance", f"{man.name}:total", m.get("total")])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="directory holding the track jsonl files")
    ap.add_argument("--eval", required=True, dest="evaldir",
                    help="benchmark data directory (gmtw_ro_v0.jsonl, gmtw_ro_hard.jsonl)")
    ap.add_argument("--out", default=None, help="also write 45_train_eval_separation.csv here")
    args = ap.parse_args()

    corpus, evaldir = pathlib.Path(args.corpus), pathlib.Path(args.evaldir)
    for p in (corpus, evaldir):
        if not p.is_dir():
            sys.exit(f"not a directory: {p}")

    rows = audit(corpus, evaldir)
    width = max(len(str(r[1])) for r in rows)
    section = None
    for check, key, *rest in rows:
        if check != section:
            print(f"\n=== {check} ===")
            section = check
        print(f"  {str(key):<{width}}  {rest[0]}")

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "45_train_eval_separation.csv", "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["check", "key", "value"])
            w.writerows(rows)
        print(f"\nwrote {out / '45_train_eval_separation.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
