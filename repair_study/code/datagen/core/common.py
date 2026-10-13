"""Shared pipeline utilities: template loading, NFC-clean JSONL I/O,
record construction per the dataset record format in CLAUDE.md."""
from __future__ import annotations

import hashlib
import json
import pathlib

from datagen.core.normalization import fix_diacritics

ROOT = pathlib.Path(__file__).resolve().parent.parent


def resolve_template(path: str | pathlib.Path) -> pathlib.Path:
    """Accept a template path either as given or relative to this package.

    Every usage line in the docs writes `templates/instructions_en.txt`, which
    only resolves when the shell happens to be sitting in datagen/. Callers run
    from the repo root, so the bare form has to work too — otherwise the
    documented command fails and the failure looks like a missing file rather
    than a wrong working directory."""
    p = pathlib.Path(path)
    if p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"{path!r} not found, and neither is {candidate} — pass a path that "
        f"exists, or one relative to {ROOT}")


def resolve_data(path: str | pathlib.Path) -> pathlib.Path:
    """Locate a corpus file named in a training config.

    The mix lists paths like `data/sft/track_a.jsonl`. Those are written by the
    track scripts relative to wherever the operator ran them -- normally the
    repo root -- so the working directory is checked first. The package
    directory is a fallback for a self-contained checkout that keeps its corpus
    inside datagen/.

    Resolving only against the package (which is what this did when it lived in
    ro-so/, where ROOT happened to be the repo) silently looked in the wrong
    place once the folder moved, and reported it as a missing file rather than
    a missing directory."""
    p = pathlib.Path(path)
    if p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"{path!r} not found from {pathlib.Path.cwd()}, nor at {candidate}. "
        f"Generate that track first, or run from the directory the corpus "
        f"paths in your config are relative to.")


def load_templates(path: str | pathlib.Path) -> list[str]:
    """Templates separated by lines containing only '---'; '#' header
    comment lines before the first template are dropped."""
    text = resolve_template(path).read_text(encoding="utf-8")
    blocks, current = [], []
    for line in text.splitlines():
        if line.strip() == "---":
            blocks.append("\n".join(current).strip())
            current = []
        elif line.startswith("#") and not blocks and not current:
            continue
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    templates = [fix_diacritics(b) for b in blocks if b]
    if not templates:
        raise ValueError(f"no templates found in {path}")
    return templates


def load_labelled_templates(path: str | pathlib.Path) -> dict[str, str]:
    """Templates keyed by a `[label]` line, for variants chosen by name.

    `load_templates` returns an unordered pool to draw from at random. These
    are picked deliberately — one per experiment arm — so positional indexing
    would let a reordered file silently swap which arm got which wording, and
    the comparison between arms is the whole point.
    """
    blocks: dict[str, list[str]] = {}
    label = None
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            label = stripped[1:-1]
            blocks[label] = []
        elif label is None:
            continue                      # header comments before block one
        else:
            blocks[label].append(line)
    out = {k: fix_diacritics("\n".join(v).strip()) for k, v in blocks.items()}
    out = {k: v for k, v in out.items() if v}
    if not out:
        raise ValueError(f"no labelled templates found in {path}")
    return out


def _deep_nfc(obj):
    """NFC + comma-below diacritics on every string, at pipeline boundaries."""
    if isinstance(obj, str):
        return fix_diacritics(obj)
    if isinstance(obj, list):
        return [_deep_nfc(v) for v in obj]
    if isinstance(obj, dict):
        return {_deep_nfc(k): _deep_nfc(v) for k, v in obj.items()}
    return obj


def append_jsonl(path: str | pathlib.Path, record: dict) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_deep_nfc(record), ensure_ascii=False) + "\n")


def read_jsonl(path: str | pathlib.Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_first_json(text: str) -> str | None:
    """First balanced JSON object/array in text, or None.

    Pipeline-side rescue for generator preamble ("Iată JSON-ul: {...}") —
    the training target is the parsed JSON anyway, so trimming prose around
    it doesn't weaken the gate: the candidate must still parse AND validate.
    Never used by the eval matcher, whose parse-rate metric stays strict.
    """
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1),
                default=None)
    if start is None:
        return None
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def approx_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate (~4 chars/token for JSON/ASCII text);
    used only for the Track A length policy thresholds, which are themselves
    approximate ("~2k"/"~4k" in CLAUDE.md)."""
    return max(1, len(text) // 4)


def record_id(track: str, source: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{track.lower()}-{source}-{h}"


def make_record(*, rid: str, track: str, instruction: str, schema: dict,
                output, explanation: str, document: str | None = None,
                meta: dict) -> dict:
    """One training record. `explanation` is the Romanian prose that precedes
    the JSON in the completion, and it is required on every track.

    This is the change the rebuild exists for. The benchmark asks for an
    explanation first and the JSON block last; the previous corpus emitted bare
    JSON everywhere except Track C, and its prompts actively said "no
    explanations". So we trained one answer shape and measured another.

    The explanation is rendered from the answer the record already carries, not
    written by a second model call, so it cannot contradict the JSON. See
    README.md for the two caveats that follow from that choice."""
    if not explanation or not explanation.strip():
        raise ValueError(f"{rid}: empty explanation — every track must render "
                         f"one, or the completion is the wrong shape")
    rec = {"id": rid, "track": track, "instruction": instruction,
           "schema": schema}
    if document is not None:
        rec["document"] = document
    rec["explanation"] = explanation
    rec["output"] = output
    rec["meta"] = meta
    return rec


def completion_text(rec: dict) -> str:
    """The supervised target: explanation, blank line, JSON block.

    One definition, used by the trainer and by any check that wants to see what
    the model is actually taught. Keeping it here stops the ordering from being
    re-derived (and quietly inverted) at the call site — the benchmark's parser
    treats the text before the first JSON block as the explanation, so a
    completion built the other way round scores zero on faithfulness."""
    return (rec["explanation"].strip() + "\n\n"
            + json.dumps(rec["output"], ensure_ascii=False, indent=2))
