"""Track B alignment filter: every gold leaf value must appear in the
generated document verbatim or via a declared normalization rule
(filters/normalization.py). Any miss -> reject the DOCUMENT and regenerate;
the gold is never patched (misaligned pairs teach hallucination)."""
from __future__ import annotations

from dataclasses import dataclass, field

from filters.normalization import value_in_text


def flatten_gold(obj, prefix: str = "") -> list[tuple[str, object]]:
    """(json-path, leaf) pairs; None leaves are skipped (nothing to align)."""
    out: list[tuple[str, object]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten_gold(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += flatten_gold(v, f"{prefix}[{i}]")
    elif obj is not None:
        out.append((prefix, obj))
    return out


@dataclass
class AlignmentResult:
    passed: bool
    misses: list[tuple[str, object]] = field(default_factory=list)


def check_alignment(gold: dict, document: str) -> AlignmentResult:
    misses = [
        (path, value)
        for path, value in flatten_gold(gold)
        if not value_in_text(value, document)
    ]
    return AlignmentResult(passed=not misses, misses=misses)
