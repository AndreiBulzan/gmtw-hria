"""Declared normalization rules — the single source of truth.

Both the Track B alignment filter and the eval field matcher import from
here (CLAUDE.md rule 5: any new rule lands in this file AND the eval matcher
together — the matcher delegates to these functions, so one edit covers both).

Declared rules:
  R1  Unicode NFC + whitespace collapse before any comparison.
  R2  Cedilla diacritics (U+015F/U+0163) are treated as equal to comma-below
      (U+0219/U+021B) when *comparing*; emitted text must be comma-below.
  R3  Dates: gold ISO YYYY-MM-DD ↔ surface DD.MM.YYYY / D.M.YYYY /
      "15 martie 2026" / DD/MM/YYYY.
  R4  Numbers: gold 1234.56 ↔ surface "1.234,56" (dot thousands, decimal
      comma), "1234,56", "1234.56"; integral values also match without
      decimals ("1234", "1.234").
"""
from __future__ import annotations

import itertools
import re
import unicodedata

# ---------------------------------------------------------------- R1 + R2

_CEDILLA_TO_COMMA = str.maketrans({
    "ş": "ș",  # ş -> ș
    "Ş": "Ș",  # Ş -> Ș
    "ţ": "ț",  # ţ -> ț
    "Ţ": "Ț",  # Ţ -> Ț
})


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def fix_diacritics(s: str) -> str:
    """Rewrite cedilla ș/ț to comma-below. Use at every pipeline boundary."""
    return nfc(s).translate(_CEDILLA_TO_COMMA)


def norm_text(s: str) -> str:
    """Comparison form: NFC, comma-below diacritics, collapsed whitespace."""
    return re.sub(r"\s+", " ", fix_diacritics(s)).strip()


# -------------------------------------------------------------------- R3

RO_MONTHS = [
    "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie",
    "august", "septembrie", "octombrie", "noiembrie", "decembrie",
]
_MONTH_NUM = {m: i + 1 for i, m in enumerate(RO_MONTHS)}

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DMY_RE = re.compile(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$")
_LONG_RE = re.compile(r"^(\d{1,2})\s+([a-zăâîșț]+)\s+(\d{4})$", re.IGNORECASE)


def is_iso_date(s: str) -> bool:
    return bool(_ISO_RE.match(s))


def date_surface_forms(iso: str) -> list[str]:
    """All accepted Romanian surface forms of an ISO gold date."""
    m = _ISO_RE.match(iso)
    if not m:
        raise ValueError(f"not an ISO date: {iso!r}")
    y, mo, d = m.groups()
    month_name = RO_MONTHS[int(mo) - 1]
    return [
        iso,
        f"{d}.{mo}.{y}",
        f"{int(d)}.{int(mo)}.{y}",
        f"{d}/{mo}/{y}",
        f"{int(d)} {month_name} {y}",
        f"{d} {month_name} {y}",
    ]


def parse_ro_date(s: str) -> str | None:
    """Any declared surface form -> ISO, else None."""
    s = norm_text(s)
    if _ISO_RE.match(s):
        return s
    m = _DMY_RE.match(s)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    m = _LONG_RE.match(s)
    if m:
        d, name, y = m.groups()
        mo = _MONTH_NUM.get(name.lower())
        if mo:
            return f"{y}-{mo:02d}-{int(d):02d}"
    return None


# -------------------------------------------------------------------- R4

def format_ro_number(x: float | int) -> str:
    """1234.56 -> '1.234,56' (dot thousands, decimal comma)."""
    if isinstance(x, int) or float(x).is_integer():
        return f"{int(x):,}".replace(",", ".")
    return f"{x:,.2f}".translate(str.maketrans({",": ".", ".": ","}))


def number_surface_forms(x: float | int) -> list[str]:
    forms = [format_ro_number(x)]
    if isinstance(x, int) or float(x).is_integer():
        forms += [str(int(x))]
    else:
        forms += [f"{x:.2f}", f"{x:.2f}".replace(".", ",")]
    return forms


_NUM_CLEAN_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+([.,]\d+)?$")


def parse_ro_number(s: str) -> float | None:
    """Romanian or plain surface form -> float, else None."""
    s = norm_text(s).replace(" ", "")
    if not _NUM_CLEAN_RE.match(s):
        return None
    if "," in s:                       # 1.234,56 or 1234,56
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") > 1:             # 1.234.567 thousands only
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


# ------------------------------------------------- shared surface matching

def value_surface_forms(value) -> list[str]:
    """All comparison forms under which a gold leaf value may appear in text."""
    if isinstance(value, bool):
        return ["true", "da"] if value else ["false", "nu"]
    if isinstance(value, (int, float)):
        return number_surface_forms(value)
    if isinstance(value, str):
        if is_iso_date(value):
            return [norm_text(f) for f in date_surface_forms(value)]
        return [norm_text(value)]
    raise TypeError(f"unsupported gold leaf type: {type(value).__name__}")


def value_in_text(value, text: str) -> bool:
    """True if the gold value appears in text verbatim or via a declared rule."""
    hay = norm_text(text)
    return any(form in hay for form in value_surface_forms(value))


def values_match(gold, pred) -> bool:
    """Field-level equality for eval scoring. Deterministic, no LLM judge."""
    if isinstance(gold, bool) or isinstance(pred, bool):
        return gold is pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(gold) - float(pred)) < 1e-9
    if isinstance(gold, (int, float)) and isinstance(pred, str):
        p = parse_ro_number(pred)
        return p is not None and abs(float(gold) - p) < 1e-9
    if isinstance(gold, str) and isinstance(pred, str):
        if is_iso_date(gold):
            return parse_ro_date(pred) == gold
        return norm_text(gold) == norm_text(pred)
    if gold is None or pred is None:
        return gold is pred
    if isinstance(gold, list) and isinstance(pred, list):
        if len(gold) != len(pred):
            return False
        if all(values_match(g, p) for g, p in zip(gold, pred)):
            return True
        # Order-insensitive fallback: most JSON arrays in extraction tasks
        # are unordered collections of facts (attachments, quotes, line
        # items, ...), not meaningful sequences, and a model has no reason
        # to reproduce the gold's internal ordering. Try any permutation
        # before calling it a genuine mismatch. Bounded to avoid factorial
        # blowup on pathological inputs (8! = 40320, still instant).
        if len(gold) <= 8:
            return any(
                all(values_match(g, p) for g, p in zip(gold, perm))
                for perm in itertools.permutations(pred)
            )
        return False
    if isinstance(gold, dict) and isinstance(pred, dict):
        return set(gold) == set(pred) and all(
            values_match(gold[k], pred[k]) for k in gold
        )
    return False
