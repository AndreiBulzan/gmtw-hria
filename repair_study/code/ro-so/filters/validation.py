"""Track A validation gate: raw model text -> parsed JSON -> schema check.

PASS is required for a pair to enter SFT data. FAIL is archived (prompt,
bad_output, validator_errors) for future DPO — never discarded silently,
never "fixed" by weakening the gate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from jsonschema import FormatChecker, validators

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$", re.MULTILINE)


def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ``` fences (with optional language tag)."""
    text = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def get_validator(schema: dict):
    """Validator class matching the schema's declared $schema draft
    (falls back to the library's latest supported draft)."""
    cls = validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, format_checker=FormatChecker())


@dataclass
class GateResult:
    passed: bool
    stage: str                 # "parse" | "schema" | "ok"
    output: object = None      # parsed JSON when parse succeeded
    errors: list[str] = field(default_factory=list)


def run_gate(schema: dict, raw_text: str) -> GateResult:
    cleaned = strip_markdown_fences(raw_text)
    try:
        instance = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return GateResult(False, "parse", errors=[f"json_parse_error: {e}"])

    try:
        validator = get_validator(schema)
    except Exception as e:  # unloadable/invalid schema: never train on it
        return GateResult(False, "schema", output=instance,
                          errors=[f"schema_load_error: {e}"])

    try:
        errs = [
            f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in validator.iter_errors(instance)
        ]
    except Exception as e:
        # Some malformed schemas survive check_schema and only break during
        # traversal -- e.g. a subschema whose draft-04 "id" is an object
        # instead of a string, which raises inside the resolver but ONLY once
        # an instance leads it into that branch. Instance-dependent, so it can
        # lie dormant for hundreds of records and then abort a whole run.
        #
        # Fail CLOSED: a schema we cannot validate against must never yield a
        # training pair. This is stricter than before, not weaker (hard rule 1)
        # -- the record is archived to data/rejected like any other failure.
        return GateResult(False, "schema", output=instance,
                          errors=[f"schema_validation_error: "
                                  f"{type(e).__name__}: {e}"])
    if errs:
        return GateResult(False, "schema", output=instance, errors=errs)
    return GateResult(True, "ok", output=instance)
