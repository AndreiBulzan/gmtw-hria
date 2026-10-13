"""Structural schema fingerprinting for train/eval dedupe.

Fingerprint = hash of the schema's structure: types, constraint kinds, and
child structures (sorted), with property NAMES, titles, descriptions and
concrete constraint VALUES ignored. Renaming keys must not change the
fingerprint; changing types or adding constraint kinds must."""
from __future__ import annotations

import hashlib
import json

_IGNORED = {"title", "description", "$id", "$comment", "examples", "default",
            "$schema", "definitions", "$defs"}
_CHILD_KEYS = {"properties", "patternProperties", "items", "prefixItems",
               "additionalProperties", "additionalItems", "allOf", "anyOf",
               "oneOf", "not", "if", "then", "else", "contains",
               "propertyNames", "unevaluatedProperties", "unevaluatedItems"}


def _canon(schema) -> object:
    if isinstance(schema, bool):
        return {"bool_schema": schema}
    if not isinstance(schema, dict):
        return {"invalid": True}

    constraint_kinds = sorted(
        k for k in schema if k not in _IGNORED and k not in _CHILD_KEYS
        and k != "type"
    )
    node: dict = {
        "type": sorted(schema["type"]) if isinstance(schema.get("type"), list)
        else schema.get("type"),
        "constraints": constraint_kinds,
    }
    children = []
    for key in sorted(schema.keys() & _CHILD_KEYS):
        sub = schema[key]
        if key in ("properties", "patternProperties"):
            # names ignored: sorted multiset of child structures
            kids = sorted(json.dumps(_canon(v), sort_keys=True)
                          for v in sub.values())
            children.append([key, kids])
        elif isinstance(sub, list):
            kids = sorted(json.dumps(_canon(v), sort_keys=True) for v in sub)
            children.append([key, kids])
        else:
            children.append([key, json.dumps(_canon(sub), sort_keys=True)])
    node["children"] = children
    return node


def schema_fingerprint(schema) -> str:
    canonical = json.dumps(_canon(schema), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
