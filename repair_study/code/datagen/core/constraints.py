"""Programmatic constraint verifier for Track C (constraint-satisfaction).

Constraint specs are plain dicts stored in each record's meta, so eval can
re-verify outputs with this exact module — same checker for training gate
and scoring, no LLM judge.

Two families of kinds.

VALUE kinds read fields the model wrote itself:
  count_exact    {path, n}                len(array at path) == n
  sum_equals     {path, field, value}     sum(item[field]) == value
  date_ascending {path, field}            ISO dates non-decreasing in order
  pair_order     {path, field_a, field_b} item[field_a] <= item[field_b]
  enum_only      {path, field, allowed}   item[field] in allowed
  field_range    {path, field, min, max}  min <= item[field] <= max

SELECTION kinds read attributes from a catalogue the model was given, and
check ids it picked out of that catalogue. These need `catalogue`:
  sum_at_most    {path, attr, max}        sum of attr over picks <= max
  count_at_least {path, n}                at least n items picked
  attr_all       {path, attr, value}      every pick has attr == value
  attr_any       {path, attr, value}      at least one pick has attr == value
  attr_unique    {path, attr}             no two picks share an attr value
  valid_refs     {path}                   every picked id exists
  no_repeats     {path}                   no id picked twice
  attr_in_group  {path, attr, value,      picks with attr == value must all sit
                  group}                  in the group named `group`
  group_sums     {path, attr, group_a,    sum(attr) over group_a compared with
                  group_b, relation}      group_b ("gt", "ge", "lt", "le")

Why the second family exists: value kinds are satisfied by whatever numbers
the model invents, so a "total must equal X" rule costs nothing to obey.
Selection kinds are checked against fixed attributes the model does not
control, so a budget below the cost of everything available forces it to
actually choose. That is the shape of the rules it currently fails.

Grouped output (`[{"name": ..., "items": [ids]}]`) is addressed by adding
`item_key` to the spec; without it the array at `path` is a flat id list.
"""
from __future__ import annotations

_TOL = 1e-6
_RELATIONS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}


def _resolve(obj, path: str):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _items(output, spec):
    arr = _resolve(output, spec["path"])
    return arr if isinstance(arr, list) else None


def _ids(arr, spec):
    """Flat list of picked ids, from either output shape.

    Returns None if the shape doesn't match the spec, so the caller reports a
    shape problem rather than raising."""
    key = spec.get("item_key")
    if key is None:
        return arr if all(isinstance(x, str) for x in arr) else None
    out = []
    for group in arr:
        if not isinstance(group, dict) or not isinstance(group.get(key), list):
            return None
        out += group[key]
    return out if all(isinstance(x, str) for x in out) else None


def _attr(catalogue, iid, attr):
    """(found, value). Missing ids are not an error here — valid_refs is the
    kind that reports them, and every generated world pairs the two."""
    entry = (catalogue or {}).get(iid)
    if not isinstance(entry, dict) or attr not in entry:
        return False, None
    return True, entry[attr]


def _group_sum(arr, spec, catalogue, name):
    total, key, attr = 0.0, spec.get("item_key", "items"), spec["attr"]
    for group in arr:
        if isinstance(group, dict) and group.get("name") == name:
            for iid in group.get(key, []):
                found, v = _attr(catalogue, iid, attr)
                if found and isinstance(v, (int, float)):
                    total += v
    return total


def check_constraints(specs: list[dict], output, catalogue: dict | None = None
                      ) -> list[str]:
    """Empty list = all constraints satisfied.

    `catalogue` maps id -> attributes and is required by the selection kinds;
    value kinds ignore it, so old callers keep working unchanged."""
    violations = []
    for spec in specs:
        kind = spec["kind"]
        arr = _items(output, spec)
        if arr is None:
            violations.append(f"{kind}: no array at path {spec['path']!r}")
            continue

        if kind in ("sum_at_most", "count_at_least", "attr_all", "attr_any",
                    "attr_unique", "valid_refs", "no_repeats", "attr_in_group",
                    "group_sums"):
            picked = _ids(arr, spec)
            if picked is None:
                violations.append(
                    f"{kind}: {spec['path']!r} is not a list of ids"
                    + (f" under {spec['item_key']!r}" if spec.get("item_key")
                       else ""))
                continue

            if kind == "valid_refs":
                missing = [i for i in picked if i not in (catalogue or {})]
                if missing:
                    violations.append(
                        f"valid_refs: ids not in the catalogue: {missing}")

            elif kind == "no_repeats":
                dupes = sorted({i for i in picked if picked.count(i) > 1})
                if dupes:
                    violations.append(f"no_repeats: picked twice: {dupes}")

            elif kind == "count_at_least":
                # Deliberately pulls against sum_at_most: you must take enough
                # items that staying under the limit stops being trivial.
                if len(picked) < spec["n"]:
                    violations.append(
                        f"count_at_least: picked {len(picked)}, need at least "
                        f"{spec['n']}")

            elif kind == "sum_at_most":
                total = 0.0
                for iid in picked:
                    found, v = _attr(catalogue, iid, spec["attr"])
                    if found and isinstance(v, (int, float)):
                        total += v
                if total > spec["max"] + _TOL:
                    violations.append(
                        f"sum_at_most: sum({spec['attr']})={total:g} exceeds "
                        f"{spec['max']:g}")

            elif kind == "attr_all":
                bad = [i for i in picked
                       if _attr(catalogue, i, spec["attr"])[1] != spec["value"]]
                if bad:
                    violations.append(
                        f"attr_all: {spec['attr']}!={spec['value']!r} for {bad}")

            elif kind == "attr_any":
                if not any(_attr(catalogue, i, spec["attr"])[1] == spec["value"]
                           for i in picked):
                    violations.append(
                        f"attr_any: nothing picked with "
                        f"{spec['attr']}={spec['value']!r}")

            elif kind == "attr_unique":
                vals = [_attr(catalogue, i, spec["attr"])[1] for i in picked]
                clashes = sorted({str(v) for v in vals
                                  if v is not None and vals.count(v) > 1})
                if clashes:
                    violations.append(
                        f"attr_unique: {spec['attr']} repeated: {clashes}")

            elif kind == "attr_in_group":
                key = spec.get("item_key", "items")
                for group in arr:
                    if not isinstance(group, dict) or group.get("name") == spec["group"]:
                        continue
                    stray = [i for i in group.get(key, [])
                             if _attr(catalogue, i, spec["attr"])[1] == spec["value"]]
                    if stray:
                        violations.append(
                            f"attr_in_group: {stray} have "
                            f"{spec['attr']}={spec['value']!r} but sit in "
                            f"{group.get('name')!r}, not {spec['group']!r}")

            elif kind == "group_sums":
                a = _group_sum(arr, spec, catalogue, spec["group_a"])
                b = _group_sum(arr, spec, catalogue, spec["group_b"])
                rel = spec["relation"]
                ok = {"gt": a > b + _TOL, "ge": a >= b - _TOL,
                      "lt": a < b - _TOL, "le": a <= b + _TOL}.get(rel)
                if ok is None:
                    violations.append(f"group_sums: unknown relation {rel!r}")
                elif not ok:
                    violations.append(
                        f"group_sums: {spec['group_a']} ({a:g}) not "
                        f"{_RELATIONS[rel]} {spec['group_b']} ({b:g}) "
                        f"on {spec['attr']}")
            continue
        if kind == "count_exact":
            if len(arr) != spec["n"]:
                violations.append(
                    f"count_exact: {spec['path']} has {len(arr)} items, "
                    f"expected {spec['n']}")
        elif kind == "sum_equals":
            try:
                total = sum(float(it[spec["field"]]) for it in arr)
            except (KeyError, TypeError, ValueError):
                violations.append(f"sum_equals: unreadable {spec['field']}")
                continue
            if abs(total - spec["value"]) > _TOL:
                violations.append(
                    f"sum_equals: sum({spec['field']})={total}, "
                    f"expected {spec['value']}")
        elif kind == "date_ascending":
            vals = [it.get(spec["field"]) for it in arr]
            if any(not isinstance(v, str) for v in vals):
                violations.append(f"date_ascending: missing {spec['field']}")
            elif any(a > b for a, b in zip(vals, vals[1:])):
                violations.append(
                    f"date_ascending: {spec['field']} not sorted: {vals}")
        elif kind == "pair_order":
            for i, it in enumerate(arr):
                a, b = it.get(spec["field_a"]), it.get(spec["field_b"])
                if not (isinstance(a, str) and isinstance(b, str) and a <= b):
                    violations.append(
                        f"pair_order: item {i}: {spec['field_a']}={a!r} "
                        f"not <= {spec['field_b']}={b!r}")
        elif kind == "enum_only":
            for i, it in enumerate(arr):
                if it.get(spec["field"]) not in spec["allowed"]:
                    violations.append(
                        f"enum_only: item {i}: {it.get(spec['field'])!r} "
                        f"not in {spec['allowed']}")
        elif kind == "field_range":
            for i, it in enumerate(arr):
                v = it.get(spec["field"])
                if not isinstance(v, (int, float)) or isinstance(v, bool) \
                        or not spec["min"] <= v <= spec["max"]:
                    violations.append(
                        f"field_range: item {i}: {spec['field']}={v!r} "
                        f"outside [{spec['min']}, {spec['max']}]")
        else:
            violations.append(f"unknown constraint kind: {kind}")
    return violations
