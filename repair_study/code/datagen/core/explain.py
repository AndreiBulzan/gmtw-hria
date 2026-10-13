"""Render the Romanian prose that precedes the JSON in every completion.

The benchmark asks for an explanation first and the JSON block last. The
previous corpus emitted bare JSON and its prompts said "no explanations", so
the model was trained on one answer shape and measured on another. These
renderers close that gap.

**The explanation is derived from the answer, not written by a model.** That is
deliberate: an explanation computed from the object cannot contradict the
object, which is exactly the property the benchmark's faithfulness axis
measures. It also costs no extra generation call and needs no separate gate.

Two consequences to keep in view, both repeated in README.md:

  - naming the fields that are in the JSON is close to what the faithfulness
    metric rewards, so a gain there is partly by construction and has to be
    reported as such;
  - templated prose repeats. Each renderer draws from several phrasings; if the
    trained model still reads stereotyped, the fallback is teacher-written
    explanations gated on naming every value.

Track C does not live here — its explanation is a rule-by-rule verification
whose verdicts must come from the constraint checker, so it sits next to the
rules in `track_c.py`.

Two later corrections, both measured on trained models rather than guessed:

  - **Every explanation ends with a handoff line.** RoMistral-SFT seed 1 failed
    17% of the standard split with `No JSON block found` and no `{` anywhere in
    the output — not truncation, but a complete Romanian answer that simply
    stopped. The failures sat entirely in the two multi-constraint planning
    families, and the model was ending on our own Track C verification wording.
    Nothing in the old explanations marked where the prose stopped and the
    answer began except a blank line, so "end of explanation" was a plausible
    "end of turn". Seed 1 was the worst run in six of seven families on the
    identical mix, which is what a training-order-sensitive data defect looks
    like from the outside.

  - **Track A describes the fields instead of listing them.** At a median of 26
    words it was the shortest track and 55% of the mix, and trained models
    converged on ~46-word answers where the benchmark gives full length credit
    at 100. Describing each field from the schema is both longer and more
    informative, and it is what the teacher's own (discarded) prose does.
"""
from __future__ import annotations

import random

# Romanian joins the final item with "și", English with "and". A list of field
# names read as prose is the whole point -- "a, b și c" rather than a bare
# comma-separated dump, which reads like a database column listing.
_JOIN = {"ro": "și", "en": "and"}


def join_list(items: list[str], lang: str = "ro") -> str:
    """Human list: 'a', 'a și b', 'a, b și c'."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} {_JOIN[lang]} {items[-1]}"


# Romanian names for the JSON Schema primitives. Type names are described in
# Romanian; field names and enum values are NOT translated (CLAUDE.md rule 2),
# so they stay verbatim inside backticks/quotes.
_TYPE_NAMES = {
    "ro": {"string": "șir de caractere", "integer": "număr întreg",
           "number": "număr", "boolean": "valoare booleană",
           "array": "listă", "object": "obiect", "null": "valoare nulă"},
    "en": {"string": "string", "integer": "integer", "number": "number",
           "boolean": "boolean", "array": "array", "object": "object",
           "null": "null"},
}

# "listă de șir de caractere" is wrong Romanian; the item type has to agree in
# number. English needs no such table ("string array" is already correct).
_TYPE_PLURAL_RO = {"string": "șiruri de caractere", "integer": "numere întregi",
                   "number": "numere", "boolean": "valori booleene",
                   "array": "liste", "object": "obiecte",
                   "null": "valori nule"}

# Closing lines. The explanation must not simply trail off: see the module
# docstring for the failure this prevents. Several phrasings for the same
# reason the other renderers have several — one makes the prose stereotyped.
_HANDOFF = {
    "ro": ("Redau mai jos rezultatul în format JSON:",
           "Răspunsul în format JSON este următorul:",
           "Cu acestea stabilite, urmează obiectul JSON cerut:",
           "În continuare dau obiectul JSON propriu-zis:"),
    "en": ("The resulting JSON object follows:",
           "Here is the answer in JSON format:",
           "With that settled, the requested JSON object follows:",
           "The JSON object itself comes next:"),
}

# An object with dozens of properties would otherwise produce a wall of
# description far longer than the answer. Past this many, the rest are counted
# rather than named.
_MAX_DESCRIBED = 8


def handoff(rng: random.Random, lang: str = "ro") -> str:
    """The line that ends every explanation and hands over to the JSON."""
    choices = _HANDOFF[lang]
    return choices[rng.randrange(len(choices))]


def _describe_field(name: str, sub: dict, required: bool, lang: str) -> str:
    """One line about one top-level field, using only what the schema states.

    Returns None when the schema says nothing about the field. Saying "`x` is a
    field" adds words without adding information, and inventing a type would be
    a false statement in the training target — the same rule `_subject` follows
    for the object as a whole."""
    if not isinstance(sub, dict):
        return None
    bits = []
    names = _TYPE_NAMES[lang]
    typ = sub.get("type")
    if isinstance(typ, list):
        typ = next((x for x in typ if x != "null"), None)
    if isinstance(typ, str) and typ in names:
        if typ == "array":
            item_t = (sub.get("items") or {}).get("type") if isinstance(
                sub.get("items"), dict) else None
            if isinstance(item_t, str) and item_t in names:
                bits.append(f"{names['array']} de {_TYPE_PLURAL_RO[item_t]}"
                            if lang == "ro"
                            else f"{names[item_t]} {names['array']}")
            else:
                bits.append(names[typ])
        else:
            bits.append(names[typ])
    fmt = sub.get("format")
    if isinstance(fmt, str) and fmt:
        bits.append(f"în format {fmt}" if lang == "ro" else f"in {fmt} format")
    enum = sub.get("enum")
    if isinstance(enum, list) and enum:
        shown = ", ".join(f'"{v}"' for v in enum[:4])
        more = "" if len(enum) <= 4 else (" ș.a." if lang == "ro" else ", etc.")
        bits.append((f"una dintre valorile {shown}{more}" if lang == "ro"
                     else f"one of {shown}{more}"))
    lo, hi = sub.get("minimum"), sub.get("maximum")
    if lo is not None and hi is not None:
        bits.append(f"între {lo} și {hi}" if lang == "ro" else f"between {lo} and {hi}")
    elif lo is not None:
        bits.append(f"cel puțin {lo}" if lang == "ro" else f"at least {lo}")
    elif hi is not None:
        bits.append(f"cel mult {hi}" if lang == "ro" else f"at most {hi}")
    if not bits:
        return None
    tail = ""
    if required:
        tail = ", obligatoriu" if lang == "ro" else ", required"
    return f"`{name}` — {', '.join(bits)}{tail}."


def _field_block(obj: dict, schema: dict, lang: str) -> str:
    """The per-field description block, or "" when the schema describes none."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return ""
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    lines, described = [], 0
    for name in obj:
        if described >= _MAX_DESCRIBED:
            break
        line = _describe_field(name, props.get(name), name in required, lang)
        if line:
            lines.append(f"- {line}")
            described += 1
    if not lines:
        return ""
    left = len(obj) - described
    if left > 0:
        lines.append(f"- restul de {left} câmpuri urmează la fel schema."
                     if lang == "ro"
                     else f"- the remaining {left} fields follow the schema too.")
    return "\n".join(lines)


def _subject(schema: dict, lang: str) -> str:
    """What the object IS, taken from the schema when it says so.

    JSONSchemaBench schemas often carry a title or description; when one is
    present it is the only honest description of the object available to us.
    When it is not, say something neutral rather than inventing a domain -- a
    guessed subject is a false statement in the training target."""
    title = schema.get("title")
    if isinstance(title, str) and title.strip():
        t = title.strip()
        return (f"un obiect de tip „{t}”" if lang == "ro"
                else f"a {t} object")
    return ("un obiect conform schemei" if lang == "ro"
            else "an object matching the schema")


def explain_schema_object(obj, schema: dict, templates: list[str],
                          rng: random.Random, lang: str = "ro") -> str:
    """Track A: describe the generated object in terms of its top-level fields.

    Falls back to a bare sentence when the answer is not an object (a schema
    whose root is an array or a scalar), because "the top-level fields are ..."
    would be false there."""
    if not isinstance(obj, dict) or not obj:
        opening = ("Am generat o valoare care respectă schema cerută."
                   if lang == "ro"
                   else "I generated a value that satisfies the given schema.")
        return f"{opening}\n\n{handoff(rng, lang)}"
    fields = list(obj)
    opening = templates[rng.randrange(len(templates))].format(
        n=len(fields), fields=join_list(fields, lang),
        subject=_subject(schema, lang))
    parts = [opening]
    block = _field_block(obj, schema, lang)
    if block:
        parts.append(block)
    parts.append(handoff(rng, lang))
    return "\n\n".join(parts)


def explain_extraction(gold: dict, labels: dict, doc_label: str,
                       templates: list[str], rng: random.Random) -> str:
    """Track B: name the document and the fields extracted from it.

    Uses the field LABELS (Romanian) rather than the values. Listing the values
    back would make the explanation a second copy of the JSON, which teaches
    the model to pad rather than to explain -- and would inflate the
    faithfulness metric far harder than naming the fields does.

    The label -> key block names the schema key each label maps to. Keys are not
    translated (CLAUDE.md rule 2), so this is also the one place the Romanian
    prose and the English key names are put side by side."""
    named = [labels.get(k, k) for k in gold]
    opening = templates[rng.randrange(len(templates))].format(
        doc=doc_label, n=len(named), fields=join_list(named, "ro"))
    # Label alongside the key it lands under, so the prose says where each
    # value goes without repeating the value itself.
    block = "\n".join(f"- {labels.get(k, k)} -> `{k}`" for k in gold)
    return f"{opening}\n\n{block}\n\n{handoff(rng, 'ro')}"
