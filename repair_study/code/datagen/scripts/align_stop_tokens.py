#!/usr/bin/env python3
"""
Align a merged checkpoint's eos_token_id with the turn marker its own chat
template emits.

This is the same logic run_family.sh runs inline after every merge, lifted into
a file so a second runner can apply it without editing the script that produced
every checkpoint already in the study. The two must stay in step; if one is
changed, change both, because a checkpoint processed by only one of them is not
comparable to the rest of the table.

No-op when the template's marker is already covered, which is the Llama case --
and the run says so out loud rather than staying silent, because "nothing to do"
and "never ran" look identical in a log otherwise.
"""

import json
import pathlib
import sys

from transformers import AutoTokenizer


def align(model_dir: str) -> bool:
    d = pathlib.Path(model_dir)
    gcf = d / "generation_config.json"
    tok = AutoTokenizer.from_pretrained(str(d))
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False)
    # The LAST token is often the newline after the turn marker (Gemma renders
    # "...<end_of_turn>\n", where \n is id 108). Stopping on that would truncate
    # every answer at its first line break -- worse than the bug being fixed.
    # Take the last SPECIAL token, which is the real turn marker.
    ids = tok.encode(rendered, add_special_tokens=False)
    special = set(tok.all_special_ids) | set(
        tok.convert_tokens_to_ids(t) for t in getattr(tok, "additional_special_tokens", []))
    marker = [i for i in ids if i in special]
    if not marker:
        print("  stop token: no special token in the rendered template, leaving as-is")
        return False
    last = marker[-1]
    g = json.loads(gcf.read_text())
    cur = g.get("eos_token_id")
    allowed = cur if isinstance(cur, list) else [cur]
    if last not in allowed:
        g["eos_token_id"] = sorted({*[a for a in allowed if a is not None], last})
        gcf.write_text(json.dumps(g, indent=2) + "\n")
        print("  stop token: %s -> %s (template ends with id %s)" % (cur, g["eos_token_id"], last))
        return True
    print("  stop token: %s already covers the template id %s" % (cur, last))
    return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: align_stop_tokens.py <merged-model-dir>")
    align(sys.argv[1])
