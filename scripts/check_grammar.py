#!/usr/bin/env python3
"""
Grammar checker using LanguageTool - shows detailed errors.

Usage:
    python scripts/check_grammar.py data/outputs_8b.jsonl
    python scripts/check_grammar.py data/outputs_8b.jsonl --verbose
    python scripts/check_grammar.py data/outputs_8b.jsonl --instance recipe_000498
    python scripts/check_grammar.py data/outputs_8b.jsonl --min-errors 10
"""

import argparse
import json
import re
import sys

try:
    import language_tool_python
except ImportError:
    print("Error: language-tool-python is not installed.")
    print("Install with: pip install language-tool-python")
    print("Or: pip install rombench[grammar]")
    sys.exit(1)

SEVERITY_WEIGHTS = {
    "grammar": 3,
    "misspelling": 2,
    "typographical": 1,
    "style": 1,
}


def extract_clean_text(full_output: str) -> str:
    """Remove JSON block, keep only narrative text."""
    clean_text, _, _ = full_output.partition("{\n")
    return clean_text.strip()


def is_proper_noun_error(match, text: str) -> bool:
    """
    Check if the error is likely a false positive on a proper noun.

    Proper nouns (names, places) often get flagged as misspellings
    because they're not in the dictionary.
    """
    # Only filter MORFOLOGIK (spelling) errors
    if "MORFOLOGIK" not in match.rule_id:
        return False

    # Get the problematic word from context
    offset = match.offset
    length = match.errorLength

    if offset < len(text) and length > 0:
        word = text[offset:offset + length]
        # Check if word starts with uppercase (likely proper noun)
        if word and word[0].isupper():
            return True

    return False


def filter_matches(matches, text: str):
    """Filter out likely false positives (proper nouns, etc.)."""
    filtered = []
    skipped_proper_nouns = []

    for m in matches:
        if is_proper_noun_error(m, text):
            # Extract the word for reporting
            word = text[m.offset:m.offset + m.errorLength] if m.offset < len(text) else "?"
            skipped_proper_nouns.append(word)
        else:
            filtered.append(m)

    return filtered, skipped_proper_nouns


def compute_score(matches, word_count: int):
    if word_count == 0:
        return 0, 0, 100.0

    weighted_sum = sum(
        SEVERITY_WEIGHTS.get(m.rule_issue_type.lower(), 1)
        for m in matches
    )
    error_density = (weighted_sum / word_count) * 100
    score = max(0, 100 - error_density)

    return weighted_sum, error_density, score


def print_error_details(matches):
    """Print detailed info for each error."""
    for i, m in enumerate(matches, 1):
        issue_type = m.rule_issue_type
        weight = SEVERITY_WEIGHTS.get(issue_type.lower(), 1)

        print(f"  [{i}] {m.rule_id} ({issue_type}, weight={weight})")
        print(f"      Message: {m.message}")
        print(f"      Context: \"{m.context}\"")
        if m.replacements:
            suggestions = m.replacements[:3]  # Show max 3 suggestions
            print(f"      Suggestions: {suggestions}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Check grammar with LanguageTool")
    parser.add_argument("input_file", help="JSONL file with model outputs")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all error details for every instance")
    parser.add_argument("--instance", "-i", type=str,
                        help="Only check this specific instance ID")
    parser.add_argument("--min-errors", type=int, default=0,
                        help="Only show instances with at least N errors")
    parser.add_argument("--max", type=int,
                        help="Process at most N instances")
    parser.add_argument("--no-filter", action="store_true",
                        help="Don't filter out proper noun false positives")

    args = parser.parse_args()

    print("Loading LanguageTool (Romanian)...")
    tool = language_tool_python.LanguageTool("ro")

    print(f"Processing: {args.input_file}\n")

    count = 0
    total_skipped_proper_nouns = 0

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            instance_id = entry["instance_id"]

            # Filter by instance ID if specified
            if args.instance and instance_id != args.instance:
                continue

            output_text = entry["output"]
            clean_text = extract_clean_text(output_text)
            word_count = len(clean_text.split())

            all_matches = tool.check(clean_text)

            # Filter proper noun false positives unless --no-filter
            if args.no_filter:
                matches = all_matches
                skipped = []
            else:
                matches, skipped = filter_matches(all_matches, clean_text)
                total_skipped_proper_nouns += len(skipped)

            weighted_errors, density, score = compute_score(matches, word_count)

            # Filter by minimum errors
            if len(matches) < args.min_errors:
                continue

            print("=" * 60)
            print(f"Instance: {instance_id}")
            print(f"Words: {word_count} | Errors: {len(matches)} | Score: {score:.1f}/100")
            if skipped:
                print(f"(Skipped {len(skipped)} proper noun(s): {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})")
            print("=" * 60)

            if args.verbose or args.instance:
                print(f"\n--- Text ---\n{clean_text[:500]}{'...' if len(clean_text) > 500 else ''}\n")
                print(f"--- Errors ({len(matches)}) ---\n")
                print_error_details(matches)

            print()

            count += 1
            if args.max and count >= args.max:
                break

    print(f"Processed {count} instances.")
    if not args.no_filter:
        print(f"Total proper nouns skipped: {total_skipped_proper_nouns}")
    tool.close()


if __name__ == "__main__":
    main()
