"""
Base parser for dual-channel outputs (explanation + JSON)

This parser is language-agnostic - it extracts JSON and text
regardless of the language used.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Any

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


@dataclass
class ParseResult:
    """Result of parsing a dual-channel output"""
    plan: Optional[dict[str, Any]]
    explanation: str
    format_ok: bool
    repaired: bool
    error_message: str = ""


class BaseParser:
    """
    Language-agnostic parser for dual-channel outputs.
    
    Extracts:
    - JSON plan (structured task response)
    - Natural language explanation (free text before JSON)
    """

    def __init__(self):
        pass

    def parse(self, output: str) -> ParseResult:
        """
        Parse dual-channel output

        Args:
            output: Raw model output string

        Returns:
            ParseResult with extracted plan and explanation
        """
        # Try to extract JSON block
        json_str, json_start, json_end = self._extract_json_block(output)

        if json_str is None:
            return ParseResult(
                plan=None,
                explanation=output.strip(),
                format_ok=False,
                repaired=False,
                error_message="No JSON block found",
            )

        # Extract explanation (everything BEFORE the JSON block)
        explanation = output[:json_start].strip()

        # Try to parse JSON
        plan, repaired, error_msg = self._parse_json_with_repair(json_str)

        if plan is None:
            return ParseResult(
                plan=None,
                explanation=explanation,
                format_ok=False,
                repaired=False,
                error_message=error_msg,
            )

        return ParseResult(
            plan=plan,
            explanation=explanation,
            format_ok=not repaired,
            repaired=repaired,
        )

    def _extract_json_block(self, text: str) -> tuple[Optional[str], int, int]:
        """
        Extract JSON block from text

        Returns:
            (json_string, start_pos, end_pos) or (None, 0, 0) if not found
        """
        # Pre-process: replace Romanian-style and other non-standard quotes
        # with standard double quotes for JSON extraction.  We work on a
        # normalised copy so that start/end positions still map to text.
        normalised = text
        for ch in '\u201e\u201d\u201c\u00ab\u00bb\u201a\u2018\u2019':
            normalised = normalised.replace(ch, '"')

        # Try markdown code blocks first (with optional language tag)
        markdown_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(markdown_pattern, normalised, re.DOTALL)
        if match:
            return match.group(1), match.start(), match.end()

        # Also try markdown blocks that might have extra whitespace
        markdown_pattern2 = r"```\s*(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```"
        match = re.search(markdown_pattern2, normalised, re.DOTALL)
        if match:
            return match.group(1), match.start(), match.end()

        # Try naked JSON (find last complete brace pair)
        brace_count = 0
        last_open = -1
        matching_close = -1

        for i in range(len(normalised) - 1, -1, -1):
            if normalised[i] == '}':
                brace_count += 1
            elif normalised[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    last_open = i
                    break

        if last_open != -1:
            brace_count = 0
            for i in range(last_open, len(normalised)):
                if normalised[i] == '{':
                    brace_count += 1
                elif normalised[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        matching_close = i
                        break

            if matching_close != -1:
                json_str = normalised[last_open:matching_close + 1]
                return json_str, last_open, matching_close + 1

        return None, 0, 0

    def _parse_json_with_repair(self, json_str: str) -> tuple[Optional[dict], bool, str]:
        """
        Parse JSON with fallback to repair

        Returns:
            (parsed_dict, was_repaired, error_message)
        """
        # Stage 1: Strict parse
        try:
            plan = json.loads(json_str)
            return plan, False, ""
        except json.JSONDecodeError as e:
            strict_error = str(e)

        # Stage 2: Repair (if available)
        if HAS_JSON_REPAIR:
            try:
                repaired_str = repair_json(json_str)
                plan = json.loads(repaired_str)
                return plan, True, ""
            except Exception as e:
                repair_error = str(e)
        else:
            repair_error = "json-repair not available"

        # Stage 3: Fallback extraction (strip comments, fix trailing commas)
        plan = self._fallback_extraction(json_str)
        if plan is not None:
            return plan, True, ""

        return None, False, f"Parse failed: {strict_error}; Repair: {repair_error}"

    def _fallback_extraction(self, json_str: str) -> Optional[dict]:
        """
        Fallback extraction for simple JSON patterns.

        Handles:
        - Inline/block comments
        - Trailing commas
        - Romanian-style quotation marks
        """
        try:
            s = json_str
            # Remove comments
            s = re.sub(r'//.*?\n', '\n', s)
            s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
            # Fix trailing commas
            s = re.sub(r',\s*}', '}', s)
            s = re.sub(r',\s*]', ']', s)
            # Replace remaining non-standard quotes
            for ch in '\u201e\u201d\u201c\u00ab\u00bb\u201a\u2018\u2019':
                s = s.replace(ch, '"')
            return json.loads(s)
        except Exception:
            return None


def parse_dual_channel_output(output: str) -> ParseResult:
    """Convenience function for parsing"""
    parser = BaseParser()
    return parser.parse(output)