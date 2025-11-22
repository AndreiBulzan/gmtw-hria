"""
Dual-channel output parser for GMTW-Ro

Extracts structured JSON plan and natural language explanation from model outputs.
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


class DualChannelParser:
    """Parser for dual-channel model outputs (explanation + JSON plan)"""

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

        # Extract explanation (everything before JSON)
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
        # Try to find JSON in markdown code blocks first
        markdown_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(markdown_pattern, text, re.DOTALL)
        if match:
            return match.group(1), match.start(), match.end()

        # Try to find naked JSON block (scan from end to get the last one)
        # Find the last { and matching }
        brace_count = 0
        last_open = -1
        matching_close = -1

        for i in range(len(text) - 1, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    last_open = i
                    break

        if last_open != -1:
            # Find the matching close brace
            brace_count = 0
            for i in range(last_open, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        matching_close = i
                        break

            if matching_close != -1:
                json_str = text[last_open:matching_close + 1]
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

        # Stage 3: Fallback extraction (for simple cases)
        plan = self._fallback_extraction(json_str)
        if plan is not None:
            return plan, True, ""

        # Failed all stages
        return None, False, f"Strict parse failed: {strict_error}. Repair failed: {repair_error}"

    def _fallback_extraction(self, json_str: str) -> Optional[dict]:
        """
        Fallback extraction for simple JSON patterns

        Only handles simple cases like:
        {"day1": ["A1", "A2"], "day2": ["A3"]}
        """
        try:
            # Remove comments (sometimes models add them)
            json_str = re.sub(r'//.*?\n', '\n', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

            # Try to fix common issues
            # Fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)

            # Try parsing again
            return json.loads(json_str)
        except:
            return None


def parse_dual_channel_output(output: str) -> ParseResult:
    """Convenience function for parsing dual-channel output"""
    parser = DualChannelParser()
    return parser.parse(output)
