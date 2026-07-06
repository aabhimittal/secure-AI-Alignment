"""Structured-output (JSON / XML) conformance and repair.

Backend systems that consume LLM output need it to parse *every time*. In
practice models wrap JSON in markdown fences, add a chatty preamble, emit
trailing commas, use single quotes, or forget to close a tag. This module:

1. **extracts** the structured payload from messy model output,
2. **repairs** the most common, safe-to-fix defects,
3. **validates** the result against an optional JSON Schema / required XML tags,

and reports parse-rate, repair-rate and schema-conformance so you can measure
how much a format guard buys you. Only the ``jsonschema`` package is optional;
without it, schema checks degrade to structural checks and everything else runs
on the standard library.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # optional, only needed for full JSON Schema validation
    import jsonschema  # type: ignore
    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAS_JSONSCHEMA = False


@dataclass
class FormatResult:
    raw: str
    parsed: Optional[Any]
    valid: bool
    repaired: bool
    repairs_applied: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "repaired": self.repaired,
            "repairs_applied": self.repairs_applied,
            "error": self.error,
        }


class FormatController:
    """Extract, repair and validate structured output."""

    # ---- JSON ---------------------------------------------------------- #

    @staticmethod
    def _extract_json_span(text: str) -> Optional[str]:
        """Pull the outermost JSON object/array out of noisy text."""
        # Strip markdown code fences first.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        # Find the first balanced { … } or [ … ].
        start = min(
            [i for i in (text.find("{"), text.find("[")) if i != -1],
            default=-1,
        )
        if start == -1:
            return None
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return text[start:]  # unbalanced; let the repairer try

    @staticmethod
    def _repair_json(span: str) -> (str, List[str]):
        repairs: List[str] = []
        fixed = span
        # single -> double quotes (only when no double quotes present)
        if "'" in fixed and '"' not in fixed:
            fixed = fixed.replace("'", '"')
            repairs.append("single_to_double_quotes")
        # trailing commas before } or ]
        new = re.sub(r",(\s*[}\]])", r"\1", fixed)
        if new != fixed:
            fixed = new
            repairs.append("trailing_comma")
        # Python literals -> JSON
        for py, js in ((r"\bTrue\b", "true"), (r"\bFalse\b", "false"), (r"\bNone\b", "null")):
            new = re.sub(py, js, fixed)
            if new != fixed:
                fixed = new
                repairs.append("python_literal")
        # balance one missing closing brace/bracket
        for opener, closer in (("{", "}"), ("[", "]")):
            diff = fixed.count(opener) - fixed.count(closer)
            if diff > 0:
                fixed += closer * diff
                repairs.append("balanced_brackets")
        return fixed, repairs

    def process_json(
        self, text: str, schema: Optional[Dict] = None
    ) -> FormatResult:
        span = self._extract_json_span(text)
        if span is None:
            return FormatResult(text, None, False, False, error="no JSON found")

        repaired = False
        repairs: List[str] = []
        parsed = None
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError:
            fixed, repairs = self._repair_json(span)
            try:
                parsed = json.loads(fixed)
                repaired = True
            except json.JSONDecodeError as e:
                return FormatResult(text, None, False, False, repairs, str(e))

        valid, err = self._validate_json(parsed, schema)
        return FormatResult(text, parsed, valid, repaired, repairs, err)

    @staticmethod
    def _validate_json(parsed: Any, schema: Optional[Dict]) -> (bool, Optional[str]):
        if schema is None:
            return True, None
        if _HAS_JSONSCHEMA:
            try:
                jsonschema.validate(parsed, schema)
                return True, None
            except jsonschema.ValidationError as e:  # type: ignore
                return False, e.message
        # Fallback: check required top-level keys only.
        required = schema.get("required", [])
        if isinstance(parsed, dict):
            missing = [k for k in required if k not in parsed]
            if missing:
                return False, f"missing required keys: {missing}"
            return True, None
        return False, "expected object"

    # ---- XML ----------------------------------------------------------- #

    @staticmethod
    def _extract_xml_span(text: str) -> Optional[str]:
        fence = re.search(r"```(?:xml)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        m = re.search(r"<[^>]+>.*</[^>]+>", text, re.DOTALL)
        return m.group(0) if m else None

    def process_xml(
        self, text: str, required_tags: Optional[List[str]] = None
    ) -> FormatResult:
        span = self._extract_xml_span(text)
        if span is None:
            return FormatResult(text, None, False, False, error="no XML found")
        repaired = False
        repairs: List[str] = []
        root = None
        try:
            root = ET.fromstring(span)
        except ET.ParseError:
            fixed = span.replace("&", "&amp;")  # unescaped ampersand is the #1 cause
            repairs.append("escape_ampersand")
            try:
                root = ET.fromstring(fixed)
                repaired = True
            except ET.ParseError as e:
                return FormatResult(text, None, False, False, repairs, str(e))

        valid, err = True, None
        if required_tags:
            present = {el.tag for el in root.iter()}
            missing = [t for t in required_tags if t not in present]
            if missing:
                valid, err = False, f"missing tags: {missing}"
        return FormatResult(text, root, valid, repaired, repairs, err)
