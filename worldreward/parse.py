"""Parsing of the reward model's raw text generation into a validated payload.

The model is instructed to emit a bare JSON object, but generations occasionally
arrive wrapped in markdown fences or nested inside an outer object. The parser
therefore: strips fences, falls back to a greedy brace match, then searches the
parsed structure for the ``action_control`` / ``visual_quality`` pair before
validating it.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import validate_review_payload


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```/```json fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_object(text: str) -> Any:
    """Parse JSON from a possibly-fenced, possibly-prefixed generation."""
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("no JSON object found in response")


def find_review_payload(obj: Any) -> dict[str, Any] | None:
    """Recursively locate the ``{action_control, visual_quality}`` pair."""
    if isinstance(obj, dict):
        if "action_control" in obj and "visual_quality" in obj:
            return {
                "action_control": obj.get("action_control"),
                "visual_quality": obj.get("visual_quality"),
            }
        for value in obj.values():
            found = find_review_payload(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_review_payload(item)
            if found is not None:
                return found
    return None


def parse_response(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse + validate one generation.

    Returns ``(payload, error)``. Exactly one of the two is non-``None``.
    """
    try:
        parsed = parse_json_object(raw_text)
    except Exception as exc:
        return None, f"json_parse_error: {exc}"

    payload = find_review_payload(parsed)
    if payload is None:
        return None, "missing_action_control_or_visual_quality"

    ok, error = validate_review_payload(payload)
    if not ok:
        return None, error or "invalid_review_payload"
    return payload, None
