"""Validation of the reward model's JSON review payload.

A payload is only accepted if it is structurally complete: 4 action entries with
sequential ids and non-empty analyses, three visual sub-dimensions, and winners
drawn from the allowed vocabulary. Incomplete payloads are rejected rather than
partially salvaged so that a malformed generation never silently contributes a
vote.
"""
from __future__ import annotations

from typing import Any

WINNERS_WITH_TIE = frozenset({"A", "B", "Tie"})
WINNERS_NO_TIE = frozenset({"A", "B"})

VISUAL_DIMENSIONS = (
    "temporal_consistency",
    "dynamic_generation_quality",
    "artifacts_and_structure_integrity",
)

N_ACTION_SLOTS = 4


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_winner(value: Any, path: str, *, allow_tie: bool) -> str | None:
    allowed = WINNERS_WITH_TIE if allow_tie else WINNERS_NO_TIE
    if value not in allowed:
        return f"{path} must be one of {{{', '.join(sorted(allowed))}}}, got {value!r}"
    return None


def validate_review_payload(payload: Any) -> tuple[bool, str | None]:
    """Return ``(ok, error)``. ``error`` is ``None`` when ``ok`` is ``True``."""
    if not isinstance(payload, dict):
        return False, "payload must be a JSON object"
    if payload.get("error"):
        return False, f"error field present: {payload.get('error')}"
    if payload.get("raw_response"):
        return False, "raw_response present, parsed schema not accepted"

    action_control = payload.get("action_control")
    if not isinstance(action_control, dict):
        return False, "missing action_control object"

    actions = action_control.get("actions")
    if not isinstance(actions, list):
        return False, "action_control.actions must be a list"
    if len(actions) != N_ACTION_SLOTS:
        return False, f"action_control.actions must contain exactly {N_ACTION_SLOTS} items, got {len(actions)}"

    for index, action in enumerate(actions, start=1):
        path = f"action_control.actions[{index - 1}]"
        if not isinstance(action, dict):
            return False, f"{path} must be an object"
        if action.get("action_id") != index:
            return False, f"{path}.action_id must be {index}, got {action.get('action_id')!r}"
        for field in ("action_label", "video_a_analysis", "video_b_analysis"):
            if not _is_non_empty_string(action.get(field)):
                return False, f"{path}.{field} must be a non-empty string"
        error = _check_winner(action.get("winner"), f"{path}.winner", allow_tie=True)
        if error:
            return False, error

    if not _is_non_empty_string(action_control.get("overall_summary")):
        return False, "action_control.overall_summary must be a non-empty string"
    error = _check_winner(action_control.get("overall_winner"), "action_control.overall_winner", allow_tie=False)
    if error:
        return False, error

    visual_quality = payload.get("visual_quality")
    if not isinstance(visual_quality, dict):
        return False, "missing visual_quality object"

    for dimension in VISUAL_DIMENSIONS:
        path = f"visual_quality.{dimension}"
        value = visual_quality.get(dimension)
        if not isinstance(value, dict):
            return False, f"{path} must be an object"
        for field in ("video_a_analysis", "video_b_analysis"):
            if not _is_non_empty_string(value.get(field)):
                return False, f"{path}.{field} must be a non-empty string"
        error = _check_winner(value.get("winner"), f"{path}.winner", allow_tie=True)
        if error:
            return False, error

    if not _is_non_empty_string(visual_quality.get("overall_summary")):
        return False, "visual_quality.overall_summary must be a non-empty string"
    error = _check_winner(visual_quality.get("overall_winner"), "visual_quality.overall_winner", allow_tie=False)
    if error:
        return False, error

    return True, None
