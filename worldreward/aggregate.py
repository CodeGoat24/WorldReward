"""Aggregation of chunk-level reviews into one pair-level preference per dimension.

A pair is split into several chunks and the model reviews each chunk
independently. The three reported benchmark dimensions are recovered from the
review schema as follows:

=============  ==========================================================
Dimension      Votes
=============  ==========================================================
``action``     ``action_control.overall_winner`` per chunk; if that majority
               ties, break it with the majority over every per-action
               ``winner`` across all chunks.
``appearance`` ``visual_quality.temporal_consistency.winner`` and
               ``visual_quality.artifacts_and_structure_integrity.winner``
               pooled into a *single* vote list (2 votes per chunk). No
               tiebreak.
``motion``     ``visual_quality.dynamic_generation_quality.winner``. No
               tiebreak.
=============  ==========================================================

``A`` maps to ``left``, ``B`` to ``right``, and an unbroken tie to ``tie``.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

DIMENSIONS = ("action", "appearance", "motion")

_SIDE_BY_LETTER = {"A": "left", "B": "right", "Tie": "tie"}
_VALID_WINNERS = frozenset({"A", "B", "Tie"})


def majority(votes: Iterable[str]) -> str:
    """``"A"`` / ``"B"`` if one strictly leads, else ``"Tie"``."""
    votes = list(votes)
    a = votes.count("A")
    b = votes.count("B")
    if a > b:
        return "A"
    if b > a:
        return "B"
    return "Tie"


def _majority_with_tiebreak(primary: Sequence[str], tiebreak: Sequence[str] | None) -> str:
    winner = majority(primary)
    if winner != "Tie" or tiebreak is None:
        return winner
    fallback = majority(tiebreak)
    return fallback if fallback in ("A", "B") else "Tie"


def _letter_to_side(letter: str | None) -> str | None:
    return _SIDE_BY_LETTER.get(letter)


def collect_votes(chunk_reviews: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Gather raw vote lists from a pair's chunk review payloads.

    Args:
        chunk_reviews: the validated ``review_payload`` dicts of one pair.

    Returns:
        Keys ``action_overall`` / ``action_per_slot`` / ``appearance`` /
        ``motion``, each a list of ``"A"`` / ``"B"`` / ``"Tie"``.
    """
    votes: dict[str, list[str]] = {
        "action_overall": [],
        "action_per_slot": [],
        "appearance": [],
        "motion": [],
    }
    for payload in chunk_reviews:
        payload = payload or {}
        action_control = payload.get("action_control") or {}
        visual_quality = payload.get("visual_quality") or {}

        overall = action_control.get("overall_winner")
        if overall in _VALID_WINNERS:
            votes["action_overall"].append(overall)

        for action in action_control.get("actions") or []:
            winner = (action or {}).get("winner")
            if winner in _VALID_WINNERS:
                votes["action_per_slot"].append(winner)

        for sub_dimension in ("temporal_consistency", "artifacts_and_structure_integrity"):
            winner = (visual_quality.get(sub_dimension) or {}).get("winner")
            if winner in _VALID_WINNERS:
                votes["appearance"].append(winner)

        winner = (visual_quality.get("dynamic_generation_quality") or {}).get("winner")
        if winner in _VALID_WINNERS:
            votes["motion"].append(winner)
    return votes


def predict_pair(chunk_reviews: Iterable[dict[str, Any]]) -> dict[str, str | None]:
    """Aggregate a pair's chunk reviews into one ``left``/``right``/``tie``
    prediction per dimension. A dimension with no usable votes yields ``None``
    only if every chunk failed to parse; otherwise ties resolve to ``"tie"``."""
    votes = collect_votes(chunk_reviews)
    return {
        "action": _letter_to_side(
            _majority_with_tiebreak(votes["action_overall"], votes["action_per_slot"])
        ),
        "appearance": _letter_to_side(majority(votes["appearance"])),
        "motion": _letter_to_side(majority(votes["motion"])),
    }


def predict_pairs(chunk_records: Iterable[dict[str, Any]]) -> dict[str, dict[str, str | None]]:
    """Group flat chunk records by ``pair_id`` and aggregate each pair.

    Args:
        chunk_records: records as written by the inference runner, each with a
            ``pair_id`` and a ``review_payload``.
    """
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in chunk_records:
        by_pair.setdefault(record["pair_id"], []).append(record.get("review_payload") or {})
    return {pair_id: predict_pair(reviews) for pair_id, reviews in by_pair.items()}
