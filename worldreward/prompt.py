"""Prompt construction for WorldReward.

The prompt template lives in ``prompt_template.txt`` next to this module rather
than as a Python string literal. The template contains literal ``{`` / ``}``
braces (it embeds a JSON output schema), so it is filled by plain string
replacement of the two placeholders -- never by ``str.format``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .actions import expand_action_expectation

PROMPT_VERSION = "medium_v1"
_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt_template.txt"


@lru_cache(maxsize=1)
def prompt_template() -> str:
    """The raw SFT prompt template, with ``{caption}`` / ``{action_sequence}``
    placeholders still unfilled."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def build_action_sequence(action_labels: list[str]) -> str:
    """Render the PART I "Commanded action sequence" block.

    For each of the 4 slots this emits the bare label the model must copy back
    verbatim, followed by the indented expected-behaviour reference block.
    """
    lines: list[str] = []
    for index, label in enumerate(action_labels):
        lines.append(f"  Action {index + 1} label (copy this exactly): {label}")
        lines.extend(expand_action_expectation(label))
    return "\n".join(lines)


def build_prompt(caption: str, action_labels: list[str]) -> str:
    """Build the full evaluation prompt for one chunk.

    Args:
        caption: caption of the shared source image.
        action_labels: the 4 normalized slot labels of this chunk, each in
            ``"<Translation> | <Rotation>"`` form (see
            :func:`worldreward.actions.normalize_slot_label`).
    """
    action_sequence = build_action_sequence(list(action_labels))
    return (
        prompt_template()
        .replace("{caption}", caption)
        .replace("{action_sequence}", action_sequence)
    )
