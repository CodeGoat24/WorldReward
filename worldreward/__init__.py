"""WorldReward -- reward modeling for camera-conditioned world models.

Typical use::

    from worldreward import render_pair, OfflineRunner, predict_pair

    chunks = render_pair(pair, output_root="rendered_chunks")
    runner = OfflineRunner(model_path="CodeGoat24/WorldReward-9B")
    records = runner.run(chunks)
    prediction = predict_pair(r["review_payload"] for r in records)
"""
from __future__ import annotations

from .actions import (
    CHUNK_SIZE,
    expand_action_expectation,
    normalize_slot_label,
    pretty_action_text,
)
from .aggregate import DIMENSIONS, collect_votes, majority, predict_pair, predict_pairs
from .parse import parse_response
from .prompt import PROMPT_VERSION, build_prompt
from .schema import validate_review_payload

__version__ = "0.1.0"

__all__ = [
    "CHUNK_SIZE",
    "DIMENSIONS",
    "PROMPT_VERSION",
    "__version__",
    "build_chunks",
    "build_prompt",
    "collect_votes",
    "expand_action_expectation",
    "majority",
    "normalize_slot_label",
    "parse_response",
    "predict_pair",
    "predict_pairs",
    "pretty_action_text",
    "render_pair",
    "validate_review_payload",
]


def __getattr__(name: str):
    # Heavy dependencies are imported on first use so that each stage of the
    # pipeline only needs its own requirements: preprocessing pulls in
    # matplotlib/OpenCV, inference pulls in vLLM, and scoring needs neither.
    if name in ("OfflineRunner", "ServerRunner"):
        from . import infer

        return getattr(infer, name)
    if name in ("build_chunks", "render_pair"):
        from . import render

        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
