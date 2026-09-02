"""Small pure-function helpers to convert raw metrics into [0, 1] rewards.

The conventions:
  - Lower-is-better errors (ATE/RPE): exp_decay so that 0 error → 1.0.
  - Unbounded-above scores (aesthetic 0-10): linear_scale clamped to [0, 1].

All helpers treat NaN as a signal of scorer failure and return a neutral
0.5 so GRPO's advantage normalization doesn't blow up.
"""
from __future__ import annotations

import math


NEUTRAL_REWARD = 0.5


def exp_decay(value: float, scale: float) -> float:
    """reward = exp(-value / scale). value=0 → 1, value=scale → 0.37.

    Returns NEUTRAL_REWARD on NaN / non-finite input.
    """
    if value is None or not math.isfinite(value):
        return NEUTRAL_REWARD
    if scale <= 0:
        # Degenerate config; treat as neutral rather than divide.
        return NEUTRAL_REWARD
    # Clamp to a reasonable range: if the metric is huge, exp(-x/scale) can
    # underflow to 0.0, which is actually fine for "bad" but we guard
    # against negative value (shouldn't happen) by clamping to 0.
    value = max(0.0, float(value))
    r = math.exp(-value / scale)
    # Protect against pathological scale where r > 1 (can't happen for
    # value >= 0, but keep the invariant explicit).
    return max(0.0, min(1.0, r))


def linear_scale(value: float, scale: float) -> float:
    """reward = clamp(value / scale, 0, 1). Returns NEUTRAL on NaN."""
    if value is None or not math.isfinite(value):
        return NEUTRAL_REWARD
    if scale <= 0:
        return NEUTRAL_REWARD
    return max(0.0, min(1.0, float(value) / scale))
