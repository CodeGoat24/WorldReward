"""Registry of all available reward computer classes.

The dispatcher iterates this list to build only the rewards that are
enabled (by weight or eval flag) for a given training_args, and to
expose a canonical ordering of output keys for logging & advantage
computation.
"""
from __future__ import annotations

from fastvideo.rewards.base import BaseReward
from fastvideo.rewards.computers import (
    AestheticReward,
    HPSv3Reward,
    MonST3RReward,
    VLMPairwiseReward,
    WorldMirrorReward,
)

# Canonical order. Output keys land in sample_kwargs as
# f"{output}_reward" and in wandb as f"{output}_reward"; preserving a
# deterministic order keeps wandb panel layouts stable.
REWARD_CLASSES: tuple[type[BaseReward], ...] = (
    WorldMirrorReward,     # action, fine_action
    HPSv3Reward,           # hpsv3, hpsv3_quality, hpsv3_quality_drift
    VLMPairwiseReward,     # vlm_action, vlm_vq
    MonST3RReward,         # ate_rmse, rpe_trans, rpe_rot
    AestheticReward,       # aesthetic
)


def build_enabled_rewards(training_args) -> list[BaseReward]:
    """Instantiate all reward classes, keep those enabled (train OR eval)."""
    instances: list[BaseReward] = []
    for cls in REWARD_CLASSES:
        r = cls(training_args)
        if r.enabled:
            instances.append(r)
    return instances


def all_output_keys() -> tuple[str, ...]:
    """Flat tuple of every output declared by every registered class."""
    keys: list[str] = []
    for cls in REWARD_CLASSES:
        keys.extend(cls.OUTPUTS)
    return tuple(keys)
