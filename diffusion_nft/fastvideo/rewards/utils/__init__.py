"""Shared reward-module utilities.

Public re-exports so callers write
    `from fastvideo.rewards.utils import exp_decay, RewardRequest, extract_frames_from_video`
instead of reaching into each submodule.
"""
from fastvideo.rewards.utils.frame_utils import (
    CV2_AVAILABLE,
    cleanup_frame_paths,
    extract_frames_from_video,
    new_instance_id,
)
from fastvideo.rewards.utils.metrics_to_reward import (
    NEUTRAL_REWARD,
    exp_decay,
    linear_scale,
)
from fastvideo.rewards.utils.types import RewardMetrics, RewardRequest

__all__ = [
    "CV2_AVAILABLE",
    "NEUTRAL_REWARD",
    "RewardMetrics",
    "RewardRequest",
    "cleanup_frame_paths",
    "exp_decay",
    "extract_frames_from_video",
    "linear_scale",
    "new_instance_id",
]
