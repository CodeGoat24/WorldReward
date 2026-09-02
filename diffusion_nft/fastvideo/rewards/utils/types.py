from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardMetrics:
    action_acc: float = 0.0
    fine_action_acc: float = 0.0
    hpsv3_acc: float = 0.0
    hpsv3_quality_acc: float = 0.0
    hpsv3_quality_drift_score: float = 0.0
    mean_position_error: float | None = None
    mean_rotation_error_deg: float | None = None
    mean_step_direction_error_deg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_training_reward_dict(self) -> dict[str, float]:
        return {
            "action_reward": float(self.action_acc),
            "fine_action_reward": float(self.fine_action_acc),
            "hpsv3_reward": float(self.hpsv3_acc),
            "hpsv3_quality_reward": float(self.hpsv3_quality_acc),
            "hpsv3_quality_drift_reward": float(self.hpsv3_quality_drift_score),
        }


@dataclass
class RewardRequest:
    prompt: str | None = None
    video_path: str | None = None
    video_frames: Any | None = None
    gt_camera_pose: Any | None = None
    gt_action: Any | None = None
    interval: int = 1
    update_latent_num: int | None = None
    sample: Any | None = None
    frame_num: int | None = None
    action_num: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
