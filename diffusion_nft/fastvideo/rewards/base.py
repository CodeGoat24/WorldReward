"""Base classes for the unified reward interface.

Design:
  - One BaseReward subclass per logical reward source (MonST3R, aesthetic,
    WorldMirror, VLM pairwise, ...).
  - A subclass can emit MULTIPLE named reward outputs (e.g. MonST3R emits
    ate_rmse, rpe_trans, rpe_rot).
  - Scoring happens in two flavors:
      * PerCandidateReward.score_one(ctx) — independent per candidate
        (default; most rewards).
      * GroupReward.score_group(ctxs) — needs to see all candidates at
        once (VLM pairwise).

  - The dispatcher is responsible for:
      * Constructing + loading only enabled rewards.
      * Calling score_one / score_group and stuffing outputs into
        sample_kwargs under the canonical key `f"{output}_reward"`.
      * Computing per-reward advantages (z-score in GPU group), weighted
        sum across rewards, and the per-reward mean for logging.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class RewardContext:
    """Everything a reward computer might need about ONE candidate.

    Kept wide so subclasses can consume whatever subset they need. Unused
    fields stay None and are ignored. Tensors are left on their source
    device (CPU for gt_w2c / gt_action, as passed by rollout.py).
    """
    candidate_idx: int
    video_path: str                       # absolute path to saved mp4
    video_frames: np.ndarray | None = None  # (T, H, W, 3) uint8, optional
    prompt: str | None = None
    caption: str | None = None
    gt_w2c: torch.Tensor | np.ndarray | None = None     # [latent_t, 4, 4]
    gt_action: torch.Tensor | np.ndarray | None = None
    chunk_id: int = 0
    update_latent_num: int = 4
    latent_t: int = 32
    action_labels: list[int] | None = None
    action_texts: list[str] | None = None
    num_replicas: int = 1
    # Freeform extras populated by rollout (e.g. worldmirror's metric dict
    # for filename rendering).
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """Per-candidate result from one reward's score call.

    scores: dict of {output_name: reward ∈ [0,1]}; every key declared in
        cls.OUTPUTS must be present (fill NEUTRAL_REWARD 0.5 on failure).
    metrics: raw metric values for logging / filename rendering (not used
        for gradient). Optional.
    """
    scores: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)


class BaseReward(ABC):
    """Abstract base.

    Subclasses MUST set:
      NAME: str          — e.g. "monst3r". Used in logs.
      OUTPUTS: tuple[str, ...] — e.g. ("ate_rmse", "rpe_trans", "rpe_rot").
        Each output becomes a sample_kwargs key `f"{output}_reward"` and
        contributes one weighted term to total advantages.
    """
    NAME: str = ""
    OUTPUTS: tuple[str, ...] = ()

    def __init__(self, training_args):
        self.training_args = training_args

    # ------------------------------------------------------------
    # Enablement
    # ------------------------------------------------------------
    def weight_of(self, output: str) -> float:
        """Look up the config weight for an output key.

        Convention: training_args.<output>_reward_weight must exist for
        each declared OUTPUT. Defaults to 0.0 via getattr, so unrelated
        configs keep this reward quietly disabled.
        """
        return float(getattr(self.training_args, f"{output}_reward_weight", 0.0))

    @property
    def enabled_for_training(self) -> bool:
        """True iff any of this reward's outputs has a non-zero weight."""
        return any(self.weight_of(o) > 0.0 for o in self.OUTPUTS)

    @property
    def enabled_for_eval(self) -> bool:
        """Override when this reward is used by eval.py.

        Default: same as training enablement. Subclasses that correspond
        to eval scorers (MonST3R, aesthetic) override to check the
        eval_* flag too so eval works without training.
        """
        return self.enabled_for_training

    @property
    def enabled(self) -> bool:
        return self.enabled_for_training or self.enabled_for_eval

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def load(self, device) -> None:
        """Idempotent: bring the backing model onto `device`. Override when
        there is state to load; default is no-op.
        """
        return

    def unload(self) -> None:
        """Idempotent: release GPU resources. Default no-op."""
        return

    # ------------------------------------------------------------
    # Scoring — subclasses implement exactly one of these.
    # ------------------------------------------------------------
    def is_group(self) -> bool:
        return False

    def score_one(self, ctx: RewardContext) -> ScoreResult:
        raise NotImplementedError(
            f"{type(self).__name__}.score_one not implemented"
        )

    def score_group(self, ctxs: list[RewardContext]) -> list[ScoreResult]:
        """Override for rewards that need all candidates at once."""
        raise NotImplementedError(
            f"{type(self).__name__}.score_group not implemented"
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def neutral_result(self) -> ScoreResult:
        from fastvideo.rewards.utils.metrics_to_reward import NEUTRAL_REWARD
        return ScoreResult(scores={o: NEUTRAL_REWARD for o in self.OUTPUTS})

    def zero_result(self) -> ScoreResult:
        """For the case where scoring is skipped entirely (reward disabled
        in training_args but context still built — shouldn't normally
        happen because dispatcher filters, but defensive).
        """
        return ScoreResult(scores={o: 0.0 for o in self.OUTPUTS})


class PerCandidateReward(BaseReward):
    """Convenience: subclasses override score_one only."""

    def is_group(self) -> bool:
        return False


class GroupReward(BaseReward):
    """Convenience: subclasses override score_group only."""

    def is_group(self) -> bool:
        return True

    def score_one(self, ctx: RewardContext) -> ScoreResult:
        # Fall back to group-of-one if called individually (unusual).
        return self.score_group([ctx])[0]
