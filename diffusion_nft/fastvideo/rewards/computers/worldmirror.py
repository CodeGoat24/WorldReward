"""WorldMirror action-accuracy reward.

Outputs (both already normalized to [0, 1] by the scorer):
  action       — discrete-action classification accuracy (coarse grid)
  fine_action  — per-axis (trans/rotate) label match mean

Load-gating rules:
  * Loads WorldMirror (or DAv3) iff any of its outputs has non-zero
    weight OR an eval/prefilter flag explicitly asks for it.

Legacy CLI plumbing: a single `action_reward_weight` + `action_reward_type`
pair picks between "action" and "fine_action" as the weighted output.
Setting `fine_action_reward_weight` directly is also supported.
"""
from __future__ import annotations

from fastvideo.rewards.base import PerCandidateReward, RewardContext, ScoreResult
from fastvideo.rewards.computers._worldmirror_scorer import WorldMirrorScorer


class WorldMirrorReward(PerCandidateReward):
    NAME = "worldmirror"
    OUTPUTS = ("action", "fine_action")

    def __init__(self, training_args):
        super().__init__(training_args)
        self._scorer: WorldMirrorScorer | None = None

    # Legacy CLI routing — see module docstring.
    def weight_of(self, output: str) -> float:
        reward_type = getattr(self.training_args, "action_reward_type", "action")
        if output == "action":
            if reward_type == "fine_action":
                return 0.0
            return float(getattr(self.training_args, "action_reward_weight", 0.0))
        if output == "fine_action":
            if reward_type == "fine_action":
                return float(getattr(self.training_args, "action_reward_weight", 0.0))
            return float(getattr(self.training_args, "fine_action_reward_weight", 0.0))
        return super().weight_of(output)

    def _wants_load(self) -> bool:
        return (
            self.enabled_for_training
            or bool(getattr(self.training_args, "vlm_prefilter_enabled", False))
            or bool(getattr(self.training_args, "eval_worldmirror", False))
        )

    @property
    def enabled_for_training(self) -> bool:
        return any(self.weight_of(o) > 0 for o in self.OUTPUTS)

    @property
    def enabled_for_eval(self) -> bool:
        return self.enabled_for_training or self._wants_load()

    def load(self, device) -> None:
        if self._scorer is not None or not self._wants_load():
            return
        camera_estimator = getattr(self.training_args, "camera_estimator", "worldmirror")
        cache_dir = getattr(self.training_args, "cache_dir", None) or None
        self._scorer = WorldMirrorScorer(
            device=str(device) if device is not None else None,
            camera_estimator=camera_estimator,
            cache_dir=cache_dir,
        )

    def unload(self) -> None:
        if self._scorer is not None:
            self._scorer.offload_to_cpu()

    # ------------------------------------------------------------
    # Back-compat construction from an already-built scorer.
    # Used by the LingBot entry points which build a scorer directly
    # (they don't go through the registry).
    # ------------------------------------------------------------
    @classmethod
    def from_scorer(cls, scorer: WorldMirrorScorer) -> "WorldMirrorReward":
        from types import SimpleNamespace
        inst = cls(
            SimpleNamespace(
                camera_estimator=scorer.camera_estimator,
                action_reward_weight=1.0,
                action_reward_type="action",
                fine_action_reward_weight=0.0,
                vlm_prefilter_enabled=False,
                eval_worldmirror=False,
            )
        )
        inst._scorer = scorer
        return inst

    @property
    def scorer(self) -> WorldMirrorScorer:
        assert self._scorer is not None, "WorldMirrorReward not loaded"
        return self._scorer

    def score_one(self, ctx: RewardContext) -> ScoreResult:
        assert self._scorer is not None, "WorldMirrorReward not loaded"
        metrics = self._scorer.score(
            video_path=ctx.video_path,
            gt_action=ctx.gt_action,
            interval=1,
            update_latent_num=ctx.update_latent_num,
        )
        return ScoreResult(
            scores={
                "action": float(metrics["action_acc"]),
                "fine_action": float(metrics["fine_action_acc"]),
            },
            metrics=metrics,
        )
