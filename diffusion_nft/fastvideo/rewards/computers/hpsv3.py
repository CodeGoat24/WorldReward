"""HPSv3 human-preference reward.

Outputs (already in [0, 1]-ish raw HPSv3 range; linear passthrough):
  hpsv3               — caption / video alignment
  hpsv3_quality       — quality-prompt alignment
  hpsv3_quality_drift — -|quality_acc - first_frame_quality| (anti-drift)
"""
from __future__ import annotations

from fastvideo.rewards.base import PerCandidateReward, RewardContext, ScoreResult
from fastvideo.rewards.computers._hpsv3_scorer import HPSv3Scorer


class HPSv3Reward(PerCandidateReward):
    NAME = "hpsv3"
    OUTPUTS = ("hpsv3", "hpsv3_quality", "hpsv3_quality_drift")

    def __init__(self, training_args):
        super().__init__(training_args)
        self._scorer: HPSv3Scorer | None = None

    def _wants_load(self) -> bool:
        return (
            self.enabled_for_training
            or bool(getattr(self.training_args, "eval_hpsv3", False))
        )

    @property
    def enabled_for_eval(self) -> bool:
        return self.enabled_for_training or self._wants_load()

    def load(self, device) -> None:
        if self._scorer is not None or not self._wants_load():
            return
        cache_dir = getattr(self.training_args, "cache_dir", None) or None
        self._scorer = HPSv3Scorer(
            device=str(device) if device is not None else None,
            cache_dir=cache_dir,
        )

    def unload(self) -> None:
        if self._scorer is not None:
            self._scorer.offload_to_cpu()

    @property
    def scorer(self) -> HPSv3Scorer:
        assert self._scorer is not None, "HPSv3Reward not loaded"
        return self._scorer

    def score_one(self, ctx: RewardContext) -> ScoreResult:
        assert self._scorer is not None, "HPSv3Reward not loaded"
        metrics = self._scorer.score(
            video_path=ctx.video_path,
            caption=ctx.caption or ctx.prompt or "",
            interval=1,
            update_latent_num=ctx.update_latent_num,
            score_caption=self.weight_of("hpsv3") > 0,
        )
        return ScoreResult(
            scores={
                "hpsv3": float(metrics["hpsv3_acc"]),
                "hpsv3_quality": float(metrics["hpsv3_quality_acc"]),
                "hpsv3_quality_drift": float(metrics["hpsv3_quality_drift_score"]),
            },
            metrics=metrics,
        )
