"""Reward dispatcher — single entry point for training- and eval-side reward code.

Owns:
  * the registry-built list of enabled reward instances
  * canonical output ordering + per-output weight lookup
  * the advantage-computation loop (per-reward z-score within GPU group,
    weighted sum, global z-score across world)
"""
from __future__ import annotations

import torch

from fastvideo.rewards.base import BaseReward, RewardContext
from fastvideo.rewards.computers.aesthetic import AestheticReward
from fastvideo.rewards.computers.hpsv3 import HPSv3Reward
from fastvideo.rewards.computers.monst3r_trajectory import MonST3RReward
from fastvideo.rewards.computers.worldreward import VLMPairwiseReward
from fastvideo.rewards.computers.worldmirror import WorldMirrorReward
from fastvideo.rewards.registry import build_enabled_rewards
from fastvideo.rewards.utils.types import RewardRequest


class RewardDispatcher:
    """Owns all enabled reward instances and serves the training loop."""

    def __init__(
        self,
        *,
        training_args=None,
        rewards: list[BaseReward] | None = None,
    ) -> None:
        if rewards is None:
            if training_args is None:
                rewards = []
            else:
                rewards = build_enabled_rewards(training_args)
        self.rewards: list[BaseReward] = list(rewards)
        self.training_args = training_args

    # ------------------------------------------------------------
    # Registry / introspection
    # ------------------------------------------------------------
    def _find(self, cls: type[BaseReward]) -> BaseReward | None:
        for r in self.rewards:
            if isinstance(r, cls):
                return r
        return None

    @property
    def worldmirror(self) -> WorldMirrorReward | None:
        return self._find(WorldMirrorReward)  # type: ignore[return-value]

    @property
    def hpsv3(self) -> HPSv3Reward | None:
        return self._find(HPSv3Reward)  # type: ignore[return-value]

    @property
    def vlm(self) -> VLMPairwiseReward | None:
        return self._find(VLMPairwiseReward)  # type: ignore[return-value]

    @property
    def monst3r(self) -> MonST3RReward | None:
        return self._find(MonST3RReward)  # type: ignore[return-value]

    @property
    def aesthetic(self) -> AestheticReward | None:
        return self._find(AestheticReward)  # type: ignore[return-value]

    @property
    def enabled_outputs(self) -> list[str]:
        """Output keys with non-zero training weight, in registry order."""
        keys: list[str] = []
        for r in self.rewards:
            for o in r.OUTPUTS:
                if r.weight_of(o) > 0.0:
                    keys.append(o)
        return keys

    @property
    def all_outputs(self) -> list[str]:
        """Every output declared by loaded rewards (enabled or not)."""
        keys: list[str] = []
        for r in self.rewards:
            keys.extend(r.OUTPUTS)
        return keys

    def weight_of(self, output: str) -> float:
        for r in self.rewards:
            if output in r.OUTPUTS:
                return r.weight_of(output)
        return 0.0

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def load_all(self, device) -> None:
        for r in self.rewards:
            r.load(device)

    def unload_all(self) -> None:
        for r in self.rewards:
            r.unload()

    # ------------------------------------------------------------
    # Advantage computation (hoisted out of _prepare_grpo_inputs)
    # ------------------------------------------------------------
    def compute_advantages(
        self,
        *,
        sample_kwargs: dict,
        sample_keys: list[str],
        gpu_group,
        world_group,
        std_type: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-reward z-score → weighted sum → global z-score.

        Returns (overall_reward [len(sample_keys)], reward_means dict).
        """
        zero = torch.tensor(0.0, device=device)
        for sk in sample_keys:
            for o in self.all_outputs:
                sample_kwargs[sk].setdefault(f"{o}_advantages", zero)

        reward_means: dict[str, float] = {}
        overall_reward = torch.zeros(len(sample_keys), device=device)
        sum_weights = 0.0
        weighted_adv_sum = torch.zeros(len(sample_keys), device=device)

        for o in self.all_outputs:
            reward_key = f"{o}_reward"
            if any(reward_key not in sample_kwargs[sk] for sk in sample_keys):
                reward_means[o] = 0.0
                continue

            w = self.weight_of(o)
            rewards = torch.stack([
                torch.as_tensor(sample_kwargs[sk][reward_key], device=device)
                for sk in sample_keys
            ]).to(dtype=torch.float32)

            reward_means[o] = float(
                world_group.all_gather(rewards, dim=0).mean().item()
            )

            if w <= 0.0:
                continue

            group_rewards = gpu_group.all_gather(rewards, dim=0)
            group_mean = group_rewards.mean()
            group_std = group_rewards.std(unbiased=False) + 1e-8

            world_rewards = world_group.all_gather(group_rewards, dim=0)
            world_std_gathered = world_group.all_gather(
                group_std.unsqueeze(0), dim=0
            )
            if std_type == "sample_max":
                group_std = world_std_gathered.max()
            elif std_type == "global":
                group_std = world_rewards.std(unbiased=False) + 1e-8

            advantages = (rewards - group_mean) / group_std
            for si, sk in enumerate(sample_keys):
                sample_kwargs[sk][f"{o}_advantages"] = advantages[si]
            overall_reward += w * advantages
            weighted_adv_sum += w * advantages
            sum_weights += w

        if sum_weights <= 0:
            sum_weights = 1.0
        raw_total_adv = weighted_adv_sum / sum_weights

        all_total_adv = world_group.all_gather(raw_total_adv, dim=0)
        global_mean = all_total_adv.mean()
        global_std = all_total_adv.std(unbiased=False) + 1e-8
        total_adv = (raw_total_adv - global_mean) / global_std
        for si, sk in enumerate(sample_keys):
            sample_kwargs[sk]["total_advantages"] = total_adv[si]

        return overall_reward, reward_means

    # ------------------------------------------------------------
    # Generic scoring
    # ------------------------------------------------------------
    def score_all(
        self, ctxs: list[RewardContext]
    ) -> dict[int, dict[str, float]]:
        from fastvideo.rewards.utils.metrics_to_reward import NEUTRAL_REWARD

        result: dict[int, dict[str, float]] = {
            c.candidate_idx: {} for c in ctxs
        }
        for reward in self.rewards:
            if not reward.enabled:
                continue
            if reward.is_group():
                try:
                    group_out = reward.score_group(ctxs)
                except Exception:
                    group_out = [reward.neutral_result() for _ in ctxs]
                for ctx, sr in zip(ctxs, group_out):
                    result[ctx.candidate_idx].update(sr.scores)
            else:
                for ctx in ctxs:
                    try:
                        sr = reward.score_one(ctx)
                    except Exception:
                        sr = reward.neutral_result()
                    result[ctx.candidate_idx].update(sr.scores)
        for ctx in ctxs:
            for o in self.all_outputs:
                result[ctx.candidate_idx].setdefault(o, NEUTRAL_REWARD)
        return result


