"""Concrete reward computer classes.

Each file defines one BaseReward subclass. Adding a new reward is:
  1. drop `fastvideo/rewards/computers/<new>.py`
  2. append the class to `fastvideo.rewards.registry.REWARD_CLASSES`
  3. add the matching `<output>_reward_weight` cli arg in TrainingArgs
"""
from __future__ import annotations

from fastvideo.rewards.computers.aesthetic import AestheticReward
from fastvideo.rewards.computers.hpsv3 import HPSv3Reward
from fastvideo.rewards.computers.monst3r_trajectory import MonST3RReward
from fastvideo.rewards.computers.worldreward import VLMPairwiseReward
from fastvideo.rewards.computers.worldmirror import WorldMirrorReward

__all__ = [
    "AestheticReward",
    "HPSv3Reward",
    "MonST3RReward",
    "VLMPairwiseReward",
    "WorldMirrorReward",
]
