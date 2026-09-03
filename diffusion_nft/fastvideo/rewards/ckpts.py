"""Single source of truth for reward model checkpoint paths.

Paths resolve relative to the project root (the directory containing
`fastvideo/`) so moving the repo to a different cluster mount point
just works. Override any entry via env var REWARD_CKPT_<NAME> for
local testing or alternate checkpoints.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root: fastvideo/rewards/ckpts.py → fastvideo/rewards → fastvideo → project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CKPT_ROOT: Path = PROJECT_ROOT / "ckpt"


def _resolve(key: str, relative: str) -> str:
    """Resolve a checkpoint path.

    If REWARD_CKPT_<KEY> is set (upper-case), use that absolute path.
    Otherwise return PROJECT_ROOT / "ckpt" / relative.
    """
    override = os.environ.get(f"REWARD_CKPT_{key.upper()}", "").strip()
    if override:
        return override
    return str(CKPT_ROOT / relative)


@dataclass(frozen=True)
class RewardCheckpoints:
    # MonST3R camera trajectory model
    monst3r: str = _resolve(
        "MONST3R",
        "Junyi42--MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
    )
    # Aesthetic predictor: CLIP backbone + MLP head
    aesthetic_clip: str = _resolve(
        "AESTHETIC_CLIP",
        "openai--clip-vit-large-patch14",
    )
    aesthetic_mlp: str = _resolve(
        "AESTHETIC_MLP",
        "aesthetic/sac+logos+ava1-l14-linearMSE.pth",
    )
    # HunyuanWorldMirror (action reward backbone)
    worldmirror: str = _resolve(
        "WORLDMIRROR",
        "tencent--HY-WorldMirror",
    )
    # HPSv3 (human preference score v3)
    hpsv3: str = _resolve(
        "HPSV3",
        "hpsv3",
    )
    # VLM reward model. Served by an external vLLM instance — the training
    # side never loads it, this is only the identifier handed to the server.
    # See the "Reward Server" section of the README; override with $VLM_RM.
    vlm_rm: str = _resolve(
        "VLM_RM",
        "CodeGoat24/WorldReward-qwen35-9b",
    )


CKPTS = RewardCheckpoints()
