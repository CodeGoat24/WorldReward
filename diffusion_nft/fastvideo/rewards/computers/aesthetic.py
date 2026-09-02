"""Aesthetic reward — LAION improved-aesthetic-predictor on video frames.

OUTPUT "aesthetic" is the raw LAION score (approx 0-10, higher is
better). No squashing — advantages are per-output z-score (scale-
invariant), so linear_scale only distorts the dynamic range without
changing relative ordering. NaN on scorer failure falls back to 5.0
(dataset median) so failed scorers don't masquerade as outliers.

Wraps the LAION "improved-aesthetic-predictor" (CLIP-ViT-L/14 image
features fed into a small MLP regressor trained on SAC/AVA/LAION-logos)
and applies it frame-wise to a generated video, returning a single
scalar quality score.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
from PIL import Image

from fastvideo.rewards.base import PerCandidateReward, RewardContext, ScoreResult
from fastvideo.rewards.ckpts import CKPTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator (private to this module; exposed as .evaluator on the computer)
# ---------------------------------------------------------------------------

class _AestheticMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class AestheticVideoEvaluator:
    """Lazy-loaded per-video aesthetic scorer."""

    def __init__(
        self,
        mlp_ckpt_path: str | None = None,
        clip_path: str | None = None,
        device: str | None = None,
        num_frames: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        env_mlp = os.environ.get("AESTHETIC_CHECKPOINT", "").strip()
        self.mlp_ckpt_path = (
            mlp_ckpt_path or (env_mlp if env_mlp else None) or CKPTS.aesthetic_mlp
        )
        env_clip = os.environ.get("AESTHETIC_CLIP_PATH", "").strip()
        self.clip_path = (
            clip_path or (env_clip if env_clip else None) or CKPTS.aesthetic_clip
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = int(num_frames)
        self.dtype = dtype
        self._clip = None
        self._processor = None
        self._mlp = None

    def _ensure_loaded(self):
        if self._mlp is not None:
            return
        if not os.path.isfile(self.mlp_ckpt_path):
            raise FileNotFoundError(
                f"Aesthetic MLP checkpoint not found at {self.mlp_ckpt_path}. "
                "Set $AESTHETIC_CHECKPOINT or place "
                "`sac+logos+ava1-l14-linearMSE.pth` under ckpt/aesthetic/"
            )
        from transformers import CLIPModel, CLIPProcessor

        logger.info(
            "[aesthetic] loading CLIP from %s and MLP from %s",
            self.clip_path, self.mlp_ckpt_path,
        )
        clip = CLIPModel.from_pretrained(self.clip_path)
        processor = CLIPProcessor.from_pretrained(self.clip_path)
        mlp = _AestheticMLP()
        state_dict = torch.load(
            self.mlp_ckpt_path, map_location="cpu", weights_only=True
        )
        mlp.load_state_dict(state_dict)

        clip.to(device=self.device, dtype=self.dtype).eval()
        mlp.to(device=self.device, dtype=self.dtype).eval()

        self._clip = clip
        self._processor = processor
        self._mlp = mlp

    def _sample_frames(self, video_path: str) -> list[Image.Image]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"video has no frames: {video_path}")

        if total <= self.num_frames:
            idxs = list(range(total))
        else:
            idxs = np.linspace(0, total - 1, self.num_frames).round().astype(int).tolist()

        frames: list[Image.Image] = []
        target = set(idxs)
        max_idx = max(idxs)
        cur = 0
        while cur <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if cur in target:
                frames.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
            cur += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"failed to read any frames from {video_path}")
        return frames

    @torch.no_grad()
    def _score_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype)
            for k, v in inputs.items()
        }
        embed = self._clip.get_image_features(**inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self._mlp(embed).squeeze(1)

    def score_video(self, video_path: str) -> float:
        """Single aesthetic score (float) for the video, NaN on failure."""
        try:
            self._ensure_loaded()
            frames = self._sample_frames(video_path)
            scores = self._score_images(frames)
            return float(scores.mean().item())
        except Exception as e:
            logger.warning(
                "[aesthetic] scoring failed for %s: %s", video_path, e
            )
            return float("nan")

    def unload(self) -> None:
        """Free GPU memory. Safe to call even if nothing is loaded."""
        if self._clip is None and self._mlp is None:
            return
        logger.info("[aesthetic] unloading CLIP+MLP to free GPU memory")
        try:
            if self._clip is not None:
                self._clip.to("cpu")
            if self._mlp is not None:
                self._mlp.to("cpu")
        except Exception as e:
            logger.warning("[aesthetic] unload .to(cpu) failed: %s", e)
        self._clip = None
        self._processor = None
        self._mlp = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# BaseReward subclass
# ---------------------------------------------------------------------------

class AestheticReward(PerCandidateReward):
    NAME = "aesthetic"
    OUTPUTS = ("aesthetic",)

    def __init__(self, training_args):
        super().__init__(training_args)
        self._evaluator: AestheticVideoEvaluator | None = None

    def load(self, device) -> None:
        if self._evaluator is not None:
            return
        self._evaluator = AestheticVideoEvaluator(
            mlp_ckpt_path=CKPTS.aesthetic_mlp,
            clip_path=CKPTS.aesthetic_clip,
            device=str(device),
            num_frames=int(
                getattr(self.training_args, "eval_aesthetic_num_frames", 16)
            ),
        )

    def unload(self) -> None:
        if self._evaluator is not None:
            self._evaluator.unload()

    @property
    def evaluator(self) -> AestheticVideoEvaluator:
        assert self._evaluator is not None, "AestheticReward not loaded"
        return self._evaluator

    @property
    def enabled_for_eval(self) -> bool:
        return (
            self.enabled_for_training
            or bool(getattr(self.training_args, "eval_aesthetic", False))
        )

    def score_one(self, ctx: RewardContext) -> ScoreResult:
        assert self._evaluator is not None, "AestheticReward not loaded"
        raw = self._evaluator.score_video(ctx.video_path)
        value = raw if math.isfinite(raw) else 5.0
        return ScoreResult(
            scores={"aesthetic": float(value)},
            metrics={"aesthetic_raw": float(raw)},
        )
