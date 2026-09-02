"""Shared video saving and path construction utilities."""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEFAULT_VIDEO_FPS = 16


def get_generated_videos_base(training_args) -> Path:
    """Return the base directory for generated videos from training args."""
    base = training_args.generated_videos_dir
    if not base:
        base = os.path.join(training_args.output_dir, "generated_videos")
    return Path(base) / Path(training_args.output_dir).name


def save_video(video_frames: np.ndarray, save_path: str | Path, fps: int = DEFAULT_VIDEO_FPS) -> None:
    imageio.mimsave(str(save_path), video_frames, fps=fps)
