"""Shared frame-extraction helpers for WorldMirror and HPSv3 scorers.

Kept deliberately tiny so each scorer is otherwise fully decoupled.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import imageio.v2 as imageio
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


def extract_frames_from_video(
    video_path: str,
    temp_dir: str,
    instance_id: str,
    interval: int = 4,
    max_frames: Optional[int] = None,
) -> list[str]:
    """Extract every `interval`-th frame of `video_path` to JPEGs in `temp_dir`.

    Returns the list of absolute JPEG paths in order. Caller is responsible
    for deleting them after use.
    """
    frame_paths: list[str] = []
    frame_count = 0

    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(0, total_frames, interval):
            if max_frames is not None and frame_count >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = os.path.join(
                    temp_dir, f"frame_{instance_id}_{frame_count}.jpg"
                )
                Image.fromarray(frame_rgb).save(frame_path)
                frame_paths.append(frame_path)
                frame_count += 1
        cap.release()
        return frame_paths

    reader = imageio.get_reader(video_path)
    try:
        for frame_idx, frame_rgb in enumerate(reader):
            if frame_idx % interval != 0:
                continue
            if max_frames is not None and frame_count >= max_frames:
                break
            frame_path = os.path.join(
                temp_dir, f"frame_{instance_id}_{frame_count}.jpg"
            )
            Image.fromarray(frame_rgb).save(frame_path)
            frame_paths.append(frame_path)
            frame_count += 1
    finally:
        reader.close()
    return frame_paths


def cleanup_frame_paths(paths: list[str], *extra_paths: str) -> None:
    """Best-effort delete the given frame paths; swallow errors."""
    for p in list(paths) + list(extra_paths):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def new_instance_id() -> str:
    return str(uuid.uuid4())[:8]
