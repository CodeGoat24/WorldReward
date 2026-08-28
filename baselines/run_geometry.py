#!/usr/bin/env python3
"""Depth-Anything-3 baseline: judge action following by estimating the camera.

    DEPTH_ANYTHING_3_SRC=/path/to/depth-anything-3 \\
    python baselines/run_geometry.py \\
        --bench data/WorldReward-Bench/bench.jsonl \\
        --model-path depth-anything/DA3-GIANT-1.1

Unlike the quality baselines, this one *does* observe the commanded
trajectory, and answers only ``action``:

1. run DA3 over the video's frames to recover per-frame extrinsics
2. quantize consecutive relative poses into the same discrete
   translation/rotation vocabulary the benchmark's action tokens use
3. accuracy = fraction of frames whose recovered action matches the command
4. the video with higher accuracy wins ``action``

Because a single motion threshold does not suit every scene scale, three
``move_norm_valid`` values are tried per video and the best accuracy is kept.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.common import (
    Pair,
    add_common_args,
    merge_shards,
    resolve_io,
    run_pairs,
    verdict_from_scores,
    write_pairs_jsonl,
)

# One-hot flag tuple -> 9-class id, for translation and rotation alike.
_FLAGS_TO_CLASS: dict[tuple[int, int, int, int], int] = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def _flags(*indices: int) -> tuple[int, int, int, int]:
    flags = [0, 0, 0, 0]
    for index in indices:
        flags[index] = 1
    return flags[0], flags[1], flags[2], flags[3]


# Translation slots: (Forward, Backward, Right, Left)
TRANSLATION_TOKENS: dict[str, int] = {
    "": _FLAGS_TO_CLASS[_flags()],
    "forward": _FLAGS_TO_CLASS[_flags(0)],
    "backward": _FLAGS_TO_CLASS[_flags(1)],
    "right": _FLAGS_TO_CLASS[_flags(2)],
    "left": _FLAGS_TO_CLASS[_flags(3)],
    "forward_right": _FLAGS_TO_CLASS[_flags(0, 2)],
    "forward_left": _FLAGS_TO_CLASS[_flags(0, 3)],
    "backward_right": _FLAGS_TO_CLASS[_flags(1, 2)],
    "backward_left": _FLAGS_TO_CLASS[_flags(1, 3)],
}

# Rotation slots: (YawRight, YawLeft, PitchUp, PitchDown)
ROTATION_TOKENS: dict[str, int] = {
    "": _FLAGS_TO_CLASS[_flags()],
    "camera_r": _FLAGS_TO_CLASS[_flags(0)],
    "camera_l": _FLAGS_TO_CLASS[_flags(1)],
    "camera_up": _FLAGS_TO_CLASS[_flags(2)],
    "camera_down": _FLAGS_TO_CLASS[_flags(3)],
    "camera_ur": _FLAGS_TO_CLASS[_flags(0, 2)],
    "camera_dr": _FLAGS_TO_CLASS[_flags(0, 3)],
    "camera_ul": _FLAGS_TO_CLASS[_flags(1, 2)],
    "camera_dl": _FLAGS_TO_CLASS[_flags(1, 3)],
}

# Thresholds swept per video, matching the internal evaluation protocol.
MOVE_NORM_CANDIDATES = (0.002, 0.005, 0.01)
ROTATION_THRESHOLD_DEG = 0.2


def split_token(token: str) -> tuple[str, str]:
    """Split an action token into (translation, rotation) parts."""
    parts = token.split("+")
    if len(parts) == 1:
        part = parts[0]
        if part in TRANSLATION_TOKENS:
            return part, ""
        if part in ROTATION_TOKENS:
            return "", part
        raise ValueError(f"unknown action token {token!r}")
    if len(parts) == 2:
        first, second = parts
        if first in TRANSLATION_TOKENS and second in ROTATION_TOKENS:
            return first, second
        if second in TRANSLATION_TOKENS and first in ROTATION_TOKENS:
            return second, first
    raise ValueError(f"cannot parse action token {token!r}")


def token_to_label(token: str) -> int:
    translation, rotation = split_token(token)
    return TRANSLATION_TOKENS[translation] * 9 + ROTATION_TOKENS[rotation]


def frame_labels(pair: Pair) -> list[int]:
    """Expand the action sequence to one ground-truth label per frame."""
    if not pair.actions:
        return []
    per_action = pair.frames_per_action or (
        pair.num_frames // len(pair.actions) if pair.num_frames else 0
    )
    if per_action <= 0:
        raise ValueError(f"{pair.pair_id}: cannot infer frames per action")
    total = pair.num_frames or per_action * len(pair.actions)
    labels = [token_to_label(token) for token in pair.actions]
    last = len(labels) - 1
    return [labels[min(index // per_action, last)] for index in range(total)]


def normalize_extrinsics(extrinsics, expected_frames: int):
    """Coerce estimator extrinsics into a [T, 4, 4] world-to-camera tensor."""
    import torch

    poses = torch.as_tensor(extrinsics, dtype=torch.float32)
    while poses.ndim > 3 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
    while poses.ndim > 3 and poses.shape[1] == 1:
        poses = poses.squeeze(1)
    if poses.ndim != 3:
        raise ValueError(f"unsupported extrinsics shape {tuple(poses.shape)}")

    if poses.shape[-2:] == (3, 4):
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4).expand(poses.shape[0], 1, 4)
        poses = torch.cat([poses, bottom], dim=1)
    elif poses.shape[-2:] != (4, 4):
        raise ValueError(f"unsupported extrinsics matrix {tuple(poses.shape[-2:])}")

    if poses.shape[0] < expected_frames:
        raise ValueError(f"got {poses.shape[0]} poses for {expected_frames} frames")
    return poses[-expected_frames:] if poses.shape[0] != expected_frames else poses


def poses_to_actions(poses, move_norm_valid: float, rotation_threshold: float):
    """Quantize a [N,4,4] w2c sequence into [N] discrete action labels.

    Frame 0 has no predecessor and is always labelled idle; callers drop it.
    """
    import torch

    count = poses.shape[0]
    c2w = torch.inverse(poses)
    relative = torch.zeros_like(c2w)
    relative[0] = c2w[0]
    relative[1:] = torch.bmm(torch.inverse(c2w[:-1]), c2w[1:])

    translation_flags = torch.zeros((count, 4), dtype=torch.int32)
    rotation_flags = torch.zeros((count, 4), dtype=torch.int32)

    for index in range(1, count):
        offset = relative[index, :3, 3]
        distance = torch.norm(offset)
        if distance > move_norm_valid:
            direction = torch.clamp(offset / distance, -1.0, 1.0)
            angles = torch.acos(direction) * (180.0 / math.pi)
            # angles[2] is the angle to +z (forward/back); angles[0] to +x (right/left).
            if angles[2] < 60:
                translation_flags[index, 0] = 1
            elif angles[2] > 120:
                translation_flags[index, 1] = 1
            if angles[0] < 60:
                translation_flags[index, 2] = 1
            elif angles[0] > 120:
                translation_flags[index, 3] = 1

        rotation = relative[index, :3, :3]
        sy = torch.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        if sy > 1e-6:
            pitch = torch.atan2(rotation[2, 1], rotation[2, 2])
            yaw = torch.atan2(-rotation[2, 0], sy)
        else:
            pitch = torch.atan2(-rotation[1, 2], rotation[1, 1])
            yaw = torch.atan2(-rotation[2, 0], sy)
        pitch_deg = float(pitch) * (180.0 / math.pi)
        yaw_deg = float(yaw) * (180.0 / math.pi)

        if yaw_deg > rotation_threshold:
            rotation_flags[index, 0] = 1
        elif yaw_deg < -rotation_threshold:
            rotation_flags[index, 1] = 1
        if pitch_deg > rotation_threshold:
            rotation_flags[index, 2] = 1
        elif pitch_deg < -rotation_threshold:
            rotation_flags[index, 3] = 1

    labels = torch.zeros(count, dtype=torch.long)
    for index in range(count):
        translation = _FLAGS_TO_CLASS.get(tuple(translation_flags[index].tolist()), 0)
        rotation = _FLAGS_TO_CLASS.get(tuple(rotation_flags[index].tolist()), 0)
        labels[index] = translation * 9 + rotation
    return labels


def extract_frames(video: Path, directory: Path, max_frames: int) -> list[str]:
    """Write every frame (up to ``max_frames``) to disk; DA3 reads file paths."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    paths: list[str] = []
    index = 0
    while index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        path = directory / f"{index:05d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))
        index += 1
    capture.release()
    if len(paths) < 2:
        raise RuntimeError(f"only {len(paths)} frames decoded from {video}")
    return paths


class DepthAnything3Scorer:
    def __init__(self, args) -> None:
        import importlib

        source = args.depth_anything_3_src or os.environ.get("DEPTH_ANYTHING_3_SRC")
        if source:
            sys.path.insert(0, str(Path(source).resolve()))
        try:
            api = importlib.import_module("depth_anything_3.api")
        except ImportError as exc:
            raise SystemExit(
                "cannot import depth_anything_3. Clone "
                "https://github.com/ByteDance-Seed/Depth-Anything-3 and pass "
                "--depth-anything-3-src (or set $DEPTH_ANYTHING_3_SRC)."
            ) from exc
        self.model = api.DepthAnything3.from_pretrained(args.model_path).to(args.device)
        self.model.eval()
        self.max_frames = args.max_frames

    def accuracy(self, video: Path, ground_truth: list[int]) -> dict[str, Any]:
        """Best action accuracy for one video over the threshold sweep."""
        import torch

        with tempfile.TemporaryDirectory() as directory:
            frames = extract_frames(video, Path(directory), self.max_frames)
            with torch.no_grad():
                predictions = self.model.inference(frames)

        usable = min(len(frames), len(ground_truth))
        if usable < 2:
            raise RuntimeError(f"{video}: only {usable} frames overlap the label sequence")
        poses = normalize_extrinsics(predictions.extrinsics, len(frames))[:usable]
        truth = torch.tensor(ground_truth[:usable], dtype=torch.long)[1:]

        best: dict[str, Any] | None = None
        for move_norm in MOVE_NORM_CANDIDATES:
            predicted = poses_to_actions(poses.cpu(), move_norm, ROTATION_THRESHOLD_DEG)[1:]
            accuracy = float((predicted == truth).float().mean())
            if best is None or accuracy > best["action_accuracy"]:
                best = {
                    "action_accuracy": accuracy,
                    "move_norm_valid": move_norm,
                    "translation_accuracy": float(((predicted // 9) == (truth // 9)).float().mean()),
                    "rotation_accuracy": float(((predicted % 9) == (truth % 9)).float().mean()),
                    "frames": usable,
                }
        assert best is not None
        return best


def make_score_one(scorer: DepthAnything3Scorer, args):
    def score_one(pair: Pair) -> dict[str, Any]:
        truth = frame_labels(pair)
        if not truth:
            raise ValueError(f"{pair.pair_id}: no actions in bench.jsonl")
        left = scorer.accuracy(pair.left_video, truth)
        right = scorer.accuracy(pair.right_video, truth)
        return {
            "scorer": "dav3",
            "action": verdict_from_scores(
                left["action_accuracy"], right["action_accuracy"], eps=args.eps
            ),
            "appearance": None,  # not modelled: geometry says nothing about looks
            "motion": None,
            "metrics_left": left,
            "metrics_right": right,
        }

    return score_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser, default_output="outputs/baselines/dav3.pairs.jsonl")
    parser.add_argument("--model-path", default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument(
        "--depth-anything-3-src",
        default=None,
        help="Checkout of Depth-Anything-3 (or set $DEPTH_ANYTHING_3_SRC)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=184,
        help="Frame cap per video; benchmark videos are 184 frames",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Tie band on the accuracy gap. Estimated accuracies are noisy, so "
        "the reported run uses 0.01 rather than 0.",
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge per-shard caches into --output and exit"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs, cache, output = resolve_io(args)

    if args.merge:
        base = Path(args.output).resolve()
        shards = sorted(base.parent.glob(f"{base.stem}.shard*.cache.jsonl"))
        if not shards:
            print(f"no shard caches next to {base}", file=sys.stderr)
            return 1
        count = merge_shards(shards, base)
        print(f"merged {len(shards)} shards -> {base} ({count} pairs)")
        return 0

    scorer = DepthAnything3Scorer(args)
    for _ in run_pairs(pairs, make_score_one(scorer, args), cache, label="dav3"):
        pass

    count = write_pairs_jsonl(output, cache.records())
    print(f"wrote {output} ({count} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
