#!/usr/bin/env python3
"""Baselines that score each video independently, then compare the scalars.

    # visual-quality scorers
    python baselines/run_quality_scorer.py --scorer aesthetic --bench data/WorldReward-Bench/bench.jsonl
    python baselines/run_quality_scorer.py --scorer hpsv3     --bench data/WorldReward-Bench/bench.jsonl
    python baselines/run_quality_scorer.py --scorer videoalign --bench data/WorldReward-Bench/bench.jsonl

Which axes each scorer can speak to:

=============  ==========================  =====================================
``aesthetic``  appearance                  one aesthetic scalar per video
``hpsv3``      appearance                  HPSv3 caption-free quality head
``videoalign`` appearance, motion          VQ head -> appearance, MQ -> motion
=============  ==========================  =====================================

None of them observe the commanded camera trajectory, so none predicts
``action``; that axis is left ``null`` and reported as ``--`` by ``score.py``.

Multi-GPU: run one process per GPU with ``--shard-id``/``--num-shards``, then
``--merge``. See ``baselines/README`` section of the top-level README.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

# Axes each scorer is allowed to answer. Anything else stays null so that
# score.py reports "not modelled" rather than counting it as a wrong answer.
SCORER_AXES = {
    "aesthetic": {"appearance": "aesthetic"},
    "hpsv3": {"appearance": "quality"},
    "videoalign": {"appearance": "VQ", "motion": "MQ"},
}


class AestheticScorer:
    """CLIP ViT-L/14 image embedding + the public LAION aesthetic MLP head.

    Scores ``num_frames`` frames sampled uniformly and averages, so the result
    is a whole-video aesthetic scalar.
    """

    def __init__(self, args) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = args.device
        self.num_frames = args.num_frames
        self.clip = CLIPModel.from_pretrained(args.clip_path).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(args.clip_path)

        mlp = _aesthetic_mlp(torch)
        state = torch.load(args.aesthetic_mlp, map_location="cpu", weights_only=True)
        mlp.load_state_dict(state)
        self.mlp = mlp.to(self.device).eval()

    def score(self, video: Path, caption: str) -> dict[str, float]:
        frames = sample_frames(video, self.num_frames)
        with self.torch.no_grad():
            inputs = self.processor(images=frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embed = self.clip.get_image_features(**inputs)
            # transformers >=5 returns a model-output object here; <5 returns the
            # tensor directly.
            if not isinstance(embed, self.torch.Tensor):
                embed = getattr(embed, "image_embeds", None) or embed.pooler_output
            embed = embed / self.torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
            scores = self.mlp(embed.to(self.torch.float32)).squeeze(1)
        return {"aesthetic": float(scores.mean().item())}


class HPSv3Scorer:
    """HPSv3 reward model, using its caption-free *quality* prompt."""

    QUALITY_PROMPT = "a high quality, detailed, and aesthetically pleasing image"

    def __init__(self, args) -> None:
        import torch
        from hpsv3 import HPSv3RewardInferencer

        self.torch = torch
        self.num_frames = args.num_frames
        self.inferencer = HPSv3RewardInferencer(
            checkpoint_path=args.hpsv3_checkpoint, device=args.device
        )

    def score(self, video: Path, caption: str) -> dict[str, float]:
        import tempfile

        frames = sample_frames(video, self.num_frames)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, frame in enumerate(frames):
                path = Path(tmp) / f"{index:04d}.png"
                frame.save(path)
                paths.append(str(path))
            with self.torch.no_grad():
                rewards = self.inferencer.reward(paths, [self.QUALITY_PROMPT] * len(paths))
        values = [float(r[0].item()) for r in rewards]
        return {"quality": sum(values) / len(values)}


class VideoAlignScorer:
    """KlingTeam VideoReward: VQ maps to appearance, MQ to motion; TA unused."""

    def __init__(self, args) -> None:
        import torch

        if not args.videoalign_src:
            raise SystemExit(
                "--videoalign-src is required for --scorer videoalign: pass the "
                "checkout of https://github.com/KwaiVGI/VideoAlign so its "
                "inference module is importable."
            )
        sys.path.insert(0, str(Path(args.videoalign_src).resolve()))
        from inference import VideoVLMRewardInference

        self.torch = torch
        self.inferencer = VideoVLMRewardInference(
            args.videoalign_ckpt,
            device=args.device,
            dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        )

    def score_pair(self, pair: Pair) -> tuple[dict[str, float], dict[str, float]]:
        """Both videos in one call -- the model is built for batched pairs."""
        with self.torch.no_grad():
            rewards = self.inferencer.reward(
                [str(pair.left_video), str(pair.right_video)],
                [pair.caption, pair.caption],
                use_norm=True,
            )
        keys = ("VQ", "MQ", "TA", "Overall")
        return (
            {k: float(rewards[0].get(k, 0.0)) for k in keys},
            {k: float(rewards[1].get(k, 0.0)) for k in keys},
        )


def _aesthetic_mlp(torch):
    """The LAION `sac+logos+ava1-l14-linearMSE` head.

    Layer shapes and the ``layers.*`` parameter names are both fixed by the
    published checkpoint, so the Sequential must stay wrapped in a module with a
    ``.layers`` attribute for ``load_state_dict`` to match.
    """

    class _MLP(torch.nn.Module):
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

        def forward(self, embed):
            return self.layers(embed)

    return _MLP()


def sample_frames(video: Path, num_frames: int):
    """Uniformly sample ``num_frames`` frames as PIL images."""
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"no frames in {video}")

    if total <= num_frames:
        wanted = list(range(total))
    else:
        wanted = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()

    target = set(wanted)
    frames = []
    index = 0
    while index <= max(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index in target:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"failed to decode any frame from {video}")
    return frames


def build_scorer(args):
    if args.scorer == "aesthetic":
        return AestheticScorer(args)
    if args.scorer == "hpsv3":
        return HPSv3Scorer(args)
    if args.scorer == "videoalign":
        return VideoAlignScorer(args)
    raise SystemExit(f"unknown scorer {args.scorer}")


def make_score_one(scorer, args):
    axes = SCORER_AXES[args.scorer]

    def score_one(pair: Pair) -> dict[str, object]:
        if hasattr(scorer, "score_pair"):
            left_scores, right_scores = scorer.score_pair(pair)
        else:
            left_scores = scorer.score(pair.left_video, pair.caption)
            right_scores = scorer.score(pair.right_video, pair.caption)

        record: dict[str, object] = {
            "scorer": args.scorer,
            "scores_left": left_scores,
            "scores_right": right_scores,
        }
        for axis, key in axes.items():
            record[axis] = verdict_from_scores(
                left_scores[key], right_scores[key], eps=args.eps
            )
        return record

    return score_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scorer", required=True, choices=sorted(SCORER_AXES))
    add_common_args(parser, default_output="outputs/baselines/quality.pairs.jsonl")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Tie band on the score gap. 0 (default, and what the published "
        "numbers use) means the baseline never predicts tie.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-frames", type=int, default=16, help="Frames sampled per video"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing per-shard caches into --output and exit",
    )

    aesthetic = parser.add_argument_group("aesthetic")
    aesthetic.add_argument("--clip-path", default="openai/clip-vit-large-patch14")
    aesthetic.add_argument(
        "--aesthetic-mlp",
        default="ckpt/sac+logos+ava1-l14-linearMSE.pth",
        help="LAION aesthetic head; download from "
        "https://github.com/christophschuhmann/improved-aesthetic-predictor",
    )

    hpsv3 = parser.add_argument_group("hpsv3")
    hpsv3.add_argument(
        "--hpsv3-checkpoint",
        default=None,
        help="HPSv3.safetensors path; omit to let the hpsv3 package resolve it",
    )

    videoalign = parser.add_argument_group("videoalign")
    videoalign.add_argument(
        "--videoalign-src",
        default=None,
        help="Checkout of https://github.com/KwaiVGI/VideoAlign (added to sys.path)",
    )
    videoalign.add_argument("--videoalign-ckpt", default="KwaiVGI/VideoReward")
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

    scorer = build_scorer(args)
    for _ in run_pairs(pairs, make_score_one(scorer, args), cache, label=args.scorer):
        pass

    count = write_pairs_jsonl(output, cache.records())
    print(f"wrote {output} ({count} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
