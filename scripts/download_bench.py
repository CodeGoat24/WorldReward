#!/usr/bin/env python3
"""Download WorldReward-Bench from the Hugging Face Hub.

    python scripts/download_bench.py --output-dir data/WorldReward-Bench

Downloads ``bench.jsonl`` plus the paired videos and source images.
Resumable: re-running fetches only what is missing.

Pass ``--videos-only`` to skip the ``*_overlay.mp4`` files, which are a
visualisation aid (camera-trajectory annotations burned into the frames) and are
not read by the evaluation pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_REPO = "CodeGoat24/WorldReward-Bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/WorldReward-Bench", help="Local destination directory")
    parser.add_argument("--repo-id", default=BENCH_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--revision", default=None, help="Branch, tag or commit sha")
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Skip *_overlay.mp4 (visualisation only, not used for evaluation)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch only bench.jsonl and README.md, no media",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 1

    ignore_patterns: list[str] | None = None
    if args.metadata_only:
        ignore_patterns = ["videos/*"]
    elif args.videos_only:
        ignore_patterns = ["videos/*/*_overlay.mp4"]

    output_dir = Path(args.output_dir).resolve()
    print(f"downloading {args.repo_id} -> {output_dir}")
    if ignore_patterns:
        print(f"  skipping: {', '.join(ignore_patterns)}")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
        ignore_patterns=ignore_patterns,
        max_workers=args.max_workers,
    )

    bench_file = Path(path) / "bench.jsonl"
    if bench_file.is_file():
        n_pairs = sum(1 for line in bench_file.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"done: {n_pairs} pairs listed in {bench_file}")
    else:
        print(f"done: {path} (bench.jsonl not present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
