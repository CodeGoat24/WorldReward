#!/usr/bin/env python3
"""Reproduce the WorldReward-Bench evaluation end to end.

    python worldreward-bench/run_bench.py --model CodeGoat24/WorldReward-qwen35-9b

Runs four stages -- download, preprocess, inference, score -- each of which is
independently resumable, so an interrupted run continues where it stopped. Use
``--stage`` to run one stage at a time, e.g. to render on a CPU machine and infer
on a GPU one.

Rendering all 760 pairs is CPU-bound; raise ``--workers`` on a many-core machine.
Inference is GPU-bound: for multi-GPU throughput, run the inference stage once
per GPU with ``CUDA_VISIBLE_DEVICES`` set and ``--shard i --num-shards N``; the
score stage merges the per-shard prediction files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STAGES = ("download", "preprocess", "infer", "score")


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}\n", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"stage failed with exit code {result.returncode}")


def shard_pairs(bench_path: Path, output_path: Path, shard: int, num_shards: int) -> Path:
    """Write the subset of ``bench.jsonl`` belonging to one shard.

    Pairs are assigned round-robin over the sorted pair ids, so every shard gets
    a comparable mix of trajectory types and the split is stable across runs.
    """
    records = [
        json.loads(line)
        for line in bench_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda r: r["pair_id"])
    selected = records[shard::num_shards]
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in selected) + "\n",
        encoding="utf-8",
    )
    print(f"shard {shard}/{num_shards}: {len(selected)}/{len(records)} pairs -> {output_path}")
    return output_path


def shard_ids_path(pairs_path: Path, work_dir: Path, args: argparse.Namespace) -> Path:
    """Write this shard's pair ids, one per line, for ``--pair-id-file``."""
    ids = [
        json.loads(line)["pair_id"]
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path = work_dir / f"shard{args.shard}of{args.num_shards}.ids.txt"
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return path


def merge_predictions(work_dir: Path, merged_path: Path) -> Path:
    """Combine per-shard prediction files into one, keyed by pair and chunk.

    Later files win on collision, which only happens if shard assignments changed
    between runs; identical reruns are idempotent.
    """
    shards = sorted(work_dir.glob("chunk_predictions.shard*of*.json"))
    if not shards:
        return merged_path
    combined: dict[tuple[str, int], dict] = {}
    for path in shards:
        for record in json.loads(path.read_text(encoding="utf-8")):
            combined[(record["pair_id"], record["chunk_id"])] = record
    records = [combined[key] for key in sorted(combined)]
    merged_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    print(f"merged {len(shards)} shard files -> {merged_path} ({len(records)} chunks)")
    return merged_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="CodeGoat24/WorldReward-qwen35-9b", help="HF repo id or local checkpoint")
    parser.add_argument("--work-dir", default="outputs/bench", help="Directory for renders and predictions")
    parser.add_argument("--bench-dir", default="data/WorldReward-Bench", help="Where the dataset lives")
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        default=[],
        help="Run only these stages (repeatable). Default: all four.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel render workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per inference batch")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N pairs (smoke test)")
    parser.add_argument("--shard", type=int, default=0, help="This shard's index, for multi-GPU runs")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards, for multi-GPU runs")
    parser.add_argument("--name", default=None, help="Label for the report (default: derived from --model)")
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Download only the videos needed for evaluation, not the overlay visualisations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = args.stage or list(STAGES)
    work_dir = Path(args.work_dir).resolve()
    bench_dir = Path(args.bench_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    bench_path = bench_dir / "bench.jsonl"
    render_root = work_dir / "rendered_chunks"
    predictions_path = work_dir / "chunk_predictions.json"
    if args.num_shards > 1:
        # The cache is rewritten whole on each flush, so shards need separate files.
        predictions_path = work_dir / f"chunk_predictions.shard{args.shard}of{args.num_shards}.json"
    report_path = work_dir / "report.md"
    metrics_path = work_dir / "metrics.json"

    if "download" in stages:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/download_bench.py"),
            "--output-dir", str(bench_dir),
        ]
        if args.skip_overlays:
            command.append("--videos-only")
        run(command)

    if not bench_path.is_file():
        raise SystemExit(f"{bench_path} not found -- run the download stage first")

    # Multi-GPU runs render and infer only their own slice of the benchmark.
    pairs_path = bench_path
    if args.num_shards > 1:
        pairs_path = shard_pairs(
            bench_path, work_dir / f"shard{args.shard}_of_{args.num_shards}.jsonl", args.shard, args.num_shards
        )

    if "preprocess" in stages:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/preprocess.py"),
            "--pairs", str(pairs_path),
            "--base-dir", str(bench_dir),
            "--output-root", str(render_root),
            "--workers", str(args.workers),
        ]
        if args.limit:
            command += ["--limit", str(args.limit)]
        run(command)

    if "infer" in stages:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_inference.py"),
            "--render-root", str(render_root),
            "--output", str(predictions_path),
            "--model", args.model,
            "--batch-size", str(args.batch_size),
        ]
        if args.num_shards > 1:
            # All shards render into one root, so restrict this process to its own
            # pairs; otherwise every shard would re-infer the whole benchmark.
            command += ["--pair-id-file", str(shard_ids_path(pairs_path, work_dir, args))]
        run(command)

    if "score" in stages:
        name = args.name or Path(args.model).name
        # Score the whole benchmark, gathering every shard's predictions.
        scored_path = merge_predictions(work_dir, work_dir / "chunk_predictions.json")
        run([
            sys.executable,
            str(REPO_ROOT / "worldreward-bench/score.py"),
            "--bench", str(bench_path),
            "--predictions", str(scored_path),
            "--name", name,
            "--output-md", str(report_path),
            "--output-json", str(metrics_path),
        ])
        print(f"\nreport: {report_path}\nmetrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
