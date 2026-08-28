"""Shared plumbing for the baseline predictors.

Every baseline reads ``bench.jsonl``, emits one ``left``/``right``/``tie``
verdict per pair per axis, and writes a ``*.pairs.jsonl`` that
``worldreward-bench/score.py`` consumes directly -- so baselines and
WorldReward are scored by the same code on the same denominators.

Two families of baseline:

*score-based*
    produce a scalar per video (HPSv3, Aesthetic, VideoAlign, DAv3). The
    verdict comes from the sign of ``score_left - score_right``, with a
    tie band of ``eps``.
*preference-based*
    compare the two videos in one forward pass and name a winner directly
    (UnifiedReward-Think / -Flex).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

AXES = ("action", "appearance", "motion")
SIDES = ("left", "right")


@dataclass
class Pair:
    """One benchmark pair, with video paths resolved against the data root."""

    pair_id: str
    caption: str
    left_video: Path
    right_video: Path
    actions: list[str]
    frames_per_action: int
    num_frames: int
    raw: dict[str, Any]

    def video(self, side: str) -> Path:
        return self.left_video if side == "left" else self.right_video


def load_bench(bench_path: Path, data_root: Path | None = None) -> list[Pair]:
    """Read ``bench.jsonl``. Video paths inside it are relative to its parent."""
    root = data_root or bench_path.parent
    pairs: list[Pair] = []
    missing = 0
    for line in bench_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        left = root / row["left"]["video"]
        right = root / row["right"]["video"]
        if not left.is_file() or not right.is_file():
            missing += 1
            continue
        pairs.append(
            Pair(
                pair_id=row["pair_id"],
                caption=row.get("input_caption") or "",
                left_video=left,
                right_video=right,
                actions=row.get("actions") or [],
                frames_per_action=row.get("frames_per_action") or 0,
                num_frames=row.get("num_frames") or 0,
                raw=row,
            )
        )
    if missing:
        raise SystemExit(
            f"{missing} pairs in {bench_path} have no video on disk under {root}. "
            "Did the dataset download finish?"
        )
    return pairs


def shard(pairs: list[Pair], shard_id: int, num_shards: int) -> list[Pair]:
    """Round-robin split, so every shard sees a similar mix of pair lengths."""
    if num_shards < 1 or not (0 <= shard_id < num_shards):
        raise SystemExit(f"invalid shard {shard_id}/{num_shards}")
    return [p for i, p in enumerate(pairs) if i % num_shards == shard_id]


def verdict_from_scores(
    score_left: float, score_right: float, eps: float = 0.0
) -> str:
    """Sign of the score gap, with ``eps`` as a symmetric tie band."""
    delta = score_left - score_right
    if delta > eps:
        return "left"
    if delta < -eps:
        return "right"
    return "tie"


class Cache:
    """Append-only per-pair JSONL cache, so a killed run resumes for free.

    One line per pair. The line is written after the pair is fully scored, so a
    truncated final line (process killed mid-write) is dropped on reload rather
    than being silently treated as a result.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.done[record["pair_id"]] = record

    def pending(self, pairs: Iterable[Pair]) -> list[Pair]:
        return [p for p in pairs if p.pair_id not in self.done]

    def append(self, record: dict[str, Any]) -> None:
        self.done[record["pair_id"]] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[dict[str, Any]]:
        return [self.done[k] for k in sorted(self.done)]


def write_pairs_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write the axis verdicts in the layout ``score.py`` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "pair_id": record["pair_id"],
                        **{axis: record.get(axis) for axis in AXES},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(records)


def merge_shards(cache_paths: Iterable[Path], output: Path) -> int:
    """Combine per-shard caches into one ``*.pairs.jsonl``."""
    merged: dict[str, dict[str, Any]] = {}
    for path in cache_paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            merged[record["pair_id"]] = record
    return write_pairs_jsonl(output, [merged[k] for k in sorted(merged)])


def run_pairs(
    pairs: list[Pair],
    score_one: Callable[[Pair], dict[str, Any]],
    cache: Cache,
    *,
    label: str,
) -> Iterator[dict[str, Any]]:
    """Score pairs one at a time, caching each result and surviving failures.

    A pair that raises is reported and skipped rather than killing the run;
    ``score.py`` reports skipped pairs as missing.
    """
    pending = cache.pending(pairs)
    print(f"[{label}] {len(pending)} pairs to score ({len(cache.done)} cached)", flush=True)
    failures = 0
    for index, pair in enumerate(pending, start=1):
        try:
            record = score_one(pair)
        except Exception as exc:  # noqa: BLE001 - one bad pair must not end the run
            failures += 1
            print(f"[{label}] {pair.pair_id} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        record["pair_id"] = pair.pair_id
        cache.append(record)
        yield record
        if index % 20 == 0 or index == len(pending):
            print(f"[{label}] {index}/{len(pending)} (failed {failures})", flush=True)
    if failures:
        print(f"[{label}] {failures} pairs failed and are absent from the output", flush=True)


def add_common_args(parser, *, default_output: str) -> None:
    parser.add_argument("--bench", required=True, help="Path to bench.jsonl")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root the video paths in bench.jsonl resolve against (default: its parent)",
    )
    parser.add_argument("--output", default=default_output, help="Where to write *.pairs.jsonl")
    parser.add_argument(
        "--cache",
        default=None,
        help="Per-pair JSONL cache with full scores (default: <output>.cache.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only score the first N pairs")
    parser.add_argument(
        "--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0"))
    )
    parser.add_argument(
        "--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1"))
    )


def resolve_io(args) -> tuple[list[Pair], Cache, Path]:
    """Shared setup: load, shard, and open the cache."""
    bench_path = Path(args.bench).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else None
    pairs = load_bench(bench_path, data_root)
    if args.limit:
        pairs = pairs[: args.limit]
    pairs = shard(pairs, args.shard_id, args.num_shards)

    output = Path(args.output).resolve()
    if args.num_shards > 1:
        output = output.with_suffix(f".shard{args.shard_id}of{args.num_shards}.jsonl")
    cache_path = Path(args.cache).resolve() if args.cache else output.with_suffix(".cache.jsonl")
    return pairs, Cache(cache_path), output
