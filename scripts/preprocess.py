#!/usr/bin/env python3
"""Render video pairs into the 6-image chunks WorldReward consumes.

    python scripts/preprocess.py \
        --pairs my_pairs.jsonl \
        --output-root outputs/rendered_chunks

Each input line describes one pair; see the repository README for the schema. Relative paths inside a record resolve against ``--base-dir`` (defaults
to the directory containing the pairs file), so a downloaded benchmark snapshot
works without rewriting any paths.

Already-rendered pairs are skipped unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldreward.render import render_pair


def read_pairs(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True, help="JSONL file of pair records")
    parser.add_argument("--output-root", required=True, help="Directory to write rendered chunks into")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Root that relative paths inside the pairs file resolve against "
        "(default: the directory containing --pairs)",
    )
    parser.add_argument("--pair-id", action="append", default=[], help="Render only this pair_id (repeatable)")
    parser.add_argument("--limit", type=int, default=0, help="Render at most N pairs (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-render pairs that are already complete")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Rendering is CPU-bound (video decode + matplotlib), "
        "so this scales well on multi-core machines.",
    )
    return parser.parse_args()


def _render_one(task: tuple[dict[str, Any], str, str, bool]) -> tuple[str, int, str | None]:
    pair, output_root, base_dir, force = task
    try:
        chunks = render_pair(pair, output_root, base_dir=base_dir, force=force)
        return pair["pair_id"], len(chunks), None
    except Exception:
        return pair.get("pair_id", "<unknown>"), 0, traceback.format_exc(limit=3)


def main() -> int:
    args = parse_args()
    pairs_path = Path(args.pairs).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else pairs_path.parent
    output_root = Path(args.output_root).resolve()

    pairs = read_pairs(pairs_path)
    if args.pair_id:
        wanted = set(args.pair_id)
        pairs = [p for p in pairs if p["pair_id"] in wanted]
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        print("no pairs selected", file=sys.stderr)
        return 1

    print(f"rendering {len(pairs)} pairs -> {output_root} (base_dir={base_dir})")
    tasks = [(pair, str(output_root), str(base_dir), args.force) for pair in pairs]

    results: list[tuple[str, int, str | None]] = []
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(processes=args.workers) as pool:
            for index, result in enumerate(pool.imap_unordered(_render_one, tasks), start=1):
                results.append(result)
                if index % 25 == 0 or index == len(tasks):
                    print(f"  {index}/{len(tasks)} pairs done", flush=True)
    else:
        for index, task in enumerate(tasks, start=1):
            results.append(_render_one(task))
            if index % 25 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} pairs done", flush=True)

    failures = [(pid, err) for pid, _, err in results if err]
    total_chunks = sum(count for _, count, err in results if not err)
    print(
        json.dumps(
            {
                "pairs_rendered": len(results) - len(failures),
                "pairs_failed": len(failures),
                "chunks_rendered": total_chunks,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    for pid, err in failures[:10]:
        print(f"\n[failed] {pid}\n{err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
