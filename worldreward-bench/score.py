#!/usr/bin/env python3
"""Score predictions against WorldReward-Bench ground truth.

    python worldreward-bench/score.py \
        --bench data/WorldReward-Bench/bench.jsonl \
        --predictions outputs/chunk_predictions.json

Accepts either chunk-level predictions (``chunk_predictions.json``, which get
aggregated here) or already-aggregated pair-level predictions
(``*.pairs.jsonl``). Reports overall accuracy plus trajectory-group and style
slices, in two denominators:

``all``
    every pair in the benchmark. A pair whose ground truth is ``tie`` is counted
    correct only if the predictor also says ``tie``.
``non-tie``
    only pairs whose ground truth is ``left`` or ``right``. Removes the tie
    calibration question and isolates directional accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldreward.aggregate import DIMENSIONS, predict_pair

TRAJECTORY_GROUPS = ("pure_translation", "pure_rotation", "compound")
STYLES = ("photo", "game_anime", "traditional")
VALID_CHOICES = frozenset({"left", "right", "tie"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_predictions(path: Path) -> dict[str, dict[str, str | None]]:
    """Load pair-level predictions from either supported prediction format."""
    if path.suffix == ".jsonl":
        return {
            record["pair_id"]: {dimension: record.get(dimension) for dimension in DIMENSIONS}
            for record in read_jsonl(path)
        }

    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("chunks") or records.get("records") or []
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record.get("review_payload") or {})
    return {pair_id: predict_pair(reviews) for pair_id, reviews in by_pair.items()}


class Tally:
    """Correct/total counters for one slice, in both denominators."""

    def __init__(self) -> None:
        self.counts = {
            dimension: {"all": [0, 0], "non-tie": [0, 0]} for dimension in DIMENSIONS
        }

    def add(self, dimension: str, predicted: str | None, truth: str) -> None:
        correct = int(predicted == truth)
        bucket = self.counts[dimension]
        bucket["all"][0] += correct
        bucket["all"][1] += 1
        if truth != "tie":
            bucket["non-tie"][0] += correct
            bucket["non-tie"][1] += 1

    def accuracy(self, dimension: str, mode: str) -> tuple[int, int, float | None]:
        correct, total = self.counts[dimension][mode]
        return correct, total, (correct / total if total else None)


def evaluate(
    bench: list[dict[str, Any]],
    predictions: dict[str, dict[str, str | None]],
) -> tuple[dict[str, Tally], list[str]]:
    """Tally accuracy over the whole benchmark and each slice.

    Pairs present in the benchmark but absent from the predictions are reported
    as missing and excluded from every denominator, so a partial run yields
    honest numbers over the subset it actually covered.
    """
    slices: dict[str, Tally] = {"overall": Tally()}
    missing: list[str] = []

    # A dimension a predictor never answers is "not modelled" and reported as
    # not-applicable. A missing answer on a *particular* pair, for a dimension
    # the predictor does model, counts as wrong and stays in the denominator.
    modelled = {
        dimension
        for dimension in DIMENSIONS
        if any(prediction.get(dimension) is not None for prediction in predictions.values())
    }

    for pair in bench:
        pair_id = pair["pair_id"]
        prediction = predictions.get(pair_id)
        if prediction is None:
            missing.append(pair_id)
            continue

        keys = [
            "overall",
            f"trajectory_group:{pair['trajectory_group']}",
            f"style:{pair['style']}",
        ]
        for key in keys:
            slices.setdefault(key, Tally())

        for dimension in DIMENSIONS:
            if dimension not in modelled:
                continue
            truth = (pair.get("label") or {}).get(dimension)
            if truth not in VALID_CHOICES:
                continue
            for key in keys:
                slices[key].add(dimension, prediction.get(dimension), truth)

    return slices, missing


def format_cell(correct: int, total: int, accuracy: float | None) -> str:
    if accuracy is None:
        return "--"
    return f"{accuracy * 100:.2f}% ({correct}/{total})"


def render_table(title: str, rows: list[tuple[str, Tally]], mode: str) -> str:
    lines = [f"### {title}", "", "| Slice | Action | Appearance | Motion |", "|---|---:|---:|---:|"]
    for label, tally in rows:
        cells = [format_cell(*tally.accuracy(dimension, mode)) for dimension in DIMENSIONS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(slices: dict[str, Tally], name: str, missing: list[str]) -> str:
    parts: list[str] = [f"# WorldReward-Bench results: {name}", ""]

    if missing:
        parts += [
            f"> **Warning:** {len(missing)} benchmark pairs had no prediction and were "
            f"excluded from all denominators (e.g. `{missing[0]}`). "
            f"Numbers below cover only the scored subset.",
            "",
        ]

    for mode, blurb in (
        ("all", "All pairs. A `tie` ground truth counts as correct only on a `tie` prediction."),
        ("non-tie", "Pairs whose ground truth is `left` or `right`."),
    ):
        parts += [f"## {mode}", "", blurb, ""]
        parts.append(render_table("Overall", [("overall", slices["overall"])], mode))
        parts.append("")

        group_rows = [
            (group, slices[key])
            for group in TRAJECTORY_GROUPS
            if (key := f"trajectory_group:{group}") in slices
        ]
        if group_rows:
            parts += [render_table("By trajectory group", group_rows, mode), ""]

        style_rows = [
            (style, slices[key]) for style in STYLES if (key := f"style:{style}") in slices
        ]
        if style_rows:
            parts += [render_table("By style", style_rows, mode), ""]

    return "\n".join(parts).rstrip() + "\n"


def build_summary(slices: dict[str, Tally], name: str, missing: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "pairs_missing": len(missing),
        "slices": {
            key: {
                dimension: {
                    mode: {
                        "correct": (result := tally.accuracy(dimension, mode))[0],
                        "total": result[1],
                        "accuracy": round(result[2], 6) if result[2] is not None else None,
                    }
                    for mode in ("all", "non-tie")
                }
                for dimension in DIMENSIONS
            }
            for key, tally in slices.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True, help="Path to bench.jsonl")
    parser.add_argument(
        "--predictions",
        required=True,
        help="chunk_predictions.json, or an aggregated *.pairs.jsonl",
    )
    parser.add_argument("--name", default=None, help="Label for the report (default: predictions filename)")
    parser.add_argument("--output-md", default=None, help="Also write the markdown report here")
    parser.add_argument("--output-json", default=None, help="Also write machine-readable metrics here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bench_path = Path(args.bench).resolve()
    predictions_path = Path(args.predictions).resolve()

    bench = read_jsonl(bench_path)
    predictions = load_predictions(predictions_path)
    name = args.name or predictions_path.stem

    slices, missing = evaluate(bench, predictions)
    report = render_report(slices, name, missing)
    print(report)

    if args.output_md:
        Path(args.output_md).write_text(report, encoding="utf-8")
        print(f"wrote {args.output_md}", file=sys.stderr)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(build_summary(slices, name, missing), indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
