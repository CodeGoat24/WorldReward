#!/usr/bin/env python3
"""Score one video pair end to end -- the smallest complete example.

    python scripts/score_pair.py \
        --input-image  path/to/source.jpg \
        --left-video   path/to/candidate_a.mp4 \
        --right-video  path/to/candidate_b.mp4 \
        --caption      "A sunlit street lined with colorful buildings." \
        --actions      forward,forward,left+camera_down \
        --frames-per-action 8

Renders into a temporary directory unless ``--work-dir`` is given. Needs a GPU;
see the README for the vLLM requirement. Prefer the batch path
(``scripts/preprocess.py`` + ``scripts/run_inference.py``) for more than a
handful of pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldreward.aggregate import DIMENSIONS, collect_votes, predict_pair
from worldreward.render import render_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-image", required=True, help="Source image both videos were generated from")
    parser.add_argument("--left-video", required=True, help="Video to present on the left")
    parser.add_argument("--right-video", required=True, help="Video to present on the right")
    parser.add_argument("--caption", required=True, help="Short English description of the source scene")
    parser.add_argument(
        "--actions",
        required=True,
        help="Comma-separated camera-action tokens, one per action step "
        "(e.g. 'forward,forward,left+camera_down'). See the repository README.",
    )

    spans = parser.add_mutually_exclusive_group(required=True)
    spans.add_argument("--frames-per-action", type=int, help="Frames each action occupies")
    spans.add_argument("--segment-frames", help="Comma-separated per-action frame counts (unequal durations)")

    parser.add_argument("--model", default="CodeGoat24/WorldReward-qwen35-9b", help="HF repo id or local checkpoint")
    parser.add_argument("--left-name", default="left", help="Label for the left system in the output")
    parser.add_argument("--right-name", default="right", help="Label for the right system in the output")
    parser.add_argument("--work-dir", default=None, help="Keep intermediate renders here instead of a temp dir")
    parser.add_argument("--show-reasoning", action="store_true", help="Print the model's per-chunk analyses")
    return parser.parse_args()


def build_pair(args: argparse.Namespace) -> dict:
    pair = {
        "pair_id": "single_pair",
        "input_image": str(Path(args.input_image).resolve()),
        "input_caption": args.caption,
        "actions": [token.strip() for token in args.actions.split(",") if token.strip()],
        "left": {"video": str(Path(args.left_video).resolve()), "model": args.left_name},
        "right": {"video": str(Path(args.right_video).resolve()), "model": args.right_name},
    }
    if args.frames_per_action:
        pair["frames_per_action"] = args.frames_per_action
    else:
        pair["segment_frames"] = [int(v) for v in args.segment_frames.split(",") if v.strip()]
    return pair


def report(records: list[dict], pair: dict, args: argparse.Namespace) -> None:
    verdict = predict_pair(record.get("review_payload") or {} for record in records)
    votes = collect_votes(record.get("review_payload") or {} for record in records)

    names = {"left": args.left_name, "right": args.right_name, "tie": "tie", None: "no verdict"}
    print(f"\n{'=' * 62}\nVerdict  ({len(records)} chunks, "
          f"{sum(1 for r in records if r.get('parsed_ok'))} parsed)\n{'=' * 62}")
    tally = {
        "action": votes["action_overall"],
        "appearance": votes["appearance"],
        "motion": votes["motion"],
    }
    for dimension in DIMENSIONS:
        chunk_votes = tally[dimension]
        counts = f"A={chunk_votes.count('A')} B={chunk_votes.count('B')} Tie={chunk_votes.count('Tie')}"
        print(f"  {dimension:<11} {names[verdict[dimension]]:<14} ({counts})")
    print("\n  A = left, B = right. Votes are per chunk; appearance pools two")
    print("  sub-dimensions, so it has twice as many votes as the others.")

    if args.show_reasoning:
        for record in sorted(records, key=lambda r: r["chunk_id"]):
            payload = record.get("review_payload")
            print(f"\n{'-' * 62}\nchunk {record['chunk_id']}  actions: {', '.join(record['action_texts'])}")
            if payload is None:
                print(f"  FAILED TO PARSE: {record.get('parse_error')}")
                continue
            action_control = payload["action_control"]
            print(f"  action  -> {action_control['overall_winner']}: {action_control['overall_summary']}")
            for item in action_control["actions"]:
                print(f"    [{item['action_id']}] {item['action_label']} -> {item['winner']}")
            visual = payload["visual_quality"]
            print(f"  visual  -> {visual['overall_winner']}: {visual['overall_summary']}")
            for key in ("temporal_consistency", "dynamic_generation_quality", "artifacts_and_structure_integrity"):
                print(f"    {key} -> {visual[key]['winner']}")


def main() -> int:
    args = parse_args()
    for path in (args.input_image, args.left_video, args.right_video):
        if not Path(path).is_file():
            print(f"not found: {path}", file=sys.stderr)
            return 1

    pair = build_pair(args)
    if not pair["actions"]:
        print("--actions is empty", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as temporary:
        work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(temporary)
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"rendering {len(pair['actions'])} actions into chunks ...")
        chunks = render_pair(pair, work_dir / "rendered_chunks", force=True)
        print(f"  {len(chunks)} chunks x 6 images")

        from worldreward.infer import OfflineRunner

        print(f"loading {args.model} ...")
        runner = OfflineRunner(model_path=args.model)
        records = runner.run(chunks, output_path=work_dir / "chunk_predictions.json")

        report(records, pair, args)
        if args.work_dir:
            print(f"\nintermediates kept in {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
