#!/usr/bin/env python3
"""UnifiedReward-Think / -Flex baselines, against an OpenAI-compatible server.

Both are generative pairwise judges: one request shows frames from both videos
and the model names a winner per category. Serve the checkpoint first, e.g.

    vllm serve CodeGoat24/UnifiedReward-Think-qwen35-9b \\
        --served-model-name UnifiedReward --port 8080 \\
        --limit-mm-per-prompt '{"image": 16}' --max-model-len 32768

then

    python baselines/run_unified_reward.py --variant think \\
        --bench data/WorldReward-Bench/bench.jsonl --url http://127.0.0.1:8080

Axis mapping, from each model's own output schema:

============  =========================================  ============
``think``     ``Temporal coherence`` scores              motion
              ``Authenticity`` scores                    appearance
``flex``      category B (video quality / dynamics)       motion
              category C (narrative / aesthetics)         appearance
============  =========================================  ============

Neither is shown the commanded camera trajectory, so neither predicts
``action``.

Frames are 8 uniformly sampled per video, sent left first then right ("first
half" / "second half" in the prompt).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.common import (
    Pair,
    add_common_args,
    merge_shards,
    resolve_io,
    run_pairs,
    write_pairs_jsonl,
)
from baselines.run_quality_scorer import sample_frames

THINK_PROMPT = """You are an objective and precise evaluator for video quality comparison. I will provide you with a text caption and a sequence of consecutive frames extracted from two generated videos based on this caption. The first half of the frames belong to Video 1, and the second half of the frames belong to Video 2. You must analyze these two videos carefully and determine which video is better.

        Instructions (MUST follow strictly):
        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.
        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:
        - Video 1 is better
        - Video 2 is better
        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.

        Evaluation procedure:

        1. The caption for the generated videos is: \u300c{prompt}\u300d. The provided frames represent two candidate videos:
        - First half: Video 1
        - Second half: Video 2

        2. You must evaluate the two videos across these core dimensions:
        - Semantic consistency (how closely the video content aligns with the caption)
        - Temporal coherence (smoothness and logical flow of motion across frames)
        - Authenticity (realism and attention to detail)

        3. You may also add up to two additional evaluation dimensions if they are clearly relevant (e.g., camera stability, lighting consistency, creativity). If no extra dimensions are relevant, keep only the three core dimensions.

        4. For each evaluation dimension:
        - Provide a score between 1\u201310 for both Video 1 and Video 2.
        - Provide a short rationale for each score (2\u20135 short sentences).
        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:
            N. Dimension name:
                Video 1 (x/10) - rationale;
                Video 2 (y/10) - rationale

        5. After evaluating all dimensions, calculate the total score for each video and show the calculation explicitly, following this exact format:
            Total score:
            Video 1: x+x+x(+...)=total
            Video 2: y+y+y(+...)=total

        6. All reasoning, analysis, scoring, and totals must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.

        Required output format (follow this exactly, including line breaks and indentation):

        <think>
        1. Semantic consistency:
            Video 1 (9/10) - ...;
            Video 2 (7/10) - ...
        2. Temporal coherence:
            Video 1 (8/10) - ...;
            Video 2 (6/10) - ...
        3. Authenticity:
            Video 1 (7/10) - ...;
            Video 2 (5/10) - ...
        [Additional dimension if any]:
            Video 1 (8/10) - ...;
            Video 2 (6/10) - ...
        [Additional dimension if any]:
            Video 1 (7/10) - ...;
            Video 2 (7/10) - ...
        Total score:
        Video 1: 9+8+7+8+7=39
        Video 2: 7+6+5+6+7=31
        </think>
        <answer>Video 1 is better</answer>

        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given caption and the two provided videos (frames divided into two halves).
        """

FLEX_PROMPT = """## Identity
You are a top-tier AI Video Evaluation Expert. Perform a hierarchical, multi-dimensional comparative analysis of Video 1 and Video 2 based on the provided Prompt.

## Evaluation Framework

### 1. Mandatory Categories
For each, independently define **3-5 specific sub-dimensions** based on the videos' actual content:
- **A. Semantic Alignment & Accuracy**: Accuracy of subjects, attributes, spatial relationships, and environment as defined by the prompt.
- **B. Video Quality & Dynamic Realism**: Technical fidelity, temporal stability (no flickering/warping), subject identity persistence, and physical plausibility of motion.
- **C. Narrative, Aesthetics & Cinematography**: Composition, color harmony, camera movement quality (smoothness/intent), and narrative flow.
*Note: If the prompt involves unique traits, you are encouraged to add a personalized Category D.*

### 2. Core Rules
- **Dynamic Selection**: Do NOT simply copy a fixed list. Choose sub-dimensions that most effectively differentiate the two videos.
- **Sum-of-10 Scoring**: For every sub-dimension, the total score (Video 1 + Video 2) MUST strictly equal 10 points (e.g., 6+4, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis pointing to specific visual/temporal evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Video 1]
[Video 2]

## Output Format
Return a single, valid JSON object in English.

```json
{{
  "prompt": "[Original Prompt]",
  "categories": [
    {{
      "name": "[Category Name]",
      "dims": [
        {{
          "name": "[Custom Sub-dimension]",
          "reason_1": "[Specific evidence]",
          "reason_2": "[Specific evidence]",
          "score_1": 0-10,
          "score_2": 0-10
        }}
      ],
      "cat_reason": "[Category-level analysis]",
      "cat_winner": "Video 1/2"
    }}
  ],
  "reason": "[Overall analysis]",
  "winner": "Video 1/2"
}}
"""

def encode_frame(image, max_side: int, quality: int = 90) -> str:
    image = image.convert("RGB")
    width, height = image.size
    if max(width, height) > max_side:
        from PIL import Image

        scale = max_side / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _winner_side(text: str | None) -> str | None:
    """Map a free-text 'Video 1/2' winner string onto left/right."""
    if not text:
        return None
    lowered = text.lower()
    if "video 1" in lowered or "video1" in lowered or lowered.strip() in {"1", "a"}:
        return "left"
    if "video 2" in lowered or "video2" in lowered or lowered.strip() in {"2", "b"}:
        return "right"
    return None


def _compare(score_left: int | None, score_right: int | None) -> str | None:
    if score_left is None or score_right is None:
        return None
    if score_left > score_right:
        return "left"
    if score_right > score_left:
        return "right"
    return "tie"


# --- think parsing ----------------------------------------------------------
_SCORE_RE = re.compile(r"Video\s*([12])\s*\(\s*(\d+)\s*/\s*10\s*\)", re.IGNORECASE)


def _think_dimension(think_text: str, keywords: tuple[str, ...]) -> tuple[int | None, int | None]:
    """Pull the two ``Video k (n/10)`` scores belonging to one named dimension."""
    lines = think_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = re.match(r"\s*(\d+)\.\s*(.+?):", line)
        if match and any(k in match.group(2).strip().lower() for k in keywords):
            start = index
            break

    if start is None:
        # The model dropped the numbered-block format; fall back to the first
        # score pair that follows a mention of the dimension anywhere.
        for keyword in keywords:
            match = re.search(rf"{re.escape(keyword)}[:\s]", think_text, re.IGNORECASE)
            if match:
                found = _SCORE_RE.findall(think_text[match.end() : match.end() + 500])
                if len(found) >= 2:
                    scores = {v: int(s) for v, s in found[:2]}
                    return scores.get("1"), scores.get("2")
        return None, None

    block: list[str] = []
    for index in range(start, min(start + 8, len(lines))):
        if index > start and re.match(r"\s*(\d+)\.\s|Total score", lines[index], re.IGNORECASE):
            break
        block.append(lines[index])
    found = _SCORE_RE.findall("\n".join(block))
    if len(found) < 2:
        return None, None
    scores = {v: int(s) for v, s in found[:2]}
    return scores.get("1"), scores.get("2")


def parse_think(raw: str) -> dict[str, Any]:
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL | re.IGNORECASE)
    think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    think_text = think.group(1) if think else raw

    temporal = _think_dimension(think_text, ("temporal coherence", "temporal"))
    authenticity = _think_dimension(think_text, ("authenticity", "realism"))
    return {
        "overall": _winner_side(answer.group(1) if answer else None),
        "temporal_scores": temporal,
        "authenticity_scores": authenticity,
        "motion": _compare(*temporal),
        "appearance": _compare(*authenticity),
    }


# --- flex parsing -----------------------------------------------------------
def _flex_json(raw: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    bare = re.search(r"\{.*\}", raw, re.DOTALL)
    if bare:
        candidates.append(bare.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object in model output")


def parse_flex(raw: str) -> dict[str, Any]:
    payload = _flex_json(raw)
    winners: dict[str, str | None] = {"A": None, "B": None, "C": None}
    for category in payload.get("categories") or []:
        name = str(category.get("name", ""))
        match = re.match(r"^\s*([ABC])\.", name)
        if match:
            letter = match.group(1)
        else:
            lowered = name.lower()
            if "semantic" in lowered or "alignment" in lowered:
                letter = "A"
            elif "video quality" in lowered or "dynamic" in lowered or "realism" in lowered:
                letter = "B"
            elif "narrative" in lowered or "aesthetic" in lowered or "cinemato" in lowered:
                letter = "C"
            else:
                continue
        if winners.get(letter) is None:
            winners[letter] = _winner_side(category.get("cat_winner"))
    return {
        "overall": _winner_side(payload.get("winner")),
        "category_winners": winners,
        "motion": winners["B"],
        "appearance": winners["C"],
        "parsed": payload,
    }


VARIANTS = {
    "think": {"prompt": THINK_PROMPT, "parse": parse_think, "max_tokens": 3072},
    "flex": {"prompt": FLEX_PROMPT, "parse": parse_flex, "max_tokens": 2048},
}


def make_score_one(args, session):
    spec = VARIANTS[args.variant]
    url = args.url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = f"{url}/v1/chat/completions"

    def score_one(pair: Pair) -> dict[str, Any]:
        left = sample_frames(pair.left_video, args.num_frames)
        right = sample_frames(pair.right_video, args.num_frames)
        # Video 1 == left. If this order ever changes, every verdict flips.
        frames = left + right
        assert len(frames) == len(left) + len(right)

        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_frame(frame, args.max_side)}"
                },
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": spec["prompt"].format(prompt=pair.caption)})

        response = session.post(
            url,
            json={
                "model": args.served_model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": args.max_tokens or spec["max_tokens"],
                "temperature": 0.0,
            },
            timeout=args.request_timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]

        parsed = spec["parse"](raw)
        record: dict[str, Any] = {
            "variant": args.variant,
            "action": None,  # not modelled: the judge never sees the trajectory
            "appearance": parsed.get("appearance"),
            "motion": parsed.get("motion"),
            "overall": parsed.get("overall"),
        }
        if args.keep_raw:
            record["raw_output"] = raw
        return record

    return score_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    add_common_args(parser, default_output="outputs/baselines/unified_reward.pairs.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--served-model-name", default="UnifiedReward")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames per video")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = per-variant default")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--keep-raw", action="store_true", help="Store the raw generation in the cache"
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

    import requests

    session = requests.Session()
    for _ in run_pairs(pairs, make_score_one(args, session), cache, label=args.variant):
        pass

    count = write_pairs_jsonl(output, cache.records())
    print(f"wrote {output} ({count} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
