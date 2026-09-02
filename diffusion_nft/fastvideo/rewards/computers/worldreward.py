"""VLM-based pairwise reward model backend.

Uses an external vLLM server (Qwen3.5 fine-tuned, served as ``WorldReward``)
to compare pairs of generated videos and produce per-candidate win-rate
rewards for action_control and visual_quality.

The prompt template is the ``medium_v1`` one the WorldReward model was
fine-tuned on (see https://github.com/CodeGoat24/WorldReward). Image
construction (6 images per pair: input_image, frame_grid, action1..4) uses
the slot-indexed layout defined locally in this module.

No model is loaded onto the training GPU – all inference happens via HTTP.
"""
from __future__ import annotations

import base64
import io
import itertools
import json
import logging
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter

# Image-building utilities for the RM evaluation 6-image layout. These were
# inlined here rather than imported, to keep this module self-contained
# so the reward path has no dependency outside fastvideo. We keep the image
# layout and chunk/frame mapping (chunk0 with leading IDLE; VAE temporal
# stride 4) identical to the RM training setup so prompt-side inputs match the
# model's training distribution.
#
# NOTE: the WorldReward training data uses a slot-indexed
# layout where each of the 4 action groups in the chunk renders 3 frames in
# order with its own border color and a centered "Ak: <label>" title above the
# middle frame. save_frame_grid (defined below) replicates that.
ACTIONS_PER_GROUP = 4
ACTION_COLORS = [(210, 60, 60), (60, 170, 60), (60, 100, 210), (200, 140, 0)]
_ACTION_COLORS_F = [tuple(c / 255.0 for c in rgb) for rgb in ACTION_COLORS]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _chunk_frame_pairs(chunk_id, chunk_size=4):
    """Return (vf_start, vf_end) for each action in the chunk.

    Video frame mapping (VAE temporal stride=4, first latent → 1 frame):
      latent 0 → frame 0 (IDLE, single frame)
      latent k → frames (k-1)*4+1 .. k*4   for k >= 1
    """
    pairs = []
    for i in range(chunk_size):
        latent_idx = chunk_id * chunk_size + i
        if latent_idx == 0:
            pairs.append((0, 0))  # IDLE: repeat frame 0
        else:
            pairs.append(((latent_idx - 1) * 4 + 1, latent_idx * 4))
    return pairs


def _frame_ar(*frames) -> "float | None":
    """Best-effort H/W aspect ratio from the first available frame."""
    for f in frames:
        if f is not None:
            h, w = f.shape[:2]
            if w > 0:
                return float(h) / float(w)
    return None


def _avg_ar(ar_a, ar_b, default: float = 480 / 832, cap: float = 0.85) -> float:
    """Average of two row aspect ratios, capped to avoid overly tall figures.

    Mirrors the WorldReward data pipeline so reward-model inputs match the
    training distribution for non-832x480 videos.
    """
    vals = [v for v in (ar_a, ar_b) if v is not None and v > 0]
    if not vals:
        return min(default, cap)
    return min(sum(vals) / len(vals), cap)


def save_action_pair(a_frames, b_frames, local_action_idx, action_text, vf_start, vf_end, out_path):
    acolor_f = _ACTION_COLORS_F[local_action_idx]
    # Adaptive cell height from the two rows' real aspect ratios (capped 0.85),
    # matching the WorldReward pair renderer (aspect="equal").
    ar_a = _frame_ar(a_frames.get(vf_start), a_frames.get(vf_end))
    ar_b = _frame_ar(b_frames.get(vf_start), b_frames.get(vf_end))
    IMG_AR = _avg_ar(ar_a, ar_b)
    CELL_W = 6.0
    CELL_H = CELL_W * IMG_AR
    fig_w = 2 * CELL_W + 0.7
    fig_h = 2 * CELL_H + 0.22

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.03})
    fig.suptitle(f"Action {local_action_idx+1}: {action_text}",
                 fontsize=22, fontweight="bold", color=acolor_f, y=1.0)

    row_labels = ["Video A", "Video B"]
    row_colors = ["#1a4fa0", "#c0281a"]
    data = [
        [a_frames.get(vf_start), a_frames.get(vf_end)],
        [b_frames.get(vf_start), b_frames.get(vf_end)],
    ]

    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            img = data[row][col]
            if img is not None:
                ax.imshow(img, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(acolor_f); sp.set_linewidth(3.0)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=13, fontweight="bold",
                              color=row_colors[row], labelpad=4)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_frame_grid(
    a_frames: dict[int, "np.ndarray"],
    b_frames: dict[int, "np.ndarray"],
    action_texts: list[str],
    out_buf,
    key_vfs: list[int],
    ref_vf: int,
) -> None:
    """Render the 2 x (1 + 4*3) frame grid in WorldReward training format.

    Layout per row: [Start | A1 frame_s, A1 frame_m, A1 frame_e | A2 ... | A4 ...].
    Action group k (slot 0..3) gets ACTION_COLORS[k] border, and the middle
    frame of row 0 carries a "Ak: <action_texts[k]>" title.

    `key_vfs` is the flat list of 12 video-frame indices, in groups of 3
    per slot. `ref_vf` is the leftmost "Start" reference frame index.
    """
    if len(key_vfs) != ACTIONS_PER_GROUP * 3:
        raise ValueError(
            f"expected {ACTIONS_PER_GROUP * 3} key_vfs, got {len(key_vfs)}"
        )

    def _row_ar(fd: dict) -> "float | None":
        # Prefer the Start/ref frame, then any key frame, matching build _row_ar.
        cand = fd.get(ref_vf)
        if cand is None:
            for vf in key_vfs:
                if vf in fd:
                    cand = fd[vf]
                    break
        if cand is None:
            return None
        h, w = cand.shape[:2]
        return float(h) / float(w) if w > 0 else None

    img_ar = _avg_ar(_row_ar(a_frames), _row_ar(b_frames))
    cell_w = 2.0
    cell_h = cell_w * img_ar
    n_actions = ACTIONS_PER_GROUP
    frames_per_action = 3
    n_cols = 1 + n_actions * frames_per_action
    n_rows = 2
    fig_w = n_cols * cell_w + 0.4
    fig_h = n_rows * cell_h + 0.38

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.025, "hspace": 0.05},
    )

    row_labels = ["Video A", "Video B"]
    row_colors = ["#1a4fa0", "#c0281a"]
    rows = [a_frames, b_frames]

    for row in range(n_rows):
        fd = rows[row]
        # Col 0: Start reference frame.
        ax = axes[row, 0]
        if ref_vf in fd:
            ax.imshow(fd[ref_vf], aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#999999"); sp.set_linewidth(2.5)
        ax.set_ylabel(
            row_labels[row], fontsize=13, fontweight="bold",
            color=row_colors[row], labelpad=6,
        )
        if row == 0:
            ax.set_title("Start", fontsize=10, color="#888888", pad=3)

        # Cols 1..: 4 action groups x 3 frames each.
        for slot_idx in range(n_actions):
            color_f = _ACTION_COLORS_F[slot_idx % len(_ACTION_COLORS_F)]
            for sub in range(frames_per_action):
                col = 1 + slot_idx * frames_per_action + sub
                vf = key_vfs[slot_idx * frames_per_action + sub]
                ax = axes[row, col]
                if vf in fd:
                    ax.imshow(fd[vf], aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(color_f); sp.set_linewidth(2.5)
                # Title on top of the middle frame in row 0 only.
                if row == 0 and sub == 1:
                    ax.set_title(
                        f"A{slot_idx + 1}: {action_texts[slot_idx]}",
                        fontsize=10, color=color_f, fontweight="bold", pad=3,
                    )

    fig.savefig(out_buf, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prompt template — must match the WorldReward SFT format (medium_v1)
# ---------------------------------------------------------------------------
#
# Source of truth: the WorldReward repository's training prompt template
# (https://github.com/CodeGoat24/WorldReward). Keep this string byte-identical to the
# template the RM was trained on. {caption} and {action_sequence} are filled
# via str.replace (not str.format) so the JSON braces inside the template
# don't need to be escaped.

PROMPT_TEMPLATE = """\
You are an expert evaluator of AI-generated world simulation videos. Compare Video A and Video B from the 6 images and return one JSON review containing `action_control` and `visual_quality`.

========================================================================
PART I: SCENE CONTEXT
========================================================================

Original input image caption: "{caption}"

Commanded action sequence:
{action_sequence}

Both videos were generated from the same source image and the same action sequence.

========================================================================
PART II: IMAGE LAYOUT
========================================================================

You receive 6 images:
- Image 1: the original input image for the shared source scene.
- Image 2: frame-grid overview. Top row = Video A, bottom row = Video B. Each action spans start / mid / end key frames.
- Images 3-6: per-action detail for Actions 1-4. In each image, top row = Video A, bottom row = Video B, left column = action start, right column = action end.

Important chunk note:
- This chunk may start later in the full video, so the local start frame can differ from Image 1 because of earlier camera motion.
- Newly revealed content is normal if it follows plausible camera motion.
- Penalize source-scene drift only when the world identity itself changes incompatibly with Image 1.

========================================================================
PART III: ACTION REFERENCE
========================================================================

Actions use the form "<Translation> | <Rotation>".
Each primitive is described with TWO anchors: where EXISTING content goes, and where NEW content enters.

Translation reference:
- Forward: zoom in, objects larger.
- Backward: zoom out, objects smaller.
- Right: camera moves right; existing objects slide toward the LEFT edge, NEW content enters from the RIGHT.
- Left: camera moves left; existing objects slide toward the RIGHT edge, NEW content enters from the LEFT.
- IDLE: no translation.

Rotation reference:
- YawRight: camera turns right; existing objects swing toward the LEFT edge, NEW content enters from the RIGHT.
- YawLeft: camera turns left; existing objects swing toward the RIGHT edge, NEW content enters from the LEFT.
- PitchUp: camera tilts UP to look higher; existing subject sinks toward the BOTTOM and may exit there, NEW content enters at the TOP edge.
- PitchDown: camera tilts DOWN to look lower; existing subject rises toward the TOP and may exit there, NEW content enters at the BOTTOM edge.
- IDLE: no rotation.

Pitch reliable test (scene-independent): PitchUp => NEW content enters at the TOP edge, existing exits at the BOTTOM; PitchDown => NEW content enters at the BOTTOM edge, existing exits at the TOP. Sky-at-top / ground-at-bottom is only a hint when such regions exist; for indoor/close-up/abstract scenes judge purely by which EDGE reveals new content, not a vague up/down impression.

For EACH action, PART I already lists the explicit expected per-component visual behaviour and the strict correctness criterion. Use that per-action block as the authoritative reference; do not re-derive pitch/yaw direction against it.

Combined actions execute simultaneously. Motion should be visible but still subtle and temporally smooth, not a giant jump.

========================================================================
PART IV: TASK OVERVIEW
========================================================================

Return one review with two sections:
1. `action_control`
2. `visual_quality`

The two sections are conceptually separate, but their factual claims must stay mutually consistent.

GLOBAL CONSISTENCY RULES:
1. Different winners are allowed for action and visual.
2. If `action_control` says a video is static, near-static, frozen, or has no discernible camera motion, `dynamic_generation_quality` must not describe that same video as smooth, convincing, natural, or meaningful motion.
3. Large pixel change caused by warping, melting, hallucinated reshaping, or chaotic jumps is not good dynamics.
4. If the chunk start differs from Image 1 only because the camera has already moved, do not call that source-scene drift.
5. Source-scene drift means the world identity changes incompatibly: scene category change, key entity replacement, strong attribute or material change, or incompatible global layout.
6. Every analysis field must be non-empty. Every winner must be exactly `A`, `B`, or `Tie` where allowed, and exactly `A` or `B` where tie is forbidden.

========================================================================
PART V: ACTION CONTROL
========================================================================

Judge only camera motion: direction, magnitude, purity, temporal smoothness, and plausible 3D motion.
Do not judge blur, artifacts, realism, texture fidelity, or source-scene preservation here.

Action-control rules:
1. Evaluate each action independently. It is normal for A and B to split wins across actions.
2. Use Image 2 and Images 3-6 to judge motion. Do not treat mismatch between Image 1 and the local chunk start as an action failure.
3. A blurry or artifacted video can still have correct motion. Only motion evidence matters.
4. In action analyses, describe only observed motion. Good evidence includes statements like: "the building shifts right", "foreground objects grow larger", "new content appears on the left", or "No discernible camera motion."
5. Avoid visual-quality words in action analyses, including terms such as blur, artifact, distortion, degraded, smear, collapse, melted, fidelity, realistic, or unrealistic.
6. Use `Tie` only when motion execution is genuinely comparable or too ambiguous to separate.
7. `action_control.overall_winner` must be consistent with the per-action results. If one video wins more actions, it should normally win overall. If counts tie, use finer judgment on motion correctness, magnitude, and purity.
8. PART I gives, for every action, the expected per-component behaviour and strict correctness criterion. That block is authoritative; do not re-derive pitch/yaw direction against it.
9. For a COMPOUND action (more than one commanded component, e.g. translation plus rotation, or a Yaw+Pitch pair), each per-video analysis MUST address EVERY commanded component separately: one clause for the translation and one clause for EACH rotation sub-axis. Do not judge a compound action from only one component.
10. A video is CORRECT on an action only if (a) every commanded component matches its expected direction AND (b) there is no obvious un-commanded extra motion (e.g. zoom during a pure rotation, tilt during a pure translation, sideways drift during a pure pitch). Getting only one of several commanded components right is not correct.
11. Name any un-commanded extra motion explicitly, and let it count against that video when choosing the per-action winner. Prefer the video that satisfies more commanded components with fewer purity violations.

For each action, consider:
- Direction correctness: does the observed content motion match the expected direction of EACH commanded component in PART I? For compound actions, all components must match.
- Magnitude appropriateness: visible but not exaggerated; near-static usually means the action was not really executed.
- Motion purity (required check): name and penalize unwanted extra zoom, lateral drift, tilt, or rotation that was not commanded.
- Temporal smoothness: does the motion progress steadily rather than jumping or stalling?
- 3D plausibility: parallax, scale change, and scene shift should look like camera motion.

Action-analysis writing requirements:
- Each per-action analysis should usually be 1-2 sentences.
- Mention concrete motion evidence, not vague impressions.
- If motion is absent or too weak to judge, say so directly.
- `action_id` must be 1, 2, 3, 4 in order.
- `action_label` must be EXACTLY the bare label from the matching `Action N label (copy this exactly): ...` line, i.e. strictly `<Translation> | <Rotation>` (e.g. `Right | YawLeft`). Never include the indented expected-behaviour reference lines or any other text in `action_label`.

========================================================================
PART VI: VISUAL QUALITY
========================================================================

Judge only how good the generated video looks over the whole chunk.
Ignore action correctness as much as possible.

You must judge exactly these three sub-dimensions:
1. `temporal_consistency`
2. `dynamic_generation_quality`
3. `artifacts_and_structure_integrity`

Visual-quality rules:
1. Use Image 1 as the global source-scene reference.
2. The local chunk start does not need to match Image 1 pixel-by-pixel; later camera viewpoint is normal.
3. However, both videos should still look like the same underlying world as Image 1.
4. Penalize source-scene drift such as:
   - the scene category flips from indoor to outdoor, city to forest, beach to snowfield, desert to urban street, or any similarly incompatible world swap;
   - the main subject or key entity is replaced;
   - strong incompatible changes in object identity, material, color, texture, or semantics;
   - the chunk start already looks like a different world with no plausible camera-motion explanation.
5. Temporal consistency is mainly about frame-to-frame stability inside this chunk.
6. Source-scene drift belongs mainly under `artifacts_and_structure_integrity`, unless it also flickers or changes over time within the chunk.
7. A static or near-static video is a major penalty for `dynamic_generation_quality`, but it is not an automatic overall override if the moving video is catastrophically corrupted.
8. Do not reward destructive deformation as good dynamics.
9. `visual_quality.overall_winner` must respect sub-dimension evidence: a 3:0 lead must win overall; a 2:1 lead should normally win unless `overall_summary` explicitly explains a serious override reason.
10. If all three sub-dimensions tie, still choose `A` or `B` based on the smaller but concrete overall edge.

Sub-dimension guidance:

`temporal_consistency`
- Judge stability of appearance, lighting, texture, color, local details, and identity across frames.
- Penalize flicker, sudden jumps, unstable local details, disappearing or reappearing content, and frame-to-frame attribute changes that are not explained by smooth camera motion.
- If a video looks stable only because it barely moves at all, that stability does not automatically make it strong overall visual quality.

`dynamic_generation_quality`
- Judge whether the result feels like a genuinely generated video rather than a frozen image, shallow texture slide, or fake motion.
- Reward meaningful, smooth, coherent dynamic update.
- Penalize static output, near-static output, superficial 2D panning, mechanical sliding, and motion that is mostly destructive warping rather than true scene update.
- A video may contain lots of visible change and still lose this sub-dimension if the change is mostly collapse, melting, or hallucinated motion instead of coherent generation.

`artifacts_and_structure_integrity`
- Jointly judge visible artifacts, structural stability, and compatibility with the source scene.
- Penalize blur, ghosting, smearing, melting, tearing, duplicated parts, hallucinated fragments, identity drift, source-scene drift, structural collapse, and incompatible world changes.
- The video with fewer and less severe failures should win this sub-dimension.

Visual-analysis writing requirements:
- Each sub-dimension analysis should usually be 1-2 sentences per video.
- `overall_summary` should synthesize the three sub-dimensions explicitly.
- If overall winner does not follow a 2:1 majority, explain the override clearly in `overall_summary`.

========================================================================
PART VII: OUTPUT FORMAT
========================================================================

Return exactly one JSON object and nothing else.
No markdown, no surrounding explanation, no comments.

The JSON must have this structure:
{
  "action_control": {
    "actions": [
      {
        "action_id": 1,
        "action_label": "<exact bare action 1 label, strictly '<Translation> | <Rotation>'>",
        "video_a_analysis": "<for a COMPOUND action cover EVERY commanded component (translation + each rotation sub-axis), none omitted>",
        "video_b_analysis": "<for a COMPOUND action cover EVERY commanded component, none omitted>",
        "winner": "A|B|Tie"
      },
      {
        "action_id": 2,
        "action_label": "<exact bare action 2 label, strictly '<Translation> | <Rotation>'>",
        "video_a_analysis": "<for a COMPOUND action cover EVERY commanded component (translation + each rotation sub-axis), none omitted>",
        "video_b_analysis": "<for a COMPOUND action cover EVERY commanded component, none omitted>",
        "winner": "A|B|Tie"
      },
      {
        "action_id": 3,
        "action_label": "<exact bare action 3 label, strictly '<Translation> | <Rotation>'>",
        "video_a_analysis": "<for a COMPOUND action cover EVERY commanded component (translation + each rotation sub-axis), none omitted>",
        "video_b_analysis": "<for a COMPOUND action cover EVERY commanded component, none omitted>",
        "winner": "A|B|Tie"
      },
      {
        "action_id": 4,
        "action_label": "<exact bare action 4 label, strictly '<Translation> | <Rotation>'>",
        "video_a_analysis": "<for a COMPOUND action cover EVERY commanded component (translation + each rotation sub-axis), none omitted>",
        "video_b_analysis": "<for a COMPOUND action cover EVERY commanded component, none omitted>",
        "winner": "A|B|Tie"
      }
    ],
    "overall_summary": "...",
    "overall_winner": "A|B"
  },
  "visual_quality": {
    "temporal_consistency": {
      "video_a_analysis": "...",
      "video_b_analysis": "...",
      "winner": "A|B|Tie"
    },
    "dynamic_generation_quality": {
      "video_a_analysis": "...",
      "video_b_analysis": "...",
      "winner": "A|B|Tie"
    },
    "artifacts_and_structure_integrity": {
      "video_a_analysis": "...",
      "video_b_analysis": "...",
      "winner": "A|B|Tie"
    },
    "overall_summary": "...",
    "overall_winner": "A|B"
  }
}

Output constraints:
- Every required field must be present.
- Winner values must use only `A`, `B`, or `Tie` where tie is allowed.
- `action_control.overall_winner` and `visual_quality.overall_winner` must be only `A` or `B`.
- Do not omit `video_b_analysis` just because Video B is weaker.
- Write detailed English analyses rather than short labels only.
"""


# ---------------------------------------------------------------------------
# Per-component expected visual behaviour (mirrored from WorldReward
# api_eval/core/reward_eval_utils.py :: COMPONENT_EXPECTATION /
# expand_action_expectation). Kept inline so the training reward path has no
# cross-repo import dependency. Must stay byte-identical to the SFT export.
# ---------------------------------------------------------------------------

COMPONENT_EXPECTATION: dict[str, str] = {
    "Forward": "Forward (translation): the camera moves deeper in -> the scene zooms IN, foreground objects grow LARGER and detail at the center expands outward.",
    "Backward": "Backward (translation): the camera retreats -> the scene zooms OUT, objects shrink SMALLER and more surrounding context appears around the edges.",
    "Left": "Left (translation): the camera slides left -> the EXISTING objects slide toward the RIGHT edge (some exit on the right) while NEW content enters from the LEFT edge.",
    "Right": "Right (translation): the camera slides right -> the EXISTING objects slide toward the LEFT edge (some exit on the left) while NEW content enters from the RIGHT edge.",
    "YawLeft": "YawLeft (rotation): the camera turns left -> the EXISTING objects swing toward the RIGHT edge while NEW content is revealed entering from the LEFT edge.",
    "YawRight": "YawRight (rotation): the camera turns right -> the EXISTING objects swing toward the LEFT edge while NEW content is revealed entering from the RIGHT edge.",
    "PitchUp": "PitchUp (camera tilts UP to look higher): the camera's view rises, so the EXISTING subject sinks toward the BOTTOM (content near the bottom edge slides down and may exit), while NEW higher content appears entering from the TOP edge. KEY CHECK: new content enters at the TOP and existing content exits at the BOTTOM. (If the scene has sky/ceiling, more of it shows at the top; if it is indoor/close-up with no sky, still judge purely by: new stuff in at TOP, old stuff out at BOTTOM.)",
    "PitchDown": "PitchDown (camera tilts DOWN to look lower): the camera's view drops, so the EXISTING subject rises toward the TOP (content near the top edge slides up and may exit), while NEW lower content appears entering from the BOTTOM edge. KEY CHECK: new content enters at the BOTTOM and existing content exits at the TOP. (If the scene has ground/floor, more of it shows at the bottom; if it is indoor/close-up with no ground, still judge purely by: new stuff in at BOTTOM, old stuff out at TOP.)",
}


def _split_pretty_label(pretty_label: str) -> tuple[list[str], list[str]]:
    trans_part, _, rot_part = pretty_label.partition("|")
    trans_part = trans_part.strip()
    rot_part = rot_part.strip()

    def _atoms(part: str) -> list[str]:
        if not part or part.upper() == "IDLE":
            return []
        return [a.strip() for a in part.split("+") if a.strip() and a.strip().upper() != "IDLE"]

    return _atoms(trans_part), _atoms(rot_part)


def expand_action_expectation(pretty_label: str) -> list[str]:
    """Expand one action label into explicit per-component expected visuals plus
    a strict correctness criterion. Returns indented sub-lines for the prompt.
    """
    trans_atoms, rot_atoms = _split_pretty_label(pretty_label)
    commanded = trans_atoms + rot_atoms
    lines: list[str] = ["    (expected behaviour, reference only -- do NOT copy into action_label):"]

    if not commanded:
        lines.append("    - IDLE: no translation and no rotation. The scene CONTENT should stay essentially still.")
        lines.append("    - Correct only if there is NO discernible camera motion (no shift, no zoom, no tilt).")
        return lines

    for atom in commanded:
        desc = COMPONENT_EXPECTATION.get(atom, f"{atom}: (motion as named).")
        lines.append(f"    - {desc}")

    forbidden: list[str] = []
    has_depth = any(a in ("Forward", "Backward") for a in trans_atoms)
    has_lateral = any(a in ("Left", "Right") for a in trans_atoms)
    has_yaw = any(a in ("YawLeft", "YawRight") for a in rot_atoms)
    has_pitch = any(a in ("PitchUp", "PitchDown") for a in rot_atoms)
    if not has_depth:
        forbidden.append("no forward/backward zoom (objects must not grow or shrink)")
    if not has_lateral:
        forbidden.append("no left/right translation drift")
    if not has_yaw:
        forbidden.append("no left/right yaw turn")
    if not has_pitch:
        forbidden.append("no up/down pitch tilt")

    if len(commanded) > 1:
        lines.append(
            "    - Expected combined: ALL of the above components must be visible simultaneously; "
            "a video that performs only one of them is NOT fully correct."
        )
    crit = (
        "    - Counts as CORRECT only if EVERY commanded component above matches its direction"
    )
    if forbidden:
        crit += ", AND there is " + ", ".join(forbidden) + "."
    else:
        crit += "."
    lines.append(crit)
    return lines


def build_prompt_text(caption, action_texts: list[str]) -> str:
    """Render the WorldReward training prompt with the current chunk's actions.

    Uses str.replace (not str.format) so the JSON braces inside the template
    body don't need to be escaped. Each action is followed by its indented
    per-component expected-behaviour block from ``expand_action_expectation``.
    """
    if isinstance(caption, (list, tuple)):
        caption = caption[0] if caption else ""
    if not isinstance(caption, str):
        caption = str(caption)
    lines: list[str] = []
    for idx, text in enumerate(action_texts):
        lines.append(f"  Action {idx + 1} label (copy this exactly): {text}")
        lines.extend(expand_action_expectation(text))
    action_sequence = "\n".join(lines)
    return (
        PROMPT_TEMPLATE
        .replace("{caption}", caption)
        .replace("{action_sequence}", action_sequence)
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateInfo:
    """Lightweight container for one GRPO candidate."""
    index: int                 # position in the rollout batch
    video_frames: np.ndarray   # (T, H, W, 3) uint8 from VAE decode


@dataclass
class VLMRewardResult:
    """Per-candidate aggregated win-rates."""
    ac_win_rate: float   # action-control win-rate  [0, 1]
    vq_win_rate: float   # visual-quality win-rate  [0, 1]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class VLMRewardBackend:
    """Orchestrates pairwise VLM comparisons across GRPO candidates.

    Supports multiple vLLM replicas for higher throughput.  When
    *num_replicas* > 1, requests are distributed round-robin across ports
    ``[port, port+1, ..., port+num_replicas-1]``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        model_name: str = "UnifiedReward",
        timeout: int = 600,
        max_retries: int = 3,
        num_replicas: int = 1,
        urls: list[str] | None = None,
    ) -> None:
        # Two deployment modes:
        #   * colocated (urls is None): every training node runs its own
        #     vLLM at host:port..port+num_replicas-1. wake/sleep is called
        #     so vLLM releases GPU to the local training rollout.
        #   * dedicated (urls is set): vLLM runs on separate nodes, each
        #     replica has its own full URL. wake/sleep are no-ops because
        #     vLLM has exclusive GPUs on those nodes.
        self.host = host
        self.base_port = port
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        if urls:
            self.urls = list(urls)
            self.colocated = False
        else:
            self.urls = [
                f"http://{host}:{port + i}"
                for i in range(max(num_replicas, 1))
            ]
            self.colocated = True
        self.num_replicas = len(self.urls)
        self._call_counter = 0  # round-robin counter
        self._call_counter_lock = threading.Lock()
        self._session = requests.Session()
        self._session.trust_env = False
        # Increase pool size to avoid "Connection pool is full" warnings
        # when many pairs are sent concurrently (C(8,2)=28 > default 10).
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=64)
        self._session.mount("http://", adapter)
        # Bypass proxy for localhost
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
        # Separate gloo CPU process group for reward-phase sync. NCCL's
        # C++ watchdog hard-caps collective timeouts at 10 min; the reward
        # phase can exceed that when rank completion is skewed. gloo has
        # no such cap, so we can barrier across the full reward phase
        # without tripping the NCCL watchdog.
        self._cpu_pg = None
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self._cpu_pg = dist.new_group(backend="gloo")
                logger.info("VLM reward: created gloo CPU PG for barriers")
        except Exception as e:
            logger.warning("VLM reward: failed to create gloo CPU PG: %s", e)

    def _next_url(self) -> str:
        """Return the next replica URL in round-robin order (thread-safe)."""
        with self._call_counter_lock:
            url = self.urls[self._call_counter % self.num_replicas]
            self._call_counter += 1
        return url

    def _cpu_barrier(self, tag: str) -> None:
        """Cross-rank barrier on the gloo CPU PG (no NCCL watchdog cap)."""
        if self._cpu_pg is None:
            return
        try:
            import torch.distributed as dist
            dist.barrier(group=self._cpu_pg)
        except Exception as e:
            logger.warning("VLM reward: gloo barrier (%s) failed: %s", tag, e)

    @staticmethod
    def _is_global_rank0() -> bool:
        """True only on world-rank 0 — for log emission gating."""
        return int(os.environ.get("RANK", "0")) == 0

    @staticmethod
    def _is_local_rank0() -> bool:
        # Each node runs its own vLLM replica at host="localhost".
        # The local rank 0 on every node is responsible for waking/sleeping
        # that node's local vLLM. Using global rank 0 only would leave the
        # vLLM replicas on other nodes sleeping, so ranks on those nodes
        # would hit every request with a sleeping server.
        return int(os.environ.get("LOCAL_RANK", "0")) == 0

    def _wake_all_cleanup_only(self) -> None:
        """Phase 1 of wake: EVERY rank empties its CUDA caching allocator.

        Colocated only — dedicated deployment leaves vLLM awake full-time
        and never needs to reclaim GPU from the training process.
        """
        if not self.colocated:
            return
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning("pre-wake empty_cache failed: %s", e)

    def _wake_all_post_only(self) -> None:
        """Phase 2 of wake: local_rank 0 POSTs /wake_up to its node's vLLM.

        Colocated only — dedicated deployment stays awake, no POST needed.
        """
        if not self.colocated:
            return
        if not self._is_local_rank0():
            return
        for url in self.urls:
            try:
                self._session.post(f"{url}/wake_up", timeout=120)
            except Exception as e:
                logger.warning("VLM wake_up %s failed: %s", url, e)

    def _sleep_all(self, level: int = 1) -> None:
        """Sleep local vLLM replica (one call per node) to free GPU memory.

        Colocated only — dedicated deployment has vLLM on separate GPUs,
        no reason to sleep between rollouts. Skips both the /sleep POST
        and the training-side empty_cache (no fragmentation issue because
        vLLM never allocates on training GPUs).
        """
        if not self.colocated:
            return
        # Only local_rank 0 on each node issues the HTTP /sleep call.
        if self._is_local_rank0():
            for url in self.urls:
                try:
                    self._session.post(
                        f"{url}/sleep?level={level}", timeout=60,
                    )
                except Exception as e:
                    logger.warning("VLM sleep %s failed: %s", url, e)
        # Every training rank drops its caching-allocator reservations so that
        # physical memory freed by vLLM's /sleep is available to the next
        # rollout step's VAE decode. Without this, each training process holds
        # onto ~10 GiB of fragmented reservations across the wake/sleep cycle
        # and rank3 GPU 5 VAE pad OOM'd at step 16.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning("empty_cache after vLLM sleep failed: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_pairwise_rewards(
        self,
        candidates: list[CandidateInfo],
        caption: str,
        action_labels: list[int],
        action_texts: list[str],
        chunk_id: int,
        chunk_size: int = ACTIONS_PER_GROUP,
    ) -> dict[int, VLMRewardResult]:
        """Run round-robin pairwise comparison, return per-candidate win-rates.

        Parameters
        ----------
        candidates : list[CandidateInfo]
            Video candidates to compare.
        caption : str
            Scene caption for the prompt.
        action_labels : list[int]
            Integer action labels for the current chunk's latents (length=chunk_size).
            These are the raw per-latent labels from the training data, where
            latent 0 is already IDLE (label=0) for chunk_id=0.
        action_texts : list[str]
            Human-readable action strings corresponding to action_labels
            (e.g. "IDLE | IDLE", "Forward | YawRight").  Length = chunk_size.
        chunk_id : int
            0-based chunk index (matches build_eval_data convention).
        chunk_size : int
            Number of latents per chunk (default 4).

        Returns
        -------
        dict mapping candidate.index → VLMRewardResult
        """
        n = len(candidates)
        if n < 2:
            return {c.index: VLMRewardResult(0.5, 0.5) for c in candidates}

        # action_labels and action_texts are already the chunk-level 4 actions
        # from the training data (latent 0 = IDLE for chunk_id=0).
        # Use them directly as the prompt's action sequence.
        chunk_texts = action_texts
        prompt_text = build_prompt_text(caption, chunk_texts)

        # Frame pairs for this chunk
        frame_pairs = _chunk_frame_pairs(chunk_id, chunk_size)

        # Reference frame index for the grid
        if chunk_id == 0:
            ref_vf = 0
        else:
            prev_last_latent = chunk_id * chunk_size - 1
            ref_vf = prev_last_latent * 4

        # key_vfs for the overview grid (3 frames per action)
        key_vfs: list[int] = []
        for vf_s, vf_e in frame_pairs:
            if vf_s == vf_e:
                key_vfs.extend([vf_s, vf_s, vf_s])
            else:
                key_vfs.extend([vf_s, vf_s + 1, vf_e])

        # Collect all frame indices we need
        all_vfs: set[int] = {0, ref_vf}  # frame 0 for input_image, ref_vf for grid
        for vf_s, vf_e in frame_pairs:
            all_vfs.update([vf_s, vf_e])
        all_vfs.update(key_vfs)

        # Two-phase wake_up:
        # 1. Every rank empty_cache + sync (inside _wake_all).
        # 2. Barrier so ALL ranks' caching allocators have returned memory
        #    to the driver before vLLM tries to cuMemMap.
        # 3. local_rank 0 POSTs /wake_up while other ranks wait.
        # 4. Barrier so every rank sees vLLM awake before sending requests.
        if self._is_global_rank0():
            logger.info("Waking up VLM replicas before pairwise reward")
        # Phase 1: all-rank empty_cache (no HTTP call inside for non-rank0)
        self._wake_all_cleanup_only()
        self._cpu_barrier("all ranks cleaned caches")
        # Phase 2: local_rank 0 POST /wake_up
        self._wake_all_post_only()
        self._cpu_barrier("after wake")

        # Build all pair images first (in memory), then fire HTTP requests in parallel.
        pair_results: list[tuple[int, int, int | None, int | None]] = []
        # How each pair got scored, reported in the phase summary below.
        score_mode: dict[str, int] = {
            "ac_soft": 0, "ac_hard": 0, "ac_tie": 0,
            "vq_soft": 0, "vq_hard": 0, "vq_tie": 0,
        }
        try:
            # Phase 1: build images for all pairs (no disk I/O)
            pair_meta: list[tuple[int, int, bool, list[bytes]]] = []
            for idx_i, idx_j in itertools.combinations(range(n), 2):
                ci = candidates[idx_i]
                cj = candidates[idx_j]

                swap = random.random() < 0.5
                disp_a, disp_b = (cj, ci) if swap else (ci, cj)

                a_frames = self._extract_frames_dict(disp_a.video_frames, all_vfs)
                b_frames = self._extract_frames_dict(disp_b.video_frames, all_vfs)

                image_bytes_list = self._build_pair_images(
                    a_frames, b_frames, chunk_texts, frame_pairs,
                    key_vfs, ref_vf, pair_dir=None,  # unused
                )
                pair_meta.append((idx_i, idx_j, swap, image_bytes_list))

            # Phase 2: fire all vLLM requests in parallel
            def _call_one(meta):
                idx_i, idx_j, swap, image_bytes_list = meta
                parsed, winner_probs = self._call_vllm(
                    image_bytes_list, prompt_text
                )
                return idx_i, idx_j, swap, parsed, winner_probs

            # 2 concurrent requests per vLLM replica. On colocated L20Z
            # with 8 ranks/node each submitting workers in parallel, 4
            # workers/rank produced Running=32 on one vLLM — compute
            # bound, not KV-bound (usage 37%), which ballooned single
            # request latency to 120-150s and pushed 3-7% of pairs past
            # the client timeout even with read_timeout=600. Halving to
            # 2 brings the steady state to Running=16: generation
            # throughput per request roughly doubles (less contention
            # for tensor cores and HBM bandwidth), tail latency drops
            # well below the timeout, and total reward-phase wall time
            # stays similar because the server was saturated either way.
            max_workers = min(len(pair_meta), self.num_replicas * 2)
            phase_timeout = 1500
            seen_pairs: set[tuple[int, int]] = set()
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_call_one, m): m for m in pair_meta}
                    for fut in as_completed(futures, timeout=phase_timeout):
                        idx_i, idx_j, swap, parsed, winner_probs = fut.result()
                        ci = candidates[idx_i]
                        cj = candidates[idx_j]
                        disp_a, disp_b = (cj, ci) if swap else (ci, cj)

                        ac_si: float | None = None
                        ac_sj: float | None = None
                        vq_si: float | None = None
                        vq_sj: float | None = None
                        ac_winner_idx: int | None = None
                        vq_winner_idx: int | None = None
                        ac_tie_override = False
                        vq_tie_override = False
                        if parsed is not None:
                            ac_raw = self._safe_get(parsed, ["action_control", "overall_winner"])
                            vq_raw = self._safe_get(parsed, ["visual_quality", "overall_winner"])
                            ac_winner_idx = self._map_winner(ac_raw, disp_a.index, disp_b.index)
                            vq_winner_idx = self._map_winner(vq_raw, disp_a.index, disp_b.index)
                            # Sub-dimension all-tie override: when every
                            # per-action winner (AC) or every VQ sub-winner
                            # is "Tie", split the pair credit 0.5/0.5
                            # instead of using overall_winner.
                            ac_tie_override = self._action_control_all_tie(parsed)
                            vq_tie_override = self._visual_quality_all_tie(parsed)
                            # If vLLM returned per-token logprobs for the two
                            # overall_winner positions, use the softmax(A,B)
                            # probability as a *soft* pairwise reward (0-1).
                            # This is lower-variance than the hard 1/0 signal
                            # and lets the policy see model uncertainty. Fall
                            # back to hard winner when logprobs missing.
                            ac_prob = None
                            vq_prob = None
                            if winner_probs:
                                ac_p = winner_probs.get("action_control")
                                vq_p = winner_probs.get("visual_quality")
                                if ac_p is not None:
                                    ac_prob = ac_p["p_A_norm"]
                                if vq_p is not None:
                                    vq_prob = vq_p["p_A_norm"]
                            # Track how each pair was actually scored. Without
                            # this the soft path can be dead for a whole run
                            # and nothing shows it: hard win-rates still look
                            # perfectly reasonable. Distinguish tie_override
                            # (logprob deliberately discarded) from a genuine
                            # logprob miss (silent fallback = a bug).
                            if ac_tie_override:
                                score_mode["ac_tie"] += 1
                            elif ac_prob is not None:
                                score_mode["ac_soft"] += 1
                            else:
                                score_mode["ac_hard"] += 1
                            if vq_tie_override:
                                score_mode["vq_tie"] += 1
                            elif vq_prob is not None:
                                score_mode["vq_soft"] += 1
                            else:
                                score_mode["vq_hard"] += 1
                            ac_si, ac_sj = self._pair_scores(
                                ac_winner_idx, ci.index, cj.index, ac_tie_override,
                                prob_a=ac_prob, disp_a_idx=disp_a.index, disp_b_idx=disp_b.index,
                            )
                            vq_si, vq_sj = self._pair_scores(
                                vq_winner_idx, ci.index, cj.index, vq_tie_override,
                                prob_a=vq_prob, disp_a_idx=disp_a.index, disp_b_idx=disp_b.index,
                            )

                        pair_results.append(
                            (ci.index, cj.index, ac_si, ac_sj, vq_si, vq_sj)
                        )
                        seen_pairs.add((idx_i, idx_j))
                        # Per-pair log would emit 32 ranks * 28 pairs = 896
                        # lines per reward phase; demote to DEBUG so only
                        # rank 0 global opt-in can see it, and log a phase
                        # summary below instead.
                        logger.debug(
                            "VLM pair %d vs %d (swap=%s): AC=%s%s VQ=%s%s",
                            ci.index, cj.index, swap,
                            ac_winner_idx, " (all-tie→0.5/0.5)" if ac_tie_override else "",
                            vq_winner_idx, " (all-tie→0.5/0.5)" if vq_tie_override else "",
                        )
            except (FuturesTimeoutError, TimeoutError):
                logger.warning(
                    "VLM pairwise phase hit global timeout after %ds; filling %d missing pairs with None",
                    phase_timeout, len(pair_meta) - len(seen_pairs),
                )
                # Fill missing pairs with None so rank ordering doesn't diverge.
                for meta in pair_meta:
                    idx_i, idx_j, _, _ = meta
                    if (idx_i, idx_j) not in seen_pairs:
                        ci = candidates[idx_i]
                        cj = candidates[idx_j]
                        pair_results.append(
                            (ci.index, cj.index, None, None, None, None)
                        )
        finally:
            # gloo-barrier so every rank finishes its requests before any
            # local_rank0 sleeps the shared vLLM. Otherwise fast ranks put
            # their node's vLLM to sleep while slow ranks still have in-
            # flight requests, causing 5xx cascades and None fallbacks.
            # NCCL can't be used here because reward_phase > 10min trips
            # the C++ watchdog; gloo (CPU) has no such cap.
            self._cpu_barrier("before sleep")
            # Rank-0-only structured summary; no more 896 per-pair lines.
            if self._is_global_rank0():
                total = len(pair_meta)
                got = len(seen_pairs)
                # pair_results entries are 6-tuples
                # (idx_i, idx_j, ac_si, ac_sj, vq_si, vq_sj). A pair has a
                # None fallback when EITHER dimension's per-pair score is
                # missing (ac_si or vq_si being None implies the partner
                # score is None too).
                none_pairs = sum(
                    1 for r in pair_results if r[2] is None or r[4] is None
                )
                logger.info(
                    "VLM reward phase done: %d/%d pairs fetched, %d None fallbacks; "
                    "AC soft/hard/tie=%d/%d/%d VQ soft/hard/tie=%d/%d/%d; sleeping vLLM",
                    got, total, none_pairs,
                    score_mode["ac_soft"], score_mode["ac_hard"], score_mode["ac_tie"],
                    score_mode["vq_soft"], score_mode["vq_hard"], score_mode["vq_tie"],
                )
                # hard > 0 means logprobs were requested but could not be read,
                # i.e. the soft reward silently degraded to 1/0. tie is a
                # deliberate 0.5/0.5 and is not a problem.
                if score_mode["ac_hard"] or score_mode["vq_hard"]:
                    logger.warning(
                        "VLM soft scoring unavailable on %d AC / %d VQ pairs "
                        "(logprobs missing or unparseable); those pairs used "
                        "hard 1/0 instead of softmax(A,B)",
                        score_mode["ac_hard"], score_mode["vq_hard"],
                    )
            else:
                logger.info("Putting VLM replicas to sleep to free GPU")
            self._sleep_all(level=1)

        return self._aggregate_win_rates(candidates, pair_results)

    # ------------------------------------------------------------------
    # Image construction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frames_dict(
        video_frames: np.ndarray, needed_vfs: set[int],
    ) -> dict[int, np.ndarray]:
        """Index into (T, H, W, 3) numpy array to build {vf: frame} dict."""
        result: dict[int, np.ndarray] = {}
        max_t = video_frames.shape[0]
        for vf in needed_vfs:
            if vf < max_t:
                result[vf] = video_frames[vf]
        return result

    @staticmethod
    def _build_pair_images(
        a_frames: dict[int, np.ndarray],
        b_frames: dict[int, np.ndarray],
        chunk_texts: list[str],
        frame_pairs: list[tuple[int, int]],
        key_vfs: list[int],
        ref_vf: int,
        pair_dir: Path,  # unused, kept for API compat
    ) -> list[bytes]:
        """Create the 6 images required by the RM prompt, in memory.

        Returns a list of PNG bytes (one per image). Avoids filesystem
        round-trip so 28 pairs × 6 images don't touch disk.
        """
        images: list[bytes] = []

        # 1. input_image (always frame 0) - encode via cv2 in memory
        f0 = a_frames.get(0, b_frames.get(0))
        if f0 is not None:
            bgr = cv2.cvtColor(f0, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", bgr)
            images.append(buf.tobytes() if ok else b"")
        else:
            images.append(b"")

        # 2. frame_grid - matplotlib savefig accepts BytesIO
        grid_buf = io.BytesIO()
        save_frame_grid(a_frames, b_frames, chunk_texts, grid_buf, key_vfs, ref_vf)
        images.append(grid_buf.getvalue())

        # 3-6. action pair images
        for la in range(ACTIONS_PER_GROUP):
            vf_start, vf_end = frame_pairs[la]
            buf = io.BytesIO()
            save_action_pair(
                a_frames, b_frames, la, chunk_texts[la],
                vf_start, vf_end, buf,
            )
            images.append(buf.getvalue())

        return images

    # ------------------------------------------------------------------
    # vLLM HTTP call
    # ------------------------------------------------------------------

    def _call_vllm(
        self, image_bytes_list: list[bytes], prompt_text: str,
    ) -> dict | None:
        """Send a single pairwise comparison to the vLLM server.

        image_bytes_list: list of PNG bytes (6 images per pair).
        """
        # Build multimodal content parts
        full_text = "<image>" * len(image_bytes_list) + prompt_text
        parts = re.split(r"<image>", full_text)
        content_parts: list[dict] = []
        for i, part in enumerate(parts):
            if i > 0 and i <= len(image_bytes_list):
                b64 = base64.b64encode(image_bytes_list[i - 1]).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            if part:
                content_parts.append({"type": "text", "text": part})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": 2048,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            # Enable per-token logprobs so downstream can extract the A vs B
            # relative probability at each overall_winner position, and use
            # the soft prob as the pairwise reward instead of a hard 1/0.
            "logprobs": True,
            "top_logprobs": 20,
        }

        url = f"{self._next_url()}/v1/chat/completions"
        # (connect_timeout, read_timeout). On slower GPUs with a loaded vLLM
        # queue a single pairwise inference can reach 2-3 min end-to-end, so
        # give read_timeout enough headroom that transient contention doesn't
        # trigger a retry storm that only makes queueing worse.
        req_timeout = (10, 600)
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=req_timeout)
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                raw = choice["message"]["content"].strip()
                parsed = self._parse_json_response(raw)
                logprobs_content = (choice.get("logprobs") or {}).get("content") or []
                winner_probs = self._extract_winner_probs(raw, logprobs_content)
                return parsed, winner_probs
            except Exception as e:
                logger.warning("VLM call attempt %d failed: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None, None

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> dict | None:
        """Parse RM JSON output, handling markdown wrapping.

        Falls back to regex extraction if JSON is malformed (truncated,
        missing braces, etc). Only recovers the overall_winner fields since
        that's all downstream code reads.
        """
        if raw.startswith("```"):
            lines = raw.split("\n")
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            raw = "\n".join(lines[1:end])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try appending closing braces (common for truncated output)
        for extra in ["}", "}}", "}}}", "}}}}"]:
            try:
                return json.loads(raw + extra)
            except json.JSONDecodeError:
                pass
        # Regex fallback: extract overall_winner for the two top-level
        # sections. The WorldReward (medium_v1) schema has no final_verdict
        # block, so we only recover action_control / visual_quality.
        result: dict = {}
        for section in ("action_control", "visual_quality"):
            pat = rf'"{section}"\s*:\s*\{{.*?"overall_winner"\s*:\s*"([AB])"'
            m = re.search(pat, raw, re.DOTALL)
            if m:
                result[section] = {"overall_winner": m.group(1)}
        if result:
            logger.info(
                "Parse fallback: regex recovered %d/2 sections from malformed JSON",
                len(result),
            )
            return result
        return None

    @staticmethod
    def _extract_winner_probs(
        raw_text: str,
        logprobs_content: list[dict],
    ) -> dict[str, dict[str, float]]:
        """Extract normalized P(A) vs P(B) at each `overall_winner` position.

        Uses the top_logprobs at the token that holds the actual A/B letter
        after the `"overall_winner": "` prefix. Because tokenization is
        model-dependent, we walk the token stream, cumulatively rebuild the
        emitted text, and locate the token whose start-offset falls just past
        the pattern `"overall_winner": "`. That token's top_logprobs list is
        scanned for any entry whose token text starts with `A` or `B` (with
        optional leading whitespace/quotes) so both `"A"`-as-one-token and
        split-tokenization work.

        Returns
        -------
        dict with keys among {"action_control", "visual_quality"}. Each maps
        to {"p_A": float, "p_B": float, "p_A_norm": float}. p_A_norm is the
        softmax-of-two: exp(logp_A) / (exp(logp_A) + exp(logp_B)). Missing
        section = missing key. Missing key downstream => fall back to hard
        winner (existing 1.0/0.0 path).
        """
        if not logprobs_content or not raw_text:
            return {}

        # Precompute char offsets of each token in the emitted text.
        offsets: list[int] = []
        cursor = 0
        recon: list[str] = []
        for tok in logprobs_content:
            offsets.append(cursor)
            piece = tok.get("token", "")
            recon.append(piece)
            cursor += len(piece)
        reconstructed = "".join(recon)

        # Find both "overall_winner" positions in the emitted text.
        # We search on the reconstructed stream so char offsets line up with
        # the token offsets we recorded above (raw_text after strip() may
        # have leading whitespace trimmed vs reconstructed).
        result: dict[str, dict[str, float]] = {}
        # Locate the A/B letter that follows each `"overall_winner"` key. This
        # MUST be whitespace-tolerant: _parse_json_response matches with \s*,
        # so a model emitting minified JSON ("overall_winner":"A") must parse
        # here too, or soft scoring drops out while hard win-rates still look
        # healthy.
        # We scan in text order and assign the first match to action_control
        # and the second to visual_quality, matching the JSON schema.
        anchor_re = re.compile(r'"overall_winner"\s*:\s*"')
        section_names = ["action_control", "visual_quality"]
        search_from = 0
        for section in section_names:
            m_anchor = anchor_re.search(reconstructed, search_from)
            if m_anchor is None:
                break
            winner_char_pos = m_anchor.end()
            search_from = winner_char_pos + 1
            # Find the token whose character offset covers winner_char_pos.
            tok_idx: int | None = None
            for i, off in enumerate(offsets):
                tok_end = off + len(logprobs_content[i].get("token", ""))
                if off <= winner_char_pos < tok_end:
                    tok_idx = i
                    break
            if tok_idx is None:
                continue
            tok = logprobs_content[tok_idx]
            # Sanity: the located token must actually contain 'A' or 'B' near
            # the winner char offset. If tokenization put the letter after
            # a leading quote in a separate token, advance by one.
            candidate_text = tok.get("token", "")
            if candidate_text.strip('"\'` \t\n') in ("", ":", ",", "{"):
                if tok_idx + 1 < len(logprobs_content):
                    tok_idx += 1
                    tok = logprobs_content[tok_idx]
            top = tok.get("top_logprobs") or []
            logp_A: float | None = None
            logp_B: float | None = None
            for entry in top:
                letter = entry.get("token", "").strip('"\'` \t\n')
                if not letter:
                    continue
                first = letter[0]
                if first == "A" and logp_A is None:
                    logp_A = float(entry.get("logprob", float("-inf")))
                elif first == "B" and logp_B is None:
                    logp_B = float(entry.get("logprob", float("-inf")))
                if logp_A is not None and logp_B is not None:
                    break
            if logp_A is None and logp_B is None:
                continue
            # Missing side => treat as very low prob (log-prob = -50 clamps
            # exp() to effectively 0 without underflow issues).
            eff_A = logp_A if logp_A is not None else -50.0
            eff_B = logp_B if logp_B is not None else -50.0
            # Softmax over {A, B} — normalize away all other mass.
            m = max(eff_A, eff_B)
            e_A = math.exp(eff_A - m)
            e_B = math.exp(eff_B - m)
            p_A_norm = e_A / (e_A + e_B)
            result[section] = {
                "p_A": math.exp(eff_A) if eff_A > -30 else 0.0,
                "p_B": math.exp(eff_B) if eff_B > -30 else 0.0,
                "p_A_norm": p_A_norm,
            }
        return result

    @staticmethod
    def _safe_get(d: dict, keys: list[str]):
        """Safely traverse nested dict."""
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
            if d is None:
                return None
        return d

    @staticmethod
    def _map_winner(raw_winner: str | None, idx_a: int, idx_b: int) -> int | None:
        """Map "A"/"B" string to candidate index."""
        if raw_winner == "A":
            return idx_a
        elif raw_winner == "B":
            return idx_b
        return None

    # ------------------------------------------------------------------
    # Win-rate aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _action_control_all_tie(parsed: dict) -> bool:
        """True iff every per-action winner under action_control is "Tie".

        WorldReward (medium_v1) schema puts a "Tie" winner on each of the 4
        per-action items. When every one is "Tie", the model is signalling
        that the two videos' camera motion is genuinely indistinguishable
        across the chunk — credit the pair 0.5/0.5 instead of forcing a
        winner via the (still A/B-only) overall_winner.
        """
        actions = VLMRewardBackend._safe_get(parsed, ["action_control", "actions"])
        if not isinstance(actions, list) or not actions:
            return False
        for item in actions:
            if not isinstance(item, dict):
                return False
            if item.get("winner") != "Tie":
                return False
        return True

    @staticmethod
    def _visual_quality_all_tie(parsed: dict) -> bool:
        """True iff all 3 VQ sub-dimension winners are "Tie"."""
        vq = parsed.get("visual_quality") if isinstance(parsed, dict) else None
        if not isinstance(vq, dict):
            return False
        for sub in ("temporal_consistency",
                    "dynamic_generation_quality",
                    "artifacts_and_structure_integrity"):
            block = vq.get(sub)
            if not isinstance(block, dict):
                return False
            if block.get("winner") != "Tie":
                return False
        return True

    @staticmethod
    def _pair_scores(
        winner_idx: int | None,
        idx_i: int,
        idx_j: int,
        tie_override: bool,
        prob_a: float | None = None,
        disp_a_idx: int | None = None,
        disp_b_idx: int | None = None,
    ) -> tuple[float | None, float | None]:
        """Return per-pair (score_i, score_j) from an overall_winner index.

        Hard-winner semantics (used when prob_a is None):
        - tie_override → (0.5, 0.5) regardless of winner_idx
        - winner_idx == idx_i → (1.0, 0.0)
        - winner_idx == idx_j → (0.0, 1.0)
        - winner_idx is None and not tie_override → (None, None) (drop pair)

        Soft-winner semantics (when prob_a is provided, in [0,1] = P(A) after
        softmax over the A/B logprobs at the overall_winner token):
        - tie_override → (0.5, 0.5) (still honor the "all sub-Tie" signal)
        - Otherwise map disp_a/disp_b → idx_i/idx_j and return the smooth
          (p_i, p_j) with p_i+p_j=1. No pair is dropped as long as a prob
          exists; hard-winner routing is only used as fallback when logprobs
          were missing on that section.
        """
        if tie_override:
            return 0.5, 0.5
        if prob_a is not None and disp_a_idx is not None and disp_b_idx is not None:
            prob_b = 1.0 - prob_a
            if disp_a_idx == idx_i:
                return prob_a, prob_b
            if disp_a_idx == idx_j:
                return prob_b, prob_a
        if winner_idx is None:
            return None, None
        if winner_idx == idx_i:
            return 1.0, 0.0
        if winner_idx == idx_j:
            return 0.0, 1.0
        return None, None

    @staticmethod
    def _aggregate_win_rates(
        candidates: list[CandidateInfo],
        pair_results: list[tuple[int, int, float | None, float | None,
                                 float | None, float | None]],
    ) -> dict[int, VLMRewardResult]:
        """Convert per-pair (ac_si, ac_sj, vq_si, vq_sj) to candidate win-rates.

        Each valid AC/VQ pair contributes:
          - 1.0 to the winner, 0.0 to the loser    (overall_winner = A or B)
          - 0.5 to BOTH                            (all-tie sub-dim override)
        Pairs where the score is None for a dimension are dropped from
        that dimension's denominator. AC and VQ have independent counts so
        a parse failure in one doesn't poison the other.
        """
        idx_set = {c.index for c in candidates}
        ac_score: dict[int, float] = {i: 0.0 for i in idx_set}
        vq_score: dict[int, float] = {i: 0.0 for i in idx_set}
        ac_count: dict[int, int] = {i: 0 for i in idx_set}
        vq_count: dict[int, int] = {i: 0 for i in idx_set}

        for idx_i, idx_j, ac_si, ac_sj, vq_si, vq_sj in pair_results:
            if ac_si is not None and ac_sj is not None:
                ac_score[idx_i] += ac_si
                ac_score[idx_j] += ac_sj
                ac_count[idx_i] += 1
                ac_count[idx_j] += 1
            if vq_si is not None and vq_sj is not None:
                vq_score[idx_i] += vq_si
                vq_score[idx_j] += vq_sj
                vq_count[idx_i] += 1
                vq_count[idx_j] += 1

        results: dict[int, VLMRewardResult] = {}
        for c in candidates:
            i = c.index
            ac_wr = ac_score[i] / ac_count[i] if ac_count[i] > 0 else 0.5
            vq_wr = vq_score[i] / vq_count[i] if vq_count[i] > 0 else 0.5
            results[i] = VLMRewardResult(ac_win_rate=ac_wr, vq_win_rate=vq_wr)
        summary = {
            i: f"AC={results[i].ac_win_rate:.2f}/VQ={results[i].vq_win_rate:.2f}"
            for i in sorted(results)
        }
        logger.info("VLM win_rates: %s", summary)
        return results


# ---------------------------------------------------------------------------
# BaseReward wrapper
# ---------------------------------------------------------------------------

from fastvideo.rewards.base import GroupReward, RewardContext, ScoreResult


class VLMPairwiseReward(GroupReward):
    """GroupReward wrapping the HTTP vLLM pairwise backend.

    OUTPUTS=("vlm_action", "vlm_vq") — action-control / visual-quality
    win rates over C(N,2) pairwise comparisons. Has no GPU footprint on
    the training side. Rollout still calls `_backend.compute_pairwise_rewards`
    directly via the dispatcher facade, because it needs frames + chunk_id
    which don't fit cleanly into `RewardContext` semantics.
    """
    NAME = "worldreward"
    OUTPUTS = ("vlm_action", "vlm_vq")

    def __init__(self, training_args):
        super().__init__(training_args)
        self._backend: VLMRewardBackend | None = None

    def load(self, device) -> None:
        if self._backend is not None:
            return
        # Optional `vlm_rm_urls` (list[str]) switches to dedicated mode:
        # requests go directly to those absolute URLs, wake/sleep is a
        # no-op. If absent, fall back to colocated `host:port + N replicas`.
        urls = getattr(self.training_args, "vlm_rm_urls", None) or None
        self._backend = VLMRewardBackend(
            host=self.training_args.vlm_rm_host,
            port=self.training_args.vlm_rm_port,
            model_name=self.training_args.vlm_rm_model_name,
            num_replicas=int(
                getattr(self.training_args, "vlm_rm_num_replicas", 1)
            ),
            urls=urls,
        )

    def unload(self) -> None:
        # External vLLM: nothing to unload on the training side.
        self._backend = None

    @property
    def backend(self) -> VLMRewardBackend:
        assert self._backend is not None, "VLMPairwiseReward not loaded"
        return self._backend

    def score_group(self, ctxs: list[RewardContext]) -> list[ScoreResult]:
        assert self._backend is not None, "VLMPairwiseReward not loaded"
        n = len(ctxs)
        if n == 0:
            return []
        caption = ctxs[0].caption or ctxs[0].prompt or ""
        action_labels = ctxs[0].action_labels or []
        action_texts = ctxs[0].action_texts or []
        chunk_id = int(ctxs[0].chunk_id)
        candidates = [
            CandidateInfo(index=i, video_frames=ctxs[i].video_frames)
            for i in range(n)
            if ctxs[i].video_frames is not None
        ]
        if len(candidates) < 2:
            return [self.neutral_result() for _ in ctxs]
        results = self._backend.compute_pairwise_rewards(
            candidates=candidates,
            caption=caption,
            action_labels=action_labels,
            action_texts=action_texts,
            chunk_id=chunk_id,
            chunk_size=int(ctxs[0].update_latent_num),
        )
        out: list[ScoreResult] = []
        for i in range(n):
            r = results.get(i)
            if r is None:
                out.append(self.neutral_result())
            else:
                out.append(
                    ScoreResult(
                        scores={
                            "vlm_action": float(r.ac_win_rate),
                            "vlm_vq": float(r.vq_win_rate),
                        }
                    )
                )
        return out
