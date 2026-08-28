"""Preprocessing: turn a pair of generated videos into per-chunk image sets.

Each pair is split into chunks of 4 action slots; a chunk renders to 6 images:
``input_image.png`` (shared source), ``frame_grid.png`` (2 rows x (1 reference +
4 actions x start/mid/end)), and ``action1..4.png`` (per-action 2x2 close-ups).
A synthetic ``IDLE`` slot is prepended and a short final chunk is padded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .actions import CHUNK_SIZE, IDLE_LABEL, normalize_slot_label
from .prompt import PROMPT_VERSION
IMAGES_PER_CHUNK = 6

# frame_grid / action-image border colours, one per action slot (RGB 0-255).
ACTION_COLORS = [
    (210, 60, 60),    # red
    (60, 170, 60),    # green
    (60, 100, 210),   # blue
    (200, 140, 0),    # orange
]
ACTION_COLORS_F = [tuple(c / 255.0 for c in rgb) for rgb in ACTION_COLORS]

ROW_LABELS = ("Video A", "Video B")
ROW_COLORS = ("#1a4fa0", "#c0281a")

# Cap on cell aspect ratio (height/width) so near-square videos don't produce
# absurdly tall figures.
_MAX_CELL_ASPECT = 0.85
_DEFAULT_CELL_ASPECT = 480 / 832
_FIGURE_DPI = 200

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f8f8f8",
        "axes.facecolor": "#f8f8f8",
    }
)


# --------------------------------------------------------------------------
# Video / frame handling
# --------------------------------------------------------------------------

def read_video_frames(video_path: str | Path) -> np.ndarray:
    """Decode every frame of a video into an ``(N, H, W, 3)`` uint8 RGB array."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video decoded to 0 frames: {video_path}")
    return np.stack(frames)


def segment_boundaries(segment_frames: list[int]) -> list[tuple[int, int]]:
    """Convert per-action frame counts into ``(start, end_exclusive)`` spans."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for count in segment_frames:
        spans.append((cursor, cursor + int(count)))
        cursor += int(count)
    return spans


def resolve_segment_frames(
    n_actions: int,
    *,
    segment_frames: list[int] | None = None,
    frames_per_action: int | None = None,
    n_frames: int | None = None,
) -> list[int]:
    """Determine the per-action frame counts for one video.

    Resolution order:

    1. explicit ``segment_frames`` (must have one entry per action);
    2. ``frames_per_action`` repeated uniformly;
    3. ``n_frames // n_actions`` repeated uniformly.
    """
    if segment_frames is not None:
        if len(segment_frames) != n_actions:
            raise ValueError(
                f"segment_frames has {len(segment_frames)} entries but there are {n_actions} actions"
            )
        return [int(v) for v in segment_frames]
    if frames_per_action is not None:
        return [int(frames_per_action)] * n_actions
    if n_frames is not None:
        per_action = int(n_frames) // n_actions
        if per_action < 1:
            raise ValueError(f"{n_frames} frames cannot cover {n_actions} actions")
        return [per_action] * n_actions
    raise ValueError("provide one of: segment_frames, frames_per_action, n_frames")


def extract_action_frames(
    frames: np.ndarray,
    spans: list[tuple[int, int]],
    action_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(start, mid, end)`` frames of one action's span.

    Spans are clamped to the decoded frame count, so a video that is shorter than
    its declared segments degrades gracefully to its last frame.
    """
    if action_index >= len(spans):
        last = frames[-1]
        return last, last, last
    total = len(frames)
    start, end_exclusive = spans[action_index]
    start = min(start, total - 1)
    end_exclusive = min(end_exclusive, total)
    if end_exclusive <= start:
        end_exclusive = start + 1
    end = end_exclusive - 1
    return frames[start], frames[(start + end) // 2], frames[end]


# --------------------------------------------------------------------------
# Chunk layout
# --------------------------------------------------------------------------

def build_chunks(actions: list[str]) -> list[list[tuple[int, str]]]:
    """Lay actions out into chunks of :data:`CHUNK_SIZE` slots.

    Each slot is ``(action_index, raw_label)`` where ``action_index == -1``
    marks a synthetic slot: the leading ``IDLE`` reference, or trailing ``PAD``.

    Chunk 0 is ``[IDLE, action0, action1, action2]``; later chunks hold 4 real
    actions each.
    """
    slots: list[tuple[int, str]] = [(-1, "IDLE")]
    slots.extend((index, token) for index, token in enumerate(actions))

    chunks: list[list[tuple[int, str]]] = []
    for start in range(0, len(slots), CHUNK_SIZE):
        chunk = list(slots[start : start + CHUNK_SIZE])
        while len(chunk) < CHUNK_SIZE:
            chunk.append((-1, "PAD"))
        chunks.append(chunk)
    return chunks or [[(-1, "PAD")] * CHUNK_SIZE]


# --------------------------------------------------------------------------
# Figure rendering
# --------------------------------------------------------------------------

def _row_aspect(reference: np.ndarray | None, frame_groups: list[list[np.ndarray]]) -> float | None:
    """Best-effort height/width ratio for one row of the figure."""
    candidate = reference
    if candidate is None:
        for group in frame_groups:
            for frame in group:
                if frame is not None:
                    candidate = frame
                    break
            if candidate is not None:
                break
    if candidate is None:
        return None
    height, width = candidate.shape[:2]
    return float(height) / float(width) if width > 0 else None


def _cell_aspect(aspect_a: float | None, aspect_b: float | None) -> float:
    """Average the two rows' aspect ratios, capped to avoid overly tall cells."""
    values = [v for v in (aspect_a, aspect_b) if v is not None and v > 0]
    if not values:
        return min(_DEFAULT_CELL_ASPECT, _MAX_CELL_ASPECT)
    return min(sum(values) / len(values), _MAX_CELL_ASPECT)


def _save(figure, output_path: str | Path) -> None:
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    if not Path(output_path).is_file():
        raise RuntimeError(f"savefig produced no file: {output_path}")


def save_frame_grid(
    a_frames_per_action: list[list[np.ndarray]],
    b_frames_per_action: list[list[np.ndarray]],
    action_labels: list[str],
    a_reference: np.ndarray | None,
    b_reference: np.ndarray | None,
    output_path: str | Path,
) -> None:
    """Render the wide temporal overview: 2 rows x (1 reference + 4 x 3 frames)."""
    cell_aspect = _cell_aspect(
        _row_aspect(a_reference, a_frames_per_action),
        _row_aspect(b_reference, b_frames_per_action),
    )
    cell_width = 2.0
    cell_height = cell_width * cell_aspect

    n_actions = len(a_frames_per_action)
    frames_per_slot = 3
    n_cols = 1 + n_actions * frames_per_slot

    figure, axes = plt.subplots(
        2,
        n_cols,
        figsize=(n_cols * cell_width + 0.4, 2 * cell_height + 0.38),
        gridspec_kw={"wspace": 0.025, "hspace": 0.05},
    )

    references = (a_reference, b_reference)
    rows = (a_frames_per_action, b_frames_per_action)

    for row in range(2):
        axis = axes[row, 0]
        if references[row] is not None:
            axis.imshow(references[row], aspect="equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#999999")
            spine.set_linewidth(2.5)
        axis.set_ylabel(ROW_LABELS[row], fontsize=13, fontweight="bold", color=ROW_COLORS[row], labelpad=6)
        if row == 0:
            axis.set_title("Start", fontsize=10, color="#888888", pad=3)

        for action_index in range(n_actions):
            color = ACTION_COLORS_F[action_index % len(ACTION_COLORS_F)]
            group = rows[row][action_index]
            for sub in range(frames_per_slot):
                axis = axes[row, 1 + action_index * frames_per_slot + sub]
                frame = group[sub] if sub < len(group) else None
                if frame is not None:
                    axis.imshow(frame, aspect="equal")
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                if row == 0 and sub == 1:
                    axis.set_title(
                        f"A{action_index + 1}: {action_labels[action_index]}",
                        fontsize=10,
                        color=color,
                        fontweight="bold",
                        pad=3,
                    )

    _save(figure, output_path)


def save_action_pair(
    a_start: np.ndarray,
    a_end: np.ndarray,
    b_start: np.ndarray,
    b_end: np.ndarray,
    slot_index: int,
    action_label: str,
    output_path: str | Path,
) -> None:
    """Render one action's 2x2 close-up (rows A/B, columns start/end)."""
    color = ACTION_COLORS_F[slot_index % len(ACTION_COLORS_F)]

    def aspect(primary: np.ndarray | None, fallback: np.ndarray | None) -> float | None:
        frame = primary if primary is not None else fallback
        if frame is None:
            return None
        return float(frame.shape[0]) / float(frame.shape[1])

    cell_aspect = _cell_aspect(aspect(a_start, a_end), aspect(b_start, b_end))
    cell_width = 6.0
    cell_height = cell_width * cell_aspect

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(2 * cell_width + 0.7, 2 * cell_height + 0.22),
        gridspec_kw={"wspace": 0.02, "hspace": 0.03},
    )
    figure.suptitle(
        f"Action {slot_index + 1}: {action_label}",
        fontsize=22,
        fontweight="bold",
        color=color,
        y=1.0,
    )

    cells = ((a_start, a_end), (b_start, b_end))
    for row in range(2):
        for col in range(2):
            axis = axes[row, col]
            frame = cells[row][col]
            if frame is not None:
                axis.imshow(frame, aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3.0)
            if col == 0:
                axis.set_ylabel(
                    ROW_LABELS[row], fontsize=13, fontweight="bold", color=ROW_COLORS[row], labelpad=4
                )

    _save(figure, output_path)


# --------------------------------------------------------------------------
# Pair rendering
# --------------------------------------------------------------------------

def load_chunk_infos(pair_render_dir: Path, expected_chunks: int) -> list[dict[str, Any]] | None:
    """Load a previously rendered pair, or ``None`` if it is absent/incomplete.

    Image paths recorded in ``group_info.json`` are rebased onto the current
    directory, so a rendered tree stays usable after being moved or downloaded.
    """
    if not pair_render_dir.is_dir():
        return None
    infos: list[dict[str, Any]] = []
    for chunk_index in range(expected_chunks):
        chunk_dir = pair_render_dir / f"chunk_{chunk_index:02d}"
        info_path = chunk_dir / "group_info.json"
        if not info_path.is_file():
            return None
        info = json.loads(info_path.read_text(encoding="utf-8"))
        rebased: list[Path] = []
        for recorded in info.get("images", []):
            candidate = Path(recorded)
            if not candidate.is_file():
                candidate = chunk_dir / Path(recorded).name
            rebased.append(candidate)
        if len(rebased) != IMAGES_PER_CHUNK or any(not p.is_file() for p in rebased):
            return None
        info["images"] = [str(p) for p in rebased]
        infos.append(info)
    return infos


def _side_spans(pair: dict[str, Any], side: dict[str, Any], n_actions: int) -> list[tuple[int, int]]:
    frames = resolve_segment_frames(
        n_actions,
        segment_frames=side.get("segment_frames") or pair.get("segment_frames"),
        frames_per_action=side.get("frames_per_action") or pair.get("frames_per_action"),
        n_frames=side.get("num_frames") or pair.get("num_frames"),
    )
    return segment_boundaries(frames)


def render_pair(
    pair: dict[str, Any],
    output_root: str | Path,
    *,
    base_dir: str | Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Render one pair into ``output_root/<pair_id>/chunk_XX/``.

    Args:
        pair: a pair record. Required keys: ``pair_id``, ``input_image``,
            ``input_caption``, ``actions``, and ``left`` / ``right`` objects each
            holding a ``video`` path. Frame spans come from ``segment_frames``,
            ``frames_per_action`` or ``num_frames`` (per side, else on the pair).
        output_root: directory to write rendered chunks into.
        base_dir: root that relative paths inside ``pair`` resolve against.
            Defaults to the current working directory.
        force: re-render even if a complete rendering already exists.

    Returns:
        One ``group_info`` dict per chunk, in chunk order. Each has an
        ``images`` list of 6 absolute paths in model-input order.
    """
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (base / candidate)

    pair_id = pair["pair_id"]
    actions = list(pair["actions"])
    if not actions:
        raise ValueError(f"{pair_id}: 'actions' is empty")
    chunks = build_chunks(actions)

    pair_render_dir = Path(output_root).resolve() / pair_id
    if not force:
        cached = load_chunk_infos(pair_render_dir, len(chunks))
        if cached is not None:
            return cached

    left, right = pair["left"], pair["right"]
    left_frames = read_video_frames(resolve(left["video"]))
    right_frames = read_video_frames(resolve(right["video"]))
    left_spans = _side_spans(pair, left, len(actions))
    right_spans = _side_spans(pair, right, len(actions))

    input_image = np.array(Image.open(resolve(pair["input_image"])).convert("RGB"))
    caption = str(pair.get("input_caption") or "").strip()

    def reference_frame(frames: np.ndarray, spans: list[tuple[int, int]], chunk_index: int) -> np.ndarray:
        """The frame_grid reference: chunk 0 uses the very first frame; later
        chunks use the end frame of the previous chunk's last real action."""
        if chunk_index == 0:
            return frames[0]
        previous_real = [index for index, _ in chunks[chunk_index - 1] if index != -1]
        if not previous_real:
            return frames[0]
        return extract_action_frames(frames, spans, previous_real[-1])[2]

    pair_render_dir.mkdir(parents=True, exist_ok=True)
    chunk_infos: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_dir = pair_render_dir / f"chunk_{chunk_index:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        input_image_path = chunk_dir / "input_image.png"
        Image.fromarray(input_image).save(input_image_path)

        left_groups: list[list[np.ndarray]] = []
        right_groups: list[list[np.ndarray]] = []
        action_indices: list[int] = []
        raw_labels: list[str] = []
        action_labels: list[str] = []

        for slot_index, (action_index, raw_label) in enumerate(chunk):
            action_indices.append(action_index)
            raw_labels.append(raw_label)
            label = normalize_slot_label(raw_label)
            action_labels.append(label)
            action_path = chunk_dir / f"action{slot_index + 1}.png"

            if action_index == -1:
                # IDLE reference slot shows each video's FIRST frame; a trailing
                # PAD slot shows the LAST frame.
                position = 0 if raw_label.upper() == "IDLE" else -1
                left_frame = left_frames[position]
                right_frame = right_frames[position]
                left_groups.append([left_frame] * 3)
                right_groups.append([right_frame] * 3)
                save_action_pair(
                    left_frame, left_frame, right_frame, right_frame, slot_index, label, action_path
                )
                continue

            l_start, l_mid, l_end = extract_action_frames(left_frames, left_spans, action_index)
            r_start, r_mid, r_end = extract_action_frames(right_frames, right_spans, action_index)
            left_groups.append([l_start, l_mid, l_end])
            right_groups.append([r_start, r_mid, r_end])
            save_action_pair(l_start, l_end, r_start, r_end, slot_index, label, action_path)

        frame_grid_path = chunk_dir / "frame_grid.png"
        save_frame_grid(
            left_groups,
            right_groups,
            action_labels,
            reference_frame(left_frames, left_spans, chunk_index),
            reference_frame(right_frames, right_spans, chunk_index),
            frame_grid_path,
        )

        images = [
            input_image_path.resolve(),
            frame_grid_path.resolve(),
            *((chunk_dir / f"action{i}.png").resolve() for i in range(1, CHUNK_SIZE + 1)),
        ]
        group_info = {
            "pair_id": pair_id,
            "chunk_id": chunk_index,
            "caption": caption,
            "action_indices": action_indices,
            "action_labels": raw_labels,
            "action_texts": action_labels,
            "left": {"model": left.get("model"), "video": str(resolve(left["video"]))},
            "right": {"model": right.get("model"), "video": str(resolve(right["video"]))},
            "images": [str(p) for p in images],
            "prompt_version": PROMPT_VERSION,
        }
        (chunk_dir / "group_info.json").write_text(
            json.dumps(group_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        chunk_infos.append(group_info)

    return chunk_infos
