"""Shared action overlay utilities for rendering WASD-style action keys on video frames."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ACTION_LABEL_TO_FLAGS = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0),
    2: (0, 1, 0, 0),
    3: (0, 0, 1, 0),
    4: (0, 0, 0, 1),
    5: (1, 0, 1, 0),
    6: (1, 0, 0, 1),
    7: (0, 1, 1, 0),
    8: (0, 1, 0, 1),
}

ACTION_FLAG_NAMES = {
    "move": ["Forward", "Backward", "Right", "Left"],
    "rot": ["Yaw Right", "Yaw Left", "Pitch Up", "Pitch Down"],
}

# RM-facing action names (NO spaces) — must stay byte-identical to the strings
# the WorldReward RM was trained on. Distinct from ACTION_FLAG_NAMES above,
# which uses spaced names ("Yaw Right") purely for the visual overlay display.
_RM_MOVE_NAMES = ["Forward", "Backward", "Right", "Left"]
_RM_ROT_NAMES = ["YawRight", "YawLeft", "PitchUp", "PitchDown"]


def label_to_text(label: int) -> str:
    """Convert an 81-class action label (trans*9 + rot) into the RM prompt
    string "<Translation> | <Rotation>", e.g. label 5 -> "IDLE | YawRight+PitchUp".

    Compound flags are joined with '+'; an empty side renders as "IDLE".
    Kept byte-identical to the legacy reward_construct.utils.prepare_data
    implementation so the action text fed to the reward model is unchanged.
    """
    t, r = label // 9, label % 9
    tf = ACTION_LABEL_TO_FLAGS.get(t, (0, 0, 0, 0))
    rf = ACTION_LABEL_TO_FLAGS.get(r, (0, 0, 0, 0))
    moves = "+".join([_RM_MOVE_NAMES[i] for i, v in enumerate(tf) if v]) or "IDLE"
    rots = "+".join([_RM_ROT_NAMES[i] for i, v in enumerate(rf) if v]) or "IDLE"
    return f"{moves} | {rots}"


def decode_action_label(
    action_label: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    trans_label = action_label // 9
    rot_label = action_label % 9
    trans_flags = ACTION_LABEL_TO_FLAGS.get(int(trans_label), (0, 0, 0, 0))
    rot_flags = ACTION_LABEL_TO_FLAGS.get(int(rot_label), (0, 0, 0, 0))
    return trans_flags, rot_flags


def expand_action_labels_to_video_frames(
    action_labels: np.ndarray | torch.Tensor,
    frame_count: int,
) -> list[int]:
    labels = torch.as_tensor(action_labels, dtype=torch.long).view(-1).cpu()
    if labels.numel() == 0:
        return [0] * frame_count

    expanded = labels.repeat_interleave(4)
    expanded = expanded[4:]
    per_frame = [0]
    per_frame.extend(expanded.tolist()[: max(frame_count - 1, 0)])
    if len(per_frame) < frame_count:
        pad_value = per_frame[-1] if per_frame else 0
        per_frame.extend([pad_value] * (frame_count - len(per_frame)))
    return per_frame[:frame_count]


def draw_action_keys_on_frame(frame: np.ndarray, action_label: int) -> np.ndarray:
    trans_flags, rot_flags = decode_action_label(int(action_label))

    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    key_size = 58
    gap = 8
    margin = 40
    base_fill = (0, 0, 0, 115)
    active_fill = (67, 139, 255, 200)
    text_fill = (255, 255, 255, 255)
    panel_fill = (0, 0, 0, 150)
    idle_fill = (120, 120, 120, 190)

    h = img.height
    left_x = margin
    left_y = h - margin - (2 * key_size + gap)

    move_keys = {
        "W": (left_x + key_size + gap, left_y),
        "A": (left_x, left_y + key_size + gap),
        "S": (left_x + key_size + gap, left_y + key_size + gap),
        "D": (left_x + 2 * (key_size + gap), left_y + key_size + gap),
    }
    move_active = {
        "W": bool(trans_flags[0]),
        "S": bool(trans_flags[1]),
        "D": bool(trans_flags[2]),
        "A": bool(trans_flags[3]),
    }

    right_x = img.width - margin - (3 * key_size + 2 * gap)
    right_y = left_y
    rot_keys = {
        "U": (right_x + key_size + gap, right_y),
        "L": (right_x, right_y + key_size + gap),
        "D": (right_x + key_size + gap, right_y + key_size + gap),
        "R": (right_x + 2 * (key_size + gap), right_y + key_size + gap),
    }
    rot_active = {
        "R": bool(rot_flags[0]),
        "L": bool(rot_flags[1]),
        "U": bool(rot_flags[2]),
        "D": bool(rot_flags[3]),
    }

    def draw_key(pos: tuple[int, int], active: bool, glyph: str) -> None:
        x, y = pos
        fill = active_fill if active else base_fill
        draw.rounded_rectangle([x, y, x + key_size, y + key_size], radius=12, fill=fill)
        bbox = draw.textbbox((0, 0), glyph, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = x + (key_size - text_w) / 2
        ty = y + (key_size - text_h) / 2 - 1
        draw.text((tx, ty), glyph, fill=text_fill, font=font)

    def flags_to_text(flags: tuple[int, int, int, int], names: list[str]) -> str:
        active_names = [name for flag, name in zip(flags, names) if flag]
        return " + ".join(active_names) if active_names else "IDLE"

    def draw_status_panel(x: int, y: int, title: str, value: str, active: bool) -> None:
        label = f"{title}: {value}"
        bbox = draw.textbbox((0, 0), label, font=font)
        pad_x = 8
        pad_y = 6
        panel_w = (bbox[2] - bbox[0]) + pad_x * 2
        panel_h = (bbox[3] - bbox[1]) + pad_y * 2
        fill = active_fill if active else idle_fill if value == "IDLE" else panel_fill
        draw.rounded_rectangle([x, y, x + panel_w, y + panel_h], radius=10, fill=fill)
        draw.text((x + pad_x, y + pad_y - 1), label, fill=text_fill, font=font)

    for key, pos in move_keys.items():
        draw_key(pos, move_active[key], key)

    draw_key(rot_keys["U"], rot_active["U"], "^")
    draw_key(rot_keys["L"], rot_active["L"], "<")
    draw_key(rot_keys["D"], rot_active["D"], "v")
    draw_key(rot_keys["R"], rot_active["R"], ">")

    move_status = flags_to_text(trans_flags, ACTION_FLAG_NAMES["move"])
    rot_status = flags_to_text(rot_flags, ACTION_FLAG_NAMES["rot"])
    draw_status_panel(margin, margin, "Move", move_status, any(trans_flags))
    draw_status_panel(margin, margin + 34, "Rot", rot_status, any(rot_flags))

    return np.array(Image.alpha_composite(img, overlay).convert("RGB"))


def render_action_overlay_video(
    video_frames: np.ndarray,
    action_labels: np.ndarray | torch.Tensor,
    title: str | None = None,
) -> np.ndarray:
    """Render action overlay on all frames of a video.

    Args:
        video_frames: (T, H, W, 3) uint8 array
        action_labels: action label per action step
        title: optional header title; if provided, adds a header overlay panel
    """
    frame_actions = expand_action_labels_to_video_frames(action_labels, video_frames.shape[0])
    if title is not None:
        rendered_frames = [
            add_overlay_header(
                draw_action_keys_on_frame(frame, action_label),
                title=title,
                action_label=action_label,
                frame_idx=idx,
            )
            for idx, (frame, action_label) in enumerate(zip(video_frames, frame_actions))
        ]
    else:
        rendered_frames = [
            draw_action_keys_on_frame(frame, action_label)
            for frame, action_label in zip(video_frames, frame_actions)
        ]
    return np.stack(rendered_frames, axis=0)


def _action_label_text(action_label: int) -> str:
    trans_flags, rot_flags = decode_action_label(int(action_label))

    def flags_to_text(flags: tuple[int, int, int, int], names: list[str]) -> str:
        active_names = [name for flag, name in zip(flags, names) if flag]
        return " + ".join(active_names) if active_names else "IDLE"

    move_text = flags_to_text(trans_flags, ACTION_FLAG_NAMES["move"])
    rot_text = flags_to_text(rot_flags, ACTION_FLAG_NAMES["rot"])
    return f"Move={move_text} | Rot={rot_text}"


def add_overlay_header(
    frame: np.ndarray,
    title: str,
    action_label: int,
    frame_idx: int | None = None,
) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    title_text = title if frame_idx is None else f"{title} | Frame {frame_idx + 1}"
    action_text = _action_label_text(action_label)

    padding_x = 10
    padding_y = 8
    gap = 4
    title_bbox = draw.textbbox((0, 0), title_text, font=font)
    action_bbox = draw.textbbox((0, 0), action_text, font=font)
    panel_w = max(title_bbox[2] - title_bbox[0], action_bbox[2] - action_bbox[0]) + 2 * padding_x
    panel_h = (
        (title_bbox[3] - title_bbox[1])
        + (action_bbox[3] - action_bbox[1])
        + 2 * padding_y
        + gap
    )
    x = img.width - panel_w - 24
    y = 24
    draw.rounded_rectangle(
        [x, y, x + panel_w, y + panel_h],
        radius=12,
        fill=(16, 20, 28, 210),
        outline=(90, 160, 255, 220),
        width=2,
    )
    draw.text((x + padding_x, y + padding_y - 1), title_text, fill=(255, 255, 255, 255), font=font)
    draw.text(
        (x + padding_x, y + padding_y + (title_bbox[3] - title_bbox[1]) + gap - 1),
        action_text,
        fill=(195, 220, 255, 255),
        font=font,
    )
    return np.array(Image.alpha_composite(img, overlay).convert("RGB"))
