"""Camera-action label normalization and expected-behaviour expansion.

The reward model is prompted with, for every action slot, both a *bare label*
(``"<Translation> | <Rotation>"``) and an indented block spelling out the
expected visual behaviour of each commanded component plus a strict correctness
criterion. This module produces both.

Keeping these strings byte-identical to training is required for reproducing the
published numbers -- the model was supervised on this exact phrasing.
"""
from __future__ import annotations

# Action slots per chunk.
CHUNK_SIZE = 4

# Raw manifest token -> canonical translation atom.
TRANSLATION_TOKENS: dict[str, str] = {
    "forward": "Forward",
    "backward": "Backward",
    "left": "Left",
    "right": "Right",
    "forward_left": "Forward+Left",
    "forward_right": "Forward+Right",
    "backward_left": "Backward+Left",
    "backward_right": "Backward+Right",
}

# Raw manifest token -> canonical rotation atom. Diagonal camera tokens keep BOTH
# their yaw and pitch components so compound rotations are not silently degraded
# to a single axis.
ROTATION_TOKENS: dict[str, str] = {
    "camera_up": "PitchUp",
    "camera_down": "PitchDown",
    "camera_left": "YawLeft",
    "camera_right": "YawRight",
    # Single-letter abbreviations used by some generators.
    "camera_u": "PitchUp",
    "camera_d": "PitchDown",
    "camera_l": "YawLeft",
    "camera_r": "YawRight",
    "yaw_left": "YawLeft",
    "yaw_right": "YawRight",
    "left_turn": "YawLeft",
    "right_turn": "YawRight",
    "turn_left": "YawLeft",
    "turn_right": "YawRight",
    "camera_ul": "YawLeft+PitchUp",
    "camera_ur": "YawRight+PitchUp",
    "camera_dl": "YawLeft+PitchDown",
    "camera_dr": "YawRight+PitchDown",
}

IDLE_LABEL = "IDLE | IDLE"


def pretty_action_text(raw: str | None) -> str:
    """Normalize a raw action token into ``"<Translation> | <Rotation>"``.

    Both axes are always spelled out (``IDLE`` when absent) so the model cannot
    misread a missing axis as "unspecified" rather than "no motion".

        None / "" / "idle"     -> "IDLE | IDLE"
        "forward"              -> "Forward | IDLE"
        "forward_left"         -> "Forward+Left | IDLE"
        "left+camera_down"     -> "Left | PitchDown"
        "camera_up"            -> "IDLE | PitchUp"
        "camera_dl"            -> "IDLE | YawLeft+PitchDown"

    Unknown tokens fall back to a Title-cased rendering so nothing is lost.
    """
    if raw is None:
        return IDLE_LABEL
    key = str(raw).strip()
    if not key or key.lower() == "idle":
        return IDLE_LABEL

    components = [c.strip().lower() for c in key.split("+") if c.strip()]
    translation: str | None = None
    rotation: str | None = None
    unknown: list[str] = []
    for component in components:
        if component == "idle":
            continue
        if component in TRANSLATION_TOKENS and translation is None:
            translation = TRANSLATION_TOKENS[component]
        elif component in ROTATION_TOKENS and rotation is None:
            rotation = ROTATION_TOKENS[component]
        else:
            unknown.append(component)

    if unknown and translation is None and rotation is None:
        return "+".join(u.replace("_", "+").title() for u in unknown)

    return f"{translation or 'IDLE'} | {rotation or 'IDLE'}"


def normalize_slot_label(raw: str | None) -> str:
    """Like :func:`pretty_action_text`, but ``IDLE`` / ``PAD`` slots collapse to
    the idle label. Used for the synthetic reference / padding slots."""
    text = str(raw or "").strip()
    if not text or text.upper() in {"IDLE", "PAD"}:
        return IDLE_LABEL
    return pretty_action_text(text)


# Per-component expected visual behaviour. Each entry describes TWO anchors:
# where EXISTING content goes, and where NEW content enters. This dual anchoring
# is what makes pitch direction judgeable in scenes without sky/ground.
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


def split_pretty_label(pretty_label: str) -> tuple[list[str], list[str]]:
    """Return ``(translation_atoms, rotation_atoms)`` from ``"<Trans> | <Rot>"``."""
    translation_part, _, rotation_part = pretty_label.partition("|")

    def atoms(part: str) -> list[str]:
        part = part.strip()
        if not part or part.upper() == "IDLE":
            return []
        return [a.strip() for a in part.split("+") if a.strip() and a.strip().upper() != "IDLE"]

    return atoms(translation_part), atoms(rotation_part)


def expand_action_expectation(pretty_label: str) -> list[str]:
    """Expand one action label into explicit per-component expected visuals plus
    a strict correctness criterion. Returns indented sub-lines for the prompt.

    These lines are REFERENCE guidance only. The value the model must copy into
    the output ``action_label`` field is always just the bare
    ``"<Translation> | <Rotation>"`` label, never any of this expansion text.
    """
    translation_atoms, rotation_atoms = split_pretty_label(pretty_label)
    commanded = translation_atoms + rotation_atoms
    lines: list[str] = ["    (expected behaviour, reference only -- do NOT copy into action_label):"]

    if not commanded:
        lines.append("    - IDLE: no translation and no rotation. The scene CONTENT should stay essentially still.")
        lines.append("    - Correct only if there is NO discernible camera motion (no shift, no zoom, no tilt).")
        return lines

    for atom in commanded:
        lines.append(f"    - {COMPONENT_EXPECTATION.get(atom, f'{atom}: (motion as named).')}")

    # Components that are NOT commanded must NOT appear (motion purity). Each
    # motion sub-axis is treated independently so that e.g. a commanded Forward
    # (depth) still forbids un-commanded lateral drift, and a commanded yaw still
    # forbids un-commanded pitch.
    forbidden: list[str] = []
    if not any(a in ("Forward", "Backward") for a in translation_atoms):
        forbidden.append("no forward/backward zoom (objects must not grow or shrink)")
    if not any(a in ("Left", "Right") for a in translation_atoms):
        forbidden.append("no left/right translation drift")
    if not any(a in ("YawLeft", "YawRight") for a in rotation_atoms):
        forbidden.append("no left/right yaw turn")
    if not any(a in ("PitchUp", "PitchDown") for a in rotation_atoms):
        forbidden.append("no up/down pitch tilt")

    if len(commanded) > 1:
        lines.append(
            "    - Expected combined: ALL of the above components must be visible simultaneously; "
            "a video that performs only one of them is NOT fully correct."
        )
    criterion = "    - Counts as CORRECT only if EVERY commanded component above matches its direction"
    criterion += (", AND there is " + ", ".join(forbidden) + ".") if forbidden else "."
    lines.append(criterion)
    return lines
