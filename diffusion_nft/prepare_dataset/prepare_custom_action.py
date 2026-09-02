"""Action to Pose Conversion Module Converts camera motion actions (forward, backward, left, right,
yaw, pitch) into camera poses represented as 4x4 transformation matrices for video generation
training."""

import numpy as np
import json
import torch
from scipy.spatial.transform import Rotation as R
import random
from tqdm import tqdm


def rot_x(theta):
    """Generate rotation matrix around X-axis (pitch).

    Args:
        theta: Rotation angle in radians

    Returns:
        3x3 rotation matrix for X-axis rotation
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    """Generate rotation matrix around Y-axis (yaw).

    Args:
        theta: Rotation angle in radians

    Returns:
        3x3 rotation matrix for Y-axis rotation
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    """Generate rotation matrix around Z-axis (roll).

    Args:
        theta: Rotation angle in radians

    Returns:
        3x3 rotation matrix for Z-axis rotation
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def action_to_pose(motions, existing_pose=None):
    """Convert a sequence of camera motion actions into camera poses.

    Args:
        motions: List of motion dictionaries, each containing:
                 - "forward": Move along camera's positive Z-axis
                 - "backward": Move along camera's negative Z-axis
                 - "right": Move along camera's positive X-axis
                 - "left": Move along camera's negative X-axis
                 - "yaw": Rotate around camera's Y-axis (OpenCV convention)
                 - "pitch": Rotate around camera's X-axis
                 - "roll": Rotate around camera's Z-axis
                 Example: [{"forward": 1.0}, {"yaw": np.pi/2}, {"pitch": np.pi/6}]
        existing_pose: Optional list of existing poses. If provided, the transformation
                      continues from the last pose in the list.

    Returns:
        List of 4x4 transformation matrices representing camera poses for each action.
        Each matrix is in world coordinates.
    """
    # Initialize transformation matrix
    if existing_pose is not None:
        T = existing_pose[-1].copy()
    else:
        T = np.eye(4)  # Start with identity matrix (origin)

    poses = []

    for move in motions:
        # Apply rotations first (in camera's local coordinate system)
        if "yaw" in move:
            R = rot_y(move["yaw"])
            T[:3, :3] = T[:3, :3] @ R
        if "pitch" in move:
            R = rot_x(move["pitch"])
            T[:3, :3] = T[:3, :3] @ R

        # Apply translations (in camera's local coordinate system, then transform to world)
        forward = move.get("forward", 0.0)
        right = move.get("right", 0.0)
        backward = move.get("backward", 0.0)
        left = move.get("left", 0.0)

        # Forward: positive Z in camera space
        if forward != 0:
            local_t = np.array([0, 0, forward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Right: positive X in camera space
        if right != 0:
            local_t = np.array([right, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Backward: negative Z in camera space
        if backward != 0:
            local_t = np.array([0, 0, -backward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Left: negative X in camera space
        if left != 0:
            local_t = np.array([-left, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        poses.append(T.copy())

    return poses


def one_hot_to_one_dimension(one_hot):
    """Convert one-hot encoded action vectors to single dimension labels.

    Args:
        one_hot: Tensor of one-hot encoded vectors with 4 dimensions representing
                 [forward, backward, left, right] and their combinations

    Returns:
        Tensor of integer labels (0-8) representing action classes:
        0: no action, 1: forward, 2: backward, 3: left, 4: right,
        5: forward+left, 6: forward+right, 7: backward+left, 8: backward+right
    """
    mapping = {
        (0, 0, 0, 0): 0,  # No action
        (1, 0, 0, 0): 1,  # Forward only
        (0, 1, 0, 0): 2,  # Backward only
        (0, 0, 1, 0): 3,  # Left only
        (0, 0, 0, 1): 4,  # Right only
        (1, 0, 1, 0): 5,  # Forward + Left
        (1, 0, 0, 1): 6,  # Forward + Right
        (0, 1, 1, 0): 7,  # Backward + Left
        (0, 1, 0, 1): 8,  # Backward + Right
    }
    y = torch.tensor([mapping[tuple(row.tolist())] for row in one_hot])
    return y


def generate_revisit_action(num):
    """
    Generate a revisit trajectory pattern: turn left for first half, then turn right.
    Useful for training camera return-to-origin behaviors.

    Args:
        num: Total number of actions to generate

    Returns:
        List of action dictionaries with yaw rotations
    """
    trajectory = []
    ANGLE = np.deg2rad(2)  # 2-degree rotation per step

    # First half: turn left (negative yaw)
    for _ in range(num // 2):
        trajectory.append({"yaw": -ANGLE})

    # Second half: turn right (positive yaw)
    for _ in range(num - num // 2):
        trajectory.append({"yaw": ANGLE})

    return trajectory


def generate_random_action(num):
    """Generates a single, unique basic trajectory of 50 steps with fixed parameters."""
    trajectory = []

    ANGLE = np.deg2rad(2)
    STEP = 0.1

    basic_chunks = [
        # single translation, weight 2
        {"forward": STEP},
        {"backward": STEP},
        {"left": STEP},
        {"right": STEP},
        # single rotation, weight 2
        {"yaw": ANGLE},
        {"yaw": -ANGLE},
        {"pitch": ANGLE},
        {"pitch": -ANGLE},
        # translation-only combos, weight 4
        {"forward": STEP, "right": STEP},
        {"forward": STEP, "left": STEP},
        {"backward": STEP, "right": STEP},
        {"backward": STEP, "left": STEP},
        # rotation-only combos, weight 4
        {"yaw": ANGLE, "pitch": ANGLE},
        {"yaw": ANGLE, "pitch": -ANGLE},
        {"yaw": -ANGLE, "pitch": ANGLE},
        {"yaw": -ANGLE, "pitch": -ANGLE},
        # translation + yaw combos, weight 4
        {"forward": STEP, "yaw": ANGLE},
        {"forward": STEP, "yaw": -ANGLE},
        {"backward": STEP, "yaw": ANGLE},
        {"backward": STEP, "yaw": -ANGLE},
        {"left": STEP, "yaw": ANGLE},
        {"left": STEP, "yaw": -ANGLE},
        {"right": STEP, "yaw": ANGLE},
        {"right": STEP, "yaw": -ANGLE},
        # translation + pitch combos, weight 4
        {"forward": STEP, "pitch": ANGLE},
        {"forward": STEP, "pitch": -ANGLE},
        {"backward": STEP, "pitch": ANGLE},
        {"backward": STEP, "pitch": -ANGLE},
        {"left": STEP, "pitch": ANGLE},
        {"left": STEP, "pitch": -ANGLE},
        {"right": STEP, "pitch": ANGLE},
        {"right": STEP, "pitch": -ANGLE},
    ]

    weights = [
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        6,
        6,
        6,
        6,
        2,
        2,
        2,
        2,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]

    trajectory = []
    total_steps = 0
    while total_steps < num:
        chunk_to_add = random.choices(basic_chunks, weights=weights, k=1)[0]

        num_steps = random.randint(2, 16)

        num_steps = min(num_steps, num - total_steps)

        for _ in range(num_steps):
            trajectory.append(chunk_to_add.copy())

        total_steps += num_steps

    return trajectory


intrinsic = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Generate random camera trajectories for training. Writes the "
            "JSON consumed by data.random_pose_path (see docs/DATA_FORMAT.md)."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./dataset/random_poses.json",
        help="output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-trajectories",
        type=int,
        default=1000,
        help="how many trajectories to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=128,
        help=(
            "frames per trajectory; must be >= model.window_frames in your "
            "training config (default: %(default)s)"
        ),
    )
    args = parser.parse_args()

    all_rand_poses = []
    for _ in tqdm(range(args.num_trajectories)):
        random_action = generate_random_action(args.frames)
        pose = action_to_pose(random_action)
        custom_w2c = {}
        for i, p in enumerate(pose):
            custom_w2c[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

        all_rand_poses.append(custom_w2c)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_rand_poses, f, indent=4, ensure_ascii=False)
    print(
        f"wrote {len(all_rand_poses)} trajectories "
        f"x {args.frames} frames -> {args.output}"
    )
