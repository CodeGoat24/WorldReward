# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/utils.py
"""Utils for model executor."""

from typing import Any

import torch


# TODO(PY): move it elsewhere
def auto_attributes(init_func):
    """Decorator that automatically adds all initialization arguments as object attributes.

    Example:
        @auto_attributes
        def __init__(self, a=1, b=2):
            pass

        # This will automatically set:
        # - self.a = 1 and self.b = 2
        # - self.config.a = 1 and self.config.b = 2
    """

    def wrapper(self, *args, **kwargs):
        # Get the function signature
        import inspect

        signature = inspect.signature(init_func)
        parameters = signature.parameters

        # Get parameter names (excluding 'self')
        param_names = list(parameters.keys())[1:]

        # Bind arguments to parameters
        bound_args = signature.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Create config object if it doesn't exist
        if not hasattr(self, "config"):
            self.config = type("Config", (), {})()

        # Set attributes on self and self.config
        for name in param_names:
            if name in bound_args.arguments:
                value = bound_args.arguments[name]
                setattr(self, name, value)
                setattr(self.config, name, value)

        # Call the original __init__ function
        return init_func(self, *args, **kwargs)

    return wrapper


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key
        ), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from fastvideo.platforms import current_platform

        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader) -> Any:

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        torch._sync(param)

    return _synced_weight_loader


def extract_layer_index(layer_name: str) -> int:
    """Extract the layer index from the module name.

    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert (
        len(int_vals) == 1
    ), f"layer name {layer_name} should only contain one integer"
    return int_vals[0]


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Modulate by shift and scale.

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))  # type: ignore[union-attr]
    elif scale is None:
        return x + shift.unsqueeze(1)  # type: ignore[union-attr]
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # type: ignore[union-attr]


import json
import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.spatial.transform import Rotation as R_scipy
import math

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from scipy.spatial.transform import Rotation as R
from fastvideo.distributed import (
    get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.distributed import (
    get_sp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
)

from fastvideo.utils.logger import init_logger

logger = init_logger(__name__)


def camera_center_normalization(w2c, nframe):
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
    w2c = c2w_view0 @ w2c
    return w2c


# action keys
ACTION_KEYS = [
    "D",
    "DL",
    "DR",
]


# *******************************Add the memory part********************************
def calculate_pose_distance_from_w2c(
    w2c_1: np.ndarray,
    w2c_2: np.ndarray,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> float:
    """Compute a combined pose distance between two 4x4 W2C (World-to-Camera) matrices.

    The distance quantifies how similar two camera poses are, akin to their
    FOV overlap.

    Args:
        w2c_1 (np.ndarray): First camera's 4x4 World-to-Camera matrix.
        w2c_2 (np.ndarray): Second camera's 4x4 World-to-Camera matrix.
        pos_weight (float): Weight of the positional distance.
        ang_weight (float): Weight of the angular distance.

    Returns:
        float: Combined distance between the two poses.
    """

    def w2c_to_6d_pose(w2c_matrix: np.ndarray) -> np.ndarray:
        """Convert a 4x4 World-to-Camera (W2C) matrix into a 6D pose.

        The 6D pose tuple is (x, y, z, pitch, yaw, roll).
        """
        # Extract rotation matrix R and translation vector t
        R_cw = w2c_matrix[:3, :3]
        t_cw = w2c_matrix[:3, 3]

        # Camera position in world coordinates C_world
        # C_world = -R_cw.T @ t_cw
        C_world = -np.dot(R_cw.T, t_cw)

        # Convert the rotation matrix to Euler angles (pitch, yaw, roll).
        # Note: scipy's default Euler order is ZYX (yaw, pitch, roll); we pick
        # the axes explicitly to match the common (pitch, yaw, roll) order.
        r = R_scipy.from_matrix(R_cw)
        pitch, yaw, roll = r.as_euler("yxz", degrees=True)

        return np.array([C_world[0], C_world[1], C_world[2], pitch, yaw, roll])

    # 1. Convert both W2C matrices to 6D poses
    pose1_6d = w2c_to_6d_pose(w2c_1)
    pose2_6d = w2c_to_6d_pose(w2c_2)

    # 2. Positional distance (Euclidean)
    pos1 = pose1_6d[:3]
    pos2 = pose2_6d[:3]
    spatial_distance = np.linalg.norm(pos1 - pos2)

    # 3. Angular distance (accounting for wrap-around)
    angles1 = pose1_6d[3:]
    angles2 = pose2_6d[3:]

    angle_diff = np.abs(angles1 - angles2)
    # Wrap the angle difference to the smallest circular distance
    angular_distance_vector = np.minimum(angle_diff, 360 - angle_diff)
    # Euclidean norm as the combined angular distance
    angular_distance = np.linalg.norm(angular_distance_vector)

    # 4. Combine both distances into the final pose distance
    total_distance = (
        pos_weight * spatial_distance + ang_weight * angular_distance
    )

    return total_distance


# TO DO: add a chunk mechanism to chunk the long sequence to the window size
# --- Helper: composite clip distance ---
def calculate_complex_clip_distance(
    w2c_list: List[np.ndarray],
    query_clip_indices: List[int],
    historical_clip_indices: List[int],
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> float:
    """Compute a composite pose distance between a query clip and a historical clip.

    The distance averages, over sampled query frames, the mean distance to
    every frame of the historical clip.
    """
    # 1. Pick the sampled frame indices of the query clip: start from the
    # second element (index 1) and take every other frame, e.g. for
    # query_clip_indices = [10, 11, 12, 13, 14, 15] sample 11, 13, 15.

    # The query clip needs at least 2 frames to sample from
    if len(query_clip_indices) < 2:
        # Prediction sequence too short: use the first and last frames
        sample_indices = [query_clip_indices[0], query_clip_indices[-1]]
    else:
        # Local sample indices within query_clip_indices: start 1, step 2
        sample_indices = [
            query_clip_indices[i]
            for i in np.arange(1, len(query_clip_indices), 2)
        ]

    total_avg_distance = 0.0

    # 2. Iterate over the sampled frames
    for query_idx in sample_indices:
        query_w2c = w2c_list[query_idx]

        dists_from_query_frame = []

        # 3. Distance from this sampled frame to every historical frame
        for hist_idx in historical_clip_indices:
            hist_w2c = w2c_list[hist_idx]
            dist = calculate_pose_distance_from_w2c(
                query_w2c, hist_w2c, pos_weight, ang_weight
            )
            dists_from_query_frame.append(dist)

        # Accumulate this frame's mean distance to the historical clip
        total_avg_distance += np.mean(dists_from_query_frame)

    # 4. Final distance: average over all sampled frames
    final_clip_distance = total_avg_distance / len(sample_indices)

    return final_clip_distance


# --- Utility 1: rotation matrix to Pitch/Yaw ---
def rotation_matrix_to_angles(
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate pitch and yaw from a 3x3 camera rotation matrix R.

    Uses the common computer-vision convention with the Z axis pointing
    forward. The angle conventions match is_inside_fov_3d_hv:
    - Yaw/azimuth in the XZ plane: atan2(x, z).
    - Pitch/elevation relative to the horizontal plane: atan2(y, sqrt(x^2 + z^2)).

    The forward vector is the Z axis of the C2W matrix (C2W = W2C_inv), i.e.
    the camera's viewing direction expressed in world coordinates.
    """

    # 1. Forward vector in world coordinates.
    # R is R_c2w.T; we want the Z axis of R_c2w = R_w2c.T
    R_c2w = R.T

    # Forward vector: third column (Z axis) of the C2W matrix
    fwd = R_c2w[:, 2]  # shape (3,)

    x = fwd[0]
    y = fwd[1]
    z = fwd[2]

    # 2. Yaw = atan2(x, z), with Z forward and X lateral
    yaw_rad = torch.atan2(x, z)
    yaw_deg = yaw_rad * (180.0 / math.pi)

    # 3. Pitch = atan2(y, sqrt(x^2 + z^2))
    pitch_rad = torch.atan2(y, torch.sqrt(x**2 + z**2))
    pitch_deg = pitch_rad * (180.0 / math.pi)

    return pitch_deg, yaw_deg


def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """Uniformly sample points inside a sphere of the given radius.

    :param n_points: Number of points to generate.
    :param radius: Sphere radius.
    :return: Tensor of shape (n_points, 3) with (x, y, z) coordinates.
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    # Uniform volume sampling: r = R * u^(1/3)
    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    # Uniform polar-angle sampling: theta = arccos(1 - 2*u)
    theta = torch.acos(1 - 2 * samples_u)

    # To Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def is_inside_fov_3d_hv(
    points: torch.Tensor,
    center: torch.Tensor,
    center_pitch: torch.Tensor,
    center_yaw: torch.Tensor,
    fov_half_h: torch.Tensor,
    fov_half_v: torch.Tensor,
) -> torch.Tensor:
    """Check whether points fall inside a 3D view frustum defined by a center, pitch, and yaw.

    :param points: (N, 3) or (N, B, 3) sample point coordinates.
    :param center: (3) or (B, 3) camera center coordinates.
    :param center_pitch: (1) or (B) pitch of the central view direction (degrees).
    :param center_yaw: (1) or (B) yaw of the central view direction (degrees).
    :param fov_half_h: Horizontal half-FOV (degrees).
    :param fov_half_v: Vertical half-FOV (degrees).
    :return: (N) or (N, B) boolean tensor, True where the point is inside the FOV.
    """
    # Make center, pitch, yaw broadcastable
    if points.ndim == 2:  # N, 3
        vectors = points - center[None, :]
        C = 1  # batch size 1
    elif points.ndim == 3:  # N, B, 3
        vectors = points - center[None, ...]
        # Reshape center_pitch/yaw to (1, B) for broadcasting
        center_pitch = (
            center_pitch[None, :] if center_pitch.ndim == 1 else center_pitch
        )
        center_yaw = center_yaw[None, :] if center_yaw.ndim == 1 else center_yaw
    else:
        raise ValueError("points must have shape (N, 3) or (N, B, 3)")

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    # Horizontal angle (yaw/azimuth), Z axis forward
    azimuth = torch.atan2(x, z) * (180 / math.pi)

    # Vertical angle (pitch/elevation)
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)

    # Angle difference to the central view direction (with wrap-around)
    diff_azimuth = azimuth - center_yaw
    # Normalize the difference to [-180, 180]
    diff_azimuth = torch.remainder(diff_azimuth + 180, 360) - 180

    diff_elevation = elevation - center_pitch
    diff_elevation = torch.remainder(diff_elevation + 180, 360) - 180

    # Check against the FOV limits
    in_fov_h = diff_azimuth.abs() < fov_half_h
    in_fov_v = diff_elevation.abs() < fov_half_v

    return in_fov_h & in_fov_v


# --- Utility 2: Euclidean distance between W2C camera centers ---
def calculate_euclidean_distance(
    w2c_matrix_1: torch.Tensor, w2c_matrix_2: torch.Tensor
) -> torch.Tensor:
    """Euclidean distance between two camera centers."""
    # Camera position in world coordinates: P_w = -R^T * t
    R1 = w2c_matrix_1[:3, :3]
    t1 = w2c_matrix_1[:3, 3]
    R2 = w2c_matrix_2[:3, :3]
    t2 = w2c_matrix_2[:3, 3]

    P_w1 = -R1.T @ t1
    P_w2 = -R2.T @ t2

    distance = torch.linalg.norm(P_w1 - P_w2)
    return distance


def calculate_fov_overlap_similarity_refine(
    w2c_matrix_curr: torch.Tensor,
    w2c_matrix_hist: torch.Tensor,
    fov_h_deg: float = 105.0,
    fov_v_deg: float = 75.0,
    device=None,
    points_local=None,
) -> float:
    """Estimate FOV overlap similarity between two W2C poses via Monte Carlo sampling.

    similarity = #points in (Curr_FOV intersect Hist_FOV) / #points in Curr_FOV.

    :param w2c_matrix_curr: (4, 4) W2C matrix of the current frame.
    :param w2c_matrix_hist: (4, 4) W2C matrix of the historical frame.
    :param fov_h_deg, fov_v_deg: FOV parameters.
    :return: Overlap ratio, a float in [0.0, 1.0].
    """
    w2c_matrix_curr = torch.tensor(w2c_matrix_curr, device=device)
    w2c_matrix_hist = torch.tensor(w2c_matrix_hist, device=device)

    # Convert to relative poses
    c2w_matrix_curr = torch.linalg.inv(w2c_matrix_curr)
    c2w_matrix_hist = torch.linalg.inv(w2c_matrix_hist)
    C_inv = w2c_matrix_curr

    w2c_matrix_curr = torch.linalg.inv(C_inv @ c2w_matrix_curr)
    w2c_matrix_hist = torch.linalg.inv(C_inv @ c2w_matrix_hist)

    # --- 1. Decompose W2C matrices into positions and angles ---

    # P_w = -R^T * t
    R_curr, t_curr = w2c_matrix_curr[:3, :3], w2c_matrix_curr[:3, 3]
    R_hist, t_hist = w2c_matrix_hist[:3, :3], w2c_matrix_hist[:3, 3]

    P_w_curr = -R_curr.T @ t_curr
    P_w_hist = -R_hist.T @ t_hist

    # pitch, yaw in degrees
    pitch_curr, yaw_curr = rotation_matrix_to_angles(R_curr)
    pitch_hist, yaw_hist = rotation_matrix_to_angles(R_hist)

    # FOV parameters
    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device)

    # Translate the points to the current camera center P_w_curr (N, 3)
    points_world = points_local + P_w_curr[None, :]

    # --- 3. FOV checks ---

    # Points inside the current FOV (denominator)
    in_fov_curr = is_inside_fov_3d_hv(
        points_world,
        P_w_curr[None, :],
        pitch_curr[None],
        yaw_curr[None],
        fov_half_h,
        fov_half_v,
    )

    # Points inside the historical FOV, by angle only
    in_fov_hist = is_inside_fov_3d_hv(
        points_world,
        P_w_hist[None, :],
        pitch_hist[None],
        yaw_hist[None],
        fov_half_h,
        fov_half_v,
    )

    # Additionally gate by point-to-camera distance
    dist = torch.norm(points_world - P_w_hist.reshape(1, -1), dim=1) < 8.0
    in_fov_hist = in_fov_hist.bool() & dist.reshape(1, -1).bool()
    # --- 4. Overlap ratio ---

    # Numerator: points in the intersection
    overlap_count = (in_fov_curr.bool() & in_fov_hist.bool()).sum().float()

    # Denominator: total points in the current FOV
    fov_curr_count = in_fov_curr.sum().float()
    # print(overlap_count, fov_curr_count)

    if fov_curr_count == 0:
        return 0.0  # avoid division by zero

    # Overlap ratio
    overlap_ratio = overlap_count / fov_curr_count

    return overlap_ratio.item()


def select_aligned_memory_frames_context_per_chunk_w_latent_sink_fov_refine_hunyuan(
    w2c_list: List[np.ndarray],
    current_frame_idx: int,
    memory_frames: int,
    temporal_context_size: int,
    pred_latent_size: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
    device=None,
    points_local=None,
) -> List[int]:
    """Select memory and context frames for a given frame based on 4-frame clip distances.

    Args:
        w2c_list (List[np.ndarray]): All N 4x4 extrinsic matrices.
        current_frame_idx (int): Index of the frame being processed.
        memory_frames (int): Total number of memory frames to select.
        temporal_context_size (int): Total number of context frames to select.
        pos_weight (float): Weight of the positional distance.
        ang_weight (float): Weight of the angular distance.

    Returns:
        List[int]: Indices of the selected memory and context frames.
    """
    if current_frame_idx <= memory_frames:
        return list(range(0, current_frame_idx))

    num_total_frames = len(w2c_list)
    # Check the current frame can form a complete 4-frame clip
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        # Graceful fallback: return all available frames instead of crashing.
        # This can happen when generate_latent_num exceeds the dataset's w2c
        # length (e.g. selected_chunk_id overshoots max_chunk_id by 1).
        safe_idx = min(current_frame_idx, num_total_frames - 1)
        return list(range(0, safe_idx))

    # 1. Select context frames
    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    # 2. Build the candidate pool of memory frames
    candidate_distances = []
    query_clip_indices = list(
        range(
            current_frame_idx,
            (
                current_frame_idx + pred_latent_size
                if current_frame_idx + pred_latent_size <= num_total_frames
                else num_total_frames
            ),
        )
    )

    historical_clip_indices = list(
        range(4, current_frame_idx - temporal_context_size, 4)
    )

    # 3. Pick the most similar `memory_frames` frames
    memory_frames_indices = [
        0,
        1,
        2,
        3,
    ]  # add the first latent frame as context
    memory_frames = (
        memory_frames - temporal_context_size - len(memory_frames_indices)
    )

    # Treat every historical clip as a memory-frame candidate. Clip start
    # indices must be multiples of 4 and must not overlap the context frames.
    for hist_idx in historical_clip_indices:
        total_dist = 0
        hist_w2c_1 = w2c_list[hist_idx]
        hist_w2c_2 = w2c_list[hist_idx + 2]
        for query_idx in query_clip_indices:
            dist_1_for_query_idx = (
                1.0
                - calculate_fov_overlap_similarity_refine(
                    w2c_list[query_idx],
                    hist_w2c_1,
                    fov_h_deg=60.0,
                    fov_v_deg=35.0,
                    device=device,
                    points_local=points_local,
                )
            )
            dist_2_for_query_idx = (
                1.0
                - calculate_fov_overlap_similarity_refine(
                    w2c_list[query_idx],
                    hist_w2c_2,
                    fov_h_deg=60.0,
                    fov_v_deg=35.0,
                    device=device,
                    points_local=points_local,
                )
            )
            dist_for_query_idx = (
                dist_1_for_query_idx + dist_2_for_query_idx
            ) / 2.0
            total_dist += dist_for_query_idx
        # Average the per-frame means into the final clip distance
        final_clip_distance = total_dist / len(query_clip_indices)
        # Store (clip start index, mean distance)
        candidate_distances.append((hist_idx, final_clip_distance))

        # Sort by mean distance, ascending
    candidate_distances.sort(key=lambda x: x[1])

    # Walk the sorted candidates until enough memory frames are collected
    for start_idx, _ in candidate_distances:
        if start_idx not in memory_frames_indices:
            memory_frames_indices.extend(range(start_idx, start_idx + 4))

        # Stop once the memory budget is met
        if len(memory_frames_indices) >= memory_frames:
            break

    # 4. Merge and deduplicate the selections
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(memory_frames_indices)

    final_selected_frames = sorted(list(selected_frames_set))
    # assert len(final_selected_frames) == memory_frames + temporal_context_size
    return final_selected_frames


# --- Process all frames ---
def process_all_frames_for_memory(
    w2c_list: List[np.ndarray],
    window_size: int,
    memory_frames: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> Dict[int, List[int]]:
    """Select memory and context frames for every frame beyond window_size.

    Args:
        w2c_list: All N 4x4 extrinsic matrices.
        window_size: Initial window size; processing starts from this frame.
        memory_frames: Total number of memory frames to select.
        pos_weight: Weight of the positional distance.
        ang_weight: Weight of the angular distance.

    Returns:
        Dict[int, List[int]]: Maps frame index to its selected memory and
        context frames.
    """
    if window_size >= len(w2c_list) or window_size < 4:
        print("Warning: window_size must be < total frames and >= 4; nothing to process.")
        return {}

    pred_latent_size = window_size - memory_frames  # number of predicted frames
    step = 4  # advance 4 frames at a time

    all_selections = {}

    # Nothing to do if w2c_list fits within window_size - memory_frames
    if len(w2c_list) <= window_size - memory_frames:
        print(
            "Warning: total frames <= window_size - memory_frames; "
            "no frames need memory selection."
        )
        return all_selections

    # Iterate from window_size to the end of the list
    for current_frame_idx in range(
        memory_frames, len(w2c_list) - pred_latent_size + step, step
    ):


        selected_frames = select_memory_frames(
            w2c_list, current_frame_idx, memory_frames, pos_weight, ang_weight
        )
        all_selections[current_frame_idx] = selected_frames

    return all_selections


# *******************************Add the memory part********************************


def get_normalized_dir_diff(x_inv):
    # Extract direction vectors
    dirs = x_inv[:, :3, 3]  # shape: (N, 3)

    # Adjacent-frame differences
    diff = torch.zeros_like(dirs)
    diff[1:] = dirs[1:] - dirs[:-1]

    # Normalize non-zero vectors
    norms = torch.norm(diff, dim=1, keepdim=True)
    norms[0] = 1.0  # avoid division by zero on the first row
    diff_norm = diff / norms

    # Keep the first entry at (0,0,0)
    diff_norm[0] = torch.tensor([0.0, 0.0, 0.0], dtype=diff.dtype)

    return norms, diff_norm


def relative_rotations(c2w_mats):
    """
    c2w_mats: (N, 4, 4) camera c2w matrices
    Returns (N-1, 3): per-row relative rotation angles (degrees) around x, y, z
    between adjacent frames.
    """
    rots = c2w_mats[:, :3, :3]  # rotation part
    rel_angles = []

    for i in range(len(rots) - 1):
        R_i = rots[i]
        R_j = rots[i + 1]
        R_rel = R_i.T @ R_j  # relative rotation matrix

        # --- Convert R_rel to Euler angles (XYZ order) ---
        # Assumes a right-handed frame; angles in [-180, 180] degrees
        sy = torch.sqrt(R_rel[0, 0] ** 2 + R_rel[1, 0] ** 2)

        if sy > 1e-6:  # regular case
            x = torch.atan2(R_rel[2, 1], R_rel[2, 2])
            y = torch.atan2(-R_rel[2, 0], sy)
            z = torch.atan2(R_rel[1, 0], R_rel[0, 0])
        else:  # singular case (gimbal lock)
            x = torch.atan2(-R_rel[1, 2], R_rel[1, 1])
            y = torch.atan2(-R_rel[2, 0], sy)
            z = 0.0

        angles = torch.rad2deg(torch.stack([x, y, z]))
        rel_angles.append(angles)

    return torch.stack(rel_angles)
