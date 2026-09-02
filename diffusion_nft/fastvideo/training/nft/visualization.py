"""Visualization — task mask, cond latents, and action overlay.

Action overlay functions are re-exported from the shared action_overlay module.
"""
import torch

from fastvideo.training.nft_train_pipeline_base import merge_tensor_by_mask
from fastvideo.utils.action_overlay import (
    ACTION_FLAG_NAMES,
    ACTION_LABEL_TO_FLAGS,
    decode_action_label,
    draw_action_keys_on_frame,
    expand_action_labels_to_video_frames,
    render_action_overlay_video,
)


def get_task_mask(task_type, latent_target_length):
    if task_type == "t2v":
        mask = torch.zeros(latent_target_length)
    elif task_type == "i2v":
        mask = torch.zeros(latent_target_length)
        mask[0] = 1.0
    else:
        raise ValueError(f"{task_type} is not supported !")
    return mask


def prepare_cond_latents(task_type, cond_latents, latents, multitask_mask):
    if cond_latents is not None and task_type == "i2v":
        latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
        latents_concat[:, :, 1:, :, :] = 0.0
    else:
        latents_concat = torch.zeros(
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
        ).to(latents.device)

    mask_zeros = torch.zeros(
        latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
    )
    mask_ones = torch.ones(
        latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
    )
    mask_concat = merge_tensor_by_mask(
        mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2
    ).to(device=latents.device)

    return torch.concat([latents_concat, mask_concat], dim=1)


__all__ = [
    "ACTION_FLAG_NAMES",
    "ACTION_LABEL_TO_FLAGS",
    "decode_action_label",
    "draw_action_keys_on_frame",
    "expand_action_labels_to_video_frames",
    "get_task_mask",
    "prepare_cond_latents",
    "render_action_overlay_video",
]
