import logging
import math
import os

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Gate tqdm bars on global rank 0 only (otherwise every one of 32 ranks
# emits its own \r-based progress bar and the log becomes 32x noisier).
# Also slow the refresh so log is readable when tailed as plain text.
_IS_RANK0 = int(os.environ.get("RANK", "0")) == 0
_TQDM_KW = dict(disable=not _IS_RANK0, mininterval=30.0, miniters=1)


def decode_fp32_resilient(vae, latents):
    """fp32 VAE decode that tolerates caching-allocator fragmentation.

    Both decode sites in a rollout sit immediately after an ODE loop that has
    churned through sampling_steps x chunk activations, so the allocator is left
    full of small reserved-but-unallocated blocks. The decode then needs one
    large contiguous fp32 buffer and can fail while plenty of memory is free in
    aggregate. We have seen a run die exactly this way at the first step the
    curriculum assigns a deeper chunk:

        Tried to allocate 1.15 GiB, 325.44 MiB free
        72.02 GiB allocated by PyTorch
        5.93 GiB reserved by PyTorch but unallocated   <- 5x the request

    Both the peak and the fragmentation grow with chunk depth, and max_chunk_id
    is 12, so this has to hold well past chunk 6 rather than just survive it.

    The retry is not redundant with the first empty_cache: that one can only
    return blocks that were already free, whereas everything the *failed*
    attempt reserved is released as the exception unwinds, so the second call
    sees strictly more headroom.

    NOT usable here: PYTORCH_CUDA_ALLOC_CONF=expandable_segments, which is what
    the PyTorch error message recommends. It keeps large virtual reservations
    alive across empty_cache(), starving vLLM's cumem_allocator on /wake_up and
    reproducing the cumem_allocator.cpp:139 OOM documented in scripts/common.sh.

    Deliberately does NOT fall back to a lower precision: vae_precision is fp32
    and decoding in bf16 would change the pixels, hence every reward computed on
    them. Memory must not be bought with signal quality.
    """
    torch.cuda.empty_cache()
    try:
        return vae.decode(latents.to(torch.float32), return_dict=False)[0]
    except torch.OutOfMemoryError:
        logger.warning(
            "fp32 VAE decode hit OOM on shape %s; retrying after empty_cache()",
            tuple(latents.shape),
        )
        torch.cuda.empty_cache()
        return vae.decode(latents.to(torch.float32), return_dict=False)[0]


from fastvideo.distributed import get_local_torch_device
from fastvideo.utils.video_utils import get_generated_videos_base, save_video
from fastvideo.training.nft.visualization import render_action_overlay_video
from fastvideo.models.utils import (
    select_aligned_memory_frames_context_per_chunk_w_latent_sink_fov_refine_hunyuan,
)


def create_sample_kv_cache(pipeline):
    _kv_cache = []
    _kv_cache_neg = []
    for _ in range(len(pipeline.transformer.double_blocks)):
        _kv_cache.append({"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None})
        _kv_cache_neg.append({"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None})
    return {"positive": _kv_cache, "negative": _kv_cache_neg}


def build_kv_cache_from_previous_chunks(
    pipeline,
    current_kv_cache,
    latents_curr,
    training_batch,
    generate_latent_num,
    update_latent_num,
    stabilization_level,
    negative=False,
):
    device = latents_curr.device
    kv_cache_latent_num = generate_latent_num - update_latent_num
    if generate_latent_num >= 20:
        selected_frame_indices = select_aligned_memory_frames_context_per_chunk_w_latent_sink_fov_refine_hunyuan(
            training_batch.w2c[:, :generate_latent_num, :, :][0].to(torch.float32).cpu().detach().numpy(),
            generate_latent_num - 4,
            memory_frames=20,
            temporal_context_size=12,
            pred_latent_size=4,
            points_local=pipeline.points_local.to(device),
            device=device,
        )
    else:
        selected_frame_indices = list(range(0, kv_cache_latent_num))
    if kv_cache_latent_num <= 0:
        return current_kv_cache
    latents_for_cache = latents_curr[:, :, selected_frame_indices, :, :]
    cond_latents_for_cache = training_batch.cond_latents[:, :, selected_frame_indices, :, :]
    latents_input_for_cache = torch.cat([latents_for_cache, cond_latents_for_cache], dim=1)
    timestep_for_cache = torch.full((len(selected_frame_indices),), stabilization_level - 1, device=latents_curr.device, dtype=torch.long)
    viewmats_for_cache = training_batch.w2c[:, selected_frame_indices].to(device)
    Ks_for_cache = training_batch.intrinsic[:, selected_frame_indices].to(device)
    action_for_cache = training_batch.action[:, selected_frame_indices].to(device)
    for kv_cache_layer in current_kv_cache["positive"]:
        kv_cache_layer["k_vision"] = None
        kv_cache_layer["v_vision"] = None
    cache_kwargs = {
        "hidden_states": latents_input_for_cache,
        "timestep": timestep_for_cache,
        "timestep_r": None,
        "return_dict": False,
        "mask_type": "i2v",
        "action": action_for_cache,
        "viewmats": viewmats_for_cache,
        "Ks": Ks_for_cache,
        "kv_cache": current_kv_cache["positive"],
        "cache_vision": True,
        "rope_temporal_size": latents_for_cache.shape[2],
        "start_rope_start_idx": 0,
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), torch.no_grad():
        current_kv_cache["positive"] = pipeline.transformer(txt_branch=False, input_dict=cache_kwargs)
    return current_kv_cache, selected_frame_indices


def flux_step(model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output
    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)
    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma
        prev_sample_mean = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
    return prev_sample_mean, pred_original_sample


def flow_grpo_step(model_output, latents, eta, sigmas, index, prev_sample=None, shift=5.0):
    """Rectified-flow SDE step with σ-adaptive noise scale.

    Flow-GRPO's std_dev_t = sqrt(s / (1 - s)) * eta assumes an *unshifted*
    rectified-flow sigma s ∈ [0, 1] uniformly. We use a shifted schedule
    (shift=5 in the caller) where high-σ region is compressed; directly
    plugging the shifted σ into sqrt(σ / (1 - σ)) makes the denominator
    explode to ~0.005 at index=0 → noise scale ~50× too large → blown-out
    frames.

    Fix: recover the unshifted s from the shifted σ and use s everywhere
    the original formula references the rectified-flow time variable.
      σ = s * shift / (1 + (shift - 1) * s)
      s = σ / (shift - (shift - 1) * σ)
    With shift=5: σ=1.0 → s=1.0, σ=0.95 → s=0.792, σ=0.1 → s=0.022, σ=0 → s=0.
    At s=1 we still need a small clamp to avoid 1/0.

    Advantages over flux_step's SDE branch:
      - noise auto-shrinks near end of sampling (s → 0) without hard cutoff
      - Fokker-Planck-consistent drift, marginals stay closer to training

    No log_prob output — NFT's DPO-style loss doesn't need it.
    """
    device = model_output.device
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    dt = sigma_prev - sigma  # negative (sigma decreases toward 0)

    pred_original_sample = latents - sigma * model_output

    # Recover pre-shift rectified-flow time variable s. With shift=5:
    #   σ=1.0  → s=1.0     (clamp below)
    #   σ=0.95 → s=0.792
    #   σ=0.5  → s=0.167   (compared to σ/(1-σ)=1.0, this prevents blow-up)
    #   σ=0.1  → s=0.0217
    #   σ=0    → s=0
    s = sigma / (shift - (shift - 1) * sigma)
    # Upper clamp tames the blow-up near s=1 and keeps early-step noise
    # comparable to flux_step's; beyond the clamp, std decays naturally.
    s_clamped = s.clamp(min=1e-3, max=0.7)

    # σ-adaptive noise scale on unshifted time axis.
    std_dev_t = torch.sqrt(s_clamped / (1 - s_clamped)) * eta

    # FP-consistent drift. The (1 - ·) and 1/(2·) factors in the original
    # flow_grpo formula both refer to the unshifted time variable too.
    prev_sample_mean = (
        latents * (1 + std_dev_t ** 2 / (2 * s_clamped) * dt)
        + model_output * (1 + std_dev_t ** 2 * (1 - s_clamped) / (2 * s_clamped)) * dt
    )

    if prev_sample is None:
        variance_noise = torch.randn_like(model_output)
        noise_scale = std_dev_t * torch.sqrt(-dt)
        prev_sample = prev_sample_mean + noise_scale * variance_noise

    return prev_sample, pred_original_sample


@torch.no_grad()
def sample_model_ode(pipeline, training_batch, selected_chunk_id, noise_latents, sampling_steps=None):
    pipeline.transformer.eval()
    chunk_latent_num = pipeline.training_args.single_chunk_size
    stabilization_level = 15
    if sampling_steps is None:
        sampling_steps = pipeline.training_args.sampling_steps
    kv_cache = create_sample_kv_cache(pipeline)
    t_expand_txt = torch.tensor([0]).to(get_local_torch_device()).to(noise_latents.dtype)
    input_dict = {
        "timestep_txt": t_expand_txt,
        "text_states": training_batch.prompt_embed,
        "encoder_attention_mask": training_batch.prompt_mask,
        "vision_states": training_batch.vision_states,
        "mask_type": "i2v",
        "extra_kwargs": training_batch.extra_kwargs,
        "kv_cache": kv_cache["positive"],
        "cache_txt": True,
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), torch.no_grad():
        kv_cache["positive"] = pipeline.transformer(txt_branch=True, input_dict=input_dict)
    sigma_schedule = torch.linspace(1, 0, sampling_steps + 1)
    shift = 5
    sigma_schedule = (shift * sigma_schedule) / (1 + (shift - 1) * sigma_schedule)
    for chunk_i in tqdm(range(0, selected_chunk_id - 1), desc="ODE Chunk", leave=True, **_TQDM_KW):
        torch.cuda.empty_cache()
        generate_latent_num = (chunk_i + 1) * chunk_latent_num
        latents_curr = noise_latents[:, :, :generate_latent_num, :, :]
        cond_latent_curr = training_batch.cond_latents[:, :, :generate_latent_num, :, :]
        w2c_curr = training_batch.w2c[:, :generate_latent_num, :, :]
        intrinsic_curr = training_batch.intrinsic[:, :generate_latent_num, :, :]
        action_curr = training_batch.action[:, :generate_latent_num]
        update_latent_num = chunk_latent_num
        selected_frame_indices = []
        if chunk_i > 0:
            kv_cache, selected_frame_indices = build_kv_cache_from_previous_chunks(
                pipeline, kv_cache, latents_curr, training_batch, generate_latent_num, update_latent_num, stabilization_level
            )
        for i in range(0, sampling_steps):
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timestep_input = torch.full((1, update_latent_num), timestep_value, device=latents_curr.device, dtype=torch.long)
            latent_concat_curr = torch.cat([latents_curr, cond_latent_curr], dim=1)
            input_dict = {
                "hidden_states": latent_concat_curr[:, :, -update_latent_num:, :, :],
                "timestep": timestep_input,
                "timestep_r": None,
                "return_dict": False,
                "mask_type": "i2v",
                "action": action_curr[:, -update_latent_num:],
                "viewmats": w2c_curr[:, -update_latent_num:, :, :],
                "Ks": intrinsic_curr[:, -update_latent_num:, :, :],
                "kv_cache": kv_cache["positive"],
                "cache_vision": False,
                "rope_temporal_size": len(selected_frame_indices) + update_latent_num,
                "start_rope_start_idx": len(selected_frame_indices),
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), torch.no_grad():
                model_pred = pipeline.transformer(txt_branch=False, input_dict=input_dict)[0]
            z, pred_original = flux_step(
                model_pred[:, :, -update_latent_num:, :, :].to(torch.float32),
                latents_curr[:, :, -update_latent_num:, :, :].to(torch.float32),
                eta=0.0,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                grpo=False,
                sde_solver=False,
            )
            latents_curr[:, :, -update_latent_num:, :, :] = z
        latents_curr[:, :, -update_latent_num:, :, :] = pred_original
    return latents_curr


@torch.no_grad()
def sample_reference_model(pipeline, training_batch):
    with pipeline.ema_generator.apply_policy_shadow_to_model(pipeline.transformer):
        pipeline.transformer.eval()
        chunk_latent_num = pipeline.training_args.single_chunk_size
        stabilization_level = 15
        sampling_steps = pipeline.training_args.sampling_steps
        sampling_batch_size = pipeline.training_args.sampling_batch_size
        sigma_schedule = torch.linspace(1, 0, pipeline.training_args.sampling_steps + 1)
        shift = 5
        sigma_schedule = (shift * sigma_schedule) / (1 + (shift - 1) * sigma_schedule)
        latents = training_batch.latents
        _, latent_channels, _, latent_h, latent_w = latents.shape
        latent_t = pipeline.training_args.window_frames
        min_chunk_id = pipeline.training_args.min_chunk_id
        max_chunk_id = pipeline.training_args.max_chunk_id
        chunk_strategy = pipeline.training_args.chunk_selection_strategy
        t = training_batch.current_timestep
        if chunk_strategy == "min2max":
            selected_chunk_id = min_chunk_id + ((t - 1) % (max_chunk_id - min_chunk_id + 1))
        elif chunk_strategy == "max2min":
            selected_chunk_id = max_chunk_id - ((t - 1) % (max_chunk_id - min_chunk_id + 1))
        elif chunk_strategy == "progressive_min2max":
            # Unlock one deeper chunk position every `target` training steps;
            # Round-robin on t within the unlocked bucket [min, frontier], so
            # revisits are deterministic. chunk_sample_alpha > 0 switches to
            # weighted sampling P(k) proportional to
            # 1/(k - min_chunk_id + 1)**alpha, biasing toward shallow chunks.
            target = int(getattr(pipeline.training_args, "chunk_target_repeats", 3))
            target = max(1, target)
            alpha = float(getattr(pipeline.training_args, "chunk_sample_alpha", 0.0))
            frontier = min(min_chunk_id + (t - 1) // target, max_chunk_id)
            bucket_size = frontier - min_chunk_id + 1
            if alpha > 0.0 and bucket_size > 1:
                import random as _random
                # depth = 1 for min_chunk_id, 2 for next, ...
                weights = [1.0 / (d ** alpha) for d in range(1, bucket_size + 1)]
                rng = _random.Random(int(t))
                selected_chunk_id = rng.choices(range(min_chunk_id, frontier + 1), weights=weights, k=1)[0]
            else:
                selected_chunk_id = min_chunk_id + ((t - 1) % bucket_size)
        else:
            raise ValueError(f'Invalid chunk_selection_strategy: {chunk_strategy}, must be one of "min2max", "max2min", or "progressive_min2max".')
        training_batch.selected_chunk_id = selected_chunk_id
        noise = torch.randn((1, latent_channels, latent_t, latent_h, latent_w), device=latents.device, dtype=latents.dtype)
        noise = pipeline.gpu_group.all_gather(noise, dim=0)
        context_latents = None
        context_num = 0
        if selected_chunk_id > 1:
            context_latents = sample_model_ode(pipeline, training_batch, selected_chunk_id, noise)
            context_num = context_latents.shape[2]
        training_batch.sample_kwargs = {}
        pipeline.vae.to(pipeline.device)
        if selected_chunk_id > 1:
            if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor:
                context_latents = context_latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
            else:
                context_latents = context_latents / pipeline.vae.config.scaling_factor
            context_video_frames = decode_fp32_resilient(pipeline.vae, context_latents)
            context_video_frames = ((context_video_frames / 2 + 0.5).clamp(0, 1).cpu().float())
            context_video_frames = np.transpose(context_video_frames[0], (1, 2, 3, 0))
            context_video_frames = (context_video_frames * 255).numpy().astype(np.uint8)
        else:
            context_video_frames = None
        kv_cache = create_sample_kv_cache(pipeline)
        t_expand_txt = torch.tensor([0]).to(get_local_torch_device()).to(noise.dtype)
        input_dict = {
            "timestep_txt": t_expand_txt,
            "text_states": training_batch.prompt_embed,
            "encoder_attention_mask": training_batch.prompt_mask,
            "vision_states": training_batch.vision_states,
            "mask_type": "i2v",
            "extra_kwargs": training_batch.extra_kwargs,
            "kv_cache": kv_cache["positive"],
            "cache_txt": True,
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), torch.no_grad():
            kv_cache["positive"] = pipeline.transformer(txt_branch=True, input_dict=input_dict)
        if selected_chunk_id > 1:
            noise_latents = torch.cat([context_latents, noise[:, :, context_num:, :, :]], dim=2)
            start_chunk_id = selected_chunk_id - 1
            generate_latent_num = selected_chunk_id * chunk_latent_num
            kv_cache, selected_frame_indices = build_kv_cache_from_previous_chunks(
                pipeline, kv_cache, noise_latents[:, :, :generate_latent_num, :, :], training_batch, generate_latent_num, chunk_latent_num, stabilization_level
            )
            training_batch.kv_cache = kv_cache
        else:
            start_chunk_id = 0
            selected_frame_indices = []
            training_batch.kv_cache = kv_cache
        # Check if WorldMirror / HPSv3 rewards are needed (independent gates)
        vlm_prefilter = getattr(pipeline.training_args, "vlm_prefilter_enabled", False)
        wm_enabled = (
            any(pipeline.reward_weights.get(k, 0) > 0 for k in ("action", "fine_action"))
            or vlm_prefilter  # prefilter needs WorldMirror scoring
        )
        hpsv3_enabled_train = any(
            pipeline.reward_weights.get(k, 0) > 0
            for k in ("hpsv3", "hpsv3_quality", "hpsv3_quality_drift")
        )
        original_reward_enabled = wm_enabled or hpsv3_enabled_train
        worldmirror_reward = pipeline.reward_dispatcher.worldmirror
        hpsv3_computer = pipeline.reward_dispatcher.hpsv3
        vlm_reward_enabled = getattr(pipeline, "vlm_reward_enabled", False)
        # Per-candidate reward log bag (mirrors VLM win_rates style):
        # maps local candidate idx -> {reward_name: float}. Printed once
        # after all GRPO candidates are scored, per rank.
        _rollout_rewards: dict[int, dict[str, float]] = {}
        candidate_video_frames = [] if vlm_reward_enabled else None
        for sample_idx in tqdm(range(0, pipeline.training_args.grpo_generation_num), desc="GRPO Batch Sampling", leave=False, **_TQDM_KW):
            new_noise = torch.randn_like(noise)
            noise_latents = torch.cat([context_latents, new_noise[:, :, context_num:, :, :]], dim=2) if context_latents is not None else new_noise
            video_num_in_each_gpu = sampling_batch_size // pipeline.training_args.gpu_para
            rank_in_gpu_group = pipeline.gpu_group.rank_in_group
            for chunk_i in range(start_chunk_id, selected_chunk_id):
                generate_latent_num = (chunk_i + 1) * chunk_latent_num
                latents_curr = noise_latents[:, :, :generate_latent_num, :, :]
                cond_latent_curr = training_batch.cond_latents[:, :, :generate_latent_num, :, :]
                w2c_curr = training_batch.w2c[:, :generate_latent_num, :, :]
                intrinsic_curr = training_batch.intrinsic[:, :generate_latent_num, :, :]
                action_curr = training_batch.action[:, :generate_latent_num]
                update_latent_num = chunk_latent_num
                for i in range(0, sampling_steps):
                    sigma = sigma_schedule[i]
                    timestep_value = int(sigma * 1000)
                    latent_concat_curr = torch.cat([latents_curr, cond_latent_curr], dim=1)
                    timestep_input = torch.full((1, update_latent_num), timestep_value, device=latents_curr.device, dtype=torch.long)
                    if i == 0:
                        tmp_kwargs = {
                            "hidden_states": latent_concat_curr[:, :, -update_latent_num:, :, :],
                            "timestep": timestep_input,
                            "timestep_r": None,
                            "return_dict": False,
                            "mask_type": "i2v",
                            "action": action_curr[:, -update_latent_num:],
                            "viewmats": w2c_curr[:, -update_latent_num:, :, :],
                            "Ks": intrinsic_curr[:, -update_latent_num:, :, :],
                            "kv_cache": kv_cache["positive"],
                            "cache_vision": False,
                            "rope_temporal_size": len(selected_frame_indices) + update_latent_num,
                            "start_rope_start_idx": len(selected_frame_indices),
                        }
                    input_dict = dict(tmp_kwargs)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), torch.no_grad():
                        model_pred = pipeline.transformer(txt_branch=False, input_dict=input_dict)[0]
                    # SDE noise injection for candidate diversity
                    # (see grpo_eta / grpo_eta_cutoff / grpo_sde / grpo_sde_solver).
                    # SDE active only while i / sampling_steps < cutoff.
                    # When solver == "flow_grpo", noise scale is σ-adaptive and
                    # auto-shrinks near σ=0, so cutoff=1.0 is safe (no end-frame blur).
                    # When solver == "flux", noise scale is constant w.r.t. σ, so a
                    # cutoff < 1.0 is needed to keep late-step frames sharp.
                    sde_on = (
                        getattr(pipeline.training_args, "grpo_sde", False)
                        and (i / sampling_steps)
                        < getattr(pipeline.training_args, "grpo_eta_cutoff", 1.0)
                    )
                    solver = getattr(pipeline.training_args, "grpo_sde_solver", "flux")
                    mp = model_pred[:, :, -update_latent_num:, :, :].to(torch.float32)
                    lc = latents_curr[:, :, -update_latent_num:, :, :].to(torch.float32)
                    if sde_on and solver == "flow_grpo":
                        eta_val = float(getattr(pipeline.training_args, "grpo_eta", 0.0))
                        z, pred_original = flow_grpo_step(
                            mp, lc, eta=eta_val, sigmas=sigma_schedule, index=i,
                            prev_sample=None,
                        )
                    else:
                        active_eta = (
                            float(getattr(pipeline.training_args, "grpo_eta", 0.0))
                            if sde_on else 0.0
                        )
                        z, pred_original = flux_step(
                            mp, lc,
                            eta=active_eta,
                            sigmas=sigma_schedule,
                            index=i,
                            prev_sample=None,
                            grpo=False,
                            sde_solver=sde_on,
                        )
                    latents_curr[:, :, -update_latent_num:, :, :] = z
                latents_curr[:, :, -update_latent_num:, :, :] = pred_original
                tmp_kwargs["pred_latents"] = latents_curr[:, :, -chunk_latent_num:, :, :].detach().clone()
            with torch.no_grad():
                if latents_curr.shape[2] > chunk_latent_num:
                    latents_curr = torch.cat([latents_curr[:, :, :1, :, :], latents_curr[:, :, -(chunk_latent_num + 2):, :, :]], dim=2)
                if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor:
                    latents = latents_curr / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
                else:
                    latents = latents_curr / pipeline.vae.config.scaling_factor
                # See decode_fp32_resilient: empty_cache + one retry, because the
                # ODE loop above leaves the allocator fragmented and this decode
                # needs one large contiguous fp32 buffer.
                video_frames = decode_fp32_resilient(pipeline.vae, latents)
                video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
                video_frames = np.transpose(video_frames[0], (1, 2, 3, 0))
                video_frames = (video_frames * 255).numpy().astype(np.uint8)
                if context_video_frames is not None:
                    video_frames = np.concatenate([context_video_frames, video_frames[-16:]], axis=0)
                video_path = str(
                    get_generated_videos_base(pipeline.training_args)
                    / f"step_{training_batch.current_timestep}"
                )
                os.makedirs(video_path, exist_ok=True)
                # Collect frames for VLM pairwise comparison
                if candidate_video_frames is not None:
                    candidate_video_frames.append(video_frames.copy())

                for local_video_idx in range(video_num_in_each_gpu):
                    global_video_idx = rank_in_gpu_group * video_num_in_each_gpu + sample_idx * sampling_batch_size + local_video_idx
                    save_video_path = os.path.join(video_path, f"{pipeline.global_rank // pipeline.training_args.gpu_para}_{global_video_idx}.mp4")
                    save_video(video_frames, save_video_path)
                    absolute_path = os.path.abspath(save_video_path)

                    # WorldMirror / HPSv3 rewards — each gated independently
                    # and called via its own scorer (no shared backend).
                    action_reward = torch.tensor(0.0).to(latents_curr)
                    fine_action_reward = torch.tensor(0.0).to(latents_curr)
                    hpsv3_reward = torch.tensor(0.0).to(latents_curr)
                    hpsv3_quality_reward = torch.tensor(0.0).to(latents_curr)
                    hpsv3_quality_drift_reward = torch.tensor(0.0).to(latents_curr)

                    wm_action_acc = 0.0
                    wm_fine_action_acc = 0.0
                    if wm_enabled and worldmirror_reward is not None and worldmirror_reward._scorer is not None:
                        wm_out = worldmirror_reward._scorer.score(
                            video_path=absolute_path,
                            gt_action=action_curr[:, -update_latent_num:],
                            interval=1,
                            update_latent_num=chunk_latent_num,
                        )
                        wm_action_acc = float(wm_out["action_acc"])
                        wm_fine_action_acc = float(wm_out["fine_action_acc"])
                        action_reward = torch.tensor(wm_action_acc).to(latents_curr)
                        fine_action_reward = torch.tensor(wm_fine_action_acc).to(latents_curr)

                    hps_acc = 0.0
                    hps_quality_acc = 0.0
                    hps_quality_drift = 0.0
                    if hpsv3_enabled_train and hpsv3_computer is not None and hpsv3_computer._scorer is not None:
                        hps_out = hpsv3_computer._scorer.score(
                            video_path=absolute_path,
                            caption=training_batch.prompt,
                            interval=1,
                            update_latent_num=chunk_latent_num,
                            # Caption alignment costs as many HPSv3 forwards as
                            # the quality term; skip it unless it is weighted.
                            score_caption=pipeline.reward_weights.get("hpsv3", 0) > 0,
                        )
                        hps_acc = float(hps_out["hpsv3_acc"])
                        hps_quality_acc = float(hps_out["hpsv3_quality_acc"])
                        hps_quality_drift = float(hps_out["hpsv3_quality_drift_score"])
                        hpsv3_reward = torch.tensor(hps_acc).to(latents_curr)
                        hpsv3_quality_reward = torch.tensor(hps_quality_acc).to(latents_curr)
                        hpsv3_quality_drift_reward = torch.tensor(hps_quality_drift).to(latents_curr)

                    new_absolute_path = absolute_path  # default when no rename happened
                    if original_reward_enabled:
                        new_filename = (
                            f"{pipeline.global_rank // pipeline.training_args.gpu_para}"
                            f"_{global_video_idx}_chunk_{selected_chunk_id}"
                            f"_action_{round(wm_action_acc, 1)}"
                            f"_Faction_{round(wm_fine_action_acc, 1)}"
                            f"_hpsv3_{round(hps_acc, 1)}"
                            f"_quality_{round(hps_quality_acc, 1)}"
                            f"_drift_{round(hps_quality_drift, 1)}.mp4"
                        )
                        new_absolute_path = os.path.join(os.path.dirname(absolute_path), new_filename)
                        if os.path.exists(absolute_path):
                            os.rename(absolute_path, new_absolute_path)

                    # Absolute-value reward anchors (anti-reward-hacking).
                    # Use the video mp4 just written to disk (original path if no
                    # WorldMirror rename happened, else the renamed path).
                    scorer_path = (
                        new_absolute_path
                        if original_reward_enabled and os.path.exists(new_absolute_path)
                        else absolute_path
                    )
                    ate_reward = torch.tensor(0.0).to(latents_curr)
                    rpe_trans_reward = torch.tensor(0.0).to(latents_curr)
                    rpe_rot_reward = torch.tensor(0.0).to(latents_curr)
                    aesthetic_reward = torch.tensor(0.0).to(latents_curr)

                    # Keep raw metric magnitudes — advantages are computed
                    # per-output via z-score (scale-invariant) then weighted
                    # sum, so squashing to [0,1] via exp_decay / linear_scale
                    # only distorts the dynamic range non-linearly. For
                    # lower-is-better metrics (ate/rpe) we negate so that
                    # "better candidate → larger reward" holds everywhere.
                    if getattr(pipeline, "monst3r_for_train", False):
                        try:
                            m = pipeline.monst3r_evaluator.compute_ate_rpe(
                                video_path=scorer_path,
                                gt_w2c=training_batch.w2c[0, :latent_t],
                            )
                            ate_val = m["ate_rmse"]
                            rpe_t_val = m["rpe_trans_rmse"]
                            rpe_r_val = m["rpe_rot_median"]
                            # NaN → pick a "bad-ish" value so failed scorers
                            # don't masquerade as outlier-good candidates.
                            ate_reward = torch.tensor(
                                -(ate_val if math.isfinite(ate_val) else 1.0)
                            ).to(latents_curr)
                            rpe_trans_reward = torch.tensor(
                                -(rpe_t_val if math.isfinite(rpe_t_val) else 1.0)
                            ).to(latents_curr)
                            rpe_rot_reward = torch.tensor(
                                -(rpe_r_val if math.isfinite(rpe_r_val) else 10.0)
                            ).to(latents_curr)
                        except Exception as e:
                            logger.warning("monst3r reward failed for %s: %s", scorer_path, e)

                    if getattr(pipeline, "aesthetic_for_train", False):
                        try:
                            aes = pipeline.aesthetic_evaluator.score_video(scorer_path)
                            aesthetic_reward = torch.tensor(
                                aes if math.isfinite(aes) else 5.0
                            ).to(latents_curr)
                        except Exception as e:
                            logger.warning("aesthetic reward failed for %s: %s", scorer_path, e)

                    try:
                        gt_overlay_frames = render_action_overlay_video(video_frames, action_curr[0])
                        overlay_path = (absolute_path if not original_reward_enabled else new_absolute_path)
                        overlay_dir = os.path.join(os.path.dirname(overlay_path), "overlays")
                        os.makedirs(overlay_dir, exist_ok=True)
                        overlay_out = os.path.join(
                            overlay_dir,
                            os.path.basename(overlay_path).replace(".mp4", "_gt_overlay.mp4"),
                        )
                        save_video(gt_overlay_frames, overlay_out)
                    except Exception:
                        pass  # overlay is optional, don't break training
                    training_batch.sample_kwargs[f"sample_{global_video_idx}"] = {
                        "pred_latents": tmp_kwargs["pred_latents"][local_video_idx].unsqueeze(0).detach().clone(),
                        "action": tmp_kwargs["action"] if pipeline.action else None,
                        "viewmats": tmp_kwargs["viewmats"] if pipeline.action else None,
                        "Ks": tmp_kwargs["Ks"] if pipeline.action else None,
                        "rope_temporal_size": tmp_kwargs["rope_temporal_size"],
                        "start_rope_start_idx": tmp_kwargs["start_rope_start_idx"],
                        "cache_vision": tmp_kwargs["cache_vision"],
                        "return_dict": False,
                        "action_reward": action_reward,
                        "fine_action_reward": fine_action_reward,
                        "hpsv3_reward": hpsv3_reward,
                        "hpsv3_quality_reward": hpsv3_quality_reward,
                        "hpsv3_quality_drift_reward": hpsv3_quality_drift_reward,
                        "vlm_action_reward": torch.tensor(0.0).to(latents_curr),
                        "vlm_vq_reward": torch.tensor(0.0).to(latents_curr),
                        "ate_rmse_reward": ate_reward,
                        "rpe_trans_reward": rpe_trans_reward,
                        "rpe_rot_reward": rpe_rot_reward,
                        "aesthetic_reward": aesthetic_reward,
                        "selected_chunk_id": selected_chunk_id,
                    }
                    # Record raw per-candidate scores for end-of-rollout
                    # printout (mirrors VLM win_rates log style).
                    _rollout_rewards[sample_idx] = {
                        "act": wm_action_acc,
                        "Fact": wm_fine_action_acc,
                        "hps": hps_acc,
                        "hpsQ": hps_quality_acc,
                        "hpsD": hps_quality_drift,
                        "ate": float(ate_reward.item()),
                        "rpeT": float(rpe_trans_reward.item()),
                        "rpeR": float(rpe_rot_reward.item()),
                        "aes": float(aesthetic_reward.item()),
                    }
            torch.cuda.empty_cache()

        # Per-candidate reward summary (one line, same spirit as VLM
        # win_rates). Shows each enabled scorer's raw reward for the 8
        # candidates this rank produced, so regressions/reward-hacking
        # patterns are visible without tailing wandb.
        if _rollout_rewards:
            _any_wm = any(v["act"] or v["Fact"] for v in _rollout_rewards.values())
            _any_hps = any(v["hps"] or v["hpsQ"] or v["hpsD"] for v in _rollout_rewards.values())
            _any_anchor = any(
                v["ate"] or v["rpeT"] or v["rpeR"] or v["aes"]
                for v in _rollout_rewards.values()
            )
            _fmt = lambda d, keys: "{" + ", ".join(
                f"{i}: '" + "/".join(f"{k}={d[i][k]:.2f}" for k in keys) + "'"
                for i in sorted(d)
            ) + "}"
            if _any_wm:
                logger.info("WM scores: %s", _fmt(_rollout_rewards, ("act", "Fact")))
            if _any_hps:
                logger.info(
                    "HPSv3 scores: %s",
                    _fmt(_rollout_rewards, ("hps", "hpsQ", "hpsD")),
                )
            if _any_anchor:
                logger.info(
                    "Anchor scores: %s",
                    _fmt(_rollout_rewards, ("ate", "rpeT", "rpeR", "aes")),
                )

        # ── VLM pairwise comparison (after all candidates generated) ──
        if vlm_reward_enabled and candidate_video_frames:
            from fastvideo.rewards.computers.worldreward import CandidateInfo
            from fastvideo.utils.action_overlay import label_to_text

            candidates = [
                CandidateInfo(index=idx, video_frames=frames)
                for idx, frames in enumerate(candidate_video_frames)
            ]

            # Extract current chunk's action labels → text
            chunk_action_labels = action_curr[0, -update_latent_num:].cpu().long().tolist()
            chunk_action_texts = [label_to_text(int(lbl)) for lbl in chunk_action_labels]
            eval_chunk_id = selected_chunk_id - 1  # 0-based for build_eval_data

            # Two-stage prefiltering: WorldMirror scores → keep top+bottom K → VLM on K
            if vlm_prefilter and original_reward_enabled:
                prefilter_topk = getattr(pipeline.training_args, "vlm_prefilter_topk", 8)
                existing_rewards = []
                for idx in range(len(candidates)):
                    sk = f"sample_{rank_in_gpu_group * video_num_in_each_gpu + idx * sampling_batch_size}"
                    r = float(training_batch.sample_kwargs[sk].get("fine_action_reward",
                              training_batch.sample_kwargs[sk].get("action_reward", 0.0)))
                    existing_rewards.append(r)
                sorted_idx = sorted(range(len(existing_rewards)), key=lambda i: -existing_rewards[i])
                half_k = prefilter_topk // 2
                top_idx = sorted_idx[:half_k]
                bottom_idx = sorted_idx[-half_k:]
                sel_idx = list(dict.fromkeys(top_idx + bottom_idx))  # deduplicate, preserve order
                selected_candidates = [candidates[i] for i in sel_idx]
                logger.info(
                    "VLM prefilter: %d → %d candidates (WorldMirror top%d + bottom%d)",
                    len(candidates), len(selected_candidates), half_k, half_k,
                )
            elif pipeline.training_args.vlm_pair_mode == "topk" and original_reward_enabled:
                # Legacy topk mode: top+bottom by existing reward
                existing_rewards = []
                for idx in range(len(candidates)):
                    sk = f"sample_{rank_in_gpu_group * video_num_in_each_gpu + idx * sampling_batch_size}"
                    existing_rewards.append(float(training_batch.sample_kwargs[sk]["action_reward"]))
                sorted_idx = sorted(range(len(existing_rewards)), key=lambda i: existing_rewards[i])
                half_k = max(pipeline.training_args.bestofn // 2, 3)
                sel_idx = list(set(sorted_idx[:half_k] + sorted_idx[-half_k:]))
                selected_candidates = [candidates[i] for i in sel_idx]
            else:
                selected_candidates = candidates

            vlm_results = pipeline.reward_dispatcher.vlm._backend.compute_pairwise_rewards(
                candidates=selected_candidates,
                caption=training_batch.prompt,
                action_labels=chunk_action_labels,
                action_texts=chunk_action_texts,
                chunk_id=eval_chunk_id,
                chunk_size=chunk_latent_num,
            )

            selected_indices = {c.index for c in selected_candidates}
            for idx in range(len(candidate_video_frames)):
                global_idx = rank_in_gpu_group * video_num_in_each_gpu + idx * sampling_batch_size
                sk = f"sample_{global_idx}"
                if idx in selected_indices and idx in vlm_results:
                    training_batch.sample_kwargs[sk]["vlm_action_reward"] = torch.tensor(
                        vlm_results[idx].ac_win_rate
                    ).to(latents_curr)
                    training_batch.sample_kwargs[sk]["vlm_vq_reward"] = torch.tensor(
                        vlm_results[idx].vq_win_rate
                    ).to(latents_curr)
                else:
                    # Not selected by prefilter → neutral VLM reward
                    training_batch.sample_kwargs[sk]["vlm_action_reward"] = torch.tensor(0.5).to(latents_curr)
                    training_batch.sample_kwargs[sk]["vlm_vq_reward"] = torch.tensor(0.5).to(latents_curr)

            del candidate_video_frames

        pipeline.vae.to("cpu")
        training_batch.sigma_schedule = sigma_schedule
        training_batch.selected_chunk_id = selected_chunk_id
    return training_batch
