import glob
import json
import logging
import math
import os

import cv2
import numpy as np
import torch
from fastvideo.training.nft.rollout import decode_fp32_resilient
import torch.distributed as dist

from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.pipelines import TrainingBatch
from fastvideo.utils.video_utils import get_generated_videos_base, save_video
from fastvideo.training.nft.visualization import (
    get_task_mask,
    prepare_cond_latents,
    render_action_overlay_video,
)

logger = logging.getLogger(__name__)


def _load_video_frames(video_path: str) -> np.ndarray | None:
    """Load mp4 as numpy (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None


def run_nft_eval(pipeline, step: int):
    with torch.no_grad():
        action_acc = []
        fine_action_acc = []
        hpsv3_acc = []
        hpsv3_quality_acc = []
        hpsv3_drift_score = []
        vlm_ac_win_list = []
        vlm_vq_win_list = []
        ate_rmse_list = []
        rpe_trans_rmse_list = []
        rpe_rot_median_list = []
        aesthetic_list = []

        vlm_reward = pipeline.reward_dispatcher.vlm
        vlm_eval_enabled = (
            step > 0
            and getattr(pipeline, "vlm_reward_enabled", False)
            and vlm_reward is not None
            and getattr(vlm_reward, "_backend", None) is not None
        )
        monst3r_enabled = getattr(pipeline, "monst3r_evaluator", None) is not None
        aesthetic_enabled = (
            getattr(pipeline, "aesthetic_evaluator", None) is not None
        )
        worldmirror_reward = pipeline.reward_dispatcher.worldmirror
        hpsv3_reward = pipeline.reward_dispatcher.hpsv3
        wm_loaded = worldmirror_reward is not None and worldmirror_reward._scorer is not None
        hpsv3_loaded = hpsv3_reward is not None and hpsv3_reward._scorer is not None
        eval_wm_cfg = bool(
            getattr(pipeline.training_args, "eval_worldmirror", False)
        )
        action_enabled = eval_wm_cfg and wm_loaded
        hpsv3_enabled = hpsv3_loaded

        with pipeline.ema_generator.apply_ckpt_shadow_to_model(pipeline.transformer):
            pipeline.vae.eval()
            pipeline.vae.to(pipeline.device)
            pipeline.transformer.eval()
            eval_batch = TrainingBatch()
            sampling_steps = int(getattr(pipeline.training_args, "eval_sampling_steps", 20))
            latent_t = int(getattr(pipeline.training_args, "eval_window_frames", 32))
            chunk_latent_num = pipeline.training_args.single_chunk_size
            video_chunk_num = latent_t // chunk_latent_num

            for batch_idx, batch in enumerate(pipeline.eval_dataloader):
                latents = batch["latent"]
                prompt = batch["prompt"]
                w2c = batch["w2c"]
                action = batch["action"]
                _, latent_channels, _, latent_h, latent_w = latents.shape
                noise = torch.randn(
                    (1, latent_channels, latent_t, latent_h, latent_w),
                    device=get_local_torch_device(),
                    dtype=latents.dtype,
                )
                extra_kwargs = {
                    "byt5_text_states": batch["byt5_text_states"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    ),
                    "byt5_text_mask": batch["byt5_text_mask"].to(
                        get_local_torch_device(), dtype=torch.bfloat16
                    ),
                }
                multitask_mask = get_task_mask("i2v", batch["latent"].shape[2])
                cond_latents = prepare_cond_latents(
                    "i2v", batch["image_cond"], batch["latent"], multitask_mask
                )
                eval_batch.cond_latents = cond_latents.to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.latents_concat = torch.concat([batch["latent"], cond_latents], dim=1).to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.prompt_embed = batch["prompt_embed"].to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.prompt_mask = batch["prompt_mask"].to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.vision_states = batch["vision_states"].to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.extra_kwargs = extra_kwargs
                eval_batch.w2c = batch["w2c"].to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.intrinsic = batch["intrinsic"].to(get_local_torch_device(), dtype=torch.bfloat16)
                eval_batch.action = batch["action"].to(get_local_torch_device(), dtype=torch.bfloat16)

                latents = pipeline._sample_model_ode(eval_batch, video_chunk_num + 1, noise, sampling_steps=sampling_steps)
                if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor:
                    latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
                else:
                    latents = latents / pipeline.vae.config.scaling_factor

                # Shared with the rollout path: empty_cache + one retry against
                # allocator fragmentation left by _sample_model_ode. Eval has not
                # OOMed here (eval_window_frames is fixed and does not follow the
                # chunk curriculum) but the exposure is identical.
                video_frames = decode_fp32_resilient(pipeline.vae, latents)
                video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
                video_frames = np.transpose(video_frames[0], (1, 2, 3, 0))
                video_frames = (video_frames * 255).numpy().astype(np.uint8)

                sample_dir_name = (
                    f"sample_{pipeline.global_rank + (batch_idx * pipeline.world_size)}"
                )
                video_path = str(
                    get_generated_videos_base(pipeline.training_args)
                    / "000_eval"
                    / sample_dir_name
                )
                os.makedirs(video_path, exist_ok=True)

                # Shared step_0 baseline location. Keyed by the user-chosen
                # directory name rather than by run name, because the baseline
                # is a function of (base checkpoint, eval set) only. Empty
                # string (default) => behave exactly as before, i.e. the
                # baseline lives inside this run's own output.
                #
                # NOTE the cache is only valid while the base checkpoint, the
                # eval json and eval_window_frames stay the same. Nothing
                # enforces that, so name the directory after those inputs.
                _bdir = str(getattr(pipeline.training_args, "eval_baseline_dir", "") or "")
                if _bdir:
                    baseline_path = os.path.join(_bdir, sample_dir_name)
                    os.makedirs(baseline_path, exist_ok=True)
                else:
                    baseline_path = video_path

                # Write a one-time meta.json per eval sample (sample-level info
                # is fixed across steps, so skip if it already exists). Holds
                # the input image path, caption, GT action sequence, and the
                # continuous camera trajectory (w2c + intrinsics) — the latter
                # is what the model actually conditions on via PRoPE, so it is
                # required to reproduce this sample's generation from a ckpt.
                meta_path = os.path.join(video_path, "meta.json")
                if not os.path.exists(meta_path):
                    try:
                        from fastvideo.utils.action_overlay import label_to_text

                        sample_id = pipeline.global_rank + batch_idx * pipeline.world_size
                        gt_action_labels = [int(x) for x in action[0].cpu().long().tolist()]
                        gt_action_texts = [label_to_text(x) for x in gt_action_labels]
                        img_path = batch.get("image_path", [""])
                        img_path = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
                        lat_path = batch.get("latent_path", [""])
                        lat_path = lat_path[0] if isinstance(lat_path, (list, tuple)) else lat_path
                        caption_str = prompt[0] if isinstance(prompt, (list, tuple)) else prompt

                        # Save continuous camera trajectory (the real conditioning
                        # signal) as .npy so generation is reproducible even if the
                        # source random_pose file later changes.
                        w2c_np = w2c[0].cpu().float().numpy()
                        intrinsic_np = batch["intrinsic"][0].cpu().float().numpy()
                        np.save(os.path.join(video_path, "w2c.npy"), w2c_np)
                        np.save(os.path.join(video_path, "intrinsic.npy"), intrinsic_np)

                        meta = {
                            "sample_id": sample_id,
                            "input_image": img_path,
                            "latent_path": lat_path,
                            "caption": caption_str,
                            "action_labels": gt_action_labels,
                            "action_texts": gt_action_texts,
                            "eval_window_frames": latent_t,
                            "chunk_latent_num": chunk_latent_num,
                            "video_chunk_num": video_chunk_num,
                            # continuous trajectory for reproduction (model conditions on these)
                            "w2c_npy": "w2c.npy",
                            "w2c_shape": list(w2c_np.shape),
                            "intrinsic_npy": "intrinsic.npy",
                            "intrinsic_shape": list(intrinsic_np.shape),
                        }
                        tmp_meta = meta_path + ".tmp"
                        with open(tmp_meta, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_meta, meta_path)
                    except Exception as e:
                        logger.warning("failed to write meta.json for sample dir %s: %s", video_path, e)

                save_video_path = os.path.join(video_path, f"step_{step}.mp4")
                save_video(video_frames, save_video_path)
                absolute_path = os.path.abspath(save_video_path)

                score: dict | None = None
                if action_enabled or hpsv3_enabled:
                    score = {}
                    if action_enabled:
                        try:
                            wm_out = worldmirror_reward._scorer.score_eval(
                                video_path=absolute_path,
                                gt_action=action,
                                interval=1,
                                latent_num=latent_t,
                            )
                            score["action_acc"] = wm_out["action_acc"]
                            # fine_action_acc is the output that actually
                            # carries the training weight when
                            # action_reward_type == "fine_action"; report both.
                            if "fine_action_acc" in wm_out:
                                score["fine_action_acc"] = wm_out["fine_action_acc"]
                        except Exception as e:
                            logger.warning(
                                "WorldMirror score_eval failed for sample %d: %s",
                                pipeline.global_rank + batch_idx * pipeline.world_size,
                                e,
                            )
                    if hpsv3_enabled:
                        try:
                            caption = prompt[0] if isinstance(prompt, (list, tuple)) else prompt
                            hps_out = hpsv3_reward._scorer.score_eval(
                                video_path=absolute_path,
                                caption=caption,
                                interval=1,
                                latent_num=latent_t,
                            )
                            score["hps_acc"] = hps_out["hps_acc"]
                            score["hps_quality_acc"] = hps_out["hps_quality_acc"]
                            score["hps_drift_score"] = hps_out["hps_drift_score"]
                        except Exception as e:
                            logger.warning(
                                "HPSv3 score_eval failed for sample %d: %s",
                                pipeline.global_rank + batch_idx * pipeline.world_size,
                                e,
                            )
                    if not score:
                        score = None

                monst3r_suffix = ""
                if monst3r_enabled:
                    metrics = pipeline.monst3r_evaluator.compute_ate_rpe(
                        video_path=absolute_path,
                        gt_w2c=w2c[0],
                    )
                    # Skip NaN entries so a single failed sample doesn't poison
                    # the cross-rank all_reduce mean. monst3r returns NaN on
                    # OpenCV PnP failure (see monst3r_trajectory.py:352). When
                    # per-rank sample count is small (e.g. 1 sample per rank at
                    # 96 GPU / 128 eval samples), a rank whose only sample fails
                    # yields nanmean([NaN])=NaN, which then contaminates the
                    # AVG across all ranks.
                    if not math.isnan(metrics["ate_rmse"]):
                        ate_rmse_list.append(metrics["ate_rmse"])
                        rpe_trans_rmse_list.append(metrics["rpe_trans_rmse"])
                        rpe_rot_median_list.append(metrics["rpe_rot_median"])
                    monst3r_suffix = (
                        f"_ate_[{metrics['ate_rmse']:.3f}]"
                        f"_rpetR_[{metrics['rpe_trans_rmse']:.3f}]"
                        f"_rperM_[{metrics['rpe_rot_median']:.2f}]"
                    )

                aesthetic_suffix = ""
                if aesthetic_enabled:
                    aes_score = pipeline.aesthetic_evaluator.score_video(
                        absolute_path
                    )
                    aesthetic_list.append(aes_score)
                    aesthetic_suffix = f"_aes_[{aes_score:.2f}]"

                name_parts = [f"step_{step}"]
                if action_enabled and score is not None and "action_acc" in score:
                    name_parts.append(
                        f"_action_[{torch.tensor(score['action_acc']).mean().item():.3f}]"
                    )
                    if "fine_action_acc" in score:
                        name_parts.append(
                            f"_fine_[{torch.tensor(score['fine_action_acc']).mean().item():.3f}]"
                        )
                if hpsv3_enabled and score is not None and "hps_acc" in score:
                    name_parts.append(
                        f"_hpsv3_[{torch.tensor(score['hps_acc']).mean().item():.3f}]"
                        f"_quality_[{torch.tensor(score['hps_quality_acc']).mean().item():.3f}]"
                        f"_drift_[{score['hps_drift_score']:.3f}]"
                    )
                if monst3r_suffix:
                    name_parts.append(monst3r_suffix)
                if aesthetic_suffix:
                    name_parts.append(aesthetic_suffix)
                name_parts.append(".mp4")
                new_filename = "".join(name_parts)
                new_absolute_path = os.path.join(os.path.dirname(absolute_path), new_filename)
                if os.path.exists(absolute_path):
                    os.rename(absolute_path, new_absolute_path)

                # Populate the shared baseline on the step-0 pass so later runs
                # can reuse it. Copy (not move/symlink): the run keeps its own
                # copy for inspection, and a symlink would break if this run's
                # output dir is ever deleted.
                if (
                    step == 0
                    and baseline_path != video_path
                    and os.path.exists(new_absolute_path)
                ):
                    try:
                        import shutil
                        dst = os.path.join(baseline_path, os.path.basename(new_absolute_path))
                        if not os.path.exists(dst):
                            shutil.copy2(new_absolute_path, dst)
                    except Exception as e:
                        logger.warning(
                            "failed to seed eval baseline %s: %s", baseline_path, e
                        )
                try:
                    overlay_frames = render_action_overlay_video(video_frames, action[0])
                    overlay_dir = os.path.join(os.path.dirname(new_absolute_path), "overlays")
                    os.makedirs(overlay_dir, exist_ok=True)
                    overlay_out = os.path.join(
                        overlay_dir,
                        os.path.basename(new_absolute_path).replace(".mp4", "_overlay.mp4"),
                    )
                    save_video(overlay_frames, overlay_out)
                except Exception:
                    pass  # overlay is optional, don't break eval
                if score is not None:
                    if "action_acc" in score:
                        action_acc += score["action_acc"]
                    if "fine_action_acc" in score:
                        fine_action_acc += score["fine_action_acc"]
                    if "hps_acc" in score:
                        hpsv3_acc += score["hps_acc"]
                        hpsv3_quality_acc += score["hps_quality_acc"]
                        # scalar per sample now, not a per-chunk list
                        hpsv3_drift_score.append(score["hps_drift_score"])

                # VLM eval: compare current model vs step_0 baseline.
                # Sample 3 chunks (early / mid / late) and average the win
                # rates so a single noisy chunk doesn't dominate the eval
                # signal. Late chunk is kept because autoregressive drift
                # is most visible there; early/mid add coverage for the
                # parts of the rollout where the AC errors compound less.
                if vlm_eval_enabled:
                    try:
                        # The step_0 clips are the fixed reference every later
                        # eval is compared against, and they depend only on
                        # (base checkpoint, eval set) — not on which run is
                        # executing. When eval_baseline_dir is set we read them
                        # from there, so a new run can reuse an existing
                        # baseline and skip regenerating it via
                        # skip_initial_eval / WC_SKIP_INITIAL_EVAL=1.
                        #
                        # This also makes the comparison sounder: eval rollout
                        # noise is NOT seeded (torch.randn above), so a
                        # regenerated step_0 differs slightly every time. A
                        # pinned baseline keeps vlm_*_win_rate comparable both
                        # across steps and across runs.
                        step0_vids = [
                            v for v in glob.glob(os.path.join(baseline_path, "step_0_*.mp4"))
                            if "overlay" not in v
                        ]
                        if not step0_vids and baseline_path != video_path:
                            # Baseline dir configured but this sample is not in
                            # it yet — fall back to the clips this run wrote.
                            step0_vids = [
                                v for v in glob.glob(os.path.join(video_path, "step_0_*.mp4"))
                                if "overlay" not in v
                            ]
                        if step0_vids:
                            step0_frames = _load_video_frames(step0_vids[0])
                            if step0_frames is not None:
                                from fastvideo.rewards.computers.worldreward import CandidateInfo
                                from fastvideo.utils.action_overlay import label_to_text

                                # Pick 3 chunk indices spread across the
                                # rollout. Dedup and clamp so very short
                                # rollouts (video_chunk_num < 3) still
                                # work without hitting duplicate chunks.
                                if video_chunk_num <= 0:
                                    chunk_ids_to_eval = []
                                elif video_chunk_num == 1:
                                    chunk_ids_to_eval = [0]
                                elif video_chunk_num == 2:
                                    chunk_ids_to_eval = [0, 1]
                                else:
                                    chunk_ids_to_eval = sorted({
                                        max(0, video_chunk_num // 4),
                                        max(0, video_chunk_num // 2),
                                        video_chunk_num - 1,
                                    })

                                caption = prompt[0] if isinstance(prompt, (list, tuple)) else prompt
                                ac_per_chunk: list[float] = []
                                vq_per_chunk: list[float] = []
                                for eval_chunk_id in chunk_ids_to_eval:
                                    chunk_action = (
                                        eval_batch.action[0,
                                            eval_chunk_id * chunk_latent_num:
                                            (eval_chunk_id + 1) * chunk_latent_num
                                        ].cpu().long().tolist()
                                    )
                                    chunk_action_texts = [label_to_text(int(lbl)) for lbl in chunk_action]
                                    candidates = [
                                        CandidateInfo(index=0, video_frames=video_frames),   # current
                                        CandidateInfo(index=1, video_frames=step0_frames),   # step_0
                                    ]
                                    results = vlm_reward._backend.compute_pairwise_rewards(
                                        candidates=candidates,
                                        caption=caption,
                                        action_labels=chunk_action,
                                        action_texts=chunk_action_texts,
                                        chunk_id=eval_chunk_id,
                                        chunk_size=chunk_latent_num,
                                    )
                                    # index 0 = current model; win_rate > 0.5 means current beats step_0
                                    ac_per_chunk.append(float(results[0].ac_win_rate))
                                    vq_per_chunk.append(float(results[0].vq_win_rate))
                                    logger.info(
                                        "VLM eval sample %d step %d chunk %d: AC_wr=%.3f VQ_wr=%.3f",
                                        pipeline.global_rank + batch_idx * pipeline.world_size,
                                        step, eval_chunk_id,
                                        results[0].ac_win_rate,
                                        results[0].vq_win_rate,
                                    )

                                if ac_per_chunk:
                                    sample_ac = float(np.mean(ac_per_chunk))
                                    sample_vq = float(np.mean(vq_per_chunk))
                                    vlm_ac_win_list.append(sample_ac)
                                    vlm_vq_win_list.append(sample_vq)
                                    logger.info(
                                        "VLM eval sample %d step %d overall (chunks=%s): AC_wr=%.3f VQ_wr=%.3f",
                                        pipeline.global_rank + batch_idx * pipeline.world_size,
                                        step, chunk_ids_to_eval, sample_ac, sample_vq,
                                    )
                    except Exception as e:
                        logger.warning("VLM eval comparison failed for sample %d: %s",
                                       pipeline.global_rank + batch_idx * pipeline.world_size, e)

            pipeline.vae.to("cpu")
            world_group = get_world_group()

            # Only populate result keys for scorers that actually ran.
            # If an eval scorer is disabled (e.g. eval_worldmirror=false,
            # hpsv3_reward_weight=0), don't clutter the result dict or
            # wandb with zero/NaN placeholders.
            result: dict = {}

            if action_enabled:
                all_action_acc = world_group.all_gather(torch.tensor(action_acc).to(get_local_torch_device()).unsqueeze(0), dim=0)
                ave_action_acc = world_group.all_reduce(torch.tensor(action_acc).mean().to(get_local_torch_device()), op=dist.ReduceOp.AVG)
                result["ave_action_acc"] = ave_action_acc.cpu().item()
                result["all_action_acc"] = all_action_acc.cpu().tolist()

                # Unconditional, exactly like action_acc above: these two lists
                # are filled from the same score_eval() dict on every rank, so
                # gating this on `if fine_action_acc` would risk a collective
                # mismatch if one rank ended up with an empty list.
                all_fine_action_acc = world_group.all_gather(torch.tensor(fine_action_acc).to(get_local_torch_device()).unsqueeze(0), dim=0)
                ave_fine_action_acc = world_group.all_reduce(torch.tensor(fine_action_acc).mean().to(get_local_torch_device()), op=dist.ReduceOp.AVG)
                result["ave_fine_action_acc"] = ave_fine_action_acc.cpu().item()
                result["all_fine_action_acc"] = all_fine_action_acc.cpu().tolist()

            if hpsv3_enabled:
                all_hpsv3_acc = world_group.all_gather(torch.tensor(hpsv3_acc).to(get_local_torch_device()).unsqueeze(0), dim=0)
                all_hpsv3_quality_acc = world_group.all_gather(torch.tensor(hpsv3_quality_acc).to(get_local_torch_device()).unsqueeze(0), dim=0)
                all_hpsv3_drift_score = world_group.all_gather(torch.tensor(hpsv3_drift_score).to(get_local_torch_device()).unsqueeze(0), dim=0)
                ave_hpsv3_acc = world_group.all_reduce(torch.tensor(hpsv3_acc).mean().to(get_local_torch_device()), op=dist.ReduceOp.AVG)
                ave_hpsv3_quality_acc = world_group.all_reduce(torch.tensor(hpsv3_quality_acc).mean().to(get_local_torch_device()), op=dist.ReduceOp.AVG)
                ave_hpsv3_drift_score = world_group.all_reduce(torch.tensor(hpsv3_drift_score).mean().to(get_local_torch_device()), op=dist.ReduceOp.AVG)
                result["ave_hpsv3_acc"] = ave_hpsv3_acc.cpu().item()
                result["ave_hpsv3_quality_acc"] = ave_hpsv3_quality_acc.cpu().item()
                result["ave_hpsv3_drift_score"] = ave_hpsv3_drift_score.cpu().item()
                result["all_hpsv3_acc"] = all_hpsv3_acc.cpu().tolist()
                result["all_hpsv3_quality_acc"] = all_hpsv3_quality_acc.cpu().tolist()
                result["all_hpsv3_drift_score"] = all_hpsv3_drift_score.cpu().tolist()

            if vlm_ac_win_list:
                ave_vlm_ac_win = world_group.all_reduce(
                    torch.tensor(float(np.mean(vlm_ac_win_list))).to(get_local_torch_device()),
                    op=dist.ReduceOp.AVG,
                )
                ave_vlm_vq_win = world_group.all_reduce(
                    torch.tensor(float(np.mean(vlm_vq_win_list))).to(get_local_torch_device()),
                    op=dist.ReduceOp.AVG,
                )
                result["ave_vlm_ac_win_rate"] = ave_vlm_ac_win.cpu().item()
                result["ave_vlm_vq_win_rate"] = ave_vlm_vq_win.cpu().item()

            if monst3r_enabled and ate_rmse_list:
                ave_ate_rmse = world_group.all_reduce(
                    torch.tensor(float(np.nanmean(ate_rmse_list))).to(get_local_torch_device()),
                    op=dist.ReduceOp.AVG,
                )
                ave_rpe_trans_rmse = world_group.all_reduce(
                    torch.tensor(float(np.nanmean(rpe_trans_rmse_list))).to(get_local_torch_device()),
                    op=dist.ReduceOp.AVG,
                )
                ave_rpe_rot_median = world_group.all_reduce(
                    torch.tensor(float(np.nanmean(rpe_rot_median_list))).to(get_local_torch_device()),
                    op=dist.ReduceOp.AVG,
                )
                result["ave_ate_rmse"] = ave_ate_rmse.cpu().item()
                result["ave_rpe_trans_rmse"] = ave_rpe_trans_rmse.cpu().item()
                result["ave_rpe_rot_median"] = ave_rpe_rot_median.cpu().item()

            if aesthetic_enabled and aesthetic_list:
                ave_aesthetic = world_group.all_reduce(
                    torch.tensor(float(np.nanmean(aesthetic_list))).to(get_local_torch_device()),
                    op=dist.ReduceOp.AVG,
                )
                result["ave_aesthetic"] = ave_aesthetic.cpu().item()

            # Free eval-only models so the next reward/rollout phase has
            # full GPU headroom. vLLM wake_up at VLM_MEM_UTIL=0.30 needs
            # every GB it can get on this colocated setup.
            if monst3r_enabled:
                try:
                    pipeline.monst3r_evaluator.unload()
                except Exception as e:
                    logger.warning("monst3r unload failed: %s", e)
            if aesthetic_enabled:
                try:
                    pipeline.aesthetic_evaluator.unload()
                except Exception as e:
                    logger.warning("aesthetic unload failed: %s", e)

            return result
