# SPDX-License-Identifier: Apache-2.0
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

# Gate inner tqdm bars (NFT Training Over Samples/Timesteps) on global
# rank 0. Otherwise each of 32 ranks emits the same progress bar and
# the log becomes unreadable when tailed as plain text.
_IS_RANK0 = int(os.environ.get("RANK", "0")) == 0
_TQDM_KW = dict(disable=not _IS_RANK0, mininterval=30.0, miniters=1)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastvideo.dataset import build_hy_camera_dataloader
from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.rewards import RewardDispatcher
import fastvideo.utils.envs as envs
from fastvideo.utils.logger import init_logger
from fastvideo.models.utils import generate_points_in_sphere
from fastvideo.pipelines import TrainingBatch
from fastvideo.utils.activation_checkpoint import apply_activation_checkpointing
from fastvideo.utils.forward_context import set_forward_context
from fastvideo.training.nft_train_pipeline_base import (
    NFTTrainPipelineBase,
    vsa_available,
)
from fastvideo.training.nft import (
    build_kv_cache_from_previous_chunks,
    create_sample_kv_cache,
    flux_step,
    get_task_mask,
    prepare_cond_latents,
    run_nft_eval,
    sample_model_ode,
    sample_reference_model,
)

logger = init_logger(__name__)


class NFTInTrainPipeline(NFTTrainPipelineBase):
    """WorldPlay training pipeline.

    The shared training/runtime implementation lives in
    `fastvideo.training.nft_train_pipeline_base`. This file intentionally stays
    thin so WorldPlay-specific and LingBot-specific training entries do not
    depend on each other.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points_local = generate_points_in_sphere(50000, 8.0)
        self.reward_weights = {}

    def _build_reward_weights(self, training_args: TrainingArgs) -> dict[str, float]:
        return {
            "action": float(training_args.action_reward_weight),
            "hpsv3": float(training_args.hpsv3_reward_weight),
            "hpsv3_quality": float(training_args.hpsv3_quality_reward_weight),
            "hpsv3_quality_drift": float(
                training_args.hpsv3_quality_drift_reward_weight
            ),
            "vlm_action": float(training_args.vlm_action_reward_weight),
            "vlm_vq": float(training_args.vlm_vq_reward_weight),
            "ate_rmse": float(getattr(training_args, "ate_rmse_reward_weight", 0.0)),
            "rpe_trans": float(getattr(training_args, "rpe_trans_reward_weight", 0.0)),
            "rpe_rot": float(getattr(training_args, "rpe_rot_reward_weight", 0.0)),
            "aesthetic": float(getattr(training_args, "aesthetic_reward_weight", 0.0)),
        }

    def _reward_enabled(self, reward_name: str) -> bool:
        return self.reward_weights.get(reward_name, 0.0) > 0.0

    def _initialize_domain_specific_components(
        self, training_args
    ) -> None:
        self.reward_weights = self._build_reward_weights(training_args)

        # Dispatcher builds all enabled reward instances (WorldMirror /
        # HPSv3 / VLM / MonST3R / Aesthetic) from the registry based on
        # weights + eval flags. Adding a new reward is a new computer
        # class + a registry entry — no edits here.
        self.reward_dispatcher = RewardDispatcher(training_args=training_args)
        self.reward_dispatcher.load_all(self.device)

        # Evaluator shortcuts that rollout.py / eval.py still read directly.
        self.monst3r_evaluator = None
        monst3r = self.reward_dispatcher.monst3r
        if monst3r is not None and monst3r._evaluator is not None:
            self.monst3r_evaluator = monst3r._evaluator
        self.monst3r_for_train = (
            self._reward_enabled("ate_rmse")
            or self._reward_enabled("rpe_trans")
            or self._reward_enabled("rpe_rot")
        )

        self.aesthetic_evaluator = None
        aesthetic = self.reward_dispatcher.aesthetic
        if aesthetic is not None and aesthetic._evaluator is not None:
            self.aesthetic_evaluator = aesthetic._evaluator
        self.aesthetic_for_train = self._reward_enabled("aesthetic")

        self.vlm_reward_enabled = (
            self._reward_enabled("vlm_action")
            or self._reward_enabled("vlm_vq")
        )

        # Park the eval-only scorers back on the host. load_all() puts every
        # reward that is enabled for training OR eval on the GPU, so an
        # eval-only one would otherwise stay resident through every rollout
        # step. Both re-load lazily when eval needs them.
        for _name, _evaluator, _needed in (
            ("monst3r", self.monst3r_evaluator, self.monst3r_for_train),
            ("aesthetic", self.aesthetic_evaluator, self.aesthetic_for_train),
        ):
            if _evaluator is None or _needed:
                continue
            try:
                _evaluator.unload()
                logger.info(
                    "%s has weight 0 for training: parked on CPU after load, "
                    "eval will page it back in on demand", _name,
                )
            except Exception as e:
                logger.warning("%s park-after-load failed: %s", _name, e)

        logger.info(
            "Rewards initialized: %s (enabled outputs: %s)",
            [r.NAME for r in self.reward_dispatcher.rewards],
            self.reward_dispatcher.enabled_outputs,
        )

        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type,
            )

    def _initialize_domain_specific_dataloaders(self, training_args) -> None:
        if not self.action:
            return
        self.train_dataset, self.train_dataloader = build_hy_camera_dataloader(
            json_path=training_args.json_path,
            causal=training_args.causal,
            window_frames=training_args.window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            cfg_rate=training_args.training_cfg_rate,
            i2v_rate=training_args.i2v_rate,
            random_pose_path=training_args.random_pose_path,
            neg_prompt_path=training_args.neg_prompt_path,
            neg_byt5_prompt_path=training_args.neg_byt5_prompt_path,
        )
        self.eval_dataset, self.eval_dataloader = build_hy_camera_dataloader(
            json_path=training_args.eval_json_path,
            causal=training_args.causal,
            window_frames=training_args.window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            cfg_rate=training_args.training_cfg_rate,
            i2v_rate=training_args.i2v_rate,
            random_pose_path=training_args.random_pose_path,
            neg_prompt_path=training_args.neg_prompt_path,
            neg_byt5_prompt_path=training_args.neg_byt5_prompt_path,
        )

    def get_task_mask(self, task_type, latent_target_length):
        return get_task_mask(task_type, latent_target_length)

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        return prepare_cond_latents(task_type, cond_latents, latents, multitask_mask)

    def _eval(self, step: int) -> None:
        return run_nft_eval(self, step)

    def _create_sample_kv_cache(self):
        return create_sample_kv_cache(self)

    def _build_kv_cache_from_previous_chunks(
        self,
        current_kv_cache,
        latents_curr,
        training_batch,
        generate_latent_num,
        update_latent_num,
        stabilization_level,
        negative=False,
    ):
        return build_kv_cache_from_previous_chunks(
            self,
            current_kv_cache,
            latents_curr,
            training_batch,
            generate_latent_num,
            update_latent_num,
            stabilization_level,
            negative=negative,
        )

    def flux_step(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        sigmas: torch.Tensor,
        index: int,
        prev_sample: torch.Tensor,
        grpo: bool,
        sde_solver: bool,
    ):
        return flux_step(
            model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver
        )

    @torch.no_grad()
    def _sample_model_ode(
        self,
        training_batch: TrainingBatch,
        selected_chunk_id,
        noise_latents,
        sampling_steps=None,
    ) -> TrainingBatch:
        return sample_model_ode(
            self, training_batch, selected_chunk_id, noise_latents, sampling_steps
        )

    @torch.no_grad()
    def _sample_reference_model(self, training_batch: TrainingBatch) -> TrainingBatch:
        return sample_reference_model(self, training_batch)

    def _prepare_grpo_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        sample_keys = [
            key
            for key in training_batch.sample_kwargs.keys()
            if key.startswith("sample_")
        ]
        world_group = get_world_group()

        # Delegate per-reward z-score → weighted sum → global z-score to
        # the dispatcher. It iterates its own enabled output list, so
        # adding a new reward does not require editing this file.
        overall_reward, reward_means = self.reward_dispatcher.compute_advantages(
            sample_kwargs=training_batch.sample_kwargs,
            sample_keys=sample_keys,
            gpu_group=self.gpu_group,
            world_group=world_group,
            std_type=self.training_args.std_type,
            device=self.device,
        )

        # Populate both the generic dict and the legacy per-field means
        # (wandb still reads some of them).
        training_batch.reward_means = reward_means
        training_batch.action_reward_mean = reward_means.get("action", 0.0)
        training_batch.fine_action_reward_mean = reward_means.get("fine_action", 0.0)
        training_batch.hpsv3_reward_mean = reward_means.get("hpsv3", 0.0)
        training_batch.hpsv3_quality_reward_mean = reward_means.get("hpsv3_quality", 0.0)
        training_batch.hpsv3_quality_drift_reward_mean = reward_means.get("hpsv3_quality_drift", 0.0)
        training_batch.vlm_action_reward_mean = reward_means.get("vlm_action", 0.0)
        training_batch.vlm_vq_reward_mean = reward_means.get("vlm_vq", 0.0)
        training_batch.ate_rmse_reward_mean = reward_means.get("ate_rmse", 0.0)
        training_batch.rpe_trans_reward_mean = reward_means.get("rpe_trans", 0.0)
        training_batch.rpe_rot_reward_mean = reward_means.get("rpe_rot", 0.0)
        training_batch.aesthetic_reward_mean = reward_means.get("aesthetic", 0.0)

        # Selection still uses the non-normalized weighted sum as the
        # "best-of-N" ordering: overall_reward[i] is the policy-relative
        # advantage accumulated across enabled rewards.
        #
        # Special-case: when action_reward_type == "fine_action" the old
        # code used fine_action advantages in place of action in the
        # overall_reward term. That's now automatically handled as long
        # as fine_action has its own weight; otherwise it's a no-op.
        sorted_indices = torch.argsort(overall_reward)
        half = max((self.training_args.bestofn // 2) // self.gpu_group.world_size, 1)
        top_indices = sorted_indices[-half:]
        bottom_indices = sorted_indices[:half]
        selected_indices = torch.cat([top_indices, bottom_indices]).unique()
        shuffled_order = torch.randperm(
            len(selected_indices), device=selected_indices.device
        )
        selected_indices = selected_indices[shuffled_order]
        training_batch.sample_kwargs["shuffled_sample_keys"] = [
            sample_keys[idx] for idx in selected_indices.tolist()
        ]
        return training_batch

    def _nft_forward_and_compute_loss(
        self, training_batch: TrainingBatch
    ) -> TrainingBatch:
        self.transformer.train()
        if vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            assert training_batch.attn_metadata is not None
        else:
            assert training_batch.attn_metadata is None

        with set_forward_context(
            current_timestep=training_batch.current_timestep,
            attn_metadata=training_batch.attn_metadata,
        ):
            sigma_schedule = torch.linspace(1, 0, self.training_args.sampling_steps + 1)
            shift = 5
            sigma_schedule = (shift * sigma_schedule) / (
                1 + (shift - 1) * sigma_schedule
            )
            training_num = int(
                self.training_args.sampling_steps
                * self.training_args.train_timestep_fraction
            )
            sigma_schedule = sigma_schedule[:-1]

            for sample_idx, sample_key in tqdm(
                list(enumerate(training_batch.sample_kwargs["shuffled_sample_keys"])),
                desc="NFT Training Over Samples",
                leave=False,
                **_TQDM_KW,
            ):
                train_sigma_schedule = random.sample(
                    list(sigma_schedule.cpu().numpy()), training_num
                )
                random.shuffle(train_sigma_schedule)

                for timestep_idx in tqdm(
                    range(len(train_sigma_schedule)),
                    desc="NFT Training Over Timesteps",
                    leave=False,
                    **_TQDM_KW,
                ):
                    adv_clip_max = self.training_args.adv_clip_max
                    sigma = train_sigma_schedule[timestep_idx]

                    x0 = training_batch.sample_kwargs[sample_key]["pred_latents"]
                    noise = torch.randn_like(x0)

                    device = training_batch.sample_kwargs[sample_key]["viewmats"].device
                    dtype = training_batch.sample_kwargs[sample_key]["viewmats"].dtype

                    update_latent_num = self.training_args.single_chunk_size
                    latent_num = x0.shape[2]
                    stabilization_level = 15
                    timestep_value = int(sigma * 1000)
                    timestep_input = torch.full(
                        (latent_num,),
                        stabilization_level - 1,
                        device=get_local_torch_device(),
                        dtype=torch.long,
                    )
                    timestep_input[-update_latent_num:] = timestep_value

                    noisy_latents = (1 - sigma) * x0 + sigma * noise
                    noisy_latents = noisy_latents.to(device).to(dtype)

                    start_idx = training_batch.sample_kwargs[sample_key][
                        "start_rope_start_idx"
                    ]
                    end_idx = training_batch.sample_kwargs[sample_key][
                        "rope_temporal_size"
                    ]

                    cond_latent_curr = training_batch.cond_latents[
                        :, :, start_idx:end_idx, :, :
                    ]

                    latents_for_model = torch.cat(
                        [noisy_latents, cond_latent_curr], dim=1
                    )

                    input_dict = {
                        "hidden_states": latents_for_model,
                        "timestep": timestep_input,
                        "timestep_r": None,
                        "return_dict": False,
                        "mask_type": "i2v",
                        "action": training_batch.sample_kwargs[sample_key]["action"]
                        if self.action
                        else None,
                        "viewmats": training_batch.sample_kwargs[sample_key][
                            "viewmats"
                        ][:, -update_latent_num:, :, :],
                        "Ks": training_batch.sample_kwargs[sample_key]["Ks"][
                            :, -update_latent_num:, :, :
                        ],
                        "kv_cache": training_batch.kv_cache["positive"],
                        "cache_vision": False,
                        "rope_temporal_size": training_batch.sample_kwargs[sample_key][
                            "rope_temporal_size"
                        ],
                        "start_rope_start_idx": training_batch.sample_kwargs[
                            sample_key
                        ]["start_rope_start_idx"],
                    }

                    with torch.no_grad():
                        with self.ema_generator.apply_policy_shadow_to_model(
                            self.transformer
                        ):
                            model_pred_old = self.transformer(
                                txt_branch=False,
                                input_dict=input_dict,
                            )[0]

                    model_pred = self.transformer(
                        txt_branch=False,
                        input_dict=input_dict,
                    )[0]

                    # Use pre-computed globally-normalized total_advantages from
                    # _prepare_grpo_inputs (weighted sum then all-group normalization).
                    total_advantages = training_batch.sample_kwargs[sample_key][
                        "total_advantages"
                    ]

                    total_advantages = torch.clamp(
                        total_advantages,
                        -adv_clip_max,
                        adv_clip_max,
                    )
                    normalized_advantages_clip = (
                        total_advantages / adv_clip_max
                    ) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)

                    positive_prediction = model_pred
                    negative_prediction = 2 * model_pred_old.detach() - model_pred

                    positive_x0 = (
                        noisy_latents[:, :, -update_latent_num:, :, :]
                        - sigma * positive_prediction[:, :, -update_latent_num:, :, :]
                    )
                    x0 = x0[:, :, -update_latent_num:, :, :]
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(positive_x0.double() - x0.double())
                            .mean(dim=tuple(range(1, positive_x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((positive_x0 - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, positive_x0.ndim))
                    )

                    negative_x0 = (
                        noisy_latents[:, :, -update_latent_num:, :, :]
                        - sigma * negative_prediction[:, :, -update_latent_num:, :, :]
                    )
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(negative_x0.double() - x0.double())
                            .mean(dim=tuple(range(1, negative_x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0 - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, negative_x0.ndim))
                    )

                    policy_loss = r * positive_loss + (1.0 - r) * negative_loss
                    final_loss = policy_loss / (
                        self.training_args.gradient_accumulation_steps
                    )

                    final_loss.backward()
                    avg_loss = final_loss.detach().clone()
                    world_group = get_world_group()
                    avg_loss = world_group.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                    training_batch.total_loss += avg_loss.item()

                if (
                    sample_idx + 1
                ) % self.training_args.gradient_accumulation_steps == 0:
                    training_batch = self._clip_grad_norm(training_batch)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    # Verify LoRA params are being updated (log lora_B norm; starts at 0)
                    if self.global_rank == 0 and not getattr(self, "_lora_norm_logged", False):
                        lora_b_norms = []
                        for name, layer in self.lora_layers.items():
                            b = layer.lora_B
                            if hasattr(b, "to_local"):
                                b = b.to_local()
                            lora_b_norms.append(b.norm().item())
                        if lora_b_norms:
                            mean_norm = sum(lora_b_norms) / len(lora_b_norms)
                            max_norm = max(lora_b_norms)
                            logger.info(
                                "LoRA lora_B norm check (first optimizer step): mean=%.6f max=%.6f over %d layers",
                                mean_norm, max_norm, len(lora_b_norms),
                            )
                            self._lora_norm_logged = True
                    self.ema_generator.update_ckpt_shadow(self.transformer)
                    self.ema_generator.update_policy_shadow(
                        self.transformer, training_batch.current_timestep
                    )

                    training_batch.samples_grad_norm[sample_key] = (
                        training_batch.grad_norm
                    )

        return training_batch


class NFTTrainPipeline(NFTInTrainPipeline):
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = NFTTrainPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    import sys

    from fastvideo.utils import FlexibleArgumentParser

    # Capture the original --config yaml path before the parser consumes it,
    # so we can archive the training config next to each checkpoint.
    _config_path = ""
    if "--config" in sys.argv:
        _ci = sys.argv.index("--config")
        if _ci + 1 < len(sys.argv):
            _config_path = sys.argv[_ci + 1]

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.config_path = _config_path
    args.dit_cpu_offload = False
    main(args)
