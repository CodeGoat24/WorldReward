# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    get_local_torch_device,
    get_sp_group,
    get_world_group,
    get_gpu_group,
)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.utils.logger import init_logger
from fastvideo.pipelines import (
    ComposedPipelineBase,
    LoRAPipeline,
    TrainingBatch,
)
from fastvideo.utils.training_utils import (
    EMA_FSDP_schedule,
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
    load_checkpoint,
    save_checkpoint,
)
from fastvideo.utils import is_vsa_available, set_random_seed
from fastvideo.utils.muon import get_muon_optimizer
from fastvideo.models.hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)

# Rewards derived from pairwise VLM judgements. Their group mean is fixed by
# the round-robin structure (see the console reward summary in train()), so it
# must not be read as a quality signal.
_PAIRWISE_REWARD_KEYS = frozenset({"vlm_action", "vlm_vq"})


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp


def _get_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NFTTrainPipelineBase(LoRAPipeline, ABC):
    """Shared base class for NFT-style training pipelines.

    Concrete training entries should inherit from this class and keep
    model-specific rollout or evaluation logic in their own files.
    """

    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0
    wandb_enabled: bool = False

    def _hpsv3_enabled(self) -> bool:
        ta = self.training_args
        return any(
            float(w) > 0.0
            for w in (
                ta.hpsv3_reward_weight,
                ta.hpsv3_quality_reward_weight,
                ta.hpsv3_quality_drift_reward_weight,
            )
        )

    def __init__(
        self,
        model_path: str,
        fastvideo_args: TrainingArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> None:
        self.lora_training = fastvideo_args.lora_training
        if self.lora_training and fastvideo_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")

        set_random_seed(fastvideo_args.seed)  # for lora param init
        super().__init__(
            model_path, fastvideo_args, required_config_modules, loaded_modules
        )  # type: ignore

        self.sample_kv_caches = {}  # KV caches keyed by sample/chunk identifiers.

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline"
        )

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_t2v

    def _initialize_domain_specific_components(
        self, training_args: TrainingArgs
    ) -> None:
        pass

    def _initialize_domain_specific_dataloaders(
        self, training_args: TrainingArgs
    ) -> None:
        pass

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.gpu_group = get_gpu_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        # self.vae = self.get_module("vae")

        # Load the latent decoder used by the training pipeline.
        vae_path = training_args.vae_path
        if not vae_path:
            raise ValueError("vae_path must be provided in training_args")

        self.vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
            vae_path, torch_dtype=torch.float32
        ).to("cpu")
        self.vae = self.vae.to(torch.float32)

        self.seed = training_args.seed
        self.set_schemas()
        self.action = training_args.action

        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()

        self.set_trainable()
        params_to_optimize = self.transformer.parameters()
        params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

        self.vae.requires_grad = False


        self.optimizer = get_muon_optimizer(
            model=self.transformer,
            lr=training_args.learning_rate,  # Learning rate
            weight_decay=training_args.weight_decay,  # Weight decay
            adamw_betas=(0.9, 0.999),  # AdamW betas for 1D parameters
            adamw_eps=1e-8,  # AdamW epsilon
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        self._initialize_domain_specific_components(training_args)
        self._initialize_domain_specific_dataloaders(training_args)

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader)
            / training_args.gradient_accumulation_steps
            * training_args.sp_size
            / training_args.train_sp_batch_size
        )
        self.num_train_epochs = math.ceil(
            training_args.max_train_steps / self.num_update_steps_per_epoch
        )

        self.current_epoch = 0

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "fastvideo"
            wandb_config = dataclasses.asdict(training_args)
            wandb_key = training_args.wandb_key or os.environ.get("WANDB_API_KEY", "")
            if wandb_key:
                wandb.login(key=wandb_key)
                wandb.init(
                    config=wandb_config,
                    name=training_args.output_dir.split("/")[-1],
                    entity=training_args.wandb_entity,
                    project=project,
                )
                self.wandb_enabled = True
            else:
                logger.info(
                    "WANDB_API_KEY/wandb_key not set; skipping Weights & Biases logging."
                )

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError("Training pipelines must implement this method")

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        self.vae.eval()
        self.transformer.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0
        training_batch.samples_loss = {}
        training_batch.samples_grad_norm = {}
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        extra_kwargs = {
            "byt5_text_states": batch["byt5_text_states"].to(
                get_local_torch_device(), dtype=torch.bfloat16
            ),
            "byt5_text_mask": batch["byt5_text_mask"].to(
                get_local_torch_device(), dtype=torch.bfloat16
            ),
        }
        multitask_mask = self.get_task_mask("i2v", batch["latent"].shape[2])
        cond_latents = self._prepare_cond_latents(
            "i2v", batch["image_cond"], batch["latent"], multitask_mask
        )

        training_batch.latents = batch["latent"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.cond_latents = cond_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.latents_concat = torch.concat(
            [batch["latent"], cond_latents], dim=1
        ).to(get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.prompt = batch["prompt"]
        training_batch.prompt_embed = batch["prompt_embed"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.prompt_mask = batch["prompt_mask"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.vision_states = batch["vision_states"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.extra_kwargs = extra_kwargs
        training_batch.w2c = batch["w2c"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.intrinsic = batch["intrinsic"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.action = batch["action"].to(
            get_local_torch_device(), dtype=torch.bfloat16
        )

        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): move this behind a transformer API so pipeline code does
        # not need to know parameter ownership details.
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            model_parts = [self.transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float("nan") or grad_norm is not float("inf")
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def _clear_cache(self, training_batch: TrainingBatch) -> TrainingBatch:
        del training_batch
        torch.cuda.empty_cache()

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        training_batch = self._get_next_batch(training_batch)

        training_batch = self._sample_reference_model(training_batch)

        training_batch = self._prepare_grpo_inputs(training_batch)

        training_batch = self._nft_forward_and_compute_loss(training_batch)

        self._clear_cache(training_batch)

        training_batch.total_loss = training_batch.total_loss
        training_batch.grad_norm = training_batch.grad_norm
        return training_batch

    def _resolve_resume_path(self) -> str:
        """Turn resume_from_checkpoint into a concrete checkpoint dir.

        Accepts either an explicit path or the literal "latest". "latest" picks
        the highest-numbered checkpoint under output_dir that still has a
        distributed_checkpoint/ (only the newest one does — see
        remove_previous_state). Sorting is numeric, so checkpoint-100 beats
        checkpoint-90.

        Returns "" to mean "nothing to resume, start fresh". That only happens
        for "latest" on a run that has not checkpointed yet, which is
        unambiguous. If checkpoints DO exist but none is loadable we raise
        instead, because that means progress exists and silently discarding it
        would also rewind the curriculum.
        """
        raw = self.training_args.resume_from_checkpoint.rstrip("/")
        if raw != "latest":
            return raw

        out_dir = self.training_args.output_dir
        if not os.path.isdir(out_dir):
            return ""

        found, resumable = [], []
        for name in os.listdir(out_dir):
            if not name.startswith("checkpoint-"):
                continue
            try:
                step = int(name.split("-", 1)[1])
            except ValueError:
                continue
            found.append(step)
            if os.path.isdir(
                os.path.join(out_dir, name, "distributed_checkpoint")
            ):
                resumable.append((step, os.path.join(out_dir, name)))

        if not found:
            return ""
        if not resumable:
            raise FileNotFoundError(
                f'resume_from_checkpoint="latest": {out_dir} has checkpoints '
                f"{sorted(found)} but none keeps a distributed_checkpoint/, so "
                "none can be resumed. Refusing to discard that progress and "
                "rewind the curriculum silently."
            )
        resumable.sort(key=lambda x: x[0])
        return resumable[-1][1]

    def _resume_from_checkpoint(self) -> None:
        path = self._resolve_resume_path()
        if not path:
            logger.info(
                'resume_from_checkpoint="latest" but %s holds no checkpoint '
                "yet — starting from step 0",
                self.training_args.output_dir,
            )
            self.init_steps = 0
            return

        logger.info("Loading checkpoint from %s", path)

        # Muon allocates momentum_buffer / moment1 / moment2 lazily on its first
        # step(), but we load BEFORE any step, so the saved buffers would have
        # no destination tensors. Pre-create them.
        if hasattr(self.optimizer, "materialize_state"):
            self.optimizer.materialize_state()

        resumed_step = load_checkpoint(
            self.transformer,
            self.global_rank,
            path,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.noise_random_generator,
            self.ema_generator,
        )
        if resumed_step <= 0:
            # Never fall through to a from-scratch run: init_steps=0 would also
            # rewind the progressive_min2max curriculum to chunk 1, silently
            # invalidating the run instead of failing it.
            raise RuntimeError(
                f"load_checkpoint({path}) returned step {resumed_step}; "
                "refusing to continue as a fresh run."
            )
        self.init_steps = resumed_step
        logger.info("Successfully resumed from step %s", resumed_step)

    def _eval_baseline_is_complete(self) -> bool:
        """True iff eval_baseline_dir already holds a step_0 clip for every eval
        sample this run would evaluate, agreed across ALL ranks.

        Lets `eval_baseline_dir` alone imply skip-the-step-0-eval, so the two
        settings cannot drift out of sync.

        The cross-rank agreement is not optional. _eval() runs all_gather /
        all_reduce over world_group, so if one rank skipped and another did not
        the job would deadlock instead of failing. Each rank checks only the
        sample ids it will actually be handed
        (global_rank + batch_idx * world_size, matching eval.py), then the
        decision is min-reduced so a single incomplete rank makes everyone run
        the eval. A partially populated baseline is therefore never used.
        """
        import glob

        import torch.distributed as dist

        baseline_dir = str(getattr(self.training_args, "eval_baseline_dir", "") or "")
        if not baseline_dir:
            return False

        # eval_dataloader is built by the subclass, so a
        # pipeline without one simply has no baseline to reuse.
        loader = getattr(self, "eval_dataloader", None)
        if loader is None:
            return False

        local_ok = True
        try:
            num_batches = len(loader)
        except TypeError:
            # Iterable-style dataloader with no __len__: cannot enumerate the
            # sample ids up front, so do not try to infer anything.
            return False

        missing = None
        for batch_idx in range(num_batches):
            sample_id = self.global_rank + batch_idx * self.world_size
            clips = [
                v for v in glob.glob(
                    os.path.join(baseline_dir, f"sample_{sample_id}", "step_0_*.mp4")
                )
                if "overlay" not in v
            ]
            if not clips:
                local_ok = False
                missing = sample_id
                break

        flag = torch.tensor(
            [1 if local_ok else 0], device=get_local_torch_device(), dtype=torch.int32
        )
        get_world_group().all_reduce(flag, op=dist.ReduceOp.MIN)
        complete = bool(flag.item() == 1)

        if self.global_rank == 0:
            if complete:
                logger.info(
                    "eval baseline complete at %s -> skipping the step-0 eval. "
                    "vlm_*_win_rate will reference these pinned clips.",
                    baseline_dir,
                )
            else:
                logger.info(
                    "eval baseline at %s is missing clips -> running the step-0 "
                    "eval, which will populate it for later runs.",
                    baseline_dir,
                )
        if not complete and missing is not None:
            logger.debug(
                "rank %d: no step_0 clip for sample_%d in %s",
                self.global_rank, missing, baseline_dir,
            )
        return complete

    def train(self) -> None:
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info(
            "rank: %s: start training", self.global_rank, local_main_process_only=False
        )

        if not self.post_init_called:
            self.post_init()
        num_trainable_params = _get_trainable_params(self.transformer)
        logger.info(
            "Starting training with %s B trainable parameters",
            round(num_trainable_params / 1e9, 3),
        )

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed
        )
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(self.seed)
        self.validation_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed
        )
        logger.info("Initialized random seeds with seed: %s", self.seed)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # EMA must be constructed BEFORE _resume_from_checkpoint so the
        # stored ckpt_shadow / policy_shadow tensors can be loaded into it.
        # _init_shadow populates from current model weights, which is the
        # correct fallback if the checkpoint has no EMA state (legacy).
        self.ema_generator = EMA_FSDP_schedule(
            self.transformer,
            min_decay=self.training_args.ema_min_decay,
            max_decay=self.training_args.ema_max_decay,
            step_decay=self.training_args.ema_step_decay,
            ckpt_decay=self.training_args.ema_ckpt_decay,
        )

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()

        # Train!
        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=self.local_rank > 0,
        )
        skip_eval = bool(getattr(self.training_args, "skip_eval", False))
        skip_initial_eval = bool(getattr(self.training_args, "skip_initial_eval", False)) or os.environ.get("WC_SKIP_INITIAL_EVAL", "") == "1"
        if not skip_eval and not skip_initial_eval:
            skip_initial_eval = self._eval_baseline_is_complete()
        for step in range(self.init_steps + 1, self.training_args.max_train_steps + 1):
            if step == 1 and not skip_eval and not skip_initial_eval:
                eval_results = self._eval(step - 1)
                if self.global_rank == 0:
                    logger.info("Eval step %d: %s", step - 1, {k: round(v, 4) if isinstance(v, float) else v for k, v in eval_results.items() if not k.startswith("all_")})
                if self.global_rank == 0 and self.wandb_enabled:
                    # Key-based conditional logging — eval.py only populates
                    # each ave_* when the corresponding scorer actually ran.
                    # Skipping missing keys keeps wandb clean of placeholder
                    # zeros/NaNs for disabled scorers.
                    eval_wandb = {}
                    for k in (
                        "ave_action_acc", "ave_fine_action_acc",
                        "ave_hpsv3_acc", "ave_hpsv3_quality_acc", "ave_hpsv3_drift_score",
                        "ave_vlm_ac_win_rate", "ave_vlm_vq_win_rate",
                        "ave_ate_rmse", "ave_rpe_trans_rmse", "ave_rpe_rot_median",
                        "ave_aesthetic",
                    ):
                        if k in eval_results:
                            eval_wandb[f"eval_{k[4:]}"] = eval_results[k]
                    if eval_wandb:
                        wandb.log(eval_wandb, step=step)

            start_time = time.perf_counter()
            if vsa_available:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                current_decay_times = min(
                    step // vsa_decay_interval_steps, vsa_sparsity // vsa_decay_rate
                )
                current_vsa_sparsity = current_decay_times * vsa_decay_rate
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity
            training_batch = self.train_one_step(training_batch)

            samples_loss = training_batch.samples_loss
            samples_grad_norm = training_batch.samples_grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            total_loss = training_batch.total_loss / len(samples_grad_norm)
            total_grad_norm = sum(samples_grad_norm.values()) / len(samples_grad_norm)

            progress_bar.set_postfix(
                {
                    "loss": total_loss,
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": total_grad_norm,
                }
            )
            progress_bar.update(1)

            # Single-line structured summary on rank 0 so tailing the log
            # gives a clean per-step trace without having to parse tqdm
            # \r-updated progress bars.
            if self.global_rank == 0:
                logger.info(
                    "[step %d/%d] loss=%.4f grad=%.3f step_time=%.1fs avg=%.1fs",
                    step, self.training_args.max_train_steps,
                    float(total_loss),
                    float(total_grad_norm),
                    step_time, avg_step_time,
                )
                # One-line reward summary; disabled rewards give 0.0 and are
                # skipped. Pairwise win-rate rewards are omitted because their
                # group mean is identically 0.500 under vlm_pair_mode="all" —
                # read the per-candidate spread from wandb instead.
                _reward_means = getattr(training_batch, "reward_means", None) or {}
                _nonzero = {
                    k: float(v)
                    for k, v in _reward_means.items()
                    if v and k not in _PAIRWISE_REWARD_KEYS
                }
                if _nonzero:
                    _parts = [f"{k}={v:.3f}" for k, v in _nonzero.items()]
                    logger.info("[step %d/%d] rewards: %s", step,
                                self.training_args.max_train_steps,
                                " ".join(_parts))

            if self.global_rank == 0 and self.wandb_enabled:
                wandb.log(
                    {
                        f"ave_loss": training_batch.total_loss / len(samples_grad_norm),
                        f"ave_grad_norm": sum(samples_grad_norm.values())
                        / len(samples_grad_norm),
                    },
                    step=step,
                )

                train_wandb = {
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                }
                # Prefer the generic dispatcher-populated dict so adding
                # a new reward requires no edits here.
                reward_means = getattr(training_batch, "reward_means", None) or {}
                if reward_means:
                    for name, value in reward_means.items():
                        if value == 0.0:
                            continue  # skip disabled outputs
                        train_wandb[f"{name}_reward"] = value
                    # Always log action/fine_action when present (even if 0.0).
                    for always in ("action", "fine_action"):
                        if always in reward_means and f"{always}_reward" not in train_wandb:
                            train_wandb[f"{always}_reward"] = reward_means[always]
                else:
                    train_wandb["action_reward"] = training_batch.action_reward_mean
                    train_wandb["fine_action_reward"] = training_batch.fine_action_reward_mean
                    if self._hpsv3_enabled():
                        train_wandb["hpsv3_reward"] = training_batch.hpsv3_reward_mean
                        train_wandb["hpsv3_quality_reward"] = training_batch.hpsv3_quality_reward_mean
                wandb.log(train_wandb, step=step)

            training_batch = TrainingBatch()

            if step % self.training_args.checkpointing_steps == 0 and not skip_eval:
                eval_results = self._eval(step - 1)
                if self.global_rank == 0:
                    logger.info("Eval step %d: %s", step - 1, {k: round(v, 4) if isinstance(v, float) else v for k, v in eval_results.items() if not k.startswith("all_")})
                if self.global_rank == 0 and self.wandb_enabled:
                    # Key-based conditional logging — eval.py only populates
                    # each ave_* when the corresponding scorer actually ran.
                    # Skipping missing keys keeps wandb clean of placeholder
                    # zeros/NaNs for disabled scorers.
                    eval_wandb = {}
                    for k in (
                        "ave_action_acc", "ave_fine_action_acc",
                        "ave_hpsv3_acc", "ave_hpsv3_quality_acc", "ave_hpsv3_drift_score",
                        "ave_vlm_ac_win_rate", "ave_vlm_vq_win_rate",
                        "ave_ate_rmse", "ave_rpe_trans_rmse", "ave_rpe_rot_median",
                        "ave_aesthetic",
                    ):
                        if k in eval_results:
                            eval_wandb[f"eval_{k[4:]}"] = eval_results[k]
                    if eval_wandb:
                        wandb.log(eval_wandb, step=step)

                with self.ema_generator.apply_ckpt_shadow_to_model(self.transformer):
                    save_checkpoint(
                        self.transformer,
                        self.global_rank,
                        self.training_args.output_dir,
                        step,
                        self.optimizer,
                        self.train_dataloader,
                        self.lr_scheduler,
                        self.noise_random_generator,
                        self.ema_generator,
                        checkpoints_total_limit=self.training_args.checkpoints_total_limit,
                        config_path=getattr(self.training_args, "config_path", ""),
                    )
                self.transformer.train()
                self.sp_group.barrier()

        if self.global_rank == 0 and self.wandb_enabled:
            wandb.finish()
        save_checkpoint(
            self.transformer,
            self.global_rank,
            self.training_args.output_dir,
            self.training_args.max_train_steps,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.noise_random_generator,
            self.ema_generator,
            checkpoints_total_limit=self.training_args.checkpoints_total_limit,
            config_path=getattr(self.training_args, "config_path", ""),
        )

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def _log_training_info(self) -> None:
        total_batch_size = (
            self.world_size
            * self.training_args.gradient_accumulation_steps
            / self.training_args.sp_size
            * self.training_args.train_sp_batch_size
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(self.train_dataset))
        logger.info("  Dataloader size = %s", len(self.train_dataloader))
        logger.info("  Num Epochs = %s", self.num_train_epochs)
        logger.info("  Resume training from step %s", self.init_steps)  # type: ignore
        logger.info(
            "  Instantaneous batch size per device = %s",
            self.training_args.train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %s",
            self.training_args.gradient_accumulation_steps,
        )
        logger.info(
            "  Total optimization steps = %s", self.training_args.max_train_steps
        )
        logger.info(
            "  Total training parameters per FSDP shard = %s B",
            round(_get_trainable_params(self.transformer) / 1e9, 3),
        )
        # print dtype
        logger.info(
            "  Master weight dtype: %s", self.transformer.parameters().__next__().dtype
        )

        gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory usage before train_one_step: %s MB", gpu_memory_usage)
        logger.info("VSA validation sparsity: %s", self.training_args.VSA_sparsity)
