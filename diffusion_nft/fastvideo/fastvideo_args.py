# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of FastVideo Inference."""

import argparse
import dataclasses
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from typing import Any

from fastvideo.configs.configs import PreprocessConfig
from fastvideo.configs.pipelines.base import PipelineConfig, STA_Mode
from fastvideo.configs.utils import clean_cli_args
from fastvideo.common_args import (
    add_shared_data_and_rollout_cli_args,
    add_shared_dataset_and_reward_cli_args,
    add_shared_logging_cli_args,
    add_shared_training_cli_args,
)
from fastvideo.utils.logger import init_logger
from fastvideo.platforms import current_platform
from fastvideo.training.nft_args import add_worldplay_cli_args
from fastvideo.utils import FlexibleArgumentParser, StoreBoolean

logger = init_logger(__name__)


class ExecutionMode(str, Enum):
    """Enumeration for different pipeline modes.

    Inherits from str to allow string comparison for backward compatibility.
    """

    INFERENCE = "inference"
    PREPROCESS = "preprocess"
    FINETUNING = "finetuning"
    DISTILLATION = "distillation"

    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Convert string to ExecutionMode enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [mode.value for mode in cls]


class WorkloadType(str, Enum):
    """Enumeration for different workload types.

    Inherits from str to allow string comparison for backward compatibility.
    """

    I2V = "i2v"  # Image to Video
    T2V = "t2v"  # Text to Video
    T2I = "t2i"  # Text to Image
    I2I = "i2i"  # Image to Image

    @classmethod
    def from_string(cls, value: str) -> "WorkloadType":
        """Convert string to WorkloadType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid workload type: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [workload.value for workload in cls]


# args for fastvideo framework
@dataclasses.dataclass
class FastVideoArgs:
    # Model and path configuration (for convenience)
    model_path: str = ""

    # Running mode
    mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy
    cache_strategy: str = "none"

    # Distributed executor backend
    distributed_executor_backend: str = "mp"

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    gpu_para: int = 1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # timeout for torch.distributed

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    preprocess_config: PreprocessConfig | None = None

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = True
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # Master port for distributed training/inference
    master_port: int | None = None

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_txt: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
        }
    )

    # When True, all training-loop _eval() calls are skipped (step-1 baseline
    # eval and the per-checkpointing_steps mid-training eval). Saves hours
    # when eval dataset is large (e.g. 128 prompts × MonST3R/aesthetic/VLM).
    skip_eval: bool = False
    # When True, only the step-1 baseline eval (before any training step) is
    # skipped; the per-checkpointing_steps mid-training eval still runs. Useful
    # when the initial eval reliably hangs/crashes but periodic evals are fine.
    skip_initial_eval: bool = False
    # Directory holding the step_0 baseline clips that every later eval is
    # compared against for vlm_ac_win_rate / vlm_vq_win_rate. Empty (default)
    # keeps the historical behaviour: the baseline lives inside this run's own
    # generated_videos_dir/000_eval/, so each run regenerates it (~33 min).
    #
    # Point several runs at the same directory to generate it once and reuse
    # it. The first run to reach its step-0 eval fills the directory; later
    # runs can then set skip_initial_eval (or WC_SKIP_INITIAL_EVAL=1) and read
    # the clips straight from here.
    #
    # The baseline is a function of (base checkpoint, eval json,
    # eval_window_frames) only — NOT of num_gpus, since eval sample ids run
    # 0..N-1 regardless of how ranks shard them. Nothing validates the key, so
    # name the directory after those inputs. Reusing a pinned baseline is also
    # more correct than regenerating: the eval rollout noise is not seeded
    # (eval.py calls torch.randn directly), so a fresh step_0 differs slightly
    # every time and would shift the win-rate reference between runs.
    eval_baseline_dir: str = ""
    cls_name: str | None = None
    load_from_dir: str | None = None
    module_name: str | None = None
    ar_action_load_from_dir: str | None = None

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = None

    @property
    def training_mode(self) -> bool:
        return not self.inference_mode

    def __post_init__(self):
        self.check_fastvideo_args()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )

        # Running mode
        parser.add_argument(
            "--mode",
            type=str,
            choices=ExecutionMode.choices(),
            default=FastVideoArgs.mode.value,
            help="The mode to run FastVideo",
        )

        # Workload type
        parser.add_argument(
            "--workload-type",
            type=str,
            choices=WorkloadType.choices(),
            default=FastVideoArgs.workload_type.value,
            help="The workload type",
        )

        # distributed_executor_backend
        parser.add_argument(
            "--distributed-executor-backend",
            type=str,
            choices=["mp"],
            default=FastVideoArgs.distributed_executor_backend,
            help="The distributed executor backend to use",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=FastVideoArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=FastVideoArgs.revision,
            help="The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=FastVideoArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=FastVideoArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sp-size",
            type=int,
            default=FastVideoArgs.sp_size,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--gpu-para",
            type=int,
            default=FastVideoArgs.gpu_para,
            help="The GPU parallelism size.",
        )
        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=FastVideoArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=FastVideoArgs.hsdp_shard_dim,
            help="The data parallelism shards.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=FastVideoArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Output type
        parser.add_argument(
            "--output-type",
            type=str,
            default=FastVideoArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )

        # Prompt text file for batch processing
        parser.add_argument(
            "--prompt-txt",
            type=str,
            default=FastVideoArgs.prompt_txt,
            help="Path to a text file containing prompts (one per line) for batch processing",
        )

        # STA (Sliding Tile Attention) parameters
        parser.add_argument(
            "--STA-mode",
            type=str,
            default=FastVideoArgs.STA_mode.value,
            choices=[mode.value for mode in STA_Mode],
            help="STA mode contains STA_inference, STA_searching, STA_tuning, STA_tuning_cfg, None",
        )
        parser.add_argument(
            "--skip-time-steps",
            type=int,
            default=FastVideoArgs.skip_time_steps,
            help="Number of time steps to warmup (full attention) for STA",
        )
        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=FastVideoArgs.enable_torch_compile,
            help="Use torch.compile to speed up DiT inference."
            + "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )

        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help="Use FSDP for inference by sharding the model weights. Latency is very low due to prefetch--enable if run out of memory.",
        )
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help='Pin memory for CPU offload. Only added as a temp workaround if it throws "CUDA error: invalid argument". '
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help="Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # VSA parameters
        parser.add_argument(
            "--VSA-sparsity",
            type=float,
            default=FastVideoArgs.VSA_sparsity,
            help="Validation sparsity for VSA",
        )

        # Master port for distributed training/inference
        parser.add_argument(
            "--master-port",
            type=int,
            default=FastVideoArgs.master_port,
            help="Master port for distributed training/inference",
        )

        # Stage verification
        parser.add_argument(
            "--enable-stage-verification",
            action=StoreBoolean,
            default=FastVideoArgs.enable_stage_verification,
            help="Enable input/output verification for pipeline stages",
        )

        parser.add_argument(
            "--cls-name",
            type=str,
            help="model package name",
        )
        parser.add_argument(
            "--load-from-dir",
            type=str,
            help="ar model checkpoint directory",
        )
        parser.add_argument(
            "--ar-action-load-from-dir",
            type=str,
            help="ar action model checkpoint directory",
        )
        parser.add_argument(
            "--skip-eval",
            action=StoreBoolean,
            help="Skip all mid-training _eval() calls (step-1 baseline + per-checkpointing_steps). Useful when eval set is large.",
        )
        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

        # Add preprocessing configuration arguments
        PreprocessConfig.add_cli_args(parser)

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "FastVideoArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs: dict[str, Any] = {}
        for attr in attrs:
            if attr == "pipeline_config":
                if "HunyuanTransformer" in provided_args.get("cls_name", ""):
                    continue
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs["pipeline_config"] = pipeline_config
            elif attr == "preprocess_config":
                preprocess_config = PreprocessConfig.from_kwargs(provided_args)
                kwargs["preprocess_config"] = preprocess_config
            elif attr == "mode":
                # Convert string to ExecutionMode enum
                mode_value = getattr(args, attr, FastVideoArgs.mode.value)
                kwargs["mode"] = (
                    ExecutionMode.from_string(mode_value)
                    if isinstance(mode_value, str)
                    else mode_value
                )
            elif attr == "workload_type":
                # Convert string to WorkloadType enum
                workload_type_value = getattr(
                    args, "workload_type", FastVideoArgs.workload_type.value
                )
                kwargs["workload_type"] = (
                    WorkloadType.from_string(workload_type_value)
                    if isinstance(workload_type_value, str)
                    else workload_type_value
                )
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                # Get the field to check if it has a default_factory
                field = dataclasses.fields(cls)[
                    next(
                        i
                        for i, f in enumerate(dataclasses.fields(cls))
                        if f.name == attr
                    )
                ]
                if field.default_factory is not dataclasses.MISSING:
                    # Use the default_factory to create the default value
                    default_value = field.default_factory()
                else:
                    default_value = getattr(cls, attr, None)
                value = getattr(args, attr, default_value)
                kwargs[attr] = value  # type: ignore

        return cls(**kwargs)  # type: ignore

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "FastVideoArgs":
        # Convert mode string to enum if necessary
        if "mode" in kwargs and isinstance(kwargs["mode"], str):
            kwargs["mode"] = ExecutionMode.from_string(kwargs["mode"])

        # Convert workload_type string to enum if necessary
        if "workload_type" in kwargs and isinstance(
            kwargs["workload_type"], str
        ):
            kwargs["workload_type"] = WorkloadType.from_string(
                kwargs["workload_type"]
            )

        kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
        kwargs["preprocess_config"] = PreprocessConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    def check_fastvideo_args(self) -> None:
        """Validate inference arguments for consistency."""
        if current_platform.is_mps():
            self.use_fsdp_inference = False

        # Validate mode and inference_mode consistency
        assert isinstance(
            self.mode, ExecutionMode
        ), f"Mode must be an ExecutionMode enum, got {type(self.mode)}"
        assert (
            self.mode in ExecutionMode.choices()
        ), f"Invalid execution mode: {self.mode}"

        # Validate workload type
        assert isinstance(
            self.workload_type, WorkloadType
        ), f"Workload type must be a WorkloadType enum, got {type(self.workload_type)}"
        assert (
            self.workload_type in WorkloadType.choices()
        ), f"Invalid workload type: {self.workload_type}"

        self.inference_mode = False

        if not self.inference_mode:
            assert (
                self.hsdp_replicate_dim != -1
            ), "hsdp_replicate_dim must be set for training"
            assert (
                self.hsdp_shard_dim != -1
            ), "hsdp_shard_dim must be set for training"
            assert self.sp_size != -1, "sp_size must be set for training"

        if self.tp_size == -1:
            self.tp_size = 1
        if self.sp_size == -1:
            self.sp_size = self.num_gpus
        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_gpus

        assert (
            self.sp_size <= self.num_gpus and self.num_gpus % self.sp_size == 0
        ), "num_gpus must >= and be divisible by sp_size"
        assert (
            self.hsdp_replicate_dim <= self.num_gpus
            and self.num_gpus % self.hsdp_replicate_dim == 0
        ), "num_gpus must >= and be divisible by hsdp_replicate_dim"
        assert (
            self.hsdp_shard_dim <= self.num_gpus
            and self.num_gpus % self.hsdp_shard_dim == 0
        ), "num_gpus must >= and be divisible by hsdp_shard_dim"

        if self.num_gpus < max(self.tp_size, self.sp_size):
            self.num_gpus = max(self.tp_size, self.sp_size)

        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in FastVideoArgs")

        self.pipeline_config.check_pipeline_config()

        # Add preprocessing config validation if needed
        if self.mode == ExecutionMode.PREPROCESS:
            if self.preprocess_config is None:
                raise ValueError(
                    "preprocess_config is not set in FastVideoArgs when mode is PREPROCESS"
                )
            if self.preprocess_config.model_path == "":
                self.preprocess_config.model_path = self.model_path
            if not self.pipeline_config.vae_config.load_encoder:
                self.pipeline_config.vae_config.load_encoder = True
            self.preprocess_config.check_preprocess_config()


_current_fastvideo_args = None


def prepare_fastvideo_args(argv: list[str]) -> FastVideoArgs:
    """Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    FastVideoArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    fastvideo_args = FastVideoArgs.from_cli_args(raw_args)
    global _current_fastvideo_args
    _current_fastvideo_args = fastvideo_args
    return fastvideo_args


@contextmanager
def set_current_fastvideo_args(fastvideo_args: FastVideoArgs):
    """Temporarily set the current fastvideo config.

    Used during model initialization. We save the current fastvideo config in a global variable, so
    that all modules can access it, e.g. custom ops can access the fastvideo config to determine how
    to dispatch.
    """
    global _current_fastvideo_args
    old_fastvideo_args = _current_fastvideo_args
    try:
        _current_fastvideo_args = fastvideo_args
        yield
    finally:
        _current_fastvideo_args = old_fastvideo_args


def get_current_fastvideo_args() -> FastVideoArgs:
    if _current_fastvideo_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the fastvideo config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current fastvideo args is not set.")
    return _current_fastvideo_args


@dataclasses.dataclass
class TrainingArgs(FastVideoArgs):
    """Training arguments.

    Inherits from FastVideoArgs and adds training-specific arguments. If there are any conflicts,
    the training arguments will take precedence.
    """

    # Shared data loading controls
    data_path: str = ""
    dataloader_num_workers: int = 0
    train_batch_size: int = 0
    group_frame: bool = False
    group_resolution: bool = False

    # Path to the training YAML passed via --config; archived next to each
    # checkpoint for reproducibility. Populated at the training entrypoint.
    config_path: str = ""

    # Shared model path controls
    pretrained_model_name_or_path: str = ""
    dit_model_name_or_path: str = ""

    # Shared diffusion configuration
    ema_decay: float = 0.0
    ema_start_step: int = 0
    training_cfg_rate: float = 0.0
    precondition_outputs: bool = False

    # Shared NFT/GRPO training controls
    std_type: str = "sample"
    adv_clip_max: float = 5.0

    # Shared rollout/sampling controls
    sampling_steps: int = 10
    sampling_batch_size: int = 2
    grpo_generation_num: int = 2
    bestofn: int = 2
    train_timestep_fraction: float = 0.5

    # Shared EMA scheduling parameters
    ema_min_decay: float = 0.4
    ema_max_decay: float = 0.9
    ema_step_decay: float = 0.002
    ema_ckpt_decay: float = 0.9

    # Shared validation and logging controls
    validation_dataset_file: str = ""
    validation_preprocessed_path: str = ""
    validation_sampling_steps: str = ""
    validation_guidance_scale: str = ""
    validation_steps: float = 0.0
    log_validation: bool = False
    tracker_project_name: str = ""
    wandb_run_name: str = ""
    seed: int | None = None

    # Shared output controls
    output_dir: str = ""
    # If > 0, keep at most this many on-disk checkpoint directories,
    # deleting the oldest when a new one would exceed the cap. Set to 0
    # to keep all checkpoints (legacy behavior). Default 4 is enough to
    # have the most recent ckpt + 3 older ones for rollback/comparison.
    checkpoints_total_limit: int = 4
    checkpointing_steps: int = 0
    resume_from_checkpoint: str = (
        ""  # specify the checkpoint folder to resume from
    )

    # Shared optimizer and scheduler controls
    num_train_epochs: int = 0
    max_train_steps: int = 0
    gradient_accumulation_steps: int = 0
    learning_rate: float = 0.0
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 0.0
    enable_gradient_checkpointing_type: str | None = None
    selective_checkpointing: float = 0.0
    mixed_precision: str = ""
    train_sp_batch_size: int = 0
    fsdp_sharding_startegy: str = ""

    weighting_scheme: str = ""
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 0.0

    num_euler_timesteps: int = 0
    lr_num_cycles: int = 0
    lr_power: float = 0.0
    min_lr_ratio: float = (
        0.5  # minimum learning rate ratio for cosine_with_min_lr scheduler
    )
    not_apply_cfg_solver: bool = False
    distill_cfg: float = 0.0
    scheduler_type: str = ""
    linear_quadratic_threshold: float = 0.0
    linear_range: float = 0.0
    weight_decay: float = 0.0
    use_ema: bool = False
    multi_phased_distill_schedule: str = ""
    pred_decay_weight: float = 0.0
    pred_decay_type: str = ""
    hunyuan_teacher_disable_cfg: bool = False

    # Shared teacher/master weight controls
    master_weight_type: str = ""

    # Shared VSA training decay parameters
    VSA_decay_rate: float = 0.01  # decay rate -> 0.02
    VSA_decay_interval_steps: int = 1  # decay interval steps -> 50

    # Shared LoRA training parameters
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_training: bool = False

    # Shared camera-conditioned dataset and path controls
    json_path: str = ""
    eval_json_path: str = ""
    wandb_key: str = ""
    wandb_entity: str = ""
    vae_path: str = ""  # VAE model path
    random_pose_path: str = ""  # Random pose json file path
    neg_prompt_path: str = ""  # Negative prompt pt file path
    neg_byt5_prompt_path: str = ""  # Negative byt5 prompt pt file path
    generated_videos_dir: str = (
        ""  # Directory to save generated videos during training
    )

    # Shared reward model arguments
    camera_estimator: str = (
        "dav3"  # Camera pose estimator: "dav3" or "worldmirror"
    )
    cache_dir: str = ""  # Cache directory for downloading models

    # Shared distillation arguments
    generator_update_interval: int = 5
    min_timestep_ratio: float = 0.2
    max_timestep_ratio: float = 0.98
    real_score_guidance_scale: float = 3.5
    fake_score_learning_rate: float = (
        0.0  # separate learning rate for fake_score_transformer, if 0.0, use learning_rate
    )
    fake_score_lr_scheduler: str = (
        "constant"  # separate lr scheduler for fake_score_transformer, if not set, use lr_scheduler
    )
    training_state_checkpointing_steps: int = 0  # for resuming training
    weight_only_checkpointing_steps: int = 0  # for inference
    log_visualization: bool = False
    # simulate generator forward to match inference
    simulate_generator_forward: bool = False

    # WorldPlay-specific rollout and reward controls
    single_chunk_size: int = 1
    action_reward_type: bool = False
    hpsv3_reward_weight: float = 0.2
    hpsv3_quality_reward_weight: float = 0.2
    action_reward_weight: float = 1.0
    hpsv3_quality_drift_reward_weight: float = 0.2

    # VLM Reward Model
    vlm_action_reward_weight: float = 0.0
    vlm_vq_reward_weight: float = 0.0

    # Absolute-value reward anchors (prevent VLM reward hacking).
    # MonST3R trajectory metrics are turned into rewards via exp(-metric/scale)
    # so that smaller error -> larger reward in [0, 1].
    ate_rmse_reward_weight: float = 0.0
    rpe_trans_reward_weight: float = 0.0
    rpe_rot_reward_weight: float = 0.0
    ate_rmse_scale: float = 0.3
    rpe_trans_scale: float = 0.15
    rpe_rot_scale: float = 1.5
    # Aesthetic: raw CLIP+MLP score ∈ [~0, 10]; reward = score / aesthetic_reward_scale.
    aesthetic_reward_weight: float = 0.0
    aesthetic_reward_scale: float = 10.0
    vlm_rm_host: str = "localhost"
    vlm_rm_port: int = 8080
    vlm_rm_model_name: str = "UnifiedReward"
    vlm_pair_mode: str = "all"
    vlm_rm_num_replicas: int = 1
    # Optional list of full vLLM endpoint URLs (e.g. ["http://ip:9080", ...]).
    # When set, switches VLMRewardBackend to dedicated mode: requests are
    # round-robin distributed across these URLs and wake/sleep is disabled
    # (assumes vLLM runs on separate GPUs, not colocated with training).
    vlm_rm_urls: list[str] = field(default_factory=list)
    vlm_prefilter_enabled: bool = False
    vlm_prefilter_topk: int = 8

    # Eval-only scorers (independent of whether these models are loaded
    # as part of the training reward).
    eval_worldmirror: bool = False
    eval_monst3r: bool = False
    eval_monst3r_n_iter: int = 300
    eval_monst3r_stride: int = 4
    eval_aesthetic: bool = False
    eval_aesthetic_num_frames: int = 16
    # Number of latents (temporal axis) to generate during eval; the decoded
    # video has (eval_window_frames - 1) * VAE_temporal_compression + 1 pixel
    # frames. Default 32 (half of training window_frames=64) keeps eval fast;
    # raise for full-length evaluation.
    eval_window_frames: int = 32
    # Number of denoising steps used during eval (ODE, no SDE). Default 20
    # matches the training rollout cadence; raise (e.g. 40) for sharper eval
    # videos at the cost of eval wall-clock, lower (e.g. 10) for faster eval
    # if the rollout schedule is short.
    eval_sampling_steps: int = 20

    # WorldPlay-specific chunk scheduling
    chunk_selection_strategy: str = (
        "min2max"  # Options: "min2max", "max2min", "progressive_min2max"
    )
    min_chunk_id: int = 1
    max_chunk_id: int = 16
    # Progressive curriculum: unlock one deeper chunk position every
    # `chunk_target_repeats` training steps. Only used when
    # chunk_selection_strategy == "progressive_min2max".
    chunk_target_repeats: int = 3
    # Optional bias within the progressive_min2max bucket toward shallower
    # chunks. 0.0 (default) preserves deterministic round-robin. When > 0,
    # chunk k is sampled with P ∝ 1/(k - min_chunk_id + 1)**alpha using the
    # step counter as seed (reproducible). alpha=0.5 gives moderate shallow
    # bias; alpha=1.0 gives strong shallow bias. Useful when max_chunk_id is
    # large (e.g. 16) and deep chunks have high advantage variance from
    # autoregressive error accumulation.
    chunk_sample_alpha: float = 0.0

    # GRPO rollout SDE controls (diversity vs. sharpness trade-off).
    # Default preserves the historical ODE-only behavior (eta=0,
    # sde_solver=False). Setting grpo_sde=True + grpo_eta>0 injects noise
    # during the first `grpo_eta_cutoff` fraction of sampling_steps and
    # then switches to pure ODE so Euler-Maruyama discretization error
    # at small sigma doesn't blur the final frames.
    # Empirically-validated sweet spot (offline eta sweep): eta=0.3,
    # cutoff=0.3 (sde=True). Gives diversity_ratio ~0.27 vs. ODE baseline
    # ~0.19, with end-frame sharpness only 15% below ODE.
    grpo_sde: bool = False
    grpo_eta: float = 0.0
    grpo_eta_cutoff: float = 1.0
    # "flux"      — Euler-Maruyama on flux-style SDE with simple drift
    #                correction. Constant noise scale w.r.t. sigma → must
    #                set grpo_eta_cutoff < 1.0 to avoid end-frame blur.
    # "flow_grpo" — Rectified-flow SDE with σ-adaptive noise scale. Noise
    #                auto-shrinks near sigma=0; cutoff=1.0 is safe. Uses the
    #                variance-preserving Fokker-Planck drift from flow_grpo.
    grpo_sde_solver: str = "flux"

    # WorldPlay-specific temporal controls
    causal: bool = False
    window_frames: int = 9
    action: bool = False
    i2v_rate: float = 0.0
    use_mem: bool = False
    memory_nframes: int = 4

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TrainingArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        logger.info(provided_args)
        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs: dict[str, Any] = {}
        for attr in attrs:
            if attr == "pipeline_config":
                if "HunyuanTransformer" in provided_args.get("cls_name", ""):
                    continue
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs[attr] = pipeline_config
            elif attr == "mode":
                # Convert string to ExecutionMode enum
                mode_value = getattr(args, attr, ExecutionMode.FINETUNING.value)
                kwargs[attr] = (
                    ExecutionMode.from_string(mode_value)
                    if isinstance(mode_value, str)
                    else mode_value
                )
            elif attr == "workload_type":
                # Convert string to WorkloadType enum
                workload_type_value = getattr(
                    args, "workload_type", WorkloadType.T2V.value
                )
                kwargs[attr] = (
                    WorkloadType.from_string(workload_type_value)
                    if isinstance(workload_type_value, str)
                    else workload_type_value
                )
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                # Get the field to check its default value
                field = dataclasses.fields(cls)[
                    next(
                        i
                        for i, f in enumerate(dataclasses.fields(cls))
                        if f.name == attr
                    )
                ]

                # Check if the attribute is provided in args
                if hasattr(args, attr):
                    value = getattr(args, attr)
                else:
                    # Use the field's default value
                    if field.default_factory is not dataclasses.MISSING:
                        value = field.default_factory()
                    elif field.default is not dataclasses.MISSING:
                        value = field.default
                    else:
                        # No default value, use None
                        value = None

                kwargs[attr] = value

        return cls(**kwargs)  # type: ignore

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = TrainingArgs._add_shared_data_and_rollout_cli_args(parser)
        parser = TrainingArgs._add_worldplay_cli_args(parser)
        parser = TrainingArgs._add_shared_logging_cli_args(parser)
        parser = TrainingArgs._add_shared_training_cli_args(parser)
        parser = TrainingArgs._add_shared_dataset_and_reward_cli_args(parser)
        return parser

    @staticmethod
    def _add_shared_data_and_rollout_cli_args(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        return add_shared_data_and_rollout_cli_args(parser, TrainingArgs)

    @staticmethod
    def _add_worldplay_cli_args(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        return add_worldplay_cli_args(parser, TrainingArgs)

    @staticmethod
    def _add_shared_logging_cli_args(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        return add_shared_logging_cli_args(parser, TrainingArgs)

    @staticmethod
    def _add_shared_training_cli_args(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        return add_shared_training_cli_args(parser, TrainingArgs)

    @staticmethod
    def _add_shared_dataset_and_reward_cli_args(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        return add_shared_dataset_and_reward_cli_args(parser, TrainingArgs)


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]
