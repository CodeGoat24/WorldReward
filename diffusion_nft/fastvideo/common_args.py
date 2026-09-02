# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from typing import Any

from fastvideo.utils import StoreBoolean


def add_shared_data_and_rollout_cli_args(parser, defaults: Any):
    parser.add_argument("--data-path", type=str, default="", help="Path to parquet files")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        required=True,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--std-type",
        type=str,
        default="sample",
        choices=["sample", "global", "sample_max"],
        help="Standard deviation type",
    )
    parser.add_argument("--adv-clip-max", type=float, default=5.0, help="Advantage clip max")
    parser.add_argument("--sampling-steps", type=int, required=True, help="Number of sampling steps")
    parser.add_argument(
        "--sampling-batch-size", type=int, required=True, help="Number of sampling batch size"
    )
    parser.add_argument("--grpo-generation-num", type=int, required=True, help="GRPO generation number")
    parser.add_argument("--bestofn", type=int, required=True, help="GRPO best of n")
    parser.add_argument(
        "--train-timestep-fraction", type=float, required=True, help="GRPO train timestep fraction"
    )
    return parser


def add_shared_logging_cli_args(parser, defaults: Any):
    parser.add_argument("--ema-min-decay", type=float, default=defaults.ema_min_decay, help="Minimum decay rate for EMA")
    parser.add_argument("--ema-max-decay", type=float, default=defaults.ema_max_decay, help="Maximum decay rate for EMA")
    parser.add_argument("--ema-step-decay", type=float, default=defaults.ema_step_decay, help="Step decay rate for EMA")
    parser.add_argument("--ema-ckpt-decay", type=float, default=defaults.ema_ckpt_decay, help="Checkpoint decay rate for EMA")
    return parser


def add_shared_training_cli_args(parser, defaults: Any):
    parser.add_argument("--train-batch-size", type=int, required=True, help="Training batch size")
    parser.add_argument("--group-frame", action=StoreBoolean, help="Whether to group frames during training")
    parser.add_argument("--group-resolution", action=StoreBoolean, help="Whether to group resolutions during training")
    parser.add_argument("--pretrained-model-name-or-path", type=str, required=True, help="Path to pretrained model or model name")
    parser.add_argument("--dit-model-name-or-path", type=str, required=False, help="Path to DiT model or model name")
    parser.add_argument("--cache-dir", type=str, help="Directory to cache models")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--ema-start-step", type=int, default=0, help="Step to start EMA")
    parser.add_argument("--training-cfg-rate", type=float, help="Classifier-free guidance scale")
    parser.add_argument("--precondition-outputs", action=StoreBoolean, help="Whether to precondition the outputs of the model")
    parser.add_argument("--validation-dataset-file", type=str, help="Path to unprocessed validation dataset")
    parser.add_argument("--validation-preprocessed-path", type=str, help="Path to processed validation dataset")
    parser.add_argument("--validation-sampling-steps", type=str, help="Validation sampling steps")
    parser.add_argument("--validation-guidance-scale", type=str, help="Validation guidance scale")
    parser.add_argument("--validation-steps", type=float, help="Number of validation steps")
    parser.add_argument("--log-validation", action=StoreBoolean, help="Whether to log validation results")
    parser.add_argument("--tracker-project-name", type=str, help="Project name for tracking")
    parser.add_argument("--wandb-run-name", type=str, help="Run name for wandb")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic training")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--checkpoints-total-limit", type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--checkpointing-steps", type=int, help="Steps between checkpoints")
    parser.add_argument("--training-state-checkpointing-steps", type=int, help="Steps between training state checkpoints (for resuming training)")
    parser.add_argument("--weight-only-checkpointing-steps", type=int, help="Steps between weight-only checkpoints (for inference)")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--logging-dir", type=str, help="Directory for logging")
    parser.add_argument("--num-train-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max-train-steps", type=int, help="Maximum number of training steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of steps to accumulate gradients")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--scale-lr", action=StoreBoolean, help="Whether to scale learning rate")
    parser.add_argument("--lr-scheduler", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--lr-warmup-steps", type=int, default=10, help="Number of warmup steps for learning rate")
    parser.add_argument("--max-grad-norm", type=float, help="Maximum gradient norm")
    parser.add_argument("--enable-gradient-checkpointing-type", type=str, choices=["full", "ops", "block_skip"], default=None, help="Gradient checkpointing type")
    parser.add_argument("--selective-checkpointing", type=float, help="Selective checkpointing threshold")
    parser.add_argument("--mixed-precision", type=str, help="Mixed precision training type")
    parser.add_argument("--train-sp-batch-size", type=int, help="Training spatial parallelism batch size")
    parser.add_argument("--fsdp-sharding-strategy", type=str, help="FSDP sharding strategy")
    parser.add_argument("--weighting_scheme", type=str, default="uniform", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.")
    parser.add_argument("--num-euler-timesteps", type=int, help="Number of Euler timesteps")
    parser.add_argument("--lr-num-cycles", type=int, help="Number of learning rate cycles")
    parser.add_argument("--lr-power", type=float, help="Learning rate power")
    parser.add_argument("--min-lr-ratio", type=float, default=defaults.min_lr_ratio, help="Minimum learning rate ratio for cosine_with_min_lr scheduler")
    parser.add_argument("--not-apply-cfg-solver", action=StoreBoolean, help="Whether to not apply CFG solver")
    parser.add_argument("--distill-cfg", type=float, help="Distillation CFG scale")
    parser.add_argument("--scheduler-type", type=str, help="Scheduler type")
    parser.add_argument("--linear-quadratic-threshold", type=float, help="Linear quadratic threshold")
    parser.add_argument("--linear-range", type=float, help="Linear range")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--use-ema", action=StoreBoolean, help="Whether to use EMA")
    parser.add_argument("--multi-phased-distill-schedule", type=str, help="Multi-phased distillation schedule")
    parser.add_argument("--pred-decay-weight", type=float, help="Prediction decay weight")
    parser.add_argument("--pred-decay-type", type=str, help="Prediction decay type")
    parser.add_argument("--hunyuan-teacher-disable-cfg", action=StoreBoolean, help="Whether to disable CFG for Hunyuan teacher")
    parser.add_argument("--master-weight-type", type=str, help="Master weight type")
    parser.add_argument("--VSA-decay-rate", type=float, default=defaults.VSA_decay_rate, help="VSA decay rate")
    parser.add_argument("--VSA-decay-interval-steps", type=int, default=defaults.VSA_decay_interval_steps, help="VSA decay interval steps")
    parser.add_argument("--lora-training", action=StoreBoolean, help="Whether to use LoRA training")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--generator-update-interval", type=int, default=defaults.generator_update_interval, help="Ratio of student updates to critic updates.")
    parser.add_argument("--min-timestep-ratio", type=float, default=defaults.min_timestep_ratio, help="Minimum step ratio")
    parser.add_argument("--max-timestep-ratio", type=float, default=defaults.max_timestep_ratio, help="Maximum step ratio")
    parser.add_argument("--real-score-guidance-scale", type=float, default=defaults.real_score_guidance_scale, help="Teacher guidance scale")
    parser.add_argument("--fake-score-learning-rate", type=float, default=defaults.fake_score_learning_rate, help="Learning rate for fake score transformer")
    parser.add_argument("--fake-score-lr-scheduler", type=str, default=defaults.fake_score_lr_scheduler, help="Learning rate scheduler for fake score transformer")
    parser.add_argument("--log-visualization", action=StoreBoolean, help="Whether to log visualization")
    parser.add_argument("--simulate-generator-forward", action=StoreBoolean, help="Whether to simulate generator forward to match inference")
    return parser


def add_shared_dataset_and_reward_cli_args(parser, defaults: Any):
    parser.add_argument("--json-path", type=str, default="", help="camera json path")
    parser.add_argument("--eval-json-path", type=str, default="", help="eval camera json path")
    parser.add_argument("--wandb-key", type=str, default="", help="key")
    parser.add_argument("--wandb-entity", type=str, default="", help="entity")
    parser.add_argument("--vae-path", type=str, default="", help="VAE model path")
    parser.add_argument("--random-pose-path", type=str, default="", help="Random pose json file path")
    parser.add_argument("--neg-prompt-path", type=str, default="", help="Negative prompt pt file path")
    parser.add_argument("--neg-byt5-prompt-path", type=str, default="", help="Negative byt5 prompt pt file path")
    parser.add_argument("--generated-videos-dir", type=str, default="", help="Directory to save generated videos during training")
    parser.add_argument("--camera-estimator", type=str, default="dav3", choices=["dav3", "worldmirror"], help="Shared camera-pose reward backend used by training/eval.")
    parser.add_argument("--eval-worldmirror", action=argparse.BooleanOptionalAction, default=False, help="Run WorldMirror action_acc on eval-step videos. Also forces WorldMirror to be loaded even when action_reward_weight=0.")
    # skip_initial_eval and eval_baseline_dir need explicit CLI flags here so
    # yaml configs can set them (the yaml loader forwards keys as CLI flags).
    # --skip-eval is NOT registered here: FastVideoArgs.add_cli_args already
    # defines it on the same parser, and registering it twice raises
    # "conflicting option string".
    parser.add_argument("--skip-initial-eval", action=StoreBoolean, default=False, help="Skip only the step-0 baseline eval; mid-training evals still run. Pair with --eval-baseline-dir so vlm_*_win_rate still has a reference. Also settable via WC_SKIP_INITIAL_EVAL=1.")
    parser.add_argument("--eval-baseline-dir", type=str, default="", help="Shared directory holding the step_0 reference clips that vlm_ac/vq_win_rate compare against. Empty = keep them inside this run's generated_videos_dir (regenerated every run, ~33 min). Point multiple runs at one directory to generate once and reuse; valid only while base ckpt + eval json + eval_window_frames are unchanged.")
    parser.add_argument("--eval-monst3r", action=argparse.BooleanOptionalAction, default=True, help="Run MonST3R-based ATE/RPE evaluation on eval-step videos and append scores to the filename. Use --no-eval-monst3r to disable.")
    parser.add_argument("--eval-monst3r-n-iter", type=int, default=300, help="MonST3R global-alignment iterations (eval only).")
    parser.add_argument("--eval-monst3r-stride", type=int, default=4, help="Frame stride for MonST3R eval (keep every k-th frame of the video).")
    parser.add_argument("--eval-aesthetic", action=argparse.BooleanOptionalAction, default=True, help="Run LAION aesthetic predictor on eval-step videos and append score to the filename. Use --no-eval-aesthetic to disable.")
    parser.add_argument("--eval-aesthetic-num-frames", type=int, default=16, help="Number of frames uniformly sampled from each eval video for aesthetic scoring.")
    parser.add_argument("--eval-window-frames", type=int, default=32, help="Number of latents (temporal axis) to generate during eval. Decoded video has (N-1)*4+1 pixel frames for HunyuanVAE. Default 32 = half of training window_frames=64 (fast eval).")
    parser.add_argument("--eval-sampling-steps", type=int, default=20, help="Denoising steps used during eval (ODE). Default 20 matches training rollout; raise (e.g. 40) for sharper videos at the cost of eval wall-clock.")
    return parser
