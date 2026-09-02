# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from fastvideo.utils import StoreBoolean


def add_worldplay_cli_args(parser, defaults: Any):
    parser.add_argument("--single-chunk-size", type=int, default=4, help="WorldPlay chunk length used by chunked rollout/training.")
    parser.add_argument("--action-reward-type", type=str, default="action", help="Whether to use fine action reward")
    parser.add_argument("--hpsv3-reward-weight", type=float, default=0.2, help="Hpsv3 reward weight")
    parser.add_argument("--hpsv3-quality-reward-weight", type=float, default=0.2, help="Hpsv3 quality reward weight")
    parser.add_argument("--action-reward-weight", type=float, default=1.0, help="Action reward weight")
    parser.add_argument("--hpsv3-quality-drift-reward-weight", type=float, default=0.2, help="Hpsv3 quality drift reward weight")
    parser.add_argument("--chunk-selection-strategy", type=str, default=defaults.chunk_selection_strategy, choices=["min2max", "max2min", "progressive_min2max"], help="WorldPlay chunk scheduling strategy for chunked rollout. progressive_min2max unlocks one deeper chunk position every --chunk-target-repeats steps and picks from the unlocked bucket (round-robin).")
    parser.add_argument("--min-chunk-id", type=int, default=defaults.min_chunk_id, help="WorldPlay minimum chunk id used by chunk scheduling.")
    parser.add_argument("--max-chunk-id", type=int, default=defaults.max_chunk_id, help="WorldPlay maximum chunk id used by chunk scheduling.")
    parser.add_argument("--chunk-target-repeats", type=int, default=defaults.chunk_target_repeats, help="For progressive_min2max: number of training steps before unlocking the next chunk position (default 3).")
    parser.add_argument("--chunk-sample-alpha", type=float, default=defaults.chunk_sample_alpha, help="For progressive_min2max: when > 0, sample chunk from unlocked bucket with P(k) proportional to 1/(k-min_chunk_id+1)**alpha instead of deterministic round-robin. alpha=0 (default) preserves round-robin. alpha=0.5 = moderate shallow bias; alpha=1.0 = strong shallow bias. Recommended when max_chunk_id is large (>=12).")
    parser.add_argument("--causal", action=StoreBoolean, help="Enable autoregressive training behavior for WorldPlay-style rollout.")
    parser.add_argument("--use-mem", action=StoreBoolean, help="Enable memory-frame usage for WorldPlay-style training.")
    parser.add_argument("--window-frames", type=int, default=defaults.window_frames, help="WorldPlay temporal window length in latent/video frames.")
    parser.add_argument("--action", action=StoreBoolean, help="Enable action-conditioned training data fields.")
    parser.add_argument("--i2v-rate", type=float, default=defaults.i2v_rate, help="Fraction of image-to-video samples used by the shared training dataloader.")
    # VLM Reward Model
    parser.add_argument("--vlm-action-reward-weight", type=float, default=0.0, help="VLM action control reward weight (0 = disabled)")
    parser.add_argument("--vlm-vq-reward-weight", type=float, default=0.0, help="VLM visual quality reward weight (0 = disabled)")
    parser.add_argument("--vlm-rm-host", type=str, default="localhost", help="VLM RM vLLM server host")
    parser.add_argument("--vlm-rm-port", type=int, default=8080, help="VLM RM vLLM server port")
    parser.add_argument("--vlm-rm-model-name", type=str, default="UnifiedReward", help="VLM RM served model name")
    parser.add_argument("--vlm-pair-mode", type=str, default="all", choices=["all", "topk"], help="VLM pairwise comparison strategy: all pairs or topk pre-filtered")
    parser.add_argument("--vlm-rm-num-replicas", type=int, default=1, help="Number of vLLM replicas on consecutive ports for round-robin load balancing")
    parser.add_argument("--vlm-rm-urls", type=str, nargs="*", default=[], help="Optional list of full vLLM endpoint URLs for dedicated (non-colocated) deployment. When set, overrides host/port/num-replicas and disables wake/sleep.")
    parser.add_argument("--vlm-prefilter-enabled", action=StoreBoolean, help="Enable WorldMirror prefiltering before VLM pairwise comparison")
    parser.add_argument("--vlm-prefilter-topk", type=int, default=8, help="Keep top K candidates by WorldMirror score before VLM comparison")
    # Absolute-value reward anchors (anti-reward-hacking)
    parser.add_argument("--ate-rmse-reward-weight", type=float, default=0.0, help="MonST3R ATE RMSE reward weight (0 = disabled)")
    parser.add_argument("--rpe-trans-reward-weight", type=float, default=0.0, help="MonST3R RPE translation RMSE reward weight (0 = disabled)")
    parser.add_argument("--rpe-rot-reward-weight", type=float, default=0.0, help="MonST3R RPE rotation median reward weight (0 = disabled)")
    parser.add_argument("--ate-rmse-scale", type=float, default=0.3, help="ATE reward = exp(-ate_rmse / scale)")
    parser.add_argument("--rpe-trans-scale", type=float, default=0.15, help="RPE trans reward = exp(-rpe_trans / scale)")
    parser.add_argument("--rpe-rot-scale", type=float, default=1.5, help="RPE rot reward = exp(-rpe_rot / scale)")
    parser.add_argument("--aesthetic-reward-weight", type=float, default=0.0, help="Aesthetic CLIP reward weight (0 = disabled)")
    parser.add_argument("--aesthetic-reward-scale", type=float, default=10.0, help="Aesthetic reward = score / scale")
    # GRPO rollout SDE (diversity probe). Default preserves ODE-only behavior.
    parser.add_argument("--grpo-sde", action=StoreBoolean, help="Enable SDE solver (with noise injection) in the GRPO rollout loop. Default False = pure ODE.")
    parser.add_argument("--grpo-eta", type=float, default=0.0, help="SDE eta (noise injection strength). Only active when --grpo-sde and step fraction < grpo-eta-cutoff. Sweet spot: 0.3.")
    parser.add_argument("--grpo-eta-cutoff", type=float, default=1.0, help="Fraction of sampling_steps during which SDE is active; beyond this fraction the solver reverts to ODE to avoid end-of-trajectory blur. Sweet spot: 0.3.")
    parser.add_argument("--grpo-sde-solver", type=str, default="flux", choices=["flux", "flow_grpo"], help="SDE solver for GRPO rollout. 'flux' = constant noise, needs cutoff<1. 'flow_grpo' = σ-adaptive noise, safe with cutoff=1.")
    return parser
