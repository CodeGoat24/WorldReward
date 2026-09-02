"""HPSv3-based image quality & prompt-alignment scorer.

Produces three scalar metrics from a generated video + caption:
  hpsv3_acc                 — text-video alignment (caption vs frame)
  hpsv3_quality_acc         — quality prompt alignment (constant prompt)
  hpsv3_quality_drift_score — least-squares slope of quality over the
                              generated frames (higher = holds up better)

Self-contained: does not depend on WorldMirror. Frame extraction uses
the shared helpers in `fastvideo.rewards.utils.frame_utils`.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import tempfile
import uuid

import torch
from hpsv3 import HPSv3RewardInferencer

from fastvideo.rewards.utils.frame_utils import (
    cleanup_frame_paths,
    extract_frames_from_video,
    new_instance_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local checkpoint discovery
# ---------------------------------------------------------------------------

def _find_local_hpsv3_checkpoint(cache_dir=None):
    env_path = os.environ.get("HPSV3_CHECKPOINT", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    cwd = os.getcwd()
    candidate_paths = [
        os.path.join(cwd, "ckpt", "HPSv3.safetensors"),
        os.path.join(cwd, "ckpt", "HPSv3", "HPSv3.safetensors"),
        os.path.join(cwd, "ckpt", "hpsv3", "HPSv3.safetensors"),
        os.path.join(cwd, "ckpt", "MizzenAI--HPSv3", "HPSv3.safetensors"),
    ]
    if cache_dir:
        candidate_paths.extend([
            os.path.join(cache_dir, "HPSv3.safetensors"),
            os.path.join(cache_dir, "HPSv3", "HPSv3.safetensors"),
            os.path.join(cache_dir, "MizzenAI--HPSv3", "HPSv3.safetensors"),
        ])
    for candidate in candidate_paths:
        if os.path.isfile(candidate):
            return candidate

    search_roots = [os.path.join(cwd, "ckpt")]
    if cache_dir:
        search_roots.append(cache_dir)
    search_roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

    patterns = [
        "**/HPSv3.safetensors",
        "**/models--MizzenAI--HPSv3/**/HPSv3.safetensors",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            for match in glob.glob(os.path.join(root, pattern), recursive=True):
                if os.path.isfile(match):
                    return match
    return None


def _find_local_qwen2_vl_dir(cache_dir=None):
    env_path = os.environ.get("QWEN2_VL_7B_INSTRUCT_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path
    cwd = os.getcwd()
    candidate_dirs = [
        os.path.join(cwd, "ckpt", "Qwen2-VL-7B-Instruct"),
        os.path.join(cwd, "ckpt", "Qwen", "Qwen2-VL-7B-Instruct"),
    ]
    if cache_dir:
        candidate_dirs.extend([
            os.path.join(cache_dir, "Qwen2-VL-7B-Instruct"),
            os.path.join(cache_dir, "Qwen", "Qwen2-VL-7B-Instruct"),
        ])
    for candidate in candidate_dirs:
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def _build_local_hpsv3_config(local_model_dir):
    import hpsv3

    hpsv3_pkg_dir = os.path.dirname(hpsv3.__file__)
    src_config_path = os.path.join(hpsv3_pkg_dir, "config", "HPSv3_7B.yaml")
    rank = os.environ.get("RANK", "0")
    pid = os.getpid()
    dst_config_path = os.path.join(
        tempfile.gettempdir(),
        f"hpsv3_7b_local_rank{rank}_pid{pid}_{uuid.uuid4().hex[:8]}.yaml",
    )
    with open(src_config_path, "r", encoding="utf-8") as f:
        config_text = f.read()
    replaced_text = config_text.replace(
        'model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"',
        f'model_name_or_path: "{local_model_dir}"',
    )
    if replaced_text == config_text:
        raise ValueError(
            "Failed to patch HPSv3 config: expected model_name_or_path entry not found "
            f"in template {src_config_path}"
        )
    with open(dst_config_path, "w", encoding="utf-8") as f:
        f.write(replaced_text)
    return dst_config_path


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

_QUALITY_PROMPT = "A high-quality, ultra-detailed and well-structured image."


def _quality_slope(q: torch.Tensor) -> float:
    """Least-squares slope of `q` against frame index, as a plain float.

    Used as the temporal-stability reward: negative = quality decaying across
    the generated chunk, ~0 = holding steady, positive = improving. Only the
    within-group ordering matters (GRPO z-scores each reward), so the units
    are irrelevant.

    Returns 0.0 for fewer than 2 points instead of NaN — a NaN reward would
    propagate through the group mean/std and destroy every candidate's
    advantage, not just its own.
    """
    m = q.numel()
    if m < 2:
        return 0.0
    t = torch.arange(m, device=q.device, dtype=torch.float32)
    t = t - t.mean()
    slope = torch.dot(t, q.float() - q.float().mean()) / torch.dot(t, t)
    return float(slope.item())


class HPSv3Scorer:
    """Wraps HPSv3RewardInferencer for per-video alignment / quality scoring."""

    def __init__(self, device: str | None = None, cache_dir: str | None = None) -> None:
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint_path = _find_local_hpsv3_checkpoint(cache_dir=cache_dir)
        qwen_dir = _find_local_qwen2_vl_dir(cache_dir=cache_dir)
        if checkpoint_path:
            logger.info(f"Loading HPSv3 checkpoint from local path: {checkpoint_path}")
            if qwen_dir:
                logger.info(f"Loading HPSv3 base model from local path: {qwen_dir}")
                local_config = _build_local_hpsv3_config(qwen_dir)
                self.model = HPSv3RewardInferencer(
                    config_path=local_config,
                    checkpoint_path=checkpoint_path,
                    device=self.device,
                )
            else:
                self.model = HPSv3RewardInferencer(
                    checkpoint_path=checkpoint_path, device=self.device,
                )
        else:
            logger.info("Loading HPSv3 checkpoint from remote source")
            self.model = HPSv3RewardInferencer(device=self.device)

        self.temp_dir = tempfile.mkdtemp(prefix="hpsv3_")
        self.instance_id = new_instance_id()

        # HPSv3RewardInferencer builds the Qwen2-VL-7B base model directly on
        # self.device, so it starts out resident.
        self._on_gpu = True
        # Park it immediately. This scorer is only needed inside score() /
        # score_eval() -- a few seconds per candidate -- whereas the ~16 GB of
        # Qwen2-VL-7B weights would otherwise sit on every one of the 64
        # training GPUs for the whole run, including the denoising loop, which
        # is where the per-step memory peak actually happens.
        self._park_on_cpu()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def cleanup(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ------------------------------------------------------------
    # GPU residency. The base model is parked in host RAM between scoring
    # calls (mirrors WorldMirrorScorer's `_on_gpu` parking), so a reload is
    # a pure H2D copy with no disk reads.
    # ------------------------------------------------------------
    def _ensure_on_gpu(self) -> None:
        """Bring the base model onto self.device. Idempotent."""
        if self._on_gpu or self.model is None or not hasattr(self.model, "model"):
            return
        self.model.model.to(self.device)
        self._on_gpu = True

    def _park_on_cpu(self) -> None:
        """Move the base model to host RAM and release its GPU blocks. Idempotent."""
        if not self._on_gpu or self.model is None or not hasattr(self.model, "model"):
            return
        self.model.model.to("cpu")
        self._on_gpu = False
        torch.cuda.empty_cache()

    # Back-compat entry points. HPSv3Reward.unload() calls offload_to_cpu();
    # keeping them as thin aliases means `_on_gpu` can never drift out of sync
    # with where the weights actually are.
    def offload_to_cpu(self):
        self._park_on_cpu()

    def reload_to_gpu(self):
        self._ensure_on_gpu()

    # ------------------------------------------------------------
    # Internal: one HPSv3 forward pass against a prompt list
    # ------------------------------------------------------------
    def _reward(self, images: list[str], prompts: list[str]) -> torch.Tensor:
        score = self.model.reward(images, prompts)
        if score.ndim == 2:
            score = score[:, 0]
        return score.cpu()

    # ------------------------------------------------------------
    # Main scoring entry points
    # ------------------------------------------------------------
    @torch.no_grad()
    def score(
        self,
        video_path: str,
        caption: str,
        interval: int = 1,
        update_latent_num: int = 4,
        score_caption: bool = True,
    ) -> dict:
        """Score one video → {hpsv3_acc, hpsv3_quality_acc, hpsv3_quality_drift_score}.

        Scores every 2nd frame of the newly generated chunk with the fixed
        quality prompt. `hpsv3_acc` (caption alignment) is only computed when
        `score_caption` is set, because it costs as many HPSv3 forwards as the
        quality term and is usually carried at weight 0.
        """
        expected_frame_num = (update_latent_num * 4) + 1
        # max_frames must stay None: extract_frames_from_video() fills from the
        # START of the video and breaks once max_frames is reached, so passing
        # expected_frame_num here would return the FIRST 17 frames and make the
        # trailing [-expected_frame_num:] slice a no-op. Those leading frames
        # are the ODE-sampled context (shared by every GRPO candidate), not the
        # newly generated chunk, so quality/drift came out identical across a
        # candidate group and their z-scored advantages were exactly zero.
        # Extract everything and slice the tail, matching WorldMirrorScorer.
        frame_paths = extract_frames_from_video(
            video_path, self.temp_dir, self.instance_id,
            interval=interval, max_frames=None,
        )
        images = frame_paths[-expected_frame_num:]
        # Score 12 of the 17 generated frames. Both quality and drift are
        # estimated from these samples and 4 (the original images[3::4]) is far
        # too few. Measured on 656 real candidates via odd/even split-half:
        #
        #   frames   quality reliability   drift reliability
        #      4           0.680                0.603
        #      8           0.791                0.830
        #     12           0.851                0.938
        #     17           0.895                0.994
        #
        # 12 costs 12 HPSv3 forwards/candidate vs the original 8 (4 frames x
        # both prompts), i.e. only 1.5x, because skipping the unweighted
        # caption term above pays for most of the extra frames.
        idx = torch.linspace(0, len(images) - 1, 12).round().long().tolist()
        hps_images = [images[i] for i in sorted(set(idx))]

        # Weights live on the host between calls (see _park_on_cpu); page them
        # in only for the forwards themselves. try/finally so a failure inside
        # _reward cannot strand 16 GB on the GPU for the rest of the run.
        # _reward() returns .cpu() tensors, so nothing below this block holds a
        # reference into the parked model.
        self._ensure_on_gpu()
        try:
            hps_quality_score = self._reward(
                hps_images, [_QUALITY_PROMPT] * len(hps_images)
            )
            # Skipping this halves the HPSv3 cost when hpsv3_reward_weight == 0.
            hps_mean = (
                float(self._reward(
                    hps_images, [caption] * len(hps_images)
                ).mean().item())
                if score_caption
                else 0.0
            )
        finally:
            self._park_on_cpu()
        # Temporal-stability reward: least-squares slope of quality over the
        # generated frames. Larger (less negative) = quality holds up better.
        drift_score = _quality_slope(hps_quality_score.flatten())

        cleanup_frame_paths(frame_paths)
        return {
            "hpsv3_acc": hps_mean,
            "hpsv3_quality_acc": float(hps_quality_score.mean().item()),
            "hpsv3_quality_drift_score": drift_score,
        }

    @torch.no_grad()
    def score_eval(
        self,
        video_path: str,
        caption: str,
        interval: int = 1,
        latent_num: int = 64,
    ) -> dict:
        """Per-chunk evaluation → {hps_acc, hps_quality_acc, hps_drift_score}.

        HPSv3 is run only on the last frame of each 4-frame chunk, matching
        the legacy eval behaviour.
        """
        images = extract_frames_from_video(
            video_path, self.temp_dir, self.instance_id,
            interval=interval, max_frames=(latent_num * 4) - 3,
        )
        num_frames = len(images)
        chunk_size = 4
        num_chunks = num_frames // chunk_size

        hps_acc_list: list[float] = []
        hps_quality_acc_list: list[float] = []
        # One page-in for the whole chunk sweep rather than one per chunk.
        self._ensure_on_gpu()
        try:
            for chunk_idx in range(num_chunks):
                chunk_end = ((chunk_idx + 1) * chunk_size) + 1
                last_in_chunk = images[:chunk_end][-chunk_size:][-1]
                hps_acc_list.append(
                    float(self._reward([last_in_chunk], [caption]).mean().item())
                )
                hps_quality_acc_list.append(
                    float(self._reward([last_in_chunk], [_QUALITY_PROMPT]).mean().item())
                )
        finally:
            self._park_on_cpu()

        # Same definition as score(): least-squares slope of quality across the
        # generated chunk, so the metric reported here is exactly the quantity
        # training optimizes. The old anchor-based -|q - first_frame_quality|
        # is gone — it was degenerate under GRPO (see score()).
        drift_score = _quality_slope(torch.tensor(hps_quality_acc_list))
        cleanup_frame_paths(images)
        return {
            "hps_acc": hps_acc_list,
            "hps_quality_acc": hps_quality_acc_list,
            "hps_drift_score": drift_score,
        }
