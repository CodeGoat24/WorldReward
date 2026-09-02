"""MonST3R-based trajectory ATE/RPE evaluator for eval-step videos.

Runs MonST3R inference + global alignment on a generated video, produces a
TUM-format predicted camera trajectory, and compares it against a GT
trajectory (given as [N, 4, 4] world-to-camera tensor).

Metric conventions:
  - Umeyama Sim3 alignment on camera-center point cloud
  - rpe_trans_rmse = RMSE of ‖Δt_est - Δt_gt‖ over adjacent frames (m)
  - rpe_rot_median = median of relative rotation angle over adjacent frames (deg)
  - ate_rmse       = RMSE of per-frame position distance after Sim3 align (m)

Any failure (short sequence, Umeyama degeneracy, OOM, missing weights, ...)
is caught and reported as NaN metrics so the eval loop never crashes.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Silence the vendored MonST3R's deprecated torch.cuda.amp.autocast warnings
# and similar known-harmless futurewarnings that would otherwise print 32x
# (once per rank) at import time.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*is deprecated.*",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
# MonST3R is not vendored; see README section 2.2. Prepended to sys.path so
# `from dust3r.xxx import ...` inside that code resolves.
_MONST3R_ROOT = Path(
    os.environ.get("MONST3R_SRC", "").strip()
    or Path(__file__).resolve().parent / "monst3r_lib"
)
MONST3R_AVAILABLE = _MONST3R_ROOT.is_dir()
if MONST3R_AVAILABLE and str(_MONST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONST3R_ROOT))

_MONST3R_MISSING_MSG = (
    f"The MonST3R code package was not found at {_MONST3R_ROOT}. Run\n"
    "  git clone https://github.com/Junyi42/monst3r "
    "fastvideo/rewards/computers/monst3r_lib\n"
    "(or set MONST3R_SRC to an existing checkout). To run without it, set "
    "eval_monst3r false and leave the ate_rmse/rpe_* reward weights unset."
)


def _find_monst3r_ckpt() -> str | None:
    env = os.environ.get("MONST3R_CHECKPOINT", "").strip()
    if env and (os.path.isdir(env) or os.path.isfile(env)):
        return env
    candidates = [
        _REPO_ROOT / "ckpt" / "Junyi42--MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
    ]
    for c in candidates:
        if (c / "config.json").is_file() or (c / "model.safetensors").is_file():
            return str(c)
    return None


def w2c_to_tum(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N,4,4] world-to-camera (numpy) -> (tum [N,7] xyz+wxyz, timestamps [N])."""
    c2w = np.linalg.inv(w2c)
    xyz = c2w[:, :3, 3]
    quats = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()  # xyzw
    quat_wxyz = np.stack(
        [quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]], axis=1
    )
    tum = np.column_stack([xyz, quat_wxyz])
    ts = np.arange(c2w.shape[0]).astype(float)
    return tum, ts


# ============================================================
# ATE / RPE
#   - Umeyama Sim3 alignment on camera-center point cloud
#   - rpe_trans = RMSE of ‖Δt_est - Δt_gt‖ over adjacent frames
#   - rpe_rot   = median of relative rotation angle (deg)
#   - ate       = RMSE of position distance after Sim3 alignment
# ============================================================

def _tum_to_c2w_mat(tum: np.ndarray) -> np.ndarray:
    """TUM [N,7] (xyz + wxyz quat) -> [N,3,4] c2w pose matrices."""
    xyz = tum[:, :3]
    q_wxyz = tum[:, 3:]
    q_xyzw = np.stack(
        [q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], axis=1
    )
    R = Rotation.from_quat(q_xyzw).as_matrix()
    mat = np.zeros((tum.shape[0], 3, 4), dtype=np.float64)
    mat[:, :, :3] = R
    mat[:, :, 3] = xyz
    return mat


def _umeyama_alignment(est_pts: np.ndarray, gt_pts: np.ndarray):
    """Sim3 alignment: returns (s, R, t) mapping est -> gt."""
    centroid_est = np.mean(est_pts, axis=0)
    centroid_gt = np.mean(gt_pts, axis=0)
    est_c = est_pts - centroid_est
    gt_c = gt_pts - centroid_gt
    denom = np.sum(est_c ** 2)
    s = np.sqrt(np.sum(gt_c ** 2) / max(denom, 1e-12))
    H = est_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_gt - s * (R @ centroid_est.T).T
    return s, R, t


def _align_sim3(est: np.ndarray, gt: np.ndarray):
    """Per-frame Sim3 align est to gt; requires equal frame count."""
    assert len(est) == len(gt), f"est({len(est)}) vs gt({len(gt)}) mismatched"
    s, R, t = _umeyama_alignment(est[:, :, 3], gt[:, :, 3])
    est_aligned = est.copy()
    est_aligned[:, :, 3] = s * (R @ est[:, :, 3].T).T + t
    est_aligned[:, :3, :3] = np.einsum("ij,njk->nik", R, est[:, :3, :3])
    return est_aligned, float(s)


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_diff = R1 @ R2.T
    tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.abs(np.degrees(np.arccos(tr))))


def _compute_rpe(est: np.ndarray, gt: np.ndarray, delta: int = 1):
    """RPE: (trans RMSE in metric units, rot median in degrees)."""
    trans_errs, rot_errs = [], []
    for i in range(len(gt) - delta):
        P_est_i = np.vstack([est[i],           [0, 0, 0, 1]])
        P_est_j = np.vstack([est[i + delta],   [0, 0, 0, 1]])
        dP_est = np.linalg.inv(P_est_i) @ P_est_j
        P_gt_i = np.vstack([gt[i],             [0, 0, 0, 1]])
        P_gt_j = np.vstack([gt[i + delta],     [0, 0, 0, 1]])
        dP_gt = np.linalg.inv(P_gt_i) @ P_gt_j
        trans_errs.append(float(np.linalg.norm(dP_est[:3, 3] - dP_gt[:3, 3])))
        rot_errs.append(_rotation_angle_deg(dP_est[:3, :3], dP_gt[:3, :3]))
    trans_rmse = float(np.sqrt(np.mean(np.asarray(trans_errs) ** 2)))
    rot_median = float(np.median(rot_errs))
    return trans_rmse, rot_median


def _compute_ate(est_aligned: np.ndarray, gt: np.ndarray) -> float:
    diffs = est_aligned[:, :, 3] - gt[:, :, 3]
    return float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))


class MonST3RTrajectoryEvaluator:
    """Lazy-loaded MonST3R evaluator for per-video ATE/RPE scoring."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        device: str | None = None,
        n_iter: int = 300,
        stride: int = 4,
        scene_graph_type: str = "swinstride-5-noncyclic",
        img_size: int = 512,
    ):
        self.ckpt_path = ckpt_path or _find_monst3r_ckpt()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_iter = int(n_iter)
        self.stride = int(stride)
        self.scene_graph_type = scene_graph_type
        self.img_size = int(img_size)
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not MONST3R_AVAILABLE:
            raise ImportError(_MONST3R_MISSING_MSG)
        if self.ckpt_path is None:
            raise FileNotFoundError(
                "MonST3R checkpoint not found. Set $MONST3R_CHECKPOINT or place "
                "weights under ckpt/Junyi42--MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt/"
            )
        from dust3r.model import AsymmetricCroCo3DStereo

        # The huggingface PyTorchModelHubMixin and dust3r's set_freeze both use
        # bare `print(...)` calls during load. On 32 ranks that spams 64 lines.
        # Let only local_rank 0 see them.
        is_local_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if is_local_rank0:
            logger.info("[monst3r] loading model from %s", self.ckpt_path)
            model = AsymmetricCroCo3DStereo.from_pretrained(self.ckpt_path).to(self.device)
        else:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                model = AsymmetricCroCo3DStereo.from_pretrained(self.ckpt_path).to(self.device)
        model.eval()
        self._model = model

    def unload(self) -> None:
        """Free GPU memory for MonST3R ViT-L. Safe to call when unloaded."""
        if self._model is None:
            return
        logger.info("[monst3r] unloading model to free GPU memory")
        try:
            self._model.to("cpu")
        except Exception as e:
            logger.warning("[monst3r] unload .to(cpu) failed: %s", e)
        self._model = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _dump_and_subsample_frames(
        self, video_path: str, tmp_dir: str
    ) -> tuple[list[str], list[int]]:
        """Dump only the strided frames to tmp_dir.

        Returns (kept_paths, kept_indices) where kept_indices are the absolute
        frame indices within the original video.
        """
        import cv2

        stride = max(1, self.stride)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        kept_paths: list[str] = []
        kept_indices: list[int] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                path = os.path.join(tmp_dir, f"{idx:05d}.jpg")
                cv2.imwrite(path, frame)
                kept_paths.append(path)
                kept_indices.append(idx)
            idx += 1
        cap.release()
        return kept_paths, kept_indices

    @torch.no_grad()
    def _predict_trajectory_from_files(
        self, kept_files: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        imgs = load_images(kept_files, size=self.img_size, verbose=False)
        n = len(imgs)
        if n < 2:
            raise RuntimeError(f"need at least 2 frames, got {n}")

        # clamp window size so scene graph doesn't exceed available frames
        sg_type = self.scene_graph_type
        parts = sg_type.split("-")
        if len(parts) >= 2 and parts[1].isdigit():
            max_winsize = max(1, math.ceil((n - 1) / 2))
            win = min(int(parts[1]), max_winsize)
            parts[1] = str(win)
            sg_type = "-".join(parts)

        pairs = make_pairs(imgs, scene_graph=sg_type, prefilter=None, symmetrize=True)
        output = inference(pairs, self._model, self.device, batch_size=1, verbose=False)

        with torch.enable_grad():
            if n > 2:
                scene = global_aligner(
                    output,
                    device=self.device,
                    mode=GlobalAlignerMode.PointCloudOptimizer,
                    verbose=False,
                    shared_focal=True,
                    flow_loss_weight=0.0,
                    flow_loss_fn="smooth_l1",
                    depth_regularize_weight=0.0,
                    num_total_iter=self.n_iter,
                    temporal_smoothing_weight=0.01,
                    translation_weight=1.0,
                    motion_mask_thre=0.35,
                    flow_loss_start_epoch=0.1,
                    flow_loss_thre=25,
                    use_self_mask=True,
                    sam2_mask_refine=False,
                    empty_cache=False,
                    pxl_thre=50,
                )
                _ = scene.compute_global_alignment(
                    init="mst", niter=self.n_iter, schedule="linear", lr=0.01
                )
            else:
                scene = global_aligner(
                    output, device=self.device, mode=GlobalAlignerMode.PairViewer,
                    verbose=False,
                )

        pred_tum, pred_ts = scene.get_tum_poses()
        pred_tum = np.asarray(pred_tum)
        pred_ts = np.asarray(pred_ts)

        # Free the optimizer graph + intermediate tensors so repeated eval
        # calls don't accumulate GPU memory across samples.
        del scene, output, pairs, imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pred_tum, pred_ts

    def predict_trajectory(
        self, video_path: str
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Predict per-frame camera trajectory for a video.

        Returns (pred_tum [N,7], timestamps [N], kept_indices) where kept_indices
        are the absolute frame indices within the original video that were kept
        after stride subsampling (caller can align GT with these indices).
        """
        self._ensure_loaded()
        with tempfile.TemporaryDirectory(prefix="monst3r_eval_") as tmp:
            kept_files, kept_indices = self._dump_and_subsample_frames(
                video_path, tmp
            )
            pred_tum, pred_ts = self._predict_trajectory_from_files(kept_files)
        return pred_tum, pred_ts, kept_indices

    def compute_ate_rpe(
        self, video_path: str, gt_w2c: torch.Tensor | np.ndarray
    ) -> dict:
        """Compute ATE/RPE for a video against a GT w2c trajectory.

        Args:
            video_path: path to generated mp4 (decoded from ``latent_t`` latents,
                containing ``(latent_t - 1) * 4 + 1`` pixel frames for HunyuanVAE).
            gt_w2c: [M, 4, 4] world-to-camera matrices indexed along the **latent
                axis** — i.e. one pose per latent frame, not per pixel frame.
                Typically M = training ``window_frames`` (e.g. 64). Pred kept
                pixel-frame indices are mapped back to latent indices via
                ``kept_idx // stride`` (stride == VAE temporal compression == 4),
                yielding a clean 1:1 alignment with GT.

        Returns:
            dict with float keys "ate_rmse", "rpe_trans_rmse", "rpe_rot_median";
            NaN on failure.
        """
        nan = float("nan")
        fail = {"ate_rmse": nan, "rpe_trans_rmse": nan, "rpe_rot_median": nan}

        try:
            pred_tum, pred_ts, kept_indices = self.predict_trajectory(video_path)
        except Exception as e:
            logger.warning("[monst3r] prediction failed for %s: %s", video_path, e)
            return fail

        try:
            if isinstance(gt_w2c, torch.Tensor):
                gt_w2c_np = gt_w2c.detach().to(torch.float32).cpu().numpy()
            else:
                gt_w2c_np = np.asarray(gt_w2c, dtype=np.float32)
            if gt_w2c_np.ndim != 3 or gt_w2c_np.shape[-2:] != (4, 4):
                raise ValueError(
                    f"gt_w2c must be [M,4,4], got {gt_w2c_np.shape}"
                )
            # Map pixel-frame kept_indices -> latent-axis indices.
            # HunyuanVideo VAE: pixel_frame f = latent_t * 4 (approx; first
            # latent covers frame 0, subsequent latents expand 4 px frames each).
            # kept_indices are pixel-frame indices on the generated video.
            # Since `_dump_and_subsample_frames` uses stride=self.stride, and
            # the training-time eval pipeline has stride==4 == VAE compression,
            # kept_indices // 4 gives the corresponding latent index, which is
            # the axis along which gt_w2c is defined (one w2c per latent frame).
            M = gt_w2c_np.shape[0]
            idx_raw = np.asarray(kept_indices, dtype=np.int64)
            idx_latent = idx_raw // max(1, self.stride)
            if idx_latent.max(initial=-1) >= M:
                logger.warning(
                    "[monst3r] GT has %d latent poses but mapped indices reach "
                    "%d; truncating both pred and gt to the overlap.",
                    M, int(idx_latent.max()),
                )
            keep_mask = idx_latent < M
            if not keep_mask.all():
                pred_tum = pred_tum[keep_mask]
                idx_latent = idx_latent[keep_mask]
            gt_w2c_subset = gt_w2c_np[idx_latent]
            # Convert both pred (already c2w via TUM) and gt (w2c) into [N,3,4]
            # c2w pose matrices, Sim3-align, then compute ATE/RPE per
            # the metric conventions documented in this module's docstring.
            pred_c2w = _tum_to_c2w_mat(pred_tum)
            gt_c2w = np.linalg.inv(gt_w2c_subset.astype(np.float64))[:, :3, :]
        except Exception as e:
            logger.warning("[monst3r] GT preparation failed for %s: %s", video_path, e)
            return fail

        try:
            est_aligned, _scale = _align_sim3(pred_c2w, gt_c2w)
            ate = _compute_ate(est_aligned, gt_c2w)
            rpe_trans, rpe_rot = _compute_rpe(est_aligned, gt_c2w, delta=1)
            return {
                "ate_rmse": float(ate),
                "rpe_trans_rmse": float(rpe_trans),
                "rpe_rot_median": float(rpe_rot),
            }
        except Exception as e:
            logger.warning(
                "[monst3r] ATE/RPE computation failed for %s: %s", video_path, e
            )
            return fail


# ---------------------------------------------------------------------------
# BaseReward wrapper
# ---------------------------------------------------------------------------

import math

from fastvideo.rewards.base import PerCandidateReward, RewardContext, ScoreResult
from fastvideo.rewards.ckpts import CKPTS


class MonST3RReward(PerCandidateReward):
    """ATE/RPE reward via MonST3R Sim3-aligned camera centers.

    Emits three outputs (raw-metric magnitudes, NEGATED so larger=better):
      ate_rmse    — -RMSE of per-frame position error after Sim3 alignment
      rpe_trans   — -RMSE of ‖Δt_est - Δt_gt‖ over adjacent frames
      rpe_rot     — -median relative rotation angle (deg) over adjacent frames
    No squashing to [0,1]: advantages are computed per-output via z-score
    (scale-invariant) then weighted sum, so linear_scale / exp_decay only
    distort the dynamic range.
    """
    NAME = "monst3r"
    OUTPUTS = ("ate_rmse", "rpe_trans", "rpe_rot")
    _METRIC_KEYS = {
        "ate_rmse": "ate_rmse",
        "rpe_trans": "rpe_trans_rmse",
        "rpe_rot": "rpe_rot_median",
    }
    # NaN fallback: pick a "mediocre-bad" magnitude so failed scorers
    # don't masquerade as outlier-good candidates after negation.
    _NAN_FALLBACK = {
        "ate_rmse": 1.0,
        "rpe_trans": 1.0,
        "rpe_rot": 10.0,
    }

    def __init__(self, training_args):
        super().__init__(training_args)
        self._evaluator: MonST3RTrajectoryEvaluator | None = None

    def load(self, device) -> None:
        if self._evaluator is not None:
            return
        self._evaluator = MonST3RTrajectoryEvaluator(
            ckpt_path=CKPTS.monst3r,
            device=str(device),
            n_iter=int(getattr(self.training_args, "eval_monst3r_n_iter", 300)),
            stride=int(getattr(self.training_args, "eval_monst3r_stride", 4)),
        )

    def unload(self) -> None:
        if self._evaluator is not None:
            self._evaluator.unload()

    @property
    def evaluator(self) -> "MonST3RTrajectoryEvaluator":
        assert self._evaluator is not None, "MonST3RReward not loaded"
        return self._evaluator

    @property
    def enabled_for_eval(self) -> bool:
        return (
            self.enabled_for_training
            or bool(getattr(self.training_args, "eval_monst3r", False))
        )

    def score_one(self, ctx: RewardContext) -> ScoreResult:
        assert self._evaluator is not None, "MonST3RReward not loaded"
        if ctx.gt_w2c is None:
            return self.neutral_result()
        metrics = self._evaluator.compute_ate_rpe(
            video_path=ctx.video_path, gt_w2c=ctx.gt_w2c
        )
        scores: dict[str, float] = {}
        for out in self.OUTPUTS:
            raw = metrics.get(self._METRIC_KEYS[out], float("nan"))
            if not math.isfinite(raw):
                raw = self._NAN_FALLBACK[out]
            # Negate so "better candidate → larger reward" (ate/rpe are
            # all lower-is-better distances / angles).
            scores[out] = -float(raw)
        return ScoreResult(scores=scores, metrics=dict(metrics))
