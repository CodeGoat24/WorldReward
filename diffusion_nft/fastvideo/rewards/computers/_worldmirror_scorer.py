"""WorldMirror / DAv3 camera-trajectory-based action accuracy scorer.

Produces two scalar metrics from a generated video + GT action labels:
  action_acc      — discrete-action classification accuracy (coarse grid)
  fine_action_acc — mean of per-axis (trans/rotate) label match rate

Self-contained: does not depend on HPSv3. Frame extraction uses the
shared helpers in `fastvideo.rewards.utils.frame_utils`.
"""
from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from fastvideo.rewards.utils.frame_utils import (
    cleanup_frame_paths,
    extract_frames_from_video,
    new_instance_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAv3 import shim (optional)
# ---------------------------------------------------------------------------

def _try_import_depth_anything_3():
    candidate_roots = []
    env_src = os.environ.get("DEPTH_ANYTHING_3_SRC", "").strip()
    if env_src:
        env_path = Path(env_src)
        if env_path.name == "depth_anything_3":
            candidate_roots.append(str(env_path.parent))
        else:
            candidate_roots.append(str(env_path))

    cwd = Path(os.getcwd())
    candidate_roots.extend(
        [
            str(cwd),
            str(cwd / "fastvideo" / "rewards" / "computers"),
            str(cwd / "DepthAnythingV3" / "src"),
        ]
    )
    for root in candidate_roots:
        if root and os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
    try:
        return importlib.import_module("depth_anything_3.api").DepthAnything3
    except Exception:
        return None


DepthAnything3 = _try_import_depth_anything_3()
DEPTH_ANYTHING_3_AVAILABLE = DepthAnything3 is not None


# HunyuanWorld-Mirror import shim. Not vendored; see README section 2.2.

def _try_import_world_mirror():
    """Import WorldMirror if a checkout is reachable, else return (None, None)."""
    env_src = os.environ.get("WORLD_MIRROR_SRC", "").strip()
    if env_src and os.path.isdir(env_src) and env_src not in sys.path:
        sys.path.insert(0, env_src)
    for module_path in (
        "fastvideo.rewards.computers.HunyuanWorldMirror",
        "HunyuanWorldMirror",
    ):
        try:
            model_cls = importlib.import_module(module_path).WorldMirror
            preprocess = importlib.import_module(
                f"{module_path}.src.utils.inference_utils"
            ).extract_load_and_preprocess_images
            return model_cls, preprocess
        except Exception:
            continue
    return None, None


WorldMirror, extract_load_and_preprocess_images = _try_import_world_mirror()
WORLD_MIRROR_AVAILABLE = WorldMirror is not None

_WORLD_MIRROR_MISSING_MSG = (
    "camera_estimator='worldmirror' needs the HunyuanWorld-Mirror code "
    "package, which is not bundled. Run\n"
    "  git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror "
    "fastvideo/rewards/computers/HunyuanWorldMirror\n"
    "(or set WORLD_MIRROR_SRC to an existing checkout), or switch to "
    "camera_estimator='dav3'."
)


def _find_local_da3_model_dir(cache_dir=None):
    env_path = os.environ.get("DEPTH_ANYTHING_3_MODEL_PATH", "").strip()
    if env_path and os.path.isfile(os.path.join(env_path, "config.json")):
        return env_path
    cwd = os.getcwd()
    candidate_dirs = [
        os.path.join(cwd, "ckpt", "depth-anything--DA3-GIANT-1.1"),
        os.path.join(cwd, "ckpt", "DA3-GIANT-1.1"),
        os.path.join(cwd, "ckpt", "depth-anything", "DA3-GIANT-1.1"),
    ]
    if cache_dir:
        candidate_dirs.extend([
            os.path.join(cache_dir, "depth-anything--DA3-GIANT-1.1"),
            os.path.join(cache_dir, "DA3-GIANT-1.1"),
        ])
    for candidate in candidate_dirs:
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


# ---------------------------------------------------------------------------
# pose → discrete action
# ---------------------------------------------------------------------------

def _normalize_predicted_camera_pose(
    extrinsics, expected_frame_num: int | None = None
) -> torch.Tensor:
    """Normalize model-specific extrinsics to a [T, 4, 4] w2c tensor."""
    camera_pose = torch.as_tensor(extrinsics, dtype=torch.float32)

    while camera_pose.ndim > 3 and camera_pose.shape[0] == 1:
        camera_pose = camera_pose.squeeze(0)
    while camera_pose.ndim > 3 and camera_pose.shape[1] == 1:
        camera_pose = camera_pose.squeeze(1)
    if camera_pose.ndim != 3:
        raise ValueError(
            f"Unsupported camera pose shape: {tuple(camera_pose.shape)}"
        )

    if camera_pose.shape[-2:] == (3, 4):
        bottom_row = (
            torch.tensor([0, 0, 0, 1], dtype=camera_pose.dtype)
            .view(1, 1, 4)
            .expand(camera_pose.shape[0], 1, 4)
        )
        camera_pose = torch.cat([camera_pose, bottom_row], dim=1)
    elif camera_pose.shape[-2:] != (4, 4):
        raise ValueError(
            f"Unsupported extrinsics matrix shape: {tuple(camera_pose.shape[-2:])}"
        )

    if expected_frame_num is not None:
        if camera_pose.shape[0] < expected_frame_num:
            raise ValueError(
                "Predicted camera pose count is smaller than extracted frame count: "
                f"{camera_pose.shape[0]} < {expected_frame_num}"
            )
        if camera_pose.shape[0] != expected_frame_num:
            logger.warning(
                "Predicted camera pose count (%s) != extracted frame count (%s); "
                "using the last %s poses for reward computation.",
                camera_pose.shape[0],
                expected_frame_num,
                expected_frame_num,
            )
            camera_pose = camera_pose[-expected_frame_num:]
    return camera_pose


def _camera_pose_to_discrete_action(
    camera_poses: torch.Tensor,
    move_norm_valid: float,
    rot_threshold: float,
) -> torch.Tensor:
    """Quantize [N,4,4] w2c pose sequence into [N] discrete action labels.

    Label encoding matches camera_dataset.py: `trans_one_label * 9 + rotate_one_label`.
    """
    N = camera_poses.shape[0]
    c2ws = torch.inverse(camera_poses)

    C_inv = torch.inverse(c2ws[:-1])
    relative_c2w = torch.zeros_like(c2ws)
    relative_c2w[0] = c2ws[0]
    relative_c2w[1:] = torch.bmm(C_inv, c2ws[1:])

    trans_one_hot = torch.zeros((N, 4), dtype=torch.int32, device=camera_poses.device)
    rotate_one_hot = torch.zeros((N, 4), dtype=torch.int32, device=camera_poses.device)

    for i in range(1, N):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = torch.norm(move_dirs)
        if move_norms > move_norm_valid:
            move_norm_dirs = move_dirs / move_norms
            move_norm_dirs = torch.clamp(move_norm_dirs, -1.0, 1.0)
            angles_rad = torch.acos(move_norm_dirs)
            trans_angles_deg = angles_rad * (180.0 / torch.pi)
            if trans_angles_deg[2] < 60:
                trans_one_hot[i, 0] = 1
            elif trans_angles_deg[2] > 120:
                trans_one_hot[i, 1] = 1
            if trans_angles_deg[0] < 60:
                trans_one_hot[i, 2] = 1
            elif trans_angles_deg[0] > 120:
                trans_one_hot[i, 3] = 1

        R_rel = relative_c2w[i, :3, :3]
        sy = torch.sqrt(R_rel[0, 0] ** 2 + R_rel[1, 0] ** 2)
        if sy > 1e-6:
            x = torch.atan2(R_rel[2, 1], R_rel[2, 2])
            y = torch.atan2(-R_rel[2, 0], sy)
            z = torch.atan2(R_rel[1, 0], R_rel[0, 0])
        else:
            x = torch.atan2(-R_rel[1, 2], R_rel[1, 1])
            y = torch.atan2(-R_rel[2, 0], sy)
            z = torch.tensor(0.0, device=camera_poses.device)
        rot_angles_deg = torch.stack([x, y, z]) * (180.0 / torch.pi)
        if rot_angles_deg[1] > rot_threshold:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -rot_threshold:
            rotate_one_hot[i, 1] = 1
        if rot_angles_deg[0] > rot_threshold:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -rot_threshold:
            rotate_one_hot[i, 3] = 1

    mapping = {
        (0, 0, 0, 0): 0,
        (1, 0, 0, 0): 1,
        (0, 1, 0, 0): 2,
        (0, 0, 1, 0): 3,
        (0, 0, 0, 1): 4,
        (1, 0, 1, 0): 5,
        (1, 0, 0, 1): 6,
        (0, 1, 1, 0): 7,
        (0, 1, 0, 1): 8,
    }
    trans_one_label = torch.zeros(N, dtype=torch.long, device=camera_poses.device)
    rotate_one_label = torch.zeros(N, dtype=torch.long, device=camera_poses.device)
    for i in range(N):
        trans_one_label[i] = mapping.get(tuple(trans_one_hot[i].tolist()), 0)
        rotate_one_label[i] = mapping.get(tuple(rotate_one_hot[i].tolist()), 0)
    return trans_one_label * 9 + rotate_one_label


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class WorldMirrorScorer:
    """Camera-trajectory → discrete-action accuracy.

    Supports two camera estimators:
      worldmirror — Tencent HunyuanWorldMirror (default for GRPO training)
      dav3        — DepthAnythingV3 GIANT

    Park model on CPU by default; move to GPU around inference so vLLM
    wake_up in the same colocated setup has memory headroom.
    """

    def __init__(
        self,
        device: str | None = None,
        camera_estimator: str = "worldmirror",
        cache_dir: str | None = None,
    ) -> None:
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.camera_estimator = camera_estimator.lower()
        self.worldmirror_model = None
        self.dav3_model = None
        self._on_gpu = False

        if self.camera_estimator == "worldmirror":
            if not WORLD_MIRROR_AVAILABLE:
                raise ImportError(_WORLD_MIRROR_MISSING_MSG)
            local_dir = os.path.join(
                os.getcwd(), "ckpt", "tencent--HunyuanWorld-Mirror"
            )
            if os.path.exists(os.path.join(local_dir, "config.json")):
                model_path = local_dir
            else:
                model_path = "tencent/HunyuanWorld-Mirror"
            if int(os.environ.get("RANK", "0")) == 0:
                logger.info(f"Loading WorldMirror model from: {model_path}")
            self.worldmirror_model = WorldMirror.from_pretrained(
                model_path, cache_dir=cache_dir
            )
            self.worldmirror_model.eval()
            self.worldmirror_model.enable_gs = False
            self.worldmirror_model.to("cpu")
        elif self.camera_estimator == "dav3":
            da3_path = _find_local_da3_model_dir(cache_dir=cache_dir)
            if not DEPTH_ANYTHING_3_AVAILABLE:
                raise ImportError(
                    "camera_estimator='dav3' needs the `depth_anything_3` code "
                    "package, which is not bundled with this repository. "
                    f"Local DA3 weights: {da3_path or 'not found'} — weights "
                    "alone are not enough. Clone the upstream source:\n"
                    "  git clone https://github.com/ByteDance-Seed/"
                    "Depth-Anything-3 DepthAnythingV3\n"
                    "then either leave it at ./DepthAnythingV3/src, move "
                    "src/depth_anything_3 to "
                    "./fastvideo/rewards/computers/depth_anything_3, or point "
                    "DEPTH_ANYTHING_3_SRC at the directory that contains it. "
                    "Alternatively use camera_estimator='worldmirror'."
                )
            if da3_path is None:
                da3_path = "depth-anything/DA3-GIANT-1.1"
            logger.info(f"Loading DepthAnything3 model from: {da3_path}")
            self.dav3_model = DepthAnything3.from_pretrained(
                da3_path, cache_dir=cache_dir
            ).to(self.device)
            self._on_gpu = True
        else:
            raise ValueError(
                f"Unsupported camera_estimator: {camera_estimator}. Choose 'dav3' or 'worldmirror'"
            )

        self.temp_dir = tempfile.mkdtemp(prefix="worldmirror_")
        self.instance_id = new_instance_id()

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def cleanup(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def offload_to_cpu(self):
        for m in (self.worldmirror_model, self.dav3_model):
            if m is not None:
                m.to("cpu")
        self._on_gpu = False
        torch.cuda.empty_cache()

    def reload_to_gpu(self):
        for m in (self.worldmirror_model, self.dav3_model):
            if m is not None:
                m.to(self.device)
        self._on_gpu = True

    # ------------------------------------------------------------
    # Frame extraction + prediction
    # ------------------------------------------------------------
    def _extract(self, video_path: str, interval: int, max_frames: int):
        return extract_frames_from_video(
            video_path, self.temp_dir, self.instance_id,
            interval=interval, max_frames=max_frames,
        )

    @torch.no_grad()
    def _predict_dav3(self, images: list[str]):
        predictions = self.dav3_model.inference(images)
        return predictions

    @torch.no_grad()
    def _predict_worldmirror(self, images: list[str], fps: int = 1, target_size: int = 518):
        # WorldMirror wants a directory of images.
        tmp_input = os.path.join(self.temp_dir, f"wm_in_{self.instance_id}")
        if os.path.exists(tmp_input):
            shutil.rmtree(tmp_input)
        os.makedirs(tmp_input)
        for idx, p in enumerate(images):
            shutil.copy(p, os.path.join(tmp_input, f"{idx:05d}.jpg"))

        inputs = {
            "img": extract_load_and_preprocess_images(
                Path(tmp_input), fps=fps, target_size=target_size
            ).to(self.device)
        }
        cond_flags = [0, 0, 0]

        try:
            if not self._on_gpu:
                self.worldmirror_model.to(self.device)
                self._on_gpu = True
            predictions = self.worldmirror_model(views=inputs, cond_flags=cond_flags)
        finally:
            # Park back on CPU so vLLM wake_up next phase has headroom.
            if self._on_gpu and self.worldmirror_model is not None:
                try:
                    self.worldmirror_model.to("cpu")
                    self._on_gpu = False
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning("WorldMirror .to(cpu) after eval failed: %s", e)

        camera_poses = predictions["camera_poses"][0]  # [S,4,4]
        w2c_poses = torch.inverse(camera_poses)
        extrinsics = w2c_poses[:, :3, :].cpu().numpy()  # [S,3,4]

        class _Pred:
            def __init__(self, ex):
                self.extrinsics = ex
        return _Pred(extrinsics)

    def process_video(
        self, video_path: str, interval: int = 4, max_frames: int | None = None,
        last_frames: int = 16,
    ):
        """Returns (first_frame_path, kept_image_paths, predictions, all_paths).

        `all_paths` is every JPEG written to temp_dir. Callers must pass it to
        cleanup_frame_paths(): `images` is only the trailing slice, so cleaning
        that alone leaks the leading frames (~150 JPEGs / ~6 MB per call).
        """
        frame_paths = self._extract(video_path, interval, max_frames)
        first_frame = frame_paths[0]
        images = frame_paths[-last_frames:]
        if self.camera_estimator == "worldmirror":
            predictions = self._predict_worldmirror(images)
        else:
            predictions = self._predict_dav3(images)
        return first_frame, images, predictions, frame_paths

    # ------------------------------------------------------------
    # Main scoring entry points
    # ------------------------------------------------------------
    @torch.no_grad()
    def score(
        self,
        video_path: str,
        gt_action: torch.Tensor,
        interval: int = 1,
        update_latent_num: int = 4,
    ) -> dict:
        """Score one video → {action_acc, fine_action_acc, pred/gt label tensors}.

        The returned labels are only used for filename rendering / debug.
        """
        expected_frame_num = (update_latent_num * 4) + 1
        first_frame, images, predictions, all_frame_paths = self.process_video(
            video_path, interval, last_frames=expected_frame_num
        )
        actual_frame_num = len(images)
        camera_pose = _normalize_predicted_camera_pose(
            predictions.extrinsics, expected_frame_num=actual_frame_num
        )

        expanded_gt_action = gt_action.repeat_interleave(4, dim=1)[0]
        target_action_num = min(expanded_gt_action.shape[0], camera_pose.shape[0] - 1)
        expanded_gt_action = expanded_gt_action[-target_action_num:].cpu()

        preds = [
            _camera_pose_to_discrete_action(camera_pose, mnv, 0.2)[-target_action_num:].cpu()
            for mnv in (0.002, 0.005, 0.01)
        ]
        accs = torch.stack([
            torch.mean((p == expanded_gt_action).float()) for p in preds
        ])
        best_idx = int(torch.argmax(accs).item())
        action_acc = accs[best_idx]
        pred_action = preds[best_idx]

        pred_trans_one_label = pred_action // 9
        pred_rotate_one_label = pred_action % 9
        gt_trans_one_label = (expanded_gt_action // 9).to(torch.long)
        gt_rotate_one_label = (expanded_gt_action % 9).to(torch.long)

        fine_action_acc = (
            torch.mean((pred_trans_one_label == gt_trans_one_label.cpu()).float())
            + torch.mean((pred_rotate_one_label == gt_rotate_one_label.cpu()).float())
        ) / 2

        # Clean up extracted frames (our temp only — caller doesn't see them).
        cleanup_frame_paths(all_frame_paths, first_frame)

        return {
            "action_acc": float(action_acc.item()),
            "fine_action_acc": float(fine_action_acc.item()),
            "pred_trans_one_label": pred_trans_one_label.tolist(),
            "pred_rotate_one_label": pred_rotate_one_label.tolist(),
            "gt_trans_one_label": gt_trans_one_label.tolist(),
            "gt_rotate_one_label": gt_rotate_one_label.tolist(),
        }

    @torch.no_grad()
    def score_eval(
        self,
        video_path: str,
        gt_action: torch.Tensor,
        interval: int = 1,
        latent_num: int = 64,
    ) -> dict:
        """Per-chunk evaluation.

        Returns {action_acc, fine_action_acc, actions_summary, num_chunks}
        where both accuracy lists have one entry per chunk:

          action_acc      — exact match on the composite 81-class label
                            (9 trans x 9 rotate), i.e. both axes must agree.
          fine_action_acc — mean of the two per-axis match rates
                            (trans = label // 9, rotate = label % 9), so a
                            prediction that gets one axis right still scores.
                            This is the quantity trained against when
                            action_reward_type == "fine_action"; see score().

        Both are reported so the eval log matches whichever output carries
        the training weight. fine >= coarse always holds.
        """
        first_frame, images, predictions, all_frame_paths = self.process_video(
            video_path, interval, last_frames=(latent_num * 4) - 3
        )
        pose_pred = _normalize_predicted_camera_pose(
            predictions.extrinsics, expected_frame_num=len(images)
        ).unsqueeze(0)

        expanded_gt_action = gt_action.repeat_interleave(4, dim=1)[:, 4:]
        num_frames = len(images)
        chunk_size = 4
        num_chunks = num_frames // chunk_size

        action_acc_list: list[float] = []
        fine_action_acc_list: list[float] = []
        actions_summary: list[int] = []
        for chunk_idx in range(num_chunks):
            chunk_end = ((chunk_idx + 1) * chunk_size) + 1
            chunk_gt_action = expanded_gt_action[0, :chunk_end].cpu()[-chunk_size:]
            pred_camera_pose = pose_pred[0, :chunk_end, :, :].to(torch.float32)[-(chunk_size + 1):]
            preds = [
                _camera_pose_to_discrete_action(pred_camera_pose, mnv, 0.2)[-chunk_size:].cpu()
                for mnv in (0.002, 0.005, 0.01)
            ]
            accs = [
                float(torch.mean((p == chunk_gt_action).float()).item()) for p in preds
            ]
            # score() picks the mnv threshold by best coarse acc and derives
            # fine from that same prediction; mirror that here so the two
            # numbers describe the same decoded action sequence.
            best_idx = max(range(len(accs)), key=accs.__getitem__)
            best_pred = preds[best_idx]
            gt_trans = (chunk_gt_action // 9).to(torch.long)
            gt_rotate = (chunk_gt_action % 9).to(torch.long)
            fine_acc = (
                torch.mean((best_pred // 9 == gt_trans).float())
                + torch.mean((best_pred % 9 == gt_rotate).float())
            ) / 2
            action_acc_list.append(accs[best_idx])
            fine_action_acc_list.append(float(fine_acc.item()))
            actions_summary += preds[0].cpu().tolist()

        cleanup_frame_paths(all_frame_paths, first_frame)
        return {
            "action_acc": action_acc_list,
            "fine_action_acc": fine_action_acc_list,
            "actions_summary": actions_summary,
            "num_chunks": num_chunks,
        }
