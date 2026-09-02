#!/usr/bin/env python3
"""Download script for HY-WorldPlay models.

Optimizations in this version:
- Prefer aria2c via hfd.sh for Hugging Face downloads.
- Avoid copy-after-download for large repos by downloading directly to final targets.
- Run independent downloads concurrently.
- Keep hf_hub as an explicit fallback backend.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HFD_SCRIPT = SCRIPT_DIR / "hfd.sh"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DEFAULT_MODEL_ROOT = os.path.expanduser("~/.cache/huggingface/hfd")
DEFAULT_MAX_WORKERS = min(16, max(8, os.cpu_count() or 8))
DEFAULT_JOBS = 2
DEFAULT_ARIA2_THREADS = 8
DEFAULT_ARIA2_JOBS = 8


def check_dependencies():
    """Check and install required Python dependencies."""
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -U 'huggingface_hub[cli]'")

    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope")


def check_hfd_dependencies():
    """Validate shell dependencies required by hfd.sh + aria2c."""
    missing = []
    if not HFD_SCRIPT.is_file():
        missing.append(str(HFD_SCRIPT))

    for command in ("bash", "aria2c", "curl"):
        if shutil.which(command) is None:
            missing.append(command)

    if missing:
        raise RuntimeError(
            "hfd backend requires the following dependencies: " + ", ".join(missing)
        )


def configure_download_env():
    """Enable faster transport features when supported by the local environment."""
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        print("Enabled hf_transfer for hf_hub fallback downloads")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)


def print_banner(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def ensure_clean_dir(path):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.exists(path):
        shutil.rmtree(path)


def has_required_files(path, required_relpaths=None, min_entries=None):
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False

    if required_relpaths:
        for relpath in required_relpaths:
            if not (p / relpath).exists():
                return False

    if min_entries is not None:
        try:
            if len(list(p.iterdir())) < min_entries:
                return False
        except OSError:
            return False

    return True


def model_root_dir(cache_dir):
    root = os.path.abspath(os.path.expanduser(cache_dir or DEFAULT_MODEL_ROOT))
    os.makedirs(root, exist_ok=True)
    return root


def repo_local_dir(repo_id, root_dir):
    return os.path.join(root_dir, repo_id.replace("/", "--"))


def normalize_hf_username(hf_token, hf_username):
    if hf_username:
        return hf_username
    if hf_token:
        return os.environ.get("HF_USERNAME") or "hf_user"
    return None


def download_hf_repo_hfd(
    repo_id,
    *,
    local_dir,
    allow_patterns=None,
    exclude_patterns=None,
    token=None,
    hf_username=None,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
    revision="main",
):
    """Download a HF repo via hfd.sh + aria2c."""
    check_hfd_dependencies()
    os.makedirs(local_dir, exist_ok=True)

    cmd = [
        "bash",
        str(HFD_SCRIPT),
        repo_id,
        "--local-dir",
        local_dir,
        "--tool",
        "aria2c",
        "-x",
        str(aria2_threads),
        "-j",
        str(aria2_jobs),
        "--revision",
        revision,
    ]
    if allow_patterns:
        cmd.extend(["--include", *allow_patterns])
    if exclude_patterns:
        cmd.extend(["--exclude", *exclude_patterns])
    if token:
        cmd.extend(
            [
                "--hf_token",
                token,
                "--hf_username",
                normalize_hf_username(token, hf_username),
            ]
        )

    env = os.environ.copy()
    env["HF_ENDPOINT"] = HF_ENDPOINT
    subprocess.run(cmd, check=True, env=env)
    return local_dir


def download_hf_repo_hub(
    repo_id,
    *,
    cache_dir=None,
    local_dir=None,
    allow_patterns=None,
    token=None,
    local_files_only=False,
    max_workers=DEFAULT_MAX_WORKERS,
):
    """Download a HF repo via huggingface_hub snapshot_download."""
    from huggingface_hub import snapshot_download

    kwargs = {
        "cache_dir": cache_dir,
        "local_dir": local_dir,
        "allow_patterns": allow_patterns,
        "token": token,
        "local_files_only": local_files_only,
        "max_workers": max_workers,
        "endpoint": HF_ENDPOINT,
    }
    return snapshot_download(repo_id, **kwargs)


def download_hf_repo(
    repo_id,
    *,
    backend,
    cache_dir=None,
    local_dir=None,
    allow_patterns=None,
    token=None,
    hf_username=None,
    local_files_only=False,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    if backend == "hfd":
        if local_files_only:
            if local_dir and has_required_files(local_dir, min_entries=1):
                return local_dir
            raise FileNotFoundError(f"Local repo not found: {local_dir}")
        if not local_dir:
            raise ValueError("hfd backend requires local_dir")
        return download_hf_repo_hfd(
            repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=token,
            hf_username=hf_username,
            aria2_threads=aria2_threads,
            aria2_jobs=aria2_jobs,
        )

    return download_hf_repo_hub(
        repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=token,
        local_files_only=local_files_only,
        max_workers=max_workers,
    )


def download_ms_repo(
    repo_id,
    *,
    cache_dir=None,
    local_dir=None,
    allow_patterns=None,
    local_files_only=False,
    max_workers=DEFAULT_MAX_WORKERS,
):
    from modelscope import snapshot_download as ms_snapshot_download

    kwargs = {
        "cache_dir": cache_dir,
        "local_dir": local_dir,
        "local_files_only": local_files_only,
        "max_workers": max_workers,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    return ms_snapshot_download(repo_id, **kwargs)


def download_hy_worldplay(
    root_dir,
    *,
    backend,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download only ar_model/diffusion_pytorch_model.safetensors from tencent/HY-WorldPlay."""
    print_banner(
        "[1/8] Downloading ar_model/diffusion_pytorch_model.safetensors from tencent/HY-WorldPlay..."
    )

    worldplay_dir = repo_local_dir("tencent/HY-WorldPlay", root_dir)
    model_path = os.path.join(
        worldplay_dir, "ar_model", "diffusion_pytorch_model.safetensors"
    )
    if os.path.exists(model_path):
        print(f"HY-WorldPlay already exists at: {model_path}")
        return model_path

    download_hf_repo(
        "tencent/HY-WorldPlay",
        backend=backend,
        cache_dir=cache_dir,
        local_dir=worldplay_dir,
        allow_patterns=["ar_model/diffusion_pytorch_model.safetensors"],
        max_workers=max_workers,
        aria2_threads=aria2_threads,
        aria2_jobs=aria2_jobs,
    )
    print(f"Downloaded file: {model_path}")
    return model_path


def download_hunyuan_video(
    root_dir,
    *,
    backend,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download HunyuanVideo-1.5 base models (vae, scheduler, transformer)."""
    print_banner(
        "[2/8] Downloading tencent/HunyuanVideo-1.5 (vae, scheduler, transformer)..."
    )

    hunyuan_dir = repo_local_dir("tencent/HunyuanVideo-1.5", root_dir)
    if has_required_files(
        hunyuan_dir,
        required_relpaths=[
            "vae/config.json",
            "scheduler/scheduler_config.json",
            "transformer/480p_i2v/config.json",
        ],
        min_entries=3,
    ):
        print(f"HunyuanVideo already exists at: {hunyuan_dir}")
        return hunyuan_dir

    download_hf_repo(
        "tencent/HunyuanVideo-1.5",
        backend=backend,
        cache_dir=cache_dir,
        local_dir=hunyuan_dir,
        allow_patterns=["vae/*", "scheduler/*", "transformer/480p_i2v/*"],
        max_workers=max_workers,
        aria2_threads=aria2_threads,
        aria2_jobs=aria2_jobs,
    )
    print(f"Downloaded to: {hunyuan_dir}")
    return hunyuan_dir


def download_llm_text_encoder(
    hunyuan_path,
    *,
    backend,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download Qwen2.5-VL-7B-Instruct as the LLM text encoder."""
    print_banner("[3/8] Downloading LLM text encoder (Qwen2.5-VL-7B-Instruct)...")

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    llm_target = os.path.join(text_encoder_base, "llm")
    if has_required_files(llm_target, required_relpaths=["config.json"], min_entries=5):
        print(f"LLM text encoder already exists at: {llm_target}")
        return llm_target

    ensure_clean_dir(llm_target)
    print("Downloading Qwen/Qwen2.5-VL-7B-Instruct directly to target...")
    download_hf_repo(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        backend=backend,
        cache_dir=cache_dir,
        local_dir=llm_target,
        max_workers=max_workers,
        aria2_threads=aria2_threads,
        aria2_jobs=aria2_jobs,
    )
    print(f"Downloaded to: {llm_target}")
    return llm_target


def download_byt5_encoders(
    hunyuan_path,
    *,
    backend,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download ByT5 text encoders (byt5-small and Glyph-SDXL-v2)."""
    print_banner("[4/8] Downloading ByT5 text encoders...")

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    byt5_target = os.path.join(text_encoder_base, "byt5-small")
    if has_required_files(byt5_target, required_relpaths=["config.json"], min_entries=3):
        print(f"byt5-small already exists at: {byt5_target}")
    else:
        ensure_clean_dir(byt5_target)
        print("Downloading google/byt5-small directly to target...")
        download_hf_repo(
            "google/byt5-small",
            backend=backend,
            cache_dir=cache_dir,
            local_dir=byt5_target,
            max_workers=max_workers,
            aria2_threads=aria2_threads,
            aria2_jobs=aria2_jobs,
        )
        print(f"Downloaded to: {byt5_target}")

    glyph_target = os.path.join(text_encoder_base, "Glyph-SDXL-v2")
    if has_required_files(
        glyph_target, required_relpaths=["checkpoints/byt5_model.pt"]
    ):
        print(f"Glyph-SDXL-v2 already exists at: {glyph_target}")
    else:
        ensure_clean_dir(glyph_target)
        print("Downloading AI-ModelScope/Glyph-SDXL-v2 directly to target...")
        download_ms_repo(
            "AI-ModelScope/Glyph-SDXL-v2",
            cache_dir=cache_dir or "/tmp/glyph_cache",
            local_dir=glyph_target,
            max_workers=max_workers,
        )
        print(f"Downloaded to: {glyph_target}")

    return text_encoder_base


def download_vision_encoder(
    hunyuan_path,
    hf_token,
    *,
    backend,
    hf_username=None,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download SigLIP vision encoder from FLUX.1-Redux-dev."""
    print_banner("[5/8] Downloading Vision Encoder (SigLIP from FLUX.1-Redux-dev)...")

    if not hf_token:
        print("WARNING: No HF token provided!")
        print(
            "The vision encoder requires access to: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
        )
        print("Skipping vision encoder download.")
        print("\nYou can download it manually later.")
        return None

    vision_encoder_base = os.path.join(hunyuan_path, "vision_encoder")
    os.makedirs(vision_encoder_base, exist_ok=True)

    siglip_target = os.path.join(vision_encoder_base, "siglip")
    if has_required_files(
        siglip_target,
        required_relpaths=[
            "image_encoder/config.json",
            "feature_extractor/preprocessor_config.json",
        ],
        min_entries=2,
    ):
        print(f"siglip already exists at: {siglip_target}")
        return siglip_target

    ensure_clean_dir(siglip_target)

    print("Downloading black-forest-labs/FLUX.1-Redux-dev image encoder via selected backend...")
    try:
        download_hf_repo(
            "black-forest-labs/FLUX.1-Redux-dev",
            backend=backend,
            cache_dir=cache_dir,
            local_dir=siglip_target,
            allow_patterns=["image_encoder/*", "feature_extractor/*"],
            token=hf_token,
            hf_username=hf_username,
            max_workers=max_workers,
            aria2_threads=aria2_threads,
            aria2_jobs=aria2_jobs,
        )
        print(f"Downloaded to: {siglip_target}")
        return siglip_target
    except Exception as e:
        print(f"ERROR: Failed to download vision encoder: {e}")
        print(
            "Make sure you have requested access to FLUX.1-Redux-dev and your token is valid."
        )
        return None


def download_worldmirror(
    root_dir,
    *,
    backend,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download HunyuanWorld-Mirror for camera pose estimation."""
    print_banner("[6/8] Downloading WorldMirror (tencent/HunyuanWorld-Mirror)...")

    worldmirror_dir = repo_local_dir("tencent/HunyuanWorld-Mirror", root_dir)
    if has_required_files(worldmirror_dir, min_entries=3):
        print(f"WorldMirror already exists at: {worldmirror_dir}")
        return worldmirror_dir

    try:
        print("Downloading tencent/HunyuanWorld-Mirror...")
        download_hf_repo(
            "tencent/HunyuanWorld-Mirror",
            backend=backend,
            cache_dir=cache_dir,
            local_dir=worldmirror_dir,
            max_workers=max_workers,
            aria2_threads=aria2_threads,
            aria2_jobs=aria2_jobs,
        )
        print(f"Downloaded to: {worldmirror_dir}")
        print("WorldMirror weights are ready; module import/loading stays lazy at first use.")
        return worldmirror_dir
    except Exception as e:
        print(f"WARNING: Failed to download WorldMirror: {e}")
        print("The model will be downloaded on first use.")
        return None


def download_depth_anything_3(
    root_dir,
    *,
    backend,
    cache_dir=None,
    max_workers=DEFAULT_MAX_WORKERS,
    aria2_threads=DEFAULT_ARIA2_THREADS,
    aria2_jobs=DEFAULT_ARIA2_JOBS,
):
    """Download DepthAnything3 model for camera pose estimation (optional)."""
    print_banner("[7/8] Downloading DepthAnything3 (depth-anything/DA3-GIANT-1.1)...")

    da3_dir = repo_local_dir("depth-anything/DA3-GIANT-1.1", root_dir)
    if has_required_files(da3_dir, min_entries=3):
        print(f"DepthAnything3 already exists at: {da3_dir}")
        return da3_dir

    print("Downloading depth-anything/DA3-GIANT-1.1...")
    download_hf_repo(
        "depth-anything/DA3-GIANT-1.1",
        backend=backend,
        cache_dir=cache_dir,
        local_dir=da3_dir,
        max_workers=max_workers,
        aria2_threads=aria2_threads,
        aria2_jobs=aria2_jobs,
    )
    print(f"Downloaded to: {da3_dir}")
    return da3_dir


def print_paths(root_dir):
    """Print the model paths for run.sh configuration."""
    print_banner("[8/8] Verifying downloads...")

    hunyuan_path = repo_local_dir("tencent/HunyuanVideo-1.5", root_dir)
    worldplay_path = repo_local_dir("tencent/HY-WorldPlay", root_dir)
    worldmirror_path = repo_local_dir("tencent/HunyuanWorld-Mirror", root_dir)
    da3_path = repo_local_dir("depth-anything/DA3-GIANT-1.1", root_dir)

    if not has_required_files(hunyuan_path, min_entries=1):
        raise FileNotFoundError(f"Missing HunyuanVideo download: {hunyuan_path}")
    if not has_required_files(worldplay_path, min_entries=1):
        raise FileNotFoundError(f"Missing HY-WorldPlay download: {worldplay_path}")

    if not has_required_files(worldmirror_path, min_entries=1):
        worldmirror_path = None
    if not has_required_files(da3_path, min_entries=1):
        da3_path = None

    print(f"\n{'=' * 60}")
    print("ALL DOWNLOADS COMPLETE!")
    print("=" * 60)
    print("\nADD these paths to your prepare_dataset/prepare_image_text_latent_simple.py:\n")
    print(f"--hunyuan_checkpoint_path {hunyuan_path}")

    print("\nModify these paths in your scripts/full_hy_nft.sh:\n")
    print(f"CACHE_DIR={root_dir}")
    print(f"HUNYUAN_CHECKPOINT={hunyuan_path}")
    print(f"WORLDPLAY_CHECKPOINT={worldplay_path}")
    if worldmirror_path:
        print(f"WORLDMIRROR_CHECKPOINT={worldmirror_path}")
    if da3_path:
        print(f"DEPTH_ANYTHING_3_CHECKPOINT={da3_path}")

    print("\nYou can now run: bash prepare_dataset/extract_latents.sh to prepare your dataset")
    print("\nAnd then run: bash scripts/full_hy_nft.sh to start training")


def run_tasks(tasks, jobs):
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        future_to_name = {executor.submit(func): name for name, func in tasks.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            results[name] = future.result()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download all required models for HY-WorldPlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Download with aria2/hfd backend
    python download_models.py --hf_token hf_xxxxxxxxxxxxx --hf_backend hfd

    # Download to custom directory with more aria2 concurrency
    python download_models.py --hf_token hf_xxxxxxxxxxxxx --cache_dir /mnt/data/models --aria2_threads 8 --aria2_jobs 8

Note:
    The HuggingFace token is required for downloading the vision encoder
    from black-forest-labs/FLUX.1-Redux-dev. You need to:
    1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    2. Wait for approval (usually instant)
    3. Create a token at: https://huggingface.co/settings/tokens (select "Read" permission)

Default HF endpoint: {HF_ENDPOINT}
Default model root: {DEFAULT_MODEL_ROOT}
        """,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for downloading gated models (required for vision encoder)",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default=None,
        help="HuggingFace username for hfd gated-repo auth checks (optional; defaults to HF_USERNAME or a placeholder)",
    )
    parser.add_argument(
        "--hf_backend",
        type=str,
        choices=["hfd", "hf_hub"],
        default="hfd",
        help="Backend for Hugging Face downloads (default: hfd)",
    )
    parser.add_argument(
        "--skip_vision_encoder",
        action="store_true",
        help="Skip downloading the vision encoder (if you don't have FLUX access yet)",
    )
    parser.add_argument(
        "--skip_worldmirror",
        action="store_true",
        help="Skip downloading WorldMirror model (optional, used for camera pose estimation)",
    )
    parser.add_argument(
        "--skip_depth_anything_3",
        action="store_true",
        help="Skip downloading DepthAnything3 model (optional, alternative to WorldMirror)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to store downloaded model repos (default: ~/.cache/huggingface/hfd)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"How many independent model downloads to run in parallel (default: {DEFAULT_JOBS})",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "How many file workers hf_hub fallback can use per repo "
            f"(default: {DEFAULT_MAX_WORKERS})"
        ),
    )
    parser.add_argument(
        "--aria2_threads",
        type=int,
        default=DEFAULT_ARIA2_THREADS,
        help=f"aria2 -x/-s thread count per repo when using hfd (default: {DEFAULT_ARIA2_THREADS})",
    )
    parser.add_argument(
        "--aria2_jobs",
        type=int,
        default=DEFAULT_ARIA2_JOBS,
        help=f"aria2 -j concurrent file count per repo when using hfd (default: {DEFAULT_ARIA2_JOBS})",
    )

    args = parser.parse_args()

    root_dir = model_root_dir(args.cache_dir)
    print(f"\n{'=' * 60}")
    print(f"Using model root: {root_dir}")
    print(f"{'=' * 60}\n")

    print("=" * 60)
    print("HY-WorldPlay Model Download Script")
    print("=" * 60)
    print(f"HF endpoint: {HF_ENDPOINT}")
    print(f"HF backend: {args.hf_backend}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"hf_hub fallback workers: {args.max_workers}")
    print(f"aria2 threads per repo: {args.aria2_threads}")
    print(f"aria2 jobs per repo: {args.aria2_jobs}")

    check_dependencies()
    configure_download_env()
    if args.hf_backend == "hfd":
        check_hfd_dependencies()

    stage1_results = run_tasks(
        {
            "worldplay": lambda: download_hy_worldplay(
                root_dir,
                backend=args.hf_backend,
                cache_dir=args.cache_dir,
                max_workers=args.max_workers,
                aria2_threads=args.aria2_threads,
                aria2_jobs=args.aria2_jobs,
            ),
            "hunyuan": lambda: download_hunyuan_video(
                root_dir,
                backend=args.hf_backend,
                cache_dir=args.cache_dir,
                max_workers=args.max_workers,
                aria2_threads=args.aria2_threads,
                aria2_jobs=args.aria2_jobs,
            ),
        },
        jobs=min(args.jobs, 2),
    )
    worldplay_path = stage1_results["worldplay"]
    hunyuan_path = stage1_results["hunyuan"]

    stage2_tasks = {
        "llm": lambda: download_llm_text_encoder(
            hunyuan_path,
            backend=args.hf_backend,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            aria2_threads=args.aria2_threads,
            aria2_jobs=args.aria2_jobs,
        ),
        "byt5": lambda: download_byt5_encoders(
            hunyuan_path,
            backend=args.hf_backend,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            aria2_threads=args.aria2_threads,
            aria2_jobs=args.aria2_jobs,
        ),
    }

    if not args.skip_vision_encoder:
        stage2_tasks["vision"] = lambda: download_vision_encoder(
            hunyuan_path,
            args.hf_token,
            backend=args.hf_backend,
            hf_username=args.hf_username,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            aria2_threads=args.aria2_threads,
            aria2_jobs=args.aria2_jobs,
        )
    else:
        print("\n[5/8] Skipping vision encoder download (--skip_vision_encoder flag)")

    if not args.skip_worldmirror:
        stage2_tasks["worldmirror"] = lambda: download_worldmirror(
            root_dir,
            backend=args.hf_backend,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            aria2_threads=args.aria2_threads,
            aria2_jobs=args.aria2_jobs,
        )
    else:
        print("\n[6/8] Skipping WorldMirror download (--skip_worldmirror flag)")

    if not args.skip_depth_anything_3:
        stage2_tasks["depth_anything_3"] = lambda: download_depth_anything_3(
            root_dir,
            backend=args.hf_backend,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            aria2_threads=args.aria2_threads,
            aria2_jobs=args.aria2_jobs,
        )
    else:
        print("\n[7/8] Skipping DepthAnything3 download (--skip_depth_anything_3 flag)")

    run_tasks(stage2_tasks, jobs=args.jobs)

    _ = worldplay_path
    print_paths(root_dir)


if __name__ == "__main__":
    main()
