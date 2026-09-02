# Data Format

Training and eval data are **not** distributed with this repository — you
prepare your own. This document is the contract the loader
(`fastvideo/dataset/camera_dataset.py`) expects.

You need three things, corresponding to three config keys:

| Config key | What it is |
|---|---|
| `data.json_path` | index of pre-encoded training samples |
| `data.eval_json_path` | same format, held-out samples |
| `data.random_pose_path` | camera trajectories sampled during rollout |

## 1. Latent index — `json_path` / `eval_json_path`

A JSON array of objects. Only three fields are read
(`camera_dataset.py:245,272,421`):

```json
[
  {
    "latent_path": "/abs/or/relative/path/to/sample_00000.pt",
    "caption": "a first-person walk through a stone courtyard",
    "image_path": "/path/to/source_frame.jpg"
  }
]
```

- **`latent_path`** (required) — the `.pt` file described in §2.
- **`caption`** (required) — must be non-empty; the loader raises if it is
  missing or blank.
- **`image_path`** (optional) — carried through to logs and generated-video
  filenames only; defaults to `""`.

Keep the eval set disjoint from train. Prefer an eval sample count divisible by
your world size: sample ids are assigned as `rank + batch_idx * world_size`, so
an indivisible count leaves some ranks with fewer samples.

## 2. Per-sample `.pt`

Each `latent_path` is a `torch.save`'d dict, loaded with `weights_only=True` on
CPU. Seven keys are read:

| Key | Shape | Notes |
|---|---|---|
| `latent` | `[C, T, H, W]` | VAE latent. A leading batch dim (`[1, C, T, H, W]`) is squeezed. `T` is zero-padded up to, or truncated down to, `model.window_frames`. |
| `prompt_embeds` | `[L, D]` | Text embedding from the LLM encoder. Leading batch dim squeezed. |
| `prompt_mask` | `[L]` | Attention mask for the above. Leading batch dim squeezed. |
| `image_cond` | — | First-frame conditioning latent. |
| `vision_states` | — | Vision-encoder states for the conditioning frame. |
| `byt5_text_states` | — | ByT5 glyph-encoder states. |
| `byt5_text_mask` | — | Attention mask for the above. |

Exact channel counts and sequence lengths follow from the encoders you use, so
rather than hardcoding them, generate these files with the provided encoder:

```bash
python prepare_dataset/prepare_image_text_latent_simple.py --help
```

It takes an input JSON of `{image_path, caption}` records, runs the
HunyuanVideo-1.5 VAE plus the text/vision encoders from Step 2 of the README,
writes one `.pt` per sample, and emits the `latents.json` index from §1. Using it
guarantees the shapes match the model you are training.

## 3. Camera trajectories — `random_pose_path`

A JSON array of trajectories. Each trajectory is an object keyed by frame index
(as strings), and the loader reads the first `model.window_frames` keys **in
insertion order** (`camera_dataset.py`), so keep them ordered `"0"`, `"1"`, ….

```json
[
  {
    "0": {"extrinsic": [[4x4 camera-to-world]], "K": [[3x3 intrinsics]]},
    "1": {"extrinsic": [[...]],                 "K": [[...]]}
  }
]
```

- **`extrinsic`** — 4×4 **camera-to-world** matrix. The loader inverts it to get
  world-to-camera.
- **`K`** — 3×3 pixel-space intrinsics. Values above `2000.0` are rejected. The
  loader then normalises to a unit image: `fx /= 2*cx`, `fy /= 2*cy`, and
  `cx = cy = 0.5`.

Trajectories are assigned round-robin (`idx % len(random_pose)`), so the file
does not need as many entries as you have samples. Each trajectory must contain
at least `window_frames` frames.

Discrete actions (forward / back / left / right …) are derived from consecutive
poses, which is what the action reward scores — so the trajectories determine
which action distribution the model is trained against. `prepare_dataset/prepare_custom_action.py`
generates trajectories for a chosen action sequence.

## Directory layout

Nothing forces a particular layout, since every path is explicit in the config
and the index. A conventional one:

```
dataset/
├── train_latents/
│   ├── latents.json
│   └── sample_00000.pt ...
├── eval_latents/
│   ├── latents.json
│   └── sample_00000.pt ...
└── random_poses.json
```
