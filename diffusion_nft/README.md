# Diffusion-NFT

RL post-training code for video world models with **WorldReward** as the reward model.
Built on HunyuanVideo-1.5 + HY-WorldPlay.

- [⚙️ 1. Environment](#️-1-environment)
- [📦 2. Models](#-2-models)
- [📊 3. Data](#-3-data)
- [🚀 4. Training](#-4-training)

---

## ⚙️ 1. Environment

```bash
conda create -n diffusion_nft python=3.10 -y
conda activate diffusion_nft

pip install -r requirements.txt
pip install transformers==4.50.0

# recommended for faster training
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## 📦 2. Models

### 2.1 Download checkpoints

```bash
python scripts/download_models.py \
    --hf_token <your_huggingface_token> --cache_dir <your_cache_dir>
```

The script downloads:

- **HunyuanVideo-1.5** base model (VAE, scheduler, 480p transformer)
- **HY-WorldPlay** action models (AR, bidirectional, distilled)
- **Qwen2.5-VL-7B-Instruct** text encoder
- **ByT5** encoders (byt5-small + Glyph-SDXL-v2)
- **SigLIP** vision encoder (from FLUX.1-Redux-dev)
- **DepthAnythingV3** camera pose estimator — option 1
- **Hunyuan-WorldMirror** camera pose estimator — option 2

### 2.2 Optional code packages

```bash
# DepthAnything3 — required by camera_estimator: dav3 (the default)
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git DepthAnythingV3
mv ./DepthAnythingV3/src/depth_anything_3 ./fastvideo/rewards/computers/depth_anything_3
# or leave it where it is and: export DEPTH_ANYTHING_3_SRC=$PWD/DepthAnythingV3/src

# HunyuanWorld-Mirror — required by camera_estimator: worldmirror
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror \
    ./fastvideo/rewards/computers/HunyuanWorldMirror
# or: export WORLD_MIRROR_SRC=/path/to/checkout

# MonST3R — required only by eval_monst3r (the geometry eval metrics)
git clone https://github.com/Junyi42/monst3r \
    ./fastvideo/rewards/computers/monst3r_lib
# or: export MONST3R_SRC=/path/to/checkout
```

MonST3R also needs its checkpoint:

```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', \
    local_dir='ckpt/Junyi42--MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt')"
```

### 🏆 2.3 Reward server (optional)

The `vlm_action` / `vlm_vq` rewards are pairwise judgements from a reward model
the trainer reaches over HTTP. Set both weights to `0.0` to drop this dependency
entirely.

**Run it on GPUs the trainer is not using.** This is a separate process with its
own weights and KV cache, not something the training ranks load. One replica per spare GPU, `TP=1`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model CodeGoat24/WorldReward-9B \
    --served-model-name WorldReward \
    --tensor-parallel-size 1 \
    --port 9080 --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code --enable-prefix-caching
```

Repeat with a different `CUDA_VISIBLE_DEVICES` and `--port` for more replicas.
Point the trainer at a single one with `vlm_rm_host` / `vlm_rm_port`, or list
several under `vlm_rm_urls` (requests round-robin across the list). 

Confirm the server is up before training:

```bash
curl -m 15 http://localhost:9080/v1/models
```

---

## 📊 3. Data

The full on-disk contract — index schema, per-sample tensor keys, camera
trajectory format — is in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). The steps
below are the happy path.

### 3.1 Input JSON

```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "caption": "A serene park with trees and a bridge over water"
  },
  {
    "image_path": "/path/to/image2.png",
    "caption": "A modern city street at sunset"
  }
]
```

### 3.2 Encode latents

Encodes images and text into latent features via the VAE and text encoders.

**Single GPU**:

```bash
python prepare_dataset/prepare_image_text_latent_simple.py \
    --input_json /path/to/train.json \
    --output_dir /path/to/train_latents \
    --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5
```

**Multi-GPU (recommended)**:

```bash
torchrun --nproc_per_node=8 prepare_dataset/prepare_image_text_latent_simple.py \
    --input_json /path/to/train.json \
    --output_dir /path/to/train_latents \
    --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5
```

**Output**:

```
/path/to/train_latents/
├── latents/
│   ├── 0_000000.pt
│   ├── 0_000001.pt
│   └── ...
└── latents.json          # index file, point data.json_path at this
```

Each `.pt` contains:

- `latent`: VAE-encoded features `[1, C, T, H, W]`
- `image_cond`: first-frame condition `[1, C, 1, H, W]`
- `prompt_embeds`: text embeddings `[1, L, D]`
- `prompt_mask`: text attention mask `[1, L]`
- `vision_states`: visual features `[1, N, D]`
- `byt5_text_states`: ByT5 features `[1, 256, 1472]`
- `byt5_text_mask`: ByT5 attention mask `[1, 256]`

### 3.3 Evaluation set

Build a small held-out set so you can watch a fixed subset during training. Same
JSON format as training. Keep the sample count divisible by your GPU count
(sample ids are `rank + batch_idx * world_size`); 16 or 32 samples is usually
enough.

```bash
python prepare_dataset/prepare_image_text_latent_simple.py \
    --input_json /path/to/eval.json \
    --output_dir /path/to/eval_latents \
    --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5
```

### 3.4 Action trajectories

```bash
python prepare_dataset/prepare_custom_action.py \
    -o dataset/random_poses.json -n 1000 --frames 128
```

This writes the JSON that `data.random_pose_path` points at — 1000 trajectories
of 128 frames by default, weighted towards complex composite actions. `--frames`
must be at least `model.window_frames` from your config. Use `--help` for the
options; edit the script to change the synthesis rules themselves.

---

## 🚀 4. Training

### 4.1 Write a config

Training is driven entirely by a YAML config passed to the launcher — there is
nothing to edit inside the shell script. Two configs are provided; they differ
only in `num_gpus` / `hsdp_replicate_dim` and the output paths, so every
hyperparameter is shared:

| Config | Training | Reward server |
|---|---|---|
| `train_grpo_16gpu.yaml` | 2 nodes × 8 | 2 nodes × 8 replicas — see §4.3 |
| `train_grpo_64gpu.yaml` | 8 nodes × 8 | as many replicas as you can spare |

Start from whichever matches your allocation:

```bash
cp configs/train/train_grpo_16gpu.yaml configs/train/my_run.yaml
```

Fill in the six paths marked `SET ME` (three data paths from §3, three
checkpoint paths from §2), then size it for your cluster:
`distributed.num_gpus` must equal `hsdp_replicate_dim` (node count) times
`hsdp_shard_dim` (GPUs per node).

Which rewards train, and which are only measured:

| Reward | Trains? |
|---|---|
| `action` 2.0, `hpsv3_quality_drift` 1.0 | yes |
| `vlm_action` 2.0, `vlm_vq` 1.0 | yes — needs the reward server (§2.3) |
| `hpsv3`, `hpsv3_quality` | no — weight 0.0, but still logged every step |
| `ate_rmse`, `rpe_trans`, `rpe_rot`, `aesthetic` | no — eval-only |

Geometry and aesthetic are computed and logged at every eval but never enter the
objective, so MonST3R is only needed for `eval_monst3r`; set it to `false` to
train without MonST3R.

Environment variables the launcher reads (all optional):

```bash
export MASTER_ADDR=<master hostname or IP>   # required for multi-node
export MASTER_PORT=29500                     # must be free on the master
export CONDA_ROOT=$HOME/miniconda3           # only if auto-discovery fails
export CONDA_ENV_NAME=diffusion_nft           # conda env name
export WANDB_API_KEY=<your key>              # omit to train without wandb
```

### 🔥 4.2 Launch

Usage: `bash scripts/train.sh <config> <node_rank> <num_nodes>`

**Single node (8 GPUs)**:

```bash
bash scripts/train.sh configs/train/my_run.yaml 0 1
```

**Multi-node** — run on every node with its own rank:

```bash
# Node 0 (master):
bash scripts/train.sh configs/train/my_run.yaml 0 4
# Node 1:
bash scripts/train.sh configs/train/my_run.yaml 1 4
# Node 2:
bash scripts/train.sh configs/train/my_run.yaml 2 4
# Node 3:
bash scripts/train.sh configs/train/my_run.yaml 3 4
```

All nodes must reach each other on `MASTER_ADDR:MASTER_PORT`, and
`hsdp_replicate_dim` in the config must equal `<num_nodes>`.

If your cluster uses bonded InfiniBand HCAs, `export NCCL_FABRIC_PRESET=ib_bond`
(see `scripts/common.sh`); otherwise set your own `NCCL_SOCKET_IFNAME` /
`NCCL_IB_HCA`. The defaults are fabric-agnostic.

Training on 8 GPUs shows initial improvements, but more nodes generally give
more stable training and better final results.

### 4.3 Four-node reference layout

`train_grpo_16gpu.yaml` assumes four nodes split into two roles. Keep them
separate: vLLM does not release the memory it reserves, and the trainer runs
close enough to the ceiling that sharing devices with it will OOM the run.

```
nodes 1-2   16 training ranks   bash scripts/train.sh ... <rank> 2
nodes 3-4   16 vLLM replicas    one per GPU, ports 9080-9087
```

On each of the two reward nodes, start one replica per GPU:

```bash
for GPU in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
      --model CodeGoat24/WorldReward-9B \
      --served-model-name WorldReward \
      --tensor-parallel-size 1 \
      --port $((9080 + GPU)) --host 0.0.0.0 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.85 \
      --trust-remote-code --enable-prefix-caching \
      > /tmp/vllm_gpu${GPU}.log 2>&1 &
done
```

Allow ~5 minutes for load and warmup, then confirm all eight answer:

```bash
for p in $(seq 9080 9087); do curl -s http://localhost:$p/v1/models | grep -o WorldReward; done
```

Then uncomment `vlm_rm_urls` in the config and put the two reward nodes'
addresses in. That gives 16 replicas for 16 training ranks; requests
round-robin, so the reward phase stays off the critical path.

### 4.4 Monitor

Checkpoints and sample videos go to the `output_dir` /
`generated_videos_dir` from your config:

```
<output_dir>/
├── checkpoint-{step}/
│   ├── transformer/
│   │   └── diffusion_pytorch_model.safetensors   # consolidated weights
│   └── distributed_checkpoint/                   # resumable state
└── ...
```

Every `checkpointing_steps` steps the trainer runs an eval, logs it as
`Eval step <step-1>` (the state of `checkpoint-<step>`), then writes the
checkpoint.

---

## 🙏 Acknowledgement

Our RL training code is built upon [WorldCompass](https://github.com/Tencent-Hunyuan/HY-WorldPlay/tree/main/worldcompass). We sincerely appreciate their excellent work.