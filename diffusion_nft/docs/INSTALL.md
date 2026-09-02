# Environment Setup

Two independent conda environments are required:

| env | role | Python | torch | key deps |
|---|---|---|---|---|
| **`diffusion_nft`** | training, MonST3R / aesthetic eval | 3.10 | 2.5–2.7 | transformers 4.x, numpy 1.26 |
| **`reward`** | VLM reward-model server, reached over HTTP | 3.11 | 2.8–2.10 | vllm, transformers 5.x, numpy 2.x |

They must be separate — numpy 1.x / 2.x and transformers 4.x / 5.x cannot
coexist. The second one is only needed if the `vlm_*` reward weights are
non-zero; see section 2.

---

## Prerequisites

Match the CUDA build of PyTorch to your host driver (`nvidia-smi` shows the max supported CUDA):

| driver shows | use PyTorch wheel |
|---|---|
| ≥ 12.6 | `cu126` |
| ≥ 12.4 | `cu124` |
| ≥ 12.1 | `cu121` |
| ≥ 11.8 | `cu118` |

Commands below use `cu124` — change it to match your setup.

---

## 1. Training environment

```bash
# 1.1 create
conda create -n diffusion_nft python=3.10 -y
conda activate diffusion_nft

# 1.2 install PyTorch (pick the cuXXX that matches your driver)
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.* torchvision==0.21.* torchaudio==2.6.*

# 1.3 install the rest
pip install -r requirements.txt

# 1.4 OPTIONAL: flash-attn (prebuilt wheels; must match torch + CUDA)
#     If the pinned version in requirements.txt does not match your
#     torch/CUDA combo, grab a matching wheel from
#     https://github.com/Dao-AILab/flash-attention/releases and install it:
#       pip install flash_attn-X.Y.Z+cuXXXtorchYYY-cp310-linux_x86_64.whl

# 1.5 OPTIONAL: HPSv3 reward (already listed in requirements.txt)
#     If `pip install hpsv3` fails on your index, install from source:
#       pip install git+https://github.com/MizzenAI/HPSv3
```

### MonST3R trajectory eval

MonST3R is **not** bundled with this repository. Clone the code, then fetch
the checkpoint:

```bash
# Code (review the NonCommercial terms first)
git clone https://github.com/Junyi42/monst3r fastvideo/rewards/computers/monst3r_lib

# Download MonST3R weights (~2.2 GB)
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', \
    local_dir='ckpt/Junyi42--MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt')"
```

### Aesthetic video eval

Two checkpoints (~4 MB + ~2.5 GB):

```bash
mkdir -p ckpt/aesthetic
wget "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth" \
    -O "ckpt/aesthetic/sac+logos+ava1-l14-linearMSE.pth"

python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openai/clip-vit-large-patch14', \
    local_dir='ckpt/openai--clip-vit-large-patch14')"
```

If `huggingface.co` is slow or unreachable from your network, prefix the two HF commands with a mirror, e.g. `HF_ENDPOINT=https://hf-mirror.com`.

### Sanity check

```bash
conda activate diffusion_nft
python -c "import torch, transformers, diffusers, evo, cv2; \
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
    print('transformers', transformers.__version__)"

python -c "from fastvideo.rewards.computers.aesthetic import AestheticVideoEvaluator; \
    from fastvideo.rewards.computers.monst3r_trajectory import MonST3RTrajectoryEvaluator; \
    print('reward modules import OK')"
```

---

## 2. Reward server (optional)

The pairwise VLM reward is served by a standalone vLLM process that the trainer
only talks to over HTTP. It needs its own environment — vLLM's pinned torch and
transformers conflict with the training env's, so do not install it into the
training env.

```bash
conda create -n reward python=3.11 -y
conda activate reward
pip install vllm
```

Launch it on GPUs the trainer is not using; the command and the reasoning are in
the README's "Reward server" section. Health check:

```bash
curl -m 15 http://localhost:9080/v1/models
```

Set the `vlm_action` / `vlm_vq` reward weights to 0.0 to skip this entirely.

---

## 3. Switching between envs

```bash
# train (this repository)
conda activate diffusion_nft
bash scripts/train.sh configs/train/my_run.yaml 0 1

# reward server, separately and on another set of GPUs — see section 2 above
conda activate reward
python -m vllm.entrypoints.openai.api_server --model CodeGoat24/WorldReward-9B ...
```

The two envs never need to see each other's site-packages — they communicate only over HTTP.

---

## Troubleshooting

**`ImportError: No module named 'roma'`** (training env)
MonST3R dep. `pip install roma` — already listed in `requirements.txt`.

**`flash_attn` wheel ABI mismatch**
Your torch + CUDA combo has no prebuilt wheel. Download the closest match from the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) and install it explicitly, or drop `flash-attn` from the file.

**reward server fails to start**
It is installed from the parent repository, not here — see section 2 and that
repository's own instructions.

**`numpy.ndarray has no attribute 'bool'` (training env)**
Some downstream pinned numpy 2.x. Force `numpy<2`: `pip install 'numpy<2'`.

**HF downloads hang or time out**
Try a mirror (`HF_ENDPOINT=https://hf-mirror.com <cmd>`) or pre-download with `bash scripts/hfd.sh <repo_id>`.
