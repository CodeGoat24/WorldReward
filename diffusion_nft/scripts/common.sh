COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_SCRIPT_DIR}/.." && pwd)"
source "${COMMON_SCRIPT_DIR}/env.sh"

# --- Fabric-independent NCCL settings -------------------------------------
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_PXN_DISABLE=0
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export _CHECK_PEFT="0"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Extended NCCL timeouts so allreduce/allgather don't trigger the watchdog
# during the (slow) VLM reward phase. The 10 min default is too short.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=1800

# Fabric-specific settings, opt-in. Tuned for bonded mlx5 InfiniBand HCAs and
# wrong for other fabrics, so off by default: set NCCL_FABRIC_PRESET=ib_bond to
# enable, or export your own NCCL_SOCKET_IFNAME / NCCL_IB_* instead.
if [ "${NCCL_FABRIC_PRESET:-}" = "ib_bond" ]; then
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_IB_TC=160
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_MPI_PROFILE_PRIMS_ENABLE=1
fi

setup_world_env() {
    set +u
    source ~/.bashrc
    set -u
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    # Keep localhost off any HTTP proxy so intra-node vLLM
    # sleep/wake/infer calls don't get intercepted.
    export no_proxy="${no_proxy:-},localhost,127.0.0.1"
    export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1"
    # MPS for colocated vLLM + training on same GPUs
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log

    # ---- log noise suppression ----
    # DA3 builds its logger at import time, so this must precede python.
    export DA3_LOG_LEVEL=WARN
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"

    if [ ! -f "${CONDA_SH}" ]; then
        echo "conda.sh not found at: ${CONDA_SH}"
        echo "Point CONDA_ROOT at your conda installation, e.g."
        echo "  export CONDA_ROOT=\$HOME/miniconda3"
        exit 1
    fi

    source "${CONDA_SH}"
    conda activate "${CONDA_ENV_NAME}"
    PYTHON_BIN="$(which python)"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
        echo "Failed to resolve python after 'conda activate ${CONDA_ENV_NAME}'."
        echo "Create the env first (see docs/INSTALL.md) or set CONDA_ENV_NAME."
        exit 1
    fi

    # Prefer the env's pip-installed NCCL over any system one, whose CUDA
    # version usually does not match the torch build. Must run after the
    # activate above, or PYTHON_BIN is not the env's interpreter.
    _NCCL_LIB="$("${PYTHON_BIN}" -c 'import os,nvidia.nccl;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),"lib"))' 2>/dev/null || true)"
    if [ -n "${_NCCL_LIB}" ] && [ -d "${_NCCL_LIB}" ]; then
        export LD_LIBRARY_PATH="${_NCCL_LIB}:${LD_LIBRARY_PATH:-}"
    fi

    # MPS warmup: registers this UID with the MPS daemon so the torchrun
    # workers don't hit Error 807.
    if [ -d "${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_pipe}" ]; then
        echo "Warming up MPS client (single-process CUDA init)..."
        "${PYTHON_BIN}" -c "import torch; torch.cuda.init(); print('MPS warmup OK, device_count=', torch.cuda.device_count())" 2>&1 | tail -2 || true
    fi
}

setup_rank_and_nodes() {
    if [ $# -ge 1 ]; then
        RANK=$1
    elif [ -n "${INDEX:-}" ]; then
        RANK=$INDEX
    else
        RANK=0
    fi

    if [ $# -ge 2 ]; then
        NODES=$2
    elif [ -n "${HOST_NUM:-}" ]; then
        NODES=$HOST_NUM
    else
        NODES=1
    fi
}

setup_distributed_env() {
    NUM_GPUS="8"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export WORLD_SIZE=$NODES
    export RANK=$RANK
    export MASTER_PORT="${MASTER_PORT:-27859}"
    export NUM_NODES=$WORLD_SIZE
}

cd_repo_root() {
    cd "${REPO_ROOT}"
}

yaml_get() {
    local config_path="$1"
    local key="$2"
    local default_value="${3:-}"
    python - "$config_path" "$key" "$default_value" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
key = sys.argv[2]
default_value = sys.argv[3]

if not config_path.exists():
    print(default_value)
    raise SystemExit(0)

with config_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

value = data.get(key, default_value)
if "." in key:
    value = data
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            value = default_value
            break
else:
    value = data.get(key, default_value)
if isinstance(value, bool):
    print("True" if value else "False")
else:
    print(value)
PY
}
