#!/bin/bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

setup_world_env

CONFIG_PATH="configs/train/train_grpo_64gpu.yaml"
if [ $# -ge 1 ] && [[ "$1" == *.yaml || "$1" == *.yml || "$1" == *.json ]]; then
  CONFIG_PATH="$1"
  shift
fi

setup_rank_and_nodes "$@"
setup_distributed_env
cd_repo_root

# Weights & Biases is optional. Export WANDB_API_KEY yourself to enable it;
# with no key the trainer logs a notice and carries on without wandb.
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TOTAL_GPUS="$(yaml_get "${CONFIG_PATH}" "distributed.num_gpus" "8")"
GPUS_PER_NODE="${GPUS_PER_NODE:-$((TOTAL_GPUS / NUM_NODES))}"

echo "Begin training"
echo "Using python: ${PYTHON_BIN}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "TOTAL_GPUS=${TOTAL_GPUS} GPUS_PER_NODE=${GPUS_PER_NODE} RANK=${RANK} NODES=${NUM_NODES}"

# Per-rank log files:
#   --tee 3                 all workers' stdout+stderr -> per-rank files AND console
#   --local-ranks-filter 0  console keeps ONLY local rank 0
# The caller's log is rank-0-only; a crash on any other rank is still
# recoverable from ${TORCHRUN_LOG_DIR}/<job>/attempt_0/<local_rank>/stderr.log
# --local-ranks-filter needs torch >= 2.2.
CONFIG_NAME="$(basename "${CONFIG_PATH}" | sed 's/\.[^.]*$//')"
# Include node${RANK} in the path: on shared storage every node writes under
# logs/, and local_rank is 0..7 on all of them, so the node id is the only way
# to tell which node a given rank-0 log belongs to.
# Global rank = RANK * GPUS_PER_NODE + local_rank.
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-logs/torchrun/${CONFIG_NAME}/node${RANK}}"
mkdir -p "${TORCHRUN_LOG_DIR}"
echo "TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR} (per-rank stdout/stderr; global rank = ${RANK}*${GPUS_PER_NODE}+local_rank)"

"${PYTHON_BIN}" -u -m torch.distributed.run \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${RANK}" \
    --log-dir="${TORCHRUN_LOG_DIR}" \
    --tee 3 \
    --local-ranks-filter 0 \
    fastvideo/training/nft/in_train_pipeline.py \
    --config "${CONFIG_PATH}"
