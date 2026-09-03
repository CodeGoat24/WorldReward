#!/usr/bin/env bash
# Serve WorldReward behind an OpenAI-compatible HTTP endpoint.
#
#   bash scripts/launch_vllm_server.sh --model-path CodeGoat24/WorldReward-qwen35-9b
#
# Then point the client at it:
#
#   python scripts/run_inference.py --backend server \
#       --render-root outputs/rendered_chunks \
#       --output outputs/chunk_predictions.json
#
# For a single-node evaluation the offline backend (scripts/run_inference.py with
# the default --backend offline, one process per GPU) is faster, since it skips
# HTTP and image transport entirely. Use this server when the model is shared
# across machines or already running.
set -euo pipefail

MODEL_PATH="CodeGoat24/WorldReward-qwen35-9b"
HOST="0.0.0.0"
PORT="8080"
SERVED_MODEL_NAME="WorldReward"
TENSOR_PARALLEL_SIZE="1"
DATA_PARALLEL_SIZE="1"
GPU_MEMORY_UTILIZATION="0.90"
MAX_MODEL_LEN="49152"
# The model consumes exactly 6 images per chunk.
LIMIT_MM_PER_PROMPT='{"image": 6}'
# Required for file:// image URLs (--use-file-url on the client). Narrow this to
# your render root if you prefer.
ALLOWED_LOCAL_MEDIA_PATH="/"
EXTRA_ARGS=()

usage() {
  sed -n '2,16p' "$0"
  cat <<'EOF'

Options:
  --model-path PATH               HF repo id or local checkpoint (default: CodeGoat24/WorldReward-qwen35-9b)
  --host HOST                     Bind address (default: 0.0.0.0)
  --port PORT                     Bind port (default: 8080)
  --served-model-name NAME        Model name clients must send (default: WorldReward)
  --tensor-parallel-size N        Shard one model replica across N GPUs (default: 1)
  --data-parallel-size N          Run N independent replicas (default: 1)
  --gpu-memory-utilization F      KV-cache budget, 0-1 (default: 0.90)
  --max-model-len N               Context length (default: 49152)
  --allowed-local-media-path PATH Root the server may read images from (default: /)
  -- ARGS...                      Pass everything after -- straight to vLLM
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)                MODEL_PATH="$2"; shift 2 ;;
    --host)                      HOST="$2"; shift 2 ;;
    --port)                      PORT="$2"; shift 2 ;;
    --served-model-name)         SERVED_MODEL_NAME="$2"; shift 2 ;;
    --tensor-parallel-size)      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --data-parallel-size)        DATA_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu-memory-utilization)    GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len)             MAX_MODEL_LEN="$2"; shift 2 ;;
    --limit-mm-per-prompt)       LIMIT_MM_PER_PROMPT="$2"; shift 2 ;;
    --allowed-local-media-path)  ALLOWED_LOCAL_MEDIA_PATH="$2"; shift 2 ;;
    -h|--help)                   usage; exit 0 ;;
    --)                          shift; EXTRA_ARGS+=("$@"); break ;;
    *)                           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  "${EXTRA_ARGS[@]}"
