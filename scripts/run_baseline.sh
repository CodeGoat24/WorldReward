#!/usr/bin/env bash
# Evaluate one baseline on WorldReward-Bench: shard over GPUs, merge, score.
#
#   bash scripts/run_baseline.sh --baseline hpsv3
#   bash scripts/run_baseline.sh --baseline dav3 --num-gpus 8
#   bash scripts/run_baseline.sh --baseline all
#
# Wraps the adapters in baselines/ so a full comparison run is one command. Each
# baseline writes <out-dir>/<baseline>.pairs.jsonl plus a scored report, all
# through the same worldreward-bench/score.py that scores WorldReward itself.
#
# Baselines and the axes they answer:
#
#   aesthetic     appearance                   CLIP ViT-L/14 + LAION head
#   hpsv3         appearance                   pip install hpsv3
#   videoalign    appearance, motion           needs --videoalign-src
#   think         appearance, motion           needs a vLLM server (see below)
#   flex          appearance, motion           needs a vLLM server (see below)
#   dav3          action                       needs --depth-anything-3-src
#
# The UnifiedReward variants are HTTP clients, so serve the checkpoint first:
#
#   vllm serve CodeGoat24/UnifiedReward-Think-qwen35-9b \
#       --served-model-name UnifiedReward --port 8080 \
#       --limit-mm-per-prompt '{"image": 16}' --max-model-len 32768
#
# Runs are cached per pair, so re-running after an interruption resumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASELINE=""
BENCH="data/WorldReward-Bench/bench.jsonl"
OUT_DIR="outputs/baselines"
NUM_GPUS="1"
EPS=""
URL="http://127.0.0.1:8080"
SERVED_MODEL_NAME="UnifiedReward"
CLIP_PATH=""
AESTHETIC_MLP=""
HPSV3_CHECKPOINT=""
VIDEOALIGN_SRC=""
VIDEOALIGN_CKPT=""
DA3_SRC="${DEPTH_ANYTHING_3_SRC:-}"
DA3_MODEL=""
LIMIT=""
SKIP_SCORE=0
EXTRA_ARGS=()

ALL_BASELINES=(aesthetic hpsv3 videoalign think flex dav3)

usage() {
  sed -n '2,29p' "$0"
  cat <<'EOF'

Options:
  --baseline NAME            One of: aesthetic hpsv3 videoalign think flex dav3, or "all"
  --bench PATH               bench.jsonl (default: data/WorldReward-Bench/bench.jsonl)
  --out-dir DIR              Where predictions and reports go (default: outputs/baselines)
  --num-gpus N               Shard across N GPUs, one process each (default: 1)
  --eps F                    Tie band on the score gap (default: per-baseline)
  --limit N                  Only score the first N pairs (smoke test)
  --url URL                  vLLM endpoint for think/flex (default: http://127.0.0.1:8080)
  --served-model-name NAME   Model name to send to that endpoint (default: UnifiedReward)
  --clip-path PATH           CLIP checkpoint for aesthetic
  --aesthetic-mlp PATH       LAION aesthetic head (.pth) for aesthetic
  --hpsv3-checkpoint PATH    HPSv3.safetensors
  --videoalign-src DIR       VideoAlign checkout
  --videoalign-ckpt PATH     VideoReward weights
  --depth-anything-3-src DIR Depth-Anything-3 checkout (or set $DEPTH_ANYTHING_3_SRC)
  --da3-model PATH           DA3 weights (default: depth-anything/DA3-GIANT-1.1)
  --skip-score               Only produce predictions, do not score
  -- ARGS...                 Pass everything after -- to the adapter
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)               BASELINE="$2"; shift 2 ;;
    --bench)                  BENCH="$2"; shift 2 ;;
    --out-dir)                OUT_DIR="$2"; shift 2 ;;
    --num-gpus)               NUM_GPUS="$2"; shift 2 ;;
    --eps)                    EPS="$2"; shift 2 ;;
    --limit)                  LIMIT="$2"; shift 2 ;;
    --url)                    URL="$2"; shift 2 ;;
    --served-model-name)      SERVED_MODEL_NAME="$2"; shift 2 ;;
    --clip-path)              CLIP_PATH="$2"; shift 2 ;;
    --aesthetic-mlp)          AESTHETIC_MLP="$2"; shift 2 ;;
    --hpsv3-checkpoint)       HPSV3_CHECKPOINT="$2"; shift 2 ;;
    --videoalign-src)         VIDEOALIGN_SRC="$2"; shift 2 ;;
    --videoalign-ckpt)        VIDEOALIGN_CKPT="$2"; shift 2 ;;
    --depth-anything-3-src)   DA3_SRC="$2"; shift 2 ;;
    --da3-model)              DA3_MODEL="$2"; shift 2 ;;
    --skip-score)             SKIP_SCORE=1; shift ;;
    -h|--help)                usage; exit 0 ;;
    --)                       shift; EXTRA_ARGS+=("$@"); break ;;
    *)                        echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$BASELINE" ]]; then
  echo "--baseline is required" >&2
  usage
  exit 2
fi
if [[ ! -f "$BENCH" ]]; then
  echo "bench file not found: $BENCH" >&2
  echo "Download it first: python scripts/download_bench.py" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Map a baseline name onto its adapter plus the flags only it understands.
adapter_args() {
  case "$1" in
    aesthetic|hpsv3|videoalign)
      ADAPTER="$REPO_ROOT/baselines/run_quality_scorer.py"
      ARGS=(--scorer "$1")
      [[ -n "$CLIP_PATH"        ]] && ARGS+=(--clip-path "$CLIP_PATH")
      [[ -n "$AESTHETIC_MLP"    ]] && ARGS+=(--aesthetic-mlp "$AESTHETIC_MLP")
      [[ -n "$HPSV3_CHECKPOINT" ]] && ARGS+=(--hpsv3-checkpoint "$HPSV3_CHECKPOINT")
      [[ -n "$VIDEOALIGN_SRC"   ]] && ARGS+=(--videoalign-src "$VIDEOALIGN_SRC")
      [[ -n "$VIDEOALIGN_CKPT"  ]] && ARGS+=(--videoalign-ckpt "$VIDEOALIGN_CKPT")
      ;;
    think|flex)
      ADAPTER="$REPO_ROOT/baselines/run_unified_reward.py"
      ARGS=(--variant "$1" --url "$URL" --served-model-name "$SERVED_MODEL_NAME")
      ;;
    dav3)
      ADAPTER="$REPO_ROOT/baselines/run_geometry.py"
      ARGS=()
      [[ -n "$DA3_SRC"   ]] && ARGS+=(--depth-anything-3-src "$DA3_SRC")
      [[ -n "$DA3_MODEL" ]] && ARGS+=(--model-path "$DA3_MODEL")
      ;;
    *)
      echo "unknown baseline: $1 (expected one of ${ALL_BASELINES[*]} or all)" >&2
      exit 2
      ;;
  esac
  [[ -n "$EPS"   ]] && ARGS+=(--eps "$EPS")
  [[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
}

run_one() {
  local name="$1"
  adapter_args "$name"
  local output="$OUT_DIR/${name}.pairs.jsonl"
  mkdir -p "$OUT_DIR"

  echo
  echo "=== $name ($NUM_GPUS GPU(s)) ==="

  if [[ "$NUM_GPUS" -le 1 ]]; then
    python "$ADAPTER" --bench "$BENCH" --output "$output" \
      "${ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  else
    # think/flex are HTTP clients: one server serves every shard, and the shards
    # only differ in which pairs they send.
    local pids=()
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
      CUDA_VISIBLE_DEVICES="$gpu" python "$ADAPTER" \
        --bench "$BENCH" --output "$output" \
        --shard-id "$gpu" --num-shards "$NUM_GPUS" \
        "${ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        >"$OUT_DIR/${name}.shard${gpu}.log" 2>&1 &
      pids+=($!)
    done
    local failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=$((failed + 1))
    done
    if [[ "$failed" -gt 0 ]]; then
      echo "$failed/$NUM_GPUS shards failed; see $OUT_DIR/${name}.shard*.log" >&2
      # Still merge: the surviving shards hold real results, and score.py reports
      # the missing pairs rather than counting them as wrong.
    fi
    python "$ADAPTER" --merge --bench "$BENCH" --output "$output" "${ARGS[@]}"
  fi

  if [[ "$SKIP_SCORE" -eq 0 ]]; then
    python "$REPO_ROOT/worldreward-bench/score.py" \
      --bench "$BENCH" \
      --predictions "$output" \
      --name "$name" \
      --output-md "$OUT_DIR/${name}.report.md" \
      --output-json "$OUT_DIR/${name}.metrics.json"
  fi
}

if [[ "$BASELINE" == "all" ]]; then
  for name in "${ALL_BASELINES[@]}"; do
    run_one "$name"
  done
else
  run_one "$BASELINE"
fi
