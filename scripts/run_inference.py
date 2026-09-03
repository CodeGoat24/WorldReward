#!/usr/bin/env python3
"""Score rendered chunks with WorldReward.

Offline backend (recommended, one process per GPU):

    CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py \
        --render-root outputs/rendered_chunks \
        --output outputs/chunk_predictions.json \
        --model CodeGoat24/WorldReward-qwen35-9b

Server backend (against ``scripts/launch_vllm_server.sh``):

    python scripts/run_inference.py \
        --render-root outputs/rendered_chunks \
        --output outputs/chunk_predictions.json \
        --backend server --base-url http://127.0.0.1:8080/v1

Writes two files:

``--output``
    one record per chunk, including the raw generation and parsed review. Doubles
    as a resume cache: re-running skips chunks that already parsed.
``--output`` with a ``.pairs.jsonl`` suffix
    one aggregated ``left``/``right``/``tie`` verdict per pair per dimension.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldreward.aggregate import DIMENSIONS, predict_pair
from worldreward.infer import DEFAULT_BATCH_SIZE, DEFAULT_MAX_MODEL_LEN, DEFAULT_MAX_TOKENS, DEFAULT_MODEL


def discover_chunks(render_root: Path, pair_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Collect every ``group_info.json`` under ``render_root``, in stable order."""
    chunk_infos: list[dict[str, Any]] = []
    for pair_dir in sorted(p for p in render_root.iterdir() if p.is_dir()):
        if pair_ids and pair_dir.name not in pair_ids:
            continue
        for chunk_dir in sorted(c for c in pair_dir.iterdir() if c.is_dir()):
            info_path = chunk_dir / "group_info.json"
            if not info_path.is_file():
                continue
            info = json.loads(info_path.read_text(encoding="utf-8"))
            # Rebase image paths so a rendering produced elsewhere still resolves.
            info["images"] = [
                str(p if (p := Path(recorded)).is_file() else chunk_dir / Path(recorded).name)
                for recorded in info["images"]
            ]
            chunk_infos.append(info)
    return chunk_infos


def write_pair_predictions(path: Path, records: list[dict[str, Any]]) -> int:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record)

    lines = []
    for pair_id in sorted(by_pair):
        chunks = by_pair[pair_id]
        prediction = predict_pair(c.get("review_payload") or {} for c in chunks)
        lines.append(
            json.dumps(
                {
                    "pair_id": pair_id,
                    **{dimension: prediction[dimension] for dimension in DIMENSIONS},
                    "chunks": len(chunks),
                    "chunks_parsed_ok": sum(1 for c in chunks if c.get("parsed_ok")),
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render-root", required=True, help="Directory produced by scripts/preprocess.py")
    parser.add_argument("--output", required=True, help="Path to write chunk_predictions.json")
    parser.add_argument("--backend", choices=("offline", "server"), default="offline")
    parser.add_argument("--pair-id", action="append", default=[], help="Score only this pair_id (repeatable)")
    parser.add_argument(
        "--pair-id-file",
        default=None,
        help="Text file with one pair_id per line; scored in addition to any --pair-id. "
        "Useful for sharding a run across GPUs.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Re-run chunks that already parsed successfully")

    offline = parser.add_argument_group("offline backend")
    offline.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local checkpoint path")
    offline.add_argument("--tensor-parallel-size", type=int, default=1)
    offline.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    offline.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    offline.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    offline.add_argument("--max-num-seqs", type=int, default=0, help="0 = let vLLM decide")

    server = parser.add_argument_group("server backend")
    server.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    server.add_argument("--served-model-name", default="WorldReward")
    server.add_argument(
        "--use-file-url",
        action="store_true",
        help="Send file:// paths instead of base64. Requires a same-host server started "
        "with --allowed-local-media-path.",
    )
    server.add_argument("--max-workers", type=int, default=8, help="Concurrent in-flight requests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_root = Path(args.render_root).resolve()
    output_path = Path(args.output).resolve()

    wanted = set(args.pair_id)
    if args.pair_id_file:
        wanted |= {
            line.strip()
            for line in Path(args.pair_id_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    chunk_infos = discover_chunks(render_root, wanted or None)
    if not chunk_infos:
        print(f"no group_info.json found under {render_root}", file=sys.stderr)
        return 1
    pair_count = len({info["pair_id"] for info in chunk_infos})
    print(f"found {len(chunk_infos)} chunks across {pair_count} pairs")

    if args.backend == "offline":
        from worldreward.infer import OfflineRunner

        runner = OfflineRunner(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            max_num_seqs=args.max_num_seqs or None,
        )
    else:
        from worldreward.infer import ServerRunner

        runner = ServerRunner(
            base_url=args.base_url,
            model=args.served_model_name,
            use_file_url=args.use_file_url,
            max_workers=args.max_workers,
        )

    records = runner.run(
        chunk_infos,
        output_path=output_path,
        batch_size=args.batch_size,
        force=args.force,
    )

    parsed_ok = sum(1 for r in records if r.get("parsed_ok"))
    pairs_path = output_path.with_suffix(".pairs.jsonl")
    n_pairs = write_pair_predictions(pairs_path, records)

    print(
        json.dumps(
            {
                "chunks": len(records),
                "chunks_parsed_ok": parsed_ok,
                "pairs": n_pairs,
                "chunk_predictions": str(output_path),
                "pair_predictions": str(pairs_path),
            },
            indent=2,
        )
    )
    if parsed_ok < len(records):
        print(
            f"\n{len(records) - parsed_ok} chunks failed to parse. "
            f"Re-run the same command to retry only those chunks.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
