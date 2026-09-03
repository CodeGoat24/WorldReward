"""Inference: run WorldReward over rendered chunks.

Two backends are provided:

:class:`OfflineRunner`
    Loads ``vllm.LLM`` in-process and submits every pending chunk as one batched
    ``llm.chat`` call. This is the fastest option and the one used to produce the
    published numbers. One process per GPU (set ``CUDA_VISIBLE_DEVICES``).

:class:`ServerRunner`
    Talks to an OpenAI-compatible endpoint (``scripts/launch_vllm_server.sh``).
    Useful when the model is already served, or served on another machine.

Both write a chunk-level cache to ``chunk_predictions.json`` and skip chunks that
already parsed successfully, so an interrupted run resumes cheaply and a failed
generation is retried on the next invocation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from .parse import parse_response
from .prompt import build_prompt

# vLLM defaults that reproduce the published evaluation.
DEFAULT_MODEL = "CodeGoat24/WorldReward-qwen35-9b"
DEFAULT_MAX_MODEL_LEN = 49152
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BATCH_SIZE = 32
IMAGES_PER_PROMPT = 6

# Images larger than this are downscaled once and cached before inference.
COMPRESS_MAX_SIDE = 2048
COMPRESS_QUALITY = 92


class _ImageCache:
    """Caches downscaled copies of oversized images on disk, keyed by path."""

    def __init__(self, cache_dir: str | Path, max_side: int = COMPRESS_MAX_SIDE) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_side = max_side
        self._resolved: dict[str, Path] = {}

    def get(self, image_path: str | Path) -> Path:
        image_path = Path(image_path)
        key = str(image_path.resolve())
        cached = self._resolved.get(key)
        if cached is not None:
            return cached

        resolved = image_path
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                if max(width, height) > self.max_side:
                    scale = self.max_side / max(width, height)
                    resized = image.convert("RGB").resize(
                        (round(width * scale), round(height * scale)), Image.LANCZOS
                    )
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.md5(key.encode()).hexdigest()[:16]
                    out_path = self.cache_dir / f"{digest}.jpg"
                    resized.save(out_path, "JPEG", quality=COMPRESS_QUALITY)
                    resolved = out_path
        except OSError:
            # Unreadable image: pass the original through, the backend will error.
            resolved = image_path

        self._resolved[key] = resolved
        return resolved


def chunk_key(pair_id: str, chunk_id: int) -> str:
    return f"{pair_id}|chunk_{int(chunk_id):02d}"


def load_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a ``chunk_predictions.json`` into a ``chunk_key -> record`` map.

    The key is recomputed from ``pair_id`` / ``chunk_id`` rather than read from
    the record, so a stale stored ``chunk_key`` cannot mis-key the cache.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {chunk_key(record["pair_id"], record["chunk_id"]): record for record in records}


def write_cache(path: str | Path, cache: dict[str, dict[str, Any]]) -> None:
    """Write the cache back as a flat list, sorted and indented for stable diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(cache.values(), key=lambda r: (r["pair_id"], r["chunk_id"]))
    path.write_text(json.dumps(records, ensure_ascii=False, indent=4), encoding="utf-8")


def build_record(
    chunk_info: dict[str, Any],
    raw_response: str | None,
    image_paths: Sequence[Path],
    elapsed_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    """Normalize one generation into the on-disk chunk record schema.

    ``prediction_text`` is kept only when parsing failed; on success it would
    just duplicate ``review_payload``.
    """
    payload: dict[str, Any] | None = None
    if raw_response is not None:
        payload, parse_error = parse_response(raw_response)
        error = error or parse_error

    action_control = (payload or {}).get("action_control") or {}
    visual_quality = (payload or {}).get("visual_quality") or {}
    return {
        "pair_id": chunk_info["pair_id"],
        "chunk_id": int(chunk_info["chunk_id"]),
        "chunk_key": chunk_key(chunk_info["pair_id"], chunk_info["chunk_id"]),
        "action_indices": list(chunk_info.get("action_indices") or []),
        "action_labels": list(chunk_info.get("action_labels") or []),
        "action_texts": list(chunk_info.get("action_texts") or []),
        "image_paths": [str(p) for p in image_paths],
        "response_seconds": round(elapsed_seconds, 3),
        "parsed_ok": payload is not None,
        "parse_error": error,
        "prediction_text": None if payload is not None else raw_response,
        "action_winners": [item.get("winner") for item in action_control.get("actions") or []],
        "action_overall_winner": action_control.get("overall_winner"),
        "visual_overall_winner": visual_quality.get("overall_winner"),
        "review_payload": payload,
    }


class _BaseRunner:
    """Shared caching / batching logic for both backends."""

    def __init__(self, *, image_cache_dir: str | Path = "/tmp/worldreward_images") -> None:
        self._images = _ImageCache(image_cache_dir)

    def _prompt_inputs(self, chunk_info: dict[str, Any]) -> tuple[list[Path], str]:
        images = [self._images.get(p) for p in chunk_info["images"]]
        if len(images) != IMAGES_PER_PROMPT:
            raise ValueError(
                f"{chunk_info['pair_id']} chunk {chunk_info['chunk_id']}: "
                f"expected {IMAGES_PER_PROMPT} images, got {len(images)}"
            )
        prompt = build_prompt(chunk_info["caption"], list(chunk_info["action_texts"]))
        return images, prompt

    def _pending(
        self,
        chunk_infos: Sequence[dict[str, Any]],
        cache: dict[str, dict[str, Any]],
        force: bool,
    ) -> list[dict[str, Any]]:
        if force:
            return list(chunk_infos)
        pending = []
        for info in chunk_infos:
            cached = cache.get(chunk_key(info["pair_id"], info["chunk_id"]))
            if cached is None or not cached.get("parsed_ok"):
                pending.append(info)
        return pending

    def run(
        self,
        chunk_infos: Iterable[dict[str, Any]],
        *,
        output_path: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        force: bool = False,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Score every chunk, reusing and updating the cache at ``output_path``.

        Args:
            chunk_infos: ``group_info`` dicts from :func:`worldreward.render_pair`.
            output_path: where to read/write ``chunk_predictions.json``. When
                ``None`` nothing is persisted.
            batch_size: chunks per backend call. Larger batches are more
                efficient; smaller batches checkpoint progress sooner.
            force: re-run chunks even if they already parsed successfully.

        Returns:
            All chunk records, cached and freshly generated, in pair/chunk order.
        """
        chunk_infos = list(chunk_infos)
        cache = load_cache(output_path) if output_path else {}
        pending = self._pending(chunk_infos, cache, force)

        if verbose:
            print(f"[worldreward] {len(chunk_infos)} chunks, {len(pending)} to infer")

        step = batch_size if batch_size and batch_size > 0 else max(len(pending), 1)
        for start in range(0, len(pending), step):
            batch = pending[start : start + step]
            started = time.time()
            for record in self._run_batch(batch):
                cache[record["chunk_key"]] = record
            if output_path:
                write_cache(output_path, cache)
            if verbose:
                ok = sum(1 for info in batch if cache[chunk_key(info["pair_id"], info["chunk_id"])]["parsed_ok"])
                print(
                    f"[worldreward] batch {start // step + 1}: {len(batch)} chunks "
                    f"in {time.time() - started:.1f}s, parsed_ok={ok}/{len(batch)}",
                    flush=True,
                )

        wanted = {chunk_key(info["pair_id"], info["chunk_id"]) for info in chunk_infos}
        records = [cache[key] for key in wanted if key in cache]
        return sorted(records, key=lambda r: (r["pair_id"], r["chunk_id"]))

    def _run_batch(self, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError


class OfflineRunner(_BaseRunner):
    """In-process vLLM backend. Requires a GPU and a vLLM build supporting the Qwen3.5 architecture."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_num_seqs: int | None = None,
        image_cache_dir: str | Path = "/tmp/worldreward_images",
        **llm_kwargs: Any,
    ) -> None:
        super().__init__(image_cache_dir=image_cache_dir)
        from vllm import LLM, SamplingParams

        kwargs: dict[str, Any] = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": IMAGES_PER_PROMPT},
            disable_custom_all_reduce=True,
            enable_prefix_caching=True,
            # Images are passed as file:// URLs, so vLLM must be allowed to read
            # them from the local filesystem.
            allowed_local_media_path="/",
        )
        if max_num_seqs:
            kwargs["max_num_seqs"] = max_num_seqs
        kwargs.update(llm_kwargs)

        self.llm = LLM(**kwargs)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def _run_batch(self, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        conversations = []
        image_paths_per_chunk = []
        for info in batch:
            images, prompt = self._prompt_inputs(info)
            image_paths_per_chunk.append(images)
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"file://{p.resolve()}"}} for p in images
            ]
            content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])

        started = time.time()
        outputs = self.llm.chat(
            conversations,
            sampling_params=self.sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        per_chunk_seconds = (time.time() - started) / max(len(batch), 1)

        return [
            build_record(
                info,
                output.outputs[0].text if output.outputs else None,
                images,
                per_chunk_seconds,
            )
            for info, output, images in zip(batch, outputs, image_paths_per_chunk)
        ]


class ServerRunner(_BaseRunner):
    """OpenAI-compatible HTTP backend.

    Args:
        base_url: e.g. ``http://127.0.0.1:8080/v1``.
        model: the ``--served-model-name`` of the endpoint.
        use_file_url: send ``file://`` references instead of inlined base64. Only
            works when the server runs on this machine and was started with
            ``--allowed-local-media-path``. Much cheaper than base64.
        max_workers: number of concurrent in-flight requests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        *,
        model: str = "WorldReward",
        api_key: str = "EMPTY",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 600.0,
        use_file_url: bool = False,
        max_workers: int = 8,
        image_cache_dir: str | Path = "/tmp/worldreward_images",
    ) -> None:
        super().__init__(image_cache_dir=image_cache_dir)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.use_file_url = use_file_url
        self.max_workers = max(1, max_workers)
        self._base64_cache: dict[str, str] = {}

    def _image_url(self, image_path: Path) -> str:
        if self.use_file_url:
            return f"file://{image_path.resolve()}"
        key = str(image_path)
        encoded = self._base64_cache.get(key)
        if encoded is None:
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            self._base64_cache[key] = encoded
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        return f"data:{mime};base64,{encoded}"

    def _request(self, info: dict[str, Any]) -> dict[str, Any]:
        import requests

        images, prompt = self._prompt_inputs(info)
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": self._image_url(p)}} for p in images
        ]
        content.append({"type": "text", "text": prompt})

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return build_record(info, text, images, time.time() - started)
        except Exception as exc:
            return build_record(info, None, images, time.time() - started, error=f"request_failed: {exc}")

    def _run_batch(self, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._request, batch))
