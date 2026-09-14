"""Image generation processor implementation using pluggable backends."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import socket
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, override

import jinja2

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry, TokenCounts
from mmirage.core.process.processors.image_gen.backends.base import (
    ImageGenerationBackend,
)
from mmirage.core.process.processors.image_gen.config import (
    ImageGenConfig,
    ImageGenOutputVar,
    ImageOutputMode,
)
from mmirage.core.process.variables import VariableEnvironment

logger = logging.getLogger(__name__)

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RECOVERABLE_IMAGE_GEN_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    jinja2.TemplateError,
)


def _sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename stem."""
    normalized = _SAFE_FILENAME_RE.sub("_", filename).strip("._")
    return normalized or "image"


def _create_backend(config: ImageGenConfig) -> ImageGenerationBackend:
    """Instantiate the configured image generation backend."""
    from mmirage.core.process.processors.image_gen.backends.sglang_backend import (
        SGLangImageBackend,
    )

    if config.backend == "external":
        assert config.external is not None  # validated in __post_init__
        client = config.external
    elif config.backend == "sglang":
        from mmirage.core.process.processors.image_gen.sglang_server import (
            MMIRAGE_SGLANG_BASE_URL,
        )

        assert config.sglang is not None  # validated in __post_init__
        client = config.sglang
        base_url = os.environ.get(MMIRAGE_SGLANG_BASE_URL)
        if not base_url:
            raise RuntimeError(
                "backend='sglang' requires the MMIRAGE runner to launch the "
                "shared SGLang server first. No MMIRAGE_SGLANG_BASE_URL was provided."
            )
    else:
        raise ValueError(f"Unknown image_gen backend={config.backend!r}")

    return SGLangImageBackend(
        base_url=client.base_url if config.backend == "external" else base_url,
        api_key=client.api_key,
        timeout_seconds=client.timeout_seconds,
        request_model=client.request_model,
        max_concurrent_requests=client.max_concurrent_requests,
    )


@ProcessorRegistry.register("image_gen", ImageGenConfig, ImageGenOutputVar)
class ImageGenProcessor(BaseProcessor[ImageGenOutputVar]):
    """Processor that generates images from prompts using a pluggable backend.

    Supported backends: ``external`` and ``sglang`` HTTP servers.

    Responsibilities of this processor:
    - Render Jinja2 prompt and filename templates.
    - Compute deterministic, shard-aware seeds.
    - Chunk batches and call the backend's ``generate_batch`` method.
    - Fall back to per-sample sequential generation if a batch chunk fails.
    - Save images atomically to disk (``output_mode="path"``) or pass PIL
      images through directly (``output_mode="pil"``).
    """

    def __init__(self, config: ImageGenConfig, shard_id: int = 0, **kwargs) -> None:
        super().__init__(config, shard_id=shard_id, **kwargs)

        _load_start = time.monotonic()
        self._backend: ImageGenerationBackend = _create_backend(config)
        self._model_load_seconds = time.monotonic() - _load_start
        self._default_sampling_params = dict(config.default_sampling_params)
        self._parallel_inference = config.parallel_inference
        self._parallel_chunk_size = config.parallel_chunk_size

        self._output_dir = config.get_output_dir()
        self._file_format = config.file_format.lower()
        os.makedirs(self._output_dir, exist_ok=True)

        self._shard_id = shard_id
        # Counts the total number of samples processed by this instance.
        # Used to derive shard-local sample indices for filenames and seeds.
        self._sample_counter = 0
        run_token = uuid.uuid4().hex[:8]
        self._run_id = f"{socket.gethostname()}.{os.getpid()}.{run_token}"

    # ------------------------------------------------------------------
    # Seed and param helpers
    # ------------------------------------------------------------------

    def _compute_seeds(
        self,
        base_seed: int,
        batch_offset: int,
        count: int,
    ) -> List[int]:
        """Compute per-sample deterministic seeds that are unique across shards.

        The seed for a sample is:

            ``base_seed + shard_id * 1_000_000_000 + sample_counter + batch_offset + i``

        This guarantees that different shards with the same ``base_seed``
        produce different images even when their local sample indices overlap.

        Args:
            base_seed: The ``seed`` value from the output variable config.
            batch_offset: Position of the first sample in this call within the
                current mapper batch (0 for the first chunk).
            count: Number of seeds to produce.
        """
        base = (
            base_seed
            + self._shard_id * 1_000_000_000
            + self._sample_counter
            + batch_offset
        )
        return [base + i for i in range(count)]

    def _build_params(self, output_var: ImageGenOutputVar) -> Dict[str, Any]:
        """Build the generation kwargs dict from config defaults and per-var overrides."""
        params = dict(self._default_sampling_params)
        if output_var.width is not None:
            params["width"] = output_var.width
        if output_var.height is not None:
            params["height"] = output_var.height
        if output_var.num_inference_steps is not None:
            params["num_inference_steps"] = output_var.num_inference_steps
        if output_var.guidance_scale is not None:
            params["guidance_scale"] = output_var.guidance_scale
        return params

    # ------------------------------------------------------------------
    # Filename and file saving helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_source_hash(env: VariableEnvironment) -> str:
        """Return an 8-character SHA-256 hex digest of all input variable values."""
        items = sorted(env.to_dict().items())
        if not items:
            return hashlib.sha256(b"empty").hexdigest()[:8]
        return hashlib.sha256(str(items).encode()).hexdigest()[:8]

    def _render_filename(
        self,
        filename_template: jinja2.Template,
        output_var: ImageGenOutputVar,
        env: VariableEnvironment,
        sample_index: int,
    ) -> str:
        """Render the output filename stem and return ``stem.ext``.

        ``sample_index`` is the shard-local index (``self._sample_counter`` +
        position within the current batch).  Combine with ``__shard_id`` in
        the template for globally unique filenames.
        """
        context = dict(env.to_dict())
        context["__sample_index"] = sample_index
        context["__output_name"] = output_var.name
        context["__shard_id"] = self._shard_id
        context["__source_hash"] = self._compute_source_hash(env)
        stem = _sanitize_filename(filename_template.render(**context))
        return f"{stem}.{self._file_format}"

    def _save_image(self, image: Any, filename: str) -> str:
        """Persist a PIL image atomically and return the absolute path."""
        stem, ext = os.path.splitext(filename)
        path = os.path.join(self._output_dir, filename)
        if os.path.exists(path):
            path = os.path.join(self._output_dir, f"{stem}.{self._run_id}{ext}")

        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._output_dir, suffix=ext)
        saved = False
        try:
            os.close(tmp_fd)
            image.save(tmp_path)
            os.replace(tmp_path, path)
            saved = True
        finally:
            if not saved:
                try:
                    os.unlink(tmp_path)
                except OSError as cleanup_err:
                    logger.warning(
                        "Failed to clean up temp file %r: %s",
                        tmp_path,
                        cleanup_err,
                    )
        return path

    # ------------------------------------------------------------------
    # Chunk-level generation
    # ------------------------------------------------------------------

    def _collect_results(
        self,
        chunk: List[VariableEnvironment],
        images: List[Any],
        output_var: ImageGenOutputVar,
        filename_template: jinja2.Template,
        batch_offset: int,
    ) -> List[VariableEnvironment]:
        """Map backend images back to updated VariableEnvironments."""
        updated: List[VariableEnvironment] = []
        for i, (env, image) in enumerate(zip(chunk, images)):
            sample_index = self._sample_counter + batch_offset + i
            if output_var.output_mode == ImageOutputMode.PIL:
                value = image
            else:
                filename = self._render_filename(
                    filename_template, output_var, env, sample_index
                )
                value = self._save_image(image, filename)
            updated.append(env.with_variable(output_var.name, value, is_image=True))
        return updated

    def _generate_chunk_batch(
        self,
        chunk: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        filename_template: jinja2.Template,
        batch_offset: int,
    ) -> List[VariableEnvironment]:
        """Generate an entire chunk with a single batched backend call."""
        prompts = [prompt_template.render(**env.to_dict()) for env in chunk]
        neg_prompts: Optional[List[Optional[str]]] = (
            [negative_prompt_template.render(**env.to_dict()) for env in chunk]
            if negative_prompt_template is not None
            else None
        )
        seeds: List[Optional[int]] = (
            self._compute_seeds(int(output_var.seed), batch_offset, len(chunk))
            if output_var.seed is not None
            else [None] * len(chunk)
        )
        params = self._build_params(output_var)
        images = self._backend.generate_batch(prompts, neg_prompts, params, seeds)
        return self._collect_results(
            chunk, images, output_var, filename_template, batch_offset
        )

    def _generate_chunk_sequential(
        self,
        chunk: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        filename_template: jinja2.Template,
        batch_offset: int,
    ) -> List[VariableEnvironment]:
        """Generate samples one-by-one, tolerating per-sample failures."""
        updated: List[VariableEnvironment] = []
        params = self._build_params(output_var)

        for i, env in enumerate(chunk):
            sample_index = self._sample_counter + batch_offset + i
            try:
                prompt = prompt_template.render(**env.to_dict())
                neg: Optional[str] = (
                    negative_prompt_template.render(**env.to_dict())
                    if negative_prompt_template is not None
                    else None
                )
                seed_val: Optional[int] = (
                    self._compute_seeds(int(output_var.seed), batch_offset + i, 1)[0]
                    if output_var.seed is not None
                    else None
                )
                image = self._backend.generate_one(
                    prompt=prompt,
                    negative_prompt=neg,
                    params=params,
                    seed=seed_val,
                )
                if output_var.output_mode == ImageOutputMode.PIL:
                    value = image
                else:
                    filename = self._render_filename(
                        filename_template, output_var, env, sample_index
                    )
                    value = self._save_image(image, filename)
                updated.append(env.with_variable(output_var.name, value, is_image=True))
            except _RECOVERABLE_IMAGE_GEN_ERRORS as exc:
                logger.error(
                    "Image generation failed for output '%s' at sample %d: %s",
                    output_var.name,
                    sample_index,
                    exc,
                )
                updated.append(env.with_variable(output_var.name, None, is_image=True))

        return updated

    # ------------------------------------------------------------------
    # Batch-level orchestration
    # ------------------------------------------------------------------

    def _batch_process_parallel(
        self,
        batch: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        filename_template: jinja2.Template,
    ) -> List[VariableEnvironment]:
        """Process the full mapper batch in chunks with per-chunk fallback.

        For each chunk the processor tries a single batched backend call.
        If that call fails only the failing chunk is retried sample-by-sample;
        already-successful chunks are never re-generated.
        """
        chunk_size = self._parallel_chunk_size or len(batch)
        updated: List[VariableEnvironment] = []

        for batch_offset in range(0, len(batch), chunk_size):
            chunk = batch[batch_offset : batch_offset + chunk_size]
            try:
                updated.extend(
                    self._generate_chunk_batch(
                        chunk,
                        output_var,
                        prompt_template,
                        negative_prompt_template,
                        filename_template,
                        batch_offset,
                    )
                )
            except _RECOVERABLE_IMAGE_GEN_ERRORS as exc:
                logger.warning(
                    "Batch generation failed for chunk at offset %d "
                    "(samples %d–%d); falling back to sequential for this chunk. "
                    "Reason: %s",
                    batch_offset,
                    self._sample_counter + batch_offset,
                    self._sample_counter + batch_offset + len(chunk) - 1,
                    exc,
                )
                updated.extend(
                    self._generate_chunk_sequential(
                        chunk,
                        output_var,
                        prompt_template,
                        negative_prompt_template,
                        filename_template,
                        batch_offset,
                    )
                )

        self._sample_counter += len(batch)
        return updated

    def _batch_process_sequential(
        self,
        batch: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        filename_template: jinja2.Template,
    ) -> List[VariableEnvironment]:
        """Process all samples one by one (used when parallel_inference=False)."""
        result = self._generate_chunk_sequential(
            batch,
            output_var,
            prompt_template,
            negative_prompt_template,
            filename_template,
            batch_offset=0,
        )
        self._sample_counter += len(batch)
        return result

    # ------------------------------------------------------------------
    # Public processor interface
    # ------------------------------------------------------------------

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: ImageGenOutputVar
    ) -> List[VariableEnvironment]:
        """Generate images for each sample in the batch."""
        prompt_template = jinja2.Template(output_var.prompt)
        negative_prompt_template = (
            jinja2.Template(output_var.negative_prompt)
            if output_var.negative_prompt
            else None
        )
        filename_template = jinja2.Template(output_var.filename_template)

        if self._parallel_inference and len(batch) > 1:
            return self._batch_process_parallel(
                batch,
                output_var,
                prompt_template,
                negative_prompt_template,
                filename_template,
            )
        else:
            return self._batch_process_sequential(
                batch,
                output_var,
                prompt_template,
                negative_prompt_template,
                filename_template,
            )

    @override
    def get_token_counts(self) -> TokenCounts:
        """Return token counts for this processor.

        Image generation does not produce LLM token accounting, so this
        processor always reports zero tokens.
        """
        return TokenCounts(input_tokens=0, output_tokens=0)

    @override
    def get_load_time(self) -> float:
        """Return the time spent initializing the image generation backend."""
        return self._model_load_seconds

    @override
    def shutdown(self) -> None:
        """Release backend resources (GPU memory, HTTP connections, …)."""
        self._backend.shutdown()
