"""Configuration for image generation processor in MMIRAGE."""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from jinja2 import Environment, meta

from mmirage.core.process.base import BaseProcessorConfig, ProcessorRegistry
from mmirage.core.process.variables import BaseVar, OutputVar

logger = logging.getLogger(__name__)
env = Environment()


class ImageOutputMode(str, Enum):
    """Output representation for generated images."""

    PATH = "path"
    PIL = "pil"


@dataclass
class ExternalImageBackendConfig:
    """HTTP client configuration for an already-running image server."""

    base_url: str
    api_key: Optional[str] = None
    timeout_seconds: int = 900
    request_model: Optional[str] = None
    max_concurrent_requests: int = 1

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("external.base_url must be non-empty")
        if self.max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be a positive integer, got {self.max_concurrent_requests!r}."
            )


@dataclass
class SGLangBackendConfig:
    """Configuration for one MMIRAGE-launched shared SGLang server."""

    api_key: Optional[str] = None
    timeout_seconds: int = 900
    model_path: str = ""
    request_model: Optional[str] = None

    port: Optional[int] = None
    num_gpus: int = 1
    dtype: Optional[str] = None
    startup_timeout_seconds: int = 900
    extra_server_args: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 1

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError(
                "backend='sglang' requires model_path to be set so MMIRAGE "
                "knows which model to pass to the SGLang server."
            )
        if self.max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be a positive integer, got {self.max_concurrent_requests!r}."
            )

    def resolved_base_url(self) -> str:
        """Return the HTTP client URL for the shared local server."""
        if self.port is None:
            raise RuntimeError("SGLang server port has not been resolved yet.")
        return f"http://127.0.0.1:{self.port}/v1"


@dataclass
class ImageGenConfig(BaseProcessorConfig):
    """Configuration for the backend-neutral image generation processor.

    Attributes:
        backend: Image generation backend to use.  One of ``"external"`` or
            ``"sglang"``.
        external: HTTP client configuration for ``backend="external"``.
        sglang: SGLang server configuration (used when ``backend="sglang"``).
        default_sampling_params: Default generation kwargs forwarded to every
            pipeline/server call (e.g. ``num_inference_steps``,
            ``guidance_scale``).
        parallel_inference: If ``True``, generate a full chunk of prompts in a
            concurrent server calls. Chunks that fail are retried
            sample-by-sample.
        parallel_chunk_size: Maximum number of samples per batched call.
            ``None`` means use the full mapper batch size.
        output_dir: Directory where generated images are written when
            ``output_mode="path"``.  Supports ``~`` expansion.
        file_format: Image file format for saved outputs (e.g. ``"png"``,
            ``"jpg"``).
    """

    type: Literal["image_gen"] = "image_gen"
    backend: Literal["external", "sglang"] = "external"
    external: Optional[ExternalImageBackendConfig] = None
    sglang: Optional[SGLangBackendConfig] = None
    default_sampling_params: Dict[str, Any] = field(default_factory=dict)
    parallel_inference: bool = True
    parallel_chunk_size: Optional[int] = 4
    output_dir: str = "~/.cache/MMIRAGE/generated_images"
    file_format: str = "png"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend not in ("external", "sglang"):
            raise ValueError(
                f"Unsupported image_gen backend={self.backend!r}. "
                "Choose 'external' or 'sglang'."
            )
        if self.backend == "external" and self.external is None:
            raise ValueError(
                "backend='external' requires an 'external:' configuration block."
            )
        if self.backend == "sglang" and self.sglang is None:
            raise ValueError(
                "backend='sglang' requires a 'sglang:' configuration block."
            )
        if self.parallel_chunk_size is not None and self.parallel_chunk_size <= 0:
            raise ValueError(
                f"parallel_chunk_size must be a positive integer, got {self.parallel_chunk_size!r}. "
                "Set to None to use the full batch size."
            )
        if not self.file_format:
            logger.warning("file_format is empty; defaulting to 'png'.")
            self.file_format = "png"

    def get_output_dir(self) -> str:
        """Return the normalised absolute output directory path."""
        return os.path.abspath(os.path.expanduser(self.output_dir))


@dataclass
class ImageGenOutputVar(OutputVar):
    """Output variable generated by image generation processor.

    Attributes:
        prompt: Jinja2 template used as positive prompt.
        negative_prompt: Optional Jinja2 template used as negative prompt.
        output_mode: Output representation: "path" (default) or "pil".
        filename_template: Optional Jinja2 template used for saved image filename stem.
            Supported internal variables: ``__sample_index`` (shard-local row index,
            i.e. the position within this shard's output — combine with ``__shard_id``
            for global uniqueness), ``__output_name``, ``__shard_id``,
            ``__source_hash`` (8-char SHA-256 of input values).
            All input variables (e.g. ``text``) are also available.
        width: Optional image width override.
        height: Optional image height override.
        num_inference_steps: Optional sampling steps override.
        guidance_scale: Optional guidance scale override.
        seed: Optional deterministic seed. If set, sample index is added for uniqueness.
    """

    prompt: str = ""
    negative_prompt: str = ""
    output_mode: ImageOutputMode = ImageOutputMode.PATH
    filename_template: str = (
        "generated_{{ __shard_id }}_{{ __sample_index }}_{{ __source_hash }}"
    )
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None

    def is_computable(self, vars: List[BaseVar]) -> bool:
        """Check if all variables referenced in templates are available."""
        reserved = {"__sample_index", "__output_name", "__shard_id", "__source_hash"}
        var_names = {v.name for v in vars}

        # Prompt/negative_prompt are rendered from env.to_dict() only — reserved
        # vars are not injected there, so treat them as undeclared in those templates.
        prompt_templates: List[str] = [self.prompt]
        if self.negative_prompt:
            prompt_templates.append(self.negative_prompt)

        undeclared: set[str] = set()
        for template in prompt_templates:
            parsed_content = env.parse(template)
            template_vars = meta.find_undeclared_variables(parsed_content)
            undeclared |= template_vars - var_names

        # filename_template is rendered with reserved vars injected, so allow them.
        if self.filename_template:
            parsed_content = env.parse(self.filename_template)
            template_vars = meta.find_undeclared_variables(parsed_content)
            undeclared |= template_vars - var_names - reserved

        if undeclared:
            logger.warning(
                f"⚠️ Undeclared variables found for {self.name}: {undeclared}"
            )
            return False

        try:
            self.output_mode = ImageOutputMode(self.output_mode)
        except ValueError:
            logger.warning(
                f"⚠️ Invalid output_mode for {self.name}: {self.output_mode}. "
                f"Expected one of {[m.value for m in ImageOutputMode]}"
            )
            return False

        return True


ProcessorRegistry.register_types("image_gen", ImageGenConfig, ImageGenOutputVar)
