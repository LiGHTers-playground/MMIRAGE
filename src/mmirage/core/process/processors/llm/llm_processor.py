"""LLM processor implementation using SGLang with multimodal support."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, List, override

import jinja2
from pydantic import BaseModel, ValidationError

try:
    import sglang as sgl

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

    class DummySGL:
        class Engine:
            pass

    sgl = DummySGL

from transformers import AutoTokenizer

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry, TokenCounts
from mmirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig
from mmirage.core.process.variables import VariableEnvironment

logger = logging.getLogger(__name__)

# Cap on raw generations echoed into WARNING logs; the full text goes to DEBUG.
RAW_OUTPUT_LOG_LIMIT = 500

# Common image tokens for known templates
IMAGE_TOKENS = {
    "qwen2-vl": "<|vision_start|><|image_pad|><|vision_end|>",
    "llava": "<image>",
    "internvl": "<image>",
    "phi3_v": "<|image_1|>",
}


@ProcessorRegistry.register("llm", SGLangLLMConfig, LLMOutputVar)
class LLMProcessor(BaseProcessor[LLMOutputVar]):
    """LLM processor for generating text using SGLang.

    Supports both plain text and JSON output formats, with automatic
    chat template formatting and structured output validation.

    Also supports multimodal (vision-language) inputs. For SGLang, `image_data`
    is expected to be *aligned with prompts*: a list where each element is
    either None (text-only), a single image (path/URL/PIL), or (optionally)
    a list of images for that prompt.

    Attributes:
        llm: SGLang engine for text generation.
        tokenizer: Hugging Face tokenizer for chat template formatting.
        sampling_params: Default sampling parameters for generation.
    """

    def __init__(
        self, engine_args: SGLangLLMConfig, shard_id: int = 0, **kwargs
    ) -> None:
        """Initialize the LLM processor.

        Args:
            engine_args: SGLang runtime configuration.
            shard_id: Shard index for this worker.
        """
        super().__init__(engine_args, shard_id=shard_id, **kwargs)

        if not SGLANG_AVAILABLE:
            raise RuntimeError(
                "SGLang is not installed. Install with: pip install 'mmirage[gpu]' "
                "or, from a source checkout, pip install -e '.[gpu]'"
            )

        server_kwargs = asdict(engine_args.server_args)
        extra = server_kwargs.pop("extra_engine_args", {}) or {}
        server_kwargs.update(extra)
        _load_start = time.monotonic()
        self.llm = sgl.Engine(**server_kwargs)
        self._model_load_seconds = time.monotonic() - _load_start
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.server_args.model_path,
            trust_remote_code=getattr(
                engine_args.server_args, "trust_remote_code", False
            ),
        )

        self.sampling_params = engine_args.default_sampling_params
        self.chat_template = engine_args.chat_template

        # Cumulative token counts across all generate() calls in this processor's lifetime.
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def get_load_time(self) -> float:
        """Return the wall-clock seconds spent initializing the SGLang engine."""
        return self._model_load_seconds

    def get_token_counts(self) -> TokenCounts:
        """Return cumulative token counts for this processor.

        Returns:
            TokenCounts object containing input and output token counts accumulated since this processor was created.
        """
        return TokenCounts(
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
        )

    def _accumulate_tokens(self, outputs: list) -> None:
        """Add token counts from a list of SGLang generate() outputs."""
        for out in outputs:
            meta = out.get("meta_info") or {}
            self._total_input_tokens += int(meta.get("prompt_tokens") or 0)
            self._total_output_tokens += int(meta.get("completion_tokens") or 0)

    def build_prompt(
        self, prompt_template: str, vars_samples: List[VariableEnvironment]
    ) -> List[str]:
        """Build formatted prompts from a Jinja2 template and variable environments.

        Args:
            prompt_template: Jinja2 template string for the prompt.
            vars_samples: List of variable environments containing values.

        Returns:
            List of formatted prompts with chat template applied.
        """
        prompts_for_output: List[str] = []
        jinja_template = jinja2.Template(prompt_template)

        for var in vars_samples:
            user_prompt = [
                {"role": "user", "content": jinja_template.render(**var.to_dict())}
            ]
            formatted = self.tokenizer.apply_chat_template(
                user_prompt, tokenize=False, add_generation_prompt=True
            )
            prompts_for_output.append(formatted)

        return prompts_for_output

    def _get_image_token(self) -> str:
        """Get the image token for the current chat template."""
        if not self.chat_template:
            return "<image>"

        # Import chat templates from sglang if available
        try:
            from sglang.srt.conversation import chat_templates

            if self.chat_template in chat_templates:
                conv = chat_templates[self.chat_template].copy()
                return conv.image_token
        except Exception:
            pass

        return IMAGE_TOKENS.get(self.chat_template, "<image>")

    @staticmethod
    def _warn_on_schema_violation(
        name: str, model: type[BaseModel] | None, value: Any
    ) -> None:
        """Warn when a parsed JSON output does not match its declared schema."""
        if model is None:
            return

        try:
            model.model_validate(value)
        except ValidationError as exc:
            details = []
            for err in exc.errors():
                location = ".".join(str(part) for part in err["loc"]) or "<root>"
                if err["type"] == "missing":
                    details.append(f"{location}: {err['msg']}")
                else:
                    details.append(f"{location}={err.get('input')!r}: {err['msg']}")
            logger.warning(
                f"Schema validation failed for '{name}'; keeping parsed value. "
                + "; ".join(details)
            )

    def _decode_json_value(
        self, name: str, raw: str, constraint_model: type[BaseModel] | None
    ) -> Any:
        """Parse a JSON generation, falling back to an empty dict if unusable."""
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            preview = raw[:RAW_OUTPUT_LOG_LIMIT]
            suffix = (
                ""
                if len(raw) <= RAW_OUTPUT_LOG_LIMIT
                else f" (truncated, {len(raw)} chars)"
            )
            logger.warning(
                f"Failed to parse JSON output for '{name}'; "
                f"falling back to empty dict. Raw model output: {preview!r}{suffix}"
            )
            logger.debug("Full unparsable output for '%s': %r", name, raw)
            return {}
        self._warn_on_schema_violation(name, constraint_model, value)
        return value

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: LLMOutputVar
    ) -> List[VariableEnvironment]:
        """Process a batch of variable environments to generate LLM outputs.

        Args:
            batch: List of variable environments to process.
            output_var: Output variable defining prompt and output format.

        Returns:
            List of updated variable environments with LLM-generated values.

        Raises:
            ValueError: If output_type is JSON but no output_schema is defined.
            RuntimeError: If output batch size doesn't match input batch size.
        """
        nb_samples = len(batch)

        # Prepare sampling params
        sampling_params_output = self.sampling_params.copy()

        constraint_model = None
        if output_var.output_type == "JSON":
            json_schema = output_var.get_output_schema()
            if json_schema is None:
                raise ValueError(
                    f"Output variable {output_var.name} has output_type=JSON but no output_schema defined."
                )
            sampling_params_output["json_schema"] = json.dumps(
                json_schema.model_json_schema()
            )
            if output_var.has_schema_constraints():
                constraint_model = json_schema

        # Separate samples into text-only and multimodal groups
        text_only_indices: List[int] = []
        multimodal_indices: List[int] = []
        for i in range(nb_samples):
            if batch[i].has_images():
                multimodal_indices.append(i)
            else:
                text_only_indices.append(i)

        results: dict[int, VariableEnvironment] = {}

        # Text-only batch
        if text_only_indices:
            text_only_envs = [batch[i] for i in text_only_indices]
            text_only_prompts = self.build_prompt(output_var.prompt, text_only_envs)

            try:
                text_only_outputs = self.llm.generate(
                    prompt=text_only_prompts,
                    sampling_params=sampling_params_output,
                )

                if not isinstance(text_only_outputs, list) or len(
                    text_only_outputs
                ) != len(text_only_indices):
                    raise RuntimeError(
                        f"Mismatch between text-only prompts and outputs for '{output_var.name}': "
                        f"{len(text_only_prompts)} vs "
                        f"{len(text_only_outputs) if isinstance(text_only_outputs, list) else 'non-list'}"
                    )

                self._accumulate_tokens(text_only_outputs)

                for local_idx, global_i in enumerate(text_only_indices):
                    value = text_only_outputs[local_idx].get("text", "").strip()
                    if output_var.output_type == "JSON":
                        value = self._decode_json_value(
                            output_var.name, value, constraint_model
                        )
                    results[global_i] = batch[global_i].with_variable(
                        output_var.name, value
                    )

            except Exception as e:
                logger.error(
                    f"Batch generation failed for text-only samples in output '{output_var.name}': {e}"
                )
                for global_i in text_only_indices:
                    empty_val = {} if output_var.output_type == "JSON" else ""
                    results[global_i] = batch[global_i].with_variable(
                        output_var.name, empty_val
                    )

        # Multimodal batch
        if multimodal_indices:
            image_token = self._get_image_token()
            jinja_template = jinja2.Template(output_var.prompt)

            multimodal_prompts: List[str] = []
            multimodal_image_data: List[Any] = []

            for global_i in multimodal_indices:
                var_env = batch[global_i]
                base_prompt = jinja_template.render(**var_env.to_dict())

                # Format prompt with chat template
                user_prompt = [{"role": "user", "content": base_prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    user_prompt, tokenize=False, add_generation_prompt=True
                )

                # Append image token (common pattern for VL templates)
                formatted_prompt = f"{formatted_prompt}\n{image_token}\n"
                multimodal_prompts.append(formatted_prompt)

                # `image_data` must be aligned 1:1 with prompts.
                imgs = var_env.get_images()
                if not imgs:
                    multimodal_image_data.append(None)
                elif len(imgs) == 1:
                    multimodal_image_data.append(imgs[0])
                else:
                    multimodal_image_data.append(imgs)

            try:
                multimodal_outputs = self.llm.generate(
                    prompt=multimodal_prompts,
                    sampling_params=sampling_params_output,
                    image_data=multimodal_image_data,
                )

                if not isinstance(multimodal_outputs, list) or len(
                    multimodal_outputs
                ) != len(multimodal_indices):
                    raise RuntimeError(
                        f"Mismatch between multimodal prompts and outputs for '{output_var.name}': "
                        f"{len(multimodal_prompts)} vs "
                        f"{len(multimodal_outputs) if isinstance(multimodal_outputs, list) else 'non-list'}"
                    )

                self._accumulate_tokens(multimodal_outputs)

                for local_idx, global_i in enumerate(multimodal_indices):
                    value = multimodal_outputs[local_idx].get("text", "").strip()
                    if output_var.output_type == "JSON":
                        value = self._decode_json_value(
                            output_var.name, value, constraint_model
                        )
                    results[global_i] = batch[global_i].with_variable(
                        output_var.name, value
                    )

            except Exception as e:
                logger.error(
                    f"Batch generation failed for multimodal samples in output '{output_var.name}': {e}"
                )
                for global_i in multimodal_indices:
                    empty_val = {} if output_var.output_type == "JSON" else ""
                    results[global_i] = batch[global_i].with_variable(
                        output_var.name, empty_val
                    )

        return [results[i] for i in range(nb_samples)]

    def shutdown(self) -> None:
        """Shutdown the LLM engine."""
        if self.llm is None:
            return

        try:
            self.llm.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down LLM: {e}")
