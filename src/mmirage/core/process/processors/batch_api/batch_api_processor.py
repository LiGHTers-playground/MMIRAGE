"""Batch API processor implementation for provider batch submission."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import jinja2
from PIL import Image

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry, TokenCounts
from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator
from mmirage.core.process.batch.registry import BatchAdapterRegistry
from mmirage.core.process.processors.batch_api.config import (
    BATCH_API_PROCESSOR_TYPE,
    BatchApiOutputVar,
    BatchApiProcessorConfig,
)
from mmirage.core.process.variables import VariableEnvironment

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover
    from typing_extensions import override  # type: ignore


logger = logging.getLogger(__name__)


@ProcessorRegistry.register(
    BATCH_API_PROCESSOR_TYPE, BatchApiProcessorConfig, BatchApiOutputVar
)
class BatchApiProcessor(BaseProcessor[BatchApiOutputVar]):
    """Processor that submits generation requests to a provider batch API.

    No model runs locally: each sample is serialized into a provider request and
    accumulated by an orchestrator, which uploads chunks and writes metadata
    receipts. Processed samples receive a ``__BATCH_SUBMITTED__`` placeholder
    that the receiver utilities later replace with the provider results.

    Text-only and multimodal requests are accumulated separately so each batch
    job stays homogeneous.
    """

    def __init__(
        self,
        config: BatchApiProcessorConfig,
        shard_id: int = 0,
        **kwargs,
    ) -> None:
        """Initialize the batch API processor.

        Args:
            config: Batch API configuration holding the resolved provider config.
            shard_id: Shard index for this worker.
        """
        super().__init__(config, shard_id=shard_id, **kwargs)

        provider_cfg = config.provider_config
        if provider_cfg is None:
            raise ValueError("batch_api processor requires a provider configuration")

        export_prompts_dir = config.export_prompts_dir
        if export_prompts_dir is not None and not export_prompts_dir.strip():
            raise ValueError(
                "export_prompts_dir is set but empty; give it a path or leave it unset"
            )
        # A real submission needs a receipt for `mmirage merge`; a prompt export does not.
        if export_prompts_dir is None and not provider_cfg.metadata_output_path:
            raise ValueError(
                "metadata_output_path is required to submit batches; set it in the "
                "provider config or export prompts instead with --export-prompts"
            )

        self._batch_provider_config = provider_cfg
        self._export_prompts_dir = export_prompts_dir
        # When export_prompts_dir is set we are in dry-run mode and should not
        # require valid provider credentials because no network calls will be
        # performed.
        self._batch_adapter = BatchAdapterRegistry.create(
            provider_cfg, allow_missing_credentials=bool(export_prompts_dir)
        )
        self._batch_request_counter = 0
        self._global_row_offset = 0
        run_id = uuid.uuid4().hex[:6]
        export_prompts_path = self._resolve_export_prompts_path(
            export_prompts_dir,
            run_id,
        )
        if export_prompts_path:
            self._prepare_export_file(export_prompts_path)

        self._text_orchestrator = BatchSubmissionOrchestrator(
            adapter=self._batch_adapter,
            config=replace(
                provider_cfg,
                metadata_output_path=self._with_metadata_suffix(
                    provider_cfg.metadata_output_path,
                    "text",
                    run_id,
                    dry_run=bool(export_prompts_path),
                ),
            ),
            export_prompts_path=export_prompts_path,
            shard_id=self.shard_id,
            export_batch_prefix="text-",
        )
        self._multimodal_orchestrator = BatchSubmissionOrchestrator(
            adapter=self._batch_adapter,
            config=replace(
                provider_cfg,
                metadata_output_path=self._with_metadata_suffix(
                    provider_cfg.metadata_output_path,
                    "multimodal",
                    run_id,
                    dry_run=bool(export_prompts_path),
                ),
            ),
            export_prompts_path=export_prompts_path,
            shard_id=self.shard_id,
            export_batch_prefix="multimodal-",
        )

    @staticmethod
    def _with_metadata_suffix(
        path: str, suffix: str, run_id: str, dry_run: bool
    ) -> str:
        if not path:
            return ""
        base_path = path.removesuffix(".jsonl")
        # '.dry-run' goes before the suffix, '<base>.<suffix>.*.jsonl' would match it after.
        marker = ".dry-run" if dry_run else ""
        return f"{base_path}{marker}.{suffix}.{run_id}.jsonl"

    @staticmethod
    def _prepare_export_file(path: str) -> None:
        """Open the export file now so an unusable path fails before the dataset runs."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            raise ValueError(f"Cannot write exported prompts to {path}: {exc}") from exc

    @staticmethod
    def _resolve_export_prompts_path(path: Optional[str], run_id: str) -> Optional[str]:
        if not path:
            return None
        # Both forms carry the run id, one run per file.
        if path.endswith(".jsonl"):
            return f"{path.removesuffix('.jsonl')}.{run_id}.jsonl"
        return os.path.join(path, f"exported_prompts.{run_id}.jsonl")

    def _next_custom_id(self, output_name: str, modality: str) -> str:
        self._batch_request_counter += 1
        # Only [a-zA-Z0-9_-] to stay valid for every provider, anthropic rejects the rest.
        # The counter restarts in every shard process, so the shard id keeps ids unique.
        return (
            f"{output_name}-{modality}-s{self.shard_id}-{self._batch_request_counter}"
        )

    def get_load_time(self) -> float:
        """Return 0: no model is loaded in batch submission mode."""
        return 0.0

    def get_token_counts(self) -> TokenCounts:
        """Return zero counts: no generation happens at submission time.

        Provider usage is reported with the batch results, so it is read by the
        receiver (``mmirage.core.process.batch.collector``) instead.
        """
        return TokenCounts(input_tokens=0, output_tokens=0)

    def build_multimodal_prompt(
        self, prompt_template: str, var_env: VariableEnvironment
    ) -> Tuple[str, List[Image.Image | str]]:
        """Build a prompt and extract its images.

        Returns:
            (formatted_prompt, images)
        """
        jinja_template = jinja2.Template(prompt_template)
        base_prompt = jinja_template.render(**var_env.to_dict())

        return base_prompt, var_env.get_images()

    @override
    def batch_process_sample(
        self,
        batch: List[VariableEnvironment],
        output_var: BatchApiOutputVar,
    ) -> List[VariableEnvironment]:
        """Serialize a batch of samples into provider requests.

        Args:
            batch: List of variable environments to process.
            output_var: Output variable defining prompt and output format.

        Returns:
            The variable environments with a placeholder set for ``output_var``.
        """
        nb_samples = len(batch)
        text_only_indices: List[int] = []
        multimodal_indices: List[int] = []
        index_to_custom_id: Dict[int, str] = {}
        for i in range(nb_samples):
            if batch[i].has_images():
                multimodal_indices.append(i)
            else:
                text_only_indices.append(i)

        if text_only_indices:
            jinja_template = jinja2.Template(output_var.prompt)
            requests: List[Dict[str, Any]] = []
            source_indices: List[int] = []
            for global_i in text_only_indices:
                base_prompt = jinja_template.render(**batch[global_i].to_dict())
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": base_prompt,
                        }
                    ]
                }
                if output_var.output_type == "JSON" and output_var.output_schema:
                    payload["expected_schema"] = list(output_var.output_schema)
                custom_id = self._next_custom_id(output_var.name, "text")
                index_to_custom_id[global_i] = custom_id
                request = self._batch_adapter.build_request(
                    custom_id=custom_id,
                    payload=payload,
                    config=self._batch_provider_config,
                )
                requests.append(request)
                source_indices.append(self._global_row_offset + global_i)

            self._text_orchestrator.add_requests(
                requests=requests,
                source_indices=source_indices,
                model_params_snapshot={
                    "output_name": output_var.name,
                    "output_type": output_var.output_type,
                    "modality": "text",
                },
            )

        if multimodal_indices:
            requests = []
            source_indices = []
            for global_i in multimodal_indices:
                base_prompt, images = self.build_multimodal_prompt(
                    output_var.prompt, batch[global_i]
                )
                content: List[Dict[str, Any]] = [{"type": "text", "text": base_prompt}]

                for image_ref in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": str(image_ref)},
                        }
                    )

                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                }
                if output_var.output_type == "JSON" and output_var.output_schema:
                    payload["expected_schema"] = list(output_var.output_schema)

                custom_id = self._next_custom_id(output_var.name, "multimodal")
                index_to_custom_id[global_i] = custom_id
                request = self._batch_adapter.build_request(
                    custom_id=custom_id,
                    payload=payload,
                    config=self._batch_provider_config,
                )
                requests.append(dict(request))
                source_indices.append(self._global_row_offset + global_i)

            self._multimodal_orchestrator.add_requests(
                requests=requests,
                source_indices=source_indices,
                model_params_snapshot={
                    "output_name": output_var.name,
                    "output_type": output_var.output_type,
                    "modality": "multimodal",
                },
            )

        placeholders: List[VariableEnvironment] = []
        for i in range(nb_samples):
            unique_id = index_to_custom_id.get(i, f"unknown-{i}")
            placeholder = f"__BATCH_SUBMITTED__:{unique_id}"
            placeholders.append(batch[i].with_variable(output_var.name, placeholder))

        self._global_row_offset += nb_samples

        return placeholders

    def finalize(self) -> None:
        """Flush both accumulators, submitting any remaining requests."""
        self._text_orchestrator.finalize(
            model_params_snapshot={
                "modality": "text",
                "phase": "finalize",
            }
        )
        self._multimodal_orchestrator.finalize(
            model_params_snapshot={
                "modality": "multimodal",
                "phase": "finalize",
            }
        )
