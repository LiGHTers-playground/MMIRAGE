"""Concrete Anthropic implementation of batch submission contracts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import mimetypes
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from anthropic import Anthropic

from mmirage.config.anthropic_batch import AnthropicBatchConfig
from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)

logger = logging.getLogger(__name__)

# The only media types anthropic accepts in a base64 image source.
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


class AnthropicBatchAdapter(BatchSubmissionAdapter):
    """Provider adapter for Anthropic Messages Batches API."""

    required_credentials = ("api_key",)

    def build_request(
        self,
        custom_id: str,
        payload: Dict[str, Any],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        anthropic_config = self._require_anthropic_config(config)
        normalized_custom_id = self._normalize_custom_id(custom_id)
        body = copy.deepcopy(payload)
        expected_schema = body.pop("expected_schema", None)

        if expected_schema is not None and (
            not isinstance(expected_schema, list)
            or not all(isinstance(key, str) for key in expected_schema)
        ):
            raise ValueError(
                "expected_schema must be a list of strings, "
                f"got {type(expected_schema).__name__}"
            )

        body.setdefault("model", anthropic_config.model)
        body.setdefault("max_tokens", anthropic_config.max_tokens)

        if anthropic_config.temperature is not None:
            body.setdefault("temperature", anthropic_config.temperature)
        elif anthropic_config.top_p is not None:
            body.setdefault("top_p", anthropic_config.top_p)

        messages = body.get("messages")
        if isinstance(messages, list):
            body["messages"] = self._normalize_messages(messages)

        if expected_schema:
            properties = {key: {"type": "string"} for key in expected_schema}
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": expected_schema,
                        "additionalProperties": False,
                    },
                }
            }

        payload_request = {
            "custom_id": normalized_custom_id,
            "params": body,
        }
        return payload_request

    def estimate_request_bytes(self, request: Dict[str, Any]) -> int:
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        return len(serialized.encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Dict[str, Any]],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        anthropic_config = self._require_anthropic_config(config)
        client = self._create_client(anthropic_config)
        batches_client = self._resolve_batches_client(client)

        response = batches_client.create(requests=requests)

        return {
            "id": self._attr_or_get(response, "id", ""),
            "status": self._attr_or_get(response, "processing_status", None),
            "chunk_id": chunk_id,
        }

    def parse_submission_result(
        self, raw_result: Dict[str, Any]
    ) -> BatchSubmissionResult:
        batch_id = str(self._attr_or_get(raw_result, "id", "") or "")
        # submit_chunk hands over a dict with 'status'; the SDK's MessageBatch
        # only has 'processing_status'.
        status = self._attr_or_get(raw_result, "status", None) or self._attr_or_get(
            raw_result, "processing_status", None
        )
        return BatchSubmissionResult(
            provider_batch_id=batch_id,
            status=self._normalize_status(status),
            raw_response=raw_result,
        )

    def check_batch_status(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> BatchSubmissionResult:
        anthropic_config = self._require_anthropic_config(config)
        client = self._create_client(anthropic_config)
        batches_client = self._resolve_batches_client(client)
        retrieved = batches_client.retrieve(provider_batch_id)
        return self.parse_submission_result(raw_result=retrieved)

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        """Download completed Anthropic batch rows and normalize text into ``generated_text``."""
        anthropic_config = self._require_anthropic_config(config)
        client = self._create_client(anthropic_config)
        batches_client = self._resolve_batches_client(client)

        retrieved = batches_client.retrieve(provider_batch_id)
        status = self.parse_submission_result(raw_result=retrieved).status
        if status != "completed":
            raise ValueError(
                f"Batch '{provider_batch_id}' is not completed yet (status={status})."
            )

        results_response = batches_client.results(provider_batch_id)
        rows = self._parse_results_response(results_response)
        normalized: List[Dict[str, Any]] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            error_message = self._extract_error_message(row)
            if error_message:
                row.setdefault("status", "error")
                row["error_message"] = error_message
            if "generated_text" not in row:
                generated_text = self._extract_generated_text(row)
                if generated_text:
                    row["generated_text"] = generated_text
            normalized.append(row)

        return normalized

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                content_blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                content_blocks = AnthropicBatchAdapter._normalize_content_blocks(
                    content
                )
            else:
                raise ValueError(
                    f"message content must be a string or a list of blocks, got {type(content).__name__}"
                )
            normalized.append({"role": role, "content": content_blocks})
        return normalized

    @staticmethod
    def _normalize_content_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for part in blocks:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        normalized.append({"type": "text", "text": text})
                        continue
                elif part_type == "image_url":
                    url = part.get("image_url", {}).get("url")
                    if isinstance(url, str):
                        source = AnthropicBatchAdapter._image_source_from_url(url)
                        normalized.append({"type": "image", "source": source})
                        continue
                elif part_type == "image" and isinstance(part.get("source"), dict):
                    normalized.append({"type": "image", "source": dict(part["source"])})
                    continue
            logger.warning("Dropping content block anthropic cannot use: %.200r", part)
        return normalized

    @staticmethod
    def _normalize_custom_id(custom_id: str) -> str:
        # Anthropic requires custom_id to match ^[a-zA-Z0-9_-]{1,64}$
        raw = str(custom_id).strip()
        if not raw:
            raise ValueError("custom_id must be a non-empty string")

        safe_chars = []
        for ch in raw:
            if ch.isalnum() or ch in {"_", "-"}:
                safe_chars.append(ch)
            else:
                safe_chars.append("_")

        normalized = "".join(safe_chars)
        if normalized == raw and len(normalized) <= 64:
            return normalized

        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        max_base = 64 - (1 + len(digest))
        if max_base < 1:
            return digest[:64]

        base = normalized[:max_base]
        return f"{base}-{digest}"

    @staticmethod
    def _image_source_from_url(url: str) -> Dict[str, str]:
        if url.startswith("http://") or url.startswith("https://"):
            # Anthropic fetches remote images itself, like openai does.
            return {"type": "url", "url": url}

        if url.startswith("data:"):
            media_type, data = AnthropicBatchAdapter._parse_data_uri(url)
        else:
            if not os.path.exists(url):
                raise ValueError(f"Image path does not exist: {url}")
            media_type = AnthropicBatchAdapter._guess_mime_type(url)
            with open(url, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")

        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"Unsupported image media type '{media_type}' for '{url[:80]}', "
                f"anthropic accepts {', '.join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))}."
            )

        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    @staticmethod
    def _parse_data_uri(data_uri: str) -> tuple[str, str]:
        try:
            header, encoded = data_uri.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data URI format") from exc

        if ";base64" not in header:
            raise ValueError("Data URI must be base64 encoded")

        media_type = (
            header.removeprefix("data:").split(";", 1)[0] or "application/octet-stream"
        )
        return media_type, encoded

    @staticmethod
    def _guess_mime_type(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type or ""

    @staticmethod
    def _extract_generated_text(row: Dict[str, Any]) -> str:
        # Anthropic batch results contain result.message.content blocks.
        try:
            result = row.get("result", {})
            message = result.get("message", {})
            content = message.get("content", [])
            if isinstance(content, list):
                texts = [
                    block.get("text", "")
                    for block in content
                    if block.get("type") == "text"
                ]
                return "".join(texts)
        except AttributeError:
            # if provider response no longer has the shape we expect.
            logger.warning(
                "Could not read the generated text, the Anthropic response shape may have changed: %.200r",
                row,
            )
        return ""

    @staticmethod
    def _extract_error_message(row: Dict[str, Any]) -> str:
        result = row.get("result")
        if not isinstance(result, Mapping):
            return ""
        result_type = result.get("type")
        if result_type not in {"errored", "canceled", "expired"}:
            return ""
        try:
            # result.error is an ErrorResponse wrapper, the message is one level deeper.
            message = result["error"]["error"]["message"]
        except (KeyError, TypeError):
            message = None
        return message if isinstance(message, str) else f"Request {result_type}"

    @staticmethod
    def _parse_results_response(response: Iterable[Any]) -> List[Dict[str, Any]]:
        # The SDK streams the results file as MessageBatchIndividualResponse models.
        return [AnthropicBatchAdapter._coerce_mapping(item) for item in response]

    @staticmethod
    def _normalize_status(status: Any) -> str:
        # processing_status is only 'in_progress', 'canceling' or 'ended', the SDK
        # rejects the rest. A failed batch is still 'ended', its errors are per request.
        value = str(status or "").strip().lower()
        if value == "ended":
            return "completed"
        if value in ("in_progress", "canceling"):
            return "in_progress"
        return "unknown"

    @staticmethod
    def _coerce_mapping(item: Any) -> Any:
        if isinstance(item, Mapping):
            return dict(item)
        if hasattr(item, "model_dump"):
            try:
                return dict(item.model_dump())
            except Exception:
                logger.debug("model_dump() failed on %s", type(item).__name__)
                return item
        return item

    @staticmethod
    def _require_anthropic_config(config: BatchProviderConfig) -> AnthropicBatchConfig:
        if isinstance(config, AnthropicBatchConfig):
            return config
        raise TypeError("AnthropicBatchAdapter requires AnthropicBatchConfig")

    @staticmethod
    def _create_client(config: AnthropicBatchConfig) -> Anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Anthropic API key is missing. set ANTHROPIC_API_KEY. with `export ANTHROPIC_API_KEY=your_api_key` ."
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.timeout_seconds is not None:
            client_kwargs["timeout"] = config.timeout_seconds
        return Anthropic(**client_kwargs)

    @staticmethod
    def _resolve_batches_client(client: Anthropic) -> Any:
        # Typed Any: the adapter's dict-shaped contract does not match the SDK's
        # typed client yet, so its calls stay unchecked here.
        return client.messages.batches

    @staticmethod
    def _attr_or_get(obj: Any, attr: str, default: Any = None) -> Any:
        try:
            val = getattr(obj, attr)
        except AttributeError:
            # Attribute not found; fall back to checking if obj is a dictionary/mapping.
            val = None
        if val is not None:
            return val
        if isinstance(obj, Mapping):
            return obj.get(attr, default)
        return default
