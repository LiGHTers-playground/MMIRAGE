"""Concrete OpenAI implementation of batch submission contracts."""

import base64
import copy
import io
import json
import logging
import mimetypes
import os
from typing import Any, Dict, List, Mapping, Sequence

from openai import OpenAI

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)

logger = logging.getLogger(__name__)


class OpenAIBatchAdapter(BatchSubmissionAdapter):
    """Provider adapter for OpenAI Batch API."""

    required_credentials = ("api_key",)

    def build_request(
        self,
        custom_id: str,
        payload: Dict[str, Any],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        openai_config = self._require_openai_config(config)
        body = copy.deepcopy(payload)
        expected_schema = body.pop(
            "expected_schema", None
        )  # expected_schema needs to be popped from body before submission, as it was in the normalized request but is not an OpenAI API parameter.
        if expected_schema is not None and (
            not isinstance(expected_schema, list)
            or not all(isinstance(key, str) for key in expected_schema)
        ):
            raise ValueError(
                "expected_schema must be a list of strings, "
                f"got {type(expected_schema).__name__}"
            )
        body.setdefault("model", openai_config.model)
        self._convert_local_images_to_data_uris(body)

        if expected_schema:
            properties = {key: {"type": "string"} for key in expected_schema}
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": expected_schema,
                        "additionalProperties": False,
                    },
                },
            }

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": openai_config.batch_endpoint,
            "body": body,
        }

    @staticmethod
    def _convert_local_images_to_data_uris(body: Dict[str, Any]) -> None:
        # If the payload shape is different, swallow the exception and leave the body untouched.
        try:
            for message in body["messages"]:
                for part in message["content"]:
                    if part.get("type") != "image_url":
                        continue
                    url = part["image_url"]["url"]
                    if not isinstance(url, str):
                        continue
                    # Keep remote/data URLs untouched.
                    if (
                        url.startswith("http://")
                        or url.startswith("https://")
                        or url.startswith("data:")
                    ):
                        continue
                    if os.path.exists(url):
                        part["image_url"]["url"] = (
                            OpenAIBatchAdapter._local_file_to_data_uri(url)
                        )
        except (KeyError, IndexError, TypeError, AttributeError):
            # Ignore malformed shapes.
            pass

    @staticmethod
    def _local_file_to_data_uri(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def estimate_request_bytes(self, request: Dict[str, Any]) -> int:
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        return len(serialized.encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Dict[str, Any]],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        openai_config = self._require_openai_config(config)
        client = self._create_client(openai_config)

        jsonl_lines = [
            json.dumps(req, ensure_ascii=False, separators=(",", ":"))
            for req in requests
        ]
        jsonl_payload = "\n".join(jsonl_lines).encode("utf-8")

        file_response = client.files.create(
            file=(f"batch_chunk-{chunk_id}.jsonl", io.BytesIO(jsonl_payload)),
            purpose="batch",
        )

        metadata = dict(openai_config.metadata)
        metadata["chunk_id"] = chunk_id

        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint=openai_config.batch_endpoint,
            completion_window=openai_config.completion_window,
            metadata=metadata,
        )

        return {
            "id": batch_response.id,
            "status": getattr(batch_response, "status", None),
            "endpoint": getattr(batch_response, "endpoint", None),
            "input_file_id": file_response.id,
            "chunk_id": chunk_id,
        }

    def check_batch_status(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> BatchSubmissionResult:
        openai_config = self._require_openai_config(config)
        client = self._create_client(openai_config)
        retrieved = client.batches.retrieve(provider_batch_id)
        return self.parse_submission_result(raw_result=retrieved)

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        """Download completed OpenAI batch rows and normalize text into ``generated_text``.

        OpenAI batch outputs can surface the assistant payload in nested
        response bodies, so this method flattens the provider-specific shape
        before returning rows to the provider-agnostic collector.
        """
        openai_config = self._require_openai_config(config)
        client = self._create_client(openai_config)

        retrieved = client.batches.retrieve(provider_batch_id)
        status = getattr(retrieved, "status", None) or "unknown"
        output_file_id = getattr(retrieved, "output_file_id", None)
        error_file_id = getattr(retrieved, "error_file_id", None)

        if status != "completed":
            raise ValueError(
                f"Batch '{provider_batch_id}' is not completed yet (status={status}). "
                "Please retry after the provider marks it completed and produces an output file."
            )

        # A partially failed batch has both files; read the output first, then the errors.
        content_file_ids = [
            file_id for file_id in (output_file_id, error_file_id) if file_id
        ]
        if not content_file_ids:
            raise ValueError(
                f"Batch '{provider_batch_id}' completed, but neither output_file_id nor error_file_id was returned."
            )

        rows: List[Dict[str, Any]] = []
        for content_file_id in content_file_ids:
            content_response = client.files.content(content_file_id)
            jsonl_text = self._extract_content_text(content_response)

            for line in jsonl_text.splitlines():
                raw = line.strip()
                if not raw:
                    continue
                row = dict(json.loads(raw))
                error_message = self._extract_error_message(row)
                if error_message:
                    row.setdefault("status", "error")
                    row["error_message"] = error_message
                if "generated_text" not in row:
                    generated_text = self._extract_generated_text(row)
                    if generated_text:
                        row["generated_text"] = generated_text
                row.update(self._extract_usage(row))
                rows.append(row)

        return rows

    def parse_submission_result(
        self,
        raw_result: Dict[str, Any],
    ) -> BatchSubmissionResult:
        # Prefer attribute access for OpenAI SDK objects, fall back to mapping access.
        def _attr_or_get(obj: Any, attr: str, default: Any = None) -> Any:
            try:
                val = getattr(obj, attr)
            except Exception:
                val = None
            if val is not None:
                return val
            if isinstance(obj, Mapping):
                return obj.get(attr, default)
            return default

        batch_id = str(
            _attr_or_get(raw_result, "id") or _attr_or_get(raw_result, "batch_id", "")
        )
        return BatchSubmissionResult(
            provider_batch_id=batch_id,
            status=self._normalize_status(_attr_or_get(raw_result, "status")),
            raw_response=raw_result,
        )

    @staticmethod
    def _normalize_status(status: Any) -> str:
        value = str(status or "").strip().lower()
        if value == "completed":
            return "completed"
        if value in ("failed", "expired", "cancelled"):
            return "failed"
        if value in ("validating", "in_progress", "finalizing", "cancelling"):
            return "in_progress"
        return "unknown"

    @staticmethod
    def _require_openai_config(config: BatchProviderConfig) -> OpenAIBatchConfig:
        """Validate that `config` is an `OpenAIBatchConfig` and return it.

        Raises `TypeError` when the provided `config` is not an
        `OpenAIBatchConfig`.
        """
        if isinstance(config, OpenAIBatchConfig):
            return config
        raise TypeError("OpenAIBatchAdapter requires OpenAIBatchConfig")

    @staticmethod
    def _extract_generated_text(row: Dict[str, Any]) -> str:
        # prefer chat `message.content`, then `choices[0].text`,
        # then `body.text`. Return empty string if none match.
        try:
            content = row["response"]["body"]["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
        except (KeyError, IndexError, TypeError):
            pass

        try:
            text = row["response"]["body"]["choices"][0]["text"]
            if isinstance(text, str):
                return text
        except (KeyError, IndexError, TypeError):
            pass

        try:
            body_text = row["response"]["body"]["text"]
            if isinstance(body_text, str):
                return body_text
        except (KeyError, TypeError):
            pass

        return ""

    @staticmethod
    def _extract_usage(row: Dict[str, Any]) -> Dict[str, int]:
        # Return an empty mapping when the provider reports no usage, e.g. on failed rows
        try:
            usage = row["response"]["body"]["usage"]
            return {
                "input_tokens": int(usage["prompt_tokens"]),
                "output_tokens": int(usage["completion_tokens"]),
            }
        except (KeyError, TypeError, ValueError):
            return {}

    @staticmethod
    def _extract_error_message(row: Dict[str, Any]) -> str:
        try:
            error = row["response"]["body"]["error"]
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return message
        except (KeyError, TypeError):
            pass

        return ""

    @staticmethod
    def _create_client(config: OpenAIBatchConfig) -> OpenAI:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. set OPENAI_API_KEY. with `export OPENAI_API_KEY=your_api_key` ."
            )

        client_kwargs = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        return OpenAI(**client_kwargs)

    @staticmethod
    def _extract_content_text(content_response: Any) -> str:
        # Assume `content_response` is an httpx.Response (OpenAI SDK v1).
        # Prefer `.text`, fallback to `.content` bytes decode.
        try:
            text = content_response.text
        except Exception:
            text = None

        if isinstance(text, str):
            return text

        content = getattr(content_response, "content", None)
        if isinstance(content, bytes):
            return content.decode("utf-8")

        logger.debug(
            "Unable to extract content from response of type %s", type(content_response)
        )
        raise ValueError(
            "Unable to parse OpenAI files.content response: missing text or content bytes"
        )
