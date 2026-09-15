"""Stateful provider-agnostic orchestration for batch submission."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from mmirage.core.process.batch.chunking import BatchRequestChunker, RequestChunk


@dataclass
class _PendingRequest:
    request: Mapping[str, Any]
    source_index: int  # original row index of the data sample within the input dataset


class BatchSubmissionOrchestrator:
    """Accumulate requests across map iterations and submit full-ready chunks."""

    def __init__(
        self,
        adapter: BatchSubmissionAdapter,
        config: BatchProviderConfig,
        export_prompts_path: Optional[str] = None,
        export_batch_prefix: str = "",
        shard_id: int = 0,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.shard_id = shard_id
        self.chunker = BatchRequestChunker(adapter=adapter, config=config)
        self._export_prompts_path = export_prompts_path
        self._export_batch_prefix = export_batch_prefix
        self._pending: List[_PendingRequest] = []
        self._chunk_counter = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add_requests(
        self,
        requests: Sequence[Mapping[str, Any]],
        source_indices: Sequence[int],
        model_params_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> List[BatchSubmissionResult]:
        """Append requests and submit only chunks that are ready mid-stream."""
        if len(requests) != len(source_indices):
            raise ValueError("requests and source_indices must have identical lengths")

        for request, source_index in zip(requests, source_indices):
            self._pending.append(
                _PendingRequest(request=request, source_index=source_index)
            )

        return self._emit_ready_chunks(
            model_params_snapshot=model_params_snapshot,
            finalize=False,
        )

    def finalize(
        self,
        model_params_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> List[BatchSubmissionResult]:
        """Flush all remaining requests at end-of-dataset lifecycle."""
        return self._emit_ready_chunks(
            model_params_snapshot=model_params_snapshot,
            finalize=True,
        )

    def _emit_ready_chunks(
        self,
        model_params_snapshot: Optional[Mapping[str, Any]],
        finalize: bool = False,
    ) -> List[BatchSubmissionResult]:
        if not self._pending:
            return []

        pending_requests = [entry.request for entry in self._pending]
        chunks = self.chunker.chunk_requests(pending_requests)
        chunk_groups = self._split_pending_entries_by_chunks(chunks)

        groups_to_submit: List[tuple[List[_PendingRequest], RequestChunk]] = []
        groups_to_keep: List[_PendingRequest] = []

        if finalize:
            groups_to_submit = chunk_groups
        elif chunk_groups:
            groups_to_submit = chunk_groups[:-1]
            tail_entries, tail_chunk = chunk_groups[-1]
            if self._is_complete_chunk(tail_chunk):
                groups_to_submit.append((tail_entries, tail_chunk))
            else:
                groups_to_keep = list(tail_entries)

        self._pending = groups_to_keep

        submission_results: List[BatchSubmissionResult] = []
        for chunk_entries, request_chunk in groups_to_submit:
            chunk_id = self._next_chunk_id()
            requests = [entry.request for entry in chunk_entries]
            if self._export_prompts_path:
                parsed_result = self._export_chunk_requests(
                    chunk_id=chunk_id,
                    requests=requests,
                    export_path=self._export_prompts_path,
                )
            else:
                raw_result = self.adapter.submit_chunk(
                    chunk_id=chunk_id,
                    requests=requests,
                    config=self.config,
                )
                parsed_result = self.adapter.parse_submission_result(
                    raw_result=raw_result,
                )
            submission_results.append(parsed_result)

            self._persist_metadata(
                chunk_id=chunk_id,
                chunk_entries=chunk_entries,
                chunk=request_chunk,
                parsed_result=parsed_result,
                model_params_snapshot=model_params_snapshot,
                flush_reason="finalize" if finalize else "full_chunk",
            )

        return submission_results

    def _export_chunk_requests(
        self,
        chunk_id: str,
        requests: Sequence[Mapping[str, Any]],
        export_path: str,
    ) -> BatchSubmissionResult:
        batch_id = f"{self._export_batch_prefix}{chunk_id}"

        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

        with open(export_path, "a", encoding="utf-8") as f:
            for request in requests:
                # The request stays untouched so the line can be submitted as-is.
                row = {"batch_id": batch_id, "request": dict(request)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return BatchSubmissionResult(
            provider_batch_id=f"dry-run-{batch_id}",
            status="dry_run",
            raw_response={
                "dry_run": True,
                "chunk_id": chunk_id,
                "batch_id": batch_id,
                "export_path": export_path,
                "request_count": len(requests),
            },
        )

    def _split_pending_entries_by_chunks(
        self,
        chunks: Sequence[RequestChunk],
    ) -> List[tuple[List[_PendingRequest], RequestChunk]]:
        grouped: List[tuple[List[_PendingRequest], RequestChunk]] = []
        cursor = 0
        for chunk in chunks:
            size = len(chunk.requests)
            grouped.append((self._pending[cursor : cursor + size], chunk))
            cursor += size
        return grouped

    def _is_complete_chunk(self, chunk: RequestChunk) -> bool:
        if chunk.has_oversized_request:
            return True
        if chunk.total_bytes >= self.config.max_chunk_bytes:
            return True
        if self.config.max_requests_per_chunk is not None:
            return chunk.total_requests >= self.config.max_requests_per_chunk
        return False

    def _next_chunk_id(self) -> str:
        self._chunk_counter += 1
        return f"chunk-{self._chunk_counter:06d}"

    def _persist_metadata(
        self,
        chunk_id: str,
        chunk_entries: Sequence[_PendingRequest],
        chunk: RequestChunk,
        parsed_result: BatchSubmissionResult,
        model_params_snapshot: Optional[Mapping[str, Any]],
        flush_reason: str,
    ) -> None:
        if not self.config.metadata_output_path:
            return

        custom_to_source = {
            str(
                entry.request.get("custom_id", f"idx-{entry.source_index}")
            ): entry.source_index
            for entry in chunk_entries
        }

        request_hash = hashlib.sha256(
            json.dumps(
                [entry.request for entry in chunk_entries],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        metadata_record: Dict[str, Any] = {
            "provider": self.config.provider,
            "shard_id": self.shard_id,
            "chunk_id": chunk_id,
            "provider_batch_id": parsed_result.provider_batch_id,
            "status": parsed_result.status,
            "custom_id_to_source_index": custom_to_source,
            "request_hash": request_hash,
            "model_params_snapshot": dict(model_params_snapshot or {}),
            "submitted_request_count": chunk.total_requests,
            "total_bytes": chunk.total_bytes,
            "has_oversized_request": chunk.has_oversized_request,
            "flush_reason": flush_reason,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path = self.config.metadata_output_path
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        with open(metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")
