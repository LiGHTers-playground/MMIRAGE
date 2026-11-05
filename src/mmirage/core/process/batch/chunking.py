"""Provider-agnostic request chunking utilities for batch submission."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from mmirage.config.batch_provider import BatchProviderConfig, OversizedRequestPolicy
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter

logger = logging.getLogger(__name__)


@dataclass
class RequestChunk:
    """Chunk of provider-ready requests with aggregate metadata."""

    requests: List[Dict[str, Any]]
    total_bytes: int
    has_oversized_request: bool = False

    @property
    def total_requests(self) -> int:
        return len(self.requests)


class BatchRequestChunker:
    """Split request sequences into chunks using serialized-byte limits."""

    def __init__(
        self, adapter: BatchSubmissionAdapter, config: BatchProviderConfig
    ) -> None:
        self.adapter = adapter
        self.config = config

    def chunk_requests(self, requests: List[Dict[str, Any]]) -> List[RequestChunk]:
        """Chunk requests according to max bytes, max requests, and oversize policy."""

        chunks: List[RequestChunk] = []
        current_requests: List[Dict[str, Any]] = []
        current_total_bytes = 0
        max_chunk_bytes = self.config.max_chunk_bytes

        def append_current_chunk() -> None:
            if current_requests:
                chunks.append(
                    RequestChunk(
                        requests=list(current_requests),
                        total_bytes=current_total_bytes,
                    )
                )

        for request in requests:
            request_size = self.adapter.estimate_request_bytes(request)

            if request_size > max_chunk_bytes:
                if (
                    self.config.oversized_request_policy
                    is OversizedRequestPolicy.REJECT
                ):
                    raise ValueError(
                        "Encountered oversized request: "
                        f"{request_size} bytes exceeds max_chunk_bytes={max_chunk_bytes}"
                    )

                logger.warning(
                    "Encountered oversized request (%s bytes > %s); isolating into its own chunk.",
                    request_size,
                    max_chunk_bytes,
                )

                append_current_chunk()
                current_requests = []
                current_total_bytes = 0

                chunks.append(
                    RequestChunk(
                        requests=[request],
                        total_bytes=request_size,
                        has_oversized_request=True,
                    )
                )
                continue

            would_exceed_bytes = current_total_bytes + request_size > max_chunk_bytes
            would_exceed_count = self._would_exceed_count_limit(current_requests)

            if current_requests and (would_exceed_bytes or would_exceed_count):
                append_current_chunk()
                current_requests = []
                current_total_bytes = 0

            current_requests.append(request)
            current_total_bytes += request_size

        if current_requests:
            append_current_chunk()

        return chunks

    def _would_exceed_count_limit(self, current_requests: List[Dict[str, Any]]) -> bool:
        if self.config.max_requests_per_chunk is None:
            return False
        return len(current_requests) >= self.config.max_requests_per_chunk
