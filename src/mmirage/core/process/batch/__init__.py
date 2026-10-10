"""Provider-agnostic batch processing contracts and registry."""

from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from mmirage.core.process.batch.chunking import BatchRequestChunker, RequestChunk
from mmirage.core.process.batch.collector import collect_and_merge
from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter
from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator
from mmirage.core.process.batch.registry import (
    BatchAdapterRegistry,
)
from mmirage.core.process.batch.status_checker import (
    extract_unique_provider_batches,
    run_status_checker,
)

__all__ = [
    "BatchSubmissionAdapter",
    "BatchSubmissionResult",
    "collect_and_merge",
    "BatchRequestChunker",
    "RequestChunk",
    "BatchSubmissionOrchestrator",
    "OpenAIBatchAdapter",
    "OpenAIBatchConfig",
    "BatchAdapterRegistry",
    "extract_unique_provider_batches",
    "run_status_checker",
]
