"""Provider-agnostic batch processing contracts and registry.

Adapters are reached through the registry's lazy bootstrap only, so importing
this package never imports a provider SDK.
"""

from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from mmirage.core.process.batch.chunking import BatchRequestChunker, RequestChunk
from mmirage.core.process.batch.collector import collect_and_merge
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
    "BatchAdapterRegistry",
    "extract_unique_provider_batches",
    "run_status_checker",
]
