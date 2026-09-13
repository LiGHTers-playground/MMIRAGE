"""Provider double used by the batch tests, registered by the ``unit_provider`` fixture."""

from dataclasses import dataclass

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from mmirage.core.process.batch.provider_resolution import (
    resolve_single_provider_config,
)


@dataclass
class UnitBatchConfig(BatchProviderConfig):
    provider: str = "unit"
    unit_setting: str = "default"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.unit_setting.strip():
            raise ValueError("unit_setting must be a non-empty string")


class RecordingAdapter(BatchSubmissionAdapter):
    def __init__(self) -> None:
        self.submissions = []

    def build_request(self, custom_id, payload, config):
        return {"custom_id": custom_id, **dict(payload)}

    def estimate_request_bytes(self, request):
        return int(request.get("size_bytes", 0))

    def submit_chunk(self, chunk_id, requests, config):
        self.submissions.append({"chunk_id": chunk_id, "requests": list(requests)})
        return {"id": f"batch-{chunk_id}", "status": "submitted"}

    def parse_submission_result(self, raw_result):
        return BatchSubmissionResult(
            provider_batch_id=str(raw_result["id"]),
            status=str(raw_result["status"]),
            raw_response=raw_result,
        )

    def check_batch_status(self, provider_batch_id, config):
        return BatchSubmissionResult(
            provider_batch_id=provider_batch_id,
            status="submitted",
            raw_response={"id": provider_batch_id, "status": "submitted"},
        )

    def retrieve_results(self, provider_batch_id, config):
        return []


def unit_provider_config(**overrides) -> BatchProviderConfig:
    """Build a unit provider config the way the loader does: through resolution.

    The provider must be registered first, so use it with the ``unit_provider``
    fixture.
    """
    return resolve_single_provider_config({"provider": "unit", **overrides})
