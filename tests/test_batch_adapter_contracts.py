import pytest
from batch_fixtures import UnitBatchConfig

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from mmirage.core.process.batch.provider_resolution import (
    BatchProviderConfigRegistry,
    resolve_single_provider_config,
)
from mmirage.core.process.batch.registry import (
    BatchAdapterRegistry,
)


class CompleteTestAdapter(BatchSubmissionAdapter):
    required_credentials = tuple()

    @property
    def adapter_name(self) -> str:
        return "complete-test-adapter"

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    def build_request(self, custom_id, payload, config):
        return {
            "custom_id": custom_id,
            "payload": dict(payload),
            "provider": config.provider,
        }

    def estimate_request_bytes(self, request):
        # Deterministic approximation for tests.
        return len(str(request).encode("utf-8"))

    def submit_chunk(self, chunk_id, requests, config):
        return {
            "batch_id": f"{config.provider}-{chunk_id}",
            "status": "submitted",
            "requests": len(requests),
        }

    def parse_submission_result(self, raw_result):
        return BatchSubmissionResult(
            provider_batch_id=str(raw_result["batch_id"]),
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


class CredentialedTestAdapter(CompleteTestAdapter):
    required_credentials = ("api_key",)


class IncompleteTestAdapter(BatchSubmissionAdapter):
    @property
    def adapter_name(self) -> str:
        return "incomplete"

    @property
    def adapter_version(self) -> str:
        return "0.0.0"

    def build_request(self, custom_id, payload, config):
        return {}

    def estimate_request_bytes(self, request):
        return 0

    def submit_chunk(self, chunk_id, requests, config):
        return {}


def test_adapter_interface_is_abstract():
    with pytest.raises(TypeError):
        BatchSubmissionAdapter()


def test_incomplete_adapter_fails_interface_compliance():
    with pytest.raises(TypeError):
        IncompleteTestAdapter()


def test_complete_adapter_is_interface_compliant():
    adapter = CompleteTestAdapter()
    config = BatchProviderConfig(provider="unit")

    request = adapter.build_request(custom_id="req-1", payload={"x": 1}, config=config)
    assert request["custom_id"] == "req-1"

    estimated_bytes = adapter.estimate_request_bytes(request)
    assert estimated_bytes > 0

    raw_result = adapter.submit_chunk(
        chunk_id="chunk-1", requests=[request], config=config
    )
    parsed = adapter.parse_submission_result(raw_result=raw_result)

    assert parsed.provider_batch_id == "unit-chunk-1"
    assert parsed.status == "submitted"


def test_factory_resolves_registered_provider():
    BatchAdapterRegistry.register("unit", CompleteTestAdapter)
    config = BatchProviderConfig(provider="unit")

    adapter = BatchAdapterRegistry.create(config)

    assert isinstance(adapter, CompleteTestAdapter)


def test_factory_raises_for_unknown_provider():
    config = BatchProviderConfig(provider="not-registered")

    with pytest.raises(ValueError, match="Unknown batch provider"):
        BatchAdapterRegistry.create(config)


def test_factory_raises_when_credential_env_var_is_missing(monkeypatch):
    BatchAdapterRegistry.register("unit", CredentialedTestAdapter)
    monkeypatch.delenv("UNIT_API_KEY", raising=False)
    config = BatchProviderConfig(provider="unit")

    with pytest.raises(ValueError, match="UNIT_API_KEY"):
        BatchAdapterRegistry.create(config)


def test_factory_creates_adapter_when_credential_env_var_is_set(monkeypatch):
    BatchAdapterRegistry.register("unit", CredentialedTestAdapter)
    monkeypatch.setenv("UNIT_API_KEY", "from-env")
    config = BatchProviderConfig(provider="unit")

    assert isinstance(BatchAdapterRegistry.create(config), CredentialedTestAdapter)


def test_resolve_single_provider_config_raises_for_missing_provider():
    with pytest.raises(
        ValueError, match="batch config must include a non-empty provider"
    ):
        resolve_single_provider_config({})


def test_resolve_single_provider_config_resolves_custom_provider():
    BatchProviderConfigRegistry.register("unit", UnitBatchConfig)

    config = resolve_single_provider_config(
        {"provider": "unit", "unit_setting": "custom"}
    )

    assert isinstance(config, UnitBatchConfig)
    assert config.provider == "unit"
    assert config.unit_setting == "custom"


def test_resolve_single_provider_config_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Unknown batch provider"):
        resolve_single_provider_config({"provider": "not-registered"})
