"""Error paths of provider resolution, for both the config and the adapter registry."""

import pytest

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.provider_resolution import (
    resolve_single_provider_config,
)
from mmirage.core.process.batch.registry import BatchAdapterRegistry


def test_adapter_registry_raises_for_unknown_provider():
    config = BatchProviderConfig(provider="not-registered")

    with pytest.raises(ValueError, match="Unknown batch provider"):
        BatchAdapterRegistry.create(config)


def test_resolve_single_provider_config_raises_for_missing_provider():
    with pytest.raises(
        ValueError, match="batch config must include a non-empty provider"
    ):
        resolve_single_provider_config({})


def test_resolve_single_provider_config_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Unknown batch provider"):
        resolve_single_provider_config({"provider": "not-registered"})
