"""Resolve batch provider configs from YAML and metadata inputs.

These helpers decouple config parsing from metadata inspection so the same
provider configuration logic can be reused by receiver and submission flows.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Type

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.metadata_utils import BatchMetadataRecord

if TYPE_CHECKING:
    from mmirage.config.config import MMirageConfig


class BatchProviderConfigRegistry:
    """Registry for provider-specific batch config classes.

    Built-ins are registered lazily to avoid import cycles and keep config
    resolution available in lightweight contexts.
    """

    _registry: Dict[str, Type[BatchProviderConfig]] = {}
    _bootstrapped: bool = False

    @classmethod
    def _bootstrap_builtin_configs(cls) -> None:
        if cls._bootstrapped:
            return

        from mmirage.config.anthropic_batch import AnthropicBatchConfig
        from mmirage.config.openai_batch import OpenAIBatchConfig

        cls.register("anthropic", AnthropicBatchConfig)
        cls.register("openai", OpenAIBatchConfig)
        cls._bootstrapped = True

    @classmethod
    def register(cls, provider: str, config_cls: Type[BatchProviderConfig]) -> None:
        provider_key = provider.strip().lower()
        if not provider_key:
            raise ValueError("provider must be a non-empty string")
        cls._registry[provider_key] = config_cls

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._bootstrapped = False

    @classmethod
    def get_config_cls(
        cls,
        provider: str,
        default: Type[BatchProviderConfig] | None = None,
    ) -> Type[BatchProviderConfig]:
        cls._bootstrap_builtin_configs()
        provider_key = provider.strip().lower()
        if not provider_key:
            raise ValueError("provider must be a non-empty string")
        if provider_key in cls._registry:
            return cls._registry[provider_key]
        if default is not None:
            return default
        raise ValueError(
            f"Unknown batch provider '{provider}'. Available providers: {list(cls._registry.keys())}"
        )


def _discover_required_providers(
    metadata_records: Sequence[BatchMetadataRecord],
) -> List[str]:
    providers: List[str] = []
    seen = set()
    for record in metadata_records:
        provider = record.provider
        if not provider or provider in seen:
            continue
        seen.add(provider)
        providers.append(provider)
    return providers


def _extract_batch_provider_blocks(cfg: MMirageConfig) -> Dict[str, Dict[str, Any]]:
    """Collect raw batch blocks keyed by provider.

    Raises ValueError on duplicate provider definitions to avoid ambiguous
    config resolution.
    """
    provider_blocks: Dict[str, Dict[str, Any]] = {}
    for processor_cfg in cfg.processors:
        raw_provider = getattr(processor_cfg, "provider_config", None)
        if raw_provider is None:
            continue

        if isinstance(raw_provider, BatchProviderConfig):
            raw_block = asdict(raw_provider)
        else:
            raw_block = dict(raw_provider or {})

        if not raw_block:
            continue

        provider = str(raw_block.get("provider", "")).strip().lower()
        if not provider:
            raise ValueError("batch config must include a non-empty provider")

        if provider in provider_blocks:
            raise ValueError(
                f"Duplicate batch blocks found for provider '{provider}' in config processors."
            )

        provider_blocks[provider] = raw_block

    return provider_blocks


def _instantiate_provider_config(
    provider: str, raw_block: Mapping[str, Any]
) -> BatchProviderConfig:
    """Instantiate the provider config, falling back to the shared base config."""
    payload = dict(raw_block)
    payload.setdefault("provider", provider)

    config_cls = BatchProviderConfigRegistry.get_config_cls(
        provider,
        default=BatchProviderConfig,
    )
    return config_cls(**payload)


def resolve_single_provider_config(raw_block: Mapping[str, Any]) -> BatchProviderConfig:
    """Resolve a single provider config from a raw batch block.

    Raises ValueError for missing providers, unknown providers, or invalid
    config payloads.
    """
    payload = dict(raw_block or {})
    provider = str(payload.get("provider", "")).strip().lower()
    if not provider:
        raise ValueError("batch config must include a non-empty provider")
    payload["provider"] = provider

    # Rejects an unknown provider before the payload is instantiated.
    BatchProviderConfigRegistry.get_config_cls(provider)

    try:
        return _instantiate_provider_config(provider, payload)
    except Exception as exc:
        raise ValueError(
            f"Failed to instantiate batch provider config for '{provider}': {exc}"
        ) from exc


def build_all_provider_configs(cfg: "MMirageConfig") -> Dict[str, BatchProviderConfig]:
    """Build provider configs for every batch block in the YAML.

    Raises ValueError when any provider config fails to instantiate.
    """
    provider_blocks = _extract_batch_provider_blocks(cfg)
    if not provider_blocks:
        return {}

    resolved: Dict[str, BatchProviderConfig] = {}
    for provider, raw_block in provider_blocks.items():
        try:
            resolved[provider] = _instantiate_provider_config(provider, raw_block)
        except Exception as exc:
            raise ValueError(
                f"Failed to instantiate batch provider config for '{provider}': {exc}"
            ) from exc

    return resolved


def resolve_provider_configs(
    metadata_records: Sequence[BatchMetadataRecord],
    cfg: "MMirageConfig",
) -> Dict[str, BatchProviderConfig]:
    """Resolve provider configs required by receiver metadata.

    Args:
        metadata_records: Parsed metadata JSONL records used to discover which
            providers are required by the receiver command.
        cfg: Loaded YAML config object from ``load_mmirage_config``.

    Returns:
        Mapping from normalized provider name to instantiated provider config.

    Raises:
        ValueError: If metadata references a provider missing from config
            ``batch_api`` processor blocks or if provider config
            instantiation fails.
    """
    available_configs = build_all_provider_configs(cfg)
    required_providers = _discover_required_providers(metadata_records)
    if not required_providers:
        return {}

    missing = [
        provider for provider in required_providers if provider not in available_configs
    ]
    if missing:
        raise ValueError(
            "Metadata references provider(s) missing from YAML batch config: "
            f"{missing}. Check the batch_api processor block(s)."
        )

    return {provider: available_configs[provider] for provider in required_providers}
