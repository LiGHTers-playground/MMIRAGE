"""Registry and factory for provider batch adapters."""

import os
from typing import Dict, Type

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter


class BatchAdapterRegistry:
    """Provider-to-adapter registry with factory helpers.

    This class centralizes provider registration and fail-fast adapter
    instantiation with credential validation.
    """

    _registry: Dict[str, Type[BatchSubmissionAdapter]] = dict()
    _bootstrapped: bool = False

    @classmethod
    def _bootstrap_builtin_adapters(cls) -> None:
        if cls._bootstrapped:
            return

        # Local import avoids import cycles while ensuring built-ins are available.
        from mmirage.core.process.batch.anthropic_adapter import AnthropicBatchAdapter
        from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

        cls.register("anthropic", AnthropicBatchAdapter)
        cls.register("openai", OpenAIBatchAdapter)
        cls._bootstrapped = True

    @classmethod
    def register(cls, provider: str, adapter_cls: Type[BatchSubmissionAdapter]) -> None:
        """Register an adapter class under a provider key."""
        provider_key = provider.strip().lower()
        if not provider_key:
            raise ValueError("provider must be a non-empty string")
        cls._registry[provider_key] = adapter_cls

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters.

        Intended for tests and isolated bootstrapping logic.
        """
        cls._registry.clear()
        cls._bootstrapped = False

    @classmethod
    def resolve(cls, provider: str) -> Type[BatchSubmissionAdapter]:
        """Resolve a provider key to a registered adapter class."""
        cls._bootstrap_builtin_adapters()
        provider_key = provider.strip().lower()
        if provider_key not in cls._registry:
            raise ValueError(
                f"Unknown batch provider '{provider}'. "
                f"Available providers: {list(cls._registry.keys())}"
            )
        return cls._registry[provider_key]

    @classmethod
    def create(
        cls,
        config: BatchProviderConfig,
        allow_missing_credentials: bool = False,
    ) -> BatchSubmissionAdapter:
        """Instantiate an adapter, checking its credentials are set in the environment.

        When ``allow_missing_credentials`` is True the check is skipped because
        submission calls are expected to be bypassed.
        """
        adapter_cls = cls.resolve(config.provider)

        if not allow_missing_credentials:
            # Provider names may contain characters that are illegal in env var
            # names, e.g. 'azure-openai' -> AZURE_OPENAI_API_KEY.
            prefix = "".join(
                char if char.isalnum() else "_" for char in config.provider.upper()
            )
            missing = [
                env_var
                for env_var in (
                    f"{prefix}_{req_key.upper()}"
                    for req_key in adapter_cls.required_credentials
                )
                if not os.environ.get(env_var, "").strip()
            ]
            if missing:
                raise ValueError(
                    f"Missing environment variable(s) for provider '{config.provider}': {missing}"
                )
        return adapter_cls()
