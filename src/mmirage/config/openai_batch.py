"""OpenAI-specific batch configuration."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from mmirage.config.batch_provider import BatchProviderConfig


@dataclass
class OpenAIBatchConfig(BatchProviderConfig):
    """OpenAI Batch API configuration.

    Attributes:
        provider: Fixed provider identifier for OpenAI.
        model: Model name used in each chat completion request body.
        batch_endpoint: Target endpoint used by OpenAI batch jobs.
        completion_window: OpenAI completion window value.
        base_url: Optional base URL, useful for API-compatible gateways.
        metadata: Metadata sent on batch creation.
    """

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    batch_endpoint: str = "/v1/chat/completions"
    completion_window: Literal["24h"] = "24h"
    base_url: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        allowed_windows = {"24h"}
        if self.completion_window not in allowed_windows:
            raise ValueError(f"completion_window must be one of {allowed_windows}")

        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not self.batch_endpoint.startswith("/"):
            raise ValueError("batch_endpoint must start with '/'")
