"""A registered in-process processor for mapper and shard tests.

Sets ``output_var.name`` to ``len(text)`` and records the batch sizes it saw.
"""

from dataclasses import dataclass
from typing import List, Literal, Sequence

from mmirage.core.process.base import (
    BaseProcessor,
    BaseProcessorConfig,
    ProcessorRegistry,
    TokenCounts,
)
from mmirage.core.process.variables import BaseVar, OutputVar


@dataclass
class FakeConfig(BaseProcessorConfig):
    type: Literal["fake_len"] = "fake_len"


@dataclass
class FakeOutputVar(OutputVar):
    type: str = "fake_len"

    def is_computable(self, vars: Sequence[BaseVar]) -> bool:
        return True


@ProcessorRegistry.register("fake_len", FakeConfig, FakeOutputVar)
class FakeLenProcessor(BaseProcessor[FakeOutputVar]):
    def __init__(self, config, shard_id: int = 0) -> None:
        super().__init__(config, shard_id=shard_id)
        self.batch_sizes: List[int] = []

    def batch_process_sample(self, batch, output_var):
        self.batch_sizes.append(len(batch))
        return [
            env.with_variable(output_var.name, len(env.get("text"))) for env in batch
        ]

    def get_token_counts(self) -> TokenCounts:
        return TokenCounts(input_tokens=0, output_tokens=0)

    def get_load_time(self) -> float:
        return 0.0
