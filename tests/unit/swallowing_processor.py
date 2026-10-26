"""A fake processor that swallows every Exception, like the llm and custom processors.

Imported by the SIGTERM test's subprocess before it runs ``shard_process.main``.
"""

import os
import pathlib
import time
from dataclasses import dataclass
from typing import List, Literal, Sequence

from mmirage.core.process.base import (
    BaseProcessor,
    BaseProcessorConfig,
    ProcessorRegistry,
    TokenCounts,
)
from mmirage.core.process.variables import BaseVar, OutputVar, VariableEnvironment


@dataclass
class SwallowingConfig(BaseProcessorConfig):
    type: Literal["swallowing"] = "swallowing"


@dataclass
class SwallowingOutputVar(OutputVar):
    type: str = "swallowing"

    def is_computable(self, vars: Sequence[BaseVar]) -> bool:
        return True


@ProcessorRegistry.register("swallowing", SwallowingConfig, SwallowingOutputVar)
class SwallowingProcessor(BaseProcessor[SwallowingOutputVar]):
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: SwallowingOutputVar
    ) -> List[VariableEnvironment]:
        pathlib.Path(os.environ["MMIRAGE_TEST_STARTED"]).touch()
        try:
            time.sleep(60)
            value = "done"
        except Exception:
            value = "fallback"
        return [env.with_variable(output_var.name, value) for env in batch]

    def get_token_counts(self) -> TokenCounts:
        return TokenCounts(input_tokens=0, output_tokens=0)

    def get_load_time(self) -> float:
        return 0.0
