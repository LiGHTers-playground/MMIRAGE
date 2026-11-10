"""Mapper for orchestrating variable transformations."""

import logging
from typing import Any, Dict, List, Optional, cast

from mmirage.core.process.base import (
    AutoProcessor,
    BaseProcessor,
    BaseProcessorConfig,
    TokenCounts,
)
from mmirage.core.process.variables import (
    BaseVar,
    InputVar,
    OutputVar,
    VariableEnvironment,
)

logger = logging.getLogger(__name__)


class MMIRAGEMapper:
    """Mapper for orchestrating variable transformations in the MMIRAGE pipeline.

    Manages processors, validates variable dependencies, and applies
    transformations to batches of data. Supports multimodal inputs.

    Attributes:
        processors: Dictionary mapping processor types to processor instances.
        output_vars: List of output variables to generate.
        input_vars: List of input variables to extract.
    """

    def __init__(
        self,
        processor_configs: List[BaseProcessorConfig],
        input_vars: List[InputVar],
        output_vars: List[OutputVar],
        export_prompts_dir: Optional[str] = None,
        shard_id: int = 0,
    ) -> None:
        """Initialize the MMIRAGE mapper.

        Args:
            processor_configs: List of processor configurations.
            input_vars: List of input variable definitions.
            output_vars: List of output variable definitions.
            export_prompts_dir: Value of --export-prompts.
            shard_id: Shard index for this worker, forwarded to processors.
        """
        self.processors: Dict[str, BaseProcessor] = dict()
        self.input_vars = input_vars
        self.output_vars = output_vars

        for config in processor_configs:
            processor_cls = AutoProcessor.from_name(config.type)
            logger.info(f"✅ Successfully loaded processor of type {config.type}")

            if hasattr(config, "export_prompts_dir"):
                # Item 8 of #61 hands this to the processor instead of writing
                # it onto a config object that has no such field.
                config.export_prompts_dir = export_prompts_dir  # pyright: ignore[reportAttributeAccessIssue]
            self.processors[config.type] = processor_cls(config, shard_id=shard_id)

    def validate_vars(self) -> bool:
        """Validate that all output variables are computable.

        Checks that each output variable can be computed given the
        available variables (inputs and previously computed outputs).

        Returns:
            True if all variables are computable, False otherwise.
        """
        vars = cast(List[BaseVar], self.input_vars.copy())

        for output_var in self.output_vars:
            if not output_var.is_computable(vars):
                context = list(map(lambda v: v.name, vars))
                logger.info(
                    f"⚠️ Variable {output_var.name} not computable given current context: {context}"
                )
                return False

            vars.append(output_var)

        return True

    def rewrite_batch(
        self,
        batch: Dict[str, List[Any]],
        image_base_path: Optional[str] = None,
    ) -> List[VariableEnvironment]:
        """Transform a batch of samples by computing output variables.

        Args:
            batch: Dictionary mapping column names to lists of values.
            image_base_path: Optional base directory for resolving relative image paths.

        Returns:
            List of VariableEnvironments with all output variables computed.

        Raises:
            RuntimeError: If an output variable type has no registered processor.
        """
        batch_environment = VariableEnvironment.from_batch_input_variables(
            batch, self.input_vars, image_base_path
        )

        for output_var in self.output_vars:
            if output_var.type not in self.processors:
                raise RuntimeError(
                    f"Output {output_var.type} not in registered processors: {self.processors.keys()}"
                )

            processor = self.processors[output_var.type]
            batch_environment = processor.batch_process_sample(
                batch_environment, output_var
            )

        return batch_environment

    def get_token_counts(self) -> TokenCounts:
        """Return cumulative token counts aggregated across all LLM processors.

        Sums ``input_tokens`` and ``output_tokens`` from every processor that
        exposes a ``get_token_counts()`` method (i.e., ``LLMProcessor``).

        Returns:
            TokenCounts with ``input_tokens`` and ``output_tokens`` fields.
        """
        total_input = 0
        total_output = 0
        for proc in self.processors.values():
            if hasattr(proc, "get_token_counts"):
                counts = proc.get_token_counts()
                total_input += counts.input_tokens
                total_output += counts.output_tokens
        return TokenCounts(input_tokens=total_input, output_tokens=total_output)

    def get_load_time(self) -> float:
        """Return total model-loading time (seconds) summed across all LLM processors."""
        total = 0.0
        for proc in self.processors.values():
            if hasattr(proc, "get_load_time"):
                total += proc.get_load_time()
        return total

    def finalize_processors(self) -> None:
        """Finalize processors that expose a finalize lifecycle hook."""
        for processor in self.processors.values():
            processor.finalize()

    def shutdown(self) -> None:
        """Shut down all processors and release their resources."""
        for processor in self.processors.values():
            processor.shutdown()
