"""Custom processor implementation for executing user-defined Python functions."""

import concurrent.futures
import copy
import logging
import multiprocessing
import os
import time
from typing import Any, List

from pebble import ProcessExpired, ProcessPool

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry, TokenCounts
from mmirage.core.process.processors.custom.config import (
    CustomOutputVar,
    CustomProcessorConfig,
)
from mmirage.core.process.processors.custom.worker import (
    execute_custom_function,
    initialize_worker,
)
from mmirage.core.process.variables import VariableEnvironment

logger = logging.getLogger(__name__)


class CustomProcessor(BaseProcessor[CustomOutputVar]):
    """Processor that runs user-provided Python scripts in a persistent, isolated process pool.

    Workers are started with the configured ``start_method`` ('spawn' by default, which
    isolates them from the parent's state) and misbehaving scripts trip a circuit breaker.
    """

    def __init__(self, config: CustomProcessorConfig, shard_id: int = 0) -> None:
        """Initialize the processor and eagerly start the worker pool.

        Args:
            config: Custom module configuration.
            shard_id: Shard index for this worker.

        Raises:
            FileNotFoundError: If ``config.script_path`` does not exist.
        """
        start_time = time.time()
        super().__init__(config=config, shard_id=shard_id)
        self.config: CustomProcessorConfig = config
        self._error_count = 0
        self._timeout_count = 0
        self._is_broken = False
        self._trip_reason = ""

        # check user script file existence
        if not os.path.exists(self.config.script_path):
            raise FileNotFoundError(
                f"CustomProcessor failed to boot: script_path '{self.config.script_path}' does not exist."
            )

        self._pool = ProcessPool(
            max_workers=self.config.max_workers,
            context=multiprocessing.get_context(self.config.start_method),
            initializer=initialize_worker,
            initargs=(self.config.script_path, self.config.function_name),
        )

        self._load_time = time.time() - start_time
        logger.info(
            f"Initialized CustomProcessor pool with {self.config.max_workers} '{self.config.start_method}' workers "
            f"(target: {self.config.function_name} in {self.config.script_path})"
        )

    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: CustomOutputVar
    ) -> List[VariableEnvironment]:
        """Process a batch of data rows through the user's custom function.

        Rows are scheduled in parallel and reassembled in input order. Timeouts and
        script exceptions are absorbed with ``fallback_value``; their counters are
        cumulative over the shard and never reset, so once ``max_timeouts`` or
        ``max_errors`` is crossed the pool is stopped and every later call raises.

        Args:
            batch: Rows to process.
            output_var: Variable the function's return value is bound to.

        Returns:
            The input rows, each extended with ``output_var``.

        Raises:
            RuntimeError: If the circuit breaker is or becomes tripped.
        """

        if self._is_broken:
            raise RuntimeError(
                f"Custom processor circuit breaker tripped: {self._trip_reason}"
            )

        # to guarantee strict input ordering
        results: List[Any] = [None] * len(batch)

        row_dicts = [dict(env.to_dict()) for env in batch]

        # None disables the per-row timeout
        timeout_seconds = (
            self.config.timeout_ms / 1000.0
            if self.config.timeout_ms is not None
            else None
        )
        future_to_index = {}
        for index, row_dict in enumerate(row_dicts):
            future = self._pool.schedule(
                execute_custom_function, args=(row_dict,), timeout=timeout_seconds
            )
            future_to_index[future] = index

        for future in concurrent.futures.as_completed(future_to_index.keys()):
            index = future_to_index[future]

            try:
                results[index] = future.result()

            except concurrent.futures.TimeoutError:
                self._timeout_count += 1
                logger.warning(f"Row {index} timed out. Applying fallback value.")
                results[index] = copy.deepcopy(self.config.fallback_value)

                if self._timeout_count >= self.config.max_timeouts:
                    self._trip_circuit_breaker(
                        "Max timeouts reached. Tripping circuit breaker and shutting down pool."
                    )
            except ProcessExpired as e:
                # critical error : process died unexpectedly during its execution
                self._trip_circuit_breaker(f"Worker process crashed fatally : {e}")
            except Exception as e:
                # if user script raises exception
                logger.error(
                    f"User script raised an exception on row {index}: {e}. Applying fallback value."
                )
                results[index] = copy.deepcopy(self.config.fallback_value)
                self._error_count += 1
                if self._error_count >= self.config.max_errors:
                    self._trip_circuit_breaker(
                        "Max errors reached. Tripping circuit breaker and shutting down pool."
                    )

        updated_batch = []
        for env, result in zip(batch, results):
            updated_batch.append(env.with_variable(output_var.name, result))

        return updated_batch

    def shutdown(self) -> None:
        """Stop the persistent worker pool.

        Teardown belongs here rather than in ``finalize()``: the latter runs after every
        dataset while the same pool is reused for the next one.
        """
        if not self._is_broken:
            self._pool.stop()
            self._pool.join()

    def _trip_circuit_breaker(self, reason: str) -> None:
        """Helper to centralize circuit breaker hard-fail logic."""
        self._is_broken = True
        self._trip_reason = reason
        logger.error(f"Tripping circuit breaker: {reason}")
        self._pool.stop()
        self._pool.join()
        raise RuntimeError(f"Custom processor circuit breaker tripped: {reason}")

    def get_token_counts(self) -> TokenCounts:
        """Return zero token counts as this is not an LLM processor."""
        return TokenCounts(input_tokens=0, output_tokens=0)

    def get_load_time(self) -> float:
        """Return the time taken to initialize the processor and its pool."""
        return self._load_time


ProcessorRegistry.register("custom", CustomProcessorConfig, CustomOutputVar)(
    CustomProcessor
)
