"""Configuration dataclasses for MMIRAGE pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from mmirage.config.loading import LoadingParams
from mmirage.core.process.base import BaseProcessorConfig
from mmirage.core.process.processors.filter.config import FilterStep
from mmirage.core.process.variables import BaseVar, InputVar, OutputVar


@dataclass
class ExecutionParams:
    """Parameters for executing the MMIRAGE pipeline.

    Defines how the pipeline is executed, including local or SLURM-based
    distributed execution, retry logic, and resource allocation.

    Attributes:
        mode: Execution mode: "local" or "slurm". Defaults to "local".
        retry: Whether automatic retry orchestration is enabled. Defaults to False.
        merge: Whether to merge shard outputs after a successful run. Defaults to False.
        max_retries: Maximum number of retries for failed shards. Defaults to 3.
        poll_interval_seconds: Seconds to wait between polling job status. Defaults to 30.
        settle_time_seconds: Seconds to wait after job completes before checking results. Defaults to 60.

        # SLURM-specific parameters
        account: HPC account/partition to charge. Required for SLURM mode.
        job_name: SLURM job name. Defaults to "mmirage-sharded".
        reservation: Optional SLURM reservation name.
        nodes: Number of nodes. Defaults to 1.
        ntasks_per_node: Number of tasks per node. Defaults to 1.
        gpus: Number of GPUs per node. Defaults to 4.
        cpus_per_task: Number of CPUs per task. Defaults to 288.
        time_limit: Job time limit (HH:MM:SS). Defaults to "11:59:59".

        # Paths
        project_root: Base project directory. Can use environment variables with ${VAR}.
        report_dir: Directory for SLURM output/error files. Defaults to ~/reports.
        hf_home: HuggingFace cache directory. Defaults to ~/hf.
        edf_env: Optional EDF environment file path.
    """

    mode: str = "local"
    retry: bool = False
    merge: bool = False
    max_retries: int = 3
    poll_interval_seconds: int = 30
    settle_time_seconds: int = 60

    # Paths (can contain environment variables like ${VAR} or $VAR)
    project_root: Optional[str] = None
    report_dir: str = "~/reports"
    hf_home: str = "~/hf"
    edf_env: Optional[str] = None

    # SLURM parameters
    account: Optional[str] = None
    job_name: str = "mmirage-sharded"
    reservation: Optional[str] = None
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus: int = 4
    cpus_per_task: int = 288
    time_limit: str = "11:59:59"

    def __post_init__(self):
        """Validate execution parameters."""
        if self.mode not in ("local", "slurm"):
            raise ValueError(
                f"Invalid execution mode: {self.mode!r}. Must be 'local' or 'slurm'."
            )
        if self.mode == "slurm" and not self.account:
            raise ValueError("account is required when mode='slurm'")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    def is_slurm(self) -> bool:
        """Check if execution mode is SLURM."""
        return self.mode == "slurm"


@dataclass
class ProcessingParams:
    """Parameters for processing dataset samples.

    Defines how input variables are extracted, outputs are generated,
    and the final output schema is constructed.

    Attributes:
        inputs: List of input variables to extract from source datasets.
        outputs: List of output variables to generate using processors.
        output_schema: Dictionary defining the structure of output samples.
        remove_columns: If True, removes all columns from original dataset.
        cast_images: If True, cast generated image path columns to the
            HuggingFace Image feature before saving.
    """

    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]
    remove_columns: bool = False
    cast_images: bool = True

    def __post_init__(self) -> None:
        # Outputs are ordered steps: each one may only read the inputs and the
        # outputs declared before it. Fail at config load rather than in the shard.
        context: List[BaseVar] = list(self.inputs)
        for i, output_var in enumerate(self.outputs):
            if not output_var.is_computable(context):
                raise ValueError(self._uncomputable_message(i, output_var, context))
            # A filter step declares no variable of its own.
            if not isinstance(output_var, FilterStep):
                context.append(output_var)

    @staticmethod
    def _uncomputable_message(
        i: int, output_var: OutputVar, context: List[BaseVar]
    ) -> str:
        # Only some steps (e.g. a filter) can say which names they are missing.
        missing: Set[str] = getattr(output_var, "missing_names", lambda _: set())(
            context
        )
        if missing:
            return (
                f"outputs[{i}] '{output_var.name}' references variables not "
                f"declared before it: {sorted(missing)}"
            )
        deferred: Set[str] = getattr(output_var, "deferred_names", lambda _: set())(
            context
        )
        if deferred:
            names = ", ".join(f"'{v}'" for v in sorted(deferred))
            return (
                f"outputs[{i}]: {names} is a batch_api output, "
                "only available after `mmirage merge`"
            )
        context_names = [v.name for v in context]
        return (
            f"outputs[{i}] '{output_var.name}' is not computable given {context_names}"
        )


@dataclass
class MMirageConfig:
    """Main configuration class for MMIRAGE pipeline.

    Contains all configuration needed to run a MMIRAGE processing pipeline,
    including processor configurations, dataset loading parameters, processing
    parameters, and execution parameters.

    Attributes:
        processors: List of processor configurations for data transformation.
        loading_params: Parameters for loading input datasets.
        processing_params: Parameters for processing dataset samples.
        execution_params: Parameters for executing the pipeline (local/SLURM).
    """

    processors: List[BaseProcessorConfig]
    loading_params: LoadingParams
    processing_params: ProcessingParams
    execution_params: ExecutionParams = field(default_factory=ExecutionParams)
