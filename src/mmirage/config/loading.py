"""Data loading configuration for MMIRAGE pipeline."""

import os
import re
from dataclasses import dataclass, field
from typing import List, Union, cast

from mmirage.core.loader.base import BaseDataLoaderConfig

DEFAULT_STATE_DIR = "~/.cache/MMIRAGE/state_dir"

_UNRESOLVED_ENV_VAR_PATTERN = re.compile(
    r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$"
)


@dataclass
class LoadingParams:
    """Parameters for loading and distributing datasets across shards.

    Defines how datasets are loaded and processed in a distributed manner,
    supporting sharding for parallel processing.

    Attributes:
        datasets: List of dataset configurations to load.
        state_dir: Shared directory for logical shard state/markers/retry tracking.
        output_dir: Legacy top-level output directory. Prefer per-dataset output_dir.
        num_shards: Total number of shards to split the dataset into.
        shard_id: ID of this shard (0-indexed).
        batch_size: Batch size for processing samples.

    Raises:
        ValueError: If num_shards, shard_id, or batch_size is not an int, if
            num_shards or batch_size is < 1, if shard_id is outside
            [0, num_shards), or if num_shards references an unset environment
            variable. An unset shard_id variable falls back to 0; see
            is_shard_id_unresolved().
    """

    datasets: List[BaseDataLoaderConfig] = field(default_factory=list)
    state_dir: str = DEFAULT_STATE_DIR
    output_dir: str = ""
    num_shards: Union[int, str] = 1
    shard_id: Union[int, str] = 0
    batch_size: Union[int, str] = 1

    def __post_init__(self):
        def to_int(name: str, value: Union[int, str]) -> int:
            if isinstance(value, str) and _UNRESOLVED_ENV_VAR_PATTERN.fullmatch(
                value.strip()
            ):
                raise ValueError(
                    f"{name}={value!r} references an environment variable that is not set"
                )
            try:
                return int(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {name}: {value!r}") from None

        # The launcher loads the config before SLURM_ARRAY_TASK_ID exists, so an
        # unresolved shard_id falls back to 0 here; shard_process rejects it.
        # Plain attribute, not a dataclass field: dacite resets non-init fields.
        self._shard_id_unresolved = isinstance(self.shard_id, str) and bool(
            _UNRESOLVED_ENV_VAR_PATTERN.fullmatch(self.shard_id.strip())
        )
        if self._shard_id_unresolved:
            self.shard_id = 0

        self.num_shards = to_int("num_shards", self.num_shards)
        self.shard_id = to_int("shard_id", self.shard_id)
        self.batch_size = to_int("batch_size", self.batch_size)

        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
        if not 0 <= self.shard_id < self.num_shards:
            raise ValueError(
                f"shard_id must be in [0, {self.num_shards}), got {self.shard_id}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        raw_state_dir = "" if self.state_dir is None else str(self.state_dir)
        self.state_dir = raw_state_dir.strip()
        if not self.state_dir:
            self.state_dir = DEFAULT_STATE_DIR

        self.state_dir = os.path.expanduser(self.state_dir)

    def get_state_root(self) -> str:
        """Get the state root path.

        Returns:
            str: State root path.
        """
        return self.state_dir

    def get_num_shards(self) -> int:
        """Get the total number of shards.

        Returns:
            int: Total number of shards.
        """
        return cast(int, self.num_shards)

    def get_shard_id(self) -> int:
        """Get the ID of this shard.

        Returns:
            int: Shard ID (0-indexed).
        """
        return cast(int, self.shard_id)

    def is_shard_id_unresolved(self) -> bool:
        """Whether shard_id fell back to 0 because its environment variable was unset.

        Returns:
            bool: True if shard_id came from an unresolved environment variable.
        """
        return self._shard_id_unresolved

    def get_batch_size(self) -> int:
        """Get the batch size for processing.

        Returns:
            int: Batch size (minimum 1).
        """
        return cast(int, self.batch_size)
