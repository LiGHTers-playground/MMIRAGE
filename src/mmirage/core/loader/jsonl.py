"""JSONL data loader implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union, override

from datasets import (
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from mmirage.core.loader.base import (
    BaseDataLoader,
    BaseDataLoaderConfig,
    DataLoaderRegistry,
    DatasetLike,
)


@dataclass
class JSONLDataConfig(BaseDataLoaderConfig):
    """Configuration for loading JSONL datasets.

    Attributes:
        type: Type identifier (must be "JSONL").
        path: File path to the JSONL file, or dict mapping split names to paths.
        output_dir: Directory for saving processed output.
    """

    path: Union[str, Dict[str, str]] = ""


@DataLoaderRegistry.register("JSONL", JSONLDataConfig)
class JSONLDataLoader(BaseDataLoader[JSONLDataConfig]):
    """Data loader for JSONL (JSON Lines) formatted datasets.

    Loads datasets from JSONL files using the Hugging Face datasets library.
    Supports both single files and split-based loading.

    Note:
        Iterable datasets are not supported by this loader.
    """

    def __init__(self) -> None:
        """Initialize the JSONL data loader."""
        super().__init__()

    @override
    def from_config(self, ds_config: JSONLDataConfig) -> Optional[DatasetLike]:
        """Load a dataset from a JSONL file.

        Args:
            ds_config: Configuration containing the path to the JSONL file.

        Returns:
            A Hugging Face Dataset or a DatasetDict containing the JSONL data.

        Raises:
            RuntimeError: If the loaded dataset is an iterable dataset.
        """
        path = ds_config.path

        ds = load_dataset("json", data_files=path, streaming=False)

        if isinstance(ds, (IterableDatasetDict, IterableDataset)):
            raise RuntimeError(f"Iterable datasets are not supported for path: {path}")

        if isinstance(path, str):
            # If we only have a single split, we load it as a standard Dataset
            ds = ds["train"]

        return ds
