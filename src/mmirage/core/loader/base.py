"""Base classes and registry for data loaders in MMIRAGE."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from datasets import Dataset, DatasetDict


@dataclass
class BaseDataLoaderConfig:
    """Base configuration class for data loaders.

    All data loader configurations must inherit from this class and
    specify a type identifier.

    Attributes:
        type: String identifier for the loader type (e.g., "JSONL", "loadable").
        output_dir: Directory path for saving processed output shards.
        image_base_path: Optional base directory for resolving relative image paths in this dataset.
    """

    type: str
    output_dir: str
    image_base_path: Optional[str] = None


C = TypeVar("C", bound=BaseDataLoaderConfig)

DatasetLike = Dataset | DatasetDict


class BaseDataLoader(abc.ABC, Generic[C]):
    """Abstract base class for data loaders.

    Data loaders are responsible for loading datasets from various sources
    (JSONL files, Hugging Face datasets, etc.) and returning them as
    Hugging Face Dataset objects.

    Type Parameters:
        C: The configuration class type for this loader.

    Methods:
        from_config: Load a dataset from the given configuration.
    """

    @abc.abstractmethod
    def from_config(self, ds_config: C) -> Optional[DatasetLike]:
        """Load a dataset from the given configuration.

        Args:
            ds_config: Configuration object for loading the dataset.

        Returns:
            A Hugging Face Dataset or DatasetDict, or None if loading fails.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError()


class DataLoaderRegistry:
    """Registry for managing and accessing available data loaders.

    Provides a centralized registry for data loader classes and their
    associated configuration classes, allowing dynamic loader instantiation
    based on type names.

    Attributes:
        _registry: Mapping from loader name to registered loader class.
        _config_registry: Mapping from loader name to its configuration class.
    """

    _registry = dict()
    _config_registry = dict()

    @classmethod
    def register(cls, name: str, config_cls: Type[BaseDataLoaderConfig]) -> Callable:
        """Register a data loader class.

        Args:
            name: String identifier for the loader.
            config_cls: Configuration class associated with this loader.

        Returns:
            Decorator function to register the loader class.
        """

        def inner_register(clazz: Any):
            cls._registry[name] = clazz
            cls._config_registry[name] = config_cls
            return clazz

        return inner_register

    @classmethod
    def get_processor(cls, name: str) -> Type[BaseDataLoader]:
        """Get a registered loader class by name.

        Args:
            name: String identifier of the loader.

        Returns:
            The registered loader class.

        Raises:
            ValueError: If no loader is registered under the given name.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Loader {name} not registered. Available loaders are {list(cls._registry.keys())}"
            )

        return cls._registry[name]

    @classmethod
    def get_config_cls(cls, name: str) -> Type[BaseDataLoaderConfig]:
        """Get a registered configuration class by loader name.

        Args:
            name: String identifier of the loader.

        Returns:
            The registered configuration class.

        Raises:
            ValueError: If no loader is registered under the given name.
        """
        if name not in cls._config_registry:
            raise ValueError(
                f"Loader {name} not registered. Available loaders are {list(cls._config_registry.keys())}"
            )

        return cls._config_registry[name]


class AutoDataLoader:
    """Factory class for instantiating data loaders by name."""

    @classmethod
    def from_name(cls, name: str) -> Type[BaseDataLoader]:
        """Retrieve a data loader class by its registered name.

        Args:
            name: The registry name of the data loader.

        Returns:
            The registered data loader class.

        Raises:
            ValueError: If no data loader is registered under the given name.
        """
        return DataLoaderRegistry.get_processor(name)
