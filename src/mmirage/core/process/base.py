"""Base classes and registry for processors in MMIRAGE."""

import abc
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Generic, List, Type, TypeVar

from mmirage.core.process.variables import OutputVar, VariableEnvironment


@dataclass
class BaseProcessorConfig:
    """Base configuration class for processors.

    All processor configurations must inherit from this class.

    Attributes:
        type: Registry key of the processor. Concrete configs narrow this to
            their own literal (e.g. ``Literal["llm"] = "llm"``).
    """

    type: str


C = TypeVar("C", bound=OutputVar)


@dataclass
class TokenCounts:
    """Cumulative token counts from LLM processors."""

    input_tokens: int
    output_tokens: int


class BaseProcessor(abc.ABC, Generic[C]):
    """Abstract base class for data processors.

    Processors are responsible for transforming data by generating
    new output variables from existing variables.

    Type Parameters:
        C: The output variable type this processor works with.

    Attributes:
        config: Configuration object for this processor.
    """

    def __init__(
        self, config: BaseProcessorConfig, shard_id: int = 0, **kwargs
    ) -> None:
        """Initialize the processor with configuration.

        Args:
            config: Configuration object for this processor.
            shard_id: Optional shard identifier accepted for compatibility
                with callers that forward it during processor construction.
            **kwargs: Additional keyword arguments. Any unexpected keyword
                arguments will raise ``TypeError``.

        Raises:
            TypeError: If unexpected keyword arguments are provided.
        """
        if kwargs:
            unexpected_args = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected keyword argument(s) for "
                f"{self.__class__.__name__}: {unexpected_args}"
            )
        super().__init__()
        self.config = config
        self.shard_id = shard_id

    @abc.abstractmethod
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: C
    ) -> List[VariableEnvironment]:
        """Process a batch of variable environments.

        Args:
            batch: List of variable environments to process.
            output_var: Output variable definition to generate.

        Returns:
            List of updated variable environments with the new output variable.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError()

    def begin_batch(self, source_indices: List[int]) -> None:
        """Called before ``batch_process_sample`` when the rows' origin is known.

        Args:
            source_indices: For each row of the coming batch, its position in
                the shard. Rows dropped by an earlier filter step leave gaps.
        """

    def finalize(self) -> None:
        """Optional lifecycle hook; override when a processor buffers state."""
        pass

    @abc.abstractmethod
    def get_token_counts(self) -> TokenCounts:
        """Get cumulative token counts from this processor.

        Returns:
            TokenCounts object containing input and output token counts.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_load_time(self) -> float:
        """Get the time taken to load any necessary resources (e.g., models).

        Returns:
            Time in seconds taken to load resources.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError()

    def shutdown(self) -> None:
        """Release any resources held by this processor.

        Override in subclasses that hold GPU memory, open file handles, or
        network connections.  The default implementation is a no-op.
        """


class ProcessorRegistry:
    """Registry for managing and accessing available processors.

    Provides a centralized registry for processor classes, their
    configuration classes, and their output variable classes.

    Attributes:
        _registry: Mapping from processor name to registered processor class.
        _config_registry: Mapping from processor name to its configuration class.
        _output_var_registry: Mapping from processor name to its output variable class.
    """

    _registry = dict()
    _config_registry = dict()
    _output_var_registry = dict()

    # Import processor implementations lazily because they may depend on heavy
    # libraries (torch/transformers). Config/output-var types are registered via
    # mmirage.config.utils importing the relevant config modules.
    _lazy_processor_imports = {
        "llm": "mmirage.core.process.processors.llm.llm_processor",
        "image_gen": "mmirage.core.process.processors.image_gen.image_gen_processor",
        "custom": "mmirage.core.process.processors.custom.custom_processor",
        "batch_api": "mmirage.core.process.processors.batch_api.batch_api_processor",
    }

    @classmethod
    def register_types(
        cls,
        name: str,
        config_cls: Type[BaseProcessorConfig],
        output_var_cls: Type[OutputVar],
    ) -> None:
        """Register config/output-var types without importing processor implementations."""
        cls._config_registry[name] = config_cls
        cls._output_var_registry[name] = output_var_cls

    @classmethod
    def _maybe_import_processor(cls, name: str) -> None:
        module = cls._lazy_processor_imports.get(name)
        if module:
            import_module(module)

    @classmethod
    def register(
        cls,
        name: str,
        config_cls: Type[BaseProcessorConfig],
        output_var_cls: Type[OutputVar],
    ) -> Callable:
        """Register a processor class with its associated classes.

        Args:
            name: String identifier for the processor.
            config_cls: Configuration class associated with this processor.
            output_var_cls: Output variable class associated with this processor.

        Returns:
            Decorator function to register the processor class.
        """

        def inner_register(clazz):
            cls._registry[name] = clazz
            cls._config_registry[name] = config_cls
            cls._output_var_registry[name] = output_var_cls
            return clazz

        return inner_register

    @classmethod
    def get_processor(cls, name: str) -> Type[BaseProcessor]:
        """Get a registered processor class by name.

        Args:
            name: String identifier of the processor.

        Returns:
            The registered processor class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        if name not in cls._registry:
            cls._maybe_import_processor(name)

        if name not in cls._registry:
            raise ValueError(
                f"Processor {name} not registered. Available processors are {list(cls._registry.keys())}"
            )

        return cls._registry[name]

    @classmethod
    def get_config_cls(cls, name: str) -> Type[BaseProcessorConfig]:
        """Get a registered configuration class by processor name.

        Args:
            name: String identifier of the processor.

        Returns:
            The registered configuration class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        if name not in cls._config_registry:
            cls._maybe_import_processor(name)

        if name not in cls._config_registry:
            raise ValueError(
                f"Processor {name} not registered. Available processors are {list(cls._config_registry.keys())}"
            )

        return cls._config_registry[name]

    @classmethod
    def get_output_var_cls(cls, name: str) -> Type[OutputVar]:
        """Get a registered output variable class by processor name.

        Args:
            name: String identifier of the processor.

        Returns:
            The registered output variable class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        if name not in cls._output_var_registry:
            cls._maybe_import_processor(name)

        if name not in cls._output_var_registry:
            raise ValueError(
                f"Processor {name} not registered. Available processors are {list(cls._output_var_registry.keys())}"
            )

        return cls._output_var_registry[name]


class AutoProcessor:
    """Factory class for instantiating processors by name."""

    @classmethod
    def from_name(cls, name: str) -> Type[BaseProcessor]:
        """Retrieve a processor class by its registered name.

        Args:
            name: The registry name of the processor.

        Returns:
            The registered processor class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        return ProcessorRegistry.get_processor(name)
