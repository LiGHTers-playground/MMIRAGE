"""Configuration for LLM processor in MMIRAGE."""

import builtins
import logging
import os
import re
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Type

from jinja2 import Environment, meta
from pydantic import BaseModel, create_model
from pydantic import Field as PydanticField

from mmirage.core.process.base import BaseProcessorConfig, ProcessorRegistry
from mmirage.core.process.variables import BaseVar, OutputVar

logger = logging.getLogger(__name__)
env = Environment()


def _parse_tp_size_from_env() -> int:
    """Parse tensor parallelism size from SLURM_GPUS_ON_NODE environment variable.

    Defensively parses the environment variable, handling invalid values:
    - Returns 1 if the variable is None or empty
    - Strips whitespace before parsing
    - Returns 1 for non-integer values
    - Returns 1 for values <= 0

    Returns:
        Tensor parallelism size (>= 1), defaults to 1 on any parsing error.
    """
    env_value = os.environ.get("SLURM_GPUS_ON_NODE")
    if not env_value:
        return 1

    try:
        tp_size = int(env_value.strip())
        # Ensure tp_size is positive (must be >= 1)
        if tp_size <= 0:
            logger.warning(
                f"Invalid SLURM_GPUS_ON_NODE value '{env_value}' (must be > 0), defaulting tp_size to 1"
            )
            return 1
        return tp_size
    except ValueError:
        # ValueError: invalid integer format
        logger.warning(
            f"Invalid SLURM_GPUS_ON_NODE value '{env_value}', defaulting tp_size to 1"
        )
        return 1


@dataclass
class SGLangServerArgs:
    """Server arguments for SGLang engine.

    Attributes:
        model_path: Path to the model or HuggingFace model ID.
        tp_size: Tensor parallelism size.
        trust_remote_code: Whether to trust remote code from HuggingFace.
        disable_custom_all_reduce: Whether to disable custom all reduce.
        extra_engine_args: Any additional keyword arguments forwarded verbatim
            to ``sgl.Engine``. Use this to pass SGLang-specific options that
            are not listed above, e.g.::

                extra_engine_args:
                  max_running_requests: 512
                  chunked_prefill_size: 32768
                  mem_fraction_static: 0.88
    """

    model_path: str = "none"
    tp_size: int = field(default_factory=_parse_tp_size_from_env)
    trust_remote_code: bool = True
    disable_custom_all_reduce: bool = False
    extra_engine_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SGLangLLMConfig(BaseProcessorConfig):
    """Configuration for LLM processor using SGLang.

    Supports both text-only and multimodal (vision-language) models.

    Attributes:
        server_args: SGLang server arguments including model path and TP size.
        default_sampling_params: Default sampling parameters for generation.
        chat_template: Chat template name for vision-language models (e.g., "qwen2-vl").
    """

    type: Literal["llm"] = "llm"
    server_args: SGLangServerArgs = field(default_factory=SGLangServerArgs)
    default_sampling_params: Dict[str, Any] = field(default_factory=dict)
    chat_template: str = ""  # Empty means use tokenizer's default


_NUMERIC_BOUND_RE = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class LLMSchemaField:
    """One field of a JSON `output_schema`: a type name plus optional bounds.

    Validates itself on construction, so an invalid spec fails as soon as it
    is built.

    Attributes:
        type: Schema type name ("str", "int", "float" or "bool", or one of
            their aliases "string", "integer", "number", "boolean").
        min: Inclusive lower bound, numeric types only. Numeric strings
            (e.g. produced by `${ENV_VAR}` expansion) are coerced.
        max: Inclusive upper bound, same rules as `min`.
    """

    TYPE_MAP: ClassVar[dict[str, type]] = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
    }

    type: str
    min: int | float | str | None = None
    max: int | float | str | None = None

    def __post_init__(self) -> None:
        py_type = self.python_type()
        if self.min is None and self.max is None:
            return

        if py_type not in (int, float):
            raise ValueError(
                f"'min'/'max' are only allowed for numeric types, "
                f"got type '{self.type}'."
            )
        self.min = min_val = self._coerce_bound("min", self.min)
        self.max = max_val = self._coerce_bound("max", self.max)
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min {min_val} cannot be greater than max {max_val}.")

    @classmethod
    def from_spec(cls, spec: "SchemaFieldSpec") -> "LLMSchemaField":
        """Normalize any accepted `output_schema` entry form into a field."""
        if isinstance(spec, LLMSchemaField):
            return spec
        if isinstance(spec, str):
            return cls(type=spec)

        allowed = {f.name for f in fields(cls)}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(
                f"Unknown key(s) {sorted(unknown)}. Allowed keys: {sorted(allowed)}."
            )
        if "type" not in spec:
            raise ValueError("missing required key 'type'.")
        return cls(type=str(spec["type"]), min=spec.get("min"), max=spec.get("max"))

    def python_type(self) -> builtins.type:
        """Map the schema type name ("int", "str", ...) to its Python type."""
        py_type = self.TYPE_MAP.get(str(self.type).strip().lower())
        if py_type is None:
            raise ValueError(
                f"Unsupported type '{self.type}'. "
                f"Supported types: {sorted(self.TYPE_MAP)}."
            )
        return py_type

    def field_type(self) -> object:
        """The field's type for `create_model`, carrying any declared bounds."""
        py_type = self.python_type()
        if not self.has_bounds:
            return py_type
        return Annotated[py_type, PydanticField(ge=self.min, le=self.max)]

    @property
    def has_bounds(self) -> bool:
        """Whether the field declares a `min`/`max` bound."""
        return self.min is not None or self.max is not None

    def _coerce_bound(
        self, key: str, value: int | float | str | None
    ) -> int | float | None:
        """Validate one bound and coerce it to the field's numeric type."""
        if value is None:
            return None
        number: int | float
        if isinstance(value, str):
            if not _NUMERIC_BOUND_RE.match(value.strip()):
                raise ValueError(f"'{key}' must be a number, got {value!r}.")
            number = float(value)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be a number, got {value!r}.")
        else:
            number = value
        if self.python_type() is int:
            if int(number) != number:
                raise ValueError(
                    f"'{key}' must be a whole number for an int field, got {value!r}."
                )
            return int(number)
        return float(number)


SchemaFieldSpec = str | dict[str, str | int | float | None] | LLMSchemaField
"""Accepted forms for one `output_schema` entry: a type-name shorthand, a raw
`type`/`min`/`max` mapping (kept raw so unknown keys are still rejected), or an
already-built `LLMSchemaField`."""


@dataclass
class LLMOutputVar(OutputVar):
    """Output variable generated by LLM processor.

    Uses Jinja2 templating for prompts and supports both plain text
    and structured JSON outputs.

    Attributes:
        name: Name of the variable.
        type: Type identifier (must be "llm").
        prompt: Jinja2 template for the LLM prompt.
        output_schema: JSON output fields, either a list of field names
            (all typed as strings) or a mapping of field name to an
            `LLMSchemaField` spec: a type name ("str", "int", "float" or
            "bool") or a nested mapping with keys `type` (required) and, for
            numeric types, optional `min`/`max` bounds enforced during
            constrained decoding. Empty for plain text.
        output_type: Output format - "JSON" or "plain".
    """

    prompt: str = ""
    output_schema: list[str] | dict[str, SchemaFieldSpec] = field(default_factory=list)
    output_type: str = ""

    def __post_init__(self) -> None:
        # Surface invalid schemas at config load, not mid-shard.
        self.get_output_schema()

    def _parse_spec(self, var: str, spec: SchemaFieldSpec) -> LLMSchemaField:
        """Validate one `output_schema` entry and normalize it to a field."""
        try:
            return LLMSchemaField.from_spec(spec)
        except ValueError as exc:
            raise ValueError(
                f"Field '{var}' in output_schema of '{self.name}': {exc}"
            ) from exc

    def has_schema_constraints(self) -> bool:
        """Whether any field declares a `min`/`max` bound."""
        if not isinstance(self.output_schema, dict):
            return False
        return any(
            self._parse_spec(var, spec).has_bounds
            for var, spec in self.output_schema.items()
        )

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        """Generate a Pydantic model for JSON output validation.

        Returns:
            A Pydantic BaseModel class if output_type is "JSON" and
            output_schema is non-empty, otherwise None.

        Raises:
            ValueError: If output_schema maps a field to an unsupported type
                name or to an invalid constraint mapping.
        """
        if self.output_type == "JSON" and self.output_schema:
            if isinstance(self.output_schema, dict):
                field_defs: dict[str, Any] = {
                    var: (self._parse_spec(var, spec).field_type(), ...)
                    for var, spec in self.output_schema.items()
                }
            else:
                field_defs = {var: (str, ...) for var in self.output_schema}
            return create_model("OutputSchema", **field_defs)
        return None

    def is_computable(self, vars: List[BaseVar]) -> bool:
        """Check if all variables referenced in the prompt are available.

        Args:
            vars: Sequence of currently available variables.

        Returns:
            True if all template variables are declared, False otherwise.
        """
        parsed_content = env.parse(self.prompt)
        template_vars = meta.find_undeclared_variables(parsed_content)

        var_names = set(map(lambda v: v.name, vars))
        undeclared_vars = template_vars - var_names

        if len(undeclared_vars) > 0:
            logger.info(
                f"⚠️ Undeclared variables found for {self.name}: {undeclared_vars}"
            )
            return False

        return True


ProcessorRegistry.register_types("llm", SGLangLLMConfig, LLMOutputVar)
