"""Configuration for the filter step.

A filter is a step in ``processing_params.outputs`` that drops rows whose
Jinja2 expression predicate is falsy. It produces no variable of its own.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Set, Tuple

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, meta

from mmirage.core.process.base import BaseProcessorConfig, ProcessorRegistry
from mmirage.core.process.variables import BaseVar, OutputVar, VariableEnvironment

# Strict so that a missing variable, a `None` value, or a misspelled attribute
# raises instead of evaluating to `None` (which the default environment does
# silently, turning a typo into "drop every row").
env = Environment(undefined=StrictUndefined)


@dataclass
class FilterResult:
    """Outcome of applying a filter step to a batch of variable environments."""

    kept: List[VariableEnvironment] = field(default_factory=list)
    n_filtered: int = 0
    n_errored: int = 0
    # Row position within the batch and the exception, for the first row whose
    # predicate raised.
    first_error: Optional[Tuple[int, BaseException]] = None


@dataclass
class FilterStep(OutputVar):
    """Step that keeps only the rows whose predicate is truthy.

    Attributes:
        predicate: Jinja2 expression over the variables declared before this
            step, e.g. ``score > 0.5`` or ``verdict.keep``. A surrounding
            ``{{ ... }}`` is accepted and stripped.
        name: Unused; a filter step declares no variable.
    """

    type: Literal["filter"] = "filter"
    predicate: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        expr = self.predicate.strip()
        if expr.startswith("{{") and expr.endswith("}}"):
            expr = expr[2:-2].strip()
        if not expr:
            raise ValueError("filter step needs a `predicate`")
        self.predicate = expr
        try:
            self._expr = env.compile_expression(expr, undefined_to_none=False)
        except TemplateSyntaxError as exc:
            raise ValueError(
                f"filter step predicate {self.predicate!r} is not a valid Jinja2 "
                f"expression: {exc}"
            ) from exc

    def referenced_names(self) -> Set[str]:
        """Names of the variables the predicate reads."""
        # `meta` only reports names inside a template; a bare expression yields none.
        return meta.find_undeclared_variables(env.parse("{{ " + self.predicate + " }}"))

    def missing_names(self, vars: Sequence[BaseVar]) -> Set[str]:
        """Referenced names that no earlier variable declares."""
        return self.referenced_names() - {v.name for v in vars}

    def deferred_names(self, vars: Sequence[BaseVar]) -> Set[str]:
        """Referenced names whose value only exists after the shard map."""
        referenced = self.referenced_names()
        return {
            v.name
            for v in vars
            if v.name in referenced and not getattr(v, "available_at_map_time", True)
        }

    def is_computable(self, vars: Sequence[BaseVar]) -> bool:
        return not self.missing_names(vars) and not self.deferred_names(vars)

    def apply(self, envs: Sequence[VariableEnvironment]) -> FilterResult:
        """Evaluate the predicate on each environment.

        A row whose predicate raises (missing value, `None` in a comparison,
        misspelled attribute) is dropped and counted separately from a falsy one.
        """
        result = FilterResult()
        for position, env_ in enumerate(envs):
            try:
                keep = bool(self._expr(**env_.to_dict()))
            except Exception as exc:
                result.n_errored += 1
                if result.first_error is None:
                    result.first_error = (position, exc)
                continue
            if keep:
                result.kept.append(env_)
            else:
                result.n_filtered += 1
        return result


@dataclass
class FilterProcessorConfig(BaseProcessorConfig):
    """Rejects a `type: filter` entry under `processors` at config load."""

    type: Literal["filter"] = "filter"

    def __post_init__(self) -> None:
        raise ValueError(
            "`filter` is a step in `processing_params.outputs`, not a processor"
        )


ProcessorRegistry.register_types("filter", FilterProcessorConfig, FilterStep)
