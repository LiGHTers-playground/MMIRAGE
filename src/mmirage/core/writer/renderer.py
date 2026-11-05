"""Template renderer for generating output from variable environments."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from jinja2 import Environment, Template, meta
from jinja2.nodes import Name, Output
from PIL import Image

from mmirage.core.process.variables import VariableEnvironment

logger = logging.getLogger(__name__)

JINJA_ENV = Environment()


class TemplateRenderer:
    """Renderer for generating output from variable environments using Jinja2 templates.

    Supports nested templates (dicts and lists), optimized handling of
    simple variable references, and proper handling of non-string values
    like PIL Images.

    Attributes:
        output_schema: Dictionary defining the structure of output samples.
    """

    def __init__(self, output_schema: Dict[str, Any]) -> None:
        """Initialize the template renderer.

        Args:
            output_schema: Dictionary defining the structure of output samples.
                Values can be strings (Jinja2 templates), lists, or dicts.
        """
        self.output_schema = output_schema

    def batch_render(self, batch: List[VariableEnvironment]) -> Dict[str, List[Any]]:
        """Render a batch of variable environments.

        Args:
            batch: List of variable environments to render.

        Returns:
            Dictionary mapping output keys to lists of rendered values.
        """
        rendered_batch = defaultdict(list)
        for env in batch:
            for key, template_obj in self.output_schema.items():
                rendered_batch[key].append(
                    self._fill_template_recursive(template_obj, env)
                )

        return rendered_batch

    def undeclared_variables(self, available: Set[str]) -> Set[str]:
        """Return top-level variable names used by output_schema but not available.

        Args:
            available: Names of the variables that will be defined at render time.

        Returns:
            Names referenced by templates anywhere in output_schema (nested dicts
            and lists included) that are not in ``available``.
        """

        def collect(obj: Any) -> Set[str]:
            if isinstance(obj, dict):
                return set().union(*map(collect, obj.values()))
            if isinstance(obj, list):
                return set().union(*map(collect, obj))
            if isinstance(obj, str):
                return meta.find_undeclared_variables(JINJA_ENV.parse(obj))
            return set()

        return collect(self.output_schema) - available

    def is_single_variable_template(self, s: str) -> Optional[str]:
        """Check if a string is a simple single-variable template.

        Args:
            s: String to check.

        Returns:
            The variable name if s is exactly '{{ var }}', otherwise None.
        """
        ast = JINJA_ENV.parse(s)

        if len(ast.body) != 1:
            return None

        node = ast.body[0]

        if not isinstance(node, Output) or len(node.nodes) != 1:
            return None

        expr = node.nodes[0]

        if isinstance(expr, Name):
            return expr.name

        return None

    def _fill_template_recursive(
        self, template_obj: Any, context: VariableEnvironment
    ) -> Any:
        """Recursively fill a template object with values from a variable environment.

        Properly handles non-string values like PIL Images by returning them
        directly when the template is a simple variable reference.

        Args:
            template_obj: Template object (str, dict, list, or other type).
            context: Variable environment containing variable values.

        Returns:
            The rendered object with templates filled with variable values.
        """
        if isinstance(template_obj, dict):
            return {
                k: self._fill_template_recursive(v, context)
                for k, v in template_obj.items()
            }

        elif isinstance(template_obj, list):
            return [self._fill_template_recursive(v, context) for v in template_obj]

        elif isinstance(template_obj, str):
            # Check if this is a simple variable reference like "{{ image }}"
            # If so and it's a special type (PIL Image, etc.), return it directly
            var_name = self.is_single_variable_template(template_obj)
            if var_name is not None:
                value = context.get(var_name)
                if value is not None:
                    # Preserve PIL Images and other non-string objects
                    if isinstance(value, Image.Image):
                        return value
                    # For complex types that shouldn't be stringified
                    if not isinstance(value, (str, int, float, bool)):
                        return value

            # Normal Jinja2 template rendering
            template = Template(template_obj)
            return template.render(**context.to_dict())

        else:
            # Non-string, non-dict, non-list: return as-is
            return template_obj
