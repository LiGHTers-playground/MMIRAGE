"""Worker functions for the custom processor's process pool.

These functions are designed to be safe for a 'spawn' multiprocessing context.
The user's module is dynamically loaded within the worker process upon initialization
to avoid PicklingError when passing complex objects across process boundaries.
"""

import importlib.util
import os
import sys
import uuid
from typing import Any, Callable, Dict

# global state within the isolated worker process
_USER_FUNCTION: Callable[[Dict[str, Any]], Any] | None = None


def initialize_worker(script_path: str, function_name: str) -> None:
    """Initialize the worker process by dynamically loading the user's script.

    This runs exactly once per worker process when it boots.

    Args:
        script_path: The path to the user's Python script.
        function_name: The name of the function to execute inside the script.

    Raises:
        FileNotFoundError: If the script path does not exist.
        RuntimeError: If the module cannot be loaded.
        AttributeError: If the function name is not found.
        TypeError: If the target attribute is not callable.
    """
    global _USER_FUNCTION

    abs_script_path = os.path.abspath(script_path)
    if not os.path.exists(abs_script_path):
        raise FileNotFoundError(f"Custom script not found at: {abs_script_path}")

    module_name = (
        f"mmirage_custom_worker_{uuid.uuid4().hex}"  # ensure uniqueness in sys.modules
    )

    try:
        # load the module
        spec = importlib.util.spec_from_file_location(module_name, abs_script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {abs_script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Add the module's directory to sys.path so the user script can
        # seamlessly import other local files relative to its location
        script_dir = os.path.dirname(abs_script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        spec.loader.exec_module(module)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load or execute custom module from {abs_script_path}"
        ) from e
    # get the callable user function
    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in script '{abs_script_path}'"
        )

    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"'{function_name}' in '{abs_script_path}' is not callable")

    _USER_FUNCTION = func


def execute_custom_function(row_dict: Dict[str, Any]) -> Any:
    """Execute the dynamically loaded user function on a single data row.

    Args:
        row_dict: A dictionary representation of the VariableEnvironment for the current row.

    Returns:
        The result of the user's function.

    Raises:
        RuntimeError: If the worker was not initialized properly.
        Exception: Any standard Python exception raised by the user's function
                   will bubble up to be handled by the parent process.
    """
    if _USER_FUNCTION is None:
        raise RuntimeError(
            "Worker not initialized. 'initialize_worker' must be called first."
        )

    return _USER_FUNCTION(row_dict)
