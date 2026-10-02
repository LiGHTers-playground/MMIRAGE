"""Utility functions for loading datasets and handling images."""

from __future__ import annotations

import os
from typing import List, Optional, Union

from PIL import Image

from mmirage.core.loader.base import AutoDataLoader, BaseDataLoaderConfig, DatasetLike


def load_datasets_from_configs(
    configs: List[BaseDataLoaderConfig],
) -> List[DatasetLike]:
    """Load multiple datasets from configurations.

    Datasets are returned in config order, so callers can pair them with
    their configs by position. Any failed load aborts the whole call.

    Args:
        configs: List of dataset configuration objects.

    Returns:
        List of Hugging Face Datasets/DatasetDicts.

    Raises:
        RuntimeError: If any dataset fails to load (or its loader returns
            None), or if no configs are given.
    """

    valid_ds: List[DatasetLike] = []
    loader_by_type = {}
    for idx, ds_config in enumerate(configs):
        loader = loader_by_type.get(ds_config.type)
        if loader is None:
            loader = AutoDataLoader.from_name(ds_config.type)()
            loader_by_type[ds_config.type] = loader

        where = f"dataset at index {idx} (type={ds_config.type!r}, path={getattr(ds_config, 'path', None)!r})"
        try:
            ds = loader.from_config(ds_config)
        except Exception as e:
            raise RuntimeError(f"Failed to load {where}: {e}") from e
        if ds is None:
            raise RuntimeError(f"Loader returned no data for {where}")
        valid_ds.append(ds)

    if not valid_ds:
        raise RuntimeError("No valid datasets loaded from the provided configs.")

    return valid_ds


def resolve_image_input(
    value: Union[Image.Image, str], image_base_path: Optional[str] = None
) -> Union[Image.Image, str]:
    """Resolve image input to a format SGLang can use.

    Handles multiple image input formats:
    - PIL Image objects: passed through directly
    - URLs (http/https): passed through as-is
    - Absolute file paths: validated and passed through
    - Relative file paths: resolved using image_base_path

    Args:
        value: The image value to resolve (PIL Image, path string, or URL).
        image_base_path: Optional base directory for resolving relative paths.

    Returns:
        Resolved image value suitable for SGLang processing.

    Raises:
        FileNotFoundError: If a relative path cannot be resolved.
        RuntimeError: If an absolute path exists but is not a file.
    """
    # Case 1: Already a PIL Image - pass through
    if isinstance(value, Image.Image):
        return value

    # Case 2: Not a string - pass through (might be other image format)
    if not isinstance(value, str):
        return value

    # Case 3: URL - pass through as-is
    if value.startswith(("http://", "https://")):
        return value

    # Case 4: Absolute path that exists - pass through
    if os.path.isabs(value) and os.path.exists(value):
        if os.path.isfile(value):
            return value
        elif os.path.islink(value):
            return os.path.realpath(value)
        else:
            raise RuntimeError(f"The provided path {value} exists but is not a file")

    # Case 5: Relative path - try to resolve with base path
    if image_base_path:
        resolved_path = os.path.join(image_base_path, value)
        if os.path.exists(resolved_path):
            return resolved_path
        raise FileNotFoundError(
            f"Resolved image path '{resolved_path}' does not exist "
            f"(from base '{image_base_path}' and relative path '{value}')."
        )

    # Case 6: No base path - return as-is and let SGLang handle it
    return value
