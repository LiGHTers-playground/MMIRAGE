"""Helpers for resolving batch metadata receipt paths."""

from __future__ import annotations

import glob
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from mmirage.config.config import MMirageConfig

_METADATA_SUFFIXES = ("text", "multimodal")


def _base_path_to_patterns(base_path: str) -> List[str]:
    trimmed = base_path[:-6] if base_path.endswith(".jsonl") else base_path
    return [f"{trimmed}.{suffix}.*.jsonl" for suffix in _METADATA_SUFFIXES]


def resolve_metadata_paths_from_config(
    metadata_output_paths: List[str],
) -> List[str]:
    """Return metadata receipt paths for config-provided base paths.

    Submission writes suffixed receipts using .text.<run>.jsonl and
    .multimodal.<run>.jsonl, so we expand base paths into matching globs.
    """
    patterns: List[str] = []
    resolved: List[str] = []

    for base_path in metadata_output_paths:
        for pattern in _base_path_to_patterns(base_path):
            patterns.append(pattern)
            matches = sorted(glob.glob(pattern))
            if matches:
                resolved.extend(matches)

    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        pattern_list = ", ".join(patterns) if patterns else "<none>"
        raise ValueError(
            "No metadata receipts matched config metadata_output_path patterns. "
            f"Tried: {pattern_list}. Expected suffixed files like "
            "'<base>.text.<run>.jsonl' or '<base>.multimodal.<run>.jsonl'."
        )

    return resolved


def resolve_metadata_paths(
    cfg: "MMirageConfig",
    metadata_paths: Optional[List[str]] = None,
) -> List[str]:
    """Return receipt paths from explicit values, or from the config batch blocks."""
    if metadata_paths:
        return list(metadata_paths)

    from mmirage.core.process.batch.provider_resolution import (
        build_all_provider_configs,
    )

    base_paths = list(
        dict.fromkeys(
            config.metadata_output_path
            for config in build_all_provider_configs(cfg).values()
            if config.metadata_output_path
        )
    )
    if not base_paths:
        raise ValueError(
            "No metadata paths provided and none found in config batch_api processor blocks."
        )

    return resolve_metadata_paths_from_config(base_paths)
