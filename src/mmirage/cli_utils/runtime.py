"""Runtime/path helpers for the MMIRAGE CLI."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

from mmirage.config.config import MMirageConfig

logger = logging.getLogger(__name__)


def non_empty_path(value: str) -> str:
    """Reject a path argument given as an empty or blank string.

    An empty value reads as "flag not passed" further down, which would turn a
    dry run into a real submission.
    """
    if not value.strip():
        raise argparse.ArgumentTypeError("expected a non-empty path")
    return value


def expand_path(path: str, project_root: Optional[str] = None) -> str:
    """Expand environment variables, user home and relative paths."""
    expanded = Path(os.path.expandvars(os.path.expanduser(path)))
    if not expanded.is_absolute() and project_root:
        expanded = Path(project_root) / expanded
    return str(expanded.resolve())


def get_project_root(cfg: MMirageConfig) -> str:
    """Return the configured project root, or the current working directory."""
    project_root = cfg.execution_params.project_root
    if project_root:
        return expand_path(project_root)
    return os.getcwd()


def create_directories(paths: List[str]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def validate_edf_env_path(cfg: MMirageConfig) -> None:
    """Validate the optional EDF environment file path."""
    edf_env = cfg.execution_params.edf_env
    if not edf_env:
        return

    resolved = expand_path(edf_env, get_project_root(cfg))
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"EDF environment file not found: {resolved}")


def add_file_logging(log_file: str, level: str) -> None:
    """Add a file handler so logs are also written to disk."""
    resolved_log_file = Path(expand_path(log_file))
    try:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Unable to create log directory for %s: %s", resolved_log_file, exc
        )
        return

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved_log_file
        ):
            return

    try:
        file_handler = logging.FileHandler(
            resolved_log_file, mode="a", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Unable to open log file %s: %s", resolved_log_file, exc)
        return

    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)


def setup_runtime(cfg: MMirageConfig, log_level: str) -> None:
    """Initialize runtime-level logging."""
    report_dir = Path(
        expand_path(cfg.execution_params.report_dir, get_project_root(cfg))
    )
    log_file = report_dir / f"{cfg.execution_params.job_name}.out"
    add_file_logging(str(log_file), log_level)
    logger.info("Writing logs to %s", log_file)
