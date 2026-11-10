"""SLURM helpers for the MMIRAGE CLI."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from typing import List, Optional

from mmirage.cli_utils.runtime import create_directories, expand_path, get_project_root
from mmirage.config.config import MMirageConfig
from mmirage.core.process.processors.image_gen.sglang_server import (
    get_sglang_server_config,
)

logger = logging.getLogger(__name__)


def _bash_double_quote(value: str) -> str:
    """Return a double-quoted bash string literal.

    We intentionally do NOT escape '$' so that $VARS from config can expand on
    compute nodes (e.g. $SCRATCH). This matches typical SLURM job scripts.

    To avoid command injection, we reject values containing shell command
    substitution syntax such as ``$(...)`` or backticks. Variable expansion
    using ``$VAR`` or ``${VAR}`` is still allowed.
    """
    # Disallow command substitution while still allowing $VAR expansion.
    if "`" in value or "$(" in value:
        raise ValueError(
            "Config value contains unsupported shell command substitution "
            "(` or '$('). Command substitution is not allowed in SLURM-generated scripts."
        )
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _shell_path(value: str, project_root: str) -> str:
    """Expand user home and make relative paths project-rooted.

    If the path starts with '$' we assume it will expand on the compute node and
    therefore do not attempt to join it with project_root.
    """
    raw = value.strip()
    if not raw:
        return raw

    raw = os.path.expanduser(raw)
    if raw.startswith("$"):
        return raw
    if not os.path.isabs(raw):
        raw = os.path.join(project_root, raw)
    return raw


def build_sbatch_script(
    cfg: MMirageConfig,
    config_path: str,
    collect_stats: bool = False,
    shard_ids: Optional[List[int]] = None,
) -> str:
    """Build the sbatch payload for shard processing."""
    project_root = get_project_root(cfg)
    hf_home = _shell_path(cfg.execution_params.hf_home, project_root)
    state_root = _shell_path(cfg.loading_params.get_state_root(), project_root)
    src_root = os.path.join(project_root, "src")
    shard_process_path = os.path.join(src_root, "mmirage", "shard_process.py")

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"export PYTHONPATH={_bash_double_quote(src_root)}:${{PYTHONPATH:-}}",
        f"export SHARD_PROCESS={_bash_double_quote(shard_process_path)}",
        f"export HF_HOME={_bash_double_quote(hf_home)}",
        f"export MMIRAGE_CONFIG={_bash_double_quote(config_path)}",
    ]
    if collect_stats:
        lines.append("export MMIRAGE_COLLECT_STATS=1")
    sglang = get_sglang_server_config(cfg)
    if sglang is not None:
        lines.append(
            f"export MMIRAGE_SHARD_IDS={_bash_double_quote(','.join(map(str, shard_ids or [])))}"
        )
    lines.extend(
        [
            f"mkdir -p {_bash_double_quote(hf_home)}",
            f"mkdir -p {_bash_double_quote(state_root)}",
            "srun_args=(--cpus-per-task ${SLURM_CPUS_PER_TASK:-1} --wait 60)",
        ]
    )

    if cfg.execution_params.edf_env:
        edf_env = expand_path(cfg.execution_params.edf_env, project_root)
        lines.append(f"srun_args+=(--environment={shlex.quote(edf_env)})")

    account = cfg.execution_params.account
    if not account:
        raise ValueError("execution_params.account must be set in slurm mode")
    lines.append(f"srun_args+=(-A {shlex.quote(account)})")

    if cfg.execution_params.reservation:
        lines.append(
            f"srun_args+=(--reservation={shlex.quote(cfg.execution_params.reservation)})"
        )

    python_setup = (
        "if command -v python3 >/dev/null 2>&1; then PYTHON_CMD=python3; "
        "elif command -v python >/dev/null 2>&1; then PYTHON_CMD=python; "
        'else echo "python3/python not found in PATH" >&2; exit 127; fi; '
        'echo "Using Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1))"; '
        '${PYTHON_CMD} -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 2)" '
        '|| { echo "MMIRAGE requires Python >= 3.10 on compute nodes" >&2; exit 2; }; '
    )
    if sglang is not None:
        lines.append(
            'srun "${srun_args[@]}" bash -c \''
            + python_setup
            + 'exec ${PYTHON_CMD} -m mmirage.sglang_job --config "$MMIRAGE_CONFIG" '
            '--shard-ids "$MMIRAGE_SHARD_IDS"\''
        )
        lines.append('echo "Shared SGLang MMIRAGE job completed"')
    else:
        lines.append(
            'srun "${srun_args[@]}" bash -c \''
            + python_setup
            + 'exec ${PYTHON_CMD} "$SHARD_PROCESS" --config "$MMIRAGE_CONFIG"\''
        )
        lines.append('echo "Shard ${SLURM_ARRAY_TASK_ID:-0} completed"')
    return "\n".join(lines) + "\n"


def submit_slurm_job(
    cfg: MMirageConfig,
    config_path: str,
    shard_ids: Optional[List[int]] = None,
    collect_stats: bool = False,
) -> Optional[int]:
    """Submit a SLURM shard-processing job and return its job ID."""
    project_root = get_project_root(cfg)
    report_dir = expand_path(cfg.execution_params.report_dir, project_root)
    create_directories([report_dir])
    sglang = get_sglang_server_config(cfg)
    output_name = "R-%x.%j.out" if sglang is not None else "R-%x.%A_%a.out"
    error_name = "R-%x.%j.err" if sglang is not None else "R-%x.%A_%a.err"

    command = [
        "sbatch",
        "--parsable",
        f"--job-name={cfg.execution_params.job_name}",
        f"--chdir={project_root}",
        f"--output={os.path.join(report_dir, output_name)}",
        f"--error={os.path.join(report_dir, error_name)}",
        f"--nodes={cfg.execution_params.nodes}",
        f"--ntasks-per-node={cfg.execution_params.ntasks_per_node}",
        f"--gres=gpu:{cfg.execution_params.gpus}",
        f"--cpus-per-task={cfg.execution_params.cpus_per_task}",
        f"--time={cfg.execution_params.time_limit}",
        f"--account={cfg.execution_params.account}",
    ]

    if cfg.execution_params.reservation:
        command.append(f"--reservation={cfg.execution_params.reservation}")

    requested_shards = list(shard_ids or [])
    if sglang is not None:
        if cfg.execution_params.nodes != 1:
            raise ValueError(
                "backend='sglang' currently supports only execution_params.nodes=1"
            )
    elif requested_shards:
        command.append(
            f"--array={','.join(str(shard_id) for shard_id in requested_shards)}"
        )
    else:
        num_shards = cfg.loading_params.get_num_shards()
        last_shard_id = num_shards - 1
        command.append(f"--array=0-{last_shard_id}")

    logger.info("Submitting SLURM job: %s", " ".join(command))
    result = subprocess.run(
        command,
        input=build_sbatch_script(
            cfg,
            config_path,
            collect_stats=collect_stats,
            shard_ids=requested_shards,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("sbatch failed: %s", result.stderr.strip())
        return None

    raw_job_id = result.stdout.strip().split(";", 1)[0]
    try:
        return int(raw_job_id)
    except ValueError:
        logger.error(
            "Unable to parse job id from sbatch output: %s", result.stdout.strip()
        )
        return None


def slurm_job_gone(job_id: int | str) -> bool:
    """Return True only when squeue confirms the job is no longer queued.

    Any other squeue failure counts as "still queued".
    """
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return not result.stdout.strip()
    # Once Slurm purges a finished job (after MinJobAge), squeue rejects its id.
    return "Invalid job id specified" in result.stderr


def wait_for_slurm_job(job_id: int, cfg: MMirageConfig) -> None:
    """Wait for a SLURM job array to leave the queue."""
    logger.info("Waiting for SLURM job %s", job_id)
    while not slurm_job_gone(job_id):
        time.sleep(cfg.execution_params.poll_interval_seconds)

    if cfg.execution_params.settle_time_seconds > 0:
        logger.info(
            "Waiting %ss for state files to settle",
            cfg.execution_params.settle_time_seconds,
        )
        time.sleep(cfg.execution_params.settle_time_seconds)


def require_slurm(cfg: MMirageConfig, command_name: str) -> int:
    """Ensure command can only run in SLURM mode."""
    if cfg.execution_params.is_slurm():
        return 0
    logger.error("%s requires execution_params.mode=slurm", command_name)
    return 1
