"""Shard status and retry helpers for the MMIRAGE CLI."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from mmirage.cli_utils.slurm import submit_slurm_job
from mmirage.config.config import MMirageConfig
from mmirage.shard_utils import (
    ShardStatus,
    format_duration,
    read_status,
    shard_state_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class ShardSummary:
    """Compact status summary for shard execution."""

    total: int
    successful: int
    running: int
    failed: int
    max_retries_exceeded: int


def max_allowed_attempts(max_retries: int) -> int:
    """Return max allowed total attempts for a shard.

    Total attempts = initial attempt + max_retries.
    """
    return max_retries + 1


def is_retry_budget_exceeded(attempt_count: int, max_retries: int) -> bool:
    """Return whether a shard has exceeded the retry budget."""
    return attempt_count > max_allowed_attempts(max_retries)


def get_shard_status(state_dir: str) -> Tuple[str, int]:
    """Read the current status and attempt counter for a shard."""
    status_file = os.path.join(state_dir, "status.json")
    if not os.path.exists(status_file):
        return ("missing", 0)

    try:
        with open(status_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, dict):
                logger.warning(
                    "Invalid shard status format in %s; expected object", status_file
                )
                return ("unknown", 0)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read shard status from %s: %s", status_file, exc)
        return ("unknown", 0)

    parsed = ShardStatus.from_dict(data)
    return (parsed.status, parsed.retry_count)


def check_failed_shards(cfg: MMirageConfig) -> Tuple[List[int], ShardSummary]:
    """Return retryable failed shards and a compact summary."""
    state_root = cfg.loading_params.get_state_root()

    num_shards = cfg.loading_params.get_num_shards()
    max_retries = cfg.execution_params.max_retries
    failed_shards: List[int] = []
    success_count = 0
    running_count = 0
    exhausted_count = 0
    allowed_attempts = max_allowed_attempts(max_retries)

    for shard_id in range(num_shards):
        status, attempt_count = get_shard_status(shard_state_dir(shard_id, state_root))
        if status == "success":
            success_count += 1
        elif status == "running":
            running_count += 1
        elif is_retry_budget_exceeded(attempt_count, max_retries):
            exhausted_count += 1
            logger.warning(
                "Shard %s exceeded retry budget (attempts=%s, max_allowed_attempts=%s)",
                shard_id,
                attempt_count,
                allowed_attempts,
            )
        else:
            failed_shards.append(shard_id)

    summary = ShardSummary(
        total=num_shards,
        successful=success_count,
        running=running_count,
        failed=len(failed_shards),
        max_retries_exceeded=exhausted_count,
    )
    return failed_shards, summary


def confirm_retry(count: int, confirm_mode: Literal["prompt", "yes"]) -> bool:
    """Return whether retry submission is confirmed.

    Modes:
    - prompt: ask the user interactively
    - yes: submit without prompting
    """
    if confirm_mode == "yes":
        return True

    if not sys.stdin.isatty():
        logger.error(
            "Interactive confirmation requested but stdin is not a TTY; use --yes"
        )
        return False

    response = input(f"Retry {count} shard(s)? (y/N) ")
    return response.strip().lower() == "y"


def status_exit_code(failed_shards: Sequence[int], summary: ShardSummary) -> int:
    """Map shard status to an exit code."""
    return (
        0
        if not failed_shards
        and summary.max_retries_exceeded == 0
        and summary.running == 0
        and summary.successful == summary.total
        else 1
    )


def submit_failed_shards(
    cfg: MMirageConfig,
    config_path: str,
    failed_shards: Sequence[int],
    confirm_mode: Literal["prompt", "yes"],
    collect_stats: bool = False,
) -> int:
    """Submit retry jobs for failed shards when requested."""
    if not failed_shards:
        return 0

    if not confirm_retry(len(failed_shards), confirm_mode):
        return 1

    job_id = submit_slurm_job(
        cfg, config_path, failed_shards, collect_stats=collect_stats
    )
    if job_id is None:
        return 1

    return 0


def collect_bench_stats(cfg: MMirageConfig) -> Dict[str, Any]:
    """Collect per-shard benchmark statistics and compute aggregate totals.

    Returns a dict with two keys:

    - ``per_shard``: list of dicts, one per shard, each containing the full
      :class:`~mmirage.shard_utils.ShardStatus` payload plus a flattened
      ``stats`` sub-dict.
    - ``aggregate``: rolled-up totals across all completed shards.

    Shards without ``stats`` (e.g. still running or from older runs) are
    included in ``per_shard`` but excluded from aggregate calculations.
    """
    state_root = cfg.loading_params.get_state_root()
    num_shards = cfg.loading_params.get_num_shards()

    per_shard: List[Dict[str, Any]] = []

    total_rows: int = 0
    # Output-side row counts; None until at least one shard reports the value.
    total_rows_written: Optional[int] = None
    total_rows_filtered: Optional[int] = None
    total_rows_dropped_by_error: Optional[int] = None
    sum_runtime: float = 0.0
    runtimes: List[float] = []
    gpu_util_weighted: List[float] = []  # util * rows for weighted mean
    gpu_total_rows_for_weight: int = 0
    earliest_start: Optional[str] = None
    latest_finish: Optional[str] = None
    # Token-level aggregates (DataTrove-compatible benchmark format).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    has_token_data: bool = False
    sum_model_load_seconds: float = 0.0
    num_gpus: Optional[int] = None  # taken from first shard that has it

    for shard_id in range(num_shards):
        state_dir = shard_state_dir(shard_id, state_root)
        status = read_status(state_dir)
        entry: Dict[str, Any] = status.to_dict()
        per_shard.append(entry)

        if status.status != "success" or status.stats is None:
            continue

        s = status.stats
        if s.runtime_seconds is not None:
            sum_runtime += s.runtime_seconds
            runtimes.append(s.runtime_seconds)
        if s.rows_processed is not None:
            total_rows += s.rows_processed
        if s.rows_written is not None:
            total_rows_written = (total_rows_written or 0) + s.rows_written
        if s.rows_filtered is not None:
            total_rows_filtered = (total_rows_filtered or 0) + s.rows_filtered
        if s.rows_dropped_by_error is not None:
            total_rows_dropped_by_error = (
                total_rows_dropped_by_error or 0
            ) + s.rows_dropped_by_error
        if s.gpu_util_mean is not None and s.rows_processed:
            gpu_util_weighted.append(s.gpu_util_mean * s.rows_processed)
            gpu_total_rows_for_weight += s.rows_processed

        # Accumulate token counts.
        if s.input_tokens is not None:
            total_input_tokens += s.input_tokens
            has_token_data = True
        if s.output_tokens is not None:
            total_output_tokens += s.output_tokens
            has_token_data = True
        if s.model_load_seconds is not None:
            sum_model_load_seconds += s.model_load_seconds
        if num_gpus is None and s.num_gpus is not None:
            num_gpus = s.num_gpus

        # Track earliest start / latest finish for wall-clock runtime.
        if status.started_at:
            if earliest_start is None or status.started_at < earliest_start:
                earliest_start = status.started_at
        if status.finished_at:
            if latest_finish is None or status.finished_at > latest_finish:
                latest_finish = status.finished_at

    # Wall-clock runtime: time from first shard start to last shard finish.
    wall_clock: Optional[float] = None
    if earliest_start and latest_finish:
        try:
            from datetime import datetime as _dt

            wall_clock = round(
                (
                    _dt.fromisoformat(latest_finish) - _dt.fromisoformat(earliest_start)
                ).total_seconds(),
                3,
            )
        except (ValueError, TypeError):
            pass

    overall_throughput: Optional[float] = None
    if total_rows > 0 and wall_clock and wall_clock > 0:
        overall_throughput = round(total_rows / wall_clock, 2)

    mean_gpu_util: Optional[float] = None
    if gpu_util_weighted and gpu_total_rows_for_weight > 0:
        mean_gpu_util = round(sum(gpu_util_weighted) / gpu_total_rows_for_weight, 1)

    # Aggregate token-throughput metrics (DataTrove-compatible benchmark format).
    # Uses sum of inference runtimes (total minus model loading) for a per-GPU token rate
    # that excludes one-time model initialisation overhead.
    agg_tokens_per_sec_per_gpu: Optional[float] = None
    agg_gpu_days_per_billion_tokens: Optional[float] = None
    agg_inference_runtime: Optional[float] = None
    if (
        has_token_data
        and total_output_tokens > 0
        and runtimes
        and num_gpus
        and num_gpus > 0
    ):
        agg_inference_runtime = max(0.0, sum_runtime - sum_model_load_seconds)
        if agg_inference_runtime > 0:
            total_gpu_seconds = agg_inference_runtime * num_gpus
            agg_tokens_per_sec_per_gpu = round(
                total_output_tokens / total_gpu_seconds, 2
            )
            total_gpu_days = total_gpu_seconds / 86_400
            agg_gpu_days_per_billion_tokens = round(
                total_gpu_days / (total_output_tokens / 1e9), 4
            )

    aggregate: Dict[str, Any] = {
        "total_shards": num_shards,
        "completed_shards": sum(1 for e in per_shard if e.get("status") == "success"),
        "total_rows_processed": total_rows if total_rows > 0 else None,
        "total_rows_written": total_rows_written,
        "total_rows_filtered": total_rows_filtered,
        "total_rows_dropped_by_error": total_rows_dropped_by_error,
        "wall_clock_runtime_seconds": wall_clock,
        "wall_clock_runtime_human": format_duration(wall_clock),
        "sum_shard_runtime_seconds": round(sum_runtime, 3) if runtimes else None,
        "sum_shard_runtime_human": format_duration(
            round(sum_runtime, 3) if runtimes else None
        ),
        "min_shard_runtime_seconds": round(min(runtimes), 3) if runtimes else None,
        "min_shard_runtime_human": format_duration(
            round(min(runtimes), 3) if runtimes else None
        ),
        "max_shard_runtime_seconds": round(max(runtimes), 3) if runtimes else None,
        "max_shard_runtime_human": format_duration(
            round(max(runtimes), 3) if runtimes else None
        ),
        "overall_throughput_rows_per_sec": overall_throughput,
        "mean_gpu_util_pct": mean_gpu_util,
        # Token-level benchmark metrics (DataTrove-compatible).
        "num_gpus": num_gpus,
        "total_input_tokens": total_input_tokens if has_token_data else None,
        "total_output_tokens": total_output_tokens if has_token_data else None,
        "sum_model_load_seconds": round(sum_model_load_seconds, 3)
        if sum_model_load_seconds > 0
        else None,
        "sum_inference_runtime_seconds": round(agg_inference_runtime, 3)
        if agg_inference_runtime is not None
        else None,
        "tokens_per_sec_per_gpu": agg_tokens_per_sec_per_gpu,
        "gpu_days_per_billion_tokens": agg_gpu_days_per_billion_tokens,
    }

    return {"per_shard": per_shard, "aggregate": aggregate}
