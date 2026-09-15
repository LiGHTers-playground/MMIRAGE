"""Command-line interface for MMIRAGE pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from typing import List, Optional

from mmirage.cli_utils.runtime import (
    non_empty_path,
    setup_runtime,
    validate_edf_env_path,
)
from mmirage.cli_utils.slurm import require_slurm, submit_slurm_job, wait_for_slurm_job
from mmirage.cli_utils.status import (
    check_failed_shards,
    collect_bench_stats,
    get_shard_status,
    is_retry_budget_exceeded,
    shard_state_dir,
    status_exit_code,
    submit_failed_shards,
)
from mmirage.config.config import MMirageConfig
from mmirage.config.utils import load_mmirage_config
from mmirage.core.process.batch.collector import collect_batches
from mmirage.core.process.batch.status_checker import check_batches
from mmirage.core.process.processors.batch_api.config import BATCH_API_PROCESSOR_TYPE
from mmirage.core.process.processors.image_gen.sglang_server import (
    MMIRAGE_SGLANG_BASE_URL,
    get_sglang_server_config,
    shared_sglang_server,
)
from mmirage.merge_shards import MergeReport, merge_from_config, merge_input_dir

logger = logging.getLogger(__name__)


def run_local(
    config_path: str,
    shard_id: Optional[int] = None,
    collect_stats: bool = False,
    export_prompts_path: Optional[str] = None,
) -> int:
    """Run one shard in the current Python environment.

    Args:
        config_path: Absolute path to the MMIRAGE YAML config file.
        shard_id: Optional shard id to inject via SLURM_ARRAY_TASK_ID.
        collect_stats: If True, enable GPU utilization polling in the shard process.
        export_prompts_path: Optional file path for exporting batch prompts instead of submitting them.
    Returns:
        Process return code from shard execution.
    """
    cfg = load_mmirage_config(config_path)
    sglang = get_sglang_server_config(cfg)
    if sglang is not None and not os.environ.get(MMIRAGE_SGLANG_BASE_URL):
        with shared_sglang_server(sglang):
            return run_local(config_path, shard_id, collect_stats, export_prompts_path)

    command = [sys.executable, "-m", "mmirage.shard_process", "--config", config_path]
    if export_prompts_path is not None:
        command.extend(["--export-prompts", export_prompts_path])
    env = os.environ.copy()
    if shard_id is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(shard_id)
    if collect_stats:
        env["MMIRAGE_COLLECT_STATS"] = "1"

    logger.info("Running local shard processing: %s", " ".join(command))
    result = subprocess.run(command, env=env, check=False)
    return result.returncode


def launch_pipeline(
    cfg: MMirageConfig,
    config_path: str,
    force_retry: bool = False,
    require_completion: bool = False,
    collect_stats: bool = False,
    export_prompts_path: Optional[str] = None,
) -> int:
    """Launch the pipeline according to execution mode and retry settings.

    Args:
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.
        force_retry: If True, enable retry orchestration regardless of config flag.
        require_completion: If True, wait for completion and verify shard
            status before returning success in SLURM mode when auto-retry is off.
        collect_stats: If True, enable GPU utilization polling on compute nodes.
        export_prompts_path: Optional file path for exporting batch prompts instead of submitting them.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    auto_retry = force_retry or cfg.execution_params.retry

    if export_prompts_path is not None and cfg.execution_params.is_slurm():
        logger.error(
            "--export-prompts is not supported in slurm mode. Run it in local mode."
        )
        return 1

    if not cfg.execution_params.is_slurm():
        sglang = get_sglang_server_config(cfg)
        if sglang is not None and not os.environ.get(MMIRAGE_SGLANG_BASE_URL):
            with shared_sglang_server(sglang):
                return launch_pipeline(
                    cfg,
                    config_path,
                    force_retry=force_retry,
                    require_completion=require_completion,
                    collect_stats=collect_stats,
                    export_prompts_path=export_prompts_path,
                )

        initial_shard_id = cfg.loading_params.get_shard_id()
        if not auto_retry:
            exit_code = run_local(
                config_path,
                initial_shard_id,
                collect_stats=collect_stats,
                export_prompts_path=export_prompts_path,
            )
            if exit_code == 0:
                logger.info("All shards completed successfully")
            return exit_code

        shard_ids: List[int] = [initial_shard_id]
        attempts_by_shard = {initial_shard_id: 0}
        state_root = cfg.loading_params.get_state_root()
        while True:
            run_exit_codes = {}
            for shard_id in shard_ids:
                attempts_by_shard[shard_id] = attempts_by_shard.get(shard_id, 0) + 1
                run_exit_codes[shard_id] = run_local(
                    config_path,
                    shard_id,
                    collect_stats=collect_stats,
                    export_prompts_path=export_prompts_path,
                )

            failed_shards, summary = check_failed_shards(cfg)
            if status_exit_code(failed_shards, summary) == 0:
                logger.info("All shards completed successfully")
                return 0

            runtime_failed = [
                shard_id for shard_id, rc in run_exit_codes.items() if rc != 0
            ]
            candidates = sorted(set(failed_shards) | set(runtime_failed))
            retryable_shards: List[int] = []
            for shard_id in candidates:
                _, state_attempt_count = get_shard_status(
                    shard_state_dir(shard_id, state_root)
                )
                memory_attempt_count = attempts_by_shard.get(shard_id, 0)
                effective_attempt_count = max(state_attempt_count, memory_attempt_count)

                if not is_retry_budget_exceeded(
                    effective_attempt_count,
                    cfg.execution_params.max_retries,
                ):
                    retryable_shards.append(shard_id)

            if not retryable_shards:
                logger.error("Pipeline ended with shards that exceeded max retries")
                return 1

            logger.warning(
                "Retrying failed shards locally: %s",
                ",".join(map(str, retryable_shards)),
            )
            shard_ids = retryable_shards

    shard_ids: List[int] = []

    while True:
        job_id = submit_slurm_job(
            cfg, config_path, shard_ids, collect_stats=collect_stats
        )
        if job_id is None:
            return 1

        logger.info(f"Submitted SLURM job {job_id} for shard ids: {shard_ids or 'ALL'}")

        if not auto_retry:
            if not require_completion:
                return 0

            wait_for_slurm_job(job_id, cfg)
            failed_shards, summary = check_failed_shards(cfg)
            status_code = status_exit_code(failed_shards, summary)
            if status_code == 0:
                logger.info("All shards completed successfully")
            else:
                logger.error(
                    "SLURM run completed with failed shards; merge will not start"
                )
            return status_code

        wait_for_slurm_job(job_id, cfg)
        failed_shards, summary = check_failed_shards(cfg)

        if status_exit_code(failed_shards, summary) == 0:
            logger.info("All shards completed successfully")
            return 0

        if not failed_shards:
            logger.error("Pipeline ended with shards that exceeded max retries")
            return 1

        logger.warning("Retrying failed shards: %s", ",".join(map(str, failed_shards)))
        shard_ids = failed_shards


def configure_logging(level: str) -> None:
    """Configure root logging.

    Args:
        level: Root log level name.
    """
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach common CLI arguments to a subcommand parser.

    Args:
        parser: Subcommand parser receiving shared arguments.
    """
    parser.add_argument(
        "--config", required=True, help="Path to a MMIRAGE YAML config file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )


def add_metadata_path_argument(parser: argparse.ArgumentParser) -> None:
    """Attach the batch receipt path argument to a subcommand parser.

    Args:
        parser: Subcommand parser receiving the argument.
    """
    parser.add_argument(
        "--metadata-path",
        nargs="+",
        help=(
            "batch_api configs only: path(s) to metadata JSONL receipt file(s). "
            "When omitted, receipts are resolved from the metadata_output_path of "
            "each batch_api processor block"
        ),
    )


def uses_batch_api(cfg: MMirageConfig) -> bool:
    """Return True when the config declares at least one batch_api processor.

    Args:
        cfg: Parsed MMIRAGE configuration object.
    """
    return any(
        processor.type == BATCH_API_PROCESSOR_TYPE for processor in cfg.processors
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured top-level argparse parser.
    """
    parser = argparse.ArgumentParser(description="MMIRAGE command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit one SLURM array job")
    add_shared_arguments(submit_parser)
    submit_parser.add_argument(
        "--shard-ids",
        help="Comma-separated shard ids to submit instead of the full array",
    )
    submit_parser.add_argument(
        "--wait", action="store_true", help="Wait for the submitted job"
    )
    submit_parser.add_argument(
        "--stats",
        action="store_true",
        help="Enable GPU utilization and throughput collection on compute nodes",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Inspect shard status, or provider batch status for batch_api configs",
    )
    add_shared_arguments(check_parser)
    add_metadata_path_argument(check_parser)
    check_parser.add_argument(
        "--retry",
        dest="retry",
        action="store_true",
        help="Submit a retry job for failed shards.",
    )
    check_parser.set_defaults(retry=False)
    check_parser.add_argument(
        "-y",
        "--yes",
        dest="confirm_mode",
        action="store_const",
        const="yes",
        help="Submit retries without prompting.",
    )
    check_parser.set_defaults(confirm_mode="prompt")
    check_parser.add_argument(
        "--stats",
        action="store_true",
        help="Enable GPU utilization and throughput collection on retried compute nodes",
    )

    retry_parser = subparsers.add_parser("retry", help="Submit only failed shards")
    add_shared_arguments(retry_parser)
    retry_parser.add_argument(
        "-y",
        "--yes",
        dest="confirm_mode",
        action="store_const",
        const="yes",
        help="Submit retries without prompting.",
    )
    retry_parser.set_defaults(confirm_mode="prompt")
    retry_parser.add_argument(
        "--stats",
        action="store_true",
        help="Enable GPU utilization and throughput collection on retried compute nodes",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run according to execution_params.mode and execution_params.retry",
    )
    add_shared_arguments(run_parser)
    run_parser.add_argument(
        "--force-retry",
        action="store_true",
        help="Enable retry orchestration even if execution_params.retry is false",
    )
    run_parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Run a single shard locally (overrides execution mode)",
    )
    run_parser.add_argument(
        "--stats",
        action="store_true",
        help="Enable GPU utilization and throughput collection during shard execution",
    )
    run_parser.add_argument(
        "--export-prompts",
        type=non_empty_path,
        help="Directory or .jsonl path for exporting batch prompts instead of submitting them",
        default=None,
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help=(
            "Merge shard outputs listed in config.loading_params.datasets, or "
            "retrieve and merge completed provider batches for batch_api configs"
        ),
    )
    add_shared_arguments(merge_parser)
    merge_parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_dir",
        default=None,
        help=(
            "Optional root directory for merged outputs. MMIRAGE creates one "
            "subdirectory per configured dataset under this root. If omitted, "
            "each dataset is merged into <dataset.output_dir>/merged"
        ),
    )
    merge_parser.add_argument(
        "--output-path",
        default=None,
        help="batch_api configs only: JSONL file path for the merged batch results (required)",
    )
    add_metadata_path_argument(merge_parser)

    merge_dir_parser = subparsers.add_parser(
        "merge-dir",
        help="Merge shards directly from an input directory into an output directory",
    )
    merge_dir_parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Input directory containing one dataset with shard_* folders, or "
            "multiple dataset subdirectories each containing shard_* folders"
        ),
    )
    merge_dir_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for merged dataset(s)",
    )
    merge_dir_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )

    stats_parser = subparsers.add_parser(
        "stats",
        help="Show per-shard benchmark statistics (runtime, throughput, GPU utilization)",
    )
    add_shared_arguments(stats_parser)

    return parser


def log_merge_reports(reports: List[MergeReport]) -> None:
    """Log merge summary for one or more datasets."""
    for report in reports:
        skipped_total = report.skipped_invalid_dirs + report.skipped_zero_rows
        logger.info(
            "Merged dataset %s: shards=%d rows=%d output=%s skipped=%d "
            "(invalid=%d, zero_rows=%d)",
            report.dataset_name,
            report.used_shards,
            report.merged_rows,
            report.output_dir,
            skipped_total,
            report.skipped_invalid_dirs,
            report.skipped_zero_rows,
        )


def parse_shard_ids(
    raw_value: Optional[str], num_shards: Optional[int] = None
) -> List[int]:
    """Parse a comma-separated shard id list.

    Args:
        raw_value: Comma-separated shard ids, or None/empty for full array.
        num_shards: Optional upper bound used for range validation.

    Returns:
        Parsed shard ids.
    """
    if not raw_value:
        return []

    shard_ids: List[int] = []
    for raw_shard_id in raw_value.split(","):
        candidate = raw_shard_id.strip()
        if not candidate:
            continue

        if candidate.isdigit():
            shard_id = int(candidate)
        else:
            raise ValueError(f"Invalid shard id {candidate!r}; expected integers")

        if num_shards is not None and shard_id >= num_shards:
            raise ValueError(
                f"Invalid shard id {shard_id}; expected 0 <= shard_id < {num_shards}"
            )

        shard_ids.append(shard_id)

    return shard_ids


def handle_run(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Handle the canonical run command.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code for the run operation.
    """
    if args.shard_id is not None:
        return run_local(
            config_path,
            args.shard_id,
            collect_stats=args.stats,
            export_prompts_path=args.export_prompts,
        )

    exit_code = launch_pipeline(
        cfg,
        config_path,
        force_retry=args.force_retry,
        require_completion=cfg.execution_params.merge,
        collect_stats=args.stats,
        export_prompts_path=args.export_prompts,
    )
    if exit_code != 0:
        return exit_code

    if cfg.execution_params.merge:
        logger.info("Execution_params.merge is true; merging shard outputs")
        reports = merge_from_config(cfg)
        log_merge_reports(reports)

    return 0


def handle_submit(
    args: argparse.Namespace, cfg: MMirageConfig, config_path: str
) -> int:
    """Submit a SLURM array job and optionally wait.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code for submission/wait outcome.
    """
    if require_slurm(cfg, "submit") != 0:
        return 1

    shard_ids = parse_shard_ids(args.shard_ids, cfg.loading_params.get_num_shards())
    job_id = submit_slurm_job(cfg, config_path, shard_ids, collect_stats=args.stats)
    if job_id is None:
        return 1

    logger.info(f"Submitted SLURM job {job_id} for shard ids: {shard_ids or 'ALL'}")

    if not args.wait:
        return 0

    wait_for_slurm_job(job_id, cfg)
    failed_shards, summary = check_failed_shards(cfg)
    status_code = status_exit_code(failed_shards, summary)
    if status_code == 0:
        logger.info("All shards completed successfully")
    return status_code


def handle_check(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Inspect shard status and optionally submit retries.

    For batch_api configs, shard state only records submission, so the provider
    batch status is reported from the metadata receipts instead.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code based on shard status and optional retry submission.
    """
    if uses_batch_api(cfg):
        if args.retry or args.stats:
            logger.warning("--retry and --stats are ignored for batch_api configs.")
        return check_batches(cfg, args.metadata_path)

    failed_shards, summary = check_failed_shards(cfg)
    print(json.dumps(asdict(summary), indent=2))

    status_code = status_exit_code(failed_shards, summary)
    if not cfg.execution_params.is_slurm():
        return status_code

    if not args.retry:
        return status_code

    if not failed_shards:
        return status_code

    return submit_failed_shards(
        cfg=cfg,
        config_path=config_path,
        failed_shards=failed_shards,
        confirm_mode=args.confirm_mode,
        collect_stats=args.stats,
    )


def handle_retry(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Submit retries for failed shards only.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code for retry submission outcome.
    """
    if require_slurm(cfg, "retry") != 0:
        return 1

    failed_shards, summary = check_failed_shards(cfg)
    print(json.dumps(asdict(summary), indent=2))

    if not failed_shards:
        if summary.max_retries_exceeded > 0:
            logger.error("No retryable shards remain")
            return 1
        logger.info("All shards already succeeded.")
        return 0

    return submit_failed_shards(
        cfg=cfg,
        config_path=config_path,
        failed_shards=failed_shards,
        confirm_mode=args.confirm_mode,
        collect_stats=args.stats,
    )


def handle_stats(
    args: argparse.Namespace, cfg: MMirageConfig, _config_path: str
) -> int:
    """Print per-shard benchmark statistics and aggregate totals.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        _config_path: Absolute path to the MMIRAGE YAML config file (not needed here).

    Returns:
        Exit code: 0 always (stats are informational).
    """
    report = collect_bench_stats(cfg)
    print(json.dumps(report, indent=2))
    return 0


def handle_merge(
    args: argparse.Namespace, cfg: MMirageConfig, _config_path: str
) -> int:
    """Merge shard outputs defined in config.loading_params.datasets.

    For batch_api configs, the shards only hold submission placeholders, so the
    completed provider results are retrieved and merged into --output-path.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        _config_path: Absolute path to the MMIRAGE YAML config file (not needed here).

    Returns:
        Exit code for merge outcome.
    """
    if uses_batch_api(cfg):
        if not args.output_path:
            logger.error("--output-path is required to merge batch_api results")
            return 1
        return collect_batches(cfg, args.output_path, args.metadata_path)

    reports = merge_from_config(cfg, output_root=args.output_dir)
    log_merge_reports(reports)
    return 0


def handle_merge_dir(args: argparse.Namespace) -> int:
    """Merge shard outputs directly from input/output directory arguments.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Exit code for merge outcome.
    """
    reports = merge_input_dir(args.input_dir, args.output_dir)
    log_merge_reports(reports)
    return 0


def main() -> None:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        if args.command == "merge-dir":
            sys.exit(handle_merge_dir(args))

        config_path = os.path.abspath(args.config)
        cfg = load_mmirage_config(config_path)

        setup_runtime(cfg, args.log_level)
        validate_edf_env_path(cfg)

        handlers = {
            "run": handle_run,
            "submit": handle_submit,
            "check": handle_check,
            "retry": handle_retry,
            "merge": handle_merge,
            "stats": handle_stats,
        }
        handler = handlers.get(args.command)
        if handler is None:
            logger.error("Unknown command: %s", args.command)
            sys.exit(2)

        sys.exit(handler(args, cfg, config_path))

    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        sys.exit(1)


if __name__ == "__main__":
    main()
