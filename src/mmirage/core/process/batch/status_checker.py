"""Receiver-side helper to check provider batch status from metadata receipts.

Designed for CLI use against JSONL receipt files. Skips malformed lines and
missing keys to keep status checks resilient to partial metadata corruption.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult
from mmirage.core.process.batch.metadata_paths import resolve_metadata_paths
from mmirage.core.process.batch.metadata_utils import (
    BatchMetadataRecord,
    _read_metadata_records,
)
from mmirage.core.process.batch.provider_resolution import resolve_provider_configs

if TYPE_CHECKING:
    from mmirage.config.config import MMirageConfig

from mmirage.core.process.batch.registry import BatchAdapterRegistry

logger = logging.getLogger(__name__)


def extract_unique_provider_batches(
    metadata_records: Sequence[BatchMetadataRecord],
) -> List[Tuple[str, str]]:
    """Return unique ``(provider, provider_batch_id)`` pairs.

    Provider names are already lowercased and records missing either key are
    already dropped by ``_read_metadata_records``, so both are assumed here.
    """
    unique_pairs: List[Tuple[str, str]] = []
    seen = set()

    for record in metadata_records:
        provider = record.provider
        provider_batch_id = record.provider_batch_id

        pair = (provider, provider_batch_id)
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    return unique_pairs


def run_status_checker(
    metadata_records: Sequence[BatchMetadataRecord],
    provider_configs: Mapping[str, BatchProviderConfig],
) -> List[BatchSubmissionResult]:
    """Check batch status for each referenced provider batch.

    Prints a per-batch line and a per-provider summary. A batch the provider
    cannot resolve is counted as ``lookup_failed`` so one stale receipt does
    not hide the status of every batch after it.
    """
    results: List[BatchSubmissionResult] = []
    counter: Dict[str, Dict[str, int]] = {}

    for provider, provider_batch_id in extract_unique_provider_batches(
        metadata_records
    ):
        config = provider_configs[provider]
        adapter = BatchAdapterRegistry.create(config)
        provider_counts = counter.setdefault(provider, {})

        try:
            result = adapter.check_batch_status(
                provider_batch_id=provider_batch_id, config=config
            )
        except Exception as error:
            logger.warning(
                f"Batch {provider_batch_id} ({provider}) lookup failed: {error}"
            )
            provider_counts["lookup_failed"] = (
                provider_counts.get("lookup_failed", 0) + 1
            )
            continue

        results.append(result)
        logger.info(f"Batch {provider_batch_id} ({provider}): {result.status}")
        provider_counts[result.status] = provider_counts.get(result.status, 0) + 1

    print("\n------------ Batch status summary ------------")
    for provider, status_counts in counter.items():
        total = sum(status_counts.values())
        print(f"Provider '{provider}' (Total: {total}):")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}/{total}")

    return results


def check_batches(
    cfg: "MMirageConfig",
    metadata_paths: Optional[Sequence[str]] = None,
) -> int:
    """Check every batch referenced by the receipts of ``cfg``.

    Args:
        cfg: Loaded MMIRAGE configuration declaring the batch_api processor(s).
        metadata_paths: Explicit receipt paths; resolved from the config when omitted.

    Returns:
        Exit code: 1 when a batch-level failure occurred, a lookup failed or the
        provider configs cannot be built, 0 otherwise. Batches still running are
        not a failure, and per request errors only show up when the results are read.
    """
    metadata_paths = resolve_metadata_paths(cfg, metadata_paths)
    records = _read_metadata_records(metadata_paths)
    unique_batches = extract_unique_provider_batches(records)
    if not unique_batches:
        logger.info(
            f"No provider batch IDs found in metadata file(s): {metadata_paths}"
        )
        return 0

    provider_configs = resolve_provider_configs(records, cfg)

    results = run_status_checker(
        metadata_records=records, provider_configs=provider_configs
    )
    # A missing result is a batch whose lookup failed, its status is unknown.
    if len(results) != len(unique_batches):
        return 1

    for result in results:
        if result.status == "failed":
            return 1
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the status-check entry point."""
    parser = argparse.ArgumentParser(
        description="Check provider batch statuses from metadata receipts."
    )
    parser.add_argument(
        "--metadata-path",
        nargs="+",
        help=(
            "Path(s) to metadata JSONL receipt file(s). Supports multiple files. "
            "When omitted, uses metadata_output_path from the config batch_api processor blocks "
            "and resolves suffixed receipts like '<base>.text.<run>.jsonl'."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that returns a process-style status code.

    Returns 0 on success or no batches found, and 1 on configuration or
    provider resolution failures.
    """
    args = _build_arg_parser().parse_args(argv)
    from mmirage.config.utils import load_mmirage_config

    try:
        return check_batches(load_mmirage_config(args.config), args.metadata_path)
    except Exception:
        logger.exception("Status checker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
