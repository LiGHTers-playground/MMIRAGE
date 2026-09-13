"""Collect provider batch receipts and merge completed rows by shard and source index.

The receiver consumes one or more metadata receipt files, resolves the provider
configuration for each recorded batch, fetches the provider results, and writes a
single JSONL file ordered by shard, then by the original source row index within
the shard. When a retried shard left several receipts for the same row, the row
from the receipt submitted last is kept.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.metadata_paths import resolve_metadata_paths
from mmirage.core.process.batch.metadata_utils import (
    BatchMetadataRecord,
    _read_metadata_records,
)
from mmirage.core.process.batch.provider_resolution import resolve_provider_configs
from mmirage.core.process.batch.registry import BatchAdapterRegistry

if TYPE_CHECKING:
    from mmirage.config.config import MMirageConfig

logger = logging.getLogger(__name__)


def _aggregate_batch_mappings(
    records: Sequence[BatchMetadataRecord],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Group source-index mappings by provider and provider batch ID.

    Later receipt lines for the same provider batch overwrite earlier entries for
    the same custom ID. Choosing between different batches that cover the same
    row is left to ``collect_and_merge``, which keeps the newest receipt.
    """
    aggregated: Dict[Tuple[str, str], Dict[str, int]] = {}

    for record in records:
        provider = record.provider
        provider_batch_id = record.provider_batch_id
        mapping = record.custom_id_to_source_index

        if not provider or not provider_batch_id or not mapping:
            continue

        key = (provider, provider_batch_id)
        aggregated.setdefault(key, {})

        for custom_id, source_index in mapping.items():
            aggregated[key][str(custom_id)] = source_index

    return aggregated


def _latest_receipt_by_batch(
    records: Sequence[BatchMetadataRecord],
) -> Dict[Tuple[str, str], BatchMetadataRecord]:
    """Keep one receipt per provider batch: the last submitted, then the last read."""
    latest: Dict[Tuple[str, str], BatchMetadataRecord] = {}

    for record in records:
        key = (record.provider, record.provider_batch_id)
        current = latest.get(key)
        if current is None or record.submitted_at_utc >= current.submitted_at_utc:
            latest[key] = record

    return latest


def collect_and_merge(
    records: Sequence[BatchMetadataRecord],
    provider_configs: Mapping[str, BatchProviderConfig],
    output_path: str,
) -> List[Dict[str, Any]]:
    """Fetch provider outputs and write merged rows in shard, then source index order.

    Args:
        records: Parsed receipt metadata containing provider batch references.
        provider_configs: Provider-specific configuration keyed by normalized
            provider name.
        output_path: Destination JSONL path for the merged output.

    Returns:
        The ordered rows that were written to disk.

    Raises:
        ValueError: If a receipt references a provider that cannot be resolved.
    """
    pair_to_mapping = _aggregate_batch_mappings(records)
    pair_to_receipt = _latest_receipt_by_batch(records)

    adapters: Dict[str, Any] = {}
    pair_to_results: Dict[Tuple[str, str], Sequence[Dict[str, Any]]] = {}

    for provider, provider_batch_id in pair_to_mapping.keys():
        if provider not in provider_configs:
            raise ValueError(f"No provider config found for '{provider}'.")

        if provider not in adapters:
            adapters[provider] = BatchAdapterRegistry.create(provider_configs[provider])

        pair = (provider, provider_batch_id)
        pair_to_results[pair] = adapters[provider].retrieve_results(
            provider_batch_id=provider_batch_id,
            config=provider_configs[provider],
        )

    # One row per (shard_id, source_index): a retried shard resubmits its rows under
    # new batch ids, and the receipt submitted last is the one that counts.
    rows_by_position: Dict[Tuple[int, int], Tuple[str, Dict[str, Any]]] = {}
    for pair, mapping in pair_to_mapping.items():
        receipt = pair_to_receipt[pair]
        results = pair_to_results.get(pair, [])
        for result_row in results:
            custom_id = str(result_row.get("custom_id", "")).strip()
            if not custom_id or custom_id not in mapping:
                continue
            row_payload = _build_output_payload(result_row, custom_id=custom_id)
            usage = {
                key: result_row[key]
                for key in ("input_tokens", "output_tokens")
                if key in result_row
            }
            row = {
                "shard_id": receipt.shard_id,
                "source_index": int(mapping[custom_id]),
                "custom_id": custom_id,
                **row_payload,
                **usage,
            }
            position = (receipt.shard_id, row["source_index"])
            current = rows_by_position.get(position)
            if current is None or receipt.submitted_at_utc >= current[0]:
                rows_by_position[position] = (receipt.submitted_at_utc, row)

    # Every shard counts its rows from 0, so order by shard first; custom_id keeps
    # the order deterministic should a position ever hold more than one row.
    ordered_rows = sorted(
        (row for _, row in rows_by_position.values()),
        key=lambda row: (row["shard_id"], row["source_index"], row["custom_id"]),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ordered_rows


def _build_output_payload(
    result_row: Mapping[str, Any], custom_id: str = ""
) -> Dict[str, Any]:
    """Convert provider content into the receiver's output schema.

    The collector preserves raw text for opaque generations, but maps structured
    question/answer JSON into a conversation format expected by downstream
    consumers.
    """
    error_message = str(result_row.get("error_message", "")).strip()
    if error_message:
        return {
            "status": str(result_row.get("status", "error") or "error"),
            "error_message": error_message,
        }

    raw_content = _extract_content_string(result_row)
    if not raw_content:
        return {"caption": ""}

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        stripped_content = raw_content.lstrip()
        if stripped_content.startswith(("{", "[")):
            logger.warning(
                f"Failed to parse JSON for result row (custom_id={custom_id}). "
                f"Treating as raw text. Content: {raw_content[:100]}"
            )
        return {"caption": raw_content}

    if isinstance(parsed, dict) and ("question" in parsed or "answer" in parsed):
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": str(parsed.get("question", "")),
                },
                {
                    "role": "assistant",
                    "content": str(parsed.get("answer", "")),
                },
            ]
        }

    return {"caption": raw_content}


def _extract_content_string(result_row: Mapping[str, Any]) -> str:
    """Return the generated text payload as a string.

    The collector treats missing content as empty output rather than a hard
    failure so incomplete provider responses do not block the merge.
    """
    return str(result_row.get("generated_text", ""))


def collect_batches(
    cfg: "MMirageConfig",
    output_path: str,
    metadata_paths: Optional[Sequence[str]] = None,
) -> int:
    """Retrieve every batch referenced by the receipts of ``cfg`` and merge them.

    Args:
        cfg: Loaded MMIRAGE configuration declaring the batch_api processor(s).
        output_path: Destination JSONL path for the merged output.
        metadata_paths: Explicit receipt paths; resolved from the config when omitted.

    Returns:
        Exit code: 0 on success.
    """
    metadata_paths = resolve_metadata_paths(cfg, metadata_paths)
    records = _read_metadata_records(metadata_paths)
    provider_configs = resolve_provider_configs(records, cfg)

    rows = collect_and_merge(records, provider_configs, output_path)
    logger.info(f"Merged {len(rows)} rows and saved to {output_path}")

    input_tokens = sum(row.get("input_tokens", 0) for row in rows)
    output_tokens = sum(row.get("output_tokens", 0) for row in rows)
    if input_tokens or output_tokens:
        logger.info(
            f"Provider usage: {input_tokens} input tokens, {output_tokens} output tokens"
        )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect provider batch outputs and merge rows by source index."
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
        "--output-path",
        required=True,
        help="Path to write merged JSONL output.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collector CLI.

    Reads receipt metadata, resolves provider configs from the supplied MMIRAGE
    configuration, and writes the merged JSONL output path passed on the command
    line.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from mmirage.config.utils import load_mmirage_config

    try:
        return collect_batches(
            load_mmirage_config(args.config),
            args.output_path,
            args.metadata_path,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    except Exception:
        logger.exception("Collector failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
