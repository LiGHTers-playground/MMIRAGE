"""Shared helpers for batch metadata receipt files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchMetadataRecord:
    """Typed batch receipt row shared by collector and status checker."""

    provider: str
    provider_batch_id: str
    custom_id_to_source_index: Dict[str, int] = field(default_factory=dict)
    shard_id: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BatchMetadataRecord":
        provider = str(payload.get("provider", "")).strip().lower()
        provider_batch_id = str(payload.get("provider_batch_id", "")).strip()

        raw_mapping = payload.get("custom_id_to_source_index", {})
        custom_id_to_source_index: Dict[str, int] = {}
        if isinstance(raw_mapping, dict):
            for custom_id, source_index in raw_mapping.items():
                try:
                    custom_id_to_source_index[str(custom_id)] = int(source_index)
                except (TypeError, ValueError):
                    continue

        # Receipts written before shard ids were recorded all belong to shard 0.
        try:
            shard_id = int(payload.get("shard_id", 0))
        except (TypeError, ValueError):
            shard_id = 0

        return cls(
            provider=provider,
            provider_batch_id=provider_batch_id,
            custom_id_to_source_index=custom_id_to_source_index,
            shard_id=shard_id,
        )


def _normalize_metadata_paths(metadata_paths: str | Sequence[str]) -> List[str]:
    """Return metadata paths as a concrete list."""
    if isinstance(metadata_paths, str):
        return [metadata_paths]
    return list(metadata_paths)


def _read_metadata_records(
    metadata_output_paths: str | Sequence[str],
) -> List[BatchMetadataRecord]:
    """Load valid JSON objects from one or more receipt files.

    Malformed lines are skipped with a warning so partially written or noisy
    receipt files do not stop collection. Only JSON objects are retained and
    converted into typed records with required provider identifiers.
    """
    records: List[BatchMetadataRecord] = []
    for metadata_output_path in _normalize_metadata_paths(metadata_output_paths):
        with open(metadata_output_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed metadata JSON line in %s: %s",
                        metadata_output_path,
                        exc,
                    )
                    continue
                # defensive check to ensure only dicts are included (useful against partial/corrupt metadata)
                if isinstance(parsed, dict):
                    record = BatchMetadataRecord.from_mapping(parsed)
                    if not record.provider or not record.provider_batch_id:
                        continue
                    records.append(record)
    return records
