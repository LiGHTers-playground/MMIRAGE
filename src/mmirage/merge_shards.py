"""Merge processed dataset shards."""

import argparse
import logging
import os
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from mmirage.config.config import MMirageConfig
from mmirage.core.loader.base import DatasetLike
from mmirage.shard_utils import (
    MergeReport,
    _count_rows,
    _dataset_dirs,
    _list_shard_dirs,
    _save_dataset_atomic,
    _validate_input_dir,
    _validate_safe_output_dir,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """Configure logging for direct module execution.

    Keeps existing logging configuration intact when this module is invoked
    from another CLI entrypoint that already configured handlers.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _merge_datasetdict(shard_dsets: List[DatasetDict]) -> DatasetDict:
    """Merge multiple DatasetDicts by concatenating each split."""
    split_names = sorted({split for ds in shard_dsets for split in ds.keys()})
    merged: Dict[str, Dataset] = {}
    for split in split_names:
        split_dsets = [ds[split] for ds in shard_dsets if split in ds]
        if not split_dsets:
            continue
        merged[str(split)] = concatenate_datasets(split_dsets)
    if not merged:
        raise RuntimeError("All splits were empty after merging.")
    return DatasetDict(**merged)


def _merge_shards(shard_dsets: List[DatasetLike]) -> DatasetLike:
    """Merge shard datasets into a single dataset."""
    if not shard_dsets:
        raise RuntimeError("No shard datasets to merge.")
    if all(isinstance(ds, DatasetDict) for ds in shard_dsets):
        return _merge_datasetdict(
            [ds for ds in shard_dsets if isinstance(ds, DatasetDict)]
        )
    if any(isinstance(ds, DatasetDict) for ds in shard_dsets):
        raise RuntimeError("Cannot merge mix of Dataset and DatasetDict shards.")
    return concatenate_datasets([ds for ds in shard_dsets if isinstance(ds, Dataset)])


def merge_dataset_dir(dataset_dir: str, output_dir: str) -> MergeReport:
    """Merge one dataset directory containing shard_* folders.

    Args:
        dataset_dir: Input directory containing shard_* folders.
        output_dir: Destination directory for merged dataset.

    Returns:
        MergeReport with summary details.
    """
    dataset_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(dataset_dir)))
    normalized_output_dir = os.path.abspath(
        os.path.expandvars(os.path.expanduser(output_dir))
    )
    _validate_input_dir(dataset_dir, "dataset_dir")
    _validate_safe_output_dir(dataset_dir, normalized_output_dir)

    shard_dirs = _list_shard_dirs(dataset_dir)
    if not shard_dirs:
        raise RuntimeError(
            f"No shard_* folders found in {dataset_dir}. "
            "A shard that wrote 0 rows creates no shard_* folder; "
            "see rows_written in `mmirage stats`."
        )

    shard_dsets: List[DatasetLike] = []
    skipped_invalid_dirs = 0
    skipped_zero_rows = 0

    for shard_dir in shard_dirs:
        try:
            ds = load_from_disk(shard_dir)
        except FileNotFoundError as e:
            logger.warning(
                f"{shard_dir} is not a valid HF dataset directory, skipping. "
                f"Reason: {e}"
            )
            skipped_invalid_dirs += 1
            continue

        row_count = _count_rows(ds)
        if row_count == 0:
            logger.warning(f"Shard dataset has 0 rows, skipping: {shard_dir}")
            skipped_zero_rows += 1
            continue

        logger.info(f"Using {os.path.basename(shard_dir)} with {row_count} rows.")
        shard_dsets.append(ds)

    if not shard_dsets:
        raise RuntimeError(
            f"No non-empty shards found in {dataset_dir}. "
            f"empty/invalid dirs: {skipped_invalid_dirs}, "
            f"zero-row datasets: {skipped_zero_rows}. "
            "A shard that wrote 0 rows creates no shard_* folder; "
            "see rows_written in `mmirage stats`."
        )

    ds_merged = _merge_shards(shard_dsets)
    merged_rows = _count_rows(ds_merged)

    _save_dataset_atomic(ds_merged, normalized_output_dir)

    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    return MergeReport(
        dataset_name=dataset_name,
        input_dir=dataset_dir,
        output_dir=normalized_output_dir,
        used_shards=len(shard_dsets),
        merged_rows=merged_rows,
        skipped_invalid_dirs=skipped_invalid_dirs,
        skipped_zero_rows=skipped_zero_rows,
    )


def merge_input_dir(input_dir: str, output_dir: str) -> List[MergeReport]:
    """Merge all shard datasets found under an input directory.

    The input can be either:
    - one dataset dir containing shard_* folders directly
    - a parent dir containing multiple dataset subdirectories, each with shard_*
    """
    input_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(input_dir)))
    output_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(output_dir)))
    _validate_input_dir(input_dir, "input_dir")

    root_shards = _list_shard_dirs(input_dir)
    dataset_dirs = _dataset_dirs(input_dir)

    # If shards are present at the input root, treat it as a single dataset.
    # This avoids accidentally picking internal subdirectories (for example
    # pipeline state folders that may also contain shard_* entries).
    if root_shards:
        dataset_dirs = [input_dir]

    if not dataset_dirs:
        raise RuntimeError(
            f"No dataset directories with shard_* folders found in {input_dir}."
        )

    reports: List[MergeReport] = []
    for dataset_dir in dataset_dirs:
        if dataset_dir == input_dir:
            ds_output_dir = output_dir
        else:
            dataset_name = os.path.basename(dataset_dir)
            ds_output_dir = os.path.join(output_dir, dataset_name)

        reports.append(merge_dataset_dir(dataset_dir, ds_output_dir))

    return reports


def merge_from_config(
    cfg: MMirageConfig,
    output_root: Optional[str] = None,
) -> List[MergeReport]:
    """Merge shard outputs described in config.loading_params.datasets.

    Args:
        cfg: Loaded MMIRAGE config.
        output_root: Optional destination root. If omitted, each dataset writes
            into <dataset.output_dir>/merged.

    Returns:
        Merge reports for each dataset entry.
    """
    reports: List[MergeReport] = []
    datasets = cfg.loading_params.datasets
    if not datasets:
        raise RuntimeError("No datasets configured in loading_params.datasets.")

    dataset_names = [
        os.path.basename(os.path.normpath(ds_config.output_dir)) or f"dataset_{index}"
        for index, ds_config in enumerate(datasets)
    ]
    name_counts: Dict[str, int] = {}
    for dataset_name in dataset_names:
        name_counts[dataset_name] = name_counts.get(dataset_name, 0) + 1

    for index, ds_config in enumerate(datasets):
        dataset_dir = ds_config.output_dir
        dataset_name = dataset_names[index]
        if output_root is None:
            output_dir = os.path.join(dataset_dir, "merged")
        else:
            folder_name = dataset_name
            if name_counts[dataset_name] > 1:
                folder_name = f"{dataset_name}_{index}"
            output_dir = os.path.join(output_root, folder_name)

        reports.append(merge_dataset_dir(dataset_dir, output_dir))

    return reports


def main():
    """CLI entrypoint for directory-based shard merging.
    Scans --input-dir for dataset subdirectories containing shard_* folders.
    For each dataset directory, merges shard datasets and writes directly to
    the provided `--output-dir`.
    """
    ap = argparse.ArgumentParser("Merge processed shard datasets into HF datasets.")
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing dataset subdirectories with shard_* folders.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write merged datasets into.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for merge summaries.",
    )
    args = ap.parse_args()
    _configure_logging(args.log_level)

    reports = merge_input_dir(args.input_dir, args.output_dir)
    for report in reports:
        skipped_total = report.skipped_invalid_dirs + report.skipped_zero_rows
        logger.info(
            f"Concatenated {report.used_shards} shards for {report.dataset_name} "
            f"with {report.merged_rows} rows.\n"
            f"     Output: {report.output_dir}\n"
            f"     Skipped shards: {skipped_total} total "
            f"(empty/invalid dir: {report.skipped_invalid_dirs}, "
            f"zero rows: {report.skipped_zero_rows})."
        )


if __name__ == "__main__":
    main()
