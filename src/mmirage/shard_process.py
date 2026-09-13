"""Main script for processing dataset shards with MMIRAGE.

Supports both text-only and multimodal (vision-language) processing.
"""

import argparse
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets

from mmirage.cli_utils.runtime import non_empty_path
from mmirage.config.utils import load_mmirage_config
from mmirage.core.loader.base import DatasetLike
from mmirage.core.loader.utils import load_datasets_from_configs
from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.process.processors.filter.config import FilterStep
from mmirage.core.process.variables import OutputVar
from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.shard_utils import (
    GpuUtilizationPoller,
    ShardStats,
    _cleanup_old_shard_data,
    _count_rows,
    _dataset_out_dir,
    _drop_empty_splits,
    _mark_failure,
    _mark_running,
    _mark_success,
    _remove_columns,
    _save_dataset_atomic,
    _shard_dataset,
    shard_state_dir,
)

logger = logging.getLogger(__name__)


def _image_path_schema_cols(
    output_vars: List[OutputVar],
    output_schema: Dict[str, Any],
    renderer: TemplateRenderer,
) -> List[str]:
    """Return output-schema column names that map directly to image-path output variables.

    Uses duck typing on ``output_mode`` so no concrete processor import is needed.
    """
    image_path_var_names = {
        v.name
        for v in output_vars
        if getattr(v, "output_mode", None) == "path"
        or getattr(getattr(v, "output_mode", None), "value", None) == "path"
    }
    return [
        key
        for key, tmpl in output_schema.items()
        if isinstance(tmpl, str)
        and renderer.is_single_variable_template(tmpl) in image_path_var_names
    ]


def _cast_image_columns(ds: DatasetLike, cols: List[str]) -> DatasetLike:
    """Cast image-path string columns to the HuggingFace Image feature.

    Empty strings (failure fallbacks) are normalised to ``None`` so that
    HuggingFace stores them as missing rather than raising a decode error.
    When ``save_to_disk`` is called, HuggingFace reads each path from disk
    and embeds the raw bytes in the Arrow file, making the shard portable.
    """
    try:
        from datasets import Image as HFImage
    except ImportError as exc:
        raise RuntimeError(
            "Generated image path columns require the optional HuggingFace "
            "`datasets.Image` feature when processing_params.cast_images is true. "
            "Install `datasets` with image support or set "
            "`processing_params.cast_images: false` to keep paths as strings."
        ) from exc

    def _normalise_col(batch: Dict[str, Any], col: str) -> Dict[str, Any]:
        normalized: List[Any] = []
        for v in batch[col]:
            if v is None:
                normalized.append(None)
            elif isinstance(v, str) and v.strip().lower() in ("", "none"):
                normalized.append(None)
            else:
                normalized.append(v)
        return {col: normalized}

    def _cast_column(dataset: DatasetLike, col: str) -> DatasetLike:
        dataset = dataset.map(
            _normalise_col,
            batched=True,
            fn_kwargs={"col": col},
            desc=f"Normalising {col}",
            load_from_cache_file=False,
        )
        return dataset.cast_column(col, HFImage())

    if isinstance(ds, DatasetDict):
        for col in cols:
            for split in list(ds.keys()):
                if col in ds[split].column_names:
                    ds[split] = _cast_column(ds[split], col)
    else:
        for col in cols:
            if col in ds.column_names:
                ds = _cast_column(ds, col)
    return ds


def rewrite_batch(
    batch: Dict[str, List[Any]],
    indices: Optional[List[int]] = None,
    *,
    mapper: MMIRAGEMapper,
    renderer: TemplateRenderer,
    image_base_path: Optional[str] = None,
    kept: Optional[List[int]] = None,
) -> Dict[str, List[Any]]:
    """Rewrite a batch of samples by applying transformations.
    Args:
        batch: Dictionary mapping column names to lists of values.
        indices: Dataset positions of the rows; ``Dataset.map(with_indices=True)``
            passes them as the second positional argument.
        mapper: MMIRAGEMapper for processing transformations.
        renderer: TemplateRenderer for generating output.
        image_base_path: Optional base directory for resolving relative image paths.
        kept: When given, the ``row_index`` of every row that survived the
            filter steps is appended to it, in dataset order.
    Returns:
        Dictionary mapping output keys to lists of rendered values.
    Raises:
        ValueError: If variables are not computable given the configuration.
    """
    if not mapper.validate_vars():
        raise ValueError(
            "Uncomputable variables detected. Verify your configuration and make sure that there is no undefined variables"
        )

    batch_environment = mapper.rewrite_batch(batch, image_base_path, indices=indices)
    if kept is not None:
        kept.extend(
            env.row_index for env in batch_environment if env.row_index is not None
        )
    if not batch_environment:
        # The renderer returns {} for no rows, which `Dataset.map` rejects as a
        # schema mismatch; an empty list per output key is a valid 0-row batch.
        return {key: [] for key in renderer.output_schema}
    rendered_list = renderer.batch_render(batch_environment)
    return rendered_list


def _map_split(
    split_ds: Dataset,
    mapper: MMIRAGEMapper,
    renderer: TemplateRenderer,
    image_base_path: Optional[str],
    batch_size: int,
    remove_columns: bool,
    desc: str,
) -> Dataset:
    """Map one dataset (or one split of a dataset dict) through ``rewrite_batch``.

    Columns are removed per split so a ``DatasetDict`` whose splits have
    different columns does not fail on a column missing from one of them.

    With a filter step the map may return fewer rows than it received, which
    ``Dataset.map`` only accepts when every input column is removed. The kept
    row positions are collected instead, and when the caller wants the input
    columns they are selected back and joined column-wise onto the output.
    """
    has_filter = any(isinstance(o, FilterStep) for o in mapper.output_vars)
    if not has_filter:
        return split_ds.map(
            rewrite_batch,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False,
            desc=desc,
            fn_kwargs={
                "mapper": mapper,
                "renderer": renderer,
                "image_base_path": image_base_path,
            },
            remove_columns=_remove_columns(split_ds) if remove_columns else [],
        )

    # Filled in-process by rewrite_batch, one call per batch in dataset order.
    # Safe because num_proc is unset: the map runs in this process.
    kept: List[int] = []
    ds_processed = split_ds.map(
        rewrite_batch,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        load_from_cache_file=False,
        desc=desc,
        fn_kwargs={
            "mapper": mapper,
            "renderer": renderer,
            "image_base_path": image_base_path,
            "kept": kept,
        },
        remove_columns=split_ds.column_names,
    )
    if remove_columns or len(ds_processed) == 0:
        return ds_processed

    # Join the surviving input rows back on. `concatenate_datasets(axis=1)`
    # keeps features such as ClassLabel and Image as-is but raises on a
    # duplicated column name, so output-schema columns are dropped from the
    # input side first (the map would have overwritten them anyway).
    kept_ds = split_ds.select(kept).remove_columns(
        [c for c in split_ds.column_names if c in renderer.output_schema]
    )
    return concatenate_datasets([kept_ds, ds_processed], axis=1)


def main():
    """
    Process a single shard of the dataset.
    Loads configuration, datasets, processes the shard using MMIRAGE
    transformations (including multimodal), and saves the result to disk.
    """
    ap = argparse.ArgumentParser("Process dataset shards using MMIRAGE with SGLang.")
    ap.add_argument(
        "--config",
        help="YAML config for MMIRAGE pipeline.",
        required=True,
    )
    ap.add_argument(
        "--export-prompts",
        type=non_empty_path,
        help="Directory or .jsonl path for exporting batch prompts instead of submitting them",
        default=None,
    )
    args = ap.parse_args()

    cfg = load_mmirage_config(args.config)
    loading_params = cfg.loading_params
    processing_params = cfg.processing_params
    datasets_config = loading_params.datasets

    if not datasets_config:
        raise ValueError("No datasets provided in config.loading_params.datasets")

    shard_id = loading_params.get_shard_id()
    num_shards = loading_params.get_num_shards()
    last_shard_id = num_shards - 1

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    state_dir = shard_state_dir(shard_id, loading_params.get_state_root())

    gpu_poller: Optional[GpuUtilizationPoller] = None

    collect_stats = os.environ.get("MMIRAGE_COLLECT_STATS", "") == "1"
    if collect_stats:
        # Determine which physical GPU indices SGLang will use so the poller
        # measures only the active GPU(s) — not all GPUs on the node.
        # SLURM may allocate more GPUs than tp_size (e.g. gpus=4, tp_size=1).
        # We take only the first tp_size entries from CUDA_VISIBLE_DEVICES so
        # nvidia-smi --id receives exactly the GPUs SGLang is using.
        tp_size = 1
        for proc_cfg in cfg.processors:
            tp = getattr(getattr(proc_cfg, "server_args", None), "tp_size", None)
            if tp and int(tp) > 0:
                tp_size = int(tp)
                break
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and cuda_visible.lower() not in ("all", "nodevfiles"):
            all_visible = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            # Fall back to range-based indices if CUDA_VISIBLE_DEVICES was set
            # but contained only whitespace/empty entries after stripping.
            gpu_indices_for_polling: List[str] = (
                all_visible[:tp_size]
                if all_visible
                else [str(i) for i in range(tp_size)]
            )
        else:
            gpu_indices_for_polling = [str(i) for i in range(tp_size)]

        gpu_poller = GpuUtilizationPoller(
            interval_seconds=5.0, gpu_indices=gpu_indices_for_polling
        )

    try:
        retry_count = _mark_running(state_dir, shard_id, datasets_config)
        logger.info(
            f"Starting shard {shard_id}/{last_shard_id} (attempt #{retry_count})"
        )

        if retry_count > 1:
            for ds_config in datasets_config:
                out_dir = _dataset_out_dir(shard_id, ds_config)
                _cleanup_old_shard_data(out_dir)

        ds_all = load_datasets_from_configs(datasets_config)
        total_rows = sum(_count_rows(ds) for ds in ds_all)

        ds_all_shard = [_shard_dataset(ds, num_shards, shard_id) for ds in ds_all]
        shard_rows = sum(_count_rows(ds) for ds in ds_all_shard)

        logger.info(
            f"Loaded {len(datasets_config)} dataset(s): {datasets_config} "
            f"→ {total_rows} total rows; this logical shard has {shard_rows} rows."
        )

        if shard_rows == 0:
            # Nothing to process: skip model loading and record an empty success.
            logger.info(
                f"Logical shard {shard_id} has no input rows; marking success "
                "without loading processors."
            )
            _mark_success(state_dir, stats=ShardStats(rows_processed=0, rows_written=0))
            return

        mapper = MMIRAGEMapper(
            cfg.processors,
            processing_params.inputs,
            processing_params.outputs,
            export_prompts_dir=args.export_prompts,
            shard_id=shard_id,
        )
        renderer = TemplateRenderer(processing_params.output_schema)

        try:
            # Start GPU polling after model loading so utilisation samples reflect
            # inference only, not weight transfers during sgl.Engine() init.
            if collect_stats and gpu_poller is not None:
                gpu_poller.start()

            ds_processed_all: List[Optional[DatasetLike]] = []
            for ds_idx, ds_shard in enumerate(ds_all_shard):
                ds_config = datasets_config[ds_idx]

                logger.info(
                    f"Processing dataset {ds_idx} for shard {shard_id}: "
                    f"image_base_path={ds_config.image_base_path}, output_dir={ds_config.output_dir}"
                )

                desc = f"Shard {shard_id}/{last_shard_id} dataset {ds_idx}"
                ds_processed: DatasetLike
                if isinstance(ds_shard, DatasetDict):
                    ds_processed = DatasetDict(
                        {
                            split: _map_split(
                                split_ds,
                                mapper=mapper,
                                renderer=renderer,
                                image_base_path=ds_config.image_base_path,
                                batch_size=loading_params.get_batch_size(),
                                remove_columns=processing_params.remove_columns,
                                desc=f"{desc} split {split}",
                            )
                            for split, split_ds in ds_shard.items()
                        }
                    )
                else:
                    ds_processed = _map_split(
                        ds_shard,
                        mapper=mapper,
                        renderer=renderer,
                        image_base_path=ds_config.image_base_path,
                        batch_size=loading_params.get_batch_size(),
                        remove_columns=processing_params.remove_columns,
                        desc=desc,
                    )
                # Drain stateful batch accumulators once every split of this dataset is mapped.
                mapper.finalize_processors()

                # A 0-row dataset cannot be cast or saved; None means "nothing to write".
                ds_to_save = _drop_empty_splits(ds_processed)
                if ds_to_save is None:
                    logger.info(
                        f"Shard {shard_id} wrote 0 rows for dataset {ds_idx}; "
                        "no output folder will be created."
                    )
                    ds_processed_all.append(None)
                    continue
                ds_processed = ds_to_save

                image_cols = _image_path_schema_cols(
                    processing_params.outputs,
                    processing_params.output_schema,
                    renderer,
                )
                if image_cols and processing_params.cast_images:
                    ds_processed = _cast_image_columns(ds_processed, image_cols)
                    logger.info(
                        f"Cast image column(s) to HF Image feature: {image_cols}"
                    )
                elif image_cols:
                    logger.info(
                        "Leaving generated image column(s) as paths because "
                        "processing_params.cast_images is false: %s",
                        image_cols,
                    )

                ds_processed_all.append(ds_processed)

            # Before any save, so a two-dataset shard never writes dataset 0
            # and then fails on dataset 1.
            mapper.check_filter_errors()
            for idx, seen in mapper.rows_seen.items():
                step = mapper.output_vars[idx]
                predicate = step.predicate if isinstance(step, FilterStep) else ""
                logger.info(
                    f"Filter step outputs[{idx}] ({predicate!r}): {seen} rows in, "
                    f"{mapper.rows_filtered[idx]} filtered, "
                    f"{mapper.rows_dropped_by_error[idx]} dropped by error"
                )

            for ds_idx, (ds_config, ds_processed) in enumerate(
                zip(datasets_config, ds_processed_all)
            ):
                if ds_processed is None:
                    continue
                out_dir = _dataset_out_dir(shard_id, ds_config)
                _save_dataset_atomic(ds_processed, out_dir)
                logger.info(f"✅ Saved dataset {ds_idx} shard in: {out_dir}")

            gpu_info = (
                gpu_poller.stop()
                if collect_stats and gpu_poller is not None
                else {"mean": None, "min": None, "max": None, "samples": 0}
            )

            # Collect token counts accumulated by LLM processor(s).
            token_counts = mapper.get_token_counts()
            input_tokens = token_counts.input_tokens or None
            output_tokens = token_counts.output_tokens or None
            model_load_seconds = mapper.get_load_time() or None

            # Resolve num_gpus from the first processor config that exposes tp_size.
            num_gpus: Optional[int] = None
            for proc_cfg in cfg.processors:
                tp = getattr(getattr(proc_cfg, "server_args", None), "tp_size", None)
                if tp and tp > 0:
                    num_gpus = int(tp)
                    break

            # Datasets that produced 0 rows were replaced by None and never saved.
            rows_written = sum(
                _count_rows(ds) for ds in ds_processed_all if ds is not None
            )

            stats = ShardStats(
                rows_processed=shard_rows,
                rows_written=rows_written,
                rows_filtered=mapper.total_rows_filtered(),
                rows_dropped_by_error=mapper.total_rows_dropped_by_error(),
                gpu_util_mean=gpu_info["mean"],
                gpu_util_min=gpu_info["min"],
                gpu_util_max=gpu_info["max"],
                gpu_util_samples=gpu_info["samples"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                num_gpus=num_gpus,
                model_load_seconds=model_load_seconds,
            )
            _mark_success(state_dir, stats=stats)
            logger.info(f"✅ Logical shard {shard_id} completed successfully")

        finally:
            mapper.shutdown()
            logger.info("Processors shut down.")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Shard {shard_id} failed: {error_msg}")
        logger.error(traceback.format_exc())
        if collect_stats and gpu_poller is not None:
            gpu_poller.stop()
        _mark_failure(state_dir, error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
