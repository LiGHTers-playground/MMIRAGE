"""Utility functions for shard and merge processing.

This module contains helper functions for dataset sharding, state management,
and file operations used in the MMIRAGE shard processing pipeline.
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import humanize
from datasets import Dataset, DatasetDict

from mmirage.core.loader.base import BaseDataLoaderConfig, DatasetLike

logger = logging.getLogger(__name__)


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Format a duration given in seconds as a human-readable string."""
    if seconds is None:
        return None
    return humanize.precisedelta(seconds)


@dataclass
class ShardStats:
    """Per-shard benchmark statistics recorded at completion."""

    runtime_seconds: Optional[float] = None
    # rows_processed: input rows in this shard (drives throughput).
    # rows_written: output rows saved to disk; 0 means no shard_* folder was written.
    # rows_filtered: input rows intentionally dropped by a processor.
    # rows_dropped_by_error: input rows lost to processor errors.
    rows_processed: Optional[int] = None
    rows_written: Optional[int] = None
    rows_filtered: Optional[int] = None
    rows_dropped_by_error: Optional[int] = None
    throughput_rows_per_sec: Optional[float] = None
    gpu_util_mean: Optional[float] = None
    gpu_util_min: Optional[float] = None
    gpu_util_max: Optional[float] = None
    gpu_util_samples: Optional[int] = None
    # Token-level throughput metrics (DataTrove-compatible benchmark format).
    # input_tokens: total prompt tokens consumed across all LLM calls in this shard.
    # output_tokens: total completion tokens generated across all LLM calls in this shard.
    # num_gpus: number of GPUs used (tensor-parallel size from the LLM processor config).
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    num_gpus: Optional[int] = None
    model_load_seconds: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ShardStats"]:
        """Build a ShardStats from a JSON payload, or return None if data is missing."""
        if not isinstance(data, dict):
            return None

        def _opt_float(v: Any) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _opt_int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            runtime_seconds=_opt_float(data.get("runtime_seconds")),
            rows_processed=_opt_int(data.get("rows_processed")),
            rows_written=_opt_int(data.get("rows_written")),
            rows_filtered=_opt_int(data.get("rows_filtered")),
            rows_dropped_by_error=_opt_int(data.get("rows_dropped_by_error")),
            throughput_rows_per_sec=_opt_float(data.get("throughput_rows_per_sec")),
            gpu_util_mean=_opt_float(data.get("gpu_util_mean")),
            gpu_util_min=_opt_float(data.get("gpu_util_min")),
            gpu_util_max=_opt_float(data.get("gpu_util_max")),
            gpu_util_samples=_opt_int(data.get("gpu_util_samples")),
            input_tokens=_opt_int(data.get("input_tokens")),
            output_tokens=_opt_int(data.get("output_tokens")),
            num_gpus=_opt_int(data.get("num_gpus")),
            model_load_seconds=_opt_float(data.get("model_load_seconds")),
        )

    def to_dict(self) -> Dict[str, Any]:
        # Derived token-throughput metrics (DataTrove-compatible benchmark format).
        # Use inference_runtime (total minus model loading) so metrics reflect
        # pure generation speed, excluding one-time model initialisation overhead.
        tokens_per_sec_per_gpu: Optional[float] = None
        gpu_days_per_billion_tokens: Optional[float] = None
        inference_runtime: Optional[float] = None
        if self.runtime_seconds is not None:
            if self.model_load_seconds is not None:
                inference_runtime = max(
                    0.0, self.runtime_seconds - self.model_load_seconds
                )
            else:
                inference_runtime = self.runtime_seconds
        if (
            self.output_tokens is not None
            and self.output_tokens > 0
            and inference_runtime is not None
            and inference_runtime > 0
            and self.num_gpus is not None
            and self.num_gpus > 0
        ):
            tokens_per_sec_per_gpu = round(
                self.output_tokens / (inference_runtime * self.num_gpus), 2
            )
            gpu_days_per_billion_tokens = round(
                (self.num_gpus * inference_runtime / 86_400)
                / (self.output_tokens / 1e9),
                4,
            )

        return {
            "runtime_seconds": self.runtime_seconds,
            "runtime_human": format_duration(self.runtime_seconds),
            "model_load_seconds": round(self.model_load_seconds, 3)
            if self.model_load_seconds is not None
            else None,
            "inference_runtime_seconds": round(inference_runtime, 3)
            if inference_runtime is not None
            else None,
            "rows_processed": self.rows_processed,
            "rows_written": self.rows_written,
            "rows_filtered": self.rows_filtered,
            "rows_dropped_by_error": self.rows_dropped_by_error,
            "throughput_rows_per_sec": self.throughput_rows_per_sec,
            "gpu_util_mean": self.gpu_util_mean,
            "gpu_util_min": self.gpu_util_min,
            "gpu_util_max": self.gpu_util_max,
            "gpu_util_samples": self.gpu_util_samples,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "num_gpus": self.num_gpus,
            "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
            "gpu_days_per_billion_tokens": gpu_days_per_billion_tokens,
        }


class GpuUtilizationPoller:
    """Polls ``nvidia-smi`` in a background daemon thread.

    Usage::

        poller = GpuUtilizationPoller()
        poller.start()
        # ... do work ...
        gpu_info = poller.stop()  # {"mean": 85.2, "min": 70.0, "max": 98.0, "samples": 24}

    If ``nvidia-smi`` is unavailable all values are ``None`` and samples is 0.
    """

    def __init__(
        self, interval_seconds: float = 5.0, gpu_indices: Optional[List[str]] = None
    ) -> None:
        self._interval = interval_seconds
        self._samples: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Explicit GPU indices take priority over CUDA_VISIBLE_DEVICES.
        # Pass the indices SGLang will use (0..tp_size-1 in local mode).
        self._gpu_indices = gpu_indices

    def start(self) -> None:
        """Start background polling."""
        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop polling and return a summary dict."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2.0)
        samples = self._samples
        if not samples:
            return {"mean": None, "min": None, "max": None, "samples": 0}
        return {
            "mean": round(sum(samples) / len(samples), 1),
            "min": float(min(samples)),
            "max": float(max(samples)),
            "samples": len(samples),
        }

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            util = self._query_gpu_util()
            if util is not None:
                self._samples.append(util)

    def _query_gpu_util(self) -> Optional[float]:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            # Restrict to the GPUs this process actually uses so we don't
            # dilute utilization by averaging over idle GPUs on the same node.
            # Priority: explicit gpu_indices > CUDA_VISIBLE_DEVICES > all GPUs.
            if self._gpu_indices is not None:
                if not self._gpu_indices:
                    # Empty list would produce --id= which is invalid; skip filtering.
                    pass
                else:
                    cmd += [f"--id={','.join(str(i) for i in self._gpu_indices)}"]
            else:
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible and cuda_visible.lower() not in ("all", "nodevfiles"):
                    cmd += [f"--id={cuda_visible}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                values = []
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            values.append(float(line))
                        except ValueError:
                            pass
                if values:
                    return sum(values) / len(values)
        except Exception:
            pass
        return None


@dataclass
class ShardStatus:
    """Typed representation of the shard status.json payload."""

    status: str = "unknown"
    retry_count: int = 0
    shard_id: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    hostname: Optional[str] = None
    pid: Optional[int] = None
    slurm_job_id: Optional[str] = None
    slurm_array_task_id: Optional[str] = None
    datasets: Optional[List[Dict[str, Any]]] = None
    stats: Optional[ShardStats] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ShardStatus":
        """Build a status object from a JSON payload."""
        data = payload or {}
        try:
            retry_count = int(data.get("retry_count", 0))
        except (TypeError, ValueError):
            retry_count = 0

        shard_id = data.get("shard_id")
        if shard_id is not None:
            try:
                shard_id = int(shard_id)
            except (TypeError, ValueError):
                shard_id = None

        pid = data.get("pid")
        if pid is not None:
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None

        datasets = data.get("datasets")
        if not isinstance(datasets, list):
            datasets = None

        stats = ShardStats.from_dict(data.get("stats"))

        return cls(
            status=str(data.get("status", "unknown")),
            retry_count=retry_count,
            shard_id=shard_id,
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
            hostname=data.get("hostname"),
            pid=pid,
            slurm_job_id=data.get("slurm_job_id"),
            slurm_array_task_id=data.get("slurm_array_task_id"),
            datasets=datasets,
            stats=stats,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize status to the JSON payload written on disk."""
        return {
            "status": self.status,
            "retry_count": self.retry_count,
            "shard_id": self.shard_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "hostname": self.hostname,
            "pid": self.pid,
            "slurm_job_id": self.slurm_job_id,
            "slurm_array_task_id": self.slurm_array_task_id,
            "datasets": self.datasets,
            "stats": self.stats.to_dict() if self.stats is not None else None,
        }


@dataclass
class MergeReport:
    """Summary of a merge operation for one dataset directory."""

    dataset_name: str
    input_dir: str
    output_dir: str
    used_shards: int
    merged_rows: int
    skipped_invalid_dirs: int
    skipped_zero_rows: int


def _count_rows(ds: DatasetLike) -> int:
    """Count total rows in a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)


def _drop_empty_splits(ds: DatasetLike) -> Optional[DatasetLike]:
    """Return the dataset without its 0-row splits, or None if nothing is left.

    A 0-row dataset must never reach `_save_dataset_atomic`: `save_to_disk`
    either fails outright (with an `Image` column) or writes a folder that
    `load_from_disk` cannot reload.
    """
    if isinstance(ds, DatasetDict):
        kept = {name: split for name, split in ds.items() if len(split) > 0}
        dropped = [name for name in ds if name not in kept]
        if dropped:
            logger.info(f"Dropping 0-row split(s) before save: {dropped}")
        return DatasetDict(kept) if kept else None
    return ds if len(ds) > 0 else None


def _shard_split(split_ds: Dataset, num_shards: int, shard_id: int) -> Dataset:
    """Shard a single dataset, returning an empty dataset for out-of-range shards.

    `Dataset.shard` raises `IndexError` when the dataset has fewer rows than
    `num_shards` and `shard_id` falls past its last row.
    """
    if shard_id >= len(split_ds):
        return split_ds.select([])
    return split_ds.shard(num_shards=num_shards, index=shard_id)


def _shard_dataset(ds: DatasetLike, num_shards: int, shard_id: int) -> DatasetLike:
    """Shard a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return DatasetDict(
            {
                split: _shard_split(split_ds, num_shards, shard_id)
                for split, split_ds in ds.items()
            }
        )
    return _shard_split(ds, num_shards, shard_id)


def _remove_columns(ds: Dataset) -> List[str]:
    """Get columns to remove from a dataset (or one split of a dataset dict) if enabled.

    Each split of a `DatasetDict` must be mapped with its own column names:
    `DatasetDict.map(remove_columns=...)` raises when a split lacks a column.
    """
    return ds.column_names


def _save_dataset_atomic(ds_processed: DatasetLike, out_dir: str):
    """Save dataset atomically via temporary directory + rename."""
    parent_dir = os.path.dirname(out_dir)
    os.makedirs(parent_dir, exist_ok=True)

    tmp_dir = f"{out_dir}.tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    ds_processed.save_to_disk(tmp_dir)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.replace(tmp_dir, out_dir)


def _validate_safe_output_dir(dataset_dir: str, output_dir: str) -> None:
    """Reject output paths that could delete input data.

    We forbid output directories that are the same as, or ancestors of,
    the input dataset directory. This prevents accidental deletion when
    clearing pre-existing output_dir before writing merged data.
    """
    dataset_real = os.path.realpath(os.path.abspath(dataset_dir))
    output_real = os.path.realpath(os.path.abspath(output_dir))

    if output_real == dataset_real:
        raise RuntimeError(
            "Unsafe merge output path: output_dir equals dataset_dir "
            f"(dataset_dir={dataset_real}, output_dir={output_real})."
        )

    try:
        common = os.path.commonpath([dataset_real, output_real])
    except ValueError:
        # Different drives (Windows) -> no ancestor relationship possible
        return

    if common == output_real:
        raise RuntimeError(
            "Unsafe merge output path: output_dir contains dataset_dir "
            f"(dataset_dir={dataset_real}, output_dir={output_real})."
        )


def _dataset_out_dir(shard_idx: int, ds_config: BaseDataLoaderConfig) -> str:
    """Get dataset-specific output directory for a shard."""
    return os.path.join(ds_config.output_dir, f"shard_{shard_idx}")


def shard_state_dir(shard_idx: int, state_root: str) -> str:
    """Get central state directory for a logical shard."""
    return os.path.join(state_root, f"shard_{shard_idx}")


def _cleanup_old_shard_data(out_dir: str):
    """Remove old dataset shard output before retry."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        logger.info(f"Removed old shard output: {out_dir}")


def _status_file(state_dir: str) -> str:
    """Canonical status file path."""
    return os.path.join(state_dir, "status.json")


def read_status(state_dir: str) -> ShardStatus:
    """Read status.json if present."""
    path = _status_file(state_dir)
    if not os.path.exists(path):
        return ShardStatus(status="missing")
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(f"Invalid status format in {path}; expected object")
                return ShardStatus(status="unknown")
            return ShardStatus.from_dict(data)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read status file {path}: {e}")
        return ShardStatus(status="unknown")


def _write_status(state_dir: str, payload: ShardStatus):
    """Atomically write status.json."""
    os.makedirs(state_dir, exist_ok=True)
    tmp_path = _status_file(state_dir) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload.to_dict(), f, indent=2, sort_keys=True)
    os.replace(tmp_path, _status_file(state_dir))


def _clear_markers(state_dir: str):
    """Remove status marker files."""
    for name in (".RUNNING", ".SUCCESS", ".FAILED"):
        path = os.path.join(state_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Failed to remove marker {path}: {e}")


def _touch_marker(state_dir: str, name: str):
    """Create a marker file."""
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, name)
    with open(path, "w") as f:
        f.write(f"{datetime.now().isoformat()}\n")


def _mark_running(
    state_dir: str,
    shard_id: int,
    datasets_config: List[BaseDataLoaderConfig],
) -> int:
    """Mark shard as running and increment retry count."""
    prev = read_status(state_dir)
    retry_count = prev.retry_count + 1

    payload = ShardStatus(
        status="running",
        retry_count=retry_count,
        shard_id=shard_id,
        started_at=datetime.now().isoformat(),
        finished_at=None,
        error=None,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        datasets=[
            {
                "image_base_path": ds_config.image_base_path,
                "output_dir": ds_config.output_dir,
            }
            for ds_config in datasets_config
        ],
    )

    _write_status(state_dir, payload)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".RUNNING")
    return retry_count


def _mark_success(state_dir: str, stats: Optional[ShardStats] = None):
    """Mark shard as successful and record benchmark statistics.

    Args:
        state_dir: Shard state directory.
        stats: Optional benchmark stats; ``runtime_seconds`` and
            ``throughput_rows_per_sec`` are computed from the stored timestamps
            when not already set.
    """
    prev = read_status(state_dir)
    prev.status = "success"
    now = datetime.now()
    prev.finished_at = now.isoformat()
    prev.error = None

    if stats is not None:
        # Derive runtime from stored start timestamp when not already supplied.
        if stats.runtime_seconds is None and prev.started_at:
            try:
                started = datetime.fromisoformat(prev.started_at)
                stats.runtime_seconds = round((now - started).total_seconds(), 3)
            except (ValueError, TypeError):
                pass

        # Derive throughput once we have both rows and runtime.
        # Use inference_runtime (total minus model loading) so the metric
        # reflects pure generation speed, consistent with tokens_per_sec_per_gpu.
        if (
            stats.throughput_rows_per_sec is None
            and stats.rows_processed is not None
            and stats.runtime_seconds is not None
            and stats.runtime_seconds > 0
        ):
            inference_runtime = (
                max(0.0, stats.runtime_seconds - stats.model_load_seconds)
                if stats.model_load_seconds is not None
                else stats.runtime_seconds
            )
            if inference_runtime > 0:
                stats.throughput_rows_per_sec = round(
                    stats.rows_processed / inference_runtime, 2
                )

    prev.stats = stats
    _write_status(state_dir, prev)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".SUCCESS")


def _mark_failure(state_dir: str, error_msg: str):
    """Mark shard as failed."""
    prev = read_status(state_dir)
    prev.status = "failed"
    prev.finished_at = datetime.now().isoformat()
    prev.error = error_msg
    _write_status(state_dir, prev)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".FAILED")


def _list_shard_dirs(dataset_dir: str) -> List[str]:
    """List shard directories in a dataset directory."""
    shard_dirs: List[str] = []
    for name in os.listdir(dataset_dir):
        if not name.startswith("shard_"):
            continue
        # Only accept canonical shard directories of the form "shard_<int>"
        # and explicitly skip atomic-save temp dirs like
        # "shard_0.tmp.<host>.<pid>.<uuid>".
        if ".tmp" in name:
            continue
        suffix = name[len("shard_") :]
        if not suffix.isdigit():
            continue
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            shard_dirs.append(path)

    def _shard_key(path: str) -> int:
        base = os.path.basename(path)
        suffix = base.removeprefix("shard_")
        return int(suffix) if suffix.isdigit() else 0

    shard_dirs.sort(key=_shard_key)
    return shard_dirs


def _dataset_dirs(input_dir: str) -> List[str]:
    """Find dataset directories containing shard folders."""
    candidates: List[str] = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        if _list_shard_dirs(path):
            candidates.append(path)
    return sorted(candidates)


def _validate_input_dir(path: str, arg_name: str) -> None:
    """Ensure a user-provided input path exists and is a directory."""
    normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    if not os.path.isdir(normalized):
        raise RuntimeError(
            f"{arg_name} does not exist or is not a directory: {normalized}"
        )
