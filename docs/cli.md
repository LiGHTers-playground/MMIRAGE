# 💻 CLI Reference

This page documents every `mmirage` subcommand and its flags.

For background on what each command does in the context of the pipeline,
see [Pipeline](pipeline.md) and [SLURM & Cluster Deployment](slurm.md).

All MMIRAGE commands share the pattern:

```
mmirage <subcommand> [flags]
```

Common flags available on most subcommands:

| Flag | Description |
|---|---|
| `--config PATH` | Path to a MMIRAGE YAML config file (**required**) |
| `--log-level LEVEL` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

---

## `mmirage run`

Run the pipeline according to `execution_params.mode` and `execution_params.retry`.

```bash
mmirage run --config configs/config.yaml [--force-retry] [--shard-id N] [--stats] [--export-prompts PATH]
```

| Flag | Description |
|---|---|
| `--force-retry` | Enable retry orchestration even if `execution_params.retry` is `false` |
| `--shard-id N` | Run a single specific shard locally, ignoring the execution mode |
| `--stats` | Enable GPU utilization and throughput collection during shard execution |
| `--export-prompts PATH` | Dry-run for `batch_api` processors: write the provider-ready requests to `PATH` instead of submitting them — see [Dry run](batch_api.md#dry-run) |

**Behaviour summary:**

- `mode: local` — Runs shards in the current Python environment.
- `mode: slurm` — Submits an sbatch array job.
- `retry: true` — After each run, automatically retries failed shards until all succeed or the retry budget is exhausted.
- `merge: true` — After all shards succeed, merges outputs into `<output_dir>/merged/`.

---

## `mmirage submit`

Submit a single SLURM array job without the retry/merge orchestration loop.

```bash
mmirage submit --config configs/config.yaml [--shard-ids 0,2,3] [--wait] [--stats]
```

| Flag | Description |
|---|---|
| `--shard-ids N,N,...` | Comma-separated shard IDs to submit instead of the full array |
| `--wait` | Block until the submitted job finishes |
| `--stats` | Enable GPU utilization and throughput collection on compute nodes |

---

## `mmirage check`

Inspect shard status from the state directory and optionally submit retries.

```bash
mmirage check --config configs/config.yaml [--retry] [-y] [--stats] [--metadata-path PATH ...]
```

| Flag | Description |
|---|---|
| `--retry` | Submit a retry job for any failed shards |
| `-y`, `--yes` | Submit retries without prompting for confirmation |
| `--stats` | Enable GPU utilization and throughput collection on retried compute nodes |
| `--metadata-path PATH [PATH ...]` | `batch_api` configs only: metadata receipt file(s). If omitted, receipts are resolved from each processor's `metadata_output_path` |

Exits with code `0` if all shards succeeded, `1` otherwise.

For a config declaring a `batch_api` processor, `check` reports the provider batch status from the metadata receipts instead of the shard state — see [Batch API](batch_api.md). It exits `1` only when a batch failed at provider level, batches still running are not a failure, and `--retry`, `-y` and `--stats` are ignored.

---

## `mmirage retry`

Submit a retry job for failed shards without inspecting status interactively.

```bash
mmirage retry --config configs/config.yaml [-y] [--stats]
```

| Flag | Description |
|---|---|
| `-y`, `--yes` | Submit retries without prompting |
| `--stats` | Enable GPU utilization and throughput collection on retried compute nodes |

---

## `mmirage merge`

Merge shard outputs for all datasets listed in `loading_params.datasets`.

```bash
mmirage merge --config configs/config.yaml [--output-root /path/to/merged]
mmirage merge --config configs/batch_config.yaml --output-path merged.jsonl [--metadata-path PATH ...]
```

| Flag | Description |
|---|---|
| `--output-root PATH` | Root directory for merged outputs. MMIRAGE creates one subdirectory per dataset. If omitted, each dataset is merged into `<dataset.output_dir>/merged` |
| `--output-path PATH` | `batch_api` configs only: JSONL file for the merged batch results (**required**) |
| `--metadata-path PATH [PATH ...]` | `batch_api` configs only: metadata receipt file(s). If omitted, receipts are resolved from each processor's `metadata_output_path` |

For a config declaring a `batch_api` processor, `merge` retrieves the completed provider results and merges them by source index into `--output-path`, since the shards only hold submission placeholders — see [Batch API](batch_api.md).

---

## `mmirage merge-dir`

Merge shard outputs directly from a directory, without a config file.

```bash
mmirage merge-dir --input-dir /path/to/shards --output-dir /path/to/merged
```

| Flag | Description |
|---|---|
| `--input-dir PATH` | Directory containing `shard_*` folders (one dataset), or a parent directory containing multiple dataset subdirectories |
| `--output-dir PATH` | Output directory for the merged dataset(s) |

If `shard_*` folders are present **directly** in `--input-dir`, MMIRAGE treats it as a single dataset and merges it there, ignoring nested internal folders such as `_pipeline_state`.

---

## `mmirage stats`

Print per-shard benchmark statistics including runtime, throughput, and GPU utilization.

```bash
mmirage stats --config configs/config.yaml
```

Stats are only available for shards that were run with `--stats` enabled (or `MMIRAGE_COLLECT_STATS=1`). Output is a JSON report:

```json
{
  "per_shard": [
    {
      "shard_id": 0,
      "status": "success",
      "started_at": "2026-04-30T10:00:00",
      "finished_at": "2026-04-30T10:01:05",
      "stats": {
        "runtime_seconds": 65.2,
        "runtime_human": "1m 5s",
        "rows_processed": 1024,
        "rows_written": 1024,
        "rows_filtered": 0,
        "rows_dropped_by_error": 0,
        "throughput_rows_per_sec": 15.7,
        "gpu_util_mean": 88.4,
        "gpu_util_min": 72.0,
        "gpu_util_max": 98.0,
        "tokens_per_sec_per_gpu": 753.1,
        "gpu_days_per_billion_tokens": 0.0015
      }
    }
  ],
  "aggregate": {
    "total_shards": 1,
    "completed_shards": 1,
    "total_rows_processed": 1024,
    "total_rows_written": 1024,
    "total_rows_filtered": 0,
    "total_rows_dropped_by_error": 0,
    "overall_throughput_rows_per_sec": 15.7,
    "mean_gpu_util_pct": 88.4,
    "tokens_per_sec_per_gpu": 753.1,
    "gpu_days_per_billion_tokens": 0.0015
  }
}
```

Key metrics:

| Metric | Description |
|---|---|
| `runtime_seconds` | Shard wall-clock time (excludes SLURM queue wait) |
| `rows_processed` / `rows_written` | Input rows in the shard vs. output rows saved to disk; a shard with `rows_written: 0` creates no `shard_*` folder |
| `rows_filtered` / `rows_dropped_by_error` | Input rows dropped by a `filter` step's predicate vs. rows whose predicate raised (both `0` without a filter step) |
| `overall_throughput_rows_per_sec` | Total rows / wall-clock time across all parallel shards |
| `tokens_per_sec_per_gpu` | Output tokens per second per GPU — primary throughput metric |
| `gpu_days_per_billion_tokens` | GPU-days to generate 1B output tokens — useful for cost comparison |
| `mean_gpu_util_pct` | Mean GPU utilization across shards |

Token metrics are `null` when no LLM processor was active. GPU stats are `null` when `nvidia-smi` is unavailable or `--stats` was not passed.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | All shards completed successfully |
| `1` | One or more shards failed or exceeded the retry budget |

---

## See also

- [Pipeline](pipeline.md) — understand the stages each command drives
- [SLURM & Cluster Deployment](slurm.md) — `submit`, `check`, `retry` in context
- [Benchmarking](benchmarking.md) — `--stats` flag and `mmirage stats` in depth
- [Configuration Reference](configuration.md) — the YAML config every command reads
