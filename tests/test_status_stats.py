from mmirage.cli_utils.status import collect_bench_stats
from mmirage.config.config import MMirageConfig, ProcessingParams
from mmirage.config.loading import LoadingParams
from mmirage.shard_utils import (
    ShardStats,
    ShardStatus,
    _write_status,
    shard_state_dir,
)


def _write_success(state_root: str, shard_id: int, stats: ShardStats) -> None:
    _write_status(
        shard_state_dir(shard_id, state_root),
        ShardStatus(status="success", shard_id=shard_id, stats=stats),
    )


def test_shard_stats_row_counts_round_trip():
    stats = ShardStats(
        rows_processed=10, rows_written=7, rows_filtered=2, rows_dropped_by_error=1
    )
    payload = stats.to_dict()
    assert payload["rows_written"] == 7
    assert payload["rows_filtered"] == 2
    assert payload["rows_dropped_by_error"] == 1

    restored = ShardStats.from_dict(payload)
    assert restored is not None
    assert restored.rows_written == 7
    assert restored.rows_filtered == 2
    assert restored.rows_dropped_by_error == 1

    partial = ShardStats.from_dict({"rows_processed": 3})
    assert partial is not None
    assert partial.rows_written is None
    assert partial.rows_filtered is None
    assert partial.rows_dropped_by_error is None


def test_aggregate_sums_row_counts_over_reporting_shards(tmp_path):
    state_root = str(tmp_path / "state")
    _write_success(
        state_root,
        0,
        ShardStats(
            rows_processed=10, rows_written=7, rows_filtered=2, rows_dropped_by_error=1
        ),
    )
    # A shard from before these counters existed reports only rows_processed.
    _write_success(state_root, 1, ShardStats(rows_processed=5))

    cfg = MMirageConfig(
        processors=[],
        loading_params=LoadingParams(state_dir=state_root, num_shards=2),
        processing_params=ProcessingParams(inputs=[], outputs=[], output_schema={}),
    )
    aggregate = collect_bench_stats(cfg)["aggregate"]

    assert aggregate["completed_shards"] == 2
    assert aggregate["total_rows_processed"] == 15
    assert aggregate["total_rows_written"] == 7
    assert aggregate["total_rows_filtered"] == 2
    assert aggregate["total_rows_dropped_by_error"] == 1


def test_aggregate_row_counts_are_none_when_no_shard_reports_them(tmp_path):
    state_root = str(tmp_path / "state")
    _write_success(state_root, 0, ShardStats(rows_processed=5))

    cfg = MMirageConfig(
        processors=[],
        loading_params=LoadingParams(state_dir=state_root, num_shards=1),
        processing_params=ProcessingParams(inputs=[], outputs=[], output_schema={}),
    )
    aggregate = collect_bench_stats(cfg)["aggregate"]

    assert aggregate["total_rows_written"] is None
    assert aggregate["total_rows_filtered"] is None
    assert aggregate["total_rows_dropped_by_error"] is None
