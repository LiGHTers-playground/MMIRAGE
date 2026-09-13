"""The filter step from YAML to merged output, through shard_process.main()."""

import json
import os
import sys

import pytest
import yaml
from datasets import load_from_disk

from mmirage import shard_process
from mmirage.cli_utils.status import collect_bench_stats
from mmirage.config.utils import load_mmirage_config
from mmirage.merge_shards import merge_from_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_FILTER_CONFIG = os.path.join(REPO_ROOT, "configs", "config_mock_filter.yaml")
MOCK_SCRIPT = os.path.join(REPO_ROOT, "tests", "mock_script", "mock_script.py")

# Dataset.shard is contiguous: with two shards, shard 0 gets the long texts
# and shard 1 the short ones.
LONG_TEXTS = ["a sentence long enough to keep", "another sentence that is kept"]
SHORT_TEXTS = ["tiny", "also tiny"]


def _write_config(tmp_path, predicate: str) -> str:
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for text in LONG_TEXTS + SHORT_TEXTS:
            f.write(json.dumps({"text": text}) + "\n")

    with open(MOCK_FILTER_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["processors"][0]["script_path"] = MOCK_SCRIPT
    cfg["loading_params"]["num_shards"] = 2
    cfg["loading_params"]["batch_size"] = 2
    # Resolved from the environment at load time, like a SLURM array job.
    cfg["loading_params"]["shard_id"] = "${SLURM_ARRAY_TASK_ID}"
    cfg["loading_params"]["state_dir"] = str(tmp_path / "state")
    cfg["loading_params"]["datasets"][0]["path"] = str(data_path)
    cfg["loading_params"]["datasets"][0]["output_dir"] = str(tmp_path / "output")
    cfg["processing_params"]["outputs"][1]["predicate"] = predicate

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(config_path)


def _run_shard(monkeypatch, config_path: str, shard_id: int) -> None:
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(shard_id))
    monkeypatch.setattr(sys, "argv", ["shard_process", "--config", config_path])
    shard_process.main()


def _status(tmp_path, shard_id: int) -> dict:
    with open(tmp_path / "state" / f"shard_{shard_id}" / "status.json") as f:
        return json.load(f)


def test_filter_keeps_long_texts_and_drops_the_short_shard(tmp_path, monkeypatch):
    # Long texts score 29 and 30, short ones 4 and 9.
    config_path = _write_config(tmp_path, predicate="score > 20")

    for shard_id in (0, 1):
        _run_shard(monkeypatch, config_path, shard_id)

    assert (tmp_path / "output" / "shard_0").is_dir()
    assert not (tmp_path / "output" / "shard_1").exists()

    for shard_id, written, filtered in ((0, 2, 0), (1, 0, 2)):
        status = _status(tmp_path, shard_id)
        assert status["status"] == "success", status.get("error")
        assert status["stats"]["rows_processed"] == 2
        assert status["stats"]["rows_written"] == written
        assert status["stats"]["rows_filtered"] == filtered
        assert status["stats"]["rows_dropped_by_error"] == 0

    cfg = load_mmirage_config(config_path)
    aggregate = collect_bench_stats(cfg)["aggregate"]
    assert aggregate["completed_shards"] == 2
    assert aggregate["total_rows_processed"] == 4
    assert aggregate["total_rows_written"] == 2
    assert aggregate["total_rows_filtered"] == 2
    assert aggregate["total_rows_dropped_by_error"] == 0

    (report,) = merge_from_config(cfg)
    assert report.merged_rows == 2
    merged = load_from_disk(report.output_dir)
    assert sorted(merged["text"]) == sorted(LONG_TEXTS)
    assert merged.column_names == ["text", "score"]


def test_predicate_that_errors_on_every_row_fails_the_shard(tmp_path, monkeypatch):
    # `score` is a float; `.nope` raises under the strict Jinja environment.
    config_path = _write_config(tmp_path, predicate="score.nope")

    with pytest.raises(SystemExit) as exc_info:
        _run_shard(monkeypatch, config_path, 0)
    assert exc_info.value.code == 1

    status = _status(tmp_path, 0)
    assert status["status"] == "failed"
    assert "RuntimeError" in status["error"]
    assert "Filter step outputs[1] ('score.nope')" in status["error"]
    assert "raised on every one of its 2 rows" in status["error"]
    assert not (tmp_path / "output" / "shard_0").exists()
