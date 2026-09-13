"""In-process runs of shard_process.main() on shards with no input rows."""

import json
import os
import sys

import pytest
import yaml

from mmirage import shard_process

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_CONFIG = os.path.join(REPO_ROOT, "configs", "config_mock_custom_module.yaml")
MOCK_SCRIPT = os.path.join(REPO_ROOT, "tests", "mock_script", "mock_script.py")


class _ExplodingMapper:
    """Stand-in for MMIRAGEMapper that fails if a model would be loaded."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("MMIRAGEMapper must not be built for an empty shard")


def _write_config(tmp_path, num_rows: int, num_shards: int, shard_id: int) -> str:
    """Copy the mock custom-module config with all paths rewritten under tmp_path."""
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for i in range(num_rows):
            f.write(json.dumps({"text": f"row {i}"}) + "\n")

    with open(MOCK_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["processors"][0]["script_path"] = MOCK_SCRIPT
    cfg["loading_params"]["num_shards"] = num_shards
    cfg["loading_params"]["shard_id"] = shard_id
    cfg["loading_params"]["state_dir"] = str(tmp_path / "state")
    cfg["loading_params"]["datasets"][0]["path"] = str(data_path)
    cfg["loading_params"]["datasets"][0]["output_dir"] = str(tmp_path / "output")
    cfg["execution_params"]["mode"] = "local"

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(config_path)


def test_empty_shard_marks_success_without_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(shard_process, "MMIRAGEMapper", _ExplodingMapper)
    config_path = _write_config(tmp_path, num_rows=1, num_shards=4, shard_id=3)
    monkeypatch.setattr(sys, "argv", ["shard_process", "--config", config_path])

    shard_process.main()

    with open(tmp_path / "state" / "shard_3" / "status.json") as f:
        status = json.load(f)
    assert status["status"] == "success"
    assert status["stats"]["rows_processed"] == 0
    assert status["stats"]["rows_written"] == 0
    assert not (tmp_path / "output" / "shard_3").exists()


def test_non_empty_shard_still_builds_mapper(tmp_path, monkeypatch):
    monkeypatch.setattr(shard_process, "MMIRAGEMapper", _ExplodingMapper)
    config_path = _write_config(tmp_path, num_rows=1, num_shards=4, shard_id=0)
    monkeypatch.setattr(sys, "argv", ["shard_process", "--config", config_path])

    # main() reports failures through sys.exit(1) after marking the shard failed.
    with pytest.raises(SystemExit):
        shard_process.main()

    with open(tmp_path / "state" / "shard_0" / "status.json") as f:
        status = json.load(f)
    assert status["status"] == "failed"
