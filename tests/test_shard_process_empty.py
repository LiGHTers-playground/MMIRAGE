"""In-process runs of shard_process.main() on empty shards and uneven splits."""

import json
import os
import sys
from typing import Optional

import pytest
import yaml
from datasets import Dataset, DatasetDict, load_from_disk

from mmirage import shard_process
from mmirage.core.writer.renderer import TemplateRenderer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_CONFIG = os.path.join(REPO_ROOT, "configs", "config_mock_custom_module.yaml")
MOCK_SCRIPT = os.path.join(REPO_ROOT, "tests", "mock_script", "mock_script.py")


class _ExplodingMapper:
    """Stand-in for MMIRAGEMapper that fails if a model would be loaded."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("MMIRAGEMapper must not be built for an empty shard")


def _write_config(
    tmp_path,
    num_rows: int,
    num_shards: int,
    shard_id: int,
    remove_columns: bool = False,
    dataset_dict: Optional[DatasetDict] = None,
) -> str:
    """Copy the mock custom-module config with all paths rewritten under tmp_path.

    By default the input is a JSONL file of ``num_rows`` rows; pass
    ``dataset_dict`` to save it to disk and load it through the ``loadable`` loader.
    """
    with open(MOCK_CONFIG) as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg["loading_params"]["datasets"][0]
    if dataset_dict is not None:
        data_path = tmp_path / "input"
        dataset_dict.save_to_disk(str(data_path))
        dataset_cfg["type"] = "loadable"
    else:
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for i in range(num_rows):
                f.write(json.dumps({"text": f"row {i}"}) + "\n")

    cfg["processors"][0]["script_path"] = MOCK_SCRIPT
    cfg["loading_params"]["num_shards"] = num_shards
    cfg["loading_params"]["shard_id"] = shard_id
    cfg["loading_params"]["state_dir"] = str(tmp_path / "state")
    dataset_cfg["path"] = str(data_path)
    dataset_cfg["output_dir"] = str(tmp_path / "output")
    cfg["execution_params"]["mode"] = "local"
    if remove_columns:
        cfg["processing_params"]["remove_columns"] = True

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


def test_map_yielding_zero_rows_writes_no_output_folder(tmp_path, monkeypatch):
    # Let the real mapper and custom processor run, but render every batch to
    # 0 rows so the map filters out all input.
    def _render_nothing(self, batch):
        return {key: [] for key in self.output_schema}

    monkeypatch.setattr(TemplateRenderer, "batch_render", _render_nothing)
    config_path = _write_config(
        tmp_path, num_rows=2, num_shards=1, shard_id=0, remove_columns=True
    )
    monkeypatch.setattr(sys, "argv", ["shard_process", "--config", config_path])

    shard_process.main()

    with open(tmp_path / "state" / "shard_0" / "status.json") as f:
        status = json.load(f)
    assert status["status"] == "success"
    assert status["stats"]["rows_processed"] == 2
    assert not (tmp_path / "output" / "shard_0").exists()


def test_dataset_dict_splits_with_different_columns_are_removed_per_split(
    tmp_path, monkeypatch
):
    # Removing the union of all splits' columns would raise on the split
    # that lacks "extra"; each split must be mapped with its own columns.
    dataset_dict = DatasetDict(
        {
            "with_extra": Dataset.from_dict({"text": ["a", "b"], "extra": [1, 2]}),
            "text_only": Dataset.from_dict({"text": ["c"]}),
        }
    )
    config_path = _write_config(
        tmp_path,
        num_rows=0,
        num_shards=1,
        shard_id=0,
        remove_columns=True,
        dataset_dict=dataset_dict,
    )
    monkeypatch.setattr(sys, "argv", ["shard_process", "--config", config_path])

    shard_process.main()

    with open(tmp_path / "state" / "shard_0" / "status.json") as f:
        status = json.load(f)
    assert status["status"] == "success"

    saved = load_from_disk(str(tmp_path / "output" / "shard_0"))
    assert isinstance(saved, DatasetDict)
    assert sorted(saved.keys()) == ["text_only", "with_extra"]
    assert len(saved["with_extra"]) == 2
    assert len(saved["text_only"]) == 1
    for split in saved.values():
        assert sorted(split.column_names) == ["processed", "source"]
