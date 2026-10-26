"""Tests for shard_id / num_shards / batch_size validation in LoadingParams."""

import sys

import pytest

from mmirage import shard_process
from mmirage.config.loading import LoadingParams
from mmirage.config.utils import load_mmirage_config

CONFIG_TEMPLATE = """\
processors:
  - type: batch_api
    provider: openai
    model: gpt-4o-mini
    metadata_output_path: {tmp}/batch_metadata.jsonl

loading_params:
  datasets:
    - path: {tmp}/data.jsonl
      type: JSONL
      output_dir: {tmp}/out
  state_dir: {tmp}/state
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"

processing_params:
  inputs:
    - name: text
      key: text
  outputs:
    - name: answer
      type: batch_api
      prompt: "{{{{ text }}}}"
  output_schema:
    text: "{{{{ answer }}}}"
"""


@pytest.fixture
def config_path(tmp_path):
    (tmp_path / "data.jsonl").write_text('{"text": "hello"}\n')
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_TEMPLATE.format(tmp=tmp_path))
    return str(path)


def test_unresolved_num_shards_raises(monkeypatch):
    monkeypatch.delenv("N", raising=False)
    with pytest.raises(ValueError, match="num_shards"):
        LoadingParams(num_shards="${N}")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_shards": 0},
        {"num_shards": -1},
        {"num_shards": "0"},
        {"shard_id": -1},
        {"shard_id": 4, "num_shards": 4},
        {"shard_id": "4", "num_shards": "4"},
        {"batch_size": 0},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        LoadingParams(**kwargs)


def test_unresolved_shard_id_falls_back_to_zero(monkeypatch):
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    params = LoadingParams(num_shards=4, shard_id="$SLURM_ARRAY_TASK_ID")
    assert params.get_shard_id() == 0
    assert params.is_shard_id_unresolved() is True


def test_shard_id_resolved_from_env(monkeypatch, config_path):
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    params = load_mmirage_config(config_path).loading_params
    assert params.get_shard_id() == 2
    assert params.is_shard_id_unresolved() is False


def test_worker_refuses_unresolved_shard_id(monkeypatch, tmp_path, config_path):
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["mmirage.shard_process", "--config", config_path])

    with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
        shard_process.main()

    assert not list((tmp_path / "state").rglob("status.json"))
