"""Tests for dataset loading from configs and the loader registry."""

import json

import pytest

from mmirage.core.loader.base import BaseDataLoaderConfig, DataLoaderRegistry
from mmirage.core.loader.jsonl import JSONLDataConfig, JSONLDataLoader
from mmirage.core.loader.local_hf import LocalHFDataLoader
from mmirage.core.loader.utils import load_datasets_from_configs


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_failed_dataset_raises_with_index(tmp_path):
    good = _write_jsonl(tmp_path / "good.jsonl", [{"text": "a"}])
    configs = [
        JSONLDataConfig(
            type="JSONL", output_dir="out0", path=str(tmp_path / "missing.jsonl")
        ),
        JSONLDataConfig(type="JSONL", output_dir="out1", path=good),
    ]

    with pytest.raises(RuntimeError, match="index 0") as exc_info:
        load_datasets_from_configs(configs)
    assert "missing.jsonl" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_loader_returning_none_raises(monkeypatch):
    class NoneLoader:
        def from_config(self, ds_config):
            return None

    monkeypatch.setitem(DataLoaderRegistry._registry, "none_loader", NoneLoader)

    with pytest.raises(RuntimeError, match="index 0"):
        load_datasets_from_configs(
            [BaseDataLoaderConfig(type="none_loader", output_dir="out")]
        )


def test_datasets_loaded_in_config_order(tmp_path):
    first = _write_jsonl(tmp_path / "first.jsonl", [{"text": "first"}])
    second = _write_jsonl(tmp_path / "second.jsonl", [{"text": "second"}])
    configs = [
        JSONLDataConfig(type="JSONL", output_dir="out0", path=first),
        JSONLDataConfig(type="JSONL", output_dir="out1", path=second),
    ]

    datasets = load_datasets_from_configs(configs)

    assert [ds[0]["text"] for ds in datasets] == ["first", "second"]


def test_register_decorator_returns_class():
    assert isinstance(JSONLDataLoader, type)
    assert DataLoaderRegistry.get_processor("JSONL") is JSONLDataLoader
    assert isinstance(LocalHFDataLoader, type)
    assert DataLoaderRegistry.get_processor("loadable") is LocalHFDataLoader
