from datasets import Dataset, DatasetDict, load_from_disk

from mmirage.shard_utils import (
    _drop_empty_splits,
    _save_dataset_atomic,
    _shard_dataset,
)


def _rows(n: int) -> Dataset:
    return Dataset.from_dict({"x": list(range(n))})


def test_shard_past_last_row_is_empty():
    ds = _rows(3)
    assert len(_shard_dataset(ds, num_shards=5, shard_id=4)) == 0


def test_shard_within_range_unchanged():
    ds = _rows(3)
    assert len(_shard_dataset(ds, num_shards=5, shard_id=2)) == 1


def test_shard_dataset_dict_per_split():
    ds = DatasetDict({"small": _rows(3), "large": _rows(10)})
    sharded = _shard_dataset(ds, num_shards=5, shard_id=4)
    assert isinstance(sharded, DatasetDict)
    assert len(sharded["small"]) == 0
    assert len(sharded["large"]) == 2


def test_drop_empty_splits_on_dataset():
    assert _drop_empty_splits(_rows(0)) is None
    kept = _drop_empty_splits(_rows(2))
    assert kept is not None and len(kept) == 2


def test_drop_empty_splits_all_empty_dict():
    ds = DatasetDict({"train": _rows(0), "test": _rows(0)})
    assert _drop_empty_splits(ds) is None


def test_dataset_dict_with_empty_split_saves_and_reloads(tmp_path):
    ds = DatasetDict({"train": _rows(2), "test": _rows(0)})
    kept = _drop_empty_splits(ds)
    assert kept is not None
    assert list(kept.keys()) == ["train"]

    out_dir = str(tmp_path / "shard_0")
    _save_dataset_atomic(kept, out_dir)

    reloaded = load_from_disk(out_dir)
    assert isinstance(reloaded, DatasetDict)
    assert list(reloaded.keys()) == ["train"]
    assert len(reloaded["train"]) == 2
