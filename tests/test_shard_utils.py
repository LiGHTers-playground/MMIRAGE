from datasets import Dataset, DatasetDict

from mmirage.shard_utils import _shard_dataset


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
