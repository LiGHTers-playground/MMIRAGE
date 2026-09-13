"""Real ``Dataset.map`` through ``_map_split`` with a filter step."""

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from fake_processor import FakeConfig, FakeOutputVar
from PIL import Image as PILImage

from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.process.processors.filter.config import FilterStep
from mmirage.core.process.variables import InputVar
from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.shard_process import _map_split, rewrite_batch

OUTPUT_SCHEMA = {"source": "{{ text }}", "score": "{{ score }}"}
TEXTS = ["long text", "no", "also long", "x"]


def _mapper(predicate: str = "score > 3") -> MMIRAGEMapper:
    return MMIRAGEMapper(
        processor_configs=[FakeConfig()],
        input_vars=[InputVar(name="text", key="text")],
        output_vars=[FakeOutputVar(name="score"), FilterStep(predicate=predicate)],
    )


def _run(split_ds: Dataset, remove_columns: bool, batch_size: int = 500) -> Dataset:
    return _map_split(
        split_ds,
        mapper=_mapper(),
        renderer=TemplateRenderer(OUTPUT_SCHEMA),
        image_base_path=None,
        batch_size=batch_size,
        remove_columns=remove_columns,
        desc="test",
    )


def test_filter_with_remove_columns_keeps_only_output_columns():
    out = _run(Dataset.from_dict({"text": TEXTS, "extra": [1, 2, 3, 4]}), True)

    assert out.column_names == ["source", "score"]
    assert out["source"] == ["long text", "also long"]
    assert out["score"] == ["9", "9"]


def test_filter_without_remove_columns_joins_kept_input_rows_back():
    out = _run(Dataset.from_dict({"text": TEXTS, "extra": [1, 2, 3, 4]}), False)

    assert out.column_names == ["text", "extra", "source", "score"]
    assert out["text"] == ["long text", "also long"]
    assert out["extra"] == [1, 3]
    assert out["source"] == ["long text", "also long"]


def test_input_column_named_like_an_output_is_replaced_not_duplicated():
    ds = Dataset.from_dict({"text": TEXTS, "score": ["old"] * 4})
    out = _run(ds, False)

    assert sorted(out.column_names) == ["score", "source", "text"]
    assert out["score"] == ["9", "9"]


def test_class_label_and_image_features_survive_the_join(tmp_path):
    paths = []
    for i, color in enumerate(["red", "green", "blue", "white"]):
        path = tmp_path / f"{i}.png"
        PILImage.new("RGB", (2, 2), color).save(path)
        paths.append(str(path))
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["neg", "pos"]),
            "img": Image(),
        }
    )
    ds = Dataset.from_dict(
        {"text": TEXTS, "label": [0, 1, 1, 0], "img": paths}, features=features
    )

    out = _run(ds, False)

    assert out.features["label"] == features["label"]
    assert out.features["img"] == features["img"]
    # select() is lazy; materialise so .data holds only the two rows.
    expected = ds.select([0, 2]).flatten_indices()
    assert (
        out.data.column("label").to_pylist()
        == expected.data.column("label").to_pylist()
    )
    assert out.data.column("img").to_pylist() == expected.data.column("img").to_pylist()


def test_later_batch_fully_dropped_with_remove_columns():
    # With batch_size 2 the second batch is entirely short, so the map has to
    # return a 0-row batch after already having produced rows.
    ds = Dataset.from_dict({"text": ["long text", "also long", "no", "x"]})
    out = _run(ds, True, batch_size=2)

    assert out["source"] == ["long text", "also long"]
    assert len(out) == 2


def test_dataset_dict_selects_the_right_rows_per_split():
    dd = DatasetDict(
        {
            "a": Dataset.from_dict({"text": ["no", "long text", "x"], "n": [0, 1, 2]}),
            "b": Dataset.from_dict({"text": ["also long", "y"], "n": [10, 11]}),
        }
    )
    mapper = _mapper()
    renderer = TemplateRenderer(OUTPUT_SCHEMA)
    out = DatasetDict(
        {
            split: _map_split(
                split_ds,
                mapper=mapper,
                renderer=renderer,
                image_base_path=None,
                batch_size=500,
                remove_columns=False,
                desc=split,
            )
            for split, split_ds in dd.items()
        }
    )

    assert out["a"]["n"] == [1]
    assert out["a"]["text"] == ["long text"]
    assert out["b"]["n"] == [10]
    assert out["b"]["source"] == ["also long"]
    assert mapper.rows_seen == {1: 5}
    assert mapper.rows_filtered == {1: 3}


def test_no_filter_path_is_unchanged():
    mapper = MMIRAGEMapper(
        processor_configs=[FakeConfig()],
        input_vars=[InputVar(name="text", key="text")],
        output_vars=[FakeOutputVar(name="score")],
    )
    renderer = TemplateRenderer(OUTPUT_SCHEMA)
    ds = Dataset.from_dict({"text": TEXTS, "extra": [1, 2, 3, 4]})
    out = _map_split(
        ds,
        mapper=mapper,
        renderer=renderer,
        image_base_path=None,
        batch_size=500,
        remove_columns=False,
        desc="test",
    )
    assert out.column_names == ["text", "extra", "source", "score"]
    assert len(out) == 4

    # Same rows as a plain Dataset.map with the pre-filter kwargs.
    direct = ds.map(
        rewrite_batch,
        batched=True,
        batch_size=500,
        load_from_cache_file=False,
        fn_kwargs={"mapper": mapper, "renderer": renderer, "image_base_path": None},
        remove_columns=[],
    )
    assert out.to_list() == direct.to_list()
