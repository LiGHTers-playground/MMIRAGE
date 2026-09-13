"""Filter steps inside MMIRAGEMapper.rewrite_batch."""

import logging

import pytest
from fake_processor import FakeConfig, FakeLenProcessor, FakeOutputVar

from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.process.processors.filter.config import FilterStep
from mmirage.core.process.variables import InputVar, VariableEnvironment


def _mapper(predicate: str) -> MMIRAGEMapper:
    return MMIRAGEMapper(
        processor_configs=[FakeConfig()],
        input_vars=[InputVar(name="text", key="text")],
        output_vars=[
            FakeOutputVar(name="score"),
            FilterStep(predicate=predicate),
            FakeOutputVar(name="score_again"),
        ],
    )


def _fake(mapper: MMIRAGEMapper) -> FakeLenProcessor:
    proc = mapper.processors["fake_len"]
    assert isinstance(proc, FakeLenProcessor)
    return proc


def test_validate_vars_accepts_a_filter_between_two_steps():
    assert _mapper("score > 3").validate_vars()
    assert not _mapper("nope > 3").validate_vars()


def test_filter_counts_rows_and_later_steps_see_fewer_envs():
    mapper = _mapper("score > 3")
    envs = mapper.rewrite_batch({"text": ["long text", "no", "also long", "x"]})

    assert [e.get("text") for e in envs] == ["long text", "also long"]
    assert all(isinstance(e, VariableEnvironment) for e in envs)
    assert _fake(mapper).batch_sizes == [4, 2]
    assert mapper.rows_seen == {1: 4}
    assert mapper.rows_filtered == {1: 2}
    assert mapper.rows_dropped_by_error == {1: 0}
    assert mapper.total_rows_filtered() == 2
    assert mapper.total_rows_dropped_by_error() == 0


def test_counters_accumulate_across_batches():
    mapper = _mapper("score > 3")
    mapper.rewrite_batch({"text": ["long text", "no"]})
    mapper.rewrite_batch({"text": ["x", "y"]})
    assert mapper.rows_seen == {1: 4}
    assert mapper.rows_filtered == {1: 3}


def test_fully_dropped_batch_short_circuits_later_steps():
    mapper = _mapper("score > 100")
    envs = mapper.rewrite_batch({"text": ["short", "tiny"]})

    assert envs == []
    # The first step ran on 2 rows; the step after the filter never ran.
    assert _fake(mapper).batch_sizes == [2]


def test_erroring_rows_are_counted_and_warned_once(caplog):
    # `score.nope` raises on every row under the strict environment.
    mapper = _mapper("score.nope")
    with caplog.at_level(logging.WARNING, logger="mmirage.core.process.mapper"):
        mapper.rewrite_batch({"text": ["a", "b"]})
        mapper.rewrite_batch({"text": ["c"]})

    assert mapper.rows_dropped_by_error == {1: 3}
    warnings = [r for r in caplog.records if "Filter step outputs[1]" in r.message]
    assert len(warnings) == 1
    assert "'score.nope'" in warnings[0].message


def test_check_filter_errors_raises_only_when_every_row_errored():
    all_bad = _mapper("score.nope")
    all_bad.rewrite_batch({"text": ["a", "b"]})
    with pytest.raises(RuntimeError, match=r"outputs\[1\].*'score.nope'.*2 rows"):
        all_bad.check_filter_errors()

    # `text` is a string; `text.nope` raises, but the `or` keeps long rows valid.
    some_bad = _mapper("score > 3 or text.nope")
    some_bad.rewrite_batch({"text": ["long text", "no"]})
    assert some_bad.rows_dropped_by_error == {1: 1}
    some_bad.check_filter_errors()

    # A mapper whose filter never saw a row has nothing to report.
    _mapper("score.nope").check_filter_errors()
