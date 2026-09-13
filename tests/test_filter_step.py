import os
from dataclasses import fields

import pytest
import yaml

from mmirage.config.utils import load_mmirage_config
from mmirage.core.process.processors.batch_api.config import BatchApiOutputVar
from mmirage.core.process.processors.filter.config import FilterStep
from mmirage.core.process.processors.llm.config import LLMOutputVar
from mmirage.core.process.variables import InputVar, OutputVar, VariableEnvironment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_CONFIG = os.path.join(REPO_ROOT, "configs", "config_mock_custom_module.yaml")


def test_output_vars_are_available_at_map_time_by_default():
    assert OutputVar.available_at_map_time is True
    assert LLMOutputVar.available_at_map_time is True


def test_batch_api_output_is_deferred_until_merge():
    assert BatchApiOutputVar.available_at_map_time is False
    # A ClassVar, so it is not a dataclass field and dacite never sees it.
    assert "available_at_map_time" not in {f.name for f in fields(BatchApiOutputVar)}


# --- FilterStep construction ---------------------------------------------------


def test_predicate_compiles_and_surrounding_braces_are_stripped():
    assert FilterStep(predicate="score > 0.5").predicate == "score > 0.5"
    assert FilterStep(predicate="{{ score > 0.5 }}").predicate == "score > 0.5"
    assert FilterStep(predicate="  {{score > 0.5}}  ").predicate == "score > 0.5"


def test_empty_predicate_is_rejected():
    with pytest.raises(ValueError, match="needs a `predicate`"):
        FilterStep(predicate="")
    with pytest.raises(ValueError, match="needs a `predicate`"):
        FilterStep(predicate="{{ }}")


def test_syntax_error_message_quotes_the_predicate():
    with pytest.raises(ValueError, match=r"'score >'"):
        FilterStep(predicate="score >")


# --- name resolution ------------------------------------------------------------


def _declared_vars():
    return [
        InputVar(name="text", key="text"),
        LLMOutputVar(name="verdict", type="llm"),
        BatchApiOutputVar(name="later", type="batch_api"),
    ]


def test_missing_names_are_those_not_declared_before_the_step():
    step = FilterStep(predicate="verdict.keep and text | length > 3 and typo")
    assert step.missing_names(_declared_vars()) == {"typo"}
    assert FilterStep(predicate="verdict.keep").missing_names(_declared_vars()) == set()


def test_deferred_names_are_batch_api_outputs():
    step = FilterStep(predicate="later.keep and verdict.keep")
    assert step.deferred_names(_declared_vars()) == {"later"}
    assert FilterStep(predicate="text").deferred_names(_declared_vars()) == set()


def test_is_computable_requires_no_missing_and_no_deferred_names():
    declared = _declared_vars()
    assert FilterStep(predicate="verdict.keep").is_computable(declared)
    assert not FilterStep(predicate="typo").is_computable(declared)
    assert not FilterStep(predicate="later.keep").is_computable(declared)


# --- apply ----------------------------------------------------------------------


def _envs(*payloads):
    return [VariableEnvironment(dict(p)) for p in payloads]


def test_apply_keeps_truthy_and_counts_falsy():
    step = FilterStep(predicate="verdict.keep")
    result = step.apply(
        _envs({"verdict": {"keep": True}}, {"verdict": {"keep": False}})
    )
    assert [e.get("verdict") for e in result.kept] == [{"keep": True}]
    assert result.n_filtered == 1
    assert result.n_errored == 0
    assert result.first_error is None


def test_apply_counts_raising_rows_as_errors_and_records_the_first():
    step = FilterStep(predicate="verdict.keep")
    result = step.apply(
        _envs(
            {"verdict": {"keep": True}},
            {"verdict": {}},  # missing key
            {"verdict": None},  # attribute of None
            {"verdict": {"kep": True}},  # typo, and the attribute is missing
        )
    )
    assert len(result.kept) == 1
    assert result.n_filtered == 0
    assert result.n_errored == 3
    assert result.first_error is not None
    position, exc = result.first_error
    assert position == 1
    assert isinstance(exc, Exception)


def test_apply_counts_none_in_a_comparison_as_an_error():
    result = FilterStep(predicate="score >= 3").apply(
        _envs({"score": 5}, {"score": None})
    )
    assert len(result.kept) == 1
    assert result.n_errored == 1
    assert result.first_error is not None
    assert isinstance(result.first_error[1], TypeError)


def test_apply_supports_jinja_filters():
    result = FilterStep(predicate="text | length > 3").apply(
        _envs({"text": "long enough"}, {"text": "no"})
    )
    assert [e.get("text") for e in result.kept] == ["long enough"]
    assert result.n_filtered == 1


# --- config load ----------------------------------------------------------------


def test_filter_under_processors_is_rejected_at_config_load(tmp_path):
    with open(MOCK_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["processors"].append({"type": "filter"})
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError, match="not a processor"):
        load_mmirage_config(str(config_path))
