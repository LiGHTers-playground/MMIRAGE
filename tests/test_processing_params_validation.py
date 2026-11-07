"""Output variables are validated in order when the config loads."""

import glob
import os

import pytest
import yaml

from mmirage.config.utils import load_mmirage_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
MOCK_CUSTOM = os.path.join(CONFIGS_DIR, "config_mock_custom_module.yaml")
MOCK_OPENAI_BATCH = os.path.join(CONFIGS_DIR, "config_mock_openai_batch.yaml")
MOCK_FILTER = os.path.join(CONFIGS_DIR, "config_mock_filter.yaml")


@pytest.fixture(autouse=True)
def dummy_provider_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")


def _load_with_extra_output(tmp_path, base_config: str, output: dict):
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg["processing_params"]["outputs"].append(output)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return load_mmirage_config(str(config_path))


def test_unknown_name_is_rejected_at_load_with_the_name_in_the_message(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"outputs\[1\] \(filter step\) references variables not "
        r"declared before it: \['nope'\]",
    ):
        _load_with_extra_output(
            tmp_path, MOCK_CUSTOM, {"type": "filter", "predicate": "nope > 1"}
        )


def test_batch_api_output_is_rejected_with_its_own_message(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"outputs\[1\]: 'formatted_answer' is a batch_api output, "
        r"only available after `mmirage merge`",
    ):
        _load_with_extra_output(
            tmp_path,
            MOCK_OPENAI_BATCH,
            {"type": "filter", "predicate": "formatted_answer.question | length > 0"},
        )


def test_uncomputable_non_filter_output_is_rejected_with_the_context(tmp_path):
    with open(MOCK_CUSTOM) as f:
        cfg = yaml.safe_load(f)
    cfg["processors"] = [{"type": "llm", "server_args": {"model_path": "m"}}]
    cfg["processing_params"]["outputs"] = [
        {"name": "answer", "type": "llm", "prompt": "{{ missing }}"}
    ]
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(
        ValueError, match=r"outputs\[0\] 'answer' is not computable given \['text'\]"
    ):
        load_mmirage_config(str(config_path))


def test_valid_filter_after_its_inputs_loads(tmp_path):
    cfg = _load_with_extra_output(
        tmp_path,
        MOCK_CUSTOM,
        {"type": "filter", "predicate": "custom_result | length > 0 and text"},
    )
    assert [o.type for o in cfg.processing_params.outputs] == ["custom", "filter"]


def test_step_after_a_filter_resolves_against_the_pre_filter_context(tmp_path):
    # A filter declares no variable of its own, so the step after it still sees
    # exactly the inputs and the outputs declared before the filter.
    with open(MOCK_CUSTOM) as f:
        cfg = yaml.safe_load(f)
    cfg["processors"].append({"type": "llm", "server_args": {"model_path": "m"}})
    cfg["processing_params"]["outputs"] += [
        {"type": "filter", "predicate": "custom_result | length > 0"},
        {"name": "answer", "type": "llm", "prompt": "{{ custom_result }}"},
    ]
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    outputs = load_mmirage_config(str(config_path)).processing_params.outputs
    assert [o.type for o in outputs] == ["custom", "filter", "llm"]


def test_context_in_the_message_does_not_include_the_filter_step(tmp_path):
    # A filter declares no variable, so it must not show up as an empty name in
    # the context a later step is reported against.
    with open(MOCK_CUSTOM) as f:
        cfg = yaml.safe_load(f)
    cfg["processors"].append({"type": "llm", "server_args": {"model_path": "m"}})
    cfg["processing_params"]["outputs"] += [
        {"type": "filter", "predicate": "custom_result | length > 0"},
        {"name": "answer", "type": "llm", "prompt": "{{ missing }}"},
    ]
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(
        ValueError,
        match=r"outputs\[2\] 'answer' is not computable given "
        r"\['text', 'custom_result'\]$",
    ):
        load_mmirage_config(str(config_path))


def test_mock_filter_config_declares_a_scorer_then_a_filter():
    cfg = load_mmirage_config(MOCK_FILTER)
    outputs = cfg.processing_params.outputs

    assert [o.type for o in outputs] == ["custom", "filter"]
    assert outputs[0].name == "score"
    assert outputs[1].predicate == "score > 33"
    assert cfg.processing_params.remove_columns is True
    assert cfg.processors[0].function_name == "score_text"


@pytest.mark.parametrize(
    "config_path", sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.yaml")))
)
def test_shipped_configs_still_load(config_path):
    load_mmirage_config(config_path)
