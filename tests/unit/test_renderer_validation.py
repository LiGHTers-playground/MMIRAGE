"""Tests for static validation of output_schema template variables."""

import json
import sys

import pytest

from mmirage import shard_process
from mmirage.config.config import ProcessingParams
from mmirage.config.utils import load_mmirage_config
from mmirage.core.process.variables import InputVar
from mmirage.core.writer.renderer import TemplateRenderer


def test_typo_variable_reported():
    renderer = TemplateRenderer({"a": "Q: {{ typo_var }}"})
    assert renderer.undeclared_variables({"text"}) == {"typo_var"}


def test_typo_variable_in_nested_schema_reported():
    renderer = TemplateRenderer(
        {
            "conversations": [
                {"role": "user", "content": "{{ text }}"},
                {"role": "assistant", "content": "{{ answr.answer }}"},
            ],
            "meta": {"n": 3},
        }
    )
    assert renderer.undeclared_variables({"text", "answer"}) == {"answr"}


def test_attribute_access_on_known_variable_is_fine():
    renderer = TemplateRenderer({"q": "{{ formatted.question }}"})
    assert renderer.undeclared_variables({"formatted"}) == set()


CUSTOM_CONFIG = """\
processors:
  - type: custom
    script_path: {script}
    function_name: process
    max_workers: 1
    start_method: spawn

loading_params:
  state_dir: {tmp}/state
  datasets:
    - type: JSONL
      path: {tmp}/data.jsonl
      output_dir: {tmp}/out

processing_params:
  inputs:
    - name: text
      key: text
  outputs:
    - name: custom_result
      type: custom
  output_schema:
    source: "{{{{ text }}}}"
    processed: "{{{{ custom_reslut }}}}"
"""


def _write_config(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("def process(row):\n    return row['text']\n")
    (tmp_path / "data.jsonl").write_text(json.dumps({"text": "hi"}) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(CUSTOM_CONFIG.format(script=script, tmp=tmp_path))
    return config


def test_processing_params_rejects_unknown_schema_variable():
    with pytest.raises(ValueError, match=r"typo_var.*Available variables: \['text'\]"):
        ProcessingParams(
            inputs=[InputVar(name="text", key="text")],
            outputs=[],
            output_schema={"a": "{{ text }}", "b": "Q: {{ typo_var }}"},
        )


def test_config_load_rejects_schema_typo(tmp_path):
    with pytest.raises(ValueError, match="custom_reslut") as exc_info:
        load_mmirage_config(str(_write_config(tmp_path)))

    assert "processed" in str(exc_info.value)
    assert not (tmp_path / "state").exists()


def test_shard_fails_before_writing_state_on_schema_typo(tmp_path, monkeypatch):
    # Rejected at config load: not recorded as a shard failure, so not retried.
    config = _write_config(tmp_path)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(sys, "argv", ["mmirage.shard_process", "--config", str(config)])

    with pytest.raises(ValueError, match="custom_reslut"):
        shard_process.main()

    assert not (tmp_path / "state").exists()
