"""Tests for static validation of output_schema template variables."""

import json
import sys

import pytest

from mmirage import shard_process
from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.shard_utils import read_status, shard_state_dir


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


def test_shard_fails_before_processors_start_on_schema_typo(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("def process(row):\n    return row['text']\n")
    (tmp_path / "data.jsonl").write_text(json.dumps({"text": "hi"}) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(CUSTOM_CONFIG.format(script=script, tmp=tmp_path))

    class MapperMustNotBeBuilt(shard_process.MMIRAGEMapper):
        def __init__(self, *args, **kwargs):
            raise AssertionError("processors were started before schema validation")

    monkeypatch.setattr(shard_process, "MMIRAGEMapper", MapperMustNotBeBuilt)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(sys, "argv", ["mmirage.shard_process", "--config", str(config)])

    with pytest.raises(SystemExit):
        shard_process.main()

    status = read_status(shard_state_dir(0, str(tmp_path / "state")))
    assert status.status == "failed"
    assert "custom_reslut" in status.error
    assert "processed" in status.error
