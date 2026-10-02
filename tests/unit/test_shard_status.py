"""Tests for stale "running" shard handling and SIGTERM reporting."""

import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mmirage import cli
from mmirage.cli_utils.status import check_failed_shards
from mmirage.config.config import ExecutionParams
from mmirage.config.loading import LoadingParams
from mmirage.shard_utils import ShardStatus, _write_status, read_status, shard_state_dir


def _cfg(state_root, max_retries=3):
    return SimpleNamespace(
        loading_params=LoadingParams(num_shards=1, state_dir=str(state_root)),
        execution_params=ExecutionParams(
            mode="slurm", account="test", retry=True, max_retries=max_retries
        ),
    )


def _write(state_root, status, retry_count):
    _write_status(
        shard_state_dir(0, str(state_root)),
        ShardStatus(status=status, retry_count=retry_count, shard_id=0),
    )


def test_running_shard_not_failed_by_default(tmp_path):
    _write(tmp_path, "running", retry_count=1)

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == []
    assert summary.running == 1


def test_running_shard_treated_as_failed(tmp_path):
    _write(tmp_path, "running", retry_count=1)

    failed, summary = check_failed_shards(_cfg(tmp_path), treat_running_as_failed=True)

    assert failed == [0]
    assert summary.running == 0
    assert summary.failed == 1
    status = read_status(shard_state_dir(0, str(tmp_path)))
    assert status.status == "failed"
    assert "stale running state" in status.error


def test_stale_running_shard_over_budget(tmp_path):
    _write(tmp_path, "running", retry_count=5)

    failed, summary = check_failed_shards(
        _cfg(tmp_path, max_retries=3), treat_running_as_failed=True
    )

    assert failed == []
    assert summary.max_retries_exceeded == 1


def test_slurm_retry_loop_relaunches_stale_running_shard(tmp_path, monkeypatch):
    submitted = []

    def fake_submit(cfg, config_path, shard_ids, collect_stats=False):
        submitted.append(list(shard_ids))
        return len(submitted)

    def fake_wait(job_id, cfg):
        _write(tmp_path, "running" if job_id == 1 else "success", retry_count=job_id)

    monkeypatch.setattr(cli, "submit_slurm_job", fake_submit)
    monkeypatch.setattr(cli, "wait_for_slurm_job", fake_wait)

    assert cli.launch_pipeline(_cfg(tmp_path), "cfg.yaml") == 0
    assert submitted == [[], [0]]


SLEEPY_SCRIPT = """\
import pathlib, time

def process(row):
    pathlib.Path({started!r}).touch()
    time.sleep(60)
    return "done"
"""

SLEEPY_CONFIG = """\
processors:
  - type: custom
    script_path: {script}
    function_name: process
    max_workers: 1
    start_method: {start_method}
    timeout_ms: 120000

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
    - name: result
      type: custom
  output_schema:
    result: "{{{{ result }}}}"
"""


@pytest.mark.parametrize("start_method", ["spawn", "fork"])
def test_sigterm_marks_shard_failed(tmp_path, start_method):
    started = tmp_path / "started"
    script = tmp_path / "sleepy.py"
    script.write_text(SLEEPY_SCRIPT.format(started=str(started)))
    (tmp_path / "data.jsonl").write_text(json.dumps({"text": "hi"}) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        SLEEPY_CONFIG.format(script=script, start_method=start_method, tmp=tmp_path)
    )

    proc = subprocess.Popen(
        [sys.executable, "-m", "mmirage.shard_process", "--config", str(config)],
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
    )
    try:
        deadline = time.monotonic() + 60
        while not started.exists():
            assert proc.poll() is None, "shard process exited before processing"
            assert time.monotonic() < deadline, "worker never started"
            time.sleep(0.1)

        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=30)
    finally:
        proc.kill()

    assert returncode != 0
    status = read_status(shard_state_dir(0, str(tmp_path / "state")))
    assert status.status == "failed"
    assert "SIGTERM" in status.error
