"""Tests for stale "running" shard handling and SIGTERM reporting."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mmirage import cli
from mmirage.cli_utils import status as status_module
from mmirage.cli_utils.status import check_failed_shards
from mmirage.config.config import ExecutionParams
from mmirage.config.loading import LoadingParams
from mmirage.shard_utils import (
    ShardStatus,
    _write_status,
    proc_start_time,
    read_status,
    shard_state_dir,
)


def _cfg(state_root, max_retries=3):
    return SimpleNamespace(
        loading_params=LoadingParams(num_shards=1, state_dir=str(state_root)),
        execution_params=ExecutionParams(
            mode="slurm", account="test", retry=True, max_retries=max_retries
        ),
    )


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _write(state_root, status, retry_count, **identity):
    _write_status(
        shard_state_dir(0, str(state_root)),
        ShardStatus(status=status, retry_count=retry_count, shard_id=0, **identity),
    )


def _local(pid, start_time=None):
    return {"pid": pid, "hostname": socket.gethostname(), "pid_start_time": start_time}


def test_running_shard_with_live_process_stays_running(tmp_path):
    pid = os.getpid()
    _write(tmp_path, "running", 1, **_local(pid, proc_start_time(pid)))

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == []
    assert summary.running == 1


def test_running_shard_with_dead_process_is_failed(tmp_path):
    _write(tmp_path, "running", 1, **_local(_dead_pid()))

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == [0]
    assert summary.running == 0
    assert summary.failed == 1
    status = read_status(shard_state_dir(0, str(tmp_path)))
    assert status.status == "failed"
    assert "stale running state" in status.error


@pytest.mark.skipif(proc_start_time(os.getpid()) is None, reason="needs /proc")
def test_reused_pid_is_detected_by_start_time(tmp_path):
    _write(tmp_path, "running", 1, **_local(os.getpid(), "1"))

    failed, _ = check_failed_shards(_cfg(tmp_path))

    assert failed == [0]


@pytest.mark.parametrize("gone", [True, False])
def test_slurm_shard_uses_squeue(tmp_path, monkeypatch, gone):
    monkeypatch.setattr(status_module, "slurm_job_gone", lambda job_id: gone)
    _write(tmp_path, "running", 1, pid=1, hostname="other-node", slurm_job_id="42")

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == ([0] if gone else [])
    assert summary.running == (0 if gone else 1)


def test_running_shard_without_identity_stays_running(tmp_path):
    _write(tmp_path, "running", 1)

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == []
    assert summary.running == 1


def test_stale_running_shard_over_budget(tmp_path):
    _write(tmp_path, "running", 5, **_local(_dead_pid()))

    failed, summary = check_failed_shards(_cfg(tmp_path, max_retries=3))

    assert failed == []
    assert summary.max_retries_exceeded == 1


def test_slurm_retry_loop_relaunches_stale_running_shard(tmp_path, monkeypatch):
    submitted = []
    dead = _dead_pid()

    def fake_submit(cfg, config_path, shard_ids, collect_stats=False):
        submitted.append(list(shard_ids))
        return len(submitted)

    def fake_wait(job_id, cfg):
        if job_id == 1:
            _write(tmp_path, "running", 1, **_local(dead))
        else:
            _write(tmp_path, "success", job_id)

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


SWALLOWING_CONFIG = """\
processors:
  - type: swallowing

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
      type: swallowing
  output_schema:
    result: "{{{{ result }}}}"
"""

RUN_WITH_SWALLOWING_PROCESSOR = """\
import sys

import swallowing_processor  # noqa: F401  (registers the processor)
from mmirage import shard_process

sys.argv = ["mmirage.shard_process", "--config", sys.argv[1]]
shard_process.main()
"""


def test_sigterm_is_not_swallowed_by_a_processor(tmp_path):
    # The llm and custom processors wrap their work in `except Exception`; SIGTERM
    # must still end the shard as failed instead of becoming a fallback value.
    started = tmp_path / "started"
    (tmp_path / "data.jsonl").write_text(json.dumps({"text": "hi"}) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(SWALLOWING_CONFIG.format(tmp=tmp_path))
    unit_tests_dir = os.path.dirname(os.path.abspath(__file__))
    pythonpath = os.pathsep.join(
        filter(None, [unit_tests_dir, os.environ.get("PYTHONPATH")])
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", RUN_WITH_SWALLOWING_PROCESSOR, str(config)],
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
            "SLURM_ARRAY_TASK_ID": "0",
            "MMIRAGE_TEST_STARTED": str(started),
        },
    )
    try:
        deadline = time.monotonic() + 60
        while not started.exists():
            assert proc.poll() is None, "shard process exited before processing"
            assert time.monotonic() < deadline, "processor never started"
            time.sleep(0.1)

        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=10)
    finally:
        proc.kill()

    assert returncode != 0
    status = read_status(shard_state_dir(0, str(tmp_path / "state")))
    assert status.status == "failed"
    assert "SIGTERM" in status.error
