"""Tests for stale "running" shard handling."""

import os
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
    proc_stat,
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


ME = os.getpid()
DEAD = _dead_pid()
CASES = {
    "live pid": (_local(ME, proc_start_time(ME)), False),
    "dead pid": (_local(DEAD), True),
    "no identity": ({}, False),
    "slurm job gone": ({"hostname": "node", "slurm_job_id": "gone"}, True),
    "slurm job queued": ({"hostname": "node", "slurm_job_id": "queued"}, False),
}
if proc_start_time(ME) is not None:
    CASES["reused pid"] = (_local(ME, "1"), True)


@pytest.mark.parametrize("identity,dead", CASES.values(), ids=CASES.keys())
def test_running_shard_liveness(tmp_path, monkeypatch, identity, dead):
    monkeypatch.setattr(status_module, "slurm_job_gone", lambda job: job == "gone")
    _write(tmp_path, "running", 1, **identity)

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == ([0] if dead else [])
    assert summary.running == (0 if dead else 1)
    status = read_status(shard_state_dir(0, str(tmp_path)))
    assert (status.status == "failed") == dead


@pytest.mark.skipif(proc_start_time(ME) is None, reason="needs /proc")
def test_zombie_process_is_dead(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])  # never reaped below
    try:
        while (stat := proc_stat(child.pid)) is not None and stat[0] != "Z":
            time.sleep(0.01)
        _write(tmp_path, "running", 1, **_local(child.pid, proc_start_time(child.pid)))

        failed, _ = check_failed_shards(_cfg(tmp_path))
    finally:
        child.wait()

    assert failed == [0]


def test_success_written_during_liveness_check_is_kept(tmp_path, monkeypatch):
    _write(tmp_path, "running", 1, **_local(_dead_pid()))

    def finish_then_die(st):
        _write(tmp_path, "success", 1)
        return False

    monkeypatch.setattr(status_module, "shard_process_alive", finish_then_die)

    failed, summary = check_failed_shards(_cfg(tmp_path))

    assert failed == []
    assert summary.successful == 1
    assert read_status(shard_state_dir(0, str(tmp_path))).status == "success"


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
