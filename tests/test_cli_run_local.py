"""Tests for the local shard runner in the CLI."""

from contextlib import contextmanager
from types import SimpleNamespace

from mmirage import cli


def test_run_local_forwards_export_prompts_through_shared_server(monkeypatch, tmp_path):
    """A dry run must still export prompts when the shared SGLang server is started."""
    export_path = str(tmp_path / "prompts.jsonl")
    commands = []
    server_entries = []

    @contextmanager
    def fake_shared_sglang_server(config):
        server_entries.append(config)
        monkeypatch.setenv(cli.MMIRAGE_SGLANG_BASE_URL, "http://127.0.0.1:30010/v1")
        yield

    def fake_run(command, env, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv(cli.MMIRAGE_SGLANG_BASE_URL, raising=False)
    monkeypatch.setattr(cli, "load_mmirage_config", lambda path: object())
    monkeypatch.setattr(cli, "get_sglang_server_config", lambda cfg: object())
    monkeypatch.setattr(cli, "shared_sglang_server", fake_shared_sglang_server)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    returncode = cli.run_local("config.yaml", export_prompts_path=export_path)

    assert returncode == 0
    assert len(server_entries) == 1
    assert len(commands) == 1
    command = commands[0]
    assert "--export-prompts" in command
    assert command[command.index("--export-prompts") + 1] == export_path
