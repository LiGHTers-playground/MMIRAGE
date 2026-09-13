from types import SimpleNamespace

from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult


def test_extract_unique_provider_batches_handles_malformed_and_duplicates(tmp_path):
    from mmirage.core.process.batch.status_checker import (
        _read_metadata_records,
        extract_unique_provider_batches,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                '{"provider":"openai","provider_batch_id":"batch_1"}',
                '{"provider":"openai","provider_batch_id":"batch_1"}',
                "not-json",
                '{"provider":"openai"}',
                '{"provider":"openai","provider_batch_id":"batch_2"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    pairs = extract_unique_provider_batches(
        _read_metadata_records([str(metadata_path)])
    )

    assert pairs == [("openai", "batch_1"), ("openai", "batch_2")]


def test_run_status_checker_prints_summary_with_factory_dispatch(tmp_path, monkeypatch):
    from unittest.mock import patch

    from mmirage.core.process.batch.status_checker import (
        _read_metadata_records,
        run_status_checker,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                '{"provider":"openai","provider_batch_id":"batch_1"}',
                '{"provider":"openai","provider_batch_id":"batch_2"}',
                '{"provider":"openai","provider_batch_id":"batch_1"}',
            ]
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def check_batch_status(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            status = "completed" if provider_batch_id == "batch_1" else "in_progress"
            return BatchSubmissionResult(
                provider_batch_id=provider_batch_id,
                status=status,
                raw_response={"id": provider_batch_id, "status": status},
            )

    fake_adapter = FakeAdapter()

    monkeypatch.setattr(
        "mmirage.core.process.batch.status_checker.BatchAdapterRegistry.create",
        lambda config: fake_adapter,
    )

    config_map = {
        "openai": OpenAIBatchConfig(),
    }
    records = _read_metadata_records([str(metadata_path)])

    with patch("mmirage.core.process.batch.status_checker.logger") as mock_logger:
        results = run_status_checker(
            metadata_records=records,
            provider_configs=config_map,
        )

    assert [(r.provider_batch_id, r.status) for r in results] == [
        ("batch_1", "completed"),
        ("batch_2", "in_progress"),
    ]
    assert fake_adapter.calls == [
        ("batch_1", "openai"),
        ("batch_2", "openai"),
    ]

    # Verify logger.info was called with expected messages
    logger_calls = [call[0][0] for call in mock_logger.info.call_args_list]
    assert any(
        "Batch batch_1 (openai): completed" in str(call) for call in logger_calls
    )
    assert any(
        "Batch batch_2 (openai): in_progress" in str(call) for call in logger_calls
    )


def test_status_checker_main_uses_config_and_runs(tmp_path, monkeypatch):
    from mmirage.core.process.batch import status_checker

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        '{"provider":"openai","provider_batch_id":"batch_1"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[SimpleNamespace(provider_config=OpenAIBatchConfig())]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    called = {}

    def _fake_run_status_checker(metadata_records, provider_configs, output=None):
        called["metadata_records"] = metadata_records
        called["provider_configs"] = provider_configs
        return [BatchSubmissionResult(provider_batch_id="batch_1", status="completed")]

    monkeypatch.setattr(
        "mmirage.core.process.batch.status_checker.run_status_checker",
        _fake_run_status_checker,
    )

    rc = status_checker.main(
        [
            "--metadata-path",
            str(metadata_path),
            "--config",
            str(config_path),
        ]
    )

    assert rc == 0
    assert len(called["metadata_records"]) == 1
    assert called["metadata_records"][0].provider == "openai"
    assert "openai" in called["provider_configs"]


def test_status_checker_main_returns_error_when_metadata_provider_missing_in_config(
    tmp_path, monkeypatch
):
    from unittest.mock import patch

    from mmirage.core.process.batch import status_checker

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        '{"provider":"mistral","provider_batch_id":"batch_m1"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    # Config intentionally only defines openai, not mistral.
    cfg = SimpleNamespace(
        processors=[SimpleNamespace(provider_config=OpenAIBatchConfig())]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    with patch("mmirage.core.process.batch.status_checker.logger") as mock_logger:
        rc = status_checker.main(
            [
                "--metadata-path",
                str(metadata_path),
                "--config",
                str(config_path),
            ]
        )

    assert rc == 1
    assert mock_logger.error.called or mock_logger.exception.called


def test_status_checker_main_returns_error_when_api_key_missing(tmp_path, monkeypatch):
    from unittest.mock import patch

    from mmirage.core.process.batch import status_checker

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        '{"provider":"openai","provider_batch_id":"batch_1"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[SimpleNamespace(provider_config=OpenAIBatchConfig())]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("mmirage.core.process.batch.status_checker.logger") as mock_logger:
        rc = status_checker.main(
            [
                "--metadata-path",
                str(metadata_path),
                "--config",
                str(config_path),
            ]
        )

    assert rc == 1
    assert mock_logger.error.called or mock_logger.exception.called


def test_status_checker_main_uses_config_metadata_path_when_missing_cli_arg(
    tmp_path, monkeypatch
):
    from mmirage.core.process.batch import status_checker

    metadata_base = tmp_path / "batch_metadata.jsonl"
    metadata_path = tmp_path / "batch_metadata.text.abc123.jsonl"
    metadata_path.write_text(
        '{"provider":"openai","provider_batch_id":"batch_1"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(
                provider_config=OpenAIBatchConfig(
                    metadata_output_path=str(metadata_base)
                )
            )
        ]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    called = {}

    def _fake_run_status_checker(metadata_records, provider_configs, output=None):
        called["metadata_records"] = metadata_records
        called["provider_configs"] = provider_configs
        return [BatchSubmissionResult(provider_batch_id="batch_1", status="completed")]

    monkeypatch.setattr(
        "mmirage.core.process.batch.status_checker.run_status_checker",
        _fake_run_status_checker,
    )

    rc = status_checker.main(["--config", str(config_path)])

    assert rc == 0
    assert len(called["metadata_records"]) == 1
    assert called["metadata_records"][0].provider == "openai"
    assert "openai" in called["provider_configs"]


def test_status_checker_main_returns_error_when_config_metadata_paths_missing(
    tmp_path, monkeypatch
):
    from unittest.mock import patch

    from mmirage.core.process.batch import status_checker

    metadata_base = tmp_path / "batch_metadata.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(
                provider_config=OpenAIBatchConfig(
                    metadata_output_path=str(metadata_base)
                )
            )
        ]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    with patch("mmirage.core.process.batch.status_checker.logger") as mock_logger:
        rc = status_checker.main(["--config", str(config_path)])

    assert rc == 1
    assert mock_logger.exception.called
