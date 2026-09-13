import json
from types import SimpleNamespace

from mmirage.config.openai_batch import OpenAIBatchConfig


def test_metadata_record_reads_shard_id_and_defaults_it_for_old_receipts():
    from mmirage.core.process.batch.metadata_utils import BatchMetadataRecord

    old_receipt = {
        "provider": "openai",
        "provider_batch_id": "batch_1",
        "custom_id_to_source_index": {"c1": 0},
    }
    assert BatchMetadataRecord.from_mapping(old_receipt).shard_id == 0

    new_receipt = {**old_receipt, "shard_id": 2}
    assert BatchMetadataRecord.from_mapping(new_receipt).shard_id == 2


def test_collect_and_merge_reconstructs_rows_deterministically(
    tmp_path, monkeypatch, caplog
):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_1",
                        "custom_id_to_source_index": {"c1": 2, "c2": 0},
                    }
                ),
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_1",
                        "custom_id_to_source_index": {"c1": 2, "c2": 0},
                    }
                ),
                "malformed-line",
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_2",
                        "custom_id_to_source_index": {"c3": 1},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.jsonl"

    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def retrieve_results(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            if provider_batch_id == "batch_1":
                return [
                    {
                        "custom_id": "c1",
                        "generated_text": '{"question":"q2","answer":"a2"}',
                    },
                    {
                        "custom_id": "c2",
                        "generated_text": '{"question":"q0","answer":"a0"}',
                    },
                ]
            return [
                {
                    "custom_id": "c3",
                    "generated_text": '{"question":"q1","answer":"a1"}',
                }
            ]

    fake_adapter = FakeAdapter()
    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: fake_adapter,
    )

    provider_configs = {"openai": OpenAIBatchConfig()}
    records = _read_metadata_records(str(metadata_path))
    assert "Skipping malformed metadata JSON line" in caplog.text
    rows = collect_and_merge(
        records=records,
        provider_configs=provider_configs,
        output_path=str(output_path),
    )

    assert [r["source_index"] for r in rows] == [0, 1, 2]
    assert [r["custom_id"] for r in rows] == ["c2", "c3", "c1"]
    assert [r["conversations"][0]["content"] for r in rows] == ["q0", "q1", "q2"]
    assert [r["conversations"][1]["content"] for r in rows] == ["a0", "a1", "a2"]
    assert fake_adapter.calls == [("batch_1", "openai"), ("batch_2", "openai")]

    written = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["source_index"] for r in written] == [0, 1, 2]
    assert [r["conversations"][0]["content"] for r in written] == ["q0", "q1", "q2"]


def test_collect_and_merge_carries_provider_usage(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_1",
                "custom_id_to_source_index": {"c1": 0, "c2": 1},
            }
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def retrieve_results(self, provider_batch_id, config):
            return [
                {
                    "custom_id": "c1",
                    "generated_text": "first",
                    "input_tokens": 67,
                    "output_tokens": 22,
                },
                {"custom_id": "c2", "generated_text": "second"},
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: FakeAdapter(),
    )

    output_path = tmp_path / "merged.jsonl"
    rows = collect_and_merge(
        records=_read_metadata_records(str(metadata_path)),
        provider_configs={"openai": OpenAIBatchConfig()},
        output_path=str(output_path),
    )

    assert rows[0]["input_tokens"] == 67
    assert rows[0]["output_tokens"] == 22
    # Adapters that report no usage leave the merged row without the keys.
    assert "input_tokens" not in rows[1]
    assert "output_tokens" not in rows[1]


def test_collect_and_merge_raises_for_missing_provider_config(tmp_path):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_1",
                "custom_id_to_source_index": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        records = _read_metadata_records(str(metadata_path))
        collect_and_merge(
            records=records,
            provider_configs={},
            output_path=str(tmp_path / "out.jsonl"),
        )
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "No provider config" in str(e)


def test_collect_and_merge_outputs_caption_for_plain_text_content(
    tmp_path, monkeypatch
):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_plain",
                "custom_id_to_source_index": {"img_1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "merged_plain.jsonl"

    class FakeAdapter:
        def retrieve_results(self, provider_batch_id, config):
            return [
                {
                    "custom_id": "img_1",
                    "generated_text": "A black cat sitting on a sofa.",
                }
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: FakeAdapter(),
    )

    records = _read_metadata_records(str(metadata_path))
    rows = collect_and_merge(
        records=records,
        provider_configs={"openai": OpenAIBatchConfig()},
        output_path=str(output_path),
    )

    assert rows == [
        {
            "shard_id": 0,
            "source_index": 0,
            "custom_id": "img_1",
            "caption": "A black cat sitting on a sofa.",
        }
    ]


def test_collect_and_merge_keeps_newest_receipt_for_retried_rows(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    # A retried shard resubmits the same rows under new batch ids; the retry
    # receipt is listed first so submission time, not file order, must decide.
    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "unit",
                        "shard_id": 0,
                        "provider_batch_id": "batch_retry",
                        "custom_id_to_source_index": {"answer-text-s0-1": 0},
                        "submitted_at_utc": "2026-09-13T11:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "provider": "unit",
                        "shard_id": 0,
                        "provider_batch_id": "batch_first",
                        "custom_id_to_source_index": {"answer-text-s0-1": 0},
                        "submitted_at_utc": "2026-09-13T10:00:00+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "merged_retry.jsonl"

    class UnitAdapter:
        def retrieve_results(self, provider_batch_id, config):
            text = {"batch_first": "first", "batch_retry": "retry"}[provider_batch_id]
            return [{"custom_id": "answer-text-s0-1", "generated_text": text}]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: UnitAdapter(),
    )

    records = _read_metadata_records(str(metadata_path))
    rows = collect_and_merge(
        records=records,
        provider_configs={"unit": SimpleNamespace(provider="unit")},
        output_path=str(output_path),
    )

    assert rows == [
        {
            "shard_id": 0,
            "source_index": 0,
            "custom_id": "answer-text-s0-1",
            "caption": "retry",
        }
    ]

    written = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == rows


def test_collect_and_merge_orders_rows_by_shard_then_source_index(
    tmp_path, monkeypatch
):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    # Every shard counts its rows from 0, and shard 1 is listed first.
    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "unit",
                        "shard_id": 1,
                        "provider_batch_id": "batch_shard_1",
                        "custom_id_to_source_index": {
                            "answer-text-s1-1": 0,
                            "answer-text-s1-2": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "provider": "unit",
                        "shard_id": 0,
                        "provider_batch_id": "batch_shard_0",
                        "custom_id_to_source_index": {
                            "answer-text-s0-1": 0,
                            "answer-text-s0-2": 1,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class UnitAdapter:
        def retrieve_results(self, provider_batch_id, config):
            shard = provider_batch_id.removeprefix("batch_shard_")
            return [
                {
                    "custom_id": f"answer-text-s{shard}-{n}",
                    "generated_text": f"s{shard}-{n}",
                }
                for n in (1, 2)
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: UnitAdapter(),
    )

    rows = collect_and_merge(
        records=_read_metadata_records(str(metadata_path)),
        provider_configs={"unit": SimpleNamespace(provider="unit")},
        output_path=str(tmp_path / "merged_shards.jsonl"),
    )

    assert [(row["shard_id"], row["source_index"]) for row in rows] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [row["caption"] for row in rows] == ["s0-1", "s0-2", "s1-1", "s1-2"]


def test_collect_and_merge_uses_openai_adapter_generated_text(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_openai",
                "custom_id_to_source_index": {"o1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "merged_openai.jsonl"

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"
                output_file_id = "file_output_1"

            return _RetrieveResp()

    class FakeFiles:
        def content(self, output_file_id):
            class _ContentResp:
                text = '{"custom_id":"o1","response":{"body":{"choices":[{"message":{"content":"{\\"question\\":\\"Q\\",\\"answer\\":\\"A\\"}"}}]}}}\n'

            return _ContentResp()

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()
            self.files = FakeFiles()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )
    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: OpenAIBatchAdapter(),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    records = _read_metadata_records(str(metadata_path))
    rows = collect_and_merge(
        records=records,
        provider_configs={"openai": OpenAIBatchConfig()},
        output_path=str(output_path),
    )

    assert rows == [
        {
            "shard_id": 0,
            "source_index": 0,
            "custom_id": "o1",
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        }
    ]

    written = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == rows


def test_collector_main_uses_config_and_records(tmp_path, monkeypatch):
    from mmirage.core.process.batch import collector

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_main",
                "custom_id_to_source_index": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[SimpleNamespace(provider_config={"provider": "openai"})]
    )
    captured = {}

    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    def _fake_collect_and_merge(records, provider_configs, output_path_arg):
        captured["records"] = records
        captured["provider_configs"] = provider_configs
        captured["output_path"] = output_path_arg
        return [{"source_index": 0, "custom_id": "c1", "caption": "ok"}]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.collect_and_merge",
        _fake_collect_and_merge,
    )

    rc = collector.main(
        [
            "--metadata-path",
            str(metadata_path),
            "--output-path",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert rc == 0
    assert len(captured["records"]) == 1
    assert captured["records"][0].provider == "openai"
    assert "openai" in captured["provider_configs"]
    assert captured["output_path"] == str(output_path)


def test_collector_main_uses_config_metadata_path_when_missing_cli_arg(
    tmp_path, monkeypatch
):
    from mmirage.core.process.batch import collector

    metadata_base = tmp_path / "batch_metadata.jsonl"
    metadata_path = tmp_path / "batch_metadata.text.abc123.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_main",
                "custom_id_to_source_index": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(
                provider_config={
                    "provider": "openai",
                    "metadata_output_path": str(metadata_base),
                }
            )
        ]
    )
    captured = {}

    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    def _fake_collect_and_merge(records, provider_configs, output_path_arg):
        captured["records"] = records
        captured["provider_configs"] = provider_configs
        captured["output_path"] = output_path_arg
        return [{"source_index": 0, "custom_id": "c1", "caption": "ok"}]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.collect_and_merge",
        _fake_collect_and_merge,
    )

    rc = collector.main(
        [
            "--output-path",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert rc == 0
    assert len(captured["records"]) == 1
    assert captured["records"][0].provider == "openai"
    assert "openai" in captured["provider_configs"]
    assert captured["output_path"] == str(output_path)


def test_collector_main_raises_when_config_metadata_paths_missing(
    tmp_path, monkeypatch, caplog
):
    from mmirage.core.process.batch import collector

    metadata_base = tmp_path / "batch_metadata.jsonl"
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(
                provider_config={
                    "provider": "openai",
                    "metadata_output_path": str(metadata_base),
                }
            )
        ]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    rc = collector.main(
        [
            "--output-path",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )
    assert rc == 1
    assert (
        "No metadata receipts matched config metadata_output_path patterns"
        in caplog.text
    )


def test_collector_main_raises_when_metadata_provider_missing_in_config(
    tmp_path, monkeypatch, caplog
):
    from mmirage.core.process.batch import collector

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "mistral",
                "provider_batch_id": "batch_mistral",
                "custom_id_to_source_index": {"m1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    # Config intentionally only defines openai, not mistral.
    cfg = SimpleNamespace(
        processors=[SimpleNamespace(provider_config={"provider": "openai"})]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    rc = collector.main(
        [
            "--metadata-path",
            str(metadata_path),
            "--output-path",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )
    assert rc == 1
    assert "missing from YAML batch config" in caplog.text


def test_collect_and_merge_routes_multiple_providers(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )
    from mmirage.core.process.batch.provider_resolution import resolve_provider_configs

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_openai",
                        "custom_id_to_source_index": {"o1": 1},
                    }
                ),
                json.dumps(
                    {
                        "provider": "unit",
                        "provider_batch_id": "batch_unit",
                        "custom_id_to_source_index": {"u1": 0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(provider_config={"provider": "openai"}),
            SimpleNamespace(provider_config={"provider": "unit"}),
        ]
    )

    records = _read_metadata_records(str(metadata_path))
    provider_configs = resolve_provider_configs(records, cfg)

    class OpenAIAdapter:
        def __init__(self):
            self.calls = []

        def retrieve_results(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            return [{"custom_id": "o1", "generated_text": "openai"}]

    class UnitAdapter:
        def __init__(self):
            self.calls = []

        def retrieve_results(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            return [{"custom_id": "u1", "generated_text": "unit"}]

    adapters = {
        "openai": OpenAIAdapter(),
        "unit": UnitAdapter(),
    }

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: adapters[config.provider],
    )

    output_path = tmp_path / "merged.jsonl"
    rows = collect_and_merge(
        records=records,
        provider_configs=provider_configs,
        output_path=str(output_path),
    )

    assert [row["custom_id"] for row in rows] == ["u1", "o1"]
    assert [row["caption"] for row in rows] == ["unit", "openai"]
    assert ("batch_openai", "openai") in adapters["openai"].calls
    assert ("batch_unit", "unit") in adapters["unit"].calls


def test_collector_main_raises_for_invalid_batch_provider_config(
    tmp_path, monkeypatch, caplog
):
    from mmirage.core.process.batch import collector

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_1",
                "custom_id_to_source_index": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(
                provider_config={"provider": "openai", "batch_endpoint": "v1"}
            )
        ]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    rc = collector.main(
        [
            "--metadata-path",
            str(metadata_path),
            "--output-path",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )
    assert rc == 1
    assert "batch_endpoint must start with '/'" in caplog.text


def test_collect_and_merge_keeps_one_row_per_position(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_tie",
                "custom_id_to_source_index": {"a": 0, "b": 0, "c": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "merged_tie.jsonl"

    class FakeAdapter:
        def retrieve_results(self, provider_batch_id, config):
            # Rows come back out of order; a and b both claim source row 0.
            return [
                {"custom_id": "b", "generated_text": "B"},
                {"custom_id": "a", "generated_text": "A"},
                {"custom_id": "c", "generated_text": "C"},
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: FakeAdapter(),
    )

    records = _read_metadata_records(str(metadata_path))
    rows = collect_and_merge(
        records=records,
        provider_configs={"openai": OpenAIBatchConfig()},
        output_path=str(output_path),
    )

    # Same receipt, so the row read last wins the shared position.
    assert [r["source_index"] for r in rows] == [0, 1]
    assert [r["custom_id"] for r in rows] == ["a", "c"]


def test_build_output_payload_logs_malformed_json(caplog):
    from mmirage.core.process.batch.collector import _build_output_payload

    malformed_json = '{"question": "incomplete'
    result_row = {
        "custom_id": "row_123",
        "generated_text": malformed_json,
    }

    with caplog.at_level("WARNING"):
        output = _build_output_payload(result_row, custom_id="row_123")

    assert output == {"caption": malformed_json}
    assert "Failed to parse JSON for result row" in caplog.text
    assert "custom_id=row_123" in caplog.text
    assert "Treating as raw text" in caplog.text


def test_build_output_payload_keeps_plain_text_silent(caplog):
    from mmirage.core.process.batch.collector import _build_output_payload

    result_row = {
        "custom_id": "caption:multimodal:1",
        "generated_text": 'The image features a solid orange background with the text "A Cat" displayed in a bold font.',
    }

    with caplog.at_level("WARNING"):
        output = _build_output_payload(result_row, custom_id="caption:multimodal:1")

    assert output == {
        "caption": 'The image features a solid orange background with the text "A Cat" displayed in a bold font.'
    }
    assert "Failed to parse JSON for result row" not in caplog.text


def test_build_output_payload_preserves_provider_error_status():
    from mmirage.core.process.batch.collector import _build_output_payload

    result_row = {
        "custom_id": "formatted_answer:text:50",
        "status": "error",
        "error_message": "Unrecognized request argument supplied: expected_schema",
    }

    output = _build_output_payload(result_row, custom_id="formatted_answer:text:50")

    assert output == {
        "status": "error",
        "error_message": "Unrecognized request argument supplied: expected_schema",
    }
