import json

from mmirage.config.openai_batch import OpenAIBatchConfig


def test_integration_receiver_reads_receipt_and_writes_merged_output(
    tmp_path, monkeypatch
):
    from mmirage.core.process.batch.collector import (
        _read_metadata_records,
        collect_and_merge,
    )

    metadata_path = tmp_path / "receipt.text.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_a",
                        "custom_id_to_source_index": {"id_a": 1, "id_b": 0},
                    }
                ),
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_b",
                        "custom_id_to_source_index": {"id_c": 2},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.jsonl"

    class FakeAdapter:
        def retrieve_results(self, provider_batch_id, config):
            if provider_batch_id == "batch_a":
                return [
                    {
                        "custom_id": "id_a",
                        "generated_text": '{"question":"What is id_a?","answer":"one"}',
                    },
                    {
                        "custom_id": "id_b",
                        "generated_text": '{"question":"What is id_b?","answer":"zero"}',
                    },
                ]
            return [
                {
                    "custom_id": "id_c",
                    "generated_text": '{"question":"What is id_c?","answer":"two"}',
                }
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterRegistry.create",
        lambda config: FakeAdapter(),
    )

    records = _read_metadata_records([str(metadata_path)])
    rows = collect_and_merge(
        records=records,
        provider_configs={"openai": OpenAIBatchConfig()},
        output_path=str(output_path),
    )

    assert [r["source_index"] for r in rows] == [0, 1, 2]
    assert [r["custom_id"] for r in rows] == ["id_b", "id_a", "id_c"]
    assert [r["conversations"][1]["content"] for r in rows] == ["zero", "one", "two"]

    written = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["custom_id"] for r in written] == ["id_b", "id_a", "id_c"]
    assert [r["conversations"][1]["content"] for r in written] == ["zero", "one", "two"]
