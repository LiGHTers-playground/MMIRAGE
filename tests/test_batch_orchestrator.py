import json

from batch_fixtures import RecordingAdapter, UnitBatchConfig

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.base import ProcessorRegistry
from mmirage.core.process.processors.batch_api.config import BatchApiProcessorConfig


def test_orchestrator_buffers_across_iterations_and_avoids_tiny_midstream_flush(
    tmp_path,
):
    from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator

    adapter = RecordingAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        metadata_output_path=str(tmp_path / "metadata.jsonl"),
    )
    orchestrator = BatchSubmissionOrchestrator(adapter=adapter, config=config)

    # Iteration 1: only 9 bytes total, should remain buffered and submit nothing.
    r1 = [{"custom_id": "a", "size_bytes": 6}, {"custom_id": "b", "size_bytes": 3}]
    out1 = orchestrator.add_requests(r1, [10, 11], {"phase": "iter1"})
    assert out1 == []
    assert len(adapter.submissions) == 0

    # Iteration 2: appending 2 bytes should emit one full chunk [6,3] and keep [2].
    r2 = [{"custom_id": "c", "size_bytes": 2}]
    out2 = orchestrator.add_requests(r2, [12], {"phase": "iter2"})
    assert len(out2) == 1
    assert len(adapter.submissions) == 1
    assert [x["size_bytes"] for x in adapter.submissions[0]["requests"]] == [6, 3]
    assert orchestrator.pending_count == 1

    # Finalize: emits the remaining tiny tail exactly once.
    out3 = orchestrator.finalize({"phase": "finalize"})
    assert len(out3) == 1
    assert len(adapter.submissions) == 2
    assert [x["size_bytes"] for x in adapter.submissions[1]["requests"]] == [2]
    assert orchestrator.pending_count == 0


def test_orchestrator_writes_provider_neutral_metadata_with_flush_reason(tmp_path):
    from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator

    metadata_path = tmp_path / "batch_metadata.jsonl"
    adapter = RecordingAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        metadata_output_path=str(metadata_path),
    )
    orchestrator = BatchSubmissionOrchestrator(adapter=adapter, config=config)

    orchestrator.add_requests(
        requests=[
            {"custom_id": "x1", "size_bytes": 8},
            {"custom_id": "x2", "size_bytes": 8},
        ],
        source_indices=[0, 1],
        model_params_snapshot={"model": "unit-model"},
    )
    orchestrator.finalize({"model": "unit-model"})

    lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["provider"] == "unit"
    assert first["flush_reason"] == "full_chunk"
    assert first["custom_id_to_source_index"] == {"x1": 0}
    assert isinstance(first["request_hash"], str) and len(first["request_hash"]) == 64

    assert second["flush_reason"] == "finalize"
    assert second["custom_id_to_source_index"] == {"x2": 1}
    assert second["provider_batch_id"].startswith("batch-chunk-")


def test_orchestrator_records_shard_id_in_metadata(tmp_path):
    from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator

    metadata_path = tmp_path / "batch_metadata.jsonl"
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        metadata_output_path=str(metadata_path),
    )
    orchestrator = BatchSubmissionOrchestrator(
        adapter=RecordingAdapter(), config=config, shard_id=3
    )

    orchestrator.add_requests(
        requests=[{"custom_id": "x1", "size_bytes": 8}],
        source_indices=[0],
    )
    orchestrator.finalize()

    lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["shard_id"] == 3


def test_batch_api_processor_passes_shard_id_to_orchestrators(tmp_path, unit_provider):
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(
            provider="unit",
            metadata_output_path=str(tmp_path / "metadata.jsonl"),
        ),
    )

    processor = ProcessorRegistry.get_processor("batch_api")(config, shard_id=2)

    assert processor._text_orchestrator.shard_id == 2
    assert processor._multimodal_orchestrator.shard_id == 2


def test_orchestrator_exports_prompts_and_skips_submit(tmp_path):
    from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator

    export_path = tmp_path / "exported_prompts.jsonl"
    metadata_path = tmp_path / "batch_metadata.jsonl"
    adapter = RecordingAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        metadata_output_path=str(metadata_path),
    )
    orchestrator = BatchSubmissionOrchestrator(
        adapter=adapter,
        config=config,
        export_prompts_path=str(export_path),
    )

    out = orchestrator.add_requests(
        requests=[
            {"custom_id": "r1", "size_bytes": 8, "payload": {"text": "hello"}},
            {"custom_id": "r2", "size_bytes": 8, "payload": {"text": "world"}},
        ],
        source_indices=[0, 1],
        model_params_snapshot={"mode": "dry-run"},
    )

    assert len(out) == 1
    assert len(adapter.submissions) == 0

    assert export_path.exists()
    exported_lines = [
        json.loads(line)
        for line in export_path.read_text(encoding="utf-8").splitlines()
    ]
    assert exported_lines == [
        {
            "batch_id": "chunk-000001",
            "request": {
                "custom_id": "r1",
                "size_bytes": 8,
                "payload": {"text": "hello"},
            },
        },
    ]

    result = out[0]
    assert result.provider_batch_id.startswith("dry-run-")
    assert result.status == "dry_run"
    assert orchestrator.pending_count == 1


def test_batch_api_processor_initializes_with_custom_provider(tmp_path, unit_provider):
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(
            provider="unit",
            unit_setting="custom",
            metadata_output_path=str(tmp_path / "metadata.jsonl"),
        ),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")
    processor = processor_cls(config)

    assert isinstance(processor._batch_provider_config, UnitBatchConfig)
    assert processor._batch_provider_config.provider == "unit"
    assert processor._batch_provider_config.unit_setting == "custom"
    assert isinstance(processor._batch_adapter, RecordingAdapter)
