import json
from pathlib import Path

import pytest
from batch_fixtures import UnitBatchConfig

from mmirage.core.process.base import ProcessorRegistry
from mmirage.core.process.processors.batch_api.config import (
    BatchApiOutputVar,
    BatchApiProcessorConfig,
)
from mmirage.core.process.variables import VariableEnvironment


def test_batch_api_processor_exports_to_single_file_with_batch_ids(
    tmp_path, unit_provider
):
    export_file = tmp_path / "exports" / "prompts.jsonl"
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(
            provider="unit",
            max_chunk_bytes=10,
            metadata_output_path=str(tmp_path / "meta.jsonl"),
        ),
        export_prompts_dir=str(export_file),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")
    processor = processor_cls(config)

    assert processor._text_orchestrator is not None
    assert processor._multimodal_orchestrator is not None

    text_path = processor._text_orchestrator._export_prompts_path
    multi_path = processor._multimodal_orchestrator._export_prompts_path
    assert text_path is not None and multi_path is not None
    assert text_path.startswith(str(export_file).removesuffix(".jsonl") + ".")
    assert multi_path == text_path
    export_file = Path(text_path)

    # Submit one chunk to each orchestrator
    processor._text_orchestrator.add_requests(
        requests=[
            {"custom_id": "t1", "size_bytes": 6},
            {"custom_id": "t2", "size_bytes": 6},
        ],
        source_indices=[0, 1],
    )

    processor._multimodal_orchestrator.add_requests(
        requests=[
            {"custom_id": "m1", "size_bytes": 6},
            {"custom_id": "m2", "size_bytes": 6},
        ],
        source_indices=[2, 3],
    )

    assert export_file.exists()

    lines = [
        json.loads(line)
        for line in export_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    assert {line["request"]["custom_id"] for line in lines} == {"t1", "m1"}
    assert {line["batch_id"] for line in lines} == {
        "text-chunk-000001",
        "multimodal-chunk-000001",
    }


def test_batch_api_processor_exports_what_batch_process_sample_buffered(
    tmp_path, unit_provider
):
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(
            provider="unit",
            max_chunk_bytes=1000,
            metadata_output_path=str(tmp_path / "meta.jsonl"),
        ),
        export_prompts_dir=str(tmp_path / "exports"),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")
    processor = processor_cls(config)
    output_var = BatchApiOutputVar(
        name="answer",
        type="batch_api",
        prompt="Question about {{ text }}",
        output_type="plain",
    )

    batch = [
        VariableEnvironment({"text": "Berne"}),
        VariableEnvironment({"text": "Paris"}),
    ]
    placeholders = processor.batch_process_sample(batch, output_var)

    assert [p.get("answer") for p in placeholders] == [
        "__BATCH_SUBMITTED__:answer-text-s0-1",
        "__BATCH_SUBMITTED__:answer-text-s0-2",
    ]
    # The chunk is not full, so both requests wait for the end of the dataset.
    assert processor._text_orchestrator.pending_count == 2

    export_file = Path(processor._text_orchestrator._export_prompts_path)
    # created up front to fail early on a bad path, still empty until the flush
    assert export_file.read_text(encoding="utf-8") == ""

    processor.finalize()

    assert processor._text_orchestrator.pending_count == 0
    lines = [
        json.loads(line)
        for line in export_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["request"]["custom_id"] for line in lines] == [
        "answer-text-s0-1",
        "answer-text-s0-2",
    ]
    assert lines[0]["request"]["messages"][0]["content"] == "Question about Berne"


def test_batch_api_processor_requires_metadata_output_path_for_submission(
    unit_provider,
):
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(provider="unit", metadata_output_path=""),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")

    with pytest.raises(ValueError, match="metadata_output_path"):
        processor_cls(config)


def test_batch_api_processor_export_mode_allows_empty_metadata_output_path(
    tmp_path, unit_provider
):
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(provider="unit", metadata_output_path=""),
        export_prompts_dir=str(tmp_path / "exports"),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")
    processor = processor_cls(config)

    assert processor._text_orchestrator._export_prompts_path is not None


def test_batch_api_processor_request_ids_differ_across_shards(tmp_path, unit_provider):
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(
            provider="unit",
            metadata_output_path=str(tmp_path / "meta.jsonl"),
        ),
        export_prompts_dir=str(tmp_path / "exports"),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")
    shard_0 = processor_cls(config, shard_id=0)
    shard_1 = processor_cls(config, shard_id=1)

    first_id_shard_0 = shard_0._next_custom_id("answer", "text")
    first_id_shard_1 = shard_1._next_custom_id("answer", "text")

    assert first_id_shard_0 == "answer-text-s0-1"
    assert first_id_shard_1 == "answer-text-s1-1"
    assert first_id_shard_0 != first_id_shard_1
