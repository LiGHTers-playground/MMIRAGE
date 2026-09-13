import json
from pathlib import Path

from datasets import load_dataset

from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.process.processors.batch_api.config import (
    BatchApiOutputVar,
    BatchApiProcessorConfig,
)
from mmirage.core.process.variables import InputVar
from mmirage.core.writer.renderer import TemplateRenderer


def test_integration_batch_pipeline_with_stateful_accumulator(monkeypatch, tmp_path):
    captured = {
        "file_uploads": [],
        "batch_creates": [],
    }

    class FakeFiles:
        def create(self, *, file, purpose):
            file_name, file_obj = file
            payload = file_obj.read().decode("utf-8")
            captured["file_uploads"].append(
                {
                    "file_name": file_name,
                    "purpose": purpose,
                    "payload": payload,
                }
            )

            class _FileResp:
                id = f"file_{len(captured['file_uploads'])}"

            return _FileResp()

    class FakeBatches:
        def create(self, **kwargs):
            captured["batch_creates"].append(kwargs)

            class _BatchResp:
                id = f"batch_{len(captured['batch_creates'])}"
                status = "validating"
                endpoint = kwargs["endpoint"]

            return _BatchResp()

    class FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.files = FakeFiles()
            self.batches = FakeBatches()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeOpenAIClient,
    )

    metadata_base = tmp_path / "batch_receipts.jsonl"
    batch_cfg = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=OpenAIBatchConfig(
            model="gpt-4.1-mini",
            max_chunk_bytes=500,
            max_requests_per_chunk=None,
            metadata_output_path=str(metadata_base),
            metadata={"pipeline": "integration-test"},
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mapper = MMIRAGEMapper(
        processor_configs=[batch_cfg],
        input_vars=[InputVar(name="text", key="text")],
        output_vars=[
            BatchApiOutputVar(
                name="answer",
                type="batch_api",
                prompt="{{ text }}",
                output_type="plain",
            )
        ],
    )
    renderer = TemplateRenderer(output_schema={"answer": "{{ answer }}"})

    data_path = Path(__file__).parent / "mock_data" / "data.jsonl"
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def rewrite_batch(batch, mapper, renderer):
        envs, _positions = mapper.rewrite_batch(batch)
        return renderer.batch_render(envs)

    ds_out = dataset.map(
        rewrite_batch,
        batched=True,
        batch_size=7,
        fn_kwargs={"mapper": mapper, "renderer": renderer},
        load_from_cache_file=False,
    )

    # Explicit lifecycle flush required by the architecture.
    mapper.finalize_processors()

    # 1) Multiple provider submissions prove byte-based chunking with carry-over.
    assert len(captured["file_uploads"]) > 1
    assert len(captured["batch_creates"]) > 1

    # 2) Map output is placeholder-based and does not wait for completion.
    answers = ds_out["answer"]
    assert len(answers) == len(dataset)
    assert all(
        isinstance(v, str) and v.startswith("__BATCH_SUBMITTED__:answer-")
        for v in answers
    )

    # 3) Metadata receipts are written and include both full_chunk and finalize flush reasons.
    metadata_text_matches = sorted(tmp_path.glob("batch_receipts.text.*.jsonl"))
    assert len(metadata_text_matches) == 1
    metadata_text_path = metadata_text_matches[0]

    records = [
        json.loads(line)
        for line in metadata_text_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) > 1

    flush_reasons = {record["flush_reason"] for record in records}
    assert "full_chunk" in flush_reasons
    assert "finalize" in flush_reasons

    assert all(record["provider"] == "openai" for record in records)
    assert all("custom_id_to_source_index" in record for record in records)
