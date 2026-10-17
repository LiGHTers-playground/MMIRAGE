import base64
import json

import pytest

from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult
from mmirage.core.process.batch.registry import (
    BatchAdapterRegistry,
)


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch):
    """The API key is only ever read from the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_openai_build_request_matches_expected_structure():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    config = OpenAIBatchConfig(model="gpt-4.1-mini")
    adapter = OpenAIBatchAdapter()
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }

    request = adapter.build_request(custom_id="row-001", payload=payload, config=config)

    assert request == {
        "custom_id": "row-001",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    }


def test_openai_build_request_injects_structured_output_format():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    config = OpenAIBatchConfig(model="gpt-4.1-mini")
    adapter = OpenAIBatchAdapter()
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "expected_schema": ["question", "answer"],
    }

    request = adapter.build_request(custom_id="row-002", payload=payload, config=config)

    assert "expected_schema" not in request["body"]

    assert request["body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        },
    }


def test_openai_build_request_converts_local_image_path_to_data_uri(tmp_path):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    image_bytes = b"\xff\xd8\xff\xe0testjpeg"
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(image_bytes)

    config = OpenAIBatchConfig(model="gpt-4o-mini")
    adapter = OpenAIBatchAdapter()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe image"},
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                ],
            }
        ]
    }

    request = adapter.build_request(
        custom_id="vision-1", payload=payload, config=config
    )

    url = request["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")

    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == image_bytes


def test_openai_estimate_request_bytes_matches_utf8_json_size():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    adapter = OpenAIBatchAdapter()
    request = {
        "custom_id": "accented",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"message": "caf\u00e9"},
    }

    expected = len(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    assert adapter.estimate_request_bytes(request) == expected


def test_openai_submit_chunk_uses_mocked_openai_client(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    captured = {}

    class FakeFiles:
        def create(self, *, file, purpose):
            file_name, file_obj = file
            assert file_name == "batch_chunk-chunk-01.jsonl"
            assert purpose == "batch"
            file_content = file_obj.read().decode("utf-8")
            captured["jsonl"] = file_content

            class _FileResp:
                id = "file_123"

            return _FileResp()

    class FakeBatches:
        def create(self, **kwargs):
            captured["batch_create_kwargs"] = kwargs

            class _BatchResp:
                id = "batch_123"
                status = "validating"
                endpoint = kwargs["endpoint"]

            return _BatchResp()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.files = FakeFiles()
            self.batches = FakeBatches()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig(
        model="gpt-4.1-mini",
        completion_window="24h",
        batch_endpoint="/v1/chat/completions",
        metadata={"pipeline": "unit"},
    )
    adapter = OpenAIBatchAdapter()
    requests = [
        adapter.build_request(
            custom_id="r1",
            payload={"messages": [{"role": "user", "content": "Hi"}]},
            config=config,
        )
    ]

    raw_result = adapter.submit_chunk(
        chunk_id="chunk-01", requests=requests, config=config
    )

    assert captured["client_kwargs"]["api_key"] == "test-key"
    assert captured["batch_create_kwargs"] == {
        "input_file_id": "file_123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"pipeline": "unit", "chunk_id": "chunk-01"},
    }
    assert raw_result["id"] == "batch_123"
    assert raw_result["status"] == "validating"
    assert raw_result["input_file_id"] == "file_123"

    jsonl_lines = [line for line in captured["jsonl"].split("\n") if line.strip()]
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0]) == requests[0]


def test_openai_parse_submission_result_normalizes_payload():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    adapter = OpenAIBatchAdapter()
    raw = {
        "id": "batch_123",
        "status": "in_progress",
        "input_file_id": "file_123",
    }

    result = adapter.parse_submission_result(raw_result=raw)

    assert isinstance(result, BatchSubmissionResult)
    assert result.provider_batch_id == "batch_123"
    assert result.status == "in_progress"
    assert result.raw_response == raw


def test_factory_resolves_openai_adapter_from_registry():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    BatchAdapterRegistry.clear()
    config = OpenAIBatchConfig(model="gpt-4.1-mini")

    adapter = BatchAdapterRegistry.create(config)

    assert isinstance(adapter, OpenAIBatchAdapter)


def test_openai_check_batch_status_uses_mocked_openai_client(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    captured = {}

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            captured["retrieved_id"] = provider_batch_id

            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"

            return _RetrieveResp()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.batches = FakeBatches()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig(
        base_url="https://example.test/v1",
    )
    adapter = OpenAIBatchAdapter()

    result = adapter.check_batch_status(provider_batch_id="batch_456", config=config)

    assert captured["client_kwargs"] == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }
    assert captured["retrieved_id"] == "batch_456"
    assert isinstance(result, BatchSubmissionResult)
    assert result.provider_batch_id == "batch_456"
    assert result.status == "completed"


def test_openai_check_batch_status_falls_back_to_env_api_key(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    captured = {}

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"

            return _RetrieveResp()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.batches = FakeBatches()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    result = adapter.check_batch_status(provider_batch_id="batch_env", config=config)

    assert captured["client_kwargs"]["api_key"] == "env-test-key"
    assert result.provider_batch_id == "batch_env"


def test_openai_check_batch_status_raises_when_no_api_key(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        adapter.check_batch_status(provider_batch_id="batch_missing", config=config)


def test_openai_retrieve_results_downloads_and_parses_jsonl(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    captured = {}

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            captured["retrieved_id"] = provider_batch_id

            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"
                output_file_id = "file_output_1"

            return _RetrieveResp()

    class FakeFiles:
        def content(self, output_file_id):
            captured["output_file_id"] = output_file_id

            class _ContentResp:
                text = (
                    '{"custom_id":"c1","response":{"body":{"text":"A"}}}\n'
                    '{"custom_id":"c2","response":{"body":{"text":"B"}}}\n'
                )

            return _ContentResp()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.batches = FakeBatches()
            self.files = FakeFiles()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_abc", config=config)

    assert captured["retrieved_id"] == "batch_abc"
    assert captured["output_file_id"] == "file_output_1"
    assert len(rows) == 2
    assert rows[0]["custom_id"] == "c1"
    assert rows[1]["custom_id"] == "c2"
    assert rows[0]["response"]["body"]["text"] == "A"
    assert rows[1]["response"]["body"]["text"] == "B"
    assert rows[0]["generated_text"] == "A"
    assert rows[1]["generated_text"] == "B"


def test_openai_retrieve_results_normalizes_usage(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

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
                text = (
                    '{"custom_id":"c1","response":{"body":{"text":"A","usage":'
                    '{"prompt_tokens":67,"completion_tokens":22,"total_tokens":89}}}}\n'
                    '{"custom_id":"c2","response":{"body":{"text":"B"}}}\n'
                )

            return _ContentResp()

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()
            self.files = FakeFiles()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    adapter = OpenAIBatchAdapter()

    rows = adapter.retrieve_results(
        provider_batch_id="batch_usage", config=OpenAIBatchConfig()
    )

    assert rows[0]["input_tokens"] == 67
    assert rows[0]["output_tokens"] == 22
    # Rows without provider usage keep the keys absent rather than reporting zero.
    assert "input_tokens" not in rows[1]
    assert "output_tokens" not in rows[1]


def test_openai_retrieve_results_prefers_message_content(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"
                output_file_id = "file_output_choices"

            return _RetrieveResp()

    class FakeFiles:
        def content(self, output_file_id):
            class _ContentResp:
                text = (
                    '{"custom_id":"c1","response":{"body":{"choices":['
                    '{"message":{"content":"{\\"question\\":\\"Q\\",\\"answer\\":\\"A\\"}"}}'
                    "]}}}\n"
                )

            return _ContentResp()

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()
            self.files = FakeFiles()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_choices", config=config)

    assert rows[0]["custom_id"] == "c1"
    assert rows[0]["generated_text"] == '{"question":"Q","answer":"A"}'


def test_openai_retrieve_results_normalizes_error_rows(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"
                output_file_id = "file_output_error"

            return _RetrieveResp()

    class FakeFiles:
        def content(self, output_file_id):
            class _ContentResp:
                text = (
                    '{"id":"batch_req_1","custom_id":"formatted_answer:text:50",'
                    '"response":{"status_code":400,"request_id":"req_1",'
                    '"body":{"error":{"message":"Unrecognized request argument supplied: expected_schema",'
                    '"type":"invalid_request_error","param":null,"code":null}}},"error":null}\n'
                )

            return _ContentResp()

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()
            self.files = FakeFiles()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_error", config=config)

    assert rows == [
        {
            "id": "batch_req_1",
            "custom_id": "formatted_answer:text:50",
            "response": {
                "status_code": 400,
                "request_id": "req_1",
                "body": {
                    "error": {
                        "message": "Unrecognized request argument supplied: expected_schema",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
            },
            "error": None,
            "status": "error",
            "error_message": "Unrecognized request argument supplied: expected_schema",
        }
    ]


def test_openai_retrieve_results_raises_if_batch_not_completed(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "in_progress"
                output_file_id = None

            return _RetrieveResp()

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()

            class _Files:
                def content(self, output_file_id):
                    raise AssertionError(
                        "content() should not be called when batch is not completed"
                    )

            self.files = _Files()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    with pytest.raises(ValueError, match="not completed"):
        adapter.retrieve_results(provider_batch_id="batch_abc", config=config)


def test_openai_retrieve_results_uses_error_file_when_output_missing(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"
                output_file_id = None
                error_file_id = "file_error_1"

            return _RetrieveResp()

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()

            class _Files:
                def content(self, output_file_id):
                    captured["output_file_id"] = output_file_id

                    class _ContentResp:
                        text = (
                            '{"custom_id":"c1","response":{"body":{"error":{'
                            '"message":"Unrecognized request argument supplied: expected_schema"}}}}\n'
                        )

                    return _ContentResp()

            self.files = _Files()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_abc", config=config)

    assert captured["output_file_id"] == "file_error_1"
    assert rows == [
        {
            "custom_id": "c1",
            "response": {
                "body": {
                    "error": {
                        "message": "Unrecognized request argument supplied: expected_schema"
                    }
                }
            },
            "status": "error",
            "error_message": "Unrecognized request argument supplied: expected_schema",
        }
    ]


def test_openai_retrieve_results_reads_output_and_error_files(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            class _RetrieveResp:
                id = provider_batch_id
                status = "completed"
                output_file_id = "file_output_1"
                error_file_id = "file_error_1"

            return _RetrieveResp()

    captured = {"file_ids": []}

    class FakeClient:
        def __init__(self, **kwargs):
            self.batches = FakeBatches()

            class _Files:
                def content(self, file_id):
                    captured["file_ids"].append(file_id)

                    class _ContentResp:
                        text = {
                            "file_output_1": (
                                '{"custom_id":"c1","response":{"body":{"text":"A"}}}\n'
                            ),
                            "file_error_1": (
                                '{"custom_id":"c2","response":{"body":{"error":{'
                                '"message":"Unrecognized request argument supplied: expected_schema"}}}}\n'
                            ),
                        }[file_id]

                    return _ContentResp()

            self.files = _Files()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig()
    adapter = OpenAIBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_abc", config=config)

    assert captured["file_ids"] == ["file_output_1", "file_error_1"]
    assert len(rows) == 2
    assert rows[0]["custom_id"] == "c1"
    assert rows[0]["generated_text"] == "A"
    assert rows[1] == {
        "custom_id": "c2",
        "response": {
            "body": {
                "error": {
                    "message": "Unrecognized request argument supplied: expected_schema"
                }
            }
        },
        "status": "error",
        "error_message": "Unrecognized request argument supplied: expected_schema",
    }
