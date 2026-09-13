import base64
import json
import logging
from types import SimpleNamespace

import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from mmirage.config.anthropic_batch import AnthropicBatchConfig
from mmirage.core.process.batch import anthropic_adapter
from mmirage.core.process.batch.anthropic_adapter import AnthropicBatchAdapter
from mmirage.core.process.batch.provider_resolution import (
    resolve_single_provider_config,
)
from mmirage.core.process.batch.registry import BatchAdapterRegistry


@pytest.fixture(autouse=True)
def anthropic_api_key(monkeypatch):
    """The API key is only ever read from the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _patch_anthropic_client(monkeypatch, fake_client_cls):
    """Swap the SDK entry point the adapter holds, leaving sys.modules alone."""
    monkeypatch.setattr(anthropic_adapter, "Anthropic", fake_client_cls)


def _message_batch(batch_id="batch_1", processing_status="ended"):
    """A real MessageBatch, so a field the SDK does not have cannot be faked in."""
    return MessageBatch.model_validate(
        {
            "id": batch_id,
            "archived_at": None,
            "cancel_initiated_at": None,
            "created_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T01:00:00Z",
            "expires_at": "2026-01-02T00:00:00Z",
            "processing_status": processing_status,
            "request_counts": {
                "canceled": 0,
                "errored": 0,
                "expired": 0,
                "processing": 0,
                "succeeded": 1,
            },
            "results_url": f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
            "type": "message_batch",
        }
    )


def _batch_results(rows):
    """results() hands back a lazy iterator of models, not a list of dicts."""
    return iter([MessageBatchIndividualResponse.model_validate(row) for row in rows])


def test_anthropic_build_request_normalizes_messages_and_images(tmp_path, monkeypatch):
    image_bytes = b"\xff\xd8\xff\xe0testjpeg"
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(image_bytes)

    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig(model="claude-haiku-4-5")
    adapter = AnthropicBatchAdapter()
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

    assert request["custom_id"] == "vision-1"
    assert request["params"]["model"] == "claude-haiku-4-5"
    assert request["params"]["max_tokens"] == config.max_tokens

    content = request["params"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe image"}
    assert content[1]["type"] == "image"

    encoded = content[1]["source"]["data"]
    assert base64.b64decode(encoded) == image_bytes


def test_anthropic_build_request_keeps_remote_image_urls(monkeypatch):
    """Remote URLs go to the provider untouched, same as the openai adapter."""

    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    url = "https://example.com/cat.png"
    request = AnthropicBatchAdapter().build_request(
        custom_id="vision-2",
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": url}}],
                }
            ]
        },
        config=AnthropicBatchConfig(),
    )

    content = request["params"]["messages"][0]["content"]
    assert content[0] == {"type": "image", "source": {"type": "url", "url": url}}


@pytest.mark.parametrize(
    "url",
    [
        "sample.bmp",
        "sample.svg",
        "sample",
        "data:application/pdf;base64,AAA",
    ],
)
def test_anthropic_rejects_unsupported_image_media_types(tmp_path, url):
    """Anthropic only accepts jpeg, png, gif and webp, fail before submitting."""
    if not url.startswith("data:"):
        image_path = tmp_path / url
        image_path.write_bytes(b"x")
        url = str(image_path)

    with pytest.raises(ValueError, match="Unsupported image media type"):
        AnthropicBatchAdapter._image_source_from_url(url)


@pytest.mark.parametrize("content", [None, 123, {"text": "hi"}])
def test_anthropic_rejects_content_that_is_not_a_string_or_a_list(content):
    """normalization must fail if the content is not a string or a list of structured items."""
    with pytest.raises(ValueError, match="message content must be"):
        AnthropicBatchAdapter._normalize_messages(
            [{"role": "user", "content": content}]
        )


def test_anthropic_keeps_an_empty_prompt_as_a_normal_request():
    """An empty dataset row must cost one row, not the whole shard."""
    normalized = AnthropicBatchAdapter._normalize_messages(
        [{"role": "user", "content": ""}]
    )
    assert normalized == [{"role": "user", "content": [{"type": "text", "text": ""}]}]


@pytest.mark.parametrize(
    "part",
    [
        "not a dict",
        {"type": "text", "text": 123},
        {"type": "image_url", "image_url": {"url": None}},
        {"type": "document", "source": {}},
        {"type": "image", "source": "not a dict"},
    ],
)
def test_anthropic_warns_on_every_dropped_content_block(caplog, part):
    """A block we cannot translate must leave a trace, it used to vanish silently."""
    with caplog.at_level(logging.WARNING):
        assert AnthropicBatchAdapter._normalize_content_blocks([part]) == []

    assert "Dropping content block" in caplog.text


def test_anthropic_keeps_supported_content_blocks_without_warning(caplog):
    blocks = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]

    with caplog.at_level(logging.WARNING):
        normalized = AnthropicBatchAdapter._normalize_content_blocks(blocks)

    assert normalized == [
        {"type": "text", "text": "hi"},
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.png"},
        },
    ]
    assert caplog.text == ""


def test_anthropic_build_request_injects_structured_output_format(monkeypatch):
    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "expected_schema": ["question", "answer"],
    }

    request = adapter.build_request(custom_id="row-002", payload=payload, config=config)

    assert "expected_schema" not in request["params"]
    assert request["params"]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        }
    }


def test_anthropic_estimate_request_bytes_matches_utf8_json_size(monkeypatch):
    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    _patch_anthropic_client(monkeypatch, FakeAnthropic)
    adapter = AnthropicBatchAdapter()

    request = {"custom_id": "accented", "params": {"message": "caf\u00e9"}}
    expected = len(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    assert adapter.estimate_request_bytes(request) == expected


def test_anthropic_submit_chunk_uses_messages_batches(monkeypatch):
    captured = {}

    class FakeBatches:
        # Same signature as the SDK, so passing anything else fails here.
        def create(self, *, requests):
            captured["create_kwargs"] = {"requests": requests}
            return _message_batch(batch_id="batch_123", processing_status="in_progress")

    class FakeMessages:
        def __init__(self):
            self.batches = FakeBatches()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()
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
    assert captured["create_kwargs"]["requests"] == requests
    assert raw_result["id"] == "batch_123"
    assert raw_result["status"] == "in_progress"


def test_anthropic_check_batch_status_reads_the_env_api_key(monkeypatch):
    captured = {}

    class FakeBatches:
        def retrieve(self, message_batch_id):
            return _message_batch(batch_id=message_batch_id)

    class FakeMessages:
        def __init__(self):
            self.batches = FakeBatches()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-test-key")
    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()

    result = adapter.check_batch_status(provider_batch_id="batch_env", config=config)

    assert captured["client_kwargs"]["api_key"] == "env-test-key"
    assert result.provider_batch_id == "batch_env"
    # The status comes from processing_status, MessageBatch has no 'status' field.
    assert result.status == "completed"


def test_anthropic_retrieve_results_joins_the_generated_text_blocks(monkeypatch):
    class FakeBatches:
        def retrieve(self, message_batch_id):
            return _message_batch(batch_id=message_batch_id)

        def results(self, message_batch_id):
            return _batch_results(
                [
                    {
                        "custom_id": "c1",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "id": "msg_1",
                                "type": "message",
                                "role": "assistant",
                                "model": "claude-haiku-4-5",
                                "content": [
                                    {"type": "text", "text": "Hello"},
                                    {"type": "text", "text": " world"},
                                ],
                                "usage": {"input_tokens": 1, "output_tokens": 2},
                            },
                        },
                    }
                ]
            )

    class FakeMessages:
        def __init__(self):
            self.batches = FakeBatches()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_1", config=config)

    assert len(rows) == 1
    assert rows[0]["custom_id"] == "c1"
    assert rows[0]["generated_text"] == "Hello world"


@pytest.mark.parametrize(
    ("result", "expected_message"),
    [
        (
            {
                "type": "errored",
                "error": {
                    "type": "error",
                    "request_id": "req_1",
                    "error": {"type": "invalid_request_error", "message": "boom"},
                },
            },
            "boom",
        ),
        ({"type": "canceled"}, "Request canceled"),
        ({"type": "expired"}, "Request expired"),
    ],
)
def test_anthropic_retrieve_results_reports_failed_rows(
    monkeypatch, result, expected_message
):
    """Failures must surface a message, not an empty row that looks successful."""

    class FakeBatches:
        def retrieve(self, message_batch_id):
            return _message_batch(batch_id=message_batch_id)

        def results(self, message_batch_id):
            return _batch_results([{"custom_id": "c1", "result": result}])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(batches=FakeBatches())

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    rows = AnthropicBatchAdapter().retrieve_results(
        provider_batch_id="batch_1", config=AnthropicBatchConfig()
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_message"] == expected_message
    assert "generated_text" not in rows[0]


def test_anthropic_keeps_generated_custom_ids_unchanged():
    """Ids built by the processor are already valid, the hash branch is only a fallback."""
    processor_ids = [
        "answer-text-1",
        "formatted_answer-text-12",
        "caption-multimodal-1",
    ]
    for custom_id in processor_ids:
        assert AnthropicBatchAdapter._normalize_custom_id(custom_id) == custom_id

    # An output name with an illegal character still falls back to the hash form.
    assert AnthropicBatchAdapter._normalize_custom_id("my.answer-text-1").startswith(
        "my_answer-text-1-"
    )


def test_resolve_single_provider_config_accepts_anthropic():
    config = resolve_single_provider_config({"provider": "anthropic"})
    assert isinstance(config, AnthropicBatchConfig)


def test_factory_raises_when_anthropic_api_key_env_var_is_missing(monkeypatch):
    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    _patch_anthropic_client(monkeypatch, FakeAnthropic)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        BatchAdapterRegistry.create(AnthropicBatchConfig())
