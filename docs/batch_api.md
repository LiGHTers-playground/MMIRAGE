# 🗂️ Batch API

This page explains how to run MMIRAGE inference asynchronously using a provider batch API — OpenAI or Anthropic are built in — which is useful for large-scale processing at lower cost.

---

## Overview

Use the `batch_api` processor instead of `llm`, and give its outputs `type: batch_api`.

1. **Request serialization:** MMIRAGE serializes inference requests into JSONL chunks.
2. **Batch submission:** Each chunk is uploaded and submitted as a provider batch job.
3. **Execution completion:** The pipeline run exits immediately after submission, saving placeholder values (e.g. `__BATCH_SUBMITTED__:<output_name>-<modality>-<request_number>`) in the output dataset shards.
4. **Asynchronous retrieval:** The user manually polls status and downloads/merges the completed results using separate Python utility modules once the provider completes the batch jobs.

This mode is useful when:

- you do not have access to GPUs locally or on a cluster
- you are processing very large datasets where cost matters
- the pipeline tolerates asynchronous completion (up to 24 h per batch)

---

## When to use each mode

| Criterion | Local (SGLang) | Batch API |
|---|---|---|
| Latency | Low | High (up to 24 h) |
| Cost | GPU compute cost | batch API price |
| GPU requirement | Required | Not required |
| Vision / multimodal | Depends on model | Depends on model |
| Streaming output | ✓ | ✗ |

---

## Configuration

Declare a `batch_api` processor and its provider settings in your YAML config. `provider` selects the built-in adapter (`openai` or `anthropic`) and determines which provider-specific fields are accepted.

```yaml
processors:
  - type: batch_api
    provider: openai
    model: gpt-4o-mini
    max_chunk_bytes: 52428800      # Max bytes per uploaded JSONL file (50 MB)
    max_requests_per_chunk: 50000  # Max requests per batch job
    metadata_output_path: /path/to/batch_metadata.jsonl
    completion_window: 24h
    base_url: https://api.openai.com/v1
    oversized_request_policy: isolate  # isolate | reject
```

### Shared field reference

These fields apply to every provider.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | — | Provider identifier: `"openai"` or `"anthropic"`. |
| `max_chunk_bytes` | `int` | `52428800` | Maximum JSONL file size per batch upload (50 MB). |
| `max_requests_per_chunk` | `int` | `null` | Optional hard cap on number of requests in a chunk. |
| `metadata_output_path` | `str` | `""` | Base path for batch job metadata receipt files. Suffixes like `.text.<run_id>.jsonl` and `.multimodal.<run_id>.jsonl` will be appended. |
| `oversized_request_policy` | `str` | `"isolate"` | Policy for requests exceeding `max_chunk_bytes`: `"isolate"` (dedicated chunk) or `"reject"` (fail fast). |

### `provider: openai`

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"gpt-4.1-mini"` | Model name for chat completion requests. |
| `batch_endpoint` | `str` | `"/v1/chat/completions"` | Target endpoint used by OpenAI batch jobs. |
| `completion_window` | `str` | `"24h"` | OpenAI batch completion window (only `"24h"` is supported). |
| `base_url` | `str` | `null` | Optional base URL, useful for API-compatible gateways. |
| `metadata` | `dict` | `{}` | Key-value pairs sent on batch creation. |

### `provider: anthropic`

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-haiku-4-5"` | Model name used in each Messages request body. |
| `max_tokens` | `int` | `8192` | Maximum tokens for the generated response. |
| `temperature` | `float` | `null` | Sampling temperature, in `[0, 1]`. Mutually exclusive with `top_p`. |
| `top_p` | `float` | `null` | Nucleus sampling probability, in `(0, 1]`. Mutually exclusive with `temperature`. |
| `base_url` | `str` | `null` | Optional base URL, useful for API-compatible gateways. |
| `timeout_seconds` | `float` | `null` | Optional request timeout. |

Setting both `temperature` and `top_p` is rejected at config load; setting neither leaves sampling at the provider default.

---

## API key

Each provider reads its API key from the environment. Set it before running:

```bash
export OPENAI_API_KEY=sk-...        # provider: openai
export ANTHROPIC_API_KEY=sk-ant-... # provider: anthropic
```

Keys cannot be supplied in the YAML config. The key is checked when the processor is built, so a missing one fails before the dataset is processed — except under `--export-prompts`, which skips the check because nothing is submitted.

---

## Images

Local image paths are read and base64-encoded into the request, so they count towards `max_chunk_bytes`. `http(s)` URLs are sent as-is and the provider fetches them.

Accepted image types depend on the provider. `provider: anthropic` only encodes `image/jpeg`, `image/png`, `image/gif` and `image/webp`, and raises before submission on anything else, so give your images a correct extension. `provider: openai` sends the type as-is and lets the provider reject it.

---

## Request chunking

MMIRAGE automatically splits requests into chunks that respect both `max_chunk_bytes` and `max_requests_per_chunk`.

For very large prompts (e.g. with long contexts), you may need to reduce `max_requests_per_chunk` so that individual chunks stay within the size limit. Set `oversized_request_policy: isolate` to submit oversized requests as a dedicated chunk, or `oversized_request_policy: reject` to fail fast on requests exceeding the limit.

---

## Batch Submission & Lifecycle Workflow

Running the batch pipeline is an asynchronous, three-step process:

### Step 1: Submit the Batch Jobs
Execute your MMIRAGE pipeline with a configuration that declares a `batch_api` processor:

```bash
mmirage run --config configs/batch_config.yaml
```

During this run, MMIRAGE maps over your datasets, generates request payloads, writes them to serialized JSONL chunks, and submits them to the provider batch API.
- The pipeline execution completes immediately after submission.
- The output files in the dataset's `output_dir` shards will contain temporary placeholder variables of the format `__BATCH_SUBMITTED__:<output_name>-<modality>-s<shard_id>-<request_number>`. The part after the prefix is the `custom_id` used in the receipt and in the provider results; every shard numbers its requests from 1, so the shard id is what keeps ids unique across shards.
- MMIRAGE generates **metadata receipt files** named `<metadata_output_path>.<modality>.<run_id>.jsonl` (e.g., `batch_metadata.text.abc123.jsonl`). These receipt files store the API batch IDs and map each API request's `custom_id` to its original dataset `source_index`. Each receipt also records the `shard_id` that wrote it; receipts from before this field existed are read as shard 0.

### Step 2: Check Batch Job Status
Because batch jobs run asynchronously on the provider's server and can take up to 24 hours to complete, monitor their status with `mmirage check`, which reports provider batch status instead of shard status when the config declares a `batch_api` processor:

```bash
mmirage check --config configs/batch_config.yaml
```

By default the metadata receipt files are resolved from your configuration. You can also specify them manually:

```bash
mmirage check \
  --config configs/batch_config.yaml \
  --metadata-path /path/to/batch_metadata.text.abc123.jsonl
```

### Step 3: Retrieve and Merge Results
Once all batch jobs show a status of `completed`, retrieve the generated outputs, map them back to their original row positions, and merge them into a single, ordered JSONL file with `mmirage merge`:

```bash
mmirage merge \
  --config configs/batch_config.yaml \
  --output-path /path/to/final_merged_output.jsonl
```

Just like step 2, the metadata receipts are located from the config unless given explicitly:

```bash
mmirage merge \
  --config configs/batch_config.yaml \
  --metadata-path /path/to/batch_metadata.text.abc123.jsonl \
  --output-path /path/to/final_merged_output.jsonl
```

The collector prints the run totals, and each merged row carries `input_tokens` and `output_tokens` when the provider reports usage. Token counts are unknown at submission time, so they never appear in the [benchmark report](benchmarking.md).

Each merged row also carries its `shard_id`, and rows are ordered by shard, then by `source_index` within the shard, since every shard counts its rows from 0. When a shard was retried, its rows appear in more than one receipt: the row from the receipt submitted last is kept.

---

## Dry run

```bash
mmirage run --config configs/batch_config.yaml --export-prompts /tmp/prompts.jsonl
```

Every provider-ready request is written to the given path instead of being submitted, so no API key is needed. Each line is `{"batch_id": ..., "request": ...}`, where `request` is the untouched payload and can be submitted as-is.

The run id is added to the file name, `/tmp/prompts.a4f9c2.jsonl` above, so two runs never land in the same file. A path without `.jsonl` is treated as a directory and gets `exported_prompts.<run_id>.jsonl`.

Receipts are still written, named `<metadata_output_path>.dry-run.<modality>.<run_id>.jsonl`, and `mmirage check` skips them.

Only `mode: local` is supported, `mode: slurm` refuses the run.

---

## Provider-Agnostic Architecture & Custom Providers

MMIRAGE's batch processing system is designed to be provider-agnostic. OpenAI and Anthropic ship built in; developers can add other providers (Mistral, private gateways, ...) by implementing and registering a provider configuration and an adapter.

### Extension Contracts

To integrate a new provider, you need to implement two classes:

1. **Provider Config Subclass**: Defines the configuration schema. Must inherit from `BatchProviderConfig` ([BatchProviderConfig](../src/mmirage/config/batch_provider.py)).
2. **Submission Adapter Subclass**: Implements request construction, size estimation, chunk submission, status checking, and result retrieval. Must inherit from `BatchSubmissionAdapter` ([BatchSubmissionAdapter](../src/mmirage/core/process/batch/adapter.py#L29)).

#### 1. Custom Provider Config

A custom provider configuration class extends `BatchProviderConfig` with fields specific to that provider:

```python
from dataclasses import dataclass
from mmirage.config.batch_provider import BatchProviderConfig


@dataclass
class MistralBatchConfig(BatchProviderConfig):
    provider: str = "mistral"
    model: str = "mistral-small-latest"
```

#### 2. Custom Submission Adapter

A custom adapter implements the core lifecycle logic for the custom provider:

```python
from typing import Any, Dict, Sequence
from mmirage.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from mmirage.config.batch_provider import BatchProviderConfig

class MistralBatchAdapter(BatchSubmissionAdapter):
    # Each key must be set as an <PROVIDER>_<KEY> environment variable, e.g. MISTRAL_API_KEY.
    # Non-alphanumeric characters in the provider name become '_': azure-openai -> AZURE_OPENAI_API_KEY.
    required_credentials = ("api_key",)

    def build_request(
        self,
        custom_id: str,
        payload: Dict[str, Any],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        # Formats the internal request payload into the provider's API request format
        ...
        return {
            "custom_id": custom_id,
            "params": {
                "model": config.model,
                "messages": payload["messages"],
            },
        }

    def estimate_request_bytes(self, request: Dict[str, Any]) -> int:
        # Returns the estimated serialized UTF-8 bytes for request size-based chunking
        import json

        return len(json.dumps(request).encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Dict[str, Any]],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        # Submits the chunk requests to the provider API and returns the raw response
        ...

    def parse_submission_result(
        self,
        raw_result: Dict[str, Any],
    ) -> BatchSubmissionResult:
        # Wraps the raw submission response in a normalized BatchSubmissionResult.
        # Map your provider statuses to 'completed', 'failed', 'in_progress' or
        # 'unknown', mmirage check reads them without knowing your vocabulary.
        return BatchSubmissionResult(
            provider_batch_id=raw_result["id"],
            status=self._normalize_status(raw_result["status"]),
            raw_response=raw_result,
        )

    def check_batch_status(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> BatchSubmissionResult:
        # Queries the provider and returns the latest status
        ...

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        # Downloads/retrieves completed outputs and normalizes each row.
        # Ensure text generations are mapped to the "generated_text" key so
        # the collector can reconstruct the original dataset rows neutrally.
        # Expose any reported usage as "input_tokens" and "output_tokens",
        # omitting both keys when the provider reports none.
        ...
```

### Registry Integration

Once you have defined your config and adapter classes, register them with the MMIRAGE batch system at runtime (typically inside your application's bootstrap or initialization code):

```python
from mmirage.core.process.batch.provider_resolution import BatchProviderConfigRegistry
from mmirage.core.process.batch.registry import BatchAdapterRegistry

# Register the provider configuration class
BatchProviderConfigRegistry.register("mistral", MistralBatchConfig)

# Register the provider submission adapter
BatchAdapterRegistry.register("mistral", MistralBatchAdapter)
```

### Config Usage

After registering your custom provider, you can reference it in your MMIRAGE pipeline YAML configuration:

```yaml
processors:
  - type: batch_api
    provider: mistral
    model: mistral-small-latest
    metadata_output_path: /scratch/mistral_meta.jsonl
```

---

## Complete example config

Shown for `provider: openai`; see `configs/config_mock_anthropic_batch.yaml` for the Anthropic equivalent.

```yaml
processors:
  - type: batch_api
    provider: openai
    model: gpt-4o-mini
    max_chunk_bytes: 52428800
    max_requests_per_chunk: 50000
    metadata_output_path: /scratch/batch_meta.jsonl
    completion_window: 24h
    base_url: https://api.openai.com/v1
    oversized_request_policy: isolate

loading_params:
  state_dir: /scratch/state
  datasets:
    - path: /data/my_dataset.jsonl
      type: JSONL
      output_dir: /scratch/output/shards
  num_shards: 1
  shard_id: 0
  batch_size: 512

processing_params:
  inputs:
    - name: question
      key: question

  outputs:
    - name: answer
      type: batch_api
      output_type: plain
      prompt: |
        Answer the following question concisely:
        {{ question }}

  output_schema:
    question: "{{ question }}"
    answer: "{{ answer }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

---

## See also

- [Concepts](concepts.md) — processor types
- [Configuration Reference](configuration.md) — full `batch_api` parameter reference
- [Pipeline](pipeline.md) — where batch inference fits in the data flow
- [CLI Reference](cli.md) — CLI command reference for local and SLURM pipeline execution
