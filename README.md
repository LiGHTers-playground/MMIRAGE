<h1 align="center">

![image](https://raw.githubusercontent.com/LiGHTers-playground/MMIRAGE/main/mmirage_logo_with_text.png)

</h1>

<p align="center">
  <a href="https://github.com/LiGHTers-playground/MMIRAGE/actions/workflows/ruff.yml"><img src="https://github.com/LiGHTers-playground/MMIRAGE/actions/workflows/ruff.yml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/LiGHTers-playground/MMIRAGE/actions/workflows/tests.yml"><img src="https://github.com/LiGHTers-playground/MMIRAGE/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

# MMIRAGE

MMIRAGE, which stands for **M**odular **M**ultimodal **I**ntelligent **R**eformatting and **A**ugmentation **G**eneration **E**ngine, is an advanced platform designed to streamline the processing of datasets using generative models, including vision-language models (VLMs). It is engineered to handle large-scale data reformatting and augmentation tasks with efficiency and precision. By leveraging state-of-the-art generative models, MMIRAGE enables users to perform complex dataset transformations, ensuring compatibility across various formats and schemas. Its multi-node support and parallel processing capabilities make it an ideal choice for scenarios demanding substantial computational power, such as distributed training and inference workflows. MMIRAGE not only simplifies the integration of powerful language models but also provides a customizable framework for diverse use cases, from reformatting conversational datasets to generating Q/A pairs from plain text.

## How to install

To install the library, clone it from GitHub and install it with pip.

### Base install

The base install does **not** include the local SGLang runtime:

```bash
git clone git@github.com:EPFLiGHT/MMIRAGE.git
cd MMIRAGE
pip install -e .
```

### GPU install (SGLang-backed `llm` processor)

MMIRAGE requires a **CUDA-enabled PyTorch installation** before installing the GPU extra.

Install a PyTorch build matching your CUDA runtime (example for CUDA 12.9):

```bash
pip install --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output should include:

- `+cu129`
- `12.9`
- `True` (when run on a GPU node)

Then install MMIRAGE GPU support:

```bash
pip install -e ".[gpu]"
```

Install the SGLang diffusion extra when MMIRAGE should launch an image-generation server:

```bash
pip install -e ".[image_gen]"
```

> **Note:** On some platforms (e.g. clusters or ARM/aarch64), `pip install -e ".[gpu]"` alone may resolve to a CPU-only PyTorch build.

### Environment file

For testing and scripts that make use of the library, it is advised to create a `.env` file:

```bash
./scripts/generate_env.sh
```

## Docker

The `docker-compose.yml` defines two services, `mmirage` (GPU) and `mmirage-cpu`.

### Prebuilt images

Prebuilt images are published to GHCR for each push to `main`:

- `ghcr.io/epflight/mmirage:latest-gpu` (linux/amd64)
- `ghcr.io/epflight/mmirage:latest-cpu` (linux/amd64, linux/arm64)

How to use them:

```bash
# GPU
docker pull ghcr.io/epflight/mmirage:latest-gpu
docker run --rm -it --gpus all ghcr.io/epflight/mmirage:latest-gpu

# CPU
docker pull ghcr.io/epflight/mmirage:latest-cpu
docker run --rm -it ghcr.io/epflight/mmirage:latest-cpu
```

### GPU

The container requires an NVIDIA GPU. The `docker-compose.yml` is configured to request GPU access, but the host must have:
- NVIDIA GPU drivers installed
- NVIDIA Container Toolkit / `nvidia-container-runtime` configured for Docker
- A recent Docker Engine and Docker Compose version with GPU support enabled

Commands:

```bash
# Build
docker compose build mmirage

# Run
docker compose run --rm -it mmirage
```

### CPU-only

The CPU image installs MMIRAGE without the GPU extra. It is suitable for workflows that do not instantiate the SGLang-backed `llm` processor, such as the `batch_api` processor. See `configs/config_mock_openai_batch.yaml` and `configs/config_mock_anthropic_batch.yaml`.

Commands:

```bash
# Build
docker compose build mmirage-cpu

# Run
docker compose run --rm -it mmirage-cpu
```

## Key features

- **Multimodal Support**: Process both text and images with vision-language models
- Easily configurable with a YAML file which configures the following parameters:
    - The prompt to the LLM (using Jinja2 templating)
    - Variables with the name and their JMESPath key to a JSON
    - Image inputs for multimodal processing
- Parallelizable with multi-node support
    - The training pipeline uses distributed inference with sharding
- Support a variety of LLMs, VLMs (Vision-Language Models), and image generation models
- Support any dataset schemas (configurable with the YAML format)
- The ability to either output a JSON (or any other structured format) or plain text
- Modular architecture with pluggable processors, loaders, and writers

## Example usage

### Running (single command)

Run the pipeline via the CLI. Retry behavior is driven by your YAML config:

- `execution_params.retry: true` → automatically retries failed shards until completion or `max_retries`
- `execution_params.retry: false` → submits/runs once; you can later trigger retries via `check`
- `execution_params.merge: true` → after a successful run, automatically merges shard outputs

```bash
mmirage run --config configs/config_mock.yaml
```

To check status only:

```bash
mmirage check --config configs/config_mock.yaml
```

To check status and submit retries for failed shards:

```bash
mmirage check --config configs/config_mock.yaml --retry
```

To merge shards from the CLI directly:

```bash
mmirage merge --config configs/config_mock.yaml
```

To merge shards without a config file (input directory + output directory only):

```bash
mmirage merge-dir --input-dir /path/to/shards --output-dir /path/to/merged
```

`--input-dir` can point either to a single dataset directory that contains `shard_*`
folders, or to a parent directory containing multiple dataset subdirectories.
If `shard_*` folders are present directly in `--input-dir`, MMIRAGE merges that
root dataset directly and ignores nested internal folders.

For multiple datasets, you can also choose a shared merge root:

```bash
mmirage merge --config configs/config_mock.yaml --output-root /path/to/merged
```

MMIRAGE still keeps datasets separate by creating one subdirectory per dataset under the root.

### Text-only: Reformatting dataset

Suppose you have a dataset with samples of the following format

```json
{
    "conversations" : [{"role": "user", "content": "Describe the image"}, {"role": "assistant", "content": "This is a badly formmatted answer"}],
    "modalities" : ["<the images>"]
}
```

The dataset contains assistant answers that are badly formatted. The goal would be to use a LLM to format our answer in Markdown. With MMIRAGE, it would be as simple as defining a YAML configuration file:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen3-8B
      tp_size: 4
      trust_remote_code: true
    default_sampling_params:
      temperature: 0.1
      top_p: 1.0
      max_new_tokens: 384

loading_params:
  state_dir: /path/to/state/dir
  datasets:
    - path: /path/to/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 64

processing_params:
  inputs:
    - name: assistant_answer
      key: conversations[1].content
    - name: user_prompt
      key: conversations[0].content
    - name: modalities
      key: modalities

  outputs:
    - name: formatted_answer
      type: llm
      output_type: plain
      prompt: |
        Reformat the answer in a markdown format without adding anything else:
        {{ assistant_answer }}

  remove_columns: false
  output_schema:
    conversations:
      - role: user
        content: "{{ user_prompt }}"
      - role: assistant
        content: "{{ formatted_answer }}"
    modalities: "{{ modalities }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

Configuration explanation:

- `processors`: List of processor configurations. Currently supports :
  - `llm` (text/VLM generation, run locally with SGLang).
  - `batch_api` (text/VLM generation submitted to a provider batch API, see [Batch API](docs/batch_api.md)).
  - `image_gen` (text-to-image generation).
- `loading_params`: Parameters for loading and sharding datasets.
  - `state_dir`: Optional shared directory for shard status/retry state. Defaults to `~/.cache/MMIRAGE/state_dir`.
  - `datasets`: List of dataset configurations with path, type, and output directory.
- `processing_params`:
  - `inputs`: Variables extracted from the input dataset using JMESPath queries.
  - `outputs`: Variables created by processors. Prompts use Jinja2 templating (`{{ variable }}`).
  - `output_schema`: Defines the structure of output samples.
- `execution_params`:
  - `mode`: "local" to run shard processing in the current Python environment or "slurm" to run through SLURM by submitting an sbatch array job.
  - `retry`: If true, MMIRAGE automatically retries failed shards until they succeed or `max_retries` is reached. If false, the pipeline runs/submits once, and retries can be triggered later via the check/retry CLI commands.
  - `merge`: If true, MMIRAGE merges shard outputs after a successful `run`. Merged datasets are written under each dataset `output_dir` in a `merged` subdirectory.

Merge output behavior with multiple datasets:
- Default (`run` with `execution_params.merge: true`, or `merge` without `--output-root`): each dataset is merged to its own `<dataset.output_dir>/merged`.
- Shared root (`merge --output-root ...`): one merged subdirectory is created per dataset under the root.

### Multimodal: Processing images with VLMs

MMIRAGE supports multimodal processing with vision-language models:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen2-VL-7B-Instruct
      tp_size: 4
      trust_remote_code: true
    chat_template: qwen2-vl  # Required for VLMs
    default_sampling_params:
      temperature: 0.1
      top_p: 0.95
      max_new_tokens: 768

loading_params:
  state_dir: path/to/state/dir
  datasets:
    - path: /path/to/image/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 32

processing_params:
  inputs:
    - name: medical_image
      key: image
      type: image  # Mark as image input
      image_base_path: /path/to/images  # Base directory for relative paths
    - name: original_caption
      key: caption
      type: text

  outputs:
    - name: enhanced_caption
      type: llm
      output_type: plain
      prompt: |
        Describe the medical image in detail.
        Original caption for context: {{ original_caption }}

  remove_columns: false
  output_schema:
    image: "{{ medical_image }}"
    caption: "{{ enhanced_caption }}"
    original_caption: "{{ original_caption }}"

execution_params:
  mode: local
  retry: false
```

### Image generation: Text-to-image pipeline

MMIRAGE supports image generation through an already-running HTTP server:

```yaml
processors:
  - type: image_gen
    backend: external
    external:
      base_url: http://127.0.0.1:30010/v1
      timeout_seconds: 900
      max_concurrent_requests: 4
      request_model: null
    default_sampling_params:
      num_inference_steps: 20
      guidance_scale: 7.5
    output_dir: /path/to/generated/images
    file_format: png

loading_params:
  state_dir: /path/to/state/dir
  datasets:
    - path: /path/to/prompts.jsonl
      type: JSONL
      output_dir: /path/to/output/shards
  num_shards: 1
  shard_id: 0
  batch_size: 8

processing_params:
  inputs:
    - name: prompt_text
      key: text

  outputs:
    - name: generated_image
      type: image_gen
      output_mode: path          # "path" or "pil"
      filename_template: "generated_{{ __shard_id }}_{{ __sample_index }}_{{ __source_hash }}"
      width: 512
      height: 512
      prompt: |
        Create an illustration of:
        {{ prompt_text }}

  remove_columns: false
  output_schema:
    text: "{{ prompt_text }}"
    image: "{{ generated_image }}"

execution_params:
  mode: local
  retry: false
```

Install optional image generation dependencies before running this config:

```bash
pip install -e .[image_gen]
```

Key multimodal features:
- `chat_template`: Specify the VLM chat template (e.g., `qwen2-vl`)
- `type: image`: Mark input variables as images
- `image_base_path`: Base directory for resolving relative image paths
- Supports PIL Images, URLs, and file paths

### Batch provider mode (OpenAI or Anthropic)

Use the `batch_api` processor, give its outputs `type: batch_api`, and supply a
provider config. The same prompt/template logic applies, but requests are
submitted asynchronously via the provider batch API. Set the provider API key in
the environment (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).

```yaml
processors:
  - type: batch_api
    provider: anthropic
    model: claude-haiku-4-5
    max_chunk_bytes: 52428800
    metadata_output_path: tests/output/batch_metadata.jsonl

...
```

For OpenAI, set `provider: openai` and use `OPENAI_API_KEY` instead.

Batch runs are asynchronous: `run` submits and exits, then `mmirage check` polls
and `mmirage merge` retrieves — see [Batch API](docs/batch_api.md).

### Dry-run / Export prompts

MMIRAGE supports a simple dry-run mode that exports the provider-ready
prompt payloads to disk instead of submitting them to the provider. This is
useful for cost estimation, inspection, and offline review.

- CLI: pass `--export-prompts <path>` to `mmirage run`. When set, MMIRAGE appends
  all emitted requests to a single JSONL file and skips all network submission
  calls.
- Path behavior:

  - If `<path>` ends with `.jsonl`, the run id is inserted before the extension,
    so `/tmp/export.jsonl` becomes `/tmp/export.<run_id>.jsonl`.
  - Otherwise, MMIRAGE treats `<path>` as a directory and creates one file like
    `<path>/exported_prompts.<run_id>.jsonl`.

- Format: each line is `{"batch_id": ..., "request": ...}`, where `request` is the
  untouched payload that would have been submitted and `batch_id` identifies which
  emitted batch chunk produced the row.
- Credentials: when `--export-prompts` is used, MMIRAGE does not require
  provider credentials (API keys) because network submission is intentionally
  skipped. This makes it convenient to export prompts without setting up
  provider credentials.
- Metadata: MMIRAGE still writes the metadata receipts to `metadata_output_path`
  as configured on your `batch_api` processor, marked `.dry-run` in their name and
  skipped by `mmirage check`; export mode only short-circuits the submission step.

Example:

```bash
mmirage run --config configs/config_mock_openai_batch.yaml \
  --export-prompts /tmp/mmirage_export.jsonl
```

This writes all emitted chunks into `/tmp/mmirage_export.<run_id>.jsonl`, with one
row per request and a `batch_id` field for chunk traceability.

### Benchmarking shard performance

Pass `--stats` to `run` or `submit` to enable per-shard benchmarking. This activates GPU
utilization polling and throughput tracking on compute nodes — disabled by default to
avoid unnecessary overhead.

```bash
# Local run with stats collection
mmirage run --config configs/config_mock.yaml --stats

```

After the run completes, inspect the results with:

```bash
mmirage stats --config configs/config_mock.yaml
```

This prints a JSON report with per-shard details and an aggregate summary:

```json
{
  "per_shard": [
    {
      "shard_id": 0,
      "status": "success",
      "started_at": "2026-04-30T10:00:00",
      "finished_at": "2026-04-30T10:01:05",
      "stats": {
        "runtime_seconds": 65.2,
        "runtime_human": "1m 5s",
        "rows_processed": 1024,
        "rows_written": 1024,
        "rows_filtered": null,
        "rows_dropped_by_error": null,
        "throughput_rows_per_sec": 15.7,
        "gpu_util_mean": 88.4,
        "gpu_util_min": 72.0,
        "gpu_util_max": 98.0,
        "gpu_util_samples": 13,
        "input_tokens": 512000,
        "output_tokens": 196608,
        "num_gpus": 4,
        "tokens_per_sec_per_gpu": 753.1,
        "gpu_days_per_billion_tokens": 0.0015
      }
    }
  ],
  "aggregate": {
    "total_shards": 1,
    "completed_shards": 1,
    "total_rows_processed": 1000,
    "total_rows_written": 1000,
    "total_rows_filtered": null,
    "total_rows_dropped_by_error": null,
    "wall_clock_runtime_seconds": 133.04,
    "wall_clock_runtime_human": "2m 13s",
    "sum_shard_runtime_seconds": 133.04,
    "sum_shard_runtime_human": "2m 13s",
    "min_shard_runtime_seconds": 133.04,
    "min_shard_runtime_human": "2m 13s",
    "max_shard_runtime_seconds": 133.04,
    "max_shard_runtime_human": "2m 13s",
    "overall_throughput_rows_per_sec": 7.52,
    "mean_gpu_util_pct": 86.2,
    "num_gpus": 4,
    "total_input_tokens": 146214,
    "total_output_tokens": 1022046,
    "sum_model_load_seconds": 38.272,
    "sum_inference_runtime_seconds": 94.768,
    "tokens_per_sec_per_gpu": 10784.72,
    "gpu_days_per_billion_tokens": 1.0732
  }
}
```

Key metrics:
- **`runtime_seconds`** / **`runtime_human`**: time from when the shard started on the cluster (after dispatch), excluding queue wait time.
- **`overall_throughput_rows_per_sec`**: total rows / wall-clock time across all shards running in parallel.
- **`mean_gpu_util_pct`**: mean percentage GPU utilization across shards.
- **`tokens_per_sec_per_gpu`**: output tokens generated per second per GPU — the primary throughput metric used by frameworks such as [DataTrove](https://github.com/huggingface/datatrove).
- **`gpu_days_per_billion_tokens`**: total GPU-days consumed to generate 1 billion output tokens — useful for cost and scaling comparisons across different hardware configurations.
- Token metrics are `null` when no LLM processor was active, and GPU stats are `null` when `nvidia-smi` is unavailable or `--stats` was not passed.

Reference benchmark:
- [DataTrove Benchmark](https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark)

The config `configs/config_benchmark_datatrove.yaml` mirrors the DataTrove inference benchmark conditions:

| Setting | Value |
|---|---|
| Dataset | `simplescaling/s1K-1.1` (train split, 1 000 samples) |
| Prompt | raw `question` field, no system prompt |
| Output | up to 1 024 tokens per sample |
| Context | 2 048-token model max context |
| Model | `Qwen/Qwen3-4B` (DataTrove baseline: tp=1 on a single GPU) |

Download the dataset before running:

```python
from datasets import load_dataset

ds = load_dataset("simplescaling/s1K-1.1", split="train")
ds.save_to_disk("data/s1K-1.1")
```

Then run with stats collection enabled:

```bash
mmirage run --config configs/config_benchmark_datatrove.yaml --stats
```

Inspect results:

```bash
mmirage stats --config configs/config_benchmark_datatrove.yaml
```

## Architecture

MMIRAGE uses a modular architecture:

```
mmirage/
├── config/           # Configuration loading and validation
├── core/
│   ├── loader/       # Dataset loaders (JSONL, HuggingFace)
│   ├── process/      # Processors (LLM, etc.) and variable system
│   │   └── processors/
│   │       └── llm/  # LLM processor with multimodal support
│   └── writer/       # Output rendering with Jinja2
├── shard_process.py  # Main processing script
└── merge_shards.py   # Shard merging utility
```

## Useful tools

- Jinja2 for template processing: [link](https://jinja.palletsprojects.com/en/stable/)
- JMESPath for JSON queries: [link](https://jmespath.org/)
- SGLang for fast inference: [link](https://github.com/sgl-project/sglang)
- Performance paper: [link](https://arxiv.org/abs/2408.02442)
- DataTrove Benchmark: [link](https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark)
