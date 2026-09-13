"""Configuration loading utilities for MMIRAGE pipeline."""

import os
from typing import Any, Dict, List, TypeAlias, Union, cast

import yaml
from dacite import Config, from_dict

# Register built-in processors/loaders.
#
# We import configuration modules (lightweight) here so the registries know how
# to construct config/output-var objects from YAML without importing heavy
# processor implementations (e.g. torch/transformers).
import mmirage.core.loader.jsonl  # noqa: F401
import mmirage.core.loader.local_hf  # noqa: F401
import mmirage.core.process.processors.batch_api.config  # noqa: F401
import mmirage.core.process.processors.custom.config  # noqa: F401
import mmirage.core.process.processors.image_gen.config  # noqa: F401
import mmirage.core.process.processors.llm.config  # noqa: F401
from mmirage.config.config import MMirageConfig
from mmirage.core.loader.base import BaseDataLoaderConfig, DataLoaderRegistry
from mmirage.core.process.base import BaseProcessorConfig, OutputVar, ProcessorRegistry
from mmirage.core.process.processors.image_gen.config import ImageOutputMode

EnvValue: TypeAlias = Union[str, List["EnvValue"], Dict[str, "EnvValue"]]


def load_mmirage_config(config_path: str) -> MMirageConfig:
    """
    Load MMIRAGE configuration from a YAML file.

    Supports environment variable expansion and dynamic processor/loader
    configuration based on registered types.

    Example config:

    processors:
      - type: llm
        server_args:
          model_path: Qwen/Qwen2-VL-7B-Instruct
          tp_size: 4
          trust_remote_code: true
        chat_template: qwen2-vl
        default_sampling_params:
          temperature: 0.1
          top_p: 0.9
          max_new_tokens: 1024

    loading_params:
      datasets:
        - path: /path/to/dataset.jsonl
          type: JSONL
          output_dir: /path/to/output
          image_base_path: /path/to/images
      num_shards: 4
      shard_id: 0
      batch_size: 64

    processing_params:
      inputs:
        - name: text
          key: text
        - name: image
          key: image_path
          type: image

      outputs:
        - name: formatted_answer
          type: llm
          output_type: JSON
          output_schema:
            - question
            - answer
          prompt: |
            Generate a Q&A pair from:
            {{ text }}

      remove_columns: True
      cast_images: True
      output_schema:
        conversations:
          - role: "user"
            content: "{{ formatted_answer.question }}"
          - role: "assistant"
            content: "{{ formatted_answer.answer }}"

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        MMirageConfig: Parsed and validated configuration object.
    """

    with open(config_path, "r") as f:
        cfg: EnvValue = yaml.safe_load(f) or {}

    def expand_env_vars(obj: EnvValue) -> EnvValue:
        if isinstance(obj, dict):
            return {key: expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj

    def image_output_mode_hook(value: Any) -> ImageOutputMode:
        if isinstance(value, ImageOutputMode):
            return value
        return ImageOutputMode(value)

    def processor_config_hook(data: Dict[str, Any]) -> BaseProcessorConfig:
        clz = ProcessorRegistry.get_config_cls(data["type"])
        from_raw = getattr(clz, "from_raw", None)
        if from_raw is not None:
            return from_raw(data)
        return from_dict(clz, data, config=config)

    def loader_config_hook(data: Dict[str, Any]) -> BaseDataLoaderConfig:
        clz = DataLoaderRegistry.get_config_cls(data["type"])
        return from_dict(clz, data, config=config)

    def output_var_hook(data: Dict[str, Any]) -> OutputVar:
        clz = ProcessorRegistry.get_output_var_cls(data["type"])
        return from_dict(clz, data, config=config)

    cfg = expand_env_vars(cfg)
    config = Config(
        type_hooks={
            ImageOutputMode: image_output_mode_hook,
            BaseProcessorConfig: processor_config_hook,
            BaseDataLoaderConfig: loader_config_hook,
            OutputVar: output_var_hook,
        }
    )
    cfg_obj = from_dict(MMirageConfig, cast(dict, cfg), config=config)

    return cfg_obj
