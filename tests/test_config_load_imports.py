"""Loading a config must not import a provider SDK or the custom worker pool."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIGS = sorted((REPO_ROOT / "configs").glob("config_mock*.yaml"))
HEAVY_MODULES = ("openai", "anthropic", "pebble")

# Runs in a fresh interpreter: pytest's own imports already pollute this one.
PROBE = """
import json
import sys

from mmirage.config.utils import load_mmirage_config

load_mmirage_config(sys.argv[1])
print(json.dumps([name for name in {heavy!r} if name in sys.modules]))
"""


@pytest.mark.parametrize("config_path", MOCK_CONFIGS, ids=lambda path: path.name)
def test_loading_a_config_imports_no_sdk(config_path):
    env = {
        **os.environ,
        "OPENAI_API_KEY": "dummy",
        "ANTHROPIC_API_KEY": "dummy",
        "USER": os.environ.get("USER", "mmirage"),
    }

    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(heavy=HEAVY_MODULES), str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == []
