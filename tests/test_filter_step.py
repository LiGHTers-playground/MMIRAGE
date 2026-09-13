from dataclasses import fields

from mmirage.core.process.processors.batch_api.config import BatchApiOutputVar
from mmirage.core.process.processors.llm.config import LLMOutputVar
from mmirage.core.process.variables import OutputVar


def test_output_vars_are_available_at_map_time_by_default():
    assert OutputVar.available_at_map_time is True
    assert LLMOutputVar.available_at_map_time is True


def test_batch_api_output_is_deferred_until_merge():
    assert BatchApiOutputVar.available_at_map_time is False
    # A ClassVar, so it is not a dataclass field and dacite never sees it.
    assert "available_at_map_time" not in {f.name for f in fields(BatchApiOutputVar)}
