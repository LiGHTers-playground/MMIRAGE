"""Batch requests after a filter step record the source rows they came from."""

import json

from batch_fixtures import UnitBatchConfig

from mmirage import shard_process
from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.process.processors.batch_api.config import (
    BatchApiOutputVar,
    BatchApiProcessorConfig,
)
from mmirage.core.process.processors.filter.config import FilterStep
from mmirage.core.process.variables import InputVar
from mmirage.core.writer.renderer import TemplateRenderer


def test_receipts_point_at_the_rows_that_survived_the_filter(tmp_path, unit_provider):
    mapper = MMIRAGEMapper(
        processor_configs=[
            BatchApiProcessorConfig(
                type="batch_api",
                provider_config=UnitBatchConfig(
                    provider="unit", metadata_output_path=str(tmp_path / "meta.jsonl")
                ),
            )
        ],
        input_vars=[InputVar(name="text", key="text")],
        output_vars=[
            FilterStep(predicate="text != 'drop'"),
            BatchApiOutputVar(name="caption", type="batch_api", prompt="{{ text }}"),
        ],
    )
    renderer = TemplateRenderer({"caption": "{{ caption }}"})

    # Two batches of a split that starts at row 10 of the shard (earlier splits
    # or datasets hold rows 0-9); the filter drops rows 11 and 13.
    for texts, indices in ((["a", "drop", "c"], [0, 1, 2]), (["drop", "e"], [3, 4])):
        shard_process.rewrite_batch(
            {"text": texts},
            indices,
            mapper=mapper,
            renderer=renderer,
            kept=[],
            row_offset=10,
        )
    mapper.finalize_processors()

    receipts = [
        json.loads(line)
        for path in sorted(tmp_path.glob("meta.*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    mapping = {
        cid: idx
        for r in receipts
        for cid, idx in r["custom_id_to_source_index"].items()
    }
    assert sorted(mapping.values()) == [10, 12, 14]
