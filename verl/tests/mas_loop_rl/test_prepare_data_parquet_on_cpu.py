from __future__ import annotations

import pandas as pd

from mas_loop_rl.prepare_data import (
    _sanitize_rows_for_parquet,
    _write_parquet_via_local_staging,
)


def test_nested_mixed_leaf_types_write_without_flattening_extra_info(tmp_path):
    rows = [
        {
            "data_source": "mas_loop_editor",
            "extra_info": {
                "index": 0,
                "enabled": True,
                "reward": 0.25,
                "src_mas": {"agents": [{"id": "solver", "budget": 1}]},
                "src_trajectory": {
                    "success": True,
                    "metadata": {"judge": {"tokens": 17}},
                },
            },
        },
        {
            "data_source": "mas_loop_editor",
            "extra_info": {
                "index": 1,
                "enabled": False,
                "reward": 0.5,
                "src_mas": {"agents": [{"id": "solver", "budget": 2}]},
                "src_trajectory": {
                    "success": False,
                    "metadata": {"judge": {"tokens": "..."}},
                },
            },
        },
    ]

    normalized = _sanitize_rows_for_parquet(rows)
    assert isinstance(normalized[0]["extra_info"], dict)
    assert isinstance(normalized[0]["extra_info"]["src_mas"], dict)
    assert normalized[0]["extra_info"]["src_trajectory"]["metadata"]["judge"]["tokens"] == "17"
    assert normalized[1]["extra_info"]["src_trajectory"]["metadata"]["judge"]["tokens"] == "..."
    assert normalized[0]["extra_info"]["enabled"] is True
    assert normalized[0]["extra_info"]["index"] == 0
    assert normalized[0]["extra_info"]["reward"] == 0.25

    destination = tmp_path / "mixed.parquet"
    _write_parquet_via_local_staging(pd.DataFrame(normalized), destination)
    restored = pd.read_parquet(destination)
    assert len(restored) == 2
    extra = restored.iloc[0]["extra_info"]
    assert isinstance(extra, dict)
    assert isinstance(extra["src_mas"], dict)
    assert extra["src_trajectory"]["metadata"]["judge"]["tokens"] == "17"
    assert extra["enabled"] is True


def test_mixed_list_element_kinds_are_normalized_at_the_element_path():
    rows = [
        {"extra_info": {"diagnosis": {"evidence": [{"location": 3}]}}},
        {"extra_info": {"diagnosis": {"evidence": [{"location": "agent-3"}]}}},
    ]
    normalized = _sanitize_rows_for_parquet(rows)
    assert normalized[0]["extra_info"]["diagnosis"]["evidence"][0]["location"] == "3"
    assert normalized[1]["extra_info"]["diagnosis"]["evidence"][0]["location"] == "agent-3"
