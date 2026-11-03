from __future__ import annotations

import copy

import pytest

from mas_loop_rl.build_editor_data_from_logs import _pad_complete_groups_to_expected_rows


def _rows(group_count: int, group_size: int) -> list[dict]:
    rows = []
    for group_index in range(group_count):
        uid = f"editor-replay:group-{group_index}"
        for member_index in range(group_size):
            rows.append(
                {
                    "extra_info": {
                        "editor_group_uid": uid,
                        "index": len(rows),
                        "rollout_sample_index": member_index,
                    }
                }
            )
    return rows


def test_exact_padding_preserves_complete_groups_and_unique_uids() -> None:
    group_size = 16
    original = _rows(group_count=224, group_size=group_size)
    original_snapshot = copy.deepcopy(original)

    padded = _pad_complete_groups_to_expected_rows(original, expected_rows=4096, group_size=group_size)

    assert len(padded) == 256 * group_size
    assert original[: len(original_snapshot)] == original_snapshot
    groups = [padded[i : i + group_size] for i in range(0, len(padded), group_size)]
    group_uids = [group[0]["extra_info"]["editor_group_uid"] for group in groups]
    assert len(group_uids) == len(set(group_uids)) == 256
    for group in groups:
        assert len({row["extra_info"]["editor_group_uid"] for row in group}) == 1
    for group in groups[224:]:
        assert all(row["extra_info"]["is_padding_duplicate"] is True for row in group)
        assert all(row["extra_info"]["padding_duplicate_of_group"] for row in group)


def test_exact_padding_rejects_rows_above_target() -> None:
    with pytest.raises(ValueError, match="refusing to truncate"):
        _pad_complete_groups_to_expected_rows(_rows(257, 16), expected_rows=4096, group_size=16)


def test_exact_padding_rejects_partial_groups() -> None:
    rows = _rows(2, 16)[:-1]
    with pytest.raises(ValueError, match="only complete groups"):
        _pad_complete_groups_to_expected_rows(rows, expected_rows=48, group_size=16)


def test_exact_padding_requires_divisible_target() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _pad_complete_groups_to_expected_rows(_rows(2, 16), expected_rows=47, group_size=16)
