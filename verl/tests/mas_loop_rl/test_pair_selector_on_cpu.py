from mas_loop_rl.pair_selector import deterministic_mas_cost, select_pair_winner


def _mas(budget: int = 1):
    return {
        "agents": [{"id": "a", "role": "solve", "budget": budget}],
        "edges": [],
        "aggregation": "a",
        "max_iterations": 1,
    }


def test_pair_selector_prefers_strict_task_improvement():
    result = select_pair_winner(
        source_score=0.0, updated_score=1.0, source_mas=_mas(), updated_mas=_mas(2),
    )
    assert result.selected_origin == "updated"
    assert result.selection_reason == "higher_task_score"
    assert result.is_strict_improvement


def test_pair_selector_keeps_source_on_equal_wrong_or_explicit_accept():
    tie = select_pair_winner(
        source_score=0.0, updated_score=0.0, source_mas=_mas(2), updated_mas=_mas(),
    )
    assert tie.selected_origin == "source"
    assert tie.selection_reason == "stable_source_tie"
    accepted = select_pair_winner(
        source_score=1.0, updated_score=1.0, source_mas=_mas(2), updated_mas=_mas(), explicit_accept=True,
    )
    assert accepted.selected_origin == "source"
    assert accepted.selection_reason == "explicit_accept"


def test_pair_selector_uses_deterministic_cost_only_after_equal_positive_quality():
    result = select_pair_winner(
        source_score=1.0, updated_score=1.0, source_mas=_mas(2), updated_mas=_mas(1),
    )
    assert deterministic_mas_cost(_mas(1)) < deterministic_mas_cost(_mas(2))
    assert result.selected_origin == "updated"
    assert result.selection_reason == "equal_quality_lower_cost"


def test_invalid_updated_pair_falls_back_to_source():
    result = select_pair_winner(
        source_score=0.5, updated_score=1.0, source_mas=_mas(), updated_mas=None, updated_valid=False,
    )
    assert result.selected_origin == "source"
    assert result.selection_reason == "updated_invalid"
