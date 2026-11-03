from __future__ import annotations

"""Dataset-neutral selection for a source/updated MAS execution pair.

The selector is deliberately separate from policy reward attribution.  It records
which *state* should be carried forward or inspected, while Orchestra is still
trained only on the source MAS tokens it actually sampled.
"""

from dataclasses import asdict, dataclass
from typing import Any

SELECTOR_VERSION = "mas_pair_selector_v1"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def trajectory_confidence(trajectory: dict[str, Any] | None) -> float:
    """Extract an optional confidence signal without benchmark-specific logic."""
    metadata = trajectory.get("metadata", {}) if isinstance(trajectory, dict) else {}
    judge = metadata.get("judge", {}) if isinstance(metadata, dict) else {}
    for container in (judge, metadata):
        if not isinstance(container, dict):
            continue
        for key in ("confidence", "score_confidence", "judge_confidence", "probability"):
            if key in container:
                return _finite_float(container.get(key), 0.0)
    return 0.0


def deterministic_mas_cost(mas: dict[str, Any] | None) -> float:
    """Stable structural cost; wall-clock latency is intentionally excluded."""
    if not isinstance(mas, dict):
        return float("inf")
    agents = mas.get("agents") if isinstance(mas.get("agents"), list) else []
    edges = mas.get("edges") if isinstance(mas.get("edges"), list) else []
    budget = sum(max(0, int(agent.get("budget", 1) or 1)) for agent in agents if isinstance(agent, dict))
    iterations = max(1, int(mas.get("max_iterations", 1) or 1))
    # Budget dominates; agents/edges/iterations only resolve equal-quality ties.
    return float(budget * 1000 + len(agents) * 100 + len(edges) * 10 + iterations)


@dataclass(frozen=True)
class PairSelection:
    selected_origin: str
    selection_reason: str
    primary_margin: float
    confidence_margin: float
    cost_margin: float
    selector_version: str
    is_strict_improvement: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_pair_winner(
    *,
    source_score: float,
    updated_score: float,
    source_mas: dict[str, Any] | None,
    updated_mas: dict[str, Any] | None,
    source_trajectory: dict[str, Any] | None = None,
    updated_trajectory: dict[str, Any] | None = None,
    source_valid: bool = True,
    updated_valid: bool = True,
    explicit_accept: bool = False,
    score_epsilon: float = 1e-8,
    confidence_epsilon: float = 1e-8,
) -> PairSelection:
    """Choose a winner lexicographically: validity, score, confidence, cost.

    Equal wrong answers remain source. Equal good answers may choose the cheaper
    state, otherwise source.  Since generic scorers need not be binary, the cost
    tie-break is allowed only when both scores are positive.
    """
    src = _finite_float(source_score)
    upd = _finite_float(updated_score)
    primary_margin = upd - src
    src_conf = trajectory_confidence(source_trajectory)
    upd_conf = trajectory_confidence(updated_trajectory)
    confidence_margin = upd_conf - src_conf
    src_cost = deterministic_mas_cost(source_mas)
    upd_cost = deterministic_mas_cost(updated_mas)
    cost_margin = src_cost - upd_cost  # positive means updated is cheaper

    if explicit_accept:
        origin, reason = "source", "explicit_accept"
    elif source_valid and not updated_valid:
        origin, reason = "source", "updated_invalid"
    elif updated_valid and not source_valid:
        origin, reason = "updated", "source_invalid"
    elif not source_valid and not updated_valid:
        origin, reason = "source", "both_invalid_source_fallback"
    elif primary_margin > score_epsilon:
        origin, reason = "updated", "higher_task_score"
    elif primary_margin < -score_epsilon:
        origin, reason = "source", "higher_task_score"
    elif confidence_margin > confidence_epsilon:
        origin, reason = "updated", "higher_judge_confidence"
    elif confidence_margin < -confidence_epsilon:
        origin, reason = "source", "higher_judge_confidence"
    elif min(src, upd) > 0.0 and cost_margin > 0.0:
        origin, reason = "updated", "equal_quality_lower_cost"
    else:
        origin, reason = "source", "stable_source_tie"

    return PairSelection(
        selected_origin=origin,
        selection_reason=reason,
        primary_margin=primary_margin,
        confidence_margin=confidence_margin,
        cost_margin=cost_margin,
        selector_version=SELECTOR_VERSION,
        is_strict_improvement=origin == "updated" and primary_margin > score_epsilon,
    )
