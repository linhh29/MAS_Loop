"""Reward computation for MAS-Loop RL training.

Defines the reward functions for Orchestra and Editor as specified in
Section 3.4 of the MAS-Loop Expanded Idea.

Orchestra reward:
    R_gen = Q(M_src) + alpha * Q(M_upd) - beta * Cost(M_src) - gamma * Complexity(M_src)

Editor reward:
    R_edit = delta_Q - lambda_c * delta_Cost - lambda_s * SizePenalty
             - lambda_v * ConstraintViolation + edit_type_bonus

The edit_type_bonus rewards the Editor for producing structurally meaningful
patches (e.g., "repairing" earns more than "restructuring").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from mas_loop.mas_schema import MASConfig, Patch


# Bonus weights for different edit types (Section 4.3)
EDIT_TYPE_BONUS: dict[str, float] = {
    "repairing":     0.15,   # Fixing a broken MAS is good
    "upgrading":     0.10,   # Adding stronger capabilities
    "refining":      0.08,   # Subtle improvements
    "simplifying":   0.10,   # Reducing complexity while maintaining quality
    "constraining":  0.05,   # Adding constraints
    "extending":     0.10,   # Adding new agents
    "compressing":   0.08,   # Merging redundant agents
    "restructuring": 0.05,   # Reorganizing topology
    "mixed":         0.05,   # Mixed operations
}


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Orchestra reward weights
    alpha_editable: float = 0.3    # Weight for Q(M_upd) in orchestra reward
    beta_cost: float = 0.1         # Cost penalty for orchestra
    gamma_complexity: float = 0.05  # Complexity penalty for orchestra

    # Editor reward weights
    lambda_c: float = 0.1          # Cost delta penalty
    lambda_s: float = 0.05         # Size penalty (per operation)
    lambda_v: float = 1.0          # Constraint violation penalty

    # Edit type bonus
    edit_type_bonus_enabled: bool = True
    edit_type_bonus_scale: float = 0.1  # Scale factor for edit type bonus

    # Score normalization
    normalize_scores: bool = True
    score_max: float = 1.0


def compute_task_quality(
    final_output: str,
    expected_answer: str,
    *,
    exact_match: bool = False,
) -> float:
    """Compute a task quality score in [0, 1].

    Default: normalized string similarity.
    If exact_match=True: 1.0 for exact match, 0.0 otherwise.
    """
    if exact_match:
        return 1.0 if final_output.strip() == expected_answer.strip() else 0.0

    # Normalized Levenshtein-like similarity
    out = final_output.strip().lower()
    exp = expected_answer.strip().lower()

    if out == exp:
        return 1.0
    if not out or not exp:
        return 0.0

    # Simple character Jaccard similarity
    out_set = set(out)
    exp_set = set(exp)
    intersection = out_set & exp_set
    union = out_set | exp_set
    if not union:
        return 0.0
    return len(intersection) / len(union)


def compute_mas_complexity(mas: MASConfig) -> float:
    """Compute a complexity score for a MAS config.

    Higher values mean more complex (more agents, more edges, larger budget).
    """
    if not mas.agents:
        return 0.0
    n_agents = len(mas.agents)
    n_edges = len(mas.edges)
    total_budget = sum(a.budget for a in mas.agents)

    # Normalized complexity: log scale to avoid explosion
    complexity = math.log(1 + n_agents) + math.log(1 + n_edges) + math.log(1 + total_budget)
    return complexity


def compute_orchestra_reward(
    src_score: float,
    upd_score: float,
    src_cost: float,
    mas: MASConfig,
    *,
    config: Optional[RewardConfig] = None,
) -> float:
    """Compute the Orchestra reward.

    R_gen = Q(M_src) + alpha * Q(M_upd) - beta * Cost(M_src) - gamma * Complexity(M_src)

    Args:
        src_score: Task quality of source MAS
        upd_score: Task quality of updated MAS (after edit)
        src_cost: Cost of running the source MAS (USD)
        mas: Source MAS configuration
        config: Reward weights
    """
    cfg = config or RewardConfig()

    complexity = compute_mas_complexity(mas)

    reward = (
        src_score
        + cfg.alpha_editable * upd_score
        - cfg.beta_cost * src_cost
        - cfg.gamma_complexity * complexity
    )
    return reward


def compute_editor_reward(
    src_score: float,
    upd_score: float,
    src_cost: float,
    upd_cost: float,
    patch_size: int,
    constraint_violations: int = 0,
    *,
    config: Optional[RewardConfig] = None,
    edit_type: Optional[str] = None,
) -> float:
    """Compute the Editor reward.

    R_edit = delta_Q - lambda_c * delta_Cost(+) - lambda_s * SizePenalty
             - lambda_v * ConstraintViolation + edit_type_bonus

    Args:
        src_score: Task quality before edit
        upd_score: Task quality after edit
        src_cost: Cost before edit
        upd_cost: Cost after edit
        patch_size: Number of patch operations
        constraint_violations: Number of constraint violations
        config: Reward weights
        edit_type: Optional edit type from Patch.edit_type() for bonus
    """
    cfg = config or RewardConfig()

    delta_q = upd_score - src_score
    delta_cost = upd_cost - src_cost

    # Cost penalty: only penalize if cost increased
    cost_penalty = max(0.0, delta_cost) * cfg.lambda_c

    # Edit type bonus: reward structurally meaningful edits
    edit_bonus = 0.0
    if cfg.edit_type_bonus_enabled and edit_type:
        edit_bonus = EDIT_TYPE_BONUS.get(edit_type, 0.0) * cfg.edit_type_bonus_scale

    reward = (
        delta_q
        - cost_penalty
        - cfg.lambda_s * patch_size
        - cfg.lambda_v * constraint_violations
        + edit_bonus
    )
    return reward


def compute_combined_sample_reward(
    src_score: float,
    upd_score: float,
    src_cost: float,
    upd_cost: float,
    mas: MASConfig,
    patch_size: int,
    constraint_violations: int = 0,
    *,
    config: Optional[RewardConfig] = None,
) -> dict[str, float]:
    """Compute both orchestra and editor rewards for a single sample."""
    cfg = config or RewardConfig()

    orchestra_r = compute_orchestra_reward(
        src_score=src_score,
        upd_score=upd_score,
        src_cost=src_cost,
        mas=mas,
        config=cfg,
    )

    editor_r = compute_editor_reward(
        src_score=src_score,
        upd_score=upd_score,
        src_cost=src_cost,
        upd_cost=upd_cost,
        patch_size=patch_size,
        constraint_violations=constraint_violations,
        config=cfg,
    )

    return {
        "orchestra_reward": orchestra_r,
        "editor_reward": editor_r,
        "delta_q": upd_score - src_score,
        "delta_cost": upd_cost - src_cost,
    }
