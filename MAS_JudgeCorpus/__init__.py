"""MAS Judge Corpus — Bradley-Terry RM training data for MAS judge model.

Builds preference pairs from MAS-Loop execution trajectories for training
a reward model that evaluates MAS graph structures without running them.

Each pair compares orchestra-generated MAS vs editor-modified MAS for the
same task, ordered by score. Higher-score = chosen, lower-score = rejected.

Output: {chosen_text, rejected_text, metadata} JSONL for TRL RewardTrainer.
The chosen_text/rejected_text explicitly marks the MAS source (orchestra/editor).

Inspired by OrchRM (Tsang et al., 2026).
"""

from .schema import TrainingPair, DatasetStatistics, MasSource
from .constructor import construct_rm_dataset
from .mas_formatter import format_mas_to_text, MAS_FORMAT_TYPE
from .utils import load_pairs, compute_statistics, print_statistics, split_train_val

__all__ = [
    "TrainingPair",
    "DatasetStatistics",
    "MasSource",
    "construct_rm_dataset",
    "format_mas_to_text",
    "MAS_FORMAT_TYPE",
    "load_pairs",
    "compute_statistics",
    "print_statistics",
    "split_train_val",
]