"""MAS-Loop dataset runners: task quality scoring and execute-and-score pipelines.

Each dataset runner provides:
- Task loading from dataset JSONL files
- Scoring function: (final_output, expected_answer) -> float (0-1)
- MAS-Loop compatible execute-and-score pipeline
"""

from __future__ import annotations

from .aime_runner import AimeRunner, AimeConfig
from .bcp_runner import BcpRunner, BcpConfig
from .drb_runner import DrbRunner, DrbConfig

__all__ = [
    "AimeRunner", "AimeConfig",
    "BcpRunner", "BcpConfig",
    "DrbRunner", "DrbConfig",
]
