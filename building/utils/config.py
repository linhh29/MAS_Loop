"""Paths and defaults for the building package (repo-root relative)."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Return the MAS_Loop repository root (parent of ``building/``).

    Works regardless of CWD: walks up from this file's location until
    it finds the directory that contains both ``building/`` and ``datasets/``.
    """
    p = Path(__file__).resolve()
    for ancestor in [p.parent, *p.parents]:
        if (ancestor / "building" / "skill_mas").is_dir() and (ancestor / "datasets").is_dir():
            return ancestor
    # Fallback to the old heuristic (parents[2] from building/utils/)
    return p.parents[2]


REPO_ROOT = _repo_root()
BUILDING_HOME = REPO_ROOT / "building"
INIT_SKILL_DIR = BUILDING_HOME / "init_skill"
OPTIMIZED_SKILL_DIR = BUILDING_HOME / "optimized_skill"


def print_traces_enabled() -> bool:
    return os.environ.get("MASKILL_PRINT_TRACES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_init_skill_path(explicit: Path | None) -> Path:
    """
    Resolve a skill file path for MAS build.

    Accepts either a directory containing ``SKILL.md`` or a direct ``*.md`` skill file
    (e.g. ``building/optimized_skill/bcp.md``).
    """
    if explicit is not None:
        p = explicit.resolve()
        if p.is_file():
            return p
        if not p.is_dir():
            raise FileNotFoundError(p)
        skill_md = p / "SKILL.md"
        if skill_md.is_file():
            return skill_md
        raise FileNotFoundError(f"Expected SKILL.md or *.md skill file under: {p}")
    default = INIT_SKILL_DIR / "SKILL.md"
    if default.is_file():
        return default.resolve()
    raise FileNotFoundError(
        f"Expected SKILL.md under {INIT_SKILL_DIR}. "
        "Pass --init-skill /path/to/SKILL.md or /path/to/skill.md."
    )
