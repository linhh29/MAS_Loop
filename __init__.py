"""MAS-Loop: Multi-Agent System evolution framework.

Root package init that ensures MAS_Loop/ is on sys.path so all
sub-packages (building/, datasets/, mas_loop/) can be imported
regardless of the CWD or launch script location.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MAS_LOOP_ROOT = Path(__file__).resolve().parent

if str(_MAS_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAS_LOOP_ROOT))

# Also add the building/ sibling so that legacy imports like
# ``from building.skill_mas.build import ...`` still work.
_BUILDING_ROOT = _MAS_LOOP_ROOT / "building"
if str(_BUILDING_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUILDING_ROOT))