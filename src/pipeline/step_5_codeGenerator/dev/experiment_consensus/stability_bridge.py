"""De opgeslagen partities terugvertalen naar wat `stability.py` verwacht.

`measure_stability` telt per attribuutpaar in hoeveel runs de twee samen zaten —
precies de co-associatiematrix die dit experiment nodig heeft. Het verwacht
alleen `Group`-objecten, en op schijf staan tuples. Deze brug bestaat zodat de
telling zelf niet nagebouwd hoeft te worden.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, FrozenSet, Sequence, Tuple

SRC = Path(__file__).resolve().parents[4]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.step_5_codeGenerator.grouping import Group  # noqa: E402
from pipeline.step_5_codeGenerator.stability import measure_stability  # noqa: E402


def together_from_runs(
    runs: Sequence[Sequence[Tuple[str, ...]]],
    attribute_ids: Sequence[str],
) -> Dict[FrozenSet[str], int]:
    """Co-associatiematrix uit opgeslagen partities."""
    as_groups = [
        [Group(member_ids=tuple(cluster), proposed_name="", explanation="")
         for cluster in run]
        for run in runs
    ]
    return measure_stability(as_groups, list(attribute_ids)).together
