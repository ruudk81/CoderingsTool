"""De ruwe partities op schijf, zodat de analyse gratis herhaalbaar is.

Het vorige consolidatie-experiment logde proza; daardoor kostte elke heranalyse
opnieuw LLM-calls. Hier gaan de partities zelf naar JSON, dus een andere tau of
een andere meting kost achteraf niets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RunSet:
    """Alle runs van één configuratie op één dataset."""
    model: str
    effort: str
    attribute_ids: List[str]
    attribute_names: Dict[str, str]
    n_respondents: int
    runs: List[List[Tuple[str, ...]]]
    # Standaardwaarde omdat `consensus_luna_set0.json` (geschreven vóór dit veld
    # bestond) `salted` niet in zijn payload heeft — `load_runset` moet dat
    # bestand blijven laden, en elke run vóór de `--no-salt`-vlag was gezouten.
    salted: bool = True


def save_runset(runset: RunSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(runset)
    payload["runs"] = [[list(cluster) for cluster in run] for run in runset.runs]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_runset(path: Path) -> RunSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"] = [[tuple(cluster) for cluster in run] for run in payload["runs"]]
    return RunSet(**payload)
