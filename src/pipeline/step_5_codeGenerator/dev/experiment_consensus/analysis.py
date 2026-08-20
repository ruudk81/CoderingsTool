"""Wat er uit de runs af te lezen valt — pure functies, geen LLM, geen IO.

Drie metingen, elk met een eigen doel:
- `pairwise_ari`: hoe onstabiel is deel 1 op deze configuratie? Tien runs geven
  45 vergelijkingen, dus een verdeling in plaats van één punt.
- `histogram`: heeft de matrix de goede vorm? Bijna-binair betekent dat
  consensus niets toevoegt; overwegend middenin betekent dat er niets te
  koppelen valt. Consensus werkt alleen bij een duidelijke kern met een dunne
  onzekere schil.
- `tau_sweep`: welke drempel levert een bruikbare indeling?
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence

SRC = Path(__file__).resolve().parents[4]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.step_3_ideaExtractor.measure_stability import adjusted_rand_index  # noqa: E402

from consensus import consensus_partition, labels_from_clusters  # noqa: E402


def pairwise_ari(runs: Sequence[Sequence[Sequence[str]]]) -> List[float]:
    """ARI tussen elk paar runs. Bij tien runs zijn dat 45 waarden.

    Dat is bewust geen gemiddelde: de spreiding is de boodschap. Één
    vergelijking tussen twee runs is een punt, en een punt zegt niets over hoe
    breed de wolk is.
    """
    labellings = [labels_from_clusters(clusters) for clusters in runs]
    return [adjusted_rand_index(a, b) for a, b in combinations(labellings, 2)]


def histogram(together: Dict[FrozenSet[str], int], runs: int) -> List[int]:
    """Hoeveel paren zaten 0, 1, ... runs samen. Index = aantal runs samen."""
    counts = [0] * (runs + 1)
    for n in together.values():
        counts[n] += 1
    return counts


def tau_sweep(
    together: Dict[FrozenSet[str], int],
    attribute_ids: Sequence[str],
    runs: int,
    taus: Sequence[float],
) -> List[dict]:
    """Vorm van de indeling per drempel: hoeveel groepen, hoe groot de grootste,
    hoeveel attributen blijven alleen."""
    rows = []
    for tau in taus:
        clusters = consensus_partition(together, attribute_ids, runs, tau)
        rows.append({
            "tau": tau,
            "n_groups": len(clusters),
            "largest": max((len(c) for c in clusters), default=0),
            "n_solo": sum(1 for c in clusters if len(c) == 1),
        })
    return rows
