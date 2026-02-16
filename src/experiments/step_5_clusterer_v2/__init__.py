"""
Clusterer V2 - Theme Generation + MECE Topic Consolidation

V2 changes from V1:
- Phase B: Per-cluster theme generation (replaces "checkbox code" labels)
- Phase C: MECE topic consolidation (merges overlapping themes)

Usage:
    cd src && python -m experiments.step_5_clusterer_v2.run_experiment
"""

from .clusterer_exp import Clusterer
from .config_clusterer_exp import ClustererConfig
from .mece_consolidator import MECEConsolidator

__all__ = ["Clusterer", "ClustererConfig", "MECEConsolidator"]
