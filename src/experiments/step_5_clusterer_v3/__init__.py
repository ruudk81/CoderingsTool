"""
Clusterer V3 - Map-Reduce MECE per Cluster

V3 changes from V2:
- Phase 6 (keyword extraction): REMOVED
- Phase 7: Map-Reduce MECE per cluster (replaces V2's single-theme + cross-cluster MECE)
  - MAP: batch all ideas, find ALL atomic themes per batch
  - REDUCE: consolidate themes across batches
  - MECE: apply inclusion/exclusion boundaries
- Phase 8 (cross-cluster MECE): REMOVED

Usage:
    cd src && python -m experiments.step_5_clusterer_v3.run_experiment
"""

from .clusterer_exp import Clusterer
from .config_clusterer_exp import ClustererConfig
from .map_reduce_mece import MapReduceMECE

__all__ = ["Clusterer", "ClustererConfig", "MapReduceMECE"]
