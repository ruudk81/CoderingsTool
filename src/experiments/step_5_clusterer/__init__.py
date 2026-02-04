"""
Clusterer V3 Module

NOTE: After migration, v3 is now the PRODUCTION clusterer.
Import from utils.clusterer and config_clusterer instead.

This module provides backward-compatible aliases.
"""

# Import from production (migrated v3)
from utils.clusterer import Clusterer
from config_clusterer import ClustererConfig

# Backward-compatible aliases
ClustererV3 = Clusterer
ClustererV3Config = ClustererConfig

__all__ = ["Clusterer", "ClustererConfig", "ClustererV3", "ClustererV3Config"]
