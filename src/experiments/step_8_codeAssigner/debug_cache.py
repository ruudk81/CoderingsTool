"""
Debug script for Step 8: Code Assigner - Cache Inspection
Loads various cached data and inspects cluster/code mappings.

Usage:
    cd src && python -m experiments.step_8_codeAssigner.debug_cache
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from collections import defaultdict
from experiments import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 500

# Optional: specific cluster to inspect
INSPECT_CLUSTER = None  # e.g., "14"


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load expanded clusters
    print("Loading expanded_clusters...")
    cluster_results = cache_manager.load_from_cache(
        FILENAME, "expanded_clusters", variable_key, models.ClusterModel
    )
    print(f"Loaded {len(cluster_results)} cluster results")

    # Collect cluster mappings
    cluster_mapping = defaultdict(set)
    idea_counts = defaultdict(int)

    for result in cluster_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                initial = idea.initial_cluster
                expanded = idea.expanded_cluster

                if expanded:
                    cluster_mapping[initial].add(expanded)
                    idea_counts[expanded] += 1

    # Print expansion summary
    print("\n" + "=" * 70)
    print("CLUSTER EXPANSION SUMMARY")
    print("=" * 70)
    print(f"{'Initial Cluster':<15} | Expanded Clusters")
    print("-" * 60)

    for initial_cluster in sorted(cluster_mapping.keys(), key=lambda x: (isinstance(x, str), x)):
        expanded_list = sorted(cluster_mapping[initial_cluster])

        if len(expanded_list) == 1 and str(expanded_list[0]) == str(initial_cluster):
            print(f"{initial_cluster:<15} | {expanded_list[0]} (single-theme, {idea_counts[expanded_list[0]]} ideas)")
        else:
            print(f"{initial_cluster:<15} | {', '.join(str(e) for e in expanded_list)} (multi-theme)")
            for exp in expanded_list:
                print(f"{'':15} |   - {exp}: {idea_counts[exp]} ideas")

    # Summary
    total_initial = len(cluster_mapping)
    total_expanded = len(idea_counts)
    multi_theme = sum(1 for v in cluster_mapping.values() if len(v) > 1)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Initial clusters: {total_initial}")
    print(f"Expanded clusters: {total_expanded}")
    print(f"Multi-theme clusters: {multi_theme}")
    print(f"Single-theme clusters: {total_initial - multi_theme}")

    # Inspect specific cluster if requested
    if INSPECT_CLUSTER:
        print(f"\n{'=' * 70}")
        print(f"IDEAS IN CLUSTER {INSPECT_CLUSTER}")
        print("=" * 70)
        for result in cluster_results:
            if result.response_ideas:
                for idea in result.response_ideas:
                    if idea.expanded_cluster == INSPECT_CLUSTER:
                        print(f"  - {idea.idea}")


if __name__ == "__main__":
    main()
