#%%

"""
View Cluster Ideas - Display ideas from a randomly selected cluster.

Loads cached clustering results and displays idea.idea or idea.taxonomy_phrase
for a randomly selected cluster.

Usage:
    cd src/experiments/clusterer_v2
    python view_cluster_ideas.py
"""

import os
import sys
import random
import re
from collections import defaultdict
from typing import Optional, List

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from config import CacheConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = "M250480 Associatiemonitor ASN Bank net databestand.sav"
VARIABLE = "Qd1_combined"
SAMPLE_SIZE = 2000
N_SAMPLES = None  # Max ideas to display (None = ALL ideas from cluster)

# Display format: "response_idea", "taxonomy_phrase", or "both"
# - "response_idea": shows idea.idea
# - "taxonomy_phrase": shows idea.taxonomy_phrase
# - "both": shows idea.idea / idea.taxonomy_phrase
DATA_FORMAT_DISPLAYED = "both"


# =============================================================================
# CACHE LOADING
# =============================================================================

def load_cached_clusters() -> Optional[List[models.ClusterModel]]:
    """Load cached clustering results from Step 5 (initial_clusters)."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_manager = CacheManager(CacheConfig())
    step = "initial_clusters"

    if not cache_manager.is_cache_valid(FILENAME, step, variable_key):
        print(f"Cache not found for step '{step}'")
        print(f"   Filename: {FILENAME}")
        print(f"   Variable key: {variable_key}")
        return None

    cluster_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step=step,
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )

    return cluster_results


# =============================================================================
# DISPLAY
# =============================================================================

def display_cluster_ideas(
    cluster_results: List[models.ClusterModel],
    n_samples: Optional[int] = None,
    data_format: str = "response_idea"
):
    """
    Display ideas from ONE randomly selected cluster.

    Args:
        cluster_results: List of ClusterModel objects
        n_samples: Max ideas to display (None = ALL ideas from cluster)
        data_format: "response_idea" (idea.idea) or "taxonomy_phrase" (idea.taxonomy_phrase)
    """
    # Group ideas by cluster ID
    clusters = defaultdict(list)
    for response in cluster_results:
        if response.response_ideas:
            for idea in response.response_ideas:
                cluster_id = idea.initial_cluster
                if cluster_id is not None:
                    clusters[cluster_id].append(idea)

    if not clusters:
        print("No clustered ideas found")
        return

    # Pick 1 random cluster (exclude noise cluster -1)
    valid_cluster_ids = [cid for cid in clusters.keys() if cid != -1]

    if not valid_cluster_ids:
        print("No valid clusters found (only noise)")
        return

    selected_cluster = random.choice(valid_cluster_ids)

    # Get ideas from selected cluster (sample or all)
    cluster_ideas = clusters[selected_cluster]
    if n_samples is None:
        # Show ALL ideas
        sampled_ideas = cluster_ideas
        sample_count = len(cluster_ideas)
    else:
        # Random sample up to n_samples
        sample_count = min(n_samples, len(cluster_ideas))
        sampled_ideas = random.sample(cluster_ideas, sample_count)

    # Display header
    print("\n" + "=" * 80)
    print(f"CLUSTER {selected_cluster} - {'All Ideas' if n_samples is None else 'Random Sample'}")
    print(f"Showing {sample_count} of {len(cluster_ideas)} ideas")
    print(f"Display format: {data_format}")
    print("=" * 80)

    # Display ideas
    for i, idea in enumerate(sampled_ideas, 1):
        if data_format == "response_idea":
            text = idea.idea
        elif data_format == "taxonomy_phrase":
            text = idea.taxonomy_phrase
        else:  # "both"
            idea_text = idea.idea
            taxonomy_text = idea.taxonomy_phrase
            # Remove context identifiers from both
            idea_text = re.sub(r"\[.*?\]", "", idea_text)
            idea_text = re.sub(r"\s+", " ", idea_text).strip()
            taxonomy_text = re.sub(r"\[.*?\]", "", taxonomy_text)
            taxonomy_text = re.sub(r"\s+", " ", taxonomy_text).strip()
            print(f"[{i}] {idea_text} / {taxonomy_text}")
            continue

        # Remove context identifiers like [lang=nl-NL][domain=...]
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        print(f"[{i}] {text}")

    print("\n" + "-" * 80)

    # Show cluster summary
    print(f"\nCluster summary:")
    print(f"  Total clusters: {len(valid_cluster_ids)}")
    print(f"  Noise ideas: {len(clusters.get(-1, []))}")
    print(f"  Cluster IDs: {sorted(valid_cluster_ids)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Loading cached clustering data...")
    print(f"  Filename: {FILENAME}")
    print(f"  Variable: {VARIABLE}")
    print(f"  Sample size: {SAMPLE_SIZE}")

    results = load_cached_clusters()

    if results:
        total_ideas = sum(
            len(r.response_ideas) for r in results if r.response_ideas
        )
        print(f"\nLoaded {len(results)} responses with {total_ideas} total ideas")
        display_cluster_ideas(results, N_SAMPLES, DATA_FORMAT_DISPLAYED)
    else:
        print("\nFailed to load cached data")
        sys.exit(1)
