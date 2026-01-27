#%%
"""
Cluster Analysis v3: Parent Category → Cluster Theme Mapping

Analyze the relationship between LLM-assigned parent_category and
embedding-based clusters. Generates three reports:
  A) Detailed report: parent_category → cluster themes with counts
  B) Cross-tabulation matrix: parent_category × cluster
  C) Purity analysis: alignment between categories and clusters

Usage:
    cd src && python -m experiments.codeGenerator_v2.analyze_clusters_v3
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

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

FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VARIABLE = "Q20"
SAMPLE_SIZE = 500

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IdeaInfo:
    """Simplified idea info for analysis."""
    idea_id: str
    idea: str
    taxonomy_phrase: str
    parent_category: str
    initial_cluster: int
    cluster_probability: float


@dataclass
class ClusterInfo:
    """Cluster metadata for display."""
    cluster_id: int
    label_theme: str
    label_description: str
    size: int


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[List[IdeaInfo], Dict[int, ClusterInfo]]:
    """Load embeddings data and clustering metadata."""

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_manager = CacheManager(CacheConfig())

    # Load embeddings (has parent_category populated)
    embeddings_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step="embeddings",
        variable_key=variable_key,
        model_cls=models.EmbeddingsModel
    )

    # Load clustering metadata (has cluster themes)
    clustering_metadata = None
    if cache_manager.is_cache_valid(FILENAME, "clustering_metadata", variable_key):
        results = cache_manager.load_from_cache(
            filename=FILENAME,
            step="clustering_metadata",
            variable_key=variable_key,
            model_cls=models.ClusteringMetadataModel
        )
        if results and len(results) > 0:
            clustering_metadata = results[0]

    # Load cluster assignments (has initial_cluster per idea)
    cluster_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step="initial_clusters",
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )

    # Build cluster info dict
    cluster_info: Dict[int, ClusterInfo] = {}
    if clustering_metadata:
        for cluster_id, cluster_data in clustering_metadata.clusters.items():
            cluster_info[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                label_theme=cluster_data.label_theme or f"Cluster {cluster_id}",
                label_description=cluster_data.label_description or "",
                size=cluster_data.size
            )

    # Build idea list with cluster assignments
    # Create lookup from idea_id to cluster info
    idea_clusters: Dict[str, Tuple[int, float]] = {}
    for result in cluster_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                cluster_id = idea.initial_cluster if idea.initial_cluster is not None else -1
                prob = idea.cluster_probability or 0.0
                idea_clusters[idea.idea_id] = (cluster_id, prob)

    # Build final idea list from embeddings (which has parent_category)
    ideas: List[IdeaInfo] = []
    for result in embeddings_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                cluster_id, prob = idea_clusters.get(idea.idea_id, (-1, 0.0))
                ideas.append(IdeaInfo(
                    idea_id=idea.idea_id,
                    idea=idea.idea,
                    taxonomy_phrase=idea.taxonomy_phrase or "",
                    parent_category=idea.parent_category or "(empty)",
                    initial_cluster=cluster_id,
                    cluster_probability=prob
                ))

    return ideas, cluster_info


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def build_category_cluster_matrix(ideas: List[IdeaInfo]) -> Dict[str, Dict[int, int]]:
    """Build parent_category → cluster → count matrix."""
    matrix: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for idea in ideas:
        matrix[idea.parent_category][idea.initial_cluster] += 1
    return dict(matrix)


def build_cluster_category_matrix(ideas: List[IdeaInfo]) -> Dict[int, Dict[str, int]]:
    """Build cluster → parent_category → count matrix."""
    matrix: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for idea in ideas:
        matrix[idea.initial_cluster][idea.parent_category] += 1
    return dict(matrix)


def calculate_purity(distribution: Dict[any, int]) -> float:
    """Calculate purity (max fraction) of a distribution."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    max_count = max(distribution.values())
    return max_count / total


def calculate_entropy(distribution: Dict[any, int]) -> float:
    """Calculate normalized entropy (0 = pure, 1 = uniform)."""
    import math
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    n_classes = len(distribution)
    if n_classes <= 1:
        return 0.0

    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalize by max possible entropy
    max_entropy = math.log2(n_classes)
    return entropy / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# REPORT A: DETAILED PARENT CATEGORY → CLUSTER THEMES
# =============================================================================

def print_report_a(ideas: List[IdeaInfo], cluster_info: Dict[int, ClusterInfo]):
    """Print detailed report: parent_category → cluster themes."""

    print("\n" + "=" * 80)
    print("REPORT A: PARENT CATEGORY → CLUSTER THEMES (Detailed)")
    print("=" * 80)

    matrix = build_category_cluster_matrix(ideas)

    # Sort categories by total count
    category_totals = {cat: sum(clusters.values()) for cat, clusters in matrix.items()}
    sorted_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)

    for category in sorted_categories:
        clusters = matrix[category]
        total = category_totals[category]

        print(f"\n{'─' * 80}")
        print(f"PARENT CATEGORY: {category} ({total} ideas)")
        print(f"{'─' * 80}")

        # Sort clusters by count within this category
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)

        for cluster_id, count in sorted_clusters:
            pct = count / total * 100

            # Get cluster theme
            if cluster_id in cluster_info:
                theme = cluster_info[cluster_id].label_theme
            elif cluster_id == -1:
                theme = "(noise)"
            else:
                theme = f"(no theme)"

            print(f"  → Cluster {cluster_id:2d}: {theme[:50]:50s} | {count:3d} ideas ({pct:5.1f}%)")


# =============================================================================
# REPORT B: CROSS-TABULATION MATRIX
# =============================================================================

def print_report_b(ideas: List[IdeaInfo], cluster_info: Dict[int, ClusterInfo]):
    """Print cross-tabulation matrix: parent_category × cluster."""

    print("\n" + "=" * 80)
    print("REPORT B: CROSS-TABULATION MATRIX (parent_category × cluster)")
    print("=" * 80)

    matrix = build_category_cluster_matrix(ideas)

    # Get all clusters (sorted)
    all_clusters = sorted(set(
        cluster_id for clusters in matrix.values() for cluster_id in clusters.keys()
    ))

    # Sort categories by total count
    category_totals = {cat: sum(clusters.values()) for cat, clusters in matrix.items()}
    sorted_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)

    # Print header
    print(f"\n{'':20s}", end="")
    for cluster_id in all_clusters:
        print(f" | {cluster_id:>4d}", end="")
    print(" | Total")

    print("-" * 20, end="")
    for _ in all_clusters:
        print("-" * 7, end="")
    print("-" * 8)

    # Print rows
    for category in sorted_categories:
        # Truncate category name
        cat_display = category[:19] if len(category) <= 19 else category[:16] + "..."
        print(f"{cat_display:20s}", end="")

        row_total = 0
        for cluster_id in all_clusters:
            count = matrix[category].get(cluster_id, 0)
            row_total += count
            if count > 0:
                print(f" | {count:4d}", end="")
            else:
                print(f" |    .", end="")

        print(f" | {row_total:4d}")

    # Print cluster themes legend
    print(f"\n{'─' * 80}")
    print("CLUSTER THEMES:")
    for cluster_id in all_clusters:
        if cluster_id in cluster_info:
            theme = cluster_info[cluster_id].label_theme
        elif cluster_id == -1:
            theme = "(noise)"
        else:
            theme = "(no theme)"
        print(f"  Cluster {cluster_id:2d}: {theme}")


# =============================================================================
# REPORT C: PURITY ANALYSIS
# =============================================================================

def print_report_c(ideas: List[IdeaInfo], cluster_info: Dict[int, ClusterInfo]):
    """Print purity analysis: alignment between categories and clusters."""

    print("\n" + "=" * 80)
    print("REPORT C: PURITY ANALYSIS")
    print("=" * 80)

    cat_to_cluster = build_category_cluster_matrix(ideas)
    cluster_to_cat = build_cluster_category_matrix(ideas)

    # Part 1: Parent Category Purity (how focused is each category on specific clusters?)
    print(f"\n{'─' * 80}")
    print("PARENT CATEGORY PURITY (1.0 = all ideas in one cluster)")
    print(f"{'─' * 80}")
    print(f"{'Category':20s} | {'Ideas':>5s} | {'Purity':>6s} | {'Entropy':>7s} | Dominant Cluster(s)")
    print("-" * 80)

    category_totals = {cat: sum(clusters.values()) for cat, clusters in cat_to_cluster.items()}
    sorted_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)

    for category in sorted_categories:
        clusters = cat_to_cluster[category]
        total = category_totals[category]
        purity = calculate_purity(clusters)
        entropy = calculate_entropy(clusters)

        # Find top 2 clusters
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:2]
        dominant_str = ", ".join([
            f"Cl.{cid}({cnt}/{total})" for cid, cnt in sorted_clusters
        ])

        cat_display = category[:19] if len(category) <= 19 else category[:16] + "..."
        print(f"{cat_display:20s} | {total:5d} | {purity:6.2f} | {entropy:7.2f} | {dominant_str}")

    # Part 2: Cluster Purity (how focused is each cluster on specific categories?)
    print(f"\n{'─' * 80}")
    print("CLUSTER PURITY (1.0 = all ideas from one parent_category)")
    print(f"{'─' * 80}")
    print(f"{'Cluster':8s} | {'Theme':30s} | {'Ideas':>5s} | {'Purity':>6s} | {'Entropy':>7s} | Dominant Category")
    print("-" * 100)

    sorted_clusters = sorted(cluster_to_cat.keys())

    for cluster_id in sorted_clusters:
        categories = cluster_to_cat[cluster_id]
        total = sum(categories.values())
        purity = calculate_purity(categories)
        entropy = calculate_entropy(categories)

        # Find dominant category
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:1]
        dominant_cat, dominant_cnt = sorted_cats[0] if sorted_cats else ("", 0)

        # Get cluster theme
        if cluster_id in cluster_info:
            theme = cluster_info[cluster_id].label_theme[:30]
        elif cluster_id == -1:
            theme = "(noise)"
        else:
            theme = "(no theme)"

        print(f"Cl.{cluster_id:>4d} | {theme:30s} | {total:5d} | {purity:6.2f} | {entropy:7.2f} | {dominant_cat} ({dominant_cnt}/{total})")

    # Summary statistics
    print(f"\n{'─' * 80}")
    print("SUMMARY")
    print(f"{'─' * 80}")

    cat_purities = [calculate_purity(cat_to_cluster[cat]) for cat in cat_to_cluster]
    cluster_purities = [calculate_purity(cluster_to_cat[cid]) for cid in cluster_to_cat]

    print(f"Average Category Purity:  {sum(cat_purities)/len(cat_purities):.3f}")
    print(f"Average Cluster Purity:   {sum(cluster_purities)/len(cluster_purities):.3f}")

    # Interpretation
    avg_cat_purity = sum(cat_purities)/len(cat_purities)
    avg_cluster_purity = sum(cluster_purities)/len(cluster_purities)

    print(f"\nInterpretation:")
    if avg_cat_purity > 0.7 and avg_cluster_purity > 0.7:
        print("  → Strong alignment: LLM categories and embedding clusters agree well.")
    elif avg_cat_purity > 0.5 and avg_cluster_purity > 0.5:
        print("  → Moderate alignment: Some agreement, but categories/clusters capture different aspects.")
    else:
        print("  → Weak alignment: LLM categories and embedding clusters capture different structure.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CLUSTER ANALYSIS v3: Parent Category → Cluster Theme Mapping")
    print("=" * 80)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")

    # Load data
    print("\nLoading data...")
    ideas, cluster_info = load_data()
    print(f"  Loaded {len(ideas)} ideas")
    print(f"  Loaded {len(cluster_info)} cluster themes")

    # Count categories and clusters
    categories = set(idea.parent_category for idea in ideas)
    clusters = set(idea.initial_cluster for idea in ideas)
    print(f"  Unique parent_categories: {len(categories)}")
    print(f"  Unique clusters: {len(clusters)}")

    # Generate all reports
    print_report_a(ideas, cluster_info)
    print_report_b(ideas, cluster_info)
    print_report_c(ideas, cluster_info)

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()