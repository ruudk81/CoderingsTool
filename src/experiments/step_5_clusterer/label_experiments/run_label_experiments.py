#%%
"""
Label Experiments Runner

Extension layer for cluster label generation tuning.
Loads cached data from step_5_clusterer and experiments with:
- Low-probability cluster members
- HDBSCAN tree structures
- Prompt construction and formatting variations

Usage:
    cd src && python -m experiments.step_5_clusterer.label_experiments.run_label_experiments

Or open this file in VS Code and run cells interactively (using #%% markers).
"""

#%% ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle

import numpy as np

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Local imports
try:
    from .config_label_exp import LabelExperimentConfig
except ImportError:
    from config_label_exp import LabelExperimentConfig

print(f"Project root: {project_root}")


#%% ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset configuration from test_data
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

# Label experiment configuration
CONFIG = LabelExperimentConfig()

print(f"Dataset: {FILENAME}")
print(f"Variable: {VARIABLE}")
print(f"Sample size: {SAMPLE_SIZE}")


#%% ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cluster_models() -> List[models.ClusterModel]:
    """Load cluster models (per-idea data) from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem

    # Cache file uses prefix "006" for initial_clusters
    cache_path = cache_dir / f"006_initial_clusters_{base_name}_{variable_key}.pkl"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run step_5_clusterer.run_experiment first."
        )

    print(f"Loading cluster models from: {cache_path.name}")

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    cluster_models = [models.ClusterModel.model_validate(item) for item in data]
    print(f"Loaded {len(cluster_models)} cluster models")

    return cluster_models


def load_clustering_metadata() -> Optional[models.ClusteringMetadataModel]:
    """Load clustering metadata (per-cluster summaries) from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem

    # This step doesn't have a prefix in CacheConfig, so it uses "999"
    cache_path = cache_dir / f"999_clustering_metadata_{base_name}_{variable_key}.pkl"

    if not cache_path.exists():
        print(f"Clustering metadata not found: {cache_path.name}")
        return None

    print(f"Loading clustering metadata from: {cache_path.name}")

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    metadata = models.ClusteringMetadataModel.model_validate(data[0])
    print(f"Loaded metadata for {len(metadata.clusters)} clusters")

    return metadata


def load_hdbscan_artifacts() -> Optional[Dict]:
    """Load HDBSCAN artifacts (trees, probabilities) from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem

    # Direct pickle (no prefix) as these are not Pydantic models
    cache_path = cache_dir / f"hdbscan_artifacts_{base_name}_{variable_key}.pkl"

    if not cache_path.exists():
        print(f"HDBSCAN artifacts not found: {cache_path.name}")
        print("Re-run step_5_clusterer.run_experiment to generate artifacts.")
        return None

    print(f"Loading HDBSCAN artifacts from: {cache_path.name}")

    with open(cache_path, 'rb') as f:
        artifacts = pickle.load(f)

    print(f"Loaded artifacts: {list(artifacts.keys())}")

    return artifacts


def load_all_cached_data() -> Dict:
    """Load all cached data from main experiment."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    return {
        "cluster_models": load_cluster_models(),
        "clustering_metadata": load_clustering_metadata(),
        "hdbscan_artifacts": load_hdbscan_artifacts(),
        "variable_key": variable_key
    }


#%% ============================================================================
# LOAD DATA
# ============================================================================

data = load_all_cached_data()

print(f"\n{'='*60}")
print("DATA LOADED")
print(f"{'='*60}")
print(f"Cluster models: {len(data['cluster_models'])} ideas")
if data['clustering_metadata']:
    print(f"Clusters: {sorted(data['clustering_metadata'].clusters.keys())}")
if data['hdbscan_artifacts']:
    print(f"HDBSCAN artifacts available: {list(data['hdbscan_artifacts'].keys())}")


#%% ============================================================================
# ORGANIZE IDEAS BY CLUSTER
# ============================================================================

def organize_by_cluster(cluster_models: List[models.ClusterModel]) -> Dict[int, List[Tuple]]:
    """
    Organize ideas by cluster ID.

    Returns:
        Dict mapping cluster_id to list of (idea_text, probability, model) tuples
    """
    clusters = defaultdict(list)

    for model in cluster_models:
        if model.response_ideas:
            for idea in model.response_ideas:
                cluster_id = idea.initial_cluster
                prob = idea.cluster_probability or 0.0
                clusters[cluster_id].append((idea.idea, prob, idea))

    # Sort by probability within each cluster
    for cluster_id in clusters:
        clusters[cluster_id].sort(key=lambda x: x[1], reverse=True)

    return clusters


clusters_by_id = organize_by_cluster(data["cluster_models"])
print(f"\nOrganized {sum(len(v) for v in clusters_by_id.values())} ideas into {len(clusters_by_id)} clusters")


#%% ============================================================================
# PROBABILITY DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_probability_distribution(
    clusters_by_id: Dict[int, List[Tuple]],
    threshold: float = 0.8
) -> None:
    """Analyze how many ideas fall below threshold per cluster."""
    print(f"\nProbability Distribution (threshold={threshold}):")
    print("-" * 70)
    print(f"{'Cluster':>8} | {'Total':>6} | {'High':>6} (>={threshold}) | {'Low':>6} (<{threshold}) | {'Mean':>6}")
    print("-" * 70)

    for cluster_id in sorted(clusters_by_id.keys()):
        if cluster_id == -1:
            continue

        members = clusters_by_id[cluster_id]
        probs = [p for _, p, _ in members]
        high = sum(1 for p in probs if p >= threshold)
        low = sum(1 for p in probs if p < threshold)
        mean_prob = np.mean(probs) if probs else 0.0

        print(f"{cluster_id:>8} | {len(members):>6} | {high:>6} | {low:>6} | {mean_prob:>6.3f}")

    # Summary
    total_ideas = sum(len(v) for k, v in clusters_by_id.items() if k != -1)
    all_probs = [p for k, members in clusters_by_id.items() if k != -1 for _, p, _ in members]
    high_total = sum(1 for p in all_probs if p >= threshold)
    low_total = sum(1 for p in all_probs if p < threshold)

    print("-" * 70)
    print(f"{'TOTAL':>8} | {total_ideas:>6} | {high_total:>6} | {low_total:>6} | {np.mean(all_probs):.3f}")
    print(f"\nLow-prob ratio: {low_total}/{total_ideas} = {low_total/total_ideas:.1%}")


analyze_probability_distribution(clusters_by_id, threshold=CONFIG.high_prob_threshold)


#%% ============================================================================
# GET LOW-PROBABILITY MEMBERS FOR A CLUSTER
# ============================================================================

def get_low_prob_members(
    cluster_id: int,
    threshold: float = 0.8,
    min_prob: float = 0.0,
    max_display: int = 15
) -> List[Tuple]:
    """
    Get low-probability members for a cluster.

    Args:
        cluster_id: Cluster to analyze
        threshold: Upper bound (ideas with prob < threshold)
        min_prob: Lower bound (ideas with prob >= min_prob)
        max_display: Max ideas to display

    Returns:
        List of (text, probability, idea) tuples
    """
    if cluster_id not in clusters_by_id:
        print(f"Cluster {cluster_id} not found")
        return []

    members = clusters_by_id[cluster_id]
    low_prob = [(text, p, idea) for text, p, idea in members if min_prob <= p < threshold]

    print(f"\nCluster {cluster_id}: {len(low_prob)} low-probability members ({min_prob:.1f} <= p < {threshold})")
    print("-" * 70)

    for text, prob, idea in low_prob[:max_display]:
        display_text = text[:65] + "..." if len(text) > 65 else text
        print(f"  [{prob:.3f}] {display_text}")

    if len(low_prob) > max_display:
        print(f"  ... and {len(low_prob) - max_display} more")

    return low_prob


# Example: Get low-prob members for cluster 0
if 0 in clusters_by_id:
    low_prob_0 = get_low_prob_members(0, threshold=0.8, min_prob=0.5)


#%% ============================================================================
# GET HIGH-PROBABILITY MEMBERS FOR A CLUSTER
# ============================================================================

def get_high_prob_members(
    cluster_id: int,
    threshold: float = 0.8,
    max_display: int = 10
) -> List[Tuple]:
    """Get high-probability members for a cluster."""
    if cluster_id not in clusters_by_id:
        print(f"Cluster {cluster_id} not found")
        return []

    members = clusters_by_id[cluster_id]
    high_prob = [(text, p, idea) for text, p, idea in members if p >= threshold]

    print(f"\nCluster {cluster_id}: {len(high_prob)} high-probability members (p >= {threshold})")
    print("-" * 70)

    for text, prob, idea in high_prob[:max_display]:
        display_text = text[:65] + "..." if len(text) > 65 else text
        print(f"  [{prob:.3f}] {display_text}")

    if len(high_prob) > max_display:
        print(f"  ... and {len(high_prob) - max_display} more")

    return high_prob


# Example
if 0 in clusters_by_id:
    high_prob_0 = get_high_prob_members(0, threshold=0.8)


#%% ============================================================================
# COMPARE PROBABILITY THRESHOLDS
# ============================================================================

def compare_thresholds(cluster_id: int, thresholds: List[float] = None) -> None:
    """Compare how different thresholds affect sample selection."""
    if thresholds is None:
        thresholds = CONFIG.probability_thresholds

    if cluster_id not in clusters_by_id:
        print(f"Cluster {cluster_id} not found")
        return

    members = clusters_by_id[cluster_id]
    probs = [p for _, p, _ in members]

    print(f"\nCluster {cluster_id} - Threshold Comparison (n={len(members)})")
    print("-" * 50)

    for threshold in thresholds:
        high = sum(1 for p in probs if p >= threshold)
        low = sum(1 for p in probs if p < threshold)
        print(f"  Threshold {threshold}: {high} high, {low} low ({high/len(members):.0%} / {low/len(members):.0%})")


if 0 in clusters_by_id:
    compare_thresholds(0)


#%% ============================================================================
# HDBSCAN HIERARCHY ANALYSIS
# ============================================================================

def analyze_condensed_tree(hdbscan_artifacts: Optional[Dict]) -> None:
    """Analyze the condensed tree for cluster relationships."""
    if not hdbscan_artifacts:
        print("No HDBSCAN artifacts available")
        return

    condensed_tree = hdbscan_artifacts.get("condensed_tree")
    if condensed_tree is None:
        print("No condensed tree available")
        return

    # condensed_tree is an HDBSCAN CondensedTree object
    # It has methods: to_pandas(), plot(), get_plot_data()

    try:
        df = condensed_tree.to_pandas()
        print(f"\nCondensed Tree Structure:")
        print(f"  Total edges: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst 20 edges:")
        print(df.head(20).to_string())
    except Exception as e:
        print(f"Error analyzing condensed tree: {e}")


if data['hdbscan_artifacts']:
    analyze_condensed_tree(data['hdbscan_artifacts'])


#%% ============================================================================
# VISUALIZE CLUSTER HIERARCHY
# ============================================================================

def visualize_hierarchy(hdbscan_artifacts: Optional[Dict], save_path: Optional[Path] = None) -> None:
    """Generate hierarchy visualization."""
    if not hdbscan_artifacts:
        print("No HDBSCAN artifacts available")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    single_linkage_tree = hdbscan_artifacts.get("single_linkage_tree")
    if single_linkage_tree is not None:
        print("\nGenerating Single Linkage Tree visualization...")
        fig, ax = plt.subplots(figsize=(14, 8))
        single_linkage_tree.plot(axis=ax)
        ax.set_title("Single Linkage Tree - Cluster Hierarchy")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        plt.show()
    else:
        print("No single linkage tree available")

    condensed_tree = hdbscan_artifacts.get("condensed_tree")
    if condensed_tree is not None:
        print("\nGenerating Condensed Tree visualization...")
        fig, ax = plt.subplots(figsize=(14, 8))
        condensed_tree.plot(axis=ax)
        ax.set_title("Condensed Tree - Cluster Selection")
        plt.show()


# Uncomment to visualize:
visualize_hierarchy(data['hdbscan_artifacts'])


#%% ============================================================================
# CLUSTER PERSISTENCE ANALYSIS
# ============================================================================

def analyze_cluster_persistence(hdbscan_artifacts: Optional[Dict]) -> None:
    """Analyze cluster persistence (stability) scores."""
    if not hdbscan_artifacts:
        print("No HDBSCAN artifacts available")
        return

    persistence = hdbscan_artifacts.get("cluster_persistence")
    if persistence is None:
        print("No cluster persistence data available")
        return

    print(f"\nCluster Persistence Scores:")
    print("-" * 40)
    for i, p in enumerate(persistence):
        print(f"  Cluster {i}: {p:.4f}")

    print(f"\nMean persistence: {np.mean(persistence):.4f}")
    print(f"Min persistence: {np.min(persistence):.4f}")
    print(f"Max persistence: {np.max(persistence):.4f}")


if data['hdbscan_artifacts']:
    analyze_cluster_persistence(data['hdbscan_artifacts'])


#%% ============================================================================
# OUTLIER SCORES ANALYSIS
# ============================================================================

def analyze_outlier_scores(hdbscan_artifacts: Optional[Dict], clusters_by_id: Dict) -> None:
    """Analyze outlier scores per cluster."""
    if not hdbscan_artifacts:
        print("No HDBSCAN artifacts available")
        return

    outlier_scores = hdbscan_artifacts.get("outlier_scores")
    if outlier_scores is None:
        print("No outlier scores available")
        return

    labels = hdbscan_artifacts.get("labels")
    if labels is None:
        print("No labels available")
        return

    print(f"\nOutlier Scores by Cluster:")
    print("-" * 50)

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        cluster_scores = outlier_scores[mask]

        print(f"  Cluster {cluster_id}: mean={np.mean(cluster_scores):.3f}, "
              f"max={np.max(cluster_scores):.3f}, "
              f"high_outlier_count={np.sum(cluster_scores > 0.5)}")


if data['hdbscan_artifacts']:
    analyze_outlier_scores(data['hdbscan_artifacts'], clusters_by_id)


#%% ============================================================================
# PRINT CURRENT LABELS (from clustering_metadata)
# ============================================================================

def print_current_labels(metadata: Optional[models.ClusteringMetadataModel]) -> None:
    """Print the current cluster labels from cache."""
    if not metadata:
        print("No clustering metadata available")
        return

    print(f"\nCurrent Cluster Labels:")
    print("=" * 70)

    for cluster_id in sorted(metadata.clusters.keys()):
        cluster = metadata.clusters[cluster_id]
        print(f"\nCluster {cluster_id} (n={cluster.size}):")
        print(f"  Theme: {cluster.label_theme or '(not set)'}")
        print(f"  Description: {cluster.label_description or '(not set)'}")
        if cluster.label_key_concepts:
            print(f"  Key concepts: {', '.join(cluster.label_key_concepts)}")
        print(f"  Mean probability: {cluster.mean_probability:.3f}" if cluster.mean_probability else "")
        print(f"  Coherence: {cluster.coherence:.3f}" if cluster.coherence else "")


print_current_labels(data['clustering_metadata'])


#%% ============================================================================
# EXPERIMENTAL LABEL GENERATION (placeholder)
# ============================================================================

# This section will use label_helpers.py once implemented

def generate_experimental_label(
    cluster_id: int,
    high_threshold: float = 0.8,
    low_threshold: float = 0.5,
    include_low_prob: bool = True
) -> None:
    """
    Generate a label using experimental prompts.

    This is a placeholder - will be implemented in label_helpers.py
    """
    print(f"\n[Experimental Label Generation - Cluster {cluster_id}]")

    # Get samples
    high_prob = get_high_prob_members(cluster_id, threshold=high_threshold, max_display=5)
    if include_low_prob:
        low_prob = get_low_prob_members(cluster_id, threshold=high_threshold, min_prob=low_threshold, max_display=5)

    # TODO: Use label_helpers.py to generate label with prompts_label_exp.py
    print("\n  [Label generation will use label_helpers.py - coming soon]")


# Example:
# generate_experimental_label(0)


#%% ============================================================================
# MAIN (when run as script)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LABEL EXPERIMENTS READY")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - analyze_probability_distribution(clusters_by_id, threshold)")
    print("  - get_low_prob_members(cluster_id, threshold, min_prob)")
    print("  - get_high_prob_members(cluster_id, threshold)")
    print("  - compare_thresholds(cluster_id, thresholds)")
    print("  - analyze_condensed_tree(hdbscan_artifacts)")
    print("  - visualize_hierarchy(hdbscan_artifacts)")
    print("  - analyze_cluster_persistence(hdbscan_artifacts)")
    print("  - analyze_outlier_scores(hdbscan_artifacts, clusters_by_id)")
    print("  - print_current_labels(metadata)")
    print("\nData available in:")
    print("  - data['cluster_models']")
    print("  - data['clustering_metadata']")
    print("  - data['hdbscan_artifacts']")
    print("  - clusters_by_id")
