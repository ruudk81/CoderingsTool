#%%
"""
HDBSCAN Condensed Tree Analysis

Dedicated module for exploring and visualizing the HDBSCAN hierarchy.

The condensed tree shows:
- How clusters form and split at different density levels
- Which points are core members vs boundary/noise
- Cluster stability (persistence)

Usage:
    Open in VS Code and run cells interactively.
"""

#%% ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import pandas as pd

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import generate_enhanced_variable_key
import models

print(f"Project root: {project_root}")


#%% ============================================================================
# CONFIGURATION
# ============================================================================

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

print(f"Dataset: {FILENAME}")
print(f"Variable: {VARIABLE}")
print(f"Sample size: {SAMPLE_SIZE}")


#%% ============================================================================
# LOAD HDBSCAN ARTIFACTS
# ============================================================================

def load_hdbscan_artifacts() -> Optional[Dict]:
    """Load HDBSCAN artifacts from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem
    cache_path = cache_dir / f"hdbscan_artifacts_{base_name}_{variable_key}.pkl"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN artifacts not found: {cache_path}\n"
            f"Run step_5_clusterer.run_experiment first."
        )

    print(f"Loading from: {cache_path.name}")

    with open(cache_path, 'rb') as f:
        artifacts = pickle.load(f)

    print(f"Artifacts loaded: {list(artifacts.keys())}")
    return artifacts


def load_cluster_models() -> List[models.ClusterModel]:
    """Load cluster models to map point IDs to texts."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem
    cache_path = cache_dir / f"006_initial_clusters_{base_name}_{variable_key}.pkl"

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    return [models.ClusterModel.model_validate(item) for item in data]


# Load data
artifacts = load_hdbscan_artifacts()
cluster_models = load_cluster_models()

# Extract key components
condensed_tree = artifacts.get("condensed_tree")
single_linkage_tree = artifacts.get("single_linkage_tree")
labels = artifacts.get("labels")
probabilities = artifacts.get("probabilities")
persistence = artifacts.get("cluster_persistence")
outlier_scores = artifacts.get("outlier_scores")

n_samples = len(labels)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"\nSamples: {n_samples}")
print(f"Clusters: {n_clusters}")
print(f"Noise points: {sum(labels == -1)}")


#%% ============================================================================
# BUILD POINT-TO-TEXT MAPPING
# ============================================================================

def load_template_prefix_from_cluster_models() -> Optional[str]:
    """Load template prefix from cluster models (already loaded)."""
    if cluster_models and len(cluster_models) > 0:
        return cluster_models[0].template_prefix
    return None


def strip_template_prefix(text: str, template_prefix: Optional[str]) -> str:
    """Strip template prefix from text for cleaner display."""
    if template_prefix and text.startswith(template_prefix):
        stripped = text[len(template_prefix):].strip()
        return stripped if stripped else text
    return text


def build_point_mapping(
    cluster_models: List[models.ClusterModel],
    template_prefix: Optional[str] = None
) -> List[Dict]:
    """
    Build a list mapping point index to idea details.

    Returns list where index = point_id, value = dict with idea info.
    If template_prefix is provided, 'display_text' contains the stripped version.
    """
    points = []

    for model in cluster_models:
        if model.response_ideas:
            for idea in model.response_ideas:
                raw_text = idea.idea
                display_text = strip_template_prefix(raw_text, template_prefix)
                points.append({
                    "text": raw_text,
                    "display_text": display_text,
                    "cluster": idea.initial_cluster,
                    "probability": idea.cluster_probability or 0.0,
                    "respondent_id": model.respondent_id,
                })

    return points


# Load template prefix from cluster models
template_prefix = load_template_prefix_from_cluster_models()
if template_prefix:
    prefix_display = template_prefix[:50] + "..." if len(template_prefix) > 50 else template_prefix
    print(f"Template prefix: '{prefix_display}'")
else:
    print("Template prefix: (none found)")

point_mapping = build_point_mapping(cluster_models, template_prefix)
print(f"Point mapping built: {len(point_mapping)} points")


#%% ============================================================================
# CONDENSED TREE: BASIC STRUCTURE
# ============================================================================

def get_tree_dataframe() -> pd.DataFrame:
    """Get condensed tree as DataFrame."""
    if condensed_tree is None:
        print("No condensed tree available")
        return pd.DataFrame()

    df = condensed_tree.to_pandas()
    print(f"Condensed tree: {len(df)} edges")
    print(f"Columns: {list(df.columns)}")
    return df


tree_df = get_tree_dataframe()
print(f"\nFirst 10 edges:")
print(tree_df.head(10).to_string())


#%% ============================================================================
# ANALYZE TREE STRUCTURE
# ============================================================================

def analyze_tree_structure(tree_df: pd.DataFrame, n_samples: int) -> Dict:
    """
    Analyze the condensed tree structure.

    Returns dict with analysis results.
    """
    # Separate cluster splits from point fallouts
    cluster_nodes = tree_df[tree_df['child'] >= n_samples].copy()
    point_fallouts = tree_df[tree_df['child'] < n_samples].copy()

    print(f"{'='*60}")
    print("CONDENSED TREE ANALYSIS")
    print(f"{'='*60}")
    print(f"\nTotal samples: {n_samples}")
    print(f"Cluster splits (internal nodes): {len(cluster_nodes)}")
    print(f"Point fallouts (leaves): {len(point_fallouts)}")

    # Cluster hierarchy
    print(f"\n--- Cluster Hierarchy ---")
    for _, row in cluster_nodes.sort_values('lambda_val').iterrows():
        parent = int(row['parent'])
        child = int(row['child'])
        size = int(row['child_size'])
        lam = row['lambda_val']
        print(f"  {parent} -> {child} (size={size:3d}, lambda={lam:.2f})")

    # Lambda distribution for fallouts
    print(f"\n--- Point Fallout Distribution ---")
    lambda_bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
    labels_bins = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100+']
    point_fallouts['lambda_bin'] = pd.cut(
        point_fallouts['lambda_val'],
        bins=lambda_bins,
        labels=labels_bins
    )
    bin_counts = point_fallouts['lambda_bin'].value_counts().sort_index()
    for bin_label, count in bin_counts.items():
        pct = count / len(point_fallouts) * 100
        bar = '#' * int(pct / 2)
        print(f"  lambda {bin_label:>7}: {count:4d} ({pct:5.1f}%) {bar}")

    return {
        "cluster_nodes": cluster_nodes,
        "point_fallouts": point_fallouts,
        "n_cluster_splits": len(cluster_nodes),
        "n_point_fallouts": len(point_fallouts),
    }


analysis = analyze_tree_structure(tree_df, n_samples)


#%% ============================================================================
# IDENTIFY EARLY FALLOUTS (WEAK MEMBERS)
# ============================================================================

def get_early_fallouts(
    tree_df: pd.DataFrame,
    n_samples: int,
    lambda_threshold: float = 15.0,
    show_texts: bool = True,
    max_display: int = 20
) -> pd.DataFrame:
    """
    Get points that fell out early (low lambda = weak cluster membership).

    These are boundary/noise points that left clusters at low density.
    """
    point_fallouts = tree_df[tree_df['child'] < n_samples].copy()
    early = point_fallouts[point_fallouts['lambda_val'] < lambda_threshold].copy()
    early = early.sort_values('lambda_val')

    print(f"\n{'='*60}")
    print(f"EARLY FALLOUTS (lambda < {lambda_threshold})")
    print(f"{'='*60}")
    print(f"Found {len(early)} points that fell out early")

    if show_texts and len(early) > 0:
        print(f"\nEarliest {min(max_display, len(early))} fallouts:")
        print("-" * 60)

        for i, (_, row) in enumerate(early.head(max_display).iterrows()):
            point_id = int(row['child'])
            lam = row['lambda_val']
            parent = int(row['parent'])

            if point_id < len(point_mapping):
                info = point_mapping[point_id]
                text = info['display_text'][:60] + "..." if len(info['display_text']) > 60 else info['display_text']
                cluster = info['cluster']
                prob = info['probability']
                print(f"  [{lam:5.1f}] cluster={cluster}, prob={prob:.2f}: {text}")
            else:
                print(f"  [{lam:5.1f}] point_id={point_id} (parent={parent})")

    return early


early_fallouts = get_early_fallouts(tree_df, n_samples, lambda_threshold=15.0)


#%% ============================================================================
# IDENTIFY LATE FALLOUTS (STRONG CORE MEMBERS)
# ============================================================================

def get_late_fallouts(
    tree_df: pd.DataFrame,
    n_samples: int,
    lambda_threshold: float = 40.0,
    show_texts: bool = True,
    max_display: int = 20
) -> pd.DataFrame:
    """
    Get points that fell out late (high lambda = strong cluster membership).

    These are core members that persisted until high density levels.
    """
    point_fallouts = tree_df[tree_df['child'] < n_samples].copy()
    late = point_fallouts[point_fallouts['lambda_val'] > lambda_threshold].copy()
    late = late.sort_values('lambda_val', ascending=False)

    print(f"\n{'='*60}")
    print(f"LATE FALLOUTS (lambda > {lambda_threshold})")
    print(f"{'='*60}")
    print(f"Found {len(late)} points that persisted until high density")

    if show_texts and len(late) > 0:
        print(f"\nLatest {min(max_display, len(late))} fallouts (strongest members):")
        print("-" * 60)

        for i, (_, row) in enumerate(late.head(max_display).iterrows()):
            point_id = int(row['child'])
            lam = row['lambda_val']

            if point_id < len(point_mapping):
                info = point_mapping[point_id]
                text = info['display_text'][:60] + "..." if len(info['display_text']) > 60 else info['display_text']
                cluster = info['cluster']
                prob = info['probability']
                print(f"  [{lam:5.1f}] cluster={cluster}, prob={prob:.2f}: {text}")

    return late


late_fallouts = get_late_fallouts(tree_df, n_samples, lambda_threshold=40.0)


#%% ============================================================================
# ANALYZE FALLOUTS BY CLUSTER
# ============================================================================

def analyze_fallouts_by_cluster(
    tree_df: pd.DataFrame,
    n_samples: int
) -> pd.DataFrame:
    """
    Analyze when points from each cluster fell out.

    Shows cluster "tightness" - clusters where points fall out late are more cohesive.
    """
    point_fallouts = tree_df[tree_df['child'] < n_samples].copy()

    # Add cluster info
    point_fallouts['cluster'] = point_fallouts['child'].apply(
        lambda x: point_mapping[x]['cluster'] if x < len(point_mapping) else -1
    )

    print(f"\n{'='*60}")
    print("FALLOUT TIMING BY CLUSTER")
    print(f"{'='*60}")
    print("\nHigher mean lambda = tighter/more cohesive cluster")
    print("-" * 60)

    cluster_stats = point_fallouts.groupby('cluster')['lambda_val'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    cluster_stats = cluster_stats.sort_values('mean', ascending=False)

    print(cluster_stats.to_string())

    return cluster_stats


cluster_fallout_stats = analyze_fallouts_by_cluster(tree_df, n_samples)


#%% ============================================================================
# VISUALIZE CONDENSED TREE
# ============================================================================

def visualize_condensed_tree(save_path: Optional[Path] = None):
    """Visualize the condensed tree."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if condensed_tree is None:
        print("No condensed tree available")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    condensed_tree.plot(
        axis=ax,
        select_clusters=True,  # Highlight selected clusters
        label_clusters=True,   # Label cluster nodes
    )
    ax.set_title("HDBSCAN Condensed Tree\n(Selected clusters highlighted)")
    ax.set_xlabel("Lambda (1/distance) - higher = denser")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


# Uncomment to visualize:
visualize_condensed_tree()

# Or save to file:
# visualize_condensed_tree(save_path=project_root / "exports" / "condensed_tree.png")


#%% ============================================================================
# VISUALIZE SINGLE LINKAGE TREE (DENDROGRAM)
# ============================================================================

def visualize_single_linkage_tree(
    truncate_mode: str = None,
    p: int = 20,
    save_path: Optional[Path] = None
):
    """
    Visualize the single linkage tree (full dendrogram).

    Args:
        truncate_mode: 'lastp' to show only last p merges, None for full tree
        p: Number of merges to show if truncate_mode='lastp'
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if single_linkage_tree is None:
        print("No single linkage tree available")
        return

    fig, ax = plt.subplots(figsize=(16, 10))

    single_linkage_tree.plot(
        axis=ax,
        truncate_mode=truncate_mode,
        p=p,
    )

    title = "HDBSCAN Single Linkage Tree (Full Dendrogram)"
    if truncate_mode == 'lastp':
        title = f"HDBSCAN Single Linkage Tree (Last {p} merges)"
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


# Uncomment to visualize:
visualize_single_linkage_tree()

# Or show truncated version (last 30 merges):
# visualize_single_linkage_tree(truncate_mode='lastp', p=30)


#%% ============================================================================
# CLUSTER PERSISTENCE / STABILITY
# ============================================================================

def analyze_cluster_persistence():
    """Analyze cluster persistence (stability) scores."""
    if persistence is None:
        print("No persistence data available")
        return

    print(f"\n{'='*60}")
    print("CLUSTER PERSISTENCE (STABILITY)")
    print(f"{'='*60}")
    print("\nHigher persistence = cluster exists over wider density range = more stable")
    print("-" * 60)

    for i, p in enumerate(persistence):
        # Get cluster size
        cluster_size = sum(labels == i)
        bar = '#' * int(p * 50)
        print(f"  Cluster {i}: {p:.4f} (n={cluster_size:3d}) {bar}")

    print(f"\n  Mean: {np.mean(persistence):.4f}")
    print(f"  Std:  {np.std(persistence):.4f}")


analyze_cluster_persistence()


#%% ============================================================================
# LAMBDA VALUE VS PROBABILITY CORRELATION
# ============================================================================

def correlate_lambda_and_probability(tree_df: pd.DataFrame, n_samples: int):
    """
    Check if lambda (fallout timing) correlates with cluster probability.

    We expect: higher lambda (late fallout) = higher probability
    """
    try:
        from scipy import stats
    except ImportError:
        print("scipy not available for correlation")
        return

    point_fallouts = tree_df[tree_df['child'] < n_samples].copy()

    # Get probabilities for each point
    point_fallouts['probability'] = point_fallouts['child'].apply(
        lambda x: point_mapping[x]['probability'] if x < len(point_mapping) else 0.0
    )

    # Calculate correlation
    corr, p_value = stats.pearsonr(
        point_fallouts['lambda_val'],
        point_fallouts['probability']
    )

    print(f"\n{'='*60}")
    print("LAMBDA vs PROBABILITY CORRELATION")
    print(f"{'='*60}")
    print(f"\nCorrelation: {corr:.3f} (p-value: {p_value:.2e})")

    if corr > 0.5:
        print("Strong positive correlation - lambda is a good proxy for probability")
    elif corr > 0.3:
        print("Moderate positive correlation")
    else:
        print("Weak correlation - lambda and probability capture different aspects")

    # Visualize
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            point_fallouts['lambda_val'],
            point_fallouts['probability'],
            alpha=0.5,
            s=10
        )
        ax.set_xlabel("Lambda (fallout timing)")
        ax.set_ylabel("Cluster Probability")
        ax.set_title(f"Lambda vs Probability (r={corr:.3f})")
        plt.tight_layout()
        plt.show()

    except ImportError:
        pass


correlate_lambda_and_probability(tree_df, n_samples)


#%% ============================================================================
# LEAF CLUSTER ANALYSIS (ATOMIC MICRO-THEMES)
# ============================================================================

def identify_leaf_nodes(tree_df: pd.DataFrame, n_samples: int) -> List[int]:
    """
    Find leaf nodes - clusters whose children are all single points.

    Leaf clusters are the atomic micro-themes at the bottom of the hierarchy.
    They represent the finest-grained groupings before everything splits
    into individual data points.

    A leaf is a node that:
    - Appears as a parent at least once
    - All its children have child_size = 1 (single points)

    Args:
        tree_df: Condensed tree DataFrame
        n_samples: Number of original data points

    Returns:
        List of leaf node IDs (sorted)
    """
    parents = set(tree_df["parent"].unique())
    leaf_nodes = []

    for p in parents:
        children_of_p = tree_df[tree_df["parent"] == p]
        p_size = children_of_p["child_size"].sum()

        if p_size > 1:  # real cluster, not single point
            if all(children_of_p["child_size"] == 1):
                leaf_nodes.append(int(p))

    return sorted(leaf_nodes)


def map_leaves_to_points(tree_df: pd.DataFrame, leaf_nodes: List[int]) -> Dict[int, set]:
    """
    Build mapping: leaf_node -> set of original data point indices.

    Propagates points up the tree for leaves built from intermediate nodes.

    Args:
        tree_df: Condensed tree DataFrame
        leaf_nodes: List of leaf node IDs

    Returns:
        Dict mapping leaf_id -> set of point indices
    """
    node_to_points = {}

    # Start with direct children (single points)
    for _, row in tree_df.iterrows():
        parent = int(row["parent"])
        child = int(row["child"])
        size = int(row["child_size"])

        if size == 1:  # child is a single data point
            node_to_points.setdefault(parent, set()).add(child)

    # Propagate points upward through the tree
    changed = True
    iterations = 0
    while changed and iterations < 100:  # safety limit
        changed = False
        iterations += 1
        for _, row in tree_df.iterrows():
            p, c, size = int(row["parent"]), int(row["child"]), int(row["child_size"])
            if size > 1 and c in node_to_points:
                if p not in node_to_points:
                    node_to_points[p] = set()
                before = len(node_to_points[p])
                node_to_points[p] |= node_to_points[c]
                if len(node_to_points[p]) > before:
                    changed = True

    return {l: node_to_points.get(l, set()) for l in leaf_nodes}


def map_leaves_to_parent_clusters(
    leaf_to_points: Dict[int, set],
    labels: np.ndarray
) -> Dict[int, int]:
    """
    Assign each leaf to the parent HDBSCAN cluster that dominates its points.

    Uses majority voting: the parent cluster is whichever cluster contains
    the most points from this leaf.

    Args:
        leaf_to_points: Dict mapping leaf_id -> set of point indices
        labels: HDBSCAN cluster labels for all points

    Returns:
        Dict mapping leaf_id -> parent cluster_id
    """
    from collections import Counter

    leaf_parent = {}
    for leaf, pts in leaf_to_points.items():
        if not pts:
            leaf_parent[leaf] = -1
            continue

        labs = labels[list(pts)]
        labs = labs[labs != -1]  # ignore noise points
        if len(labs) == 0:
            leaf_parent[leaf] = -1
        else:
            leaf_parent[leaf] = int(Counter(labs).most_common(1)[0][0])

    return leaf_parent


def print_leaf_samples(
    point_mapping: List[Dict],
    leaf_to_points: Dict[int, set],
    leaf_parent: Dict[int, int],
    probabilities: np.ndarray = None,
    n_samples: int = 5
):
    """
    Print text samples organized by parent cluster -> leaf.

    Shows the atomic micro-themes within each parent cluster.

    Args:
        point_mapping: List mapping point_id -> {text, cluster, probability}
        leaf_to_points: Dict mapping leaf_id -> set of point indices
        leaf_parent: Dict mapping leaf_id -> parent cluster_id
        probabilities: Optional array of cluster probabilities
        n_samples: Number of samples to show per leaf
    """
    for parent in sorted(set(leaf_parent.values())):
        if parent == -1:
            continue

        # Get all leaves for this parent, sorted by size (largest first)
        leaves = [l for l, p in leaf_parent.items() if p == parent]
        leaves = sorted(leaves, key=lambda l: len(leaf_to_points[l]), reverse=True)

        total_points = sum(len(leaf_to_points[l]) for l in leaves)

        print(f"\n{'='*80}")
        print(f"PARENT CLUSTER {parent} ({len(leaves)} leaves, {total_points} total points)")
        print(f"{'='*80}")

        for leaf in leaves:
            pts = list(leaf_to_points[leaf])
            print(f"\n  Leaf {leaf} (n={len(pts)})")
            print(f"  {'-'*70}")

            # Sort by probability if available
            if probabilities is not None:
                pts = sorted(pts, key=lambda i: probabilities[i] if i < len(probabilities) else 0, reverse=True)

            for i in pts[:n_samples]:
                if i < len(point_mapping):
                    info = point_mapping[i]
                    text = info['display_text'][:65] + "..." if len(info['display_text']) > 65 else info['display_text']
                    prob = info['probability']
                    print(f"    [{prob:.2f}] {text}")


def analyze_leaf_statistics(
    leaf_to_points: Dict[int, set],
    leaf_parent: Dict[int, int]
) -> None:
    """
    Print statistics about leaf clusters.
    """
    print(f"\n{'='*60}")
    print("LEAF CLUSTER STATISTICS")
    print(f"{'='*60}")

    sizes = [len(pts) for pts in leaf_to_points.values()]
    print(f"\nTotal leaves: {len(leaf_to_points)}")
    print(f"Total points in leaves: {sum(sizes)}")
    print(f"Leaf sizes: min={min(sizes)}, median={np.median(sizes):.0f}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    # Leaves per parent cluster
    parent_counts = {}
    for leaf, parent in leaf_parent.items():
        parent_counts[parent] = parent_counts.get(parent, 0) + 1

    print(f"\nLeaves per parent cluster:")
    for parent in sorted(parent_counts.keys()):
        if parent == -1:
            continue
        count = parent_counts[parent]
        total_pts = sum(len(leaf_to_points[l]) for l, p in leaf_parent.items() if p == parent)
        print(f"  Cluster {parent}: {count} leaves, {total_pts} points")


#%% ============================================================================
# INTERMEDIATE NODES (hierarchy ABOVE leaves)
# ============================================================================

def identify_intermediate_nodes(tree_df: pd.DataFrame, n_samples: int) -> List[Tuple[int, int]]:
    """
    Find intermediate nodes - clusters that have multi-point children.

    These are nodes ABOVE the leaves in the hierarchy - they represent
    potential higher-level groupings that HDBSCAN didn't select.

    Returns:
        List of (node_id, total_size) tuples, sorted by size descending
    """
    parents = set(tree_df["parent"].unique())
    intermediate_nodes = []

    for p in parents:
        children_of_p = tree_df[tree_df["parent"] == p]

        # Has at least one multi-point child (not a leaf)
        has_multipoint_child = any(children_of_p["child_size"] > 1)

        if has_multipoint_child:
            total_size = int(children_of_p["child_size"].sum())
            intermediate_nodes.append((int(p), total_size))

    # Sort by size descending
    intermediate_nodes.sort(key=lambda x: x[1], reverse=True)
    return intermediate_nodes


def analyze_hierarchy_depth(tree_df: pd.DataFrame, n_samples: int) -> None:
    """
    Analyze the full hierarchy structure - leaves vs intermediate vs root.

    This shows WHY each HDBSCAN cluster = 1 leaf:
    - If there are intermediate nodes, HDBSCAN chose NOT to use them
    - The algorithm preferred the leaf-level clusters (more "excess mass")
    """
    print(f"\n{'='*60}")
    print("HIERARCHY STRUCTURE ANALYSIS")
    print(f"{'='*60}")

    # Identify node types
    leaf_nodes = identify_leaf_nodes(tree_df, n_samples)
    intermediate = identify_intermediate_nodes(tree_df, n_samples)

    print(f"\nLeaf nodes (all children are single points): {len(leaf_nodes)}")
    print(f"Intermediate nodes (have multi-point children): {len(intermediate)}")

    if len(intermediate) == 0:
        print("\n  -> No intermediate nodes = hierarchy is flat")
        print("     HDBSCAN found clusters at the finest granularity only")
    else:
        print(f"\n  -> There ARE {len(intermediate)} intermediate nodes!")
        print("     HDBSCAN chose leaf-level clusters over these higher-level groupings")
        print("     This suggests leaf clusters have better 'excess mass' (persistence)")

    print(f"\n--- Intermediate Nodes (potential higher-level clusters) ---")
    print(f"{'Node':>8} | {'Size':>6} | Structure")
    print("-" * 60)

    for node, size in intermediate[:15]:
        # Check what children this node has
        children = tree_df[tree_df["parent"] == node]
        child_sizes = children["child_size"].tolist()
        multipoint = [int(s) for s in child_sizes if s > 1]
        singlepoint = len([s for s in child_sizes if s == 1])

        if multipoint:
            desc = f"sub-clusters: {multipoint}"
            if singlepoint:
                desc += f" + {singlepoint} singles"
        else:
            desc = f"{singlepoint} single points"
        print(f"{node:>8} | {size:>6} | {desc}")

    if len(intermediate) > 15:
        print(f"  ... and {len(intermediate) - 15} more intermediate nodes")


def show_cluster_hierarchy_path(
    tree_df: pd.DataFrame,
    cluster_id: int,
    labels: np.ndarray,
    n_samples: int
) -> None:
    """
    Show the hierarchy path for a specific HDBSCAN cluster.

    Traces from leaf up to root to show where this cluster sits in the tree.
    """
    print(f"\n{'='*60}")
    print(f"HIERARCHY PATH FOR CLUSTER {cluster_id}")
    print(f"{'='*60}")

    # Find which leaf node corresponds to this cluster
    leaf_nodes_list = identify_leaf_nodes(tree_df, n_samples)
    leaf_to_pts = map_leaves_to_points(tree_df, leaf_nodes_list)

    cluster_leaf = None
    for leaf, pts in leaf_to_pts.items():
        if pts:
            leaf_labels = labels[list(pts)]
            if np.sum(leaf_labels == cluster_id) > len(pts) / 2:
                cluster_leaf = leaf
                break

    if cluster_leaf is None:
        print(f"  Could not find leaf for cluster {cluster_id}")
        return

    print(f"\nCluster {cluster_id} corresponds to leaf node {cluster_leaf}")

    # Trace upward to root
    current = cluster_leaf
    path = [(current, len(leaf_to_pts.get(current, set())))]

    for _ in range(20):  # safety limit
        parent_row = tree_df[tree_df["child"] == current]
        if parent_row.empty:
            break
        parent = int(parent_row.iloc[0]["parent"])

        # Get size of parent
        parent_children = tree_df[tree_df["parent"] == parent]
        parent_size = int(parent_children["child_size"].sum())

        path.append((parent, parent_size))
        current = parent

    print(f"\nPath from leaf to root (bottom-up):")
    for i, (node, size) in enumerate(path):
        indent = "  " * i
        if i == 0:
            marker = f"[LEAF = Cluster {cluster_id}]"
        elif i == len(path) - 1:
            marker = "[ROOT]"
        else:
            marker = "[intermediate]"
        print(f"{indent}└── Node {node} (n={size}) {marker}")


#%% RUN HIERARCHY ANALYSIS
analyze_hierarchy_depth(tree_df, n_samples)


#%% RUN LEAF ANALYSIS
leaf_nodes = identify_leaf_nodes(tree_df, n_samples)
print(f"\nFound {len(leaf_nodes)} leaf clusters (atomic micro-themes)")

leaf_to_points = map_leaves_to_points(tree_df, leaf_nodes)
leaf_parent = map_leaves_to_parent_clusters(leaf_to_points, labels)

analyze_leaf_statistics(leaf_to_points, leaf_parent)


#%% SHOW HIERARCHY PATH FOR EACH CLUSTER
for cluster_id in sorted(set(labels) - {-1}):
    show_cluster_hierarchy_path(tree_df, cluster_id, labels, n_samples)


#%% PRINT LEAF SAMPLES
print_leaf_samples(point_mapping, leaf_to_points, leaf_parent, probabilities, n_samples=5)


#%% ============================================================================
# VISUALIZE CLUSTER HIERARCHY AS TREE DIAGRAM
# ============================================================================

def build_hierarchy_graph(tree_df: pd.DataFrame, n_samples: int, labels: np.ndarray):
    """
    Build a graph representation of the cluster hierarchy.

    Returns:
        Tuple of (nodes, edges, node_info) where:
        - nodes: list of node IDs
        - edges: list of (parent, child) tuples
        - node_info: dict mapping node_id -> {size, type, cluster_id}
    """
    leaf_nodes_set = set(identify_leaf_nodes(tree_df, n_samples))
    leaf_to_pts = map_leaves_to_points(tree_df, list(leaf_nodes_set))

    # Build node info
    node_info = {}
    edges = []

    # Get all multi-point relationships (ignore single-point fallouts)
    for _, row in tree_df.iterrows():
        parent = int(row["parent"])
        child = int(row["child"])
        size = int(row["child_size"])

        if size > 1:  # Only multi-point nodes
            edges.append((parent, child))

            # Add child info
            if child not in node_info:
                if child in leaf_nodes_set:
                    # Find which HDBSCAN cluster this leaf corresponds to
                    pts = leaf_to_pts.get(child, set())
                    if pts:
                        cluster_labels = labels[list(pts)]
                        cluster_id = int(np.bincount(cluster_labels[cluster_labels >= 0]).argmax()) if any(cluster_labels >= 0) else -1
                    else:
                        cluster_id = -1
                    node_info[child] = {"size": size, "type": "leaf", "cluster_id": cluster_id}
                else:
                    node_info[child] = {"size": size, "type": "intermediate", "cluster_id": None}

            # Add parent info
            if parent not in node_info:
                parent_children = tree_df[tree_df["parent"] == parent]
                parent_size = int(parent_children["child_size"].sum())
                node_info[parent] = {"size": parent_size, "type": "root" if parent_size == n_samples else "intermediate", "cluster_id": None}

    nodes = list(node_info.keys())
    return nodes, edges, node_info


def visualize_cluster_tree(
    tree_df: pd.DataFrame,
    n_samples: int,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Visualize the cluster hierarchy as a tree diagram.

    Shows:
    - Root node at top
    - Intermediate nodes in middle
    - Leaf nodes (HDBSCAN clusters) at bottom
    - Node sizes and cluster IDs labeled

    Args:
        tree_df: Condensed tree DataFrame
        n_samples: Number of original data points
        labels: HDBSCAN cluster labels
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available")
        return

    nodes, edges, node_info = build_hierarchy_graph(tree_df, n_samples, labels)

    if not nodes:
        print("No hierarchy to visualize")
        return

    # Build adjacency for layout
    children_of = {}
    for parent, child in edges:
        children_of.setdefault(parent, []).append(child)

    # Find root (node with largest size)
    root = max(nodes, key=lambda n: node_info[n]["size"])

    # Compute positions using recursive layout
    positions = {}
    x_counter = [0]  # Use list for mutable counter in nested function

    def layout_node(node, depth):
        """Recursively layout nodes, leaves get x positions, parents center over children."""
        if node not in children_of or not children_of[node]:
            # Leaf node - assign next x position
            positions[node] = (x_counter[0], -depth)
            x_counter[0] += 1
        else:
            # Internal node - layout children first
            child_xs = []
            for child in sorted(children_of[node], key=lambda c: -node_info.get(c, {}).get("size", 0)):
                layout_node(child, depth + 1)
                child_xs.append(positions[child][0])
            # Center parent over children
            positions[node] = (sum(child_xs) / len(child_xs), -depth)

    layout_node(root, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    for parent, child in edges:
        if parent in positions and child in positions:
            px, py = positions[parent]
            cx, cy = positions[child]
            ax.plot([px, cx], [py, cy], 'k-', linewidth=1, alpha=0.5, zorder=1)

    # Draw nodes
    for node, (x, y) in positions.items():
        info = node_info.get(node, {})
        node_type = info.get("type", "unknown")
        size = info.get("size", 0)
        cluster_id = info.get("cluster_id")

        # Node color based on type
        if node_type == "root":
            color = "#2E86AB"  # Blue
            marker_size = 800
        elif node_type == "leaf":
            color = "#A23B72"  # Magenta/pink
            marker_size = 600
        else:
            color = "#F18F01"  # Orange
            marker_size = 400

        # Draw node
        ax.scatter([x], [y], s=marker_size, c=[color], zorder=2, edgecolors='white', linewidths=2)

        # Label
        if node_type == "leaf" and cluster_id is not None and cluster_id >= 0:
            label = f"C{cluster_id}\n(n={size})"
        else:
            label = f"{node}\n(n={size})"

        ax.annotate(label, (x, y), ha='center', va='center', fontsize=8,
                   fontweight='bold', color='white' if node_type != "intermediate" else 'black',
                   zorder=3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2E86AB', edgecolor='white', label='Root'),
        mpatches.Patch(facecolor='#F18F01', edgecolor='white', label='Intermediate'),
        mpatches.Patch(facecolor='#A23B72', edgecolor='white', label='Leaf (HDBSCAN Cluster)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Styling
    ax.set_title("Cluster Hierarchy Tree\n(Root at top, HDBSCAN clusters at bottom)", fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")

    plt.show()

    return positions


def visualize_cluster_tree_with_siblings(
    tree_df: pd.DataFrame,
    n_samples: int,
    labels: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Visualize tree highlighting sibling relationships.

    Siblings are clusters that share an intermediate parent node.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available")
        return

    nodes, edges, node_info = build_hierarchy_graph(tree_df, n_samples, labels)

    # Find siblings (leaves that share a parent)
    parent_to_leaves = {}
    for parent, child in edges:
        child_info = node_info.get(child, {})
        if child_info.get("type") == "leaf":
            parent_to_leaves.setdefault(parent, []).append(child)

    sibling_groups = {p: leaves for p, leaves in parent_to_leaves.items() if len(leaves) > 1}

    print(f"\n{'='*60}")
    print("SIBLING CLUSTERS (share intermediate parent)")
    print(f"{'='*60}")

    if not sibling_groups:
        print("\nNo sibling clusters found - all clusters have unique paths")
    else:
        for parent, leaves in sibling_groups.items():
            parent_info = node_info.get(parent, {})
            cluster_ids = [node_info[l].get("cluster_id") for l in leaves]
            print(f"\nParent node {parent} (n={parent_info.get('size', '?')}):")
            for leaf, cid in zip(leaves, cluster_ids):
                leaf_info = node_info.get(leaf, {})
                print(f"  - Cluster {cid} (leaf {leaf}, n={leaf_info.get('size', '?')})")

    # Visualize with siblings highlighted
    visualize_cluster_tree(tree_df, n_samples, labels, save_path)


#%% VISUALIZE THE TREE
visualize_cluster_tree(tree_df, n_samples, labels, save_path=project_root / "exports" / "cluster_hierarchy_tree.png")


#%% SHOW SIBLING RELATIONSHIPS
visualize_cluster_tree_with_siblings(tree_df, n_samples, labels)


#%% ============================================================================
# SINGLE LINKAGE TREE ANALYSIS (FULL DENDROGRAM - FINER GRANULARITY)
# ============================================================================
"""
The single_linkage_tree is the FULL hierarchical dendrogram, unlike condensed_tree
which only shows clusters meeting min_cluster_size.

This section provides tools to explore internal sub-structure within EOM clusters
that the condensed_tree hides.
"""

def get_single_linkage_array() -> Optional[np.ndarray]:
    """
    Get single linkage tree as numpy array (scipy linkage format).

    Format: Each row is [child1, child2, distance, n_points]
    - child1, child2: indices of merged clusters (< n_samples = original points)
    - distance: distance at which merge occurred
    - n_points: total points in the new cluster

    Returns:
        numpy array of shape (n_samples-1, 4) or None if not available
    """
    if single_linkage_tree is None:
        print("No single linkage tree available")
        return None

    # HDBSCAN's single_linkage_tree has a to_numpy() method
    linkage_array = single_linkage_tree.to_numpy()
    print(f"Single linkage tree: {len(linkage_array)} merges")
    print(f"Format: [child1, child2, distance, n_points]")
    return linkage_array


def extract_cluster_points(cluster_id: int, labels: np.ndarray) -> np.ndarray:
    """Get indices of all points belonging to a specific cluster."""
    return np.where(labels == cluster_id)[0]


def get_cluster_subtree(
    cluster_id: int,
    labels: np.ndarray,
    linkage_array: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Extract the portion of the single linkage tree for points in a specific cluster.

    This creates a NEW linkage matrix containing only the merges relevant to
    the given cluster's points.

    Args:
        cluster_id: The EOM cluster to analyze
        labels: HDBSCAN cluster labels
        linkage_array: Full single linkage tree

    Returns:
        Tuple of:
        - subtree: Linkage array for just this cluster (scipy format)
        - point_map: Dict mapping original point indices to subtree indices
    """
    from scipy.cluster.hierarchy import fcluster

    # Get points in this cluster
    cluster_points = extract_cluster_points(cluster_id, labels)
    n_cluster = len(cluster_points)

    if n_cluster < 2:
        print(f"Cluster {cluster_id} has only {n_cluster} points - no subtree possible")
        return np.array([]), {}

    # Create mapping: original index -> subtree index (0 to n_cluster-1)
    point_map = {orig: new for new, orig in enumerate(cluster_points)}

    # Filter linkage to only include merges within this cluster
    # This is tricky because linkage references both original points AND
    # intermediate clusters (indices >= n_samples)

    n_samples = len(labels)
    cluster_set = set(cluster_points)

    # Track which nodes (original points and intermediate clusters) belong to our cluster
    node_belongs_to_cluster = {p: True for p in cluster_points}

    subtree_rows = []
    next_cluster_id = n_cluster  # New cluster IDs start after original points

    # Map old intermediate node IDs to new ones
    old_to_new_node = dict(point_map)  # Start with point mappings

    for row in linkage_array:
        c1, c2, dist, count = int(row[0]), int(row[1]), row[2], int(row[3])

        # Check if both children belong to our cluster
        c1_in = node_belongs_to_cluster.get(c1, False)
        c2_in = node_belongs_to_cluster.get(c2, False)

        if c1_in and c2_in:
            # This merge is within our cluster
            new_c1 = old_to_new_node[c1]
            new_c2 = old_to_new_node[c2]

            # New merged cluster gets next available ID
            new_merged_id = next_cluster_id
            old_merged_id = n_samples + len(subtree_rows)

            subtree_rows.append([new_c1, new_c2, dist, count])
            old_to_new_node[old_merged_id] = new_merged_id
            node_belongs_to_cluster[old_merged_id] = True
            next_cluster_id += 1

        elif c1_in or c2_in:
            # Merge crosses cluster boundary - mark the merged node
            old_merged_id = n_samples + len([r for r in linkage_array
                                              if r[0] <= row[0] and r[1] <= row[1]])
            # This merged node is not purely in our cluster

    if len(subtree_rows) == 0:
        print(f"No internal structure found for cluster {cluster_id}")
        return np.array([]), point_map

    subtree = np.array(subtree_rows)
    print(f"Cluster {cluster_id}: {n_cluster} points, {len(subtree_rows)} internal merges")

    return subtree, point_map


def analyze_cluster_internal_structure(
    cluster_id: int,
    labels: np.ndarray,
    linkage_array: np.ndarray,
    point_mapping: List[Dict],
    distance_percentiles: List[float] = [25, 50, 75, 90],
    show_samples: bool = True,
    n_samples_per_group: int = 3
) -> Dict:
    """
    Analyze the internal hierarchical structure of a specific EOM cluster.

    This reveals sub-groupings that the condensed tree hides.

    Args:
        cluster_id: The EOM cluster to analyze
        labels: HDBSCAN cluster labels
        linkage_array: Full single linkage tree
        point_mapping: List mapping point indices to text/info
        distance_percentiles: Distance thresholds to cut the tree (as percentiles)
        show_samples: Whether to print text samples
        n_samples_per_group: Samples to show per sub-group

    Returns:
        Dict with analysis results
    """
    from scipy.cluster.hierarchy import fcluster

    print(f"\n{'='*70}")
    print(f"INTERNAL STRUCTURE OF CLUSTER {cluster_id}")
    print(f"{'='*70}")

    cluster_points = extract_cluster_points(cluster_id, labels)
    n_cluster = len(cluster_points)

    if n_cluster < 2:
        print(f"Cluster {cluster_id} has only {n_cluster} points")
        return {"cluster_id": cluster_id, "n_points": n_cluster, "sub_groups": []}

    print(f"Cluster size: {n_cluster} points")

    # Get subtree
    subtree, point_map = get_cluster_subtree(cluster_id, labels, linkage_array)

    if len(subtree) == 0:
        # Fallback: use fcluster on full linkage but filter to cluster points
        print("Using fallback: cutting full tree and filtering to cluster points")

        # Get distances in the tree for this cluster's merges
        cluster_set = set(cluster_points)
        relevant_distances = []
        for row in linkage_array:
            c1, c2 = int(row[0]), int(row[1])
            if c1 in cluster_set or c2 in cluster_set:
                relevant_distances.append(row[2])

        if not relevant_distances:
            print("No relevant distances found")
            return {"cluster_id": cluster_id, "n_points": n_cluster, "sub_groups": []}

        # Use percentiles of these distances
        dist_thresholds = np.percentile(relevant_distances, distance_percentiles)
    else:
        # Use percentiles from subtree distances
        dist_thresholds = np.percentile(subtree[:, 2], distance_percentiles)

    print(f"\n--- Sub-groupings at different distance thresholds ---")
    print(f"Distance percentiles: {distance_percentiles}")
    print(f"Distance values: {[f'{d:.4f}' for d in dist_thresholds]}")

    results = {
        "cluster_id": cluster_id,
        "n_points": n_cluster,
        "cuts": []
    }

    # Cut at each threshold
    for pct, thresh in zip(distance_percentiles, dist_thresholds):
        # Use fcluster on full tree, then filter to our cluster
        all_sub_labels = fcluster(linkage_array, t=thresh, criterion='distance')

        # Get sub-labels for just our cluster's points
        sub_labels = all_sub_labels[cluster_points]
        unique_subs = set(sub_labels)
        n_subs = len(unique_subs)

        print(f"\n  At {pct}th percentile (dist={thresh:.4f}): {n_subs} sub-groups")

        cut_result = {
            "percentile": pct,
            "distance": thresh,
            "n_sub_groups": n_subs,
            "sub_groups": []
        }

        if show_samples and n_subs > 1:
            # Show samples from each sub-group
            for sub_id in sorted(unique_subs):
                sub_mask = sub_labels == sub_id
                sub_points = cluster_points[sub_mask]
                sub_size = len(sub_points)

                print(f"\n    Sub-group {sub_id} (n={sub_size}):")

                cut_result["sub_groups"].append({
                    "sub_id": sub_id,
                    "size": sub_size,
                    "points": sub_points.tolist()
                })

                for pt_idx in sub_points[:n_samples_per_group]:
                    if pt_idx < len(point_mapping):
                        text = point_mapping[pt_idx]["display_text"]
                        text = text[:65] + "..." if len(text) > 65 else text
                        print(f"      - {text}")

        results["cuts"].append(cut_result)

    return results


def find_optimal_n_subclusters(
    cluster_id: int,
    labels: np.ndarray,
    linkage_array: np.ndarray,
    max_clusters: int = 10
) -> List[Tuple[int, float]]:
    """
    Find what the internal structure looks like at different granularities.

    Cuts the dendrogram to get 2, 3, 4, ... max_clusters sub-groups and
    reports the distance threshold for each.

    Args:
        cluster_id: The EOM cluster to analyze
        labels: HDBSCAN cluster labels
        linkage_array: Full single linkage tree
        max_clusters: Maximum number of sub-clusters to try

    Returns:
        List of (n_clusters, distance_threshold) tuples
    """
    from scipy.cluster.hierarchy import fcluster

    cluster_points = extract_cluster_points(cluster_id, labels)
    n_cluster = len(cluster_points)

    if n_cluster < 2:
        return []

    print(f"\n{'='*60}")
    print(f"OPTIMAL SUB-CLUSTER ANALYSIS: Cluster {cluster_id}")
    print(f"{'='*60}")
    print(f"Cluster size: {n_cluster} points")

    results = []
    max_k = min(max_clusters, n_cluster)

    # Get all unique distances in the tree
    distances = sorted(set(linkage_array[:, 2]))

    print(f"\n{'n_sub':>6} | {'distance':>10} | {'sizes'}")
    print("-" * 50)

    for target_k in range(2, max_k + 1):
        # Binary search for distance that gives target_k clusters for our points
        best_dist = None
        best_n = 0

        for dist in distances:
            all_labels = fcluster(linkage_array, t=dist, criterion='distance')
            sub_labels = all_labels[cluster_points]
            n_subs = len(set(sub_labels))

            if n_subs >= target_k:
                best_dist = dist
                best_n = n_subs
                break

        if best_dist is not None and best_n == target_k:
            all_labels = fcluster(linkage_array, t=best_dist, criterion='distance')
            sub_labels = all_labels[cluster_points]

            # Get sizes
            from collections import Counter
            sizes = sorted(Counter(sub_labels).values(), reverse=True)
            sizes_str = ", ".join(str(s) for s in sizes[:5])
            if len(sizes) > 5:
                sizes_str += f", ... ({len(sizes)} total)"

            print(f"{target_k:>6} | {best_dist:>10.4f} | {sizes_str}")
            results.append((target_k, best_dist))

    return results


def visualize_cluster_dendrogram(
    cluster_id: int,
    labels: np.ndarray,
    linkage_array: np.ndarray,
    point_mapping: List[Dict],
    truncate_mode: str = 'lastp',
    p: int = 30,
    save_path: Optional[Path] = None
):
    """
    Visualize the internal dendrogram of a specific cluster.

    Shows the hierarchical structure within the cluster that HDBSCAN
    condensed into a single leaf.

    Args:
        cluster_id: The EOM cluster to visualize
        labels: HDBSCAN cluster labels
        linkage_array: Full single linkage tree
        point_mapping: For labeling leaves with text snippets
        truncate_mode: 'lastp' to show last p merges, None for full
        p: Number of merges if truncate_mode='lastp'
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
        from scipy.spatial.distance import pdist
    except ImportError:
        print("matplotlib or scipy not available")
        return

    cluster_points = extract_cluster_points(cluster_id, labels)
    n_cluster = len(cluster_points)

    if n_cluster < 2:
        print(f"Cluster {cluster_id} has only {n_cluster} points")
        return

    print(f"\nGenerating dendrogram for Cluster {cluster_id} ({n_cluster} points)")

    # Get subtree
    subtree, point_map = get_cluster_subtree(cluster_id, labels, linkage_array)

    # Check if subtree is valid (should have n_cluster - 1 rows)
    subtree_valid = len(subtree) == n_cluster - 1

    if subtree_valid:
        use_subtree = subtree
        n_leaves = n_cluster

        # Create labels for the subtree leaves
        reverse_map = {v: k for k, v in point_map.items()}
        leaf_labels = []
        for i in range(n_cluster):
            if i in reverse_map:
                orig_id = reverse_map[i]
                if orig_id < len(point_mapping):
                    text = point_mapping[orig_id]["display_text"][:30]
                    leaf_labels.append(text)
                else:
                    leaf_labels.append(str(i))
            else:
                leaf_labels.append(str(i))

        print(f"Using extracted subtree: {len(subtree)} merges for {n_cluster} points")
    else:
        # Fallback: use full linkage tree, truncated view only
        print(f"Subtree incomplete ({len(subtree)} merges vs expected {n_cluster - 1})")
        print("Falling back to truncated view of full tree")
        use_subtree = linkage_array
        n_leaves = len(labels)
        leaf_labels = None
        truncate_mode = 'lastp'  # Force truncation for full tree

    fig, ax = plt.subplots(figsize=(14, 8))

    if truncate_mode == 'lastp':
        dend = dendrogram(
            use_subtree,
            ax=ax,
            truncate_mode='lastp',
            p=min(p, len(use_subtree)),
            leaf_rotation=90,
            leaf_font_size=8,
        )
        title = f"Cluster {cluster_id} Dendrogram (last {min(p, len(use_subtree))} merges)"
    else:
        dend = dendrogram(
            use_subtree,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=8,
            labels=leaf_labels,
        )
        title = f"Cluster {cluster_id} Internal Dendrogram ({n_cluster} points)"

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Points in cluster")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


def compare_condensed_vs_single_linkage(
    labels: np.ndarray,
    tree_df: pd.DataFrame,
    linkage_array: np.ndarray,
    n_samples: int
) -> None:
    """
    Compare what condensed_tree shows vs what single_linkage_tree reveals.

    This highlights the "hidden" structure in each cluster.
    """
    from scipy.cluster.hierarchy import fcluster

    print(f"\n{'='*70}")
    print("CONDENSED TREE vs SINGLE LINKAGE TREE COMPARISON")
    print(f"{'='*70}")

    # Get leaves from condensed tree
    condensed_leaves = identify_leaf_nodes(tree_df, n_samples)

    # For each cluster, show how many sub-groups exist at 50th percentile distance
    cluster_ids = sorted(set(labels) - {-1})

    print(f"\n{'Cluster':>8} | {'Size':>6} | {'Condensed':>10} | {'SL (50%)':>10} | {'SL (75%)':>10}")
    print("-" * 60)

    for cid in cluster_ids:
        cluster_points = extract_cluster_points(cid, labels)
        n_pts = len(cluster_points)

        # Condensed tree: how many leaves map to this cluster
        n_condensed = sum(1 for leaf in condensed_leaves
                         if leaf in tree_df['parent'].values)  # Approximate

        # Single linkage: sub-groups at different cuts
        if n_pts >= 2:
            # Get relevant distances
            all_dists = linkage_array[:, 2]
            d50 = np.percentile(all_dists, 50)
            d75 = np.percentile(all_dists, 75)

            labels_50 = fcluster(linkage_array, t=d50, criterion='distance')
            labels_75 = fcluster(linkage_array, t=d75, criterion='distance')

            n_sl_50 = len(set(labels_50[cluster_points]))
            n_sl_75 = len(set(labels_75[cluster_points]))
        else:
            n_sl_50 = n_sl_75 = 1

        print(f"{cid:>8} | {n_pts:>6} | {'1 leaf':>10} | {n_sl_50:>10} | {n_sl_75:>10}")


#%% GET SINGLE LINKAGE ARRAY
sl_array = get_single_linkage_array()


#%% ANALYZE INTERNAL STRUCTURE FOR EACH CLUSTER
if sl_array is not None:
    for cluster_id in sorted(set(labels) - {-1})[:3]:  # First 3 clusters
        analyze_cluster_internal_structure(
            cluster_id=cluster_id,
            labels=labels,
            linkage_array=sl_array,
            point_mapping=point_mapping,
            distance_percentiles=[50, 75, 90],
            show_samples=True,
            n_samples_per_group=3
        )


#%% FIND OPTIMAL SUB-CLUSTERS
if sl_array is not None:
    # Example: analyze first cluster
    first_cluster = min(set(labels) - {-1})
    find_optimal_n_subclusters(first_cluster, labels, sl_array, max_clusters=8)


#%% VISUALIZE CLUSTER DENDROGRAM
if sl_array is not None:
    first_cluster = min(set(labels) - {-1})
    visualize_cluster_dendrogram(
        cluster_id=first_cluster,
        labels=labels,
        linkage_array=sl_array,
        point_mapping=point_mapping,
        truncate_mode='lastp',
        p=30,
        save_path=project_root / "exports" / f"cluster_{first_cluster}_dendrogram.png"
    )


#%% COMPARE CONDENSED VS SINGLE LINKAGE
if sl_array is not None:
    compare_condensed_vs_single_linkage(labels, tree_df, sl_array, n_samples)


#%% ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*60}")
print("AVAILABLE FUNCTIONS")
print(f"{'='*60}")
print("""
Data:
  - tree_df: Condensed tree as DataFrame
  - point_mapping: List mapping point_id -> {text, cluster, probability}
  - labels, probabilities, persistence, outlier_scores
  - leaf_nodes, leaf_to_points, leaf_parent
  - sl_array: Single linkage tree as numpy array (full dendrogram)

Analysis (Condensed Tree):
  - analyze_tree_structure(tree_df, n_samples)
  - get_early_fallouts(tree_df, n_samples, lambda_threshold)
  - get_late_fallouts(tree_df, n_samples, lambda_threshold)
  - analyze_fallouts_by_cluster(tree_df, n_samples)
  - analyze_cluster_persistence()
  - correlate_lambda_and_probability(tree_df, n_samples)

Leaf Clusters (Condensed Tree):
  - identify_leaf_nodes(tree_df, n_samples)
  - map_leaves_to_points(tree_df, leaf_nodes)
  - map_leaves_to_parent_clusters(leaf_to_points, labels)
  - print_leaf_samples(point_mapping, leaf_to_points, leaf_parent, probs, n)
  - analyze_leaf_statistics(leaf_to_points, leaf_parent)

Hierarchy Analysis (Condensed Tree):
  - identify_intermediate_nodes(tree_df, n_samples)
  - analyze_hierarchy_depth(tree_df, n_samples)
  - show_cluster_hierarchy_path(tree_df, cluster_id, labels, n_samples)

Single Linkage Tree (Full Dendrogram - Finer Granularity):
  - get_single_linkage_array() -> numpy linkage matrix
  - extract_cluster_points(cluster_id, labels) -> point indices
  - get_cluster_subtree(cluster_id, labels, linkage_array) -> subtree
  - analyze_cluster_internal_structure(cluster_id, labels, linkage_array, ...)
  - find_optimal_n_subclusters(cluster_id, labels, linkage_array, max_clusters)
  - compare_condensed_vs_single_linkage(labels, tree_df, linkage_array, n_samples)

Visualization:
  - visualize_condensed_tree(save_path)
  - visualize_single_linkage_tree(truncate_mode, p, save_path)
  - visualize_cluster_tree(tree_df, n_samples, labels, save_path)
  - visualize_cluster_tree_with_siblings(tree_df, n_samples, labels)
  - visualize_cluster_dendrogram(cluster_id, labels, linkage_array, ...)  # NEW
""")
