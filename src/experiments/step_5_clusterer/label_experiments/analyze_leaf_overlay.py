#%% ============================================================================
# EOM vs LEAF OVERLAY ANALYSIS - LAYER 2
# ============================================================================
"""
Layer 2 clustering experiment:
- Loads cached UMAP embeddings + EOM results from Layer 1
- Runs HDBSCAN with LEAF method (configurable params)
- Overlays leaf clusters onto EOM parent clusters

This enables exploring micro-themes within each stable EOM cluster.

Usage: Open in VS Code and run cells interactively.
"""

#%% IMPORTS AND SETUP
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import hdbscan

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import generate_enhanced_variable_key
import models

# Import format_ontology_text for point_mapping (used for ontology-based keywords)
try:
    from experiments.step_5_clusterer.clusterer_helpers_exp import format_ontology_text
except ImportError:
    from step_5_clusterer.clusterer_helpers_exp import format_ontology_text

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


#%% LOAD CACHED DATA FROM LAYER 1
def load_layer1_cache():
    """Load UMAP embeddings, EOM results, and params from Layer 1."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem

    # Load UMAP embeddings + winning params
    umap_path = cache_dir / f"umap_embeddings_{base_name}_{variable_key}.pkl"
    if not umap_path.exists():
        raise FileNotFoundError(
            f"UMAP cache not found: {umap_path}\n"
            f"Run step_5_clusterer.run_experiment first to generate cache."
        )

    print(f"Loading UMAP cache from: {umap_path.name}")
    with open(umap_path, 'rb') as f:
        umap_cache = pickle.load(f)

    # Load HDBSCAN artifacts (EOM labels, probs)
    artifacts_path = cache_dir / f"hdbscan_artifacts_{base_name}_{variable_key}.pkl"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN artifacts not found: {artifacts_path}\n"
            f"Run step_5_clusterer.run_experiment first to generate cache."
        )

    print(f"Loading HDBSCAN artifacts from: {artifacts_path.name}")
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    # Load cluster models for text mapping
    clusters_path = cache_dir / f"006_initial_clusters_{base_name}_{variable_key}.pkl"
    with open(clusters_path, 'rb') as f:
        cluster_data = pickle.load(f)
    cluster_models = [models.ClusterModel.model_validate(item) for item in cluster_data]

    return {
        "umap_embeddings": umap_cache["embeddings"],
        "eom_params": umap_cache["params"],
        "eom_labels": artifacts["labels"],
        "eom_probs": artifacts["probabilities"],
        "cluster_models": cluster_models,
    }


data = load_layer1_cache()
print(f"\nLoaded UMAP embeddings: {data['umap_embeddings'].shape}")
print(f"EOM params: {data['eom_params']}")
print(f"EOM clusters: {len(set(data['eom_labels']) - {-1})}")
print(f"EOM noise points: {sum(data['eom_labels'] == -1)}")


#%% EXTRACT EOM LABELS
labels_eom = data["eom_labels"]
probs_eom = data["eom_probs"]
eom_mcs = data["eom_params"].get("min_cluster_size", 5)
eom_ms = data["eom_params"].get("min_samples", 3)

print(f"\nEOM params: mcs={eom_mcs}, ms={eom_ms}")


#%% BUILD POINT MAPPING
def load_template_prefix_from_cluster_models(cluster_models: List[models.ClusterModel]) -> Optional[str]:
    """Load template prefix from cluster models."""
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
    """Build list mapping point index to idea details.

    Includes multiple text representations for keyword extraction:
    - text: raw idea text (idea.idea)
    - display_text: idea text with template prefix stripped
    - ontology_text: formatted ontology (instance - node (category))
    - taxonomy_phrase: 2-4 word categorization phrase
    """
    points = []
    for model in cluster_models:
        if model.response_ideas:
            for idea in model.response_ideas:
                raw_text = idea.idea
                display_text = strip_template_prefix(raw_text, template_prefix)
                ontology_text = format_ontology_text(idea)
                taxonomy_phrase = getattr(idea, 'taxonomy_phrase', '') or ''
                points.append({
                    "text": raw_text,
                    "display_text": display_text,
                    "ontology_text": ontology_text,
                    "taxonomy_phrase": taxonomy_phrase,
                    "respondent_id": model.respondent_id,
                })
    return points


# Load template prefix from cluster models
template_prefix = load_template_prefix_from_cluster_models(data["cluster_models"])
if template_prefix:
    prefix_display = template_prefix[:50] + "..." if len(template_prefix) > 50 else template_prefix
    print(f"Template prefix: '{prefix_display}'")
else:
    print("Template prefix: (none found)")

point_mapping = build_point_mapping(data["cluster_models"], template_prefix)
print(f"Point mapping: {len(point_mapping)} points")


#%% MAP LEAF CLUSTERS TO EOM PARENTS
def map_leaf_to_eom(
    labels_leaf: np.ndarray,
    labels_eom: np.ndarray
) -> Dict[int, Dict]:
    """
    Map each leaf cluster to its corresponding EOM parent via point overlap.

    Returns dict: leaf_id -> {
        "eom_parent": dominant EOM cluster,
        "overlap_ratio": fraction of points in that EOM cluster,
        "size": number of points in leaf
    }
    """
    from collections import Counter

    leaf_to_eom = {}
    leaf_clusters = set(labels_leaf) - {-1}

    for leaf_id in leaf_clusters:
        leaf_points = np.where(labels_leaf == leaf_id)[0]
        eom_labels_for_leaf = labels_eom[leaf_points]
        eom_labels_non_noise = eom_labels_for_leaf[eom_labels_for_leaf != -1]

        if len(eom_labels_non_noise) == 0:
            leaf_to_eom[leaf_id] = {
                "eom_parent": -1,
                "overlap_ratio": 0.0,
                "size": len(leaf_points)
            }
        else:
            counter = Counter(eom_labels_non_noise)
            dominant = counter.most_common(1)[0]
            leaf_to_eom[leaf_id] = {
                "eom_parent": int(dominant[0]),
                "overlap_ratio": dominant[1] / len(leaf_points),
                "size": len(leaf_points)
            }

    return leaf_to_eom


#%% PRINT OVERLAY SUMMARY
def print_overlay_summary(leaf_to_eom: Dict[int, Dict], labels_eom: np.ndarray):
    """Print the overlay: EOM clusters with their leaf sub-clusters."""
    print(f"\n{'='*80}")
    print("EOM -> LEAF OVERLAY")
    print(f"{'='*80}")

    # Group leaves by EOM parent
    eom_to_leaves = {}
    for leaf_id, info in leaf_to_eom.items():
        eom_parent = info["eom_parent"]
        eom_to_leaves.setdefault(eom_parent, []).append((leaf_id, info))

    for eom_id in sorted(eom_to_leaves.keys()):
        if eom_id == -1:
            continue

        leaves = eom_to_leaves[eom_id]
        eom_size = sum(labels_eom == eom_id)
        total_leaf_pts = sum(info["size"] for _, info in leaves)

        print(f"\nEOM Cluster {eom_id} (n={eom_size})")
        print(f"  Contains {len(leaves)} leaf clusters ({total_leaf_pts} points)")
        print(f"  {'-'*60}")

        # Sort by size descending
        for leaf_id, info in sorted(leaves, key=lambda x: x[1]["size"], reverse=True):
            overlap = info["overlap_ratio"]
            size = info["size"]
            leakage_flag = " ⚠️ LEAKAGE" if overlap < 1.0 else ""
            print(f"    Leaf {leaf_id}: n={size:3d}, purity={overlap:.0%}{leakage_flag}")

    # Handle leaves without EOM parent (from noise points)
    orphan_leaves = [(l, i) for l, i in leaf_to_eom.items() if i["eom_parent"] == -1]
    if orphan_leaves:
        print(f"\n  [Orphan leaves - from EOM noise points: {len(orphan_leaves)}]")




#%% PRINT SAMPLES PER LEAF PER EOM CLUSTER
def print_samples_per_leaf(
    leaf_to_eom: Dict[int, Dict],
    labels_leaf: np.ndarray,
    point_mapping: List[Dict],
    probs_leaf: np.ndarray,
    n_samples: Optional[int] = 3
):
    """Print text samples organized by EOM cluster -> leaf clusters.

    Args:
        n_samples: Number of samples per leaf. Use None to show ALL points.
    """
    # Group by EOM parent
    eom_to_leaves = {}
    for leaf_id, info in leaf_to_eom.items():
        eom_parent = info["eom_parent"]
        eom_to_leaves.setdefault(eom_parent, []).append(leaf_id)

    for eom_id in sorted(eom_to_leaves.keys()):
        if eom_id == -1:
            continue

        print(f"\n{'='*80}")
        print(f"EOM CLUSTER {eom_id}")
        print(f"{'='*80}")

        for leaf_id in sorted(
            eom_to_leaves[eom_id],
            key=lambda l: leaf_to_eom[l]["size"],
            reverse=True
        ):
            leaf_points = np.where(labels_leaf == leaf_id)[0]
            info = leaf_to_eom[leaf_id]

            print(f"\n  Leaf {leaf_id} (n={info['size']}, overlap={info['overlap_ratio']:.0%})")
            print(f"  {'-'*70}")

            # Sort by probability, show samples (None = all)
            sorted_pts = sorted(
                leaf_points,
                key=lambda i: probs_leaf[i] if i < len(probs_leaf) else 0,
                reverse=True
            )
            pts_to_show = sorted_pts if n_samples is None else sorted_pts[:n_samples]
            for pt_idx in pts_to_show:
                if pt_idx < len(point_mapping):
                    text = point_mapping[pt_idx]["display_text"]
                    prob = probs_leaf[pt_idx] if pt_idx < len(probs_leaf) else 0
                    print(f"    [{prob:.2f}] {text}")




#%% VISUALIZE OVERLAY
def visualize_overlay(
    umap_embeddings: np.ndarray,
    labels_eom: np.ndarray,
    labels_leaf: np.ndarray,
    leaf_to_eom: Dict[int, Dict],
    save_path: Optional[Path] = None
):
    """Visualize EOM clusters with leaf boundaries overlaid."""
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError:
        print("matplotlib or umap not available")
        return

    # 2D UMAP for visualization
    print("\nComputing 2D UMAP for visualization...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42, min_dist=0.0)
    coords_2d = reducer_2d.fit_transform(umap_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: EOM clusters
    ax = axes[0]
    for cluster_id in sorted(set(labels_eom) - {-1}):
        mask = labels_eom == cluster_id
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            label=f"EOM {cluster_id}",
            alpha=0.6,
            s=20
        )
    noise_mask = labels_eom == -1
    ax.scatter(
        coords_2d[noise_mask, 0],
        coords_2d[noise_mask, 1],
        c='gray',
        alpha=0.3,
        s=10,
        label='Noise'
    )
    ax.set_title(f"EOM Clusters ({len(set(labels_eom) - {-1})} clusters)")
    ax.legend(loc='best', fontsize=8)

    # Right: Leaf clusters (colored by EOM parent)
    ax = axes[1]
    eom_colors = plt.cm.tab10.colors
    for leaf_id, info in leaf_to_eom.items():
        eom_parent = info["eom_parent"]
        if eom_parent == -1:
            color = 'gray'
        else:
            color = eom_colors[eom_parent % len(eom_colors)]
        mask = labels_leaf == leaf_id
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[color],
            alpha=0.6,
            s=20,
            marker='.'
        )

    noise_mask = labels_leaf == -1
    ax.scatter(
        coords_2d[noise_mask, 0],
        coords_2d[noise_mask, 1],
        c='gray',
        alpha=0.2,
        s=5
    )
    ax.set_title(f"Leaf Clusters ({len(leaf_to_eom)} clusters, colored by EOM parent)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()




#%% SUMMARY STATISTICS
def print_comparison_stats(
    labels_eom: np.ndarray,
    labels_leaf: np.ndarray,
    leaf_to_eom: Dict
):
    """Print comparison statistics between EOM and Leaf methods."""
    print(f"\n{'='*60}")
    print("COMPARISON STATISTICS")
    print(f"{'='*60}")

    n_eom = len(set(labels_eom) - {-1})
    n_leaf = len(set(labels_leaf) - {-1})
    noise_eom = sum(labels_eom == -1)
    noise_leaf = sum(labels_leaf == -1)

    print(f"\nEOM method:  {n_eom:3d} clusters, {noise_eom:4d} noise points "
          f"({noise_eom/len(labels_eom)*100:.1f}%)")
    print(f"Leaf method: {n_leaf:3d} clusters, {noise_leaf:4d} noise points "
          f"({noise_leaf/len(labels_leaf)*100:.1f}%)")

    if n_eom > 0:
        print(f"Ratio: {n_leaf/n_eom:.1f}x more leaf clusters than EOM")

    # Average leaves per EOM cluster
    eom_to_leaves = {}
    for leaf_id, info in leaf_to_eom.items():
        eom_parent = info["eom_parent"]
        if eom_parent != -1:
            eom_to_leaves.setdefault(eom_parent, []).append(leaf_id)

    if eom_to_leaves:
        avg_leaves = np.mean([len(v) for v in eom_to_leaves.values()])
        print(f"\nAverage leaves per EOM cluster: {avg_leaves:.1f}")

        # Distribution of leaves per EOM
        leaf_counts = [len(v) for v in eom_to_leaves.values()]
        print(f"Min/Max leaves per EOM: {min(leaf_counts)} / {max(leaf_counts)}")

        # EOM clusters with most sub-structure
        sorted_eom = sorted(eom_to_leaves.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\nEOM clusters with most sub-structure:")
        for eom_id, leaves in sorted_eom[:5]:
            print(f"  EOM {eom_id}: {len(leaves)} leaves")




#%% LEAKAGE ANALYSIS
def analyze_leakage(
    leaf_to_eom: Dict[int, Dict],
    labels_leaf: np.ndarray,
    labels_eom: np.ndarray,
    point_mapping: List[Dict],
    verbose: bool = True
) -> Dict:
    """
    Analyze leakage: when LEAF clusters contain points from multiple EOM clusters.

    Leakage occurs when overlap_ratio < 1.0, meaning some points in the leaf
    cluster belong to different EOM clusters than the dominant one.

    Returns dict with leakage statistics and details.
    """
    from collections import Counter

    # Collect leakage stats
    total_leaves = len(leaf_to_eom)
    leaves_with_leakage = []
    pure_leaves = []
    total_leaked_points = 0
    leakage_details = []

    for leaf_id, info in leaf_to_eom.items():
        overlap = info["overlap_ratio"]
        size = info["size"]
        eom_parent = info["eom_parent"]

        if overlap < 1.0 and eom_parent != -1:
            leaves_with_leakage.append(leaf_id)
            n_leaked = int(size * (1 - overlap))
            total_leaked_points += n_leaked

            # Get detailed breakdown of which EOM clusters contributed
            leaf_points = np.where(labels_leaf == leaf_id)[0]
            eom_labels_for_leaf = labels_eom[leaf_points]
            eom_counter = Counter(eom_labels_for_leaf[eom_labels_for_leaf != -1])

            # Build breakdown: which EOM clusters and how many points
            breakdown = []
            for eom_id, count in eom_counter.most_common():
                pct = count / len(leaf_points) * 100
                breakdown.append({
                    "eom_id": eom_id,
                    "count": count,
                    "pct": pct,
                    "is_dominant": eom_id == eom_parent
                })

            leakage_details.append({
                "leaf_id": leaf_id,
                "size": size,
                "eom_parent": eom_parent,
                "overlap_ratio": overlap,
                "n_leaked": n_leaked,
                "breakdown": breakdown
            })
        else:
            pure_leaves.append(leaf_id)

    # Calculate summary stats
    n_leaky = len(leaves_with_leakage)
    leakage_rate = n_leaky / total_leaves * 100 if total_leaves > 0 else 0
    total_leaf_points = sum(info["size"] for info in leaf_to_eom.values())
    leaked_pct = total_leaked_points / total_leaf_points * 100 if total_leaf_points > 0 else 0

    stats = {
        "total_leaves": total_leaves,
        "pure_leaves": len(pure_leaves),
        "leaves_with_leakage": n_leaky,
        "leakage_rate": leakage_rate,
        "total_leaf_points": total_leaf_points,
        "total_leaked_points": total_leaked_points,
        "leaked_point_pct": leaked_pct,
        "details": sorted(leakage_details, key=lambda x: x["n_leaked"], reverse=True)
    }

    if verbose:
        print(f"\n{'='*60}")
        print("LEAKAGE ANALYSIS")
        print(f"{'='*60}")

        print(f"\nSummary:")
        print(f"  Pure leaves (100% overlap):    {len(pure_leaves):3d} / {total_leaves} "
              f"({len(pure_leaves)/total_leaves*100:.1f}%)")
        print(f"  Leaves with leakage (<100%):   {n_leaky:3d} / {total_leaves} "
              f"({leakage_rate:.1f}%)")
        print(f"  Total leaked points:           {total_leaked_points:3d} / {total_leaf_points} "
              f"({leaked_pct:.1f}%)")

        if leakage_details:
            print(f"\nLeaky Leaves (sorted by leaked points):")
            print(f"  {'Leaf':>6} | {'Size':>5} | {'Overlap':>8} | {'Leaked':>7} | {'Breakdown'}")
            print(f"  {'-'*70}")

            for detail in stats["details"][:10]:  # Top 10 leaky leaves
                breakdown_str = " + ".join(
                    f"EOM{b['eom_id']}:{b['count']}" + ("*" if b["is_dominant"] else "")
                    for b in detail["breakdown"]
                )
                print(
                    f"  {detail['leaf_id']:>6} | "
                    f"{detail['size']:>5} | "
                    f"{detail['overlap_ratio']:>7.0%} | "
                    f"{detail['n_leaked']:>7} | "
                    f"{breakdown_str}"
                )

            if len(stats["details"]) > 10:
                print(f"  ... and {len(stats['details']) - 10} more leaky leaves")

            print(f"\n  Legend: * = dominant EOM cluster")
        else:
            print(f"\n  All leaves are pure (no leakage detected)")

    return stats




#%% ============================================================================
# LAYER 2 PARETO GRID SEARCH
# ============================================================================
"""
Find optimal LEAF overlay granularity using Pareto optimization.

Objectives (all maximize):
1. Purity: Fraction of leaves with 100% EOM overlap (no leakage)
2. Coverage: Fraction of points assigned to leaves (not noise)
3. Leaves/EOM score: Target 2-5 leaves per EOM, penalize outside range

Parameter space:
- min_cluster_size: 2 to EOM_MCS
- min_samples: 1 to MCS-1

This replaces the arbitrary multiplier-based approach with principled optimization.
"""

#%% PARETO SEARCH CONFIG
PARETO_MCS_GRID = [2, 3, 5, 7, 10, 15, 20]  # Will be clipped to EOM_MCS
PARETO_MS_GRID = [1, 2, 3, 5]                # Will be clipped to MCS-1
PARETO_LEAVES_PER_EOM_TARGET = (2, 5)        # Target range for scoring

# Threshold filters (exclude candidates before Pareto frontier)
PARETO_MAX_CLUSTERS_MULTIPLIER = 3  # Max clusters = multiplier × n_eom_clusters
PARETO_MAX_NOISE_PCT = 0.20         # Max 20% noise points


#%% PARETO OBJECTIVE FUNCTIONS
def calc_purity(leaf_to_eom: Dict) -> float:
    """Fraction of leaves with 100% EOM overlap (no leakage)."""
    pure = sum(1 for info in leaf_to_eom.values()
               if info["overlap_ratio"] == 1.0 and info["eom_parent"] != -1)
    total = sum(1 for info in leaf_to_eom.values() if info["eom_parent"] != -1)
    return pure / total if total > 0 else 0.0


def calc_coverage(labels: np.ndarray) -> float:
    """Fraction of points assigned to clusters (not noise)."""
    return 1.0 - (sum(labels == -1) / len(labels))


def calc_leaves_per_eom_score(
    leaf_to_eom: Dict,
    labels_eom: np.ndarray,
    target_range: Tuple[int, int] = (2, 5)
) -> float:
    """Score based on leaves per EOM being in target range.

    Returns 1.0 if in target range, penalizes distance from range.
    """
    # Group leaves by EOM parent
    eom_to_leaves = {}
    for leaf_id, info in leaf_to_eom.items():
        if info["eom_parent"] != -1:
            eom_to_leaves.setdefault(info["eom_parent"], []).append(leaf_id)

    if not eom_to_leaves:
        return 0.0

    # Score: 1.0 if in target range, penalize distance from range
    scores = []
    lo, hi = target_range
    for eom_id, leaves in eom_to_leaves.items():
        n = len(leaves)
        if lo <= n <= hi:
            scores.append(1.0)
        elif n < lo:
            scores.append(max(0, 1.0 - (lo - n) * 0.2))  # Penalize too few
        else:
            scores.append(max(0, 1.0 - (n - hi) * 0.1))  # Penalize too many

    return np.mean(scores)


#%% PARETO GRID SEARCH FUNCTION
def run_layer2_grid_search(
    umap_embeddings: np.ndarray,
    labels_eom: np.ndarray,
    mcs_grid: List[int],
    ms_grid: List[int],
    target_range: Tuple[int, int] = (2, 5)
) -> List[Dict]:
    """Run LEAF clustering at all (MCS, MS) combinations.

    Args:
        umap_embeddings: UMAP-reduced embeddings from Layer 1
        labels_eom: EOM cluster labels from Layer 1
        mcs_grid: Grid of min_cluster_size values to try
        ms_grid: Grid of min_samples values to try
        target_range: Target range for leaves per EOM

    Returns:
        List of result dicts with params, objectives, and clustering results
    """
    results = []

    for mcs in mcs_grid:
        for ms in ms_grid:
            if ms >= mcs:
                continue  # Invalid: min_samples must be < min_cluster_size

            # Run LEAF clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_method="leaf",
                prediction_data=True,
            )
            labels = clusterer.fit_predict(umap_embeddings)
            probs = clusterer.probabilities_

            # Map to EOM
            leaf_to_eom_map = map_leaf_to_eom(labels, labels_eom)

            # Calculate objectives
            purity = calc_purity(leaf_to_eom_map)
            coverage = calc_coverage(labels)
            leaves_score = calc_leaves_per_eom_score(leaf_to_eom_map, labels_eom, target_range)

            # Calculate additional stats
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = sum(labels == -1)

            # Average leaves per EOM
            eom_to_leaves = {}
            for leaf_id, info in leaf_to_eom_map.items():
                if info["eom_parent"] != -1:
                    eom_to_leaves.setdefault(info["eom_parent"], []).append(leaf_id)
            avg_leaves = np.mean([len(v) for v in eom_to_leaves.values()]) if eom_to_leaves else 0

            results.append({
                "mcs": mcs,
                "ms": ms,
                "purity": purity,
                "coverage": coverage,
                "leaves_score": leaves_score,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": n_noise / len(labels) * 100,
                "avg_leaves_per_eom": avg_leaves,
                "labels": labels,
                "probs": probs,
                "leaf_to_eom": leaf_to_eom_map,
            })

    return results


#%% PARETO FRONTIER FUNCTION
def find_pareto_frontier(
    results: List[Dict],
    objectives: List[str] = ["purity", "coverage", "leaves_score"]
) -> List[Dict]:
    """Find non-dominated solutions on the Pareto frontier.

    A solution is non-dominated if no other solution is better in all objectives.

    Args:
        results: List of result dicts from grid search
        objectives: List of objective names to optimize (all maximize)

    Returns:
        List of non-dominated solutions
    """
    frontier = []

    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            # Check if 'other' dominates 'candidate'
            # Dominated if: all objectives >= AND at least one >
            all_ge = all(other[obj] >= candidate[obj] for obj in objectives)
            any_gt = any(other[obj] > candidate[obj] for obj in objectives)
            if all_ge and any_gt:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return frontier


#%% THRESHOLD FILTERING
def filter_grid_results(
    results: List[Dict],
    max_clusters: int,
    max_noise_pct: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter grid search results by thresholds.

    Args:
        results: All grid search results
        max_clusters: Maximum allowed clusters
        max_noise_pct: Maximum allowed noise percentage (0-1)

    Returns:
        Tuple of (passed_results, filtered_results)
    """
    passed = []
    filtered = []

    for r in results:
        # Check thresholds
        clusters_ok = r["n_clusters"] <= max_clusters
        noise_ok = (r["noise_pct"] / 100) <= max_noise_pct  # noise_pct is 0-100

        if clusters_ok and noise_ok:
            r["filter_status"] = "passed"
            passed.append(r)
        else:
            # Mark reason for filtering
            reasons = []
            if not clusters_ok:
                reasons.append(f"clusters>{max_clusters}")
            if not noise_ok:
                reasons.append(f"noise>{max_noise_pct:.0%}")
            r["filter_status"] = ", ".join(reasons)
            filtered.append(r)

    return passed, filtered


#%% PARETO DISPLAY & SELECTION FUNCTIONS
def print_pareto_results(
    results: List[Dict],
    frontier: List[Dict],
    filtered_results: List[Dict] = None,
    max_clusters: int = None,
    max_noise_pct: float = None
):
    """Print grid search results highlighting Pareto frontier and filtered candidates."""
    print(f"\n{'='*90}")
    print("LAYER 2 PARETO GRID SEARCH RESULTS")
    print(f"{'='*90}")

    # Show threshold info if provided
    if max_clusters is not None or max_noise_pct is not None:
        print(f"\nThresholds: max_clusters={max_clusters}, max_noise={max_noise_pct:.0%}")
        if filtered_results:
            print(f"Filtered out: {len(filtered_results)} candidates (shown with ✗)")

    print(f"\n{'MCS':>4} | {'MS':>3} | {'Purity':>7} | {'Coverage':>8} | {'L/EOM':>6} | "
          f"{'Clusters':>8} | {'Noise%':>7} | {'Status'}")
    print("-" * 90)

    # Combine passed and filtered for display
    all_results = results + (filtered_results or [])

    for r in sorted(all_results, key=lambda x: (-x["purity"], -x["coverage"])):
        if r.get("filter_status") == "passed" or "filter_status" not in r:
            status = "★" if r in frontier else ""
        else:
            status = f"✗ {r['filter_status']}"

        print(
            f"{r['mcs']:>4} | {r['ms']:>3} | {r['purity']:>6.1%} | "
            f"{r['coverage']:>7.1%} | {r['leaves_score']:>6.2f} | "
            f"{r['n_clusters']:>8} | {r['noise_pct']:>6.1f}% | {status}"
        )

    print(f"\nPareto frontier: {len(frontier)} solutions (from {len(results)} candidates)")


def select_balanced_solution(frontier: List[Dict]) -> Dict:
    """Select solution with best balanced score (geometric mean of objectives)."""
    for sol in frontier:
        sol["balanced_score"] = (sol["purity"] * sol["coverage"] * sol["leaves_score"]) ** (1/3)

    return max(frontier, key=lambda x: x["balanced_score"])


#%% RUN PARETO SEARCH
print("\n" + "="*90)
print("Running Layer 2 Pareto grid search...")
print("="*90)

# Calculate dynamic threshold for max clusters
n_eom_clusters = len(set(labels_eom) - {-1})
max_clusters_threshold = PARETO_MAX_CLUSTERS_MULTIPLIER * n_eom_clusters
print(f"EOM clusters: {n_eom_clusters} → max_clusters threshold: {max_clusters_threshold}")
print(f"Max noise threshold: {PARETO_MAX_NOISE_PCT:.0%}")

pareto_grid_results = run_layer2_grid_search(
    data["umap_embeddings"],
    labels_eom,
    PARETO_MCS_GRID,
    PARETO_MS_GRID,
    PARETO_LEAVES_PER_EOM_TARGET
)

# Apply threshold filtering before Pareto frontier
pareto_passed, pareto_filtered = filter_grid_results(
    pareto_grid_results,
    max_clusters=max_clusters_threshold,
    max_noise_pct=PARETO_MAX_NOISE_PCT
)

print(f"Grid search: {len(pareto_grid_results)} total, {len(pareto_passed)} passed filters, "
      f"{len(pareto_filtered)} filtered out")

# Find Pareto frontier from passed candidates only
pareto_frontier = find_pareto_frontier(pareto_passed)
print_pareto_results(
    pareto_passed,
    pareto_frontier,
    filtered_results=pareto_filtered,
    max_clusters=max_clusters_threshold,
    max_noise_pct=PARETO_MAX_NOISE_PCT
)

# Select balanced solution (with guard for empty frontier)
if not pareto_frontier:
    print(f"\n{'='*60}")
    print("WARNING: No candidates passed threshold filters!")
    print("Consider relaxing thresholds (max_clusters or max_noise).")
    print(f"{'='*60}")
    # Fallback: use best candidate from all results (ignoring filters)
    print("Falling back to best candidate from unfiltered results...")
    pareto_frontier_fallback = find_pareto_frontier(pareto_grid_results)
    pareto_selected = select_balanced_solution(pareto_frontier_fallback)
else:
    pareto_selected = select_balanced_solution(pareto_frontier)

print(f"\n{'='*60}")
print("SELECTED SOLUTION (balanced score)")
print(f"{'='*60}")
print(f"Parameters: MCS={pareto_selected['mcs']}, MS={pareto_selected['ms']}")
print(f"  Purity:           {pareto_selected['purity']:.1%}")
print(f"  Coverage:         {pareto_selected['coverage']:.1%}")
print(f"  Leaves/EOM score: {pareto_selected['leaves_score']:.2f}")
print(f"  Clusters:         {pareto_selected['n_clusters']}")
print(f"  Avg leaves/EOM:   {pareto_selected['avg_leaves_per_eom']:.1f}")
print(f"  Balanced score:   {pareto_selected['balanced_score']:.3f}")

# Update labels_leaf and leaf_to_eom with Pareto-selected solution
labels_leaf_pareto = pareto_selected["labels"]
probs_leaf_pareto = pareto_selected["probs"]
leaf_to_eom_pareto = pareto_selected["leaf_to_eom"]

print(f"\nPareto-optimized labels available as: labels_leaf_pareto, leaf_to_eom_pareto")


#%% ============================================================================
# DESCRIPTIVE ANALYSES (using Pareto-optimized overlay)
# ============================================================================

#%% OVERLAY SUMMARY
print_overlay_summary(leaf_to_eom_pareto, labels_eom)

#%% COMPARISON STATISTICS
print_comparison_stats(labels_eom, labels_leaf_pareto, leaf_to_eom_pareto)

#%% LEAKAGE ANALYSIS
leakage_stats = analyze_leakage(
    leaf_to_eom_pareto, labels_leaf_pareto, labels_eom, point_mapping, verbose=True
)

#%% SAMPLES PER LEAF (all ideas per EOM -> leaf, using Pareto winner)
print_samples_per_leaf(leaf_to_eom_pareto, labels_leaf_pareto, point_mapping, probs_leaf_pareto, n_samples=None)  # None = show ALL

#%% VISUALIZE OVERLAY
visualize_overlay(
    data["umap_embeddings"],
    labels_eom,
    labels_leaf_pareto,
    leaf_to_eom_pareto,
    save_path=project_root / "exports" / "eom_leaf_overlay_pareto.png"
)


#%% ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print("AVAILABLE DATA & HOW TO EXPERIMENT")
print(f"{'='*60}")
print("""
Cached from Layer 1:
  - data["umap_embeddings"]: UMAP-reduced embeddings
  - data["eom_params"]: Winning HDBSCAN params from optimization
  - data["eom_labels"]: EOM cluster labels (labels_eom)
  - data["eom_probs"]: EOM membership probabilities (probs_eom)

Computed via Pareto Grid Search (Layer 2):
  - labels_leaf_pareto: Pareto-optimized leaf cluster labels
  - probs_leaf_pareto: Leaf membership probabilities
  - leaf_to_eom_pareto: Mapping of leaf -> EOM parent
  - point_mapping: Text samples per point

Pareto Objectives:
  - Purity: Fraction of leaves with 100% EOM overlap
  - Coverage: Fraction of points assigned to leaves (not noise)
  - Leaves/EOM score: Target 2-5 leaves per EOM

Configuration:
  - PARETO_MCS_GRID: min_cluster_size values to try
  - PARETO_MS_GRID: min_samples values to try
  - PARETO_LEAVES_PER_EOM_TARGET: Target range for leaves per EOM

Downstream Experiments (using Pareto labels):
  - Sub-clustering: HDBSCAN within leaves + c-TF-IDF/MMR keywords
  - Lexical grouping: c-TF-IDF on leaves -> HDBSCAN -> TF-IDF keywords
""")


#%% ============================================================================
# EXPERIMENT: SUB-CLUSTERING WITHIN LEAVES + KEYWORD EXTRACTION
# ============================================================================
"""
Layer 3: Sub-clustering within each leaf cluster to discover micro-themes.

For each leaf:
1. Run HDBSCAN with allow_single_cluster=True (allows 1+ clusters)
2. Extract keywords using c-TF-IDF + MMR diversity
3. Display formatted keywords for each sub-cluster

Example output:
  Leaf 5 (n=42) → 3 sub-clusters
    Subcluster 0 (n=18) → smaken, pittig, kruiden, smaakvoller
    Subcluster 1 (n=15) → assortiment, menu, keuze, variatie
    Subcluster 2 (n=9)  → seizoenen, keukens, afwisseling
"""

# Import representation classes and spaCy helpers
try:
    from experiments.step_5_clusterer.representation.ctfidf_representation import CTfidfRepresentation
    from experiments.step_5_clusterer.representation.mmr_representation import MMRRepresentation
    from experiments.step_5_clusterer.representation.tfidf_representation import TfidfRepresentation
    from experiments.step_5_clusterer.clusterer_helpers_exp import (
        get_spacy_nlp,
        extract_noun_phrases_lemmatized
    )
except ImportError:
    exp_root = Path(__file__).parent.parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from step_5_clusterer.representation.ctfidf_representation import CTfidfRepresentation
    from step_5_clusterer.representation.mmr_representation import MMRRepresentation
    from step_5_clusterer.representation.tfidf_representation import TfidfRepresentation
    from step_5_clusterer.clusterer_helpers_exp import (
        get_spacy_nlp,
        extract_noun_phrases_lemmatized
    )


#%% SUB-CLUSTERING CONFIG
SUBCLUSTER_MIN_CLUSTER_SIZE = 3  # Minimum points for a sub-cluster
SUBCLUSTER_MIN_SAMPLES = 2       # Minimum samples for core points
KEYWORD_TOP_K = 5                # Top keywords per sub-cluster
MMR_DIVERSITY = 0.3              # MMR diversity (0=max diversity, 1=max relevance)

# Keyword extraction config
USE_LEMMATIZATION = True           # Enable spaCy lemmatization (ADJ + NOUN only)
SPACY_MODEL = "nl_core_news_lg"    # Dutch language model
NGRAM_RANGE = (1, 1)               # Unigrams only (no bigrams)
KEYWORD_TEXT_SOURCE = "taxonomy"   # Text source for keywords: "ontology", "idea", "display_text", or "taxonomy"


#%% SUB-CLUSTERING FUNCTIONS
def subcluster_leaf(
    leaf_points: np.ndarray,
    umap_embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run HDBSCAN sub-clustering on a single leaf's points.

    Args:
        leaf_points: Array of point indices belonging to this leaf
        umap_embeddings: Full UMAP embedding matrix
        min_cluster_size: Minimum sub-cluster size
        min_samples: Minimum samples for core points

    Returns:
        Tuple of (sub_labels, sub_probs) indexed by leaf_points
    """
    if len(leaf_points) < min_cluster_size:
        # Too few points for meaningful sub-clustering
        return np.zeros(len(leaf_points), dtype=int), np.ones(len(leaf_points))

    # Extract embeddings for this leaf
    leaf_embeddings = umap_embeddings[leaf_points]

    # Run HDBSCAN with allow_single_cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",  # Use EOM for sub-clusters
        allow_single_cluster=True,       # Key: allows finding just 1 cluster
        prediction_data=True,
    )
    sub_labels = clusterer.fit_predict(leaf_embeddings)
    sub_probs = clusterer.probabilities_

    return sub_labels, sub_probs


def extract_subcluster_keywords(
    sub_labels: np.ndarray,
    leaf_points: np.ndarray,
    point_mapping: List[Dict],
    top_k: int = 5,
    mmr_diversity: float = 0.3,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg",
    ngram_range: Tuple[int, int] = (1, 1),
    text_source: str = "ontology"
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extract keywords for each sub-cluster using c-TF-IDF + MMR.

    Uses spaCy lemmatization (ADJ + NOUN only) and falls back to vanilla TF-IDF
    when only 1 sub-cluster exists (c-TF-IDF needs 2+ classes for comparison).

    Args:
        sub_labels: Sub-cluster labels (indexed by position in leaf_points)
        leaf_points: Array of global point indices
        point_mapping: Global point-to-text mapping
        top_k: Number of keywords to extract
        mmr_diversity: MMR diversity parameter (0-1)
        use_lemmatization: Apply spaCy lemmatization with POS filter
        spacy_model: spaCy model name for lemmatization
        ngram_range: N-gram range for keywords (default: unigrams only)
        text_source: Which text field to use: "ontology", "idea", or "display_text"

    Returns:
        Dict mapping sub_cluster_id to list of (keyword, score) tuples
    """
    # Map text_source to point_mapping key
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(text_source, "ontology_text")

    # Build cluster_texts dict: {sub_cluster_id: [text1, text2, ...]}
    cluster_texts = {}
    unique_subclusters = set(sub_labels) - {-1}  # Exclude noise

    for sub_id in unique_subclusters:
        sub_mask = sub_labels == sub_id
        sub_point_indices = leaf_points[sub_mask]

        texts = []
        for pt_idx in sub_point_indices:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:  # Only add non-empty texts
                    texts.append(text)

        if texts:
            cluster_texts[sub_id] = texts

    if not cluster_texts:
        return {}

    # Step 1: Apply spaCy lemmatization (ADJ + NOUN only)
    if use_lemmatization:
        all_texts = []
        cluster_boundaries = []
        for cid in sorted(cluster_texts.keys()):
            cluster_boundaries.append((cid, len(all_texts), len(cluster_texts[cid])))
            all_texts.extend(cluster_texts[cid])

        lemmatized = extract_noun_phrases_lemmatized(all_texts, model_name=spacy_model)

        # Redistribute lemmatized texts back to clusters
        for cid, start, count in cluster_boundaries:
            cluster_texts[cid] = lemmatized[start:start+count]

    # Step 2: FALLBACK - Use vanilla TF-IDF if only 1 sub-cluster
    # (c-TF-IDF needs 2+ classes for cross-cluster comparison)
    if len(cluster_texts) == 1:
        tfidf = TfidfRepresentation(top_k=top_k, ngram_range=ngram_range)
        return tfidf.extract_keywords(cluster_texts, verbose=False)

    # Step 3: Extract c-TF-IDF scores (for 2+ clusters)
    ctfidf = CTfidfRepresentation(
        top_k=top_k * 3,  # Get more candidates for MMR selection
        bm25_weighting=True,
        reduce_frequent_words=True,
        ngram_range=ngram_range,  # Unigrams only by default
        min_df=1,
        max_df=0.95
    )
    ctfidf_keywords = ctfidf.extract_keywords(cluster_texts, verbose=False)

    # Step 4: Apply MMR for diversity
    mmr = MMRRepresentation(
        diversity=mmr_diversity,
        top_k=top_k,
        candidate_multiplier=3
    )

    mmr_keywords = {}
    for sub_id, texts in cluster_texts.items():
        if sub_id not in ctfidf_keywords or not ctfidf_keywords[sub_id]:
            continue

        # Get c-TF-IDF scores and vocabulary for this cluster
        kw_list = ctfidf_keywords[sub_id]
        vocabulary = [kw for kw, _ in kw_list]
        scores = np.array([score for _, score in kw_list])

        # Apply MMR selection
        keywords = mmr.extract_topics(
            cluster_id=sub_id,
            ctfidf_scores=scores,
            vocabulary=vocabulary,
            cluster_texts=texts
        )
        mmr_keywords[sub_id] = keywords

    return mmr_keywords


def analyze_leaf_subclusters(
    leaf_id: int,
    labels_leaf: np.ndarray,
    umap_embeddings: np.ndarray,
    point_mapping: List[Dict],
    min_cluster_size: int = 3,
    min_samples: int = 2,
    top_k: int = 5,
    mmr_diversity: float = 0.3,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg",
    ngram_range: Tuple[int, int] = (1, 1),
    text_source: str = "ontology"
) -> Dict:
    """
    Analyze sub-clustering structure within a single leaf.

    Returns:
        Dict with sub-clustering results and keywords
    """
    # Get points belonging to this leaf
    leaf_points = np.where(labels_leaf == leaf_id)[0]

    if len(leaf_points) < min_cluster_size:
        return {
            "leaf_id": leaf_id,
            "n_points": len(leaf_points),
            "n_subclusters": 0,
            "subclusters": {},
            "keywords": {},
            "message": "Too few points for sub-clustering"
        }

    # Run sub-clustering
    sub_labels, sub_probs = subcluster_leaf(
        leaf_points, umap_embeddings, min_cluster_size, min_samples
    )

    # Count sub-clusters
    unique_subs = set(sub_labels) - {-1}
    n_subclusters = len(unique_subs)
    n_noise = sum(sub_labels == -1)

    # Build subcluster info
    subclusters = {}
    for sub_id in unique_subs:
        sub_mask = sub_labels == sub_id
        sub_point_indices = leaf_points[sub_mask]
        sub_probabilities = sub_probs[sub_mask]

        subclusters[sub_id] = {
            "n_points": int(sum(sub_mask)),
            "point_indices": sub_point_indices,
            "probabilities": sub_probabilities,
            "texts": [
                point_mapping[i]["display_text"]
                for i in sub_point_indices if i < len(point_mapping)
            ]
        }

    # Extract keywords
    keywords = extract_subcluster_keywords(
        sub_labels, leaf_points, point_mapping, top_k, mmr_diversity,
        use_lemmatization=use_lemmatization,
        spacy_model=spacy_model,
        ngram_range=ngram_range,
        text_source=text_source
    )

    return {
        "leaf_id": leaf_id,
        "n_points": len(leaf_points),
        "n_subclusters": n_subclusters,
        "n_noise": n_noise,
        "sub_labels": sub_labels,
        "sub_probs": sub_probs,
        "subclusters": subclusters,
        "keywords": keywords
    }


#%% PRINT SUBCLUSTER ANALYSIS
def print_leaf_subclusters(
    result: Dict,
    show_samples: bool = True,
    n_samples: int = 3
):
    """
    Print formatted sub-cluster analysis for a leaf.
    """
    leaf_id = result["leaf_id"]
    n_points = result["n_points"]
    n_subclusters = result["n_subclusters"]

    if n_subclusters == 0:
        print(f"\n  Leaf {leaf_id} (n={n_points}) → {result.get('message', 'no sub-clusters')}")
        return

    noise_str = f", noise={result['n_noise']}" if result.get('n_noise', 0) > 0 else ""
    print(f"\n  Leaf {leaf_id} (n={n_points}) → {n_subclusters} sub-cluster(s){noise_str}")

    for sub_id in sorted(result["subclusters"].keys()):
        sub_info = result["subclusters"][sub_id]
        keywords = result["keywords"].get(sub_id, [])

        # Format keywords as comma-separated
        kw_str = ", ".join([kw for kw, _ in keywords[:5]]) if keywords else "(no keywords)"

        print(f"    Subcluster {sub_id} (n={sub_info['n_points']:3d}) → {kw_str}")

        if show_samples and sub_info["texts"]:
            # Show top samples by probability
            sorted_texts = sorted(
                zip(sub_info["texts"], sub_info["probabilities"]),
                key=lambda x: x[1],
                reverse=True
            )
            for text, prob in sorted_texts[:n_samples]:
                text_display = text[:80] + "..." if len(text) > 80 else text
                print(f"      [{prob:.2f}] {text_display}")


#%% RUN SUB-CLUSTERING ON ALL LEAVES
def run_subcluster_analysis(
    leaf_to_eom: Dict[int, Dict],
    labels_leaf: np.ndarray,
    labels_eom: np.ndarray,
    umap_embeddings: np.ndarray,
    point_mapping: List[Dict],
    min_cluster_size: int = 3,
    min_samples: int = 2,
    top_k: int = 5,
    mmr_diversity: float = 0.3,
    show_samples: bool = False,
    n_samples: int = 2,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg",
    ngram_range: Tuple[int, int] = (1, 1),
    text_source: str = "ontology"
) -> Dict[int, Dict]:
    """
    Run sub-clustering analysis on all leaves, organized by EOM parent.

    Returns:
        Dict mapping leaf_id to analysis results
    """
    print(f"\n{'='*80}")
    print("SUB-CLUSTERING ANALYSIS (c-TF-IDF + MMR KEYWORDS)")
    print(f"{'='*80}")
    lemma_str = f"lemmatization={use_lemmatization}" if use_lemmatization else "no lemmatization"
    print(f"Config: min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
          f"top_k={top_k}, mmr_diversity={mmr_diversity}, {lemma_str}, ngram={ngram_range}, text_source={text_source}")

    # Group leaves by EOM parent
    eom_to_leaves = {}
    for leaf_id, info in leaf_to_eom.items():
        eom_parent = info["eom_parent"]
        eom_to_leaves.setdefault(eom_parent, []).append(leaf_id)

    all_results = {}

    for eom_id in sorted(eom_to_leaves.keys()):
        if eom_id == -1:
            continue

        eom_size = sum(labels_eom == eom_id)
        n_leaves = len(eom_to_leaves[eom_id])

        print(f"\n{'='*70}")
        print(f"EOM CLUSTER {eom_id} (n={eom_size}, {n_leaves} leaves)")
        print(f"{'='*70}")

        for leaf_id in sorted(
            eom_to_leaves[eom_id],
            key=lambda l: leaf_to_eom[l]["size"],
            reverse=True
        ):
            result = analyze_leaf_subclusters(
                leaf_id=leaf_id,
                labels_leaf=labels_leaf,
                umap_embeddings=umap_embeddings,
                point_mapping=point_mapping,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                top_k=top_k,
                mmr_diversity=mmr_diversity,
                use_lemmatization=use_lemmatization,
                spacy_model=spacy_model,
                ngram_range=ngram_range,
                text_source=text_source
            )
            all_results[leaf_id] = result
            print_leaf_subclusters(result, show_samples=show_samples, n_samples=n_samples)

    return all_results


#%% RUN THE SUB-CLUSTERING EXPERIMENT (using Pareto-optimized overlay)
subcluster_results = run_subcluster_analysis(
    leaf_to_eom=leaf_to_eom_pareto,
    labels_leaf=labels_leaf_pareto,
    labels_eom=labels_eom,
    umap_embeddings=data["umap_embeddings"],
    point_mapping=point_mapping,
    min_cluster_size=SUBCLUSTER_MIN_CLUSTER_SIZE,
    min_samples=SUBCLUSTER_MIN_SAMPLES,
    top_k=KEYWORD_TOP_K,
    mmr_diversity=MMR_DIVERSITY,
    show_samples=False,  # Set True to see sample texts
    n_samples=2,
    use_lemmatization=USE_LEMMATIZATION,
    spacy_model=SPACY_MODEL,
    ngram_range=NGRAM_RANGE,
    text_source=KEYWORD_TEXT_SOURCE
)


#%% SUMMARY OF SUB-CLUSTERING
def print_subcluster_summary(results: Dict[int, Dict]):
    """Print summary statistics of sub-clustering."""
    print(f"\n{'='*60}")
    print("SUB-CLUSTERING SUMMARY")
    print(f"{'='*60}")

    total_leaves = len(results)
    leaves_with_subs = sum(1 for r in results.values() if r["n_subclusters"] > 0)
    total_subclusters = sum(r["n_subclusters"] for r in results.values())
    avg_subs_per_leaf = total_subclusters / leaves_with_subs if leaves_with_subs > 0 else 0

    print(f"\nLeaves analyzed:          {total_leaves}")
    print(f"Leaves with sub-clusters: {leaves_with_subs} ({leaves_with_subs/total_leaves*100:.0f}%)")
    print(f"Total sub-clusters:       {total_subclusters}")
    print(f"Avg sub-clusters/leaf:    {avg_subs_per_leaf:.1f}")

    # Distribution of sub-cluster counts
    sub_counts = [r["n_subclusters"] for r in results.values() if r["n_subclusters"] > 0]
    if sub_counts:
        print(f"\nSub-cluster count distribution:")
        print(f"  Min: {min(sub_counts)}, Max: {max(sub_counts)}, Median: {np.median(sub_counts):.0f}")

    # Keywords quality check
    leaves_with_keywords = sum(
        1 for r in results.values()
        if r["keywords"] and any(len(kw) > 0 for kw in r["keywords"].values())
    )
    print(f"\nLeaves with keywords:     {leaves_with_keywords}")


print_subcluster_summary(subcluster_results)


#%% ============================================================================
# EXPERIMENT: LEXICAL LEAF CLUSTERING WITHIN EOM
# ============================================================================
"""
Lexical grouping of leaves within each EOM cluster.

New approach (different from within-leaf clustering):
1. c-TF-IDF on leaves within EOM cluster (each leaf = one "class")
2. HDBSCAN on c-TF-IDF vectors (clusters leaves by distinctive vocabulary)
3. TF-IDF keywords for each leaf-cluster

This answers: "Which leaves within an EOM cluster share similar distinctive words?"

Example output:
  EOM 5 (3 leaves) → 2 lexical groups
    Lex-group 0 (leaves 12, 14): smaken, pittigheid, kruiden
    Lex-group 1 (leaf 15): assortiment, menu, aanbod
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


#%% LEXICAL LEAF CLUSTERING CONFIG
LEXICAL_NGRAM_RANGE = (1, 2)       # Include bigrams for phrases
LEXICAL_MIN_DF = 1                 # min_df=1 since we have few leaves per EOM
LEXICAL_MAX_DF = 1.0               # No max_df filtering at leaf level
LEXICAL_TOP_KEYWORDS = 8           # Keywords per leaf-cluster
LEXICAL_USE_LEMMATIZATION = True   # Use spaCy lemmatization (ADJ + NOUN only)
LEXICAL_SPACY_MODEL = "nl_core_news_lg"  # spaCy model for lemmatization
LEXICAL_MIN_LEAVES = 2             # Minimum leaves in EOM to attempt clustering


#%% LEXICAL LEAF CLUSTERING FUNCTIONS
def lemmatize_adj_noun_only(
    texts: List[str],
    model_name: str = "nl_core_news_lg"
) -> List[str]:
    """
    Extract lemmatized ADJ + NOUN tokens only (no PROPN/proper nouns).

    Args:
        texts: List of document strings
        model_name: spaCy model name

    Returns:
        List of lemmatized texts (space-separated ADJ/NOUN lemmas)
    """
    nlp = get_spacy_nlp(model_name)

    processed = []
    for doc in nlp.pipe(texts, batch_size=100):
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.pos_ in ('ADJ', 'NOUN'):
                tokens.append(token.lemma_.lower())
        processed.append(' '.join(tokens))

    return processed


def build_leaf_ctfidf_matrix(
    leaf_ids: List[int],
    labels_leaf: np.ndarray,
    point_mapping: List[Dict],
    text_source: str = "ontology",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg"
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build c-TF-IDF matrix where each leaf is a "class".

    Args:
        leaf_ids: List of leaf cluster IDs to include
        labels_leaf: Full array of leaf labels
        point_mapping: Global point-to-text mapping
        text_source: Which text field to use
        ngram_range: N-gram range for vocabulary
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        use_lemmatization: Use spaCy lemmatization
        spacy_model: spaCy model name

    Returns:
        Tuple of (ctfidf_matrix, vocabulary, leaf_ids_ordered)
        - ctfidf_matrix: shape (n_leaves, n_terms)
        - vocabulary: array of term strings
        - leaf_ids_ordered: list of leaf IDs in matrix row order
    """
    # Map text_source to point_mapping key
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(text_source, "ontology_text")

    # Build one concatenated document per leaf
    leaf_docs = []
    leaf_ids_ordered = []

    for leaf_id in sorted(leaf_ids):
        leaf_points = np.where(labels_leaf == leaf_id)[0]
        texts = []
        for pt_idx in leaf_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:
                    texts.append(text)

        if texts:
            # Concatenate all texts in the leaf
            leaf_doc = " ".join(texts)
            leaf_docs.append(leaf_doc)
            leaf_ids_ordered.append(leaf_id)

    if not leaf_docs:
        return np.array([]), np.array([]), []

    # Apply lemmatization if requested
    if use_lemmatization:
        leaf_docs = lemmatize_adj_noun_only(leaf_docs, model_name=spacy_model)

    # Build count matrix
    try:
        cv = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        X = cv.fit_transform(leaf_docs)
        vocab = np.array(cv.get_feature_names_out())
    except ValueError:
        return np.array([]), np.array([]), leaf_ids_ordered

    if X.shape[1] == 0:
        return np.array([]), np.array([]), leaf_ids_ordered

    # Compute c-TF-IDF
    tf = X.astype(float)
    row_sums = np.asarray(tf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1e-12
    tf = tf.multiply(1.0 / row_sums[:, np.newaxis])

    df = np.asarray((X > 0).sum(axis=0)).ravel()
    n_classes = X.shape[0]
    idf = np.log((n_classes + 1) / (df + 1)) + 1.0

    ctfidf = tf.multiply(idf).tocsr()

    return ctfidf.toarray(), vocab, leaf_ids_ordered


def cluster_leaves_by_ctfidf(
    ctfidf_matrix: np.ndarray,
    leaf_ids: List[int],
    min_cluster_size: int = 2
) -> Tuple[np.ndarray, int]:
    """
    Cluster leaves using HDBSCAN on c-TF-IDF vectors.

    Args:
        ctfidf_matrix: shape (n_leaves, n_terms)
        leaf_ids: List of leaf IDs corresponding to rows
        min_cluster_size: Minimum cluster size for HDBSCAN

    Returns:
        Tuple of (labels, n_clusters)
        - labels: cluster assignment for each leaf (-1 = noise)
        - n_clusters: number of clusters found
    """
    if ctfidf_matrix.shape[0] < min_cluster_size:
        # Not enough leaves to cluster
        return np.zeros(len(leaf_ids), dtype=int), 1

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",  # Use euclidean on L2-normalized vectors (equivalent to cosine)
        cluster_selection_method="eom",
        allow_single_cluster=True
    )
    # L2 normalize the matrix so euclidean distance ~ cosine distance
    from sklearn.preprocessing import normalize
    normalized_matrix = normalize(ctfidf_matrix, norm='l2')
    labels = clusterer.fit_predict(normalized_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


def extract_tfidf_keywords_for_leaf_cluster(
    leaf_cluster_ids: List[int],
    labels_leaf: np.ndarray,
    point_mapping: List[Dict],
    text_source: str = "ontology",
    top_k: int = 8,
    ngram_range: Tuple[int, int] = (1, 2),
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg"
) -> List[str]:
    """
    Extract TF-IDF keywords for a group of leaves.

    Args:
        leaf_cluster_ids: List of leaf IDs in this cluster
        labels_leaf: Full array of leaf labels
        point_mapping: Global point-to-text mapping
        text_source: Which text field to use
        top_k: Number of keywords to extract
        ngram_range: N-gram range
        use_lemmatization: Use lemmatization
        spacy_model: spaCy model name

    Returns:
        List of top keywords
    """
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(text_source, "ontology_text")

    # Collect all texts from all leaves in this cluster
    all_texts = []
    for leaf_id in leaf_cluster_ids:
        leaf_points = np.where(labels_leaf == leaf_id)[0]
        for pt_idx in leaf_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:
                    all_texts.append(text)

    if not all_texts:
        return []

    # Apply lemmatization
    if use_lemmatization:
        all_texts = lemmatize_adj_noun_only(all_texts, model_name=spacy_model)

    # Build TF-IDF and get top terms
    try:
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=1, max_df=1.0)
        X = vec.fit_transform(all_texts)
        vocab = vec.get_feature_names_out()
    except ValueError:
        return []

    if X.shape[1] == 0:
        return []

    # Average TF-IDF scores across all documents
    avg_scores = np.array(X.mean(axis=0)).flatten()
    top_indices = np.argsort(avg_scores)[-top_k:][::-1]

    return [vocab[i] for i in top_indices if avg_scores[i] > 0]


def analyze_eom_lexical_grouping(
    eom_id: int,
    leaf_ids: List[int],
    labels_leaf: np.ndarray,
    point_mapping: List[Dict],
    leaf_to_eom: Dict[int, Dict],
    text_source: str = "ontology",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
    top_keywords: int = 8,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg",
    min_leaves_for_clustering: int = 3
) -> Dict:
    """
    Analyze lexical grouping of leaves within an EOM cluster.

    Pipeline:
    1. Build c-TF-IDF matrix (each leaf = one "class")
    2. HDBSCAN on c-TF-IDF vectors (cluster leaves)
    3. Extract TF-IDF keywords per leaf-cluster

    Args:
        eom_id: EOM cluster ID
        leaf_ids: List of leaf IDs in this EOM cluster
        labels_leaf: Full array of leaf labels
        point_mapping: Global point-to-text mapping
        leaf_to_eom: Dict mapping leaf_id to {eom_parent, size, ...}
        text_source: Which text field to use
        ngram_range: N-gram range for TF-IDF
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        top_keywords: Number of keywords per leaf-cluster
        use_lemmatization: Use spaCy lemmatization
        spacy_model: spaCy model name
        min_leaves_for_clustering: Minimum leaves to attempt HDBSCAN

    Returns:
        Dict with eom_id, leaf groupings, keywords
    """
    n_leaves = len(leaf_ids)

    # Base result structure
    result = {
        "eom_id": eom_id,
        "n_leaves": n_leaves,
        "leaf_ids": leaf_ids,
        "leaf_groups": {},  # {group_id: [leaf_ids]}
        "leaf_labels": {},  # {leaf_id: group_id}
        "keywords": {},     # {group_id: [keywords]}
        "status": "success",
        "message": None
    }

    # Not enough leaves to cluster - each leaf is its own group
    if n_leaves < min_leaves_for_clustering:
        for i, leaf_id in enumerate(leaf_ids):
            result["leaf_groups"][i] = [leaf_id]
            result["leaf_labels"][leaf_id] = i
            # Get keywords for this single-leaf group
            keywords = extract_tfidf_keywords_for_leaf_cluster(
                [leaf_id], labels_leaf, point_mapping, text_source,
                top_keywords, ngram_range, use_lemmatization, spacy_model
            )
            result["keywords"][i] = keywords
        result["message"] = f"only {n_leaves} leaves (no clustering)"
        return result

    # Step 1: Build c-TF-IDF matrix for leaves
    ctfidf_matrix, vocab, leaf_ids_ordered = build_leaf_ctfidf_matrix(
        leaf_ids, labels_leaf, point_mapping, text_source,
        ngram_range, min_df, max_df, use_lemmatization, spacy_model
    )

    if ctfidf_matrix.size == 0:
        # Empty matrix - fallback to single-leaf groups
        for i, leaf_id in enumerate(leaf_ids):
            result["leaf_groups"][i] = [leaf_id]
            result["leaf_labels"][leaf_id] = i
        result["status"] = "fallback"
        result["message"] = "empty c-TF-IDF matrix"
        return result

    # Step 2: HDBSCAN on c-TF-IDF vectors
    leaf_cluster_labels, n_clusters = cluster_leaves_by_ctfidf(
        ctfidf_matrix, leaf_ids_ordered, min_cluster_size=2
    )

    # Step 3: Build leaf groups and extract keywords
    for group_id in set(leaf_cluster_labels):
        group_mask = leaf_cluster_labels == group_id
        group_leaf_ids = [leaf_ids_ordered[i] for i in range(len(leaf_ids_ordered)) if group_mask[i]]

        if group_id == -1:
            # Noise leaves - each is its own group
            for leaf_id in group_leaf_ids:
                noise_group_id = max(result["leaf_groups"].keys(), default=-1) + 1
                result["leaf_groups"][noise_group_id] = [leaf_id]
                result["leaf_labels"][leaf_id] = noise_group_id
                keywords = extract_tfidf_keywords_for_leaf_cluster(
                    [leaf_id], labels_leaf, point_mapping, text_source,
                    top_keywords, ngram_range, use_lemmatization, spacy_model
                )
                result["keywords"][noise_group_id] = keywords
        else:
            result["leaf_groups"][group_id] = group_leaf_ids
            for leaf_id in group_leaf_ids:
                result["leaf_labels"][leaf_id] = group_id

            # Get keywords for this leaf group
            keywords = extract_tfidf_keywords_for_leaf_cluster(
                group_leaf_ids, labels_leaf, point_mapping, text_source,
                top_keywords, ngram_range, use_lemmatization, spacy_model
            )
            result["keywords"][group_id] = keywords

    result["n_groups"] = len(result["leaf_groups"])
    return result


def print_eom_lexical_result(result: Dict, leaf_to_eom: Dict[int, Dict]):
    """Print formatted lexical leaf grouping for an EOM cluster."""
    eom_id = result["eom_id"]
    n_leaves = result["n_leaves"]
    n_groups = result.get("n_groups", len(result["leaf_groups"]))

    if result.get("message"):
        print(f"\n  EOM {eom_id} ({n_leaves} leaves) → {n_groups} groups ({result['message']})")
    else:
        print(f"\n  EOM {eom_id} ({n_leaves} leaves) → {n_groups} lexical groups")

    for group_id in sorted(result["leaf_groups"].keys()):
        leaf_ids = result["leaf_groups"][group_id]
        keywords = result["keywords"].get(group_id, [])

        # Format leaf info
        leaf_info = ", ".join([
            f"L{lid}({leaf_to_eom[lid]['size']})" for lid in sorted(leaf_ids)
        ])

        # Format keywords
        kw_str = ", ".join(keywords[:5]) if keywords else "(no keywords)"

        print(f"    Group {group_id}: [{leaf_info}] → {kw_str}")


def run_lexical_leaf_clustering(
    leaf_to_eom: Dict[int, Dict],
    labels_leaf: np.ndarray,
    labels_eom: np.ndarray,
    point_mapping: List[Dict],
    text_source: str = "ontology",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
    top_keywords: int = 8,
    min_leaves_for_clustering: int = 3,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg"
) -> Dict[int, Dict]:
    """
    Run lexical leaf clustering on all EOM clusters.

    Pipeline per EOM cluster:
    1. Build c-TF-IDF matrix (each leaf = one "class")
    2. HDBSCAN on c-TF-IDF vectors (cluster leaves by vocabulary)
    3. Extract TF-IDF keywords per leaf-group

    Args:
        leaf_to_eom: Dict mapping leaf_id to {eom_parent, size, ...}
        labels_leaf: Array of leaf cluster labels
        labels_eom: Array of EOM cluster labels
        point_mapping: Global point-to-text mapping
        text_source: Which text field to use
        ngram_range: N-gram range for TF-IDF
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        top_keywords: Number of keywords per leaf-group
        min_leaves_for_clustering: Minimum leaves to attempt HDBSCAN
        use_lemmatization: Use spaCy lemmatization (ADJ + NOUN only)
        spacy_model: spaCy model for lemmatization

    Returns:
        Dict mapping eom_id to analysis results
    """
    print(f"\n{'='*80}")
    print("LEXICAL LEAF CLUSTERING (c-TF-IDF on leaves → HDBSCAN → TF-IDF keywords)")
    print(f"{'='*80}")
    lemma_str = "lemmatization=ADJ+NOUN" if use_lemmatization else "no lemmatization"
    print(f"Config: ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}, "
          f"top_keywords={top_keywords}, {lemma_str}")
    print(f"Min leaves for clustering: {min_leaves_for_clustering}")
    print(f"Text source: {text_source}")

    # Group leaves by EOM parent
    eom_to_leaves = {}
    for leaf_id, info in leaf_to_eom.items():
        eom_parent = info["eom_parent"]
        eom_to_leaves.setdefault(eom_parent, []).append(leaf_id)

    all_results = {}

    for eom_id in sorted(eom_to_leaves.keys()):
        if eom_id == -1:
            continue

        eom_size = sum(labels_eom == eom_id)
        leaf_ids = eom_to_leaves[eom_id]
        n_leaves = len(leaf_ids)

        print(f"\n{'='*70}")
        print(f"EOM CLUSTER {eom_id} (n={eom_size}, {n_leaves} leaves)")
        print(f"{'='*70}")

        result = analyze_eom_lexical_grouping(
            eom_id=eom_id,
            leaf_ids=leaf_ids,
            labels_leaf=labels_leaf,
            point_mapping=point_mapping,
            leaf_to_eom=leaf_to_eom,
            text_source=text_source,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            top_keywords=top_keywords,
            use_lemmatization=use_lemmatization,
            spacy_model=spacy_model,
            min_leaves_for_clustering=min_leaves_for_clustering
        )
        all_results[eom_id] = result
        print_eom_lexical_result(result, leaf_to_eom)

    return all_results


def print_lexical_leaf_clustering_summary(results: Dict[int, Dict], leaf_to_eom: Dict[int, Dict]):
    """Print summary statistics of lexical leaf clustering."""
    print(f"\n{'='*60}")
    print("LEXICAL LEAF CLUSTERING SUMMARY")
    print(f"{'='*60}")

    total_eoms = len(results)
    total_leaves = sum(r["n_leaves"] for r in results.values())
    total_groups = sum(r.get("n_groups", 0) for r in results.values())

    # Count EOM clusters where leaves were grouped (multiple leaves in same group)
    eoms_with_grouping = 0
    for r in results.values():
        if r.get("n_groups", r["n_leaves"]) < r["n_leaves"]:
            eoms_with_grouping += 1

    print(f"\nEOM clusters analyzed:     {total_eoms}")
    print(f"Total leaves:              {total_leaves}")
    print(f"Total leaf groups:         {total_groups}")
    print(f"EOM clusters with grouping: {eoms_with_grouping} ({eoms_with_grouping/total_eoms*100:.0f}%)")

    if total_groups > 0:
        avg_leaves_per_group = total_leaves / total_groups
        print(f"Avg leaves/group:          {avg_leaves_per_group:.1f}")

    # Breakdown by group size
    group_sizes = []
    for r in results.values():
        for group_id, leaf_ids in r.get("leaf_groups", {}).items():
            group_sizes.append(len(leaf_ids))

    if group_sizes:
        single_leaf_groups = sum(1 for s in group_sizes if s == 1)
        multi_leaf_groups = sum(1 for s in group_sizes if s > 1)
        print(f"\nGroup breakdown:")
        print(f"  Single-leaf groups: {single_leaf_groups}")
        print(f"  Multi-leaf groups:  {multi_leaf_groups}")
        if multi_leaf_groups > 0:
            multi_sizes = [s for s in group_sizes if s > 1]
            print(f"  Multi-leaf sizes:   min={min(multi_sizes)}, max={max(multi_sizes)}, "
                  f"avg={np.mean(multi_sizes):.1f}")


#%% RUN LEXICAL LEAF CLUSTERING EXPERIMENT (using Pareto-optimized overlay)
lexical_results = run_lexical_leaf_clustering(
    leaf_to_eom=leaf_to_eom_pareto,
    labels_leaf=labels_leaf_pareto,
    labels_eom=labels_eom,
    point_mapping=point_mapping,
    text_source=KEYWORD_TEXT_SOURCE,  # Reuse from semantic experiment
    ngram_range=LEXICAL_NGRAM_RANGE,
    min_df=LEXICAL_MIN_DF,
    max_df=LEXICAL_MAX_DF,
    top_keywords=LEXICAL_TOP_KEYWORDS,
    min_leaves_for_clustering=LEXICAL_MIN_LEAVES,
    use_lemmatization=LEXICAL_USE_LEMMATIZATION,
    spacy_model=LEXICAL_SPACY_MODEL
)


#%% LEXICAL LEAF CLUSTERING SUMMARY
print_lexical_leaf_clustering_summary(lexical_results, leaf_to_eom_pareto)


#%% ============================================================================
# EXPERIMENT: TF-IDF LEXICAL CLUSTERING WITHIN EOM
# ============================================================================
"""
Lexical clustering of individual ideas within each EOM cluster.

Different from existing experiments:
- Uses vanilla TF-IDF on individual ideas (not c-TF-IDF on leaves)
- Clusters ideas by word similarity (not semantic embeddings)
- Lemmatizes using ADJ + NOUN only (no verbs, proper nouns)

Pipeline per EOM cluster:
1. Extract all ideas belonging to EOM cluster
2. Lemmatize texts (ADJ + NOUN only)
3. Build vanilla TF-IDF matrix (one row per idea)
4. L2-normalize and run HDBSCAN
5. Extract top keywords per sub-cluster
"""

#%% TF-IDF LEXICAL CONFIG
TFIDF_NGRAM_RANGE = (1, 1)            # Unigrams only
TFIDF_MIN_DF = 1                       # Minimum document frequency
TFIDF_MAX_DF = 0.95                    # Ignore terms in >95% of docs
TFIDF_USE_LEMMATIZATION = True         # Use spaCy lemmatization
TFIDF_SPACY_MODEL = "nl_core_news_lg"  # Dutch language model
TFIDF_HDBSCAN_MCS = 3                  # min_cluster_size
TFIDF_HDBSCAN_MS = 2                   # min_samples
TFIDF_TOP_KEYWORDS = 5                 # Keywords per sub-cluster
TFIDF_TEXT_SOURCE = "taxonomy"         # Text source for clustering
TFIDF_SHOW_SAMPLES = 3                 # Number of sample texts per cluster


#%% TF-IDF LEXICAL FUNCTIONS
def build_tfidf_matrix_for_eom(
    eom_id: int,
    labels_eom: np.ndarray,
    point_mapping: List[Dict],
    text_source: str = "ontology",
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_df: float = 0.95,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """
    Build vanilla TF-IDF matrix for ideas in an EOM cluster.

    Args:
        eom_id: EOM cluster ID
        labels_eom: Array of EOM cluster labels
        point_mapping: Global point-to-text mapping
        text_source: Which text field to use
        ngram_range: N-gram range for TF-IDF
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        use_lemmatization: Use spaCy lemmatization (ADJ+NOUN only)
        spacy_model: spaCy model name

    Returns:
        Tuple of (tfidf_matrix, vocabulary, point_indices)
        - tfidf_matrix: shape (n_ideas, n_terms) or None if failed
        - vocabulary: array of term strings or None if failed
        - point_indices: global indices of ideas in this EOM
    """
    # Map text_source to point_mapping key
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(text_source, "ontology_text")

    # Get point indices for this EOM cluster
    point_indices = np.where(labels_eom == eom_id)[0].tolist()

    if len(point_indices) < 2:
        return None, None, point_indices

    # Extract texts
    texts = []
    for pt_idx in point_indices:
        if pt_idx < len(point_mapping):
            text = point_mapping[pt_idx].get(text_key, "")
            texts.append(text if text else "")
        else:
            texts.append("")

    # Apply lemmatization (ADJ + NOUN only)
    if use_lemmatization:
        texts = lemmatize_adj_noun_only(texts, model_name=spacy_model)

    # Filter empty texts
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    if len(valid_indices) < 2:
        return None, None, point_indices

    valid_texts = [texts[i] for i in valid_indices]
    valid_point_indices = [point_indices[i] for i in valid_indices]

    # Build TF-IDF matrix
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        vocabulary = np.array(vectorizer.get_feature_names_out())
    except ValueError:
        return None, None, valid_point_indices

    if tfidf_matrix.shape[1] == 0:
        return None, None, valid_point_indices

    return tfidf_matrix.toarray(), vocabulary, valid_point_indices


def cluster_ideas_by_tfidf(
    tfidf_matrix: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2
) -> Tuple[np.ndarray, int]:
    """
    Run HDBSCAN on TF-IDF vectors.

    Args:
        tfidf_matrix: TF-IDF matrix shape (n_ideas, n_terms)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for core points

    Returns:
        Tuple of (labels, n_clusters)
    """
    if tfidf_matrix.shape[0] < min_cluster_size:
        # Not enough ideas to cluster
        return np.zeros(tfidf_matrix.shape[0], dtype=int), 1

    # L2-normalize so euclidean distance ~ cosine distance
    from sklearn.preprocessing import normalize
    normalized_matrix = normalize(tfidf_matrix, norm='l2')

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=True
    )
    labels = clusterer.fit_predict(normalized_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


def extract_tfidf_cluster_keywords(
    cluster_id: int,
    cluster_mask: np.ndarray,
    tfidf_matrix: np.ndarray,
    vocabulary: np.ndarray,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Extract top keywords for a cluster by average TF-IDF score.

    Args:
        cluster_id: Cluster ID
        cluster_mask: Boolean mask for cluster membership
        tfidf_matrix: Full TF-IDF matrix
        vocabulary: Array of term strings
        top_k: Number of keywords to extract

    Returns:
        List of (keyword, score) tuples
    """
    cluster_rows = tfidf_matrix[cluster_mask]
    if cluster_rows.shape[0] == 0:
        return []

    # Average TF-IDF scores across cluster
    avg_scores = cluster_rows.mean(axis=0)

    # Get top-k terms
    top_indices = np.argsort(avg_scores)[-top_k:][::-1]

    return [(vocabulary[i], avg_scores[i]) for i in top_indices if avg_scores[i] > 0]


def analyze_eom_tfidf_clustering(
    eom_id: int,
    labels_eom: np.ndarray,
    point_mapping: List[Dict],
    text_source: str = "ontology",
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_df: float = 0.95,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg",
    hdbscan_mcs: int = 3,
    hdbscan_ms: int = 2,
    top_keywords: int = 5
) -> Dict:
    """
    Full TF-IDF clustering pipeline for one EOM cluster.

    Args:
        eom_id: EOM cluster ID
        labels_eom: Array of EOM cluster labels
        point_mapping: Global point-to-text mapping
        text_source: Which text field to use
        ngram_range: N-gram range for TF-IDF
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        use_lemmatization: Use spaCy lemmatization
        spacy_model: spaCy model name
        hdbscan_mcs: HDBSCAN min_cluster_size
        hdbscan_ms: HDBSCAN min_samples
        top_keywords: Number of keywords per cluster

    Returns:
        Dict with clustering results
    """
    # Build TF-IDF matrix
    tfidf_matrix, vocabulary, point_indices = build_tfidf_matrix_for_eom(
        eom_id, labels_eom, point_mapping, text_source,
        ngram_range, min_df, max_df, use_lemmatization, spacy_model
    )

    result = {
        "eom_id": eom_id,
        "n_ideas": len(point_indices),
        "point_indices": point_indices,
        "n_clusters": 0,
        "n_noise": 0,
        "cluster_labels": None,
        "clusters": {},
        "keywords": {},
        "status": "success",
        "message": None
    }

    if tfidf_matrix is None:
        result["status"] = "skipped"
        result["message"] = "not enough valid texts"
        return result

    # Run HDBSCAN clustering
    cluster_labels, n_clusters = cluster_ideas_by_tfidf(
        tfidf_matrix, hdbscan_mcs, hdbscan_ms
    )

    result["n_clusters"] = n_clusters
    result["n_noise"] = int(sum(cluster_labels == -1))
    result["cluster_labels"] = cluster_labels

    # Extract keywords and sample texts per cluster
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        cluster_point_indices = [point_indices[i] for i in np.where(cluster_mask)[0]]

        # Get sample texts (display_text for readability)
        sample_texts = []
        for pt_idx in cluster_point_indices:
            if pt_idx < len(point_mapping):
                sample_texts.append(point_mapping[pt_idx].get("display_text", ""))

        result["clusters"][cluster_id] = {
            "n_ideas": int(sum(cluster_mask)),
            "point_indices": cluster_point_indices,
            "sample_texts": sample_texts
        }

        # Extract keywords (skip noise cluster)
        if cluster_id != -1:
            keywords = extract_tfidf_cluster_keywords(
                cluster_id, cluster_mask, tfidf_matrix, vocabulary, top_keywords
            )
            result["keywords"][cluster_id] = keywords

    return result


def print_eom_tfidf_result(
    result: Dict,
    show_samples: int = 3
):
    """Print formatted TF-IDF clustering result for an EOM cluster."""
    eom_id = result["eom_id"]
    n_ideas = result["n_ideas"]
    n_clusters = result["n_clusters"]
    n_noise = result["n_noise"]

    if result["status"] == "skipped":
        print(f"\n  EOM {eom_id} (n={n_ideas}) → skipped ({result['message']})")
        return

    noise_str = f" + {n_noise} noise" if n_noise > 0 else ""
    print(f"\n  Found {n_clusters} lexical cluster(s){noise_str}")

    for cluster_id in sorted(result["clusters"].keys()):
        if cluster_id == -1:
            continue  # Skip noise in output

        cluster_info = result["clusters"][cluster_id]
        keywords = result["keywords"].get(cluster_id, [])

        # Format keywords
        kw_str = ", ".join([kw for kw, _ in keywords[:5]]) if keywords else "(no keywords)"

        print(f"\n    Cluster {cluster_id} (n={cluster_info['n_ideas']}) → {kw_str}")

        # Show sample texts
        if show_samples > 0 and cluster_info["sample_texts"]:
            for text in cluster_info["sample_texts"][:show_samples]:
                text_display = text[:70] + "..." if len(text) > 70 else text
                print(f"      • {text_display}")


def run_tfidf_lexical_clustering(
    labels_eom: np.ndarray,
    point_mapping: List[Dict],
    text_source: str = "ontology",
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_df: float = 0.95,
    use_lemmatization: bool = True,
    spacy_model: str = "nl_core_news_lg",
    hdbscan_mcs: int = 3,
    hdbscan_ms: int = 2,
    top_keywords: int = 5,
    show_samples: int = 3
) -> Dict[int, Dict]:
    """
    Run TF-IDF lexical clustering on all EOM clusters.

    Pipeline per EOM cluster:
    1. Build vanilla TF-IDF matrix (one row per idea)
    2. L2-normalize and run HDBSCAN
    3. Extract top keywords per sub-cluster

    Returns:
        Dict mapping eom_id to analysis results
    """
    print(f"\n{'='*80}")
    print("TF-IDF LEXICAL CLUSTERING WITHIN EOM CLUSTERS")
    print(f"{'='*80}")
    lemma_str = "lemmatization=ADJ+NOUN" if use_lemmatization else "no lemmatization"
    print(f"Config: ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}, {lemma_str}")
    print(f"HDBSCAN: min_cluster_size={hdbscan_mcs}, min_samples={hdbscan_ms}")
    print(f"Text source: {text_source}")

    all_results = {}
    eom_ids = sorted(set(labels_eom) - {-1})

    for eom_id in eom_ids:
        eom_size = sum(labels_eom == eom_id)

        print(f"\n{'='*70}")
        print(f"EOM CLUSTER {eom_id} (n={eom_size} ideas)")
        print(f"{'='*70}")

        result = analyze_eom_tfidf_clustering(
            eom_id=eom_id,
            labels_eom=labels_eom,
            point_mapping=point_mapping,
            text_source=text_source,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            use_lemmatization=use_lemmatization,
            spacy_model=spacy_model,
            hdbscan_mcs=hdbscan_mcs,
            hdbscan_ms=hdbscan_ms,
            top_keywords=top_keywords
        )
        all_results[eom_id] = result
        print_eom_tfidf_result(result, show_samples=show_samples)

    return all_results


def print_tfidf_clustering_summary(results: Dict[int, Dict]):
    """Print summary statistics of TF-IDF lexical clustering."""
    print(f"\n{'='*60}")
    print("TF-IDF LEXICAL CLUSTERING SUMMARY")
    print(f"{'='*60}")

    total_eoms = len(results)
    successful_eoms = sum(1 for r in results.values() if r["status"] == "success")
    total_ideas = sum(r["n_ideas"] for r in results.values())
    total_clusters = sum(r["n_clusters"] for r in results.values() if r["status"] == "success")
    total_noise = sum(r["n_noise"] for r in results.values() if r["status"] == "success")

    print(f"\nEOM clusters analyzed:     {total_eoms}")
    print(f"EOM clusters with results: {successful_eoms}")
    print(f"Total ideas:               {total_ideas}")
    print(f"Total lexical clusters:    {total_clusters}")
    print(f"Total noise points:        {total_noise} ({total_noise/total_ideas*100:.1f}%)")

    if successful_eoms > 0:
        avg_clusters = total_clusters / successful_eoms
        print(f"Avg clusters per EOM:      {avg_clusters:.1f}")

    # Distribution of cluster sizes
    cluster_sizes = []
    for r in results.values():
        if r["status"] == "success":
            for cid, info in r.get("clusters", {}).items():
                if cid != -1:
                    cluster_sizes.append(info["n_ideas"])

    if cluster_sizes:
        print(f"\nCluster size distribution:")
        print(f"  Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, "
              f"Median: {np.median(cluster_sizes):.0f}, Mean: {np.mean(cluster_sizes):.1f}")


#%% RUN TF-IDF LEXICAL CLUSTERING EXPERIMENT
tfidf_lexical_results = run_tfidf_lexical_clustering(
    labels_eom=labels_eom,
    point_mapping=point_mapping,
    text_source=TFIDF_TEXT_SOURCE,
    ngram_range=TFIDF_NGRAM_RANGE,
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    use_lemmatization=TFIDF_USE_LEMMATIZATION,
    spacy_model=TFIDF_SPACY_MODEL,
    hdbscan_mcs=TFIDF_HDBSCAN_MCS,
    hdbscan_ms=TFIDF_HDBSCAN_MS,
    top_keywords=TFIDF_TOP_KEYWORDS,
    show_samples=TFIDF_SHOW_SAMPLES
)


#%% TF-IDF LEXICAL CLUSTERING SUMMARY
print_tfidf_clustering_summary(tfidf_lexical_results)