"""
ClustererV2 Test Script

Tests the ClustererV2 implementation with cached Step 4 embeddings.
Runs on both small (Vezet Q20, n=50) and medium (ASN Bank) datasets.

Usage:
    cd src/experiments/clusterer_v2
    python test_clusterer.py
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pickle

import numpy as np

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from utils.cacheManager import generate_enhanced_variable_key

# Import ClustererV2 (from experiments/clusterer_v2/)
from clusterer_v2 import ClustererV2, ClustererV2Config
from clusterer_v2.algorithm_selector import AlgorithmRecommendation


# =============================================================================
# CONFIGURATION
# =============================================================================

# Test datasets
DATASETS = [
    {
        "name": "Vezet Q20 (small)",
        "filename": "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
        "variable": "Q20",
        "sample_size": 50,
    },
    {
        "name": "ASN Bank (medium)",
        "filename": "M241011 ASN Bank 2024 databestand.sav",
        "variable": "Q18",
        "sample_size": None,  # Full dataset
    },
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step4_embeddings(
    filename: str,
    variable: str,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load Step 4 embeddings from cache.

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
    """
    import models

    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(filename).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run pipeline step 4 first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    # Convert serialized data to EmbeddingsModel objects
    data = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    # Build embeddings array and idea texts list
    embeddings_list = []
    idea_texts = []

    for response in data:
        if response.response_ideas:
            for idea in response.response_ideas:
                if idea.idea_embedding is not None:
                    embeddings_list.append(idea.idea_embedding)
                    idea_texts.append(idea.idea)

    if not embeddings_list:
        raise ValueError("No embeddings found in cached data")

    embeddings = np.vstack(embeddings_list)
    return embeddings, idea_texts


def try_load_dataset(dataset: dict) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Try to load a dataset, return (None, None) if not found."""
    try:
        embeddings, idea_texts = load_step4_embeddings(
            filename=dataset["filename"],
            variable=dataset["variable"],
            sample_size=dataset["sample_size"]
        )
        return embeddings, idea_texts
    except FileNotFoundError as e:
        print(f"  ⚠ Skipping {dataset['name']}: {e}")
        return None, None


def create_embeddings_models(
    embeddings: np.ndarray,
    idea_texts: List[str],
    ideas_per_response: int = 5
) -> List:
    """
    Create a list of EmbeddingsModel objects from raw arrays.

    This is needed because ClustererV2 expects EmbeddingsModel input,
    matching the actual pipeline structure.
    """
    import models

    embeddings_models = []

    for resp_idx in range(0, len(idea_texts), ideas_per_response):
        end_idx = min(resp_idx + ideas_per_response, len(idea_texts))
        resp_id = f"resp_{resp_idx // ideas_per_response}"

        # Create ideas for this response
        ideas = []
        for i in range(resp_idx, end_idx):
            idea_num = i - resp_idx
            idea = models.EmbeddingsSubmodel(
                idea_id=f"{resp_id}_{idea_num}",
                idea=idea_texts[i],
                idea_embedding=embeddings[i]
            )
            ideas.append(idea)

        # Create EmbeddingsModel response
        resp = models.EmbeddingsModel(
            respondent_id=resp_id,
            response="mock text",
            response_ideas=ideas
        )
        embeddings_models.append(resp)

    return embeddings_models


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def print_recommendation(rec: AlgorithmRecommendation):
    """Print algorithm recommendation details."""
    print(f"\n  Algorithm Recommendation:")
    print(f"    Recommended: {rec.recommended_algorithm} ({rec.confidence})")
    print(f"    DVC: {rec.dvc_value:.3f} → {rec.dvc_recommendation}")
    print(f"    Knee: y_diff={rec.y_difference:.2f}, sharp={rec.has_sharp_knee}")
    mean_p = rec.mean_persistence if rec.mean_persistence is not None else 0.0
    weighted_p = rec.weighted_persistence if rec.weighted_persistence is not None else 0.0
    print(f"    Persistence: mean={mean_p:.3f}, weighted={weighted_p:.3f}")
    print(f"    Combined: {rec.combined_recommendation}")


def print_metrics(metrics):
    """Print clustering metrics."""
    print(f"\n  Clustering Metrics:")
    print(f"    Clusters: {metrics.n_clusters}")
    print(f"    Noise: {metrics.noise_count} ({metrics.noise_rate:.1%})")
    print(f"    Coherence: {metrics.mean_coherence:.3f} ({metrics.coherence_breakdown})")
    if metrics.dbcv is not None:
        print(f"    DBCV: {metrics.dbcv:.3f}")
    if metrics.silhouette is not None and not np.isnan(metrics.silhouette):
        print(f"    Silhouette: {metrics.silhouette:.3f}")
    if metrics.mean_persistence is not None:
        print(f"    Persistence: mean={metrics.mean_persistence:.3f}, weighted={metrics.weighted_persistence:.3f}")
    print(f"    Cluster sizes: min={metrics.min_cluster_size}, median={metrics.median_cluster_size}, max={metrics.max_cluster_size}")


def print_keywords(keywords: dict, max_clusters: int = 5):
    """Print c-TF-IDF keywords for clusters."""
    if not keywords:
        print("\n  Keywords: (not generated)")
        return

    print(f"\n  c-TF-IDF Keywords (top {max_clusters} clusters):")
    for cluster_id, kw_list in list(keywords.items())[:max_clusters]:
        kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
        print(f"    Cluster {cluster_id}: {kw_str}")


def test_auto_mode(embeddings: np.ndarray, idea_texts: List[str], dataset_name: str):
    """Test ClustererV2 in auto mode."""
    print(f"\n{'='*60}")
    print(f"Testing AUTO mode on {dataset_name}")
    print(f"{'='*60}")

    # Create EmbeddingsModel list from raw arrays
    input_list = create_embeddings_models(embeddings, idea_texts)

    config = ClustererV2Config(
        algorithm_mode="auto",
        generate_ctfidf=True,
        verbose=True
    )

    clusterer = ClustererV2(input_list, config=config)
    clusterer.run()

    # Print results
    recommendation = clusterer.get_algorithm_recommendation()
    if recommendation:
        print_recommendation(recommendation)

    metrics = clusterer.get_metrics()
    if metrics:
        print_metrics(metrics)

    keywords = clusterer.get_cluster_keywords()
    print_keywords(keywords)

    return clusterer


def test_hdbscan_mode(embeddings: np.ndarray, idea_texts: List[str], dataset_name: str):
    """Test ClustererV2 in HDBSCAN-only mode."""
    print(f"\n{'='*60}")
    print(f"Testing HDBSCAN mode on {dataset_name}")
    print(f"{'='*60}")

    # Create EmbeddingsModel list from raw arrays
    input_list = create_embeddings_models(embeddings, idea_texts)

    config = ClustererV2Config(
        algorithm_mode="hdbscan",
        use_optuna=True,
        generate_ctfidf=False,
        verbose=True
    )

    clusterer = ClustererV2(input_list, config=config)
    clusterer.run()

    # Print results
    recommendation = clusterer.get_algorithm_recommendation()
    if recommendation:
        print_recommendation(recommendation)

    metrics = clusterer.get_metrics()
    if metrics:
        print_metrics(metrics)

    return clusterer


def test_agglomerative_mode(embeddings: np.ndarray, idea_texts: List[str], dataset_name: str):
    """Test ClustererV2 in agglomerative mode."""
    print(f"\n{'='*60}")
    print(f"Testing AGGLOMERATIVE mode on {dataset_name}")
    print(f"{'='*60}")

    # Create EmbeddingsModel list from raw arrays
    input_list = create_embeddings_models(embeddings, idea_texts)

    # Uses sqrt-based k selection by default
    config = ClustererV2Config(
        algorithm_mode="agglomerative",
        generate_ctfidf=False,
        verbose=True
    )

    clusterer = ClustererV2(input_list, config=config)
    clusterer.run()

    # Print results
    recommendation = clusterer.get_algorithm_recommendation()
    if recommendation:
        print_recommendation(recommendation)

    metrics = clusterer.get_metrics()
    if metrics:
        print_metrics(metrics)

    return clusterer


def test_pipeline_compatibility(embeddings: np.ndarray, idea_texts: List[str], dataset_name: str):
    """Test that ClustererV2 produces a valid ClusterModel for pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing Pipeline Compatibility on {dataset_name}")
    print(f"{'='*60}")

    # Create EmbeddingsModel list from raw arrays
    input_list = create_embeddings_models(embeddings, idea_texts)

    config = ClustererV2Config(
        algorithm_mode="auto",
        generate_ctfidf=False,
        verbose=False
    )

    clusterer = ClustererV2(input_list, config=config)
    clusterer.run()

    # Test to_cluster_model
    cluster_models = clusterer.to_cluster_model()

    if cluster_models:
        print(f"  ClusterModel created successfully!")
        total_ideas = sum(len(r.response_ideas) if r.response_ideas else 0 for r in cluster_models)
        clustered_ideas = sum(
            1 for r in cluster_models
            for idea in (r.response_ideas or [])
            if idea.initial_cluster is not None
        )
        noise_ideas = sum(
            1 for r in cluster_models
            for idea in (r.response_ideas or [])
            if idea.initial_cluster == -1
        )
        print(f"    Total responses: {len(cluster_models)}")
        print(f"    Total ideas: {total_ideas}")
        print(f"    Clustered ideas: {clustered_ideas}")
        print(f"    Noise ideas: {noise_ideas}")

        # Verify cluster IDs match
        labels = clusterer._labels
        assigned_labels = []
        for r in cluster_models:
            for idea in (r.response_ideas or []):
                assigned_labels.append(idea.initial_cluster)

        if len(assigned_labels) == len(labels):
            match = all(a == b for a, b in zip(assigned_labels, labels))
            print(f"    Labels match: {match}")
        else:
            print(f"    ⚠ Label count mismatch: {len(assigned_labels)} vs {len(labels)}")

        return cluster_models
    else:
        print("  ⚠ No cluster models returned")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("ClustererV2 Test Suite")
    print("=" * 60)

    # Find a dataset that exists
    test_embeddings = None
    test_idea_texts = None
    test_dataset_name = None

    for dataset in DATASETS:
        print(f"\nTrying to load: {dataset['name']}...")
        embeddings, idea_texts = try_load_dataset(dataset)
        if embeddings is not None:
            test_embeddings = embeddings
            test_idea_texts = idea_texts
            test_dataset_name = dataset['name']
            print(f"  ✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]} dims)")
            break

    if test_embeddings is None:
        print("\n⚠ No cached embeddings found. Please run pipeline step 4 first.")
        print("  Tested paths:")
        for dataset in DATASETS:
            var_key = generate_enhanced_variable_key(
                selected_variables=[dataset["variable"]],
                is_merged=False,
                sample_size=dataset["sample_size"]
            )
            base_name = Path(dataset["filename"]).stem
            cache_filename = f"005_embeddings_{base_name}_{var_key}.pkl"
            cache_path = project_root / "data" / "cache" / cache_filename
            print(f"    - {cache_path}")
        return

    # Run tests
    print(f"\n{'#'*60}")
    print(f"# Running tests on: {test_dataset_name}")
    print(f"# Embeddings: {test_embeddings.shape}")
    print(f"{'#'*60}")

    # Test 1: Auto mode
    test_auto_mode(test_embeddings, test_idea_texts, test_dataset_name)

    # Test 2: HDBSCAN mode
    test_hdbscan_mode(test_embeddings, test_idea_texts, test_dataset_name)

    # Test 3: Agglomerative mode
    test_agglomerative_mode(test_embeddings, test_idea_texts, test_dataset_name)

    # Test 4: Pipeline compatibility
    test_pipeline_compatibility(test_embeddings, test_idea_texts, test_dataset_name)

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
