#%%

"""
Step 5: Clusterer V3 Experiment Runner

Run the Clusterer V3 pipeline on a specific dataset from cached Step 4 embeddings.

V3 pipeline:
- Phases 1-5: Same as V2 (preprocessing, algorithm selection, clustering,
  post-processing, metrics)
- Phase 6: REMOVED (no keyword extraction)
- Phase 7: Map-Reduce MECE per cluster
  - MAP: batch all ideas, find ALL atomic themes per batch
  - REDUCE: consolidate themes across batches
  - MECE: apply inclusion/exclusion boundaries
- Phase 8: REMOVED (no cross-cluster consolidation)

This experiment runner uses LOCAL COPIES that can be modified
without affecting the production pipeline. Edit these files:
- clusterer_exp.py             (main clusterer class)
- clusterer_helpers_exp.py     (helper functions/classes)
- config_clusterer_exp.py      (configuration dataclass)
- prompts_exp.py               (LLM prompts for map/reduce/MECE)
- map_reduce_mece.py           (Map-Reduce MECE logic)

Dataset configuration is centralized in experiments/test_data.py.

Usage:
    cd src && python -m experiments.step_5_clusterer_v3.run_experiment
"""

import sys
import io
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import pickle

import numpy as np

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments import models_exp as models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import V3 Clusterer
# Handle both module execution (-m) and direct/notebook execution
try:
    # Module execution (python -m experiments.step_5_clusterer_v3.run_experiment)
    from .clusterer_exp import Clusterer
    from .config_clusterer_exp import ClustererConfig
except ImportError:
    # Direct/notebook execution
    from experiments.step_5_clusterer_v3.clusterer_exp import Clusterer
    from experiments.step_5_clusterer_v3.config_clusterer_exp import ClustererConfig

# =============================================================================
# DATASET CONFIGURATION (centralized in experiments/test_data.py)
# =============================================================================
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# CLUSTERER CONFIGURATION
# =============================================================================
# All defaults defined in config_clusterer_exp.py (single source of truth).
# Override individual params here only for one-off experiments.
CONFIG = ClustererConfig(
    embedding_source="category",           # cluster on category-level embeddings (on-the-fly)
    # mapreduce_text_source defaults to "idea" (idea.idea text)
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step4_embeddings(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> List[models.EmbeddingsModel]:
    """Load Step 4 embeddings from cache."""
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

    print(f"Loading embeddings from: {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run pipeline step 4 first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    embeddings_models = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    # Log cached format for visibility
    embedding_format = "idea"
    if embeddings_models and hasattr(embeddings_models[0], 'embedding_text_format'):
        embedding_format = embeddings_models[0].embedding_text_format or "idea"
    print(f"Cached embedding format: {embedding_format}")

    return embeddings_models


def load_extraction_metadata(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None
) -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata from cache (if available)."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_manager = CacheManager()
    metadata = cache_manager.load_metadata_from_cache(
        filename=filename,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )

    if metadata:
        print(f"Loaded ExtractionMetadata: taxonomy_axis={metadata.taxonomy_axis}")
        if metadata.taxonomy_axis_description:
            print(f"  Description: {metadata.taxonomy_axis_description}")
        if metadata.var_lab:
            print(f"  Survey question (var_lab): {metadata.var_lab}")
        else:
            print(f"  Survey question (var_lab): NOT SET")
    else:
        print("ExtractionMetadata not found in cache (optional)")

    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Clusterer V3 on the configured dataset."""
    print("=" * 70)
    print("Clustering Pipeline V3 (Map-Reduce MECE)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Algorithm mode: {CONFIG.algorithm_mode}")
    print(f"Batch size: {CONFIG.mapreduce_batch_size}")
    print()

    # Load embeddings
    embeddings_models = load_step4_embeddings()

    # Load extraction metadata (optional - for taxonomy context)
    extraction_metadata = load_extraction_metadata()

    # Run clusterer
    clusterer = Clusterer(embeddings_models, config=CONFIG, extraction_metadata=extraction_metadata)
    clusterer.run()

    # ==========================================================================
    # CACHE RESULTS (like pipeline.py step 5)
    # ==========================================================================
    cluster_results = clusterer.to_cluster_model()

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_manager = CacheManager()
    var_lab = extraction_metadata.var_lab if extraction_metadata else None

    cache_manager.save_to_cache(
        cluster_results,
        FILENAME,
        "initial_clusters",
        variable_key,
        0,
        var_lab=var_lab
    )

    # Layer 2: Clustering metadata
    metadata = clusterer.to_metadata_model()
    cache_manager.save_to_cache(
        [metadata],
        FILENAME,
        "clustering_metadata",
        variable_key,
        0,
        var_lab=var_lab
    )

    # Layer 3: HDBSCAN artifacts
    hdbscan_artifacts = clusterer.get_hdbscan_artifacts()
    if hdbscan_artifacts:
        base_name = Path(FILENAME).stem
        artifacts_path = project_root / "data" / "cache" / f"hdbscan_artifacts_{base_name}_{variable_key}.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(hdbscan_artifacts, f)
        print(f"CACHED: HDBSCAN artifacts to '{artifacts_path.name}'")

    # Layer 4: UMAP embeddings + winning params
    umap_embeddings = clusterer.get_umap_embeddings()
    hdbscan_params = clusterer.get_hdbscan_params()
    if umap_embeddings is not None:
        base_name = Path(FILENAME).stem
        umap_path = project_root / "data" / "cache" / f"umap_embeddings_{base_name}_{variable_key}.pkl"
        with open(umap_path, 'wb') as f:
            pickle.dump({
                "embeddings": umap_embeddings,
                "params": hdbscan_params,
            }, f)
        print(f"CACHED: UMAP embeddings + params to '{umap_path.name}'")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Algorithm recommendation
    rec = clusterer.get_algorithm_recommendation()
    if rec:
        print(f"\nAlgorithm Recommendation:")
        print(f"  Recommended: {rec.recommended_algorithm} ({rec.confidence} confidence)")
        print(f"  DVC: {rec.dvc_value:.3f} → {rec.dvc_recommendation}")
        print(f"  Knee: y_diff={rec.y_difference:.2f}, sharp={rec.has_sharp_knee}")
        if rec.is_forced:
            print(f"  FORCED: Algorithm selection was forced by hard DVC rule")
        print(f"  Reasoning: {rec.reasoning}")

    # Metrics
    metrics = clusterer.get_metrics()
    if metrics:
        print(f"\nClustering Metrics:")
        print(f"  Clusters: {metrics.n_clusters}")
        print(f"  Noise: {metrics.noise_count} ({metrics.noise_rate:.1%})")
        print(f"  Coherence: {metrics.mean_coherence:.3f} ({metrics.coherence_breakdown})")
        if metrics.dbcv is not None:
            print(f"  DBCV: {metrics.dbcv:.3f}")
        if metrics.silhouette is not None and not np.isnan(metrics.silhouette):
            print(f"  Silhouette: {metrics.silhouette:.3f}")
        if metrics.mean_persistence is not None:
            print(f"  Persistence: mean={metrics.mean_persistence:.3f}, weighted={metrics.weighted_persistence:.3f}")
        if metrics.mean_probability is not None:
            print(f"  Probability: mean={metrics.mean_probability:.3f}, low_ratio={metrics.low_prob_ratio:.1%}")
        if metrics.mean_outlier_score is not None:
            print(f"  Outliers: mean_score={metrics.mean_outlier_score:.3f}, high_ratio={metrics.high_outlier_ratio:.1%}")
        print(f"  Cluster sizes: min={metrics.min_cluster_size}, median={metrics.median_cluster_size}, max={metrics.max_cluster_size}")

    # Print detailed MECE topics
    clusterer.print_cluster_mece_topics()

    # Cache confirmation
    print(f"\n{'='*70}")
    print(f"CACHED: {len(cluster_results)} results to 'initial_clusters' (variable_key: {variable_key})")
    print(f"CACHED: {len(metadata.clusters)} clusters to 'clustering_metadata'")

    # MECE summary
    mece_results = clusterer.get_cluster_mece_results()
    if mece_results:
        total_topics = sum(len(r.topics) for r in mece_results.values())
        print(f"\nMECE: {len(mece_results)} clusters → {total_topics} MECE topics total")

    return clusterer, embeddings_models


class TeeOutput:
    """Capture stdout while also printing to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self) -> str:
        return self.buffer.getvalue()


def save_results_to_file(output: str, filename: str, variable: str, sample_size: Optional[int]) -> Path:
    """Save clustering results to a text file."""
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")

    output_filename = f"cluster_results_v3_{base_name}_{variable}_{sample_str}_{date_str}.txt"
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        clusterer, embeddings_models = main()
    finally:
        # Restore stdout
        sys.stdout = tee.original_stdout

    # Save results to file
    output_path = save_results_to_file(
        output=tee.get_output(),
        filename=FILENAME,
        variable=VARIABLE,
        sample_size=SAMPLE_SIZE
    )
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")

# %%
