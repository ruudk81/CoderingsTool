#%%

"""
Step 5: Clusterer Experiment Runner

Runs the clustering step in isolation for experimentation.
Loads Step 4 (embeddings) results from cache and performs clustering.

Usage:
    cd src && python -m development.step_5_clusterer.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental clusterer from this folder
    USE_EXPERIMENTAL = False -> Uses production clusterer from utils/
"""

USE_EXPERIMENTAL = True

import sys
import time
from pathlib import Path

import numpy as np

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
data_dir = project_root / "data"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataclasses import dataclass
from typing import Optional

# =============================================================================
# SHARED IMPORTS (from production)
# =============================================================================
from development import models_exp as models
from config import CacheConfig, ModelConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.llm import token_tracker
from utils import dataLoader

# Import centralized test data config
try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
@dataclass
class ExperimentConfig:
    # Data config from centralized test_data.py
    filename: str = TEST_DATA.filename
    id_column: str = TEST_DATA.id_column
    var_name: str = TEST_DATA.var_name
    sample_size: Optional[int] = TEST_DATA.sample_size
    # Experiment-specific settings
    use_experimental: bool = USE_EXPERIMENTAL
    verbose: bool = True
    force_recalc: bool = True


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .clusterer_exp import (
            Clusterer, ClusterLabelModel, ClusterRepresentationModel,
            ClusterRepresentationsModel,
        )
        from .config_clusterer_exp import ClustererConfig
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from clusterer_exp import (
            Clusterer, ClusterLabelModel, ClusterRepresentationModel,
            ClusterRepresentationsModel,
        )
        from config_clusterer_exp import ClustererConfig
    print("[EXPERIMENTAL] Using clusterer_exp.py from development folder")
else:
    from utils.clusterer import Clusterer
    from config_steps.config_clusterer import ClustererConfig
    print("[PRODUCTION] Using clusterer.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step4_cache(config: ExperimentConfig):
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "embeddings"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=4 first."
        )

    data = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, models.EmbeddingsModel
    )

    # Also load extraction metadata if available
    extraction_metadata = None
    try:
        extraction_metadata = cache_manager.load_metadata_from_cache(
            config.filename, "extracted_ideas", variable_key, models.ExtractionMetadata
        )
    except:
        pass

    return data, variable_key, cache_manager, extraction_metadata


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    embedded_text, variable_key, cache_manager, extraction_metadata = load_step4_cache(config)
    var_lab = get_var_lab(config)

    verbose_reporter = VerboseReporter(config.verbose)

    verbose_reporter.section_header("CLUSTERING EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")

    total_ideas = sum(item.idea_count for item in embedded_text)
    verbose_reporter.stat_line(f"Input: {len(embedded_text)} responses with {total_ideas} ideas")

    if extraction_metadata:
        prefix_display = extraction_metadata.template_prefix[:40] + "..." if extraction_metadata.template_prefix and len(extraction_metadata.template_prefix) > 40 else extraction_metadata.template_prefix or "(none)"
        verbose_reporter.stat_line(f"Extraction metadata loaded (template_prefix: '{prefix_display}')")

    start_time = time.time()

    # Initialize clusterer with config
    # Defaults: algorithm_mode="hdbscan", enable_iterative=True,
    #           iterative_accept_probability=0.8, clustering_embedding_field="ladder_embedding"
    clusterer_config = ClustererConfig(
        verbose=config.verbose,
        # Text sources: single field or composite with "+" (e.g., "idea+rung_2")
        # Supported: "idea", "instance", "rung_1", "rung_2", "concept_type", "ladder"
        keyword_text_source="idea",                         # text for c-TF-IDF / MMR keyword extraction
        label_text_source="idea+rung_1+rung_2",             # text for representative samples in LLM prompt
        verbose_text_source="idea+rung_1+rung_2",           # text for print_all_clusters() display
        text_separator=" → ",                               # separator for composite "+" fields
    )
    verbose_reporter.stat_line(f"Algorithm mode: {clusterer_config.algorithm_mode}")

    clusterer = Clusterer(embedded_text, config=clusterer_config, extraction_metadata=extraction_metadata)
    clusterer.run()

    # Convert to ClusterModel list
    cluster_results = clusterer.to_cluster_model()

    elapsed_time = time.time() - start_time

    # =========================================================================
    # CACHE RESULTS (3 layers, matching pipeline.py step 5)
    # =========================================================================

    # Layer 1: Primary cluster assignments
    cache_manager.save_to_cache(
        cluster_results, config.filename, "initial_clusters",
        variable_key, elapsed_time, var_lab=var_lab
    )

    # Layer 2: Clustering metadata (keywords, labels, distributions, metrics)
    clustering_metadata = clusterer.to_metadata_model()
    cache_manager.save_to_cache(
        [clustering_metadata], config.filename, "clustering_metadata",
        variable_key, elapsed_time, var_lab=var_lab
    )

    # Layer 3: Cluster representations (for step 6 speculative codes)
    keywords = clusterer.get_cluster_keywords() or {}
    labels = clusterer.get_cluster_labels() or {}

    if keywords or labels:
        representations = []
        all_cluster_ids = set(keywords.keys()) | set(labels.keys())

        for cluster_id in sorted(all_cluster_ids):
            llm_label = None
            if cluster_id in labels:
                label = labels[cluster_id]
                llm_label = ClusterLabelModel(
                    cluster_id=label.cluster_id,
                    theme=label.theme,
                    description=label.description,
                    key_concepts=label.key_concepts,
                    n_ideas=label.n_ideas
                )
            rep = ClusterRepresentationModel(
                cluster_id=cluster_id,
                keywords=keywords.get(cluster_id, []),
                llm_label=llm_label
            )
            representations.append(rep)

        algorithm_rec = clusterer.get_algorithm_recommendation()
        metrics = clusterer.get_metrics()

        representations_model = ClusterRepresentationsModel(
            representations=representations,
            generation_metadata={
                "algorithm": algorithm_rec.recommended_algorithm if algorithm_rec else "unknown",
                "dvc_value": algorithm_rec.dvc_value if algorithm_rec else None,
                "n_clusters": metrics.n_clusters if metrics else len(all_cluster_ids),
                "noise_rate": metrics.noise_rate if metrics else None,
                "mean_coherence": metrics.mean_coherence if metrics else None,
            }
        )
        cache_manager.save_to_cache(
            [representations_model], config.filename,
            "cluster_representations", variable_key, elapsed_time, var_lab=var_lab
        )

    verbose_reporter.stat_line(f"Output: {len(cluster_results)} results cached")
    print(f"\n'Clustering experiment' completed in {elapsed_time:.2f} seconds.\n")

    return cluster_results, clusterer


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    config = EXPERIMENT_CONFIG
    var_lab = get_var_lab(config)

    verbose_capture = VerboseCapture(
        filename=config.filename,
        variable_key=config.var_name,
        sample_size=config.sample_size,
        run_until_step=5
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 5 - Clusterer")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print("=" * 70)

    try:
        results, clusterer = run_experiment(config)

        # Print detailed summary
        rec = clusterer.get_algorithm_recommendation()
        if rec:
            print(f"\nAlgorithm Recommendation:")
            print(f"  Recommended: {rec.recommended_algorithm} ({rec.confidence} confidence)")
            print(f"  DVC: {rec.dvc_value:.3f} -> {rec.dvc_recommendation}")
            print(f"  Knee: y_diff={rec.y_difference:.2f}, sharp={rec.has_sharp_knee}")
            if rec.is_forced:
                print(f"  FORCED: Algorithm selection was forced by hard DVC rule")
            print(f"  Reasoning: {rec.reasoning}")

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

        # Keywords summary
        all_keywords = clusterer.get_all_cluster_keywords()
        if all_keywords:
            for method_name in ["ctfidf", "mmr", "tfidf"]:
                method_keywords = all_keywords.get(method_name)
                if method_keywords:
                    method_label = {"ctfidf": "c-TF-IDF", "mmr": "MMR", "tfidf": "TF-IDF"}.get(method_name, method_name)
                    print(f"\n{method_label} Keywords ({len(method_keywords)} clusters):")
                    for cluster_id in sorted(method_keywords.keys()):
                        kw_list = method_keywords[cluster_id]
                        kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
                        print(f"  Cluster {cluster_id}: {kw_str}")

        # Print all clusters with samples
        clusterer.print_all_clusters(n_samples=10)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)

# %%
