"""
Clusterer V3 - Map-Reduce MECE per Cluster

EXPERIMENTAL VERSION (step_5_clusterer_v3)
Changes here do NOT affect the production pipeline.

Based on: src/experiments/step_5_clusterer_v2/clusterer_exp.py (V2)

V3 changes from V2:
- Phase 6 (keyword extraction): REMOVED
- Phase 7: Map-Reduce MECE per cluster (replaces V2's single-theme + cross-cluster MECE)
  - MAP: batch all ideas, find ALL atomic themes per batch
  - REDUCE: consolidate themes across batches
  - MECE: apply inclusion/exclusion boundaries
- Phase 8 (cross-cluster MECE): REMOVED

Original: src/utils/clusterer.py

Pipeline Integration:
- Input: List[EmbeddingsModel] from step 4 (embeddings)
- Output: List[ClusterModel] via to_cluster_model()
- Cache step: "initial_clusters"

Usage (experimental):
    from .clusterer_exp import Clusterer
    from .config_clusterer_exp import ClustererConfig

    config = ClustererConfig()
    clusterer = Clusterer(embeddings_list, config=config)
    clusterer.run()

    cluster_results = clusterer.to_cluster_model()
    mece_results = clusterer.get_cluster_mece_results()
"""

import re
import warnings
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import umap

from experiments import models_exp as models
from .config_clusterer_exp import ClustererConfig, resolve_embedding_source
from .clusterer_helpers_exp import (
    # Preprocessing
    preprocess_embeddings, l2_normalize,
    resolve_embeddings, resolve_embeddings_unique,
    get_idea_field_text, EMBEDDING_FIELD_MAP,
    # Algorithm Selection
    AlgorithmSelector, AlgorithmRecommendation,
    # Parameter Optimization
    ParameterOptimizer, run_umap, n_neighbors_grid, mcs_grid, ms_grid,
    # Quality Metrics
    ClusterQualityMetrics, ClusteringMetrics, calculate_coherence_score,
    # Post-Processing
    merge_similar_clusters, recluster_noise, reduce_noise_by_embedding_similarity,
)

# Suppress common warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
warnings.filterwarnings("ignore", message="overflow encountered in power", module="hdbscan.validity")
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor.providers.gemini")


class Clusterer:
    """
    Enhanced clustering module with automatic algorithm selection,
    Optuna-based optimization, and per-cluster Map-Reduce MECE topics.

    Key Features:
    1. Automatic algorithm selection (DVC + kNN knee + persistence)
    2. Bayesian HDBSCAN optimization via Optuna GridSampler
    3. Coherence-based quality metrics on original embeddings
    4. Per-cluster Map-Reduce MECE topic extraction (V3)

    Usage:
        # Basic usage (auto mode)
        clusterer = Clusterer(input_list, config=ClustererConfig())
        clusterer.run()
        results = clusterer.to_cluster_model()
        mece = clusterer.get_cluster_mece_results()
    """

    def __init__(
        self,
        input_list: List[models.EmbeddingsModel],
        config: Optional[ClustererConfig] = None,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
    ):
        """
        Initialize Clusterer.

        Args:
            input_list: List of EmbeddingsModel with embeddings populated
            config: Configuration (uses defaults if None)
            extraction_metadata: Optional ExtractionMetadata for taxonomy context
        """
        self.config = config or ClustererConfig()
        self._input_list = input_list
        self._extraction_metadata = extraction_metadata
        self._verbose = self.config.verbose

        # Will be populated during run()
        self._embeddings_original: Optional[np.ndarray] = None
        self._embeddings_processed: Optional[np.ndarray] = None
        self._idea_texts: Optional[List[str]] = None
        self._ontology_texts: Optional[List[str]] = None
        self._idea_indices: Optional[List[Tuple[int, int]]] = None
        self._template_prefix: Optional[str] = None
        self._embedding_text_format: Optional[str] = None
        self._keyword_text_format: Optional[str] = None
        self._labels: Optional[np.ndarray] = None
        self._umap_embeddings: Optional[np.ndarray] = None
        self._hdbscan_model: Optional[hdbscan.HDBSCAN] = None
        self._recommendation: Optional[AlgorithmRecommendation] = None
        self._metrics: Optional[ClusteringMetrics] = None
        self._cluster_mece_results = None  # Dict[int, ClusterMECEResult] from Phase 7
        self._algorithm_used: str = ""
        self._algorithm_params: Dict[str, Any] = {}
        self._optimizer: Optional[ParameterOptimizer] = None

        # On-the-fly unique-point clustering state (populated when embedding_source
        # is on-the-fly like "category" or "root" — cluster unique values, then
        # expand labels back to all ideas before MECE phase)
        self._is_unique_point_clustering: bool = False
        self._all_idea_indices: Optional[List[Tuple[int, int]]] = None
        self._all_idea_texts: Optional[List[str]] = None
        self._all_N: Optional[int] = None
        self._idea_to_unique_idx: Optional[List[int]] = None
        self._expanded_probabilities: Optional[np.ndarray] = None

        # Components
        self._selector = AlgorithmSelector(self.config)
        self._metrics_calculator = ClusterQualityMetrics(self.config)

    def run(self) -> 'Clusterer':
        """
        Execute the complete clustering pipeline.

        Returns:
            self (for method chaining)
        """
        if self._verbose:
            print("=" * 70)
            print("Clustering Pipeline V3 (Map-Reduce MECE)")
            print("=" * 70)

        # Phase 1: Preprocessing
        self._run_preprocessing()

        # Small dataset path: skip DVC/knee analysis, use Agglomerative directly
        if self._N <= self.config.small_dataset_threshold:
            if self._verbose:
                print(f"\n[Phase 2] Skipping algorithm selection (n={self._N} <= {self.config.small_dataset_threshold})")
            self._run_agglomerative_small()
        else:
            # Phase 2: Algorithm Selection (always compute for diagnostics)
            self._run_algorithm_selection()

            # Phase 3: Clustering
            if self.config.algorithm_mode == "auto":
                algorithm = self._map_recommendation_to_algorithm()
            else:
                algorithm = self.config.algorithm_mode

            if algorithm == "hdbscan":
                self._run_hdbscan_optimized()
            elif algorithm == "agglomerative":
                self._run_agglomerative()
            elif algorithm == "kmeans":
                self._run_kmeans()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        # Phase 4: Post-processing
        self._run_post_processing()

        # Phase 5: Final metrics
        self._compute_final_metrics()

        # Phase 6: REMOVED (no keyword extraction in V3)

        # Expand labels from unique points to all ideas (if on-the-fly clustering)
        if self._is_unique_point_clustering:
            self._expand_labels_to_ideas()

        # Phase 7: Map-Reduce MECE per cluster
        if self.config.generate_mece_topics:
            self._run_map_reduce_mece()

        # Phase 8: REMOVED (no cross-cluster consolidation in V3)

        return self

    def _run_preprocessing(self):
        """Phase 1: Extract, normalize, and optionally PCA embeddings."""
        if self._verbose:
            print("\n[Phase 1] Preprocessing")

        is_on_the_fly = self.config.embedding_source in EMBEDDING_FIELD_MAP and \
            EMBEDDING_FIELD_MAP.get(self.config.embedding_source) is None

        # Always run base preprocessing (extracts idea_texts, idea_indices, etc.)
        # Use idea_embedding as base when doing on-the-fly embedding
        if is_on_the_fly:
            import copy
            preprocess_config = copy.copy(self.config)
            preprocess_config.embedding_source = "idea_embedding"
        else:
            preprocess_config = self.config

        (
            self._embeddings_original,
            self._embeddings_processed,
            self._idea_texts,
            self._ontology_texts,
            self._idea_indices,
            _,
            self._template_prefix,
            self._embedding_text_format
        ) = preprocess_embeddings(self._input_list, preprocess_config)

        self._N = len(self._embeddings_original)

        # Override embeddings if source needs on-the-fly embedding
        # V2 pattern: cluster UNIQUE values only (e.g. 312 unique categories),
        # then expand labels back to all ideas before MECE phase
        if is_on_the_fly:
            # Save original state for later expansion
            self._is_unique_point_clustering = True
            self._all_idea_indices = self._idea_indices
            self._all_idea_texts = self._idea_texts
            self._all_N = self._N

            # Embed unique values only (e.g. 312 unique categories, not 1795 ideas)
            unique_embs, unique_texts, self._idea_to_unique_idx = \
                resolve_embeddings_unique(
                    self._input_list, self._idea_indices,
                    self.config.embedding_source, verbose=self._verbose,
                )
            self._embeddings_original = l2_normalize(unique_embs)
            self._embeddings_processed = self._embeddings_original
            self._N = len(unique_embs)
            self._idea_texts = unique_texts  # for potential keyword phases

        if self._verbose:
            if is_on_the_fly:
                print(f"  Clustering {self._N} unique '{self.config.embedding_source}' points "
                      f"(from {self._all_N} ideas)")
            else:
                print(f"  {self._N} ideas, embedding source: {self.config.embedding_source}")

    def _run_algorithm_selection(self):
        """Phase 2: Compute DVC, knee detection, and recommendation."""
        if self._verbose:
            print("\n[Phase 2] Algorithm Selection Analysis")

        # Compute DVC on original embeddings
        dvc_result = self._selector.compute_dvc(self._embeddings_original)
        if self._verbose:
            dvc_val = dvc_result['dvc']
            if not np.isnan(dvc_val):
                print(f"  DVC = {dvc_val:.3f} → {dvc_result['recommendation']}")

        # Check hard DVC rule first
        force_threshold = getattr(self.config, 'force_agglomerative_below_dvc', 0.25)
        if not np.isnan(dvc_result['dvc']) and dvc_result['dvc'] < force_threshold:
            if self._verbose:
                print(f"  HARD RULE: DVC < {force_threshold} → Forcing Agglomerative")

            knee_result = {
                'K': None,
                'y_difference': 0.0,
                'has_sharp_knee': False,
                'recommendation': 'AGGLOMERATIVE_FORCED'
            }
        else:
            # Run trial UMAP for knee detection
            n_neighbors_list = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k)
            nn_idx = len(n_neighbors_list) // 2 if self.config.trial_umap_nn_index == -1 else self.config.trial_umap_nn_index
            trial_n_neighbors = n_neighbors_list[nn_idx]

            trial_n_components = self.config.umap_n_components_grid[0]
            trial_umap = run_umap(
                self._embeddings_processed,
                trial_n_neighbors,
                trial_n_components,
                self.config.umap_min_dist,
                self.config.umap_random_state
            )
            trial_umap_normalized = l2_normalize(trial_umap)

            knee_result = self._selector.detect_knee(trial_umap_normalized)
            if self._verbose:
                k_str = f"K={knee_result['K']}" if knee_result['K'] else "No knee"
                print(f"  Knee: {k_str}, y_diff={knee_result['y_difference']:.2f}, "
                      f"sharp={knee_result['has_sharp_knee']}")

        self._recommendation = self._selector.recommend(dvc_result, knee_result)

        if self._verbose:
            print(f"  Recommendation: {self._recommendation.combined_recommendation} "
                  f"({self._recommendation.confidence} confidence)")
            print(f"  Reasoning: {self._recommendation.reasoning}")

    def _map_recommendation_to_algorithm(self) -> str:
        """Map combined recommendation to algorithm name."""
        if self.config.force_hdbscan:
            return "hdbscan"
        rec = self._recommendation.combined_recommendation
        if "HDBSCAN" in rec:
            return "hdbscan"
        elif rec == "AGGLOMERATIVE_OR_KMEANS":
            return "agglomerative"
        else:
            return "agglomerative"

    def _run_hdbscan_optimized(self):
        """Phase 3a: Run Optuna-optimized HDBSCAN."""
        if self._verbose:
            print("\n[Phase 3] HDBSCAN Optimization (Optuna)")

        self._optimizer = ParameterOptimizer(
            self.config,
            self._embeddings_processed,
            self._embeddings_original,
            verbose=self._verbose
        )

        result = self._optimizer.optimize()

        self._labels = result.best_labels.copy()
        self._umap_embeddings = result.umap_embeddings
        self._hdbscan_model = result.best_model
        self._algorithm_used = "HDBSCAN"
        self._algorithm_params = result.best_params

        if self._verbose:
            dvc_reduced = self._selector.compute_dvc(result.umap_embeddings)
            dvc_val = dvc_reduced['dvc']
            if not np.isnan(dvc_val):
                print(f"  DVC (UMAP-reduced) = {dvc_val:.3f} (mean_dk={dvc_reduced['mean_dk']:.4f}, std_dk={dvc_reduced['std_dk']:.4f})")

    def _run_agglomerative(self):
        """Phase 3b: Run Agglomerative clustering."""
        if self._verbose:
            print("\n[Phase 3] Agglomerative Clustering")

        n_neighbors = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k)[self.config.agglomerative_nn_index]
        n_components = self.config.umap_n_components_grid[0]

        if self._verbose:
            print(f"  UMAP: n_neighbors={n_neighbors}, n_components={n_components}")

        self._umap_embeddings = run_umap(
            self._embeddings_processed,
            n_neighbors,
            n_components,
            self.config.umap_min_dist,
            self.config.umap_random_state
        )
        self._umap_embeddings = l2_normalize(self._umap_embeddings)

        sqrt_n = int(np.sqrt(self._N))
        k_grid = sorted(set([
            max(2, int(m * sqrt_n))
            for m in self.config.k_grid_multipliers
        ]))

        if self._verbose:
            print(f"  K grid: {k_grid}")

        best_k = k_grid[0]
        best_coherence = -1.0
        best_labels = None

        if self._verbose:
            print(f"  {'k':>3} | {'coherence':>9} | {'silhouette':>10}")
            print(f"  {'-'*3}-+-{'-'*9}-+-{'-'*10}")

        for k in k_grid:
            if k >= len(self._umap_embeddings):
                continue

            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='euclidean',
                linkage=self.config.agglomerative_linkage
            )
            labels = clusterer.fit_predict(self._umap_embeddings)

            if len(set(labels)) > 1:
                coherence = calculate_coherence_score(labels, self._embeddings_original)
                sil = silhouette_score(self._umap_embeddings, labels)
                if self._verbose:
                    print(f"  {k:>3} | {coherence:>9.3f} | {sil:>10.3f}")
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_k = k
                    best_labels = labels.copy()

        if best_labels is None:
            clusterer = AgglomerativeClustering(n_clusters=k_grid[0])
            best_labels = clusterer.fit_predict(self._umap_embeddings)
            best_k = k_grid[0]

        self._labels = best_labels
        self._algorithm_used = "Agglomerative"
        self._algorithm_params = {'n_clusters': best_k, 'linkage': self.config.agglomerative_linkage}

        if self._verbose:
            print(f"  Best: k={best_k}, coherence={best_coherence:.3f}")

    def _run_agglomerative_small(self):
        """Run Agglomerative clustering for small datasets (n <= small_dataset_threshold)."""
        if self._verbose:
            print("\n[Phase 3] Agglomerative Clustering (Small Dataset Path)")
            print(f"  Dataset size: {self._N} (threshold: {self.config.small_dataset_threshold})")
            print("  Skipping UMAP - using L2-normalized embeddings directly")

        embeddings = l2_normalize(self._embeddings_original)

        sqrt_n = np.sqrt(self._N)
        k_min = max(2, int(0.5 * sqrt_n))
        k_max = int(2 * sqrt_n)
        k_grid = list(range(k_min, k_max + 1))

        if self._verbose:
            print(f"  K grid: {k_grid} (0.5×sqrt({self._N})={k_min}, 2×sqrt({self._N})={k_max})")

        results = []

        if self._verbose:
            print(f"  {'k':>3} | {'sil':>6} | {'CH':>8} | {'DB':>6} | {'coh':>5}")
            print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}")

        for k in k_grid:
            if k >= len(embeddings):
                continue

            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='euclidean',
                linkage=self.config.agglomerative_linkage
            )
            labels = clusterer.fit_predict(embeddings)

            if len(set(labels)) > 1:
                sil = silhouette_score(embeddings, labels)
                ch = calinski_harabasz_score(embeddings, labels)
                db = davies_bouldin_score(embeddings, labels)

                coherences = []
                for cluster_id in set(labels):
                    if cluster_id < 0:
                        continue
                    mask = labels == cluster_id
                    cluster_emb = self._embeddings_original[mask]
                    if len(cluster_emb) > 1:
                        cluster_emb_norm = cluster_emb / np.linalg.norm(cluster_emb, axis=1, keepdims=True)
                        sim_matrix = cluster_emb_norm @ cluster_emb_norm.T
                        n_pts = len(cluster_emb)
                        mean_sim = (sim_matrix.sum() - n_pts) / (n_pts * (n_pts - 1))
                        coherences.append(mean_sim)
                coh = np.mean(coherences) if coherences else 0.0

                if self._verbose:
                    print(f"  {k:>3} | {sil:>6.3f} | {ch:>8.1f} | {db:>6.3f} | {coh:>5.3f}")

                results.append((k, labels.copy(), sil, ch, db, coh))

        if not results:
            clusterer = AgglomerativeClustering(n_clusters=k_grid[0])
            best_labels = clusterer.fit_predict(embeddings)
            best_k = k_grid[0]
        else:
            best_k, best_labels, selection_reason = self._select_k_by_bootstrap_ci(results)

        self._labels = best_labels
        self._umap_embeddings = embeddings
        self._algorithm_used = "Agglomerative"
        self._algorithm_params = {
            'n_clusters': best_k,
            'linkage': self.config.agglomerative_linkage,
            'small_dataset_path': True
        }

        if self._verbose:
            print(f"  Best: k={best_k} ({selection_reason})")

    def _select_k_by_bootstrap_ci(self, results: list) -> tuple:
        """Select optimal K using bootstrap confidence intervals."""
        import time
        start_time = time.time()

        if len(results) < 2:
            k, labels, sil, ch, db, coh = results[0]
            return k, labels, f"coherence={coh:.3f} (only 1 result)"

        k_values = [r[0] for r in results]
        all_labels = {r[0]: r[1] for r in results}

        n_bootstrap = self.config.agglomerative_bootstrap_iterations
        confidence_level = self.config.agglomerative_bootstrap_confidence
        alpha = 1 - confidence_level

        ci_lower = {}
        ci_upper = {}
        point_estimates = {}

        n_samples = len(self._embeddings_original)

        for k, labels, sil, ch, db, coh in results:
            point_estimates[k] = coh

            bootstrap_coherences = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                resampled_labels = labels[indices]
                resampled_embeddings = self._embeddings_original[indices]

                coherences = []
                for cluster_id in set(resampled_labels):
                    if cluster_id < 0:
                        continue
                    mask = resampled_labels == cluster_id
                    cluster_emb = resampled_embeddings[mask]
                    if len(cluster_emb) > 1:
                        cluster_emb_norm = cluster_emb / np.linalg.norm(cluster_emb, axis=1, keepdims=True)
                        sim_matrix = cluster_emb_norm @ cluster_emb_norm.T
                        n_pts = len(cluster_emb)
                        mean_sim = (sim_matrix.sum() - n_pts) / (n_pts * (n_pts - 1))
                        coherences.append(mean_sim)

                if coherences:
                    bootstrap_coherences.append(np.mean(coherences))

            if bootstrap_coherences:
                ci_lower[k] = np.percentile(bootstrap_coherences, 100 * alpha / 2)
                ci_upper[k] = np.percentile(bootstrap_coherences, 100 * (1 - alpha / 2))
            else:
                ci_lower[k] = coh
                ci_upper[k] = coh

        best_k_max = max(k_values, key=lambda k: point_estimates[k])
        best_ci_lower = ci_lower[best_k_max]

        selected_k = best_k_max
        for k in sorted(k_values):
            if ci_upper[k] >= best_ci_lower:
                selected_k = k
                break

        elapsed = time.time() - start_time

        selected_labels = all_labels[selected_k]
        selected_coh = point_estimates[selected_k]
        best_coh = point_estimates[best_k_max]

        if self._verbose:
            print(f"  Bootstrap CIs ({n_bootstrap} iterations, {elapsed:.2f}s):")
            for k in k_values:
                marker = " *" if k == selected_k else ""
                print(f"    k={k}: coh={point_estimates[k]:.3f} CI=[{ci_lower[k]:.3f}, {ci_upper[k]:.3f}]{marker}")

        reason = f"bootstrap CI overlap at k={selected_k} (coh={selected_coh:.3f}, best={best_coh:.3f} at k={best_k_max})"

        return selected_k, selected_labels, reason

    def _run_kmeans(self):
        """Phase 3c: Run K-means clustering."""
        if self._verbose:
            print("\n[Phase 3] K-means Clustering")

        n_neighbors = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k)[self.config.kmeans_nn_index]
        n_components = self.config.umap_n_components_grid[0]

        if self._verbose:
            print(f"  UMAP: n_neighbors={n_neighbors}, n_components={n_components}")

        self._umap_embeddings = run_umap(
            self._embeddings_processed,
            n_neighbors,
            n_components,
            self.config.umap_min_dist,
            self.config.umap_random_state
        )
        self._umap_embeddings = l2_normalize(self._umap_embeddings)

        sqrt_n = int(np.sqrt(self._N))
        k_grid = sorted(set([
            max(2, int(m * sqrt_n))
            for m in self.config.k_grid_multipliers
        ]))

        if self._verbose:
            print(f"  K grid: {k_grid}")

        best_k = k_grid[0]
        best_sil = -1.0
        best_labels = None

        for k in k_grid:
            if k >= len(self._umap_embeddings):
                continue

            clusterer = KMeans(n_clusters=k, random_state=self.config.kmeans_random_state, n_init=self.config.kmeans_n_init)
            labels = clusterer.fit_predict(self._umap_embeddings)

            if len(set(labels)) > 1:
                sil = silhouette_score(self._umap_embeddings, labels)
                if self._verbose:
                    print(f"    k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels.copy()

        if best_labels is None:
            clusterer = KMeans(n_clusters=k_grid[0], random_state=self.config.kmeans_random_state, n_init=self.config.kmeans_n_init)
            best_labels = clusterer.fit_predict(self._umap_embeddings)
            best_k = k_grid[0]

        self._labels = best_labels
        self._algorithm_used = "K-means"
        self._algorithm_params = {'n_clusters': best_k}

        if self._verbose:
            print(f"  Best: k={best_k}, silhouette={best_sil:.3f}")

    def _run_post_processing(self):
        """Phase 4: Cluster merging and noise reduction."""
        if self._verbose:
            print("\n[Phase 4] Post-processing")

        if self.config.enable_merging:
            self._labels = merge_similar_clusters(
                self._labels,
                self._embeddings_original,
                self.config,
                verbose=self._verbose
            )

        if self._algorithm_used == "HDBSCAN":
            strategy = self.config.noise_reduction_strategy

            if strategy == "embeddings":
                self._labels, noise_stats = reduce_noise_by_embedding_similarity(
                    self._labels,
                    self._embeddings_original,
                    threshold=self.config.noise_reduction_threshold,
                    verbose=self._verbose
                )
            elif strategy == "hdbscan" and self.config.enable_noise_reclustering:
                self._labels = recluster_noise(
                    self._labels,
                    self._umap_embeddings,
                    self._embeddings_original,
                    self.config,
                    verbose=self._verbose
                )

    def _compute_final_metrics(self):
        """Phase 5: Calculate final quality metrics."""
        if self._verbose:
            print("\n[Phase 5] Final Metrics")

        self._metrics = self._metrics_calculator.calculate_all(
            self._labels,
            self._umap_embeddings,
            self._embeddings_original,
            hdbscan_model=self._hdbscan_model if self._algorithm_used == "HDBSCAN" else None,
            algorithm_used=self._algorithm_used,
            algorithm_params=self._algorithm_params
        )

        if self._verbose:
            print(f"  Clusters: {self._metrics.n_clusters}")
            print(f"  Noise rate: {self._metrics.noise_rate:.1%}")
            print(f"  Coherence: {self._metrics.mean_coherence:.3f}")
            print(f"  Breakdown: {self._metrics.coherence_breakdown}")
            if self._metrics.mean_persistence is not None:
                print(f"  Persistence: mean={self._metrics.mean_persistence:.3f}, "
                      f"weighted={self._metrics.weighted_persistence:.3f}")

    # =========================================================================
    # LABEL EXPANSION (unique points → all ideas)
    # =========================================================================

    def _expand_labels_to_ideas(self):
        """Expand cluster labels from unique embedding points to all ideas.

        After on-the-fly clustering of N_unique points (e.g. 312 categories),
        maps each of the N_total ideas (e.g. 1795) to the cluster of its
        unique embedding point. Restores original idea state for MECE phase.
        """
        unique_labels = self._labels  # length N_unique

        # Expand: each idea gets the label of its unique embedding point
        expanded_labels = np.array([
            unique_labels[self._idea_to_unique_idx[i]]
            for i in range(self._all_N)
        ])

        # Expand HDBSCAN probabilities if available
        self._expanded_probabilities = None
        if self._hdbscan_model is not None and hasattr(self._hdbscan_model, 'probabilities_'):
            unique_probs = self._hdbscan_model.probabilities_
            self._expanded_probabilities = np.array([
                unique_probs[self._idea_to_unique_idx[i]]
                for i in range(self._all_N)
            ])

        # Restore original state
        self._labels = expanded_labels
        self._idea_indices = self._all_idea_indices
        self._idea_texts = self._all_idea_texts
        self._N = self._all_N

        if self._verbose:
            n_unique = len(unique_labels)
            n_clusters = len(set(unique_labels[unique_labels >= 0]))
            n_noise_ideas = int(np.sum(expanded_labels == -1))
            print(f"\n  Label expansion: {n_unique} unique points → {self._N} ideas "
                  f"({n_clusters} clusters, {n_noise_ideas} noise ideas)")

    # =========================================================================
    # PHASE 7: MAP-REDUCE MECE (V3)
    # =========================================================================

    def _get_idea_text(self, flat_idx: int, source: str) -> str:
        """Get text for an idea by flat index and field name.

        Args:
            flat_idx: Index into the flat arrays (idea_texts, labels, etc.)
            source: Field name - "idea", "ontology", "instance", "node",
                    "category", "root"
        """
        resp_idx, idea_idx = self._idea_indices[flat_idx]
        idea = self._input_list[resp_idx].response_ideas[idea_idx]
        return get_idea_field_text(idea, source)

    def _run_map_reduce_mece(self):
        """Phase 7: Per-cluster Map-Reduce MECE topic extraction."""
        if self._verbose:
            print("\n[Phase 7] Map-Reduce MECE Topic Extraction")
            print(f"  Text source: {self.config.mapreduce_text_source}")

        # Build cluster_texts dict (all ideas per cluster)
        text_source = self.config.mapreduce_text_source
        cluster_texts = {}
        for i, label in enumerate(self._labels):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(self._get_idea_text(i, text_source))

        # Extract survey question
        survey_question = ""
        if self._extraction_metadata and hasattr(self._extraction_metadata, 'var_lab'):
            survey_question = self._extraction_metadata.var_lab or ""

        # Extract dataset context
        dataset_context = None
        if self._extraction_metadata:
            dataset_context = {
                'domain': getattr(self._extraction_metadata, 'domain', '') or '',
                'topic': getattr(self._extraction_metadata, 'topic', '') or '',
                'entity': getattr(self._extraction_metadata, 'entity', '') or '',
                'perspective': getattr(self._extraction_metadata, 'perspective', '') or '',
                'intent': getattr(self._extraction_metadata, 'intent', '') or '',
            }

        # Extract taxonomy context
        taxonomy_axis = None
        taxonomy_description = None
        if self._extraction_metadata:
            taxonomy_axis = getattr(self._extraction_metadata, 'taxonomy_axis', None)
            taxonomy_description = getattr(self._extraction_metadata, 'taxonomy_axis_description', None)

        # Run the pipeline
        from .map_reduce_mece import MapReduceMECE
        mece_processor = MapReduceMECE(self.config)
        self._cluster_mece_results = mece_processor.process_all_clusters(
            cluster_texts=cluster_texts,
            survey_question=survey_question,
            language="Dutch",
            dataset_context=dataset_context,
            taxonomy_axis=taxonomy_axis,
            taxonomy_description=taxonomy_description,
            verbose=self._verbose,
        )

        if self._verbose:
            total_topics = sum(len(r.topics) for r in self._cluster_mece_results.values())
            print(f"\n  Total: {total_topics} MECE topics across {len(self._cluster_mece_results)} clusters")

    # =========================================================================
    # OUTPUT: to_cluster_model
    # =========================================================================

    def to_cluster_model(self) -> List[models.ClusterModel]:
        """Convert internal results to ClusterModel list (pipeline-compatible)."""
        if self._labels is None:
            raise RuntimeError("Must call run() before to_cluster_model()")

        response_results = {}
        for idx, (resp_idx, idea_idx) in enumerate(self._idea_indices):
            if resp_idx not in response_results:
                response_results[resp_idx] = {}

            cluster_id = int(self._labels[idx])
            probability = None
            if cluster_id != -1:
                if self._expanded_probabilities is not None:
                    probability = float(self._expanded_probabilities[idx])
                elif self._hdbscan_model is not None:
                    probability = float(self._hdbscan_model.probabilities_[idx])

            response_results[resp_idx][idea_idx] = (cluster_id, probability)

        output_list = []
        for resp_idx, response in enumerate(self._input_list):
            cluster_data = response.model_dump()

            if cluster_data.get('response_ideas'):
                for idea_idx, idea in enumerate(cluster_data['response_ideas']):
                    if resp_idx in response_results and idea_idx in response_results[resp_idx]:
                        cluster_id, probability = response_results[resp_idx][idea_idx]
                        idea['initial_cluster'] = cluster_id
                        idea['cluster_probability'] = probability
                    else:
                        idea['initial_cluster'] = -1
                        idea['cluster_probability'] = None

            output_list.append(models.ClusterModel.model_validate(cluster_data))

        return output_list

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_cluster_mece_results(self):
        """Get per-cluster MECE topic results (Phase 7 output)."""
        return self._cluster_mece_results

    def get_metrics(self) -> ClusteringMetrics:
        """Get comprehensive clustering quality metrics."""
        if self._metrics is None:
            raise RuntimeError("Must call run() before get_metrics()")
        return self._metrics

    def get_algorithm_recommendation(self) -> Optional[AlgorithmRecommendation]:
        """Get algorithm selection details."""
        return self._recommendation

    @property
    def labels_(self) -> np.ndarray:
        """Cluster labels for all ideas (-1 for noise)."""
        if self._labels is None:
            raise RuntimeError("Must call run() before accessing labels_")
        return self._labels

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found (excluding noise)."""
        if self._labels is None:
            raise RuntimeError("Must call run() before accessing n_clusters_")
        return len(set(self._labels)) - (1 if -1 in self._labels else 0)

    @property
    def noise_rate_(self) -> float:
        """Fraction of ideas labeled as noise."""
        if self._labels is None:
            raise RuntimeError("Must call run() before accessing noise_rate_")
        return (self._labels == -1).sum() / len(self._labels)

    def get_hdbscan_artifacts(self) -> Dict[str, Any]:
        """Return HDBSCAN tree structures for external caching."""
        if self._hdbscan_model is None:
            return {}

        return {
            "probabilities": self._hdbscan_model.probabilities_,
            "labels": self._hdbscan_model.labels_,
            "single_linkage_tree": self._hdbscan_model.single_linkage_tree_,
            "condensed_tree": self._hdbscan_model.condensed_tree_,
            "cluster_persistence": getattr(self._hdbscan_model, 'cluster_persistence_', None),
            "outlier_scores": getattr(self._hdbscan_model, 'outlier_scores_', None),
        }

    def get_umap_embeddings(self) -> Optional[np.ndarray]:
        """Return UMAP-reduced embeddings for caching."""
        return self._umap_embeddings

    def get_hdbscan_params(self) -> Dict[str, Any]:
        """Return the winning HDBSCAN parameters from optimization."""
        params = dict(self._algorithm_params) if self._algorithm_params else {}
        if 'cluster_selection_method' not in params:
            params['cluster_selection_method'] = self.config.hdbscan_cluster_selection_method
        return params

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def visualize_elbow(self, output_dir: Optional[Path] = None, filename_prefix: str = "elbow_analysis") -> Optional[Path]:
        """Generate elbow analysis visualization."""
        if self._optimizer is None:
            if self._verbose:
                print("[Elbow Visualization] Not available - HDBSCAN optimization was not used")
            return None
        return self._optimizer.visualize_elbow_analysis(output_dir=output_dir, filename_prefix=filename_prefix)

    def visualize_metrics(self, output_dir: Optional[Path] = None, filename_prefix: str = "metric_comparison") -> Optional[Path]:
        """Generate metric comparison visualization."""
        if self._optimizer is None:
            if self._verbose:
                print("[Metric Comparison] Not available - HDBSCAN optimization was not used")
            return None
        return self._optimizer.visualize_metric_comparison(output_dir=output_dir, filename_prefix=filename_prefix)

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def print_all_clusters(self, n_samples: int = 5):
        """Print all clusters with sample ideas and MECE topics."""
        if self._labels is None:
            raise RuntimeError("Must call run() before print_all_clusters()")

        # Build cluster texts dict
        cluster_texts = {}
        for i, label in enumerate(self._labels):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(self._idea_texts[i])

        print(f"\n{'='*80}")
        print(f"ALL CLUSTERS ({len(cluster_texts)} clusters)")
        print(f"{'='*80}")

        # Build per-cluster probability lookup if HDBSCAN was used
        per_cluster_prob = {}
        per_cluster_probs_array = {}
        probs = self._expanded_probabilities  # expanded to all ideas (if on-the-fly)
        if probs is None and self._hdbscan_model is not None:
            probs = self._hdbscan_model.probabilities_
        if probs is not None:
            for cluster_id_tmp in sorted(cluster_texts.keys()):
                mask = self._labels == cluster_id_tmp
                cluster_probs = probs[mask]
                mean_prob = float(np.mean(cluster_probs)) if len(cluster_probs) > 0 else 0.0
                low_ratio = float((cluster_probs < self.config.low_probability_threshold).sum() / len(cluster_probs)) if len(cluster_probs) > 0 else 0.0
                per_cluster_prob[cluster_id_tmp] = (mean_prob, low_ratio)
                per_cluster_probs_array[cluster_id_tmp] = cluster_probs

        for cluster_id in sorted(cluster_texts.keys()):
            texts = cluster_texts[cluster_id]
            n_ideas = len(texts)

            print(f"\n{'─'*80}")
            if cluster_id in per_cluster_prob:
                mean_prob, low_ratio = per_cluster_prob[cluster_id]
                print(f"CLUSTER {cluster_id} (n={n_ideas}) | prob: mean={mean_prob:.2f}, low_ratio={low_ratio:.1%}")
            else:
                print(f"CLUSTER {cluster_id} (n={n_ideas})")
            print(f"{'─'*80}")

            # Show probability distribution histogram
            if cluster_id in per_cluster_probs_array:
                cluster_probs = per_cluster_probs_array[cluster_id]
                if len(cluster_probs) > 0:
                    bins = np.linspace(0, 1, 11)
                    hist, _ = np.histogram(cluster_probs, bins=bins)
                    max_count = max(hist) if max(hist) > 0 else 1
                    bar_chars = "▁▂▃▄▅▆▇█"
                    bars = []
                    for count in hist:
                        if count == 0:
                            bars.append(" ")
                        else:
                            level = min(7, int((count / max_count) * 7.99))
                            bars.append(bar_chars[level])
                    print(f"\nProb dist: [{''.join(bars)}]  (0%→100%, n={len(cluster_probs)})")

            # Show MECE topics for this cluster (V3)
            if self._cluster_mece_results and cluster_id in self._cluster_mece_results:
                mece = self._cluster_mece_results[cluster_id]
                print(f"\nMECE Topics ({len(mece.topics)}):")
                for j, topic in enumerate(mece.topics, 1):
                    print(f"  [{j}] {topic.topic_label}")
                    print(f"      Inclusion: {topic.inclusion_definition}")
                    print(f"      Exclusion: {topic.exclusion_definition}")

            # Show sample ideas
            unique_texts = list(dict.fromkeys(texts))  # deduplicate
            sample = unique_texts[:n_samples]
            print(f"\nSample ideas ({len(sample)} of {n_ideas}):")
            for i, idea_text in enumerate(sample, 1):
                if len(idea_text) > 100:
                    idea_text = idea_text[:97] + "..."
                print(f"  {i}. {idea_text}")

        print(f"\n{'='*80}\n")

    def print_cluster_mece_topics(self):
        """Print per-cluster MECE topics (Phase 7 output)."""
        if self._cluster_mece_results is None:
            print("\n[MECE Topics] Not available (Phase 7 not run)")
            return

        results = self._cluster_mece_results
        total_topics = sum(len(r.topics) for r in results.values())

        print(f"\n{'='*80}")
        print(f"PER-CLUSTER MECE TOPICS ({len(results)} clusters, {total_topics} total topics)")
        print(f"{'='*80}")

        for cluster_id in sorted(results.keys()):
            result = results[cluster_id]
            print(f"\n{'─'*80}")
            print(f"CLUSTER {cluster_id} (n={result.n_ideas}, {result.n_batches} batch(es), "
                  f"{'reduce skipped' if result.reduce_skipped else 'reduce applied'})")
            print(f"{'─'*80}")

            for j, topic in enumerate(result.topics, 1):
                print(f"\n  [{j}] {topic.topic_label}")
                print(f"      Inclusion: {topic.inclusion_definition}")
                print(f"      Exclusion: {topic.exclusion_definition}")
                if topic.key_expressions:
                    print(f"      Expressions:")
                    for expr in topic.key_expressions[:3]:
                        truncated = expr[:80] + "..." if len(expr) > 80 else expr
                        print(f"        - {truncated}")

        print(f"\n{'='*80}\n")

    # =========================================================================
    # METADATA EXPORT
    # =========================================================================

    def _get_cluster_mean_probability(self, cluster_id: int) -> Optional[float]:
        """Get mean probability for a cluster."""
        if self._expanded_probabilities is not None:
            mask = self._labels == cluster_id
            if not np.any(mask):
                return None
            return float(np.mean(self._expanded_probabilities[mask]))
        if self._hdbscan_model is None:
            return None
        mask = self._labels == cluster_id
        if not np.any(mask):
            return None
        return float(np.mean(self._hdbscan_model.probabilities_[mask]))

    def _get_cluster_coherence(self, cluster_id: int) -> Optional[float]:
        """Get coherence for a specific cluster."""
        if self._metrics and self._metrics.per_cluster_coherence:
            for cid, size, coh in self._metrics.per_cluster_coherence:
                if cid == cluster_id:
                    return coh
        return None

    def to_metadata_model(self) -> models.ClusteringMetadataModel:
        """Export clustering metadata for caching (Layer 2)."""
        from datetime import datetime

        if self._labels is None:
            raise RuntimeError("Must call run() before to_metadata_model()")

        clusters = {}
        unique_labels = set(self._labels) - {-1}

        for cluster_id in sorted(unique_labels):
            cluster_id = int(cluster_id)

            size = int(np.sum(self._labels == cluster_id))

            # Get MECE topics for this cluster if available
            mece_result = self._cluster_mece_results.get(cluster_id) if self._cluster_mece_results else None
            label_theme = None
            label_description = None
            label_key_concepts = None

            if mece_result and mece_result.topics:
                # Use the first topic as the primary label for compatibility
                label_theme = mece_result.topics[0].topic_label
                label_description = mece_result.topics[0].inclusion_definition
                label_key_concepts = [t.topic_label for t in mece_result.topics]

            clusters[cluster_id] = models.ClusterRepresentationCacheModel(
                cluster_id=cluster_id,
                size=size,
                representative_samples=[],
                keywords_ctfidf=[],
                keywords_mmr=[],
                keywords_tfidf=[],
                label_theme=label_theme,
                label_description=label_description,
                label_key_concepts=label_key_concepts,
                mean_probability=self._get_cluster_mean_probability(cluster_id),
                coherence=self._get_cluster_coherence(cluster_id),
            )

        metrics_model = models.ClusteringMetricsModel(
            n_clusters=self._metrics.n_clusters if self._metrics else 0,
            noise_rate=self._metrics.noise_rate if self._metrics else 0.0,
            noise_count=self._metrics.noise_count if self._metrics else 0,
            mean_coherence=self._metrics.mean_coherence if self._metrics else 0.0,
            coherence_breakdown=self._metrics.coherence_breakdown if self._metrics else "",
            silhouette=self._metrics.silhouette if self._metrics else None,
            dbcv=self._metrics.dbcv if self._metrics else None,
        )

        llm_context = None
        if self._extraction_metadata:
            meta = self._extraction_metadata
            llm_context = models.LLMContextModel(
                survey_question=meta.var_lab or "",
                language=meta.lang or "Dutch",
                domain=meta.domain or None,
                entity=meta.entity or None,
                topic=meta.topic or None,
                perspective=meta.perspective or None,
                intent=meta.intent or None,
                taxonomy_axis=meta.taxonomy_axis or None,
                taxonomy_description=meta.taxonomy_axis_description or None,
                taxonomy_actionable_type=meta.taxonomy_actionable_type or None,
            )

        return models.ClusteringMetadataModel(
            clusters=clusters,
            llm_context=llm_context,
            metrics=metrics_model,
            algorithm_used=self._algorithm_used,
            algorithm_params=self._algorithm_params,
            timestamp=datetime.now().isoformat(),
        )


def clean_cluster_ideas(cluster_results: List[models.ClusterModel]) -> List[models.ClusterModel]:
    """Clean cluster idea texts by removing bracketed annotations and normalizing whitespace."""
    cleaned_results = []

    for result in cluster_results:
        cleaned_response_ideas = []

        if result.response_ideas:
            for idea_submodel in result.response_ideas:
                cleaned_idea = idea_submodel.idea
                cleaned_idea = re.sub(r"\[.*?\]", "", cleaned_idea)
                cleaned_idea = re.sub(r"\s+", " ", cleaned_idea).strip()

                cleaned_submodel = models.ClusterSubmodel(
                    idea_id=idea_submodel.idea_id,
                    idea=cleaned_idea,
                    instance=getattr(idea_submodel, 'instance', ''),
                    node=getattr(idea_submodel, 'node', ''),
                    semantic_category=getattr(idea_submodel, 'semantic_category', ''),
                    category_label=getattr(idea_submodel, 'category_label', ''),
                    root=getattr(idea_submodel, 'root', ''),
                    idea_embedding=idea_submodel.idea_embedding,
                    node_embedding=getattr(idea_submodel, 'node_embedding', None),
                    category_embedding=getattr(idea_submodel, 'category_embedding', None),
                    taxonomy_embedding=getattr(idea_submodel, 'taxonomy_embedding', None),
                    initial_cluster=idea_submodel.initial_cluster,
                    cluster_probability=idea_submodel.cluster_probability,
                    expanded_cluster=idea_submodel.expanded_cluster,
                    cluster_theme=idea_submodel.cluster_theme
                )
                cleaned_response_ideas.append(cleaned_submodel)

        cleaned_result = models.ClusterModel(
            respondent_id=result.respondent_id,
            response=result.response,
            response_type=result.response_type,
            quality_filter=result.quality_filter,
            quality_filter_code=result.quality_filter_code,
            response_ideas=cleaned_response_ideas,
            idea_count=len(cleaned_response_ideas)
        )
        cleaned_results.append(cleaned_result)

    return cleaned_results
