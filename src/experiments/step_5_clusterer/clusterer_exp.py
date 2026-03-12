"""
Clusterer - Iterative HDBSCAN Clustering Module - EXPERIMENTAL VERSION

Clustering module with iterative residual approach:
1. UMAP + HDBSCAN with grid search + Pareto frontier selection
2. Accept high-probability members (>= 0.8) → lock into clusters
3. Pool remaining (noise + low-probability) → re-cluster
4. Repeat until residual <= 10% of original N

Also includes:
- c-TF-IDF / MMR keyword extraction with spaCy lemmatization
- LLM-generated cluster labels
- Post-processing (cluster merging, noise reduction)

Pipeline Integration:
- Input: List[EmbeddingsModel] from step 4 (ladder embeddings)
- Output: List[ClusterModel] via to_cluster_model()
- Cache step: "initial_clusters"

Usage (experimental):
    from .clusterer_exp import Clusterer
    from .config_clusterer_exp import ClustererConfig

    config = ClustererConfig()  # defaults: hdbscan, iterative, prob >= 0.8
    clusterer = Clusterer(embeddings_list, config=config)
    clusterer.run()

    cluster_results = clusterer.to_cluster_model()
    keywords = clusterer.get_cluster_keywords()
    labels = clusterer.get_cluster_labels()
"""

import warnings
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

from pydantic import BaseModel, Field
from experiments import models_exp as models


# ---------------------------------------------------------------------------
# Local cache models for to_metadata_model() — not yet in models_exp.py
# ---------------------------------------------------------------------------

class ClusterRepresentationCacheModel(BaseModel):
    cluster_id: int
    size: int
    representative_samples: List[Tuple[str, float]] = Field(default_factory=list)
    keywords_ctfidf: List = Field(default_factory=list)
    keywords_mmr: List = Field(default_factory=list)
    keywords_tfidf: List = Field(default_factory=list)
    label_theme: Optional[str] = None
    label_description: Optional[str] = None
    label_key_concepts: Optional[List[str]] = None
    mean_probability: Optional[float] = None
    coherence: Optional[float] = None


class ClusteringMetricsModel(BaseModel):
    n_clusters: int = 0
    noise_rate: float = 0.0
    noise_count: int = 0
    mean_coherence: float = 0.0
    coherence_breakdown: str = ""
    silhouette: Optional[float] = None
    dbcv: Optional[float] = None


class LLMContextModel(BaseModel):
    survey_question: str = ""
    language: str = "Dutch"
    domain: Optional[str] = None
    entity: Optional[str] = None
    topic: Optional[str] = None
    perspective: Optional[str] = None
    intent: Optional[str] = None
    taxonomy_axis: Optional[str] = None
    taxonomy_description: Optional[str] = None
    taxonomy_actionable_type: Optional[str] = None


class ClusteringMetadataModel(BaseModel):
    clusters: Dict[int, ClusterRepresentationCacheModel] = Field(default_factory=dict)
    llm_context: Optional[LLMContextModel] = None
    metrics: Optional[ClusteringMetricsModel] = None
    algorithm_used: Optional[str] = None
    algorithm_params: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class ClusterLabelModel(BaseModel):
    cluster_id: int
    theme: str
    description: str = ""
    key_concepts: List[str] = Field(default_factory=list)
    n_ideas: int = 0


class ClusterRepresentationModel(BaseModel):
    cluster_id: int
    keywords: List = Field(default_factory=list)
    llm_label: Optional[ClusterLabelModel] = None


class ClusterRepresentationsModel(BaseModel):
    representations: List[ClusterRepresentationModel] = Field(default_factory=list)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


try:
    from .config_clusterer_exp import ClustererConfig
    from .clusterer_helpers_exp import (
        # Preprocessing
        preprocess_embeddings, l2_normalize,
        # Algorithm Selection
        AlgorithmSelector, AlgorithmRecommendation,
        # Parameter Optimization
        ParameterOptimizer, run_umap, n_neighbors_grid,
        # Quality Metrics
        ClusterQualityMetrics, ClusteringMetrics,
        # Post-Processing
        merge_similar_clusters, reduce_noise_by_embedding_similarity,
        # Representation
        RepresentationEngine,
        # Label Generation
        LabelGenerator, ClusterLabel,
        # Text field access
        get_idea_field_text,
    )
    from .placeholder_lookup import build_dataset_placeholders
except ImportError:
    from config_clusterer_exp import ClustererConfig
    from clusterer_helpers_exp import (
        # Preprocessing
        preprocess_embeddings, l2_normalize,
        # Algorithm Selection
        AlgorithmSelector, AlgorithmRecommendation,
        # Parameter Optimization
        ParameterOptimizer, run_umap, n_neighbors_grid,
        # Quality Metrics
        ClusterQualityMetrics, ClusteringMetrics,
        # Post-Processing
        merge_similar_clusters, reduce_noise_by_embedding_similarity,
        # Representation
        RepresentationEngine,
        # Label Generation
        LabelGenerator, ClusterLabel,
        # Text field access
        get_idea_field_text,
    )
    from placeholder_lookup import build_dataset_placeholders

# Suppress common warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
warnings.filterwarnings("ignore", message="overflow encountered in power", module="hdbscan.validity")
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor.providers.gemini")


class Clusterer:
    """
    Enhanced clustering module with automatic algorithm selection,
    Optuna-based optimization, and integrated quality metrics.

    Key Features:
    1. Automatic algorithm selection (DVC + kNN knee + persistence)
    2. Bayesian HDBSCAN optimization via Optuna GridSampler
    3. Coherence-based quality metrics on original embeddings
    4. Optional c-TF-IDF keyword extraction

    Usage:
        # Basic usage (auto mode)
        clusterer = Clusterer(input_list, config=ClustererConfig())
        clusterer.run()
        results = clusterer.to_cluster_model()

        # With manual algorithm override
        config = ClustererConfig(algorithm_mode="hdbscan")
        clusterer = Clusterer(input_list, config=config)

        # With c-TF-IDF representation
        config = ClustererConfig(generate_ctfidf=True)
        clusterer = Clusterer(input_list, config=config)
        keywords = clusterer.get_cluster_keywords()
    """

    def __init__(
        self,
        input_list: List[models.EmbeddingsModel],
        config: Optional[ClustererConfig] = None,
        extraction_metadata: Optional[models.ExtractionMetadata] = None
    ):
        """
        Initialize Clusterer.

        Args:
            input_list: List of EmbeddingsModel with idea_embedding populated
            config: Configuration (uses defaults if None)
            extraction_metadata: Optional ExtractionMetadata for facet context in LLM labels
        """
        self.config = config or ClustererConfig()
        self._input_list = input_list
        self._extraction_metadata = extraction_metadata
        self._verbose = self.config.verbose

        # Will be populated during run()
        self._embeddings_original: Optional[np.ndarray] = None
        self._embeddings_processed: Optional[np.ndarray] = None
        self._idea_texts: Optional[List[str]] = None
        self._idea_indices: Optional[List[Tuple[int, int]]] = None
        self._template_prefix: Optional[str] = None
        self._embedding_text_format: Optional[str] = None  # Text format used for embedding
        self._concept_types: Optional[List[str]] = None
        self._labels: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None
        self._iteration_assigned: Optional[np.ndarray] = None
        self._umap_embeddings: Optional[np.ndarray] = None
        self._hdbscan_model: Optional[hdbscan.HDBSCAN] = None
        self._recommendation: Optional[AlgorithmRecommendation] = None
        self._metrics: Optional[ClusteringMetrics] = None
        # Keywords stored as {"ctfidf": {...}, "mmr": {...}, "tfidf": {...}}
        self._cluster_keywords: Optional[Dict[str, Dict[int, List[Tuple[str, float]]]]] = None
        self._cluster_labels: Optional[Dict[int, ClusterLabel]] = None
        self._algorithm_used: str = ""
        self._algorithm_params: Dict[str, Any] = {}

        # Components
        self._selector = AlgorithmSelector(self.config)
        self._metrics_calculator = ClusterQualityMetrics(self.config)
        self._representation_engine = RepresentationEngine(self.config)
        self._label_generator = LabelGenerator(self.config)

    def run(self) -> 'Clusterer':
        """
        Execute the complete clustering pipeline.

        Default path: Iterative HDBSCAN with grid search + Pareto selection.
        Each iteration accepts points with probability >= threshold (default 0.8),
        then re-clusters the residual until it drops to <= 10% of original N.

        Returns:
            self (for method chaining)
        """
        if self._verbose:
            print("=" * 70)
            print("Clustering Pipeline")
            print("=" * 70)

        # Phase 1: Preprocessing
        self._run_preprocessing()

        # Phase 2-3: Clustering
        algorithm = self.config.algorithm_mode

        if algorithm == "auto":
            # Run DVC + knee analysis to select algorithm
            self._run_algorithm_selection()
            algorithm = self._map_recommendation_to_algorithm()

        if algorithm == "hdbscan":
            if self.config.enable_iterative:
                # Iterative residual clustering on all points
                if self._verbose:
                    print(f"\n[Phase 2-3] Iterative HDBSCAN "
                          f"(accept_prob >= {self.config.iterative_accept_probability}, "
                          f"stop at {self.config.iterative_residual_ratio_stop:.0%} residual)")
                all_indices = np.arange(self._N)
                self._run_iterative_hdbscan(all_indices)
                # Global UMAP for visualization
                n_neighbors_mid = n_neighbors_grid(
                    self._N, k=self.config.n_neighbors_grid_k,
                    high_min=self.config.n_neighbors_high_min
                )[1]
                n_components = self.config.umap_n_components_grid[0]
                self._umap_embeddings = run_umap(
                    self._embeddings_processed, n_neighbors_mid, n_components,
                    self.config.umap_min_dist, self.config.umap_random_state
                )
            else:
                self._run_hdbscan_optimized()
        elif algorithm == "agglomerative":
            self._run_agglomerative()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Phase 4: Post-processing
        self._run_post_processing()

        # Phase 5: Final metrics
        self._compute_final_metrics()

        # Phase 6: Representation (optional)
        if self.config.generate_ctfidf:
            self._run_representation()

        # Phase 7: LLM Labels (optional)
        if self.config.generate_llm_labels:
            self._run_llm_labels()

        return self

    def _run_preprocessing(self):
        """Phase 1: Extract embeddings, optionally PCA. No normalization."""
        if self._verbose:
            print("\n[Phase 1] Preprocessing")

        (
            self._embeddings_original,
            self._embeddings_processed,
            self._idea_texts,
            self._idea_indices,
            self._concept_types,
            _,
            self._template_prefix,
            self._embedding_text_format
        ) = preprocess_embeddings(self._input_list, self.config)

        self._N = len(self._embeddings_original)

        # Pre-allocate tracking arrays
        self._labels = np.full(self._N, -1, dtype=int)
        self._probabilities = np.full(self._N, 0.0, dtype=float)
        self._iteration_assigned = np.full(self._N, -1, dtype=int)

        if self._verbose:
            print(f"  Loaded {self._N} embeddings")

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
            # Skip knee detection - force Agglomerative
            if self._verbose:
                print(f"  HARD RULE: DVC < {force_threshold} → Forcing Agglomerative")

            # Create minimal knee result
            knee_result = {
                'K': None,
                'y_difference': 0.0,
                'has_sharp_knee': False,
                'recommendation': 'AGGLOMERATIVE_FORCED'
            }
        else:
            # Run trial UMAP for knee detection
            # Use middle of n_neighbors grid
            n_neighbors_list = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k, high_min=self.config.n_neighbors_high_min)
            trial_n_neighbors = n_neighbors_list[len(n_neighbors_list) // 2]

            # Use first n_components from grid for trial UMAP
            trial_n_components = self.config.umap_n_components_grid[0]
            trial_umap = run_umap(
                self._embeddings_processed,
                trial_n_neighbors,
                trial_n_components,
                self.config.umap_min_dist,
                self.config.umap_random_state
            )

            # Detect knee on raw UMAP output
            knee_result = self._selector.detect_knee(trial_umap)
            if self._verbose:
                k_str = f"K={knee_result['K']}" if knee_result['K'] else "No knee"
                print(f"  Knee: {k_str}, y_diff={knee_result['y_difference']:.2f}, "
                      f"sharp={knee_result['has_sharp_knee']}")

        # Combined recommendation (no persistence in Phase 2 anymore)
        self._recommendation = self._selector.recommend(dvc_result, knee_result)

        if self._verbose:
            print(f"  Recommendation: {self._recommendation.combined_recommendation} "
                  f"({self._recommendation.confidence} confidence)")
            print(f"  Reasoning: {self._recommendation.reasoning}")

    def _map_recommendation_to_algorithm(self) -> str:
        """Map combined recommendation to algorithm name."""
        rec = self._recommendation.combined_recommendation
        if "HDBSCAN" in rec:
            return "hdbscan"
        elif rec == "AGGLOMERATIVE_OR_KMEANS":
            # Use DVC to decide between agglomerative and kmeans
            # (both work similarly on uniform density data)
            return "agglomerative"
        else:
            return "agglomerative"

    def _run_grouped_clustering(self):
        """Phase 2+3: Group ideas by concept_type, cluster each group independently."""
        if self._verbose:
            print("\n[Phase 2+3] Grouped Clustering by concept_type")

        # Build groups: {concept_type: [flat_indices]}
        groups: Dict[str, List[int]] = {}
        fallback = self.config.concept_type_fallback
        for idx, ct in enumerate(self._concept_types):
            key = ct if ct else fallback
            groups.setdefault(key, []).append(idx)

        # Pool small groups into fallback
        pooled_groups: Dict[str, np.ndarray] = {}
        other_indices = list(groups.get(fallback, []))
        for key, indices in groups.items():
            if key == fallback:
                continue
            if len(indices) < self.config.concept_type_min_group_size:
                other_indices.extend(indices)
            else:
                pooled_groups[key] = np.array(indices)
        if other_indices:
            pooled_groups[fallback] = np.array(other_indices)

        if self._verbose:
            print(f"  {len(pooled_groups)} concept_type groups:")
            for key in sorted(pooled_groups, key=lambda k: -len(pooled_groups[k])):
                print(f"    {key}: {len(pooled_groups[key])} ideas")

        # Cluster each group, largest first
        cluster_offset = 0
        self._algorithm_used = "HDBSCAN"  # default, may be overridden per-group

        for group_name in sorted(pooled_groups, key=lambda k: -len(pooled_groups[k])):
            group_indices = pooled_groups[group_name]
            group_size = len(group_indices)

            if self._verbose:
                print(f"\n{'='*60}")
                print(f"  Group '{group_name}' ({group_size} ideas)")
                print(f"{'='*60}")

            # Algorithm selection for this group
            group_emb_original = self._embeddings_original[group_indices]
            dvc_result = self._selector.compute_dvc(group_emb_original)
            if self._verbose:
                dvc_val = dvc_result['dvc']
                if not np.isnan(dvc_val):
                    print(f"  DVC = {dvc_val:.3f} → {dvc_result['recommendation']}")

            # Choose algorithm based on DVC
            force_threshold = getattr(self.config, 'force_agglomerative_below_dvc', 0.25)
            use_agglomerative = (
                self.config.enable_agglomerative_fallback
                and not np.isnan(dvc_result['dvc'])
                and dvc_result['dvc'] < force_threshold
            )

            if use_agglomerative:
                if self._verbose:
                    print(f"  Algorithm: Agglomerative (DVC < {force_threshold})")
                group_stats = self._run_agglomerative_on_subset(group_indices)
            else:
                if self._verbose:
                    print(f"  Algorithm: HDBSCAN (iterative={self.config.enable_iterative})")
                if self.config.enable_iterative:
                    group_stats = self._run_iterative_hdbscan(group_indices)
                else:
                    # Single-shot: run optimizer once on the group
                    optimizer = ParameterOptimizer(
                        self.config,
                        self._embeddings_processed[group_indices],
                        self._embeddings_original[group_indices],
                        verbose=self._verbose
                    )
                    result = optimizer.optimize()
                    for local_idx, global_idx in enumerate(group_indices):
                        self._labels[global_idx] = int(result.best_labels[local_idx])
                        self._probabilities[global_idx] = float(result.best_model.probabilities_[local_idx])
                        self._iteration_assigned[global_idx] = 0
                    group_stats = {'n_clusters': len(set(result.best_labels) - {-1})}

            # Offset non-noise labels to be globally unique
            n_group_clusters = group_stats.get('n_clusters', 0)
            if cluster_offset > 0:
                for global_idx in group_indices:
                    if self._labels[global_idx] >= 0:
                        self._labels[global_idx] += cluster_offset
            cluster_offset += n_group_clusters

            if self._verbose:
                n_assigned = int((self._labels[group_indices] >= 0).sum())
                print(f"  Group result: {n_group_clusters} clusters, "
                      f"{n_assigned}/{group_size} assigned ({n_assigned/group_size:.1%})")

        # Global UMAP for visualization (on all points, using mid-range params)
        n_neighbors_mid = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k,
                                           high_min=self.config.n_neighbors_high_min)[1]
        n_components = self.config.umap_n_components_grid[0]
        self._umap_embeddings = run_umap(
            self._embeddings_processed, n_neighbors_mid, n_components,
            self.config.umap_min_dist, self.config.umap_random_state
        )

        if self._verbose:
            total_assigned = int((self._labels >= 0).sum())
            total_clusters = len(set(self._labels.tolist()) - {-1})
            print(f"\n  Total: {total_clusters} clusters, "
                  f"{total_assigned}/{self._N} assigned ({total_assigned/self._N:.1%}), "
                  f"{self._N - total_assigned} residual")

    def _run_hdbscan_optimized(self):
        """Phase 3a: Run single-shot grid search HDBSCAN with Pareto selection."""
        if self._verbose:
            print("\n[Phase 3] HDBSCAN Grid Search + Pareto Selection")

        optimizer = ParameterOptimizer(
            self.config,
            self._embeddings_processed,
            self._embeddings_original,
            verbose=self._verbose
        )

        result = optimizer.optimize()

        self._labels = result.best_labels.copy()
        self._umap_embeddings = result.umap_embeddings
        self._hdbscan_model = result.best_model
        self._algorithm_used = "HDBSCAN"
        self._algorithm_params = result.best_params

        # Populate tracking arrays for unified access
        self._probabilities = result.best_model.probabilities_.copy()
        self._iteration_assigned = np.zeros(len(self._labels), dtype=int)  # all iteration 0

        if self._verbose:
            dvc_reduced = self._selector.compute_dvc(result.umap_embeddings)
            dvc_val = dvc_reduced['dvc']
            if not np.isnan(dvc_val):
                print(f"  DVC (UMAP-reduced) = {dvc_val:.3f} (mean_dk={dvc_reduced['mean_dk']:.4f}, std_dk={dvc_reduced['std_dk']:.4f})")

    def _run_iterative_hdbscan(self, group_indices: np.ndarray) -> Dict[str, Any]:
        """
        Iterative residual clustering for a concept_type group.

        Each iteration runs grid search + Pareto selection on the residual points,
        accepts confident members (non-noise AND probability >= threshold), and
        continues on the remainder.

        Writes directly into self._labels, self._probabilities, self._iteration_assigned
        at the positions specified by group_indices.

        Args:
            group_indices: Flat indices into the full N-length arrays.

        Returns:
            Dict with stats: {n_iterations, n_clusters, accepted_per_iter, ...}
        """
        group_size = len(group_indices)
        residual_mask = np.ones(group_size, dtype=bool)
        next_cluster_id = 0
        stats = {'accepted_per_iter': [], 'clusters_per_iter': []}

        for iteration in range(self.config.iterative_max_iterations):
            residual_positions = np.where(residual_mask)[0]
            n_residual = len(residual_positions)

            # Stopping: residual ratio
            if n_residual / group_size <= self.config.iterative_residual_ratio_stop:
                if self._verbose:
                    print(f"    Stopping: residual {n_residual}/{group_size} "
                          f"({n_residual/group_size:.1%}) <= {self.config.iterative_residual_ratio_stop:.0%}")
                break

            # Stopping: too few points
            if n_residual < self.config.iterative_min_residual_size:
                if self._verbose:
                    print(f"    Stopping: residual {n_residual} < min_residual_size {self.config.iterative_min_residual_size}")
                break

            # Extract subset embeddings for this residual
            residual_global_indices = group_indices[residual_positions]
            residual_emb_processed = self._embeddings_processed[residual_global_indices]
            residual_emb_original = self._embeddings_original[residual_global_indices]

            if self._verbose:
                print(f"\n    --- Iteration {iteration} ({n_residual} points) ---")

            # Run grid search + Pareto on residual
            try:
                optimizer = ParameterOptimizer(
                    self.config,
                    residual_emb_processed,
                    residual_emb_original,
                    verbose=self._verbose
                )
                result = optimizer.optimize()
            except (RuntimeError, ValueError) as e:
                if self._verbose:
                    print(f"    Stopping: optimizer failed ({e})")
                break

            # Determine accepted points: non-noise AND probability >= threshold
            iter_labels = result.best_labels
            iter_probs = result.best_model.probabilities_
            accepted_mask = (iter_labels >= 0) & (iter_probs >= self.config.iterative_accept_probability)
            n_accepted = int(accepted_mask.sum())

            # Stopping: no progress
            if n_accepted == 0:
                if self._verbose:
                    print(f"    Stopping: no points accepted (0 with prob >= {self.config.iterative_accept_probability})")
                break

            # Remap local cluster IDs to group-local IDs
            unique_local_clusters = sorted(set(iter_labels[accepted_mask].tolist()))
            local_to_group = {}
            for local_id in unique_local_clusters:
                if local_id >= 0:
                    local_to_group[local_id] = next_cluster_id
                    next_cluster_id += 1

            # Write accepted points into global arrays
            for pos_in_residual in range(n_residual):
                if accepted_mask[pos_in_residual]:
                    global_idx = residual_global_indices[pos_in_residual]
                    local_cluster = int(iter_labels[pos_in_residual])
                    self._labels[global_idx] = local_to_group[local_cluster]
                    self._probabilities[global_idx] = float(iter_probs[pos_in_residual])
                    self._iteration_assigned[global_idx] = iteration

            # Update residual mask
            for pos_in_residual in range(n_residual):
                if accepted_mask[pos_in_residual]:
                    residual_mask[residual_positions[pos_in_residual]] = False

            n_new_clusters = len(local_to_group)
            stats['accepted_per_iter'].append(n_accepted)
            stats['clusters_per_iter'].append(n_new_clusters)

            if self._verbose:
                remaining = int(residual_mask.sum())
                print(f"    Accepted {n_accepted}/{n_residual} points, "
                      f"{n_new_clusters} new clusters, "
                      f"residual: {remaining}/{group_size} ({remaining/group_size:.1%})")

        stats['n_iterations'] = len(stats['accepted_per_iter'])
        stats['n_clusters'] = next_cluster_id
        stats['final_residual'] = int(residual_mask.sum())
        return stats

    def _run_agglomerative_on_subset(self, group_indices: np.ndarray) -> Dict[str, Any]:
        """
        Run Agglomerative clustering on a subset. No iterative loop needed
        since Agglomerative assigns all points (no noise concept).

        Writes into self._labels, self._probabilities, self._iteration_assigned
        at the positions specified by group_indices.

        Returns:
            Dict with stats: {n_clusters, ...}
        """
        group_emb_original = self._embeddings_original[group_indices]
        group_size = len(group_indices)

        # Single UMAP reduction
        n_neighbors = n_neighbors_grid(group_size, k=self.config.n_neighbors_grid_k,
                                       high_min=self.config.n_neighbors_high_min)[1]
        n_components = self.config.umap_n_components_grid[0]
        umap_reduced = run_umap(
            self._embeddings_processed[group_indices],
            n_neighbors, n_components,
            self.config.umap_min_dist,
            self.config.umap_random_state
        )

        # Normalize for ward linkage (ward + euclidean on L2-normalized ≈ cosine-based ward)
        umap_reduced_norm = l2_normalize(umap_reduced)

        # K grid based on sqrt(n)
        sqrt_n = int(np.sqrt(group_size))
        k_grid = sorted(set([
            max(2, int(m * sqrt_n))
            for m in self.config.k_grid_multipliers
        ]))

        if self._verbose:
            print(f"    Agglomerative: K grid {k_grid}, n_neighbors={n_neighbors}, n_components={n_components}")

        # Find best K by silhouette score
        best_k = k_grid[0]
        best_sil = -1.0
        best_labels = None

        for k in k_grid:
            if k >= group_size:
                continue
            clusterer = AgglomerativeClustering(
                n_clusters=k, metric='euclidean',
                linkage=self.config.agglomerative_linkage
            )
            labels = clusterer.fit_predict(umap_reduced_norm)
            try:
                sil = silhouette_score(umap_reduced_norm, labels)
            except ValueError:
                continue
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels

        if best_labels is None:
            best_labels = np.zeros(group_size, dtype=int)

        # Write to global arrays — all assigned, prob=1.0, iteration=0
        for local_idx, global_idx in enumerate(group_indices):
            self._labels[global_idx] = int(best_labels[local_idx])
            self._probabilities[global_idx] = 1.0
            self._iteration_assigned[global_idx] = 0

        return {'n_clusters': len(set(best_labels)), 'best_k': best_k, 'silhouette': best_sil}

    def _run_agglomerative(self):
        """Phase 3b: Run Agglomerative clustering."""
        if self._verbose:
            print("\n[Phase 3] Agglomerative Clustering")

        # Single UMAP reduction - use first n_components from grid
        n_neighbors = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k, high_min=self.config.n_neighbors_high_min)[1]  # Second value
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

        # K grid based on sqrt(n)
        sqrt_n = int(np.sqrt(self._N))
        k_grid = sorted(set([
            max(2, int(m * sqrt_n))
            for m in self.config.k_grid_multipliers
        ]))

        if self._verbose:
            print(f"  K grid: {k_grid}")

        # Find best K by silhouette score
        best_k = k_grid[0]
        best_sil = -1.0
        best_labels = None

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
                sil = silhouette_score(self._umap_embeddings, labels)
                if self._verbose:
                    print(f"    k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels.copy()

        if best_labels is None:
            # Fallback
            clusterer = AgglomerativeClustering(n_clusters=k_grid[0])
            best_labels = clusterer.fit_predict(self._umap_embeddings)
            best_k = k_grid[0]

        self._labels = best_labels
        self._algorithm_used = "Agglomerative"
        self._algorithm_params = {'n_clusters': best_k, 'linkage': self.config.agglomerative_linkage}

        if self._verbose:
            print(f"  Best: k={best_k}, silhouette={best_sil:.3f}")

    def _run_agglomerative_small(self):
        """
        Run Agglomerative clustering for small datasets (n <= small_dataset_threshold).

        Optimized path:
        - No UMAP: run directly on L2-normalized original embeddings
        - K grid: all integers from log(n) to sqrt(n)
        - Scoring: silhouette score
        """
        if self._verbose:
            print("\n[Phase 3] Agglomerative Clustering (Small Dataset Path)")
            print(f"  Dataset size: {self._N} (threshold: {self.config.small_dataset_threshold})")
            print("  Skipping UMAP - using L2-normalized embeddings directly")

        # Use L2-normalized original embeddings (no UMAP)
        embeddings = l2_normalize(self._embeddings_original)

        # K grid: 0.5*sqrt(n) to 2*sqrt(n), centered around expected K=sqrt(n)
        sqrt_n = np.sqrt(self._N)
        k_min = max(2, int(0.5 * sqrt_n))
        k_max = int(2 * sqrt_n)
        k_grid = list(range(k_min, k_max + 1))

        if self._verbose:
            print(f"  K grid: {k_grid} (0.5×sqrt({self._N})={k_min}, 2×sqrt({self._N})={k_max})")

        # Collect metrics for all K values
        results = []  # List of (k, labels, sil, ch, db, coh)

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

                # Compute coherence: mean pairwise cosine similarity within clusters
                # Use original embeddings for semantic coherence
                coherences = []
                for cluster_id in set(labels):
                    if cluster_id < 0:
                        continue
                    mask = labels == cluster_id
                    cluster_emb = self._embeddings_original[mask]
                    if len(cluster_emb) > 1:
                        # Cosine similarity on L2-normalized = dot product
                        cluster_emb_norm = cluster_emb / np.linalg.norm(cluster_emb, axis=1, keepdims=True)
                        sim_matrix = cluster_emb_norm @ cluster_emb_norm.T
                        # Mean of upper triangle (excluding diagonal)
                        n_pts = len(cluster_emb)
                        mean_sim = (sim_matrix.sum() - n_pts) / (n_pts * (n_pts - 1))
                        coherences.append(mean_sim)
                coh = np.mean(coherences) if coherences else 0.0

                if self._verbose:
                    print(f"  {k:>3} | {sil:>6.3f} | {ch:>8.1f} | {db:>6.3f} | {coh:>5.3f}")

                results.append((k, labels.copy(), sil, ch, db, coh))

        if not results:
            # Fallback
            clusterer = AgglomerativeClustering(n_clusters=k_grid[0])
            best_labels = clusterer.fit_predict(embeddings)
            best_k = k_grid[0]
        else:
            # Find best K using bootstrap CI overlap
            # Select smallest K whose CI overlaps with the best K's CI
            best_k, best_labels, selection_reason = self._select_k_by_bootstrap_ci(results)

        self._labels = best_labels
        self._umap_embeddings = embeddings  # Store normalized embeddings for metrics
        self._algorithm_used = "Agglomerative"
        self._algorithm_params = {
            'n_clusters': best_k,
            'linkage': self.config.agglomerative_linkage,
            'small_dataset_path': True
        }

        if self._verbose:
            print(f"  Best: k={best_k} ({selection_reason})")

    def _select_k_by_bootstrap_ci(
        self, results: list
    ) -> tuple:
        """
        Select optimal K using bootstrap confidence intervals.

        Strategy:
        1. For each K, bootstrap resample the data and recompute coherence
        2. Build 95% confidence intervals for each K's coherence
        3. Find the best K (highest coherence)
        4. Select the smallest K whose CI overlaps with the best K's CI

        This gives a statistically principled way to find the smallest K
        that's "not significantly worse" than the best.

        Args:
            results: List of (k, labels, sil, ch, db, coh) tuples

        Returns:
            (best_k, best_labels, selection_reason) tuple
        """
        import time
        start_time = time.time()

        if len(results) < 2:
            k, labels, sil, ch, db, coh = results[0]
            return k, labels, f"coherence={coh:.3f} (only 1 result)"

        # Extract k values and labels
        k_values = [r[0] for r in results]
        all_labels = {r[0]: r[1] for r in results}  # k -> labels mapping

        # Bootstrap parameters
        n_bootstrap = 100  # Number of bootstrap iterations
        confidence_level = 0.95
        alpha = 1 - confidence_level

        # Compute bootstrap CIs for each K
        ci_lower = {}
        ci_upper = {}
        point_estimates = {}

        n_samples = len(self._embeddings_original)

        for k, labels, sil, ch, db, coh in results:
            # Store point estimate
            point_estimates[k] = coh

            # Bootstrap: resample indices and recompute coherence
            bootstrap_coherences = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)

                # Compute coherence on resampled data
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

        # Find the best K (highest point estimate)
        best_k_max = max(k_values, key=lambda k: point_estimates[k])
        best_ci_lower = ci_lower[best_k_max]

        # Find smallest K whose CI overlaps with best K's CI
        # Overlap means: k's upper bound >= best's lower bound
        selected_k = best_k_max
        for k in sorted(k_values):
            if ci_upper[k] >= best_ci_lower:
                selected_k = k
                break

        elapsed = time.time() - start_time

        # Get the labels for selected K
        selected_labels = all_labels[selected_k]
        selected_coh = point_estimates[selected_k]
        best_coh = point_estimates[best_k_max]

        # Build explanation
        if self._verbose:
            print(f"  Bootstrap CIs ({n_bootstrap} iterations, {elapsed:.2f}s):")
            for k in k_values:
                marker = " *" if k == selected_k else ""
                print(f"    k={k}: coh={point_estimates[k]:.3f} CI=[{ci_lower[k]:.3f}, {ci_upper[k]:.3f}]{marker}")

        reason = f"bootstrap CI overlap at k={selected_k} (coh={selected_coh:.3f}, best={best_coh:.3f} at k={best_k_max})"

        return selected_k, selected_labels, reason

    def _run_post_processing(self):
        """Phase 4: Cluster merging and noise reduction."""
        if self._verbose:
            print("\n[Phase 4] Post-processing")

        # Cluster merging
        if self.config.enable_merging:
            self._labels = merge_similar_clusters(
                self._labels,
                self._embeddings_original,
                self.config,
                verbose=self._verbose
            )

        # Noise reduction (only for HDBSCAN): assign noise by embedding similarity
        if self._algorithm_used == "HDBSCAN":
            self._labels, noise_stats = reduce_noise_by_embedding_similarity(
                self._labels,
                self._embeddings_original,
                threshold=self.config.noise_reduction_threshold,
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
            probabilities=self._probabilities,
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

    def _run_representation(self):
        """Phase 6: Extract keywords using all enabled representation methods."""
        if self._verbose:
            print("\n[Phase 6] Keyword Extraction & Representation")
            methods = ["c-TF-IDF"]
            if self.config.generate_mmr_keywords:
                methods.append("MMR")
            if self.config.generate_tfidf_keywords:
                methods.append("TF-IDF")
            print(f"  Methods: {', '.join(methods)}")

        # Get probabilities for filtering
        probabilities = None
        min_probability = None
        if self._probabilities is not None and self._probabilities.any():
            probabilities = self._probabilities
            min_probability = self.config.representative_min_probability
            if self._verbose:
                print(f"  Using core cluster members (probability > {min_probability}) for keywords and LLM samples")

        keyword_texts = (
            [self._get_idea_text(i, self.config.keyword_text_source) for i in range(len(self._labels))]
            if self.config.keyword_text_source != "idea"
            else self._idea_texts
        )
        if self._verbose and self.config.keyword_text_source != "idea":
            print(f"  Keyword text source: {self.config.keyword_text_source}")

        self._cluster_keywords = self._representation_engine.extract_all_keywords_from_labels(
            self._labels,
            keyword_texts,
            embedding_text_format=self._embedding_text_format,
            probabilities=probabilities,
            min_probability=min_probability,
            verbose=self._verbose
        )

        if self._verbose:
            for method, keywords in self._cluster_keywords.items():
                print(f"  {method}: extracted keywords for {len(keywords)} clusters")

    def _compute_cluster_metadata_distributions(self, cluster_id: int) -> Dict[str, Dict[str, float]]:
        """
        Compute concept_type distribution for a cluster.

        Args:
            cluster_id: The cluster to analyze

        Returns:
            Dict with keys: 'concept_type'
            Each value is a dict of {value: percentage}
        """
        cluster_mask = self._labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        categories = []

        for global_idx in cluster_indices:
            resp_idx, idea_idx = self._idea_indices[global_idx]
            idea = self._input_list[resp_idx].response_ideas[idea_idx]
            if hasattr(idea, 'concept_type') and idea.concept_type:
                categories.append(idea.concept_type)

        def to_distribution(items: List[str]) -> Dict[str, float]:
            if not items:
                return {}
            counts = Counter(items)
            total = len(items)
            return {k: round(v/total, 2) for k, v in counts.most_common()}

        return {
            'concept_type': to_distribution(categories),
        }

    def _run_llm_labels(self):
        """Phase 7: Generate LLM-based cluster labels."""
        if self._verbose:
            print("\n[Phase 7] LLM Cluster Label Generation")

        # Build dataset-level placeholders once (from extraction metadata)
        metadata = self._extraction_metadata or models.ExtractionMetadata()
        dataset_placeholders = build_dataset_placeholders(metadata)

        # Build cluster_texts dict using configured label text source
        cluster_texts = {}
        for i, label in enumerate(self._labels):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(self._get_idea_text(i, self.config.label_text_source))

        # Compute representative samples for each cluster
        representative_samples = {}
        for cluster_id in cluster_texts.keys():
            representative_samples[cluster_id] = self.get_representative_ideas(
                cluster_id,
                n=self.config.llm_max_ideas_per_cluster
            )

        if self._verbose:
            use_dense_region = (
                self.config.representative_selection_method == "dense_region"
                and self._probabilities is not None
                and self._probabilities.any()
            )
            method_name = "cluster probability" if use_dense_region else "centroid similarity"
            print(f"  Selected {self.config.llm_max_ideas_per_cluster} representative samples per cluster ({method_name})")
            if self.config.label_text_source != "idea":
                print(f"  Label text source: {self.config.label_text_source}")

        # Compute per-cluster metadata distributions
        cluster_distributions = {}
        for cluster_id in cluster_texts.keys():
            cluster_distributions[cluster_id] = self._compute_cluster_metadata_distributions(cluster_id)

        # Use MMR keywords (primary) for LLM label generation, fallback to c-TF-IDF
        keywords_for_llm = (
            self._cluster_keywords.get("mmr") or self._cluster_keywords.get("ctfidf")
        ) if self._cluster_keywords else None

        self._cluster_labels = self._label_generator.generate_all_labels(
            cluster_texts=cluster_texts,
            dataset_placeholders=dataset_placeholders,
            cluster_keywords=keywords_for_llm,
            representative_samples=representative_samples,
            cluster_distributions=cluster_distributions,
            verbose=self._verbose
        )

        if self._verbose:
            print(f"  Generated labels for {len(self._cluster_labels)} clusters")

    def to_cluster_model(self) -> List[models.ClusterModel]:
        """
        Convert internal results to ClusterModel list (pipeline-compatible).

        Returns:
            List of ClusterModel instances with cluster assignments and probabilities
        """
        if self._labels is None:
            raise RuntimeError("Must call run() before to_cluster_model()")

        # Group results by response: {resp_idx: {idea_idx: (cluster_id, probability)}}
        response_results = {}
        for idx, (resp_idx, idea_idx) in enumerate(self._idea_indices):
            if resp_idx not in response_results:
                response_results[resp_idx] = {}

            cluster_id = int(self._labels[idx])
            # Get probability and iteration from tracking arrays
            probability = None
            if self._probabilities is not None and cluster_id != -1:
                probability = float(self._probabilities[idx])
            iteration = None
            if self._iteration_assigned is not None and self._iteration_assigned[idx] >= 0:
                iteration = int(self._iteration_assigned[idx])

            response_results[resp_idx][idea_idx] = (cluster_id, probability, iteration)

        # Build output list
        output_list = []
        for resp_idx, response in enumerate(self._input_list):
            # Create new ClusterModel from EmbeddingsModel
            cluster_data = response.model_dump()

            # Update ideas with cluster assignments and probabilities
            if cluster_data.get('response_ideas'):
                for idea_idx, idea in enumerate(cluster_data['response_ideas']):
                    if resp_idx in response_results and idea_idx in response_results[resp_idx]:
                        cluster_id, probability, iteration = response_results[resp_idx][idea_idx]
                        idea['initial_cluster'] = cluster_id
                        idea['cluster_probability'] = probability
                        idea['iteration_assigned'] = iteration
                    else:
                        idea['initial_cluster'] = -1  # Noise or missing
                        idea['cluster_probability'] = None
                        idea['iteration_assigned'] = None

            output_list.append(models.ClusterModel.model_validate(cluster_data))

        return output_list

    def get_cluster_keywords(self, method: str = "ctfidf") -> Optional[Dict[int, List[Tuple[str, float]]]]:
        """
        Get keywords for each cluster from a specific representation method.

        Args:
            method: Representation method ("ctfidf", "mmr", or "tfidf")

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples,
            or None if representation not enabled or method not run
        """
        if self._cluster_keywords is None:
            return None
        return self._cluster_keywords.get(method)

    def get_all_cluster_keywords(self) -> Optional[Dict[str, Dict[int, List[Tuple[str, float]]]]]:
        """
        Get all keyword representations.

        Returns:
            Dict mapping method name to keyword dict:
            {"ctfidf": {...}, "mmr": {...}, "tfidf": {...}}
            or None if representation not enabled
        """
        return self._cluster_keywords

    def _get_idea_text(self, flat_idx: int, source: str) -> str:
        """Get text for an idea by flat index and configurable field source.

        Args:
            flat_idx: Index into the flat arrays (idea_texts, labels, etc.)
            source: Field name - "idea", "ladder", "instance", "rung_1",
                    "rung_2", "concept_type",
                    or composite: "idea+rung_2"
        """
        resp_idx, idea_idx = self._idea_indices[flat_idx]
        idea = self._input_list[resp_idx].response_ideas[idea_idx]
        return get_idea_field_text(idea, source, separator=self.config.text_separator)

    def get_representative_ideas(
        self,
        cluster_id: int,
        n: int = 5,
        text_source: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get most representative ideas for a cluster.

        Supports two selection methods (configurable via representative_selection_method):

        1. "dense_region" (default): Uses HDBSCAN's probabilities_ to select ideas that
           are core members of the cluster. Only includes ideas with probability >
           representative_min_probability (default 0.8) to ensure high-confidence
           representatives. Falls back to centroid method if HDBSCAN model is not
           available (e.g., Agglomerative clustering).

        2. "centroid": Legacy method that computes the centroid (mean embedding) and
           returns ideas closest to the centroid by cosine similarity.

        Both methods first deduplicate by text to ensure diverse results - when multiple
        points have the same text, only the best instance is kept (highest probability
        for dense_region, first occurrence for centroid).

        Args:
            cluster_id: The cluster ID to get representatives for
            n: Number of representative ideas to return (default 5)
            text_source: Field source for idea text (default: config.label_text_source).
                Supports single fields ("idea", "rung_1", "rung_2") and
                composites ("idea+rung_2").

        Returns:
            List of (idea_text, score) tuples. For dense_region method, score is
            cluster probability (higher = stronger member). For centroid method,
            score is cosine similarity (higher = closer to centroid).
        """
        if self._labels is None or self._embeddings_original is None:
            raise RuntimeError("Must call run() before get_representative_ideas()")

        source = text_source or self.config.label_text_source

        # Get indices of ideas in this cluster
        cluster_mask = self._labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            return []

        # Determine which method to use
        use_dense_region = (
            self.config.representative_selection_method == "dense_region"
            and self._probabilities is not None
            and self._probabilities.any()
        )

        # Step 1: Build text -> best index mapping (deduplication)
        # For dense_region: keep the point with highest probability per unique text
        # For centroid: keep first occurrence per unique text
        text_to_best = {}  # text -> (global_idx, local_idx, score)

        for local_idx, global_idx in enumerate(cluster_indices):
            text = self._get_idea_text(global_idx, source)

            if use_dense_region:
                prob = float(self._probabilities[global_idx])
                # Filter by minimum probability threshold
                if prob <= self.config.representative_min_probability:
                    continue
                # Higher probability = stronger cluster member = better
                if text not in text_to_best or prob > text_to_best[text][2]:
                    text_to_best[text] = (global_idx, local_idx, prob)
            else:
                # Centroid method: just keep first occurrence (score computed later)
                if text not in text_to_best:
                    text_to_best[text] = (global_idx, local_idx, None)

        # Step 2: Select top N based on method
        if use_dense_region:
            # Sort by probability (descending = highest/strongest first)
            sorted_items = sorted(text_to_best.items(), key=lambda x: x[1][2], reverse=True)
            n_to_select = min(n, len(sorted_items))
            return [(text, score) for text, (_, _, score) in sorted_items[:n_to_select]]

        else:
            # Centroid method: compute similarities for deduplicated indices
            cluster_embeddings = self._embeddings_original[cluster_mask]
            centroid = cluster_embeddings.mean(axis=0)
            similarities = cosine_similarity([centroid], cluster_embeddings)[0]

            # Update scores in text_to_best
            for cleaned_text, (global_idx, local_idx, _) in text_to_best.items():
                text_to_best[cleaned_text] = (global_idx, local_idx, float(similarities[local_idx]))

            # Sort by similarity (descending = highest first)
            sorted_items = sorted(text_to_best.items(), key=lambda x: x[1][2], reverse=True)
            n_to_select = min(n, len(sorted_items))
            return [(text, score) for text, (_, _, score) in sorted_items[:n_to_select]]

    def get_metrics(self) -> ClusteringMetrics:
        """
        Get comprehensive clustering quality metrics.

        Returns:
            ClusteringMetrics dataclass with all computed metrics
        """
        if self._metrics is None:
            raise RuntimeError("Must call run() before get_metrics()")
        return self._metrics

    def get_algorithm_recommendation(self) -> Optional[AlgorithmRecommendation]:
        """
        Get algorithm selection details.

        Returns:
            AlgorithmRecommendation with DVC, knee, and combined recommendation,
            or None if algorithm selection was skipped (e.g., small dataset path)
        """
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

    def get_cluster_labels(self) -> Optional[Dict[int, ClusterLabel]]:
        """
        Get LLM-generated labels for each cluster.

        Returns:
            Dict mapping cluster_id to ClusterLabel,
            or None if LLM labels not enabled
        """
        return self._cluster_labels

    def print_all_clusters(self, n_samples: int = 5):
        """
        Print all clusters with sample ideas.

        Args:
            n_samples: Number of sample ideas to show per cluster
        """
        if self._labels is None:
            raise RuntimeError("Must call run() before print_all_clusters()")

        import random

        # Build cluster_texts dict using configured verbose text source
        cluster_texts = {}
        for i, label in enumerate(self._labels):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(self._get_idea_text(i, self.config.verbose_text_source))

        print(f"\n{'='*80}")
        print(f"ALL CLUSTERS ({len(cluster_texts)} clusters)")
        if self.config.verbose_text_source != "idea":
            print(f"  Text source: {self.config.verbose_text_source}")
        print(f"{'='*80}")

        # Build per-cluster probability lookup
        per_cluster_prob = {}
        if self._probabilities is not None and self._probabilities.any():
            probs = self._probabilities
            for cluster_id_tmp in sorted(cluster_texts.keys()):
                mask = self._labels == cluster_id_tmp
                cluster_probs = probs[mask]
                mean_prob = float(np.mean(cluster_probs)) if len(cluster_probs) > 0 else 0.0
                low_ratio = float((cluster_probs < self.config.low_probability_threshold).sum() / len(cluster_probs)) if len(cluster_probs) > 0 else 0.0
                per_cluster_prob[cluster_id_tmp] = (mean_prob, low_ratio)

        for cluster_id in sorted(cluster_texts.keys()):
            texts = cluster_texts[cluster_id]
            n_ideas = len(texts)

            print(f"\n{'─'*80}")
            # Include probability metrics in cluster header if available
            if cluster_id in per_cluster_prob:
                mean_prob, low_ratio = per_cluster_prob[cluster_id]
                print(f"CLUSTER {cluster_id} (n={n_ideas}) | prob: mean={mean_prob:.2f}, low_ratio={low_ratio:.1%}")
            else:
                print(f"CLUSTER {cluster_id} (n={n_ideas})")
            print(f"{'─'*80}")

            # Show LLM label if available
            if self._cluster_labels and cluster_id in self._cluster_labels:
                label = self._cluster_labels[cluster_id]
                print(f"\nTheme: {label.theme}")
                print(f"Description: {label.description}")
                if label.key_concepts:
                    print(f"Key concepts: {', '.join(label.key_concepts)}")

            # Show MMR keywords if available
            if self._cluster_keywords:
                mmr_keywords = self._cluster_keywords.get("mmr", {})
                if cluster_id in mmr_keywords:
                    keywords = mmr_keywords[cluster_id]
                    kw_str = ", ".join([f"{kw} ({score:.3f})" for kw, score in keywords[:8]])
                    print(f"\nKeywords: {kw_str}")

            # Show representative ideas using verbose text source
            representative = self.get_representative_ideas(
                cluster_id, n=n_samples, text_source=self.config.verbose_text_source
            )

            # Deduplicate (preserve order, keep first occurrence) - same as label_generator
            seen = set()
            unique_ideas = []
            for idea_text, score in representative:
                if idea_text not in seen:
                    seen.add(idea_text)
                    unique_ideas.append((idea_text, score))

            print(f"\nRepresentative ideas ({len(unique_ideas)} of {n_ideas}):")
            for i, (idea_text, score) in enumerate(unique_ideas, 1):
                # Truncate if too long
                if len(idea_text) > 100:
                    idea_text = idea_text[:97] + "..."
                print(f"  {i}. {idea_text} (score={score:.3f})")

        print(f"\n{'='*80}\n")

    # ==========================================================================
    # METADATA EXPORT METHODS
    # ==========================================================================

    def _get_cluster_mean_probability(self, cluster_id: int) -> Optional[float]:
        """Get mean probability for a cluster."""
        if self._probabilities is None:
            return None
        mask = self._labels == cluster_id
        if not np.any(mask):
            return None
        return float(np.mean(self._probabilities[mask]))

    def _get_cluster_coherence(self, cluster_id: int) -> Optional[float]:
        """Get coherence for a specific cluster."""
        if self._metrics and self._metrics.per_cluster_coherence:
            for cid, size, coh in self._metrics.per_cluster_coherence:
                if cid == cluster_id:
                    return coh
        return None

    def _get_cluster_distributions(self, cluster_id: int) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute concept_type distribution for a cluster (for metadata caching).

        Returns:
            Dict with 'concept_type' distribution, e.g.:
            {'concept_type': {'recommendation': 0.5, 'attribute': 0.3, 'opinion': 0.2}}
        """
        cluster_indices = np.where(self._labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            return None

        category_counts = Counter()

        for idx in cluster_indices:
            resp_idx, idea_idx = self._idea_indices[idx]
            response = self._input_list[resp_idx]
            if response.response_ideas and idea_idx < len(response.response_ideas):
                idea = response.response_ideas[idea_idx]
                cat = getattr(idea, 'concept_type', '')
                if cat:
                    category_counts[cat] += 1

        if not category_counts:
            return None

        total = sum(category_counts.values())
        return {
            'concept_type': {k: v/total for k, v in category_counts.items()},
        }

    def to_metadata_model(self):
        """
        Export clustering metadata for caching (Layer 2).

        This method captures all cluster-level data including:
        - Representative samples that were given to the LLM
        - Keywords (c-TF-IDF, MMR, TF-IDF)
        - LLM-generated labels
        - Cluster distributions
        - Global LLM context (survey question, facet, etc.)
        - Quality metrics

        Returns:
            ClusteringMetadataModel with all cluster-level data for caching
        """
        from datetime import datetime

        if self._labels is None:
            raise RuntimeError("Must call run() before to_metadata_model()")

        # Build per-cluster data
        clusters = {}
        unique_labels = set(self._labels) - {-1}

        for cluster_id in sorted(unique_labels):
            cluster_id = int(cluster_id)

            # Get representative samples (what went to LLM)
            rep_samples = self.get_representative_ideas(cluster_id, n=self.config.llm_max_ideas_per_cluster)

            # Get keywords
            kw_ctfidf = self._cluster_keywords.get('ctfidf', {}).get(cluster_id, []) if self._cluster_keywords else []
            kw_mmr = self._cluster_keywords.get('mmr', {}).get(cluster_id, []) if self._cluster_keywords else []
            kw_tfidf = self._cluster_keywords.get('tfidf', {}).get(cluster_id, []) if self._cluster_keywords else []

            # Get LLM label if available
            label = self._cluster_labels.get(cluster_id) if self._cluster_labels else None

            # Cluster size
            size = int(np.sum(self._labels == cluster_id))

            # Get cluster distributions
            distributions = self._get_cluster_distributions(cluster_id)

            clusters[cluster_id] = ClusterRepresentationCacheModel(
                cluster_id=cluster_id,
                size=size,
                representative_samples=rep_samples,
                keywords_ctfidf=kw_ctfidf,
                keywords_mmr=kw_mmr,
                keywords_tfidf=kw_tfidf,
                label_theme=label.theme if label else None,
                label_description=label.description if label else None,
                label_key_concepts=label.key_concepts if label else None,
                mean_probability=self._get_cluster_mean_probability(cluster_id),
                coherence=self._get_cluster_coherence(cluster_id),
            )

        # Build global metrics
        metrics_model = ClusteringMetricsModel(
            n_clusters=self._metrics.n_clusters if self._metrics else 0,
            noise_rate=self._metrics.noise_rate if self._metrics else 0.0,
            noise_count=self._metrics.noise_count if self._metrics else 0,
            mean_coherence=self._metrics.mean_coherence if self._metrics else 0.0,
            coherence_breakdown=self._metrics.coherence_breakdown if self._metrics else "",
            silhouette=self._metrics.silhouette if self._metrics else None,
            dbcv=self._metrics.dbcv if self._metrics else None,
        )

        # Build LLM context (global context provided to all clusters)
        llm_context = None
        if self._extraction_metadata:
            meta = self._extraction_metadata
            llm_context = LLMContextModel(
                survey_question=meta.var_lab or "",
                language=meta.lang or "Dutch",
                domain=meta.domain or None,
                entity=meta.entity or None,
                topic=meta.topic or None,
                perspective=meta.perspective or None,
                intent=meta.intent or None,
                # LLMContextModel uses legacy field names; mapped from new ExtractionMetadata
                taxonomy_axis=meta.primary_facet or None,
                taxonomy_description=meta.primary_facet_description or None,
                taxonomy_actionable_type=None,  # dead field, never populated
            )

        return ClusteringMetadataModel(
            clusters=clusters,
            llm_context=llm_context,
            metrics=metrics_model,
            algorithm_used=self._algorithm_used,
            algorithm_params=self._algorithm_params,
            timestamp=datetime.now().isoformat(),
        )
