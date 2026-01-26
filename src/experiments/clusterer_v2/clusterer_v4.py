"""
ClustererV4 Main Module (PaCMAP)

Unified clustering utility with PaCMAP dimensionality reduction.
Replaces UMAP with PaCMAP throughout the pipeline.

Key differences from ClustererV2:
- Uses PaCMAP instead of UMAP for dimensionality reduction
- Optimizes PaCMAP parameters (n_neighbors, MN_ratio, FP_ratio) via Optuna
- Uses sklearn kNN to bypass Annoy bug on macOS + Python 3.12+
"""

import warnings
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import hdbscan

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import models
from .config_v4 import ClustererV4Config
from .preprocessing import preprocess_embeddings, l2_normalize
from .algorithm_selector import AlgorithmSelector, AlgorithmRecommendation
from .parameter_optimizer_pacmap import (
    ParameterOptimizerPaCMAP,
    run_pacmap,
    n_neighbors_grid_pacmap,
    mcs_grid_sqrt
)
from .quality_metrics import ClusterQualityMetrics, ClusteringMetrics
from .post_processing import merge_similar_clusters, recluster_noise, reduce_noise_by_embedding_similarity
from .representation import RepresentationEngine
from .label_generator import LabelGenerator, ClusterLabel

# Suppress common warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor.providers.gemini")
warnings.filterwarnings("ignore", message="random state is set")


class ClustererV4:
    """
    Enhanced clustering module with PaCMAP dimensionality reduction.

    Key Features:
    1. Automatic algorithm selection (DVC + kNN knee)
    2. Bayesian PaCMAP + HDBSCAN optimization via Optuna GridSampler
    3. Coherence-based quality metrics on original embeddings
    4. Optional c-TF-IDF keyword extraction

    Usage:
        clusterer = ClustererV4(input_list, config=ClustererV4Config())
        clusterer.run()
        results = clusterer.to_cluster_model()
    """

    def __init__(
        self,
        input_list: List[models.EmbeddingsModel],
        config: Optional[ClustererV4Config] = None
    ):
        """
        Initialize ClustererV4.

        Args:
            input_list: List of EmbeddingsModel with idea_embedding populated
            config: Configuration (uses defaults if None)
        """
        self.config = config or ClustererV4Config()
        self._input_list = input_list
        self._verbose = self.config.verbose

        # Will be populated during run()
        self._embeddings_original: Optional[np.ndarray] = None
        self._embeddings_processed: Optional[np.ndarray] = None
        self._idea_texts: Optional[List[str]] = None
        self._idea_indices: Optional[List[Tuple[int, int]]] = None
        self._template_prefix: Optional[str] = None
        self._labels: Optional[np.ndarray] = None
        self._pacmap_embeddings: Optional[np.ndarray] = None  # Changed from _umap_embeddings
        self._hdbscan_model: Optional[hdbscan.HDBSCAN] = None
        self._recommendation: Optional[AlgorithmRecommendation] = None
        self._metrics: Optional[ClusteringMetrics] = None
        self._cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None
        self._cluster_labels: Optional[Dict[int, ClusterLabel]] = None
        self._algorithm_used: str = ""
        self._algorithm_params: Dict[str, Any] = {}

        # Components
        self._selector = AlgorithmSelector(self.config)
        self._metrics_calculator = ClusterQualityMetrics(self.config)
        self._representation_engine = RepresentationEngine(self.config)
        self._label_generator = LabelGenerator(self.config)

    def run(self) -> 'ClustererV4':
        """
        Execute the complete clustering pipeline.

        Returns:
            self (for method chaining)
        """
        if self._verbose:
            print("=" * 70)
            print("ClustererV4 Clustering Pipeline (PaCMAP)")
            print("=" * 70)

        # Phase 1: Preprocessing
        self._run_preprocessing()

        # Phase 2: Algorithm Selection
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

        # Phase 6: Representation (optional)
        if self.config.generate_ctfidf:
            self._run_representation()

        # Phase 7: LLM Labels (optional)
        if self.config.generate_llm_labels:
            self._run_llm_labels()

        return self

    def _run_preprocessing(self):
        """Phase 1: Extract, normalize, and optionally PCA embeddings."""
        if self._verbose:
            print("\n[Phase 1] Preprocessing")

        (
            self._embeddings_original,
            self._embeddings_processed,
            self._idea_texts,
            self._idea_indices,
            _,
            self._template_prefix
        ) = preprocess_embeddings(self._input_list, self.config)

        self._N = len(self._embeddings_original)

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
                print(f"  DVC = {dvc_val:.3f} -> {dvc_result['recommendation']}")

        # Check hard DVC rule first
        force_threshold = getattr(self.config, 'force_agglomerative_below_dvc', 0.25)
        if not np.isnan(dvc_result['dvc']) and dvc_result['dvc'] < force_threshold:
            if self._verbose:
                print(f"  HARD RULE: DVC < {force_threshold} -> Forcing Agglomerative")

            knee_result = {
                'K': None,
                'y_difference': 0.0,
                'has_sharp_knee': False,
                'recommendation': 'AGGLOMERATIVE_FORCED'
            }
        else:
            # Run trial PaCMAP for knee detection
            n_neighbors_list = list(self.config.pacmap_n_neighbors_grid)
            trial_n_neighbors = n_neighbors_list[len(n_neighbors_list) // 2]
            trial_mn_ratio = self.config.pacmap_mn_ratio_grid[len(self.config.pacmap_mn_ratio_grid) // 2]
            trial_fp_ratio = self.config.pacmap_fp_ratio_grid[len(self.config.pacmap_fp_ratio_grid) // 2]
            trial_n_components = self.config.pacmap_n_components_grid[0]

            if self._verbose:
                print(f"  Trial PaCMAP: nn={trial_n_neighbors}, MN={trial_mn_ratio}, "
                      f"FP={trial_fp_ratio}, nc={trial_n_components}")

            trial_pacmap = run_pacmap(
                self._embeddings_processed,
                trial_n_neighbors,
                trial_n_components,
                MN_ratio=trial_mn_ratio,
                FP_ratio=trial_fp_ratio,
                random_state=self.config.pacmap_random_state,
                apply_pca=self.config.pacmap_apply_pca
            )
            trial_pacmap_normalized = l2_normalize(trial_pacmap)

            # Detect knee
            knee_result = self._selector.detect_knee(trial_pacmap_normalized)
            if self._verbose:
                k_str = f"K={knee_result['K']}" if knee_result['K'] else "No knee"
                print(f"  Knee: {k_str}, y_diff={knee_result['y_difference']:.2f}, "
                      f"sharp={knee_result['has_sharp_knee']}")

        # Combined recommendation
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
            return "agglomerative"
        else:
            return "agglomerative"

    def _run_hdbscan_optimized(self):
        """Phase 3a: Run Optuna-optimized HDBSCAN with PaCMAP."""
        if self._verbose:
            print("\n[Phase 3] HDBSCAN Optimization (Optuna + PaCMAP)")

        optimizer = ParameterOptimizerPaCMAP(
            self.config,
            self._embeddings_processed,
            self._embeddings_original,
            verbose=self._verbose
        )

        result = optimizer.optimize()

        self._labels = result.best_labels.copy()
        self._pacmap_embeddings = result.pacmap_embeddings
        self._hdbscan_model = result.best_model
        self._algorithm_used = "HDBSCAN"
        self._algorithm_params = result.best_params

    def _run_agglomerative(self):
        """Phase 3b: Run Agglomerative clustering with PaCMAP."""
        if self._verbose:
            print("\n[Phase 3] Agglomerative Clustering (PaCMAP)")

        # Single PaCMAP reduction
        n_neighbors = list(self.config.pacmap_n_neighbors_grid)[1]  # Second value
        mn_ratio = self.config.pacmap_mn_ratio_grid[len(self.config.pacmap_mn_ratio_grid) // 2]
        fp_ratio = self.config.pacmap_fp_ratio_grid[len(self.config.pacmap_fp_ratio_grid) // 2]
        n_components = self.config.pacmap_n_components_grid[0]

        if self._verbose:
            print(f"  PaCMAP: nn={n_neighbors}, MN={mn_ratio}, FP={fp_ratio}, nc={n_components}")

        self._pacmap_embeddings = run_pacmap(
            self._embeddings_processed,
            n_neighbors,
            n_components,
            MN_ratio=mn_ratio,
            FP_ratio=fp_ratio,
            random_state=self.config.pacmap_random_state,
            apply_pca=self.config.pacmap_apply_pca
        )
        self._pacmap_embeddings = l2_normalize(self._pacmap_embeddings)

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
            if k >= len(self._pacmap_embeddings):
                continue

            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='euclidean',
                linkage=self.config.agglomerative_linkage
            )
            labels = clusterer.fit_predict(self._pacmap_embeddings)

            if len(set(labels)) > 1:
                sil = silhouette_score(self._pacmap_embeddings, labels)
                if self._verbose:
                    print(f"    k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels.copy()

        if best_labels is None:
            clusterer = AgglomerativeClustering(n_clusters=k_grid[0])
            best_labels = clusterer.fit_predict(self._pacmap_embeddings)
            best_k = k_grid[0]

        self._labels = best_labels
        self._algorithm_used = "Agglomerative"
        self._algorithm_params = {'n_clusters': best_k, 'linkage': self.config.agglomerative_linkage}

        if self._verbose:
            print(f"  Best: k={best_k}, silhouette={best_sil:.3f}")

    def _run_kmeans(self):
        """Phase 3c: Run K-means clustering with PaCMAP."""
        if self._verbose:
            print("\n[Phase 3] K-means Clustering (PaCMAP)")

        # Single PaCMAP reduction
        n_neighbors = list(self.config.pacmap_n_neighbors_grid)[1]
        mn_ratio = self.config.pacmap_mn_ratio_grid[len(self.config.pacmap_mn_ratio_grid) // 2]
        fp_ratio = self.config.pacmap_fp_ratio_grid[len(self.config.pacmap_fp_ratio_grid) // 2]
        n_components = self.config.pacmap_n_components_grid[0]

        if self._verbose:
            print(f"  PaCMAP: nn={n_neighbors}, MN={mn_ratio}, FP={fp_ratio}, nc={n_components}")

        self._pacmap_embeddings = run_pacmap(
            self._embeddings_processed,
            n_neighbors,
            n_components,
            MN_ratio=mn_ratio,
            FP_ratio=fp_ratio,
            random_state=self.config.pacmap_random_state,
            apply_pca=self.config.pacmap_apply_pca
        )
        self._pacmap_embeddings = l2_normalize(self._pacmap_embeddings)

        # K grid
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
            if k >= len(self._pacmap_embeddings):
                continue

            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(self._pacmap_embeddings)

            if len(set(labels)) > 1:
                sil = silhouette_score(self._pacmap_embeddings, labels)
                if self._verbose:
                    print(f"    k={k}: silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels.copy()

        if best_labels is None:
            clusterer = KMeans(n_clusters=k_grid[0], random_state=42, n_init=10)
            best_labels = clusterer.fit_predict(self._pacmap_embeddings)
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

        # Cluster merging
        if self.config.enable_merging:
            self._labels = merge_similar_clusters(
                self._labels,
                self._embeddings_original,
                self.config,
                verbose=self._verbose
            )

        # Noise reduction (only for HDBSCAN)
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
                    self._pacmap_embeddings,
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
            self._pacmap_embeddings,
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

    def _run_representation(self):
        """Phase 6: Extract c-TF-IDF keywords."""
        if self._verbose:
            print("\n[Phase 6] c-TF-IDF Keyword Extraction")

        self._cluster_keywords = self._representation_engine.extract_keywords_from_labels(
            self._labels,
            self._idea_texts,
            template_prefix=self._template_prefix,
            verbose=self._verbose
        )

        if self._verbose:
            print(f"  Extracted keywords for {len(self._cluster_keywords)} clusters")

    def _run_llm_labels(self):
        """Phase 7: Generate LLM-based cluster labels."""
        if self._verbose:
            print("\n[Phase 7] LLM Cluster Label Generation")

        cluster_texts = {}
        for i, (label, text) in enumerate(zip(self._labels, self._idea_texts)):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(text)

        self._cluster_labels = self._label_generator.generate_all_labels(
            cluster_texts=cluster_texts,
            cluster_keywords=self._cluster_keywords,
            survey_question="",
            language="Dutch",
            verbose=self._verbose
        )

        if self._verbose:
            print(f"  Generated labels for {len(self._cluster_labels)} clusters")

    def to_cluster_model(self) -> List[models.ClusterModel]:
        """Convert internal results to ClusterModel list."""
        if self._labels is None:
            raise RuntimeError("Must call run() before to_cluster_model()")

        response_results = {}
        for idx, (resp_idx, idea_idx) in enumerate(self._idea_indices):
            if resp_idx not in response_results:
                response_results[resp_idx] = {}
            response_results[resp_idx][idea_idx] = int(self._labels[idx])

        output_list = []
        for resp_idx, response in enumerate(self._input_list):
            cluster_data = response.model_dump()

            if cluster_data.get('response_ideas'):
                for idea_idx, idea in enumerate(cluster_data['response_ideas']):
                    if resp_idx in response_results and idea_idx in response_results[resp_idx]:
                        idea['initial_cluster'] = response_results[resp_idx][idea_idx]
                    else:
                        idea['initial_cluster'] = -1

            output_list.append(models.ClusterModel.model_validate(cluster_data))

        return output_list

    def get_cluster_keywords(self) -> Optional[Dict[int, List[Tuple[str, float]]]]:
        """Get c-TF-IDF keywords for each cluster."""
        return self._cluster_keywords

    def get_metrics(self) -> ClusteringMetrics:
        """Get comprehensive clustering quality metrics."""
        if self._metrics is None:
            raise RuntimeError("Must call run() before get_metrics()")
        return self._metrics

    def get_algorithm_recommendation(self) -> AlgorithmRecommendation:
        """Get algorithm selection details."""
        if self._recommendation is None:
            raise RuntimeError("Must call run() before get_algorithm_recommendation()")
        return self._recommendation

    @property
    def labels_(self) -> np.ndarray:
        """Cluster labels for all ideas."""
        if self._labels is None:
            raise RuntimeError("Must call run() before accessing labels_")
        return self._labels

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found."""
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
        """Get LLM-generated labels for each cluster."""
        return self._cluster_labels

    def print_all_clusters(self, n_samples: int = 5):
        """Print all clusters with sample ideas."""
        if self._labels is None:
            raise RuntimeError("Must call run() before print_all_clusters()")

        import random
        from .representation import extract_embedded_text

        cluster_texts = {}
        for i, (label, text) in enumerate(zip(self._labels, self._idea_texts)):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(text)

        print(f"\n{'='*80}")
        print(f"ALL CLUSTERS ({len(cluster_texts)} clusters)")
        print(f"{'='*80}")

        per_cluster_prob = {}
        if self._hdbscan_model is not None and hasattr(self._hdbscan_model, 'probabilities_'):
            probs = self._hdbscan_model.probabilities_
            for cluster_id_tmp in sorted(cluster_texts.keys()):
                mask = self._labels == cluster_id_tmp
                cluster_probs = probs[mask]
                mean_prob = float(np.mean(cluster_probs)) if len(cluster_probs) > 0 else 0.0
                low_ratio = float((cluster_probs < self.config.low_probability_threshold).sum() / len(cluster_probs)) if len(cluster_probs) > 0 else 0.0
                per_cluster_prob[cluster_id_tmp] = (mean_prob, low_ratio)

        for cluster_id in sorted(cluster_texts.keys()):
            texts = cluster_texts[cluster_id]
            n_ideas = len(texts)

            print(f"\n{'-'*80}")
            if cluster_id in per_cluster_prob:
                mean_prob, low_ratio = per_cluster_prob[cluster_id]
                print(f"CLUSTER {cluster_id} (n={n_ideas}) | prob: mean={mean_prob:.2f}, low_ratio={low_ratio:.1%}")
            else:
                print(f"CLUSTER {cluster_id} (n={n_ideas})")
            print(f"{'-'*80}")

            if self._cluster_labels and cluster_id in self._cluster_labels:
                label = self._cluster_labels[cluster_id]
                print(f"\nTheme: {label.theme}")
                print(f"Description: {label.description}")
                if label.key_concepts:
                    print(f"Key concepts: {', '.join(label.key_concepts)}")

            if self._cluster_keywords and cluster_id in self._cluster_keywords:
                keywords = self._cluster_keywords[cluster_id]
                kw_str = ", ".join([f"{kw} ({score:.3f})" for kw, score in keywords[:8]])
                print(f"\nKeywords: {kw_str}")

            print(f"\nSample ideas ({min(n_samples, n_ideas)} of {n_ideas}):")
            sample_texts = random.sample(texts, min(n_samples, len(texts)))
            for i, text in enumerate(sample_texts, 1):
                cleaned = extract_embedded_text(text, self._template_prefix)
                if len(cleaned) > 100:
                    cleaned = cleaned[:97] + "..."
                print(f"  {i}. {cleaned}")

        print(f"\n{'='*80}\n")
