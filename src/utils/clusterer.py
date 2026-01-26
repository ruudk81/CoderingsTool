"""
ClustererV2 - Unified Clustering Module

A comprehensive clustering module with:
- Automatic algorithm selection (DVC + kNN knee detection)
- Optuna-based HDBSCAN optimization with GridSampler
- Agglomerative/K-means fallback for uniform density data
- Post-processing (cluster merging, BERTopic-style noise reduction)
- c-TF-IDF keyword extraction with spaCy lemmatization
- LLM-generated cluster labels

Pipeline Integration:
- Input: List[EmbeddingsModel] from step 4 (embeddings)
- Output: List[ClusterModel] via to_cluster_model()
- Cache step: "initial_clusters"

Usage:
    from utils.clusterer import ClustererV2, ClustererV2Config

    config = ClustererV2Config(
        algorithm_mode="auto",
        generate_ctfidf=True,
        generate_llm_labels=True,
    )
    clusterer = ClustererV2(embeddings_list, config=config)
    clusterer.run()

    cluster_results = clusterer.to_cluster_model()
    keywords = clusterer.get_cluster_keywords()
    labels = clusterer.get_cluster_labels()
"""

# =============================================================================
# IMPORTS
# =============================================================================

import asyncio
import math
import random
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
from joblib import Parallel, delayed
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import hdbscan
import optuna
from optuna.samplers import GridSampler
import umap

import models
from prompts import CLUSTER_DESCRIPTION_PROMPT

# Suppress common warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor.providers.gemini")


# =============================================================================
# CONSTANTS & HELPER FUNCTIONS
# =============================================================================

# Lazy-loaded spaCy model for lemmatization
_SPACY_NLP = None


def get_spacy_nlp(model_name: str = "nl_core_news_lg"):
    """
    Get or load spaCy NLP model (lazy initialization).

    Args:
        model_name: Name of spaCy model to load

    Returns:
        spaCy Language model
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load(model_name, disable=["ner", "parser"])
        except OSError:
            from spacy.cli import download
            download(model_name)
            _SPACY_NLP = spacy.load(model_name, disable=["ner", "parser"])
    return _SPACY_NLP


def log_spaced_ints(low: int, high: int, k: int = 4) -> List[int]:
    """
    Generate k log-spaced integers between low and high.

    Args:
        low: Lower bound
        high: Upper bound
        k: Number of grid points

    Returns:
        Sorted list of unique integers
    """
    if low == high:
        return [low]
    if low <= 0:
        low = 1
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def n_neighbors_bounds(N: int) -> Tuple[int, int]:
    """
    Compute n_neighbors bounds based on dataset size.

    Args:
        N: Dataset size

    Returns:
        (low, high) bounds for n_neighbors
    """
    if N < 80:
        return 5, min(15, N - 1)
    if N < 300:
        return 5, min(30, N - 1)

    low = max(5, int(round(0.005 * N)))    # ~0.5% of N
    high = min(200, int(round(0.05 * N)))  # ~5% of N, capped
    high = max(high, low)
    high = min(high, N - 1)
    low = min(low, high)
    return low, high


def n_neighbors_grid(N: int, k: int = 4) -> List[int]:
    """
    Generate n_neighbors grid for dataset of size N.

    Args:
        N: Dataset size
        k: Number of grid points

    Returns:
        Log-spaced list of n_neighbors values
    """
    low, high = n_neighbors_bounds(N)
    return log_spaced_ints(low, high, k=k)


def mcs_bounds_sqrt(N: int) -> Tuple[int, int]:
    """
    Compute min_cluster_size bounds based on sqrt(N).

    Args:
        N: Dataset size

    Returns:
        (low, high) bounds for min_cluster_size
    """
    low = max(3, int(round(0.25 * math.sqrt(N))))   # 0.25 * sqrt(N)
    high = max(low, int(round(1.0 * math.sqrt(N))))  # 1.0 * sqrt(N)
    return low, high


def mcs_grid_sqrt(N: int, k: int = 4) -> List[int]:
    """
    Generate min_cluster_size grid for dataset of size N.

    Args:
        N: Dataset size
        k: Number of grid points

    Returns:
        Log-spaced list of min_cluster_size values
    """
    low, high = mcs_bounds_sqrt(N)
    return log_spaced_ints(low, high, k=k)


def create_search_space(N: int, k: int = 4, n_components_grid: Tuple[int, ...] = (10,)) -> Dict[str, List]:
    """
    Create Optuna search space dict for GridSampler.

    Args:
        N: Dataset size
        k: Grid density
        n_components_grid: Tuple of n_components values to search

    Returns:
        Dict with 'n_neighbors', 'n_components', and 'min_cluster_size' grids
    """
    return {
        'n_neighbors': n_neighbors_grid(N, k=k),
        'n_components': list(n_components_grid),
        'min_cluster_size': mcs_grid_sqrt(N, k=k),
    }


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.

    Args:
        embeddings: L2-normalized embeddings
        n_neighbors: UMAP n_neighbors
        n_components: Target dimensionality
        min_dist: UMAP min_dist
        random_state: Random seed

    Returns:
        UMAP-reduced embeddings
    """
    warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings to unit vectors."""
    return normalize(embeddings, norm='l2', axis=1)


def extract_embedded_text(idea_text: str, template_prefix: Optional[str] = None) -> str:
    """
    Extract the text that was actually embedded (matches embedder._get_text_for_embedding).

    The idea text format from ideaExtractor is:
    [lang=...][domain=...][topic=...][perspective=...][entity=...][intent=...]
    [sentiment=...][sense=...]
    <template_prefix><unique content>

    This function extracts only the unique content (the part that varies between ideas).

    Args:
        idea_text: Full formatted idea with specifiers and template prefix
        template_prefix: The canonical phrasing prefix (e.g., "Merk X has the association")

    Returns:
        The unique content that was embedded
    """
    lines = idea_text.split('\n')
    idea_line = lines[-1] if len(lines) >= 1 else idea_text

    if template_prefix and idea_line.startswith(template_prefix):
        unique_content = idea_line[len(template_prefix):].strip()
        return unique_content if unique_content else idea_line

    return idea_line


def extract_noun_phrases_lemmatized(
    texts: List[str],
    nlp=None,
    model_name: str = "nl_core_news_lg"
) -> List[str]:
    """
    Extract lemmatized content words: ADJ, NOUN, PROPN (standalone or in phrases).

    Pattern: (ADJ | NOUN | PROPN)+

    Args:
        texts: List of text strings to process
        nlp: Pre-loaded spaCy model (will load if None)
        model_name: spaCy model name if nlp is None

    Returns:
        List of processed texts with only lemmatized content words
    """
    if nlp is None:
        nlp = get_spacy_nlp(model_name)

    processed = []

    for doc in nlp.pipe(texts, batch_size=100):
        phrases = []
        current_phrase = []

        for token in doc:
            if token.is_punct or token.is_space:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []
                continue

            if token.pos_ in ('ADJ', 'NOUN', 'PROPN'):
                current_phrase.append(token.lemma_.lower())
            else:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []

        if current_phrase:
            phrases.append(' '.join(current_phrase))

        processed.append(' '.join(phrases))

    return processed


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ClustererV2Config:
    """
    Configuration for ClustererV2 with automatic algorithm selection.

    Design decisions:
    - Diagnostics always computed (even in manual mode) for debugging
    - Persistence metrics included in output
    - Visualization optional via generate_plots flag
    """

    # ==========================================================================
    # ALGORITHM SELECTION
    # ==========================================================================

    # Mode: "auto", "hdbscan", "agglomerative", "kmeans"
    algorithm_mode: str = "auto"

    # DVC (Density Variation Coefficient) thresholds
    dvc_high_threshold: float = 0.45   # Above this → HDBSCAN recommended
    dvc_low_threshold: float = 0.25    # Below this → Agglomerative recommended
    dvc_knn_k: int = 10                # k for DVC computation

    # Hard DVC rule: force Agglomerative when DVC < this threshold
    force_agglomerative_below_dvc: float = 0.25

    # kNN Knee detection parameters
    knee_y_diff_threshold: float = 0.6  # Minimum y_difference for sharp knee
    knee_knn_k: int = 5                 # k for knee detection

    # ==========================================================================
    # UMAP CONFIGURATION
    # ==========================================================================

    umap_n_components_grid: Tuple[int, ...] = (10, 15)  # Grid of n_components to try
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_random_state: int = 42

    # UMAP precomputation for Optuna grid search
    precompute_umap: bool = True

    # n_neighbors grid density
    n_neighbors_grid_k: int = 4  # Number of log-spaced grid points

    # ==========================================================================
    # HDBSCAN / OPTUNA OPTIMIZATION
    # ==========================================================================

    use_optuna: bool = True

    # Grid search parameters
    min_cluster_size_grid_k: int = 4  # Number of log-spaced grid points
    min_samples_strategy: str = "half_mcs"  # "half_mcs" = mcs // 2

    # Constraints for Optuna pruning
    max_noise_rate: float = 0.20  # Maximum acceptable noise rate
    min_clusters: int = 3         # Minimum number of clusters required

    # Quality thresholds for conditional re-search
    enable_research: bool = True
    research_max_noise_rate: float = 0.10           # Noise threshold for condition 1
    research_min_validity: float = 0.70             # Validity threshold for condition 1
    research_cluster_deviation_threshold: float = 0.15  # |k - sqrt(n)| / sqrt(n) > this triggers

    # Extended search grid configuration
    research_mcs_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5)
    research_ms_range_multipliers: Tuple[float, float] = (0.5, 2.0)  # (low, high)
    research_ms_grid_k: int = 4  # Number of log-spaced points for MS
    research_selection_methods: Tuple[str, ...] = ('eom', 'leaf')

    # ==========================================================================
    # AGGLOMERATIVE / KMEANS PARAMETERS
    # ==========================================================================

    k_selection_strategy: str = "sqrt"  # "sqrt" = grid based on sqrt(n)
    k_grid_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)  # Multipliers for sqrt(n)

    agglomerative_linkage: str = "ward"

    # ==========================================================================
    # COHERENCE THRESHOLDS
    # ==========================================================================

    coherence_high: float = 0.95       # >= this = high coherence
    coherence_moderate: float = 0.90   # >= this = moderate coherence
    coherence_acceptable: float = 0.70 # >= this = acceptable coherence

    # ==========================================================================
    # POST-PROCESSING
    # ==========================================================================

    # Cluster merging
    enable_merging: bool = True
    merge_centroid_threshold: float = 0.95   # Centroid similarity screening
    merge_pairwise_threshold: float = 0.98   # Final merge decision

    # Noise reduction strategy
    noise_reduction_strategy: str = "embeddings"  # "embeddings" or "hdbscan"
    noise_reduction_threshold: float = 0.5  # Min similarity to assign noise to cluster

    # Legacy noise reclustering settings
    enable_noise_reclustering: bool = True
    noise_parameter_strategy: str = "adaptive"
    noise_min_cluster_size: int = 3
    noise_cohesion_threshold: float = 0.70

    # ==========================================================================
    # REPRESENTATION (c-TF-IDF)
    # ==========================================================================

    generate_ctfidf: bool = False      # Enable c-TF-IDF keyword extraction
    ctfidf_top_k: int = 15             # Top keywords per cluster
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)  # Unigrams + bigrams
    ctfidf_min_df: int = 1  # Allow cluster-unique terms
    ctfidf_bm25_weighting: bool = True
    ctfidf_reduce_frequent_words: bool = True

    # Lemmatization with spaCy
    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"
    ctfidf_pos_pattern: str = "ADJ*_NOUN+"

    # ==========================================================================
    # LLM CLUSTER LABELS
    # ==========================================================================

    generate_llm_labels: bool = False  # Enable LLM-generated cluster labels
    llm_labels_model: str = "gpt-4.1"  # Model for label generation
    llm_max_ideas_per_cluster: int = 10  # Max ideas to include in prompt

    # ==========================================================================
    # VISUALIZATION (OPTIONAL)
    # ==========================================================================

    generate_plots: bool = False
    plots_output_dir: Optional[Path] = None

    # ==========================================================================
    # PERFORMANCE
    # ==========================================================================

    pca_threshold: int = 10_000  # Apply PCA when n > threshold
    pca_variance_retained: float = 0.99

    n_jobs: int = -1  # -1 = all cores

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClusteringMetrics:
    """Comprehensive clustering quality metrics."""

    # Core metrics
    n_clusters: int = 0
    noise_rate: float = 0.0
    noise_count: int = 0

    # Density-based metrics (HDBSCAN)
    dbcv: Optional[float] = None
    relative_validity: Optional[float] = None
    mean_persistence: Optional[float] = None
    weighted_persistence: Optional[float] = None
    min_persistence: Optional[float] = None
    max_persistence: Optional[float] = None
    std_persistence: Optional[float] = None

    # Geometry metrics
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None

    # Coherence metrics (on original embeddings)
    mean_coherence: float = 0.0
    coherence_n_unacceptable: int = 0
    coherence_n_low: int = 0
    coherence_n_moderate: int = 0
    coherence_n_high: int = 0
    coherence_breakdown: str = ""
    per_cluster_coherence: Optional[List[Tuple[int, int, float]]] = None

    # Cluster size distribution
    cluster_sizes: Optional[List[int]] = None
    median_cluster_size: Optional[int] = None
    min_cluster_size: Optional[int] = None
    max_cluster_size: Optional[int] = None

    # Algorithm info
    algorithm_used: str = ""
    algorithm_params: Optional[Dict[str, Any]] = None


@dataclass
class AlgorithmRecommendation:
    """Result of automatic algorithm selection."""

    # Final recommendation
    recommended_algorithm: str  # "HDBSCAN", "AGGLOMERATIVE", "KMEANS"
    confidence: str  # "high", "medium", "low"

    # DVC analysis
    dvc_value: float
    dvc_mean_dk: float
    dvc_std_dk: float
    dvc_recommendation: str  # "HDBSCAN", "AGGLOMERATIVE", "EITHER", "AGGLOMERATIVE_FORCED"

    # Knee analysis
    knee_K: Optional[int]
    has_sharp_knee: bool
    y_difference: float
    knee_recommendation: str

    # Combined analysis
    combined_recommendation: str
    reasoning: str

    # Flag for forced algorithm selection
    is_forced: bool = False


@dataclass
class ClusterLabel:
    """LLM-generated label for a cluster."""
    cluster_id: int
    theme: str              # Short atomic label (≤10 words)
    description: str        # 1-2 sentence description
    key_concepts: List[str] # 3-5 key concepts
    n_ideas: int


@dataclass
class OptunaResult:
    """Result container for Optuna optimization."""
    best_params: Dict[str, Any]
    best_value: float
    best_labels: np.ndarray
    best_model: hdbscan.HDBSCAN
    n_trials_completed: int
    n_trials_pruned: int
    study: optuna.Study
    umap_embeddings: np.ndarray
    search_space: Dict[str, List]
    persistence_metrics: Dict[str, float]


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def apply_pca(
    embeddings: np.ndarray,
    n_components: float = 0.99,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Variance to retain (0.99 = 99%)
        random_state: Random seed

    Returns:
        (reduced_embeddings, fitted_pca_model)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca


def extract_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererV2Config
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Optional[str]]:
    """
    Extract embeddings from EmbeddingsModel list.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererV2Config

    Returns:
        embeddings: Array of shape (n_ideas, embedding_dim)
        idea_texts: List of idea text strings
        idea_indices: List of (response_idx, idea_idx) tuples for result mapping
        template_prefix: The canonical phrasing prefix (if available)
    """
    embeddings_list = []
    idea_texts = []
    idea_indices = []
    template_prefix = None

    for resp_idx, response in enumerate(input_list):
        if template_prefix is None and hasattr(response, 'template_prefix') and response.template_prefix:
            template_prefix = response.template_prefix

        if response.response_ideas:
            for idea_idx, idea in enumerate(response.response_ideas):
                if idea.idea_embedding is not None:
                    embeddings_list.append(idea.idea_embedding)
                    idea_texts.append(idea.idea if hasattr(idea, 'idea') else str(idea))
                    idea_indices.append((resp_idx, idea_idx))

    if not embeddings_list:
        raise ValueError("No embeddings found in input data")

    embeddings = np.vstack(embeddings_list)

    if config.verbose:
        print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        if template_prefix:
            prefix_display = template_prefix[:50] + "..." if len(template_prefix) > 50 else template_prefix
            print(f"Template prefix: '{prefix_display}'")

    return embeddings, idea_texts, idea_indices, template_prefix


def preprocess_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererV2Config
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int]], Optional[PCA], Optional[str]]:
    """
    Full preprocessing pipeline: extract, normalize, optionally PCA.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererV2Config

    Returns:
        embeddings_normalized: L2-normalized original embeddings
        embeddings_processed: Processed embeddings (may be PCA-reduced)
        idea_texts: List of idea text strings
        idea_indices: List of (response_idx, idea_idx) tuples
        pca_model: Fitted PCA model (or None if not applied)
        template_prefix: The canonical phrasing prefix (if available)
    """
    embeddings, idea_texts, idea_indices, template_prefix = extract_embeddings(input_list, config)
    n_samples = len(embeddings)

    embeddings_normalized = l2_normalize(embeddings)

    if config.verbose:
        print(f"L2-normalized {n_samples} embeddings")

    pca_model = None
    if n_samples > config.pca_threshold:
        if config.verbose:
            print(f"Applying PCA (n > {config.pca_threshold})...")
        embeddings_processed, pca_model = apply_pca(
            embeddings_normalized,
            n_components=config.pca_variance_retained,
            random_state=config.umap_random_state
        )
        embeddings_processed = l2_normalize(embeddings_processed)
        if config.verbose:
            print(f"PCA reduced to {embeddings_processed.shape[1]} components")
    else:
        embeddings_processed = embeddings_normalized

    return embeddings_normalized, embeddings_processed, idea_texts, idea_indices, pca_model, template_prefix


# =============================================================================
# ALGORITHM SELECTOR
# =============================================================================

class AlgorithmSelector:
    """
    Automatic algorithm selection using DVC and kNN knee detection.

    Usage:
        selector = AlgorithmSelector(config)
        dvc_result = selector.compute_dvc(embeddings_original)
        knee_result = selector.detect_knee(embeddings_reduced)
        recommendation = selector.recommend(dvc_result, knee_result)
    """

    def __init__(self, config: ClustererV2Config):
        self.config = config

    def compute_dvc(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute Density Variation Coefficient.

        DVC = std(d_k) / mean(d_k), where d_k is distance to k-th nearest neighbor.

        High DVC (>0.45) indicates varying density → HDBSCAN better
        Low DVC (<0.25) indicates uniform density → Agglomerative better

        Args:
            embeddings: L2-normalized embeddings
            k: k-th nearest neighbor (uses config default if None)

        Returns:
            Dict with dvc, mean_dk, std_dk, recommendation
        """
        k = k or self.config.dvc_knn_k
        n = len(embeddings)

        if n < k + 1:
            return {
                'dvc': np.nan,
                'mean_dk': np.nan,
                'std_dk': np.nan,
                'recommendation': 'INSUFFICIENT_DATA'
            }

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        d_k = distances[:, -1]

        mean_dk = float(np.mean(d_k))
        std_dk = float(np.std(d_k))

        if mean_dk == 0:
            return {
                'dvc': np.nan,
                'mean_dk': mean_dk,
                'std_dk': std_dk,
                'recommendation': 'ZERO_MEAN'
            }

        dvc = std_dk / mean_dk

        if dvc > self.config.dvc_high_threshold:
            recommendation = 'HDBSCAN'
        elif dvc < self.config.dvc_low_threshold:
            recommendation = 'AGGLOMERATIVE'
        else:
            recommendation = 'EITHER'

        return {
            'dvc': float(dvc),
            'mean_dk': mean_dk,
            'std_dk': std_dk,
            'recommendation': recommendation
        }

    def detect_knee(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect knee in kNN distance curve using adaptive KneeLocator.

        Args:
            embeddings: UMAP-reduced embeddings
            k: k-th nearest neighbor (uses config default if None)

        Returns:
            Dict with K, y_difference, has_sharp_knee, recommendation, etc.
        """
        k = k or self.config.knee_knn_k
        n = len(embeddings)

        kneedle_S = max(1.0, n / 100)
        interp_method = "polynomial" if n < 200 else "interp1d"

        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        k_distances = distances[:, k]

        sorted_distances = np.sort(k_distances)

        start_idx = 1
        end_idx = n - 1

        if end_idx <= start_idx:
            return {
                'K': None,
                'y_difference': 0.0,
                'has_sharp_knee': False,
                'recommendation': 'AGGLOMERATIVE_OR_KMEANS',
                'distances': sorted_distances,
                'kneedle_S': kneedle_S,
                'interp_method': interp_method
            }

        search_distances = sorted_distances[start_idx:end_idx]
        search_x = np.arange(len(search_distances))

        kneedle = KneeLocator(
            x=search_x,
            y=search_distances,
            S=kneedle_S,
            curve="convex",
            direction="increasing",
            interp_method=interp_method
        )

        K_in_window = kneedle.knee
        if K_in_window is not None:
            K = start_idx + K_in_window
        else:
            K = None

        if K is not None and kneedle.y_difference is not None and len(kneedle.y_difference) > 0:
            y_difference = float(max(kneedle.y_difference))
            has_sharp_knee = y_difference >= self.config.knee_y_diff_threshold
        else:
            y_difference = 0.0
            has_sharp_knee = False

        recommendation = "HDBSCAN" if has_sharp_knee else "AGGLOMERATIVE_OR_KMEANS"
        knee_distance = sorted_distances[K] if K is not None else None

        return {
            'K': K,
            'y_difference': y_difference,
            'has_sharp_knee': has_sharp_knee,
            'recommendation': recommendation,
            'knee_distance': knee_distance,
            'distances': sorted_distances,
            'kneedle_S': kneedle_S,
            'interp_method': interp_method
        }

    def extract_persistence_metrics(
        self,
        clusterer: hdbscan.HDBSCAN,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract cluster persistence metrics from fitted HDBSCAN model.

        Args:
            clusterer: Fitted HDBSCAN model
            labels: Cluster labels from the model

        Returns:
            Dict with mean_persistence, weighted_persistence, etc.
        """
        persistence = getattr(clusterer, "cluster_persistence_", None)
        if persistence is None:
            persistence = getattr(clusterer, "cluster_stability_", None)

        if persistence is None or len(persistence) == 0:
            return {
                'mean_persistence': np.nan,
                'min_persistence': np.nan,
                'max_persistence': np.nan,
                'std_persistence': np.nan,
                'weighted_persistence': np.nan
            }

        persistence = np.array(persistence)

        metrics = {
            'mean_persistence': float(np.mean(persistence)),
            'min_persistence': float(np.min(persistence)),
            'max_persistence': float(np.max(persistence)),
            'std_persistence': float(np.std(persistence)) if len(persistence) > 1 else 0.0,
        }

        mask = labels >= 0
        if np.any(mask):
            labels_non_noise = labels[mask]
            n = labels_non_noise.size
            max_lab = int(labels_non_noise.max())
            counts = np.bincount(labels_non_noise, minlength=max_lab + 1).astype(float)
            k = min(len(persistence), len(counts))
            if k > 0 and n > 0:
                weighted = float(np.dot(persistence[:k], counts[:k]) / n)
            else:
                weighted = np.nan
        else:
            weighted = np.nan

        metrics['weighted_persistence'] = weighted
        return metrics

    def recommend(
        self,
        dvc_result: Dict[str, Any],
        knee_result: Dict[str, Any]
    ) -> AlgorithmRecommendation:
        """
        Generate combined algorithm recommendation.

        Decision logic:
        1. If DVC < 0.25 → FORCE Agglomerative (skip HDBSCAN entirely)
        2. Otherwise, use knee detection:
           - Sharp knee (ydiff ≥ 0.6) → HDBSCAN
           - Flat knee (ydiff < 0.6) → AGGLOMERATIVE

        Args:
            dvc_result: Result from compute_dvc()
            knee_result: Result from detect_knee()

        Returns:
            AlgorithmRecommendation with full details
        """
        dvc_value = dvc_result.get('dvc', np.nan)
        has_sharp_knee = knee_result.get('has_sharp_knee', False)
        y_difference = knee_result.get('y_difference', 0.0)

        force_threshold = getattr(self.config, 'force_agglomerative_below_dvc', 0.25)

        if not np.isnan(dvc_value) and dvc_value < force_threshold:
            return AlgorithmRecommendation(
                recommended_algorithm="AGGLOMERATIVE",
                confidence="high",
                dvc_value=dvc_value,
                dvc_mean_dk=dvc_result.get('mean_dk', np.nan),
                dvc_std_dk=dvc_result.get('std_dk', np.nan),
                dvc_recommendation="AGGLOMERATIVE_FORCED",
                knee_K=knee_result.get('K'),
                has_sharp_knee=has_sharp_knee,
                y_difference=y_difference,
                knee_recommendation=knee_result.get('recommendation', 'UNKNOWN'),
                combined_recommendation="AGGLOMERATIVE_FORCED",
                reasoning=f"DVC={dvc_value:.3f} < {force_threshold} indicates uniform density → HDBSCAN skipped",
                is_forced=True
            )

        if has_sharp_knee:
            combined_recommendation = "HDBSCAN"
            recommended_algorithm = "HDBSCAN"
            confidence = "high"
            reasoning = f"Sharp knee (ydiff={y_difference:.2f} ≥ 0.6) indicates density transitions"
        else:
            combined_recommendation = "AGGLOMERATIVE"
            recommended_algorithm = "AGGLOMERATIVE"
            confidence = "medium"
            reasoning = f"Flat knee (ydiff={y_difference:.2f} < 0.6) suggests uniform density"

        return AlgorithmRecommendation(
            recommended_algorithm=recommended_algorithm,
            confidence=confidence,
            dvc_value=dvc_value,
            dvc_mean_dk=dvc_result.get('mean_dk', np.nan),
            dvc_std_dk=dvc_result.get('std_dk', np.nan),
            dvc_recommendation=dvc_result.get('recommendation', 'UNKNOWN'),
            knee_K=knee_result.get('K'),
            has_sharp_knee=has_sharp_knee,
            y_difference=y_difference,
            knee_recommendation=knee_result.get('recommendation', 'UNKNOWN'),
            combined_recommendation=combined_recommendation,
            reasoning=reasoning,
            is_forced=False
        )


# =============================================================================
# PARAMETER OPTIMIZER (Optuna)
# =============================================================================

class ParameterOptimizer:
    """
    Optuna-based hyperparameter optimization for HDBSCAN.

    Features:
    - GridSampler for exhaustive search
    - Pre-computed UMAP reductions
    - Constraint-based pruning (noise, min clusters)
    - Maximizes relative_validity_

    Usage:
        optimizer = ParameterOptimizer(config, embeddings, original_embeddings)
        result = optimizer.optimize()
        best = optimizer.get_best_result()
    """

    def __init__(
        self,
        config: ClustererV2Config,
        embeddings: np.ndarray,
        original_embeddings: np.ndarray,
        verbose: bool = True
    ):
        self.config = config
        self._embeddings = embeddings
        self._original_embeddings = original_embeddings
        self._verbose = verbose
        self._N = len(embeddings)

        self._search_space: Dict[str, List] = {}
        self._umap_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._study: Optional[optuna.Study] = None
        self._best_result: Optional[Dict[str, Any]] = None
        self._selector = AlgorithmSelector(config)

    def precompute_umap_reductions(
        self,
        n_neighbors_list: List[int],
        n_components_list: List[int]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Pre-compute UMAP reductions for all (n_neighbors, n_components) combinations in parallel."""
        combinations = [(nn, nc) for nn in n_neighbors_list for nc in n_components_list]

        if self._verbose:
            print(f"  Pre-computing {len(combinations)} UMAP reductions in parallel...")
            print(f"    n_neighbors: {n_neighbors_list}")
            print(f"    n_components: {n_components_list}")

        def compute_single_umap(n_neighbors: int, n_components: int) -> Tuple[Tuple[int, int], np.ndarray]:
            reduced = run_umap(
                self._embeddings,
                n_neighbors,
                n_components,
                self.config.umap_min_dist,
                self.config.umap_random_state
            )
            reduced_normalized = l2_normalize(reduced)
            return (n_neighbors, n_components), reduced_normalized

        n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else -1
        results = Parallel(n_jobs=n_jobs, verbose=1 if self._verbose else 0)(
            delayed(compute_single_umap)(nn, nc) for nn, nc in combinations
        )

        return {key: reduced for key, reduced in results}

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function maximizing relative_validity_."""
        n_neighbors = trial.suggest_categorical('n_neighbors', self._search_space['n_neighbors'])
        n_components = trial.suggest_categorical('n_components', self._search_space['n_components'])
        min_cluster_size = trial.suggest_categorical('min_cluster_size', self._search_space['min_cluster_size'])
        min_samples = max(1, min_cluster_size // 2)

        if self._verbose:
            print(f"  Trial {trial.number}: nn={n_neighbors}, nc={n_components}, mcs={min_cluster_size}, ms={min_samples}")

        reduced_normalized = self._umap_cache[(n_neighbors, n_components)]

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(reduced_normalized)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).sum() / len(labels)

        if n_clusters < self.config.min_clusters:
            if self._verbose:
                print(f"    PRUNED: Too few clusters ({n_clusters})")
            raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")

        if noise_rate > self.config.max_noise_rate:
            if self._verbose:
                print(f"    PRUNED: Noise too high ({noise_rate:.1%})")
            raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

        try:
            relative_validity = clusterer.relative_validity_
        except AttributeError:
            relative_validity = self._compute_dbcv(labels, reduced_normalized)

        persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)
        coherence = self._calculate_coherence(labels, self._original_embeddings)

        trial.set_user_attr('n_clusters', n_clusters)
        trial.set_user_attr('noise_rate', noise_rate)
        trial.set_user_attr('coherence', coherence)
        trial.set_user_attr('min_samples', min_samples)
        trial.set_user_attr('mean_persistence', persistence_metrics.get('mean_persistence', np.nan))
        trial.set_user_attr('weighted_persistence', persistence_metrics.get('weighted_persistence', np.nan))

        if self._verbose:
            print(f"    → rel_validity={relative_validity:.4f}, k={n_clusters}, "
                  f"noise={noise_rate:.1%}, coh={coherence:.3f}")

        return relative_validity

    def _compute_dbcv(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
        """Compute DBCV score as fallback for relative_validity_."""
        try:
            from hdbscan import validity
            mask = labels >= 0
            if mask.sum() < 2:
                return -1.0
            embeddings_f64 = embeddings[mask].astype(np.float64)
            labels_filtered = labels[mask]
            score = validity.validity_index(embeddings_f64, labels_filtered)
            return float(score)
        except Exception:
            return -1.0

    def _calculate_coherence(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
        """Calculate mean intra-cluster cosine similarity."""
        unique_labels = [l for l in set(labels) if l >= 0]
        if not unique_labels:
            return 0.0

        coherences = []
        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]

            if len(cluster_embeddings) < 2:
                coherences.append(1.0)
                continue

            similarities = cluster_embeddings @ cluster_embeddings.T
            n = len(cluster_embeddings)
            upper_tri_indices = np.triu_indices(n, k=1)
            pairwise_sims = similarities[upper_tri_indices]
            coherences.append(np.mean(pairwise_sims))

        return np.mean(coherences)

    def optimize(self) -> OptunaResult:
        """Run Optuna grid search optimization."""
        if self._verbose:
            print(f"\n[Optuna] Starting HDBSCAN optimization (N={self._N})")

        self._search_space = create_search_space(
            self._N,
            k=self.config.min_cluster_size_grid_k,
            n_components_grid=self.config.umap_n_components_grid
        )
        n_trials = (
            len(self._search_space['n_neighbors']) *
            len(self._search_space['n_components']) *
            len(self._search_space['min_cluster_size'])
        )

        if self._verbose:
            print(f"  n_neighbors grid: {self._search_space['n_neighbors']}")
            print(f"  n_components grid: {self._search_space['n_components']}")
            print(f"  min_cluster_size grid: {self._search_space['min_cluster_size']}")
            print(f"  Total trials: {n_trials}")

        if self.config.precompute_umap:
            self._umap_cache = self.precompute_umap_reductions(
                self._search_space['n_neighbors'],
                self._search_space['n_components']
            )
            if self._verbose:
                print(f"  Cached {len(self._umap_cache)} UMAP reductions")

        sampler = GridSampler(self._search_space)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(
            study_name=f"clusterer_v2_{id(self)}",
            direction='maximize',
            sampler=sampler,
        )

        self._study.optimize(self._objective, n_trials=None)

        best = self._study.best_trial
        n_neighbors = best.params['n_neighbors']
        n_components = best.params['n_components']
        min_cluster_size = best.params['min_cluster_size']
        min_samples = max(1, min_cluster_size // 2)

        reduced_normalized = self._umap_cache[(n_neighbors, n_components)]

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        best_labels = best_clusterer.fit_predict(reduced_normalized)

        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        completed = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if self._verbose:
            print(f"\n[Optuna] Optimization complete")
            print(f"  Best: nn={n_neighbors}, nc={n_components}, mcs={min_cluster_size}, ms={min_samples}")
            print(f"  relative_validity_: {best.value:.4f}")
            print(f"  Trials: {completed} completed, {pruned} pruned")

        result = OptunaResult(
            best_params={
                'n_neighbors': n_neighbors,
                'n_components': n_components,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
            },
            best_value=best.value,
            best_labels=best_labels,
            best_model=best_clusterer,
            n_trials_completed=completed,
            n_trials_pruned=pruned,
            study=self._study,
            umap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )

        result = self._check_quality_and_research(result)

        self._best_result = result
        return result

    def get_best_result(self) -> Optional[OptunaResult]:
        """Get the best result from optimization (None if not run yet)."""
        return self._best_result

    def _check_quality_and_research(self, result: OptunaResult) -> OptunaResult:
        """Check quality and trigger re-search if needed."""
        if not self.config.enable_research:
            return result

        best_trial = self._study.best_trial
        n_clusters = best_trial.user_attrs.get('n_clusters', 0)
        noise_rate = best_trial.user_attrs.get('noise_rate', 0.0)
        validity = result.best_value

        sqrt_n = math.sqrt(self._N)
        max_noise = self.config.research_max_noise_rate
        min_validity = self.config.research_min_validity
        cluster_deviation_threshold = self.config.research_cluster_deviation_threshold

        cluster_deviation = abs(n_clusters - sqrt_n) / sqrt_n if sqrt_n > 0 else 0.0

        needs_research = False
        reasons = []

        if noise_rate > max_noise and validity < min_validity:
            needs_research = True
            reasons.append(f"noise={noise_rate:.1%}>{max_noise:.0%} AND validity={validity:.3f}<{min_validity}")

        if cluster_deviation > cluster_deviation_threshold:
            needs_research = True
            reasons.append(f"cluster_deviation={cluster_deviation:.1%}>{cluster_deviation_threshold:.0%} (k={n_clusters}, expected≈{sqrt_n:.0f})")

        if not needs_research:
            if self._verbose:
                print(f"  Quality check PASSED: k={n_clusters} (expected≈{sqrt_n:.0f}), "
                      f"noise={noise_rate:.1%}, validity={validity:.3f}")
            return result

        if self._verbose:
            print(f"\n[Research] Quality check FAILED: {', '.join(reasons)}")
            print(f"  Triggering extended search...")

        return self._run_extended_search(result)

    def _run_extended_search(self, initial_result: OptunaResult) -> OptunaResult:
        """Run extended search with expanded parameters using Optuna GridSampler."""
        best_n_neighbors = initial_result.best_params['n_neighbors']
        best_n_components = initial_result.best_params.get('n_components', self.config.umap_n_components_grid[0])
        best_mcs = initial_result.best_params['min_cluster_size']
        best_ms = initial_result.best_params.get('min_samples', best_mcs // 2)
        reduced_normalized = self._umap_cache[(best_n_neighbors, best_n_components)]

        mcs_multipliers = self.config.research_mcs_multipliers
        mcs_options = sorted(set(
            max(3, int(round(best_mcs * mult)))
            for mult in mcs_multipliers
        ))

        ms_low_mult, ms_high_mult = self.config.research_ms_range_multipliers
        ms_low = max(1, int(round(best_ms * ms_low_mult)))
        ms_high = max(ms_low, int(round(best_ms * ms_high_mult)))
        ms_options = log_spaced_ints(ms_low, ms_high, k=self.config.research_ms_grid_k)

        selection_methods = list(self.config.research_selection_methods)

        max_mcs = max(mcs_options)
        ms_options = [ms for ms in ms_options if ms <= max_mcs]

        extended_search_space = {
            'min_cluster_size': mcs_options,
            'min_samples': ms_options,
            'cluster_selection_method': selection_methods,
        }

        n_trials_total = len(mcs_options) * len(ms_options) * len(selection_methods)

        if self._verbose:
            print(f"\n[Extended Search - Optuna GridSampler]")
            print(f"  Based on best: nn={best_n_neighbors}, nc={best_n_components}, mcs={best_mcs}, ms={best_ms}")
            print(f"  MCS grid: {mcs_options}")
            print(f"  MS grid: {ms_options}")
            print(f"  Selection methods: {selection_methods}")
            print(f"  Total trials: {n_trials_total}")

        def extended_objective(trial: optuna.Trial) -> float:
            mcs = trial.suggest_categorical('min_cluster_size', extended_search_space['min_cluster_size'])
            ms = trial.suggest_categorical('min_samples', extended_search_space['min_samples'])
            method = trial.suggest_categorical('cluster_selection_method', extended_search_space['cluster_selection_method'])

            if ms > mcs:
                raise optuna.TrialPruned(f"Invalid: ms={ms} > mcs={mcs}")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric='euclidean',
                cluster_selection_method=method,
                gen_min_span_tree=True,
            )
            labels = clusterer.fit_predict(reduced_normalized)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_rate = (labels == -1).sum() / len(labels)

            if n_clusters < self.config.min_clusters:
                if self._verbose:
                    print(f"    PRUNED: Too few clusters ({n_clusters})")
                raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")

            if noise_rate > self.config.max_noise_rate:
                if self._verbose:
                    print(f"    PRUNED: Noise too high ({noise_rate:.1%})")
                raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

            try:
                validity = clusterer.relative_validity_
            except AttributeError:
                validity = self._compute_dbcv(labels, reduced_normalized)

            coherence = self._calculate_coherence(labels, self._original_embeddings)

            trial.set_user_attr('n_clusters', n_clusters)
            trial.set_user_attr('noise_rate', noise_rate)
            trial.set_user_attr('coherence', coherence)
            trial.set_user_attr('labels', labels.tolist())

            if self._verbose:
                improved = "★" if validity > initial_result.best_value else " "
                print(f"  {improved} Trial {trial.number}: {method}, mcs={mcs}, ms={ms} → "
                      f"val={validity:.4f}, k={n_clusters}, noise={noise_rate:.1%}, coh={coherence:.3f}")

            return validity

        extended_sampler = GridSampler(extended_search_space)
        extended_study = optuna.create_study(
            study_name=f"clusterer_v2_extended_{id(self)}",
            direction='maximize',
            sampler=extended_sampler,
        )
        extended_study.optimize(extended_objective, n_trials=None)

        completed = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if self._verbose:
            print(f"\n  Extended search complete: {completed} completed, {pruned} pruned")

        if completed == 0:
            if self._verbose:
                print(f"  No valid trials found, keeping initial result")
            return initial_result

        best_extended = extended_study.best_trial

        if best_extended.value <= initial_result.best_value:
            if self._verbose:
                print(f"  No improvement found (best extended: {best_extended.value:.4f} <= initial: {initial_result.best_value:.4f})")
            return initial_result

        mcs = best_extended.params['min_cluster_size']
        ms = best_extended.params['min_samples']
        method = best_extended.params['cluster_selection_method']

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric='euclidean',
            cluster_selection_method=method,
            gen_min_span_tree=True,
        )
        best_labels = best_clusterer.fit_predict(reduced_normalized)

        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        if self._verbose:
            print(f"  Found better: {method}, mcs={mcs}, ms={ms}, validity={best_extended.value:.4f}")

        return OptunaResult(
            best_params={
                'n_neighbors': best_n_neighbors,
                'n_components': best_n_components,
                'min_cluster_size': mcs,
                'min_samples': ms,
                'cluster_selection_method': method,
            },
            best_value=best_extended.value,
            best_labels=best_labels,
            best_model=best_clusterer,
            n_trials_completed=initial_result.n_trials_completed + completed,
            n_trials_pruned=initial_result.n_trials_pruned + pruned,
            study=self._study,
            umap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )


# =============================================================================
# POST-PROCESSING
# =============================================================================

class UnionFind:
    """Union-Find data structure for transitive closure in cluster merging."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def get_components(self) -> Dict[int, int]:
        """Return mapping from element to component representative."""
        return {e: self.find(e) for e in self.parent}


def compute_cluster_centroids(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Compute centroid for each cluster."""
    centroids = {}
    sizes = {}

    unique_labels = [l for l in set(labels) if l >= 0]
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        sizes[label] = len(cluster_embeddings)

        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid

    return centroids, sizes


def pairwise_cluster_similarity(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    embeddings: np.ndarray
) -> Dict[str, float]:
    """Calculate pairwise similarity statistics between two clusters."""
    emb_a = embeddings[indices_a]
    emb_b = embeddings[indices_b]

    sim_matrix = emb_a @ emb_b.T
    all_sims = sim_matrix.flatten()

    return {
        'q25': float(np.percentile(all_sims, 25)),
        'q50': float(np.percentile(all_sims, 50)),
        'q75': float(np.percentile(all_sims, 75)),
        'mean': float(np.mean(all_sims))
    }


def renumber_clusters(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels to be sequential starting from 0."""
    new_labels = labels.copy()
    unique_labels = sorted(set(labels) - {-1})

    mapping = {old: new for new, old in enumerate(unique_labels)}
    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels


def reduce_noise_by_embedding_similarity(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    BERTopic-style noise reduction: assign noise points to nearest cluster by embedding similarity.

    Args:
        labels: Cluster labels with -1 for noise points
        embeddings: L2-normalized embeddings
        threshold: Minimum cosine similarity to assign noise to cluster
        verbose: Print progress

    Returns:
        Tuple of (updated_labels, stats_dict)
    """
    noise_mask = labels == -1
    n_total_noise = noise_mask.sum()

    if n_total_noise == 0:
        if verbose:
            print("  No noise points to reduce")
        return labels, {'n_noise_initial': 0, 'n_assigned': 0, 'n_noise_final': 0}

    if verbose:
        print(f"\n[Noise Reduction - Embedding Similarity]")
        print(f"  Initial noise points: {n_total_noise}")
        print(f"  Similarity threshold: {threshold}")

    centroids, sizes = compute_cluster_centroids(labels, embeddings)

    if len(centroids) == 0:
        if verbose:
            print("  No clusters found - cannot reduce noise")
        return labels, {'n_noise_initial': n_total_noise, 'n_assigned': 0, 'n_noise_final': n_total_noise}

    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])

    noise_indices = np.where(noise_mask)[0]
    noise_embeddings = embeddings[noise_indices]

    similarities = noise_embeddings @ centroid_matrix.T

    best_cluster_indices = np.argmax(similarities, axis=1)
    best_scores = np.max(similarities, axis=1)

    new_labels = labels.copy()
    n_assigned = 0

    for i, (noise_idx, cluster_idx, score) in enumerate(zip(noise_indices, best_cluster_indices, best_scores)):
        if score >= threshold:
            new_labels[noise_idx] = cluster_ids[cluster_idx]
            n_assigned += 1

    n_noise_final = (new_labels == -1).sum()
    assignment_rate = n_assigned / n_total_noise if n_total_noise > 0 else 0.0

    if verbose:
        print(f"  Assigned: {n_assigned} ({assignment_rate:.1%})")
        print(f"  Remaining noise: {n_noise_final} ({n_noise_final/len(labels):.1%})")

    stats = {
        'n_noise_initial': n_total_noise,
        'n_assigned': n_assigned,
        'n_noise_final': n_noise_final,
        'assignment_rate': assignment_rate
    }

    return new_labels, stats


def merge_similar_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    config: ClustererV2Config,
    verbose: bool = True
) -> np.ndarray:
    """
    Merge clusters using graph-based transitive closure with union-find.

    Args:
        labels: Initial cluster assignments
        embeddings: L2-normalized embeddings
        config: ClustererV2Config
        verbose: Print progress

    Returns:
        Updated cluster labels with merged clusters
    """
    if not config.enable_merging:
        return labels

    centroids, sizes = compute_cluster_centroids(labels, embeddings)
    n_initial_clusters = len(centroids)

    if n_initial_clusters < 2:
        if verbose:
            print("  Less than 2 clusters - skipping merge")
        return labels

    if verbose:
        print(f"\n[Cluster Merging]")
        print(f"  Initial clusters: {n_initial_clusters}")
        print(f"  Centroid threshold: {config.merge_centroid_threshold}")
        print(f"  Pairwise threshold: {config.merge_pairwise_threshold}")

    cluster_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_to_indices[int(label)].append(i)

    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
    centroid_similarities = centroid_matrix @ centroid_matrix.T

    candidates = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            sim = centroid_similarities[i, j]
            if sim >= config.merge_centroid_threshold:
                candidates.append((cluster_ids[i], cluster_ids[j], sim))

    if verbose:
        print(f"  Candidate pairs (centroid >= {config.merge_centroid_threshold}): {len(candidates)}")

    if not candidates:
        if verbose:
            print("  No similar clusters found - no merging needed")
        return labels

    merge_edges = []
    for cluster_a, cluster_b, centroid_sim in candidates:
        indices_a = np.array(cluster_to_indices[cluster_a])
        indices_b = np.array(cluster_to_indices[cluster_b])

        stats = pairwise_cluster_similarity(indices_a, indices_b, embeddings)
        quantile_mean = np.mean([stats['q25'], stats['q50'], stats['q75']])

        if quantile_mean >= config.merge_pairwise_threshold:
            merge_edges.append((cluster_a, cluster_b, centroid_sim, quantile_mean))
            if verbose:
                print(f"    ✓ Merge {cluster_a}↔{cluster_b} | "
                      f"sizes: {sizes[cluster_a]}, {sizes[cluster_b]} | "
                      f"centroid: {centroid_sim:.3f} | quantile_mean: {quantile_mean:.3f}")

    if not merge_edges:
        if verbose:
            print("  No merge-worthy pairs found")
        return labels

    uf = UnionFind(cluster_ids)
    for cluster_a, cluster_b, _, _ in merge_edges:
        uf.union(cluster_a, cluster_b)

    component_map = uf.get_components()

    labels_merged = labels.copy()
    for old_id, component_id in component_map.items():
        labels_merged[labels == old_id] = component_id

    labels_final = renumber_clusters(labels_merged)

    unique_components = set(component_map.values())
    n_final_clusters = len(unique_components)
    n_merged = n_initial_clusters - n_final_clusters

    if verbose:
        print(f"  Merging complete:")
        print(f"    Initial: {n_initial_clusters} → Final: {n_final_clusters}")
        print(f"    Reduction: {n_merged} clusters removed")

    return labels_final


def assess_noise_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cohesion_threshold: float
) -> List[int]:
    """Assess quality of noise-derived clusters and return valid ones."""
    valid_clusters = []
    unique_labels = [l for l in set(labels) if l >= 0]

    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 2:
            valid_clusters.append(label)
            continue

        similarities = cluster_embeddings @ cluster_embeddings.T
        n = len(cluster_embeddings)
        upper_tri = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(similarities[upper_tri]))

        if mean_sim >= cohesion_threshold:
            valid_clusters.append(label)

    return valid_clusters


def recluster_noise(
    labels: np.ndarray,
    umap_embeddings: np.ndarray,
    original_embeddings: np.ndarray,
    config: ClustererV2Config,
    verbose: bool = True
) -> np.ndarray:
    """Two-pass clustering: Attempt to find viable clusters among noise points."""
    if not config.enable_noise_reclustering:
        return labels

    noise_mask = labels == -1
    n_total_noise = noise_mask.sum()

    min_total = 10
    if n_total_noise < min_total:
        if verbose:
            print(f"  Skipping noise reclustering: only {n_total_noise} noise points (minimum: {min_total})")
        return labels

    if verbose:
        print(f"\n[Noise Reclustering]")
        print(f"  Total noise points: {n_total_noise}")

    noise_mcs = max(3, config.noise_min_cluster_size)
    noise_ms = max(1, noise_mcs // 2)

    if verbose:
        print(f"  Parameters: min_cluster_size={noise_mcs}, min_samples={noise_ms}")

    U_noise = umap_embeddings[noise_mask]

    noise_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=noise_mcs,
        min_samples=noise_ms,
        metric='euclidean',
        cluster_selection_method='leaf',
        gen_min_span_tree=True
    )
    noise_labels = noise_hdbscan.fit_predict(U_noise)

    original_noise = original_embeddings[noise_mask]
    valid_noise_clusters = assess_noise_cluster_quality(
        original_noise, noise_labels, config.noise_cohesion_threshold
    )

    if len(valid_noise_clusters) == 0:
        if verbose:
            print(f"  No viable clusters found in noise")
        return labels

    labels_updated = labels.copy()
    max_main_cluster = labels[labels >= 0].max() if np.any(labels >= 0) else -1
    next_cluster_id = max_main_cluster + 1

    noise_indices = np.where(noise_mask)[0]
    n_recovered = 0

    for old_noise_cluster_id in valid_noise_clusters:
        cluster_mask_in_noise = noise_labels == old_noise_cluster_id
        global_indices = noise_indices[cluster_mask_in_noise]
        labels_updated[global_indices] = next_cluster_id
        n_recovered += len(global_indices)
        next_cluster_id += 1

    n_noise_clusters = len(valid_noise_clusters)
    recovery_rate = n_recovered / n_total_noise if n_total_noise > 0 else 0.0
    final_noise = (labels_updated == -1).sum()

    if verbose:
        print(f"  Viable clusters discovered: {n_noise_clusters}")
        print(f"  Points recovered: {n_recovered} ({recovery_rate:.1%})")
        print(f"  Residual noise: {final_noise} ({final_noise/len(labels):.1%})")

    return labels_updated


# =============================================================================
# QUALITY METRICS CALCULATOR
# =============================================================================

class ClusterQualityMetrics:
    """Calculator for comprehensive clustering quality metrics."""

    def __init__(self, config: ClustererV2Config):
        self.config = config

    def calculate_coherence(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[float, Dict[str, int], List[Tuple[int, int, float]]]:
        """Calculate mean coherence and per-cluster breakdown."""
        unique_labels = [l for l in set(labels) if l >= 0]

        if not unique_labels:
            return 0.0, {'n_unacceptable': 0, 'n_low': 0, 'n_moderate': 0, 'n_high': 0}, []

        per_cluster = []
        n_unacceptable = 0
        n_low = 0
        n_moderate = 0
        n_high = 0

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            size = len(cluster_embeddings)

            if size < 2:
                coherence = 1.0
            else:
                similarities = cluster_embeddings @ cluster_embeddings.T
                n = len(cluster_embeddings)
                upper_tri_indices = np.triu_indices(n, k=1)
                pairwise_sims = similarities[upper_tri_indices]
                coherence = float(np.mean(pairwise_sims))

            per_cluster.append((label, size, coherence))

            if coherence < self.config.coherence_acceptable:
                n_unacceptable += 1
            elif coherence < self.config.coherence_moderate:
                n_low += 1
            elif coherence < self.config.coherence_high:
                n_moderate += 1
            else:
                n_high += 1

        per_cluster.sort(key=lambda x: x[0])

        coherences = [coh for _, _, coh in per_cluster]
        mean_coherence = float(np.mean(coherences)) if coherences else 0.0

        breakdown = {
            'n_unacceptable': n_unacceptable,
            'n_low': n_low,
            'n_moderate': n_moderate,
            'n_high': n_high
        }

        return mean_coherence, breakdown, per_cluster

    def compute_dbcv(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
        """Compute DBCV (Density-Based Clustering Validation) score."""
        try:
            from hdbscan import validity
            mask = labels >= 0
            if mask.sum() < 2:
                return -1.0
            embeddings_f64 = embeddings[mask].astype(np.float64)
            labels_filtered = labels[mask]
            score = validity.validity_index(embeddings_f64, labels_filtered)
            return float(score)
        except Exception:
            return np.nan

    def compute_geometry_metrics(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Compute geometry-based metrics (silhouette, CH, DB)."""
        mask = labels >= 0
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        metrics = {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }

        if mask.sum() < 2 or n_clusters < 2:
            return metrics

        try:
            metrics['silhouette'] = silhouette_score(embeddings[mask], labels[mask])
        except Exception:
            pass

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings[mask], labels[mask])
        except Exception:
            pass

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings[mask], labels[mask])
        except Exception:
            pass

        return metrics

    def compute_cluster_sizes(self, labels: np.ndarray) -> Dict[str, Any]:
        """Compute cluster size distribution."""
        unique_labels = [l for l in set(labels) if l >= 0]
        sizes = [int((labels == l).sum()) for l in unique_labels]

        if not sizes:
            return {
                'cluster_sizes': [],
                'median_cluster_size': None,
                'min_cluster_size': None,
                'max_cluster_size': None
            }

        return {
            'cluster_sizes': sizes,
            'median_cluster_size': int(np.median(sizes)),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes)
        }

    def calculate_all(
        self,
        labels: np.ndarray,
        embeddings_reduced: np.ndarray,
        embeddings_original: np.ndarray,
        hdbscan_model: Optional[hdbscan.HDBSCAN] = None,
        algorithm_used: str = "",
        algorithm_params: Optional[Dict[str, Any]] = None
    ) -> ClusteringMetrics:
        """Calculate all configured metrics."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        noise_rate = noise_count / len(labels) if len(labels) > 0 else 0.0

        mean_coherence, breakdown, per_cluster = self.calculate_coherence(
            labels, embeddings_original
        )

        breakdown_parts = []
        if breakdown['n_unacceptable'] > 0:
            breakdown_parts.append(f"{breakdown['n_unacceptable']} unacceptable")
        if breakdown['n_low'] > 0:
            breakdown_parts.append(f"{breakdown['n_low']} low")
        if breakdown['n_moderate'] > 0:
            breakdown_parts.append(f"{breakdown['n_moderate']} moderate")
        if breakdown['n_high'] > 0:
            breakdown_parts.append(f"{breakdown['n_high']} high")
        coherence_breakdown_str = ", ".join(breakdown_parts) if breakdown_parts else "no clusters"

        dbcv = self.compute_dbcv(labels, embeddings_reduced)

        geometry = self.compute_geometry_metrics(labels, embeddings_reduced)

        sizes = self.compute_cluster_sizes(labels)

        persistence_metrics = {}
        relative_validity = None
        if hdbscan_model is not None:
            selector = AlgorithmSelector(self.config)
            persistence_metrics = selector.extract_persistence_metrics(hdbscan_model, labels)

            try:
                relative_validity = float(hdbscan_model.relative_validity_)
            except AttributeError:
                relative_validity = None

        return ClusteringMetrics(
            n_clusters=n_clusters,
            noise_rate=noise_rate,
            noise_count=noise_count,
            dbcv=dbcv,
            relative_validity=relative_validity,
            mean_persistence=persistence_metrics.get('mean_persistence'),
            weighted_persistence=persistence_metrics.get('weighted_persistence'),
            min_persistence=persistence_metrics.get('min_persistence'),
            max_persistence=persistence_metrics.get('max_persistence'),
            std_persistence=persistence_metrics.get('std_persistence'),
            silhouette=geometry['silhouette'],
            calinski_harabasz=geometry['calinski_harabasz'],
            davies_bouldin=geometry['davies_bouldin'],
            mean_coherence=mean_coherence,
            coherence_n_unacceptable=breakdown['n_unacceptable'],
            coherence_n_low=breakdown['n_low'],
            coherence_n_moderate=breakdown['n_moderate'],
            coherence_n_high=breakdown['n_high'],
            coherence_breakdown=coherence_breakdown_str,
            per_cluster_coherence=per_cluster,
            cluster_sizes=sizes['cluster_sizes'],
            median_cluster_size=sizes['median_cluster_size'],
            min_cluster_size=sizes['min_cluster_size'],
            max_cluster_size=sizes['max_cluster_size'],
            algorithm_used=algorithm_used,
            algorithm_params=algorithm_params
        )


# =============================================================================
# REPRESENTATION ENGINE (c-TF-IDF)
# =============================================================================

class RepresentationEngine:
    """
    c-TF-IDF keyword extraction for clusters.

    Wraps the existing CTfidfRepresentation module from experiments/representation/.
    """

    def __init__(self, config: ClustererV2Config):
        self.config = config
        self._ctfidf = None

    def _ensure_ctfidf(self):
        """Lazy initialization of c-TF-IDF model."""
        if self._ctfidf is None:
            try:
                import sys
                import os
                parent_path = os.path.dirname(os.path.dirname(__file__))
                experiments_path = os.path.join(parent_path, 'experiments')
                if experiments_path not in sys.path:
                    sys.path.insert(0, experiments_path)

                from representation.ctfidf_representation import CTfidfRepresentation

                self._ctfidf = CTfidfRepresentation(
                    top_k=self.config.ctfidf_top_k,
                    bm25_weighting=self.config.ctfidf_bm25_weighting,
                    reduce_frequent_words=self.config.ctfidf_reduce_frequent_words,
                    ngram_range=self.config.ctfidf_ngram_range,
                    min_df=self.config.ctfidf_min_df,
                    max_df=0.95,
                    language="nl"
                )
            except ImportError as e:
                raise ImportError(
                    f"Could not import CTfidfRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def extract_keywords(
        self,
        cluster_texts: Dict[int, List[str]],
        template_prefix: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top keywords for each cluster using c-TF-IDF.

        Args:
            cluster_texts: Dict mapping cluster_id to list of idea texts
            template_prefix: The canonical phrasing prefix to strip
            verbose: Print progress

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples
        """
        if not self.config.generate_ctfidf:
            return {}

        self._ensure_ctfidf()

        cleaned_clusters = {}
        for cluster_id, texts in cluster_texts.items():
            cleaned_clusters[cluster_id] = [
                extract_embedded_text(t, template_prefix) for t in texts
            ]

        if verbose and template_prefix:
            print(f"  Using template_prefix for text extraction: '{template_prefix[:50]}...'" if len(template_prefix) > 50 else f"  Using template_prefix for text extraction: '{template_prefix}'")

        if self.config.ctfidf_use_lemmatization:
            if verbose:
                print("  Applying spaCy lemmatization (ADJ | NOUN | PROPN)...")

            all_texts = []
            cluster_offsets = {}
            offset = 0
            for cluster_id, texts in cleaned_clusters.items():
                cluster_offsets[cluster_id] = (offset, offset + len(texts))
                all_texts.extend(texts)
                offset += len(texts)

            lemmatized_all = extract_noun_phrases_lemmatized(
                all_texts,
                model_name=self.config.ctfidf_spacy_model
            )

            lemmatized_clusters = {}
            for cluster_id, (start, end) in cluster_offsets.items():
                lemmatized_clusters[cluster_id] = lemmatized_all[start:end]

            cleaned_clusters = lemmatized_clusters

        keywords = self._ctfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        return keywords

    def extract_keywords_from_labels(
        self,
        labels: np.ndarray,
        idea_texts: List[str],
        template_prefix: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract keywords given cluster labels and idea texts."""
        cluster_texts = {}
        for i, (label, text) in enumerate(zip(labels, idea_texts)):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(text)

        return self.extract_keywords(cluster_texts, template_prefix=template_prefix, verbose=verbose)


# =============================================================================
# LABEL GENERATOR (LLM)
# =============================================================================

class LabelGenerator:
    """
    LLM-based cluster label generation.

    Generates short thematic labels, descriptions, and key concepts for each cluster.
    """

    def __init__(self, config: ClustererV2Config):
        self.config = config

    def generate_label(
        self,
        cluster_id: int,
        texts: List[str],
        keywords: Optional[List[Tuple[str, float]]] = None,
        survey_question: str = "",
        language: str = "Dutch",
        verbose: bool = False
    ) -> Optional[ClusterLabel]:
        """
        Generate a label for a single cluster using LLM.

        Args:
            cluster_id: Cluster ID
            texts: List of idea texts in this cluster
            keywords: Optional c-TF-IDF keywords
            survey_question: The research question/survey context
            language: Language for the label
            verbose: Print progress

        Returns:
            ClusterLabel or None on failure
        """
        try:
            from pydantic import BaseModel, Field
            from utils.llm import create_client, llm_create_async

            # Prepare keywords section
            if keywords:
                keywords_section = "\n".join([f"- {kw} (score: {score:.3f})" for kw, score in keywords[:10]])
            else:
                keywords_section = "(none available)"

            # Prepare ideas list (limit to max_ideas_per_cluster)
            max_ideas = self.config.llm_max_ideas_per_cluster
            sample_texts = texts[:max_ideas] if len(texts) > max_ideas else texts
            ideas_list = "\n".join([f"- {extract_embedded_text(t)}" for t in sample_texts])

            # Format prompt
            prompt = CLUSTER_DESCRIPTION_PROMPT.format(
                language=language,
                cluster_id=cluster_id,
                survey_question=survey_question or "(not specified)",
                num_ideas=len(texts),
                keywords_section=keywords_section,
                ideas_list=ideas_list
            )

            # Define response model
            class ClusterLabelResponse(BaseModel):
                theme: str = Field(description="Short atomic thematic label (≤10 words)")
                description: str = Field(description="Clear description (1-2 sentences)")
                key_concepts: List[str] = Field(description="List of 3-5 key concepts")

            # Call LLM
            client = create_client(model=self.config.llm_labels_model, async_mode=False)
            response = client.chat.completions.create(
                model=self.config.llm_labels_model,
                messages=[{"role": "user", "content": prompt}],
                response_model=ClusterLabelResponse,
            )

            return ClusterLabel(
                cluster_id=cluster_id,
                theme=response.theme,
                description=response.description,
                key_concepts=response.key_concepts,
                n_ideas=len(texts)
            )

        except Exception as e:
            if verbose:
                print(f"    Error generating label for cluster {cluster_id}: {e}")
            return None

    def generate_all_labels(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        survey_question: str = "",
        language: str = "Dutch",
        verbose: bool = False
    ) -> Dict[int, ClusterLabel]:
        """
        Generate labels for all clusters.

        Args:
            cluster_texts: Dict mapping cluster_id to list of idea texts
            cluster_keywords: Optional dict of c-TF-IDF keywords per cluster
            survey_question: The research question/survey context
            language: Language for labels
            verbose: Print progress

        Returns:
            Dict mapping cluster_id to ClusterLabel
        """
        labels = {}

        for cluster_id, texts in sorted(cluster_texts.items()):
            keywords = cluster_keywords.get(cluster_id) if cluster_keywords else None

            if verbose:
                print(f"  Generating label for cluster {cluster_id} ({len(texts)} ideas)...")

            label = self.generate_label(
                cluster_id=cluster_id,
                texts=texts,
                keywords=keywords,
                survey_question=survey_question,
                language=language,
                verbose=verbose
            )

            if label:
                labels[cluster_id] = label
                if verbose:
                    print(f"    Theme: {label.theme}")

        return labels


# =============================================================================
# MAIN CLUSTERER CLASS
# =============================================================================

class ClustererV2:
    """
    Enhanced clustering module with automatic algorithm selection,
    Optuna-based optimization, and integrated quality metrics.

    Key Features:
    1. Automatic algorithm selection (DVC + kNN knee + persistence)
    2. Bayesian HDBSCAN optimization via Optuna GridSampler
    3. Coherence-based quality metrics on original embeddings
    4. Optional c-TF-IDF keyword extraction
    5. Optional LLM-generated cluster labels

    Usage:
        clusterer = ClustererV2(input_list, config=ClustererV2Config())
        clusterer.run()
        results = clusterer.to_cluster_model()
        keywords = clusterer.get_cluster_keywords()
        labels = clusterer.get_cluster_labels()
    """

    def __init__(
        self,
        input_list: List[models.EmbeddingsModel],
        config: Optional[ClustererV2Config] = None
    ):
        """
        Initialize ClustererV2.

        Args:
            input_list: List of EmbeddingsModel with idea_embedding populated
            config: Configuration (uses defaults if None)
        """
        self.config = config or ClustererV2Config()
        self._input_list = input_list
        self._verbose = self.config.verbose

        # Will be populated during run()
        self._embeddings_original: Optional[np.ndarray] = None
        self._embeddings_processed: Optional[np.ndarray] = None
        self._idea_texts: Optional[List[str]] = None
        self._idea_indices: Optional[List[Tuple[int, int]]] = None
        self._template_prefix: Optional[str] = None
        self._labels: Optional[np.ndarray] = None
        self._umap_embeddings: Optional[np.ndarray] = None
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

    def run(self) -> 'ClustererV2':
        """
        Execute the complete clustering pipeline.

        Returns:
            self (for method chaining)
        """
        if self._verbose:
            print("=" * 70)
            print("ClustererV2 Clustering Pipeline")
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

        dvc_result = self._selector.compute_dvc(self._embeddings_original)
        if self._verbose:
            dvc_val = dvc_result['dvc']
            if not np.isnan(dvc_val):
                print(f"  DVC = {dvc_val:.3f} → {dvc_result['recommendation']}")

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
            n_neighbors_list = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k)
            trial_n_neighbors = n_neighbors_list[len(n_neighbors_list) // 2]

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

    def _run_agglomerative(self):
        """Phase 3b: Run Agglomerative clustering."""
        if self._verbose:
            print("\n[Phase 3] Agglomerative Clustering")

        n_neighbors = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k)[1]
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
            clusterer = AgglomerativeClustering(n_clusters=k_grid[0])
            best_labels = clusterer.fit_predict(self._umap_embeddings)
            best_k = k_grid[0]

        self._labels = best_labels
        self._algorithm_used = "Agglomerative"
        self._algorithm_params = {'n_clusters': best_k, 'linkage': self.config.agglomerative_linkage}

        if self._verbose:
            print(f"  Best: k={best_k}, silhouette={best_sil:.3f}")

    def _run_kmeans(self):
        """Phase 3c: Run K-means clustering."""
        if self._verbose:
            print("\n[Phase 3] K-means Clustering")

        n_neighbors = n_neighbors_grid(self._N, k=self.config.n_neighbors_grid_k)[1]
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

            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
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
            clusterer = KMeans(n_clusters=k_grid[0], random_state=42, n_init=10)
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
        """
        Convert internal results to ClusterModel list (pipeline-compatible).

        Returns:
            List of ClusterModel instances with cluster assignments
        """
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
        """Get LLM-generated labels for each cluster."""
        return self._cluster_labels

    def print_all_clusters(self, n_samples: int = 5):
        """
        Print all clusters with sample ideas.

        Args:
            n_samples: Number of sample ideas to show per cluster
        """
        if self._labels is None:
            raise RuntimeError("Must call run() before print_all_clusters()")

        cluster_texts = {}
        for i, (label, text) in enumerate(zip(self._labels, self._idea_texts)):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(text)

        print(f"\n{'='*80}")
        print(f"ALL CLUSTERS ({len(cluster_texts)} clusters)")
        print(f"{'='*80}")

        for cluster_id in sorted(cluster_texts.keys()):
            texts = cluster_texts[cluster_id]
            n_ideas = len(texts)

            print(f"\n{'─'*80}")
            print(f"CLUSTER {cluster_id} (n={n_ideas})")
            print(f"{'─'*80}")

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


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Keep old Clusterer class for backwards compatibility
Clusterer = ClustererV2


def clean_cluster_ideas(cluster_results: List[models.ClusterModel]) -> List[models.ClusterModel]:
    """Clean cluster idea texts by removing bracketed annotations and normalizing whitespace.

    Args:
        cluster_results: List of ClusterModel objects with idea texts to clean

    Returns:
        List of ClusterModel objects with cleaned idea texts
    """
    cleaned_results = []

    for result in cluster_results:
        cleaned_response_ideas = []

        if result.response_ideas:
            for idea_submodel in result.response_ideas:
                # Extract and clean idea text
                cleaned_idea = idea_submodel.idea
                cleaned_idea = re.sub(r"\[.*?\]", "", cleaned_idea)
                cleaned_idea = re.sub(r"\s+", " ", cleaned_idea).strip()

                # Create new ClusterSubmodel with cleaned text
                cleaned_submodel = models.ClusterSubmodel(
                    idea_id=idea_submodel.idea_id,
                    idea=cleaned_idea,
                    idea_embedding=idea_submodel.idea_embedding,
                    initial_cluster=idea_submodel.initial_cluster,
                    expanded_cluster=idea_submodel.expanded_cluster,
                    cluster_theme=idea_submodel.cluster_theme
                )
                cleaned_response_ideas.append(cleaned_submodel)

        # Create new ClusterModel with cleaned ideas
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
