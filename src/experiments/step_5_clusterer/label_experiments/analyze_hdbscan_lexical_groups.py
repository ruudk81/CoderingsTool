#%% ============================================================================
# HDBSCAN LEXICAL GROUPS ANALYSIS - LAYER 2
# ============================================================================
"""
Layer 2 clustering experiment using HDBSCAN fragmentation:
- Loads cached UMAP embeddings + EOM results from Layer 1
- Deepens each EOM cluster via HDBSCAN with DBCV-optimized grid search
- Grid: min_samples and min_cluster_size share same values (1:1 mapping)
- Grid formula: ceil(log(n)) to ceil(2*log(n)), log-spaced, k=3 points
- Builds c-TF-IDF on HDBSCAN sub-clusters for taxonomy_phrase
- Runs HDBSCAN on c-TF-IDF vectors to find lexical groups

This enables exploring micro-themes within each stable EOM cluster using
a density-based approach with DBCV-optimized parameter selection.

Usage: Open in VS Code and run cells interactively.
"""

#%% IMPORTS AND SETUP
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

import numpy as np
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

project_root = Path(__file__).parent.parent.parent.parent.parent


@dataclass
class GlobalCtfidfArtifacts:
    """Frozen global c-TF-IDF artifacts for consistent term weighting.

    This dataclass holds pre-computed vocabulary and IDF weights from the
    full corpus, enabling consistent term weighting across all EOM clusters.
    """
    vocabulary: np.ndarray           # Frozen vocabulary terms (1D array)
    vocabulary_index: Dict[str, int] # Term -> column index mapping
    global_idf: np.ndarray           # Frozen IDF weights (1D array, same size as vocabulary)
    n_total_documents: int           # Total document count used for IDF computation
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import generate_enhanced_variable_key
import models

# Import format_ontology_text for point_mapping
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


#%% CONFIGURATION
@dataclass
class HdbscanLexicalConfig:
    """Configuration for HDBSCAN-based lexical grouping."""

    # Grid search settings
    grid_k: int = 3                    # Number of grid points
    ms_low_log_mult: float = 1.0       # lower = ceil(1.0 * log(n))
    ms_high_log_mult: float = 2.0      # upper = ceil(2.0 * log(n))
    # mcs grid = ms grid (1:1 mapping per requirements)

    # HDBSCAN settings
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"
    min_cluster_size_for_fragmentation: int = 6

    # Embedding source: "umap" uses Layer 1 UMAP embeddings
    embedding_source: str = "umap"

    # c-TF-IDF settings
    text_source: str = "idea"  # Use idea text for c-TF-IDF
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)
    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"

    # Global c-TF-IDF settings (for full corpus vocabulary + IDF)
    global_ctfidf_min_df: int = 2       # Higher for global (filter rare terms)
    global_ctfidf_max_df: float = 0.8   # Lower for global (filter very common terms)

    # Lexical HDBSCAN settings (for clustering c-TF-IDF vectors)
    lexical_min_cluster_size: int = 2
    lexical_min_samples: int = 1

    # Keyword extraction
    top_keywords: int = 8

    # Representative samples
    max_samples_per_group: int = 5  # Max representative samples to show per lexical group


CONFIG = HdbscanLexicalConfig()


#%% SPACY UTILITIES
_SPACY_NLP = None


def get_spacy_nlp(model_name: str = "nl_core_news_lg"):
    """Get or load spaCy NLP model (lazy initialization)."""
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


#%% PHASE 1: DATA LOADING
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


#%% PHASE 2: HDBSCAN FRAGMENTATION WITH DBCV GRID SEARCH
def generate_ms_mcs_grid(
    n: int,
    k: int = 3,
    low_mult: float = 1.0,
    high_mult: float = 2.0
) -> List[int]:
    """
    Generate log-spaced min_samples/min_cluster_size grid for HDBSCAN.

    Formula:
        lower = ceil(low_mult * log(n))
        upper = ceil(high_mult * log(n))
        k log-spaced points

    Args:
        n: Number of points in cluster
        k: Number of grid points (default 3)
        low_mult: Multiplier for lower bound (default 1.0)
        high_mult: Multiplier for upper bound (default 2.0)

    Returns:
        Log-spaced list of unique integers
    """
    log_n = math.log(max(n, 2))
    low = max(2, int(math.ceil(low_mult * log_n)))
    high = max(low, int(math.ceil(high_mult * log_n)))

    if low == high:
        return [low]

    # Generate log-spaced values
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def compute_dbcv(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """
    Compute DBCV (Density-Based Clustering Validation) score.

    Args:
        labels: Cluster labels (-1 = noise)
        embeddings: Embedding matrix

    Returns:
        DBCV score, or -1.0 on failure
    """
    try:
        from hdbscan import validity
        mask = labels >= 0
        if mask.sum() < 2:
            return -1.0
        embeddings_f64 = embeddings[mask].astype(np.float64)
        labels_filtered = labels[mask]
        # Need at least 2 clusters for DBCV
        if len(set(labels_filtered)) < 2:
            return -1.0
        score = validity.validity_index(embeddings_f64, labels_filtered)
        return float(score)
    except Exception:
        return -1.0


def run_hdbscan_grid_search_for_cluster(
    eom_cluster_id: int,
    point_indices: np.ndarray,
    embeddings: np.ndarray,
    config: HdbscanLexicalConfig = CONFIG
) -> Dict:
    """
    Run HDBSCAN grid search on one EOM cluster, optimizing DBCV.

    Args:
        eom_cluster_id: ID of the EOM cluster
        point_indices: Indices of points belonging to this EOM cluster
        embeddings: Embedding matrix (UMAP from Layer 1)
        config: Configuration object

    Returns:
        Dict containing:
        - labels: array of sub-cluster labels (local indexing)
        - best_ms: winning min_samples value
        - best_mcs: winning min_cluster_size value
        - best_dbcv: DBCV score of winning config
        - n_subclusters: number of sub-clusters found
        - point_indices: original indices for mapping back
        - skipped: bool if cluster was too small
        - all_results: list of all grid search results
    """
    n_points = len(point_indices)

    # Skip if too small
    if n_points < config.min_cluster_size_for_fragmentation:
        return {
            "labels": np.zeros(n_points, dtype=int),
            "best_ms": None,
            "best_mcs": None,
            "best_dbcv": None,
            "n_subclusters": 1,
            "point_indices": point_indices,
            "skipped": True,
            "skip_reason": f"Too small ({n_points} < {config.min_cluster_size_for_fragmentation})",
            "all_results": []
        }

    # Generate grid (ms and mcs share same values)
    ms_mcs_grid = generate_ms_mcs_grid(
        n_points,
        k=config.grid_k,
        low_mult=config.ms_low_log_mult,
        high_mult=config.ms_high_log_mult
    )

    # Extract cluster embeddings
    cluster_embeddings = embeddings[point_indices]

    # Grid search
    best_result = None
    best_dbcv = -np.inf
    all_results = []

    for ms_mcs_value in ms_mcs_grid:
        # ms and mcs use same value (1:1 mapping)
        ms = ms_mcs_value
        mcs = ms_mcs_value

        # Safety: mcs cannot exceed n_points // 2
        mcs = min(mcs, max(2, n_points // 2))
        ms = min(ms, mcs)

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric=config.hdbscan_metric,
                cluster_selection_method=config.hdbscan_cluster_selection_method,
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(cluster_embeddings)

            # Compute DBCV
            dbcv = compute_dbcv(labels, cluster_embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = sum(labels == -1)

            result = {
                "ms": ms,
                "mcs": mcs,
                "dbcv": dbcv,
                "n_clusters": n_clusters,
                "labels": labels.copy(),
                "noise_count": noise_count
            }
            all_results.append(result)

            if dbcv > best_dbcv:
                best_dbcv = dbcv
                best_result = result

        except Exception as e:
            all_results.append({"ms": ms, "mcs": mcs, "error": str(e)})

    # Fallback if no valid result
    if best_result is None:
        return {
            "labels": np.zeros(n_points, dtype=int),
            "best_ms": None,
            "best_mcs": None,
            "best_dbcv": None,
            "n_subclusters": 1,
            "point_indices": point_indices,
            "skipped": True,
            "skip_reason": "All grid search attempts failed",
            "all_results": all_results
        }

    return {
        "labels": best_result["labels"],
        "best_ms": best_result["ms"],
        "best_mcs": best_result["mcs"],
        "best_dbcv": best_result["dbcv"],
        "n_subclusters": best_result["n_clusters"],
        "point_indices": point_indices,
        "skipped": False,
        "skip_reason": None,
        "all_results": all_results
    }


def run_hdbscan_fragmentation_all_clusters(
    labels_eom: np.ndarray,
    embeddings: np.ndarray,
    config: HdbscanLexicalConfig = CONFIG
) -> Dict[int, Dict]:
    """
    Run HDBSCAN fragmentation for all EOM clusters.

    Args:
        labels_eom: EOM cluster labels array
        embeddings: UMAP embeddings from Layer 1
        config: Configuration object

    Returns:
        Dict mapping eom_cluster_id -> fragmentation result dict
    """
    eom_cluster_ids = sorted(set(labels_eom) - {-1})
    results = {}

    print(f"\nRunning HDBSCAN fragmentation on {len(eom_cluster_ids)} EOM clusters...")

    for eom_id in eom_cluster_ids:
        point_indices = np.where(labels_eom == eom_id)[0]
        result = run_hdbscan_grid_search_for_cluster(eom_id, point_indices, embeddings, config)
        results[eom_id] = result

        if result["skipped"]:
            print(f"  EOM {eom_id}: skipped - {result['skip_reason']}")
        else:
            print(f"  EOM {eom_id}: {len(point_indices)} points -> {result['n_subclusters']} sub-clusters "
                  f"(DBCV={result['best_dbcv']:.3f}, ms={result['best_ms']}, mcs={result['best_mcs']})")

    return results


def build_global_subcluster_labels(
    labels_eom: np.ndarray,
    fragmentation_results: Dict[int, Dict]
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Build global sub-cluster labels from per-EOM fragmentation results.

    Each sub-cluster gets a unique global ID.

    Returns:
        Tuple of (global_labels, id_mapping)
        - global_labels: array same size as labels_eom with global sub-cluster IDs
        - id_mapping: dict mapping global_id -> (eom_id, local_subcluster_id)
    """
    n_points = len(labels_eom)
    global_labels = np.full(n_points, -1, dtype=int)
    id_mapping = {}

    next_global_id = 0

    for eom_id, result in sorted(fragmentation_results.items()):
        point_indices = result["point_indices"]
        local_labels = result["labels"]

        for local_id in sorted(set(local_labels)):
            if local_id == -1:
                # Skip noise points in fragmentation
                continue
            local_mask = local_labels == local_id
            global_indices = point_indices[local_mask]
            global_labels[global_indices] = next_global_id
            id_mapping[next_global_id] = (eom_id, local_id)
            next_global_id += 1

    return global_labels, id_mapping


#%% PHASE 3: c-TF-IDF ON HDBSCAN SUB-CLUSTERS
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


#%% PHASE 2B: GLOBAL c-TF-IDF ARTIFACTS
def build_global_ctfidf_artifacts(
    point_mapping: List[Dict],
    config: HdbscanLexicalConfig = CONFIG
) -> GlobalCtfidfArtifacts:
    """
    Build global c-TF-IDF artifacts (frozen vocabulary + IDF) from ALL texts.

    This function runs ONCE at startup to establish a consistent vocabulary
    and IDF weighting across all EOM clusters, eliminating IDF drift.

    Args:
        point_mapping: List of dicts with text data for each point
        config: Configuration object

    Returns:
        GlobalCtfidfArtifacts with frozen vocabulary and IDF
    """
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "display_text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(config.text_source, "display_text")

    # Step 1: Collect ALL texts from ALL points
    all_texts = []
    for pt in point_mapping:
        text = pt.get(text_key, "")
        if text:
            all_texts.append(text)

    if not all_texts:
        # Return empty artifacts
        return GlobalCtfidfArtifacts(
            vocabulary=np.array([]),
            vocabulary_index={},
            global_idf=np.array([]),
            n_total_documents=0
        )

    # Step 2: Apply lemmatization
    if config.ctfidf_use_lemmatization:
        lemmatized_texts = lemmatize_adj_noun_only(all_texts, model_name=config.ctfidf_spacy_model)
    else:
        lemmatized_texts = all_texts

    # Step 3: Fit global CountVectorizer with corpus-appropriate filtering
    cv = CountVectorizer(
        ngram_range=config.ctfidf_ngram_range,
        min_df=config.global_ctfidf_min_df,
        max_df=config.global_ctfidf_max_df
    )

    try:
        X = cv.fit_transform(lemmatized_texts)
        vocabulary = np.array(cv.get_feature_names_out())
    except ValueError:
        # No terms survived filtering
        return GlobalCtfidfArtifacts(
            vocabulary=np.array([]),
            vocabulary_index={},
            global_idf=np.array([]),
            n_total_documents=len(lemmatized_texts)
        )

    # Step 4: Compute global IDF
    # df = number of documents containing each term
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    n_docs = X.shape[0]
    global_idf = np.log((n_docs + 1) / (df + 1)) + 1.0

    # Build vocabulary index for fast lookups
    vocab_index = {term: idx for idx, term in enumerate(vocabulary)}

    print(f"  Global c-TF-IDF: {len(vocabulary)} terms from {n_docs} documents")

    return GlobalCtfidfArtifacts(
        vocabulary=vocabulary,
        vocabulary_index=vocab_index,
        global_idf=global_idf,
        n_total_documents=n_docs
    )


#%% PHASE 3: c-TF-IDF ON HDBSCAN SUB-CLUSTERS
def build_subcluster_ctfidf_matrix(
    subcluster_ids: List[int],
    global_subcluster_labels: np.ndarray,
    point_mapping: List[Dict],
    global_artifacts: GlobalCtfidfArtifacts,
    config: HdbscanLexicalConfig = CONFIG
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build c-TF-IDF matrix where each sub-cluster is a "class".

    Uses frozen global vocabulary and IDF from global_artifacts to ensure
    consistent term weighting across all EOM clusters.

    Args:
        subcluster_ids: List of global sub-cluster IDs to include
        global_subcluster_labels: Full array of global sub-cluster labels
        point_mapping: Global point-to-text mapping
        global_artifacts: Frozen vocabulary and IDF from build_global_ctfidf_artifacts()
        config: Configuration object

    Returns:
        Tuple of (ctfidf_matrix, vocabulary, subcluster_ids_ordered)
        - ctfidf_matrix: shape (n_subclusters, n_terms)
        - vocabulary: array of term strings (from global_artifacts)
        - subcluster_ids_ordered: list of sub-cluster IDs in matrix row order
    """
    # Check for empty global artifacts
    if len(global_artifacts.vocabulary) == 0:
        return np.array([]), np.array([]), []

    # Map text_source to point_mapping key
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "display_text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(config.text_source, "display_text")

    # Build one concatenated document per sub-cluster
    subcluster_docs = []
    subcluster_ids_ordered = []

    for subcluster_id in sorted(subcluster_ids):
        subcluster_points = np.where(global_subcluster_labels == subcluster_id)[0]
        texts = []
        for pt_idx in subcluster_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:
                    texts.append(text)

        if texts:
            subcluster_doc = " ".join(texts)
            subcluster_docs.append(subcluster_doc)
            subcluster_ids_ordered.append(subcluster_id)

    if not subcluster_docs:
        return np.array([]), global_artifacts.vocabulary, []

    # Apply lemmatization if requested
    if config.ctfidf_use_lemmatization:
        subcluster_docs = lemmatize_adj_noun_only(subcluster_docs, model_name=config.ctfidf_spacy_model)

    # Use FROZEN vocabulary from global artifacts
    cv = CountVectorizer(
        ngram_range=config.ctfidf_ngram_range,
        vocabulary=global_artifacts.vocabulary_index  # FROZEN vocab
    )

    try:
        X = cv.transform(subcluster_docs)  # transform, NOT fit_transform
    except ValueError:
        return np.array([]), global_artifacts.vocabulary, subcluster_ids_ordered

    if X.shape[1] == 0:
        return np.array([]), global_artifacts.vocabulary, subcluster_ids_ordered

    # Compute LOCAL TF (L1 normalized per sub-cluster)
    tf = X.astype(float)
    row_sums = np.asarray(tf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1e-12
    tf = tf.multiply(1.0 / row_sums[:, np.newaxis])

    # Apply GLOBAL IDF (frozen from global_artifacts)
    ctfidf = tf.multiply(global_artifacts.global_idf).tocsr()

    return ctfidf.toarray(), global_artifacts.vocabulary, subcluster_ids_ordered


#%% PHASE 4: HDBSCAN ON c-TF-IDF VECTORS
def cluster_subclusters_by_ctfidf(
    ctfidf_matrix: np.ndarray,
    subcluster_ids: List[int],
    config: HdbscanLexicalConfig = CONFIG
) -> Tuple[np.ndarray, int]:
    """
    Cluster sub-clusters using HDBSCAN on c-TF-IDF vectors.

    Args:
        ctfidf_matrix: shape (n_subclusters, n_terms)
        subcluster_ids: List of sub-cluster IDs corresponding to rows
        config: Configuration object

    Returns:
        Tuple of (labels, n_clusters)
        - labels: lexical group assignment for each sub-cluster (-1 = noise)
        - n_clusters: number of lexical groups found
    """
    if ctfidf_matrix.shape[0] < config.lexical_min_cluster_size:
        # Not enough sub-clusters to cluster
        return np.zeros(len(subcluster_ids), dtype=int), 1

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.lexical_min_cluster_size,
        min_samples=config.lexical_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=True
    )

    # L2 normalize so euclidean distance ~ cosine distance
    normalized_matrix = normalize(ctfidf_matrix, norm='l2')
    labels = clusterer.fit_predict(normalized_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


#%% PHASE 4B: PER-EOM LEXICAL GROUPING
def analyze_eom_lexical_grouping(
    eom_id: int,
    subcluster_local_ids: List[int],
    global_subcluster_labels: np.ndarray,
    id_mapping: Dict[int, Tuple[int, int]],
    point_mapping: List[Dict],
    global_artifacts: GlobalCtfidfArtifacts,
    config: HdbscanLexicalConfig = CONFIG,
    min_subclusters_for_clustering: int = 3
) -> Dict:
    """
    Analyze lexical grouping of sub-clusters within ONE EOM cluster.

    Args:
        eom_id: EOM cluster ID
        subcluster_local_ids: List of local sub-cluster IDs in this EOM
        global_subcluster_labels: Global sub-cluster label array
        id_mapping: Dict mapping global_id -> (eom_id, local_id)
        point_mapping: Global point-to-text mapping
        global_artifacts: Frozen vocabulary and IDF from build_global_ctfidf_artifacts()
        config: Configuration object
        min_subclusters_for_clustering: Minimum sub-clusters to attempt HDBSCAN

    Returns:
        Dict with lexical grouping results for this EOM
    """
    n_subclusters = len(subcluster_local_ids)

    # Build reverse mapping for this EOM: local_id -> global_id
    local_to_global = {}
    for global_id, (eid, lid) in id_mapping.items():
        if eid == eom_id:
            local_to_global[lid] = global_id

    # Get global IDs for this EOM's sub-clusters
    global_ids_this_eom = [local_to_global[lid] for lid in subcluster_local_ids if lid in local_to_global]

    result = {
        "eom_id": eom_id,
        "n_subclusters": n_subclusters,
        "subcluster_groups": {},   # {group_id: [local_ids]}
        "subcluster_labels": {},   # {local_id: group_id}
        "keywords": {},            # {group_id: [keywords]}
        "ctfidf_matrix": None,
        "vocab": None,
        "status": "success",
        "message": None
    }

    # Not enough sub-clusters to cluster
    if n_subclusters < min_subclusters_for_clustering:
        for i, local_id in enumerate(subcluster_local_ids):
            result["subcluster_groups"][i] = [local_id]
            result["subcluster_labels"][local_id] = i
            # Extract keywords for this single sub-cluster
            global_id = local_to_global.get(local_id)
            if global_id is not None:
                kws = extract_keywords_for_lexical_group(
                    [global_id], global_subcluster_labels, point_mapping,
                    global_artifacts, config
                )
                result["keywords"][i] = kws
        result["message"] = f"only {n_subclusters} sub-clusters (no HDBSCAN clustering)"
        return result

    # Step 1: Build c-TF-IDF matrix for just this EOM's sub-clusters
    ctfidf_matrix, vocab, ordered_global_ids = build_subcluster_ctfidf_matrix(
        global_ids_this_eom, global_subcluster_labels, point_mapping,
        global_artifacts, config
    )

    result["ctfidf_matrix"] = ctfidf_matrix
    result["vocab"] = vocab

    if ctfidf_matrix.size == 0:
        # Empty matrix - fallback
        for i, local_id in enumerate(subcluster_local_ids):
            result["subcluster_groups"][i] = [local_id]
            result["subcluster_labels"][local_id] = i
        result["status"] = "fallback"
        result["message"] = "empty c-TF-IDF matrix"
        return result

    # Step 2: HDBSCAN on c-TF-IDF vectors
    lexical_labels, n_clusters = cluster_subclusters_by_ctfidf(ctfidf_matrix, ordered_global_ids, config)

    # Step 3: Build sub-cluster groups and extract keywords
    # Map ordered_global_ids back to local_ids
    global_to_local = {v: k for k, v in local_to_global.items()}

    for group_id in set(lexical_labels):
        group_mask = lexical_labels == group_id
        group_global_ids = [ordered_global_ids[i] for i in range(len(ordered_global_ids)) if group_mask[i]]
        group_local_ids = [global_to_local[gid] for gid in group_global_ids if gid in global_to_local]

        if group_id == -1:
            # Noise sub-clusters - each is its own group
            for local_id in group_local_ids:
                noise_group_id = max(list(result["subcluster_groups"].keys()) + [-1]) + 1
                result["subcluster_groups"][noise_group_id] = [local_id]
                result["subcluster_labels"][local_id] = noise_group_id
                global_id = local_to_global.get(local_id)
                if global_id:
                    kws = extract_keywords_for_lexical_group(
                        [global_id], global_subcluster_labels, point_mapping,
                        global_artifacts, config
                    )
                    result["keywords"][noise_group_id] = kws
        else:
            result["subcluster_groups"][group_id] = group_local_ids
            for local_id in group_local_ids:
                result["subcluster_labels"][local_id] = group_id

            # Extract keywords for this lexical group
            kws = extract_keywords_for_lexical_group(
                group_global_ids, global_subcluster_labels, point_mapping,
                global_artifacts, config
            )
            result["keywords"][group_id] = kws

    result["n_lexical_groups"] = n_clusters
    return result


def run_lexical_grouping_all_eoms(
    fragmentation_results: Dict[int, Dict],
    global_subcluster_labels: np.ndarray,
    id_mapping: Dict[int, Tuple[int, int]],
    point_mapping: List[Dict],
    global_artifacts: GlobalCtfidfArtifacts,
    config: HdbscanLexicalConfig = CONFIG
) -> Dict[int, Dict]:
    """
    Run lexical grouping per EOM cluster.

    Args:
        fragmentation_results: Results from HDBSCAN fragmentation per EOM
        global_subcluster_labels: Global sub-cluster label array
        id_mapping: Dict mapping global_id -> (eom_id, local_id)
        point_mapping: Global point-to-text mapping
        global_artifacts: Frozen vocabulary and IDF from build_global_ctfidf_artifacts()
        config: Configuration object

    Returns:
        Dict mapping eom_id -> lexical grouping result
    """
    print(f"\nRunning lexical grouping per EOM cluster...")
    results = {}

    for eom_id, frag_result in sorted(fragmentation_results.items()):
        if frag_result["skipped"]:
            results[eom_id] = {
                "eom_id": eom_id,
                "status": "skipped",
                "message": frag_result["skip_reason"]
            }
            print(f"  EOM {eom_id}: skipped (fragmentation was skipped)")
            continue

        # Get unique local sub-cluster IDs (excluding noise = -1)
        local_ids = [lid for lid in set(frag_result["labels"]) if lid >= 0]
        result = analyze_eom_lexical_grouping(
            eom_id, local_ids, global_subcluster_labels,
            id_mapping, point_mapping, global_artifacts, config
        )
        results[eom_id] = result

        n_groups = result.get("n_lexical_groups", len(result["subcluster_groups"]))
        print(f"  EOM {eom_id}: {len(local_ids)} sub-clusters -> {n_groups} lexical groups")

    return results


#%% PHASE 5: KEYWORD EXTRACTION
def extract_keywords_for_lexical_group(
    subcluster_ids: List[int],
    global_subcluster_labels: np.ndarray,
    point_mapping: List[Dict],
    global_artifacts: GlobalCtfidfArtifacts,
    config: HdbscanLexicalConfig = CONFIG
) -> List[str]:
    """
    Extract c-TF-IDF keywords for a lexical group (set of sub-clusters).

    Uses frozen global vocabulary and IDF from global_artifacts to ensure
    consistent term weighting across all lexical groups.

    Args:
        subcluster_ids: List of sub-cluster IDs in this lexical group
        global_subcluster_labels: Full array of global sub-cluster labels
        point_mapping: Global point-to-text mapping
        global_artifacts: Frozen vocabulary and IDF from build_global_ctfidf_artifacts()
        config: Configuration object

    Returns:
        List of top keywords
    """
    # Check for empty global artifacts
    if len(global_artifacts.vocabulary) == 0:
        return []

    text_key_map = {
        "ontology": "ontology_text",
        "idea": "display_text",
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(config.text_source, "display_text")

    # Collect all texts from all sub-clusters in this lexical group
    all_texts = []
    for subcluster_id in subcluster_ids:
        subcluster_points = np.where(global_subcluster_labels == subcluster_id)[0]
        for pt_idx in subcluster_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:
                    all_texts.append(text)

    if not all_texts:
        return []

    # Apply lemmatization
    if config.ctfidf_use_lemmatization:
        all_texts = lemmatize_adj_noun_only(all_texts, model_name=config.ctfidf_spacy_model)

    # Aggregate all texts into single document for this lexical group
    combined_doc = " ".join(all_texts)

    # Use FROZEN vocabulary from global artifacts
    cv = CountVectorizer(
        ngram_range=config.ctfidf_ngram_range,
        vocabulary=global_artifacts.vocabulary_index
    )

    try:
        X = cv.transform([combined_doc])
    except ValueError:
        return []

    if X.shape[1] == 0:
        return []

    # Compute TF (L1 normalized)
    tf = X.astype(float)
    row_sum = tf.sum()
    if row_sum == 0:
        return []
    tf = tf / row_sum

    # Apply GLOBAL IDF - convert sparse to dense array
    ctfidf_sparse = tf.multiply(global_artifacts.global_idf)
    ctfidf_scores = np.asarray(ctfidf_sparse.toarray()).ravel()

    # Get top keywords
    top_indices = np.argsort(ctfidf_scores)[-config.top_keywords:][::-1]

    return [global_artifacts.vocabulary[i] for i in top_indices if ctfidf_scores[i] > 0]


def extract_keywords_from_ctfidf_row(
    ctfidf_matrix: np.ndarray,
    vocab: np.ndarray,
    row_idx: int,
    top_k: int = 5
) -> List[str]:
    """
    Extract top keywords from a single row of the c-TF-IDF matrix.

    Args:
        ctfidf_matrix: The full c-TF-IDF matrix (n_subclusters x n_terms)
        vocab: Vocabulary array matching matrix columns
        row_idx: Index of the row (sub-cluster) to extract keywords from
        top_k: Number of top keywords to return

    Returns:
        List of top keywords for this sub-cluster
    """
    if row_idx >= ctfidf_matrix.shape[0]:
        return []

    scores = ctfidf_matrix[row_idx]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [vocab[i] for i in top_indices if scores[i] > 0]


def get_representative_samples(
    subcluster_ids: List[int],
    global_subcluster_labels: np.ndarray,
    point_mapping: List[Dict],
    max_samples: int = 5
) -> List[str]:
    """
    Get representative sample texts for a lexical group.

    Args:
        subcluster_ids: List of sub-cluster IDs in this lexical group
        global_subcluster_labels: Full array of global sub-cluster labels
        point_mapping: Global point-to-text mapping
        max_samples: Maximum number of samples to return

    Returns:
        List of representative display_text samples
    """
    all_texts = []
    for subcluster_id in subcluster_ids:
        subcluster_points = np.where(global_subcluster_labels == subcluster_id)[0]
        for pt_idx in subcluster_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get("display_text", "")
                if text and text not in all_texts:  # Avoid duplicates
                    all_texts.append(text)

    # Return up to max_samples, prioritizing shorter/cleaner examples
    # Sort by length to get more readable examples first
    all_texts.sort(key=len)
    return all_texts[:max_samples]


#%% PHASE 6: DISPLAY FUNCTIONS
def display_per_eom_results(
    labels_eom: np.ndarray,
    fragmentation_results: Dict[int, Dict],
    lexical_results: Dict[int, Dict],
    global_subcluster_labels: np.ndarray,
    id_mapping: Dict[int, Tuple[int, int]],
    point_mapping: List[Dict],
    global_artifacts: GlobalCtfidfArtifacts,
    config: HdbscanLexicalConfig = CONFIG
) -> None:
    """Display full results: EOM -> HDBSCAN sub-clusters -> Lexical groups."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL RESULTS: EOM -> HDBSCAN SUB-CLUSTERS -> LEXICAL GROUPS")
    print("=" * 70)

    # Build reverse mapping for keyword extraction: (eom_id, local_id) -> global_id
    local_to_global = {}
    for global_id, (eid, lid) in id_mapping.items():
        local_to_global[(eid, lid)] = global_id

    for eom_id in sorted(fragmentation_results.keys()):
        frag_result = fragmentation_results[eom_id]
        lexical_result = lexical_results.get(eom_id, {})

        eom_size = sum(labels_eom == eom_id)
        print(f"\n{'='*70}")
        print(f"EOM {eom_id} ({eom_size} points)")
        print("=" * 70)

        if frag_result.get("skipped"):
            print(f"  SKIPPED: {frag_result['skip_reason']}")
            continue

        # Get fragmentation info
        n_subclusters = frag_result["n_subclusters"]
        best_dbcv = frag_result["best_dbcv"]
        best_ms = frag_result["best_ms"]
        best_mcs = frag_result["best_mcs"]
        local_labels = frag_result["labels"]
        print(f"HDBSCAN: {n_subclusters} sub-clusters (DBCV={best_dbcv:.4f}, ms={best_ms}, mcs={best_mcs})")

        # Get lexical grouping info
        n_lex_groups = lexical_result.get("n_lexical_groups", len(lexical_result.get("subcluster_groups", {})))
        lex_message = lexical_result.get("message", "")
        if lex_message:
            print(f"Lexical grouping: {n_lex_groups} groups ({lex_message})")
        else:
            print(f"Lexical grouping: {n_lex_groups} groups")

        # Get c-TF-IDF matrix for keyword extraction
        ctfidf_matrix = lexical_result.get("ctfidf_matrix")
        vocab = lexical_result.get("vocab")

        # Group sub-clusters by lexical group
        subcluster_labels = lexical_result.get("subcluster_labels", {})
        subcluster_groups = lexical_result.get("subcluster_groups", {})
        keywords_by_group = lexical_result.get("keywords", {})

        # Organize display by lexical group
        for lex_group_id in sorted(subcluster_groups.keys()):
            local_ids_in_group = subcluster_groups[lex_group_id]
            group_keywords = keywords_by_group.get(lex_group_id, [])

            # Calculate total points in this lexical group
            total_pts = 0
            for local_id in local_ids_in_group:
                total_pts += sum(local_labels == local_id)

            # Get global IDs for representative samples
            group_global_ids = [local_to_global.get((eom_id, lid))
                               for lid in local_ids_in_group
                               if local_to_global.get((eom_id, lid)) is not None]

            print(f"\n  [Lex-{lex_group_id}] {len(local_ids_in_group)} sub-clusters, {total_pts} pts")

            # Always show TF-IDF keywords for the group
            if group_keywords:
                print(f"    TF-IDF keywords: {', '.join(group_keywords[:config.top_keywords])}")
            else:
                # Extract keywords on-the-fly if not pre-computed
                if group_global_ids:
                    kws = extract_keywords_for_lexical_group(
                        group_global_ids, global_subcluster_labels, point_mapping,
                        global_artifacts, config
                    )
                    if kws:
                        print(f"    TF-IDF keywords: {', '.join(kws[:config.top_keywords])}")

            # Show representative samples for the lexical group
            if group_global_ids:
                samples = get_representative_samples(
                    group_global_ids, global_subcluster_labels, point_mapping,
                    max_samples=config.max_samples_per_group
                )
                if samples:
                    print(f"    Representative samples:")
                    for i, sample in enumerate(samples, 1):
                        # Truncate long samples for display
                        display_sample = sample[:100] + "..." if len(sample) > 100 else sample
                        print(f"      {i}. {display_sample}")

            # Show each sub-cluster with its own keywords from c-TF-IDF
            for local_id in sorted(local_ids_in_group):
                size = sum(local_labels == local_id)

                # Get sub-cluster keywords from c-TF-IDF row
                sub_keywords = []
                if ctfidf_matrix is not None and vocab is not None and ctfidf_matrix.size > 0:
                    # Find row index for this local_id
                    global_id = local_to_global.get((eom_id, local_id))
                    if global_id is not None:
                        # Find row in this EOM's ctfidf matrix
                        # The matrix rows correspond to ordered global IDs
                        ordered_globals = [local_to_global.get((eom_id, lid))
                                          for lid in sorted(set(local_labels)) if lid >= 0]
                        if global_id in ordered_globals:
                            row_idx = ordered_globals.index(global_id)
                            sub_keywords = extract_keywords_from_ctfidf_row(
                                ctfidf_matrix, vocab, row_idx, top_k=5
                            )

                kw_str = f" | {', '.join(sub_keywords)}" if sub_keywords else ""
                print(f"      Sub-{local_id}: {size} pts{kw_str}")


#%% MAIN EXECUTION
def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("HDBSCAN LEXICAL GROUPS ANALYSIS")
    print("=" * 70)

    # Phase 1: Load Layer 1 cache
    print("\n[Phase 1] Loading Layer 1 cache...")
    data = load_layer1_cache()
    print(f"  UMAP embeddings: {data['umap_embeddings'].shape}")
    print(f"  EOM clusters: {len(set(data['eom_labels']) - {-1})}")

    labels_eom = data["eom_labels"]
    umap_embeddings = data["umap_embeddings"]

    # Build point mapping
    template_prefix = load_template_prefix_from_cluster_models(data["cluster_models"])
    point_mapping = build_point_mapping(data["cluster_models"], template_prefix)
    print(f"  Point mapping: {len(point_mapping)} points")

    # Phase 1b: Build global c-TF-IDF artifacts (frozen vocabulary + IDF)
    print(f"\n[Phase 1b] Building global c-TF-IDF artifacts...")
    global_artifacts = build_global_ctfidf_artifacts(point_mapping, CONFIG)
    print(f"  Vocabulary size: {len(global_artifacts.vocabulary)}")

    # Phase 2: HDBSCAN fragmentation on UMAP embeddings
    print(f"\n[Phase 2] Running HDBSCAN fragmentation on UMAP embeddings...")
    print(f"  Grid: ms/mcs from ceil({CONFIG.ms_low_log_mult}*log(n)) to ceil({CONFIG.ms_high_log_mult}*log(n)), k={CONFIG.grid_k}")
    fragmentation_results = run_hdbscan_fragmentation_all_clusters(labels_eom, umap_embeddings, CONFIG)

    # Build global sub-cluster labels
    global_subcluster_labels, id_mapping = build_global_subcluster_labels(labels_eom, fragmentation_results)
    subcluster_ids = list(id_mapping.keys())
    print(f"  Total sub-clusters: {len(subcluster_ids)}")

    # Phase 3 & 4: Per-EOM lexical grouping (c-TF-IDF + HDBSCAN)
    print("\n[Phase 3-4] Running lexical grouping per EOM cluster...")
    lexical_results = run_lexical_grouping_all_eoms(
        fragmentation_results, global_subcluster_labels, id_mapping, point_mapping,
        global_artifacts, CONFIG
    )

    # Phase 5 & 6: Display results
    display_per_eom_results(
        labels_eom, fragmentation_results, lexical_results,
        global_subcluster_labels, id_mapping, point_mapping,
        global_artifacts, CONFIG
    )

    return {
        "fragmentation_results": fragmentation_results,
        "global_subcluster_labels": global_subcluster_labels,
        "id_mapping": id_mapping,
        "lexical_results": lexical_results,
        "point_mapping": point_mapping,
        "global_artifacts": global_artifacts,
    }


#%% RUN
if __name__ == "__main__":
    results = main()
