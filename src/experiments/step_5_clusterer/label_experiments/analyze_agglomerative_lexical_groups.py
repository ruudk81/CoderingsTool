#%% ============================================================================
# AGGLOMERATIVE LEXICAL GROUPS ANALYSIS - LAYER 2
# ============================================================================
"""
Layer 2 clustering experiment using Agglomerative clustering:
- Loads cached UMAP embeddings + EOM results from Layer 1
- Deepens each EOM cluster via Agglomerative clustering
- Selects optimal k via dendrogram leap detection (largest gap in merge distances)
- Builds c-TF-IDF on taxonomy_phrase for agglomerative sub-clusters
- Runs HDBSCAN on c-TF-IDF vectors to find lexical groups

This enables exploring micro-themes within each stable EOM cluster using
a hierarchical approach with interpretable k selection.

Usage: Open in VS Code and run cells interactively.
"""

#%% IMPORTS AND SETUP
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

import numpy as np
import hdbscan
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

project_root = Path(__file__).parent.parent.parent.parent.parent
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
class AgglomLexicalConfig:
    """Configuration for agglomerative lexical grouping."""

    # Dendrogram leap detection settings
    linkage_method: str = "ward"
    min_k: int = 3                         # Absolute minimum k
    min_k_sqrt: bool = False               # If True, min_k = max(min_k, sqrt(N)) - disabled for natural k
    max_k_fraction: float = 0.5            # max_k = cluster_size * this
    min_cluster_size_for_agglom: int = 6   # Skip agglom for clusters smaller than this

    # Embedding source for agglomerative clustering
    # Options: "idea", "taxonomy", "ontology" (must be available in cache)
    embedding_source: str = "idea"

    # c-TF-IDF settings
    text_source: str = "idea"  # Use idea text (was: taxonomy_phrase)
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)
    ctfidf_min_df: int = 1
    ctfidf_max_df: float = 1.0
    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"

    # Lexical HDBSCAN settings
    lexical_min_cluster_size: int = 2
    lexical_min_samples: int = 1

    # Keyword extraction
    top_keywords: int = 8


CONFIG = AgglomLexicalConfig()


#%% SPACY UTILITIES (copied from clusterer_helpers_exp.py)
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


#%% PHASE 1: DATA LOADING (copied from analyze_leaf_overlay.py)
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


def load_embeddings_from_cache(
    embedding_source: str = "ontology"
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Load embeddings from step 5 embeddings cache.

    Args:
        embedding_source: Which embedding to load: "idea", "taxonomy", or "ontology"

    Returns:
        Dict mapping (respondent_id, idea_id) -> embedding array
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem

    emb_path = cache_dir / f"005_embeddings_{base_name}_{variable_key}.pkl"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings cache not found: {emb_path}\n"
            f"Run step_4_generate_embeddings first."
        )

    # Map source name to cache field name
    field_map = {
        "idea": "idea_embedding",
        "taxonomy": "taxonomy_embedding",
        "ontology": "ontology_embedding",
    }
    emb_field = field_map.get(embedding_source, "ontology_embedding")

    print(f"Loading {embedding_source} embeddings from: {emb_path.name}")
    with open(emb_path, 'rb') as f:
        emb_data = pickle.load(f)

    # Check what's available
    emb_format = emb_data[0].get("embedding_text_format", "unknown") if emb_data else "unknown"
    print(f"  Cache embedding_text_format: {emb_format}")

    # Build lookup dict: (respondent_id, idea_id) -> embedding
    embeddings_lookup = {}
    for item in emb_data:
        respondent_id = item.get("respondent_id")
        ideas = item.get("response_ideas", [])
        for idea in ideas:
            idea_id = idea.get("idea_id")
            emb = idea.get(emb_field)
            if emb is not None:
                # Handle both list and numpy array
                if hasattr(emb, '__len__') and len(emb) > 0:
                    key = (respondent_id, idea_id)
                    embeddings_lookup[key] = np.array(emb)

    print(f"  Loaded {len(embeddings_lookup)} {embedding_source} embeddings")
    return embeddings_lookup


def build_embeddings_array(
    cluster_models: List[models.ClusterModel],
    embeddings_lookup: Dict[Tuple[str, int], np.ndarray],
    embedding_source: str = "ontology"
) -> np.ndarray:
    """
    Build embeddings array aligned with point_mapping order.

    Args:
        cluster_models: ClusterModel list (same order as point_mapping)
        embeddings_lookup: Dict from load_embeddings_from_cache()
        embedding_source: Name for error messages

    Returns:
        np.ndarray of shape (n_points, embedding_dim)
    """
    embeddings = []
    missing_count = 0

    for model in cluster_models:
        if model.response_ideas:
            for idea in model.response_ideas:
                key = (model.respondent_id, idea.idea_id)
                emb = embeddings_lookup.get(key)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    missing_count += 1
                    # Fallback: use zeros (will be handled downstream)
                    # Get dim from first available embedding
                    dim = next(iter(embeddings_lookup.values())).shape[0] if embeddings_lookup else 3072
                    embeddings.append(np.zeros(dim))

    if missing_count > 0:
        print(f"  Warning: {missing_count} ideas missing {embedding_source} embeddings")

    return np.array(embeddings)


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


#%% PHASE 2: AGGLOMERATIVE CLUSTERING WITH DENDROGRAM LEAP DETECTION (NEW)
def detect_dendrogram_leap(
    linkage_matrix: np.ndarray,
    min_k: int = 2,
    max_k: Optional[int] = None
) -> Tuple[int, float, np.ndarray]:
    """
    Find optimal k by detecting largest gap in merge distances.

    The linkage matrix has shape (n-1, 4) where:
    - Column 0, 1: indices of clusters being merged
    - Column 2: distance at which merge occurs
    - Column 3: number of points in new cluster

    Algorithm:
    1. Extract merge distances (column 2)
    2. Compute gaps between consecutive distances
    3. Find the largest gap within valid k range (indicates natural cluster boundary)
    4. k = n - index_of_largest_gap

    Args:
        linkage_matrix: scipy linkage matrix from hierarchical clustering
        min_k: Minimum number of clusters to consider
        max_k: Maximum number of clusters (default: n // 2)

    Returns:
        Tuple of (optimal_k, gap_size, all_gaps)
    """
    distances = linkage_matrix[:, 2]
    gaps = np.diff(distances)

    n = len(distances) + 1  # n points = n-1 merges
    if max_k is None:
        max_k = n // 2

    # Clamp k bounds
    min_k = max(2, min_k)
    max_k = min(n - 1, max_k)

    if min_k >= max_k:
        return min_k, 0.0, gaps

    # Gap at index i corresponds to the merge that goes from (n-i) clusters to (n-i-1) clusters
    # So cutting just before merge i gives us (n-i) clusters
    # For k clusters, we need to find gap at index (n-k-1)
    # Valid k range: [min_k, max_k]
    # Corresponding gap indices: [n-max_k-1, n-min_k-1]

    start_idx = max(0, n - max_k - 1)
    end_idx = min(len(gaps), n - min_k)

    if start_idx >= end_idx:
        return min_k, 0.0, gaps

    # Find largest gap in valid range
    valid_gaps = gaps[start_idx:end_idx]
    best_local_idx = np.argmax(valid_gaps)
    best_gap_idx = start_idx + best_local_idx
    gap_size = gaps[best_gap_idx]

    # Convert gap index to k
    # Gap at index i: cutting here gives k = n - i - 1 clusters
    optimal_k = n - best_gap_idx - 1

    return optimal_k, gap_size, gaps


def run_agglomerative_for_cluster(
    eom_cluster_id: int,
    point_indices: np.ndarray,
    umap_embeddings: np.ndarray,
    config: AgglomLexicalConfig = CONFIG
) -> Dict:
    """
    Run agglomerative clustering on one EOM cluster with dendrogram leap detection.

    Args:
        eom_cluster_id: ID of the EOM cluster
        point_indices: Indices of points belonging to this EOM cluster
        umap_embeddings: Full UMAP embedding matrix
        config: Configuration object

    Returns:
        Dict containing:
        - labels: array of sub-cluster labels for each point (local indexing)
        - optimal_k: number of clusters selected
        - gap_size: size of the dendrogram gap that determined k
        - linkage_matrix: scipy linkage matrix for visualization
        - point_indices: original point indices (for mapping back)
    """
    n_points = len(point_indices)

    # Too small for agglomerative?
    if n_points < config.min_cluster_size_for_agglom:
        return {
            "labels": np.zeros(n_points, dtype=int),
            "optimal_k": 1,
            "gap_size": 0.0,
            "linkage_matrix": None,
            "point_indices": point_indices,
            "skipped": True,
            "skip_reason": f"Cluster too small ({n_points} < {config.min_cluster_size_for_agglom})"
        }

    # Extract embeddings for this cluster
    cluster_embeddings = umap_embeddings[point_indices]

    # Build linkage matrix
    Z = linkage(cluster_embeddings, method=config.linkage_method)

    # Calculate min_k: use sqrt(N) if configured, otherwise use fixed min_k
    if config.min_k_sqrt:
        min_k = max(config.min_k, int(np.sqrt(n_points)))
    else:
        min_k = config.min_k

    # Detect optimal k via dendrogram leap
    max_k = max(min_k + 1, int(n_points * config.max_k_fraction))
    optimal_k, gap_size, gaps = detect_dendrogram_leap(Z, min_k=min_k, max_k=max_k)

    # Cut dendrogram at optimal k
    labels = fcluster(Z, t=optimal_k, criterion='maxclust')
    # fcluster returns 1-indexed labels, convert to 0-indexed
    labels = labels - 1

    return {
        "labels": labels,
        "optimal_k": optimal_k,
        "gap_size": gap_size,
        "linkage_matrix": Z,
        "point_indices": point_indices,
        "skipped": False,
        "skip_reason": None
    }


def run_agglomerative_all_clusters(
    labels_eom: np.ndarray,
    umap_embeddings: np.ndarray,
    config: AgglomLexicalConfig = CONFIG
) -> Dict[int, Dict]:
    """
    Run agglomerative clustering for all EOM clusters.

    Args:
        labels_eom: EOM cluster labels array
        umap_embeddings: UMAP embeddings
        config: Configuration object

    Returns:
        Dict mapping eom_cluster_id -> agglomerative result dict
    """
    eom_cluster_ids = sorted(set(labels_eom) - {-1})
    results = {}

    print(f"\nRunning agglomerative clustering on {len(eom_cluster_ids)} EOM clusters...")

    for eom_id in eom_cluster_ids:
        point_indices = np.where(labels_eom == eom_id)[0]
        result = run_agglomerative_for_cluster(eom_id, point_indices, umap_embeddings, config)
        results[eom_id] = result

        if result["skipped"]:
            print(f"  EOM {eom_id}: skipped - {result['skip_reason']}")
        else:
            print(f"  EOM {eom_id}: {len(point_indices)} points -> {result['optimal_k']} sub-clusters (gap={result['gap_size']:.3f})")

    return results


def build_global_agglom_labels(
    labels_eom: np.ndarray,
    agglom_results: Dict[int, Dict]
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Build global agglom sub-cluster labels from per-EOM results.

    Each agglom sub-cluster gets a unique global ID.

    Returns:
        Tuple of (global_labels, id_mapping)
        - global_labels: array same size as labels_eom with global sub-cluster IDs
        - id_mapping: dict mapping global_id -> (eom_id, local_agglom_id)
    """
    n_points = len(labels_eom)
    global_labels = np.full(n_points, -1, dtype=int)
    id_mapping = {}

    next_global_id = 0

    for eom_id, result in sorted(agglom_results.items()):
        point_indices = result["point_indices"]
        local_labels = result["labels"]

        for local_id in sorted(set(local_labels)):
            local_mask = local_labels == local_id
            global_indices = point_indices[local_mask]
            global_labels[global_indices] = next_global_id
            id_mapping[next_global_id] = (eom_id, local_id)
            next_global_id += 1

    return global_labels, id_mapping


#%% PHASE 3: c-TF-IDF ON AGGLOMERATIVE SUB-CLUSTERS
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


def build_agglom_ctfidf_matrix(
    agglom_ids: List[int],
    global_agglom_labels: np.ndarray,
    point_mapping: List[Dict],
    config: AgglomLexicalConfig = CONFIG
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build c-TF-IDF matrix where each agglom sub-cluster is a "class".

    Uses taxonomy_phrase as text source (configurable via config.text_source).

    Args:
        agglom_ids: List of global agglom cluster IDs to include
        global_agglom_labels: Full array of global agglom labels
        point_mapping: Global point-to-text mapping
        config: Configuration object

    Returns:
        Tuple of (ctfidf_matrix, vocabulary, agglom_ids_ordered)
        - ctfidf_matrix: shape (n_agglom_clusters, n_terms)
        - vocabulary: array of term strings
        - agglom_ids_ordered: list of agglom IDs in matrix row order
    """
    # Map text_source to point_mapping key
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "display_text",  # Use prefix-stripped idea text
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(config.text_source, "taxonomy_phrase")

    # Build one concatenated document per agglom sub-cluster
    agglom_docs = []
    agglom_ids_ordered = []

    for agglom_id in sorted(agglom_ids):
        agglom_points = np.where(global_agglom_labels == agglom_id)[0]
        texts = []
        for pt_idx in agglom_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:
                    texts.append(text)

        if texts:
            agglom_doc = " ".join(texts)
            agglom_docs.append(agglom_doc)
            agglom_ids_ordered.append(agglom_id)

    if not agglom_docs:
        return np.array([]), np.array([]), []

    # Apply lemmatization if requested
    if config.ctfidf_use_lemmatization:
        agglom_docs = lemmatize_adj_noun_only(agglom_docs, model_name=config.ctfidf_spacy_model)

    # Build count matrix
    try:
        cv = CountVectorizer(
            ngram_range=config.ctfidf_ngram_range,
            min_df=config.ctfidf_min_df,
            max_df=config.ctfidf_max_df
        )
        X = cv.fit_transform(agglom_docs)
        vocab = np.array(cv.get_feature_names_out())
    except ValueError:
        return np.array([]), np.array([]), agglom_ids_ordered

    if X.shape[1] == 0:
        return np.array([]), np.array([]), agglom_ids_ordered

    # Compute c-TF-IDF
    tf = X.astype(float)
    row_sums = np.asarray(tf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1e-12
    tf = tf.multiply(1.0 / row_sums[:, np.newaxis])

    df = np.asarray((X > 0).sum(axis=0)).ravel()
    n_classes = X.shape[0]
    idf = np.log((n_classes + 1) / (df + 1)) + 1.0

    ctfidf = tf.multiply(idf).tocsr()

    return ctfidf.toarray(), vocab, agglom_ids_ordered


#%% PHASE 4: HDBSCAN ON c-TF-IDF VECTORS
def cluster_agglom_by_ctfidf(
    ctfidf_matrix: np.ndarray,
    agglom_ids: List[int],
    config: AgglomLexicalConfig = CONFIG
) -> Tuple[np.ndarray, int]:
    """
    Cluster agglom sub-clusters using HDBSCAN on c-TF-IDF vectors.

    Args:
        ctfidf_matrix: shape (n_agglom_clusters, n_terms)
        agglom_ids: List of agglom IDs corresponding to rows
        config: Configuration object

    Returns:
        Tuple of (labels, n_clusters)
        - labels: lexical group assignment for each agglom sub-cluster (-1 = noise)
        - n_clusters: number of lexical groups found
    """
    if ctfidf_matrix.shape[0] < config.lexical_min_cluster_size:
        # Not enough sub-clusters to cluster
        return np.zeros(len(agglom_ids), dtype=int), 1

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


#%% PHASE 4B: PER-EOM LEXICAL GROUPING (NEW - matches overlay.py approach)
def analyze_eom_lexical_grouping(
    eom_id: int,
    agglom_local_ids: List[int],
    global_agglom_labels: np.ndarray,
    id_mapping: Dict[int, Tuple[int, int]],
    point_mapping: List[Dict],
    config: AgglomLexicalConfig = CONFIG,
    min_subclusters_for_clustering: int = 3
) -> Dict:
    """
    Analyze lexical grouping of agglom sub-clusters within ONE EOM cluster.

    This matches the overlay.py approach: build c-TF-IDF and run HDBSCAN
    only on sub-clusters from this single EOM, not globally.

    Args:
        eom_id: EOM cluster ID
        agglom_local_ids: List of local agglom sub-cluster IDs in this EOM
        agglom_results: Full agglom results dict
        global_agglom_labels: Global agglom label array
        id_mapping: Dict mapping global_id -> (eom_id, local_id)
        point_mapping: Global point-to-text mapping
        config: Configuration object
        min_subclusters_for_clustering: Minimum sub-clusters to attempt HDBSCAN

    Returns:
        Dict with lexical grouping results for this EOM
    """
    n_subclusters = len(agglom_local_ids)

    # Build reverse mapping for this EOM: local_id -> global_id
    local_to_global = {}
    for global_id, (eid, lid) in id_mapping.items():
        if eid == eom_id:
            local_to_global[lid] = global_id

    # Get global IDs for this EOM's sub-clusters
    global_ids_this_eom = [local_to_global[lid] for lid in agglom_local_ids if lid in local_to_global]

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
        for i, local_id in enumerate(agglom_local_ids):
            result["subcluster_groups"][i] = [local_id]
            result["subcluster_labels"][local_id] = i
            # Extract keywords for this single sub-cluster
            global_id = local_to_global.get(local_id)
            if global_id is not None:
                kws = extract_keywords_for_lexical_group(
                    [global_id], global_agglom_labels, point_mapping, config
                )
                result["keywords"][i] = kws
        result["message"] = f"only {n_subclusters} sub-clusters (no HDBSCAN clustering)"
        return result

    # Step 1: Build c-TF-IDF matrix for just this EOM's sub-clusters
    ctfidf_matrix, vocab, ordered_global_ids = build_agglom_ctfidf_matrix(
        global_ids_this_eom, global_agglom_labels, point_mapping, config
    )

    result["ctfidf_matrix"] = ctfidf_matrix
    result["vocab"] = vocab

    if ctfidf_matrix.size == 0:
        # Empty matrix - fallback
        for i, local_id in enumerate(agglom_local_ids):
            result["subcluster_groups"][i] = [local_id]
            result["subcluster_labels"][local_id] = i
        result["status"] = "fallback"
        result["message"] = "empty c-TF-IDF matrix"
        return result

    # Step 2: HDBSCAN on c-TF-IDF vectors
    lexical_labels, n_clusters = cluster_agglom_by_ctfidf(ctfidf_matrix, ordered_global_ids, config)

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
                        [global_id], global_agglom_labels, point_mapping, config
                    )
                    result["keywords"][noise_group_id] = kws
        else:
            result["subcluster_groups"][group_id] = group_local_ids
            for local_id in group_local_ids:
                result["subcluster_labels"][local_id] = group_id

            # Extract keywords for this lexical group
            kws = extract_keywords_for_lexical_group(
                group_global_ids, global_agglom_labels, point_mapping, config
            )
            result["keywords"][group_id] = kws

    result["n_lexical_groups"] = n_clusters
    return result


def run_lexical_grouping_all_eoms(
    agglom_results: Dict[int, Dict],
    global_agglom_labels: np.ndarray,
    id_mapping: Dict[int, Tuple[int, int]],
    point_mapping: List[Dict],
    config: AgglomLexicalConfig = CONFIG
) -> Dict[int, Dict]:
    """
    Run lexical grouping per EOM cluster.

    Returns:
        Dict mapping eom_id -> lexical grouping result
    """
    print(f"\nRunning lexical grouping per EOM cluster...")
    results = {}

    for eom_id, agglom_result in sorted(agglom_results.items()):
        if agglom_result["skipped"]:
            results[eom_id] = {
                "eom_id": eom_id,
                "status": "skipped",
                "message": agglom_result["skip_reason"]
            }
            print(f"  EOM {eom_id}: skipped (agglom was skipped)")
            continue

        local_ids = list(set(agglom_result["labels"]))
        result = analyze_eom_lexical_grouping(
            eom_id, local_ids, global_agglom_labels,
            id_mapping, point_mapping, config
        )
        results[eom_id] = result

        n_groups = result.get("n_lexical_groups", len(result["subcluster_groups"]))
        print(f"  EOM {eom_id}: {len(local_ids)} sub-clusters -> {n_groups} lexical groups")

    return results


#%% PHASE 5: KEYWORD EXTRACTION
def extract_keywords_for_lexical_group(
    agglom_cluster_ids: List[int],
    global_agglom_labels: np.ndarray,
    point_mapping: List[Dict],
    config: AgglomLexicalConfig = CONFIG
) -> List[str]:
    """
    Extract TF-IDF keywords for a lexical group (set of agglom sub-clusters).

    Args:
        agglom_cluster_ids: List of agglom IDs in this lexical group
        global_agglom_labels: Full array of global agglom labels
        point_mapping: Global point-to-text mapping
        config: Configuration object

    Returns:
        List of top keywords
    """
    text_key_map = {
        "ontology": "ontology_text",
        "idea": "display_text",  # Use prefix-stripped idea text
        "display_text": "display_text",
        "taxonomy": "taxonomy_phrase"
    }
    text_key = text_key_map.get(config.text_source, "taxonomy_phrase")

    # Collect all texts from all agglom sub-clusters in this lexical group
    all_texts = []
    for agglom_id in agglom_cluster_ids:
        agglom_points = np.where(global_agglom_labels == agglom_id)[0]
        for pt_idx in agglom_points:
            if pt_idx < len(point_mapping):
                text = point_mapping[pt_idx].get(text_key, "")
                if text:
                    all_texts.append(text)

    if not all_texts:
        return []

    # Apply lemmatization
    if config.ctfidf_use_lemmatization:
        all_texts = lemmatize_adj_noun_only(all_texts, model_name=config.ctfidf_spacy_model)

    # Build TF-IDF and get top terms
    try:
        vec = TfidfVectorizer(
            ngram_range=config.ctfidf_ngram_range,
            min_df=1,
            max_df=1.0
        )
        X = vec.fit_transform(all_texts)
        vocab = vec.get_feature_names_out()
    except ValueError:
        return []

    if X.shape[1] == 0:
        return []

    # Average TF-IDF scores across all documents
    avg_scores = np.array(X.mean(axis=0)).flatten()
    top_indices = np.argsort(avg_scores)[-config.top_keywords:][::-1]

    return [vocab[i] for i in top_indices if avg_scores[i] > 0]


def extract_keywords_from_ctfidf_row(
    ctfidf_matrix: np.ndarray,
    vocab: np.ndarray,
    row_idx: int,
    top_k: int = 5
) -> List[str]:
    """
    Extract top keywords from a single row of the c-TF-IDF matrix.

    This is efficient because the c-TF-IDF is already computed.

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


#%% PHASE 6: DISPLAY FUNCTIONS
def display_per_eom_results(
    labels_eom: np.ndarray,
    agglom_results: Dict[int, Dict],
    lexical_results: Dict[int, Dict],
    global_agglom_labels: np.ndarray,
    id_mapping: Dict[int, Tuple[int, int]],
    point_mapping: List[Dict],
    config: AgglomLexicalConfig = CONFIG
) -> None:
    """Display full results: EOM -> Agglom sub-clusters -> Lexical groups."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL RESULTS: EOM -> AGGLOM SUB-CLUSTERS -> LEXICAL GROUPS")
    print("=" * 70)

    # Build reverse mapping for keyword extraction: (eom_id, local_id) -> global_id
    local_to_global = {}
    for global_id, (eid, lid) in id_mapping.items():
        local_to_global[(eid, lid)] = global_id

    for eom_id in sorted(agglom_results.keys()):
        agglom_result = agglom_results[eom_id]
        lexical_result = lexical_results.get(eom_id, {})

        eom_size = sum(labels_eom == eom_id)
        print(f"\n{'='*70}")
        print(f"EOM {eom_id} ({eom_size} points)")
        print("=" * 70)

        if agglom_result.get("skipped"):
            print(f"  SKIPPED: {agglom_result['skip_reason']}")
            continue

        # Get agglom info
        k = agglom_result["optimal_k"]
        gap = agglom_result["gap_size"]
        local_labels = agglom_result["labels"]
        print(f"Agglomerative: {k} sub-clusters (dendrogram gap={gap:.4f})")

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

            print(f"\n  [Lex-{lex_group_id}] {len(local_ids_in_group)} sub-clusters, {total_pts} pts")
            if group_keywords:
                print(f"    Group keywords: {', '.join(group_keywords[:5])}")

            # Show each sub-cluster with its own keywords from c-TF-IDF
            for local_id in sorted(local_ids_in_group):
                size = sum(local_labels == local_id)

                # Get sub-cluster keywords from c-TF-IDF row
                sub_keywords = []
                if ctfidf_matrix is not None and vocab is not None:
                    # Find row index for this local_id
                    global_id = local_to_global.get((eom_id, local_id))
                    if global_id is not None:
                        # Find row in this EOM's ctfidf matrix
                        # The matrix rows correspond to ordered global IDs
                        ordered_globals = [local_to_global.get((eom_id, lid))
                                          for lid in sorted(set(local_labels))]
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
    print("AGGLOMERATIVE LEXICAL GROUPS ANALYSIS")
    print("=" * 70)

    # Phase 1: Load Layer 1 cache
    print("\n[Phase 1] Loading Layer 1 cache...")
    data = load_layer1_cache()
    print(f"  UMAP embeddings: {data['umap_embeddings'].shape}")
    print(f"  EOM clusters: {len(set(data['eom_labels']) - {-1})}")

    labels_eom = data["eom_labels"]

    # Build point mapping
    template_prefix = load_template_prefix_from_cluster_models(data["cluster_models"])
    point_mapping = build_point_mapping(data["cluster_models"], template_prefix)
    print(f"  Point mapping: {len(point_mapping)} points")

    # Load embeddings for agglomerative clustering (full dimensional, not UMAP-reduced)
    print(f"\n[Phase 1b] Loading {CONFIG.embedding_source} embeddings...")
    embeddings_lookup = load_embeddings_from_cache(CONFIG.embedding_source)
    agglom_embeddings = build_embeddings_array(data["cluster_models"], embeddings_lookup, CONFIG.embedding_source)
    print(f"  Embeddings shape: {agglom_embeddings.shape}")

    # Phase 2: Agglomerative clustering on embeddings
    print(f"\n[Phase 2] Running agglomerative clustering on {CONFIG.embedding_source} embeddings...")
    agglom_results = run_agglomerative_all_clusters(labels_eom, agglom_embeddings, CONFIG)

    # Build global agglom labels
    global_agglom_labels, id_mapping = build_global_agglom_labels(labels_eom, agglom_results)
    agglom_ids = list(id_mapping.keys())
    print(f"  Total agglom sub-clusters: {len(agglom_ids)}")

    # Phase 3 & 4: Per-EOM lexical grouping (c-TF-IDF + HDBSCAN)
    print("\n[Phase 3-4] Running lexical grouping per EOM cluster...")
    lexical_results = run_lexical_grouping_all_eoms(
        agglom_results, global_agglom_labels, id_mapping, point_mapping, CONFIG
    )

    # Phase 5 & 6: Display results
    display_per_eom_results(
        labels_eom, agglom_results, lexical_results,
        global_agglom_labels, id_mapping, point_mapping, CONFIG
    )

    return {
        "agglom_results": agglom_results,
        "global_agglom_labels": global_agglom_labels,
        "id_mapping": id_mapping,
        "lexical_results": lexical_results,
        "point_mapping": point_mapping,
    }


#%% RUN
if __name__ == "__main__":
    results = main()
