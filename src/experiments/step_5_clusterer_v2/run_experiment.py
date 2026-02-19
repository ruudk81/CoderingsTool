#%%

"""
Step 5: Clusterer V2 Experiment Runner

Run the Clusterer V2 pipeline on a specific dataset from cached Step 4 embeddings.

V2 adds:
- Phase B: Per-cluster theme generation with inclusion definitions
- Phase C: MECE topic consolidation (merge overlapping themes)

This experiment runner uses LOCAL COPIES that can be modified
without affecting the production pipeline. Edit these files:
- clusterer_exp.py             (main clusterer class)
- clusterer_helpers_exp.py     (helper functions/classes)
- config_clusterer_exp.py      (configuration dataclass)
- prompts_exp.py               (LLM prompts for theme + MECE generation)
- mece_consolidator.py         (MECE consolidation logic)

Dataset configuration is centralized in experiments/test_data.py.

Usage:
    cd src && python -m experiments.step_5_clusterer_v2.run_experiment
"""

import sys
import io
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter
from datetime import datetime
import pickle

import nest_asyncio
import numpy as np

nest_asyncio.apply()

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments import models_exp as models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.llm import create_embedding_client
from config import get_embedding_model_for_api

# Import V2 Clusterer
# Handle both module execution (-m) and direct/notebook execution
try:
    # Module execution (python -m experiments.step_5_clusterer_v2.run_experiment)
    from .clusterer_exp import Clusterer
    from .config_clusterer_exp import ClustererConfig
except ImportError:
    # Direct/notebook execution
    from experiments.step_5_clusterer_v2.clusterer_exp import Clusterer
    from experiments.step_5_clusterer_v2.config_clusterer_exp import ClustererConfig

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
# CLUSTERING UNIT
# =============================================================================
# "idea"     = cluster every extracted idea (default, current behavior)
# "node"     = deduplicate to unique nodes, then cluster
# "category" = deduplicate to unique categories, then cluster
# "root"     = deduplicate to unique roots, then cluster
CLUSTERING_UNIT = "category_label"

# Prompt mode: "topics" (default) or "objects"
# "topics"  = standard topic prompts (for idea-level clustering)
# "objects" = object-discovery prompts (for ontology-level clustering)
PROMPT_MODE = "category_label"

# Map ontology level → pre-computed embedding field (None = embed on the fly)
LEVEL_TO_EMBEDDING_FIELD = {
    "instance": "idea_embedding",
    "node": "node_embedding",
    "category": None,
    "root": None,
}


# =============================================================================
# CLUSTERER CONFIGURATION
# =============================================================================
# All defaults defined in config_clusterer_exp.py (single source of truth).
# Override individual params here only for one-off experiments.
CONFIG = ClustererConfig()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step4_embeddings(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> List[models.EmbeddingsModel]:
    """
    Load Step 4 embeddings from cache.

    Embedding field selection (idea_embedding vs taxonomy_embedding etc.) is
    handled downstream by the Clusterer via auto-resolution from the cached
    embedding_text_format.  This function only loads and deserializes.

    Args:
        filename: Dataset filename
        variable: Variable name
        sample_size: Sample size used for caching
        variable_key: Optional explicit cache key

    Returns:
        embeddings_models: list of EmbeddingsModel objects
    """
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

    # Convert serialized data to EmbeddingsModel objects
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
    """
    Load ExtractionMetadata from cache (if available).

    Returns:
        ExtractionMetadata or None if not found
    """
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
# ONTOLOGY-LEVEL CLUSTERING SUPPORT
# =============================================================================

async def embed_texts(texts: List[str]) -> Dict[str, np.ndarray]:
    """Embed a list of unique text strings using the configured embedding provider."""
    client = create_embedding_client(async_mode=True)
    model = get_embedding_model_for_api()
    response = await client.embeddings.create(input=texts, model=model)
    return {
        text: np.array(item.embedding, dtype=np.float32)
        for text, item in zip(texts, response.data)
    }


def extract_unique_nodes(
    embeddings_models: List[models.EmbeddingsModel],
    level: str = "node"
) -> Tuple[List[str], np.ndarray, Dict[str, dict]]:
    """
    Extract unique ontology items at the specified level with matching embeddings.

    Uses pre-computed embeddings when available (instance, node).
    Generates embeddings on the fly for levels without a dedicated field (category, root).

    Returns:
        (names, embeddings_matrix, metadata_dict)
    """
    embedding_field = LEVEL_TO_EMBEDDING_FIELD.get(level)

    item_ideas: Dict[str, List] = {}
    item_embeddings: Dict[str, List[np.ndarray]] = {}
    n_ideas_total = 0
    n_empty = 0
    n_missing_emb = 0

    for resp in embeddings_models:
        if not resp.response_ideas:
            continue
        for idea in resp.response_ideas:
            n_ideas_total += 1
            item_name = (getattr(idea, level, "") or "").strip()
            if not item_name:
                n_empty += 1
                continue

            if item_name not in item_ideas:
                item_ideas[item_name] = []
                if embedding_field:
                    item_embeddings[item_name] = []
            item_ideas[item_name].append(idea)

            if embedding_field:
                emb = getattr(idea, embedding_field, None)
                if emb is not None:
                    item_embeddings[item_name].append(np.array(emb, dtype=np.float32))
                else:
                    n_missing_emb += 1

    if not item_ideas:
        raise ValueError(f"No valid '{level}' texts found in the data.")

    if n_empty > 0:
        print(f"  Note: {n_empty}/{n_ideas_total} ideas have empty '{level}' field")

    names = sorted(item_ideas.keys())

    if embedding_field:
        if n_missing_emb > 0:
            print(f"  WARNING: {n_missing_emb}/{n_ideas_total} ideas have no {embedding_field}")
        names = [n for n in names if item_embeddings.get(n)]
        if not names:
            raise ValueError(
                f"All ideas have {embedding_field}=None. "
                f"Run step 4 with embedding_text_format that includes '{level}'."
            )
        averaged = [np.stack(item_embeddings[n]).mean(axis=0) for n in names]
        embeddings_matrix = np.stack(averaged)
        print(f"\n  Using pre-computed embeddings from '{embedding_field}' (averaged per unique {level})")
    else:
        print(f"\n  No pre-computed embedding for '{level}' level — generating on the fly...")
        text_to_emb = asyncio.run(embed_texts(names))
        embeddings_matrix = np.stack([text_to_emb[n] for n in names])
        print(f"  Embedded {len(names)} unique {level} texts")

    metadata = {}
    for name in names:
        ideas = item_ideas[name]
        cat_c = Counter(i.semantic_category for i in ideas if i.semantic_category)
        root_c = Counter(i.root for i in ideas if i.root)
        metadata[name] = {
            "count": len(ideas),
            "category": cat_c.most_common(1)[0][0] if cat_c else "",
            "root": root_c.most_common(1)[0][0] if root_c else "",
        }

    print(f"\nNode extraction ({level} level):")
    print(f"  Total ideas: {n_ideas_total}")
    print(f"  Unique {level}s: {len(names)}")
    print(f"  Embedding shape: {embeddings_matrix.shape}")

    return names, embeddings_matrix, metadata


def wrap_nodes_as_embeddings_models(
    names: List[str], embeddings: np.ndarray
) -> List[models.EmbeddingsModel]:
    """
    Wrap unique nodes into EmbeddingsModel format for the Clusterer.

    Each unique node becomes a "respondent" with one "idea",
    where idea_embedding = averaged node embedding.
    """
    wrapped = []
    for idx, (name, emb) in enumerate(zip(names, embeddings)):
        idea = models.EmbeddingsSubmodel(
            idea_id=f"node_{idx}_0",
            idea=name,
            node=name,
            idea_embedding=emb,
        )
        resp = models.EmbeddingsModel(
            respondent_id=f"node_{idx}",
            response=name,
            response_ideas=[idea],
            embedding_text_format="idea",
        )
        wrapped.append(resp)
    return wrapped


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Clusterer V2 on the configured dataset."""
    print("=" * 70)
    print("Clustering Pipeline V2 (Theme + MECE)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Algorithm mode: {CONFIG.algorithm_mode}")
    print(f"Clustering unit: {CLUSTERING_UNIT}")
    print(f"Prompt mode: {PROMPT_MODE}")
    print()

    # Load embeddings (field selection handled by Clusterer via auto-resolution)
    embeddings_models = load_step4_embeddings()

    # Load extraction metadata (optional - for taxonomy context in LLM labels)
    extraction_metadata = load_extraction_metadata()

    # --- Ontology-level deduplication (if not clustering at idea level) ---
    node_names = None
    node_metadata = None
    if CLUSTERING_UNIT != "idea":
        node_names, node_embeddings, node_metadata = extract_unique_nodes(
            embeddings_models, level=CLUSTERING_UNIT
        )
        embeddings_models = wrap_nodes_as_embeddings_models(node_names, node_embeddings)
        CONFIG.embedding_source = "idea_embedding"

    # --- Prompt mode: select prompts for theme generation and MECE ---
    theme_generator = None
    mece_prompt = None
    mece_model = None

    if PROMPT_MODE == "objects":
        try:
            from .prompts_exp import (
                CLUSTER_OBJECT_PROMPT, ClusterThemeDescription,
                MECE_OBJECT_CONSOLIDATION_PROMPT, MECETopicSet,
            )
            from .clusterer_helpers_exp import ThemeGenerator
        except ImportError:
            from experiments.step_5_clusterer_v2.prompts_exp import (
                CLUSTER_OBJECT_PROMPT, ClusterThemeDescription,
                MECE_OBJECT_CONSOLIDATION_PROMPT, MECETopicSet,
            )
            from experiments.step_5_clusterer_v2.clusterer_helpers_exp import ThemeGenerator
        theme_generator = ThemeGenerator(
            CONFIG,
            prompt_template=CLUSTER_OBJECT_PROMPT,
            response_model=ClusterThemeDescription,
        )
        mece_prompt = MECE_OBJECT_CONSOLIDATION_PROMPT
        mece_model = MECETopicSet

    # Run clusterer
    clusterer = Clusterer(
        embeddings_models,
        config=CONFIG,
        extraction_metadata=extraction_metadata,
        theme_generator=theme_generator,
        mece_prompt_template=mece_prompt,
        mece_response_model=mece_model,
    )
    clusterer.run()

    # ==========================================================================
    # CACHE RESULTS (like pipeline.py step 5)
    # ==========================================================================
    # Convert to ClusterModel list (preserves all fields + adds initial_cluster)
    cluster_results = clusterer.to_cluster_model()

    # Generate variable key for caching (consistent with Step 4)
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    # Initialize cache manager and save
    cache_manager = CacheManager()
    var_lab = extraction_metadata.var_lab if extraction_metadata else None

    cache_manager.save_to_cache(
        cluster_results,              # List[ClusterModel]
        FILENAME,                     # Dataset filename
        "initial_clusters",           # Step name (matches pipeline.py)
        variable_key,                 # Cache key
        0,                            # elapsed_time
        var_lab=var_lab
    )

    # Layer 2: Clustering metadata (keywords, labels, distributions, metrics)
    metadata = clusterer.to_metadata_model()
    cache_manager.save_to_cache(
        [metadata],                   # ClusteringMetadataModel (wrapped in list for save_to_cache)
        FILENAME,                     # Dataset filename
        "clustering_metadata",        # New step name
        variable_key,                 # Cache key
        0,                            # elapsed_time
        var_lab=var_lab
    )

    # Layer 3: HDBSCAN artifacts (trees for hierarchy analysis in label experiments)
    # Use direct pickle since these are numpy arrays and HDBSCAN objects, not Pydantic models
    hdbscan_artifacts = clusterer.get_hdbscan_artifacts()
    if hdbscan_artifacts:
        base_name = Path(FILENAME).stem
        artifacts_path = project_root / "data" / "cache" / f"hdbscan_artifacts_{base_name}_{variable_key}.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(hdbscan_artifacts, f)
        print(f"CACHED: HDBSCAN artifacts to '{artifacts_path.name}'")

    # Layer 4: UMAP embeddings + winning params (for Layer 2 leaf overlay experiments)
    # This enables analyze_leaf_overlay.py to run HDBSCAN with 'leaf' method
    # without recomputing the expensive UMAP step
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

    # Template prefix (used for text extraction in c-TF-IDF and display)
    template_prefix = clusterer._template_prefix
    if template_prefix:
        prefix_display = template_prefix[:60] + "..." if len(template_prefix) > 60 else template_prefix
        print(f"\nTemplate prefix: '{prefix_display}'")
    else:
        print(f"\nTemplate prefix: (none)")

    # Keywords (MMR and TF-IDF only - c-TF-IDF runs internally but not displayed in summary)
    all_keywords = clusterer.get_all_cluster_keywords()
    if all_keywords:
        for method_name in ["mmr", "tfidf"]:  # Skip ctfidf in summary (still runs, used by MMR)
            method_keywords = all_keywords.get(method_name)
            if method_keywords:
                method_label = {"mmr": "MMR", "tfidf": "TF-IDF"}.get(method_name, method_name)
                print(f"\n{method_label} Keywords ({len(method_keywords)} clusters):")
                for cluster_id in sorted(method_keywords.keys()):
                    kw_list = method_keywords[cluster_id]
                    kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
                    print(f"  Cluster {cluster_id}: {kw_str}")

    # Print ALL clusters with samples and themes
    clusterer.print_all_clusters(n_samples=10)

    # Print low-probability cluster members with their own keywords
    clusterer.print_low_probability_clusters(n_samples=10)

    # Print MECE topics (Phase C output)
    clusterer.print_mece_topics()

    # Cache confirmation (at end for visibility)
    print(f"\n{'='*70}")
    print(f"CACHED: {len(cluster_results)} results to 'initial_clusters' (variable_key: {variable_key})")
    print(f"CACHED: {len(metadata.clusters)} clusters to 'clustering_metadata'")

    # MECE summary
    mece_topics = clusterer.get_mece_topics()
    if mece_topics:
        n_topics = len(mece_topics.topics)
        n_clusters = len(metadata.clusters)
        print(f"\nMECE: {n_clusters} clusters -> {n_topics} MECE topics")

    # Return clusterer for further analysis
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
    """
    Save clustering results to a text file.

    Args:
        output: The captured console output
        filename: Original data filename
        variable: Variable name
        sample_size: Sample size (or None)

    Returns:
        Path to the saved file
    """
    # Create output directory
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: cluster_results_filename_variable_samplesize_YYYYMMDD.txt
    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")

    output_filename = f"cluster_results_{base_name}_{variable}_{sample_str}_{date_str}.txt"
    output_path = output_dir / output_filename

    # Write output
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
