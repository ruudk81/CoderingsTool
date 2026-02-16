"""
Shared helpers for MECE discovery notebooks (topic_discovery_nodes, object_discovery_category).

Provides text formatting, embedding strategy resolution, data loading,
adapter wrapping, and results printing.
"""

import asyncio
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np

from experiments import models_exp as models
from utils.cacheManager import generate_enhanced_variable_key, CacheManager
from utils.llm import create_embedding_client
from config import get_embedding_model_for_api


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

# Simple format strings that match a single pre-computed embedding field.
# If TEXT_FORMAT is one of these, we reuse the pre-computed embedding
# (averaged per unique text) instead of embedding on the fly.
PRECOMPUTED_EMBEDDING_MAP = {
    "{idea}": "idea_embedding",
    "{node}": "node_embedding",
}


def format_cluster_text(
    idea: models.EmbeddingsSubmodel,
    template_prefix: str,
    fmt: str,
) -> str:
    """
    Format a single idea's text according to the format string.

    Available placeholders: {idea}, {instance}, {node}, {category}, {root}, {template_prefix}

    Examples:
        "{node}"                      -> "duurzaamheid"
        "{root} -> {category}"        -> "Maatschappij -> Milieu"
        "{template_prefix}{node}"     -> "ASN Bank roept de associatie op duurzaamheid"
        "{category}"                  -> "Milieu"
    """
    return fmt.format(
        idea=idea.idea or "",
        instance=idea.instance or "",
        node=idea.node or "",
        category=idea.semantic_category or "",
        root=idea.root or "",
        template_prefix=template_prefix or "",
    ).strip()


def resolve_embedding_strategy(fmt: str) -> Tuple[Optional[str], str]:
    """
    Determine whether to use a pre-computed embedding or embed on the fly.

    Returns:
        (precomputed_field, embedding_source_for_clusterer)

        - precomputed_field: EmbeddingsSubmodel field to average per unique text,
          or None if on-the-fly embedding is needed.
        - embedding_source_for_clusterer: Field name the Clusterer reads from
          the synthetic records.

    Examples:
        "{node}"                -> ("node_embedding", "node_embedding")
        "{category}"            -> (None, "idea_embedding")
        "{root} -> {category}"  -> (None, "idea_embedding")
    """
    precomputed_field = PRECOMPUTED_EMBEDDING_MAP.get(fmt)
    if precomputed_field:
        return precomputed_field, precomputed_field
    else:
        # On-the-fly: store fresh embedding in idea_embedding (generic carrier)
        return None, "idea_embedding"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_texts(texts: List[str]) -> Dict[str, np.ndarray]:
    """Embed a list of unique text strings using the configured embedding provider."""
    client = create_embedding_client(async_mode=True)
    model = get_embedding_model_for_api()
    response = await client.embeddings.create(input=texts, model=model)
    return {
        text: np.array(item.embedding, dtype=np.float32)
        for text, item in zip(texts, response.data)
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_unique_items(
    embeddings_models: List[models.EmbeddingsModel],
    fmt: str,
    template_prefix: str = "",
) -> Tuple[List[str], np.ndarray, Dict[str, dict]]:
    """
    Extract unique formatted texts with embeddings (pre-computed or on-the-fly).

    Args:
        embeddings_models: Step 4 output (list of EmbeddingsModel records).
        fmt: Format string, e.g. "{node}", "{category}", "{root} -> {category}".
        template_prefix: Template prefix from extraction metadata (for {template_prefix} placeholder).

    Returns:
        (unique_texts, embeddings_matrix, metadata_dict)
        where metadata_dict[text] = {"count": N, "category": str, "root": str}
    """
    precomputed_field, _ = resolve_embedding_strategy(fmt)

    # Collect unique texts and per-item metadata
    item_ideas: Dict[str, List] = {}                     # text -> list of idea objects
    item_embeddings: Dict[str, List[np.ndarray]] = {}    # only for pre-computed path
    n_ideas_total = 0
    n_empty = 0
    n_missing_emb = 0

    for resp in embeddings_models:
        if not resp.response_ideas:
            continue
        tp = template_prefix or resp.template_prefix or ""
        for idea in resp.response_ideas:
            n_ideas_total += 1
            text = format_cluster_text(idea, tp, fmt)
            if not text:
                n_empty += 1
                continue

            if text not in item_ideas:
                item_ideas[text] = []
                if precomputed_field:
                    item_embeddings[text] = []
            item_ideas[text].append(idea)

            if precomputed_field:
                emb = getattr(idea, precomputed_field, None)
                if emb is not None:
                    item_embeddings[text].append(np.array(emb, dtype=np.float32))
                else:
                    n_missing_emb += 1

    if not item_ideas:
        raise ValueError(f"No valid texts for format '{fmt}'.")

    if n_empty > 0:
        print(f"  Note: {n_empty}/{n_ideas_total} ideas produced empty text for format '{fmt}'")

    # Build embeddings matrix
    names = sorted(item_ideas.keys())

    if precomputed_field:
        # --- Pre-computed path: average existing embeddings per unique text ---
        if n_missing_emb > 0:
            print(f"  WARNING: {n_missing_emb}/{n_ideas_total} ideas have no {precomputed_field}")
        names = [n for n in names if item_embeddings.get(n)]
        if not names:
            raise ValueError(
                f"All ideas have {precomputed_field}=None. "
                f"Run step 4 with an embedding_text_format that includes this field."
            )
        averaged = [np.stack(item_embeddings[n]).mean(axis=0) for n in names]
        embeddings_matrix = np.stack(averaged)
        print(f"\n  Using pre-computed '{precomputed_field}' (averaged per unique text)")
    else:
        # --- On-the-fly path: embed unique texts directly ---
        print(f"\n  No pre-computed embedding for format '{fmt}' — generating on the fly...")
        text_to_emb = asyncio.run(embed_texts(names))
        embeddings_matrix = np.stack([text_to_emb[n] for n in names])
        print(f"  Embedded {len(names)} unique texts")

    # Build metadata
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

    print(f"\nExtraction summary (format='{fmt}'):")
    print(f"  Total ideas: {n_ideas_total}")
    print(f"  Unique texts: {len(names)}")
    print(f"  Embedding shape: {embeddings_matrix.shape}")

    return names, embeddings_matrix, metadata


# ---------------------------------------------------------------------------
# Adapter: wrap unique items as synthetic EmbeddingsModel records
# ---------------------------------------------------------------------------

def wrap_as_embeddings_models(
    names: List[str],
    embeddings: np.ndarray,
    embedding_source: str,
) -> List[models.EmbeddingsModel]:
    """
    Wrap unique items as synthetic EmbeddingsModel records for the Clusterer.

    Each unique text becomes a "respondent" with one "idea".
    The embedding is stored in the field that ``embedding_source`` points to:

    - Pre-computed (e.g. embedding_source="node_embedding"):
      stores in node_embedding — semantically correct.
    - On-the-fly (embedding_source="idea_embedding"):
      stores in idea_embedding — the text in idea.idea is the same text
      that was embedded, so semantically consistent.
    """
    wrapped = []
    for idx, (name, emb) in enumerate(zip(names, embeddings)):
        idea_kwargs = {
            "idea_id": f"discovery_{idx}_0",
            "idea": name,
            "node": name,
            embedding_source: emb,
        }
        idea = models.EmbeddingsSubmodel(**idea_kwargs)
        resp = models.EmbeddingsModel(
            respondent_id=f"discovery_{idx}",
            response=name,
            response_ideas=[idea],
            embedding_text_format="idea",
        )
        wrapped.append(resp)
    return wrapped


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_step4_embeddings(filename: str, variable: str, sample_size, project_root: Path):
    """Load Step 4 embeddings from pickle cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable], is_merged=False, sample_size=sample_size
    )
    cache_dir = project_root / "data" / "cache"
    base_name = Path(filename).stem
    cache_path = cache_dir / f"005_embeddings_{base_name}_{variable_key}.pkl"

    print(f"Loading embeddings from: {cache_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    embeddings_models = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]
    print(f"Loaded {len(embeddings_models)} respondent records")
    return embeddings_models


def load_extraction_metadata(filename: str, variable: str, sample_size):
    """Load ExtractionMetadata from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable], is_merged=False, sample_size=sample_size
    )
    cache_manager = CacheManager()
    metadata = cache_manager.load_metadata_from_cache(
        filename=filename, step="extracted_ideas",
        variable_key=variable_key, model_cls=models.ExtractionMetadata
    )
    if metadata:
        print(f"Loaded ExtractionMetadata: var_lab='{metadata.var_lab}'")
    return metadata


def get_template_prefix(
    embeddings_models: List[models.EmbeddingsModel],
    extraction_metadata=None,
) -> str:
    """Extract template_prefix from metadata or first model that has one."""
    if extraction_metadata and getattr(extraction_metadata, "template_prefix", None):
        return extraction_metadata.template_prefix
    for resp in embeddings_models:
        if resp.template_prefix:
            return resp.template_prefix
    return ""


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def print_discovery_results(
    clusterer,
    item_names: List[str],
    item_metadata: Dict[str, dict],
    discovery_type: str = "TOPIC",
):
    """Print clusters, MECE results, and summary."""
    clusterer.print_all_clusters(n_samples=10)
    clusterer.print_mece_topics()

    mece_result = clusterer.get_mece_topics()
    themes = clusterer.get_cluster_themes()
    n_clusters = len(themes) if themes else 0
    n_mece = len(mece_result.topics) if mece_result else 0
    labels = clusterer._labels
    n_noise = int(np.sum(labels == -1)) if labels is not None else 0

    print(f"\n{'='*70}")
    print(f"{discovery_type} DISCOVERY SUMMARY")
    print(f"{'='*70}")
    print(f"  {len(item_names)} unique items -> {n_clusters} clusters -> {n_mece} MECE {discovery_type.lower()}s")
    print(f"  Noise: {n_noise} items ({n_noise/len(item_names):.1%})")

    if mece_result:
        for i, topic in enumerate(mece_result.topics):
            source_items = []
            for cid in topic.source_cluster_ids:
                mask = labels == cid
                for idx in range(len(item_names)):
                    if mask[idx]:
                        source_items.append((item_names[idx], item_metadata[item_names[idx]]["count"]))
            source_items.sort(key=lambda x: -x[1])
            item_strs = [f"{name} ({count}x)" for name, count in source_items[:10]]
            print(f"\n  [{i+1}] {topic.topic_label}")
            print(f"      Items: {', '.join(item_strs)}")
            if len(source_items) > 10:
                print(f"             ... and {len(source_items) - 10} more")
