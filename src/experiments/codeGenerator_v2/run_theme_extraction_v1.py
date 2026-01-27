#%%
"""
Theme Extraction Experiment v1

Emulates codeGenerator pipeline up to CLUSTER_SUMMARY_PROMPT (Stage 1: Theme Extraction).
Enables modifications to input data and prompt for experimentation.

Key modifications from codeGenerator:
- REMOVE template prefix from idea.idea
- FILTER to only low-probability ideas (cluster_probability < threshold)

Usage:
    cd src && python -m experiments.codeGenerator_v2.run_theme_extraction_v1
"""

import os
import sys
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import umap
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.llm import create_client, llm_create_async
from config import CacheConfig, ModelConfig

# Import local prompts (can be modified for experimentation)
from experiments.codeGenerator_v2.prompts import CLUSTER_SUMMARY_PROMPT

# Import ClusterSummaryOutput from codeGenerator
from utils.codeGenerator import ClusterSummaryOutput


# =============================================================================
# CONFIGURATION - Modify these for experimentation
# =============================================================================

FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VARIABLE = "Q20"
SAMPLE_SIZE = 500

# Input modifications (the experiment)
PROBABILITY_THRESHOLD = 0.8         # Only include ideas with prob < this value
REMOVE_TEMPLATE_PREFIX = True       # Strip template prefix from idea.idea

# Sampling configuration (same as codeGenerator)
USE_SAMPLING = True                 # Enable UMAP+HDBSCAN sampling for large clusters
MAX_IDEAS_PER_CLUSTER = 50          # Max ideas to send to LLM per cluster

# Probability band configuration
USE_PROBABILITY_BANDS = True        # Group ideas by probability bands in prompt
TOTAL_SAMPLE_BUDGET = 30            # Total ideas across all bands
PROBABILITY_BANDS = {
    'inner':  (0.6, 0.8),           # High-ish probability: 0.6 <= prob < 0.8
    'border': (0.4, 0.6),           # Medium probability: 0.4 <= prob < 0.6
    'fringe': (0.0, 0.4),           # Low probability: prob < 0.4
}
BAND_LABELS = {
    'inner':  'inner members',
    'border': 'border members',
    'fringe': 'fringe members',
}

# Output configuration
PRINT_FULL_PROMPT = True            # Print the full prompt before LLM call
PRINT_IDEAS_LIST = True             # Print the ideas list for each cluster

# LLM configuration
DEFAULT_LANGUAGE = "nl-NL"
MODEL = "gpt-4.1"                   # Model to use for theme extraction


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClusterData:
    """Container for cluster data."""
    cluster_id: Union[int, str]
    ideas: List[Any]                # Full idea objects (ClusterSubmodel)
    embeddings: List[np.ndarray]
    idea_texts: List[str]           # Processed idea texts (with modifications applied)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_variable_key() -> str:
    """Generate consistent variable key for cache lookups."""
    return generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )


def load_cluster_results(cache_manager: CacheManager, variable_key: str) -> List[models.ClusterModel]:
    """Load ClusterModel data from cache."""
    return cache_manager.load_from_cache(
        filename=FILENAME,
        step="initial_clusters",
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )


def load_extraction_metadata(cache_manager: CacheManager, variable_key: str) -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata (for template_prefix and var_lab)."""
    return cache_manager.load_metadata_from_cache(
        filename=FILENAME,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )


def load_clustering_metadata(cache_manager: CacheManager, variable_key: str) -> Optional[models.ClusteringMetadataModel]:
    """Load ClusteringMetadataModel (for existing cluster themes - for comparison)."""
    if cache_manager.is_cache_valid(FILENAME, "clustering_metadata", variable_key):
        results = cache_manager.load_from_cache(
            filename=FILENAME,
            step="clustering_metadata",
            variable_key=variable_key,
            model_cls=models.ClusteringMetadataModel
        )
        if results and len(results) > 0:
            return results[0]
    return None


# =============================================================================
# INPUT PROCESSING (Modified from codeGenerator)
# =============================================================================

def strip_template_prefix(text: str, prefix: str) -> str:
    """Remove template prefix from idea text."""
    if prefix and text.startswith(prefix):
        return text[len(prefix):].strip()
    return text


def extract_cluster_data_modified(
    cluster_results: List[models.ClusterModel],
    template_prefix: str,
    prob_threshold: float
) -> Dict[Union[int, str], ClusterData]:
    """
    Extract cluster data with modifications:
    1. Filter by cluster_probability < prob_threshold
    2. Strip template_prefix from idea.idea

    Based on codeGenerator.extract_cluster_data() but with modifications.
    """
    clusters: Dict[Union[int, str], ClusterData] = {}

    for result in cluster_results:
        ideas_list = result.response_ideas or []

        for idea in ideas_list:
            # Get cluster_id (same logic as codeGenerator)
            cluster_id = idea.expanded_cluster if idea.expanded_cluster is not None else idea.initial_cluster

            # Skip noise cluster
            if cluster_id is None or cluster_id == -1 or str(cluster_id) == "-1":
                continue

            # MODIFICATION 1: Filter by probability threshold
            prob = idea.cluster_probability or 0.0
            if prob >= prob_threshold:
                continue  # Skip high-probability ideas

            # Create cluster entry if needed
            if cluster_id not in clusters:
                clusters[cluster_id] = ClusterData(
                    cluster_id=cluster_id,
                    ideas=[],
                    embeddings=[],
                    idea_texts=[]
                )

            # Add idea data
            clusters[cluster_id].ideas.append(idea)

            # MODIFICATION 2: Strip template prefix
            idea_text = idea.idea or ""
            if REMOVE_TEMPLATE_PREFIX and template_prefix:
                idea_text = strip_template_prefix(idea_text, template_prefix)
            clusters[cluster_id].idea_texts.append(idea_text)

            # Add embedding if available
            if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                clusters[cluster_id].embeddings.append(np.asarray(idea.idea_embedding, dtype=np.float32))

    # Filter out empty clusters
    return {cid: cdata for cid, cdata in clusters.items() if len(cdata.idea_texts) > 0}


# =============================================================================
# IDEA SAMPLING (Copied from codeGenerator lines 2678-2822)
# =============================================================================

def sample_representative_ideas(
    idea_texts: List[str],
    embeddings: List[np.ndarray],
    max_ideas: int = MAX_IDEAS_PER_CLUSTER
) -> List[str]:
    """
    Sample representative ideas from a cluster.

    Copied from codeGenerator._sample_representative_ideas() with minimal changes.

    Behaviour:
      - If n <= max_ideas (or n <= 30), return all (no clustering).
      - Else:
          UMAP (10D, metric='cosine') -> HDBSCAN (euclidean).
          Exclude noise (-1). Allocate ∝ cluster size (stable rounding).
          Within each sub-cluster: random sample.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(idea_texts)

    # Early exit: small clusters — keep all
    if n <= max_ideas or n <= 30:
        return idea_texts

    # Check if we have embeddings
    have_dense_embeddings = len(embeddings) == n and all(e is not None for e in embeddings)

    if not have_dense_embeddings or not USE_SAMPLING:
        # Random sample fallback
        k = min(max_ideas, n)
        return random.sample(idea_texts, k)

    # UMAP + HDBSCAN sampling
    emb = np.vstack(embeddings).astype(np.float32)
    L2_emb = normalize(emb, norm="l2", copy=False)

    reducer = umap.UMAP(n_components=10, n_neighbors=5, metric="cosine", random_state=42)
    emb_10 = reducer.fit_transform(L2_emb)

    # Heuristic: min_cluster_size grows sublinearly with n
    min_cluster_size = max(5, int(np.sqrt(n)))
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False
    )
    labels = hdb.fit_predict(emb_10)

    # Build clusters excluding noise (-1)
    sub_clusters: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue  # exclude noise completely
        sub_clusters.setdefault(int(lbl), []).append(i)

    total_non_noise = sum(len(v) for v in sub_clusters.values())

    # If HDBSCAN yielded only noise or a degenerate result, fall back to centroid top-k
    if total_non_noise == 0:
        centroid = emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(emb, centroid).ravel()
        top_idx = np.argsort(sims)[-max_ideas:][::-1]
        return [idea_texts[i] for i in top_idx]

    # Allocation ∝ cluster size (stable rounding)
    budget = min(max_ideas, total_non_noise)
    sizes = {cid: len(idxs) for cid, idxs in sub_clusters.items()}
    total = float(total_non_noise)

    # base quota + fractional remainder for stable rounding
    raw = {cid: (sizes[cid] / total) * budget for cid in sizes}
    base = {cid: int(np.floor(raw[cid])) for cid in sizes}
    remainder = budget - sum(base.values())

    if remainder > 0:
        order = sorted(sizes.keys(), key=lambda c: (raw[c] - base[c], sizes[c]), reverse=True)
        for cid in order[:remainder]:
            base[cid] += 1

    allocation = base  # final per-cluster k

    # Sample within each cluster
    sampled_indices: List[int] = []
    for cid, idxs in sub_clusters.items():
        k = min(len(idxs), allocation.get(cid, 0))
        if k > 0:
            sampled_indices.extend(random.sample(idxs, k))

    # Safety: cap to budget and map to texts
    sampled_indices = sampled_indices[:budget]
    return [idea_texts[i] for i in sampled_indices]


def sample_ideas_by_probability_band(
    cluster_data: ClusterData,
    total_budget: int = TOTAL_SAMPLE_BUDGET
) -> Dict[str, List[str]]:
    """
    Group ideas by probability bands and sample within each band.

    Returns dict: band_name -> list of sampled idea texts
    Only non-empty bands are included.

    Budget is split evenly across non-empty bands.
    Within each band, HDBSCAN sampling is applied (via sample_representative_ideas).
    """
    # Group ideas by probability band
    bands: Dict[str, Tuple[List[str], List[np.ndarray]]] = {
        'inner':  ([], []),
        'border': ([], []),
        'fringe': ([], []),
    }

    for i, idea in enumerate(cluster_data.ideas):
        prob = idea.cluster_probability or 0.0
        text = cluster_data.idea_texts[i]
        emb = cluster_data.embeddings[i] if i < len(cluster_data.embeddings) else None

        # Determine which band this idea belongs to
        for band_name, (low, high) in PROBABILITY_BANDS.items():
            if low <= prob < high:
                bands[band_name][0].append(text)
                if emb is not None:
                    bands[band_name][1].append(emb)
                break

    # Filter to non-empty bands
    non_empty_bands = {name: data for name, data in bands.items() if len(data[0]) > 0}

    if not non_empty_bands:
        return {}

    # Split budget evenly across non-empty bands
    n_bands = len(non_empty_bands)
    per_band_budget = total_budget // n_bands
    remainder = total_budget % n_bands

    # Allocate budget (give remainder to bands in order: inner, border, fringe)
    band_budgets = {}
    remainder_idx = 0
    for band_name in ['inner', 'border', 'fringe']:
        if band_name in non_empty_bands:
            extra = 1 if remainder_idx < remainder else 0
            band_budgets[band_name] = per_band_budget + extra
            remainder_idx += 1

    # Sample within each band using HDBSCAN sampling
    result: Dict[str, List[str]] = {}

    for band_name in ['inner', 'border', 'fringe']:
        if band_name not in non_empty_bands:
            continue

        texts, embeddings = non_empty_bands[band_name]
        budget = band_budgets[band_name]

        # Apply HDBSCAN sampling
        sampled = sample_representative_ideas(texts, embeddings, budget)
        result[band_name] = sampled

    return result


def format_cluster_text_by_bands(sampled_bands: Dict[str, List[str]]) -> str:
    """
    Format sampled ideas grouped by probability bands.

    Output format:
        inner members:
        - idea 1
        - idea 2

        border members:
        - idea 3
        - idea 4

        fringe members:
        - idea 5
        - idea 6
    """
    sections = []

    for band_name in ['inner', 'border', 'fringe']:
        if band_name not in sampled_bands or not sampled_bands[band_name]:
            continue

        label = BAND_LABELS[band_name]
        ideas_list = "\n".join([f"- {idea}" for idea in sampled_bands[band_name]])
        sections.append(f"{label}:\n{ideas_list}")

    return "\n\n".join(sections)


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def format_cluster_prompt(
    cluster_id: Union[int, str],
    idea_texts_or_bands: Union[List[str], Dict[str, List[str]]],
    var_lab: str,
    extraction_metadata: Optional[models.ExtractionMetadata] = None
) -> str:
    """
    Format the CLUSTER_SUMMARY_PROMPT with cluster data and context.

    idea_texts_or_bands can be:
    - List[str]: Simple list of idea texts (old behavior)
    - Dict[str, List[str]]: Banded ideas from sample_ideas_by_probability_band()
    """
    # Format ideas based on input type
    if isinstance(idea_texts_or_bands, dict):
        # Banded format
        ideas_text = format_cluster_text_by_bands(idea_texts_or_bands)
    else:
        # Simple list format (same as codeGenerator line 2833)
        ideas_text = "\n".join([f"- {idea}" for idea in idea_texts_or_bands])

    # Base params
    params = {
        'cluster_id': str(cluster_id),
        'survey_question': var_lab,
        'language': DEFAULT_LANGUAGE,
        'cluster_text': ideas_text
    }

    # Add context specifiers and taxonomy clarifiers from extraction_metadata
    if extraction_metadata:
        params.update({
            # Context specifiers
            'domain': extraction_metadata.domain or "",
            'topic': extraction_metadata.topic or "",
            'perspective': extraction_metadata.perspective or "",
            'intent': extraction_metadata.intent or "",
            # Taxonomy clarifiers
            'taxonomy_axis': extraction_metadata.taxonomy_primary_axis or "",
            'taxonomy_axis_description': extraction_metadata.taxonomy_axis_description or "",
            'taxonomy_actionable_type': extraction_metadata.taxonomy_actionable_type or "",
        })
    else:
        # Fallback empty values
        params.update({
            'domain': "",
            'topic': "",
            'perspective': "",
            'intent': "",
            'taxonomy_axis': "",
            'taxonomy_axis_description': "",
            'taxonomy_actionable_type': "",
        })

    return CLUSTER_SUMMARY_PROMPT.format(**params)


# =============================================================================
# LLM CALLING
# =============================================================================

async def extract_single_theme(
    cluster_id: Union[int, str],
    idea_texts_or_bands: Union[List[str], Dict[str, List[str]]],
    var_lab: str,
    extraction_metadata: Optional[models.ExtractionMetadata] = None
) -> Optional[ClusterSummaryOutput]:
    """
    Extract theme for a single cluster using CLUSTER_SUMMARY_PROMPT.

    idea_texts_or_bands can be:
    - List[str]: Simple list of idea texts
    - Dict[str, List[str]]: Banded ideas from sample_ideas_by_probability_band()

    Based on codeGenerator._extract_single_theme() but simplified.
    """
    # Format prompt with context specifiers and taxonomy clarifiers
    prompt = format_cluster_prompt(cluster_id, idea_texts_or_bands, var_lab, extraction_metadata)

    # Print prompt if enabled
    if PRINT_FULL_PROMPT:
        print(f"\n{'=' * 80}")
        print(f"PROMPT FOR CLUSTER {cluster_id}")
        print('=' * 80)
        print(prompt)
        print('=' * 80)

    try:
        # Create client
        client = create_client(model=MODEL, async_mode=True)

        # Make LLM call
        response = await llm_create_async(
            client=client,
            model=MODEL,
            prompt=prompt,
            response_model=ClusterSummaryOutput,
            temperature=0.0,
            track_usage=True
        )

        return response

    except Exception as e:
        print(f"ERROR: Theme extraction failed for cluster {cluster_id}: {e}")
        return None


# =============================================================================
# OUTPUT PRINTING
# =============================================================================

def print_cluster_input(
    cluster_id: Union[int, str],
    cluster_data: ClusterData,
    existing_theme: Optional[str] = None
):
    """Print cluster input summary."""
    print(f"\n{'=' * 80}")
    print(f"CLUSTER {cluster_id} INPUT")
    print('=' * 80)
    print(f"Ideas count (after filtering): {len(cluster_data.idea_texts)}")
    print(f"Embeddings count: {len(cluster_data.embeddings)}")

    if existing_theme:
        print(f"\nExisting theme (from cache): {existing_theme}")

    # Show probability band distribution
    if USE_PROBABILITY_BANDS:
        band_counts = {'inner': 0, 'border': 0, 'fringe': 0}
        for idea in cluster_data.ideas:
            prob = idea.cluster_probability or 0.0
            for band_name, (low, high) in PROBABILITY_BANDS.items():
                if low <= prob < high:
                    band_counts[band_name] += 1
                    break

        print(f"\nProbability band distribution:")
        for band_name in ['inner', 'border', 'fringe']:
            count = band_counts[band_name]
            if count > 0:
                print(f"  {BAND_LABELS[band_name]}: {count} ideas")

    if PRINT_IDEAS_LIST:
        print(f"\nIdeas (first 20):")
        for i, text in enumerate(cluster_data.idea_texts[:20], 1):
            prob = cluster_data.ideas[i-1].cluster_probability or 0.0
            print(f"  {i}. ({prob:.2f}) {text}")
        if len(cluster_data.idea_texts) > 20:
            print(f"  ... and {len(cluster_data.idea_texts) - 20} more")


def print_theme_result(cluster_id: Union[int, str], result: ClusterSummaryOutput):
    """Print theme extraction result."""
    print(f"\n{'=' * 80}")
    print(f"RESULT FOR CLUSTER {cluster_id}")
    print('=' * 80)

    if result is None:
        print("  ERROR: No result returned")
        return

    print(f"\nAnalysis:")
    print(f"  {result.analysis}")

    print(f"\nExtracted Themes ({len(result.extracted_themes)}):")
    for theme in result.extracted_themes:
        print(f"\n  Theme {theme.theme_id}:")
        print(f"    Label: {theme.theme_label}")
        print(f"    Clarification: {theme.theme_clarification}")
        print(f"    Abstraction: {theme.abstraction_level}")
        print(f"    Inclusion examples: {theme.assignment_examples.inclusion}")
        print(f"    Exclusion examples: {theme.assignment_examples.exclusion}")
        print(f"    Near neighbor: {theme.assignment_examples.near_neighbor.label}")
        print(f"    Tell apart: {theme.assignment_examples.near_neighbor.tell_apart_rule}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("=" * 80)
    print("THEME EXTRACTION EXPERIMENT v1")
    print("=" * 80)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"\nExperimental settings:")
    print(f"  Probability threshold: < {PROBABILITY_THRESHOLD}")
    print(f"  Remove template prefix: {REMOVE_TEMPLATE_PREFIX}")
    print(f"  Use sampling: {USE_SAMPLING}")
    print(f"  Max ideas per cluster: {MAX_IDEAS_PER_CLUSTER}")
    print(f"  Use probability bands: {USE_PROBABILITY_BANDS}")
    if USE_PROBABILITY_BANDS:
        print(f"  Total sample budget: {TOTAL_SAMPLE_BUDGET}")
        print(f"  Probability bands:")
        for band_name, (low, high) in PROBABILITY_BANDS.items():
            print(f"    {BAND_LABELS[band_name]}: {low} ≤ prob < {high}")
    print(f"  Model: {MODEL}")

    # Initialize
    variable_key = get_variable_key()
    cache_manager = CacheManager(CacheConfig())

    # Load data
    print("\nLoading data...")

    cluster_results = load_cluster_results(cache_manager, variable_key)
    print(f"  Loaded {len(cluster_results)} cluster results")

    extraction_metadata = load_extraction_metadata(cache_manager, variable_key)
    template_prefix = extraction_metadata.template_prefix if extraction_metadata else ""
    var_lab = extraction_metadata.var_lab if extraction_metadata else VARIABLE
    print(f"  Template prefix: '{template_prefix[:50]}...' " if template_prefix and len(template_prefix) > 50 else f"  Template prefix: '{template_prefix}'")
    print(f"  Survey question: {var_lab}")

    clustering_metadata = load_clustering_metadata(cache_manager, variable_key)
    existing_themes = {}
    if clustering_metadata:
        for cid, cdata in clustering_metadata.clusters.items():
            existing_themes[cid] = cdata.label_theme
        print(f"  Loaded {len(existing_themes)} existing cluster themes (for comparison)")

    # Extract cluster data with modifications
    print("\nExtracting cluster data with modifications...")
    clusters = extract_cluster_data_modified(cluster_results, template_prefix, PROBABILITY_THRESHOLD)
    print(f"  Found {len(clusters)} clusters with ideas below probability threshold")

    # Summary
    print("\nCluster summary:")
    total_ideas = 0
    for cid in sorted(clusters.keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else 999):
        cdata = clusters[cid]
        total_ideas += len(cdata.idea_texts)
        existing = existing_themes.get(int(cid) if isinstance(cid, str) and cid.isdigit() else cid, "(no existing theme)")
        print(f"  Cluster {cid}: {len(cdata.idea_texts)} ideas | Existing theme: {existing}")
    print(f"\nTotal ideas to process: {total_ideas}")

    # Process each cluster
    print("\n" + "=" * 80)
    print("PROCESSING CLUSTERS")
    print("=" * 80)

    results = {}

    for cid in sorted(clusters.keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else 999):
        cluster_data = clusters[cid]
        existing_theme = existing_themes.get(int(cid) if isinstance(cid, str) and cid.isdigit() else cid)

        # Print input
        print_cluster_input(cid, cluster_data, existing_theme)

        # Sample ideas (with or without probability bands)
        if USE_PROBABILITY_BANDS:
            # Sample by probability bands
            sampled_bands = sample_ideas_by_probability_band(cluster_data, TOTAL_SAMPLE_BUDGET)
            total_sampled = sum(len(ideas) for ideas in sampled_bands.values())
            print(f"\nSampled {total_sampled} ideas across {len(sampled_bands)} bands (from {len(cluster_data.idea_texts)} total):")
            for band_name in ['inner', 'border', 'fringe']:
                if band_name in sampled_bands:
                    print(f"  {BAND_LABELS[band_name]}: {len(sampled_bands[band_name])} ideas")

            # Extract theme with banded input
            result = await extract_single_theme(cid, sampled_bands, var_lab, extraction_metadata)
        else:
            # Original behavior: simple list sampling
            sampled_ideas = sample_representative_ideas(
                cluster_data.idea_texts,
                cluster_data.embeddings,
                MAX_IDEAS_PER_CLUSTER
            )
            print(f"\nSampled {len(sampled_ideas)} ideas for LLM (from {len(cluster_data.idea_texts)})")

            # Extract theme with simple list
            result = await extract_single_theme(cid, sampled_ideas, var_lab, extraction_metadata)

        results[cid] = result

        # Print result
        print_theme_result(cid, result)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nProcessed {len(results)} clusters")

    successful = sum(1 for r in results.values() if r is not None)
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    print("\nNew themes vs existing themes:")
    for cid in sorted(results.keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else 999):
        result = results[cid]
        existing = existing_themes.get(int(cid) if isinstance(cid, str) and cid.isdigit() else cid, "(none)")

        if result and result.extracted_themes:
            new_theme = result.extracted_themes[0].theme_label
        else:
            new_theme = "(extraction failed)"

        print(f"\n  Cluster {cid}:")
        print(f"    Existing: {existing}")
        print(f"    New:      {new_theme}")

    print("\n" + "=" * 80)
    print("END OF EXPERIMENT")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
