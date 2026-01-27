#%%
"""
Cluster Analysis v4: Low Probability Cluster Members

Print low-confidence cluster members (probability ≤ threshold) for each cluster,
showing detailed context to understand edge cases and potential misclassifications.

Output format per idea:
  {idea without template prefix} - {parent_category} ({probability}) | {sentiment}

Usage:
    cd src && python -m experiments.codeGenerator_v2.analyze_clusters_v4
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from config import CacheConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VARIABLE = "Q20"
SAMPLE_SIZE = 500

PROBABILITY_THRESHOLD = 0.7  # Show ideas with prob < this value


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LowProbIdea:
    """Idea with low cluster probability."""
    idea_id: str
    idea: str
    taxonomy_phrase: str
    parent_category: str
    sense: str
    sentiment: str
    initial_cluster: int
    cluster_probability: float


@dataclass
class GlobalContext:
    """Global extraction context."""
    taxonomy_primary_axis: str
    taxonomy_actionable_type: str
    template_prefix: str


@dataclass
class ClusterTheme:
    """Cluster label info."""
    cluster_id: int
    label_theme: str


# =============================================================================
# DATA LOADING
# =============================================================================

def get_variable_key() -> str:
    """Generate consistent variable key."""
    return generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )


def load_extraction_metadata(cache_manager: CacheManager, variable_key: str) -> Optional[GlobalContext]:
    """Load ExtractionMetadata for global context."""

    metadata = cache_manager.load_metadata_from_cache(
        filename=FILENAME,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )

    if metadata:
        return GlobalContext(
            taxonomy_primary_axis=metadata.taxonomy_primary_axis or "",
            taxonomy_actionable_type=metadata.taxonomy_actionable_type or "",
            template_prefix=metadata.template_prefix or ""
        )
    return None


def load_cluster_themes(cache_manager: CacheManager, variable_key: str) -> Dict[int, str]:
    """Load cluster themes from ClusteringMetadataModel."""

    themes: Dict[int, str] = {}

    if cache_manager.is_cache_valid(FILENAME, "clustering_metadata", variable_key):
        results = cache_manager.load_from_cache(
            filename=FILENAME,
            step="clustering_metadata",
            variable_key=variable_key,
            model_cls=models.ClusteringMetadataModel
        )
        if results and len(results) > 0:
            metadata = results[0]
            for cluster_id, cluster_data in metadata.clusters.items():
                themes[cluster_id] = cluster_data.label_theme or f"Cluster {cluster_id}"

    return themes


def load_embeddings_data(cache_manager: CacheManager, variable_key: str) -> Dict[str, Tuple[str, str]]:
    """Load taxonomy_phrase and parent_category from embeddings cache.

    Returns dict: idea_id -> (taxonomy_phrase, parent_category)
    """

    data: Dict[str, Tuple[str, str]] = {}

    embeddings_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step="embeddings",
        variable_key=variable_key,
        model_cls=models.EmbeddingsModel
    )

    for result in embeddings_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                data[idea.idea_id] = (
                    idea.taxonomy_phrase or "",
                    idea.parent_category or ""
                )

    return data


def load_cluster_results(cache_manager: CacheManager, variable_key: str) -> List[LowProbIdea]:
    """Load cluster results and filter for low probability ideas.

    Returns list of LowProbIdea (filtered by PROBABILITY_THRESHOLD).
    """

    low_prob_ideas: List[LowProbIdea] = []

    cluster_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step="initial_clusters",
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )

    for result in cluster_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                prob = idea.cluster_probability or 0.0
                if prob < PROBABILITY_THRESHOLD:
                    low_prob_ideas.append(LowProbIdea(
                        idea_id=idea.idea_id,
                        idea=idea.idea or "",
                        taxonomy_phrase="",  # Will be filled from embeddings
                        parent_category="",  # Will be filled from embeddings
                        sense=idea.sense or "",
                        sentiment=idea.sentiment or "",
                        initial_cluster=idea.initial_cluster if idea.initial_cluster is not None else -1,
                        cluster_probability=prob
                    ))

    return low_prob_ideas


# =============================================================================
# PROCESSING
# =============================================================================

def merge_data(
    low_prob_ideas: List[LowProbIdea],
    embeddings_data: Dict[str, Tuple[str, str]]
) -> List[LowProbIdea]:
    """Merge embeddings data (taxonomy_phrase, parent_category) into ideas."""

    for idea in low_prob_ideas:
        if idea.idea_id in embeddings_data:
            taxonomy_phrase, parent_category = embeddings_data[idea.idea_id]
            idea.taxonomy_phrase = taxonomy_phrase
            idea.parent_category = parent_category

    return low_prob_ideas


def strip_template_prefix(idea_text: str, template_prefix: str) -> str:
    """Remove template prefix from idea text."""
    if template_prefix and idea_text.startswith(template_prefix):
        stripped = idea_text[len(template_prefix):].strip()
        return stripped
    return idea_text


def group_by_cluster(ideas: List[LowProbIdea]) -> Dict[int, List[LowProbIdea]]:
    """Group ideas by cluster_id."""
    grouped: Dict[int, List[LowProbIdea]] = defaultdict(list)
    for idea in ideas:
        grouped[idea.initial_cluster].append(idea)
    return dict(grouped)


# =============================================================================
# OUTPUT
# =============================================================================

def print_header(context: GlobalContext, total_low_prob: int, total_ideas: int):
    """Print report header."""
    print("=" * 80)
    print("CLUSTER ANALYSIS v4: Low Probability Cluster Members")
    print("=" * 80)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Probability threshold: ≤ {PROBABILITY_THRESHOLD}")
    print(f"\nLow-prob ideas: {total_low_prob} / {total_ideas} ({total_low_prob/total_ideas*100:.1f}%)")
    print(f"\nGlobal context:")
    print(f"  taxonomy_primary_axis: {context.taxonomy_primary_axis}")
    print(f"  taxonomy_actionable_type: {context.taxonomy_actionable_type}")
    print(f"  template_prefix: \"{context.template_prefix}\"")


def print_cluster_ideas(
    cluster_id: int,
    ideas: List[LowProbIdea],
    theme: str,
    context: GlobalContext
):
    """Print low-prob ideas for a single cluster."""

    print(f"\n{'=' * 80}")
    print(f"CLUSTER {cluster_id}: {theme}")
    print(f"{'=' * 80}")
    print(f"[{len(ideas)} ideas with prob < {PROBABILITY_THRESHOLD}]")

    # Sort by probability ascending (lowest first)
    ideas_sorted = sorted(ideas, key=lambda x: x.cluster_probability)

    for idea in ideas_sorted:
        stripped_idea = strip_template_prefix(idea.idea, context.template_prefix)
        print(f"{stripped_idea} - {idea.parent_category} ({idea.cluster_probability:.2f}) | {idea.sentiment}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    variable_key = get_variable_key()
    cache_manager = CacheManager(CacheConfig())

    # Load global context
    context = load_extraction_metadata(cache_manager, variable_key)
    if not context:
        print("ERROR: Could not load ExtractionMetadata")
        return

    # Load cluster themes
    themes = load_cluster_themes(cache_manager, variable_key)

    # Load embeddings data (taxonomy_phrase, parent_category)
    embeddings_data = load_embeddings_data(cache_manager, variable_key)

    # Load and filter cluster results
    low_prob_ideas = load_cluster_results(cache_manager, variable_key)

    # Get total idea count for percentage
    cluster_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step="initial_clusters",
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )
    total_ideas = sum(
        len(r.response_ideas) for r in cluster_results if r.response_ideas
    )

    # Merge embeddings data into ideas
    low_prob_ideas = merge_data(low_prob_ideas, embeddings_data)

    # Print header
    print_header(context, len(low_prob_ideas), total_ideas)

    # Group by cluster and print
    grouped = group_by_cluster(low_prob_ideas)

    for cluster_id in sorted(grouped.keys()):
        ideas = grouped[cluster_id]
        theme = themes.get(cluster_id, "(no theme)")
        if cluster_id == -1:
            theme = "(noise)"
        print_cluster_ideas(cluster_id, ideas, theme, context)

    print(f"\n{'=' * 80}")
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
