#%%
"""
Cluster Data Explorer v2

Load ALL cached cluster data and print EVERYTHING available for a target cluster.
Loads data from three cache sources:
1. ExtractionMetadata (context specifiers, taxonomy clarifiers)
2. ClusteringMetadataModel (keywords, LLM labels, distributions, metrics)
3. ClusterModel list (per-idea data)

Usage:
    cd src && python -m experiments.codeGenerator_v2.analyze_clusters_v2
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pickle

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
TARGET_CLUSTER = 0  # Which cluster to analyze in detail

# Project root for cache paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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


def load_extraction_metadata(variable_key: str) -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata from cache (context specifiers, taxonomy clarifiers)."""

    cache_manager = CacheManager(CacheConfig())

    metadata = cache_manager.load_metadata_from_cache(
        filename=FILENAME,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )

    return metadata


def load_clustering_metadata(variable_key: str) -> Optional[models.ClusteringMetadataModel]:
    """Load ClusteringMetadataModel from cache (keywords, LLM labels, distributions)."""

    cache_manager = CacheManager(CacheConfig())
    step = "clustering_metadata"

    if not cache_manager.is_cache_valid(FILENAME, step, variable_key):
        return None

    # ClusteringMetadataModel is wrapped in a list by save_to_cache
    results = cache_manager.load_from_cache(
        filename=FILENAME,
        step=step,
        variable_key=variable_key,
        model_cls=models.ClusteringMetadataModel
    )

    if results and len(results) > 0:
        return results[0]
    return None


def load_cluster_results(variable_key: str) -> Optional[List[models.ClusterModel]]:
    """Load ClusterModel list from cache (per-idea data)."""

    cache_manager = CacheManager(CacheConfig())
    step = "initial_clusters"

    if not cache_manager.is_cache_valid(FILENAME, step, variable_key):
        return None

    cluster_results = cache_manager.load_from_cache(
        filename=FILENAME,
        step=step,
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )

    return cluster_results


def get_cluster_ideas(cluster_results: List[models.ClusterModel], target_cluster: int) -> List[models.ClusterSubmodel]:
    """Extract all ideas belonging to a specific cluster."""

    ideas = []
    for result in cluster_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                cluster_id = idea.initial_cluster if idea.initial_cluster is not None else -1
                if cluster_id == target_cluster:
                    ideas.append(idea)
    return ideas


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_header():
    """Print script header with dataset info."""
    print("=" * 80)
    print("CLUSTER DATA EXPLORER v2 - Full Data Dump")
    print("=" * 80)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Target cluster: {TARGET_CLUSTER}")


def print_extraction_metadata(metadata: Optional[models.ExtractionMetadata]):
    """Print ExtractionMetadata (context specifiers, taxonomy clarifiers)."""

    print("\n" + "=" * 80)
    print("EXTRACTION METADATA (from step: extracted_ideas)")
    print("=" * 80)

    if metadata is None:
        print("  NOT FOUND - run ideaExtractor first")
        return

    # Basic info
    print("\n--- Basic Info ---")
    print(f"  filename: {metadata.filename}")
    print(f"  var_name: {metadata.var_name}")
    print(f"  var_lab: {metadata.var_lab}")
    print(f"  template_prefix: {metadata.template_prefix}")
    print(f"  extraction_timestamp: {metadata.extraction_timestamp}")

    # Context Specifiers
    print("\n--- Context Specifiers ---")
    print(f"  lang: {metadata.lang}")
    print(f"  domain: {metadata.domain}")
    print(f"  topic: {metadata.topic}")
    print(f"  perspective: {metadata.perspective}")
    print(f"  entity: {metadata.entity}")
    print(f"  intent: {metadata.intent}")

    # Taxonomy Clarifiers
    print("\n--- Taxonomy Clarifiers ---")
    print(f"  taxonomy_primary_axis: {metadata.taxonomy_primary_axis}")
    print(f"  taxonomy_secondary_axis: {metadata.taxonomy_secondary_axis}")
    print(f"  taxonomy_rationale: {metadata.taxonomy_rationale}")
    print(f"  taxonomy_axis_description: {metadata.taxonomy_axis_description}")
    print(f"  taxonomy_actionable_type: {metadata.taxonomy_actionable_type}")
    print(f"  taxonomy_sample_phrases: {metadata.taxonomy_sample_phrases}")


def print_clustering_metadata(metadata: Optional[models.ClusteringMetadataModel], target_cluster: int):
    """Print ClusteringMetadataModel (global + per-cluster data)."""

    print("\n" + "=" * 80)
    print("CLUSTERING METADATA (from step: clustering_metadata)")
    print("=" * 80)

    if metadata is None:
        print("  NOT FOUND - run clusterer first")
        return

    # Provenance
    print("\n--- Provenance ---")
    print(f"  algorithm_used: {metadata.algorithm_used}")
    print(f"  algorithm_params: {metadata.algorithm_params}")
    print(f"  timestamp: {metadata.timestamp}")

    # Global Metrics
    print("\n--- Global Metrics ---")
    m = metadata.metrics
    print(f"  n_clusters: {m.n_clusters}")
    print(f"  noise_rate: {m.noise_rate:.3f}")
    print(f"  noise_count: {m.noise_count}")
    print(f"  mean_coherence: {m.mean_coherence:.3f}")
    print(f"  coherence_breakdown: {m.coherence_breakdown}")
    print(f"  silhouette: {m.silhouette}")
    print(f"  dbcv: {m.dbcv}")

    # LLM Context (global)
    if metadata.llm_context:
        print("\n--- LLM Context (global) ---")
        ctx = metadata.llm_context
        print(f"  survey_question: {ctx.survey_question}")
        print(f"  language: {ctx.language}")
        print(f"  domain: {ctx.domain}")
        print(f"  entity: {ctx.entity}")
        print(f"  topic: {ctx.topic}")
        print(f"  perspective: {ctx.perspective}")
        print(f"  intent: {ctx.intent}")
        print(f"  taxonomy_axis: {ctx.taxonomy_axis}")
        print(f"  taxonomy_description: {ctx.taxonomy_description}")
        print(f"  taxonomy_actionable_type: {ctx.taxonomy_actionable_type}")

    # Target Cluster Data
    print("\n" + "-" * 80)
    print(f"CLUSTER {target_cluster} METADATA")
    print("-" * 80)

    if target_cluster not in metadata.clusters:
        print(f"  Cluster {target_cluster} not found in metadata")
        print(f"  Available clusters: {sorted(metadata.clusters.keys())}")
        return

    cluster = metadata.clusters[target_cluster]

    print(f"\n  cluster_id: {cluster.cluster_id}")
    print(f"  size: {cluster.size}")
    print(f"  mean_probability: {cluster.mean_probability}")
    print(f"  coherence: {cluster.coherence}")

    # LLM Label
    print("\n  --- LLM Label ---")
    print(f"  label_theme: {cluster.label_theme}")
    print(f"  label_description: {cluster.label_description}")
    print(f"  label_key_concepts: {cluster.label_key_concepts}")

    # Keywords
    print("\n  --- Keywords (c-TF-IDF) ---")
    for word, score in cluster.keywords_ctfidf[:10]:
        print(f"    {word}: {score:.4f}")

    print("\n  --- Keywords (MMR) ---")
    for word, score in cluster.keywords_mmr[:10]:
        print(f"    {word}: {score:.4f}")

    print("\n  --- Keywords (TF-IDF) ---")
    for word, score in cluster.keywords_tfidf[:10]:
        print(f"    {word}: {score:.4f}")

    # Distributions
    print("\n  --- Sentiment Distribution ---")
    if cluster.sentiment_distribution:
        for sentiment, pct in sorted(cluster.sentiment_distribution.items()):
            print(f"    {sentiment}: {pct:.1%}")
    else:
        print("    (not available)")

    print("\n  --- Sense Distribution ---")
    if cluster.sense_distribution:
        for sense, pct in sorted(cluster.sense_distribution.items()):
            print(f"    {sense}: {pct:.1%}")
    else:
        print("    (not available)")

    # Representative Samples
    print("\n  --- Representative Samples ---")
    for i, (text, prob) in enumerate(cluster.representative_samples[:10], 1):
        text_display = text[:80] + "..." if len(text) > 80 else text
        print(f"    [{i}] prob={prob:.4f} | \"{text_display}\"")


def print_cluster_ideas(ideas: List[models.ClusterSubmodel], target_cluster: int):
    """Print ALL ideas in the target cluster with FULL fields."""

    print("\n" + "=" * 80)
    print(f"CLUSTER {target_cluster} IDEAS (all {len(ideas)} ideas)")
    print("=" * 80)

    if not ideas:
        print("  No ideas found for this cluster")
        return

    # Sort by probability descending
    ideas_sorted = sorted(ideas, key=lambda x: x.cluster_probability or 0, reverse=True)

    for i, idea in enumerate(ideas_sorted, 1):
        print(f"\n[{i}] idea_id: {idea.idea_id}")
        print(f"    idea: \"{idea.idea}\"")
        print(f"    taxonomy_phrase: \"{idea.taxonomy_phrase}\"")
        print(f"    parent_category: \"{idea.parent_category}\"")
        print(f"    sentiment: {idea.sentiment}")
        print(f"    sense: {idea.sense}")
        print(f"    initial_cluster: {idea.initial_cluster}")
        print(f"    cluster_probability: {idea.cluster_probability}")
        print(f"    expanded_cluster: {idea.expanded_cluster}")
        print(f"    cluster_theme: {idea.cluster_theme}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header()

    variable_key = get_variable_key()
    print(f"\nVariable key: {variable_key}")

    # Load all three data sources
    print("\n" + "-" * 80)
    print("LOADING DATA...")
    print("-" * 80)

    extraction_metadata = load_extraction_metadata(variable_key)
    print(f"  ExtractionMetadata: {'LOADED' if extraction_metadata else 'NOT FOUND'}")

    clustering_metadata = load_clustering_metadata(variable_key)
    print(f"  ClusteringMetadataModel: {'LOADED' if clustering_metadata else 'NOT FOUND'}")

    cluster_results = load_cluster_results(variable_key)
    print(f"  ClusterModel list: {'LOADED (' + str(len(cluster_results)) + ' responses)' if cluster_results else 'NOT FOUND'}")

    # Print extraction metadata
    print_extraction_metadata(extraction_metadata)

    # Print clustering metadata (global + target cluster)
    print_clustering_metadata(clustering_metadata, TARGET_CLUSTER)

    # Get and print all ideas for target cluster
    if cluster_results:
        ideas = get_cluster_ideas(cluster_results, TARGET_CLUSTER)
        print_cluster_ideas(ideas, TARGET_CLUSTER)
    else:
        print("\n" + "=" * 80)
        print(f"CLUSTER {TARGET_CLUSTER} IDEAS")
        print("=" * 80)
        print("  Cannot load ideas - cluster results not found")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
