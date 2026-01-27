#%%
"""
Cluster Data Explorer

Load cluster data and examine probability distributions per cluster.

Usage:
    cd src && python -m experiments.codeGenerator_v2.analyze_clusters
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

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

# Display settings
# Ideas are shown grouped by probability buckets (bottom 10% and top 10%)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cluster_data(filename: str, variable: str, sample_size: int):
    """Load ClusterModel data from cache."""

    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size
    )

    cache_manager = CacheManager(CacheConfig())
    step = "initial_clusters"

    if not cache_manager.is_cache_valid(filename, step, variable_key):
        raise FileNotFoundError(
            f"Cache not found for:\n"
            f"  Filename: {filename}\n"
            f"  Variable: {variable}\n"
            f"  Sample size: {sample_size}\n"
            f"  Variable key: {variable_key}\n\n"
            f"Run the pipeline through Step 5 first."
        )

    cluster_results = cache_manager.load_from_cache(
        filename=filename,
        step=step,
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )

    # Get var_lab from cache metadata
    cache_info = cache_manager.db.get_cache_info(filename, step, variable_key)
    var_lab = cache_info.get('var_lab') if cache_info else variable

    return cluster_results, var_lab, variable_key


# =============================================================================
# ANALYSIS
# =============================================================================

def group_ideas_by_cluster(cluster_results: List[models.ClusterModel]) -> Dict[int, List]:
    """Group all ideas by their cluster ID."""

    clusters = defaultdict(list)

    for result in cluster_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                cluster_id = idea.initial_cluster if idea.initial_cluster is not None else -1
                clusters[cluster_id].append(idea)

    return dict(clusters)


def analyze_probability_distribution(ideas: List) -> Dict:
    """Analyze probability distribution for a list of ideas."""

    probs = [idea.cluster_probability for idea in ideas if idea.cluster_probability is not None]

    if not probs:
        return {
            'count': len(ideas),
            'with_prob': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'percentiles': {},
            'histogram': {}
        }

    probs_array = np.array(probs)

    # Create histogram buckets (0-0.1, 0.1-0.2, ..., 0.9-1.0)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]  # 1.01 to include 1.0
    hist_counts, _ = np.histogram(probs_array, bins=bins)
    histogram = {}
    for i in range(len(bins) - 1):
        bucket_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}" if bins[i+1] <= 1.0 else f"{bins[i]:.1f}-1.0"
        histogram[bucket_label] = int(hist_counts[i])

    return {
        'count': len(ideas),
        'with_prob': len(probs),
        'min': float(np.min(probs_array)),
        'max': float(np.max(probs_array)),
        'mean': float(np.mean(probs_array)),
        'median': float(np.median(probs_array)),
        'std': float(np.std(probs_array)),
        'percentiles': {
            '10': float(np.percentile(probs_array, 10)),
            '25': float(np.percentile(probs_array, 25)),
            '50': float(np.percentile(probs_array, 50)),
            '75': float(np.percentile(probs_array, 75)),
            '90': float(np.percentile(probs_array, 90)),
        },
        'histogram': histogram
    }


def analyze_metadata_distribution(ideas: List) -> Dict:
    """Analyze sentiment and sense distributions."""

    sentiment_counts = defaultdict(int)
    sense_counts = defaultdict(int)

    for idea in ideas:
        sentiment_counts[idea.sentiment] += 1
        sense_counts[idea.sense] += 1

    total = len(ideas)

    return {
        'sentiment': {k: {'count': v, 'pct': v/total*100} for k, v in sentiment_counts.items()},
        'sense': {k: {'count': v, 'pct': v/total*100} for k, v in sense_counts.items()}
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_cluster_analysis(cluster_id: int, ideas: List):
    """Print analysis for a single cluster."""

    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id} ({len(ideas)} ideas)")
    print('='*70)

    # Probability distribution
    prob_stats = analyze_probability_distribution(ideas)
    print(f"\nProbability Distribution:")
    if prob_stats['with_prob'] > 0:
        print(f"  Min:    {prob_stats['min']:.4f}")
        print(f"  Max:    {prob_stats['max']:.4f}")
        print(f"  Mean:   {prob_stats['mean']:.4f}")
        print(f"  Median: {prob_stats['median']:.4f}")
        print(f"  Std:    {prob_stats['std']:.4f}")

        # Histogram breakdown
        print(f"\n  Probability Histogram:")
        total = prob_stats['with_prob']
        for bucket, count in prob_stats['histogram'].items():
            bar = '#' * int(count / total * 30) if total > 0 else ''
            pct = count / total * 100 if total > 0 else 0
            print(f"    {bucket}: {count:3d} ({pct:5.1f}%) {bar}")
    else:
        print("  No probability data available")

    # Metadata distribution
    meta_stats = analyze_metadata_distribution(ideas)

    print(f"\nSentiment Distribution:")
    for sentiment, data in sorted(meta_stats['sentiment'].items()):
        print(f"  {sentiment}: {data['count']} ({data['pct']:.1f}%)")

    print(f"\nSense Distribution:")
    for sense, data in sorted(meta_stats['sense'].items()):
        print(f"  {sense}: {data['count']} ({data['pct']:.1f}%)")

    # Group ideas by probability bucket
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

    ideas_by_bucket = {i: [] for i in range(len(bins))}
    for idea in ideas:
        prob = idea.cluster_probability or 0
        for i, (low, high) in enumerate(bins):
            if low <= prob < high:
                ideas_by_bucket[i].append((idea, prob))
                break

    # Show ideas for each probability bucket
    print(f"\nIdeas by Probability Bucket:")

    for bucket_idx in range(len(bins)):
        low, high = bins[bucket_idx]
        bucket_ideas = ideas_by_bucket[bucket_idx]
        label = f"{low:.1f}-{high:.1f}" if high <= 1.0 else f"{low:.1f}-1.0"

        print(f"\n  Bucket {label}: {len(bucket_ideas)} ideas")

        if not bucket_ideas:
            continue

        # Sort by probability within bucket (descending)
        bucket_ideas_sorted = sorted(bucket_ideas, key=lambda x: x[1], reverse=True)

        for i, (idea, prob) in enumerate(bucket_ideas_sorted, 1):
            idea_text = idea.idea[:80] + "..." if len(idea.idea) > 80 else idea.idea
            print(f"    [{i}] prob={prob:.4f} | {idea.sentiment[:3]} | {idea.sense[:4]} | {idea.taxonomy_phrase}")
            print(f"        \"{idea_text}\"")
            if idea.parent_category:
                print(f"        parent_category: {idea.parent_category}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("CLUSTER DATA EXPLORER")
    print("="*70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")

    # Load data
    print("\nLoading cluster data...")
    try:
        cluster_results, var_lab, variable_key = load_cluster_data(FILENAME, VARIABLE, SAMPLE_SIZE)
        print(f"  Loaded {len(cluster_results)} responses")
        print(f"  Variable label: {var_lab}")
        print(f"  Cache key: {variable_key}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Group by cluster
    clusters = group_ideas_by_cluster(cluster_results)
    print(f"\nFound {len(clusters)} clusters")

    # Summary
    print("\n" + "-"*70)
    print("CLUSTER SUMMARY")
    print("-"*70)

    total_ideas = 0
    for cluster_id in sorted(clusters.keys()):
        ideas = clusters[cluster_id]
        total_ideas += len(ideas)
        prob_stats = analyze_probability_distribution(ideas)

        if cluster_id == -1:
            label = "NOISE"
        else:
            label = f"Cluster {cluster_id}"

        if prob_stats['mean'] is not None:
            print(f"  {label:12}: {len(ideas):4} ideas | "
                  f"prob: mean={prob_stats['mean']:.3f}, "
                  f"min={prob_stats['min']:.3f}, "
                  f"max={prob_stats['max']:.3f}")
        else:
            print(f"  {label:12}: {len(ideas):4} ideas | prob: N/A")

    print(f"\n  Total: {total_ideas} ideas")

    # Detailed analysis per cluster (excluding noise)
    for cluster_id in sorted(clusters.keys()):
        if cluster_id == -1:
            continue  # Skip noise for detailed analysis
        print_cluster_analysis(cluster_id, clusters[cluster_id])

    # Noise cluster at the end
    if -1 in clusters:
        print_cluster_analysis(-1, clusters[-1])

    return clusters


if __name__ == "__main__":
    clusters = main()
