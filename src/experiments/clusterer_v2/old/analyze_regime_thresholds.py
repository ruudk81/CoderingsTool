#!/usr/bin/env python3
"""
Deep Dive: Regime Threshold Validation

Analyzes multiple cached embedding datasets to:
1. Validate q90 thresholds (0.65 and 0.80 boundaries)
2. Check for calculation bugs or edge cases
3. Visualize similarity distributions across regimes
4. Empirically tune regime strategies

Usage:
    cd /Users/ruudkooiman/projects/Python_apps/CoderingsTool
    source .venv/bin/activate
    python analyze_regime_thresholds.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add project to path
project_root = Path("/Users/ruudkooiman/projects/Python_apps/CoderingsTool")
sys.path.insert(0, str(project_root / "src"))

# Cache directory
cache_dir = project_root / "data" / "cache"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_embeddings_from_cache(cache_file: Path) -> Tuple[List[str], np.ndarray, int]:
    """
    Load embeddings from cache file

    Returns:
        (idea_texts, embeddings_matrix, n_ideas)
    """
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    all_ideas = []
    all_embeddings = []

    for respondent in data:
        if 'response_ideas' in respondent and respondent['response_ideas']:
            for idea_obj in respondent['response_ideas']:
                idea_text = idea_obj.get('idea', '')
                embedding = idea_obj.get('idea_embedding')

                if idea_text and embedding is not None:
                    all_ideas.append(idea_text)
                    all_embeddings.append(np.array(embedding, dtype=np.float32))

    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    return all_ideas, embeddings_matrix, len(all_ideas)


def calculate_similarity_stats(embeddings: np.ndarray, sample_size: int = 100) -> Dict:
    """
    Calculate q90, q50, and full distribution of cosine similarities

    Mimics the clusterer.py calculation for comparison
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-12)

    n = len(embeddings_normalized)

    # Subsample if too large (mimic clusterer behavior)
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        sample = embeddings_normalized[idx]
    else:
        sample = embeddings_normalized

    # Compute similarity matrix
    S = sample @ sample.T

    # Extract upper triangle (exclude diagonal)
    triu_indices = np.triu_indices_from(S, k=1)
    similarities = S[triu_indices]

    # Calculate statistics
    stats = {
        'n': n,
        'n_sampled': len(sample),
        'q10': float(np.percentile(similarities, 10)),
        'q25': float(np.percentile(similarities, 25)),
        'q50': float(np.percentile(similarities, 50)),
        'q75': float(np.percentile(similarities, 75)),
        'q90': float(np.percentile(similarities, 90)),
        'q95': float(np.percentile(similarities, 95)),
        'q99': float(np.percentile(similarities, 99)),
        'min': float(similarities.min()),
        'max': float(similarities.max()),
        'mean': float(similarities.mean()),
        'std': float(similarities.std()),
        'similarities': similarities,  # For distribution plotting
    }

    # Calculate kd_cv (coefficient of variation of kNN distances)
    D = 1.0 - (sample @ sample.T)
    np.fill_diagonal(D, np.inf)
    knn_k = min(15, max(5, len(sample) - 1))
    knn_d = np.partition(D, knn_k, axis=1)[:, :knn_k]
    kd_mean = float(np.mean(knn_d))
    kd_std = float(np.std(knn_d))
    kd_cv = float(kd_std / (kd_mean + 1e-12))

    stats['kd_cv'] = kd_cv

    return stats


def classify_regime(n: int, q90: float) -> Tuple[str, str, str]:
    """
    Classify regime based on size and q90

    Returns:
        (regime_id, size_class, structure_class)
    """
    # Size classification
    if n < 100:
        size_class = "small"
    elif n <= 300:
        size_class = "medium"
    else:
        size_class = "large"

    # Structure classification
    if q90 >= 0.80:
        structure_class = "coherent"
    elif q90 < 0.65:
        structure_class = "diffuse"
    else:
        structure_class = "mixed"

    # Regime mapping
    regime_map = {
        ("small", "diffuse"): "R1",
        ("small", "mixed"): "R2",
        ("small", "coherent"): "R3",
        ("medium", "diffuse"): "R4",
        ("medium", "mixed"): "R5",
        ("medium", "coherent"): "R6",
        ("large", "diffuse"): "R7",
        ("large", "mixed"): "R8",
        ("large", "coherent"): "R9",
    }

    regime_id = regime_map[(size_class, structure_class)]
    return regime_id, size_class, structure_class


def check_for_duplicates(idea_texts: List[str]) -> Dict:
    """Check for duplicate idea texts"""
    counts = Counter(idea_texts)
    duplicates = {text: count for text, count in counts.items() if count > 1}

    return {
        'total_ideas': len(idea_texts),
        'unique_ideas': len(counts),
        'duplicate_count': len(duplicates),
        'duplicate_rate': len(duplicates) / len(counts) if counts else 0,
        'compression_ratio': len(counts) / len(idea_texts) if idea_texts else 0,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("REGIME THRESHOLD VALIDATION - DEEP DIVE ANALYSIS")
    print("=" * 80)
    print()

    # Find all embedding cache files
    embedding_files = sorted(cache_dir.glob("005_embeddings_*.pkl"))

    if not embedding_files:
        print("❌ No embedding cache files found!")
        print(f"   Searched in: {cache_dir}")
        return

    print(f"Found {len(embedding_files)} cached embedding datasets")
    print()

    # Analyze each dataset
    results = []

    for i, cache_file in enumerate(embedding_files, 1):
        print(f"[{i}/{len(embedding_files)}] Analyzing: {cache_file.name}")
        print("-" * 80)

        try:
            # Load data
            idea_texts, embeddings, n_ideas = load_embeddings_from_cache(cache_file)

            # Check duplicates
            dup_stats = check_for_duplicates(idea_texts)

            # Calculate similarity statistics
            sim_stats = calculate_similarity_stats(embeddings)

            # Classify regime
            regime_id, size_class, structure_class = classify_regime(n_ideas, sim_stats['q90'])

            # Store results
            result = {
                'filename': cache_file.name,
                'n': n_ideas,
                'regime_id': regime_id,
                'size_class': size_class,
                'structure_class': structure_class,
                'q90': sim_stats['q90'],
                'q50': sim_stats['q50'],
                'kd_cv': sim_stats['kd_cv'],
                'dup_rate': dup_stats['duplicate_rate'],
                'compression_ratio': dup_stats['compression_ratio'],
                'sim_stats': sim_stats,
                'dup_stats': dup_stats,
            }
            results.append(result)

            # Print summary
            print(f"  n={n_ideas} → size_class={size_class}")
            print(f"  q90={sim_stats['q90']:.4f} → structure_class={structure_class}")
            print(f"  Regime: {regime_id}")
            print(f"  Duplicates: {dup_stats['duplicate_count']} ({dup_stats['duplicate_rate']:.1%})")
            print(f"  q50={sim_stats['q50']:.4f}, kd_cv={sim_stats['kd_cv']:.4f}")
            print()

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            print()
            continue

    if not results:
        print("❌ No datasets successfully analyzed")
        return

    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================

    print("=" * 80)
    print("SUMMARY: Regime Distribution")
    print("=" * 80)

    regime_counts = Counter(r['regime_id'] for r in results)
    for regime_id in sorted(regime_counts.keys()):
        count = regime_counts[regime_id]
        examples = [r['filename'][:40] for r in results if r['regime_id'] == regime_id][:2]
        print(f"{regime_id}: {count} dataset(s)")
        for ex in examples:
            print(f"  - {ex}...")
    print()

    # ==========================================================================
    # THRESHOLD VALIDATION
    # ==========================================================================

    print("=" * 80)
    print("THRESHOLD VALIDATION")
    print("=" * 80)
    print()

    print("Q90 Thresholds:")
    print(f"  Diffuse:   q90 < 0.65")
    print(f"  Mixed:     0.65 ≤ q90 < 0.80")
    print(f"  Coherent:  q90 ≥ 0.80")
    print()

    # Check for edge cases near boundaries
    near_065 = [r for r in results if 0.60 <= r['q90'] <= 0.70]
    near_080 = [r for r in results if 0.75 <= r['q90'] <= 0.85]

    if near_065:
        print(f"⚠️  {len(near_065)} dataset(s) near 0.65 boundary:")
        for r in near_065:
            print(f"  - {r['filename'][:40]}: q90={r['q90']:.4f} ({r['structure_class']})")
    else:
        print("✅ No datasets near 0.65 boundary")
    print()

    if near_080:
        print(f"⚠️  {len(near_080)} dataset(s) near 0.80 boundary:")
        for r in near_080:
            print(f"  - {r['filename'][:40]}: q90={r['q90']:.4f} ({r['structure_class']})")
    else:
        print("✅ No datasets near 0.80 boundary")
    print()

    # ==========================================================================
    # Q90 DISTRIBUTION
    # ==========================================================================

    print("=" * 80)
    print("Q90 Distribution Across All Datasets")
    print("=" * 80)

    q90_values = sorted([r['q90'] for r in results])

    print(f"Min q90:  {min(q90_values):.4f}")
    print(f"Q25 q90:  {np.percentile(q90_values, 25):.4f}")
    print(f"Median:   {np.percentile(q90_values, 50):.4f}")
    print(f"Q75 q90:  {np.percentile(q90_values, 75):.4f}")
    print(f"Max q90:  {max(q90_values):.4f}")
    print()

    print("Q90 by regime:")
    for regime_id in sorted(regime_counts.keys()):
        regime_q90s = [r['q90'] for r in results if r['regime_id'] == regime_id]
        if regime_q90s:
            print(f"  {regime_id}: min={min(regime_q90s):.4f}, median={np.median(regime_q90s):.4f}, max={max(regime_q90s):.4f}")
    print()

    # ==========================================================================
    # DETAILED INSPECTION OF SPECIFIC CASES
    # ==========================================================================

    print("=" * 80)
    print("DETAILED INSPECTION: Representative Cases")
    print("=" * 80)
    print()

    # Find one example of each regime (if available)
    regime_examples = {}
    for regime_id in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]:
        examples = [r for r in results if r['regime_id'] == regime_id]
        if examples:
            regime_examples[regime_id] = examples[0]

    for regime_id, result in sorted(regime_examples.items()):
        print(f"{regime_id} ({result['size_class']}+{result['structure_class']}):")
        print(f"  File: {result['filename']}")
        print(f"  n={result['n']}, q90={result['q90']:.4f}, q50={result['q50']:.4f}")
        print(f"  kd_cv={result['kd_cv']:.4f}")
        print(f"  Duplicates: {result['dup_stats']['duplicate_count']} ({result['dup_rate']:.1%})")

        # Similarity distribution
        sims = result['sim_stats']['similarities']
        bins = [0, 0.5, 0.65, 0.80, 0.90, 0.95, 1.0]
        hist, _ = np.histogram(sims, bins=bins)
        total = len(sims)

        print(f"  Similarity distribution:")
        for j in range(len(bins) - 1):
            pct = hist[j] / total * 100
            print(f"    [{bins[j]:.2f}, {bins[j+1]:.2f}): {pct:>5.1f}%")
        print()

    # ==========================================================================
    # BUG CHECK: Compare subsample vs full calculation
    # ==========================================================================

    print("=" * 80)
    print("BUG CHECK: Subsample Stability")
    print("=" * 80)
    print()

    # Pick one small dataset and compare full vs subsample
    small_datasets = [r for r in results if r['n'] < 200]
    if small_datasets:
        test_dataset = small_datasets[0]
        cache_file = cache_dir / test_dataset['filename']

        print(f"Testing: {test_dataset['filename']} (n={test_dataset['n']})")
        print()

        idea_texts, embeddings, n = load_embeddings_from_cache(cache_file)

        # Full calculation
        full_stats = calculate_similarity_stats(embeddings, sample_size=n)

        # Subsample calculations (3 runs)
        subsample_q90s = []
        for i in range(3):
            sub_stats = calculate_similarity_stats(embeddings, sample_size=100)
            subsample_q90s.append(sub_stats['q90'])

        print(f"Full calculation (n={n}):    q90={full_stats['q90']:.4f}")
        print(f"Subsample (100) - Run 1:      q90={subsample_q90s[0]:.4f}")
        print(f"Subsample (100) - Run 2:      q90={subsample_q90s[1]:.4f}")
        print(f"Subsample (100) - Run 3:      q90={subsample_q90s[2]:.4f}")
        print(f"Subsample std dev:            {np.std(subsample_q90s):.4f}")
        print(f"Difference (full - mean sub): {full_stats['q90'] - np.mean(subsample_q90s):.4f}")
        print()

        if abs(full_stats['q90'] - np.mean(subsample_q90s)) > 0.05:
            print("⚠️  WARNING: Significant subsample bias detected!")
        else:
            print("✅ Subsample is stable")
    else:
        print("No small datasets available for subsample test")

    print()

    # ==========================================================================
    # VISUALIZATION (if matplotlib available)
    # ==========================================================================

    try:
        print("=" * 80)
        print("VISUALIZATION: Similarity Distributions")
        print("=" * 80)
        print()

        # Plot similarity distributions for representative regimes
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Cosine Similarity Distributions by Regime', fontsize=16)

        for idx, (regime_id, result) in enumerate(sorted(regime_examples.items())):
            if idx >= 9:
                break

            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            sims = result['sim_stats']['similarities']

            # Histogram
            ax.hist(sims, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

            # Threshold lines
            ax.axvline(0.65, color='orange', linestyle='--', linewidth=2, label='Diffuse/Mixed')
            ax.axvline(0.80, color='red', linestyle='--', linewidth=2, label='Mixed/Coherent')
            ax.axvline(result['q90'], color='green', linestyle='-', linewidth=2, label=f'q90={result["q90"]:.2f}')

            ax.set_title(f'{regime_id}: {result["size_class"]}+{result["structure_class"]}\nn={result["n"]}')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Hide empty subplots
        for idx in range(len(regime_examples), 9):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')

        plt.tight_layout()

        # Save plot
        output_path = project_root / "regime_analysis_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to: {output_path}")
        print()

    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
        print()

    # ==========================================================================
    # RECOMMENDATIONS
    # ==========================================================================

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Check if thresholds need adjustment
    diffuse_q90s = [r['q90'] for r in results if r['structure_class'] == 'diffuse']
    mixed_q90s = [r['q90'] for r in results if r['structure_class'] == 'mixed']
    coherent_q90s = [r['q90'] for r in results if r['structure_class'] == 'coherent']

    if diffuse_q90s and max(diffuse_q90s) >= 0.65:
        print("⚠️  Some 'diffuse' datasets have q90 ≥ 0.65!")
        print("    Consider adjusting diffuse/mixed threshold")
        print()

    if mixed_q90s and (min(mixed_q90s) < 0.65 or max(mixed_q90s) >= 0.80):
        print("⚠️  'Mixed' regime spans wide range")
        print(f"    min={min(mixed_q90s):.4f}, max={max(mixed_q90s):.4f}")
        print("    This is expected - mixed is a transitional category")
        print()

    if coherent_q90s and min(coherent_q90s) < 0.80:
        print("⚠️  Some 'coherent' datasets have q90 < 0.80!")
        print("    Consider adjusting mixed/coherent threshold")
        print()

    # Final summary
    print("Threshold validation summary:")
    if diffuse_q90s:
        print(f"  Diffuse:   {len(diffuse_q90s)} datasets, q90 range [{min(diffuse_q90s):.4f}, {max(diffuse_q90s):.4f}]")
    if mixed_q90s:
        print(f"  Mixed:     {len(mixed_q90s)} datasets, q90 range [{min(mixed_q90s):.4f}, {max(mixed_q90s):.4f}]")
    if coherent_q90s:
        print(f"  Coherent:  {len(coherent_q90s)} datasets, q90 range [{min(coherent_q90s):.4f}, {max(coherent_q90s):.4f}]")
    print()

    if all([
        not diffuse_q90s or max(diffuse_q90s) < 0.65,
        not coherent_q90s or min(coherent_q90s) >= 0.80
    ]):
        print("✅ Thresholds 0.65 and 0.80 appear appropriate!")
    else:
        print("⚠️  Consider threshold adjustment based on findings above")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
