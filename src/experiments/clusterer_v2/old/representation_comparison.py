#%%

"""
Compare representation models on cached Step 5 clustering results

Compares multiple keyword extraction methods:
- Standard TF-IDF (existing baseline)
- c-TF-IDF (BERTopic baseline)
- c-TF-IDF + MMR (diversity-aware)
- c-TF-IDF + KeyBERT (embedding-based)
- c-TF-IDF + LLM (GPT-enhanced)

Usage:
    python experiments/representation_comparison.py

    # Or import and use programmatically:
    from experiments.representation_comparison import compare_all_models

    results = compare_all_models(
        cluster_results=cached_results,
        n_sample_clusters=10,
        export_excel=True
    )
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Add src to path (works for both script and notebook execution)
if 'experiments' in os.getcwd():
    # Running from experiments directory
    src_path = str(Path(os.getcwd()).parent)
else:
    # Running from project root or src
    src_path = str(Path(__file__).parent.parent) if '__file__' in globals() else str(Path(os.getcwd()) / 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from config import CacheConfig
from experiments.tfidf_analyzer import TfidfAnalyzer, TfidfConfig
from experiments.representation.ctfidf_representation import CTfidfRepresentation
from experiments.representation.mmr_representation import MMRRepresentation
from experiments.representation.keybert_representation import KeyBERTRepresentation
from experiments.representation.llm_representation import LLMRepresentation
import models


@dataclass
class ComparisonConfig:
    """Configuration for representation comparison experiment"""
    filename: str
    var_name: str
    sample_size: int
    n_sample_clusters: int = 10  # Number of clusters to display
    export_excel: bool = True
    export_filename: str = "representation_comparison.xlsx"
    verbose: bool = True


def extract_cluster_ideas(cluster_results: List[models.ClusterModel]) -> Dict[int, List[str]]:
    """
    Extract cluster → ideas mapping from ClusterModel objects

    Args:
        cluster_results: List of ClusterModel instances from Step 5

    Returns:
        Dict mapping cluster_id to list of cleaned idea texts
    """
    import re

    def strip_context_tags(text: str) -> str:
        """Remove context tags from idea text"""
        pattern = r'\[(?:lang|domain|topic|perspective|entity|intent|sentiment|sense)=[^\]]*\]'
        cleaned = re.sub(pattern, '', text)
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()

    clusters = {}

    for result in cluster_results:
        ideas_list = result.response_ideas or []

        for idea in ideas_list:
            cluster_id = idea.initial_cluster

            if cluster_id is not None and cluster_id != -1:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []

                idea_text = idea.idea if hasattr(idea, 'idea') else str(idea)
                cleaned_text = strip_context_tags(idea_text)
                clusters[cluster_id].append(cleaned_text)

    return clusters


def run_standard_tfidf(clusters: Dict[int, List[str]], verbose: bool = False) -> Dict[int, List[Tuple[str, float]]]:
    """Run standard TF-IDF analysis"""
    if verbose:
        print("\n[1/5] Running Standard TF-IDF...")

    config = TfidfConfig(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        top_k_keywords=15
    )

    analyzer = TfidfAnalyzer(config, verbose=verbose)
    keywords = analyzer.extract_keywords(clusters)

    if verbose:
        print(f"      Extracted keywords for {len(keywords)} clusters")

    return keywords


def run_ctfidf(clusters: Dict[int, List[str]], verbose: bool = False) -> Dict[int, List[Tuple[str, float]]]:
    """Run c-TF-IDF analysis"""
    if verbose:
        print("\n[2/5] Running c-TF-IDF...")

    ctfidf = CTfidfRepresentation(
        top_k=15,
        bm25_weighting=True,
        reduce_frequent_words=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    keywords = ctfidf.extract_keywords(clusters, verbose=verbose)

    if verbose:
        print(f"      Extracted keywords for {len(keywords)} clusters")

    return keywords


def run_ctfidf_mmr(clusters: Dict[int, List[str]], verbose: bool = False) -> Dict[int, List[Tuple[str, float]]]:
    """Run c-TF-IDF + MMR analysis"""
    if verbose:
        print("\n[3/5] Running c-TF-IDF + MMR...")

    # First get c-TF-IDF baseline
    ctfidf = CTfidfRepresentation(
        top_k=50,  # Get more candidates for MMR
        bm25_weighting=True,
        reduce_frequent_words=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    # Build vocabulary and matrix
    cluster_ids = sorted(clusters.keys())
    cluster_docs = [" ".join(clusters[cid]) for cid in cluster_ids]

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True
    )

    count_matrix = vectorizer.fit_transform(cluster_docs)
    vocabulary = vectorizer.get_feature_names_out()
    ctfidf_matrix = ctfidf.transformer.fit_transform(count_matrix)

    # Apply MMR to each cluster
    mmr = MMRRepresentation(diversity=0.3, top_k=10)

    keywords = {}
    for idx, cluster_id in enumerate(cluster_ids):
        ctfidf_scores = ctfidf_matrix[idx].toarray()[0]
        keywords[cluster_id] = mmr.extract_topics(
            cluster_id=cluster_id,
            ctfidf_scores=ctfidf_scores,
            vocabulary=vocabulary,
            cluster_texts=clusters[cluster_id]
        )

    if verbose:
        print(f"      Extracted keywords for {len(keywords)} clusters")

    return keywords


def run_ctfidf_keybert(clusters: Dict[int, List[str]], embeddings_dict: Dict[int, np.ndarray] = None, verbose: bool = False) -> Dict[int, List[Tuple[str, float]]]:
    """Run c-TF-IDF + KeyBERT analysis"""
    if verbose:
        print("\n[4/5] Running c-TF-IDF + KeyBERT...")

    # First get c-TF-IDF baseline
    ctfidf = CTfidfRepresentation(
        top_k=50,
        bm25_weighting=True,
        reduce_frequent_words=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    # Build vocabulary and matrix
    cluster_ids = sorted(clusters.keys())
    cluster_docs = [" ".join(clusters[cid]) for cid in cluster_ids]

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True
    )

    count_matrix = vectorizer.fit_transform(cluster_docs)
    vocabulary = vectorizer.get_feature_names_out()
    ctfidf_matrix = ctfidf.transformer.fit_transform(count_matrix)

    # Apply KeyBERT to each cluster
    keybert = KeyBERTRepresentation(top_k=10, weight=0.5)

    keywords = {}
    for idx, cluster_id in enumerate(cluster_ids):
        ctfidf_scores = ctfidf_matrix[idx].toarray()[0]
        cluster_embeddings = embeddings_dict.get(cluster_id) if embeddings_dict else None

        keywords[cluster_id] = keybert.extract_topics(
            cluster_id=cluster_id,
            ctfidf_scores=ctfidf_scores,
            vocabulary=vocabulary,
            cluster_texts=clusters[cluster_id],
            embeddings=cluster_embeddings
        )

    if verbose:
        print(f"      Extracted keywords for {len(keywords)} clusters")

    return keywords


def run_ctfidf_llm(clusters: Dict[int, List[str]], verbose: bool = False) -> Dict[int, List[Tuple[str, float]]]:
    """Run c-TF-IDF + LLM analysis"""
    if verbose:
        print("\n[5/5] Running c-TF-IDF + LLM Enhancement...")

    # First get c-TF-IDF baseline
    ctfidf = CTfidfRepresentation(
        top_k=50,
        bm25_weighting=True,
        reduce_frequent_words=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    # Build vocabulary and matrix
    cluster_ids = sorted(clusters.keys())
    cluster_docs = [" ".join(clusters[cid]) for cid in cluster_ids]

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True
    )

    count_matrix = vectorizer.fit_transform(cluster_docs)
    vocabulary = vectorizer.get_feature_names_out()
    ctfidf_matrix = ctfidf.transformer.fit_transform(count_matrix)

    # Apply LLM to each cluster
    llm = LLMRepresentation(model="gpt-4.1-mini", top_k=10, verbose=verbose)

    keywords = {}
    for idx, cluster_id in enumerate(cluster_ids):
        ctfidf_scores = ctfidf_matrix[idx].toarray()[0]

        if verbose:
            print(f"      Processing cluster {cluster_id}...", end=" ")

        keywords[cluster_id] = llm.extract_topics(
            cluster_id=cluster_id,
            ctfidf_scores=ctfidf_scores,
            vocabulary=vocabulary,
            cluster_texts=clusters[cluster_id]
        )

        if verbose:
            print("✓")

    if verbose:
        print(f"      Extracted keywords for {len(keywords)} clusters")

    return keywords


def display_comparison_table(
    results: Dict[str, Dict[int, List[Tuple[str, float]]]],
    clusters: Dict[int, List[str]],
    n_sample: int = 10
):
    """Display side-by-side comparison of all models"""
    cluster_ids = sorted(clusters.keys())

    if n_sample and len(cluster_ids) > n_sample:
        sampled_ids = random.sample(cluster_ids, n_sample)
        sampled_ids = sorted(sampled_ids)
        print(f"\n{'='*100}")
        print(f"Displaying {n_sample} randomly selected clusters (out of {len(cluster_ids)} total)")
    else:
        sampled_ids = cluster_ids
        print(f"\n{'='*100}")
        print(f"Displaying all {len(cluster_ids)} clusters")

    print(f"{'='*100}\n")

    model_names = list(results.keys())

    for cluster_id in sampled_ids:
        cluster_size = len(clusters[cluster_id])
        print(f"\n{'─'*100}")
        print(f"Cluster {cluster_id} ({cluster_size} ideas)")
        print(f"{'─'*100}\n")

        # Display each model's keywords
        for model_name in model_names:
            keywords = results[model_name].get(cluster_id, [])
            print(f"{model_name}:")

            if keywords:
                for i, (kw, score) in enumerate(keywords[:10], 1):
                    print(f"  {i:2d}. {kw:<30} ({score:.4f})")
            else:
                print("  (no keywords)")

            print()


def calculate_comparison_metrics(
    results: Dict[str, Dict[int, List[Tuple[str, float]]]],
    clusters: Dict[int, List[str]]
) -> dict:
    """Calculate comparison metrics across all models"""
    metrics = {}

    for model_name, model_results in results.items():
        # Coverage: % of clusters with keywords
        total_clusters = len(clusters)
        clusters_with_keywords = sum(1 for kws in model_results.values() if kws)
        coverage = clusters_with_keywords / total_clusters if total_clusters > 0 else 0.0

        # Average keywords per cluster
        avg_keywords = np.mean([len(kws) for kws in model_results.values()])

        # Diversity: average uniqueness of keywords within each cluster
        diversities = []
        for cluster_id, keywords in model_results.items():
            if len(keywords) > 1:
                # Count unique tokens (simple heuristic)
                all_tokens = set()
                for kw, _ in keywords:
                    all_tokens.update(kw.lower().split())
                diversity = len(all_tokens) / len(keywords)
                diversities.append(diversity)

        avg_diversity = np.mean(diversities) if diversities else 0.0

        metrics[model_name] = {
            "coverage": coverage,
            "avg_keywords": avg_keywords,
            "avg_diversity": avg_diversity,
            "total_clusters": total_clusters
        }

    return metrics


def print_metrics_report(metrics: dict):
    """Print comparison metrics report"""
    print(f"\n{'='*100}")
    print("COMPARISON METRICS")
    print(f"{'='*100}\n")

    print(f"{'Model':<30} {'Coverage':<12} {'Avg Keywords':<15} {'Avg Diversity':<15}")
    print(f"{'-'*30} {'-'*12} {'-'*15} {'-'*15}")

    for model_name, model_metrics in metrics.items():
        print(
            f"{model_name:<30} "
            f"{model_metrics['coverage']:<12.1%} "
            f"{model_metrics['avg_keywords']:<15.1f} "
            f"{model_metrics['avg_diversity']:<15.2f}"
        )

    print(f"\n{'='*100}\n")


def export_comparison_excel(
    results: Dict[str, Dict[int, List[Tuple[str, float]]]],
    clusters: Dict[int, List[str]],
    filename: str = "representation_comparison.xlsx"
):
    """Export comparison results to Excel with multiple sheets"""
    import pandas as pd

    output_path = Path(__file__).parent.parent.parent / "exports" / filename

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Overview
        overview_data = []
        for model_name in results.keys():
            for cluster_id in sorted(clusters.keys()):
                keywords = results[model_name].get(cluster_id, [])
                kw_str = ", ".join([kw for kw, _ in keywords[:10]])

                overview_data.append({
                    "Model": model_name,
                    "Cluster ID": cluster_id,
                    "Cluster Size": len(clusters[cluster_id]),
                    "Keywords": kw_str
                })

        df_overview = pd.DataFrame(overview_data)
        df_overview.to_excel(writer, sheet_name="Overview", index=False)

        # Sheet 2-N: One sheet per model with detailed scores
        for model_name, model_results in results.items():
            sheet_name = model_name.replace(" ", "_")[:31]  # Excel sheet name limit
            model_data = []

            for cluster_id in sorted(clusters.keys()):
                keywords = model_results.get(cluster_id, [])
                for rank, (kw, score) in enumerate(keywords, 1):
                    model_data.append({
                        "Cluster ID": cluster_id,
                        "Rank": rank,
                        "Keyword": kw,
                        "Score": score
                    })

            if model_data:
                df_model = pd.DataFrame(model_data)
                df_model.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nExported comparison to: {output_path}")


def compare_all_models(
    cluster_results: Optional[List[models.ClusterModel]] = None,
    config: Optional[ComparisonConfig] = None,
    n_sample_clusters: Optional[int] = 10,
    export_excel: bool = True
) -> dict:
    """
    Run all representation models and generate comparison report

    Args:
        cluster_results: Cached Step 5 results (or None to load from cache)
        config: ComparisonConfig (or None to use defaults)
        n_sample_clusters: Number of clusters to display (None = all)
        export_excel: Export results to Excel

    Returns:
        Dict with results from each model
    """
    # Load cluster results if not provided
    if cluster_results is None:
        if config is None:
            raise ValueError("Must provide either cluster_results or config")

        print(f"Loading cached Step 5 results: {config.filename}, {config.var_name}...")

        # Initialize cache manager
        cache_config = CacheConfig()
        cache_manager = CacheManager(cache_config)

        # Generate variable key (matching cluster_analysis.py pattern)
        variable_key = generate_enhanced_variable_key(
            selected_variables=[config.var_name],
            is_merged=False,
            sample_size=config.sample_size
        )

        # Load from cache
        cluster_results = cache_manager.load_from_cache(
            filename=config.filename,
            step="step_5_cluster",
            variable_key=variable_key,
            model_cls=models.ClusterModel
        )

        if not cluster_results:
            raise ValueError("No cached Step 5 results found")

    # Extract clusters
    print(f"\nExtracting cluster → ideas mapping...")
    clusters = extract_cluster_ideas(cluster_results)
    print(f"Found {len(clusters)} clusters")

    # Run all models
    results = {}

    results['Standard TF-IDF'] = run_standard_tfidf(clusters, verbose=True)
    results['c-TF-IDF'] = run_ctfidf(clusters, verbose=True)
    results['c-TF-IDF + MMR'] = run_ctfidf_mmr(clusters, verbose=True)
    results['c-TF-IDF + KeyBERT'] = run_ctfidf_keybert(clusters, verbose=True)
    results['c-TF-IDF + LLM'] = run_ctfidf_llm(clusters, verbose=True)

    # Display comparison
    display_comparison_table(results, clusters, n_sample=n_sample_clusters)

    # Calculate metrics
    metrics = calculate_comparison_metrics(results, clusters)
    print_metrics_report(metrics)

    # Export to Excel
    if export_excel:
        export_filename = config.export_filename if config else "representation_comparison.xlsx"
        export_comparison_excel(results, clusters, filename=export_filename)

    return {
        "results": results,
        "metrics": metrics,
        "clusters": clusters
    }


if __name__ == "__main__":
    # Example usage
    config = ComparisonConfig(
        filename="your_data_file.sav",
        var_name="Q20",
        sample_size=50,
        n_sample_clusters=10,
        export_excel=True,
        verbose=True
    )

    print("="*100)
    print("REPRESENTATION MODEL COMPARISON")
    print("="*100)

    results = compare_all_models(config=config, n_sample_clusters=10, export_excel=True)

    print("\nComparison complete!")

# %%
