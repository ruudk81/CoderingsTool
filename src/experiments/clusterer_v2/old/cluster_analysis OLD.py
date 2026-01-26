#%%

"""
Cluster Analysis Experimentation Framework

Loads cached Step 5 clustering results and experiments with:
- TF-IDF keyword extraction for clusters
- LLM-based cluster descriptions enhanced with keywords
- Comparison with original Step 6 codebook generation

Usage:
    python experiments/cluster_analysis.py
"""

import os
import sys

# Path setup - add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random
import re

# Import from codebase
import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils import dataLoader
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE
from experiments.tfidf_analyzer import TfidfAnalyzer, TfidfConfig
from utils.llm import llm_create_sync, create_client
from pydantic import BaseModel, Field


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for cluster analysis experiments"""
    # Data loading
    filename: str
    var_name: str
    id_column: str = "DLNMID"
    sample_size: Optional[int] = None

    # TF-IDF settings
    tfidf_config: TfidfConfig = field(default_factory=TfidfConfig)

    # LLM description generation
    description_model: str = "gpt-4.1"
    use_keywords_in_prompt: bool = True  # Toggle keyword enhancement

    # Display settings
    n_sample_clusters: int = 5
    show_comparisons: bool = True  # Show original Step 6 codes if available
    verbose: bool = True


# Predefined experiment configurations for quick switching
EXPERIMENTS = {
    "baseline": TfidfConfig(
        max_features=1000,
        ngram_range=(1, 1),  # Unigrams only
        min_df=2,
        max_df=0.8,
        top_k_keywords=10,
        language="nl"
    ),
    "bigrams": TfidfConfig(
        max_features=2000,
        ngram_range=(1, 2),  # Unigrams + bigrams
        min_df=2,
        max_df=0.8,
        top_k_keywords=15,
        language="nl"
    ),
    "strict_filtering": TfidfConfig(
        max_features=500,
        ngram_range=(1, 2),
        min_df=3,  # Require keyword in 3+ documents
        max_df=0.6,  # Exclude very common terms
        top_k_keywords=8,
        language="nl"
    ),
}


# ============================================================================
# LLM DESCRIPTION GENERATION
# ============================================================================

class ClusterDescription(BaseModel):
    """LLM-generated cluster description"""
    theme: str = Field(..., description="Short thematic label for the cluster (3-8 words)")
    description: str = Field(..., description="Detailed description of the cluster content (1-2 sentences)")
    key_concepts: List[str] = Field(..., description="List of 3-5 key concepts/themes in this cluster")


def generate_cluster_description(
    cluster_id: int,
    ideas: List[str],
    keywords: Optional[List[Tuple[str, float]]] = None,
    var_lab: str = "",
    model: str = "gpt-4.1",
    verbose: bool = False
) -> ClusterDescription:
    """
    Generate an LLM-based description for a cluster

    Args:
        cluster_id: Cluster identifier
        ideas: List of idea texts in the cluster
        keywords: Optional list of (keyword, score) tuples from TF-IDF
        var_lab: Survey question text for context
        model: LLM model to use
        verbose: Enable verbose output

    Returns:
        ClusterDescription with theme, description, and key concepts
    """
    # Sample ideas if cluster is too large
    max_ideas_for_prompt = 20
    if len(ideas) > max_ideas_for_prompt:
        sampled_ideas = random.sample(ideas, max_ideas_for_prompt)
    else:
        sampled_ideas = ideas

    # Build prompt
    prompt_parts = [
        f"Survey question: {var_lab}\n" if var_lab else "",
        f"\nCluster {cluster_id} contains {len(ideas)} response ideas.\n",
        "\nSample ideas from this cluster:\n"
    ]

    for i, idea in enumerate(sampled_ideas, 1):
        prompt_parts.append(f"{i}. {idea}\n")

    # Add TF-IDF keywords if provided
    if keywords:
        prompt_parts.append("\nStatistical keyword analysis (TF-IDF) identified these important terms:\n")
        for i, (keyword, score) in enumerate(keywords[:10], 1):
            prompt_parts.append(f"  • {keyword}\n")

    prompt_parts.append(
        "\nBased on these ideas" + (" and keywords" if keywords else "") + ", provide:\n"
        "1. A short thematic label (3-8 words) that captures the essence of this cluster\n"
        "2. A detailed description (1-2 sentences) explaining what these responses have in common\n"
        "3. A list of 3-5 key concepts/themes present in this cluster"
    )

    prompt = "".join(prompt_parts)

    # Call LLM
    try:
        client = create_client(model=model, async_mode=False)
        description = llm_create_sync(
            client=client,
            model=model,
            prompt=prompt,
            response_model=ClusterDescription,
            temperature=0.3,
            max_tokens=1000
        )
        return description
    except Exception as e:
        # Print full error for debugging
        print(f"\n✗ LLM Error for cluster {cluster_id}: {type(e).__name__}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        # Return fallback
        return ClusterDescription(
            theme=f"Cluster {cluster_id}",
            description="LLM description generation failed",
            key_concepts=[]
        )


# ============================================================================
# CLUSTER DATA EXTRACTION
# ============================================================================

def extract_cluster_ideas(cluster_results: List[models.ClusterModel]) -> Dict[int, List[str]]:
    """
    Extract cluster ideas from ClusterModel objects (similar to codeGenerator pattern)

    Args:
        cluster_results: List of ClusterModel instances from Step 5

    Returns:
        Dict mapping cluster_id to list of idea texts
    """
    clusters = {}

    for result in cluster_results:
        ideas_list = result.response_ideas or []

        for idea in ideas_list:
            # Use initial_cluster (before Step 6 expansion)
            cluster_id = idea.initial_cluster

            if cluster_id is not None and cluster_id != -1:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []

                # Extract idea text
                idea_text = idea.idea if hasattr(idea, 'idea') else str(idea)
                clusters[cluster_id].append(idea_text)

    return clusters


def load_original_step6_codes(
    filename: str,
    variable_key: str,
    cache_manager: CacheManager
) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Load original Step 6 codebook for comparison

    Args:
        filename: SPSS filename
        variable_key: Cache variable key
        cache_manager: Cache manager instance

    Returns:
        Dict mapping cluster_id to {code, definition} or None if not available
    """
    try:
        from utils import codeGenerator

        # Try to load Step 6 reasoning results
        reasoning_models = cache_manager.load_from_cache(
            filename,
            "codebook_generation_reasoning",
            variable_key,
            codeGenerator.CodeGeneratorReasoningResults
        )

        if not reasoning_models or len(reasoning_models) == 0:
            return None

        codebook_reasoning = reasoning_models[0]
        if not hasattr(codebook_reasoning, 'codebook') or not codebook_reasoning.codebook:
            return None

        # Extract cluster -> code mapping
        cluster_codes = {}
        for entry in codebook_reasoning.codebook:
            source_clusters = entry.get('source_cluster_id', '').split(',')
            code = entry.get('code', 'Unknown')
            definition = entry.get('definition', '')

            for cluster_id_str in source_clusters:
                cluster_id_str = cluster_id_str.strip()
                if cluster_id_str and cluster_id_str.isdigit():
                    cluster_id = int(cluster_id_str)
                    cluster_codes[cluster_id] = {
                        'code': code,
                        'definition': definition
                    }

        return cluster_codes

    except Exception as e:
        print(f"Note: Could not load original Step 6 codes: {e}")
        return None


# ============================================================================
# DISPLAY UTILITIES
# ============================================================================

def display_sample_clusters(
    clusters: Dict[int, List[str]],
    cluster_keywords: Dict[int, List[Tuple[str, float]]],
    cluster_descriptions: Dict[int, ClusterDescription],
    original_codes: Optional[Dict[int, Dict[str, str]]] = None,
    n_samples: int = 5,
    ideas_per_cluster: int = 5
):
    """
    Display sample cluster analysis results

    Args:
        clusters: Dict mapping cluster_id to idea texts
        cluster_keywords: Dict mapping cluster_id to keywords
        cluster_descriptions: Dict mapping cluster_id to LLM descriptions
        original_codes: Optional dict with original Step 6 codes
        n_samples: Number of clusters to display
        ideas_per_cluster: Number of sample ideas to show per cluster
    """
    # Sample random clusters
    cluster_ids = list(clusters.keys())
    if len(cluster_ids) > n_samples:
        sampled_ids = random.sample(cluster_ids, n_samples)
    else:
        sampled_ids = cluster_ids

    # Sort for consistent display
    sampled_ids = sorted(sampled_ids)

    for cluster_id in sampled_ids:
        ideas = clusters[cluster_id]
        keywords = cluster_keywords.get(cluster_id, [])
        description = cluster_descriptions.get(cluster_id)

        print(f"\n{'─' * 80}")
        print(f"CLUSTER {cluster_id} (n={len(ideas)})")
        print(f"{'─' * 80}\n")

        # TF-IDF Keywords
        print("TF-IDF Keywords (top 10):")
        if keywords:
            for i, (keyword, score) in enumerate(keywords[:10], 1):
                print(f"  {i}. {keyword:<30} ({score:.3f})")
        else:
            print("  (No keywords extracted)")
        print()

        # LLM Description
        if description:
            print("LLM-Generated Description:")
            print(f"  Theme: {description.theme}")
            print(f"  Description: {description.description}")
            if description.key_concepts:
                print(f"  Key Concepts: {', '.join(description.key_concepts)}")
        else:
            print("LLM-Generated Description: (Not available)")
        print()

        # Sample Ideas
        print(f"Sample Ideas ({min(ideas_per_cluster, len(ideas))} of {len(ideas)}):")
        sample_ideas = random.sample(ideas, min(ideas_per_cluster, len(ideas)))
        for idea in sample_ideas:
            # Clean idea text (remove metadata brackets)
            cleaned = re.sub(r'\[.*?\]', '', idea)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            print(f"  • {cleaned}")
        print()

        # Original Step 6 Code (if available)
        if original_codes and cluster_id in original_codes:
            code_info = original_codes[cluster_id]
            print("Original Step 6 Code (for comparison):")
            print(f"  Code: {code_info['code']}")
            if code_info['definition']:
                # Truncate long definitions
                definition = code_info['definition']
                if len(definition) > 100:
                    definition = definition[:97] + "..."
                print(f"  Definition: {definition}")
            print()

    print(f"\n{'═' * 80}\n")


def display_summary(
    clusters: Dict[int, List[str]],
    cluster_keywords: Dict[int, List[Tuple[str, float]]],
    config: ExperimentConfig
):
    """Display experiment summary statistics"""
    total_ideas = sum(len(ideas) for ideas in clusters.values())
    avg_cluster_size = total_ideas / len(clusters) if clusters else 0
    cluster_sizes = [len(ideas) for ideas in clusters.values()]

    print(f"{'═' * 80}")
    print("CLUSTER ANALYSIS EXPERIMENT SUMMARY")
    print(f"{'═' * 80}")
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Total ideas: {total_ideas}")
    print(f"Average cluster size: {avg_cluster_size:.1f}")
    print(f"Cluster size range: {min(cluster_sizes)} - {max(cluster_sizes)}")
    print(f"\nTF-IDF Configuration:")
    print(f"  N-grams: {config.tfidf_config.ngram_range}")
    print(f"  Min DF: {config.tfidf_config.min_df}")
    print(f"  Max DF: {config.tfidf_config.max_df}")
    print(f"  Top K keywords: {config.tfidf_config.top_k_keywords}")
    print(f"  Language: {config.tfidf_config.language}")
    print(f"\nKeywords enhanced LLM prompts: {config.use_keywords_in_prompt}")
    print(f"{'═' * 80}\n")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiment(config: ExperimentConfig):
    """
    Run cluster analysis experiment

    Args:
        config: ExperimentConfig with all settings
    """
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)

    # Generate variable key
    selected_variables = [config.var_name]
    is_merged = False
    variable_key = generate_enhanced_variable_key(
        selected_variables,
        is_merged,
        sample_size=config.sample_size,
        merge_config=None
    )

    if config.verbose:
        print(f"\n{'═' * 80}")
        print("LOADING CACHED STEP 5 RESULTS")
        print(f"{'═' * 80}")
        print(f"Filename: {config.filename}")
        print(f"Variable: {config.var_name}")
        print(f"Variable key: {variable_key}")
        print(f"{'═' * 80}\n")

    # Step 1: Load cached Step 5 results
    try:
        cluster_results = cache_manager.load_from_cache(
            config.filename,
            "initial_clusters",
            variable_key,
            models.ClusterModel
        )
        if config.verbose:
            print(f"✓ Loaded {len(cluster_results)} ClusterModel objects from cache\n")
    except Exception as e:
        print(f"✗ Error loading cached results: {e}")
        print("\nMake sure you have run the pipeline to Step 5 first:")
        print(f"  cd src")
        print(f"  python pipeline.py  # with RUN_UNTIL_STEP = 5")
        return

    # Step 2: Extract cluster data
    clusters = extract_cluster_ideas(cluster_results)
    if config.verbose:
        print(f"✓ Extracted {len(clusters)} clusters\n")

    if not clusters:
        print("✗ No clusters found in cached results")
        return

    # Step 3: Run TF-IDF analysis
    if config.verbose:
        print(f"{'═' * 80}")
        print("RUNNING TF-IDF KEYWORD EXTRACTION")
        print(f"{'═' * 80}\n")

    tfidf_analyzer = TfidfAnalyzer(config=config.tfidf_config, verbose=config.verbose)
    cluster_keywords = tfidf_analyzer.extract_keywords(clusters)

    # Step 4: Generate LLM descriptions
    if config.verbose:
        print(f"\n{'═' * 80}")
        print("GENERATING LLM CLUSTER DESCRIPTIONS")
        print(f"{'═' * 80}\n")

    # Get variable label for context (fallback if file not accessible)
    try:
        data_loader = dataLoader.DataLoader(verbose=False)
        var_lab = data_loader.get_varlab(filename=config.filename, var_name=config.var_name)
    except FileNotFoundError:
        # Fallback: use variable name if SPSS file not accessible
        var_lab = f"Survey question for variable {config.var_name}"
        if config.verbose:
            print(f"Note: SPSS file not found, using fallback variable label\n")

    cluster_descriptions = {}
    for cluster_id, ideas in clusters.items():
        if config.verbose:
            print(f"Generating description for cluster {cluster_id}...", end=" ")

        keywords = cluster_keywords.get(cluster_id) if config.use_keywords_in_prompt else None

        description = generate_cluster_description(
            cluster_id=cluster_id,
            ideas=ideas,
            keywords=keywords,
            var_lab=var_lab,
            model=config.description_model,
            verbose=False
        )

        cluster_descriptions[cluster_id] = description

        if config.verbose:
            print(f"✓ {description.theme}")

    # Step 5: Load original Step 6 codes for comparison (if requested)
    original_codes = None
    if config.show_comparisons:
        if config.verbose:
            print(f"\n{'═' * 80}")
            print("LOADING ORIGINAL STEP 6 CODES FOR COMPARISON")
            print(f"{'═' * 80}\n")

        original_codes = load_original_step6_codes(config.filename, variable_key, cache_manager)
        if original_codes:
            if config.verbose:
                print(f"✓ Loaded {len(original_codes)} original codes\n")
        else:
            if config.verbose:
                print("Note: Original Step 6 codes not available (run full pipeline first)\n")

    # Step 6: Display results
    print(f"\n{'═' * 80}")
    print("EXPERIMENT RESULTS")
    print(f"{'═' * 80}\n")

    display_summary(clusters, cluster_keywords, config)
    display_sample_clusters(
        clusters,
        cluster_keywords,
        cluster_descriptions,
        original_codes,
        n_samples=config.n_sample_clusters,
        ideas_per_cluster=5
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configure experiment here
    config = ExperimentConfig(
        filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
        var_name="Q20",
        id_column="DLNMID",
        sample_size=50,

        # Choose TF-IDF experiment: "baseline", "bigrams", or "strict_filtering"
        tfidf_config=EXPERIMENTS["bigrams"],

        # LLM settings
        description_model="gpt-4.1",
        use_keywords_in_prompt=True,

        # Display settings
        n_sample_clusters=5,
        show_comparisons=True,
        verbose=True
    )

    run_experiment(config)


# ============================================================================
# JUPYTER/NOTEBOOK HELPER
# ============================================================================
# Uncomment and run this cell in Jupyter/VS Code notebooks instead of main:
#
# config = ExperimentConfig(
#     filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
#     var_name="Q20",
#     sample_size=50,
#     tfidf_config=EXPERIMENTS["bigrams"],  # Try: "baseline", "bigrams", "strict_filtering"
#     use_keywords_in_prompt=True,  # Toggle to compare with/without keywords
#     n_sample_clusters=5,
#     show_comparisons=True,
#     verbose=True
# )
# run_experiment(config)

# %%
