#%%

"""
Cluster Analysis Experimentation Framework

Loads cached Step 5 clustering results and experiments with:
- Keyword extraction: Standard TF-IDF or BERTopic-inspired c-TF-IDF
- LLM-based cluster descriptions enhanced with keywords
- Comparison with original Step 6 codebook generation

Keyword Extraction Methods:
    - TF-IDF: Standard term frequency-inverse document frequency
    - c-TF-IDF: Class-based TF-IDF from BERTopic (treats clusters as classes)
      * Better at identifying cluster-distinguishing terms
      * Uses BM25 weighting for improved short text performance
      * Applies frequency reduction to reduce impact of very common words

Usage:
    python experiments/cluster_analysis.py

    To switch methods, modify keyword_method in ExperimentConfig:
        keyword_method="tfidf"   # Standard TF-IDF
        keyword_method="ctfidf"  # BERTopic c-TF-IDF (recommended)
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
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE, API_PROVIDER
from experiments.tfidf_analyzer import TfidfAnalyzer, TfidfConfig
from experiments.prompts import CLUSTER_DESCRIPTION_PROMPT
from utils.llm import llm_create_sync, create_client
from pydantic import BaseModel, Field


# ============================================================================
# AZURE PROVIDER VERIFICATION
# ============================================================================

# Verify Azure provider is active at module load time
if API_PROVIDER != "azure":
    print(f"⚠️  WARNING: API_PROVIDER is set to '{API_PROVIDER}' (expected 'azure')")
    print(f"    To use Azure, set API_PROVIDER='azure' in src/config.py")
else:
    print(f"✓ Using Azure OpenAI provider")


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

    # Keyword extraction method: "tfidf" or "ctfidf"
    keyword_method: str = "tfidf"  # NEW: Choose between standard TF-IDF and c-TF-IDF

    # TF-IDF settings (for standard TF-IDF)
    tfidf_config: TfidfConfig = field(default_factory=TfidfConfig)

    # c-TF-IDF settings (for BERTopic-inspired c-TF-IDF)
    ctfidf_top_k: int = 15
    ctfidf_bm25_weighting: bool = True
    ctfidf_reduce_frequent_words: bool = True
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)
    ctfidf_min_df: int = 1
    ctfidf_max_df: float = 0.95

    # LLM description generation
    description_model: str = "gpt-5.2"
    use_keywords_in_prompt: bool = True  # Toggle keyword enhancement
    max_ideas_per_cluster: int = 10  # Maximum ideas to include in LLM prompt

    # Display settings
    n_sample_clusters: Optional[int] = None  # None = display all clusters
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

# c-TF-IDF experiment note:
# To use c-TF-IDF instead of standard TF-IDF, set keyword_method="ctfidf" in ExperimentConfig
# and configure ctfidf_* parameters (see ExperimentConfig dataclass above)


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
    max_ideas: int = 10,
    verbose: bool = False
) -> ClusterDescription:
    """
    Generate an LLM-based description for a cluster

    Args:
        cluster_id: Cluster identifier
        ideas: List of idea texts in the cluster
        keywords: Optional list of (keyword, score) tuples from TF-IDF/c-TF-IDF
        var_lab: Survey question text for context
        model: LLM model to use
        max_ideas: Maximum ideas to include in prompt
        verbose: Enable verbose output

    Returns:
        ClusterDescription with theme, description, and key concepts
    """
    # Sample ideas based on config max
    sample_ideas = ideas
    if len(ideas) > max_ideas:
        sample_ideas = random.sample(ideas, max_ideas)
        if verbose:
            print(f"  (sampled {max_ideas}/{len(ideas)} ideas)", end="")

    # Format ideas list
    ideas_formatted = "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(sample_ideas))

    # Format keywords section
    if keywords:
        keyword_lines = [f"  • {kw} (score: {score:.3f})" for kw, score in keywords[:10]]
        keywords_section = f"""Statistical keyword analysis identified these cluster-distinguishing terms:

{chr(10).join(keyword_lines)}

These keywords highlight terms that are statistically important for distinguishing this cluster from others."""
    else:
        keywords_section = "(No statistical keywords provided)"

    # Build prompt using template from experiments/prompts.py
    prompt = CLUSTER_DESCRIPTION_PROMPT.format(
        language="Dutch",  # Default language for this dataset
        survey_question=var_lab,
        cluster_id=cluster_id,
        num_ideas=len(ideas),
        keywords_section=keywords_section,
        ideas_list=ideas_formatted
    )

    # Call LLM
    try:
        if verbose:
            print(f"  [Provider: {API_PROVIDER}, Model: {model}]", end=" ")

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

def strip_context_tags(text: str) -> str:
    """
    Remove context tags from idea text

    Tags format (from ideaExtractor._format_idea_with_specifiers, line 1033):
    - Generic tags (line 1): [lang=...][domain=...][topic=...][perspective=...][entity=...][intent=...]
    - Specific tags (line 2): [sentiment=...][sense=...]
    - Text (line 3): actual idea content

    Examples:
        Input: "[lang=nl-NL][domain=voedingsmiddelenindustrie]\n[sentiment=positive][sense=suggestion]\nporties zijn te klein"
        Output: "porties zijn te klein"

    Args:
        text: Idea text possibly containing context tags

    Returns:
        Cleaned text without tags
    """
    # Pattern matches: [key=value] where key is one of the 8 tag types
    # Generic tags: lang, domain, topic, perspective, entity, intent
    # Specific tags: sentiment, sense
    pattern = r'\[(?:lang|domain|topic|perspective|entity|intent|sentiment|sense)=[^\]]*\]'
    cleaned = re.sub(pattern, '', text)

    # Clean up any extra whitespace and newlines from tag removal
    cleaned = ' '.join(cleaned.split())  # Normalize all whitespace to single spaces
    cleaned = cleaned.strip()

    return cleaned


def extract_cluster_ideas(cluster_results: List[models.ClusterModel]) -> Dict[int, List[str]]:
    """
    Extract cluster ideas from ClusterModel objects (similar to codeGenerator pattern)

    Context tags are automatically stripped from idea texts to prevent pollution
    of c-TF-IDF keyword extraction and LLM analysis.

    Args:
        cluster_results: List of ClusterModel instances from Step 5

    Returns:
        Dict mapping cluster_id to list of CLEANED idea texts (context tags removed)
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

                # Extract idea text and STRIP CONTEXT TAGS
                idea_text = idea.idea if hasattr(idea, 'idea') else str(idea)
                cleaned_text = strip_context_tags(idea_text)  # Clean tags
                clusters[cluster_id].append(cleaned_text)

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
    n_samples: Optional[int] = None,
    ideas_per_cluster: int = 5
):
    """
    Display sample cluster analysis results

    Args:
        clusters: Dict mapping cluster_id to idea texts
        cluster_keywords: Dict mapping cluster_id to keywords
        cluster_descriptions: Dict mapping cluster_id to LLM descriptions
        original_codes: Optional dict with original Step 6 codes
        n_samples: Number of clusters to display (None = all clusters)
        ideas_per_cluster: Number of sample ideas to show per cluster
    """
    # Get cluster IDs
    cluster_ids = sorted(clusters.keys())  # Sort for consistent order

    if n_samples is not None and len(cluster_ids) > n_samples:
        sampled_ids = random.sample(cluster_ids, n_samples)
        sampled_ids = sorted(sampled_ids)
        print(f"\nDisplaying {n_samples} randomly selected clusters (out of {len(cluster_ids)} total)\n")
    else:
        sampled_ids = cluster_ids
        print(f"\nDisplaying all {len(cluster_ids)} clusters\n")

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

    # Step 3: Run keyword extraction (TF-IDF or c-TF-IDF)
    if config.verbose:
        print(f"{'═' * 80}")
        if config.keyword_method == "ctfidf":
            print("RUNNING c-TF-IDF KEYWORD EXTRACTION (BERTopic)")
        else:
            print("RUNNING TF-IDF KEYWORD EXTRACTION")
        print(f"{'═' * 80}\n")

    if config.keyword_method == "ctfidf":
        # Use c-TF-IDF (BERTopic-inspired)
        from representation.ctfidf_representation import CTfidfRepresentation

        ctfidf_analyzer = CTfidfRepresentation(
            top_k=config.ctfidf_top_k,
            bm25_weighting=config.ctfidf_bm25_weighting,
            reduce_frequent_words=config.ctfidf_reduce_frequent_words,
            ngram_range=config.ctfidf_ngram_range,
            min_df=config.ctfidf_min_df,
            max_df=config.ctfidf_max_df,
            language="nl"
        )
        cluster_keywords = ctfidf_analyzer.extract_keywords(clusters, verbose=config.verbose)
    else:
        # Use standard TF-IDF
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
            max_ideas=config.max_ideas_per_cluster,
            verbose=config.verbose
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
        n_samples=config.n_sample_clusters,  # None = all clusters
        ideas_per_cluster=5
    )

    # Return data for optional comparison analysis
    return {
        "clusters": clusters,
        "cluster_keywords": cluster_keywords,
        "cluster_descriptions": cluster_descriptions,
        "cluster_results": cluster_results  # Original Step 5 data
    }


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

        # NEW: Choose keyword extraction method
        keyword_method="ctfidf",  # Options: "tfidf" or "ctfidf"

        # Standard TF-IDF settings (used when keyword_method="tfidf")
        tfidf_config=EXPERIMENTS["bigrams"],

        # c-TF-IDF settings (used when keyword_method="ctfidf")
        ctfidf_top_k=15,
        ctfidf_bm25_weighting=True,
        ctfidf_reduce_frequent_words=True,
        ctfidf_ngram_range=(1, 2),  # Unigrams + bigrams
        ctfidf_min_df=1,
        ctfidf_max_df=0.95,

        # LLM settings
        description_model="gpt-4.1",
        use_keywords_in_prompt=True,
        max_ideas_per_cluster=10,

        # Display settings
        n_sample_clusters=None,  # None = display all clusters
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

# ============================================================================
# OPTIONAL: RUN REPRESENTATION MODEL COMPARISON
# ============================================================================
# Uncomment to compare all 5 representation models side-by-side:
#
# # First run the main experiment
# experiment_data = run_experiment(config)
#
# # Then run the comparison
# from representation_comparison import compare_all_models
#
# comparison_results = compare_all_models(
#     cluster_results=experiment_data["cluster_results"],
#     n_sample_clusters=10,
#     export_excel=True
# )
#
# Output: exports/representation_comparison.xlsx

# %%
