#%%
#
"""
Debug script for Single-Idea Category Assignment prompts.

Loads cached step 3 ideas and MECE codebook, randomly picks a few ideas,
builds the single-idea assignment prompt for each, and displays it along
with the Pydantic response model schema.

Usage:
    cd src && python -m experiments.step_5_categories_v2.debug_assignment_prompt
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key

# Import centralized test data config
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

from experiments.step_3_ideaExtractor import models_exp as models
from experiments.step_5_categories.models_exp import (
    PartitionSet, PartitionMECEResultModel, MECEResultsCache,
)
from experiments.step_5_categories.prompts_exp import (
    build_single_idea_assignment_prompt,
    SingleCategoryAssignment,
    MECECategory,
)
from experiments.step_5_categories.category_assignment import CategoryAssigner
from experiments.step_5_categories.config_categories_exp import (
    AssignmentConfig, get_other_category_label,
)

# Configuration
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

N_IDEAS_PER_PARTITION = 3   # How many random ideas to pick per partition (1 prompt each)
RANDOM_SEED = 42            # Set to None for truly random


# =============================================================================
# DATA LOADING (mirrors run_experiment.py)
# =============================================================================

def load_step3_ideas() -> List[models.IdeasExtractedModel]:
    """Load Step 3 extracted ideas from cache."""
    from utils.cacheManager import CacheManager
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cache_manager = CacheManager()
    data = cache_manager.load_from_cache(
        FILENAME, "extracted_ideas", variable_key, models.IdeasExtractedModel,
    )
    if not data:
        raise FileNotFoundError(
            f"Cache not found for step 'extracted_ideas'. Run step 3 first."
        )
    total_ideas = sum(item.idea_count for item in data)
    print(f"Loaded {len(data)} responses with {total_ideas} ideas from step 3 cache")
    return data


def load_mece_cache() -> Optional[MECEResultsCache]:
    """Load cached MECE results."""
    from utils.cacheManager import CacheManager
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cache_manager = CacheManager()
    return cache_manager.load_metadata_from_cache(
        filename=FILENAME, step="mece_categories",
        variable_key=variable_key, model_cls=MECEResultsCache,
    )


def load_extraction_metadata() -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata from cache."""
    from utils.cacheManager import CacheManager
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cache_manager = CacheManager()
    return cache_manager.load_metadata_from_cache(
        filename=FILENAME, step="extracted_ideas",
        variable_key=variable_key, model_cls=models.ExtractionMetadata,
    )


# =============================================================================
# GROUPING & SAMPLING
# =============================================================================

def group_ideas_by_partition(
    ideas_models: List[models.IdeasExtractedModel],
) -> Dict[str, List[models.IdeasExtractedSubmodel]]:
    """Group all ideas by domain (partition)."""
    partitions: Dict[str, List] = {}
    for resp in ideas_models:
        if not resp.response_ideas:
            continue
        for idea in resp.response_ideas:
            ct = (idea.domain or '').strip().lower()
            if not ct:
                continue
            if ct not in partitions:
                partitions[ct] = []
            partitions[ct].append(idea)
    return partitions


def sample_ideas(
    partition_ideas: Dict[str, List],
    n_per_partition: int,
    seed: Optional[int] = None,
) -> Dict[str, List]:
    """Randomly sample n ideas per partition."""
    rng = random.Random(seed)
    sampled = {}
    for pname, ideas in sorted(partition_ideas.items()):
        k = min(n_per_partition, len(ideas))
        sampled[pname] = rng.sample(ideas, k)
    return sampled


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_prompt_for_idea(
    partition_name: str,
    idea,
    mece_results: Dict[str, PartitionMECEResultModel],
    partition_set: PartitionSet,
    extraction_metadata: Optional[models.ExtractionMetadata],
    config: AssignmentConfig,
) -> str:
    """Build a single-idea assignment prompt (flat codebook, one idea)."""
    # Survey context
    survey_question = ""
    language = "Dutch"
    dataset_context_section = ""
    if extraction_metadata:
        survey_question = extraction_metadata.var_lab or ""
        language = extraction_metadata.lang or "Dutch"
        parts = []
        for f in ('domain', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(extraction_metadata, f, None)
            if val:
                parts.append(f"{f.capitalize()}: {val}")
        if parts:
            dataset_context_section = "\n".join(parts)

    # Partition inclusion
    partition_inclusion = ""
    for p in partition_set.partitions:
        if p.partition_name == partition_name:
            partition_inclusion = p.inclusion_definition
            break

    # Resolve codebook
    if "__global__" in mece_results:
        mece_res = mece_results["__global__"]
    else:
        mece_res = mece_results.get(partition_name)

    if not mece_res or not mece_res.categories:
        return f"(No categories found for partition '{partition_name}')"

    leaf_categories = CategoryAssigner._flatten_categories(mece_res.categories)

    other_label = get_other_category_label(language)

    return build_single_idea_assignment_prompt(
        survey_question=survey_question,
        language=language,
        dataset_context_section=dataset_context_section,
        partition_name=partition_name,
        partition_inclusion=partition_inclusion,
        categories=leaf_categories,
        other_label=other_label if config.include_other_category else None,
        idea=idea,
    )


# =============================================================================
# DISPLAY
# =============================================================================

def print_prompt(
    partition_name: str,
    idea,
    prompt: str,
    index: int,
    total: int,
):
    """Display one single-idea assignment prompt."""
    print(f"\n{'='*100}")
    print(f"ASSIGNMENT PROMPT {index}/{total}: partition='{partition_name}'")
    print(f"{'='*100}")

    # Show the idea being assigned
    valence = getattr(idea, 'valence', '') or '0'
    print(f"\n[Idea]")
    print(f"  {idea.idea_id}: {idea.idea}  [valence={valence}]")
    if hasattr(idea, 'instance') and idea.instance:
        print(f"  instance: \"{idea.instance}\"")
    if hasattr(idea, 'rung_1') and idea.rung_1:
        print(f"  rung_1:   {idea.rung_1}")
    if hasattr(idea, 'rung_2') and idea.rung_2:
        print(f"  rung_2:   {idea.rung_2}")

    # Full prompt
    print(f"\n[Full Prompt]")
    print("-" * 100)
    print(prompt)
    print("-" * 100)

    # Stats
    print(f"\n[Stats]")
    print(f"  Prompt: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens)")


def print_response_schema():
    """Display the Pydantic response model schema."""
    print(f"\n{'='*100}")
    print(f"RESPONSE MODEL: SingleCategoryAssignment")
    print(f"{'='*100}")
    schema = SingleCategoryAssignment.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    print(schema_str)
    print(f"\n  Schema: {len(schema_str):,} chars (~{len(schema_str) // 4:,} tokens)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("DEBUG: Single-Idea Category Assignment Prompt Inspector")
    print("Randomly picks ideas per partition and builds one prompt per idea")
    print("=" * 100)
    print(f"Variable:              {VAR_NAME}")
    print(f"Sample size:           {SAMPLE_SIZE}")
    print(f"Ideas per partition:   {N_IDEAS_PER_PARTITION}")
    print(f"Random seed:           {RANDOM_SEED}")
    print("=" * 100)

    # Load data
    ideas_models = load_step3_ideas()
    extraction_metadata = load_extraction_metadata()

    mece_cache = load_mece_cache()
    if mece_cache is None:
        print("\nERROR: No cached MECE results found.")
        print("Run the full pipeline first:")
        print("  cd src && python -m experiments.step_5_categories_v2.run_experiment")
        return

    partition_set = mece_cache.partition_set
    mece_results = mece_cache.partition_results

    n_themes = mece_cache.total_categories
    n_partitions = len(partition_set.partitions)
    print(f"\nCodebook: {n_themes} themes, {n_partitions} partitions")

    # Group and sample ideas
    partition_ideas = group_ideas_by_partition(ideas_models)
    sampled = sample_ideas(partition_ideas, N_IDEAS_PER_PARTITION, RANDOM_SEED)

    total_sampled = sum(len(v) for v in sampled.values())
    print(f"Sampled {total_sampled} ideas across {len(sampled)} partitions")
    print(f"Will show {total_sampled} individual assignment prompts")

    # Build config
    config = AssignmentConfig()

    # Build and display one prompt per idea
    prompt_idx = 0
    partition_names = sorted(sampled.keys())
    for pname in partition_names:
        ideas = sampled[pname]
        for idea in ideas:
            prompt_idx += 1
            prompt = build_prompt_for_idea(
                partition_name=pname,
                idea=idea,
                mece_results=mece_results,
                partition_set=partition_set,
                extraction_metadata=extraction_metadata,
                config=config,
            )
            print_prompt(pname, idea, prompt, prompt_idx, total_sampled)

    # Response model schema (once)
    print_response_schema()

    print(f"\n{'='*100}")
    print("Done.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

# %%
