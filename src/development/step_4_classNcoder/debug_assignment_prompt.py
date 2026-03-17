#%%
#
"""
Debug script for Single-Idea Dual Assignment prompts.

Loads cached step 3 ideas and codebook, randomly picks a few ideas,
builds the single-idea dual assignment prompt for each, and displays it along
with the Pydantic response model schema.

Usage:
    cd src && python -m development.step_4_classNcoder.debug_assignment_prompt
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
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

from development.step_3_ideaExtractor import models_exp as models
from development.step_4_classNcoder.models_exp import (
    DomainSet, DomainResultModel, CodingResultsCache,
)
from development.step_4_classNcoder.prompts_exp import (
    build_single_dual_assignment_prompt,
    CodeAttributeAssignment,
    CodeFromAttributes,
    MECECode,
)
from development.step_4_classNcoder.code_assignment import CodeAssigner
from development.step_4_classNcoder.config_classNcoder_exp import (
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


def load_mece_cache() -> Optional[CodingResultsCache]:
    """Load cached MECE results."""
    from utils.cacheManager import CacheManager
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cache_manager = CacheManager()
    return cache_manager.load_metadata_from_cache(
        filename=FILENAME, step="mece_categories",
        variable_key=variable_key, model_cls=CodingResultsCache,
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
    idea,
    codes: List[CodeFromAttributes],
    extraction_metadata: Optional[models.ExtractionMetadata],
    config: AssignmentConfig,
    facet_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build a single-idea dual assignment prompt (flat codebook, one idea)."""
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

    other_label = get_other_category_label(language)

    return build_single_dual_assignment_prompt(
        survey_question=survey_question,
        language=language,
        dataset_context_section=dataset_context_section,
        codes=codes,
        other_label=other_label if config.include_other_category else None,
        idea=idea,
        facet_lookup=facet_lookup,
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
    if hasattr(idea, 'interpretation') and idea.interpretation:
        print(f"  interpretation:   {idea.interpretation}")
    if hasattr(idea, 'abstraction') and idea.abstraction:
        print(f"  abstraction:   {idea.abstraction}")

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
    print(f"RESPONSE MODEL: CodeAttributeAssignment")
    print(f"{'='*100}")
    schema = CodeAttributeAssignment.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    print(schema_str)
    print(f"\n  Schema: {len(schema_str):,} chars (~{len(schema_str) // 4:,} tokens)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("DEBUG: Single-Idea Dual Assignment Prompt Inspector")
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
        print("  cd src && python -m development.step_4_classNcoder.run_experiment")
        return

    partition_set = mece_cache.partition_set

    # Reconstruct codes from raw_codes cache
    codes = [CodeFromAttributes(**d) for d in mece_cache.raw_codes]

    # Build facet lookup from cached P2 facet assignments
    facet_lookup: Dict[str, str] = {}
    for mece_res in mece_cache.partition_results.values():
        if mece_res.facet_assignments:
            facet_lookup.update(mece_res.facet_assignments)

    n_codes = len(codes)
    n_partitions = len(partition_set.partitions)
    print(f"\nCodebook: {n_codes} codes, {n_partitions} partitions")
    print(f"Facet lookup: {len(facet_lookup)} entries")

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
                idea=idea,
                codes=codes,
                extraction_metadata=extraction_metadata,
                config=config,
                facet_lookup=facet_lookup,
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
