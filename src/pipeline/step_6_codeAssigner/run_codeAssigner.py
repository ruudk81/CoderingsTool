#%%

"""
Step 6: Code Assigner runner (P10)

Pipeline: load codebook from step 5 cache + ideas from step 4 (or step 3 fallback) →
assign codes to ideas (P10).
No RUN_MODE — always runs P10.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils.costTracker import CostTracker
from utils.saveVerbose import VerboseCapture

# Import step_6_codeAssigner components
from pipeline.step_6_codeAssigner.config_codeAssigner import AssignmentConfig
from pipeline.step_6_codeAssigner.code_assignment import CodeAssigner
from models import CodeAssignedModel

# Import step_5_codeGenerator (upstream output types)
from models import CodingResultsCache

# Import step_4_classifier (upstream output types)
from models import (
    DomainSet, DomainResultModel, TaxonomyClassifiedModel,
)


# =============================================================================
from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time


# =============================================================================
# CONFIGURATION
# =============================================================================
ASSIGNMENT_CONFIG = AssignmentConfig(
    assignment_temperature=0.1,
    verbose=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step3_ideas(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> List[models.IdeasExtractedModel]:
    """Load Step 3 extracted ideas from cache."""
    filename = FILENAME if filename is None else filename
    variable = VARIABLE if variable is None else variable
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_manager = CacheManager()
    data = cache_manager.load_from_cache(
        filename, "extracted_ideas", variable_key, models.IdeasExtractedModel
    )

    if not data:
        raise FileNotFoundError(
            f"Cache not found for step 'extracted_ideas' / variable_key '{variable_key}'.\n"
            f"Run step 3 (ideaExtractor) first."
        )

    total_ideas = sum(item.idea_count for item in data)
    print(f"Loaded {len(data)} responses with {total_ideas} ideas from step 3 cache")

    return data


def load_step4_enriched(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> Optional[List[TaxonomyClassifiedModel]]:
    """Load Step 4 enriched ideas (with facet/attribute/partition) from cache.

    Returns None if not cached (caller should fall back to step 3).
    """
    filename = FILENAME if filename is None else filename
    variable = VARIABLE if variable is None else variable
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_manager = CacheManager()
    data = cache_manager.load_from_cache(
        filename, "taxonomy_classified", variable_key, TaxonomyClassifiedModel
    )

    if data:
        total_ideas = sum(item.idea_count for item in data)
        print(f"Loaded {len(data)} responses with {total_ideas} ideas from step 4 cache (enriched)")

    return data


def load_extraction_metadata(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata from cache (if available)."""
    filename = FILENAME if filename is None else filename
    variable = VARIABLE if variable is None else variable
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_manager = CacheManager()
    metadata = cache_manager.load_metadata_from_cache(
        filename=filename,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )

    if metadata:
        print(f"Loaded ExtractionMetadata: primary_dimension={metadata.primary_dimension}")
        if metadata.var_lab:
            print(f"  Survey question (var_lab): {metadata.var_lab}")
    else:
        print("ExtractionMetadata not found in cache (optional)")

    return metadata


def load_mece_cache(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> Optional[CodingResultsCache]:
    """Load cached MECE results (codebook) from step 5."""
    filename = FILENAME if filename is None else filename
    variable = VARIABLE if variable is None else variable
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    cache_manager = CacheManager()
    return cache_manager.load_metadata_from_cache(
        filename=filename,
        step="mece_codes",
        variable_key=variable_key,
        model_cls=CodingResultsCache,
    )


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_assignment_results(assigned_results):
    """Print assignment summary."""
    total_ideas = sum(
        len(r.response_ideas or []) for r in assigned_results
    )
    assigned_count = sum(
        1 for r in assigned_results
        for idea in (r.response_ideas or [])
        if idea.assigned_code
    )
    print(f"\n{'='*80}")
    print(f"ASSIGNMENT SUMMARY")
    print(f"{'='*80}")
    print(f"  Responses:       {len(assigned_results)}")
    print(f"  Total ideas:     {total_ideas}")
    print(f"  Ideas assigned:  {assigned_count}")
    print(f"{'='*80}\n")


# =============================================================================
# CODE ASSIGNMENT
# =============================================================================

def run_code_assignment(
    ideas_models: List[models.IdeasExtractedModel],
    mece_results: Dict[str, DomainResultModel],
    partition_set: DomainSet,
    extraction_metadata: Optional[models.ExtractionMetadata] = None,
    config: AssignmentConfig = ASSIGNMENT_CONFIG,
    prompt_printer=None,
    codes=None,
    attribute_assignments: Optional[Dict[str, str]] = None,
    cost_tracker=None,
) -> List[CodeAssignedModel]:
    """Run code assignment and cache results."""
    assigner = CodeAssigner(
        config=config,
        ideas_models=ideas_models,
        mece_results=mece_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
        prompt_printer=prompt_printer,
        codes=codes,
        attribute_assignments=attribute_assignments,
        cost_tracker=cost_tracker,
    )

    assigned_results = assigner.assign_all()

    # Cache assignment results
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )
    cache_manager = CacheManager()
    cache_manager.save_to_cache(
        assigned_results,
        FILENAME,
        "taxonomy_codes",
        variable_key,
    )
    print(f"Category assignment results cached "
          f"({len(assigned_results)} response models)")

    return assigned_results


# =============================================================================
# PROMPT SAVING
# =============================================================================

def save_prompts_to_json(prompt_printer):
    """Save captured assignment prompts to JSON file."""
    if not prompt_printer or not prompt_printer.prompts:
        return

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )
    prompts_dir = project_root / "exports" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_printer.save_prompts(
        str(prompts_dir / f"step6_codeAssigner_{variable_key}_assignment.json"))


# =============================================================================
# MAIN
# =============================================================================

def run_assignment(filename: str = FILENAME, var_name: str = VARIABLE,
                   sample_size: Optional[int] = SAMPLE_SIZE, force_recalc: bool = False):
    """Run code assignment (P10) from cached codebook + step 3 ideas.

    Dataset params default to the module-level TEST_DATA constants (so existing
    callers like run_pipeline.py are unchanged); the UI passes them explicitly.
    Rebinds the module globals once so downstream helpers see the right dataset.
    """
    global FILENAME, VARIABLE, SAMPLE_SIZE
    FILENAME, VARIABLE, SAMPLE_SIZE = filename, var_name, sample_size
    print("=" * 70)
    print("CODE ASSIGNER (P10, loading from cache)")
    print("=" * 70)

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )

    if not force_recalc:
        cache_manager = CacheManager()
        if cache_manager.is_cache_valid(FILENAME, "taxonomy_codes", variable_key):
            print("Code-assignment cache valid — skipping P10 (use force_recalc=True to rerun).\n")
            return None

    # Load dependencies from cache — prefer step 4 enriched, fall back to step 3
    ideas_models = load_step4_enriched()
    if ideas_models is None:
        print("Step 4 growing model not cached, falling back to step 3")
        ideas_models = load_step3_ideas()
    extraction_metadata = load_extraction_metadata()

    mece_cache = load_mece_cache()
    if mece_cache is None:
        print("\nERROR: No cached MECE results found.")
        print("Run step 5 (codeGenerator) first.")
        return None

    partition_set = mece_cache.partition_set
    pydantic_results = mece_cache.partition_results

    # Reconstruct ConsolidatedCode from cached dicts
    from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
    codes = [ConsolidatedCode(**d) for d in mece_cache.raw_codes] if mece_cache.raw_codes else None

    n_themes = mece_cache.total_categories
    n_partitions = len(partition_set.partitions)
    print(f"  Loaded codebook: {n_themes} themes, {n_partitions} partitions"
          f", {len(codes) if codes else 0} raw codes")

    cost_tracker = CostTracker(filename=FILENAME, var_name=VARIABLE,
                               sample_size=SAMPLE_SIZE)

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
    )
    # Collect attribute_assignments from all domains
    all_attr_assignments = {}
    for domain_result in pydantic_results.values():
        all_attr_assignments.update(domain_result.attribute_assignments)

    assigned_results = run_code_assignment(
        ideas_models=ideas_models,
        mece_results=pydantic_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
        prompt_printer=prompt_printer,
        codes=codes,
        attribute_assignments=all_attr_assignments,
        cost_tracker=cost_tracker,
    )

    cost_tracker.finalize_step("step_6_code_assigner")

    # Print assignment summary
    print_assignment_results(assigned_results)

    # Save prompts
    save_prompts_to_json(prompt_printer)

    return assigned_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    with VerboseCapture(
        filename=FILENAME,
        var_name=VARIABLE,
        sample_size=SAMPLE_SIZE,
        step=6,
    ):
        token_tracker.reset()

        result = run_assignment()

        # Print token usage
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

# %%
