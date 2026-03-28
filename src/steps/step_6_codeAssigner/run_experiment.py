#%%

"""
Step 6: Code Assigner runner (P10)

Pipeline: load codebook from step 5 cache + ideas from step 4 (or step 3 fallback) →
assign codes to ideas (P10).
No RUN_MODE — always runs P10.
"""

import sys
import io
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "development"))

from development.step_3_ideaExtractor import models_exp as models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker

# Import step_6_codeAssigner components
from config_steps.config_codeAssigner import AssignmentConfig
from development.step_6_codeAssigner.code_assignment import CodeAssigner
from development.step_6_codeAssigner.models_codeAssigner import CodeAssignedModel

# Import step_5_codeGenerator (upstream output types)
from development.step_5_codeGenerator.models_codeGenerator import CodingResultsCache

# Import step_4_classifier (upstream output types)
from development.step_4_classifier.models_classifier import (
    DomainSet, DomainResultModel, TaxonomyClassifiedModel,
)


# =============================================================================
# DATASET CONFIGURATION (centralized in development/test_data.py)
# =============================================================================
try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time


# =============================================================================
# CONFIGURATION
# =============================================================================
ASSIGNMENT_CONFIG = AssignmentConfig(
    assignment_model="gpt-4.1-nano",
    assignment_temperature=0.1,
    verbose=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step3_ideas(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> List[models.IdeasExtractedModel]:
    """Load Step 3 extracted ideas from cache."""
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
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[List[TaxonomyClassifiedModel]]:
    """Load Step 4 enriched ideas (with facet/attribute/partition) from cache.

    Returns None if not cached (caller should fall back to step 3).
    """
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
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata from cache (if available)."""
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
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[CodingResultsCache]:
    """Load cached MECE results (codebook) from step 5."""
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
# OUTPUT CAPTURE
# =============================================================================

class TeeOutput:
    """Capture stdout while also printing to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self) -> str:
        return self.buffer.getvalue()


def save_results_to_file(
    output: str,
    filename: str,
    variable: str,
    sample_size: Optional[int],
) -> Path:
    """Save results to a text file."""
    output_dir = project_root / "exports" / "verbose_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem.replace(" ", "_")
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_filename = (
        f"{base_name}_{variable}_{sample_str}"
        f"_step6_assignment_{date_str}.txt"
    )
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


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

    base = f"step6_codeAssigner_{variable_key}"
    assignment_prompts = [
        p for p in prompt_printer.prompts
        if p.get("prompt_type") in {"taxonomy_codes", "dual_assignment"}
    ]
    if assignment_prompts:
        pp_assign = PromptPrinter(enabled=True)
        pp_assign.prompts = assignment_prompts
        pp_assign.save_prompts(str(prompts_dir / f"{base}_assignment.json"))


# =============================================================================
# MAIN
# =============================================================================

def run_assignment():
    """Run code assignment (P10) from cached codebook + step 3 ideas."""
    print("=" * 70)
    print("CODE ASSIGNER (P10, loading from cache)")
    print("=" * 70)

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
    from development.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
    codes = [ConsolidatedCode(**d) for d in mece_cache.raw_codes] if mece_cache.raw_codes else None

    n_themes = mece_cache.total_categories
    n_partitions = len(partition_set.partitions)
    print(f"  Loaded codebook: {n_themes} themes, {n_partitions} partitions"
          f", {len(codes) if codes else 0} raw codes")

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
    )

    # Print assignment summary
    print_assignment_results(assigned_results)

    # Save prompts
    save_prompts_to_json(prompt_printer)

    return assigned_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    token_tracker.reset()

    try:
        result = run_assignment()

        # Print token usage
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        sys.stdout = tee.original_stdout

    # Save full verbose report
    output_path = save_results_to_file(
        output=tee.get_output(),
        filename=FILENAME,
        variable=VARIABLE,
        sample_size=SAMPLE_SIZE
    )
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")

# %%
