#%%

"""
Step 5: Categories runnner
"""

import sys
import io
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments.step_3_ideaExtractor import models_exp as models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.promptPrinter import PromptPrinter

# Import step_5_categories_v2 components
from experiments.step_5_categories.config_categories_exp import (
    CategoriesConfig, AssignmentConfig,
)
from experiments.step_5_categories.partition_discoverer import PartitionDiscoverer, PartitionLabelMapping
from experiments.step_5_categories.qualitative_researcher import QualitativeResearcher, PipelineResult, PartitionResult
from experiments.step_5_categories.models_exp import (
    PartitionSet, PartitionMECEResultModel, MECEResultsCache,
    CategoryAssignedModel,
)
from experiments.step_5_categories.category_assignment import CategoryAssigner


# =============================================================================
# DATASET CONFIGURATION (centralized in experiments/test_data.py)
# =============================================================================
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time
RUN_ASSIGNMENT = False  # Set True to run category assignment after MECE discovery
RUN_ASSIGNMENT_ONLY = False  # Set True to skip pipeline, run assignment from cache only
EXPERIMENT_N = 100  # Limit number of responses for a test run (None = use all)


# =============================================================================
# CONFIGURATION
# =============================================================================
# All defaults defined in config_categories_exp.py.
# Override individual params here only for one-off experiments.
CONFIG = CategoriesConfig(
    label_source="ladder",              # see config_categories_exp.py for all valid values
    label_prefix="",                   # "" or any static prefix string
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


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def _print_categories(categories, indent=4):
    """Recursively print MECE categories with subcategory support."""
    prefix = " " * indent
    for j, cat in enumerate(categories, 1):
        print(f"\n{prefix}[{j}] {cat.category_label}")
        if cat.interpretive_claim:
            print(f"{prefix}    Claim: {cat.interpretive_claim}")
        print(f"{prefix}    Inclusion: {cat.inclusion_definition}")
        print(f"{prefix}    Boundary test: {cat.boundary_test}")
        signals = (", ".join(cat.diagnostic_signals[:5])
                   if cat.diagnostic_signals else "(none)")
        print(f"{prefix}    Diagnostic signals: {signals}")
        if cat.tiebreaker_rules:
            print(f"{prefix}    Tiebreaker rules:")
            for rule in cat.tiebreaker_rules:
                print(f"{prefix}      - {rule}")
        if cat.key_expressions:
            print(f"{prefix}    Key expressions:")
            for expr in cat.key_expressions[:3]:
                truncated = (expr[:80] + "..."
                             if len(expr) > 80 else expr)
                print(f"{prefix}      - {truncated}")
        if cat.subcategories:
            print(f"{prefix}    Subcategories ({len(cat.subcategories)}):")
            _print_categories(cat.subcategories, indent=indent + 6)


def print_results(
    partition_set: PartitionSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    pipeline_result: PipelineResult,
):
    """Print the complete v2 pipeline results."""
    results = pipeline_result.partition_results

    # --- Per-partition themes ---
    print(f"\n{'='*80}")
    print(f"PER-PARTITION THEMES "
          f"({len(partition_set.partitions)} partitions)")
    print(f"{'='*80}")

    for i, part in enumerate(partition_set.partitions, 1):
        name = part.partition_name
        mapping = label_mappings.get(name)
        result = results.get(name)
        themes = pipeline_result.partition_themes.get(name, [])

        print(f"\n{'─'*80}")
        n_labels = result.n_labels if result else (mapping.label_count if mapping else 0)
        n_chunks = result.n_batches if result else 0
        print(f"PARTITION {i}: {name} "
              f"(n={n_labels}, {n_chunks} chunk(s), "
              f"{len(themes)} themes)")
        print(f"{'─'*80}")

        print(f"  Inclusion: {part.inclusion_definition}")

        if mapping:
            print(f"  Labels: {mapping.label_count} unique")

        if themes:
            print(f"\n  Themes ({len(themes)}):")
            for j, theme in enumerate(themes, 1):
                print(f"    {j}. {theme}")
        else:
            print(f"\n  (no themes — processing may have failed)")

    # --- Per-partition organizing concepts (if available) ---
    if pipeline_result.partition_concepts:
        print(f"\n{'='*80}")
        total_concepts = sum(
            len(pcr.concept_discovery.compressed_concepts)
            for pcr in pipeline_result.partition_concepts.values()
        )
        print(f"PER-PARTITION ORGANIZING CONCEPTS ({total_concepts} COCs "
              f"across {len(pipeline_result.partition_concepts)} partitions)")
        print(f"{'='*80}")

        for name, pcr in sorted(pipeline_result.partition_concepts.items()):
            cd = pcr.concept_discovery
            print(f"\n  [{name}] ({len(cd.compressed_concepts)} concepts)")
            for j, concept in enumerate(cd.compressed_concepts, 1):
                print(f"    {j}. {concept}")
            if cd.meaning_dimensions:
                print(f"    Meaning dimensions:")
                for dim in cd.meaning_dimensions:
                    print(f"      - {dim.dimension_name}: "
                          f"{', '.join(dim.concepts_associated)}")

    # --- Consolidated COCs (if available) ---
    if pipeline_result.coc_consolidation:
        cons = pipeline_result.coc_consolidation
        n_input = sum(
            len(pcr.concept_discovery.compressed_concepts)
            for pcr in (pipeline_result.partition_concepts or {}).values()
        )
        print(f"\n{'='*80}")
        print(f"CONSOLIDATED ORGANIZING CONCEPTS "
              f"({n_input} per-partition → {len(cons.consolidated_concepts)} consolidated)")
        print(f"{'='*80}")

        print(f"\n  Rationale: {cons.consolidation_rationale}")

        for j, c in enumerate(cons.consolidated_concepts, 1):
            sources = ", ".join(c.source_partitions)
            print(f"\n  [{j}] {c.concept_name}")
            print(f"      {c.explanation}")
            print(f"      Sources: {sources}")
            print(f"      Merged from: {', '.join(c.source_concepts)}")

    # --- Hierarchical codebook ---
    codebook = pipeline_result.codebook
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL CODEBOOK")
    print(f"{'='*80}")

    _print_categories(codebook, indent=4)

    # --- MECE validation ---
    print(f"\n{'='*80}")
    print(f"MECE VALIDATION")
    print(f"{'='*80}")
    print(f"  {pipeline_result.codebook_narrative}")

    # --- Grand summary ---
    total_labels = sum(m.label_count for m in label_mappings.values())
    total_themes = sum(len(t) for t in pipeline_result.partition_themes.values())
    n_per_partition_concepts = sum(
        len(pcr.concept_discovery.compressed_concepts)
        for pcr in (pipeline_result.partition_concepts or {}).values()
    )
    n_consolidated = (
        len(pipeline_result.coc_consolidation.consolidated_concepts)
        if pipeline_result.coc_consolidation else 0
    )
    # Count hierarchy levels
    n_l1 = len(codebook)
    n_l2 = sum(len(c.subcategories) for c in codebook)
    n_l3 = sum(
        len(sc.subcategories) for c in codebook for sc in c.subcategories
    )
    print(f"\n{'='*80}")
    print(f"GRAND SUMMARY")
    print(f"{'='*80}")
    print(f"  Partitions:              {len(partition_set.partitions)}")
    print(f"  Total Labels:            {total_labels}")
    print(f"  Descriptive Themes:      {total_themes}")
    print(f"  Per-Partition COCs:      {n_per_partition_concepts}")
    print(f"  Consolidated COCs:       {n_consolidated}")
    print(f"  Codebook L1 (themes):    {n_l1}")
    print(f"  Codebook L2 (subthemes): {n_l2}")
    if n_l3:
        print(f"  Codebook L3 (valence):   {n_l3}")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the Category Discovery pipeline."""
    print("=" * 70)
    print("Category Discovery (Qualitative Researcher Pipeline)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Label source: {CONFIG.label_source}")
    if CONFIG.label_prefix:
        print(f"Label prefix: {CONFIG.label_prefix!r}")
    print(f"Batch sizing: {CONFIG.batch_size_min}-{CONFIG.batch_size_max} "
          f"(target {CONFIG.target_batches} chunks)")
    print()

    # Load data
    ideas_models = load_step3_ideas()
    if EXPERIMENT_N is not None and EXPERIMENT_N < len(ideas_models):
        total = len(ideas_models)
        ideas_models = ideas_models[:EXPERIMENT_N]
        print(f"Experiment subset: {EXPERIMENT_N} responses (of {total} total)")
    extraction_metadata = load_extraction_metadata()

    # =========================================================================
    # Stage 1: Partition Discovery
    # =========================================================================
    discoverer = PartitionDiscoverer(CONFIG, extraction_metadata)
    partition_set, label_mappings = discoverer.discover(
        ideas_models
    )

    # =========================================================================
    # Stage 2: Qualitative Researcher v2 pipeline
    # =========================================================================
    # Build context from extraction metadata
    survey_question = ""
    language = "Dutch"
    dataset_context = None
    dimension_name = ""
    dimension_description = ""

    if extraction_metadata:
        meta = extraction_metadata
        survey_question = getattr(meta, 'var_lab', '') or ''
        language = getattr(meta, 'lang', 'Dutch') or 'Dutch'
        dataset_context = {}
        for f in ('sector', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(meta, f, None)
            if val:
                dataset_context[f] = val
        dimension_name = getattr(meta, 'primary_dimension', '') or ''
        dimension_description = getattr(meta, 'primary_dimension_description', '') or ''

    prompt_printer = PromptPrinter(
        enabled=True,                    # Always capture prompts for debugging
        print_realtime=PRINT_PROMPTS,    # Only print to console if requested
    )
    processor = QualitativeResearcher(CONFIG, prompt_printer=prompt_printer)
    pipeline_result = processor.process_all_partitions(
        label_mappings=label_mappings,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=CONFIG.verbose,
    )

    # =========================================================================
    # Print results
    # =========================================================================
    print_results(partition_set, label_mappings, pipeline_result)

    return partition_set, label_mappings, pipeline_result, ideas_models, prompt_printer


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
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")

    # Determine run mode suffix from flags
    if RUN_ASSIGNMENT_ONLY:
        mode_suffix = "_assignment"
    elif RUN_ASSIGNMENT:
        mode_suffix = "_generation+assignment"
    else:
        mode_suffix = "_generation"

    output_filename = (
        f"category_results_{base_name}_{variable}"
        f"_{sample_str}_{date_str}{mode_suffix}.txt"
    )
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


# =============================================================================
# ASSIGNMENT CONFIGURATION
# =============================================================================
ASSIGNMENT_CONFIG = AssignmentConfig(
    assignment_model="gpt-4.1-nano",
    assignment_temperature=0.1,
    assignment_batch_size=10,
    verbose=True,
)


# =============================================================================
# MECE CACHING
# =============================================================================

def cache_mece_results(
    partition_set: PartitionSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    pipeline_result: PipelineResult,
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Dict[str, PartitionMECEResultModel]:
    """Cache codebook results for later use by category assignment.

    The codebook (hierarchical MECECategory tree) is stored as a single
    partition result keyed by "__global__", since v2 produces a single
    cross-partition codebook rather than per-partition categories.
    """
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    # Store the global codebook as a single "partition" for cache compat
    codebook = pipeline_result.codebook
    total_labels = sum(m.label_count for m in label_mappings.values())
    pydantic_results = {
        "__global__": PartitionMECEResultModel(
            partition_name="__global__",
            n_labels=total_labels,
            n_batches=0,
            categories=codebook,
        )
    }

    mece_cache = MECEResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={
            name: m.label_count for name, m in label_mappings.items()
        },
        label_source=CONFIG.label_source,
        total_categories=len(codebook),
    )

    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=mece_cache,
        filename=filename,
        step="mece_categories",
        variable_key=variable_key,
    )
    n_subthemes = sum(len(c.subcategories) for c in codebook)
    print(f"Thematic analysis cached "
          f"({len(codebook)} themes, {n_subthemes} subthemes)")

    return pydantic_results


def load_mece_cache(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[MECEResultsCache]:
    """Load cached MECE results if available."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    cache_manager = CacheManager()
    return cache_manager.load_metadata_from_cache(
        filename=filename,
        step="mece_categories",
        variable_key=variable_key,
        model_cls=MECEResultsCache,
    )


# =============================================================================
# CATEGORY ASSIGNMENT
# =============================================================================

def run_category_assignment(
    ideas_models: List[models.IdeasExtractedModel],
    mece_results: Dict[str, PartitionMECEResultModel],
    partition_set: PartitionSet,
    extraction_metadata: Optional[models.ExtractionMetadata] = None,
    config: AssignmentConfig = ASSIGNMENT_CONFIG,
    prompt_printer=None,
) -> List[CategoryAssignedModel]:
    """Run category assignment and cache results."""
    assigner = CategoryAssigner(
        config=config,
        ideas_models=ideas_models,
        mece_results=mece_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
        prompt_printer=prompt_printer,
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
        "category_assignment",
        variable_key,
    )
    print(f"Category assignment results cached "
          f"({len(assigned_results)} response models)")

    return assigned_results


def run_assignment_only():
    """Run category assignment from cached data (skip pipeline).

    Loads step 3 ideas, MECE codebook, and extraction metadata from cache,
    then runs assignment. Useful for iterating on assignment without
    re-running theme discovery.
    """
    print("=" * 70)
    print("ASSIGNMENT ONLY MODE (loading from cache)")
    print("=" * 70)

    # Load dependencies from cache
    ideas_models = load_step3_ideas()
    extraction_metadata = load_extraction_metadata()

    mece_cache = load_mece_cache()
    if mece_cache is None:
        print("\nERROR: No cached MECE results found.")
        print("Run the full pipeline first (RUN_ASSIGNMENT_ONLY = False).")
        return

    partition_set = mece_cache.partition_set
    pydantic_results = mece_cache.partition_results

    n_themes = mece_cache.total_categories
    n_partitions = len(partition_set.partitions)
    print(f"  Loaded codebook: {n_themes} themes, {n_partitions} partitions")

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
    )
    assigned_results = run_category_assignment(
        ideas_models=ideas_models,
        mece_results=pydantic_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
        prompt_printer=prompt_printer,
    )

    return assigned_results, prompt_printer


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        if RUN_ASSIGNMENT_ONLY:
            # =====================================================================
            # Assignment-only mode: load from cache, run assignment
            # =====================================================================
            result = run_assignment_only()
            if result:
                assigned_results, prompt_printer = result
            else:
                prompt_printer = PromptPrinter()
        else:
            # =====================================================================
            # Full pipeline mode
            # =====================================================================
            partition_set, label_mappings, pipeline_result, ideas_models, prompt_printer = main()

            # Cache MECE results
            pydantic_results = cache_mece_results(
                partition_set, label_mappings, pipeline_result,
            )

            # Run category assignment (optional)
            if RUN_ASSIGNMENT:
                extraction_metadata = load_extraction_metadata()
                assigned_results = run_category_assignment(
                    ideas_models=ideas_models,
                    mece_results=pydantic_results,
                    partition_set=partition_set,
                    extraction_metadata=extraction_metadata,
                    prompt_printer=prompt_printer,
                )
            else:
                print("\n  Category assignment skipped (RUN_ASSIGNMENT = False)")

        # =====================================================================
        # Save captured prompts to JSON (split by phase)
        # =====================================================================
        if prompt_printer.prompts:
            variable_key = generate_enhanced_variable_key(
                selected_variables=[VARIABLE],
                is_merged=False,
                sample_size=SAMPLE_SIZE,
            )
            prompts_dir = project_root / "exports" / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # Split prompts into pipeline (generation) vs assignment
            ASSIGNMENT_TYPES = {"category_assignment"}
            pipeline_prompts = [
                p for p in prompt_printer.prompts
                if p.get("prompt_type") not in ASSIGNMENT_TYPES
            ]
            assignment_prompts = [
                p for p in prompt_printer.prompts
                if p.get("prompt_type") in ASSIGNMENT_TYPES
            ]

            base = f"step5_categories_{VARIABLE}_{variable_key}"
            if pipeline_prompts:
                pp_pipeline = PromptPrinter(enabled=True)
                pp_pipeline.prompts = pipeline_prompts
                pp_pipeline.save_prompts(str(prompts_dir / f"{base}_pipeline.json"))
            if assignment_prompts:
                pp_assign = PromptPrinter(enabled=True)
                pp_assign.prompts = assignment_prompts
                pp_assign.save_prompts(str(prompts_dir / f"{base}_assignment.json"))
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
