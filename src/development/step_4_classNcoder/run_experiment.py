#%%

"""
Step 5: Categories runner

Pipeline: facet discovery → facet assignment → attribute discovery →
code generation from attributes → code assignment.
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

# Import step_4_classNcoder components
from development.step_4_classNcoder.config_classNcoder_exp import (
    CategoriesConfig, AssignmentConfig,
)
from development.step_4_classNcoder.domain_discoverer import DomainDiscoverer, PartitionLabelMapping
from development.step_4_classNcoder.qualitative_researcher import (
    QualitativeResearcher, PipelineResult, DomainResult, TaxonomyResult,
)
from development.step_4_classNcoder.models_exp import (
    DomainSet, DomainResultModel, CodingResultsCache, TaxonomyResultsCache,
    CodeAssignedModel,
)
from development.step_4_classNcoder.code_assignment import CodeAssigner


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
RUN_MODE = "taxonomy"  # "taxonomy" | "codebook" | "assignment" | "all"
EXPERIMENT_N = None  # Limit number of responses for a test run (None = use all)


# =============================================================================
# CONFIGURATION
# =============================================================================
# All defaults defined in config_classNcoder_exp.py.
# Override individual params here only for one-off experiments.
CONFIG = CategoriesConfig(
    label_source="idea",                # show idea text only (includes template_prefix), not full ladder
    label_prefix="",                   # "" or any static prefix string
    include_valence=True,              # prepend [+]/[-]/[0] valence tag to labels
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

def print_taxonomy_results(
    partition_set: DomainSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    taxonomy_result: TaxonomyResult,
):
    """Print taxonomy results (P1-P3.5): domains, facets, attributes."""
    print(f"\n{'='*80}")
    print(f"TAXONOMY RESULTS "
          f"({len(partition_set.partitions)} domains)")
    print(f"{'='*80}")

    for i, part in enumerate(partition_set.partitions, 1):
        name = part.partition_name
        mapping = label_mappings.get(name)
        facets = taxonomy_result.partition_facets.get(name, [])
        assignments = taxonomy_result.partition_assignments.get(name, {})
        attributes = taxonomy_result.partition_attributes.get(name, {})

        # Collect attribute assignment counts
        domain_facet_ids = set(assignments.keys())
        domain_attr_assigns = {
            iid: aname for iid, aname in taxonomy_result.attribute_assignments.items()
            if iid in domain_facet_ids
        }
        attr_counts = {}
        for attr_name in domain_attr_assigns.values():
            attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

        n_labels = taxonomy_result.partition_n_labels.get(name, 0)
        n_chunks = taxonomy_result.partition_n_batches.get(name, 0)

        print(f"\n{'─'*80}")
        print(f"DOMAIN {i}: {name} "
              f"(n={n_labels}, {n_chunks} chunk(s), "
              f"{len(facets)} facets, {len(assignments)} assigned, "
              f"{sum(len(a) for a in attributes.values())} attributes)")
        print(f"{'─'*80}")

        print(f"  Inclusion: {part.inclusion_definition}")

        if mapping:
            print(f"  Observations: {mapping.label_count} unique")

        if facets:
            print(f"\n  Facets ({len(facets)}):")
            for j, facet in enumerate(facets, 1):
                print(f"    {j}. {facet.facet_name}: {facet.facet_description}")

        if attributes:
            print(f"\n  Attributes per facet:")
            for facet_name, attrs in sorted(attributes.items()):
                print(f"    {facet_name} ({len(attrs)}):")
                for attr in attrs:
                    count = attr_counts.get(attr.attribute_name, 0)
                    print(f"      - {attr.attribute_name} [{count} ideas]: "
                          f"{attr.attribute_description}")

    # Summary
    total_labels = sum(m.label_count for m in label_mappings.values())
    total_facets = sum(
        len(taxonomy_result.partition_facets.get(name, []))
        for name in taxonomy_result.partition_facets
    )
    total_assignments = sum(
        len(a) for a in taxonomy_result.partition_assignments.values()
    )
    total_attributes = sum(
        len(attrs)
        for facet_attrs in taxonomy_result.partition_attributes.values()
        for attrs in facet_attrs.values()
    )
    total_attr_assigned = len(taxonomy_result.attribute_assignments)

    print(f"\n{'='*80}")
    print(f"TAXONOMY SUMMARY")
    print(f"{'='*80}")
    print(f"  Domains:                 {len(partition_set.partitions)}")
    print(f"  Total Observations:      {total_labels}")
    print(f"  Facets (P1):             {total_facets}")
    print(f"  Ideas assigned (P2):     {total_assignments}")
    print(f"  Attributes (P3):         {total_attributes}")
    print(f"  Ideas with attrs (P4a):  {total_attr_assigned}")
    print(f"{'='*80}\n")


def print_results(
    partition_set: DomainSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    pipeline_result: PipelineResult,
):
    """Print the complete v3 pipeline results."""
    results = pipeline_result.partition_results

    # --- Per-domain facets, assignments, and attributes ---
    print(f"\n{'='*80}")
    print(f"PER-DOMAIN RESULTS "
          f"({len(partition_set.partitions)} domains)")
    print(f"{'='*80}")

    for i, part in enumerate(partition_set.partitions, 1):
        name = part.partition_name
        mapping = label_mappings.get(name)
        result = results.get(name)

        print(f"\n{'─'*80}")
        n_labels = result.n_labels if result else (mapping.label_count if mapping else 0)
        n_chunks = result.n_batches if result else 0
        n_facets = len(result.facets) if result else 0
        n_assigned = len(result.facet_assignments) if result else 0
        n_attrs = sum(
            len(attrs) for attrs in result.attributes.values()
        ) if result else 0
        print(f"DOMAIN {i}: {name} "
              f"(n={n_labels}, {n_chunks} chunk(s), "
              f"{n_facets} facets, {n_assigned} assigned, "
              f"{n_attrs} attributes)")
        print(f"{'─'*80}")

        print(f"  Inclusion: {part.inclusion_definition}")

        if mapping:
            print(f"  Observations: {mapping.label_count} unique")

        if result and result.facets:
            print(f"\n  Facets ({len(result.facets)}):")
            for j, facet in enumerate(result.facets, 1):
                print(f"    {j}. {facet.facet_name}: {facet.facet_description}")

        if result and result.attributes:
            print(f"\n  Attributes per facet:")
            for facet_name, attrs in sorted(result.attributes.items()):
                print(f"    {facet_name} ({len(attrs)}):")
                for attr in attrs:
                    print(f"      - {attr.attribute_name}: {attr.attribute_description}")

    # --- Codebook ---
    print(f"\n{'='*80}")
    print(f"CODEBOOK "
          f"({len(pipeline_result.codes)} codes)")
    print(f"{'='*80}")

    for j, code in enumerate(pipeline_result.codes, 1):
        indicators = ", ".join(code.typical_indicators[:5])
        sources = ", ".join(code.source_attributes[:5]) if code.source_attributes else "(none)"
        print(f"\n    [{j}] {code.code_name}")
        print(f"        Definition: {code.definition}")
        print(f"        Indicators: {indicators}")
        print(f"        Source attributes: {sources}")

    # --- Evaluation ---
    print(f"\n{'='*80}")
    print(f"CODE GENERATION EVALUATION")
    print(f"{'='*80}")
    print(f"  {pipeline_result.codebook_narrative}")

    # --- Grand summary ---
    total_labels = sum(m.label_count for m in label_mappings.values())
    total_facets = sum(
        len(r.facets) for r in results.values()
    )
    total_assignments = sum(
        len(r.facet_assignments) for r in results.values()
    )
    total_attributes = sum(
        len(attrs)
        for r in results.values()
        for attrs in r.attributes.values()
    )
    n_codes = len(pipeline_result.codes)

    print(f"\n{'='*80}")
    print(f"GRAND SUMMARY")
    print(f"{'='*80}")
    print(f"  Domains:                 {len(partition_set.partitions)}")
    print(f"  Total Observations:      {total_labels}")
    print(f"  Facets (P1):             {total_facets}")
    print(f"  Ideas assigned (P2):     {total_assignments}")
    print(f"  Attributes (P3):         {total_attributes}")
    print(f"  Codes (P4):              {n_codes}")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the full Category Discovery v3 pipeline (taxonomy + codebook)."""
    print("=" * 70)
    print("Category Discovery v3 (Full Pipeline)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Observation source: {CONFIG.label_source}")
    if CONFIG.label_prefix:
        print(f"Label prefix: {CONFIG.label_prefix!r}")
    print(f"Batch sizing: {CONFIG.batch_size_min}-{CONFIG.batch_size_max} "
          f"(target {CONFIG.target_batches} chunks)")
    print()

    ideas_models, extraction_metadata, partition_set, label_mappings = _load_and_discover()
    survey_question, language, dataset_context, dimension_name, dimension_description = \
        _extract_metadata_context(extraction_metadata)

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
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

    # Print results
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
    output_dir = project_root / "exports" / "verbose_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem.replace(" ", "_")
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine run mode suffix
    mode_suffix = f"_{RUN_MODE}"

    output_filename = (
        f"{base_name}_{variable}_{sample_str}"
        f"_step4_{date_str}{mode_suffix}.txt"
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
    verbose=True,
)


# =============================================================================
# MECE CACHING
# =============================================================================

def cache_mece_results(
    partition_set: DomainSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    pipeline_result: PipelineResult,
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Dict[str, DomainResultModel]:
    """Cache codebook results for later use by category assignment.

    The codebook is stored as a single partition result keyed by "__global__",
    since v3 produces a single cross-partition codebook.

    Per-domain results (facets, assignments, attributes) are also stored
    for debugging and analysis.
    """
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    # Store per-domain results (facets, assignments, attributes)
    pydantic_results = {}
    for name, result in pipeline_result.partition_results.items():
        pydantic_results[name] = DomainResultModel(
            partition_name=name,
            n_labels=result.n_labels,
            n_batches=result.n_batches,
            facets=[f.model_dump() for f in result.facets],
            facet_assignments=result.facet_assignments,
            attributes={
                facet_name: [a.model_dump() for a in attrs]
                for facet_name, attrs in result.attributes.items()
            },
            attribute_assignments=result.attribute_assignments,
        )

    n_codes = len(pipeline_result.codes)
    mece_cache = CodingResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={
            name: m.label_count for name, m in label_mappings.items()
        },
        label_source=CONFIG.label_source,
        total_categories=n_codes,
        raw_codes=[c.model_dump() for c in pipeline_result.codes],
    )

    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=mece_cache,
        filename=filename,
        step="mece_categories",
        variable_key=variable_key,
    )
    total_facets = sum(
        len(r.facets) for r in pipeline_result.partition_results.values()
    )
    print(f"v3 results cached "
          f"({n_codes} codes, {total_facets} facets across "
          f"{len(pipeline_result.partition_results)} domains)")

    return pydantic_results


def load_mece_cache(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[CodingResultsCache]:
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
        model_cls=CodingResultsCache,
    )


# =============================================================================
# TAXONOMY CACHING
# =============================================================================

def cache_taxonomy_results(
    partition_set: DomainSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    taxonomy_result: TaxonomyResult,
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Dict[str, DomainResultModel]:
    """Cache taxonomy results (P1-P3.5) for later use by codebook generation."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    # Build per-domain pydantic results
    pydantic_results = {}
    for name in taxonomy_result.partition_facets:
        domain_facet_ids = set(taxonomy_result.partition_assignments.get(name, {}).keys())
        domain_attr_assigns = {
            iid: aname for iid, aname in taxonomy_result.attribute_assignments.items()
            if iid in domain_facet_ids
        }
        pydantic_results[name] = DomainResultModel(
            partition_name=name,
            n_labels=taxonomy_result.partition_n_labels.get(name, 0),
            n_batches=taxonomy_result.partition_n_batches.get(name, 0),
            facets=[f.model_dump() for f in taxonomy_result.partition_facets.get(name, [])],
            facet_assignments=taxonomy_result.partition_assignments.get(name, {}),
            attributes={
                facet_name: [a.model_dump() for a in attrs]
                for facet_name, attrs in taxonomy_result.partition_attributes.get(name, {}).items()
            },
            attribute_assignments=domain_attr_assigns,
        )

    taxonomy_cache = TaxonomyResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={
            name: m.label_count for name, m in label_mappings.items()
        },
        label_source=CONFIG.label_source,
    )

    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=taxonomy_cache,
        filename=filename,
        step="taxonomy",
        variable_key=variable_key,
    )

    total_facets = sum(
        len(taxonomy_result.partition_facets.get(name, []))
        for name in taxonomy_result.partition_facets
    )
    total_attrs = sum(
        len(attrs)
        for facet_attrs in taxonomy_result.partition_attributes.values()
        for attrs in facet_attrs.values()
    )
    print(f"Taxonomy results cached "
          f"({total_facets} facets, {total_attrs} attributes across "
          f"{len(pydantic_results)} domains)")

    return pydantic_results


def load_taxonomy_cache(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[TaxonomyResultsCache]:
    """Load cached taxonomy results if available."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    cache_manager = CacheManager()
    return cache_manager.load_metadata_from_cache(
        filename=filename,
        step="taxonomy",
        variable_key=variable_key,
        model_cls=TaxonomyResultsCache,
    )


# =============================================================================
# CATEGORY ASSIGNMENT
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
        "code_assignment",
        variable_key,
    )
    print(f"Category assignment results cached "
          f"({len(assigned_results)} response models)")

    return assigned_results


def _extract_metadata_context(extraction_metadata):
    """Extract survey context from extraction metadata."""
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

    return survey_question, language, dataset_context, dimension_name, dimension_description


def _load_and_discover(extraction_metadata=None):
    """Shared data loading: step 3 ideas + partition discovery."""
    ideas_models = load_step3_ideas()
    if EXPERIMENT_N is not None and EXPERIMENT_N < len(ideas_models):
        total = len(ideas_models)
        ideas_models = ideas_models[:EXPERIMENT_N]
        print(f"Experiment subset: {EXPERIMENT_N} responses (of {total} total)")
    if extraction_metadata is None:
        extraction_metadata = load_extraction_metadata()

    discoverer = DomainDiscoverer(CONFIG, extraction_metadata)
    partition_set, label_mappings = discoverer.discover(ideas_models)

    return ideas_models, extraction_metadata, partition_set, label_mappings


def run_taxonomy():
    """Run taxonomy stages only (P1-P3.5): facets, attributes, assignments."""
    print("=" * 70)
    print("TAXONOMY ONLY MODE (P1-P3.5)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print()

    ideas_models, extraction_metadata, partition_set, label_mappings = _load_and_discover()
    survey_question, language, dataset_context, dimension_name, dimension_description = \
        _extract_metadata_context(extraction_metadata)

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
    )
    processor = QualitativeResearcher(CONFIG, prompt_printer=prompt_printer)
    taxonomy_result = processor.process_taxonomy_only(
        label_mappings=label_mappings,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=CONFIG.verbose,
    )

    # Print taxonomy results
    print_taxonomy_results(partition_set, label_mappings, taxonomy_result)

    return partition_set, label_mappings, taxonomy_result, ideas_models, prompt_printer


def run_codebook_from_cache():
    """Run codebook generation (P4-P4.5) from cached taxonomy results."""
    print("=" * 70)
    print("CODEBOOK ONLY MODE (P4-P4.5, loading taxonomy from cache)")
    print("=" * 70)

    extraction_metadata = load_extraction_metadata()
    taxonomy_cache = load_taxonomy_cache()
    if taxonomy_cache is None:
        print("\nERROR: No cached taxonomy results found.")
        print("Run taxonomy first (RUN_MODE = 'taxonomy' or 'all').")
        return None

    partition_set = taxonomy_cache.partition_set
    pydantic_results = taxonomy_cache.partition_results

    n_facets = sum(len(r.facets) for r in pydantic_results.values())
    n_attrs = sum(
        len(attrs) for r in pydantic_results.values()
        for attrs in r.attributes.values()
    )
    print(f"  Loaded taxonomy: {n_facets} facets, {n_attrs} attributes "
          f"across {len(pydantic_results)} domains")

    # Reconstruct TaxonomyResult from cached data
    from development.step_4_classNcoder.prompts_exp import DiscoveredFacet, DiscoveredAttribute

    partition_facets = {}
    partition_assignments = {}
    partition_attributes = {}
    partition_n_labels = {}
    partition_n_batches = {}
    all_attr_assignments = {}

    for name, result in pydantic_results.items():
        partition_facets[name] = [DiscoveredFacet(**f) for f in result.facets]
        partition_assignments[name] = result.facet_assignments
        partition_attributes[name] = {
            facet_name: [DiscoveredAttribute(**a) for a in attrs]
            for facet_name, attrs in result.attributes.items()
        }
        partition_n_labels[name] = result.n_labels
        partition_n_batches[name] = result.n_batches
        all_attr_assignments.update(result.attribute_assignments)

    taxonomy_result = TaxonomyResult(
        partition_n_labels=partition_n_labels,
        partition_n_batches=partition_n_batches,
        partition_facets=partition_facets,
        partition_assignments=partition_assignments,
        partition_attributes=partition_attributes,
        attribute_assignments=all_attr_assignments,
    )

    survey_question, language, dataset_context, dimension_name, dimension_description = \
        _extract_metadata_context(extraction_metadata)

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
    )
    processor = QualitativeResearcher(CONFIG, prompt_printer=prompt_printer)
    pipeline_result = processor.process_codebook_only(
        taxonomy=taxonomy_result,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=CONFIG.verbose,
    )

    return partition_set, pydantic_results, pipeline_result, prompt_printer


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
        print("Run codebook generation first (RUN_MODE = 'codebook' or 'all').")
        return

    partition_set = mece_cache.partition_set
    pydantic_results = mece_cache.partition_results

    # Reconstruct ConsolidatedCode from cached dicts
    from development.step_4_classNcoder.prompts_exp import ConsolidatedCode
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

    return assigned_results, prompt_printer


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        if RUN_MODE == "taxonomy":
            # =================================================================
            # Taxonomy only: P1-P3.5
            # =================================================================
            partition_set, label_mappings, taxonomy_result, ideas_models, prompt_printer = run_taxonomy()
            cache_taxonomy_results(partition_set, label_mappings, taxonomy_result)

        elif RUN_MODE == "codebook":
            # =================================================================
            # Codebook only: P4-P4.5 from cached taxonomy
            # =================================================================
            result = run_codebook_from_cache()
            if result:
                partition_set, pydantic_results, pipeline_result, prompt_printer = result
                # Also cache as MECE (taxonomy + codes) for assignment
                # Need label_mappings for cache — reconstruct from taxonomy cache
                taxonomy_cache = load_taxonomy_cache()
                label_counts = taxonomy_cache.label_counts if taxonomy_cache else {}
                cache_mece_results(
                    partition_set, {},  # no label_mappings needed, use pipeline_result
                    pipeline_result,
                )
            else:
                prompt_printer = PromptPrinter()

        elif RUN_MODE == "assignment":
            # =================================================================
            # Assignment only: P5 from cached codebook
            # =================================================================
            result = run_assignment_only()
            if result:
                assigned_results, prompt_printer = result
            else:
                prompt_printer = PromptPrinter()

        elif RUN_MODE == "all":
            # =================================================================
            # Full pipeline: taxonomy + codebook + assignment
            # =================================================================
            partition_set, label_mappings, pipeline_result, ideas_models, prompt_printer = main()

            # Cache taxonomy results (for future codebook-only runs)
            # Reconstruct TaxonomyResult from pipeline_result for caching
            tax_pydantic = {}
            for name, result in pipeline_result.partition_results.items():
                tax_pydantic[name] = DomainResultModel(
                    partition_name=name,
                    n_labels=result.n_labels,
                    n_batches=result.n_batches,
                    facets=[f.model_dump() for f in result.facets],
                    facet_assignments=result.facet_assignments,
                    attributes={
                        facet_name: [a.model_dump() for a in attrs]
                        for facet_name, attrs in result.attributes.items()
                    },
                    attribute_assignments=result.attribute_assignments,
                )
            taxonomy_cache_obj = TaxonomyResultsCache(
                partition_set=partition_set,
                partition_results=tax_pydantic,
                label_counts={name: m.label_count for name, m in label_mappings.items()},
                label_source=CONFIG.label_source,
            )
            cache_manager = CacheManager()
            variable_key = generate_enhanced_variable_key(
                selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE,
            )
            cache_manager.save_metadata_to_cache(
                metadata=taxonomy_cache_obj,
                filename=FILENAME, step="taxonomy", variable_key=variable_key,
            )
            print("Taxonomy results cached")

            # Cache MECE results (taxonomy + codes)
            pydantic_results = cache_mece_results(
                partition_set, label_mappings, pipeline_result,
            )

            # Run code assignment
            extraction_metadata = load_extraction_metadata()
            all_attr_assignments = {}
            for domain_result in pipeline_result.partition_results.values():
                all_attr_assignments.update(domain_result.attribute_assignments)
            assigned_results = run_code_assignment(
                ideas_models=ideas_models,
                mece_results=pydantic_results,
                partition_set=partition_set,
                extraction_metadata=extraction_metadata,
                prompt_printer=prompt_printer,
                codes=pipeline_result.codes,
                attribute_assignments=all_attr_assignments,
            )

        else:
            print(f"ERROR: Unknown RUN_MODE '{RUN_MODE}'")
            print("Valid options: 'taxonomy', 'codebook', 'assignment', 'all'")
            prompt_printer = PromptPrinter()

        # =====================================================================
        # Save captured prompts to JSON (3-way split)
        # =====================================================================
        if prompt_printer.prompts:
            variable_key = generate_enhanced_variable_key(
                selected_variables=[VARIABLE],
                is_merged=False,
                sample_size=SAMPLE_SIZE,
            )
            prompts_dir = project_root / "exports" / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # 3-way prompt split by stage
            TAXONOMY_TYPES = {
                "facet_discovery", "facet_consolidation", "facet_assignment",
                "attribute_discovery", "attribute_chunk_consolidation",
                "attribute_consolidation", "attribute_assignment",
            }
            CODEBOOK_TYPES = {
                "code_generation_from_attributes", "codebook_consolidation",
            }
            ASSIGNMENT_TYPES = {"code_assignment", "dual_assignment"}

            taxonomy_prompts = [
                p for p in prompt_printer.prompts
                if p.get("prompt_type") in TAXONOMY_TYPES
            ]
            codebook_prompts = [
                p for p in prompt_printer.prompts
                if p.get("prompt_type") in CODEBOOK_TYPES
            ]
            assignment_prompts = [
                p for p in prompt_printer.prompts
                if p.get("prompt_type") in ASSIGNMENT_TYPES
            ]

            base = f"step4_classNcoder_{variable_key}"
            if taxonomy_prompts:
                pp_tax = PromptPrinter(enabled=True)
                pp_tax.prompts = taxonomy_prompts
                pp_tax.save_prompts(str(prompts_dir / f"{base}_taxonomy.json"))
            if codebook_prompts:
                pp_code = PromptPrinter(enabled=True)
                pp_code.prompts = codebook_prompts
                pp_code.save_prompts(str(prompts_dir / f"{base}_codebook.json"))
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
