#%%

"""
Step 4: Taxonomy Classifier runner (P1-P7)

Pipeline: domain discovery → facet discovery → facet assignment →
attribute discovery → attribute assignment.

Always runs the full taxonomy pipeline (P1-P7).
"""

import sys
import io
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pipeline.step_3_ideaExtractor import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker

# Import step_4_classifier components
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.domain_discoverer import DomainDiscoverer, PartitionLabelMapping
from pipeline.step_4_classifier.classifier import TaxonomyClassifier, TaxonomyResult
from pipeline.step_4_classifier.models_classifier import (
    DomainSet, DomainResultModel, TaxonomyResultsCache,
    TaxonomyClassifiedModel, TaxonomyClassifiedSubmodel,
)


# =============================================================================
# DATASET CONFIGURATION (centralized in steps/test_data.py)
# =============================================================================
try:
    from pipeline.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time
EXPERIMENT_N = None  # Limit number of responses for a test run (None = use all)


# =============================================================================
# CONFIGURATION
# =============================================================================
# All defaults defined in config_classifier.py.
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
    """Print taxonomy results (P1-P7): domains, facets, attributes."""
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
    print(f"  Ideas assigned (P3):     {total_assignments}")
    print(f"  Attributes (P4):         {total_attributes}")
    print(f"  Ideas with attrs (P6):  {total_attr_assigned}")
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
        f"_step4_taxonomy_{date_str}.txt"
    )
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


# =============================================================================
# GROWING MODEL BUILDER
# =============================================================================

def _build_taxonomy_enriched_models(encoded_text, taxonomy_cache):
    """Build TaxonomyClassifiedModel list from step 3 ideas + taxonomy results.

    Creates new model instances (does not mutate encoded_text) with facet (L3),
    attribute (L4), and partition_name populated from TaxonomyResultsCache.
    """
    # Build global lookups: idea_id -> facet/attribute/partition name
    facet_lookup = {}
    attr_lookup = {}
    partition_lookup = {}  # idea_id -> partition_name
    for domain_result in taxonomy_cache.partition_results.values():
        facet_lookup.update(domain_result.facet_assignments)
        attr_lookup.update(domain_result.attribute_assignments)
        for idea_id in domain_result.facet_assignments:
            partition_lookup[idea_id] = domain_result.partition_name

    output = []
    for resp in encoded_text:
        new_ideas = []
        if resp.response_ideas:
            for idea in resp.response_ideas:
                idea_data = idea.model_dump()
                idea_data["facet"] = facet_lookup.get(idea.idea_id, idea.facet or "")
                idea_data["attribute"] = attr_lookup.get(idea.idea_id, idea.attribute or "")
                idea_data["partition_name"] = partition_lookup.get(idea.idea_id, idea.domain or "")
                new_ideas.append(TaxonomyClassifiedSubmodel(**idea_data))

        resp_data = resp.model_dump(exclude={"response_ideas"})
        output.append(TaxonomyClassifiedModel(**resp_data, response_ideas=new_ideas))

    return output


# =============================================================================
# TAXONOMY CACHING
# =============================================================================

def cache_taxonomy_results(
    partition_set: DomainSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    taxonomy_result: TaxonomyResult,
    ideas_models: Optional[List[models.IdeasExtractedModel]] = None,
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Dict[str, DomainResultModel]:
    """Cache taxonomy results (P1-P7) for later use by codebook generation."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    # Build per-domain pydantic results
    pydantic_results = {}
    for name in taxonomy_result.partition_facets:
        # Filter out None values from assignments (ideas that weren't assigned)
        facet_assigns = {
            k: v for k, v in taxonomy_result.partition_assignments.get(name, {}).items()
            if v is not None
        }
        domain_facet_ids = set(facet_assigns.keys())
        domain_attr_assigns = {
            iid: aname for iid, aname in taxonomy_result.attribute_assignments.items()
            if iid in domain_facet_ids and aname is not None
        }
        pydantic_results[name] = DomainResultModel(
            partition_name=name,
            n_labels=taxonomy_result.partition_n_labels.get(name, 0),
            n_batches=taxonomy_result.partition_n_batches.get(name, 0),
            facets=[f.model_dump() for f in taxonomy_result.partition_facets.get(name, [])],
            facet_assignments=facet_assigns,
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

    # Build and cache growing model (enriched facet/attribute per idea)
    if ideas_models is not None:
        enriched = _build_taxonomy_enriched_models(ideas_models, taxonomy_cache)
        cache_manager.save_to_cache(enriched, filename, "taxonomy_classified", variable_key)
        print(f"Growing model cached: {len(enriched)} enriched responses")

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


# =============================================================================
# PROMPT SAVING
# =============================================================================

def save_prompts_to_json(prompt_printer):
    """Save captured prompts to JSON file."""
    if not prompt_printer or not prompt_printer.prompts:
        return

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )
    prompts_dir = project_root / "exports" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    TAXONOMY_TYPES = {
        "facet_discovery", "facet_consolidation", "facet_assignment",
        "attribute_discovery", "attribute_chunk_consolidation",
        "attribute_consolidation", "attribute_assignment",
    }

    taxonomy_prompts = [
        p for p in prompt_printer.prompts
        if p.get("prompt_type") in TAXONOMY_TYPES
    ]

    base = f"step4_classifier_{variable_key}"
    if taxonomy_prompts:
        pp_tax = PromptPrinter(enabled=True)
        pp_tax.prompts = taxonomy_prompts
        pp_tax.save_prompts(str(prompts_dir / f"{base}_taxonomy.json"))


# =============================================================================
# HELPERS
# =============================================================================

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
        print(f"Subset: {EXPERIMENT_N} responses (of {total} total)")

    if extraction_metadata is None:
        extraction_metadata = load_extraction_metadata()

    discoverer = DomainDiscoverer(CONFIG, extraction_metadata)
    partition_set, label_mappings = discoverer.discover(ideas_models)

    return ideas_models, extraction_metadata, partition_set, label_mappings


# =============================================================================
# MAIN
# =============================================================================

def run_taxonomy():
    """Run taxonomy stages (P1-P7): facets, attributes, assignments."""
    print("=" * 70)
    print("TAXONOMY PIPELINE (P1-P7)")
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
    processor = TaxonomyClassifier(CONFIG, prompt_printer=prompt_printer)
    taxonomy_result = processor.process(
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

    # Cache taxonomy results (metadata + growing model)
    cache_taxonomy_results(partition_set, label_mappings, taxonomy_result, ideas_models=ideas_models)

    # Save prompts
    save_prompts_to_json(prompt_printer)

    return partition_set, label_mappings, taxonomy_result, ideas_models, prompt_printer


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    token_tracker.reset()

    try:
        partition_set, label_mappings, taxonomy_result, ideas_models, prompt_printer = run_taxonomy()

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
