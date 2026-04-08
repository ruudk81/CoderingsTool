#%%

"""
Step 5: Code Generator runner (P8-P9)

Pipeline: load taxonomy from step 4 cache → generate codebook (P8-P9).
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
from utils.costTracker import CostTracker

# Import step_5_codeGenerator components
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.codebook_generator import CodebookGenerator, CodebookResult
from pipeline.step_5_codeGenerator.models_codeGenerator import CodingResultsCache

# Import step_4_classifier (upstream output types)
from pipeline.step_4_classifier.models_classifier import (
    DomainSet, DomainResultModel, TaxonomyResultsCache,
    TaxonomyClassifiedModel,
)


# =============================================================================
from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time

# Valence prevalence thresholds (for suppressing low-prevalence valence groups)
MIN_VALENCE_SHARE = 0.10    # Min share of attribute total (e.g., 0.10 = 10%)
# MIN_VALENCE_IDEAS: computed as floor(log(sample_size)) at runtime


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = CodebookConfig(
    min_valence_share=MIN_VALENCE_SHARE,
)


# =============================================================================
# DATA LOADING
# =============================================================================

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


def load_taxonomy_cache(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[TaxonomyResultsCache]:
    """Load cached taxonomy results from step 4."""
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


def load_classified_ideas(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> Optional[List[TaxonomyClassifiedModel]]:
    """Load step 4's taxonomy-classified growing model (ideas with attribute/valence)."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    cache_manager = CacheManager()
    data = cache_manager.load_from_cache(
        filename=filename,
        step="taxonomy_classified",
        variable_key=variable_key,
        model_cls=TaxonomyClassifiedModel,
    )

    if data:
        n_ideas = sum(
            len(r.response_ideas) for r in data if r.response_ideas
        )
        print(f"Loaded classified ideas: {len(data)} responses, {n_ideas} ideas")
    else:
        print("WARNING: taxonomy_classified growing model not found in cache")

    return data


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_codebook_results(codebook_result: CodebookResult):
    """Print codebook results (P8-P9): codes with definitions and source attributes."""
    codes = codebook_result.codes
    n_pos = sum(1 for c in codes if getattr(c, 'valence', '') == 'positive')
    n_neg = sum(1 for c in codes if getattr(c, 'valence', '') == 'negative')
    n_neu = len(codes) - n_pos - n_neg

    print(f"\n{'='*80}")
    print(f"CODEBOOK ({len(codes)} codes: {n_pos} positive, {n_neg} negative, {n_neu} neutral)")
    print(f"{'='*80}")

    for j, code in enumerate(codes, 1):
        indicators = ", ".join(code.typical_indicators[:5]) if code.typical_indicators else "(none)"
        sources = ", ".join(code.source_attributes[:5]) if code.source_attributes else "(none)"
        valence = getattr(code, 'valence', '') or ''
        diagnostic = getattr(code, 'diagnostic_test', '') or ''
        valence_tag = f" ({valence})" if valence else ""
        print(f"\n    [{j}] {code.code_name}{valence_tag}")
        print(f"        Definition: {code.definition}")
        if diagnostic:
            print(f"        Diagnostic: {diagnostic}")
        print(f"        Indicators: {indicators}")
        print(f"        Source attributes: {sources}")

    print(f"\n{'='*80}")
    print(f"Total codes: {len(codes)}")
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
        f"_step5_{date_str}.txt"
    )
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


# =============================================================================
# MECE CACHING
# =============================================================================

def cache_mece_results(
    partition_set: DomainSet,
    pydantic_results: Dict[str, DomainResultModel],
    codebook_result: CodebookResult,
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
    idea_embeddings: Optional[Dict] = None,
    embedding_code_source: str = "",
    embedding_model: str = "",
) -> None:
    """Cache codebook results for later use by code assignment (step 6)."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    # Serialize numpy arrays to lists for Pydantic compatibility
    serialized_embeddings = None
    if idea_embeddings:
        serialized_embeddings = {
            idea_id: emb.tolist() if hasattr(emb, 'tolist') else emb
            for idea_id, emb in idea_embeddings.items()
        }

    n_codes = len(codebook_result.codes)
    mece_cache = CodingResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={
            name: r.n_labels for name, r in pydantic_results.items()
        },
        total_categories=n_codes,
        raw_codes=[c.model_dump() for c in codebook_result.codes],
        idea_embeddings=serialized_embeddings,
        embedding_code_source=embedding_code_source,
        embedding_model=embedding_model,
    )

    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=mece_cache,
        filename=filename,
        step="mece_codes",
        variable_key=variable_key,
    )
    total_facets = sum(
        len(r.facets) for r in pydantic_results.values()
    )
    print(f"Codebook cached "
          f"({n_codes} codes, {total_facets} facets across "
          f"{len(pydantic_results)} domains)")


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

    base = f"step5_codeGenerator_{variable_key}"
    codebook_prompts = [
        p for p in prompt_printer.prompts
        if p.get("prompt_type") in {
            "code_generation_from_attributes", "codebook_consolidation",
        }
    ]
    if codebook_prompts:
        pp_code = PromptPrinter(enabled=True)
        pp_code.prompts = codebook_prompts
        pp_code.save_prompts(str(prompts_dir / f"{base}_codebook.json"))


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


# =============================================================================
# MAIN
# =============================================================================

def run_codebook():
    """Run codebook generation (P8-P9) from cached taxonomy results."""
    print("=" * 70)
    print("CODE GENERATOR (P8-P9, loading taxonomy from cache)")
    print("=" * 70)

    extraction_metadata = load_extraction_metadata()
    classified_ideas = load_classified_ideas()
    taxonomy_cache = load_taxonomy_cache()
    if taxonomy_cache is None:
        print("\nERROR: No cached taxonomy results found.")
        print("Run step 4 taxonomy first (step_4_classifier).")
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

    # Reconstruct taxonomy data for codebook generator
    from pipeline.step_4_classifier.prompts_classifier import DiscoveredFacet, DiscoveredAttribute

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

    from pipeline.step_5_codeGenerator.codebook_generator import TaxonomyResult
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

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )
    cost_tracker = CostTracker(filename=FILENAME, variable_key=variable_key)

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
    )
    template_prefix = ""
    if extraction_metadata and getattr(extraction_metadata, "template_prefix", None):
        template_prefix = extraction_metadata.template_prefix

    dataset_key = f"{FILENAME}:{variable_key}"
    generator = CodebookGenerator(CONFIG, prompt_printer=prompt_printer, cost_tracker=cost_tracker, dataset_key=dataset_key)
    codebook_result = generator.generate(
        taxonomy_result=taxonomy_result,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=CONFIG.verbose if hasattr(CONFIG, 'verbose') else True,
        classified_ideas=classified_ideas,
        template_prefix=template_prefix,
    )

    cost_tracker.finalize_step("step_5_code_generator")

    # Print codebook results
    print_codebook_results(codebook_result)

    # Cache for downstream use by step 6 (code assigner)
    cache_mece_results(
        partition_set, pydantic_results, codebook_result,
        idea_embeddings=getattr(generator, '_idea_embeddings', None),
        embedding_code_source=CONFIG.code_source,
        embedding_model=CONFIG.embedding_model,
    )

    # Save prompts
    save_prompts_to_json(prompt_printer)

    return codebook_result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    token_tracker.reset()

    try:
        result = run_codebook()

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
