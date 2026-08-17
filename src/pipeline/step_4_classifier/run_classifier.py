#%%

"""
Step 4: Taxonomy Classifier runner.

Domain discovery, then discovery → facet consolidation → facet assignment →
facet settle → attribute consolidation → assignment → refinement →
cross-domain, and the valence-neutral merge last. Nine phases; facets and
attributes are found together and settled apart, one call per level, and the
facet layer is settled on the ideas it actually drew rather than on how many
chunks proposed a name. See `classifier.TaxonomyClassifier`.
"""
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time

# How many RESPONSES to keep for a test run — a count, or None for all. This is
# not where you stop after a phase; that is STOP_AFTER_PHASE below.
LIMIT_N = None

# None = all nine phases. Otherwise one of TaxonomyClassifier.PHASES, which is
# the only place the names live:
#   discovery, facet_consolidation, facet_assignment, facet_settle,
#   attribute_consolidation, assignment, refinement, cross_domain, valence_merge
# An unknown name raises at construction rather than quietly running everything.
#
# A stop before `assignment` leaves the facets with their attribute pools still
# unconsolidated, and no catch-alls: those arrive after attribute consolidation.
#
# WARNING: a phase stop still writes its partial taxonomy to the cache, over the
# complete one that was there. Copy data/cache before an early-stop run.
STOP_AFTER_PHASE = "facet_settle"

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.exportNaming import export_filename
from utils.identity import ensure_taxonomy_ids, restamp_assignment_ids
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils.costTracker import CostTracker
from utils.saveVerbose import VerboseCapture

# Import step_4_classifier components
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.domain_discoverer import DomainDiscoverer, PartitionLabelMapping
from pipeline.step_4_classifier.classifier import TaxonomyClassifier, TaxonomyResult
from pipeline.step_4_classifier.taxonomy_health import (
    prune_empty_nodes, print_health, attr_structure_home,
)
from models import (
    DomainSet, DomainResultModel, TaxonomyResultsCache,
    TaxonomyClassifiedModel, TaxonomyClassifiedSubmodel,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# All defaults defined in config_classifier.py.
# Override individual params here only for one-off experiments.
CONFIG = CategoriesConfig(
    stop_after_phase=STOP_AFTER_PHASE,
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


# =============================================================================
# GROWING MODEL BUILDER
# =============================================================================

def _build_taxonomy_enriched_models(encoded_text, taxonomy_cache):
    """Build TaxonomyClassifiedModel list from step 3 ideas + taxonomy results.

    Creates new model instances (does not mutate encoded_text) with facet (L3),
    attribute (L4), and partition_name populated from TaxonomyResultsCache.
    """
    # Build global lookups: idea_id -> facet/attribute/partition name + confidence
    facet_lookup = {}
    attr_lookup = {}
    partition_lookup = {}  # idea_id -> partition_name
    facet_conf_lookup = {}
    attr_conf_lookup = {}
    facet_val_lookup = {}
    attr_val_lookup = {}
    for domain_result in taxonomy_cache.partition_results.values():
        facet_lookup.update(domain_result.facet_assignments)
        attr_lookup.update(domain_result.attribute_assignments)
        facet_conf_lookup.update(domain_result.facet_confidence)
        attr_conf_lookup.update(domain_result.attribute_confidence)
        facet_val_lookup.update(domain_result.facet_valence)
        attr_val_lookup.update(domain_result.attribute_valence)
        for idea_id in domain_result.facet_assignments:
            partition_lookup[idea_id] = domain_result.partition_name

    # Per-idea (domain, facet) is a DERIVED projection of the structure — the
    # single source of truth — so it can't drift from partition_results.attributes.
    struct_home = attr_structure_home(taxonomy_cache)

    output = []
    for resp in encoded_text:
        new_ideas = []
        if resp.response_ideas:
            for idea in resp.response_ideas:
                idea_data = idea.model_dump()
                attr_name = attr_lookup.get(idea.idea_id, idea.attribute or "")
                idea_data["attribute"] = attr_name
                dom_fac = struct_home.get(attr_name)
                if dom_fac:
                    idea_data["partition_name"], idea_data["facet"] = dom_fac
                else:
                    idea_data["facet"] = facet_lookup.get(idea.idea_id, idea.facet or "")
                    idea_data["partition_name"] = partition_lookup.get(idea.idea_id, idea.domain or "")
                idea_data["domain"] = idea_data["partition_name"]   # canonical: domain == partition_name (no casing drift)
                idea_data["facet_confidence"] = facet_conf_lookup.get(idea.idea_id)
                idea_data["attribute_confidence"] = attr_conf_lookup.get(idea.idea_id)
                # Valence cascade: P8 (most precise) > P4 > step 3 (inherited)
                idea_data["valence"] = (
                    attr_val_lookup.get(idea.idea_id)
                    or facet_val_lookup.get(idea.idea_id)
                    or idea.valence
                    or ""
                )
                new_ideas.append(TaxonomyClassifiedSubmodel(**idea_data))

        resp_data = resp.model_dump(exclude={"response_ideas"})
        output.append(TaxonomyClassifiedModel(**resp_data, response_ideas=new_ideas))

    return output


# =============================================================================
# TAXONOMY CACHING
# =============================================================================

def _write_action_log(
    taxonomy_result: TaxonomyResult,
    filename: str,
    variable_key: str,
) -> None:
    """Dump the step-4 action log to exports/experiment_logs/ as JSON.

    Every merge, split and move with the exact texts it touched — this is what
    makes a bad decision findable afterwards instead of invisible.

    Deliberately a file and not a cache field: TaxonomyResultsCache is shared
    broadly, and this log is a diagnostic side-artifact that should not widen it.
    """
    log = getattr(taxonomy_result, "consolidation_log", None)
    if not log:
        return

    out_dir = Path(__file__).resolve().parents[3] / "exports" / "experiment_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    path = out_dir / f"{stem}_{variable_key}_step4_log.json"
    path.write_text(
        json.dumps({"dataset": filename, "variable_key": variable_key, "actions": log},
                   indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    if CONFIG.verbose:
        totals = [e for e in log if e.get("action") in ("_facet_totals", "_totals")]
        print(f"  Step-4 action log written: {path.name} ({len(log)} actions)")
        for entry in totals:
            print(f"    {entry}")


def cache_taxonomy_results(
    partition_set: DomainSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    taxonomy_result: TaxonomyResult,
    ideas_models: Optional[List[models.IdeasExtractedModel]] = None,
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> Dict[str, DomainResultModel]:
    """Cache the taxonomy for later use by codebook generation."""
    filename = FILENAME if filename is None else filename
    variable = VARIABLE if variable is None else variable
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
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
        # Raw (pre-refinement) attribute assignments for this domain
        domain_raw_attr_assigns = {
            iid: aname for iid, aname in taxonomy_result.raw_attribute_assignments.items()
            if iid in domain_facet_ids and aname is not None
        }
        # Confidence scores scoped to this domain
        domain_facet_conf = {
            iid: c for iid, c in taxonomy_result.facet_confidence.items()
            if iid in domain_facet_ids
        }
        domain_attr_conf = {
            iid: c for iid, c in taxonomy_result.attribute_confidence.items()
            if iid in domain_facet_ids
        }
        # Valence scoped to this domain
        domain_facet_val = {
            iid: v for iid, v in taxonomy_result.facet_valence.items()
            if iid in domain_facet_ids
        }
        domain_attr_val = {
            iid: v for iid, v in taxonomy_result.attribute_valence.items()
            if iid in domain_facet_ids
        }
        pydantic_results[name] = DomainResultModel(
            partition_name=name,
            n_labels=taxonomy_result.partition_n_labels.get(name, 0),
            n_batches=taxonomy_result.partition_n_batches.get(name, 0),
            # Already plain dicts: the classifier converts each phase's
            # response at its own boundary, so the catch-alls it builds itself
            # are the same kind of thing as everything a model proposed.
            facets=list(taxonomy_result.partition_facets.get(name, [])),
            facet_assignments=facet_assigns,
            attributes={
                facet_name: list(attrs)
                for facet_name, attrs in taxonomy_result.partition_attributes.get(name, {}).items()
            },
            attribute_assignments=domain_attr_assigns,
            # Discovery snapshot: the state before consolidation settled the
            # inventory. Never drop these — they are what makes a bad merge
            # diagnosable after the fact.
            raw_facets=list(taxonomy_result.partition_raw_facets.get(name, [])),
            raw_attributes={
                facet_name: list(attrs)
                for facet_name, attrs in taxonomy_result.raw_partition_attributes.get(name, {}).items()
            },
            raw_attribute_assignments=domain_raw_attr_assigns,
            facet_confidence=domain_facet_conf,
            attribute_confidence=domain_attr_conf,
            facet_valence=domain_facet_val,
            attribute_valence=domain_attr_val,
        )

    taxonomy_cache = TaxonomyResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={
            name: m.label_count for name, m in label_mappings.items()
        },
        label_source=CONFIG.label_source,
    )

    # Deterministic hygiene before anything downstream sees the taxonomy: drop
    # structure nodes left behind with no ideas, then report the label-health metrics.
    prune_report = prune_empty_nodes(taxonomy_cache)
    if CONFIG.verbose:
        print_health(taxonomy_cache, prune_report)

    ensure_taxonomy_ids(taxonomy_cache)
    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=taxonomy_cache,
        filename=filename,
        step="taxonomy",
        variable_key=variable_key,
    )

    # A fresh taxonomy supersedes everything derived from the one it replaces.
    # Step 5 prefers the corrected keys while they are valid (silently feeding it
    # the superseded taxonomy), and steps 5/6 skip entirely on a valid
    # mece_codes/taxonomy_codes cache when run without force_recalc.
    for stale in ("taxonomy_corrected_metadata", "taxonomy_classified_corrected",
                  "mece_codes", "mece_codes_metadata", "taxonomy_codes"):
        cache_manager.invalidate_cache(filename, stale, variable_key)

    _write_action_log(taxonomy_result, filename, variable_key)

    # Build and cache growing model (enriched facet/attribute per idea)
    if ideas_models is not None:
        enriched = _build_taxonomy_enriched_models(ideas_models, taxonomy_cache)
        restamp_assignment_ids(enriched, taxonomy_cache)
        cache_manager.save_to_cache(enriched, filename, "taxonomy_classified", variable_key)
        print(f"Growing model cached: {len(enriched)} enriched responses")

    # Counted off what was SAVED, not off the result object that went in. Those
    # two differ whenever prune_empty_nodes drops something, and the old line read
    # the pre-prune object: a run that pruned its way down to nothing still
    # reported "55 facets, 179 attributes cached".
    total_facets = sum(len(dr.facets) for dr in taxonomy_cache.partition_results.values())
    total_attrs = sum(
        len(attrs)
        for dr in taxonomy_cache.partition_results.values()
        for attrs in dr.attributes.values()
    )
    print(f"Taxonomy results cached "
          f"({total_facets} facets, {total_attrs} attributes across "
          f"{len(pydantic_results)} domains)")

    return pydantic_results


# =============================================================================
# PROMPT SAVING
# =============================================================================

def save_prompts_to_json(prompt_printer):
    """Save captured prompts to JSON file.

    Everything the runner captured goes in. A whitelist of prompt types here
    would have to be extended by hand for every new phase, and drops what it
    does not recognise without saying so — that is how the valence merge
    went missing.
    """
    if not prompt_printer or not prompt_printer.prompts:
        return

    prompts_dir = project_root / "exports" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_printer.save_prompts(str(prompts_dir / export_filename(
        FILENAME, VARIABLE, SAMPLE_SIZE, "prompts_step4", "json")))


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
    if LIMIT_N is not None and LIMIT_N < len(ideas_models):
        total = len(ideas_models)
        ideas_models = ideas_models[:LIMIT_N]
        print(f"Subset: {LIMIT_N} responses (of {total} total)")

    if extraction_metadata is None:
        extraction_metadata = load_extraction_metadata()

    discoverer = DomainDiscoverer(CONFIG, extraction_metadata)
    partition_set, label_mappings = discoverer.discover(ideas_models)

    return ideas_models, extraction_metadata, partition_set, label_mappings


# =============================================================================
# MAIN
# =============================================================================

def run_taxonomy(filename: str = FILENAME, var_name: str = VARIABLE,
                 sample_size: Optional[int] = SAMPLE_SIZE, force_recalc: bool = False):
    """Run the taxonomy: facets and attributes discovered and consolidated
    together, assigned in one pass, refined per domain, folded across domains,
    then the valence-neutral merge.

    Dataset params default to the module-level TEST_DATA constants (so existing
    callers like run_pipeline.py are unchanged); the UI passes them explicitly.
    Rebinds the module globals once so the downstream helpers see the right dataset.
    """
    global FILENAME, VARIABLE, SAMPLE_SIZE
    FILENAME, VARIABLE, SAMPLE_SIZE = filename, var_name, sample_size
    print("=" * 70)
    print("TAXONOMY PIPELINE")
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

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )

    if not force_recalc:
        cache_manager = CacheManager()
        if (cache_manager.is_metadata_cache_valid(FILENAME, "taxonomy", variable_key)
                and cache_manager.is_cache_valid(FILENAME, "taxonomy_classified", variable_key)):
            print("Taxonomy cache valid — skipping step 4 (use force_recalc=True to rerun).\n")
            return None

    ideas_models, extraction_metadata, partition_set, label_mappings = _load_and_discover()
    survey_question, language, dataset_context, dimension_name, dimension_description = \
        _extract_metadata_context(extraction_metadata)

    prompt_printer = PromptPrinter(
        enabled=True,
        print_realtime=PRINT_PROMPTS,
    )
    cost_tracker = CostTracker(filename=FILENAME, var_name=VARIABLE,
                               sample_size=SAMPLE_SIZE)

    processor = TaxonomyClassifier(CONFIG, prompt_printer=prompt_printer, cost_tracker=cost_tracker)
    taxonomy_result = processor.process(
        label_mappings=label_mappings,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=CONFIG.verbose,
        extraction_metadata=extraction_metadata,
    )

    # Cache taxonomy results (metadata + growing model). The taxonomy is displayed
    # once at the very end, so the readout reflects the final state.
    cache_taxonomy_results(partition_set, label_mappings, taxonomy_result, ideas_models=ideas_models)

    # Valence-neutral attribute merge (collapse valence-split attribute pairs).
    # Skipped on any early stop: it judges the settled attribute layer, which a
    # partial run does not have.
    if CONFIG.stop_after_phase is None:
        import asyncio
        from pipeline.step_4_classifier.valence_consolidator import ValenceConsolidator
        cache_manager = CacheManager()
        v_taxonomy = cache_manager.load_metadata_from_cache(
            filename=FILENAME, step="taxonomy",
            variable_key=variable_key, model_cls=TaxonomyResultsCache,
        )
        v_classified = cache_manager.load_from_cache(
            filename=FILENAME, step="taxonomy_classified",
            variable_key=variable_key, model_cls=TaxonomyClassifiedModel,
        )
        if v_taxonomy and v_classified:
            new_taxonomy, new_classified, _v_report, v_stats = asyncio.run(
                ValenceConsolidator(
                    CONFIG, cost_tracker=cost_tracker,
                    prompt_printer=prompt_printer,
                ).consolidate(
                    v_taxonomy, v_classified, extraction_metadata, verbose=CONFIG.verbose,
                )
            )
            if v_stats["merges"] > 0:
                ensure_taxonomy_ids(new_taxonomy)
                restamp_assignment_ids(new_classified, new_taxonomy)
                cache_manager.save_metadata_to_cache(
                    metadata=new_taxonomy, filename=FILENAME,
                    step="taxonomy", variable_key=variable_key,
                )
                cache_manager.save_to_cache(
                    data=new_classified, filename=FILENAME,
                    step="taxonomy_classified", variable_key=variable_key,
                )

    # Display the final taxonomy
    if CONFIG.verbose:
        final_tax = CacheManager().load_metadata_from_cache(
            filename=FILENAME, step="taxonomy", variable_key=variable_key, model_cls=TaxonomyResultsCache)
        if final_tax is not None:
            _print_final_taxonomy(final_tax, use_corrected=False)

    cost_tracker.finalize_step("step_4_taxonomy_classifier")

    # Save prompts
    save_prompts_to_json(prompt_printer)

    return partition_set, label_mappings, taxonomy_result, ideas_models, prompt_printer


def _print_final_taxonomy(tax_cache, use_corrected):
    """Domain > Facet > Attribute [idea count] readout of the final taxonomy."""
    from collections import Counter
    label = "CORRECTED " if use_corrected else ""
    print(f"\n{'=' * 70}\n{label}TAXONOMY (final)\n{'=' * 70}")
    for domain in sorted(tax_cache.partition_results):
        res = tax_cache.partition_results[domain]
        attrs = res.corrected_attributes if use_corrected else res.attributes
        assigns = res.corrected_attribute_assignments if use_corrected else res.attribute_assignments
        counts = Counter(assigns.values())
        print(f"\nDOMAIN: {domain}  ({sum(counts.values())} ideas)")
        for facet in sorted(attrs):
            print(f"  {facet}:")
            for a in sorted(attrs[facet], key=lambda a: -counts.get(a.get("attribute_name"), 0)):
                print(f"    - {a.get('attribute_name')} [{counts.get(a.get('attribute_name'), 0)} ideas]")


if __name__ == "__main__":
    with VerboseCapture(
        filename=FILENAME,
        var_name=VARIABLE,
        sample_size=SAMPLE_SIZE,
        step=4,
    ):
        token_tracker.reset()

        # force_recalc=True: with production cache keys a valid taxonomy is
        # already present, so a bare run would skip step 4 entirely.
        partition_set, label_mappings, taxonomy_result, ideas_models, prompt_printer = run_taxonomy(force_recalc=True)

        # Print token usage
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

# %%
