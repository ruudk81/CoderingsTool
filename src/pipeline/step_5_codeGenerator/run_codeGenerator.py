#%%

"""
Step 5: Code Generator runner.

Pipeline: load taxonomy from step 4 cache, then build the codebook:

    taxonomy_input -> concept_inventory -> relations (2 LLM calls) ->
    consolidator (deterministic: dedup, pooling, direction) -> codebook_writer
    (1 LLM call) -> mece (2 LLM calls per round, iterating) -> codebook_writer
    (re-write of merged codes only) -> three deterministic guards (duplicate
    names, duplicate definitions, naming mismatch) -> Overig sweep ->
    scorecard -> cache for step 6.

`generate_codebook()` is the reusable orchestration entry point — also used
by run_codebook_preview.py (same chain, a different cache dir, a markdown
report instead of a cache write).
"""

import asyncio
import copy
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import models
from config import MISCELLANEOUS_CODE_LABELS
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.costTracker import CostTracker
from utils.exportNaming import export_filename
from utils.identity import ensure_codebook_ids
from utils.llm import token_tracker
from utils.promptPrinter import PromptPrinter
from utils.saveVerbose import VerboseCapture

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from pipeline.step_5_codeGenerator.codebook_verifier import (
    build_scorecard, collect_idea_assignments, collect_taxonomy_attributes, format_scorecard,
)
from pipeline.step_5_codeGenerator.codebook_writer import (
    find_duplicate_definitions, find_naming_mismatches, resolve_duplicate_names, write_codebook,
)
from pipeline.step_5_codeGenerator.concept_inventory import Concept, build_inventory, t_keep
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.consolidator import CodeShape, consolidate, normalize_relations
from pipeline.step_5_codeGenerator.mece import enforce_mece
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.prompts_mece import CodeCandidate
from pipeline.step_5_codeGenerator.prompts_umbrella_merge import umbrellas_from_relations
from pipeline.step_5_codeGenerator.relations import apply_umbrella_merge, resolve_relations, resolve_umbrella_merge
from pipeline.step_5_codeGenerator.taxonomy_input import IdeaUnit, build_attribute_refs, build_idea_units

from models import CodingResultsCache
from models import (
    DomainResultModel, DomainSet, TaxonomyClassifiedModel, TaxonomyResultsCache,
)

FALLBACK_DIAGNOSTIC = "Do responses mainly differ in qualities, traits, images, or associations?"


# =============================================================================
from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = CodebookConfig()


# =============================================================================
# DATA LOADING
# =============================================================================

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


def load_taxonomy_cache(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> Optional[TaxonomyResultsCache]:
    """Load cached taxonomy results from step 4."""
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
        step="taxonomy",
        variable_key=variable_key,
        model_cls=TaxonomyResultsCache,
    )


def load_classified_ideas(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
) -> Optional[List[TaxonomyClassifiedModel]]:
    """Load step 4's taxonomy-classified growing model (ideas with attribute/valence)."""
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
# CODEBOOK GENERATION — the reusable chain
# =============================================================================

class _RoundLog:
    """Collects `enforce_mece`'s per-round `log.add(...)` calls for the print
    line at the end of the run. No `decision_log.py` (not built yet) —
    duck-typed, like `write_codebook`'s own `log` parameter."""
    def __init__(self):
        self.rounds: List[dict] = []

    def add(self, **kwargs):
        self.rounds.append(kwargs)


class _CollisionLog:
    """Collects `resolve_duplicate_names`'s per-collision `log.add(...)` calls
    for the print line at the end of the run — the same duck-typed shape as
    `_RoundLog` above."""
    def __init__(self):
        self.collisions: List[dict] = []

    def add(self, **kwargs):
        self.collisions.append(kwargs)


def _shape_lookup(
    shapes: List[CodeShape], concept_by_id: Dict[str, Concept],
) -> Dict[tuple, CodeShape]:
    """Key shapes by (their source-attribute names, valence) — the same two
    things `write_codebook` echoes back on each `ConsolidatedCode` — so a
    returned code can be matched to the shape it came from without needing
    write_codebook to carry shape identity through the LLM round-trip."""
    lookup = {}
    for shape in shapes:
        names = frozenset(concept_by_id[m].name for m in shape.members if m in concept_by_id)
        lookup[(names, shape.valence)] = shape
    return lookup


def _match_shape(
    code: ConsolidatedCode, lookup: Dict[tuple, CodeShape],
) -> Optional[CodeShape]:
    return lookup.get((frozenset(code.source_attributes), code.valence))


def _index_codes_by_shape_key(
    codes: List[ConsolidatedCode], lookup: Dict[tuple, CodeShape],
) -> Dict[str, ConsolidatedCode]:
    """Maps each shape's own `.key` (unique per run, assigned once by
    `consolidate()`) to the `ConsolidatedCode` written for it. Indexing by
    `code_name` instead collapses the moment two different shapes are given
    the same name: a dict comprehension keyed on name keeps only the last
    code for that name, so every shape sharing it silently inherits ONE
    shape's definition, including shapes whose actual members that text does
    not describe. `.key` is never reused across shapes, so this mapping never
    collapses regardless of what name the writer chose."""
    indexed: Dict[str, ConsolidatedCode] = {}
    for code in codes:
        matched = _match_shape(code, lookup)
        if matched is not None:
            indexed[matched.key] = code
    return indexed


@dataclass
class GeneratedCodebook:
    """Everything one run of the chain produces, including the three
    deterministic guards — computed once here so every caller (production,
    the preview script) reports the same checks instead of re-deriving them."""
    shapes: List[CodeShape]
    overig_ids: List[str]
    codes: List[ConsolidatedCode]
    merge_failed: bool
    mece_rounds: List[dict]
    collisions: List[dict]
    naming_mismatches: List[dict]
    duplicate_definitions: List[dict]
    concept_by_id: Dict[str, Concept] = field(repr=False)


async def _generate_codebook_async(
    concepts: List[Concept],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    threshold: int,
    dimension_diagnostic: str,
    language: str,
    config: CodebookConfig,
    verbose: bool,
    prompt_printer=None,
) -> GeneratedCodebook:
    relations_before = await resolve_relations(
        concepts, config, language, verbose=verbose, prompt_printer=prompt_printer,
    )
    umbrellas_before = umbrellas_from_relations(relations_before)
    merge_result = await resolve_umbrella_merge(
        umbrellas_before, config, verbose=verbose, prompt_printer=prompt_printer,
    )
    relations_final = (
        apply_umbrella_merge(relations_before, merge_result)
        if merge_result is not None else relations_before
    )
    relation_map = normalize_relations(relations_final, concepts)
    shapes, overig_ids = consolidate(concepts, relation_map, threshold)
    codes = await write_codebook(
        shapes, concepts, dimension_diagnostic, language, config, verbose=verbose,
        prompt_printer=prompt_printer,
    )

    # MECE enforcement: looking at codes as a SET, not per shape.
    # `code_by_shape_key` keeps the full written text (including
    # diagnostic_test) of codes no round touches, keyed on the shape itself
    # (never on the written name — that can coincide between two different
    # shapes).
    concept_by_id = {c.attribute_id: c for c in concepts}
    shape_lookup = _shape_lookup(shapes, concept_by_id)
    code_by_shape_key = _index_codes_by_shape_key(codes, shape_lookup)
    candidates = [
        CodeCandidate(name=code.code_name, definition=code.definition,
                      indicators=tuple(code.typical_indicators), valence=code.valence,
                      shape=_match_shape(code, shape_lookup))
        for code in codes if _match_shape(code, shape_lookup) is not None
    ]
    round_log = _RoundLog()
    final_candidates = await enforce_mece(
        candidates, idea_units_by_attribute, config, log=round_log, verbose=verbose,
        prompt_printer=prompt_printer,
    )
    merged = [c for c in final_candidates if c.shape.origin == "mece_merge"]
    untouched = [c for c in final_candidates if c.shape.origin != "mece_merge"]

    # Only the merged codes get new text — unchanged codes keep their
    # previously written definition/diagnostic_test. The rewrite sees the
    # already-fixed names of the unchanged codes (taken_names) so it does not
    # write over them — a prompt rule, not a guarantee;
    # resolve_duplicate_names below is the deterministic backstop over the
    # complete, reunited codebook.
    untouched_names = [code_by_shape_key[c.shape.key].code_name for c in untouched]
    rewritten = await write_codebook(
        [c.shape for c in merged], concepts, dimension_diagnostic, language, config,
        taken_names=untouched_names, verbose=verbose, prompt_printer=prompt_printer,
    ) if merged else []
    final_shapes = [c.shape for c in untouched] + [c.shape for c in merged]
    final_codes = [code_by_shape_key[c.shape.key] for c in untouched] + rewritten

    collision_log = _CollisionLog()
    final_codes = resolve_duplicate_names(final_codes, final_shapes, log=collision_log)

    mismatches = find_naming_mismatches(final_codes, final_shapes, concept_by_id)
    duplicate_defs = find_duplicate_definitions(final_codes, final_shapes)

    return GeneratedCodebook(
        shapes=final_shapes, overig_ids=overig_ids, codes=final_codes,
        merge_failed=merge_result is None, mece_rounds=round_log.rounds,
        collisions=collision_log.collisions, naming_mismatches=mismatches,
        duplicate_definitions=duplicate_defs, concept_by_id=concept_by_id,
    )


def generate_codebook(
    concepts: List[Concept],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    threshold: int,
    dimension_diagnostic: str,
    language: str,
    config: CodebookConfig,
    verbose: bool = True,
    prompt_printer=None,
) -> GeneratedCodebook:
    """Sync wrapper around the chain — the one orchestration entry point,
    shared by `run_codebook()` and `run_codebook_preview.py`. `prompt_printer`
    is optional: None (the default, used by the preview script and every
    test) costs nothing and captures nothing."""
    return asyncio.run(_generate_codebook_async(
        concepts, idea_units_by_attribute, threshold, dimension_diagnostic, language,
        config, verbose, prompt_printer,
    ))


def report_codebook_build(result: GeneratedCodebook) -> None:
    """Print the three deterministic guards + the MECE round log — the same
    reporting the dev preview script prints, now also surfaced by a
    production run instead of being visible only from the dev loop."""
    if result.merge_failed:
        print("WAARSCHUWING: consolidatiecall mislukt — doorgegaan met ongeconsolideerde namen.")

    if result.collisions:
        print(f"WAARSCHUWING: {len(result.collisions)} dubbele codenaam/namen deterministisch opgelost:")
        for c in result.collisions:
            print(f"  '{c['name']}' ({c['kept_n_resp']} resp.) behouden; "
                  f"kleinere code ({c['renamed_n_resp']} resp.) hernoemd naar '{c['renamed_to']}'")

    if result.naming_mismatches:
        print(f"WAARSCHUWING: {len(result.naming_mismatches)} code(s) waarvan de naam geen woord deelt "
              f"met een van zijn bronattributen:")
        for m in result.naming_mismatches:
            print(f"  '{m['code_name']}' ({m['n_resp']} resp.) — leden: "
                  f"{', '.join(m['members'])}")

    if result.duplicate_definitions:
        print(f"WAARSCHUWING: {len(result.duplicate_definitions)} groep(en) codes met identieke definitie:")
        for d in result.duplicate_definitions:
            names = ", ".join(f"'{c['code_name']}' ({c['n_resp']} resp.)" for c in d["codes"])
            print(f"  {names}")

    if result.mece_rounds:
        total_merges = sum(r["merges"] for r in result.mece_rounds)
        print(f"MECE: {len(result.mece_rounds)} ronde(s), {total_merges} samenvoeging(en) totaal")
        for r in result.mece_rounds:
            acc = f"{r['mean_accuracy']:.0%}" if r["mean_accuracy"] is not None else "—"
            reason = f", reden einde: {r['reason']}" if r["reason"] else ""
            print(f"  ronde {r['round']}: {r['pairs_found']} paar/paren gevonden, "
                  f"{r['pairs_probed']} bevraagd, gem. accuracy {acc}, "
                  f"{r['merges']} samenvoeging(en){reason}")
            for p in r.get("pairs", []):
                decision = "SAMENVOEGEN" if p["merged"] else "apart"
                print(f"    {p['code_a']} vs {p['code_b']}: accuracy {p['accuracy']:.0%}, "
                      f"both_rate {p['both_rate']:.0%} -> {decision}")


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_codebook_results(codes: List[ConsolidatedCode]):
    """Print codebook results: codes with definitions and source attributes."""
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
# MECE CACHING
# =============================================================================

def cache_mece_results(
    partition_set: DomainSet,
    pydantic_results: Dict[str, DomainResultModel],
    codes: List[ConsolidatedCode],
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
    step: str = "mece_codes",
) -> None:
    """Cache codebook results for later use by code assignment (step 6).

    `step` defaults to the v1 cache key; v2 passes "mece_codes_v2" so both
    codebooks stay loadable side by side on the same taxonomy."""
    filename = FILENAME if filename is None else filename
    variable = VARIABLE if variable is None else variable
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size,
        )

    n_codes = len(codes)
    mece_cache = CodingResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={
            name: r.n_labels for name, r in pydantic_results.items()
        },
        total_categories=n_codes,
        raw_codes=[c.model_dump() for c in codes],
    )

    # Mint K# (list order: written codes, then Overig) and fill any
    # source_attribute_ids still missing — new codebooks are id-bearing on
    # disk, not just normalized at load.
    ensure_codebook_ids(mece_cache)

    cache_manager = CacheManager()
    saved = cache_manager.save_metadata_to_cache(
        metadata=mece_cache,
        filename=filename,
        step=step,
        variable_key=variable_key,
    )
    total_facets = sum(
        len(r.facets) for r in pydantic_results.values()
    )
    if saved:
        print(f"Codebook cached "
              f"({n_codes} codes, {total_facets} facets across "
              f"{len(pydantic_results)} domains)")
    else:
        print(f"ERROR: codebook NOT cached ({n_codes} codes) — downstream steps "
              f"will regenerate. See CACHE SAVE FAILED above for the cause.")


# =============================================================================
# SCORECARD
# =============================================================================

def apply_overig_sweep(
    codes: List[ConsolidatedCode],
    pydantic_results: Dict[str, DomainResultModel],
    language: str,
) -> Optional[str]:
    """Route attributes no code placed into a single catch-all 'Overig' code.

    Guarantees 100% attribute/idea coverage by construction. Mutates `codes`
    in place. Returns the Overig code name.
    """
    # Referenced = taxonomy attributes AND attributes ideas were actually assigned to
    # (the latter catches step-4 dangling assignments → guarantees 100% idea coverage).
    all_attrs = collect_taxonomy_attributes(pydantic_results)
    idea_attrs = [a for a in collect_idea_assignments(pydantic_results).values() if a]
    referenced = list(dict.fromkeys(all_attrs + idea_attrs))
    covered = set()
    for code in codes:
        covered.update(code.source_attributes or [])
    orphans = [a for a in referenced if a not in covered]
    # Always emit Overig — even with zero orphans at generation time, step 6
    # assignment can still produce an idea with no confident code match; Overig
    # must exist as a routing target instead of falling through to __UNASSIGNED__.

    # Union of ids per orphan name across ALL domains — the catch-all covers the
    # attribute wherever it lives. Dangling idea-assigned names have no id.
    name_to_ids: Dict[str, List[str]] = {}
    for r in pydantic_results.values():
        for attrs in r.attributes.values():
            for a in attrs:
                if a.get("attribute_name") and a.get("attribute_id"):
                    ids = name_to_ids.setdefault(a["attribute_name"], [])
                    if a["attribute_id"] not in ids:
                        ids.append(a["attribute_id"])

    label = MISCELLANEOUS_CODE_LABELS.get(language, "Overig")
    codes.append(ConsolidatedCode(
        code_name=label,
        definition="Catch-all voor antwoorden die geen specifieke code kregen "
                   "(o.a. diffuus of algemeen oordeel zonder concreet onderwerp).",
        diagnostic_test="valt buiten alle specifieke codes",
        valence="neutral",
        typical_indicators=[],
        source_attributes=orphans,  # may be empty list
        source_attribute_ids=[i for name in orphans for i in name_to_ids.get(name, [])],
    ))
    return label


def run_scorecard(
    codes: List[ConsolidatedCode],
    pydantic_results: Dict[str, DomainResultModel],
    overig_code_name: Optional[str] = None,
):
    """Build the post-generation verification scorecard (PASS/FAIL) and print it.

    Console only — the PASS/FAIL readout is captured in the verbose log (which is
    auto-pruned); no separate JSON file is written.
    """
    scorecard = build_scorecard(codes, pydantic_results, overig_code_name)
    print("\n" + format_scorecard(scorecard))
    return scorecard


# =============================================================================
# PROMPT SAVING
# =============================================================================

def save_prompts_to_json(prompt_printer):
    """Save captured prompts to JSON file.

    Everything the runner captured goes in, unfiltered — no doctype whitelist
    here (see run_classifier.py's save_prompts_to_json for why).
    """
    if not prompt_printer or not prompt_printer.prompts:
        return

    prompts_dir = project_root / "exports" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_printer.save_prompts(str(prompts_dir / export_filename(
        FILENAME, VARIABLE, SAMPLE_SIZE, "prompts_step5", "json")))


# =============================================================================
# MAIN
# =============================================================================

def _project_corrected(corrected_taxonomy):
    """A copy of the corrected taxonomy where the corrected_* fields are exposed as
    the plain `attributes` / `attribute_assignments` — so the whole codebook chain
    (reconstruction, mece_codes cache, step 6) consumes corrected attributes with no
    further code change."""
    proj = TaxonomyResultsCache.model_validate(copy.deepcopy(corrected_taxonomy.model_dump()))
    for r in proj.partition_results.values():
        r.attributes = r.corrected_attributes
        r.attribute_assignments = r.corrected_attribute_assignments
    return proj


def run_codebook(filename: str = FILENAME, var_name: str = VARIABLE,
                 sample_size: Optional[int] = SAMPLE_SIZE, force_recalc: bool = False):
    """Run codebook generation from cached taxonomy results.

    Dataset params default to the module-level TEST_DATA constants (so existing
    callers like run_pipeline.py are unchanged); the UI passes them explicitly.
    Rebinds the module globals once so downstream helpers see the right dataset.
    """
    global FILENAME, VARIABLE, SAMPLE_SIZE
    FILENAME, VARIABLE, SAMPLE_SIZE = filename, var_name, sample_size
    print("=" * 70)
    print("CODE GENERATOR (loading taxonomy from cache)")
    print("=" * 70)

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )

    if not force_recalc:
        cache_manager = CacheManager()
        if cache_manager.is_metadata_cache_valid(FILENAME, "mece_codes", variable_key):
            print("Codebook cache valid — skipping generation (use force_recalc=True to rerun).\n")
            return None

    extraction_metadata = load_extraction_metadata()
    classified_ideas = load_classified_ideas()
    taxonomy_cache = load_taxonomy_cache()
    if taxonomy_cache is None:
        print("\nERROR: No cached taxonomy results found.")
        print("Run step 4 taxonomy first (step_4_classifier).")
        return None

    partition_set = taxonomy_cache.partition_set

    # LEGACY READ PATH: chains built before the P7 promotion (2026-08-01) may
    # carry a corrected taxonomy from the old P9 over-merge correction. Prefer
    # it for those chains so delivered work keeps its shipped shape. New step-4
    # runs never write these keys and invalidate stale ones after saving.
    cache_manager = CacheManager()
    pydantic_results = taxonomy_cache.partition_results
    if cache_manager.is_metadata_cache_valid(FILENAME, "taxonomy_corrected", variable_key):
        corrected_tax = cache_manager.load_metadata_from_cache(
            FILENAME, "taxonomy_corrected", variable_key, TaxonomyResultsCache)
        if corrected_tax is not None:
            pydantic_results = _project_corrected(corrected_tax).partition_results
            corrected_cls = cache_manager.load_from_cache(
                FILENAME, "taxonomy_classified_corrected", variable_key, TaxonomyClassifiedModel)
            if corrected_cls:
                classified_ideas = corrected_cls
            print("  Using CORRECTED taxonomy (over-merge correction from step 4)")

    n_facets = sum(len(r.facets) for r in pydantic_results.values())
    n_attrs = sum(
        len(attrs) for r in pydantic_results.values()
        for attrs in r.attributes.values()
    )
    print(f"  Loaded taxonomy: {n_facets} facets, {n_attrs} attributes "
          f"across {len(pydantic_results)} domains")

    language = getattr(extraction_metadata, "lang", "") or "Dutch"
    dimension_name = getattr(extraction_metadata, "primary_dimension", "") or ""
    dimension_diagnostic = (
        get_dimension(dimension_name).criterion if dimension_name else FALLBACK_DIAGNOSTIC
    )

    units = build_idea_units(classified_ideas)
    refs = build_attribute_refs(pydantic_results)
    concepts = build_inventory(units, refs)
    idea_units_by_attribute: Dict[str, List[IdeaUnit]] = defaultdict(list)
    for unit in units:
        idea_units_by_attribute[unit.attribute_id].append(unit)

    n_resp_total = len(classified_ideas)
    threshold = t_keep(n_resp_total, CONFIG)
    print(f"  Attributes: {len(refs)} (concepts with an idea: {len(concepts)})")
    print(f"  T_keep = {threshold} over {n_resp_total} respondents")

    cost_tracker = CostTracker(filename=FILENAME, var_name=VARIABLE,
                               sample_size=SAMPLE_SIZE)
    snapshot_before = token_tracker.snapshot()

    prompt_printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)
    result = generate_codebook(
        concepts, idea_units_by_attribute, threshold, dimension_diagnostic, language,
        CONFIG, verbose=CONFIG.verbose, prompt_printer=prompt_printer,
    )
    report_codebook_build(result)

    # One phase for the whole chain (relations + writer + MECE): every rung
    # in STEP_MODEL currently shares one model, so a finer split buys nothing.
    cost_tracker.record_phase(
        "step_5_code_generator", "codebook_generation",
        snapshot_before, token_tracker.snapshot(), model=CONFIG.model_writer,
    )
    cost_tracker.finalize_step("step_5_code_generator")

    codes = result.codes

    # Overig sweep: route any unplaced attribute into a catch-all → 100% coverage
    overig_name = apply_overig_sweep(codes, pydantic_results, language)

    # Print codebook results (includes Overig if added)
    print_codebook_results(codes)

    # Post-generation verification scorecard (PASS/FAIL against the definition of done)
    run_scorecard(codes, pydantic_results, overig_name)

    # Cache for downstream use by step 6 (code assigner)
    cache_mece_results(partition_set, pydantic_results, codes)

    save_prompts_to_json(prompt_printer)

    return codes


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    with VerboseCapture(
        filename=FILENAME,
        var_name=VARIABLE,
        sample_size=SAMPLE_SIZE,
        step=5,
    ):
        token_tracker.reset()

        result = run_codebook()

        # Print token usage
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

# %%
