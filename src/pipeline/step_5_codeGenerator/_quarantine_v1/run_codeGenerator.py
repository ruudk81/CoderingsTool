#%%

"""De v1-keten van step 5 — MET PENSIOEN sinds de v2-promotie (2026-08-18).

Draait niet meer in productie: `run_pipeline.py` en `app_backend.py` roepen
`v2.run_codebook_v2` aan. Deze module blijft staan omdat v2 op één dataset is
gemeten; blijkt v2 elders te breken, dan is dit de keten om tegen af te zetten.

    taxonomy_input -> concept_inventory -> relations (2 LLM-calls) ->
    consolidator (deterministisch: dedup, pooling, richting) -> codebook_writer
    (1 LLM-call) -> mece (2 LLM-calls per ronde, iteratief) -> codebook_writer
    (herschrijven van samengevoegde codes) -> drie deterministische bewakers ->
    Overig-sweep -> scorecard -> cache voor step 6.

Wat v1 en v2 delen staat een map hoger: `codebook_io.py` (inlezen, wegschrijven,
rapporteren), `code_shape.py`, `prompts_common.py`, `codebook_writer.py`.

Waarom v1 met pensioen ging staat in
`.superpowers/specs/2026-08-18-step5-v2-promotienotitie.md`.

Handmatig draaien:
    cd src && python -m pipeline.step_5_codeGenerator._quarantine_v1.run_codeGenerator
"""

import asyncio
import copy
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.costTracker import CostTracker
from utils.llm import token_tracker
from utils.promptPrinter import PromptPrinter
from utils.saveVerbose import VerboseCapture

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from pipeline.step_5_codeGenerator.code_shape import CodeShape, _match_shape, _shape_lookup
from pipeline.step_5_codeGenerator.codebook_io import (
    FALLBACK_DIAGNOSTIC, FILENAME, SAMPLE_SIZE, VARIABLE, apply_overig_sweep,
    cache_mece_results, load_classified_ideas, load_extraction_metadata,
    load_taxonomy_cache, print_codebook_results, run_scorecard, save_prompts_to_json,
)
from pipeline.step_5_codeGenerator.codebook_writer import (
    find_duplicate_definitions, find_naming_mismatches, resolve_duplicate_names, write_codebook,
)
from pipeline.step_5_codeGenerator.concept_inventory import Concept, build_inventory, t_keep
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.taxonomy_input import IdeaUnit, build_attribute_refs, build_idea_units

from .consolidator import consolidate, normalize_relations
from .mece import enforce_mece
from .prompts_mece import CodeCandidate
from .prompts_umbrella_merge import umbrellas_from_relations
from .relations import apply_umbrella_merge, resolve_relations, resolve_umbrella_merge

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from models import TaxonomyResultsCache

PRINT_PROMPTS = False  # Set True to print prompts to console in real-time
CONFIG = CodebookConfig()


# =============================================================================
# DE KETEN
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
