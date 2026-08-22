#%%

"""Wat step 5 rond de keten heen doet: inlezen, wegschrijven, rapporteren.

De keten zelf staat in `run_codebook.py` en de fasemodules ernaast. Hier
staat wat elke keten nodig heeft en wat
niet aan één keten toebehoort:

    inlezen      step-3-metadata, step-4-taxonomie, geclassificeerde ideeen
    wegschrijven het codeboek onder "mece_codes", waar step 6 het opent
    afronden     de Overig-sweep (dekkingsgarantie) en de scorecard
    rapporteren  het codeboek naar console, de prompts naar exports/

Dit was de bovenhelft van `run_codeGenerator.py`. Die module was twee dingen
tegelijk — de v1-orkestratie en deze plumbing — waardoor v1 niet met pensioen
kon zonder de opvolger mee te nemen. De v1-orkestratie staat nu in
`_quarantine_v1/`.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent paths for imports
# Vastgelegde afwijking van het origineel: dit bestand ligt in `consensus/`,
# één map dieper dan `step_5_codeGenerator/codebook_io.py`, dus wijst hetzelfde
# aantal `.parent`-stappen hier een niveau te hoog. Eén extra `.parent` houdt
# `project_root` op dezelfde map (de repo-root) als in het origineel — een
# positieafhankelijke constante kan niet byte-identiek zijn op een andere
# diepte. Zie test-1-report.md, Concern 2 (2026-08-22).
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import models
from config import MISCELLANEOUS_CODE_LABELS
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.exportNaming import export_filename
from utils.identity import ensure_codebook_ids

from .codebook_verifier import (
    build_scorecard, collect_idea_assignments, collect_taxonomy_attributes, format_scorecard,
)
from models import ConsolidatedCode

from models import CodingResultsCache
from models import (
    DomainResultModel, DomainSet, TaxonomyClassifiedModel, TaxonomyResultsCache,
)

from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

# Valt terug op het associatie-stramien wanneer de metadata geen dimensie noemt.
FALLBACK_DIAGNOSTIC = "Do responses mainly differ in qualities, traits, images, or associations?"


# =============================================================================
# INLEZEN
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
# RAPPORTEREN
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
# WEGSCHRIJVEN
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
    narrative: str = "",
) -> None:
    """Cache codebook results for later use by code assignment (step 6).

    `step` names the cache key step 6 and step 7 read from. `narrative` is
    additive and defaults to empty, so production's call site is untouched;
    only the consensus chain passes it, to stamp its provenance into the
    shared cache key (see `consensus.run_codebook.provenance`)."""
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
        codebook_narrative=narrative,
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
# AFRONDEN
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
# PROMPTS BEWAREN
# =============================================================================

def save_prompts_to_json(prompt_printer, doctype: str = "prompts_step5"):
    """Save captured prompts to JSON file.

    Everything the runner captured goes in, unfiltered — no doctype whitelist
    here (see run_classifier.py's save_prompts_to_json for why).

    `doctype` defaults to production's own export name, so this call site does
    not change. The consensus runner passes `"prompts_step5c"` — without a
    distinct name it would silently overwrite production's export (`.save_prompts`
    opens the target in `'w'` mode, no merge), and the two chains' prompt
    captures are exactly what a comparison between them needs intact.
    """
    if not prompt_printer or not prompt_printer.prompts:
        return

    prompts_dir = project_root / "exports" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_printer.save_prompts(str(prompts_dir / export_filename(
        FILENAME, VARIABLE, SAMPLE_SIZE, doctype, "json")))
