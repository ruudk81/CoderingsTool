#%%

"""Bouw een volledig codeboek van begin tot eind — buiten de productiepijplijn om.

Dunne wrapper rond `run_codeGenerator.generate_codebook()` (dezelfde keten
als de productierunner), gericht op een eigen cache-map en een leesbaar
codeboek in plaats van een cache-write. Draait de hele stap-5-keten in één
script:

    taxonomy_input -> concept_inventory -> relations (2 LLM-calls) ->
    consolidator (geen LLM) -> codebook_writer (1 LLM-call) ->
    mece (2 LLM-calls per ronde, max 3 rondes) -> codebook_writer (herschrijft
    alleen de samengevoegde codes)

Drie tot tien LLM-calls, afhankelijk van de codeboekgrootte en van hoeveel
MECE-rondes iets samenvoegen (elke ronde: 1 detectiecall, plus 1 adjudicatie-
call zodra er kandidaat-paren zijn; maximaal 3 rondes).

    cd src && python -m pipeline.step_5_codeGenerator.run_codebook_preview
    cd src && python -m pipeline.step_5_codeGenerator.run_codebook_preview --cache-dir <pad>
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import MISCELLANEOUS_CODE_LABELS
from utils.llm import token_tracker

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from pipeline.step_5_codeGenerator.concept_inventory import build_inventory, t_keep
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.run_codeGenerator import (
    FALLBACK_DIAGNOSTIC, _match_shape, _shape_lookup, generate_codebook, report_codebook_build,
)
from pipeline.step_5_codeGenerator.taxonomy_input import IdeaUnit, build_attribute_refs, build_idea_units
from pipeline.step_5_codeGenerator.view_relations import load_cache

# Step 4 is being rewritten in a parallel worktree. Unlike view_relations.py's
# gate runs (which pin a frozen fixture for comparability across runs), this
# runner's whole point is to show the user a codebook from the LATEST
# taxonomy — a moving target here is the intended behaviour, not a hazard.
DEFAULT_CACHE_DIR = project_root / "data" / "cache"

CODEBOOK_OUTPUT = project_root / "exports" / "codebook" / "codebook_preview.md"

DIRECTION_SYMBOL = {"positive": "+", "negative": "−", "neutral": "neutraal"}


def _taxonomy_timestamp(cache_dir: Path) -> str:
    db_path = cache_dir / "cache.db"
    if not db_path.exists():
        return "onbekend"
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT created_at FROM cache_metadata WHERE step_name = 'taxonomy_classified' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else "onbekend"


def build_markdown(
    codes: List[ConsolidatedCode],
    shapes,
    concept_by_id,
    overig_ids: List[str],
    header: dict,
) -> str:
    lookup = _shape_lookup(shapes, concept_by_id)
    rows = []
    unmatched = []
    for code in codes:
        shape = _match_shape(code, lookup)
        if shape is None:
            unmatched.append(code.code_name)
        n_resp = len(shape.resp_ids) if shape is not None else 0
        rows.append((n_resp, code.code_name, DIRECTION_SYMBOL.get(code.valence, code.valence),
                     code.definition, ", ".join(code.source_attributes)))
    if unmatched:
        print(f"WAARSCHUWING: {len(unmatched)} code(s) niet aan hun vorm gekoppeld "
              f"voor het respondentenaantal: {unmatched}")

    overig_names = [concept_by_id[i].name for i in overig_ids if i in concept_by_id]
    overig_resp_ids = set()
    for i in overig_ids:
        if i in concept_by_id:
            overig_resp_ids |= concept_by_id[i].resp_ids
    overig_n_resp = len(overig_resp_ids)
    overig_share = overig_n_resp / header["n_resp_total"] if header["n_resp_total"] else 0.0

    if overig_names:
        label = MISCELLANEOUS_CODE_LABELS.get(header["language"], "Overig")
        rows.append((overig_n_resp, label, "neutraal",
                     "Restcategorie: attributen die geen eigen of gepoolde code haalden.",
                     ", ".join(overig_names)))

    rows.sort(key=lambda r: -r[0])

    lines = [
        f"# Codeboek — {header['survey_question'] or header['var_name']}",
        "",
        f"**Taxonomie:** {header['taxonomy_timestamp']}, {header['n_attrs']} attributen",
        f"**T_keep:** {header['t_keep']} ({header['t_keep_share']:.0%} van "
        f"{header['n_resp_total']} respondenten)",
        f"**Totaal codes:** {len(rows)}",
        f"**Overig-aandeel:** {overig_share:.1%} van de respondenten "
        f"({overig_n_resp} van {header['n_resp_total']})",
        "",
        "| Code | Richting | Definitie | Respondenten | Bronattributen |",
        "|---|---|---|---:|---|",
    ]
    for n_resp, name, direction, definition, sources in rows:
        lines.append(f"| {name} | {direction} | {definition} | {n_resp} | {sources} |")
    return "\n".join(lines), overig_n_resp, overig_share


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    print(f"Cache-map: {cache_dir}")
    taxonomy_ts = _taxonomy_timestamp(cache_dir)
    print(f"Taxonomie-timestamp: {taxonomy_ts}")

    token_tracker.reset()

    extraction_metadata, classified_ideas, taxonomy_cache = load_cache(cache_dir)
    if taxonomy_cache is None or not classified_ideas:
        print("\nERROR: cache onvolledig — is de cache-map correct?")
        return

    language = getattr(extraction_metadata, "lang", "") or "Dutch"
    dimension_name = getattr(extraction_metadata, "primary_dimension", "") or ""
    survey_question = getattr(extraction_metadata, "var_lab", "") or ""
    dimension_diagnostic = (
        get_dimension(dimension_name).criterion if dimension_name else FALLBACK_DIAGNOSTIC
    )

    units = build_idea_units(classified_ideas)
    refs = build_attribute_refs(taxonomy_cache.partition_results)
    concepts = build_inventory(units, refs)
    concept_by_id = {c.attribute_id: c for c in concepts}

    idea_units_by_attribute: Dict[str, List[IdeaUnit]] = defaultdict(list)
    for unit in units:
        idea_units_by_attribute[unit.attribute_id].append(unit)

    config = CodebookConfig()
    n_resp_total = len(classified_ideas)
    threshold = t_keep(n_resp_total, config)

    print(f"Attributen: {len(refs)} (concepten met idee: {len(concepts)})")
    print(f"T_keep = {threshold} over {n_resp_total} respondenten")
    print("LLM-calls worden uitgevoerd (relaties, consolidatie-namen, schrijven, "
          "MECE-afdwinging)...\n")

    result = generate_codebook(
        concepts, idea_units_by_attribute, threshold, dimension_diagnostic, language,
        config, verbose=True,
    )
    report_codebook_build(result)

    header = {
        "survey_question": survey_question,
        "var_name": getattr(extraction_metadata, "var_name", ""),
        "taxonomy_timestamp": taxonomy_ts,
        "n_attrs": len(refs),
        "t_keep": threshold,
        "t_keep_share": config.t_keep_share,
        "n_resp_total": n_resp_total,
        "language": language,
    }
    markdown, overig_n_resp, overig_share = build_markdown(
        result.codes, result.shapes, concept_by_id, result.overig_ids, header,
    )
    CODEBOOK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CODEBOOK_OUTPUT.write_text(markdown, encoding="utf-8")
    print(f"\nCodeboek geschreven: {CODEBOOK_OUTPUT}")

    codes = result.codes
    n_pos = sum(1 for c in codes if c.valence == "positive")
    n_neg = sum(1 for c in codes if c.valence == "negative")
    n_neu = len(codes) - n_pos - n_neg
    print(f"Codes: {len(codes)} ({n_pos} positief, {n_neg} negatief, {n_neu} neutraal)")
    print(f"Overig: {overig_n_resp}/{n_resp_total} respondenten ({overig_share:.1%})")

    if token_tracker.call_count > 0:
        print("\n" + token_tracker.get_summary())


if __name__ == "__main__":
    main()

# %%
