#%%

"""Bouw een volledig codeboek van begin tot eind — buiten de productiepijplijn om.

Draait de hele stap-5-keten in één script en schrijft een leesbaar codeboek:

    taxonomy_input -> concept_inventory -> relations (2 LLM-calls) ->
    consolidator (geen LLM) -> codebook_writer (1 LLM-call) ->
    mece (2 LLM-calls per ronde, max 3 rondes) -> codebook_writer (herschrijft
    alleen de samengevoegde codes)

Drie tot tien LLM-calls, afhankelijk van de codeboekgrootte en van hoeveel
MECE-rondes iets samenvoegen (elke ronde: 1 detectiecall, plus 1 adjudicatie-
call zodra er kandidaat-paren zijn; maximaal 3 rondes). Dit is een dev-loop-
runner — niet gewired in run_codeGenerator.py (dat is een aparte taak).

    cd src && python -m pipeline.step_5_codeGenerator.run_codebook_preview
    cd src && python -m pipeline.step_5_codeGenerator.run_codebook_preview --cache-dir <pad>
"""

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import MISCELLANEOUS_CODE_LABELS
from utils.llm import token_tracker

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from pipeline.step_5_codeGenerator.codebook_writer import write_codebook
from pipeline.step_5_codeGenerator.concept_inventory import Concept, build_inventory, t_keep
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.consolidator import CodeShape, consolidate, normalize_relations
from pipeline.step_5_codeGenerator.mece import enforce_mece
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.prompts_mece import CodeCandidate
from pipeline.step_5_codeGenerator.prompts_umbrella_merge import umbrellas_from_relations
from pipeline.step_5_codeGenerator.relations import apply_umbrella_merge, resolve_relations, resolve_umbrella_merge
from pipeline.step_5_codeGenerator.taxonomy_input import build_attribute_refs, build_idea_units
from pipeline.step_5_codeGenerator.view_relations import load_cache

# Step 4 is being rewritten in a parallel worktree. Unlike view_relations.py's
# gate runs (which pin a frozen fixture for comparability across runs), this
# runner's whole point is to show the user a codebook from the LATEST
# taxonomy — a moving target here is the intended behaviour, not a hazard.
WORKTREE_CACHE_DIR = (
    project_root / ".claude" / "worktrees" / "step4-herschrijven" / "data" / "cache"
)

CODEBOOK_OUTPUT = (
    project_root / ".superpowers" / "sdd" / "2026-08-12-step5-herbouw" / "codebook.md"
)

DIRECTION_SYMBOL = {"positive": "+", "negative": "−", "neutral": "neutraal"}


class _RoundLog:
    """Verzamelt `enforce_mece`'s per-ronde `log.add(...)`-aanroepen voor de
    printregel aan het eind van de run. Geen `decision_log.py` (nog niet
    gebouwd) — duck-typed, zoals `write_codebook`'s eigen `log`-parameter."""
    def __init__(self):
        self.rounds: List[dict] = []

    def add(self, **kwargs):
        self.rounds.append(kwargs)


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


def _shape_lookup(
    shapes: List[CodeShape], concept_by_id: Dict[str, Concept],
) -> Dict[Tuple[FrozenSet[str], str], CodeShape]:
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
    code: ConsolidatedCode, lookup: Dict[Tuple[FrozenSet[str], str], CodeShape],
) -> Optional[CodeShape]:
    return lookup.get((frozenset(code.source_attributes), code.valence))


def build_markdown(
    codes: List[ConsolidatedCode],
    shapes: List[CodeShape],
    concept_by_id: Dict[str, Concept],
    overig_ids: List[str],
    header: dict,
) -> str:
    lookup = _shape_lookup(shapes, concept_by_id)
    rows = []
    for code in codes:
        shape = _match_shape(code, lookup)
        n_resp = len(shape.resp_ids) if shape is not None else 0
        rows.append((n_resp, code.code_name, DIRECTION_SYMBOL.get(code.valence, code.valence),
                     code.definition, ", ".join(code.source_attributes)))

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
    parser.add_argument("--cache-dir", type=Path, default=WORKTREE_CACHE_DIR)
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
        get_dimension(dimension_name).criterion if dimension_name
        else "Do responses mainly differ in qualities, traits, images, or associations?"
    )

    units = build_idea_units(classified_ideas)
    refs = build_attribute_refs(taxonomy_cache.partition_results)
    concepts = build_inventory(units, refs)
    concept_by_id = {c.attribute_id: c for c in concepts}

    config = CodebookConfig()
    n_resp_total = len(classified_ideas)
    threshold = t_keep(n_resp_total, config)

    print(f"Attributen: {len(refs)} (concepten met idee: {len(concepts)})")
    print(f"T_keep = {threshold} over {n_resp_total} respondenten")
    print("LLM-calls worden uitgevoerd (relaties, consolidatie-namen, schrijven, "
          "MECE-afdwinging)...\n")

    async def _run():
        relations_before = await resolve_relations(concepts, config, language, verbose=True)
        umbrellas_before = umbrellas_from_relations(relations_before)
        merge_result = await resolve_umbrella_merge(umbrellas_before, config, verbose=True)
        relations_final = (
            apply_umbrella_merge(relations_before, merge_result)
            if merge_result is not None else relations_before
        )
        relation_map = normalize_relations(relations_final, concepts)
        shapes, overig_ids = consolidate(concepts, relation_map, threshold)
        codes = await write_codebook(
            shapes, concepts, dimension_diagnostic, language, config, verbose=True,
        )

        # MECE-afdwinging: codes als VERZAMELING bekijken, niet per vorm.
        # `code_by_name` bewaart de volledige geschreven tekst (incl.
        # diagnostic_test) van codes die geen enkele ronde aanraakt.
        shape_lookup = _shape_lookup(shapes, concept_by_id)
        code_by_name = {code.code_name: code for code in codes}
        candidates = [
            CodeCandidate(name=code.code_name, definition=code.definition,
                          indicators=tuple(code.typical_indicators), valence=code.valence,
                          shape=_match_shape(code, shape_lookup))
            for code in codes if _match_shape(code, shape_lookup) is not None
        ]
        round_log = _RoundLog()
        final_candidates = await enforce_mece(candidates, config, log=round_log, verbose=True)
        merged = [c for c in final_candidates if c.shape.origin == "mece_merge"]
        untouched = [c for c in final_candidates if c.shape.origin != "mece_merge"]

        # Alleen de samengevoegde codes krijgen nieuwe tekst — ongewijzigde
        # codes behouden hun eerder geschreven definitie/diagnostic_test.
        rewritten = await write_codebook(
            [c.shape for c in merged], concepts, dimension_diagnostic, language, config, verbose=True,
        ) if merged else []
        final_shapes = [c.shape for c in untouched] + [c.shape for c in merged]
        final_codes = [code_by_name[c.name] for c in untouched] + rewritten

        return final_shapes, overig_ids, final_codes, merge_result is None, round_log.rounds

    shapes, overig_ids, codes, merge_failed, mece_rounds = asyncio.run(_run())
    if merge_failed:
        print("WAARSCHUWING: consolidatiecall mislukt — doorgegaan met ongeconsolideerde namen.")

    total_merges = sum(r["merges"] for r in mece_rounds)
    if mece_rounds:
        rounds_desc = ", ".join(f"ronde {r['round']}: {r['merges']}" for r in mece_rounds)
        print(f"MECE: {len(mece_rounds)} ronde(s), {total_merges} samenvoeging(en) totaal ({rounds_desc})")

    lookup = _shape_lookup(shapes, concept_by_id)
    unmatched = [c.code_name for c in codes if _match_shape(c, lookup) is None]
    if unmatched:
        print(f"WAARSCHUWING: {len(unmatched)} code(s) niet aan hun vorm gekoppeld "
              f"voor het respondentenaantal: {unmatched}")

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
    markdown, overig_n_resp, overig_share = build_markdown(codes, shapes, concept_by_id, overig_ids, header)
    CODEBOOK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CODEBOOK_OUTPUT.write_text(markdown, encoding="utf-8")
    print(f"\nCodeboek geschreven: {CODEBOOK_OUTPUT}")

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
