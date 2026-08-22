"""Wat een code van vorm is, en hoe je een geschreven code terugvindt bij zijn vorm.

`CodeShape` is het scharnier tussen de groeperingsfase en de schrijffase: de
groepering levert vormen op, de schrijver geeft er tekst bij. Omdat de schrijver
door een LLM-ronde gaat en geen vormidentiteit meedraagt, matcht `_match_shape`
op de twee dingen die de schrijver wél teruggeeft — de bronattribuutnamen en de
valentie.

Dit stond in `consolidator.py` en `run_codeGenerator.py`, twee modules die met
de v1-keten in `_quarantine_v1/` zijn beland. De vorm zelf is niet v1-specifiek:
De productieketen bouwt hem in `grouping.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .concept_inventory import Concept
from models import ConsolidatedCode

# `ConsolidatedCode.valence` kent drie waarden; `build_shapes(two_pole=True)`
# levert er een vierde. Die wordt vertaald in plaats van het model te
# verruimen — `models.py` is het contract over stapgrenzen. De vertaling hoort
# hier omdat BEIDE kanten van de vorm↔code-match hem moeten gebruiken: de
# sleutel die `_shape_lookup` legt en de valentie die de geschreven code
# draagt. Spreken ze niet dezelfde vocabulaire, dan vindt `_match_shape` niets
# terug.
def stored_valence(valence: str) -> str:
    """De valentie zoals een geschreven code hem draagt.

    Tot 2026-08-22 vertaalde deze functie `non_negative` naar `neutral`, omdat
    `ConsolidatedCode.valence` maar drie waarden kende. Dat was geen afronding
    maar betekenisverlies: `neutral` betekent "beschrijvend, geen richting" en
    `non_negative` betekent "expliciet geen klacht". Step 6's `opposes()` laat
    `neutral` bewust buiten zijn tabel — beschrijvend materiaal heeft geen
    tegenpool — en dus vuurde de richtingsbewaking nooit op een codeboek dat
    met twee polen was gemaakt. Precies het pad dat dagelijks draaide.

    Het contract kent nu vier waarden en er valt niets meer te vertalen. De
    functie blijft bestaan omdat `_shape_lookup` en `codebook_writer` hem aan
    weerszijden van de hermontage aanroepen: zolang beide kanten dezelfde
    functie gebruiken kunnen ze niet uit elkaar lopen, ook niet als hier ooit
    weer een vertaling bij komt.
    """
    return valence


@dataclass(frozen=True)
class CodeShape:
    """One outcome of consolidation: how many codes, which members, which
    direction. `key` is a run-local ordering key, not a codebook id."""
    key: str
    members: Tuple[str, ...]
    valence: str
    umbrella: str
    resp_ids: frozenset[str]
    resp_pos: frozenset[str]
    resp_neg: frozenset[str]
    resp_neu: frozenset[str]
    origin: str


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
        lookup[(names, stored_valence(shape.valence))] = shape
    return lookup


def _match_shape(
    code: ConsolidatedCode, lookup: Dict[tuple, CodeShape],
) -> Optional[CodeShape]:
    return lookup.get((frozenset(code.source_attributes), code.valence))
