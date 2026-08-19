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
        lookup[(names, shape.valence)] = shape
    return lookup


def _match_shape(
    code: ConsolidatedCode, lookup: Dict[tuple, CodeShape],
) -> Optional[CodeShape]:
    return lookup.get((frozenset(code.source_attributes), code.valence))
