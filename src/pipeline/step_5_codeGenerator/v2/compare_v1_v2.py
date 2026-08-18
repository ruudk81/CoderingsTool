"""Read-only: beide codeboeken op dezelfde taxonomie naast elkaar.

Schrijft niets. Laadt `mece_codes` en `mece_codes_v2` uit dezelfde cache en zet
de cijfers naast elkaar waarop v1 aantoonbaar faalde: aantal codes, hoeveel
daarvan één enkel attribuut zijn, en het valentieprofiel.

    python -m pipeline.step_5_codeGenerator.v2.compare_v1_v2
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from models import CodingResultsCache
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

from ..prompts_codeGenerator import ConsolidatedCode
from .run_codebook_v2 import CACHE_STEP

ROWS = [
    ("codes", "n_codes"),
    ("waarvan één attribuut", "n_solo"),
    ("positief", "n_positive"),
    ("negatief", "n_negative"),
    ("neutraal", "n_neutral"),
    ("attributen gedekt", "attributes_covered"),
]


def summarise(codes: List[ConsolidatedCode]) -> Dict[str, int]:
    return {
        "n_codes": len(codes),
        "n_positive": sum(1 for c in codes if c.valence == "positive"),
        "n_negative": sum(1 for c in codes if c.valence == "negative"),
        "n_neutral": sum(1 for c in codes if c.valence == "neutral"),
        "n_solo": sum(1 for c in codes if len(c.source_attributes) == 1),
        "attributes_covered": len({a for c in codes for a in c.source_attributes}),
    }


def format_comparison(v1: Dict[str, int], v2: Dict[str, int]) -> str:
    lines = [f"{'':<24}{'v1':>8}{'v2':>8}", "-" * 40]
    for label, key in ROWS:
        lines.append(f"{label:<24}{v1.get(key, 0):>8}{v2.get(key, 0):>8}")
    return "\n".join(lines)


def where_each_attribute_landed(
    v1_codes: List[ConsolidatedCode], v2_codes: List[ConsolidatedCode],
) -> List[Tuple[str, str, str]]:
    """Per attribuut: in welke code het in v1 zat en in welke in v2. Dit is de
    vergelijking die laat zien wat de consolidatie feitelijk heeft gedaan —
    de aantallen zeggen alleen dát er iets veranderde, niet wat."""
    def index(codes):
        return {a: c.code_name for c in codes for a in c.source_attributes}

    in_v1, in_v2 = index(v1_codes), index(v2_codes)
    return [(attribute, in_v1.get(attribute, "—"), in_v2.get(attribute, "—"))
            for attribute in sorted(set(in_v1) | set(in_v2))]


def _load(filename: str, step: str, variable_key: str) -> List[ConsolidatedCode]:
    cache = CacheManager().load_metadata_from_cache(
        filename, step, variable_key, CodingResultsCache)
    if cache is None:
        return []
    return [ConsolidatedCode(**d) for d in cache.raw_codes]


if __name__ == "__main__":
    from ..run_codeGenerator import FILENAME, SAMPLE_SIZE, VARIABLE

    key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE)
    v1_codes = _load(FILENAME, "mece_codes", key)
    v2_codes = _load(FILENAME, CACHE_STEP, key)
    if not v1_codes or not v2_codes:
        print("Beide codeboeken moeten in cache staan — draai v1 en v2 eerst.")
    else:
        print(format_comparison(summarise(v1_codes), summarise(v2_codes)))
        print(f"\n{'attribuut':<40}{'v1':<28}v2")
        print("-" * 96)
        for attribute, in_v1, in_v2 in where_each_attribute_landed(v1_codes, v2_codes):
            print(f"{attribute[:38]:<40}{in_v1[:26]:<28}{in_v2[:26]}")
