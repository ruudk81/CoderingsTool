"""Read-only: beide codeboeken op dezelfde taxonomie naast elkaar.

Schrijft niets. Laadt `mece_codes` en `mece_codes_v2` uit dezelfde cache en zet
de cijfers naast elkaar waarop v1 aantoonbaar faalde: aantal codes,
grootteverdeling, valentieprofiel, Overig-aandeel, en per attribuut waar het in
beide terechtkwam.

`build_scorecard` (`codebook_verifier.py`, v1) draait op elke kant apart — het
is de bron voor Overig-aandeel en mini-codes; deze module leest dat resultaat,
het wijzigt geen v1-code. Beide cacherijen dragen ook `partition_results`, dus
er is geen aparte taxonomie-load nodig.

    python -m pipeline.step_5_codeGenerator.v2.compare_v1_v2
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config import MISCELLANEOUS_CODE_LABELS
from models import CodingResultsCache
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

from ..codebook_verifier import build_scorecard, collect_attribute_valence
from ..prompts_codeGenerator import ConsolidatedCode
from .run_codebook_v2 import CACHE_STEP

ROWS = [
    ("codes", "n_codes"),
    ("waarvan één attribuut", "n_solo"),
    ("positief", "n_positive"),
    ("negatief", "n_negative"),
    ("neutraal", "n_neutral"),
    ("Overig-aandeel (%)", "overig_share_pct"),
    ("mini-codes (< bevolkingsvloer)", "n_mini_codes"),
    ("kleinste code (resp.)", "min_code_size"),
    ("mediaan codegrootte (resp.)", "median_code_size"),
    ("grootste code (resp.)", "max_code_size"),
]

_OVERIG_NAMES = {v.strip().lower() for v in MISCELLANEOUS_CODE_LABELS.values()} | {"overig"}


def find_overig_code_name(codes: List[ConsolidatedCode]) -> Optional[str]:
    """Vindt de catch-all-code op naam, taalonafhankelijk — dezelfde opzoeking
    als step 6 (`code_assignment.py:557`); de vergelijking kent de taal van de
    run niet, alleen de gecachete codes."""
    for code in codes:
        if (code.code_name or "").strip().lower() in _OVERIG_NAMES:
            return code.code_name
    return None


def code_sizes(
    codes: List[ConsolidatedCode], partition_results: Dict[str, Any],
) -> Dict[str, int]:
    """Verwachte ideevolume per code — de bijpassende-valentiepool van zijn
    bronattributen, dezelfde afleiding als `codebook_verifier`'s mini-code-
    toets, maar voor elke code, niet alleen die onder de vloer. Dit is de
    grootteverdeling: een staart van 12-respondent-codes naast een kop van
    398 wordt hier zichtbaar."""
    attr_valence = collect_attribute_valence(partition_results)
    sizes: Dict[str, int] = {}
    for code in codes:
        total = 0
        for attr in code.source_attributes or []:
            counts = attr_valence.get(attr, {})
            if code.valence == "positive":
                total += counts.get("positive", 0)
            elif code.valence == "negative":
                total += counts.get("negative", 0)
            else:
                total += (counts.get("positive", 0) + counts.get("neutral", 0)
                          + counts.get("negative", 0))
        sizes[code.code_name] = total
    return sizes


def _median(values: List[int]) -> int:
    if not values:
        return 0
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) // 2


def summarise(
    codes: List[ConsolidatedCode], partition_results: Dict[str, Any],
) -> Dict[str, float]:
    overig_name = find_overig_code_name(codes)
    scorecard = build_scorecard(codes, partition_results, overig_name)
    sizes = sorted(code_sizes(codes, partition_results).values())
    return {
        "n_codes": len(codes),
        "n_positive": sum(1 for c in codes if c.valence == "positive"),
        "n_negative": sum(1 for c in codes if c.valence == "negative"),
        "n_neutral": sum(1 for c in codes if c.valence == "neutral"),
        "n_solo": sum(1 for c in codes if len(c.source_attributes) == 1),
        "overig_share_pct": scorecard.overig_idea_share_pct,
        "n_mini_codes": len(scorecard.mini_codes),
        "min_code_size": sizes[0] if sizes else 0,
        "median_code_size": _median(sizes),
        "max_code_size": sizes[-1] if sizes else 0,
    }


def format_comparison(v1: Dict[str, float], v2: Dict[str, float]) -> str:
    lines = [f"{'':<32}{'v1':>8}{'v2':>8}", "-" * 48]
    for label, key in ROWS:
        lines.append(f"{label:<32}{v1.get(key, 0):>8}{v2.get(key, 0):>8}")
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


def _load(
    filename: str, step: str, variable_key: str,
) -> Optional[Tuple[List[ConsolidatedCode], Dict[str, Any]]]:
    cache = CacheManager().load_metadata_from_cache(
        filename, step, variable_key, CodingResultsCache)
    if cache is None:
        return None
    codes = [ConsolidatedCode(**d) for d in cache.raw_codes]
    return codes, cache.partition_results


if __name__ == "__main__":
    from ..run_codeGenerator import FILENAME, SAMPLE_SIZE, VARIABLE

    key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE)
    v1 = _load(FILENAME, "mece_codes", key)
    v2 = _load(FILENAME, CACHE_STEP, key)
    if v1 is None or v2 is None:
        print("Beide codeboeken moeten in cache staan — draai v1 en v2 eerst.")
    else:
        v1_codes, v1_results = v1
        v2_codes, v2_results = v2
        print(format_comparison(summarise(v1_codes, v1_results), summarise(v2_codes, v2_results)))
        print(f"\n{'attribuut':<40}{'v1':<28}v2")
        print("-" * 96)
        for attribute, in_v1, in_v2 in where_each_attribute_landed(v1_codes, v2_codes):
            print(f"{attribute[:38]:<40}{in_v1[:26]:<28}{in_v2[:26]}")
