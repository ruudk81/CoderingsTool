"""Stap 5 — MECE-afdwinging over de codeverzameling. De enige plek in step 5
die codes als VERZAMELING bekijkt in plaats van per attribuut of per vorm.

De operationele toets (het hele ontwerp): twee codes zijn ÉÉN dimensie als je
geen regel kunt formuleren die een gegeven idee aan precies één van de twee
toewijst. Dat toetst Pass B, ná een gedwongen per-code opzoeking in Pass A
(zie `prompts_mece.py` voor waarom een groepeervraag hier niet werkt).

Samenvoegen is hierna volledig deterministisch: alleen dezelfde richting
(een positieve en een negatieve code zijn door hun richting alleen al
onderscheiden — dat is geen overlap, dat is het richtingsonderscheid zelf),
componenten via union-find (een keten A-B-C wordt één groep), en een
VERENIGING van leden en respondentverzamelingen — nooit een som.

Samenvoegen verandert de verzameling, dus een latere ronde kan overlap
zichtbaar maken die een eerdere ronde nog niet kon zien. Daarom itereert
`enforce_mece` pass A + pass B tot een ronde niets meer samenvoegt, met een
plafond (`config.mece_max_rounds`)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from .config_codeGenerator import CodebookConfig
from .consolidator import CodeShape
from .prompts_mece import (
    CandidatePair, CodeCandidate, OverlapDetectionResult, OverlapVerdict,
    PairAdjudicationResult, PairVerdict, build_overlap_prompt, build_pair_prompt,
    make_overlap_model, make_pair_model,
)
from .prompts_relations import _shuffled

DETECT_PHASE = "step5_mece_detect"
ADJUDICATE_PHASE = "step5_mece_adjudicate"


# ---------------------------------------------------------------------------
# Dispatch — Pass A en Pass B via SmoothRequester
# ---------------------------------------------------------------------------

async def resolve_overlap_detection(
    candidates: List[CodeCandidate],
    config: CodebookConfig,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
) -> Optional[OverlapDetectionResult]:
    """One call across the current code set. A failed call means this round
    finds no candidates — not a hard stop for the codebook (same contract as
    `resolve_umbrella_merge`, not `resolve_relations`: a missed MECE round
    gives a finer-grained codebook, not a broken one)."""

    def prepare_fn(task):
        return {
            "prompt": build_overlap_prompt(task["candidates"]),
            "response_model": make_overlap_model(task["candidates"]),
            "temperature": config.temperature_mece_detect,
            "max_tokens": config.max_tokens_mece_detect,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(config.model_mece_detect, phase="codegen_mece_detect"),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = SmoothRequester(
        model=config.model_mece_detect, phase_key=DETECT_PHASE, num_tasks=1,
        verbose=verbose, known_limits=known_limits, has_server_headers=has_server_headers,
        quiet=True,
    )
    results = await requester.process_all(
        [{"candidates": candidates}], prepare_fn, parse_fn, fallback_fn
    )
    return results[0] if results else None


async def resolve_pair_adjudication(
    pairs: List[CandidatePair],
    candidate_by_name: Dict[str, CodeCandidate],
    config: CodebookConfig,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
) -> Optional[PairAdjudicationResult]:
    """One call across the round's candidate pairs. Same no-hard-stop contract
    as `resolve_overlap_detection`."""

    def prepare_fn(task):
        return {
            "prompt": build_pair_prompt(task["pairs"], task["candidate_by_name"]),
            "response_model": make_pair_model(task["pairs"]),
            "temperature": config.temperature_mece_adjudicate,
            "max_tokens": config.max_tokens_mece_adjudicate,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(
                config.model_mece_adjudicate, phase="codegen_mece_adjudicate"
            ),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = SmoothRequester(
        model=config.model_mece_adjudicate, phase_key=ADJUDICATE_PHASE, num_tasks=1,
        verbose=verbose, known_limits=known_limits, has_server_headers=has_server_headers,
        quiet=True,
    )
    results = await requester.process_all(
        [{"pairs": pairs, "candidate_by_name": candidate_by_name}], prepare_fn, parse_fn, fallback_fn
    )
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Deterministisch — kandidaat-paren, componenten, samenvoegen
# ---------------------------------------------------------------------------

def build_candidate_pairs(
    verdicts: List[OverlapVerdict], valence_by_name: Dict[str, str]
) -> List[CandidatePair]:
    """Unieke, ongeordende, zelfde-richting paren uit Pass A. Een paar dat het
    model over een richtingsgrens heen voorstelt, of een code die zichzelf
    voorstelt, wordt hier verwijderd — deterministisch, ongeacht wat het model
    zei: een positieve en een negatieve code zijn door hun richting alleen al
    onderscheiden, nooit een samenvoegkandidaat. `pair_id` volgt de
    `_shuffled`-volgorde, zodat promptvolgorde en de `Literal`-enum in het
    responsemodel altijd gelijk lopen."""
    seen: Set[frozenset] = set()
    for verdict in verdicts:
        other = verdict.hardest_to_separate_from
        if other is None or other == verdict.code:
            continue
        if valence_by_name.get(verdict.code) != valence_by_name.get(other):
            continue
        seen.add(frozenset((verdict.code, other)))

    # Sorted here (not left to set-iteration order, which varies with Python's
    # per-process string hash randomization): _shuffled's own hash reorders
    # deterministically, but only if it starts from a deterministic sequence.
    unordered = sorted(tuple(sorted(pair)) for pair in seen)
    wrapped = [CandidatePair(pair_id=0, code_a=a, code_b=b) for a, b in unordered]
    return [CandidatePair(pair_id=i, code_a=p.code_a, code_b=p.code_b)
            for i, p in enumerate(_shuffled(wrapped), start=1)]


def merge_components(
    pair_by_id: Dict[int, CandidatePair], verdicts: List[PairVerdict]
) -> List[Set[str]]:
    """Samenhangende componenten (union-find) over `one_dimension=True`
    verdicts: een keten A-B, B-C wordt één component, ongeacht de volgorde
    waarin de paren zijn beoordeeld. Alleen componenten van meer dan één lid
    zijn een echte samenvoeging."""
    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for verdict in verdicts:
        pair = pair_by_id.get(verdict.pair_id)
        if pair is None:
            continue
        find(pair.code_a)
        find(pair.code_b)
        if verdict.one_dimension:
            union(pair.code_a, pair.code_b)

    groups: Dict[str, Set[str]] = defaultdict(set)
    for name in parent:
        groups[find(name)].add(name)
    return [group for group in groups.values() if len(group) > 1]


def _canonical_name(group: Set[str], candidate_by_name: Dict[str, CodeCandidate]) -> str:
    """Het lid met de meeste bron-attributen wint (een inhoudelijke telling,
    berekend in code — nooit aan het model getoond); bij gelijke stand de
    kortste naam, dan alfabetisch."""
    return min(group, key=lambda name: (
        -len(candidate_by_name[name].shape.members), len(name), name
    ))


def merge_candidates(
    group: Set[str], candidate_by_name: Dict[str, CodeCandidate], key: str
) -> CodeCandidate:
    """Verenigt leden-ids ÉN respondentverzamelingen over de groep — nooit een
    som (`CodeShape` draagt `frozenset`s precies hiervoor). Weigert een groep
    die niet allemaal dezelfde richting deelt: dat zou het richtingsonderscheid
    zelf tegenspreken, en mag hier nooit binnenkomen (de deterministische
    filter in `build_candidate_pairs` voorkomt het al bovenstrooms — dit is
    het vangnet)."""
    shapes = [candidate_by_name[name].shape for name in group]
    valences = {s.valence for s in shapes}
    if len(valences) > 1:
        raise ValueError(f"cannot merge across valence: {sorted(valences)}")
    valence = shapes[0].valence

    members = tuple(sorted({id_ for shape in shapes for id_ in shape.members}))
    resp_pos = frozenset().union(*(s.resp_pos for s in shapes))
    resp_neg = frozenset().union(*(s.resp_neg for s in shapes))
    resp_neu = frozenset().union(*(s.resp_neu for s in shapes))

    canonical = _canonical_name(group, candidate_by_name)
    canonical_candidate = candidate_by_name[canonical]
    merged_shape = CodeShape(
        key=key, members=members, valence=valence, umbrella=canonical_candidate.shape.umbrella,
        resp_ids=resp_pos | resp_neg | resp_neu, resp_pos=resp_pos, resp_neg=resp_neg,
        resp_neu=resp_neu, origin="mece_merge",
    )
    definition = " / ".join(sorted(candidate_by_name[name].definition for name in group))
    indicators = tuple(sorted({i for name in group for i in candidate_by_name[name].indicators}))
    return CodeCandidate(
        name=canonical_candidate.name, definition=definition,
        indicators=indicators, valence=valence, shape=merged_shape,
    )


def apply_merges(candidates: List[CodeCandidate], components: List[Set[str]]) -> List[CodeCandidate]:
    """Vervangt elke component door één samengevoegde kandidaat; codes die in
    geen enkele component zitten blijven ongewijzigd (zelfde object)."""
    candidate_by_name = {c.name: c for c in candidates}
    merged_names: Set[str] = set().union(*components) if components else set()
    merged = [merge_candidates(group, candidate_by_name, key=f"MECE{i}")
              for i, group in enumerate(components, start=1)]
    untouched = [c for c in candidates if c.name not in merged_names]
    return merged + untouched


def _log_round(log, round_num: int, merge_count: int) -> None:
    if log is None:
        return
    log.add(action="MECE_ROUND", round=round_num, merges=merge_count)


# ---------------------------------------------------------------------------
# Orkestratie — itereer pass A + pass B tot een ronde niets samenvoegt
# ---------------------------------------------------------------------------

async def enforce_mece(
    candidates: List[CodeCandidate],
    config: CodebookConfig,
    log=None,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
) -> List[CodeCandidate]:
    """Detecteer (Pass A) + adjudiceer (Pass B), herhaald tot een ronde niets
    samenvoegt of `config.mece_max_rounds` is bereikt. Geeft de finale
    kandidaten terug: samengevoegde codes hebben `shape.origin ==
    "mece_merge"` en dragen nog placeholder-tekst — de aanroeper herschrijft
    de tekst alleen voor die codes (`write_codebook` op hun `shape`)."""
    current = list(candidates)
    if len(current) < 2:
        return current

    for round_num in range(1, config.mece_max_rounds + 1):
        overlap = await resolve_overlap_detection(
            current, config, known_limits, has_server_headers, verbose
        )
        if overlap is None:
            _log_round(log, round_num, 0)
            break

        valence_by_name = {c.name: c.valence for c in current}
        pairs = build_candidate_pairs(overlap.verdicts, valence_by_name)
        if not pairs:
            _log_round(log, round_num, 0)
            break

        candidate_by_name = {c.name: c for c in current}
        adjudication = await resolve_pair_adjudication(
            pairs, candidate_by_name, config, known_limits, has_server_headers, verbose
        )
        if adjudication is None:
            _log_round(log, round_num, 0)
            break

        pair_by_id = {p.pair_id: p for p in pairs}
        components = merge_components(pair_by_id, adjudication.verdicts)
        if not components:
            _log_round(log, round_num, 0)
            break

        current = apply_merges(current, components)
        _log_round(log, round_num, len(components))
        if len(current) < 2:
            break

    return current
