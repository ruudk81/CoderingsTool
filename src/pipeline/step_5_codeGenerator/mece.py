"""Stap 5 — MECE-afdwinging over de codeverzameling. De enige plek in step 5
die codes als VERZAMELING bekijkt in plaats van per attribuut of per vorm.

De operationele toets (het hele ontwerp, bronspecificatie §4.8): twee codes
zijn ÉÉN dimensie als een blinde toewijzingsproef op ECHTE ideeën ze niet
betrouwbaar uit elkaar houdt, ÓF als ideeën uit die proef er structureel
allebei genuinely bij horen. Pass A (opzoeking) levert de kandidaat-paren;
Pass B (deze proef) meet ze op twee manieren, niet één — zie `prompts_mece.py`
voor waarom een binaire toewijzingsproef (91% gemiddelde accuracy, nul
samenvoegingen op een live run met vier codes over duurzaamheid, vier over
persoonlijk contact) het verkeerde meet: scheidbaarheid is geen
orthogonaliteit. De nuloptie is hierna samenvoegen, niet apart houden
(bronspecificatie §2.5, compressievoorkeur) — een paar houdt zich alleen apart
als de proef dat aantoont, niet omgekeerd.

Een nóg eerdere vorm van Pass B liet het model een scheidingsregel SCHRIJVEN
en daarna zelf beoordelen of die regel echt was — dat kan niet werken, want
een model dat gevraagd wordt een regel te schrijven schrijft er altijd één,
en concludeert daarna dat de codes scheidbaar zijn. Het genereren van een
verantwoording is geen toets zolang het model zelf bepaalt of hij slaagt; de
score van een blinde proef op data die het model niet zelf heeft
geproduceerd, is dat wel.

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

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from .config_codeGenerator import CodebookConfig
from .consolidator import CodeShape
from .prompts_mece import (
    CandidatePair, CodeCandidate, IdeaAssignment, OverlapDetectionResult, OverlapVerdict,
    ProbeIdea, ProbeResult, build_overlap_prompt, build_probe_prompt, make_overlap_model,
    make_probe_model,
)
from .prompts_relations import _shuffled
from .taxonomy_input import IdeaUnit

DETECT_PHASE = "step5_mece_detect"
PROBE_PHASE = "step5_mece_probe"

_KeyedIdea = namedtuple("_KeyedIdea", ["attribute_id", "unit"])


# ---------------------------------------------------------------------------
# Dispatch — Pass A en Pass B via SmoothRequester
# ---------------------------------------------------------------------------

async def resolve_overlap_detection(
    candidates: List[CodeCandidate],
    config: CodebookConfig,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
    prompt_printer=None,
) -> Optional[OverlapDetectionResult]:
    """One call across the current code set. A failed call means this round
    finds no candidates — not a hard stop for the codebook (same contract as
    `resolve_umbrella_merge`, not `resolve_relations`: a missed MECE round
    gives a finer-grained codebook, not a broken one)."""

    def prepare_fn(task):
        prompt = build_overlap_prompt(task["candidates"])
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator",
                utility_name="resolve_overlap_detection",
                prompt_content=prompt,
                prompt_type="mece_detect",
                metadata={
                    "model": config.model_mece_detect,
                    "temperature": config.temperature_mece_detect,
                    "max_tokens": config.max_tokens_mece_detect,
                    "n_candidates": len(task["candidates"]),
                    "candidate_names": [c.name for c in task["candidates"]],
                },
            )
        return {
            "prompt": prompt,
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


@dataclass(frozen=True)
class PairVerdict:
    """De uitkomst van één blinde proef: run-lokaal, nooit een LLM-
    responsemodel. `one_dimension` volgt uit `accuracy` en `both_rate` in
    Python (`is_one_dimension`) — het model claimt hier niets over zichzelf."""
    pair_id: int
    accuracy: float
    both_rate: float
    one_dimension: bool


@dataclass(frozen=True)
class PairProbe:
    """Alles om één paar te bevragen én te scoren. `truth` (idea_ref -> echte
    codenaam) is de antwoordsleutel — gebouwd in Python, nooit getoond aan het
    model (zie `prompts_mece.build_probe_prompt`, dat alleen `ideas` krijgt)."""
    pair: CandidatePair
    ideas: List[ProbeIdea]
    truth: Dict[int, str]


def _shuffled_ideas(units: List[IdeaUnit]) -> List[IdeaUnit]:
    """`units` in de gedeelde deterministische volgorde, gekeyd op elk ideetje
    zijn eigen `idea_id` — niet op de gedeelde attribuut-id (die zou ideeën
    uit hetzelfde attribuut ongesorteerd laten) en niet op invoervolgorde."""
    keyed = [_KeyedIdea(attribute_id=unit.idea_id, unit=unit) for unit in units]
    return [entry.unit for entry in _shuffled(keyed)]


def _idea_text(unit: IdeaUnit) -> str:
    if unit.interpretation and unit.interpretation != unit.instance:
        return f"{unit.instance} ({unit.interpretation})"
    return unit.instance


def _idea_pool(
    candidate: CodeCandidate, idea_units_by_attribute: Dict[str, List[IdeaUnit]], n: int,
) -> List[IdeaUnit]:
    """Tot `n` ideeën van deze code, gepoold over al haar bronattributen (ook
    na een eerdere MECE-samenvoeging: `shape.members` draagt dan de vereniging
    van de oorspronkelijke attribuut-ids) en deterministisch bemonsterd."""
    pooled = [unit for member_id in candidate.shape.members
              for unit in idea_units_by_attribute.get(member_id, [])]
    return _shuffled_ideas(pooled)[:n]


def build_pair_probe(
    pair: CandidatePair,
    candidate_by_name: Dict[str, CodeCandidate],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    ideas_per_code: int,
) -> Optional[PairProbe]:
    """Bouwt de gepoolde, geschudde ideeënlijst en de antwoordsleutel voor één
    paar. `None` als een van beide kanten geen enkel idee heeft — een proef
    zonder materiaal aan één kant meet niets, dus dat paar wordt overgeslagen
    (niet samengevoegd: hetzelfde geen-hard-stop-contract als een mislukte
    call)."""
    a = _idea_pool(candidate_by_name[pair.code_a], idea_units_by_attribute, ideas_per_code)
    b = _idea_pool(candidate_by_name[pair.code_b], idea_units_by_attribute, ideas_per_code)
    if not a or not b:
        return None

    pooled = [(unit, pair.code_a) for unit in a] + [(unit, pair.code_b) for unit in b]
    keyed = [_KeyedIdea(attribute_id=unit.idea_id, unit=(unit, code)) for unit, code in pooled]
    shuffled_pool = [entry.unit for entry in _shuffled(keyed)]

    ideas: List[ProbeIdea] = []
    truth: Dict[int, str] = {}
    for i, (unit, code) in enumerate(shuffled_pool, start=1):
        ideas.append(ProbeIdea(idea_ref=i, text=_idea_text(unit)))
        truth[i] = code
    return PairProbe(pair=pair, ideas=ideas, truth=truth)


@dataclass(frozen=True)
class ProbeScore:
    """De twee deterministische signalen uit één blinde proef — nooit een
    claim van het model over zichzelf, altijd berekend tegen `PairProbe.truth`
    in Python."""
    accuracy: float
    both_rate: float


def score_probe(assignments: List[IdeaAssignment], truth: Dict[int, str]) -> ProbeScore:
    """`accuracy` — van de ideeën die het model op één kant zette (A of B,
    dus NIET "BOTH"), het aandeel dat op de kant kwam waar het idee
    werkelijk vandaan komt. Een idee zonder toewijzing, of met "BOTH", telt
    niet mee in die teller: het is niet "op een kant gezet". `both_rate` —
    het aandeel van ALLE bevraagde ideeën (heel `truth`) dat "BOTH" kreeg,
    ongeacht welke kant dat idee werkelijk vandaan kwam. Een dubbele
    toewijzing voor eenzelfde `idea_ref` laat het laatste antwoord winnen.
    Lege `truth` (kan niet via `build_pair_probe`, wel direct getest) geeft
    beide op 0.0, niet een deling door nul."""
    if not truth:
        return ProbeScore(accuracy=0.0, both_rate=0.0)
    assigned = {a.idea_ref: a.assigned_to for a in assignments}
    sided = {ref: code for ref, code in assigned.items() if ref in truth and code != "BOTH"}
    accuracy = (sum(1 for ref, code in sided.items() if code == truth[ref]) / len(sided)
                if sided else 0.0)
    both_rate = sum(1 for ref in truth if assigned.get(ref) == "BOTH") / len(truth)
    return ProbeScore(accuracy=accuracy, both_rate=both_rate)


def is_one_dimension(
    accuracy: float, both_rate: float, accuracy_threshold: float, both_rate_threshold: float,
) -> bool:
    """Samenvoegen (`True`) wanneer scheidbaarheid ontbreekt (`accuracy` op of
    onder zijn drempel) ÓF wanneer ideeën structureel bij beide horen
    (`both_rate` op of boven zijn drempel) — de twee manieren waarop een paar
    kan mislukken in "aantoonbaar apart blijven": onscheidbaar, of scheidbaar-
    maar-beide-passen-echt. Kansniveau bij een binaire keuze (`accuracy`,
    exclusief "BOTH") is 0.50; de standaarddrempel (0.70) ligt daarboven, dus
    'geen beter dan raden' merget ruimschoots mee. De nuloptie is samenvoegen,
    niet apart houden (bronspecificatie §2.5) — dit is dan ook een OF, geen
    EN: één van de twee signalen is genoeg."""
    return accuracy <= accuracy_threshold or both_rate >= both_rate_threshold


async def resolve_pair_probes(
    pairs: List[CandidatePair],
    candidate_by_name: Dict[str, CodeCandidate],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    config: CodebookConfig,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
    prompt_printer=None,
) -> Dict[int, PairVerdict]:
    """Eén taak per paar (pairs zijn onafhankelijk), gebundeld in één
    SmoothRequester-batch voor concurrency. Een paar zonder materiaal
    (`build_pair_probe` gaf `None`) of met een mislukte call krijgt gewoon
    geen entry — hetzelfde geen-hard-stop-contract als elders in deze
    module."""
    probes = [p for p in (build_pair_probe(pair, candidate_by_name, idea_units_by_attribute,
                                            config.mece_probe_ideas_per_code)
                          for pair in pairs) if p is not None]
    if not probes:
        return {}

    def prepare_fn(task):
        probe: PairProbe = task["probe"]
        prompt = build_probe_prompt(probe.pair, candidate_by_name, probe.ideas)
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator",
                utility_name="resolve_pair_probes",
                prompt_content=prompt,
                prompt_type="mece_probe",
                metadata={
                    "model": config.model_mece_probe,
                    "temperature": config.temperature_mece_probe,
                    "max_tokens": config.max_tokens_mece_probe,
                    "pair_id": probe.pair.pair_id,
                    "code_a": probe.pair.code_a,
                    "code_b": probe.pair.code_b,
                    "idea_refs": [idea.idea_ref for idea in probe.ideas],
                },
            )
        return {
            "prompt": prompt,
            "response_model": make_probe_model(probe.pair, probe.ideas),
            "temperature": config.temperature_mece_probe,
            "max_tokens": config.max_tokens_mece_probe,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(config.model_mece_probe, phase="codegen_mece_probe"),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = SmoothRequester(
        model=config.model_mece_probe, phase_key=PROBE_PHASE, num_tasks=len(probes),
        verbose=verbose, known_limits=known_limits, has_server_headers=has_server_headers,
        quiet=True,
    )
    # SmoothRequester.process_all eist List[Dict] — een lijst objecten laat
    # _execute_task struikelen op task.get(), en de foutafhandeling daarna
    # nog eens, waardoor de echte oorzaak onzichtbaar wordt.
    results: List[Optional[ProbeResult]] = await requester.process_all(
        [{"probe": probe} for probe in probes], prepare_fn, parse_fn, fallback_fn
    )

    verdicts: Dict[int, PairVerdict] = {}
    for probe, result in zip(probes, results):
        if result is None:
            continue
        score = score_probe(result.assignments, probe.truth)
        verdicts[probe.pair.pair_id] = PairVerdict(
            pair_id=probe.pair.pair_id, accuracy=score.accuracy, both_rate=score.both_rate,
            one_dimension=is_one_dimension(
                score.accuracy, score.both_rate,
                config.mece_separability_threshold, config.mece_both_rate_threshold,
            ),
        )
    return verdicts


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
    geen enkele component zitten blijven ongewijzigd (zelfde object).

    `candidate_by_name` is onvermijdelijk naam-gekeyd: de componenten komen uit
    Pass A/B, die allebei in het naam-domein van het model werken (een prompt
    kan een code alleen bij de naam noemen die het model te zien kreeg — er is
    geen `shape.key` dat het model kent om mee te disambigueren). Deelt twee
    kandidaten toevallig een naam, dan is de opzoeking zelf al dubbelzinnig, en
    valt hier niets aan te repareren. Wat wél te repareren is: welke van de
    twee bij een samenvoeging hoorde mag niet ook zijn naamgenoot meesleuren.
    `untouched` filterde vroeger op naam (`c.name not in merged_names`) — als
    twee kandidaten dezelfde naam droegen en er ÉÉN daarvan werd samengevoegd,
    verdwenen ALLEBEI uit `untouched` terwijl alleen degene die de dict-opzoeking
    daadwerkelijk opleverde is samengevoegd. De ander — een compleet andere
    vorm die toevallig dezelfde naam draagt — werd dan stilzwijgend uit het
    codeboek verwijderd, erger dan een foute samenvoeging. `consumed_keys`
    volgt daarom welk fysiek object (`shape.key`, uniek per vorm) de opzoeking
    echt gebruikte, dus een naamgenoot die niet zelf is opgezocht blijft
    behouden."""
    candidate_by_name = {c.name: c for c in candidates}
    merged = [merge_candidates(group, candidate_by_name, key=f"MECE{i}")
              for i, group in enumerate(components, start=1)]
    consumed_keys = {candidate_by_name[name].shape.key for group in components for name in group}
    untouched = [c for c in candidates if c.shape.key not in consumed_keys]
    return merged + untouched


def _mean_accuracy(verdicts: Dict[int, PairVerdict]) -> Optional[float]:
    if not verdicts:
        return None
    return sum(v.accuracy for v in verdicts.values()) / len(verdicts)


def _pair_details(
    pair_by_id: Dict[int, CandidatePair], verdict_by_id: Dict[int, PairVerdict],
) -> List[dict]:
    """Eén entry per bevraagd paar, gesorteerd op `pair_id` voor een
    reproduceerbare printvolgorde — de gegevens voor de per-paar logregel
    (de twee codenamen, `accuracy`, `both_rate`, de samenvoegbeslissing)."""
    return [
        {"code_a": pair_by_id[pair_id].code_a, "code_b": pair_by_id[pair_id].code_b,
         "accuracy": verdict.accuracy, "both_rate": verdict.both_rate,
         "merged": verdict.one_dimension}
        for pair_id, verdict in sorted(verdict_by_id.items())
    ]


def _log_round(
    log, round_num: int, *, pairs_found: int = 0, pairs_probed: int = 0,
    mean_accuracy: Optional[float] = None, merges: int = 0, reason: Optional[str] = None,
    pairs: Optional[List[dict]] = None,
) -> None:
    """Elke afsluitreden is een apart veld — nooit alleen `merges=0`. Een
    call die crasht (`reason="detection_failed"` etc.) en een ronde die
    echt niets vond (`reason=None`, `pairs_probed > 0`) loggen allebei
    `merges=0`, maar zijn hierna nooit meer van elkaar te onderscheiden op
    dat getal alleen — dat was precies het defect dat dit verving. `pairs`
    (leeg tenzij Pass B daadwerkelijk paren scoorde) draagt de per-paar
    uitsplitsing (`_pair_details`) voor de printregel in de aanroeper."""
    if log is None:
        return
    log.add(action="MECE_ROUND", round=round_num, pairs_found=pairs_found,
            pairs_probed=pairs_probed, mean_accuracy=mean_accuracy, merges=merges, reason=reason,
            pairs=pairs or [])


# ---------------------------------------------------------------------------
# Orkestratie — itereer pass A + pass B tot een ronde niets samenvoegt
# ---------------------------------------------------------------------------

async def enforce_mece(
    candidates: List[CodeCandidate],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    config: CodebookConfig,
    log=None,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
    prompt_printer=None,
) -> List[CodeCandidate]:
    """Detecteer (Pass A) + bevraag blind (Pass B), herhaald tot een ronde
    niets samenvoegt of `config.mece_max_rounds` is bereikt. Geeft de finale
    kandidaten terug: samengevoegde codes hebben `shape.origin ==
    "mece_merge"` en dragen nog placeholder-tekst — de aanroeper herschrijft
    de tekst alleen voor die codes (`write_codebook` op hun `shape`).
    `idea_units_by_attribute` (attribuut-id -> zijn ideeën) levert het
    materiaal voor de blinde proef; zie `taxonomy_input.build_idea_units`."""
    current = list(candidates)
    if len(current) < 2:
        return current

    for round_num in range(1, config.mece_max_rounds + 1):
        overlap = await resolve_overlap_detection(
            current, config, known_limits, has_server_headers, verbose, prompt_printer,
        )
        if overlap is None:
            _log_round(log, round_num, reason="detection_failed")
            break

        valence_by_name = {c.name: c.valence for c in current}
        pairs = build_candidate_pairs(overlap.verdicts, valence_by_name)
        if not pairs:
            _log_round(log, round_num, reason="no_pairs")
            break

        candidate_by_name = {c.name: c for c in current}
        verdict_by_id = await resolve_pair_probes(
            pairs, candidate_by_name, idea_units_by_attribute, config,
            known_limits, has_server_headers, verbose, prompt_printer,
        )
        if not verdict_by_id:
            _log_round(log, round_num, pairs_found=len(pairs), reason="probe_failed")
            break

        pair_by_id = {p.pair_id: p for p in pairs}
        components = merge_components(pair_by_id, list(verdict_by_id.values()))
        round_stats = dict(pairs_found=len(pairs), pairs_probed=len(verdict_by_id),
                           mean_accuracy=_mean_accuracy(verdict_by_id),
                           pairs=_pair_details(pair_by_id, verdict_by_id))
        if not components:
            _log_round(log, round_num, **round_stats, reason="no_components")
            break

        current = apply_merges(current, components)
        _log_round(log, round_num, **round_stats, merges=len(components))
        if len(current) < 2:
            break

    return current
