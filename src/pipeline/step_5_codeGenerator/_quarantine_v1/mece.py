"""Step 5 — MECE enforcement across the code set. The only place in step 5 that
looks at codes as a SET rather than per attribute or per shape.

The operational test (the whole design, source specification §4.8): two codes are
ONE dimension when a blind assignment probe on REAL ideas cannot tell them apart
reliably, OR when ideas from that probe structurally belong to both. Pass A (the
lookup) yields the candidate pairs; Pass B (this probe) measures them in two ways,
not one — see `prompts_mece.py` for why a binary assignment probe (91% average
accuracy, zero merges on a live run that kept four codes on one theme and four on
another) measures the wrong thing: separability is not orthogonality. The null
option from here on is to merge, not to keep apart (source specification §2.5,
compression preference) — a pair stays separate only when the probe demonstrates
it, not the other way round.

An even earlier form of Pass B had the model WRITE a separation rule and then
judge for itself whether that rule was real — which cannot work, because a model
asked to write a rule always writes one, and then concludes the codes are
separable. Generating a justification is not a test as long as the model decides
whether it passes; the score of a blind probe on data the model did not produce
itself is.

Merging from here on is fully deterministic: same direction only (a positive and a
negative code are distinguished by their direction alone — that is not overlap,
that is the direction distinction itself), components via union-find (a chain
A-B-C becomes one group), and a UNION of members and respondent sets — never a
sum.

Merging changes the set, so a later round can reveal overlap an earlier round
could not yet see. That is why `enforce_mece` iterates pass A + pass B until a
round merges nothing, with a cap (`config.mece_max_rounds`)."""
from __future__ import annotations

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from ..config_codeGenerator import CodebookConfig
from ..code_shape import CodeShape
from .prompts_mece import (
    CandidatePair, CodeCandidate, IdeaAssignment, OverlapDetectionResult, OverlapVerdict,
    ProbeIdea, ProbeResult, build_overlap_prompt, build_probe_prompt, make_overlap_model,
    make_probe_model,
)
from ..prompts_common import _shuffled
from ..taxonomy_input import IdeaUnit

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
    """The outcome of one blind probe: run-local, never an LLM response model.
    `one_dimension` follows from `accuracy` and `both_rate` in Python
    (`is_one_dimension`) — the model claims nothing about itself here."""
    pair_id: int
    accuracy: float
    both_rate: float
    one_dimension: bool


@dataclass(frozen=True)
class PairProbe:
    """Everything needed to probe one pair and to score it. `truth` (idea_ref ->
    real code name) is the answer key — built in Python, never shown to the model
    (see `prompts_mece.build_probe_prompt`, which only receives `ideas`)."""
    pair: CandidatePair
    ideas: List[ProbeIdea]
    truth: Dict[int, str]


def _shuffled_ideas(units: List[IdeaUnit]) -> List[IdeaUnit]:
    """`units` in the shared deterministic order, keyed on each idea's own
    `idea_id` — not on the shared attribute id (that would leave ideas from the
    same attribute unsorted) and not on input order."""
    keyed = [_KeyedIdea(attribute_id=unit.idea_id, unit=unit) for unit in units]
    return [entry.unit for entry in _shuffled(keyed)]


def _idea_text(unit: IdeaUnit) -> str:
    if unit.interpretation and unit.interpretation != unit.instance:
        return f"{unit.instance} ({unit.interpretation})"
    return unit.instance


def _idea_pool(
    candidate: CodeCandidate, idea_units_by_attribute: Dict[str, List[IdeaUnit]], n: int,
) -> List[IdeaUnit]:
    """Up to `n` ideas from this code, pooled across all its source attributes
    (including after an earlier MECE merge: `shape.members` then carries the
    union of the original attribute ids) and sampled deterministically."""
    pooled = [unit for member_id in candidate.shape.members
              for unit in idea_units_by_attribute.get(member_id, [])]
    return _shuffled_ideas(pooled)[:n]


def build_pair_probe(
    pair: CandidatePair,
    candidate_by_name: Dict[str, CodeCandidate],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    ideas_per_code: int,
) -> Optional[PairProbe]:
    """Builds the pooled, shuffled idea list and the answer key for one pair.
    `None` when either side has no ideas at all — a probe without material on one
    side measures nothing, so that pair is skipped (not merged: the same
    no-hard-stop contract as a failed call)."""
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
    """The two deterministic signals from one blind probe — never a claim by the
    model about itself, always computed against `PairProbe.truth` in Python."""
    accuracy: float
    both_rate: float


def score_probe(assignments: List[IdeaAssignment], truth: Dict[int, str]) -> ProbeScore:
    """`accuracy` — of the ideas the model put on one side (A or B, so NOT
    "BOTH"), the share that landed on the side the idea really comes from. An
    idea without an assignment, or with "BOTH", does not count in that
    denominator: it was not "put on a side". `both_rate` — the share of ALL
    probed ideas (the whole of `truth`) that got "BOTH", regardless of which side
    that idea really came from. A duplicate assignment for the same `idea_ref`
    lets the last answer win. An empty `truth` (impossible via
    `build_pair_probe`, but tested directly) puts both at 0.0 rather than
    dividing by zero."""
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
    """Merge (`True`) when separability is absent (`accuracy` at or below its
    threshold) OR when ideas structurally belong to both (`both_rate` at or above
    its threshold) — the two ways a pair can fail at "demonstrably staying
    apart": inseparable, or separable-but-both-genuinely-fit. Chance level on a
    binary choice (`accuracy`, excluding "BOTH") is 0.50; the default threshold
    (0.70) sits above that, so 'no better than guessing' merges comfortably. The
    null option is to merge, not to keep apart (source specification §2.5) —
    which is why this is an OR, not an AND: either signal alone is enough."""
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
    """One task per pair (pairs are independent), bundled into a single
    SmoothRequester batch for concurrency. A pair without material
    (`build_pair_probe` returned `None`) or with a failed call simply gets no
    entry — the same no-hard-stop contract as elsewhere in this module."""
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
    # SmoothRequester.process_all demands List[Dict] — a list of objects trips
    # _execute_task on task.get(), and then trips the error handling after it as
    # well, which hides the real cause.
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
    """Unique, unordered, same-direction pairs from Pass A. A pair the model
    proposes across a direction boundary, or a code proposing itself, is removed
    here — deterministically, regardless of what the model said: a positive and a
    negative code are distinguished by their direction alone, and are never a
    merge candidate. `pair_id` follows the `_shuffled` order, so that prompt order
    and the `Literal` enum in the response model always stay in step."""
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
    """Connected components (union-find) over `one_dimension=True` verdicts: a
    chain A-B, B-C becomes one component, regardless of the order in which the
    pairs were judged. Only components with more than one member are a real
    merge."""
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
    """The member with the most source attributes wins (a substantive count,
    computed in code — never shown to the model); on a tie the shortest name,
    then alphabetically."""
    return min(group, key=lambda name: (
        -len(candidate_by_name[name].shape.members), len(name), name
    ))


def merge_candidates(
    group: Set[str], candidate_by_name: Dict[str, CodeCandidate], key: str
) -> CodeCandidate:
    """Unions member ids AND respondent sets across the group — never a sum
    (`CodeShape` carries `frozenset`s for exactly this). Refuses a group that
    does not all share the same direction: that would contradict the direction
    distinction itself, and must never arrive here (the deterministic filter in
    `build_candidate_pairs` already prevents it upstream — this is the safety
    net)."""
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
    """Replaces every component with one merged candidate; codes in no component
    stay unchanged (same object).

    `candidate_by_name` is unavoidably name-keyed: the components come from
    Pass A/B, which both work in the model's name domain (a prompt can only refer
    to a code by the name the model was shown — there is no `shape.key` the model
    knows to disambiguate with). If two candidates happen to share a name, the
    lookup is ambiguous in itself, and nothing here can repair that. What can be
    repaired: whichever of the two belonged to a merge must not drag its namesake
    along. `untouched` used to filter on name (`c.name not in merged_names`) — if
    two candidates carried the same name and ONE of them was merged, BOTH
    disappeared from `untouched` while only the one the dict lookup actually
    returned had been merged. The other — a completely different shape that
    happens to carry the same name — was then silently removed from the codebook,
    worse than a wrong merge. `consumed_keys` therefore tracks which physical
    object (`shape.key`, unique per shape) the lookup really used, so a namesake
    that was not itself looked up is kept."""
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
    """One entry per probed pair, sorted on `pair_id` for a reproducible print
    order — the data for the per-pair log line (the two code names, `accuracy`,
    `both_rate`, the merge decision)."""
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
    """Every termination reason is its own field — never just `merges=0`. A call
    that crashes (`reason="detection_failed"` etc.) and a round that genuinely
    found nothing (`reason=None`, `pairs_probed > 0`) both log `merges=0`, but
    are afterwards indistinguishable on that number alone — which was exactly the
    defect this replaced. `pairs` (empty unless Pass B actually scored pairs)
    carries the per-pair breakdown (`_pair_details`) for the caller's print
    line."""
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
    """Detect (Pass A) + probe blindly (Pass B), repeated until a round merges
    nothing or `config.mece_max_rounds` is reached. Returns the final candidates:
    merged codes have `shape.origin == "mece_merge"` and still carry placeholder
    text — the caller rewrites the text for those codes only (`write_codebook` on
    their `shape`). `idea_units_by_attribute` (attribute id -> its ideas) supplies
    the material for the blind probe; see `taxonomy_input.build_idea_units`."""
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
