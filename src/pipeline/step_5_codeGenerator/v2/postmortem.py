"""Post-generatie bijstelling: een groep die te breed uitviel alsnog terugsnijden.

Waarom dit een APARTE fase is en geen extra eis in de consolidatieprompt: die
opdracht is al zwaar — het model weegt daar zestig onderwerpen tegen elkaar af en
moet ze in één keer goed indelen. Er nog een voorwaarde bij zetten maakt de kans
groter dat het geheel verslechtert. Hier krijgt het model per keer één groep en
één vraag, met alles erbij wat het nodig heeft.

Wat een kandidaat maakt, zijn twee objectieve triggers, geen doelaantal codes:
  - de groep dekt een onevenredig deel van de steekproef (relatief, nooit absoluut)
  - er wisselde een paar BINNEN de groep tussen consolidatieruns (zie
    stability.py) — dan heeft het model over die indeling zelf geen vast oordeel,
    en dat is precies waar een tweede blik iets toevoegt. Niet: "de groep bevat
    een attribuut dat ergens wiebelt" — op een inventaris met veel wisseling is
    dat vrijwel elke groep, en dan legt de post-mortem het hele codeboek open

Splitsen mag alleen LANGS BESTAANDE ATTRIBUUTGRENZEN. Het model kan niets fijner
maken dan wat step 4 heeft aangeleverd, en `apply_splits` wijst elk voorstel af
dat een attribuut laat verdwijnen, verzint of dubbel plaatst. De partitie blijft
daarmee heel, ook als het oordeel onzin is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from config import get_reasoning_params
from utils.smoothRequester import SmoothRequester

from ..concept_inventory import Concept
from ..config_codeGenerator import CodebookConfig
from .attribute_cards import AttributeCard
from .grouping import Group
from .stability import StabilityReport
from .prompts_postmortem import (
    build_postmortem_prompt, make_postmortem_model,
)

PHASE = "step5_v2_postmortem"

# Beredeneerde startwaarde, niet gemeten: een groep die meer dan dit aandeel van
# de respondenten dekt, verdient een tweede blik. Bijstellen zodra er runs op
# meer dan één dataset zijn — net als de degeneratiegrenzen in grouping.py.
SHARE_THRESHOLD = 0.20


@dataclass(frozen=True)
class SplitVerdict:
    """Het oordeel over één kandidaatgroep. Lege `parts` betekent: laat staan."""
    group_name: str
    parts: Tuple[Tuple[str, ...], ...]


def group_respondents(group: Group, concept_by_id: Dict[str, Concept]) -> frozenset:
    """Union over de leden, nooit een som — een respondent die in twee attributen
    van dezelfde groep zit telt één keer."""
    sets = [concept_by_id[i].resp_ids for i in group.member_ids if i in concept_by_id]
    return frozenset().union(*sets) if sets else frozenset()


def select_candidates(
    groups: List[Group],
    concepts: List[Concept],
    report: StabilityReport,
    n_respondents: int,
    share_threshold: float = SHARE_THRESHOLD,
) -> List[Group]:
    """Groepen die een tweede blik verdienen. Een groep van één attribuut valt
    niet te splitsen langs attribuutgrenzen en komt dus nooit in aanmerking,
    hoe groot hij ook is."""
    concept_by_id = {c.attribute_id: c for c in concepts}

    picked = []
    for group in groups:
        if len(group.member_ids) < 2:
            continue
        oversized = (
            n_respondents > 0
            and len(group_respondents(group, concept_by_id)) / n_respondents > share_threshold
        )
        wobbled = report.has_unstable_pair_within(group.member_ids)
        if oversized or wobbled:
            picked.append(group)
    return picked


def apply_splits(
    groups: List[Group], verdicts: List[SplitVerdict],
) -> Tuple[List[Group], List[dict]]:
    """Vervangt elke gesplitste groep door zijn delen, op de plek waar hij stond.

    Een voorstel wordt in zijn geheel afgewezen zodra het de leden van de groep
    niet exact herverdeelt: een ontbrekend attribuut zou stil uit het codeboek
    verdwijnen, een verzonnen of dubbel attribuut zou de partitie breken. Half
    toepassen is er niet bij — dan zou de garantie afhangen van welk deel toevallig
    klopte."""
    by_name: Dict[str, SplitVerdict] = {v.group_name: v for v in verdicts}
    out: List[Group] = []
    log: List[dict] = []

    for group in groups:
        verdict = by_name.get(group.proposed_name)
        if verdict is None or not verdict.parts or len(verdict.parts) < 2:
            out.append(group)
            continue

        proposed = [attribute for part in verdict.parts for attribute in part]
        original, seen = set(group.member_ids), set(proposed)
        missing, invented = original - seen, seen - original
        if missing or invented or len(proposed) != len(seen):
            reason = []
            if missing:
                reason.append(f"laat {', '.join(sorted(missing))} vallen")
            if invented:
                reason.append(f"verzint {', '.join(sorted(invented))}")
            if len(proposed) != len(seen):
                reason.append("plaatst een attribuut dubbel")
            log.append({
                "action": "POSTMORTEM_SPLIT_REJECTED",
                "group": group.proposed_name,
                "reason": "; ".join(reason),
            })
            out.append(group)
            continue

        for part in verdict.parts:
            out.append(Group(member_ids=tuple(part), proposed_name=group.proposed_name,
                             explanation=group.explanation))
        log.append({
            "action": "POSTMORTEM_SPLIT",
            "group": group.proposed_name,
            "parts": [list(part) for part in verdict.parts],
        })
    return out, log


def format_postmortem(n_candidates: int, log: List[dict]) -> str:
    lines = [f"POST-MORTEM: {n_candidates} kandidaatgroep(en) bekeken"]
    for entry in log:
        if entry["action"] == "POSTMORTEM_SPLIT":
            parts = " | ".join(", ".join(part) for part in entry["parts"])
            lines.append(f"  gesplitst: '{entry['group']}' -> {parts}")
        else:
            lines.append(f"  VOORSTEL AFGEWEZEN voor '{entry['group']}': {entry['reason']}")
    if not log:
        lines.append("  geen enkele groep gesplitst")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _labelled(candidates: List[Group],
              card_by_id: Dict[str, AttributeCard]) -> List[Tuple[str, List[AttributeCard]]]:
    """Elke kandidaat krijgt een run-lokaal label. Dat is het label dat het model
    terugmeldt; `verdicts_from_result` vertaalt het terug naar de groepsnaam, zodat
    twee kandidaten met dezelfde voorgestelde naam niet op elkaar botsen."""
    return [
        (f"K{i}", [card_by_id[m] for m in group.member_ids if m in card_by_id])
        for i, group in enumerate(candidates, 1)
    ]


def verdicts_from_result(result, candidates: List[Group],
                         card_by_id: Dict[str, AttributeCard]) -> List[SplitVerdict]:
    """Zet de deelnummers per onderwerp om in `SplitVerdict`s. Delen komen in
    oplopende deelnummervolgorde; binnen een deel blijft de volgorde van de groep
    zelf staan, zodat twee identieke antwoorden hetzelfde resultaat geven."""
    group_by_label = {f"K{i}": g for i, g in enumerate(candidates, 1)}
    id_by_tag = {card.tag: card.attribute_id for card in card_by_id.values()}

    verdicts: List[SplitVerdict] = []
    for verdict in result.verdicts:
        group = group_by_label.get(verdict.group)
        if group is None:
            continue
        part_of: Dict[str, int] = {}
        for assignment in verdict.assignments:
            attribute_id = id_by_tag.get(assignment.topic)
            if attribute_id is not None:
                part_of[attribute_id] = assignment.part
        numbers = sorted(set(part_of.values()))
        parts = tuple(
            tuple(m for m in group.member_ids if part_of.get(m) == number)
            for number in numbers
        )
        verdicts.append(SplitVerdict(group_name=group.proposed_name,
                                     parts=tuple(p for p in parts if p)))
    return verdicts


async def resolve_postmortem(
    candidates: List[Group],
    cards: List[AttributeCard],
    survey_question: str,
    n_respondents: int,
    language: str,
    config: CodebookConfig,
    verbose: bool = False,
) -> List[SplitVerdict]:
    """Eén call over alle kandidaten. Faalt hij, dan blijft het codeboek zoals
    het was — dit is bijstelling, geen fundament, dus een mislukking mag hem niet
    tegenhouden."""
    if not candidates:
        return []

    card_by_id = {card.attribute_id: card for card in cards}
    labelled = _labelled(candidates, card_by_id)

    def prepare_fn(task):
        return {
            "prompt": build_postmortem_prompt(
                task["labelled"], survey_question, n_respondents, language),
            "response_model": make_postmortem_model(task["labelled"]),
            "temperature": config.temperature_relations,
            "max_tokens": config.max_tokens_relations,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(
                config.model_relations, phase="codegen_relations"),
        }

    requester = SmoothRequester(
        model=config.model_relations, phase_key=PHASE, num_tasks=1,
        verbose=verbose, quiet=True,
    )
    results = await requester.process_all(
        [{"labelled": labelled}], prepare_fn,
        lambda _task, response: response, lambda _task, _reason: None,
    )
    if not results or results[0] is None:
        return []
    return verdicts_from_result(results[0], candidates, card_by_id)
