"""Meet hoe vast de consolidatie ligt door fase 1 een paar keer te herhalen.

De consolidatiecall reproduceert niet: drie runs op identieke invoer gaven 26, 31
en 25 codes (ARI 0,33-0,58, gemeten 2026-08-18). Dat is een gegeven, geen bug die
hier wordt opgelost — deze module maakt het alleen zichtbaar.

De eenheid is het ATTRIBUUTPAAR, niet de code. Voor elk paar telt dit in hoeveel
runs de twee samen in één groep zaten. Altijd samen en nooit samen zijn allebei
een vast besluit; alles daartussen is een plek waar het model geen stabiel oordeel
heeft. Die lijst stuurt de post-mortem: die hoeft niet te raden waar hij moet
kijken.

Bewust NIET: een consensusindeling afleiden uit de vaste paren. Transitieve
sluiting plakt A-B-C aan elkaar zodra A-B en B-C vastliggen, óók wanneer A-C
juist vast op apart staat — dan bouw je een groep die geen enkele run heeft
voorgesteld. Het codeboek blijft van één run; deze meting stuurt alleen de
aandacht.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .concept_inventory import Concept
from .config_codeGenerator import CodebookConfig
from .attribute_cards import AttributeCard
from .consolidation import resolve_consolidation
from .grouping import Group, repair_partition

DEFAULT_RUNS = 5


def pairs_from_groups(groups: List[Group]) -> Set[FrozenSet[str]]:
    """Elk paar attributen dat in dezelfde groep zit. Een groep van één levert
    niets op — daar valt geen samen-of-apart aan af te lezen."""
    return {
        frozenset(pair)
        for group in groups
        for pair in combinations(sorted(group.member_ids), 2)
    }


@dataclass(frozen=True)
class StabilityReport:
    runs: int
    together: Dict[FrozenSet[str], int]

    def unstable_pairs(self) -> List[Tuple[str, str]]:
        """Paren die niet in élke run samen zaten en ook niet in élke run apart.
        Gesorteerd, zodat twee metingen op dezelfde runs dezelfde lijst geven."""
        return sorted(
            tuple(sorted(pair))
            for pair, n in self.together.items()
            if 0 < n < self.runs
        )

    def unstable_attributes(self) -> Set[str]:
        """Elk attribuut dat in minstens één wisselend paar zit. LET OP: op een
        inventaris met veel wisseling raakt dit vrijwel alles — gebruik
        `has_unstable_pair_within` om te toetsen of een GROEP intern wiebelt."""
        return {attribute for pair in self.unstable_pairs() for attribute in pair}

    def has_unstable_pair_within(self, attribute_ids) -> bool:
        """Wisselt er een paar BINNEN deze verzameling? Dat is iets heel anders
        dan 'bevat een attribuut dat ergens wisselt': een groep kan uit louter
        attributen bestaan die elders wiebelen terwijl hun onderlinge indeling
        in elke run identiek was — dan valt er hier niets te heroverwegen."""
        ids = sorted(attribute_ids)
        return any(
            0 < self.together.get(frozenset(pair), 0) < self.runs
            for pair in combinations(ids, 2)
        )

    def stable_share(self) -> float:
        """Aandeel van alle paren waarover de runs het eens zijn. 1.0 betekent
        dat elke samen-of-apart-beslissing in elke run hetzelfde uitviel."""
        if not self.together:
            return 1.0
        settled = sum(1 for n in self.together.values() if n in (0, self.runs))
        return settled / len(self.together)


def measure_stability(runs: List[List[Group]], attribute_ids: List[str]) -> StabilityReport:
    """Telt per attribuutpaar in hoeveel runs het samen zat.

    `attribute_ids` bepaalt welke paren überhaupt geteld worden, zodat een
    attribuut dat in één run ontbreekt toch als 'niet samen' meetelt in plaats
    van uit de meting te verdwijnen."""
    if len(runs) < 2:
        raise ValueError("stabiliteit vraagt minstens twee runs om te vergelijken")

    counts: Counter = Counter()
    for groups in runs:
        counts.update(pairs_from_groups(groups))

    together = {
        frozenset(pair): counts.get(frozenset(pair), 0)
        for pair in combinations(sorted(attribute_ids), 2)
    }
    return StabilityReport(runs=len(runs), together=together)


async def run_consolidation_repeatedly(
    cards: List[AttributeCard],
    concepts: List[Concept],
    survey_question: str,
    n_respondents: int,
    language: str,
    config: CodebookConfig,
    runs: int = DEFAULT_RUNS,
    verbose: bool = False,
    on_run=None,
    first_run_log=None,
) -> Tuple[StabilityReport, List[List[Group]]]:
    """Draait alleen fase 1 — geen schrijffase, geen cache.

    Geeft de meting terug PLUS de groepen van elke run: één daarvan wordt het
    codeboek, de rest dient alleen om te zien wat wisselde. Zo kost de meting
    geen extra consolidatiecall bovenop de run die je toch al nodig had.
    `on_run` is een optionele callback (run-nummer, groepen). `first_run_log`
    vangt de partitiereparaties van de EERSTE run — dat is de run die het
    codeboek wordt, dus alleen daarvan is de reparatielijst rapportabel."""
    all_groups: List[List[Group]] = []
    for index in range(runs):
        proposal = await resolve_consolidation(
            cards, survey_question, n_respondents, language, config, verbose=verbose,
        )
        groups = repair_partition(proposal, cards, concepts,
                                  log=first_run_log if index == 0 else None)
        all_groups.append(groups)
        if on_run is not None:
            on_run(index + 1, groups)
    report = measure_stability(all_groups, [card.attribute_id for card in cards])
    return report, all_groups


def format_stability(report: StabilityReport, name_by_id: Optional[Dict[str, str]] = None) -> str:
    """Leesbaar overzicht. Zonder `name_by_id` toont het de ids."""
    def label(attribute_id: str) -> str:
        return name_by_id.get(attribute_id, attribute_id) if name_by_id else attribute_id

    unstable = report.unstable_pairs()
    lines = [
        f"STABILITEIT over {report.runs} consolidatieruns",
        f"  paren waarover de runs het eens zijn: {report.stable_share():.0%} "
        f"({len(report.together) - len(unstable)} van {len(report.together)})",
        f"  wisselende paren: {len(unstable)}",
        f"  betrokken attributen: {len(report.unstable_attributes())}",
    ]
    for a, b in unstable:
        lines.append(f"    {report.together[frozenset({a, b})]}/{report.runs}x samen  "
                     f"{label(a)}  ↔  {label(b)}")
    return "\n".join(lines)
