"""Step 1 — the empirical inventory. Counting, nothing else.

Respondent sets rather than counts: on a merge the number of unique respondents
is the union, not the sum.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .taxonomy_input import AttributeRef, IdeaUnit

POSITIVE, NEGATIVE = "+", "-"


@dataclass(frozen=True)
class Concept:
    attribute_id: str
    name: str
    definition: str
    domain: str
    facet: str
    n_iu: int
    resp_ids: frozenset[str]
    resp_pos: frozenset[str]
    resp_neg: frozenset[str]
    resp_neu: frozenset[str]

    @property
    def n_resp(self) -> int:
        return len(self.resp_ids)

    @property
    def n_resp_pos(self) -> int:
        return len(self.resp_pos)

    @property
    def n_resp_neg(self) -> int:
        return len(self.resp_neg)

    @property
    def n_resp_neu(self) -> int:
        return len(self.resp_neu)


def t_keep(n_resp_total: int, config) -> int:
    """The threshold above which something may be a code of its own."""
    return max(config.t_keep_min_respondents,
               round(config.t_keep_share * n_resp_total))


def build_inventory(units: List[IdeaUnit],
                    refs: Dict[str, AttributeRef]) -> List[Concept]:
    """Eén Concept per attribuut dat daadwerkelijk ideeën heeft."""
    counts = defaultdict(lambda: {"n_iu": 0, "all": set(),
                                  "pos": set(), "neg": set(), "neu": set()})
    for unit in units:
        if unit.attribute_id not in refs:
            continue
        bucket = counts[unit.attribute_id]
        bucket["n_iu"] += 1
        bucket["all"].add(unit.respondent_id)
        if unit.valence == POSITIVE:
            bucket["pos"].add(unit.respondent_id)
        elif unit.valence == NEGATIVE:
            bucket["neg"].add(unit.respondent_id)
        else:
            bucket["neu"].add(unit.respondent_id)

    concepts = []
    for attribute_id, bucket in counts.items():
        ref = refs[attribute_id]
        concepts.append(Concept(
            attribute_id=attribute_id,
            name=ref.name,
            definition=ref.definition,
            domain=ref.domain,
            facet=ref.facet,
            n_iu=bucket["n_iu"],
            resp_ids=frozenset(bucket["all"]),
            resp_pos=frozenset(bucket["pos"]),
            resp_neg=frozenset(bucket["neg"]),
            resp_neu=frozenset(bucket["neu"]),
        ))
    concepts.sort(key=lambda c: (-c.n_resp, c.name))
    return concepts
