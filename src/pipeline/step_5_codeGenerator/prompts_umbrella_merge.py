"""Stap 2b — verzamelnamen consolideren voordat er gepoold wordt.

Stap 2 (prompts_relations.py) genereert per attribuut onafhankelijk een
verzamelnaam ("umbrella"). Niets dwingt twee gelijke betekenissen tot dezelfde
bewoording, dus verwante namen versplinteren de pool ("Bankdiensten" /
"Bankdiensten en aanbod").

Eerste vorm van deze module vroeg het model een partitionering te maken
("groepeer de namen die hetzelfde betekenen"). Op een echte run leverde dat
NIETS op: elke naam als eigen groep is een even geldig antwoord op die vraag, dus
er was geen dwang om iets samen te voegen. De vorm hieronder stelt in plaats
daarvan een per-naam vraag — "is er een andere naam in de lijst die hetzelfde
betekent?" — hetzelfde patroon als `synonym_of` in prompts_relations.py, dat op
diezelfde run wél drie correcte paren vond. Een per-item vraag dwingt een
opzoeking per item af; een groepeervraag niet.

De canonieke naam per groep wordt niet meer door het model gekozen: dat is
deterministiek in code (relations.py), gebaseerd op hoeveel attributen elke naam
al draagt — een telling die in een prompt een lekkanaal zou zijn, maar in code
gewoon een sorteersleutel is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, create_model

from .prompts_relations import INSTRUCTOR_HINT, _shuffled

_TAG_RE = re.compile(r"^\[[^\]]+\]\s*")


@dataclass(frozen=True)
class Umbrella:
    name: str
    definition: str
    member_names: Tuple[str, ...]

    @property
    def attribute_id(self) -> str:
        """Lets `_shuffled` (keyed on `.attribute_id`) order umbrellas by the
        same deterministic-hash mechanism as attributes, without a second
        implementation of that ordering."""
        return self.name


class UmbrellaVerdict(BaseModel):
    umbrella: str = Field(..., description="The umbrella name this verdict is about")
    same_as: Optional[str] = Field(
        None,
        description=(
            "Another umbrella name in the list that means the SAME concept as "
            "this one, or null. Use only for genuine duplicates: two different "
            "names for one concept."
        ),
    )


class UmbrellaMergeResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the verdicts")
    verdicts: List[UmbrellaVerdict] = Field(
        ..., description="Exactly one entry per umbrella name in the list"
    )


def umbrellas_from_relations(relations_result) -> List[Umbrella]:
    """Group a RelationsResult's per-attribute relations into one Umbrella per
    distinct umbrella_name, with only the name part of each member attribute."""
    grouped: dict[str, dict] = {}
    for relation in relations_result.relations:
        entry = grouped.setdefault(relation.umbrella_name, {
            "definition": relation.umbrella_definition,
            "member_names": [],
        })
        entry["member_names"].append(_TAG_RE.sub("", relation.attribute))
    return [
        Umbrella(name=name, definition=data["definition"], member_names=tuple(data["member_names"]))
        for name, data in grouped.items()
    ]


def make_umbrella_merge_model(umbrellas) -> type:
    """UmbrellaMergeResult met `umbrella` en `same_as` beperkt tot bestaande
    verzamelnamen."""
    names: Tuple[str, ...] = tuple(u.name for u in _shuffled(umbrellas))
    constrained_verdict = create_model(
        "ConstrainedUmbrellaVerdict",
        __base__=UmbrellaVerdict,
        umbrella=(Literal[names], Field(..., description=(
            UmbrellaVerdict.model_fields["umbrella"].description))),
        same_as=(Optional[Literal[names]], Field(None, description=(
            UmbrellaVerdict.model_fields["same_as"].description))),
    )
    return create_model(
        "ConstrainedUmbrellaMergeResult",
        __base__=UmbrellaMergeResult,
        verdicts=(List[constrained_verdict], Field(..., description=(
            UmbrellaMergeResult.model_fields["verdicts"].description))),
    )


def build_umbrella_merge_prompt(umbrellas) -> str:
    inventory = "\n".join(
        f'- "{u.name}": {u.definition}\n  Topics: {", ".join(u.member_names)}'
        for u in _shuffled(umbrellas)
    )

    return f"""Below is a list of broader concepts, each with the topics that were placed under
it. They were named one at a time, independently, so some of them are the same
concept under different wording.

For EVERY concept in the list, state whether another concept in the list means
the SAME thing. If one does, name it. If none does, say null.

Rules:
- Same concept only — not merely related, and not one being a special case of
  the other.
- Every concept in the list gets exactly one entry.
- Judge meaning only. You are not told how often anything occurs.

Broader concepts:
{inventory}

{INSTRUCTOR_HINT}"""
