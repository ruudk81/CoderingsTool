"""Stap 2b — verzamelnamen consolideren voordat er gepoold wordt.

Stap 2 (prompts_relations.py) genereert per attribuut onafhankelijk een
verzamelnaam ("umbrella"). Niets dwingt twee gelijke betekenissen tot dezelfde
bewoording, dus verwante namen versplinteren de pool ("Bankdiensten" /
"Bankdiensten en aanbod"). Deze module bouwt de kleine vervolgcall die dat
rechttrekt: één keer over de lijst van verzamelnamen, niet over de attributen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Tuple

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


class UmbrellaGroup(BaseModel):
    canonical_name: str = Field(..., description="The name to keep for this group")
    canonical_definition: str = Field(..., description="One sentence defining it")
    members: List[str] = Field(..., description="The umbrella names that mean the same thing")


class UmbrellaMergeResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the groups")
    groups: List[UmbrellaGroup] = Field(..., description="One entry per group of equivalent names")


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
    """UmbrellaMergeResult met `members` beperkt tot bestaande verzamelnamen."""
    names: Tuple[str, ...] = tuple(u.name for u in _shuffled(umbrellas))
    constrained_group = create_model(
        "ConstrainedUmbrellaGroup",
        __base__=UmbrellaGroup,
        members=(List[Literal[names]], Field(..., description=(
            UmbrellaGroup.model_fields["members"].description))),
    )
    return create_model(
        "ConstrainedUmbrellaMergeResult",
        __base__=UmbrellaMergeResult,
        groups=(List[constrained_group], Field(..., description=(
            UmbrellaMergeResult.model_fields["groups"].description))),
    )


def build_umbrella_merge_prompt(umbrellas, language: str) -> str:
    inventory = "\n".join(
        f'- "{u.name}": {u.definition}\n  Topics: {", ".join(u.member_names)}'
        for u in _shuffled(umbrellas)
    )

    return f"""Below is a list of broader concepts, each with the topics that were placed under
it. They were named independently of one another, so some of them are the same
concept under different wording.

Group the names that mean the SAME concept. For each group, choose one name to
keep and write a one-sentence definition. Write both in {language}.

Rules:
- Only group names that are the same concept, not names that are merely related
  or where one is a special case of the other.
- A name that matches no other name forms a group of its own. Every name must
  appear in exactly one group.
- Judge meaning only. You are not told how often anything occurs, and the number
  of topics under a name says nothing about whether it should be kept.

Broader concepts:
{inventory}

{INSTRUCTOR_HINT}"""
