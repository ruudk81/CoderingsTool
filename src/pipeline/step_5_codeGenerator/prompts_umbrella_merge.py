"""Step 2b — consolidate umbrella names before pooling.

Step 2 (prompts_relations.py) generates an umbrella name per attribute,
independently. Nothing forces two identical meanings into the same wording, so
related names splinter the pool (one name and that same name plus a trailing
"and offering").

The first form of this module asked the model for a partition ("group the names
that mean the same thing"). On a real run that yielded NOTHING: every name as its
own group is an equally valid answer to that question, so there was no pressure
to merge anything. The form below asks a per-name question instead — "is there
another name in the list that means the same thing?" — the same pattern as
`synonym_of` in prompts_relations.py, which did find three correct pairs on that
same run. A per-item question forces a lookup per item; a grouping question does
not.

The canonical name per group is no longer chosen by the model: that is
deterministic in code (relations.py), based on how many attributes each name
already carries — a count that would be a leak channel in a prompt, but is just a
sort key in code.
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
