"""Stap 2 — relaties tussen attributen. Semantiek, geen aantallen."""
from __future__ import annotations

import hashlib
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, create_model

INSTRUCTOR_HINT = (
    "provide your output as valid JSON following the response schema provided"
)


class AttributeRelation(BaseModel):
    attribute: str = Field(..., description="The attribute this statement is about")
    synonym_of: Optional[str] = Field(
        None,
        description=(
            "Another attribute that means the SAME thing as this one, or null. "
            "Use only for genuine duplicates: two names for one concept."
        ),
    )
    umbrella_name: str = Field(
        ..., description="The broader concept this attribute belongs to (2-5 words)"
    )
    umbrella_definition: str = Field(
        ..., description="One sentence defining that broader concept"
    )


class RelationsResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the relations")
    relations: List[AttributeRelation] = Field(
        ..., description="Exactly one entry per attribute in the list"
    )


def tagged(concept) -> str:
    """Attribute id + name, e.g. "[A17] Prijs". The id disambiguates duplicate
    names (like the old domain qualification did) but — unlike the domain —
    carries no groupable content: it cannot be handed back as an umbrella."""
    return f"[{concept.attribute_id}] {concept.name}"


def _shuffled(concepts):
    """Concepts in a deterministic order unrelated to prevalence OR domain.

    Both the prompt text and the response model's enum must use this instead of
    the caller's order: `concepts` arrives prevalence-sorted (build_inventory's
    `(-n_resp, name)`), and that order — whether in prose or in a JSON schema
    enum — is itself a signal about how often something occurs. Sorting by
    attribute_id instead fixes that, but opens a second, subtler channel:
    identity.py mints A# sequentially PER DOMAIN, so id order still groups
    domains into contiguous blocks — visible structure of exactly the kind this
    step exists to stop handing the model. Sorting by a hash of the id keeps the
    order reproducible across runs (the hash is a pure function of a stable id)
    while carrying neither signal."""
    return sorted(concepts, key=lambda c: hashlib.md5(c.attribute_id.encode()).hexdigest())


def make_relations_model(concepts) -> type:
    """RelationsResult met `attribute` en `synonym_of` beperkt tot bestaande namen."""
    names: Tuple[str, ...] = tuple(tagged(c) for c in _shuffled(concepts))
    constrained_relation = create_model(
        "ConstrainedAttributeRelation",
        __base__=AttributeRelation,
        attribute=(Literal[names], Field(..., description=(
            AttributeRelation.model_fields["attribute"].description))),
        synonym_of=(Optional[Literal[names]], Field(None, description=(
            AttributeRelation.model_fields["synonym_of"].description))),
    )
    return create_model(
        "ConstrainedRelationsResult",
        __base__=RelationsResult,
        relations=(List[constrained_relation], Field(..., description=(
            RelationsResult.model_fields["relations"].description))),
    )


def build_relations_prompt(concepts, language: str) -> str:
    inventory = "\n".join(
        f'- "{tagged(concept)}": {concept.definition}' for concept in _shuffled(concepts)
    )

    return f"""You are organising a list of observed topics from an open-ended survey.

For EVERY topic below, state two things:

1. `synonym_of` — another topic in the list that means the SAME thing, or null.
   Use this only for true duplicates: two different names for one concept. Do NOT
   use it for topics that are merely related, or where one is a special case of
   the other.

2. `umbrella_name` and `umbrella_definition` — the broader concept this topic
   belongs to, shared with its siblings. Topics that belong together must get the
   SAME umbrella name, spelled identically. Write both in {language}.

Rules:
- Judge meaning only. You are NOT told how often anything occurs, and you must not
  guess: how common a topic is plays no part in this task.
- Do not decide what deserves to become a code. That decision is made elsewhere.
- Every topic gets exactly one entry. Do not invent topics that are not listed.
- An umbrella that would cover the entire list is too broad; an umbrella that
  covers exactly one topic is too narrow unless that topic truly stands alone.

Topics:
{inventory}

{INSTRUCTOR_HINT}"""
