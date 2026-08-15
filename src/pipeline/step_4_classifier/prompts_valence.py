"""Valence-neutral merge — the last phase of step 4.

Collapses attribute pairs that split one concept along evaluative direction into
a single descriptive attribute. This is a safety net, not the main defence: the
lever that actually works is rule 2 of `prompts_shared.UNIVERSAL_RULES`, which
forbids the split at the point where attributes are named. On the verification
run of 2026-06-07 that rule brought the number of candidates reaching this phase
to zero. See dev/DESIGN_VALENCE_NEUTRALITY.md.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .prompts_shared import INSTRUCTOR_HINT


class ValenceNeutralAttribute(BaseModel):
    """One descriptive, valence-neutral attribute replacing a valence-split pair."""
    pair_id: int = Field(..., description="The id of the attribute pair this replaces")
    attribute_name: str = Field(
        ...,
        description=(
            "One descriptive, valence-neutral attribute name, in the survey "
            "language (at most 5 words)"
        ),
    )
    attribute_definition: str = Field(
        ...,
        description=(
            "A 1-2 sentence valence-neutral description, in the survey language"
        ),
    )


class ValenceNeutralRenameResponse(BaseModel):
    """Neutral replacements for the supplied valence-split attribute pairs."""
    attributes: List[ValenceNeutralAttribute] = Field(
        ..., description="Exactly one neutral attribute per input pair_id"
    )


def build_valence_neutral_rename_prompt(pairs: list, language: str) -> str:
    """Name one descriptive attribute per valence-split pair.

    `pairs`: dicts with pair_id, name_a, desc_a, name_b, desc_b, samples.
    """
    blocks = []
    for p in pairs:
        samples = ", ".join(f'"{s}"' for s in p.get("samples", []))
        blocks.append(
            f"[{p['pair_id']}]\n"
            f'  A: "{p["name_a"]}" — {p.get("desc_a", "")}\n'
            f'  B: "{p["name_b"]}" — {p.get("desc_b", "")}\n'
            f"  example mentions: {samples}"
        )
    pairs_block = "\n\n".join(blocks)

    return f"""You are cleaning up a taxonomy. Each numbered pair below wrongly split ONE concept by evaluative direction (valence): the two attributes mean the same thing, but one captures the positive side and the other the negative or neutral side. Valence has been baked into the attribute, which is wrong — valence is recorded separately, per response.

For each pair, produce ONE descriptive, valence-neutral attribute that covers both sides:
- The name (at most 5 words, in {language}) and description (1-2 sentences, in {language}) must be purely descriptive.
- Do NOT encode positive, negative, good or bad — that direction is captured separately as valence.
- Name the underlying subject the two share: a pair that splits an impression into a positive and a negative version becomes the impression itself.

Pairs:
{pairs_block}

Return exactly one entry per pair_id. Begin now and {INSTRUCTOR_HINT}"""
