"""De tweede schrijfprompt: de restcategorieën onder Overig.

`prompts_writer.py` schrijft de hoofdcodes. Deze module schrijft de kinderen —
facetunies van afgevallen valentiepolen die de drempel niet halen maar de bodem
wel (`grouping.pool_minority_poles`). Ze hangen onder de Overig-code en zijn
géén hoofdthema's, en dat is de enige inhoudelijke regel die van de
schrijverprompt verschilt: de naam moet de restcategorie binnen zijn onderwerp
dragen zonder een eigen kop te claimen.

De rest is geleend, niet gekopieerd: `_ordered`, `_code_block` en
`INSTRUCTOR_HINT` komen uit de schrijverkant. Twee promptmodules die dezelfde
ledenweergave elk apart uitschrijven lopen uit elkaar zodra er één wordt
bijgesteld, en dan zou het verschil tussen de twee prompts groter zijn dan het
verschil dat bedoeld is.

Eén veld ontbreekt bewust ten opzichte van `CodeText`: `nameable`. Een kind is
niet vetobaar — het bestaat omdat deze respondenten anders nergens staan (zie
`code_shape.CodeShape.origin`) — en een veto-veld dat de keten toch negeert
nodigt het model uit een oordeel te vellen dat nergens landt.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, create_model

from .prompts_common import INSTRUCTOR_HINT
from .prompts_writer import _code_block, _ordered


def _miscellaneous_block(shape, concept_by_id) -> str:
    """Het ledenblok van de schrijver, plus het gedeelde onderwerp.

    Het onderwerp (het facet, dat `grouping.build_shapes` als `umbrella` op de
    vorm zet) is voor deze prompt geen versiering maar het niveau waaróp de
    naam moet landen. Zonder die regel kan het model alleen terugvallen op het
    eerste lid, en dan claimt een restcategorie de kop van één onderwerp.
    Ontbreekt het onderwerp — per constructie onbereikbaar, want een kind heeft
    altijd één facet — dan blijft het blok gewoon dat van de schrijver.
    """
    blok = _code_block(shape, concept_by_id)
    return f"{blok}\n  Shared subject: {shape.umbrella}" if shape.umbrella else blok


class MiscellaneousText(BaseModel):
    key: str = Field(..., description="Which fixed rest category this text belongs to")
    code_name: str = Field(
        ..., description=(
            "Short code name (3-5 words), in the survey language, naming the "
            "rest category within its subject"
        )
    )
    definition: str = Field(
        ..., description=(
            "One interpretive sentence that reads like an analyst conclusion, "
            "not a restatement of the topic list."
        )
    )
    diagnostic_test: str = Field(
        ..., description=(
            "Completes the diagnostic stem — must be unique per code and must "
            "not overlap with other codes in the list."
        )
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    boundary_note: str = Field(
        ..., description="What distinguishes this code from its nearest competing code in the list"
    )


class MiscellaneousResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the codes")
    codes: List[MiscellaneousText] = Field(
        ..., description="Exactly one entry per rest category in the list"
    )


def make_miscellaneous_model(shapes) -> type:
    """MiscellaneousResult met `key` beperkt tot de aangeboden vormen, zodat het
    model er geen kan verzinnen en geen kan overslaan."""
    keys: Tuple[str, ...] = tuple(shape.key for shape in _ordered(shapes))
    constrained_text = create_model(
        "ConstrainedMiscellaneousText",
        __base__=MiscellaneousText,
        key=(Literal[keys], Field(
            ..., description=MiscellaneousText.model_fields["key"].description
        )),
    )
    return create_model(
        "ConstrainedMiscellaneousResult",
        __base__=MiscellaneousResult,
        codes=(List[constrained_text], Field(
            ..., description=MiscellaneousResult.model_fields["codes"].description
        )),
    )


def build_miscellaneous_prompt(
    shapes, concept_by_id, dimension_diagnostic: str, language: str,
    taken_names: Optional[List[str]] = None,
) -> str:
    inventory = "\n".join(
        _miscellaneous_block(shape, concept_by_id) for shape in _ordered(shapes))

    taken_block = ""
    if taken_names:
        names = "\n".join(f"- {name}" for name in sorted(set(taken_names)))
        taken_block = f"""

These code names are already used elsewhere in this codebook and are OFF LIMITS —
do not reuse any of them for a code below, even if it would otherwise fit:
{names}"""

    return f"""You are writing the rest categories of a codebook. These are not main themes: each
one holds what was left over within a single subject area, and each sits under the
codebook's catch-all rest code. The set is already fixed: there are {len(shapes)} of
them, each with its direction already decided from the data. Write the text for each
one. Do NOT add, remove or merge codes.

For every code, write:
- a name (3-5 words) in {language}
- a definition: one interpretive sentence that reads like an analyst conclusion
- a diagnostic test completing this stem: "{dimension_diagnostic}"
- typical indicators: words or phrases that signal this code
- a boundary note against the nearest competing code in this list

Three rules:
1. The name must read as a rest category within its subject, and not as a theme
   of its own. Build it on the shared subject shown with the code and make clear
   that this code holds what falls outside the main codes for that subject: a
   bare topic name would claim a heading this code does not have, and a bare
   word for "other" would hide which subject it belongs to. The members of one
   code always share that subject — that is how they were grouped — so an
   honest name exists at that level even when the members have nothing else in
   common. Never invent an umbrella that claims more than the members support.
2. The direction shown for a code is a fact from the data, not a suggestion. Where
   a code's direction is positive or negative, that evaluation must be readable in
   BOTH its name and its definition — someone reading only the name in a report
   table has to know which way it points. Where the direction is neutral or
   non_negative, the code is descriptive: name the topic itself and do not invent
   an evaluation it does not carry.
3. A code name must not claim territory another code already owns — neither
   another code in the list below, nor one of the already-taken names listed
   further down, if any.

Codes:
{inventory}
{taken_block}

{INSTRUCTOR_HINT}"""
