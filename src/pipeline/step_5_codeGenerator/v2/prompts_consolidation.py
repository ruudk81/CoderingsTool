"""Fase 1 — de consolidatievraag zoals je hem aan een onderzoeker zou stellen.

Het criterium is het rapportagedoel, niet een aantal: elke code wordt een regel
in een tabel, en de lezer moet die regel snappen zonder de rest erbij. Dat is de
tegendruk die een kale partitievraag mist — die gaf op een live run 45 namen in
en 45 groepen uit.

Richting komt hier niet voor. Die wordt deterministisch afgeleid zodra de groepen
vaststaan (grouping.py).
"""
from __future__ import annotations

from typing import List, Literal, Tuple

from pydantic import BaseModel, Field, create_model

from ..prompts_relations import INSTRUCTOR_HINT, _shuffled


class ProposedCode(BaseModel):
    code_name: str = Field(
        ..., description="Short name for this code, as it would appear in a report table"
    )
    explanation: str = Field(
        ..., description="The one sentence that explains what this code covers"
    )
    topics: List[str] = Field(..., description="The topics that belong in this code")


class ConsolidationResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the codes")
    codes: List[ProposedCode] = Field(..., description="The proposed codebook")


def make_consolidation_model(cards) -> type:
    """`topics` beperkt tot de getoonde tags, zodat het model er geen kan
    verzinnen. De enum bewaakt het vocabulaire, niet de volledigheid — een
    vergeten of dubbel geplaatst attribuut vangt `repair_partition` af."""
    tags: Tuple[str, ...] = tuple(card.tag for card in _shuffled(cards))
    constrained_code = create_model(
        "ConstrainedProposedCode",
        __base__=ProposedCode,
        topics=(List[Literal[tags]], Field(
            ..., description=ProposedCode.model_fields["topics"].description)),
    )
    return create_model(
        "ConstrainedConsolidationResult",
        __base__=ConsolidationResult,
        codes=(List[constrained_code], Field(
            ..., description=ConsolidationResult.model_fields["codes"].description)),
    )


def build_consolidation_prompt(
    cards, survey_question: str, n_respondents: int, language: str,
) -> str:
    blocks = []
    for card in _shuffled(cards):
        answers = ", ".join(f"{text} ({n})" for text, n in card.top_answers) or "—"
        blocks.append(
            f'"{card.tag}" — {card.n_resp} respondents\n'
            f"    meaning: {card.definition}\n"
            f"    answers: {answers}\n"
            f"    listed under: {card.domain} > {card.facet}"
        )
    inventory = "\n\n".join(blocks)

    return f"""These are the topics {n_respondents} respondents raised in answer to
this open-ended survey question:

"{survey_question}"

A codebook has to come out of this — the kind a researcher uses in a report. Every
code becomes one row in a table with a percentage next to it. Someone reading that
row has to understand what respondents actually said, without having to consult the
rest of the codebook.

Decide which topics form one code together.

- A topic that is raised often enough on its own AND means something of its own
  stays a code of its own.
- Topics that say too little apart belong together — but only if what you end up
  with is still ONE thing you can explain in a single sentence.
- A group you can only explain by listing what is in it is not a code. If your
  sentence needs "and" to join two unrelated things, the group is wrong.

Every topic goes in exactly one code. Do not leave any out, do not place any twice.
Write `code_name` and `explanation` in {language}.

Topics:

{inventory}

{INSTRUCTOR_HINT}"""
