"""De post-mortem-vraag: is dit één code, of zijn het er meerdere?

Eén groep per keer, met alles erbij wat een mens ook zou hebben: de onderwerpen
die erin zitten, hun definities, hun omvang en wat respondenten letterlijk
zeiden. Dat is dezelfde context als fase 1 krijgt, maar over een fractie van het
materiaal — daarom kan dit oordeel scherper zijn dan het oordeel dat de hele
inventaris in één keer moest indelen.

De prompt zegt NIET waarom een groep wordt voorgelegd. Zou hij melden dat een
groep "te groot" is of "wisselde tussen runs", dan is dat een duw richting
splitsen, en meet je je eigen trigger terug in plaats van een oordeel over de
inhoud. De selectie bepaalt wélke groepen langskomen; de vraag blijft neutraal.

Het antwoord is per onderwerp een deelnummer, niet een lijst van lijsten: zo kan
het model geen onderwerp half toewijzen, en is een vergeten onderwerp meteen
zichtbaar in plaats van verstopt in een geneste structuur.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

from pydantic import BaseModel, Field, create_model

from .prompts_common import INSTRUCTOR_HINT, _shuffled


class TopicPart(BaseModel):
    topic: str = Field(..., description="One of the topics in this group")
    part: int = Field(
        ..., description=(
            "Which part this topic belongs to: 1 for the first part, 2 for the "
            "second, and so on. Give every topic in the group the same number if "
            "they all belong together."
        ),
    )


class GroupVerdict(BaseModel):
    group: str = Field(..., description="The group this verdict is about")
    reasoning: str = Field(
        default="", description="One sentence: what distinguishes the parts, or why they belong together"
    )
    assignments: List[TopicPart] = Field(
        ..., description="Exactly one entry per topic in this group"
    )


class PostMortemResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the verdicts")
    verdicts: List[GroupVerdict] = Field(..., description="Exactly one verdict per group")


def make_postmortem_model(labelled_groups) -> type:
    """`group` en `topic` beperkt tot wat er getoond is. Dat het model niets kan
    verzinnen is hier extra belangrijk: een verzonnen onderwerp zou een splitsing
    opleveren die `apply_splits` in zijn geheel afwijst, en dan is de call weg."""
    group_labels: Tuple[str, ...] = tuple(label for label, _cards in labelled_groups)
    topic_tags: Tuple[str, ...] = tuple(
        card.tag for _label, cards in labelled_groups for card in cards
    )
    constrained_part = create_model(
        "ConstrainedTopicPart",
        __base__=TopicPart,
        topic=(Literal[topic_tags], Field(
            ..., description=TopicPart.model_fields["topic"].description)),
    )
    constrained_verdict = create_model(
        "ConstrainedGroupVerdict",
        __base__=GroupVerdict,
        group=(Literal[group_labels], Field(
            ..., description=GroupVerdict.model_fields["group"].description)),
        assignments=(List[constrained_part], Field(
            ..., description=GroupVerdict.model_fields["assignments"].description)),
    )
    return create_model(
        "ConstrainedPostMortemResult",
        __base__=PostMortemResult,
        verdicts=(List[constrained_verdict], Field(
            ..., description=PostMortemResult.model_fields["verdicts"].description)),
    )


def build_postmortem_prompt(labelled_groups, survey_question: str,
                            n_respondents: int, language: str) -> str:
    blocks = []
    for label, cards in labelled_groups:
        topics = []
        for card in _shuffled(list(cards)):
            answers = ", ".join(f"{text} ({n})" for text, n in card.top_answers) or "—"
            topics.append(
                f'    "{card.tag}" — {card.n_resp} respondents\n'
                f"        meaning: {card.definition}\n"
                f"        answers: {answers}"
            )
        blocks.append(f'GROUP "{label}"\n' + "\n".join(topics))
    inventory = "\n\n".join(blocks)

    return f"""A codebook was built from the answers {n_respondents} respondents gave to
this open-ended survey question:

"{survey_question}"

Each code becomes one row in a report table with a percentage next to it. Someone
reading that row has to understand what respondents actually said, without
consulting the rest of the codebook.

Below are groups of topics that were each put together as ONE code. For every
group, decide whether that is right.

- If the topics in a group really are one thing — something you can explain in a
  single sentence without listing them — give every topic in that group part 1.
- If the group actually covers more than one thing, divide the topics into parts.
  Each part must be one thing you can explain in a single sentence. Topics that
  belong together go in the same part.

Rules:
- Every topic listed under a group gets exactly one part number. Do not leave any
  out and do not move a topic to a different group.
- Only divide where the difference would matter to someone reading the table. A
  part that no reader would tell apart from another part is not worth separating.
- A part may contain a single topic if that topic genuinely stands on its own.

Write `reasoning` in {language}.

{inventory}

{INSTRUCTOR_HINT}"""
