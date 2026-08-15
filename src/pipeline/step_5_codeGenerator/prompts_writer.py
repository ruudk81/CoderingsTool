"""Step 4 — writing it down. The shape is already fixed: how many codes, and
which direction.

This only fills in the texts — name, definition, diagnostic test, indicators, a
boundary note against the nearest neighbouring code, and the one veto this step
may still exercise: `nameable`. Per code the prompt shows every member attribute
as name + definition (never a name alone — a pooled or MECE-merged shape can
carry ten members or more, and a bare list of names gives the model too little
ground to write a name and definition that fit the actual contents) together with
the direction already decided. Respondent counts, domain, facet and attribute ids
are judgement-irrelevant and appear nowhere in the prompt or the response model.
The direction is shown — it has already been decided and must be respected, not
re-derived.
"""
from __future__ import annotations

from collections import namedtuple
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, create_model

from .prompts_relations import INSTRUCTOR_HINT, _shuffled

_Keyed = namedtuple("_Keyed", ["attribute_id", "shape"])


def _ordered(shapes) -> List[object]:
    """Shapes in the shared deterministic, non-prevalence order. `_shuffled` is
    keyed on `.attribute_id`; a code shape carries `.key` instead, so each shape
    is wrapped in a throwaway holder that exposes `.attribute_id` for it, rather
    than copying the hashing logic."""
    keyed = [_Keyed(attribute_id=shape.key, shape=shape) for shape in shapes]
    return [entry.shape for entry in _shuffled(keyed)]


def _members(shape, concept_by_id) -> List[object]:
    return [concept_by_id[member_id] for member_id in shape.members if member_id in concept_by_id]


def _code_block(shape, concept_by_id) -> str:
    """Renders each member as its own name + definition line, not a flat name
    list — a pooled or MECE-merged shape can carry a dozen members, and a bare
    list of short attribute names gives the model too little to ground a name
    and definition in what the shape actually contains."""
    members = _members(shape, concept_by_id)
    lines = [f"  - {m.name}: {m.definition}" for m in members]
    body = "\n".join(lines) if lines else "  - —"
    return f"[{shape.key}] direction: {shape.valence}\n  Members:\n{body}"


class CodeText(BaseModel):
    key: str = Field(..., description="Which fixed code this text belongs to")
    code_name: str = Field(
        ..., description="Short code name (3-5 words), in the survey language"
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
    nameable: bool = Field(
        True, description=(
            "False only if the topics grouped under this code share nothing "
            "that can be named honestly. True for every code by default."
        )
    )


class WriterResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the codes")
    codes: List[CodeText] = Field(
        ..., description="Exactly one entry per code in the list"
    )


def make_writer_model(shapes) -> type:
    """WriterResult with `key` restricted to the shapes on offer, so the LLM can
    neither invent one nor skip one."""
    keys: Tuple[str, ...] = tuple(shape.key for shape in _ordered(shapes))
    constrained_code_text = create_model(
        "ConstrainedCodeText",
        __base__=CodeText,
        key=(Literal[keys], Field(
            ..., description=CodeText.model_fields["key"].description
        )),
    )
    return create_model(
        "ConstrainedWriterResult",
        __base__=WriterResult,
        codes=(List[constrained_code_text], Field(
            ..., description=WriterResult.model_fields["codes"].description
        )),
    )


def build_writer_prompt(
    shapes, concept_by_id, dimension_diagnostic: str, language: str,
    taken_names: Optional[List[str]] = None,
) -> str:
    inventory = "\n".join(_code_block(shape, concept_by_id) for shape in _ordered(shapes))

    taken_block = ""
    if taken_names:
        names = "\n".join(f"- {name}" for name in sorted(set(taken_names)))
        taken_block = f"""

These code names are already used elsewhere in this codebook and are OFF LIMITS —
do not reuse any of them for a code below, even if it would otherwise fit:
{names}"""

    return f"""You are writing the final codebook. The set of codes is already fixed: there are
{len(shapes)} of them, each with its direction already decided from the data. Write the
text for each one. Do NOT add, remove or merge codes.

For every code, write:
- a name (3-5 words) in {language}
- a definition: one interpretive sentence that reads like an analyst conclusion
- a diagnostic test completing this stem: "{dimension_diagnostic}"
- typical indicators: words or phrases that signal this code
- a boundary note against the nearest competing code in this list

Two rules:
1. A code name must not claim territory another code already owns — neither
   another code in the list below, nor one of the already-taken names listed
   further down, if any. If one code covers a specific topic and another
   covers the broader family it belongs to, name the broader one for what is
   left, not for the family.
2. If the source topics grouped under a code share nothing you can name honestly,
   set nameable to false. Do not invent an umbrella term to cover unrelated items.

Codes:
{inventory}
{taken_block}

{INSTRUCTOR_HINT}"""
