"""Shared building blocks for the step-4 prompts.

  1. context            build_context_block()
  2. the taxonomy       build_taxonomy_block()
  3. the task           the caller writes this
  4. rules + output     the caller writes this, ending on UNIVERSAL_RULES
                        and INSTRUCTOR_HINT
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal

from pydantic import Field, create_model

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# The sentence instructor needs
# =============================================================================

INSTRUCTOR_HINT = ("provide your output as valid JSON following the response schema provided")


# =============================================================================
# Rules that hold at every level
# =============================================================================

UNIVERSAL_RULES = """<universal_rules>
These four rules hold at every level of the taxonomy and for every decision you make here.

1. DESCRIPTIVE, NEVER EVALUATIVE.
   Every name and definition you return states WHAT is being referred to, never how good
   or bad it is. Even when every response in a group points the same way, the label names
   the subject, not the judgment. Evaluative direction is recorded separately, per
   response, as valence — never inside the taxonomy. If a candidate reads as a verdict,
   restate it as the subject being judged.
   
   REVERSAL TEST, on every name you return: would a response expressing the opposite
   direction still belong under it? A name that only fits one direction has taken a side,
   even when it looks like a plain noun — and responses pointing the other way will be
   pushed elsewhere or dropped. Rename it to the property being judged, so that both
   directions sit under it and valence tells them apart.

2. NEVER SPLIT ONE CONCEPT BY EVALUATIVE DIRECTION.
   Do not return two items that differ only in evaluative direction — a positive and a
   negative version of the same thing. Capture the concept ONCE; the direction is carried
   by valence. A response that is nothing but an overall judgment, with no descriptive
   content at all, belongs to a single residual overall-judgment item — never to a
   positive one and a negative one.

3. NO BORROWED EXAMPLES.
   Ground every judgment in the material in front of you. Do not import examples,
   category names, or rules of thumb from other studies, other sectors, or from general
   knowledge about surveys of this kind. The observations, definitions and boundaries in
   this prompt are the whole evidence base.

4. NEVER CREATE A LEFTOVER CATEGORY.
   Every item you return is defined by what its responses have IN COMMON, never by what
   they lack. Do not return an item whose real definition is "the ones that fit nowhere
   else" — names like "Other", "Various", "Miscellaneous", "Remaining" are the signature,
   and so is a definition that describes the item by exclusion.
   
   If a group of responses shares nothing statable, it does not become a category: leave
   those responses out of your grouping and let them fall through. A residual bucket is
   provided for them elsewhere, and one you invent here would compete with it.
   This is not a ban on abstraction. "Overall judgment" is a real, statable thing — those
   responses have something in common, namely that they judge without naming a subject.
   The test is whether you can say what the members ARE without using the word "other".
</universal_rules>"""


# =============================================================================
# Reading dimension_data
# =============================================================================

def _extract_definition(instruction: str) -> str:
    """Return the 'Definition: ...' sentence from a prompt_rules instruction."""
    marker = "Definition: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    rest = instruction[idx + len(marker):]
    newline = rest.find("\n")
    if newline != -1:
        rest = rest[:newline]
    return rest.strip()


def _extract_key_idea(instruction: str) -> str:
    """Return the 'Key idea: ...' sentence from a prompt_rules instruction."""
    marker = "Key idea: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    return instruction[idx + len(marker):].strip().rstrip(".")


# =============================================================================
# Block 1 — context
# =============================================================================

def build_context_block(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    dimension_name: str = "",
    dimension_description: str = "",
) -> str:
    dimension_lines = f"""
The responses vary within this dimension:
<dimension>
Dimension name: {dimension_name}
Clarification: {dimension_description}
</dimension>""" if dimension_name else ""
    return f"""Here is the language the survey responses are written in:
<language>
{language}
</language>

The respondents answered the following survey question (in {language}):
<survey_question>
{survey_question}
</survey_question>
{dimension_lines}

The following contextual information helps clarify how the survey question should be understood:
<context>
- Sector: {sector}
- Entity of interest: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}

Use this context to interpret the survey question and ambiguous responses.
Do not infer content that is not supported by the responses themselves
</context>"""

# =============================================================================
# Block 2 — the taxonomy
# =============================================================================

def build_dimension_block(
    *,
    dimension: "DimensionDefinition",
    dimension_name: str,
    dimension_description: str,
) -> str:
    rules = dimension.prompt_rules
    return f"""<taxonomy_dimension>
Dimension name: {dimension_name}
Clarification: {dimension_description}
</taxonomy_dimension>"""

def build_taxonomy_block_L3(
    *,
    dimension: "DimensionDefinition",
    dimension_name: str,
    dimension_description: str,
) -> str:
    rules = dimension.prompt_rules
    return f"""<taxonomy_structure>
L1 — Domain: {_extract_definition(rules.domain_instruction)} Key idea: {_extract_key_idea(rules.domain_instruction)}
L2 — Facet: {_extract_definition(rules.facet_instruction)} Key idea: {_extract_key_idea(rules.facet_instruction)}
L3 — Attribute: {_extract_definition(rules.attribute_instruction)} Key idea: {_extract_key_idea(rules.attribute_instruction)}
</taxonomy_structure>"""

def build_taxonomy_block(
    *,
    dimension: "DimensionDefinition",
    dimension_name: str,
    dimension_description: str,
) -> str:
    rules = dimension.prompt_rules
    return f"""<taxonomy_structure>
L1 — Dimension: {dimension_name}
     The kind of information every response in this study is read as. Fixed for
     every response, every domain and every level below.
     {dimension_description}

L2 — Domain: {_extract_definition(rules.domain_instruction)}
     Key idea: {_extract_key_idea(rules.domain_instruction)}

L3 — Facet: {_extract_definition(rules.facet_instruction)}
     Key idea: {_extract_key_idea(rules.facet_instruction)}

L4 — Attribute: {_extract_definition(rules.attribute_instruction)}
     Key idea: {_extract_key_idea(rules.attribute_instruction)}
</taxonomy_structure>"""


def build_facets_attributes_block(*, dimension: "DimensionDefinition") -> str:
    """The two levels a facet-scoped call works in, unnumbered.

    No level numbers here: this call settles what sits under one facet and never
    has to place anything relative to a level it cannot see.
    """
    rules = dimension.prompt_rules
    return f"""<definitions>
Facet:
- {_extract_definition(rules.facet_instruction)}
- Key idea: {_extract_key_idea(rules.facet_instruction)}

Attribute:
- {_extract_definition(rules.attribute_instruction)}
- Key idea: {_extract_key_idea(rules.attribute_instruction)}
</definitions>"""


# =============================================================================
# Cross-scope consolidation — the one phase that sees more than one scope
# =============================================================================

def build_cross_scope_model(item_ids: List[str], noun: str):
    """Runtime response model for a cross-scope merge over a fixed id space.

    Works on ids, never on names: the model returns groups of input ids plus the
    id whose scope the survivor keeps. That makes relocation a choice among the
    inputs instead of free text that has to be matched back, and it makes a
    dropped id detectable rather than silent.
    """
    id_literal = Literal[tuple(item_ids)]  # type: ignore[valid-type]

    item = create_model(
        f"Merged{noun.capitalize()}",
        name=(str, Field(..., description=(
            f"Short descriptive name for the merged {noun}, in the survey "
            f"language (at most 5 words)"))),
        definition=(str, Field(..., description=(
            "One sentence naming the single aspect, in the survey language"))),
        source_ids=(List[id_literal], Field(
            ..., description=(
                f"Every input id that folds into this {noun}, including its own. "
                f"A {noun} kept unchanged lists exactly one id"))),
        home_id=(id_literal, Field(
            ..., description=(
                "The id whose scope this one keeps. Must be one of the source_ids. "
                "Pick the scope where most of these responses already sit"))),
    )
    return create_model(
        "CrossScopeResult",
        scratchpad=(str, Field(..., description=(
            f"Reasoning: (1) group the {noun}s that mean the same thing across scopes, "
            f"(2) for each group pick the scope where most of its responses sit, "
            f"(3) check every id appears exactly once"))),
        items=(List[item], Field(
            ..., description="The merged inventory. Every input id appears exactly once")),
    )
