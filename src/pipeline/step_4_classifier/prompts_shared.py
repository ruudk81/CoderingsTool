"""Shared building blocks for the step-4 prompts.

Every step-4 prompt is assembled from the same four blocks, in this order:

  1. context            build_context_block()
  2. the taxonomy       build_taxonomy_block()
  3. the task           the caller writes this, using level_diagnostic()
  4. rules + output     the caller writes this, ending on UNIVERSAL_RULES
                        and INSTRUCTOR_HINT

The shape is taken from step 3, which builds the domain layer (L2) the same
way. The one thing step 4 adds is that the level being worked on varies, so
the diagnostic question has to be selected rather than fixed: step 3 always
asks `domain_diagnostic`, step 4 asks `facet_diagnostic` or
`attribute_diagnostic` depending on which layer the phase is building.

## Two things called a dimension, and why only one of them is

L1 is the **lens**: the one perspective the whole study reads every response
through, fixed in `dimension_data.py` and named in the taxonomy block. The code
still calls it `dimension_name`, because that is what `ExtractionMetadata`
carries across the step boundary — but no prompt calls it that, or the word
would mean two things on one page.

The **dimensions** the prompts ask for are the ways responses inside one scope
differ from each other: discovery finds them per domain and per facet, and the
facets respectively attributes are the values they take. They are a construction
aid, not a taxonomy level — the parse flattens them away and the cache never
sees them. Their whole job is to force the model to say what its list is a list
OF before it makes the list.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# The sentence instructor needs
# =============================================================================

# Field(description=...) on the response model is not enough on its own: without
# this sentence at the end of the prompt a large share of calls comes back
# unparseable. Every builder ends on it.
INSTRUCTOR_HINT = (
    "provide your output as valid JSON following the response schema provided"
)


# =============================================================================
# Rules that hold at every level
# =============================================================================

UNIVERSAL_RULES = """<universal_rules>
These three rules hold at every level of the taxonomy and for every decision you make here.

1. DESCRIPTIVE, NEVER EVALUATIVE.
   Every name and definition you return states WHAT is being referred to, never how good
   or bad it is. Even when every response in a group points the same way, the label names
   the subject, not the judgment. Evaluative direction is recorded separately, per
   response, as valence — never inside the taxonomy. If a candidate reads as a verdict,
   restate it as the subject being judged.

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


def level_diagnostic(dimension: "DimensionDefinition", level: str) -> str:
    """The question every item at `level` has to answer, for this dimension.

    `dimension_data.py` carries one diagnostic per level: `domain_diagnostic`,
    `facet_diagnostic`, `attribute_diagnostic`. Step 3 uses the first; step 4
    uses the other two. Passing an unknown level is a programming error, not
    something to paper over with a fallback — a phase that silently builds its
    prompt around the wrong question produces plausible output that is wrong
    all the way down.
    """
    rules = dimension.prompt_rules
    if level == "facet":
        return rules.facet_diagnostic
    if level == "attribute":
        return rules.attribute_diagnostic
    raise ValueError(
        f"level must be 'facet' or 'attribute', got {level!r}. "
        f"The domain level belongs to step 3."
    )


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
) -> str:
    """The survey context, in step 3's shape."""
    return f"""Here is the language the survey responses are written in:
<language>
{language}
</language>

Here is the survey question that was asked in {language}:
<survey_question>
{survey_question}
</survey_question>

Here is contextual information from prior analysis:
<context>
- Sector: {sector}
- Entity of interest: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>"""


# =============================================================================
# Block 2 — the taxonomy
# =============================================================================

def build_taxonomy_block(
    *,
    dimension: "DimensionDefinition",
    dimension_name: str,
    dimension_description: str,
) -> str:
    """All four levels, each described in this dimension's own words.

    Shown in full even though a phase only builds one level: an item that
    belongs one level up or one level down is the most common failure, and it
    is only recognisable against the neighbouring levels.
    """
    rules = dimension.prompt_rules
    return f"""<taxonomy_structure>
The taxonomy has four levels. All four are given so you can see where your own task
sits, and so you do not return something that belongs one level up or one level down.

L1 — Lens: {dimension_name}
     The lens the whole study looks through. Fixed for every response, every
     domain and every level below.
     {dimension_description}

L2 — Domain: {_extract_definition(rules.domain_instruction)}
     Key idea: {_extract_key_idea(rules.domain_instruction)}

L3 — Facet: {_extract_definition(rules.facet_instruction)}
     Key idea: {_extract_key_idea(rules.facet_instruction)}

L4 — Attribute: {_extract_definition(rules.attribute_instruction)}
     Key idea: {_extract_key_idea(rules.attribute_instruction)}
</taxonomy_structure>"""
