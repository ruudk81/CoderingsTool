"""Assignment prompts voor step 4."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

from pydantic import Field, create_model

from .drains import is_drain_item
from .prompts_shared import INSTRUCTOR_HINT, build_context_block


# =============================================================================
# HET MENU
# =============================================================================

def build_assignment_menu(
    facets: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """The domain-wide menu, grouped by facet, plus the id -> choice map.

    Ids run on across facet boundaries: the model picks an attribute, not a facet
    plus an attribute. The map is what the parse needs to get from an id back to
    (facet, attribute) — a choice among the inputs rather than free text that has
    to be matched back.

    A facet without attributes drops out. Assignment picks an attribute, so such
    a facet would be a line in the menu that nobody can choose.

    Catch-alls carry `[CATCH-ALL]`, as in `build_contents_block`. Their name is in
    the survey language — `Overig`, `Other`, or whatever the next language makes
    of it — so the prompt cannot point at the name; it can point at the marker.

    An attribute's boundary rules come last, after its examples: the definition
    says what it is and the examples show it, and only then is there something to
    tell apart. They are the one line here written about a pair rather than about
    a single item, which is why a definition can be restated and a boundary
    cannot. Attributes that never went through consolidation carry none.

    One labelled line per property, as in `build_facet_menu`. Two things differ,
    and both because this menu is the only nested one: the attributes stay
    indented under their facet, and the gaps are graded — one blank line between
    attributes, two between facets — so that the deeper break stays the wider
    one. Flush-left entries with an even gap would read as one flat list of
    attributes with facet headings scattered through it.
    """
    blocks: List[str] = []
    id_map: Dict[str, Dict[str, Any]] = {}
    counter = 0

    for facet in facets:
        attributes = facet.get("attributes") or []
        if not attributes:
            continue
        header = [f"Facet: {facet['facet_name']}",
                  f"Definition: {facet['facet_definition']}"]
        if is_drain_item(facet):
            header.append("[CATCH-ALL]")
        entries = ["\n".join(header)]

        for attribute in attributes:
            counter += 1
            attribute_id = f"A{counter}"
            id_map[attribute_id] = {
                "facet_name": facet["facet_name"],
                "attribute_name": attribute["attribute_name"],
                "is_drain": is_drain_item(attribute),
            }
            lines = [f"  [{attribute_id}] {attribute['attribute_name']}",
                     f"  Definition: {attribute['attribute_definition']}"]
            examples = attribute.get("example_observations") or []
            if examples:
                shown = "; ".join(f'"{e}"' for e in examples[:2])
                lines.append(f"  e.g. {shown}")
            for rule in attribute.get("boundary_rules") or []:
                lines.append(f"  Boundary: {rule}")
            if is_drain_item(attribute):
                lines.append("  [CATCH-ALL]")
            entries.append("\n".join(lines))

        blocks.append("\n\n".join(entries))

    return "\n\n\n".join(blocks), id_map

# =============================================================================
# ATTRIBUTE  
# =============================================================================

def build_assignment_model(attribute_ids: List[str]):
    """Runtime model in which the menu is the id space.

    The menu sits in the schema as a `Literal`, so an invented id is a schema
    error that instructor retries — instead of a content error that surfaces
    three phases later as an attribute nobody knows.
    """
    id_literal = Literal[tuple(attribute_ids)]  # type: ignore[valid-type]
    return create_model(
        "AssignmentResult",
        assigned_attribute_id=(id_literal, Field(..., description=(
            "The id from the [A#] prefix of the single best-fitting attribute. "
            "Return only the id, not the name"))),
        confidence=(float, Field(..., ge=0.0, le=1.0, description=(
            "How certain the assignment is, from 0.0 to 1.0"))),
        valence=(Literal["+", "-", "0"], Field(..., description=(
            "Evaluative direction relative to the chosen attribute: "
            "+ positive, - negative, 0 neutral or descriptive"))),
    )

def build_assignment_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    domain_label: str,
    domain_definition: str,
    menu_block: str,
    label: str,
) -> str:
    """Place one label in the menu of its domain."""
    return f"""You are a qualitative coding assistant. Assign the survey response below to
the single attribute that best captures what it refers to.

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

The response has already been placed in this domain:

<domain_context>
{domain_label} — {domain_definition}
</domain_context>

Here are the attributes available in this domain, grouped under the facet each one belongs
to. You are choosing an attribute; the facet follows from your choice.

<menu>
{menu_block}
</menu>

Here is the response to place:

<response>
{label}
</response>

# Choosing

Pick exactly one attribute — the one that names what this response actually refers to. Read
the definitions, not only the names. If two seem to fit, choose the more specific one.

Some attributes carry a line beginning `Boundary:`. That line names another attribute and
states what decides between the two. Where it applies to this response, it settles the
choice: it was written by the call that separated those two attributes, which had all their
material in view. Follow it over your own reading of the two definitions.

Some options are marked [CATCH-ALL]: one at the end of every facet, and one whole facet at
the end of the menu for the domain as a whole. These are real answers, for a response that
belongs here but that none of the named attributes covers. Use them as a last resort, never
as a way to avoid reading the menu: if a named attribute fits, that attribute is the answer.

# Valence

Record the evaluative direction of this response relative to the attribute you chose:
- "+" the response describes the attribute as present, sufficient, or meeting expectations
- "-" the response describes it as absent, insufficient, or failing expectations
- "0" the response is descriptive, ambiguous, or expresses no evaluation

Valence is not emotional sentiment. It is direction relative to the attribute, and it is
recorded here precisely so the taxonomy itself never has to encode it.

{INSTRUCTOR_HINT}"""


# =============================================================================
# FACET ASSIGNMENT 
# =============================================================================


def build_facet_menu(
    facets: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """The domain-wide facet menu, plus the id -> facet map.

    Names are not unique — two facets of one domain may carry the same name —
    so the choice is an id, and the map is what the parse needs to get back to
    the facet that was meant.

    The attribute pool is not rendered. It is still unconsolidated at this
    point, and tens of near-identical names per facet would make the menu
    unreadable. The facet question is the signal that makes a facet's identity
    testable.

    The boundary rules are what the question cannot be. Name, definition and
    question are each written about one facet in isolation, so all three can say
    the same thing in different words; a boundary rule has to name a sibling and
    is therefore the only line in the menu that carries a comparison. A facet
    that had no near neighbour has none, and a catch-all never has any.

    One labelled line per property, one blank line between facets, no indent:
    nothing here is nested — this menu is a flat list of facets, unlike the
    attribute menu, whose indentation carries the facet each attribute sits
    under. `[CATCH-ALL]` gets its own line for the same reason the others do.
    """
    blocks: List[str] = []
    id_map: Dict[str, Dict[str, Any]] = {}
    for counter, facet in enumerate(facets, start=1):
        facet_id = f"F{counter}"
        id_map[facet_id] = {
            "facet_name": facet["facet_name"],
            "is_drain": is_drain_item(facet),
        }
        lines = [f"[{facet_id}] {facet['facet_name']}",
                 f"Definition: {facet['facet_definition']}"]
        question = facet.get("facet_question") or ""
        if question:
            lines.append(f"The question it answers: {question}")
        for rule in facet.get("boundary_rules") or []:
            lines.append(f"Boundary: {rule}")
        if is_drain_item(facet):
            lines.append("[CATCH-ALL]")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), id_map


def build_facet_assignment_model(facet_ids: List[str]):
    """Runtime model in which the facet menu is the id space.

    No valence field: valence is judged relative to the chosen attribute, which
    this phase does not choose yet. Asking for it here would invent a judgment
    that only the later, attribute-scoped call can actually make.
    """
    id_literal = Literal[tuple(facet_ids)]  # type: ignore[valid-type]
    return create_model(
        "FacetAssignmentResult",
        assigned_facet_id=(id_literal, Field(..., description=(
            "The id from the [F#] prefix of the single best-fitting facet. "
            "Return only the id, not the name"))),
        confidence=(float, Field(..., ge=0.0, le=1.0, description=(
            "How certain the assignment is, from 0.0 to 1.0"))),
    )


def build_facet_assignment_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    domain_label: str,
    domain_definition: str,
    menu_block: str,
    observation: str,
) -> str:
    return f"""You are a taxonomy classification specialist for surveys.
Pick the one facet this observation belongs to.

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

You are working inside this domain:
<taxonomy_domain>
{domain_label} — {domain_definition}
</taxonomy_domain>

These are the facets of this domain. Pick exactly one by its id:
<facets>
{menu_block}
</facets>

This is the observation to place:
<observation>
{observation}
</observation>

# Rules

1. Pick the facet whose question this observation answers. The question is the test, not the name: an observation may mention words from a facet's name and still answer a different question.
2. Some facets carry a line beginning `Boundary:`. That line names another facet and states what decides between the two. Where it applies to this observation, it settles the choice: it was written by the call that separated those two facets, which had all their material in view. Follow it over your own reading of the two questions.
3. An observation that answers none of the questions belongs in the facet marked [CATCH-ALL]. That is a valid outcome, not a failure — forcing it into a facet it does not answer costs more than leaving it there.
4. Judge only this observation. What other observations do is another call.

{INSTRUCTOR_HINT}"""
