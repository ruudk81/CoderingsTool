"""Assignment: one label, one attribute, and with it the attribute's facet.

This used to be two gates. An idea was given a facet first, then an attribute
inside that facet. Anything that stranded on the first gate got `__UNASSIGNED__`
and by definition no attribute either — a name that was not in the structure, and
that everything downstream had to make an exception for.

Now there is one gate. The menu spans the domain but stays grouped by facet, so
the model sees the structure it is choosing within, and the facet follows from
the chosen attribute instead of being determined separately.

**The menu always has an exit.** Every facet carries an `other` attribute, and
the domain's `other` facet sits at the bottom with an attribute of its own. So
there is always a valid answer, and no second way of choosing nothing is needed —
that one kept coming back as `__UNASSIGNED__`.

Both carry the `[CATCH-ALL]` marker in the menu, and the prompt points at that
marker. Pointing at their name is impossible: the name is in the survey language.

**One call per unique normalised label**, not per idea instance: identical text
shares one judgement. This is not a batch — the model sees one label and returns
one attribute.

This module also builds the facet menu, response model and prompt for the
separate, earlier facet-assignment phase — see the FACET ASSIGNMENT section
below. That phase exists so facet consolidation can judge candidate facets by
real idea counts instead of by how many text chunks proposed a name during
discovery.
"""
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
    """
    blocks: List[str] = []
    id_map: Dict[str, Dict[str, Any]] = {}
    counter = 0

    for facet in facets:
        attributes = facet.get("attributes") or []
        if not attributes:
            continue
        facet_tag = "  [CATCH-ALL]" if is_drain_item(facet) else ""
        lines = [f"Facet: {facet['facet_name']} — "
                 f"{facet['facet_definition']}{facet_tag}"]
        for attribute in attributes:
            counter += 1
            attribute_id = f"A{counter}"
            id_map[attribute_id] = {
                "facet_name": facet["facet_name"],
                "attribute_name": attribute["attribute_name"],
                "is_drain": is_drain_item(attribute),
            }
            tag = "  [CATCH-ALL]" if is_drain_item(attribute) else ""
            lines.append(f"  [{attribute_id}] {attribute['attribute_name']}{tag}")
            lines.append(f"        {attribute['attribute_definition']}")
            examples = attribute.get("example_observations") or []
            if examples:
                shown = "; ".join(f'"{e}"' for e in examples[:2])
                lines.append(f"        e.g. {shown}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks), id_map


# =============================================================================
# HET RESPONSEMODEL
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


# =============================================================================
# DE PROMPT
# =============================================================================

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
# FACET ASSIGNMENT — one gate earlier, on real idea counts
# =============================================================================
#
# Facet consolidation needs to judge candidate facets by how many ideas actually
# belong to them, not by how many text chunks proposed the name during
# discovery. That count does not exist until ideas are placed in facets, so this
# phase runs before attribute assignment and produces it: one call per label,
# picking a facet id from the domain's menu.

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
    """
    blocks: List[str] = []
    id_map: Dict[str, Dict[str, Any]] = {}
    for counter, facet in enumerate(facets, start=1):
        facet_id = f"F{counter}"
        id_map[facet_id] = {
            "facet_name": facet["facet_name"],
            "is_drain": is_drain_item(facet),
        }
        tag = "  [CATCH-ALL]" if is_drain_item(facet) else ""
        line = (f"[{facet_id}] {facet['facet_name']} — "
                f"{facet['facet_definition']}{tag}")
        question = facet.get("facet_question") or ""
        if question:
            line += f"\n      The question it answers: {question}"
        blocks.append(line)
    return "\n".join(blocks), id_map


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
    """Place one observation in the facet menu of its domain.

    No taxonomy block and no `UNIVERSAL_RULES`: this phase picks an id from a
    menu inside a domain that is already fixed and invents no name, the same
    exception documented in `prompts_shared.py` for attribute assignment. The
    facet question in the menu already carries, concretely and derived from
    this dataset, what the taxonomy block would say abstractly.
    """
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
2. An observation that answers none of the questions belongs in the facet marked [CATCH-ALL]. That is a valid outcome, not a failure — forcing it into a facet it does not answer costs more than leaving it there.
3. Judge only this observation. What other observations do is another call.

{INSTRUCTOR_HINT}"""
