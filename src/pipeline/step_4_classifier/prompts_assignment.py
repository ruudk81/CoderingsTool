"""Toewijzing: één label, één attribuut, en daarmee ook zijn facet.

Hiervoor waren dit twee poorten. Een idee kreeg eerst een facet en daarna,
binnen dat facet, een attribuut. Wie op de eerste poort strandde kreeg
`__UNASSIGNED__` en daarmee per definitie ook geen attribuut — een naam die
niet in de structuur stond, waar alles stroomafwaarts een uitzondering voor
moest maken.

Nu is er één poort. Het menu is domeinbreed maar per facet gegroepeerd, zodat
het model de structuur ziet waarbinnen het kiest, en het facet volgt uit het
gekozen attribuut in plaats van er los van te worden bepaald.

**Het menu heeft altijd een uitgang.** Onder elk facet staat een `Overig`, en
onderaan staat het `Overig`-facet van het domein met zijn eigen attribuut. Er is
dus altijd een geldig antwoord, en er hoeft geen tweede manier te zijn om niets
te kiezen — die kwam als `__UNASSIGNED__` telkens terug.

**Eén call per uniek genormaliseerd label**, niet per idee-instantie: identieke
tekst deelt één oordeel. Dat is geen batch — het model ziet één label en geeft
één attribuut.
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
    """Het domeinbrede menu, per facet gegroepeerd, plus de id→keuze-map.

    Ids lopen door over facetgrenzen heen: het model kiest een attribuut, niet
    een facet plus een attribuut. De map is wat de parse nodig heeft om van id
    terug naar (facet, attribuut) te komen — een keuze tussen de invoer in
    plaats van vrije tekst die teruggematcht moet worden.

    Een facet zonder attributen valt weg. Toewijzing kiest een attribuut, dus
    zo'n facet zou een regel in het menu zijn die niemand kan kiezen.
    """
    blocks: List[str] = []
    id_map: Dict[str, Dict[str, Any]] = {}
    counter = 0

    for facet in facets:
        attributes = facet.get("attributes") or []
        if not attributes:
            continue
        lines = [f"Facet: {facet['facet_name']} — {facet['facet_definition']}"]
        for attribute in attributes:
            counter += 1
            attribute_id = f"A{counter}"
            id_map[attribute_id] = {
                "facet_name": facet["facet_name"],
                "attribute_name": attribute["attribute_name"],
                "is_drain": is_drain_item(attribute),
            }
            lines.append(f"  [{attribute_id}] {attribute['attribute_name']}")
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
    """Runtime model waarin het menu de id-ruimte is.

    Het menu ligt als `Literal` in het schema, dus een verzonnen id is een
    schemafout die instructor overdoet — in plaats van een inhoudsfout die drie
    fasen verderop opduikt als een attribuut dat niemand kent.
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
    """Plaats één label in het menu van zijn domein."""
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

Each facet ends with an "Overig" option, and the menu ends with an "Overig" facet for the
domain as a whole. These are real answers, for a response that belongs here but that none
of the named attributes covers. Use them as a last resort, never as a way to avoid reading
the menu: if a named attribute fits, that attribute is the answer.

# Valence

Record the evaluative direction of this response relative to the attribute you chose:
- "+" the response describes the attribute as present, sufficient, or meeting expectations
- "-" the response describes it as absent, insufficient, or failing expectations
- "0" the response is descriptive, ambiguous, or expresses no evaluation

Valence is not emotional sentiment. It is direction relative to the attribute, and it is
recorded here precisely so the taxonomy itself never has to encode it.

# Output

Return a JSON object with:
- `assigned_attribute_id`: the id from the [A#] prefix, and nothing else
- `confidence`: how certain you are, from 0.0 to 1.0
- `valence`: "+", "-" or "0"

{INSTRUCTOR_HINT}"""
