"""
Facet definitions and type system for step_3_ideaExtractor v4.

All data is expressed as frozen dataclasses. No nested dicts, no .get() chains.
Access a facet with: get_facet("EVALUATION_JUDGMENT") — KeyError if not found.

Replaces: template_lookup.py (TEMPLATE_LOOKUP nested dict)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


# ========================================================================
# Data structures
# ========================================================================

@dataclass(frozen=True)
class SlotDefinition:
    """A single slot in a template pattern (e.g., ANCHOR_SUBJECT)."""
    name: str           # e.g., "ANCHOR_SUBJECT" or "EVALUATION_JUDGMENT"
    type_name: str      # e.g., "noun_phrase" or "noun_like_phrase"
    required: bool
    guidance: str       # Contains {language} for runtime substitution


@dataclass(frozen=True)
class PromptRules:
    """Facet-specific instructions for taxonomy field extraction."""
    instance_instruction: str
    node_instruction: str
    concept_type_instruction: str


@dataclass(frozen=True)
class FacetDefinition:
    """Complete definition for one primary facet."""
    key: str
    noun_phrase_descriptor: str
    dimension_description: str
    allowed_concepts: Tuple[str, ...]
    pattern: str
    instruction: str
    prompt_rules: PromptRules
    anchor_slot: SlotDefinition
    dimension_slot: SlotDefinition
    seed_examples: Tuple[str, ...]

    @property
    def dimension_marker(self) -> str:
        """The marker token for this facet, e.g., '[EVALUATION_JUDGMENT]'."""
        return f"[{self.dimension_slot.name}]"


# ========================================================================
# Type system (flat — no nesting, no indirection)
# ========================================================================

@dataclass(frozen=True)
class TypeDefinition:
    """A concrete slot type (e.g., noun_phrase)."""
    name: str
    description: str


@dataclass(frozen=True)
class TypeAlias:
    """An alias that expands to multiple concrete types."""
    name: str
    concrete_types: Tuple[str, ...]


TYPE_DEFINITIONS: Dict[str, TypeDefinition] = {
    "noun_phrase": TypeDefinition(
        name="noun_phrase",
        description="A noun phrase (1-8 tokens) naming an entity/aspect (e.g., 'appointment availability').",
    ),
    "gerund_nominal": TypeDefinition(
        name="gerund_nominal",
        description="A nominalized -ing form used as a noun (e.g., 'overcrowding').",
    ),
    "compound_noun": TypeDefinition(
        name="compound_noun",
        description="A compound noun or noun+modifier (e.g., 'wait time variability').",
    ),
    "nominalized_adjective": TypeDefinition(
        name="nominalized_adjective",
        description="An adjective-like property expressed as a noun-ish label (e.g., 'uneven quality').",
    ),
    "quality_of_construction": TypeDefinition(
        name="quality_of_construction",
        description="A 'quality of X' phrase (e.g., 'quality of communication').",
    ),
}

TYPE_ALIASES: Dict[str, TypeAlias] = {
    "noun_like_phrase": TypeAlias(
        name="noun_like_phrase",
        concrete_types=(
            "noun_phrase", "gerund_nominal", "compound_noun",
            "nominalized_adjective", "quality_of_construction",
        ),
    ),
}


def resolve_slot_type(type_name: str) -> Tuple[bool, str, str]:
    """Resolve a slot type name.

    Returns: (is_alias, short_label, description)
    Raises KeyError if the type_name is unknown.
    """
    if type_name in TYPE_ALIASES:
        alias = TYPE_ALIASES[type_name]
        descs = [TYPE_DEFINITIONS[t].description for t in alias.concrete_types]
        return (True, ", ".join(alias.concrete_types), " | ".join(descs))

    if type_name in TYPE_DEFINITIONS:
        td = TYPE_DEFINITIONS[type_name]
        return (False, td.name, td.description)

    raise KeyError(
        f"Unknown slot type: {type_name!r}. "
        f"Known types: {sorted(TYPE_DEFINITIONS)} | aliases: {sorted(TYPE_ALIASES)}"
    )


# ========================================================================
# Facet registry — all 6 facets
# ========================================================================

FACETS: Dict[str, FacetDefinition] = {

    "DEFINITION_IDENTITY": FacetDefinition(
        key="DEFINITION_IDENTITY",
        noun_phrase_descriptor="DEFINITION/IDENTITY: what it is and why it exists",
        dimension_description=(
            "The survey question concerns what the entity IS — its definition, purpose, meaning, or framing.\n"
            "Responses differ primarily in how they define, categorize, name, or frame the entity's identity, "
            "nature, or reason for existence.\n"
            "Variations typically reflect different ways of understanding or labeling what the entity fundamentally is."
        ),
        allowed_concepts=(
            "definition", "purpose", "meaning", "framing",
            "categorization", "identity", "nature", "classification",
            "naming", "essence",
        ),
        pattern="[ANCHOR_SUBJECT] → [IDENTITY_DEFINITION]",
        instruction=(
            "Identify each distinct idea about what the entity IS, how it is defined, or why it exists. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one definition/identity concept from the response.",
            node_instruction="Create a canonical, reusable noun-phrase label for this definition/identity concept. Generalize beyond the response wording.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="IDENTITY_DEFINITION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase defining or framing what the entity is, its nature, or its purpose.",
        ),
        seed_examples=(
            "Definition", "Categorization", "Naming",
            "Purpose Statement", "Mission Framing",
        ),
    ),

    "COMPOSITION_ATTRIBUTES": FacetDefinition(
        key="COMPOSITION_ATTRIBUTES",
        noun_phrase_descriptor="COMPOSITION/ATTRIBUTES: what it has and what it is like",
        dimension_description=(
            "The survey question concerns the properties, features, components, or qualities of the entity.\n"
            "Responses differ primarily in which attributes, characteristics, or compositional elements they describe.\n"
            "Variations typically reflect different properties or aspects being highlighted."
        ),
        allowed_concepts=(
            "attribute", "feature", "property", "quality",
            "component", "characteristic", "specification",
            "aspect", "capability", "structure",
        ),
        pattern="[ANCHOR_SUBJECT] → [ATTRIBUTE_COMPONENT]",
        instruction=(
            "Identify each distinct idea about what the entity HAS or what it is LIKE. "
            "For each idea, produce one concise descriptive realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one attribute or compositional concept from the response.",
            node_instruction="Create a canonical, reusable noun-phrase label for this attribute/compositional concept. Generalize beyond the response wording.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="ATTRIBUTE_COMPONENT",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a property, feature, quality, or component of the entity.",
        ),
        seed_examples=(
            "Physical Property", "Functional Feature",
            "Quality Measure", "Component", "Specification",
        ),
    ),

    "BEHAVIOR_FUNCTION": FacetDefinition(
        key="BEHAVIOR_FUNCTION",
        noun_phrase_descriptor="BEHAVIOR/FUNCTION: what it does",
        dimension_description=(
            "The survey question concerns what the entity DOES — its actions, processes, behaviors, effects, or outcomes.\n"
            "Responses differ primarily in which behaviors, functions, or activities they describe.\n"
            "Variations typically reflect different actions, processes, or functional aspects being highlighted."
        ),
        allowed_concepts=(
            "action", "process", "behavior", "function",
            "effect", "outcome", "activity", "operation",
            "performance", "service",
        ),
        pattern="[ANCHOR_SUBJECT] → [BEHAVIOR_FUNCTION]",
        instruction=(
            "Identify each distinct idea about what the entity DOES — its actions, behaviors, or functions. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one behavior or function from the response.",
            node_instruction="Create a canonical, reusable noun-phrase label for this behavior/function concept. Generalize beyond the response wording.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="BEHAVIOR_FUNCTION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing an action, process, behavior, or functional output of the entity.",
        ),
        seed_examples=(
            "Core Service", "Process Step", "Side Effect",
            "Performance Metric", "Operational Pattern",
        ),
    ),

    "CONDITIONS_CONTEXT": FacetDefinition(
        key="CONDITIONS_CONTEXT",
        noun_phrase_descriptor="CONDITIONS/CONTEXT: when, where, and why it works or fails",
        dimension_description=(
            "The survey question concerns the conditions under which the entity operates — its environment, "
            "preconditions, constraints, triggers, timing, or situational context.\n"
            "Responses differ primarily in which conditions, contexts, or circumstances they describe.\n"
            "Variations typically reflect different situational factors, constraints, or environmental aspects."
        ),
        allowed_concepts=(
            "condition", "context", "constraint", "trigger",
            "precondition", "environment", "setting", "situation",
            "timing", "circumstance",
        ),
        pattern="[ANCHOR_SUBJECT] @ [CONDITION_CONTEXT]",
        instruction=(
            "Identify each distinct idea about the conditions, contexts, or circumstances under which the entity operates. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one condition or contextual factor from the response.",
            node_instruction="Create a canonical, reusable noun-phrase label for this condition/context concept. Generalize beyond the response wording.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity, event, or topic frame in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="CONDITION_CONTEXT",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase specifying the condition, context, setting, or circumstance.",
        ),
        seed_examples=(
            "Precondition", "Environmental Factor", "Trigger",
            "Constraint", "Temporal Pattern", "Setting",
        ),
    ),

    "RELATIONS_INTERACTIONS": FacetDefinition(
        key="RELATIONS_INTERACTIONS",
        noun_phrase_descriptor="RELATIONS/INTERACTIONS: who and what it connects to",
        dimension_description=(
            "The survey question concerns how the entity relates to other entities — its stakeholders, "
            "dependencies, comparisons, influences, or interactions.\n"
            "Responses differ primarily in which relationships, actors, or connections they describe.\n"
            "Variations typically reflect different relational aspects, stakeholders, or interaction patterns."
        ),
        allowed_concepts=(
            "stakeholder", "dependency", "comparison", "influence",
            "interaction", "connection", "partnership", "competition",
            "relationship", "collaboration",
        ),
        pattern="[ANCHOR_SUBJECT] → [RELATION_INTERACTION]",
        instruction=(
            "Identify each distinct idea about how the entity relates to, interacts with, or connects to other entities. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one relationship or interaction from the response.",
            node_instruction="Create a canonical, reusable noun-phrase label for this relationship/interaction concept. Generalize beyond the response wording.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="RELATION_INTERACTION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a relationship, interaction, connection, or stakeholder involvement.",
        ),
        seed_examples=(
            "Stakeholder Link", "Dependency", "Comparison",
            "Influence Channel", "Partnership", "Competition",
        ),
    ),

    "EVALUATION_JUDGMENT": FacetDefinition(
        key="EVALUATION_JUDGMENT",
        noun_phrase_descriptor="EVALUATION/JUDGMENT: how it is assessed or what should be done",
        dimension_description=(
            "The survey question concerns how the entity is assessed, evaluated, or judged — including "
            "opinions, value judgments, recommendations, preferences, and prescriptions.\n"
            "Responses differ primarily in the type of evaluative stance taken.\n"
            "Variations typically reflect different kinds of assessments: judgments, recommendations, "
            "comparisons, risk assessments, or priorities."
        ),
        allowed_concepts=(
            "judgment", "recommendation", "preference", "opinion",
            "assessment", "comparison", "priority", "risk_assessment",
            "criticism", "praise",
        ),
        pattern="[ANCHOR_SUBJECT] → [EVALUATION_JUDGMENT]",
        instruction=(
            "Identify each distinct evaluative or judgmental idea in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one evaluation or judgment from the response.",
            node_instruction="Create a canonical, reusable noun-phrase label for this evaluation/judgment concept. Generalize beyond the response wording.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon being evaluated in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="EVALUATION_JUDGMENT",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase expressing a judgment, recommendation, preference, or evaluative stance.",
        ),
        seed_examples=(
            "Judgment", "Recommendation", "Preference",
            "Risk Assessment", "Priority", "Comparison",
        ),
    ),
}


def get_facet(facet_key: str) -> FacetDefinition:
    """Get a facet by key. Raises KeyError with a clear message if not found."""
    try:
        return FACETS[facet_key]
    except KeyError:
        raise KeyError(
            f"Unknown facet: {facet_key!r}. "
            f"Valid facets: {sorted(FACETS.keys())}"
        ) from None
