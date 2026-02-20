"""
Facet definitions and type system for step_3_ideaExtractor v5.

v5 overhaul: 10 MECE facets with decision-tree ordering (apply in order, stop at first fit).
Replaces v4's 6-facet scoring-based system.

All data is expressed as frozen dataclasses. No nested dicts, no .get() chains.
Access a facet with: get_facet("PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS") — KeyError if not found.
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
    name: str           # e.g., "ANCHOR_SUBJECT" or "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER"
    type_name: str      # e.g., "noun_phrase" or "noun_like_phrase"
    required: bool
    guidance: str       # Contains {language} for runtime substitution


@dataclass(frozen=True)
class PromptRules:
    """Facet-specific instructions for taxonomy field extraction."""
    instance_instruction: str
    concept_instruction: str
    concept_type_instruction: str


@dataclass(frozen=True)
class FacetDefinition:
    """Complete definition for one primary facet."""
    key: str
    decision_tree_position: int              # 1-10, priority order in decision tree
    criterion: str                           # Diagnostic question from decision tree
    criterion_signals: Tuple[str, ...]       # Bullet-point signals for this facet
    exclusions: Tuple[str, ...]              # "What this facet is NOT" — disambiguation cues
    noun_phrase_descriptor: str
    dimension_description: str
    allowed_concepts: Tuple[str, ...]
    pattern: str
    instruction: str
    prompt_rules: PromptRules
    anchor_slot: SlotDefinition
    dimension_slot: SlotDefinition
    clarification: Tuple[str, ...] = ()      # Optional clarification notes (empty tuple = none)

    @property
    def dimension_marker(self) -> str:
        """The marker token for this facet, e.g., '[PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER]'."""
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
# Decision tree ordering — MECE by priority
# ========================================================================

FACET_DECISION_ORDER: Tuple[str, ...] = (
    "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS",
    "IDENTITY_DEFINITION",
    "ACTORS_TARGETS",
    "CONTEXT_CONDITIONS",
    "MOTIVATIONS_DRIVERS",
    "EXPERIENCE_PERCEPTION",
    "EVALUATION_PRIORITIZATION",
    "BEHAVIOR_FUNCTION",
    "ATTRIBUTES_ASSOCIATIONS",
    "RELATIONS_DEPENDENCIES",
)


# ========================================================================
# Facet registry — all 10 facets (decision tree order)
# ========================================================================
#
# SKELETON: criterion + criterion_signals are complete from the user's schema.
# Downstream metadata (allowed_concepts, instruction,
# prompt_rules, slot definitions) are PLACEHOLDERS awaiting user input.
# ========================================================================

FACETS: Dict[str, FacetDefinition] = {

    # ── 1. PRESCRIPTIVE CHANGE ──────────────────────────────────────────
    "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS": FacetDefinition(
        key="PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS",
        decision_tree_position=1,
        criterion="Do responses mainly differ in proposed actions, improvements, or solutions?",
        criterion_signals=(
            "Concrete or abstract improvement ideas",
            "Suggestions, interventions, recommendations, feature requests",
            "Normative or forward-looking language: 'should,' 'need to,' 'would be better if'",
        ),
        exclusions=(
            "Pure judgments without proposed change",
            "Descriptions of current state",
            "Explanations of experience without action",
        ),
        noun_phrase_descriptor="PRESCRIPTIVE CHANGE & OUTCOME ENABLERS: proposed actions, improvements, or solutions",
        dimension_description=(
            "Use this facet when the dominant variation across responses is in what should be done "
            "to change, improve, fix, or enable something. Responses focus on recommendations, "
            "ideas for improvement, or ways to achieve a desired outcome."
        ),
        allowed_concepts=(
            "suggestion", "recommendation", "improvement", "solution",
            "action_proposal", "enabler", "intervention", "measure",
            "initiative", "strategy",
        ),
        pattern="[ANCHOR_SUBJECT] → [PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER]",
        instruction=(
            "Identify each distinct proposed action, improvement, or solution in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one proposed action or improvement from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this proposed action/improvement. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a proposed action, improvement, recommendation, or solution.",
        ),
    ),

    # ── 7. JUDGMENT / PRIORITIZATION ────────────────────────────────────
    "EVALUATION_PRIORITIZATION": FacetDefinition(
        key="EVALUATION_PRIORITIZATION",
        decision_tree_position=7,
        criterion="Do responses mainly differ in opinions, judgments, or preferences?",
        criterion_signals=(
            "Good vs bad, positive vs negative",
            "Preferences, rankings, comparisons",
            "Statements of importance, value, risk, or priority",
        ),
        exclusions=(
            "Proposed changes or actions",
            "Explanations of why people care (see MOTIVATIONS)",
            "Experience narratives as such",
        ),
        noun_phrase_descriptor="EVALUATION & PRIORITIZATION: opinions, judgments, or preferences",
        dimension_description=(
            "Use this facet when the dominant variation is in how respondents assess or evaluate "
            "the entity, experience, or topic — including likes/dislikes, perceived quality, "
            "importance, or comparisons."
        ),
        allowed_concepts=(
            "judgment", "preference", "opinion", "assessment",
            "comparison", "priority", "risk_assessment", "criticism",
            "praise", "ranking",
        ),
        pattern="[ANCHOR_SUBJECT] → [EVALUATION_PRIORITY]",
        instruction=(
            "Identify each distinct evaluative opinion, preference, or prioritization in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one evaluation or preference from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this evaluation/preference concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon being evaluated in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="EVALUATION_PRIORITY",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase expressing a judgment, preference, opinion, or evaluative stance.",
        ),
    ),

    # ── 6. LIVED EXPERIENCE / PERCEPTION ────────────────────────────────
    "EXPERIENCE_PERCEPTION": FacetDefinition(
        key="EXPERIENCE_PERCEPTION",
        decision_tree_position=6,
        criterion="Do responses mainly differ in how something was experienced or perceived?",
        criterion_signals=(
            "Lived experiences (positive or negative)",
            "Feelings, atmosphere, vibe, flow",
            "Holistic narratives or impressions",
        ),
        exclusions=(
            "Explicit rankings or prioritization",
            "Isolated actions or mechanics",
            "Abstract traits without experiential framing",
        ),
        noun_phrase_descriptor="EXPERIENCE & PERCEPTION: how something was experienced or perceived",
        dimension_description=(
            "Use this facet when responses vary primarily in lived experiences, feelings, "
            "impressions, or overall sense-making. The focus is on what it was like, rather "
            "than judgments, actions, or attributes."
        ),
        allowed_concepts=(
            "experience", "perception", "impression", "feeling",
            "atmosphere", "encounter", "sensation", "observation",
            "memory", "narrative",
        ),
        pattern="[ANCHOR_SUBJECT] → [EXPERIENCE_PERCEPTION]",
        instruction=(
            "Identify each distinct experience, perception, or impression described in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one experience or perception from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this experience/perception concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon experienced in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="EXPERIENCE_PERCEPTION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing an experience, perception, impression, or feeling.",
        ),
        clarification=(
            "Includes implicit evaluation when embedded in an experience narrative",
        ),
    ),

    # ── 8. ACTION / PROCESS ─────────────────────────────────────────────
    "BEHAVIOR_FUNCTION": FacetDefinition(
        key="BEHAVIOR_FUNCTION",
        decision_tree_position=8,
        criterion="Do responses mainly differ in what happens or how something works?",
        criterion_signals=(
            "Actions, processes, step-by-step descriptions",
            "How something operates or was done",
            "Events, outcomes, observable effects",
        ),
        exclusions=(
            "Evaluative framing",
            "Experiential or emotional framing",
            "Proposed improvements or changes",
        ),
        noun_phrase_descriptor="BEHAVIOR & FUNCTION: what happened or how something works",
        dimension_description=(
            "Use this facet when responses vary primarily in descriptive accounts of actions, "
            "events, processes, or how something functions or operates — reported factually, "
            "not evaluatively or experientially."
        ),
        allowed_concepts=(
            "action", "process", "behavior", "function",
            "effect", "outcome", "activity", "operation",
            "performance", "service",
        ),
        pattern="[ANCHOR_SUBJECT] → [BEHAVIOR_FUNCTION]",
        instruction=(
            "Identify each distinct action, process, or functional behavior described in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one behavior or function from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this behavior/function concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
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
    ),

    # ── 9. DESCRIPTIVE QUALITIES / ASSOCIATIONS ─────────────────────────
    "ATTRIBUTES_ASSOCIATIONS": FacetDefinition(
        key="ATTRIBUTES_ASSOCIATIONS",
        decision_tree_position=9,
        criterion="Do responses mainly differ in qualities, traits, images, or associations?",
        criterion_signals=(
            "Descriptive traits or characteristics",
            "Product or brand associations",
            "Image, reputation, perceived qualities",
        ),
        exclusions=(
            "Category or definition",
            "Judgments of good/bad",
            "Lived experience explanations",
        ),
        noun_phrase_descriptor="ATTRIBUTES & ASSOCIATIONS: qualities, traits, images, or associations",
        dimension_description=(
            "Use this facet when the dominant variation lies in how the entity is described "
            "or perceived — its characteristics, symbolic meanings, reputation, or associations "
            "— rather than experiences or judgments."
        ),
        allowed_concepts=(
            "attribute", "trait", "quality", "property",
            "association", "characteristic", "image",
            "reputation", "perception", "symbol",
        ),
        pattern="[ANCHOR_SUBJECT] → [ATTRIBUTE_ASSOCIATION]",
        instruction=(
            "Identify each distinct quality, trait, image, or association described in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one attribute or association from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this attribute/association concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="ATTRIBUTE_ASSOCIATION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a quality, trait, image, association, or perceived characteristic of the entity.",
        ),
        clarification=(
            "Traits must be non-comparative and non-prioritized",
        ),
    ),

    # ── 5. MOTIVATION / REASON ──────────────────────────────────────────
    "MOTIVATIONS_DRIVERS": FacetDefinition(
        key="MOTIVATIONS_DRIVERS",
        decision_tree_position=5,
        criterion="Do responses mainly differ in why people care, want, or act?",
        criterion_signals=(
            "Reasons, rationales, explanations of importance",
            "Needs, values, goals, concerns",
            "Causal language tied to human intent: 'because…', 'so that…', 'in order to…'",
        ),
        exclusions=(
            "Judgments of quality",
            "Experience narratives",
            "Structural/system causality not tied to human intent",
        ),
        noun_phrase_descriptor="MOTIVATIONS & DRIVERS: why people care, want, or act",
        dimension_description=(
            "Use this facet when variation is driven by underlying reasons — needs, goals, "
            "values, concerns, or trade-offs that explain respondents' attitudes or behaviors."
        ),
        allowed_concepts=(
            "need", "goal", "value", "concern",
            "motivation", "reason", "driver", "trade_off",
            "aspiration", "priority",
        ),
        pattern="[ANCHOR_SUBJECT] → [MOTIVATION_DRIVER]",
        instruction=(
            "Identify each distinct motivation, need, goal, or reason expressed in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one motivation or reason from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this motivation/driver concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="MOTIVATION_DRIVER",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a need, goal, motivation, value, or reason.",
        ),
    ),

    # ── 4. CONTEXT / CONDITIONS ─────────────────────────────────────────
    "CONTEXT_CONDITIONS": FacetDefinition(
        key="CONTEXT_CONDITIONS",
        decision_tree_position=4,
        criterion="Do responses mainly differ in when, where, or under what conditions something applies?",
        criterion_signals=(
            "Timing, frequency, lifecycle stage",
            "Location, channel, environment",
            "Constraints, triggers, situational factors",
        ),
        exclusions=(
            "Actions or behaviors themselves",
            "Motivations or reasons",
            "Descriptive attributes",
        ),
        noun_phrase_descriptor="CONTEXT & CONDITIONS: when, where, or under what conditions something applies",
        dimension_description=(
            "Use this facet when responses vary primarily by situational factors — time, place, "
            "environment, constraints, or conditions that shape applicability or relevance."
        ),
        allowed_concepts=(
            "condition", "context", "constraint", "trigger",
            "precondition", "environment", "setting", "situation",
            "timing", "circumstance",
        ),
        pattern="[ANCHOR_SUBJECT] @ [CONTEXT_CONDITION]",
        instruction=(
            "Identify each distinct condition, context, or circumstance described in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one condition or contextual factor from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this condition/context concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity, event, or topic frame in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="CONTEXT_CONDITION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase specifying the condition, context, setting, or circumstance.",
        ),
    ),

    # ── 3. ACTORS / AFFECTED PARTIES ────────────────────────────────────
    "ACTORS_TARGETS": FacetDefinition(
        key="ACTORS_TARGETS",
        decision_tree_position=3,
        criterion="Do responses mainly differ in who is involved or impacted?",
        criterion_signals=(
            "Different user groups, stakeholders, agents",
            "Responsibility, ownership, accountability",
            "Who benefits, who is affected, who decides",
        ),
        exclusions=(
            "What happens or how it works",
            "Evaluations of actors",
            "Relationships between actors (see RELATIONS_DEPENDENCIES)",
        ),
        noun_phrase_descriptor="ACTORS & TARGETS: who is involved or impacted",
        dimension_description=(
            "Use this facet when the dominant variation is in the actors, agents, stakeholders, "
            "or affected parties being mentioned — who does what, who benefits, who is responsible."
        ),
        allowed_concepts=(
            "actor", "stakeholder", "user_group", "target",
            "beneficiary", "responsible_party", "affected_group",
            "participant", "owner", "audience",
        ),
        pattern="[ANCHOR_SUBJECT] → [ACTOR_TARGET]",
        instruction=(
            "Identify each distinct actor, stakeholder, or affected party mentioned in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one actor or affected party from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this actor/stakeholder concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="ACTOR_TARGET",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase identifying an actor, stakeholder, user group, or affected party.",
        ),
    ),

    # ── 10. RELATIONS / DEPENDENCIES ────────────────────────────────────
    "RELATIONS_DEPENDENCIES": FacetDefinition(
        key="RELATIONS_DEPENDENCIES",
        decision_tree_position=10,
        criterion="Do responses mainly differ in relationships or dependencies between entities?",
        criterion_signals=(
            "Dependencies between components or systems",
            "Trade-offs (A vs B, cost vs benefit)",
            "Influence, interaction effects, system-level causality",
        ),
        exclusions=(
            "Individual attributes",
            "Actor identity",
            "Contextual timing or location",
        ),
        noun_phrase_descriptor="RELATIONS & DEPENDENCIES: relationships or comparisons between entities",
        dimension_description=(
            "Use this facet when responses vary primarily in how entities, concepts, or topics "
            "relate to each other — dependencies, trade-offs, influence, or comparisons across options."
        ),
        allowed_concepts=(
            "dependency", "comparison", "influence", "trade_off",
            "interaction", "connection", "partnership", "competition",
            "relationship", "collaboration",
        ),
        pattern="[ANCHOR_SUBJECT] → [RELATION_DEPENDENCY]",
        instruction=(
            "Identify each distinct relationship, dependency, or comparison described in the response. "
            "For each idea, produce one concise realization formatted according to the pattern."
        ),
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one relationship or dependency from the response.",
            concept_instruction="Create a canonical, reusable noun-phrase label for this relationship/dependency concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
            concept_type_instruction="Classify into one of the discovered concept types for this facet.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        dimension_slot=SlotDefinition(
            name="RELATION_DEPENDENCY",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a relationship, dependency, comparison, or influence.",
        ),
    ),

    # ── 2. CONSTITUTIVE DEFINITION ──────────────────────────────────────
    "IDENTITY_DEFINITION": FacetDefinition(
        key="IDENTITY_DEFINITION",
        decision_tree_position=2,
        criterion="Do responses mainly differ in how the entity is defined or categorized?",
        criterion_signals=(
            "What kind of thing it is",
            "Category membership, scope, boundaries",
            "Purpose, mission, raison d'être",
        ),
        exclusions=(
            "Descriptive traits or qualities",
            "Evaluations or preferences",
            "Relationships to other entities",
        ),
        noun_phrase_descriptor="IDENTITY & DEFINITION: how the entity is defined or categorized",
        dimension_description=(
            "Use this facet when the dominant variation is in what the entity IS — its fundamental "
            "nature, purpose, meaning, or categorical identity — rather than its qualities or "
            "how it's experienced."
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
            concept_instruction="Create a canonical, reusable noun-phrase label for this definition/identity concept. Name what the respondent is really talking about in context — the underlying thing, phenomenon, or idea. Not a spelling fix or nominalization.",
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
    ),
}


# ========================================================================
# Fallback concept type guidance (used when DISCOVER_CONCEPT_TYPES = False)
# ========================================================================


CONCEPT_TYPE_FALLBACK_TABLE = (
    "Use a short, reusable label (1-4 words) that could organize many different concepts.\n\n"
    "GOOD concept types (from other surveys, for illustration):\n"
    "  appointment scheduling → access and logistics\n"
    "  schedule reliability → operations and planning\n"
    "  warmth of service → hospitality and interaction\n\n"
    "BAD concept types:\n"
    "  × Linguistic role labels: 'functional trait', 'moral attribute', 'quality measure'\n"
    "  × Paraphrases of the concept: 'scheduling issue' for 'appointment scheduling'\n"
    "  × Generic catch-alls: 'characteristics', 'properties', 'features'\n\n"
    "All labels must be in {language}."
)

CONCEPT_TYPE_FALLBACK_PRIORITY_RULES = (
    "1. Each ladder level answers a DIFFERENT question — instance (what they said), "
    "concept (what they mean), concept type (which aspect), definition (what it represents).\n"
    "2. Never repeat, nominalize, synonymize, or paraphrase the level below.\n"
    "3. Concept types must be reusable thematic domains, not per-concept labels.\n"
    "4. Stay in one language throughout the ladder.\n"
    "5. Concept type definition frames WHY this domain matters for the entity."
)


def get_facet(facet_key: str) -> FacetDefinition:
    """Get a facet by key. Raises KeyError with a clear message if not found."""
    try:
        return FACETS[facet_key]
    except KeyError:
        raise KeyError(
            f"Unknown facet: {facet_key!r}. "
            f"Valid facets: {sorted(FACETS.keys())}"
        ) from None


def get_facets_in_decision_order() -> list[FacetDefinition]:
    """Return all facets in decision tree order (position 1-10)."""
    return [FACETS[key] for key in FACET_DECISION_ORDER]
