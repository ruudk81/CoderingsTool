"""
Dimension definitions and type system for step_3_ideaExtractor v5.

Taxonomy: Dimension > Domain > Facet > Attribute (progressive narrowing).
Dimensions are the highest-level conceptual axes used to organize the problem space.

All data is expressed as frozen dataclasses. No nested dicts, no .get() chains.
Access a dimension with: get_dimension("PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS") — KeyError if not found.
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
    """Dimension-specific instructions for taxonomy field extraction."""
    instance_instruction: str
    interpretation_instruction: str
    abstraction_instruction: str
    domain_instruction: str


@dataclass(frozen=True)
class DimensionExample:
    """One worked example for the extraction prompt."""
    survey_context: str     # e.g., "City improvement survey (entity: City of Springfield)"
    response: str           # e.g., "more bike lanes and better lighting"
    instance: str           # verbatim span
    domain: str             # thematic domain (L2 in taxonomy)
    interpretation: str     # concrete interpretation (what it means)
    abstraction: str        # broader significance (why it matters)
    valence: str            # "+", "-", or "0"


@dataclass(frozen=True)
class DimensionDefinition:
    """Complete definition for one primary dimension (L1 in taxonomy: Dimension > Domain > Facet > Attribute)."""
    key: str
    decision_tree_position: int              # 1-10, priority order in decision tree
    criterion: str                           # Diagnostic question from decision tree
    criterion_signals: Tuple[str, ...]       # Bullet-point signals for this dimension
    exclusions: Tuple[str, ...]              # "What this dimension is NOT" — disambiguation cues
    noun_phrase_descriptor: str
    dimension_description: str
    allowed_concepts: Tuple[str, ...]
    pattern: str
    instruction: str
    prompt_rules: PromptRules
    anchor_slot: SlotDefinition
    domain_slot: SlotDefinition
    examples: Tuple[DimensionExample, ...] = ()  # Worked examples for the extraction prompt
    clarification: Tuple[str, ...] = ()      # Optional clarification notes (empty tuple = none)

    @property
    def domain_marker(self) -> str:
        """The marker token for this dimension, e.g., '[PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER]'."""
        return f"[{self.domain_slot.name}]"


# ========================================================================
# Decision tree ordering — MECE by priority
# ========================================================================

DIMENSION_DECISION_ORDER: Tuple[str, ...] = (
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
# Dimension registry — all 10 dimensions (decision tree order)
# ========================================================================
#
# SKELETON: criterion + criterion_signals are complete from the user's schema.
# Downstream metadata (allowed_concepts, instruction,
# prompt_rules, slot definitions) are PLACEHOLDERS awaiting user input.
# ========================================================================

DIMENSIONS: Dict[str, DimensionDefinition] = {

    # ── 1. PRESCRIPTIVE CHANGE ──────────────────────────────────────────
    "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS": DimensionDefinition(
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
            "Use this dimension when the dominant variation across responses is in what should be done "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a proposed action, improvement, recommendation, or solution.",
        ),
        examples=(
            DimensionExample(
                survey_context="City improvement survey (entity: City of Springfield)",
                response="more bike lanes and better street lighting",
                instance="more bike lanes",
                domain="infrastructure and mobility",
                interpretation="cycling infrastructure expansion",
                abstraction="sustainable urban mobility",
                valence="+",
            ),
            DimensionExample(
                survey_context="Hospital feedback (entity: City Hospital)",
                response="reduce the paperwork for admissions",
                instance="reduce the paperwork",
                domain="administrative processes",
                interpretation="admission simplification",
                abstraction="operational efficiency",
                valence="+",
            ),
        ),
    ),

    # ── 7. JUDGMENT / PRIORITIZATION ────────────────────────────────────
    "EVALUATION_PRIORITIZATION": DimensionDefinition(
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
            "Use this dimension when the dominant variation is in how respondents assess or evaluate "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon being evaluated in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="EVALUATION_PRIORITY",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase expressing a judgment, preference, opinion, or evaluative stance.",
        ),
        examples=(
            DimensionExample(
                survey_context="Airline satisfaction survey (entity: SkyAir)",
                response="the food is terrible but the seats are comfortable",
                instance="the food is terrible",
                domain="onboard services",
                interpretation="poor meal quality",
                abstraction="onboard service standards",
                valence="-",
            ),
            DimensionExample(
                survey_context="University evaluation (entity: State University)",
                response="excellent research reputation",
                instance="excellent research reputation",
                domain="academic standing",
                interpretation="research prestige",
                abstraction="institutional academic standing",
                valence="+",
            ),
        ),
    ),

    # ── 6. LIVED EXPERIENCE / PERCEPTION ────────────────────────────────
    "EXPERIENCE_PERCEPTION": DimensionDefinition(
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
            "Use this dimension when responses vary primarily in lived experiences, feelings, "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon experienced in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="EXPERIENCE_PERCEPTION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing an experience, perception, impression, or feeling.",
        ),
        examples=(
            DimensionExample(
                survey_context="Hotel stay feedback (entity: Grand Plaza Hotel)",
                response="felt rushed during checkout",
                instance="felt rushed during checkout",
                domain="guest journey",
                interpretation="hurried checkout experience",
                abstraction="guest journey quality",
                valence="-",
            ),
            DimensionExample(
                survey_context="Theme park survey (entity: FunWorld)",
                response="the atmosphere was magical",
                instance="the atmosphere was magical",
                domain="park ambiance",
                interpretation="immersive atmosphere",
                abstraction="experiential design",
                valence="+",
            ),
        ),
        clarification=(
            "Includes implicit evaluation when embedded in an experience narrative",
        ),
    ),

    # ── 8. ACTION / PROCESS ─────────────────────────────────────────────
    "BEHAVIOR_FUNCTION": DimensionDefinition(
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
            "Use this dimension when responses vary primarily in descriptive accounts of actions, "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="BEHAVIOR_FUNCTION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing an action, process, behavior, or functional output of the entity.",
        ),
        examples=(
            DimensionExample(
                survey_context="Banking usage survey (entity: QuickBank App)",
                response="I transfer money and check my balance every morning",
                instance="transfer money",
                domain="transaction services",
                interpretation="money transfers",
                abstraction="core banking functionality",
                valence="0",
            ),
            DimensionExample(
                survey_context="Software feedback (entity: ProjectHub)",
                response="the auto-save keeps overwriting my changes",
                instance="auto-save keeps overwriting my changes",
                domain="data management",
                interpretation="auto-save behavior",
                abstraction="data integrity management",
                valence="-",
            ),
        ),
    ),

    # ── 9. DESCRIPTIVE QUALITIES / ASSOCIATIONS ─────────────────────────
    "ATTRIBUTES_ASSOCIATIONS": DimensionDefinition(
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
            "Use this dimension when the dominant variation lies in how the entity is described "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="ATTRIBUTE_ASSOCIATION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a quality, trait, image, association, or perceived characteristic of the entity.",
        ),
        examples=(
            DimensionExample(
                survey_context="Brand association survey (entity: Merk X)",
                response="insurance and sustainability",
                instance="insurance",
                domain="products and services",
                interpretation="insurance products",
                abstraction="financial service offering",
                valence="0",
            ),
            DimensionExample(
                survey_context="Car brand perception (entity: Volvo)",
                response="safe but boring design",
                instance="safe",
                domain="safety and engineering",
                interpretation="vehicle safety reputation",
                abstraction="brand trust and reliability",
                valence="+",
            ),
        ),
        clarification=(
            "Traits must be non-comparative and non-prioritized",
        ),
    ),

    # ── 5. MOTIVATION / REASON ──────────────────────────────────────────
    "MOTIVATIONS_DRIVERS": DimensionDefinition(
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
            "Use this dimension when variation is driven by underlying reasons — needs, goals, "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="MOTIVATION_DRIVER",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a need, goal, motivation, value, or reason.",
        ),
        examples=(
            DimensionExample(
                survey_context="Gym membership survey (entity: FitLife Gym)",
                response="I go because it helps my mental health",
                instance="helps my mental health",
                domain="health and wellbeing",
                interpretation="mental health benefit",
                abstraction="personal wellbeing",
                valence="+",
            ),
            DimensionExample(
                survey_context="Grocery store choice (entity: FreshMart)",
                response="it's close to home and the prices are low",
                instance="close to home",
                domain="convenience and access",
                interpretation="proximity",
                abstraction="convenience and accessibility",
                valence="+",
            ),
        ),
    ),

    # ── 4. CONTEXT / CONDITIONS ─────────────────────────────────────────
    "CONTEXT_CONDITIONS": DimensionDefinition(
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
            "Use this dimension when responses vary primarily by situational factors — time, place, "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity, event, or topic frame in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="CONTEXT_CONDITION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase specifying the condition, context, setting, or circumstance.",
        ),
        examples=(
            DimensionExample(
                survey_context="Remote work survey (entity: TechCorp)",
                response="works great when internet is stable but fails during peak hours",
                instance="during peak hours",
                domain="technical infrastructure",
                interpretation="network load timing",
                abstraction="infrastructure capacity management",
                valence="-",
            ),
            DimensionExample(
                survey_context="Public transit survey (entity: Metro Line 5)",
                response="only useful for my morning commute",
                instance="morning commute",
                domain="usage patterns",
                interpretation="commute-hour dependency",
                abstraction="usage pattern constraints",
                valence="0",
            ),
        ),
    ),

    # ── 3. ACTORS / AFFECTED PARTIES ────────────────────────────────────
    "ACTORS_TARGETS": DimensionDefinition(
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
            "Use this dimension when the dominant variation is in the actors, agents, stakeholders, "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="ACTOR_TARGET",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase identifying an actor, stakeholder, user group, or affected party.",
        ),
        examples=(
            DimensionExample(
                survey_context="School policy survey (entity: Riverside School)",
                response="parents should be more involved in curriculum decisions",
                instance="parents",
                domain="school community",
                interpretation="parental involvement",
                abstraction="stakeholder engagement in education",
                valence="+",
            ),
            DimensionExample(
                survey_context="Healthcare access (entity: Regional Clinic)",
                response="elderly patients struggle with the online booking system",
                instance="elderly patients",
                domain="patient demographics",
                interpretation="senior accessibility",
                abstraction="inclusive service design",
                valence="-",
            ),
        ),
    ),

    # ── 10. RELATIONS / DEPENDENCIES ────────────────────────────────────
    "RELATIONS_DEPENDENCIES": DimensionDefinition(
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
            "Use this dimension when responses vary primarily in how entities, concepts, or topics "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="RELATION_DEPENDENCY",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase describing a relationship, dependency, comparison, or influence.",
        ),
        examples=(
            DimensionExample(
                survey_context="Energy policy survey (entity: National Grid)",
                response="wind energy depends too much on weather conditions",
                instance="depends too much on weather conditions",
                domain="supply reliability",
                interpretation="weather dependency",
                abstraction="supply reliability risk",
                valence="-",
            ),
            DimensionExample(
                survey_context="Retail ecosystem (entity: ShopLocal Platform)",
                response="small shops benefit from the shared delivery network",
                instance="benefit from the shared delivery network",
                domain="platform partnerships",
                interpretation="shared logistics advantage",
                abstraction="platform ecosystem value",
                valence="+",
            ),
        ),
    ),

    # ── 2. CONSTITUTIVE DEFINITION ──────────────────────────────────────
    "IDENTITY_DEFINITION": DimensionDefinition(
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
            "Use this dimension when the dominant variation is in what the entity IS — its fundamental "
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
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction="Classify into one of the discovered domains for this dimension.",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="IDENTITY_DEFINITION",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase defining or framing what the entity is, its nature, or its purpose.",
        ),
        examples=(
            DimensionExample(
                survey_context="Brand identity survey (entity: Patagonia)",
                response="more of an activist movement than a clothing brand",
                instance="activist movement",
                domain="brand identity",
                interpretation="activism positioning",
                abstraction="brand purpose and identity",
                valence="0",
            ),
            DimensionExample(
                survey_context="Municipal services survey (entity: Public Library)",
                response="it's a community hub, not just a book lending place",
                instance="community hub",
                domain="institutional role",
                interpretation="community function",
                abstraction="institutional social value",
                valence="+",
            ),
        ),
    ),
}


# ========================================================================
# Fallback domain guidance (used when DISCOVER_DOMAINS = False)
# ========================================================================


DOMAIN_FALLBACK_TABLE = (
    "EXAMPLES (from other surveys, for illustration):\n"
    "    - GOOD domains:\n"
    "       • appointment scheduling → access and logistics\n"
    "       • schedule reliability → operations and planning\n"
    "       • warmth of service → hospitality and interaction\n\n"
    "   - BAD domains:\n"
    "       × Linguistic role labels: 'functional trait', 'moral attribute', 'quality measure'\n"
    "       × Paraphrases of the interpretation: 'scheduling issue' for 'appointment scheduling'\n"
    "       × Generic catch-alls: 'characteristics', 'properties', 'features'\n\n"
)


def get_dimension(dimension_key: str) -> DimensionDefinition:
    """Get a dimension by key. Raises KeyError with a clear message if not found."""
    try:
        return DIMENSIONS[dimension_key]
    except KeyError:
        raise KeyError(
            f"Unknown dimension: {dimension_key!r}. "
            f"Valid dimensions: {sorted(DIMENSIONS.keys())}"
        ) from None


def get_dimensions_in_decision_order() -> list[DimensionDefinition]:
    """Return all dimensions in decision tree order (position 1-10)."""
    return [DIMENSIONS[key] for key in DIMENSION_DECISION_ORDER]
