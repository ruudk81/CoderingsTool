"""
Dimension definitions and type system for step_3_ideaExtractor.
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
    """Dimension-specific instructions for taxonomy field extraction.

    Each field maps to a taxonomy level or abstraction ladder rung:
    - instance_instruction       → Abstraction ladder rung 1: verbatim span guidance
    - interpretation_instruction → Abstraction ladder rung 2: concrete meaning
    - abstraction_instruction    → Abstraction ladder rung 3: broader significance
    - domain_instruction         → Domain (L2): dimension-specific subject question
    - domain_diagnostic          → Domain (L2): short-form question for prompt headers
    - facet_instruction          → Facet (L3): dimension-specific analytical lens question
    - facet_diagnostic           → Facet (L3): short-form question for prompt headers
    - attribute_instruction      → Attribute (L4): dimension-specific observable property question
    - attribute_diagnostic       → Attribute (L4): short-form question for prompt headers
    - code_diagnostic            → Code: sentence stem for P9 consolidation diagnostic test
    """
    instance_instruction: str
    interpretation_instruction: str
    abstraction_instruction: str
    facet_instruction: str
    domain_instruction: str
    domain_diagnostic: str
    facet_diagnostic: str
    attribute_instruction: str
    attribute_diagnostic: str
    code_diagnostic: str


@dataclass(frozen=True)
class DimensionExample:
    """One worked example for the extraction prompt.

    Fields map to taxonomy levels + abstraction ladder:
    - instance        → Abstraction ladder rung 1: verbatim span from response
    - interpretation  → Abstraction ladder rung 2: concrete meaning (survey language)
    - abstraction     → Abstraction ladder rung 3: broader significance (survey language)
    - domain          → Domain (L2): thematic domain
    - facet           → Facet (L3): analytical lens
    """
    survey_context: str     # e.g., "City improvement survey (entity: City of Springfield)"
    response: str           # e.g., "more bike lanes and better lighting"
    instance: str           # Abstraction ladder rung 1: verbatim span
    interpretation: str     # Ladder rung 2: concrete meaning
    abstraction: str        # Ladder rung 3: broader significance
    domain: str             # Domain (L2): thematic domain
    facet: str              # Facet (L3): dimension-specific aspect


@dataclass(frozen=True)
class DimensionDefinition:
    """Complete definition for one primary dimension (L1 in taxonomy: Dimension > Domain > Facet > Attribute)."""
    key: str
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
    "GENERAL_OTHER",
)


# ========================================================================
# Dimension registry — all 11 dimensions in decision tree order (1-11)
# ========================================================================

DIMENSIONS: Dict[str, DimensionDefinition] = {

    # ── 1. PRESCRIPTIVE CHANGE ──────────────────────────────────────────
    "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS": DimensionDefinition(
        key="PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS",
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
        instruction="Identify each distinct proposed action, improvement, or solution in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one proposed action or improvement from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the part of the system, organization, or context that is the target of the proposed change. \n"
                "Key idea: Domains specify what part of the system should change."
            ),
            domain_diagnostic="Question that needs to be answered: What is the target of the change?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which proposed changes are examined — specifically, the type or approach of change being proposed. "
                "Facets distinguish between different kinds of interventions aimed at the same target area; each must be independently analyzable. \n"
                "Key idea: Facets specify how the target should change."
            ),
            facet_diagnostic="What type of change is proposed?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific, concrete improvement or action being proposed. It is a named property that captures the precise nature of the suggestion. \n"
                "Key idea: Attributes name the specific proposed improvement."
            ),
            attribute_diagnostic="What exactly is the proposed improvement?",
            code_diagnostic="This code is about what should change regarding …",
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
                interpretation="cycling infrastructure expansion",
                abstraction="sustainable urban mobility",
                domain="infrastructure and mobility",
                facet="infrastructure expansion",

            ),
            DimensionExample(
                survey_context="Hospital feedback (entity: City Hospital)",
                response="reduce the paperwork for admissions",
                instance="reduce the paperwork",
                interpretation="administrative burden reduction",
                abstraction="process efficiency",
                domain="administrative processes",
                facet="process simplification",

            ),
        ),
    ),

    # ── 2. CONSTITUTIVE DEFINITION ──────────────────────────────────────
    "IDENTITY_DEFINITION": DimensionDefinition(
        key="IDENTITY_DEFINITION",
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
        instruction="Identify each distinct idea about what the entity IS, how it is defined, or why it exists.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one definition/identity concept from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the dimension of identity being articulated — what facet of the entity's nature, purpose, or classification is being described. \n"
                "Key idea: Domains specify which dimension of identity is addressed."
            ),
            domain_diagnostic="Question that needs to be answered: Which dimension of identity is being described?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which an entity's identity is examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify which aspect of identity is being defined."
            ),
            facet_diagnostic="What aspect of identity is being addressed?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific defining characteristic or identity marker being articulated. "
                "It is a named property that captures the precise feature of the entity's identity. \n"
                "Key idea: Attributes name the specific defining feature."
            ),
            attribute_diagnostic="What defining feature is mentioned?",
            code_diagnostic="This code is about how the entity is defined as …",
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
                interpretation="activism positioning",
                abstraction="brand purpose and identity",
                domain="brand identity",
                facet="purpose",

            ),
            DimensionExample(
                survey_context="Municipal services survey (entity: Public Library)",
                response="it's a community hub, not just a book lending place",
                instance="community hub",
                interpretation="community function",
                abstraction="institutional social value",
                domain="institutional role",
                facet="social function",

            ),
        ),
    ),

    # ── 3. ACTORS / AFFECTED PARTIES ────────────────────────────────────
    "ACTORS_TARGETS": DimensionDefinition(
        key="ACTORS_TARGETS",
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
        instruction="Identify each distinct actor, stakeholder, or affected party mentioned in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one actor or affected party from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the sphere of activity or responsibility in which actors are involved. \n"
                "Key idea: Domains specify in what sphere actors play a role."
            ),
            domain_diagnostic="Question that needs to be answered: In what sphere of activity are actors involved?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which actor involvement is examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what role the actor plays."
            ),
            facet_diagnostic="What role or position does the actor occupy?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific actor group or stakeholder type referenced. It is a named property that captures the precise party being discussed. \n"
                "Key idea: Attributes name the specific actor or stakeholder group."
            ),
            attribute_diagnostic="Which specific actor group is referenced?",
            code_diagnostic="This code is about which actors are involved in …",
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
                interpretation="parental involvement",
                abstraction="stakeholder engagement in education",
                domain="school community",
                facet="responsibility",

            ),
            DimensionExample(
                survey_context="Healthcare access (entity: Regional Clinic)",
                response="elderly patients struggle with the online booking system",
                instance="elderly patients",
                interpretation="senior accessibility",
                abstraction="inclusive service design",
                domain="patient demographics",
                facet="affected party",

            ),
        ),
    ),

    # ── 4. CONTEXT / CONDITIONS ─────────────────────────────────────────
    "CONTEXT_CONDITIONS": DimensionDefinition(
        key="CONTEXT_CONDITIONS",
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
        instruction="Identify each distinct condition, context, or circumstance described in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one condition or contextual factor from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the type of context or environment being described — temporal, spatial, organizational, or situational. \n"
                "Key idea: Domains specify what type of contextual environment is described."
            ),
            domain_diagnostic="Question that needs to be answered: What type of contextual environment is described?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which contextual conditions are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what type of condition is described."
            ),
            facet_diagnostic="What type of contextual dimension is described?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific condition, circumstance, or contextual factor being mentioned. It is a named property that captures the precise situational feature. \n"
                "Key idea: Attributes name the specific contextual condition."
            ),
            attribute_diagnostic="What specific condition is mentioned?",
            code_diagnostic="This code is about the condition or situation of …",
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
                interpretation="network load timing",
                abstraction="infrastructure capacity management",
                domain="technical infrastructure",
                facet="time",

            ),
            DimensionExample(
                survey_context="Public transit survey (entity: Metro Line 5)",
                response="only useful for my morning commute",
                instance="morning commute",
                interpretation="commute-hour dependency",
                abstraction="usage pattern constraints",
                domain="usage patterns",
                facet="time",

            ),
        ),
    ),

    # ── 5. MOTIVATION / REASON ──────────────────────────────────────────
    "MOTIVATIONS_DRIVERS": DimensionDefinition(
        key="MOTIVATIONS_DRIVERS",
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
        instruction="Identify each distinct motivation, need, goal, or reason expressed in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one motivation or reason from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the life area, activity, or concern that the motivation relates to. \n"
                "Key idea: Domains specify what area of life or concern drives the motivation."
            ),
            domain_diagnostic="Question that needs to be answered: What area of life or concern is the motivation about?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which motivations are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what type of motivation is expressed."
            ),
            facet_diagnostic="What type of motivation is expressed?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific reason, benefit, or motivational factor being stated. It is a named property that captures the precise driver. \n"
                "Key idea: Attributes name the specific motivational factor."
            ),
            attribute_diagnostic="What specific reason is stated?",
            code_diagnostic="This code is about why people care about …",
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
                interpretation="mental health benefit",
                abstraction="personal wellbeing",
                domain="health and wellbeing",
                facet="need",

            ),
            DimensionExample(
                survey_context="Grocery store choice (entity: FreshMart)",
                response="it's close to home and the prices are low",
                instance="close to home",
                interpretation="proximity",
                abstraction="convenience and accessibility",
                domain="convenience and access",
                facet="value",

            ),
        ),
    ),

    # ── 6. LIVED EXPERIENCE / PERCEPTION ────────────────────────────────
    "EXPERIENCE_PERCEPTION": DimensionDefinition(
        key="EXPERIENCE_PERCEPTION",
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
            "Detached judgments without narrative framing (see EVALUATION_PRIORITIZATION)",
        ),
        noun_phrase_descriptor="EXPERIENCE & PERCEPTION: how something was experienced or perceived",
        dimension_description=(
            "Use this dimension when responses vary primarily in lived experiences, feelings, impressions, or overall sense-making. "
            "The focus is on what it was like, rather than judgments, actions, or attributes."
        ),
        allowed_concepts=(
            "experience", "perception", "impression", "feeling",
            "atmosphere", "encounter", "sensation", "observation",
            "memory", "narrative",
        ),
        pattern="[ANCHOR_SUBJECT] → [EXPERIENCE_PERCEPTION]",
        instruction="Identify each distinct experience, perception, or impression described in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one experience or perception from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the part of the experience or journey being described. \n"
                "Key idea: Domains specify which part of the experience is described."
            ),
            domain_diagnostic="Question that needs to be answered: Which part of the experience?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which experiences are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what experiential quality is described."
            ),
            facet_diagnostic="What experiential dimension is being addressed?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific experiential feature observed or felt. It is a named property that captures the precise aspect of the experience. \n"
                "Key idea: Attributes name the specific observed experience feature."
            ),
            attribute_diagnostic="What specific experience feature was observed or felt?",
            code_diagnostic="This code is about the experience of …",
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
                interpretation="hurried checkout experience",
                abstraction="guest journey quality",
                domain="guest journey",
                facet="flow",

            ),
            DimensionExample(
                survey_context="Theme park survey (entity: FunWorld)",
                response="the atmosphere was magical",
                instance="the atmosphere was magical",
                interpretation="immersive atmosphere",
                abstraction="experiential design",
                domain="park ambiance",
                facet="atmosphere",

            ),
        ),
        clarification=(
            "Includes implicit evaluation when embedded in an experience narrative",
            "Experience = narrative interaction with the entity; the respondent recounts what happened or how it felt",
        ),
    ),

    # ── 7. JUDGMENT / PRIORITIZATION ────────────────────────────────────
    "EVALUATION_PRIORITIZATION": DimensionDefinition(
        key="EVALUATION_PRIORITIZATION",
        criterion="Do responses mainly differ in opinions, judgments, or preferences?",
        criterion_signals=(
            "Good vs bad, positive vs negative",
            "Preferences, rankings, comparisons",
            "Statements of importance, value, risk, or priority",
        ),
        exclusions=(
            "Proposed changes or actions",
            "Explanations of why people care (see MOTIVATIONS)",
            "Narrative interaction accounts where evaluation is implicit (see EXPERIENCE_PERCEPTION)",
            "Neutral descriptive properties without value judgment (see ATTRIBUTES_ASSOCIATIONS)",
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
        instruction="Identify each distinct evaluative opinion, preference, or prioritization in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one evaluation or preference from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the aspect or dimension of the entity being evaluated — what part of the offering, experience, or organization is being judged. \n"
                "Key idea: Domains specify what aspect is being evaluated."
            ),
            domain_diagnostic="Question that needs to be answered: What aspect of the entity is being evaluated?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which evaluations are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify on what criterion the evaluation is based."
            ),
            facet_diagnostic="What evaluation criterion is being applied?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific evaluative signal or evidence of judgment. It is a named property that captures the precise characteristic being assessed. \n"
                "Key idea: Attributes name the specific evaluation signal."
            ),
            attribute_diagnostic="What specific evidence of evaluation appears?",
            code_diagnostic="This code is about how the entity is judged on …",
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
                interpretation="poor meal quality",
                abstraction="onboard service standards",
                domain="onboard services",
                facet="quality",

            ),
            DimensionExample(
                survey_context="University evaluation (entity: State University)",
                response="excellent research reputation",
                instance="excellent research reputation",
                interpretation="research prestige",
                abstraction="institutional academic standing",
                domain="academic standing",
                facet="reputation",

            ),
        ),
        clarification=(
            "Evaluation = detached judgment or assessment, not embedded in an interaction narrative",
        ),
    ),

    # ── 8. ACTION / PROCESS ─────────────────────────────────────────────
    "BEHAVIOR_FUNCTION": DimensionDefinition(
        key="BEHAVIOR_FUNCTION",
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
            "Causal links between two variables or entities (see RELATIONS_DEPENDENCIES)",
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
        instruction="Identify each distinct action, process, or functional behavior described in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one behavior or function from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the process, system, or activity being described. \n"
                "Key idea: Domains specify what system or process is described."
            ),
            domain_diagnostic="Question that needs to be answered: What system or process?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which behaviors or functions are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify which step or function is described."
            ),
            facet_diagnostic="Which functional stage or step is described?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific action, behavior, or functional feature being described. It is a named property that captures the precise operational characteristic. \n"
                "Key idea: Attributes name the specific behavioral or functional feature."
            ),
            attribute_diagnostic="What specific action occurs?",
            code_diagnostic="This code is about what happens regarding …",
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
                interpretation="money transfers",
                abstraction="core banking functionality",
                domain="transaction services",
                facet="processing",

            ),
            DimensionExample(
                survey_context="Software feedback (entity: ProjectHub)",
                response="the auto-save keeps overwriting my changes",
                instance="auto-save keeps overwriting my changes",
                interpretation="auto-save behavior",
                abstraction="data integrity management",
                domain="data management",
                facet="output",

            ),
        ),
        clarification=(
            "Behavior = a single event, action, or process occurring; one entity acts or something happens",
        ),
    ),

    # ── 9. DESCRIPTIVE QUALITIES / ASSOCIATIONS ─────────────────────────
    "ATTRIBUTES_ASSOCIATIONS": DimensionDefinition(
        key="ATTRIBUTES_ASSOCIATIONS",
        criterion="Do responses mainly differ in qualities, traits, images, or associations?",
        criterion_signals=(
            "Descriptive traits or characteristics",
            "Product or brand associations",
            "Image, reputation, perceived qualities",
        ),
        exclusions=(
            "Category or definition",
            "Judgments of good/bad (see EVALUATION_PRIORITIZATION)",
            "Evaluative adjectives that imply positive/negative assessment (e.g., 'great', 'terrible')",
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
        instruction="Identify each distinct quality, trait, image, or association described in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one attribute or association from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the dimension or aspect of the entity being described — what part of its identity, offering, behavior, or perception is being characterized. \n"
                "Key idea: Domains specify what dimension of the entity is being described."
            ),
            domain_diagnostic="Question that needs to be answered: What dimension of the entity is being described?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which descriptive qualities are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what type of quality is described."
            ),
            facet_diagnostic="What type of quality or attribute is described?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific quality, trait, or association being described. It is a named property that captures the precise descriptive characteristic. \n"
                "Key idea: Attributes name the specific quality or trait."
            ),
            attribute_diagnostic="What specific quality is described?",
            code_diagnostic="This code is about the quality of the entity being …",
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
                interpretation="insurance products",
                abstraction="financial service offering",
                domain="products and services",
                facet="functional",

            ),
            DimensionExample(
                survey_context="Car brand perception (entity: Volvo)",
                response="safe but boring design",
                instance="safe",
                interpretation="vehicle safety reputation",
                abstraction="brand trust and reliability",
                domain="safety and engineering",
                facet="functional",

            ),
        ),
        clarification=(
            "Traits must be non-comparative and non-prioritized",
            "Attribute = descriptive property without value judgment (e.g., 'blue packaging', 'Dutch brand')",
        ),
    ),

    # ── 10. RELATIONS / DEPENDENCIES ────────────────────────────────────
    "RELATIONS_DEPENDENCIES": DimensionDefinition(
        key="RELATIONS_DEPENDENCIES",
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
            "Single-entity events or processes without inter-variable dependency (see BEHAVIOR_FUNCTION)",
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
        instruction="Identify each distinct relationship, dependency, or comparison described in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing exactly one relationship or dependency from the response.",
            interpretation_instruction="What does this instance MEAN in context? Name the concrete phenomenon or interpretation.",
            abstraction_instruction="What BROADER significance or higher-level theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the system, context, or sphere in which the relationship exists. \n"
                "Key idea: Domains specify in what sphere the relationship exists."
            ),
            domain_diagnostic="Question that needs to be answered: In what sphere does this relationship exist?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which relationships are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what type of relationship is described."
            ),
            facet_diagnostic="What type of relationship is described?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific relational feature or linkage being described. It is a named property that captures the precise nature of the connection between entities. \n"
                "Key idea: Attributes name the specific relational feature."
            ),
            attribute_diagnostic="What specific relationship is described?",
            code_diagnostic="This code is about the relationship between …",
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
                interpretation="weather dependency",
                abstraction="supply reliability risk",
                domain="supply reliability",
                facet="dependency",

            ),
            DimensionExample(
                survey_context="Retail ecosystem (entity: ShopLocal Platform)",
                response="small shops benefit from the shared delivery network",
                instance="benefit from the shared delivery network",
                interpretation="shared logistics advantage",
                abstraction="platform ecosystem value",
                domain="platform partnerships",
                facet="influence",

            ),
        ),
        clarification=(
            "Relation = dependency or influence between two or more variables; requires at least two entities connected causally or comparatively",
        ),
    ),

    # ── 11. GENERAL / OTHER (fallback) ────────────────────────────────────
    "GENERAL_OTHER": DimensionDefinition(
        key="GENERAL_OTHER",
        criterion="Does the response not clearly fit any of the above dimensions?",
        criterion_signals=(
            "Statements that do not match the criteria of the defined dimensions",
            "Very general or vague remarks",
            "Meta-responses (e.g., 'I don't know', 'no comment')",
            "Ambiguous or mixed statements without a dominant interpretation",
        ),
        exclusions=(
            "Statements that can reasonably be classified under an existing dimension",
        ),
        noun_phrase_descriptor="GENERAL / OTHER: responses that do not clearly fit a specific dimension",
        dimension_description=(
            "Use this dimension as a fallback when none of the defined dimensions "
            "clearly apply. This category captures general remarks, ambiguous responses, "
            "meta-responses, and other edge cases that cannot be reliably classified "
            "under the existing dimensions."
        ),
        allowed_concepts=(
            "general_remark",
            "comment",
            "statement",
            "note",
            "unspecified_response",
        ),
        pattern="[ANCHOR_SUBJECT] → [GENERAL_STATEMENT]",
        instruction="Identify the core idea expressed in the response.",
        prompt_rules=PromptRules(
            instance_instruction="Select the minimal verbatim span expressing the main idea of the response.",
            interpretation_instruction="What is the respondent really saying or expressing?",
            abstraction_instruction="What general type of remark or theme does this point to?",
            domain_instruction=(
                "Definition: A domain identifies the general subject area of the response. \n"
                "Key idea: Domains specify what subject the response relates to."
            ),
            domain_diagnostic="Question that needs to be answered: What is this about?",
            facet_instruction=(
                "Definition: A facet identifies the analytical lens through which general remarks are examined. Each facet must be independently analyzable. \n"
                "Key idea: Facets specify what type of general remark this is."
            ),
            facet_diagnostic="What type of remark is this?",
            attribute_instruction=(
                "Definition: An attribute identifies the specific feature or characteristic of the general remark. It is a named property that captures whatever concrete signal is present. \n"
                "Key idea: Attributes name whatever specific feature can be identified."
            ),
            attribute_diagnostic="What specific feature is mentioned?",
            code_diagnostic="This code is about …",
        ),
        anchor_slot=SlotDefinition(
            name="ANCHOR_SUBJECT",
            type_name="noun_phrase",
            required=True,
            guidance="The focal entity or phenomenon in {language}.",
        ),
        domain_slot=SlotDefinition(
            name="GENERAL_STATEMENT",
            type_name="noun_like_phrase",
            required=True,
            guidance="A concise phrase summarizing the general idea or statement.",
        ),
        examples=(
            DimensionExample(
                survey_context="Customer feedback (entity: TelcoProvider)",
                response="I don't really know what to say about them",
                instance="don't really know what to say",
                interpretation="onzekerheid over mening",
                abstraction="gebrek aan merkbetrokkenheid",
                domain="general",
                facet="uncertain response",

            ),
            DimensionExample(
                survey_context="City policy survey (entity: City Council)",
                response="they do many things, hard to summarize",
                instance="many things, hard to summarize",
                interpretation="moeilijk samen te vatten",
                abstraction="complexiteit van gemeentelijk beleid",
                domain="general",
                facet="general remark",

            ),
        ),
    ),
}


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
    """Return all dimensions in decision tree order (position 1-11)."""
    return [DIMENSIONS[key] for key in DIMENSION_DECISION_ORDER]
