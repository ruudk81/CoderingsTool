"""
Prompt builders and response models for step_3_ideaExtractor v4.

Organized by pipeline stage so each prompt builder is immediately followed
by the response model/schema that instructor injects — matching what the LLM sees.

Design principles:
- Builder functions: typed keyword-only args, return complete prompt strings
- Response models: validators REJECT invalid data (raise ValidationError), never auto-fix
- No ClassVar mutation — per-invocation model subclasses via class factory
- instructor retries with the ValidationError message, giving the LLM a chance to self-correct
"""

from __future__ import annotations
from typing import ClassVar, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, create_model

try:
    from .facet_data import FacetDefinition, PromptRules, resolve_slot_type
except ImportError:
    from facet_data import FacetDefinition, PromptRules, resolve_slot_type


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: Context Specifier Extraction
# ═══════════════════════════════════════════════════════════════════════


# --- 1a. Group 1: Speaker characteristics (lang / perspective / intent) ---

def build_context_specifier_group1_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
) -> str:
    """Build the Group 1 context specifier prompt (lang/perspective/intent)."""
    return f"""You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 1 specifiers (speaker characteristics):

1. **lang**: Language/dialect code
   - Identify the primary language and any dialect/regional variations
   - Format: ISO code
   - Examples: "nl-NL" (Dutch Netherlands), "en-GB" (British English)

2. **perspective**: Stakeholder viewpoint
   - From whose perspective are these responses given?
   - Examples "consumer", "client", "employee", "partner", "expert", "general_public", "beneficiary"

3. **intent**: Question intent (what the survey question asks respondents to do)
   - What cognitive task does the QUESTION itself ask respondents to perform?
   - Focus on the question's instruction, NOT on the content or tone of the responses
   - Examples: "associate" (name associations), "recall" (remember experiences), "enumerate" (list items), "describe" (characterize something), "suggest" (propose improvements), "evaluate" (judge/rate quality)

Provide concise answers (2-5 words each) in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class GenericSpecifierGroup1Response(BaseModel):
    """Group 1: Speaker characteristics"""
    lang: str = Field(
        description="Language/dialect code in ISO format",
        examples=["nl-NL", "en-GB", "de-DE"]
    )
    perspective: str = Field(
        description="Stakeholder viewpoint from whose perspective responses are given",
        examples=["consumer", "employee", "partner", "expert", "general_public", "beneficiary"]
    )
    intent: str = Field(
        description="Question intent - what cognitive task the survey question asks respondents to perform (not the tone of their answers)",
        examples=["associate", "recall", "enumerate", "describe", "evaluate", "suggest"]
    )


# --- 1b. Group 2: Subject matter (domain / topic / entity) ---

def build_context_specifier_group2_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
) -> str:
    """Build the Group 2 context specifier prompt (domain/topic/entity)."""
    return f"""You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 2 specifiers (subject matter):

1. **domain**: Industry or sector
   - What broad industry or sector does this survey belong to?
   - Use a high-level category (not a specific product or function)
   - If multiple industries apply, choose the dominant one
   - If none apply, use "general_consumer" or "unknown"
   - Examples:
     - "finance" (banking, insurance, investments)
     - "healthcare" (hospitals, clinics, medical services)
     - "education" (schools, universities, training)
     - "retail" (supermarkets, ecommerce, stores)

2. **topic**: Specific subject matter
   - What is the specific topic being discussed?
   - Examples: "brand_association" (brand perception), "customer_service" (support experience)

3. **entity**: Main entity of interest
   - What entity (group, person or thing) is the primary focus?
   - Use lowercase with underscores for multi-word names
   - Examples: "ing_bank", "tesla_model_3", "albert_heijn", "ns_trains"

Provide concise answers (2-5 words each) in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class GenericSpecifierGroup2Response(BaseModel):
    """Group 2: Subject matter"""
    domain: str = Field(
        description="Industry/sector domain the survey concerns",
        examples=["finance", "healthcare", "education", "retail"]
    )
    topic: str = Field(
        description="Specific subject matter being discussed",
        examples=["brand_association", "customer_service", "product_quality"]
    )
    entity: str = Field(
        description="Main entity of interest, lowercase_with_underscores",
        examples=["ing_bank", "tesla_model_3", "albert_heijn"]
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: Context Specifier Consolidation
# ═══════════════════════════════════════════════════════════════════════
# Response models: reuses GenericSpecifierGroup1Response / Group2Response
# from Stage 1 above.


# --- 2a. Consolidate Group 1 ---

def build_consolidate_specifiers_group1_prompt(
    *,
    survey_question: str,
    chunk_results: str,
) -> str:
    """Build the Group 1 specifier consolidation prompt."""
    return f"""You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 1 specifiers (speaker characteristics):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations (e.g., "evaluative" vs "assessment viewpoint" -> choose most accurate)
- For **lang**: Standardize to ISO format (e.g., "Dutch" -> "nl-NL", "English" -> "en-US")
- For **perspective**: Choose the most representative viewpoint across all chunks
- For **intent**: Choose the cognitive task the survey question asks respondents to perform (e.g., "associate", "recall", "evaluate"). Focus on the question's instruction, not on response content.

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 1 specifiers as valid JSON following the response schema provided."""


# --- 2b. Consolidate Group 2 ---

def build_consolidate_specifiers_group2_prompt(
    *,
    survey_question: str,
    chunk_results: str,
) -> str:
    """Build the Group 2 specifier consolidation prompt."""
    return f"""You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 2 specifiers (subject matter):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations (e.g., "financial services" vs "banking sector" -> choose most accurate)
- For **domain**: Standardize to lowercase, single/hyphenated word
- For **topic**: Choose the most representative subject matter across all chunks
- For **entity**: Standardize format (lowercase_with_underscores)

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 2 specifiers as valid JSON following the response schema provided."""


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: Primary Facet Selection
# ═══════════════════════════════════════════════════════════════════════


# --- 3a. Per-chunk facet scoring ---

def build_primary_facet_scoring_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
    perspective: str,
    intent: str,
    domain: str,
    entity: str,
    topic: str,
) -> str:
    """Build the primary facet scoring prompt for a single chunk."""
    return f"""You are selecting the SINGLE best primary facet for organizing a set of survey responses.

Your task is NOT to summarize responses, judge quality, or assign labels to each response.
Your ONLY goal is to decide which ONE facet best explains the MAIN way the responses DIFFER from one another.

Here is the language you will be working in:
<language>
{language}
</language>

Here is contextual information about the survey question:
<context>
- Domain: {domain}
- Entity of interest: {entity}
- Topic: {topic}
</context>

Here is the type of respondent who answered the question:
<respondent_type>
{perspective}
</respondent_type>

Here is the survey question that was asked:
<survey_question>
{survey_question}
</survey_question>

For reference, the question asks respondents to:
<question_intent>
{intent}
</question_intent>

NOTE: Question intent is background context only. Select the facet based on how responses actually DIFFER from each other, not based on the question's communicative task.

Here is a sample of SHORT, COARSE responses for you to analyze:
<sample_responses>
{chunk_responses}
</sample_responses>

------------------------------
HOW TO THINK ABOUT THE TASK
------------------------------
Ask yourself:
"If I had to cluster these responses into groups, which facet would create the most meaningful separation in answering the survey question?"

Choose the facet that explains the LARGEST share of meaningful variation across MOST responses (not just edge cases).

If multiple facets seem plausible:
- Choose the facet that would be used as the *top-level folder* to organize these responses.
- If still tied, choose the facet that applies to a larger fraction of responses.

------------------------------
Primary facets (choose exactly one)
------------------------------

1) DEFINITION_IDENTITY
- Differences are in how the entity is defined, categorized, named, or framed. What it is, why it exists.
- Excludes: properties/features (COMPOSITION_ATTRIBUTES), actions/processes (BEHAVIOR_FUNCTION), conditions/timing (CONDITIONS_CONTEXT), relationships (RELATIONS_INTERACTIONS), judgments/recommendations (EVALUATION_JUDGMENT).

2) COMPOSITION_ATTRIBUTES
- Differences are in properties, features, components, or qualities described. What it has, what it's like.
- Excludes: definitions/identity (DEFINITION_IDENTITY), actions/processes (BEHAVIOR_FUNCTION), conditions/timing (CONDITIONS_CONTEXT), relationships (RELATIONS_INTERACTIONS), judgments/recommendations (EVALUATION_JUDGMENT).

3) BEHAVIOR_FUNCTION
- Differences are in actions, processes, behaviors, effects, or outcomes described. What it does.
- Excludes: definitions/identity (DEFINITION_IDENTITY), properties/features (COMPOSITION_ATTRIBUTES), conditions/timing (CONDITIONS_CONTEXT), relationships (RELATIONS_INTERACTIONS), judgments/recommendations (EVALUATION_JUDGMENT).

4) CONDITIONS_CONTEXT
- Differences are in conditions, contexts, constraints, triggers, timing, or situations described. When, where, why it works or fails.
- Excludes: definitions/identity (DEFINITION_IDENTITY), properties/features (COMPOSITION_ATTRIBUTES), actions/processes (BEHAVIOR_FUNCTION), relationships (RELATIONS_INTERACTIONS), judgments/recommendations (EVALUATION_JUDGMENT).

5) RELATIONS_INTERACTIONS
- Differences are in relationships, stakeholders, dependencies, comparisons, or influences described. Who/what it connects to.
- Excludes: definitions/identity (DEFINITION_IDENTITY), properties/features (COMPOSITION_ATTRIBUTES), actions/processes (BEHAVIOR_FUNCTION), conditions/timing (CONDITIONS_CONTEXT), judgments/recommendations (EVALUATION_JUDGMENT).

6) EVALUATION_JUDGMENT
- Differences are in evaluative stances: judgments, recommendations, preferences, risk assessments, priorities. How it is assessed or what should be done.
- Excludes: definitions/identity (DEFINITION_IDENTITY), properties/features (COMPOSITION_ATTRIBUTES), actions/processes (BEHAVIOR_FUNCTION), conditions/timing (CONDITIONS_CONTEXT), relationships (RELATIONS_INTERACTIONS).

------------------------------
ANALYSIS PROCESS (internal)
------------------------------
Do NOT output your step-by-step reasoning.
You MUST still follow this process internally:
1) Identify the dominant pattern of variation across the sample by examining what the responses actually SAY and how they DIFFER from each other.
2) Score each facet for explanatory power over the variation: 0 = absent, 1 = present but secondary, 2 = primary.
3) Choose the single facet with the highest score (break ties using the rules above).
4) Extract 2-3 verbatim snippets from <sample_responses> that support the chosen facet.

All string values (including evidence snippets) must be in {language}.
Evidence snippets must be copied verbatim from <sample_responses>.
If fewer than 3 distinct snippets exist, include as many as possible without inventing any.
Clarification must explicitly contrast the chosen facet with at least one plausible alternative.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class PrimaryFacetChunkResponse(BaseModel):
    """LLM response for single chunk primary facet scoring."""
    primary_facet: Literal[
        "DEFINITION_IDENTITY",
        "COMPOSITION_ATTRIBUTES",
        "BEHAVIOR_FUNCTION",
        "CONDITIONS_CONTEXT",
        "RELATIONS_INTERACTIONS",
        "EVALUATION_JUDGMENT"
    ] = Field(
        description="The single best primary facet for organizing responses"
    )
    evidence: List[str] = Field(
        description="2-3 verbatim snippets from sample_responses supporting the chosen facet",
        examples=[["good service", "too expensive", "friendly staff"]]
    )
    clarification: str = Field(
        description="1-2 sentences explaining why this facet is most appropriate, contrasting with at least one alternative"
    )


# --- 3b. Facet consolidation ---

def build_primary_facet_consolidation_prompt(
    *,
    language: str,
    survey_question: str,
    domain: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    chunk_results: str,
) -> str:
    """Build the primary facet consolidation prompt."""
    return f"""You are a taxonomy consolidation specialist.
Your task is to analyze multiple chunk-level primary facet analyses and consolidate them into a single, coherent global primary facet for a survey question.

Here is the language the survey responses are written in:
<language>
{language}
</language>

Here is the survey question that was asked:
<survey_question>
{survey_question}
</survey_question>

Here is contextual information from prior analysis:
<context>
- Domain: {domain}: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

Here are the chunk-level analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

## YOUR TASK

You must consolidate these chunk-level analyses into a single global primary facet. Each chunk analysis evaluated the same survey question and produced a primary facet and supporting evidence. Your job is to synthesize these into one coherent framework.

## ANALYSIS STEPS

Follow these steps in order:

**Step 1: Review and consolidate chunk-level analyses**
Examine all chunk-level analyses carefully. Note areas of convergence and divergence. Identify which facets appear across multiple chunks and assess the quality of evidence supporting each.

**Step 2: Select the PRIMARY facet**
Choose the ONE facet (DEFINITION_IDENTITY, COMPOSITION_ATTRIBUTES, BEHAVIOR_FUNCTION, CONDITIONS_CONTEXT, RELATIONS_INTERACTIONS, or EVALUATION_JUDGMENT) that:
- Shows strong and consistent support across chunks
- Provides the clearest partition boundaries for coding responses
- Offers the best interpretability and stability for downstream use

Important: Do NOT select a facet solely because it appears most frequently. Favor partition clarity, boundary stability, and interpretability over raw frequency counts.

**Step 3: Define the GLOBAL primary facet**
Write a primary facet description that:
- Is specific to THIS survey question and response domain
- Clearly falls within the selected primary facet
- Reconciles and generalizes the chunk-level analyses without introducing new organizing principles
- Operates at a mid-level of abstraction (not too narrow, not too broad)
- Can directly seed downstream descriptive code labels
- Clearly indicates what coders should extract from each response

## DECISION RULES

When consolidating:
- If chunk analyses converge on the same facet, follow the consensus
- If chunk analyses diverge, rely on MECE quality (mutually exclusive, collectively exhaustive) to determine which facet provides the clearest boundaries
- Optimize for downstream coding usability and cross-coder consistency
- Prefer clarity and stability over cleverness or novelty

All output values must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class PrimaryFacetConsolidatedResponse(BaseModel):
    """Consolidated primary facet selection after merging all chunks."""
    primary_facet: str = Field(
        description="The selected primary facet",
        examples=["DEFINITION_IDENTITY", "COMPOSITION_ATTRIBUTES", "BEHAVIOR_FUNCTION",
                   "CONDITIONS_CONTEXT", "RELATIONS_INTERACTIONS", "EVALUATION_JUDGMENT"]
    )
    primary_facet_rationale: str = Field(
        description="2-4 sentence explanation of why this facet is the dominant organizing principle"
    )
    primary_facet_description: str = Field(
        description="Clear definition of the primary facet at proper abstraction level, specific to this survey question"
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: Concept Type Discovery
# ═══════════════════════════════════════════════════════════════════════


# --- 4a. Per-chunk discovery ---

def build_concept_type_discovery_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
    perspective: str,
    intent: str,
    domain: str,
    entity: str,
    topic: str,
    primary_facet: str,
    primary_facet_description: str,
    seed_examples: str,
) -> str:
    """Build the concept type discovery prompt for a single chunk."""
    return f"""You are a qualitative research methodologist analyzing survey responses.

Here is the language the survey responses are written in:
<language>
{language}
</language>

Here is the survey question that was asked:
<survey_question>
{survey_question}
</survey_question>

Here is contextual information from prior analysis:
<context>
- Domain: {domain}: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

The primary facet selected for this dataset is:
<primary_facet>
{primary_facet}
</primary_facet>

<primary_facet_description>
{primary_facet_description}
</primary_facet_description>

Here is a representative sample of {chunk_size} verbatim responses:
<sample_responses>
{chunk_responses}
</sample_responses>

Seed examples of concept types for this facet (for illustration only — you may deviate):
<seed_examples>
{seed_examples}
</seed_examples>

## YOUR TASK

Identify 5-15 **mutually exclusive concept types** that describe the **semantic roles** ideas can play within the {primary_facet} facet.

Concept types are SEMANTIC ROLES — they describe the KIND of thing an idea is within this facet, NOT the specific content or topic.

- GOOD: "Judgment" (a kind of evaluation — a semantic role)
- GOOD: "Recommendation" (a kind of evaluation — a semantic role)
- GOOD: "Physical Property" (a kind of attribute — a semantic role)
- BAD: "sustainability" (a specific topic, NOT a semantic role)
- BAD: "advertising" (a specific topic, NOT a semantic role)
- BAD: "customer service" (a specific topic, NOT a semantic role)

The same topic (e.g. sustainability) should fall consistently into ONE concept type (e.g. "Judgment" if it's an evaluative statement about sustainability), regardless of how the respondent phrases it.

## REQUIREMENTS

1. Concept types must be **mutually exclusive** — each idea should clearly belong to exactly one concept type
2. Concept types must be **collectively exhaustive** — every idea extractable from the sample should fit into at least one concept type
3. Each concept type needs a **snake_case key**, a **human-readable label**, and a **one-sentence definition** explaining what semantic role it captures
4. Aim for 5-15 concept types — enough to differentiate meaningfully, few enough to be useful
5. Concept types must be **semantic roles within the {primary_facet} facet**, not content topics

All output values (labels, definitions) must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class ConceptTypeItem(BaseModel):
    """A single concept type discovered from the data."""
    key: str = Field(
        description="Short snake_case identifier, e.g. 'judgment', 'recommendation'",
        examples=["judgment", "recommendation", "physical_property", "core_service"]
    )
    label: str = Field(
        description="Human-readable label in the response language, e.g. 'Oordeel', 'Aanbeveling'",
        examples=["Oordeel", "Aanbeveling", "Fysieke eigenschap", "Kerndienst"]
    )
    definition: str = Field(
        description="One-sentence definition of what semantic role this concept type captures"
    )


class ConceptTypeChunkResponse(BaseModel):
    """LLM response for single chunk concept type discovery."""
    concept_types: List[ConceptTypeItem] = Field(
        description="5-15 mutually exclusive concept types (semantic roles) discovered from the responses"
    )


# --- 4b. Consolidation ---

def build_concept_type_consolidation_prompt(
    *,
    language: str,
    survey_question: str,
    domain: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    primary_facet: str,
    chunk_results: str,
) -> str:
    """Build the concept type consolidation prompt."""
    return f"""You are a taxonomy consolidation specialist.
Your task is to merge multiple chunk-level concept type analyses into a single, coherent set of concept types.

Here is the language the survey responses are written in:
<language>
{language}
</language>

Here is the survey question that was asked:
<survey_question>
{survey_question}
</survey_question>

Here is contextual information from prior analysis:
<context>
- Domain: {domain}: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

The primary facet selected for this dataset is:
<primary_facet>
{primary_facet}
</primary_facet>

Here are the chunk-level concept type analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

## YOUR TASK

Consolidate these chunk-level concept type lists into a single set of 5-15 mutually exclusive concept types.

## CONSOLIDATION RULES

1. **Merge semantically equivalent concept types** — if multiple chunks produced similar types (e.g. "judgment" and "evaluative_stance"), merge them into one
2. **Preserve distinctions that appear across chunks** — if a concept type appears consistently across chunks, it reflects a real pattern in the data
3. **Drop concept types that only appeared in one chunk** and seem idiosyncratic
4. **Ensure MECE** — the final set must be mutually exclusive and collectively exhaustive
5. **Prefer broader, more stable concept types** over narrow ones — the goal is a robust partition that works across all responses
6. **Concept types must remain semantic roles**, not content topics — verify each consolidated type is still a semantic role within the {primary_facet} facet

All output values (labels, definitions) must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class ConceptTypeConsolidatedResponse(BaseModel):
    """Consolidated concept types after merging all chunks."""
    concept_types: List[ConceptTypeItem] = Field(
        description="5-15 mutually exclusive concept types (semantic roles), consolidated from all chunks"
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: Subject Extraction (dynamic model)
# ═══════════════════════════════════════════════════════════════════════

def build_taxonomy_subject_prompt(
    *,
    language: str,
    survey_question: str,
    perspective: str,
    domain: str,
    entity: str,
    topic: str,
    intent: str,
    facet: FacetDefinition,
) -> str:
    """Build the taxonomy-aware subject extraction prompt.

    All facet data comes from the typed FacetDefinition — no dict lookups.
    """
    # Resolve slot types for enriched guidance
    anchor_is_alias, anchor_short, _anchor_desc = resolve_slot_type(facet.anchor_slot.type_name)
    dim_is_alias, dim_short, _dim_desc = resolve_slot_type(facet.dimension_slot.type_name)

    anchor_guidance = facet.anchor_slot.guidance.replace("{language}", language)
    dim_guidance = facet.dimension_slot.guidance.replace("{language}", language)

    # Build slot lines with type hints
    anchor_line = f"- [{facet.anchor_slot.name}]: {anchor_guidance}"
    if anchor_is_alias:
        anchor_line += f" Form: {anchor_short}."

    dim_line = f"- [{facet.dimension_slot.name}]: {dim_guidance}"
    if dim_is_alias:
        dim_line += f" Form: {dim_short}."

    slots_text = f"{anchor_line}\n{dim_line}"
    allowed_concepts_str = ", ".join(facet.allowed_concepts)

    return f"""You are generating a phrasing template for structured survey response analysis in {language}.

<survey_question>{survey_question}</survey_question>

<context>
Respondent type: {perspective} | Domain: {domain} | Entity: {entity} | Topic: {topic} | Question intent: {intent}
</context>

<primary_facet>
{facet.dimension_description}
</primary_facet>

<template>
Pattern: "{facet.pattern}"
Slots:
{slots_text}
</template>

---

**Your task** (three steps):

1. **canonical_term** -- Pick a short noun phrase (in {language}) for the ANCHOR slot. Avoid respondent pronouns unless the question demands first-person framing.

2. **taxonomy_actionable_type** -- Select the single best concept type from: {allowed_concepts_str}

3. **canonical_phrasing** -- Build a single-clause sentence following the pattern above.
   - Replace ANCHOR with your canonical_term from step 1.
   - Keep the literal marker token {facet.dimension_marker} for the DIMENSION slot.
   - Must read as a natural answer to the survey question once the marker is replaced.
   - Aim for 10 words or fewer (excluding the marker).

**Rules**: Normalize meaning, not style. Do not introduce entities, motives, or outcomes absent from the survey question. Single clause only.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class SubjectExtractionResponse(BaseModel):
    """Base model for subject extraction. Use create_subject_model() for facet-specific versions."""
    canonical_term: str = Field(
        description="Short noun phrase for the anchor slot of the template"
    )
    taxonomy_actionable_type: str = Field(
        description="The chosen concept type on the primary facet",
    )
    canonical_phrasing: str = Field(
        description="Single grammatical clause with the dimension marker token as placeholder"
    )


def create_subject_model(
    *,
    facet: FacetDefinition,
) -> type[SubjectExtractionResponse]:
    """Create a facet-specific SubjectExtractionResponse with STRICT validation.

    The dimension_marker is baked into the model subclass at creation time.
    No ClassVar mutation. Each call returns a fresh class. Safe for async.
    """
    allowed_str = ", ".join(facet.allowed_concepts)
    _marker = facet.dimension_marker

    # Resolve type hints for field descriptions
    anchor_is_alias, anchor_short, anchor_desc = resolve_slot_type(facet.anchor_slot.type_name)
    dim_is_alias, dim_short, dim_desc = resolve_slot_type(facet.dimension_slot.type_name)

    anchor_hint = f" {anchor_desc}" if anchor_desc else ""
    dim_hint = f" Dimension slot must be a {facet.dimension_slot.type_name}." if dim_is_alias else ""

    phrasing_desc = (
        f"Single grammatical clause following: {facet.pattern}. "
        f"Must contain the literal marker token {_marker}."
        f"{dim_hint}"
    )

    class FacetSubjectModel(SubjectExtractionResponse):
        _dim_marker: ClassVar[str] = _marker

        canonical_term: str = Field(
            description=f"Short noun phrase: {facet.noun_phrase_descriptor}.{anchor_hint}"
        )
        taxonomy_actionable_type: str = Field(
            description=f"The actionable taxonomy type on the {facet.key} facet. Must be one of: {allowed_str}."
        )
        canonical_phrasing: str = Field(description=phrasing_desc)

        @field_validator('canonical_phrasing', mode='before')
        @classmethod
        def validate_marker_present(cls, v: str) -> str:
            """STRICT: Reject if dimension marker is missing."""
            if isinstance(v, str) and cls._dim_marker not in v:
                raise ValueError(
                    f"canonical_phrasing must contain the dimension marker token "
                    f"{cls._dim_marker!r}, but got: {v!r}. "
                    f"Please include {cls._dim_marker} in your canonical_phrasing."
                )
            return v

    FacetSubjectModel.__name__ = f"SubjectExtractionResponse_{facet.key}"
    FacetSubjectModel.__qualname__ = f"SubjectExtractionResponse_{facet.key}"
    return FacetSubjectModel


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: Idea Extraction (dynamic model)
# ═══════════════════════════════════════════════════════════════════════

def build_taxonomy_enriched_extraction_prompt(
    *,
    language: str,
    var_lab: str,
    perspective: str,
    domain: str,
    entity: str,
    topic: str,
    intent: str,
    respondent_id: str,
    response: str,
    canonical_phrasing: str,
    facet: FacetDefinition,
    concept_type_table: str,
    priority_rules: str,
) -> str:
    """Build the taxonomy-enriched idea extraction prompt.

    All facet-specific data (dimension_marker, noun_phrase_descriptor,
    slot_guidance, instruction) comes from the FacetDefinition.
    """
    # Build enriched dimension guidance
    dim_guidance = facet.dimension_slot.guidance.replace("{language}", language)
    dim_is_alias, dim_short, _dim_desc = resolve_slot_type(facet.dimension_slot.type_name)
    if dim_is_alias:
        dim_guidance += f" Form: {dim_short}."

    return f"""You are an expert in extracting structured ideas from survey responses using taxonomy-aware analysis.
Your task is to identify distinct ideas in a survey response, reformulate each idea using a canonical template, and produce a lightweight taxonomy classification for each idea.

**SURVEY CONTEXT**
<survey_context>
You will be working in the following language: {language}

Here is the survey question being analyzed: "{var_lab}"

Here is the context for the survey:
- Type of respondent: {perspective}
- Domain: {domain}
- Entity of interest: {entity}
- Topic: {topic}
- Question intent: {intent}
</survey_context>

**TAXONOMY FRAMEWORK**
<taxonomy_instructions>
Taxonomy lens: "{facet.noun_phrase_descriptor}"

Instruction: {facet.instruction}

The pattern: {canonical_phrasing}

{facet.dimension_marker} must be replaced with: {dim_guidance}
</taxonomy_instructions>

**RESPONSE TO ANALYZE**
<response>
Respondent ID: {respondent_id}
Response: {response}
</response>

---

Now follow these three steps to complete the extraction:

**STEP 1: IDENTIFY AND SPLIT IDEAS**

Identify all conceptually distinct ideas in the response, interpreted in light of the survey question. Assign each a sequential idea_id starting from "1".

CRITICAL SPLITTING RULES:
- Items joined by conjunctions ("and", "or", "en", "und", "et", "y", "ou") or commas that refer to different concepts MUST be split into separate ideas
- Example: "faster and cheaper" -> TWO ideas: "faster" (idea 1) and "cheaper" (idea 2)
- Each split idea gets its OWN template instantiation and its OWN taxonomy classification
- When in doubt, err on the side of splitting

**STEP 2: COMPLETE THE PATTERN**

For each idea, reformulate it using the pattern provided in the taxonomy framework.

Rules:
- Keep the fixed pattern prefix exactly as provided (do NOT alter it)
- Replace the marker token {facet.dimension_marker} with the shortest verbatim span from the original response that expresses the idea
- Use the exact respondent_id provided in the response

**STEP 3: CLASSIFY EACH IDEA**

For each idea, assign a semantic classification in the specified language.

1. **INSTANCE**: The verbatim span from the response expressing the idea (cleaned and minimally standardized)

2. **NODE**: A canonical, reusable concept label
   - Remove descriptive qualifiers (e.g., adjectives like "slow," "cheap," "great")
   - Reduce to the base object, action, or concept (noun phrase)
   - Must stay semantically equivalent to the instance

3. **CONCEPT TYPE**: Classify the idea into one of the discovered concept types for the {facet.key} facet:
{concept_type_table}

4. **VALENCE**: The evaluative direction of the idea
   - "positive": favorable, desirable, praising
   - "negative": unfavorable, undesirable, critical
   - "neutral_mixed": factual, balanced, or ambiguous

5. **AGENCY FOCUS**: Who or what is the primary agent or focus
   - "system_entity": the entity/system being discussed
   - "stakeholder_actor": an external stakeholder or actor
   - "respondent": the respondent themselves
   - "": leave empty if unclear or not applicable

6. **PRESCRIPTIVENESS**: Is the idea descriptive or prescriptive?
   - "descriptive": describes what is or was
   - "prescriptive": suggests what should be or could be done
   - "": leave empty if unclear

**Priority Rules** (apply in order when uncertain):
{priority_rules}

---

**EDGE CASES**
- If the response is empty, irrelevant, or nonsensical: return []
- If there is one idea: return an array with one object
- If there are multiple ideas: return an array with multiple objects

**OUTPUT**

Begin processing now and provide your output as valid JSON matching the required response schema.
CRITICAL: all output fields must be in {language}
"""


class SemanticTaxonomyResponse(BaseModel):
    """Semantic taxonomy classification for extracted ideas.
    4-layer classification + secondary facets:
    instance, node, concept_type, valence, agency_focus, prescriptiveness."""

    instance: str = ""
    node: str = ""
    concept_type: str = ""
    valence: Literal["positive", "negative", "neutral_mixed"] = "neutral_mixed"
    agency_focus: Literal["system_entity", "stakeholder_actor", "respondent", ""] = ""
    prescriptiveness: Literal["descriptive", "prescriptive", ""] = ""

    @field_validator('instance', 'node', mode='before')
    @classmethod
    def reject_invalid(cls, v: object) -> str:
        """STRICT: Reject None and non-string values instead of auto-fixing."""
        if v is None:
            raise ValueError("Field must not be None. Provide a non-empty string value.")
        if not isinstance(v, str):
            raise TypeError(f"Expected str, got {type(v).__name__}: {v!r}")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field must not be empty after stripping whitespace.")
        return stripped.lower().rstrip('.,;:!?')


class TaxonomyEnrichedIdeaResponse(BaseModel):
    """Base model for extraction. Use create_extraction_model() for facet-specific versions."""
    respondent_id: str = Field(
        description="Respondent identifier from the response context"
    )
    idea_id: str = Field(
        description="Sequential number as string",
        examples=["1", "2", "3"]
    )
    idea: str = Field(
        description="Complete idea statement beginning with the canonical_phrasing template"
    )
    taxonomy: Optional[SemanticTaxonomyResponse] = Field(
        default=None,
        description="Semantic taxonomy: instance -> node -> concept_type + valence, agency_focus, prescriptiveness"
    )


# Facet-specific Dutch examples for semantic taxonomy fields.
_TAXONOMY_EXAMPLES = {
    "DEFINITION_IDENTITY": {
        "instance": ["een betrouwbare bank", "innovatief bedrijf", "traditionele instelling"],
        "node": ["betrouwbaarheid", "innovatie", "traditie"],
        "concept_type": ["definition", "categorization", "framing"],
        "valence": ["positive", "positive", "neutral_mixed"],
    },
    "COMPOSITION_ATTRIBUTES": {
        "instance": ["goede service", "te dure producten", "modern design"],
        "node": ["klantenservice", "prijsniveau", "productontwerp"],
        "concept_type": ["quality_measure", "functional_feature", "physical_property"],
        "valence": ["positive", "negative", "positive"],
    },
    "BEHAVIOR_FUNCTION": {
        "instance": ["meer personeel inzetten", "sneller reageren"],
        "node": ["personeelsbezetting", "reactiesnelheid"],
        "concept_type": ["core_service", "process_step"],
        "valence": ["neutral_mixed", "negative"],
    },
    "CONDITIONS_CONTEXT": {
        "instance": ["te lang wachten", "in het weekend"],
        "node": ["wachttijd", "weekendperiode"],
        "concept_type": ["constraint", "temporal_pattern"],
        "valence": ["negative", "neutral_mixed"],
    },
    "RELATIONS_INTERACTIONS": {
        "instance": ["oudere klanten", "samenwerking met partners"],
        "node": ["senioren", "partnerschap"],
        "concept_type": ["stakeholder_link", "partnership"],
        "valence": ["neutral_mixed", "positive"],
    },
    "EVALUATION_JUDGMENT": {
        "instance": ["te duur voor wat je krijgt", "zou meer aandacht moeten besteden"],
        "node": ["prijskwaliteitverhouding", "aandachtsniveau"],
        "concept_type": ["judgment", "recommendation"],
        "valence": ["negative", "negative"],
    },
}


def create_extraction_model(
    *,
    facet: FacetDefinition,
    template_prefix: str,
    concept_types: list[ConceptTypeItem] | None = None,
) -> type[TaxonomyEnrichedIdeaResponse]:
    """Create facet-specific extraction model with STRICT validation.

    template_prefix and dimension_marker are baked in via class closure.
    No ClassVar. Each call returns a fresh class. Safe for async.
    """
    _prefix = template_prefix.strip() if template_prefix else ""
    _marker = facet.dimension_marker
    prompt_rules = facet.prompt_rules
    facet_key = facet.key

    # Resolve dimension type for node description enrichment
    dim_is_alias, dim_short, dim_desc = resolve_slot_type(facet.dimension_slot.type_name)
    node_desc = prompt_rules.node_instruction
    if dim_is_alias:
        node_desc += f" Must be a {facet.dimension_slot.type_name}."

    # Facet-specific examples
    ex = _TAXONOMY_EXAMPLES.get(facet_key, {})

    # Build concept_type field
    if concept_types:
        concept_type_field = (
            Literal[tuple(c.key for c in concept_types)],
            Field(
                description=(
                    "Concept type (semantic role within the facet). One of: " +
                    ", ".join(f"{c.key} ({c.definition})" for c in concept_types)
                ),
                examples=[c.key for c in concept_types[:3]]
            )
        )
    else:
        concept_type_field = (
            str,
            Field(
                description=(
                    f"Concept type: a short snake_case semantic role label describing HOW this idea "
                    f"relates to the {facet_key} facet (e.g., quality_measure, functional_feature, "
                    f"moral_attribute, symbolic_element). Must be a reusable ROLE, not a topic."
                ),
                examples=ex.get("concept_type", [])
            )
        )

    # Create facet-specific SemanticTaxonomyResponse
    FacetTaxonomy = create_model(
        f"SemanticTaxonomyResponse_{facet_key}",
        __base__=SemanticTaxonomyResponse,
        instance=(str, Field(
            description=prompt_rules.instance_instruction,
            examples=ex.get("instance", [])
        )),
        node=(str, Field(
            description=node_desc,
            examples=ex.get("node", [])
        )),
        concept_type=concept_type_field,
        valence=(
            Literal["positive", "negative", "neutral_mixed"],
            Field(
                description="Evaluative direction: positive (favorable), negative (unfavorable), neutral_mixed (factual/balanced/ambiguous)",
                examples=ex.get("valence", ["positive", "negative", "neutral_mixed"])
            )
        ),
        agency_focus=(
            Literal["system_entity", "stakeholder_actor", "respondent", ""],
            Field(
                default="",
                description="Primary agent/focus: system_entity, stakeholder_actor, respondent, or empty if unclear",
                examples=["system_entity", "stakeholder_actor", "respondent", ""]
            )
        ),
        prescriptiveness=(
            Literal["descriptive", "prescriptive", ""],
            Field(
                default="",
                description="Descriptive (what is/was) vs prescriptive (what should be). Empty if unclear.",
                examples=["descriptive", "prescriptive", ""]
            )
        ),
    )

    # Create facet-specific extraction model with strict validators
    class FacetExtractionModel(TaxonomyEnrichedIdeaResponse):
        _template_prefix: ClassVar[str] = _prefix
        _dimension_marker: ClassVar[str] = _marker

        idea: str = Field(
            description=(
                f"Idea following the {facet_key} pattern: {facet.pattern}. "
                f"Must begin with the canonical_phrasing template."
            )
        )
        taxonomy: Optional[FacetTaxonomy] = Field(
            default=None,
            description="Semantic taxonomy: instance -> node -> concept_type + valence, agency_focus, prescriptiveness"
        )

        @field_validator('idea', mode='before')
        @classmethod
        def validate_idea(cls, v: str) -> str:
            """STRICT: Reject if template prefix is missing or marker not replaced."""
            if not isinstance(v, str) or not v.strip():
                raise ValueError("idea must be a non-empty string.")
            v = v.strip()
            if cls._template_prefix and not v.lower().startswith(cls._template_prefix.lower()):
                raise ValueError(
                    f"idea must start with the canonical phrasing template "
                    f"{cls._template_prefix!r}, but starts with: {v[:len(cls._template_prefix)+10]!r}. "
                    f"Please begin your idea with: {cls._template_prefix}"
                )
            if cls._dimension_marker and cls._dimension_marker in v:
                raise ValueError(
                    f"The marker token {cls._dimension_marker!r} must be replaced with actual content. "
                    f"Do not include the literal marker in the final idea."
                )
            return v

    FacetExtractionModel.__name__ = f"TaxonomyEnrichedIdeaResponse_{facet_key}"
    FacetExtractionModel.__qualname__ = f"TaxonomyEnrichedIdeaResponse_{facet_key}"
    return FacetExtractionModel
