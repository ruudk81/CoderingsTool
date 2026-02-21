"""
Prompt builders and response models for step_3_ideaExtractor v5.

v5 overhaul: 10 MECE facets with decision-tree ordering (apply in order, stop at first fit).
Replaces v4's 6-facet scoring-based system. Prescriptiveness secondary facet removed.

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
    from .facet_data import FacetDefinition, PromptRules, resolve_slot_type, get_facets_in_decision_order
except ImportError:
    from facet_data import FacetDefinition, PromptRules, resolve_slot_type, get_facets_in_decision_order


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
# STAGE 3: Primary Facet Selection (Decision Tree)
# ═══════════════════════════════════════════════════════════════════════

# All 10 facet keys for Literal type validation
_ALL_FACET_KEYS = (
    "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS", "IDENTITY_DEFINITION",
    "ACTORS_TARGETS", "CONTEXT_CONDITIONS", "MOTIVATIONS_DRIVERS",
    "EXPERIENCE_PERCEPTION", "EVALUATION_PRIORITIZATION",
    "BEHAVIOR_FUNCTION", "ATTRIBUTES_ASSOCIATIONS", "RELATIONS_DEPENDENCIES",
)


def _build_decision_tree_block() -> str:
    """Build the decision tree block dynamically from facet_data.py definitions.

    Keeps prompt and data in sync — criterion text lives in FacetDefinition only.
    Includes criterion signals, optional clarification, and exclusions for disambiguation.
    """
    facets = get_facets_in_decision_order()
    emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    lines = []
    for facet, emoji in zip(facets, emoji_numbers):
        signals = "\n".join(f"  • {s}" for s in facet.criterion_signals)
        exclusions = "\n".join(f"  ✗ {e}" for e in facet.exclusions)
        block = (
            f"{emoji} {facet.key}\n"
            f"  {facet.criterion}\n"
            f"  **Criterion signals**\n"
            f"{signals}\n"
        )
        if facet.clarification:
            clarification = "\n".join(f"  • {c}" for c in facet.clarification)
            block += f"  **Clarification**\n{clarification}\n"
        block += (
            f"  **Exclusions**\n"
            f"{exclusions}\n"
            f"  ➡ If YES → select {facet.key}"
        )
        lines.append(block)
    return "\n\n".join(lines)


# --- 3a. Per-chunk facet selection (decision tree) ---

def build_primary_facet_decision_tree_prompt(
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
    """Build the primary facet decision tree prompt for a single chunk.

    Replaces the v4 scoring prompt. Uses ordered decision tree (stop at first fit).
    """
    decision_tree = _build_decision_tree_block()

    return f"""You are selecting the SINGLE best primary facet for organizing a set of open-ended responses.
Your task is not to summarize responses or label each one. Your task is to identify the main semantic axis along which the responses DIFFER.

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

NOTE: Question intent is background context only. Select the facet based on how responses actually DIFFER from each other, not based on the question's communicative task. Do not infer intent from the question — use actual response differences.

Here is a sample of SHORT, COARSE responses for you to analyze:
<sample_responses>
{chunk_responses}
</sample_responses>

------------------------------
HOW TO THINK ABOUT THE TASK
------------------------------
Ask yourself:
"If I had to organize these responses into groups, which facet would best explain the biggest, most meaningful differences between them?"

Choose the facet that:
* Explains variation across most responses
* Creates the clearest top-level separation
* Would naturally be used as the first folder when organizing insights

If multiple facets seem plausible:
1. Choose the facet that applies to a larger share of responses
2. If still tied, choose the facet earlier in the decision order

------------------------------
DECISION TREE (Apply in Order, Stop at First Fit)
------------------------------

{decision_tree}

------------------------------
RULES (Do Not Skip)
------------------------------
* Select exactly one facet.
* Apply the decision tree steps in order (1 through 10). Stop at the FIRST step where the answer is clearly YES for the dominant variation.
* Base your decision on dominant variation, not edge cases.
* Facets are organizational lenses, not labels for individual responses.

------------------------------
ANALYSIS PROCESS (internal)
------------------------------
Do NOT output your step-by-step reasoning.
You MUST still follow this process internally:
1) Read through the sample responses and identify the dominant pattern of variation.
2) Walk through the decision tree from step 1 to step 10.
3) For each step, ask: "Do most responses mainly differ along THIS axis?" If clearly YES, stop and select that facet.
4) Record which decision tree step triggered your selection.
5) Extract 2-3 verbatim snippets from <sample_responses> that support the chosen facet.

All string values (including evidence snippets) must be in {language}.
Evidence snippets must be copied verbatim from <sample_responses>.
If fewer than 3 distinct snippets exist, include as many as possible without inventing any.
Clarification must explicitly contrast the chosen facet with at least one plausible alternative.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class PrimaryFacetChunkResponse(BaseModel):
    """LLM response for single chunk primary facet selection (decision tree)."""
    primary_facet: Literal[
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
    ] = Field(
        description="The single best primary facet for organizing responses"
    )
    decision_tree_stop_position: int = Field(
        description="Which decision tree step (1-10) triggered the selection",
        ge=1, le=10,
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
    facet_keys_str = ", ".join(_ALL_FACET_KEYS)

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
- Domain: {domain}
- Entity of interest: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

Here are the chunk-level analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

## YOUR TASK

You must consolidate these chunk-level analyses into a single global primary facet. Each chunk analysis used a 10-step decision tree and produced a primary facet with supporting evidence. Your job is to synthesize these into one coherent framework.

## ANALYSIS STEPS

Follow these steps in order:

**Step 1: Review and consolidate chunk-level analyses**
Examine all chunk-level analyses carefully. Note areas of convergence and divergence. Identify which facets appear across multiple chunks and assess the quality of evidence supporting each.

**Step 2: Select the PRIMARY facet**
Choose the ONE facet ({facet_keys_str}) that:
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
- When chunks are split, prefer the facet that appears earlier in the decision tree order (the tree is designed so earlier steps capture more common variation patterns)
- Optimize for downstream coding usability and cross-coder consistency
- Prefer clarity and stability over cleverness or novelty

All output values must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class PrimaryFacetConsolidatedResponse(BaseModel):
    """Consolidated primary facet selection after merging all chunks."""
    primary_facet: str = Field(
        description="The selected primary facet",
        examples=list(_ALL_FACET_KEYS),
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
- Domain: {domain}
- Entity of interest: {entity} 
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

## YOUR TASK

Identify 5–15 **mutually exclusive, collectively exhaustive thematic domains** that represent **distinct domains of relevance, impact, or meaning for {entity}**, as evidenced by the responses.

These thematic domains should be defined from the point of view of the topic, describing *what it means for {entity}, rather than grouping or summarizing what respondents said.

## GUIDANCE

- Treat survey responses as **evidence**, not as the units being categorized.
- Each thematic domain should describe **a structural domain in which {entity} is situated, affected, evaluated, or understood**.
- Think of thematic domains as **section headers in a research report about {entity}**, not as topics, sentiments, or response types.
- The same underlying concept should always map to **one and only one** thematic domain, regardless of wording or opinion.

## REQUIREMENTS

1. Thematic domains must be **mutually exclusive** — each concept should clearly belong to exactly one domain.
2. Thematic domains must be **collectively exhaustive** — every idea extractable from the sample should fit into at least one domain.
3. Each thematic domain must include:
   - a **human-readable label**
   - a **one-sentence definition** explaining the domain of relevance or implication for {entity}
4. Aim for **5–15 thematic domains** — enough to differentiate meaningfully, few enough to be analytically useful.
5. Thematic domains must organize **domains of relevance for {entity}**, not linguistic forms, response styles, or abstract role labels (e.g., avoid “sentiment,” “opinion type,” “functional trait”).

All output values (labels and definitions) must be in {language}.

Begin processing now and provide your output as **valid JSON** following the response schema provided."""


class ConceptTypeItem(BaseModel):
    """A single thematic domain discovered from the data."""
    key: str = Field(
        description="Short natural-language identifier (1-4 words, no underscores)",
        examples=["access and logistics", "value proposition", "hospitality and interaction"]
    )
    label: str = Field(
        description="Human-readable label in the response language",
        examples=["Toegang en logistiek", "Waardepropositie", "Gastvrijheid en interactie"]
    )
    definition: str = Field(
        description="One-sentence definition of what ASPECT of the entity this thematic domain covers"
    )


class ConceptTypeChunkResponse(BaseModel):
    """LLM response for single chunk concept type discovery."""
    concept_types: List[ConceptTypeItem] = Field(
        description="5-15 mutually exclusive thematic domains discovered from the responses"
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

Consolidate these chunk-level thematic domain lists into a single set of 5-15 mutually exclusive thematic domains.

## CONSOLIDATION RULES

1. **Merge semantically equivalent domains** — if multiple chunks produced similar domains (e.g. "access and logistics" and "service accessibility"), merge them into one
2. **Preserve distinctions that appear across chunks** — if a domain appears consistently across chunks, it reflects a real pattern in the data
3. **Absorb, never drop** — if a domain appeared in only one chunk, either keep it (if it represents a genuine aspect of {entity}) or explicitly merge it into the most semantically related broader domain. Never silently discard a domain — every domain from every chunk must be accounted for in the final set (kept, merged, or split).
4. **Ensure MECE** — the final set must be mutually exclusive and collectively exhaustive
5. **Prefer precision over breadth** — create domains that are specific enough to be analytically useful. Only broaden a domain when two chunk-level domains genuinely describe the same aspect.
6. **Domains must organize ASPECTS of the entity**, not be linguistic role labels — verify each consolidated domain describes a thematic aspect, not a semantic role like "moral attribute" or "functional trait"

All output values (labels, definitions) must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class ConceptTypeConsolidatedResponse(BaseModel):
    """Consolidated thematic domains after merging all chunks."""
    concept_types: List[ConceptTypeItem] = Field(
        description="5-15 mutually exclusive thematic domains, consolidated from all chunks"
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
Respondent type: {perspective} 
Domain: {domain} 
Entity of interest: {entity} 
Topic: {topic} 
Question intent: {intent}
</context>

<primary_facet>
{facet.noun_phrase_descriptor}

Usage:
{facet.dimension_description}
</primary_facet>

<template>
Pattern: "{facet.pattern}"
Slots:
{slots_text}
</template>

---

**Your two tasks**:

1. **canonical_term** -- Pick a short noun phrase (in {language}) for the ANCHOR slot. Avoid respondent pronouns unless the question demands first-person framing.

2. **canonical_phrasing** -- Build a single-clause sentence following the pattern above.
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

    FacetSubjectModel.__name__ = f"SubjExtr_{facet.key}"
    FacetSubjectModel.__qualname__ = f"SubjExtr_{facet.key}"
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

First, here is the survey context you'll be working with:

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

Next, here is the taxonomy framework you should apply:

<taxonomy_lens>
Taxonomy lens: "{facet.noun_phrase_descriptor}"
Anchor: {facet.dimension_marker}
</taxonomy_lens>

Here is the survey response you need to analyze:

<response>
Respondent ID: {respondent_id}
Response: {response}
</response>

----

## TASK OVERVIEW (READ CAREFULLY)

You must convert the survey response into a structured list of ATOMIC ideas.
Each output object represents EXACTLY ONE distinct idea about {entity}, expressed using a fixed canonical template and classified using the provided taxonomy lens.

CRITICAL CONTRACT:
- One idea = one object
- When in doubt → SPLIT
- Over-splitting is preferred to under-splitting
- All outputs must strictly follow the schema and validation rules

---

## STEP 1 — IDENTIFY ATOMIC IDEAS

Carefully read the response and extract ALL distinct ideas.

{facet.instruction}

CRITICAL SPLITTING RULES (NON-NEGOTIABLE):
- Any items joined by conjunctions ("and", "or", "en", "und", "et", "y", "ou") or commas that express DIFFERENT concepts MUST be split
- Example:
  - "faster and cheaper" → TWO ideas:
    1. "faster"
    2. "cheaper"
- Each idea MUST receive:
  - its own canonical phrasing
  - its own taxonomy classification
- If unsure whether something is one idea or two → SPLIT

---

## STEP 2 — REFORMULATE EACH IDEA

For EACH atomic idea, produce an idea statement using EXACTLY this pattern:

{canonical_phrasing}

Rules (STRICT):
- Do NOT alter the template prefix
- Replace the marker token {facet.dimension_marker} with the SHORTEST verbatim span from the response that expresses the idea
- Do NOT include the literal marker token in the final output
- Use the EXACT respondent_id: {respondent_id}
- Preserve the original meaning
- Use the same language as the response ({language})

---

## STEP 3 — PROVIDE AN ABSTRACTION LADDER USING THE TAXONOMY LENS

For each idea, provide an abstraction ladder with the following fields (all in {language}):

1. INSTANCE  
   - Literal wording from the response (cleaned, minimal normalization)

2. CONCEPT (interpretive meaning in context)
   - What is the respondent REALLY talking about when they say this?
   - Name the underlying thing, phenomenon, or idea the instance refers to in the context of {entity} and {domain}
   - This requires INTERPRETATION, not just normalization or nominalization
   - Different surface expressions that point to the same underlying meaning should map to the same concept
   - NOT a spelling fix, nominalization, or synonym of the instance

3. CONCEPT TYPE (thematic domain)
   - Classify each concept into the single most specific thematic domain
   - Classification guidance:
        - A thematic domain should describe a structural domain in which {entity} is situated, affected, evaluated, or understood
        - Think of thematic domains as **section headers in a research report about {entity}**, not as topics, sentiments, or response types
        - Ask yourself: Which distinct domain of relevance, impact, or meaning for {entity} does the concept identified in step 2 belong to?
        - Must be a high-level thematic category suitable for organizing a codebook
        - Should be reusable across many different concepts that relate to the same aspect
        - All labels must be in {language}
    - {concept_type_table}

4. CONCEPT TYPE DEFINITION (contextual framing)
    - One short phrase (2–5 words) explaining what this thematic domain REPRESENTS for {entity} in this survey context
    - Frames WHY this domain matters — what larger question or concern does it address?
    - The same definition should apply to all concepts within the same concept type
    - NOT a paraphrase, synonym, or translation of the concept type

## ABSTRACTION LADDER EXAMPLES (study the PRINCIPLE, not the content)

These examples are from OTHER surveys in English. Your output must be in {language}.

### Example A — Healthcare satisfaction survey (entity: City Hospital)
Response: "long wait at the reception desk"
  WRONG: long wait at reception → long wait at reception → functional trait → operational characteristic
  RIGHT: long wait at reception → appointment scheduling → access and logistics → patient journey efficiency
  WHY: The respondent is really talking about appointment scheduling (concept). This belongs to the "access and logistics" aspect of the hospital (concept type). That aspect represents "patient journey efficiency" for this entity (definition).

### Example B — Public transport survey (entity: Metro Line 5)
Response: "always running late"
  WRONG: running late → running late → quality judgment → perception of characteristics
  RIGHT: running late → schedule reliability → operations and planning → core service promise
  WHY: "running late" points to schedule reliability (concept). This falls under "operations and planning" (concept type). For a transit line, that domain represents the "core service promise" (definition).

### Example C — Restaurant review survey (entity: Bistro Roma)
Response: "friendly staff and small portions"
  Idea 1: friendly staff
    WRONG: friendly staff → friendly staff → trait → characteristic
    RIGHT: friendly staff → warmth of service → hospitality and interaction → dining experience
  Idea 2: small portions
    WRONG: small portions → small portions → quality measure → objective property
    RIGHT: small portions → portion sizing → value proposition → price-quality balance

KEY PRINCIPLE: Each level answers a different question:
  - INSTANCE: What did they SAY?
  - CONCEPT: What are they REALLY TALKING ABOUT?
  - CONCEPT TYPE: Which ASPECT of {entity} does this relate to?
  - CONCEPT TYPE DEFINITION: What does this aspect REPRESENT in this survey context?

5. VALENCE 

For each idea, assign directional valence relative to the CONCEPT, as framed by its CONCEPT TYPE.
Valence indicates the directional effect the idea has on the concept — not sentiment, opinion, or overall evaluation of the entity.

Use exactly one of the following values:
- "+" = the idea strengthens, increases, or reinforces the concept
- "-" = the idea weakens, decreases, or undermines the concept
- "0" = the idea is non-directional with respect to the concept

Rules (STRICT):
Assign valence only after the CONCEPT and CONCEPT TYPE are determined
Valence must be evaluated within the evaluative frame defined by the CONCEPT TYPE
Do not infer sentiment, desirability, or respondent intent
If the idea describes a fact, condition, or attribute without directional effect → assign 0
If an idea contains multiple directional effects → it must have been split earlier into separate ideas


---

## PRIORITY RULES (APPLY IN ORDER)

{priority_rules}

---

## EDGE CASES
- Empty, irrelevant, or nonsensical response → return []
- One idea → array with one object
- Multiple ideas → array with multiple objects

---

## OUTPUT REQUIREMENTS (STRICT)
- Valid JSON only
- Must match the required response schema
- No explanations or extra text
- All fields must be in {language}

Begin processing now and provide your output as valid JSON matching the required response schema.
"""


class SemanticTaxonomyResponse(BaseModel):
    """Abstraction ladder for extracted ideas.
    4-layer classification:
    instance (what they said), concept (what they mean),
    concept_type (thematic domain), concept_type_definition (contextual framing)."""

    instance: str = ""
    concept: str = ""
    concept_type: str = ""
    concept_type_definition: str = ""

    @field_validator('instance', 'concept', 'concept_type_definition', mode='before')
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
    abstraction_ladder: Optional[SemanticTaxonomyResponse] = Field(
        default=None,
        description="Abstraction ladder: instance (verbatim) -> concept (interpretive meaning) -> concept_type (thematic domain) -> concept_type_definition (contextual framing)"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="directional valence of idea relative to the CONCEPT: the idea strengthens, increases, or reinforces the concept (+), the idea weakens, decreases, or undermines the concept (-),  the idea is non-directional with respect to the concept (0)"
    )

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

    # Resolve dimension type for concept description enrichment
    dim_is_alias, dim_short, dim_desc = resolve_slot_type(facet.dimension_slot.type_name)
    concept_desc = prompt_rules.concept_instruction
    if dim_is_alias:
        concept_desc += f" Must be a {facet.dimension_slot.type_name}."

    # Build concept_type field (thematic domain)
    if concept_types:
        allowed_keys = tuple(c.key for c in concept_types) + ("Other",)
        concept_type_field = (
            Literal[allowed_keys],
            Field(
                description=(
                    "Thematic domain — which aspect of the entity does this concept belong to? One of: " +
                    ", ".join(f"{c.key} ({c.definition})" for c in concept_types) +
                    ", Other (does not fit any of the above)"
                ),
                examples=[c.key for c in concept_types[:3]]
            )
        )
    else:
        concept_type_field = (
            str,
            Field(
                description=(
                    f"Thematic domain: which ASPECT of the entity does this concept belong to? "
                    f"Use a short label (1-4 words) suitable for organizing a codebook section. "
                    f"NOT a linguistic role ('moral attribute', 'functional trait') but a thematic category "
                    f"('products and services', 'marketing and communication', 'social responsibility')."
                ),
            )
        )

    # Create base facet-specific SemanticTaxonomyResponse
    _BaseFacetTaxonomy = create_model(
        f"SemTax_{facet_key}_b",
        __base__=SemanticTaxonomyResponse,
        instance=(str, Field(
            description=prompt_rules.instance_instruction,
        )),
        concept=(str, Field(
            description=concept_desc + " Name what the respondent is REALLY talking about in context, not a spelling fix or nominalization.",
        )),
        concept_type=concept_type_field,
        concept_type_definition=(str, Field(
            description=(
                "One short phrase (2-5 words) explaining what this thematic domain REPRESENTS "
                "for the entity in this survey context. Frames WHY this domain matters. "
                "NOT a paraphrase or synonym of concept_type. "
                "Example: concept_type='operations and planning' → definition='core service promise'."
            ),
        )),
    )

    # Add fuzzy-match validator for concept_type (runs before Literal validation)
    # Captures allowed_keys from enclosing scope via closure
    _key_map = {k.lower(): k for k in allowed_keys} if concept_types else {}
    # Also map underscore variants so "products_and_services" → "products and services"
    if concept_types:
        _key_map.update({k.lower().replace(' ', '_'): k for k in allowed_keys})

    class FacetTaxonomy(_BaseFacetTaxonomy):
        @field_validator('concept_type', mode='before')
        @classmethod
        def normalize_concept_type(cls, v: object) -> str:
            if not isinstance(v, str) or not _key_map:
                return v
            stripped = v.strip()
            # Exact match (case-insensitive)
            if stripped.lower() in _key_map:
                return _key_map[stripped.lower()]
            # Normalize: _ → space, & → and, collapse whitespace, strip trailing punctuation
            normalized = stripped.lower().replace('_', ' ').replace('&', 'and').replace('  ', ' ').rstrip('.,;:')
            if normalized in _key_map:
                return _key_map[normalized]
            # No match — pass through unchanged, Literal validator will reject
            return stripped

    FacetTaxonomy.__name__ = f"SemTax_{facet_key}"
    FacetTaxonomy.__qualname__ = f"SemTax_{facet_key}"

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
        abstraction_ladder: Optional[FacetTaxonomy] = Field(
            default=None,
            description="Abstraction ladder: instance -> concept -> concept_type -> concept_type_definition"
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

    FacetExtractionModel.__name__ = f"IdeaExtr_{facet_key}"
    FacetExtractionModel.__qualname__ = f"IdeaExtr_{facet_key}"
    return FacetExtractionModel
