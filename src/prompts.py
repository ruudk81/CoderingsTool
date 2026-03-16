"""
Prompts module - Contains all LLM prompt templates for the pipeline.
"""

from __future__ import annotations
from typing import Any, ClassVar, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, create_model

from facet_data import FacetDefinition, PromptRules, resolve_slot_type, get_facets_in_decision_order


# =============================================================================
# STEP 1: SPELL CHECKING
# =============================================================================

SPELLCHECK_INSTRUCTIONS = """
You are a {language} language expert specializing in correcting misspelled words in open-ended survey responses.
Your task is to process correction tasks for responses that contain placeholder tokens indicating spelling mistakes.

First, here is the survey question that the responses are answering:
<survey_question>
{var_lab}
</survey_question>

For each correction task, you will receive:
- A sentence with one or more <oov_word> placeholders
- A list of misspelled words, in the same order as the placeholders
- A list of suggested corrections, in the same order

Follow these rules when making corrections:
1. Replace each <oov_word> placeholder with the best possible correction of the corresponding misspelled word.
2. Consider the meaning and context of the survey question when choosing corrections.
3. If a better correction exists than the ones provided, use that instead.
4. You may split a misspelled word into two words only if the split preserves the intended meaning and fits grammatically.
5. If no suitable correction is possible, use "[NO RESPONSE]" as the corrected sentence for that task.

Here are the correction tasks to process:
<correction_tasks>
{tasks}
</correction_tasks>

Additional guidelines:
- Pay close attention to the context and meaning of each response when making corrections.
- Ensure that your corrections maintain the original intent of the respondent.
- If a suggested correction doesn't fit the context, consider alternative corrections that preserve the meaning.

Begin processing the correction tasks now and provide your output as valid JSON following the response schema provided.
"""


class CorrectionItem(BaseModel):
    """A single spell correction result."""
    respondent_id: Any = Field(
        description="The respondent ID from the correction task"
    )
    corrected_response: str = Field(
        description="The fully corrected response with all spelling mistakes fixed"
    )


class LLMCorrectionResponse(BaseModel):
    """Structured output for spell check corrections."""
    corrections: List[CorrectionItem] = Field(
        description="List of corrections, one for each task in the input"
    )

# =============================================================================
# STEP 2: QUALITY FILTERING 
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses.
Your task is to determine whether each response provides **usable, on-topic content in relation to the specific survey question**, and assign appropriate quality filter codes.

You will classify each response into one of three practical outcomes:

==================================================
OUTCOME A — Don't Know / Uncertainty
CODE: 99999997 | quality_filter = true
==================================================

Use this code if the respondent explicitly expresses uncertainty, lack of knowledge, or non-applicability.

This includes any response whose clear meaning in {language} is equivalent to:
- "I don't know"
- "N/A"
- "Not applicable"
- "No idea"
- "Unsure"
- "Can't say"
- "?"

If a response fits this pattern → set:
quality_filter = true
quality_filter_code = 99999997

==================================================
OUTCOME B — Nonsensical/Gibberish OR completely Off-topic
CODE: 99999999 | quality_filter = true
==================================================

Use this code for **two different kinds of unusable responses**:

A) Pure gibberish / nonsensical
   Examples:
   - Random characters: "asdfkj", "jjjjj", "x!@#%"
   - Placeholder text: "lorem ipsum", "test test"
   - Verbatim repetition of the question with no added content
   - Completely unintelligible text

B) Intelligible but completely off-topic / totally irrelevant
   The response is understandable {language} , BUT:
   - It does NOT address the actual survey question ({var_lab}) even remotely, OR
   - It obviously avoids the question.

Illustractive examples in English (if the question is about public transport):
- "Nothing"
- "I love dogs."
- "The weather is nice today."
- "Pizza is better than pasta."
- "I work in finance."
- A personal story that has nothing to do with transportation.

These are NOT "I don't know" — they are simply irrelevant to the question.

If a response fits **either A or B** → set:
quality_filter = true
quality_filter_code = 99999999


==================================================
OUTCOME C — Meaningful / On-topic Response
quality_filter = false | quality_filter_code = null
==================================================

A response is meaningful if:
- It is understandable in {language}, AND
- It engages with or relates to the survey question ({var_lab}), even if:
  - It is very short
  - It is vague
  - It is opinionated
  - It is critical
  - It is poorly written
  - It is partially incomplete

Examples (if the question is about public transport):
- "Buses are always late."
- "Too crowded."
- "Tickets are expensive."
- "The metro is unreliable."

For this category → set:
quality_filter = false
quality_filter_code = null

==================================================
SURVEY QUESTION
<survey_question>
{var_lab}
</survey_question>

RESPONSES TO EVALUATE
<responses>
{responses}
</responses>

==================================================
DECISION RULE (FOLLOW EXACTLY)

For each response, apply these steps in order:

1. Does the response explicitly express uncertainty or "I don't know"?
   - If YES → quality_filter = true, quality_filter_code = 99999997
   - If NO → go to Step 2

2. Ask:
   "Does this response provide usable content that addresses the survey question, even remotely?"

   - If NO (because it is gibberish OR off-topic) →
     quality_filter = true, quality_filter_code = 99999999

   - If YES →
     quality_filter = false, quality_filter_code = null
"""


QualityCode = Optional[Literal[99999997, 99999999]]

class QualityFilterLLMResponse(BaseModel):
    """A single quality filter assessment result."""

    respondent_id: Any = Field(
        description="The respondent's ID from the input (preserve exact type and format)"
    )

    response: Union[str, float, int, None] = Field(
        description="The exact response text being evaluated"
    )

    quality_filter: bool = Field(
        description=(
            "true if the response is unusable (don't know OR gibberish/off-topic), "
            "false if the response is meaningful and addresses the question"
        ),
        examples=[True, False],
    )

    quality_filter_code: QualityCode = Field(
        default=None,
        description=(
            "99999997 = uncertainty / don't know; "
            "99999999 = gibberish OR completely off-topic; "
            "null = meaningful response"
        ),
        examples=[99999997, 99999999, None],
    )

    @model_validator(mode="after")
    def check_consistency(self):
        if self.quality_filter and self.quality_filter_code is None:
            raise ValueError(
                "If quality_filter=true, quality_filter_code must be 99999997 or 99999999"
            )
        if not self.quality_filter and self.quality_filter_code is not None:
            raise ValueError(
                "If quality_filter=false, quality_filter_code must be null"
            )

        return self


# =============================================================================
# STEP 3: IDEA EXTRACTION (v5 — 10 MECE facets, builder functions)
# =============================================================================
#
# Prompts are builder functions returning complete prompt strings.
# Response models use Field(description=...) + validators.
# Facet data comes from facet_data.py (frozen dataclasses).
#

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



# ============================================================================
# STEP 5: MECE CATEGORY DISCOVERY & ASSIGNMENT
# =============================================================================
#
# Three-step MAP/REDUCE/MECE pipeline operating on descriptive codes
# within each concept_type partition.
#
# Terminology:
#   - partition = concept_type group (data-driven from step 3)
#   - theme     = overarching pattern of shared meaning discovered by MAP/REDUCE
#   - category  = operationalized theme with MECE boundaries (from MECE step)
#   - label     = the text string being analyzed (default: concept_type_definition field)


# --- Partition models (data-driven concept_type groups) ---

class DomainDescription(BaseModel):
    """Description of a concept_type partition."""
    partition_name: str = Field(
        ...,
        description="Concept type name (data-driven, e.g., 'recommendation', 'product_feature')"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of statements belong to this partition. "
            "Uses observable criteria."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a coder asks to determine if a statement "
            "belongs to this partition."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description="3-5 concrete words or phrases that indicate this partition"
    )


class DomainSet(BaseModel):
    """Complete set of concept_type partitions."""
    partitions: List[DomainDescription] = Field(
        ...,
        description="List of populated concept_type partitions"
    )


# --- MAP: Candidate theme extraction per batch ---

MAP_CATEGORIES_PROMPT = """You are a codebook designer specialized in thematic analysis as professed by Braun and Clarcke (2006)
Your goal is identify themes within a cluster of descriptive codes derived from responses to a survey question that share semantic meaning.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<cluster_context>
Current cluster:
{partition_name}

Cluster description:
{partition_inclusion}

CRITICAL: Only identify categories that fall WITHIN "{partition_name}".
</cluster_context>

<cluster_with_descriptive_codes>
{labels_list}
</cluster_with_descriptive_codes>

<task>
Identify the overarching theme or themes that capture patterns of shared meaning across the cluster with descriptive codes
</task>

<instruction>
- Work bottom-up from the existing codes and categories.
- Do not impose external theories, frameworks, or pre-defined analytic concepts.
- Do not simply restate or rename the existing categories.
- Identify the minimum number of themes needed to account for the patterned meanings in the data.
- If the data supports more than one overarching theme, explicitly justify why multiple themes are needed.

- For each theme:
    • Describe the central organizing idea that gives the theme coherence
    • Explain how the theme integrates multiple codes or categories
    • Provide a clear analytic theme label
    • Provide a more descriptive or stakeholder-friendly label, if different
    • Make explicit how the themes relate to one another (e.g., hierarchy, tension, conditionality, complementarity).

- Output goal: Produce a small set of inductively derived themes that represent meaningful patterns in the dataset and can be clearly communicated to non-academic audiences.
</instruction>

Provide output as valid JSON following the response schema.
All output (labels, descriptions, rationales) MUST be in {language}.
"""


class CandidateTheme(BaseModel):
    """A candidate theme identified in a batch of descriptive codes."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this theme"
    )
    central_organizing_idea: str = Field(
        ...,
        description=(
            "The core idea that gives this theme coherence — "
            "what pattern of shared meaning unites the codes"
        )
    )
    description: str = Field(
        ...,
        description="How this theme integrates multiple codes: what they share, why they cluster"
    )
    example_codes: List[str] = Field(
        ...,
        description="2-4 descriptive codes from the batch that exemplify this theme (exact quotes)"
    )
    rationale: str = Field(
        ...,
        description="Why these codes belong together under this theme"
    )


class MapBatchThemes(BaseModel):
    """Candidate themes identified in a single batch of descriptive codes."""
    themes: List[CandidateTheme] = Field(
        ...,
        description="Candidate themes identified in this batch (typically 3-8)"
    )


# --- REDUCE: Cross-batch thematic synthesis ---

REDUCE_THEMES_PROMPT = """You are a thematic analyst synthesizing candidate themes discovered across multiple batches of descriptive codes from survey responses, following the approach of Braun and Clarke (2006).

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<partition_context>
You are analyzing descriptive codes within the concept type: "{partition_name}"
Partition scope: {partition_inclusion}
Partition boundary test: {partition_boundary_test}

Peer partitions (do NOT identify themes that belong to these):
{peer_partitions_list}

CRITICAL: Only retain themes that fall WITHIN "{partition_name}".
</partition_context>

{cluster_hints}

<instruction>
You analyzed {n_batches} batches of descriptive codes from the partition "{partition_name}" and identified {n_total_themes} candidate themes across all batches. Because batches were processed independently, many themes overlap or capture different facets of the same underlying pattern of meaning. Your task is to synthesize them into a coherent set of overarching themes.
{grouping_instruction}
</instruction>

<principles>
1. INTEGRATE candidate themes that capture the same underlying pattern of shared meaning,
   even if they were labelled differently across batches.
2. For each synthesized theme, articulate its CENTRAL ORGANIZING IDEA: the core pattern
   of meaning that gives the theme coherence and explains why the codes cluster together.
3. AIM for the minimum number of themes needed to account for the patterned meanings
   in this partition. Fewer well-defined themes are better than many fragmented ones.
4. When integrating, pick the BEST theme label from the input candidates
   (do NOT invent a new label unless none of the existing labels are adequate).
5. Every input candidate theme must appear in exactly one synthesized theme's
   integrated_from list. Do not drop any.
6. Preserve the most informative central organizing idea and description when integrating.
7. DISCARD any candidate themes that actually belong to peer partitions, not to "{partition_name}".
</principles>

<candidate_themes_from_batches>
{batch_categories_list}
</candidate_themes_from_batches>

<task>
1. Read all {n_total_themes} candidate themes from {n_batches} batches.
2. Identify themes that capture the same or overlapping patterns of shared meaning.
3. Synthesize them into a coherent set of 3-12 overarching themes for this partition.
   If you have significantly more, you are likely preserving unnecessary distinctions.
4. For each synthesized theme, articulate its central organizing idea — the core pattern
   that unites the underlying codes.
5. CRITICAL: every input candidate theme label must appear in exactly one synthesized
   theme's integrated_from list. Do not drop any.

All output (labels, descriptions, central organizing ideas) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</task>
"""


class SynthesizedTheme(BaseModel):
    """A theme synthesized from candidate themes across MAP batches."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this theme"
    )
    central_organizing_idea: str = Field(
        ...,
        description=(
            "The core pattern of shared meaning that gives this theme coherence — "
            "why the underlying codes cluster together"
        )
    )
    description: str = Field(
        ...,
        description="How this theme integrates the underlying codes — what they share and why they cluster"
    )
    integrated_from: List[str] = Field(
        ...,
        description=(
            "All candidate theme labels from the MAP batches that were integrated "
            "into this theme. If standalone, list contains only the original label."
        )
    )


class SynthesizedThemeList(BaseModel):
    """Synthesized theme list from the reduce step."""
    themes: List[SynthesizedTheme] = Field(
        ...,
        description="Overarching themes after cross-batch synthesis (typically 3-12)"
    )


# --- MECE: Apply boundaries with self-verification ---

MECE_BOUNDARIES_PROMPT = """You are a codebook designer applying MECE (Mutually Exclusive, Collectively Exhaustive) constraints to a list of coding categories within a single partition of survey response labels.

Your output must be a DECISION SYSTEM — not a semantic description. Each category must be defined by criteria a human coder can independently apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<partition_context>
You are analyzing labels within the concept type: "{partition_name}"
Partition scope: {partition_inclusion}
Partition boundary test: {partition_boundary_test}

Peer partitions (do NOT identify categories that belong to these):
{peer_partitions_list}

CRITICAL: Only retain categories that fall WITHIN "{partition_name}".
</partition_context>

{cluster_hints}

<instruction>
You are given {n_categories} consolidated coding categories from the partition "{partition_name}" containing {n_labels} unique category labels. Your task is to ensure this list is MECE:
- Mutually Exclusive: each label should clearly belong to exactly one category
- Collectively Exhaustive: every label in this partition should fit one category

You may merge categories that still overlap, split categories that are too broad, or adjust boundaries.
{grouping_instruction}
CRITICAL DESIGN RULE: Define each category using POSITIVE, INDEPENDENT criteria.
- DO NOT define a category by what it is NOT or by referencing other categories.
- Each category's boundary_test must work WITHOUT knowing what other categories exist.
- Use observable characteristics of the label, not abstract semantic descriptions.
</instruction>

<principles>
1. The result must be a clean, non-overlapping set of categories WITHIN "{partition_name}".
2. Each category needs:
   - A BOUNDARY TEST: a yes/no question a coder can ask independently
   - DIAGNOSTIC SIGNALS: concrete words/phrases that trigger assignment
   - TIEBREAKER RULES: for each neighboring category, a concrete rule for ambiguous cases
3. Categories are VALENCE-NEUTRAL: name what people talk about, not how they feel.
4. Actively merge related categories into broader containers.
   Fewer well-defined categories are far better than many overlapping ones.
5. AIM for 3-10 final categories per partition. If you have significantly more,
   merge categories that a coder would confuse.
6. Every category from the input must be accounted for in the output (merged, kept, or split).
7. If a category actually belongs to a peer partition, exclude it from the final set.
</principles>

<label_constraints>
Category labels MUST:
- Be in {language}
Category labels MUST NOT:
- contain "en/and/und/et",
- contain slashes,
- stack multiple adjectives,
- list multiple attributes.
Each label must express ONE core subject only.

All output (labels, definitions, key_expressions) MUST be in {language}.
</label_constraints>

<consolidated_categories>
Partition: {partition_name}
Total unique labels in partition: {n_labels}

{categories_list}
</consolidated_categories>

<task>
1. Review the {n_categories} categories and their descriptions.
2. Check for overlaps: merge categories where a coder would hesitate between them.
3. Check for gaps: is there any common label type within "{partition_name}" not covered?
4. For each final category, define:
   - category_label: short name
   - inclusion_definition: what labels belong (observable criteria)
   - boundary_test: a yes/no question a coder asks to determine membership
     (MUST be self-contained — no references to other categories)
   - diagnostic_signals: 3-5 concrete words/phrases/framings that trigger assignment
   - key_expressions: 3-5 representative labels from the partition
   - tiebreaker_rules: for each similar/adjacent category, a rule:
     "If ambiguous with [category X], assign here when [observable condition]"
5. VERIFY your categories are MECE by testing each pair of adjacent/similar categories:
   - Construct one AMBIGUOUS example (a label that could plausibly fit either category)
   - Show which category gets it and WHY, using only your boundary_test and diagnostic_signals
   - If you cannot decide using your criteria alone, your categories are NOT MECE — merge or redefine them before finalizing.

Provide output as valid JSON following the response schema provided.
</task>
"""


class MECECode(BaseModel):
    """A MECE category with independent boundary criteria."""
    category_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this category"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of labels belong to this category. "
            "Must use observable criteria, not vague semantic descriptions."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a human coder asks to determine if a label belongs here. "
            "Must be self-contained — no references to other categories."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description=(
            "3-5 concrete words, phrases, or framings that, if present, "
            "indicate this category"
        )
    )
    key_expressions: List[str] = Field(
        ...,
        description="3-5 representative labels from the partition that exemplify this category"
    )
    tiebreaker_rules: List[str] = Field(
        ...,
        description=(
            "For each similar/adjacent category, a rule: "
            "'If ambiguous with [category X], assign here when [observable condition]'"
        )
    )


class MECEVerification(BaseModel):
    """Self-verification test for one pair of adjacent MECE categories."""
    category_a: str = Field(
        ...,
        description="First category in the pair"
    )
    category_b: str = Field(
        ...,
        description="Second category in the pair"
    )
    ambiguous_example: str = Field(
        ...,
        description="A constructed label that could plausibly fit either category"
    )
    assigned_to: str = Field(
        ...,
        description="Which category the ambiguous example is assigned to"
    )
    reasoning: str = Field(
        ...,
        description="Why this assignment is correct, using only boundary_test and diagnostic_signals"
    )


class MECECodeSet(BaseModel):
    """Complete MECE category set for a single partition, with self-verification."""
    categories: List[MECECode] = Field(
        ...,
        description="MECE categories for this partition"
    )
    mece_verifications: List[MECEVerification] = Field(
        ...,
        description=(
            "Self-verification tests: one per pair of adjacent/similar categories. "
            "Proves the boundary criteria actually work."
        )
    )


# --- Category Assignment ---

CATEGORY_ASSIGNMENT_PROMPT = """You are a coding assistant assigning survey response ideas to pre-defined MECE coding categories.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<partition_context>
Concept type: "{partition_name}"
Partition scope: {partition_inclusion}
</partition_context>

<categories>
You MUST assign each idea to exactly ONE of these categories:

{categories_block}
</categories>

<ideas_to_assign>
{ideas_block}
</ideas_to_assign>

<instructions>
For each idea:
1. Read the idea text, concept, and concept_type_definition.
2. Apply each category's boundary_test to determine which category fits.
3. Use diagnostic_signals to confirm your choice.
4. If ambiguous between two categories, apply the tiebreaker_rules.
5. If NO category's boundary_test passes and no diagnostic_signals match,
   assign "{other_category_label}" — do NOT force-fit into an ill-matching category.
6. Assign exactly ONE category per idea — use the exact category_label from the list above.
7. Rate your confidence (0.0 to 1.0):
   - 0.90-1.00: boundary_test clearly matches, diagnostic_signals confirm
   - 0.70-0.89: boundary_test matches, signals partially confirm
   - 0.50-0.69: plausible fit, decided by tiebreaker
   - below 0.50: weak fit — strongly consider "{other_category_label}" instead
8. Provide a brief rationale referencing the boundary_test or diagnostic_signals that decided it.

All output (rationale) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</instructions>
"""


class CodeAssignment(BaseModel):
    """Single idea-to-category assignment."""
    idea_id: str = Field(
        ...,
        description="The idea_id from the input"
    )
    assigned_category: str = Field(
        ...,
        description="The exact category_label of the assigned MECE category"
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assignment (0.0 to 1.0)"
    )
    rationale: str = Field(
        ...,
        description="Brief rationale referencing boundary_test or diagnostic_signals"
    )


class CodeAssignmentBatch(BaseModel):
    """Batch of category assignments for multiple ideas."""
    assignments: List[CodeAssignment] = Field(
        ...,
        description="One assignment per idea in the input batch"
    )


# =============================================================================
#  Speculative codes
# =============================================================================

INITIAL_CODEBOOK_CREATION_PROMPT = """
You are an {language} expert qualitative data analyst specializing in rigorous thematic analysis and code creation. 
Your task is to generate hypothetical codes that might be encountered when analyzing written answers to a specific survey question.

Here are the critical coding principles you must follow:
- ATOMIC: Each code must capture ONE concept only - no compound ideas with "and", "including", "with"
- PRECISE: Clear boundaries that enable reliable coding decisions
- CONCISE: Code names must be 2-5 words maximum
- OPERATIONAL: Definitions must use observable criteria, not interpretations
- MUTUALLY EXCLUSIVE: Minimal overlap between codes

You will be working with the following inputs:
- Language to use: <language> {language} </language>
- Number of codes to generate: <n_codes> {n_codes} </n_codes>
- Survey question to analyze: <survey_question> {survey_question} </survey_question>

Your task is to generate {n_codes} diverse, hypothetical codes that might emerge from analyzing responses to the given survey question. Create codes that could apply to ANY survey topic. Do not assume the survey is about education, healthcare, or any specific domain. Let the survey question guide your code generation.

Consider different code types when generating your codes:
- Attribute codes: Qualities or characteristics mentioned
- Process codes: Actions, procedures, or methods described
- Relational codes: Interactions or connections between elements
- State codes: Conditions, situations, or circumstances
- Evaluative codes: Assessments, judgments, or opinions expressed

Provide your response in {language} as a JSON array of objects, where each object has "code" and "definition" fields. 
Here's an example of the structure to follow (using generic placeholders):
<example>
[
  {{"code": "Quality assessment", "definition": "References to evaluating the quality/characteristic of topic-specific element."}},
  {{"code": "Process difficulties", "definition": "Mentions of challenges in topic-specific process."}},
  {{"code": "Actor perspectives", "definition": "Expessions of viewpoints of relevant actors/participants."}}
]
</example>

Examples of well-structured code definitions:
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these weak definitions:
- Compound: "References to [issue A] including [aspect 1], [aspect 2], and [aspect 3]"
- Vague: "Mentions of various [things] related to [topic]"
- Interpretive: "Underlying [abstract concept] manifesting in different ways"


Return ONLY the JSON array in {language}. Do not include any additional text or explanations outside of the JSON array.
"""

# =============================================================================
# STEP 6: CODEBOOK GENERATION - 4 PROMPT CHAIN
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CLUSTER_SUMMARY_PROMPT  (unified -- works for all Stage 1 input routes)
#
# Dynamic placeholders filled by the caller:
#   {data_unit}        -- "cluster" or "category"
#   {evidence_source}  -- "the provided key expressions" or "the assigned ideas"
#   {data_description} -- paragraph(s) explaining the data block structure
#   {analysis_step_2}  -- route-specific reasoning instruction
#   {analysis_step_3}  -- route-specific reasoning instruction
# -----------------------------------------------------------------------------

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher identifying meaning-based codes from survey response data for a codebook. Your work is interpretive but strictly data-bound: you may organize and name patterns, but you must not introduce concepts absent from the provided data.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES a set of responses -- the shared interpretive thread that explains WHY these expressions belong together. If a candidate can be expressed as a single everyday word (e.g., "price"), it is likely a topic, not a theme. A well-formed code names the specific meaning pattern respondents express.

## Research Context

<survey_context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Your codes must capture these expressed meanings, not just the surface topics mentioned.

- Language: {language}
- Domain: {domain}
- Topic: {topic}
- Perspective: {perspective}
- Intent: {intent}
- Entity: {entity}
</survey_context>

## Taxonomy

<taxonomy_rules>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
</taxonomy_rules>

## {data_unit} Data to Analyze

<{data_unit}_data>
<{data_unit}_id>
{cluster_id}
</{data_unit}_id>

<{data_unit}_content>
{cluster_text}
</{data_unit}_content>
</{data_unit}_data>

{data_description}

## Analysis Instructions

First, identify 1-2 Central Organizing Concepts (COCs) -- analytic statements that capture what ties this {data_unit}'s data together. COCs describe shared patterns across {evidence_source}. They are NOT code labels.

Then derive atomic themes from the data. Each theme must satisfy ALL criteria:

THEME CRITERIA
1. Atomic: one semantic nucleus, unsplittable into independently meaningful parts.
2. Within-facet: falls within the {facet_name} scope.
3. Grounded: directly supported by {evidence_source}.
4. Operational: precise enough to serve as a codebook entry.
5. Meaning-based: captures a shared meaning pattern, not just a topic.
6. Metadata-coherent: consistent with any structured metadata provided (inclusion definitions, boundary tests).

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of travel delays", "Perception of vehicle cleanliness", "Feeling of personal safety"
- RIGHT: "Travel delays", "Vehicle cleanliness", "Personal safety"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

ASSIGNMENT EXAMPLES
For each theme provide:
- inclusion (2-3): Observable cues starting with verbs ("Describes...", "Mentions..."), traceable to {evidence_source}.
- exclusion (1-2): Boundary cases that should NOT be coded here.
- near_neighbor: Closest confusable theme label + one sentence distinguishing them. Use "Unknown" if none.

## Required Analysis

Document your reasoning in the analysis field:
1. State 1-2 {data_unit}-level COCs
2. {analysis_step_2}
3. {analysis_step_3}
4. Justify single vs multiple themes

Write all output in {language}. Provide your output as valid JSON following the response schema provided.
"""


class NearNeighbor(BaseModel):
    label: str = Field(
        ...,
        min_length=1,
        description="Label of closest potentially-confusable theme, or 'Unknown' if none exists",
        examples=["Product Quality", "Unknown"]
    )
    tell_apart_rule: str = Field(
        default="",
        min_length=0,
        description="One sentence distinguishing this theme from the neighbor",
        examples=["This theme focuses on speed, not accuracy"]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AssignmentExamples(BaseModel):
    inclusion: List[str] = Field(
        ...,
        min_length=1,
        description="2-3 observable cues starting with verbs for what to include",
        examples=[["Mentions waiting time explicitly", "Describes delay in service"]]
    )
    exclusion: List[str] = Field(
        ...,
        min_length=1,
        description="1-2 boundary cases for what must NOT be included",
        examples=[["General complaints without time reference", "Mentions speed positively"]]
    )
    near_neighbor: NearNeighbor = Field(
        ...,
        description="Closest confusable theme and how to distinguish",
        examples=[{"label": "Product Quality", "tell_apart_rule": "This theme focuses on speed, not accuracy"}]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusterThemeItem(BaseModel):
    theme_id: int = Field(
        ...,
        description="Sequential theme identifier starting at 1",
        examples=[1, 2]
    )
    theme_label: str = Field(
        ...,
        max_length=60,
        description="1-5 word meaning-based noun phrase theme label",
        examples=["Waiting Time", "Product Quality", "Staff Friendliness"]
    )
    theme_clarification: str = Field(
        ...,
        max_length=250,
        description="<=30-word grounded definition describing what belongs in this theme",
        examples=["Responses mentioning the duration of waiting for service or products"]
    )
    abstraction_level: Literal["concrete-experiential", "interpretive-pattern"] = Field(
        ...,
        description="'concrete-experiential' = directly observable in responses; 'interpretive-pattern' = inferred shared meaning pattern",
        examples=["concrete-experiential", "interpretive-pattern"]
    )
    assignment_examples: AssignmentExamples = Field(
        ...,
        description="Concrete inclusion/exclusion examples for coding",
        examples=[{
            "inclusion": ["Mentions waiting time explicitly", "Describes delay in service"],
            "exclusion": ["General complaints without time reference"],
            "near_neighbor": {"label": "Service Quality", "tell_apart_rule": "This theme focuses on duration, not quality"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('theme_label')
    @classmethod
    def validate_label_length(cls, v):
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(f"theme_label must be <=10 words, got {word_count}")
        return v

    @field_validator('theme_clarification')
    @classmethod
    def validate_clarification_length(cls, v):
        word_count = len(v.split())
        if word_count > 36:
            raise ValueError(f"theme_clarification must be <=36 words, got {word_count}")
        return v


class ClusterSummaryOutput(BaseModel):
    cluster_id: str = Field(
        ...,
        description="The cluster identifier exactly as provided",
        examples=["3", "5", "12"]
    )
    analysis: str = Field(
        ...,
        description="1-3 sentence analysis: COCs identified, splits/merges made, rationale for single vs multiple themes",
        examples=["Identified 2 COCs: 'speed' and 'accuracy'. Retained both as distinct atomic concepts."]
    )
    extracted_themes: List[ClusterThemeItem] = Field(
        ...,
        description="Final theme entries for valid atomic themes"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -----------------------------------------------------------------------------
# 2. CODING_DECISION_PROMPT
# -----------------------------------------------------------------------------

CODING_DECISION_PROMPT = """
You are a qualitative research assistant maintaining a parsimonious codebook following Braun & Clarke (2006) methodology. Your task: classify a newly identified theme within the {facet_name} facet and decide whether to USE, MODIFY, or CREATE a code.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES responses -- the shared interpretive thread that explains WHY expressions belong together. The codebook must remain MECE (Mutually Exclusive, Collectively Exhaustive).

---

PARAMETERS

<context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Language: {language}
Domain: {domain} | Entity: {entity}
</context>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

<new_theme>
- name: "{theme_name}"
- description: "{theme_description}"
- included expressions:
    {inclusion}
</new_theme>

<existing_codes>
{code_text}
</existing_codes>

---

DECISION OPTIONS

- **USE** -- Existing code fully captures the new theme's meaning; use as-is.
- **MODIFY_HORIZONTAL** -- Existing code needs broader scope at the same abstraction level.
- **MODIFY_VERTICAL** -- Same conceptual family but different abstraction level; create/reference a parent code.
- **CREATE** -- Theme represents a distinct concept not covered by existing codes.

If the codebook is empty, CREATE is the only valid decision.

---

ANALYSIS FRAMEWORK

1. **MATCH**: Identify best-matching existing code(s) by core meaning and practical relevance.
2. **FAMILY TEST**: Do the theme and best match share the same conceptual family (same core concept, same practical relevance given the research question)?
3. **LEVEL TEST**: Are they at the same abstraction level on the {facet_name} axis?
4. **DECIDE**:
   - Fully covered in meaning and scope -> USE
   - Same family + same level + not fully covered -> MODIFY_HORIZONTAL
   - Same family + different level -> MODIFY_VERTICAL
   - Different family -> CREATE
   - Multi-concept theme where MODIFY would replace core meaning -> CREATE

---

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of travel delays", "Perception of vehicle cleanliness", "Feeling of personal safety"
- RIGHT: "Travel delays", "Vehicle cleanliness", "Personal safety"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

---

OUTPUT INSTRUCTIONS

Reason through the analysis framework before producing your answer. In your justification, reference: (a) matched code(s), (b) family test result, (c) level test result. Reference cosine similarity scores if provided.

Write field names in English, values in {language}. Provide your output as valid JSON following the response schema provided.
"""


class MatchedCandidate(BaseModel):
    code: str = Field(
        ...,
        description="Exact candidate code name from existing codebook",
        examples=["Product Quality", "Waiting Time"]
    )
    definition: Optional[str] = Field(
        default=None,
        description="Code definition in light of the survey question",
        examples=["References to product quality concerns", "Mentions of waiting duration"]
    )
    definition_source: Literal["provided", "inferred"] = Field(
        default="inferred",
        description="Whether definition was provided in codebook or inferred"
    )
    assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Assignment examples from existing code",
        examples=[{
            "inclusion": ["References to product quality", "Mentions satisfaction with quality"],
            "exclusion": ["Price complaints without quality mention"],
            "near_neighbor": {"label": "Product Value", "tell_apart_rule": "Quality focuses on product attributes, value on price-quality ratio"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# NOTE on modify_instruction values: Due to legacy naming, the Literal values
# are confusingly inverted relative to the decision names:
#   "vertical_broaden_same_level"       -> used for MODIFY_HORIZONTAL (broaden at same level)
#   "hierarchical_parent_diff_level"    -> used for MODIFY_VERTICAL (create parent)
# These values are checked in codeGenerator_exp.py lines ~5744-5748.
# Do NOT rename without updating the caller.
class ModifyParameters(BaseModel):
    modify_instruction: Literal["vertical_broaden_same_level", "hierarchical_parent_diff_level", "none"] = Field(
        ...,
        description="Type of modification: 'vertical_broaden_same_level' = broaden at same level (HORIZONTAL), 'hierarchical_parent_diff_level' = create parent (VERTICAL), 'none' = no modification",
        examples=["vertical_broaden_same_level", "hierarchical_parent_diff_level", "none"]
    )
    conceptual_family: Literal["same", "different", "none"] = Field(
        ...,
        description="Whether new theme belongs to same conceptual family as matched code",
        examples=["same", "different"]
    )
    abstraction_level: Literal["same", "different", "none"] = Field(
        ...,
        description="Whether theme is at same abstraction level as matched code",
        examples=["same", "different"]
    )
    abstraction_level_action: Literal["keep", "broaden_to_parent", "none"] = Field(
        ...,
        description="Action to take regarding abstraction level",
        examples=["keep", "broaden_to_parent", "none"]
    )
    inclusion_update: Optional[str] = Field(
        default=None,
        description="Concrete additions to inclusion rules if modifying",
        examples=["Add expressions about delivery speed", None]
    )
    exclusion_update: Optional[str] = Field(
        default=None,
        description="Concrete boundary clarifications for exclusion rules",
        examples=["Exclude mentions of product defects", None]
    )
    parent_theme_label: Optional[str] = Field(
        default=None,
        description="Suggested parent label for vertical modification",
        examples=["Service Experience", None]
    )
    near_neighbor_label_update: Optional[str] = Field(
        default=None,
        description="Updated neighbor label if boundaries changed",
        examples=["Response Time", None]
    )
    tell_apart_rule_update: Optional[str] = Field(
        default=None,
        description="Updated tell-apart rule if distinction changed",
        examples=["This theme focuses on duration, neighbor on frequency", None]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodingDecision(BaseModel):
    theme_number: int = Field(
        ...,
        description="Sequential theme identifier as provided",
        examples=[1, 2, 3]
    )
    theme_name: str = Field(
        ...,
        description="Theme name as provided",
        examples=["Waiting Time", "Product Quality"]
    )
    matched_candidates: List[MatchedCandidate] = Field(
        ...,
        description="Best matching existing codes from codebook (for traceability)",
        examples=[[{"code": "Service Speed", "definition": "References to speed of service"}]]
    )
    decision: Literal["USE", "MODIFY_VERTICAL", "MODIFY_HORIZONTAL", "CREATE"] = Field(
        ...,
        description="Coding decision: USE existing, MODIFY existing, or CREATE new",
        examples=["USE", "MODIFY_HORIZONTAL", "CREATE"]
    )
    source_code: Optional[str] = Field(
        default=None,
        description="Exact candidate code name if USE/MODIFY, null if CREATE",
        examples=["Service Speed", None]
    )
    modify_parameters: ModifyParameters = Field(
        ...,
        description="Parameters for modification. Set all fields to 'none'/null for USE and CREATE decisions."
    )
    justification: str = Field(
        ...,
        description="Decision explanation referencing conceptual family and abstraction level comparison",
        examples=["Theme belongs to same family (service timing) at same abstraction level - MODIFY_HORIZONTAL to broaden scope"]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# NOTE: validation_alias="coding_devision" is a workaround for LLM outputs
# that occasionally misspell "decision" as "devision". populate_by_name=True
# allows both the canonical field name and the alias to be accepted.
class CodingDecisionOutput(BaseModel):
    coding_decision: CodingDecision = Field(
        ...,
        validation_alias="coding_devision",
        description="The coding decision result"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


# -----------------------------------------------------------------------------
# 3a. CODE_CREATION_PROMPT
# -----------------------------------------------------------------------------


CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant. Your task: CREATE a new code for a newly identified theme within the {facet_name} facet.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES responses -- the shared interpretive thread. The code must be atomic (one semantic nucleus) and within the {facet_name} dimension.

---

PARAMETERS

<context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Language: {language}
Domain: {domain} | Entity: {entity}
</context>

<new_theme>
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (must be covered):
  {inclusion}
</new_theme>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

---

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of travel delays", "Perception of vehicle cleanliness", "Feeling of personal safety"
- RIGHT: "Travel delays", "Vehicle cleanliness", "Personal safety"
- VALENCE-SENSITIVE: The label should reflect the valence direction of the theme.
  If the theme covers positive/reinforcing ideas, the label should convey that direction (e.g., "Short wait times", "Good accessibility").
  If the theme covers negative/undermining ideas, use wording that reflects absence, insufficiency, or negation (e.g., "Long wait times", "Poor accessibility").
  A valence indicator will be appended automatically — focus on making the label semantically directional.

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

ASSIGNMENT EXAMPLES
- inclusion: 2-3 expressions that SHOULD be coded here.
- exclusion: 1-2 boundary cases that should NOT.
- near_neighbor: closest confusable theme + one distinguishing sentence.

---

Use theme_number and theme_name exactly as provided. Set source_code to null. Write all values in {language}. Provide your output as valid JSON following the response schema provided.
"""

# -----------------------------------------------------------------------------
# 3b. CODING_MODIFICATION_PROMPT
# -----------------------------------------------------------------------------

HORIZONTAL_INSTRUCTIONS = """
- Keep the original code's abstraction level.
- Find the single atomic shared concept that covers both the original code
  and the new theme within their shared conceptual family.
- The modified label must reflect the broadened meaning without introducing
  multiple aspects or becoming more abstract than necessary.
- The modified definition must describe the shared meaning space,
  incorporating original inclusions + inclusion_update and
  original exclusions + exclusion_update."""

VERTICAL_INSTRUCTIONS = """
- Same conceptual family but different abstraction levels: create hierarchical structure.
- Original code and new theme remain atomic child codes.
- Parent code represents the shared conceptual family.

Parent label:
  - If {parent_theme_label} is provided, use it as-is.
  - Otherwise, generate a label at a higher abstraction level.
  - Must express shared conceptual family, not blend child labels.
  - Must be broader, not vaguer.

Structure: Parent = conceptual anchor (broader), Children = distinct manifestations (narrower).
Child meanings do not change."""


CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant. Your task: MODIFY an existing code to include a new theme within the {facet_name} facet, preserving atomic meaning and clear boundaries.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. The modified code must remain atomic (one semantic nucleus) and within the {facet_name} dimension.

---

PARAMETERS

<context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Language: {language}
Domain: {domain} | Entity: {entity}
</context>

<new_theme>
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (must be covered):
  {inclusion}
</new_theme>

<original_code>
- code_label: {source_code}
- code_definition: {source_definition}
</original_code>

<current_assignment_examples>
Current inclusion examples:
  {current_inclusion}
</current_assignment_examples>

<required_modifications>
- inclusion_update: {inclusion_update}
- exclusion_update: {exclusion_update}
</required_modifications>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

---

MODIFICATION INSTRUCTIONS
{modification_instructions}

---

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of travel delays", "Perception of vehicle cleanliness", "Feeling of personal safety"
- RIGHT: "Travel delays", "Vehicle cleanliness", "Personal safety"
- VALENCE-SENSITIVE: The label should reflect the valence direction of the theme.
  If the theme covers positive/reinforcing ideas, the label should convey that direction (e.g., "Short wait times", "Good accessibility").
  If the theme covers negative/undermining ideas, use wording that reflects absence, insufficiency, or negation (e.g., "Long wait times", "Poor accessibility").
  A valence indicator will be appended automatically — focus on making the label semantically directional.

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

ASSIGNMENT EXAMPLES
- inclusion: Combine original + new expressions from inclusion_update.
- exclusion: Combine original + new boundaries from exclusion_update.
- near_neighbor: Update if boundaries changed. Identify closest confusable concept.

---

Use theme_number, theme_name, and source_code exactly as provided. Write all values in {language}. Provide your output as valid JSON following the response schema provided.
"""


class GeneratedCode(BaseModel):
    theme_number: int = Field(
        ...,
        description="Sequential theme identifier as provided",
        examples=[1, 2, 3]
    )
    theme_name: str = Field(
        ...,
        description="Theme name as provided",
        examples=["Waiting Time", "Product Quality"]
    )
    source_code: Optional[str] = Field(
        default=None,
        description="Existing code being modified (null for CREATE)",
        examples=["Service Speed", None]
    )
    code_label: str = Field(
        ...,
        max_length=60,
        description="New or modified code label (1-5 word meaning-based noun phrase)",
        examples=["Waiting Time", "Service Response"]
    )
    code_definition: str = Field(
        ...,
        max_length=250,
        description="<=30-word operational definition describing what belongs in this code",
        examples=["References to duration of waiting for service or products"]
    )
    assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Concrete inclusion/exclusion examples for coding",
        examples=[{
            "inclusion": ["Expresses concern about waiting", "Mentions delay as negative"],
            "exclusion": ["Positive comments about speed"],
            "near_neighbor": {"label": "Service Speed", "tell_apart_rule": "Waiting focuses on negative delay, speed on positive efficiency"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('code_label')
    @classmethod
    def validate_label_length(cls, v):
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(f"code_label must be <=10 words, got {word_count}")
        return v

    @field_validator('code_definition')
    @classmethod
    def validate_definition_length(cls, v):
        word_count = len(v.split())
        if word_count > 36:
            raise ValueError(f"code_definition must be <=36 words, got {word_count}")
        return v


class CodeGenerationOutput(BaseModel):
    generated_code: GeneratedCode = Field(
        ...,
        description="The generated or modified code"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -----------------------------------------------------------------------------
# 4. VALIDATION_PROMPT
# -----------------------------------------------------------------------------

USE_VALIDATION_INSTRUCTIONS = """
**Scenario: USE existing code**

Validate that the existing code fully captures this theme's meaning.
- Does the code's definition cover all expressions in the new theme?
- Would assigning this theme lose any meaningful distinctions?
- Are there uncovered expressions?

If FAILS: recommend MODIFY (horizontal or vertical) or CREATE.
"""


MODIFY_HORIZONTAL_VALIDATION_INSTRUCTIONS = """
**Scenario: MODIFY_HORIZONTAL (broaden at same abstraction level)**

Validate that the modification broadens scope while preserving the semantic core.
- Does the new label preserve the original code's central meaning?
- Do BOTH original and new expressions fit under the unified meaning?
- Is this genuinely broadening, not replacing the concept?

CRITICAL: If the label shifts the core concept -> REJECT, recommend CREATE.
"""

MODIFY_VERTICAL_VALIDATION_INSTRUCTIONS = """
**Scenario: MODIFY_VERTICAL (create parent at higher abstraction level)**

Validate that the hierarchical structure is well-formed.
- Is the parent abstract enough to encompass both children?
- Does the parent represent the shared family, not a label blend?
- Do children remain atomic and distinct?
- Is the abstraction difference genuine, not just wording variation?

If FAILS: recommend MODIFY_HORIZONTAL or CREATE.
"""

CREATE_VALIDATION_INSTRUCTIONS = """
**Scenario: CREATE new code**

Validate that this theme represents a genuinely novel concept.
- Is there truly no existing code that covers this theme?
- Does this fill a real gap, not just a wording preference?
- Is the new code sufficiently different from ALL existing codes?

If FAILS: recommend USE or MODIFY.
"""

VALIDATION_PROMPT = """
You are a codebook curator for thematic analysis following Braun & Clarke (2006) methodology. Your role: maintain a parsimonious codebook with non-overlapping, non-redundant codes by reviewing coding proposals.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES responses -- the shared interpretive thread that explains WHY expressions belong together.

---

CONTEXT

<codebook_context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Domain: {domain} | Topic: {topic}

Existing codes in codebook:
{code_text}
</codebook_context>

<coding_proposal>
Theme to evaluate:
- name: "{theme_name}"
- description: "{theme_description}"

Proposal to review:
{step3_recommendation}

Expressions the code should cover:
  {inclusion_examples}
</coding_proposal>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

---

SCENARIO-SPECIFIC VALIDATION

{validation_instructions}

---

CODE QUALITY RULES (apply to all scenarios)
A code must be:
- TAXONOMIC: falls within the {facet_name} scope.
- ATOMIC: one indivisible concept; unsplittable into independently meaningful parts.
- SINGLE-VALUED: one clear concept, no blending from different families.
- ACTIONABLE: directly identifiable and relevant to the research intent.
No conjunctions, slashes, or compound constructions in labels.

---

VALIDATION STEPS

1. Apply the scenario-specific validation questions above.
2. If the proposal fails any criterion, determine the correct decision (USE / MODIFY_HORIZONTAL / MODIFY_VERTICAL / CREATE).
3. Produce a final code (label + definition + assignment examples) per the rules below.

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of travel delays", "Perception of vehicle cleanliness", "Feeling of personal safety"
- RIGHT: "Travel delays", "Vehicle cleanliness", "Personal safety"
- VALENCE-SENSITIVE: The code label should reflect the valence direction of its source ideas. Do not neutralize directional labels.

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

---

OUTPUT INSTRUCTIONS

- Use theme_number and theme_name exactly as provided in the proposal.
- source_code: USE = exact code from proposal; MODIFY = exact existing code; CREATE = null.
- Write all values in {language}.
- Provide your output as valid JSON following the response schema provided.
"""


class ValidatedCode(BaseModel):
    code: str = Field(
        ...,
        max_length=60,
        description="Final validated code label (<=8 words, rule-compliant)",
        examples=["Waiting Time", "Service Response"]
    )
    definition: str = Field(
        ...,
        max_length=250,
        description="Final validated definition (<=30 words, operational, grounded)",
        examples=["References to duration of waiting for service or products"]
    )
    assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Validated/refined assignment examples for coding",
        examples=[{
            "inclusion": ["Validated inclusion example 1", "Validated inclusion example 2"],
            "exclusion": ["Validated boundary case to exclude"],
            "near_neighbor": {"label": "Nearby Theme", "tell_apart_rule": "This theme focuses on X, nearby theme focuses on Y"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('code')
    @classmethod
    def validate_label_length(cls, v):
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(f"code label must be <=10 words, got {word_count}")
        return v

    @field_validator('definition')
    @classmethod
    def validate_definition_length(cls, v):
        word_count = len(v.split())
        if word_count > 36:
            raise ValueError(f"definition must be <=36 words, got {word_count}")
        return v


class OriginalRecommendation(BaseModel):
    code: str = Field(
        ...,
        description="Exact recommended code label from proposal",
        examples=["Waiting Time", "Product Quality"]
    )
    definition: str = Field(
        ...,
        description="Exact recommended definition from proposal",
        examples=["References to time spent waiting"]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodeValidation(BaseModel):
    theme_number: int = Field(
        ...,
        description="Sequential theme identifier as provided in proposal",
        examples=[1, 2, 3]
    )
    theme_name: str = Field(
        ...,
        description="Theme name as provided in proposal",
        examples=["Waiting Time", "Product Quality"]
    )
    original_recommendation: OriginalRecommendation = Field(
        ...,
        description="The original coding recommendation being validated"
    )
    verdict: Literal["APPROVE", "REJECT"] = Field(
        ...,
        description="Binary accept/reject of original proposal. APPROVE = proposal is acceptable as-is; REJECT = proposal needs correction. NOT a coding decision - use validated_decision for USE/MODIFY/CREATE.",
        examples=["APPROVE", "REJECT"]
    )
    decision_rationale: str = Field(
        ...,
        description="Brief explanation of why proposal was approved or rejected",
        examples=["Proposal maintains atomicity and aligns with taxonomy axis"]
    )
    validated_decision: Literal["USE", "MODIFY_HORIZONTAL", "MODIFY_VERTICAL", "CREATE"] = Field(
        ...,
        description="Final decision after validation",
        examples=["USE", "MODIFY_HORIZONTAL", "CREATE"]
    )
    source_code: Optional[str] = Field(
        default=None,
        description="If USE: exact code from proposal; If MODIFY: exact existing code being modified; If CREATE: null",
        examples=["Service Speed", None]
    )
    validated_code: ValidatedCode = Field(
        ...,
        description="Final validated code with label, definition, and assignment examples"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationResult(BaseModel):
    code_validation: CodeValidation = Field(
        ...,
        description="The validation result for the coding proposal"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# =============================================================================
# STEP 7 THEME ORGANIZATION WITH REASONING MODELS
# =============================================================================

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative research methodologist organizing codes into a hierarchical codebook.
Your PRIMARY task is to ORGANIZE codes into themes - NOT to reduce their number.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the codes to organize:
<raw_codes>
{raw_codes}
</raw_codes>

Output your response in this language:
<language>
{language}
</language>

# CRITICAL: Preservation Over Reduction
Your goal is to ORGANIZE codes into a clear hierarchy, NOT to reduce the number of codes.
- PRESERVE all codes that represent distinct concepts
- Only MERGE codes that are TRUE DUPLICATES (identical meaning, just different wording)
- When in doubt, KEEP codes separate

# When to Merge (STRICT criteria - all must apply)
ONLY merge two codes if ALL of the following are true:
1. They describe the EXACT same concept (not just related concepts)
2. A researcher would report them as ONE finding (not as variations)
3. Their inclusion examples overlap completely
4. No exclusion examples or tell-apart rules distinguish them

If ANY doubt exists → KEEP SEPARATE

# Structure and Hierarchy
Organize codes into a 2-level or 3-level hierarchy:

**2-level (Theme → Code)**: Use when themes are simple and codes don't need sub-grouping
**3-level (Theme → Category → Code)**: Use when a theme contains multiple sub-concepts that benefit from grouping

Guidelines:
- Every code must belong to exactly one theme
- Themes should be conceptually coherent (related codes grouped together)
- Use 3-level hierarchy when ≥3 codes share a clear sub-concept within a theme
- Aim for 5-15 themes depending on codebook size

# Theme and Code Naming

**Theme Labels**
- ≤ 10 words, noun phrases preferred
- Describe the conceptual domain (e.g., "Duurzaamheid", "Klantenservice", "Prijsperceptie")
- No conjunctions or slashes

**Code Labels**
- Keep original code labels unless they violate naming rules
- ≤ 10 words, specific and atomic

**Code Descriptions**
- ≤ 20 words
- Define what belongs in this code
- Use patterns like: "Mentions of…", "References to…"

# Required Output Format

Think through the organization, then provide JSON:

{{
  "analysis": "In {language}: (1) How codes were organized into themes, (2) Any codes merged (with justification - should be very few), (3) Hierarchy structure chosen (2-level or 3-level), (4) Final count: X codes organized into Y themes.",
  "refined_codebook": [
    {{
      "theme": "Theme label",
      "codes": [
        {{
          "id": "original code_id (or comma-separated IDs if merged)",
          "code": "Code label",
          "description": "≤ 20 words explanation",
          "category": ""  // Empty for 2-level, category name for 3-level
        }}
      ]
    }}
  ]
}}

Notes:
- The number of codes in output should be close to the number of input codes (merging should be rare)
- No commentary before or after JSON
- All text in the specified output language

Begin organizing the codebook.
"""


CODEBOOK_MERGE_PROMPT = """
You are a qualitative research methodologist consolidating multiple codebooks into one unified structure.
Your PRIMARY task is to UNIFY the organization - NOT to reduce the number of codes.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the codebooks to consolidate:
<codebooks>
{codebooks_summary}
</codebooks>

All output must be in this language:
<language>
{language}
</language>

# CRITICAL: Preservation Over Reduction
You are consolidating {n_codebooks} codebooks from different batches. Your goal is to:
1. PRESERVE all unique codes from all codebooks
2. Only MERGE codes that are TRUE DUPLICATES (identical meaning appearing in multiple codebooks)
3. Create a unified theme structure that organizes ALL codes

# When to Merge (STRICT criteria)
ONLY merge codes if they are TRUE DUPLICATES:
- EXACT same concept appearing in multiple codebooks (due to batch overlap)
- A researcher would consider them identical findings

Do NOT merge codes that are:
- Related but distinct concepts
- Different aspects of the same topic
- Similar but with different nuances

When in doubt → KEEP SEPARATE

# Consolidation Steps
1. Identify TRUE duplicates across codebooks (codes with identical meaning)
2. Keep all unique codes
3. Organize all codes into a unified theme structure
4. Use 2-level or 3-level hierarchy as appropriate

# Theme Structure
**2-level (Theme → Code)**: Simple organization
**3-level (Theme → Category → Code)**: Use when themes have clear sub-groupings

Guidelines:
- Merge similar THEMES across codebooks (organizational labels), but preserve the CODES within them
- Every code must appear exactly once in the final codebook
- Aim for 5-15 themes depending on total code count

# Label Rules
- Theme labels: ≤10 words, noun phrases, no conjunctions/slashes
- Code labels: Keep original labels, ≤10 words
- Descriptions: ≤30 words, define when to use the code

# Output Format

{{
  "analysis": "In {language}: (1) How codebooks were unified, (2) Any duplicate codes merged (should be few - only true duplicates from batch overlaps), (3) Theme structure chosen, (4) Final count: X codes from Y input codebooks organized into Z themes.",
  "refined_codebook": [
    {{
      "theme": "Theme label",
      "codes": [
        {{
          "id": "original code ID(s)",
          "code": "Code label",
          "description": "Code definition (≤30 words)",
          "category": ""  // Empty for 2-level, category name for 3-level
        }}
      ]
    }}
  ]
}}

IMPORTANT: The total number of unique codes in your output should be close to the total unique codes across all input codebooks. Significant reduction indicates over-merging.

Begin consolidating the codebooks.
"""


# =============================================================================
# STEP 8: CODE ASSIGNMENT
# =============================================================================

# Stage 1: Evaluate default code from cluster
DEFAULT_CODE_EVALUATION_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses. 
Your task is to determine if there is explicit or clearly paraphrased evidence that a specific code appears in a given response text.

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response you need to analyze:
<response>
Idea ID: {idea_id}
Idea Text: {idea_text}
</response>

Here is the code you need to evaluate:
<code_details>
Code: {default_code}
Definition: {default_definition}

Inclusion Examples (valid references for this code):
    {inclusion_examples}

Exclusion Examples (invalid references for this code):
    {exclusion_examples}

Boundary: This code covers "{default_code}", which differs from "{near_neighbor_label}"
How to tell them apart: {tell_apart_rule}
</code_details>

Follow these DECISION RULES strictly:

1) Evidence types
   • Explicit: the response uses terms that directly express the target concept.
   • Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   • Do NOT infer intent beyond the text. Do not rely on general world knowledge.

2) Include vs Exclude
   • Include if the target concept is explicit or an unambiguous paraphrase appears anywhere in the response.
   • Exclude if the response:
       – Only expresses the near neighbor concept (per {tell_apart_rule});
       – Matches any Exclusion Example pattern;
       – Mentions the concept only in a negated or hypothetical/conditional way (e.g., “would/if/might” without an asserted claim);
       – Is too generic or off-topic.
   • If both Inclusion-like and Exclusion-like signals appear, Exclusion takes precedence unless the Inclusion is explicit and clearly satisfies the Definition.

3) Minimal supporting span
   • If Including, identify the shortest verbatim span in the response that demonstrates the concept.
   • If Excluding, no supporting span is needed.
   • Preserve original casing and spelling; do not correct typos.

4) Multiple claims / long answers
   • Evaluate the entire Idea Text. If any part contains qualifying evidence, Include.
   • If the answer only restates the survey question or is empty/“N/A”, Exclude.

5) Confidence (0.00–1.00)
• 0.90–1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated in the response; no interpretation needed. Another trained coder would definitely agree.
• 0.70–0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference. Another trained coder would likely agree.
• 0.50–0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment to justify. Reasonable coder disagreement is likely; discussion may be required.
• 0.00–0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response. Another trained coder would not assign this code.


6) **Confidence Threshold Rule (Critical)**
   • If the confidence score would be **below 0.70**, the decision **must be EXCLUDE**.
   • Borderline or partially implied concepts should **not** be coded as present.


IMPORTANT — RATIONALE STRUCTURE:
   • The rationale MUST begin with either "INCLUDE:" or "EXCLUDE:"
   • If INCLUDE: follow with the minimal supporting span in quotes, then a short explanation referencing the definition.
     Example: INCLUDE: "we krijgen geen begeleiding" → explicitly expresses lack of support.
   • If EXCLUDE: briefly state the rule-based reason for exclusion.
     Example: EXCLUDE: No text expresses the target concept; content is generic.

   
Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "confidence": CONFIDENCE_SCORE,
  "rationale":  "INCLUDE: \"...\" → explanation in {language}" OR "EXCLUDE: brief explanation in {language}"
}}

Critical requirements:
- The confidence score must be a number between 0.00 and 1.00
- If the confidence score is below 0.70, the rationale MUST begin with "EXCLUDE:"
- The rationale must follow the INCLUDE:/EXCLUDE: format exactly
- Focus only on the specific concept defined by the code
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""

# Stage 1b: Evaluate multiple codes from cluster family
FAMILY_CODE_EVALUATION_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses.
Your task is to evaluate MULTIPLE candidate codes from the same cluster family and determine which one (if any) best fits the response.

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response you need to analyze:
<response>
Idea ID: {idea_id}
Idea Text: {idea_text}
</response>

Here are the candidate codes from this cluster family. Evaluate EACH code against the response:
<candidate_codes>
{candidate_codes_formatted}
</candidate_codes>

**Evaluation Process:**
1. Evaluate EACH candidate code against the response independently
2. For each code, determine if there is explicit or clearly paraphrased evidence
3. If multiple codes match with confidence >= 0.70, choose the MOST SPECIFIC one that fits the evidence
4. If no code matches with confidence >= 0.70, set best_match.code to "NONE"

**Decision Rules (apply to EACH code):**

1) Evidence types
   • Explicit: the response uses terms that directly express the target concept.
   • Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   • Do NOT infer intent beyond the text. Do not rely on general world knowledge.

2) Include vs Exclude
   • Include if the target concept is explicit or an unambiguous paraphrase appears anywhere in the response.
   • Exclude if the response:
       – Only expresses a different code's concept;
       – Matches any Exclusion Example pattern;
       – Mentions the concept only in a negated or hypothetical/conditional way;
       – Is too generic or off-topic.

3) Minimal supporting span
   • If Including, identify the shortest verbatim span in the response that demonstrates the concept.
   • Preserve original casing and spelling; do not correct typos.

**Confidence Anchors (0.00–1.00):**
• 0.90–1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated in the response; no interpretation needed.
• 0.70–0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference.
• 0.50–0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment. EXCLUDE threshold.
• 0.00–0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response.

**Confidence Threshold Rule (Critical):**
• Only codes with confidence >= 0.70 qualify as matches.
• If the best-fitting code has confidence < 0.70, set best_match.code to "NONE".

**Tie-Breaking (when multiple codes have confidence >= 0.70):**
1. Choose the code with the highest confidence score.
2. If still tied, prefer the more specific code (narrower definition).
3. If still tied, prefer the code whose definition most closely matches the supporting span.

Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "evaluations": [
    {{"code": "CODE_NAME_1", "confidence": SCORE, "rationale": "INCLUDE: \"span\" → explanation" or "EXCLUDE: reason"}},
    {{"code": "CODE_NAME_2", "confidence": SCORE, "rationale": "INCLUDE: \"span\" → explanation" or "EXCLUDE: reason"}}
  ],
  "best_match": {{
    "code": "BEST_CODE_NAME or NONE",
    "confidence": SCORE,
    "rationale": "INCLUDE: \"span\" → explanation in {language}" or "EXCLUDE: brief explanation in {language}"
  }}
}}

Critical requirements:
- Evaluate ALL candidate codes in the evaluations array
- The best_match.code must be one of the evaluated codes OR "NONE" if no code reaches 0.70 confidence
- The best_match.confidence must match the confidence of the selected code (or 0.0 if NONE)
- All rationales must follow the INCLUDE:/EXCLUDE: format
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""

# Stage 2: Fallback assignment from all codes
FALLBACK_CODE_ASSIGNMENT_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses. 
Your task is to assign exactly one existing code from the provided codebook to a response, but only if there is explicit or clearly paraphrased evidence that the specific code concept appears in the response text.

Here is the survey question context:
<survey_question>
{var_lab}
</survey_question>

Here is the response you need to analyze:
<response>
Idea ID: {idea_id}
Idea Text: {idea_text}
</response>

Here are the available codes in the codebook:
<codebook>
{all_codes}
</codebook>

**Decision Rules:**
- Assign EXACTLY ONE code from the codebook if — and only if — the response explicitly states or unambiguously paraphrases the specific concept in that code’s definition.
- If the response is broader/more generic than a code’s definition, that code does NOT fit.
- Prefer codes whose definitions are most specific to the quoted evidence (not merely thematically related).
- Do not infer meaning beyond the text. Negated or hypothetical/conditional mentions (e.g., “not X”, “would/if/might”) do NOT qualify as evidence.
- If no code has clear evidence, assign "{unknown_label}" with low confidence.

**Confidence Level Anchors:**
• 0.90–1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated in the response; no interpretation needed. Another trained coder would definitely agree.
• 0.70–0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference. Another trained coder would likely agree.
• 0.50–0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment to justify. Reasonable coder disagreement is likely; discussion may be required.
• 0.00–0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response. Another trained coder would not assign this code.

Tie-breaking (when multiple candidates look plausible):
1) Choose the code supported by the strongest minimal verbatim span most closely matching its definition.
2) If still tied, choose the code with the more specific definition.
3) If still tied or evidence remains ambiguous, assign "{unknown_label}".

**Confidence Threshold Rule:**
- If the best-fitting interpretation would result in a confidence score below 0.70, assign "{unknown_label}".

**IMPORTANT — RATIONALE FORMAT:**
- The assignment_rationale MUST begin with either:
     "Match:" if assigning a code (confidence ≥ 0.70)
     "{unknown_label}:" if assigning "{unknown_label}" (confidence < 0.70 or no clear concept match)
- If MATCH: include the minimal supporting span in quotes, then explain why it fits the selected code.
- If {unknown_label}: briefly explain that no code was clearly supported.

**Analysis Process:**
1) Evidence Identification: Scan the response for candidate spans that might support specific code concepts.
2) Supporting Span Extraction: For the best-fitting code, identify the shortest verbatim span that demonstrates the concept (preserve casing/spelling).
3) Conceptual Matching: Confirm the span satisfies the chosen code’s definition (not just a related theme).
4) Confidence Assessment: Apply the anchors above.
5) Final Assignment: Output a single code, or "{unknown_label}" if none fit well.

Provide your analysis and assignment in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Match: \"...\" → explanation" OR "{unknown_label}: explanation in {language}"
}}

"""


