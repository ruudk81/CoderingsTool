"""
Experimental Prompts for Step 3: Idea Extraction

This file contains the prompts and their paired Pydantic response models
used by ideaExtractor_exp.py. Modify these prompts to experiment with
different idea extraction approaches.

Migration: Response models co-located with prompts following instructor schema pattern.
Original source: src/prompts.py (STEP 3: IDEA EXTRACTION section)
"""

from typing import Any, ClassVar, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, create_model

# =============================================================================
# STEP 3: IDEA EXTRACTION
# =============================================================================

# -----------------------------------------------------------------------------
# CONTEXT SPECIFIER PROMPTS (Group 1 & 2)
# -----------------------------------------------------------------------------

CONTEXT_SPECIFIER_PROMPT1 = """
You are analyzing survey responses to extract contextual metadata.

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
   - Common values: "consumer", "client", "employee", "partner", "expert", "general_public", "beneficiary"
   - Examples: "consumer" (customer feedback), "employee" (internal survey)

3. **intent**: Purpose/communicative function
   - What are respondents trying to do with their responses?
   - Common values: "evaluate", "describe", "suggest", "complain", "praise", "question"
   - Examples: "evaluate" (assessing brand), "suggest" (recommendations)

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
        examples=["consumer", "employee", "partner", "expert","general_public", "beneficiary"]
    )
    intent: str = Field(
        description="Communicative function - what respondents are trying to do",
        examples=["evaluate", "suggest", "complain", "describe", "explain", "clarify"]
    )


CONTEXT_SPECIFIER_PROMPT2 = """
You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 2 specifiers (subject matter):

1. **domain**: Industry/sector domain
   - What industry or sector does this survey concern?
   - Examples: "finance" (banking survey), "healthcare" (hospital satisfaction)

2. **topic**: Specific subject matter
   - What is the specific topic being discussed?
   - Examples: "brand_association" (brand perception), "customer_service" (support experience)

3. **entity**: Main entity of interest
   - What entity (group, person or thing) is the primary focus?
   - Use lowercase with underscores for multi-word names
   - Examples: "asn_bank", "tesla_model_3", "albert_heijn", "ns_trains"

Provide concise answers (2-5 words each) in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class GenericSpecifierGroup2Response(BaseModel):
    """Group 2: Subject matter"""
    domain: str = Field(
        description="Industry/sector domain the survey concerns",
        examples=["finance", "healthcare", "retail", "technology"]
    )
    topic: str = Field(
        description="Specific subject matter being discussed",
        examples=["brand_association", "customer_service", "product_quality"]
    )
    entity: str = Field(
        description="Main entity of interest, lowercase_with_underscores",
        examples=["abnamro_bank", "tesla_model_3", "albert_heijn"]
    )


# -----------------------------------------------------------------------------
# CONSOLIDATION PROMPTS
# -----------------------------------------------------------------------------

CONSOLIDATE_SPECIFIERS_GROUP1 = """
You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 1 specifiers (speaker characteristics):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations (e.g., "evaluative" vs "assessment viewpoint" → choose most accurate)
- For **lang**: Standardize to ISO format (e.g., "Dutch" → "nl-NL", "English" → "en-US")
- For **perspective**: Choose the most representative viewpoint across all chunks
- For **intent**: Choose the most common communicative goal

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 1 specifiers as valid JSON following the response schema provided."""


CONSOLIDATE_SPECIFIERS_GROUP2 = """
You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 2 specifiers (subject matter):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations (e.g., "financial services" vs "banking sector" → choose most accurate)
- For **domain**: Standardize to lowercase, single/hyphenated word
- For **topic**: Choose the most representative subject matter across all chunks
- For **entity**: Standardize format (lowercase_with_underscores)

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 2 specifiers as valid JSON following the response schema provided."""


# -----------------------------------------------------------------------------
# TAXONOMY CHUNK SCORING
# -----------------------------------------------------------------------------

CODING_DIMENSION_SCORING_PROMPT = """
You are selecting the SINGLE best coding dimension for organizing a set of survey responses.

Your task is NOT to summarize responses, judge quality, or assign labels to each response.
Your ONLY goal is to decide which ONE dimension — WHAT, WHY, HOW, WHO, WHEN, or WHERE — best explains the MAIN way the responses DIFFER from one another.

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

Here is the intent behind the responses:
<intent>
{intent}
</intent>

Here is a sample of SHORT, COARSE responses for you to analyze:
<sample_responses>
{chunk_responses}
</sample_responses>

------------------------------
HOW TO THINK ABOUT THE TASK
------------------------------
Ask yourself:
"If I had to cluster these responses into groups, which dimension would create the most meaningful separation in answering the survey question?"

Choose the dimension that explains the LARGEST share of meaningful variation across MOST responses (not just edge cases).

If multiple axes seem plausible:
- Choose the dimension that would be used as the *top-level folder* to organize these responses.
- If still tied, choose the dimension that applies to a larger fraction of responses.

------------------------------
Coding dimensions (choose exactly one)
------------------------------

1) WHY (reason_driver)
- Differences are motivations, goals, values, concerns, or trade-offs.
- Excludes: attributes (WHAT), methods (HOW), actors (WHO), timing (WHEN), context/channel (WHERE).

2) HOW (outcome_enablers)
- Differences are about how an outcome would be achieved or carried out, including:
   A) Change-enabling mechanisms: actions, changes, interventions, tools, or mechanisms that make the outcome possible
   B) Execution pathways: steps, processes, workflows, procedures, or ways of carrying something out
- Includes: recommendations, tactics, methods, implementation approaches, processes, or preferred ways of "getting from here to there."
- Excludes: what something *is or has* (WHAT), why someone wants something (WHY), who is involved (WHO), timing (WHEN), context/channel (WHERE).

3) WHO (actor_target)
- Differences are who is involved, affected, targeted, or responsible.
- Excludes: methods (HOW), motivations (WHY), attributes (WHAT), timing (WHEN), context/channel (WHERE).

4) WHEN (time_urgency)
- Differences are timing, urgency, frequency, sequence, or lifecycle stage.
- Excludes: methods (HOW), motivations (WHY), actors (WHO), attributes (WHAT), context/channel (WHERE).

5) WHERE (location_context)
- Differences are environment, setting, channel, platform, touchpoint, or situation.
- Excludes: methods (HOW), motivations (WHY), actors (WHO), timing (WHEN), attributes (WHAT).

6) WHAT (entity_descriptor)
- Differences are properties, attributes, features, or constraints of the entity as it currently exists or has existed.
- This is descriptive, not prescriptive.
- Excludes:
  - desired changes or improvements (these belong to HOW),
  - motivations (WHY),
  - actors (WHO),
  - timing (WHEN),
  - context/channel (WHERE)

------------------------------
ANALYSIS PROCESS (internal)
------------------------------
Do NOT output your step-by-step reasoning.
You MUST still follow this process internally:
1) Identify the dominant pattern of variation across the sample, in light of the intent ("{intent}") in answering the survey question.
2) Score each dimension for explanatory power over the variation: 0 = absent, 1 = present but secondary, 2 = primary.
3) Choose the single dimension with the highest score (break ties using the rules above).
4) Extract 2–3 verbatim snippets from <sample_responses> that support the chosen dimension.

All string values (including evidence snippets) must be in {language}.
Evidence snippets must be copied verbatim from <sample_responses>.
If fewer than 3 distinct snippets exist, include as many as possible without inventing any.
Clarification must explicitly contrast the chosen dimension with at least one plausible alternative.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class CodingDimensionChunkResponse(BaseModel):
    """LLM response for single chunk coding dimension scoring."""
    primary_dimension: Literal["HOW", "WHAT", "WHEN", "WHERE", "WHO", "WHY"] = Field(
        description="The single best coding dimension for organizing responses"
    )
    evidence: List[str] = Field(
        description="2-3 verbatim snippets from sample_responses supporting the chosen dimension",
        examples=[["good service", "too expensive", "friendly staff"]]
    )
    clarification: str = Field(
        description="1-2 sentences explaining why this dimension is most appropriate, contrasting with at least one alternative"
    )


# -----------------------------------------------------------------------------
# TAXONOMY CONSOLIDATION
# -----------------------------------------------------------------------------

CODING_DIMENSION_CONSOLIDATION_PROMPT = """
You are a taxonomy consolidation specialist.
Your task is to analyze multiple chunk-level coding dimension analyses and consolidate them into a single, coherent global coding dimension for a survey question.

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
- Intent by response: {intent}
</context>

Here are the chunk-level analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

## YOUR TASK

You must consolidate these chunk-level analyses into a single global coding dimension. Each chunk analysis evaluated the same survey question and produced a primary coding dimension and supporting evidence. Your job is to synthesize these into one coherent framework.

## ANALYSIS STEPS

Follow these steps in order:

**Step 1: Review and consolidate chunk-level analyses**
Examine all chunk-level analyses carefully. Note areas of convergence and divergence. Identify which dimensions appear across multiple chunks and assess the quality of evidence supporting each.

**Step 2: Select the PRIMARY coding dimension**
Choose the ONE dimension (WHAT, WHY, HOW, WHO, WHEN, or WHERE) that:
- Shows strong and consistent support across chunks
- Provides the clearest partition boundaries for coding responses
- Offers the best interpretability and stability for downstream use

Important: Do NOT select a dimension solely because it appears most frequently. Favor partition clarity, boundary stability, and interpretability over raw frequency counts.

**Step 3: Define the GLOBAL coding dimension**
Write a coding dimension description that:
- Is specific to THIS survey question and response domain
- Clearly falls within the selected primary dimension
- Reconciles and generalizes the chunk-level axes without introducing new organizing principles
- Operates at a mid-level of abstraction (not too narrow, not too broad)
- Can directly seed downstream descriptive code labels
- Clearly indicates what coders should extract from each response

## DECISION RULES

When consolidating:
- If chunk analyses converge on the same dimension, follow the consensus
- If chunk analyses diverge, rely on MECE quality (mutually exclusive, collectively exhaustive) to determine which dimension provides the clearest boundaries
- Optimize for downstream coding usability and cross-coder consistency
- Prefer clarity and stability over cleverness or novelty

All output values must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class CodingDimensionConsolidatedResponse(BaseModel):
    """Consolidated coding dimension selection after merging all chunks."""
    primary_dimension: str = Field(
        description="The selected coding dimension",
        examples=["WHAT", "WHY", "HOW", "WHO", "WHEN", "WHERE"]
    )
    primary_dimension_rationale: str = Field(
        description="2-4 sentence explanation of why this dimension is the dominant organizing principle"
    )
    primary_dimension_description: str = Field(
        description="Clear definition of the coding dimension at proper abstraction level, specific to this survey question"
    )


# -----------------------------------------------------------------------------
# TAXONOMY-AWARE SUBJECT EXTRACTION
# -----------------------------------------------------------------------------

TAXONOMY_AWARE_SUBJECT_PROMPT = """
You are a language expert tasked with generating a precise phrasing template for survey response analysis. 
Your template must fit the survey question and the selected taxonomy axis.

You will be working in the following language:
<language>
{language}
</language>

Here is the survey question you are analyzing:
<survey_question>
{survey_question}
</survey_question>

Here is the context for the survey:
<context>
Type of respondent: {perspective}
Domain: {domain}
Entity of interest: {entity}
Topic: {topic}
Intent by response: {intent}
</context>

Here is the taxonomy guidance for the selected axis:
<taxonomy_guidance>
Selected axis: {taxonomy_axis}: {axis_dimension_description}
Axis anchor: {axis_anchor}
</taxonomy_guidance>

Here is the template pattern and slot definitions you must follow:
<template>
Axis pattern: "{axis_pattern}"

Slot descriptions:
{axis_slots}
</template>

================================================
GLOBAL RULES  
================================================

You must follow these rules when creating your template:
- Semantic framing: normalize meaning, not style.
- Keep it generic: do NOT use domain-specific examples or domain nouns unless required by the survey question itself.
- Do NOT introduce new entities, motives, outcomes, comparisons, or evaluations not present in the survey question framing.
- Use minimal inference only to create a well-formed template and to select an axis-consistent anchor.
- Single clause only. Concise.

================================================
TASK
================================================

You must complete three steps:

**Step 1: Choose the canonical_term (ANCHOR slot)**
- Pick a short noun phrase in the specified language that matches the axis anchor described in the taxonomy guidance
- Avoid respondent pronouns unless the survey question explicitly requires first-person framing
- This term will serve as the anchor concept for the template

**Step 2: Choose the taxonomy_actionable_type**
- Select the single most fitting concept type from the allowed concepts listed here:
  {axis_dimension_allowed_concepts}
- This should be the concept that best captures the dimension being analyzed

**Step 3: Construct the canonical_phrasing (template sentence)**
- Follow the axis pattern provided in the template section
- Replace the ANCHOR slot with your chosen canonical_term from Step 1
- Keep the DIMENSION slot with the literal marker token: {dimension_marker}
- Must be a single grammatical clause in the specified language
- Must read as a direct, natural answer to the survey question once the marker is replaced
- Keep it short (aim for ≤ 10 words excluding the marker)
- Ensure the phrasing flows naturally and grammatically in the target language

================================================
OUTPUT FORMAT
================================================
Begin processing now and provide your output as valid JSON following the response schema provided.
"""

class SubjectExtractionResponse(BaseModel):
    """Response model for subject/actor extraction with axis-aware template."""
    # ClassVar: axis-specific dimension marker (set before LLM call)
    _dimension_marker: ClassVar[str] = "[ACTIONABLE_TAXONOMY_DIMENSION]"

    @classmethod
    def set_dimension_marker(cls, marker: str):
        """Set the axis-specific dimension marker before LLM call."""
        cls._dimension_marker = marker

    canonical_term: str = Field(
        description="Short noun phrase identifying the anchor subject (entity-of-interest) chosen for the template"
    )
    taxonomy_axis: Literal["WHAT", "WHY", "HOW", "WHO", "WHEN", "WHERE"] = Field(
        description="The primary taxonomy dimension"
    )
    taxonomy_actionable_type: str = Field(
        description="The single actionable taxonomy type on the primary axis",
    )
    canonical_phrasing: str = Field(
        description=(
            "Single grammatical clause containing the axis-specific dimension marker token, "
            "reading as a natural answer to the survey question"
        )
    )

    @field_validator('canonical_phrasing', mode='before')
    @classmethod
    def ensure_marker_present(cls, v: str) -> str:
        """Ensure the axis-specific dimension marker is present in the canonical phrasing."""
        marker = cls._dimension_marker
        if isinstance(v, str) and marker not in v:
            v = v.rstrip('.') + f" {marker}."
        return v


def create_subject_extraction_model(axis: str, axis_data: dict, schema_data: dict = None, slot_type_map: dict = None, dimension_marker: str = "[ACTIONABLE_TAXONOMY_DIMENSION]"):
    """Create axis-specific SubjectExtractionResponse with tailored Field descriptions."""
    allowed = ", ".join(axis_data["allowed_concepts"])
    excluded = ", ".join(axis_data["excluded_concepts"])
    slot_type_map = slot_type_map or {}

    # Extract structural forms and notes from schema for canonical_phrasing hint
    schema_data = schema_data or {}
    structural_forms = schema_data.get("structural_forms", [])
    forms_hint = " or ".join(f'"{f}"' for f in structural_forms[:2]) if structural_forms else ""
    notes = schema_data.get("notes", [])
    notes_hint = "; ".join(notes[:2]) if notes else ""

    # Build canonical_phrasing description with schema enrichments
    phrasing_desc = (
        f"Single grammatical clause following the axis pattern: "
        f"{axis_data['pattern']}. "
    )
    if forms_hint:
        phrasing_desc += f"Expected forms: {forms_hint}. "
    if notes_hint:
        phrasing_desc += f"Note: {notes_hint}. "
    phrasing_desc += f"Must contain the literal marker token {dimension_marker}."

    # Enrich descriptions with type_system constraints
    anchor_type = slot_type_map.get("anchor", {})
    anchor_type_hint = f" {anchor_type['description']}" if anchor_type.get("description") else ""

    dimension_type = slot_type_map.get("dimension", {})
    if dimension_type.get("is_alias"):
        dimension_type_hint = f" Dimension slot must be a {dimension_type['type_name']}."
    else:
        dimension_type_hint = ""

    # Type glossary for model-level description (definitions appear once, fields reference by name)
    glossary = _build_type_glossary()
    config_kwargs = {"__config__": ConfigDict(json_schema_extra={"description": glossary})} if glossary else {}

    # Using create_model() so $defs key and title match in JSON Schema
    return create_model(
        f"SubjectExtractionResponse_{axis}",
        __base__=SubjectExtractionResponse,
        **config_kwargs,
        canonical_term=(str, Field(
            description=f"Short noun phrase: {axis_data['noun_phrase_descriptor']}.{anchor_type_hint}"
        )),
        taxonomy_axis=(Literal[axis], Field(
            description=f"Taxonomy axis (must be '{axis}')"
        )),
        taxonomy_actionable_type=(str, Field(
            description=(
                f"The actionable taxonomy type on the {axis} axis. "
                f"Must be one of: {allowed}. "
                f"Must NOT be: {excluded}."
            )
        )),
        canonical_phrasing=(str, Field(description=phrasing_desc + dimension_type_hint)),
    )


# -----------------------------------------------------------------------------
# TAXONOMY-ENRICHED IDEA EXTRACTION
# -----------------------------------------------------------------------------

TAXONOMY_ENRICHED_EXTRACTION_PROMPT ="""You are an expert in extracting structured ideas from survey responses using taxonomy-aware analysis. 
Your task is to identify distinct ideas in a survey response, reformulate each idea using a canonical template, and produce a lightweight taxonomy classification for each idea.

You will be provided with three key pieces of information:

**SURVEY CONTEXT**
<survey_context>
You will be working in the following language: {language}

Here is the survey question being analyzed: "{var_lab}"

Here is the context for the survey:
- Type of respondent: {perspective}
- Domain: {domain}
- Entity of interest: {entity}
- Topic: {topic}
- Intent by response: {intent}
</survey_context>

**TAXONOMY FRAMEWORK**
<taxonomy_instructions>
Here is the taxonomy lens you need to apply: "{noun_phrase_descriptor}"

Instruction: {instruction}

The pattern: {canonical_phrasing} 

{marker} is defined as: {slot_guidance_second}
The selected {taxonomy_axis}-concept: {taxonomy_actionable_type}

</taxonomy_instructions>


**RESPONSE TO ANALYZE**
<response>
Respondent ID: {respondent_id}
Response: {response}
</response>

---

Now follow these five steps to complete the extraction:

**STEP 1: IDENTIFY AND SPLIT IDEAS**

Identify all conceptually distinct instances in which the response describes or refers to a unique instance of the selected {taxonomy_axis}-concept, interpreted in light of the survey question.

CRITICAL SPLITTING RULES:
- Items joined by conjunctions (such as "and", "or", "en", "und", "et", "y", "ou") or commas that refer to different concepts MUST be split into separate ideas
- Example: If the response is "faster and cheaper" → this contains TWO ideas: "faster" (idea 1) and "cheaper" (idea 2)
- Each split idea will get its OWN template instantiation and its OWN taxonomy hierarchy
- Even very short responses (2-5 words) can contain multiple ideas if they enumerate distinct concepts
- When in doubt about whether to split, err on the side of splitting. Over-splitting is preferable to under-splitting
- Assign each unique concept a sequential idea_id starting from "1"

**STEP 2: COMPLETE THE PATTERN **

For each idea, reformulate it using the pattern provided in the taxonomy framework.

Rules for template completion:
- Begin with the fixed pattern prefix exactly as provided (do NOT alter this)
- The template contains a marker token (shown in square brackets) that you must replace. 
- This is the marker token: {axis_anchor}
- Replace ONLY the marker token with  he shortest verbatim span from the original response text that expresses the unique idea identified in Step 1

**STEP 3: CLASSIFY EACH IDEA**

For each idea, assign a semantic classification in the specified language.

1. **INSTANCE**: The verbatim span from the response expressing the idea (cleaned and minimally standardized)

2. **NODE**: A canonical, reusable concept label
   - Remove descriptive qualifiers (e.g., adjectives like "slow," "cheap," "great")
   - Reduce to the base object, action, or concept (noun phrase)
   - Must stay semantically equivalent to the instance
   - If the instance already represents a base concept, the node may remain identical

3. **SEMANTIC CATEGORY**: Classify the idea as one of these types (interpreted for the {taxonomy_axis} dimension):
{semantic_category_table}

4. **CATEGORY LABEL**: A concise descriptive label for the specific classification within the semantic category

5. **ROOT**: Top-level domain framing implied by the survey question ({taxonomy_actionable_type} → {entity}), if identifiable

**Priority Rules** (apply in order when uncertain):
{priority_rules}


**STEP 4: ASSIGN IDs**

- Use the exact respondent_id provided in the response
- Assign sequential idea_id values as strings: "1", "2", "3", etc., for each distinct idea

---

**EDGE CASES**
- If the response is empty, irrelevant, or nonsensical: return []
- If there is one idea: return an array with one object
- If there are multiple ideas: return an array with multiple objects, each with its own sequential idea_id

**OUTPUT**

Begin processing now and provide your output as valid JSON matching the required response schema.
CRITICAL: all output fields must be in {language}
"""

class SemanticTaxonomyResponse(BaseModel):
    """Semantic taxonomy classification for extracted ideas.
    Field descriptions and examples are set per-axis in create_taxonomy_enriched_model()."""
    model_config = ConfigDict(populate_by_name=True)

    instance: str = ""
    node: str = ""
    semantic_category: Literal["identity", "attribute", "function", "state", "evaluation", "relation"] = "attribute"
    category_label: str = ""
    root: str = ""

    @field_validator('instance', 'node', 'category_label', 'root', mode='before')
    @classmethod
    def normalize_field(cls, v: str) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            return str(v).strip().lower()
        v = v.strip().lower().rstrip('.,;:!?')
        return v


class TaxonomyEnrichedIdeaResponse(BaseModel):
    """Response model for taxonomy-enriched idea extraction."""
    # Class variables for axis-aware validation
    _template_prefix: ClassVar[str] = ""
    _allowed_concepts: ClassVar[list] = []
    _excluded_concepts: ClassVar[list] = []
    _axis: ClassVar[str] = ""
    _node_instruction: ClassVar[str] = ""
    _dimension_marker: ClassVar[str] = "[ACTIONABLE_TAXONOMY_DIMENSION]"

    @classmethod
    def set_template_prefix(cls, prefix: str):
        """Set template prefix before LLM call for validation."""
        cls._template_prefix = prefix.strip() if prefix else ""

    @classmethod
    def set_axis_context(cls, axis: str, axis_data: dict, dimension_marker: str = ""):
        """Inject axis-specific rules for post-parse validation."""
        cls._axis = axis
        cls._allowed_concepts = axis_data.get("allowed_concepts", [])
        cls._excluded_concepts = axis_data.get("excluded_concepts", [])
        cls._node_instruction = axis_data.get("prompt_rules", {}).get("node_instruction", "")
        if dimension_marker:
            cls._dimension_marker = dimension_marker

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
        description="Semantic taxonomy: instance → node → semantic_category (category_label) → root"
    )

    @field_validator('idea', mode='before')
    @classmethod
    def enforce_template_prefix(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            return v or ""

        v = v.strip()
        prefix = cls._template_prefix

        if not prefix:
            return v  # No prefix to enforce

        # Check if already compliant (case-insensitive)
        if v.lower().startswith(prefix.lower()):
            pass  # prefix OK
        else:
            # Fix: prepend template prefix
            v = f"{prefix} {v}"

        # Check that marker token was replaced (should not appear in final idea)
        if cls._dimension_marker and cls._dimension_marker in v:
            import logging
            logging.getLogger(__name__).warning(
                f"Marker token not replaced in idea: {v[:80]}..."
            )

        return v


def _build_type_glossary() -> str | None:
    """Build a noun_like_phrase glossary string for schema-level description.

    Reads type_system from TEMPLATE_LOOKUP. Returns a string like:
      "noun_like_phrase = noun_phrase (...) | gerund_nominal (...) | ..."
    or None if no aliases exist.
    """
    from experiments.step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
    ts = TEMPLATE_LOOKUP.get("type_system", {})
    aliases = ts.get("aliases", {})
    definitions = ts.get("definitions", {})
    if not aliases:
        return None
    parts = []
    for alias_name, concrete_types in aliases.items():
        entries = []
        for t in concrete_types:
            defn = definitions.get(t, t)
            entries.append(f"{t}: {defn}")
        parts.append(f"{alias_name} = {' | '.join(entries)}")
    return " ".join(parts) if parts else None


# Axis-specific Dutch examples for semantic taxonomy fields.
# These reach the LLM via JSON Schema (unlike base class examples which get overridden).
_TAXONOMY_EXAMPLES = {
    "WHAT": {
        "instance": ["goede service", "te dure producten", "modern design"],
        "node": ["klantenservice", "prijsniveau", "productontwerp"],
        "semantic_category": ["attribute", "evaluation", "attribute"],
        "category_label": ["servicekwaliteit", "prijsbeoordeling", "vormgeving"],
        "root": ["klanttevredenheid"],
    },
    "HOW": {
        "instance": ["meer personeel inzetten", "sneller reageren"],
        "node": ["personeelsbezetting", "reactiesnelheid"],
        "semantic_category": ["function", "function"],
        "category_label": ["capaciteitsverbetering", "communicatiesnelheid"],
        "root": ["dienstverlening"],
    },
    "WHY": {
        "instance": ["te lang wachten", "geen alternatief"],
        "node": ["wachttijd", "beperkt aanbod"],
        "semantic_category": ["state", "state"],
        "category_label": ["wachtervaring", "marktbeperking"],
        "root": ["klanttevredenheid"],
    },
    "WHO": {
        "instance": ["oudere klanten", "nieuw personeel"],
        "node": ["senioren", "medewerkers"],
        "semantic_category": ["identity", "identity"],
        "category_label": ["leeftijdsgroep", "personeelstype"],
        "root": ["betrokken partijen"],
    },
    "WHERE": {
        "instance": ["op de website", "in de winkel"],
        "node": ["website", "fysieke winkel"],
        "semantic_category": ["identity", "identity"],
        "category_label": ["digitaal kanaal", "fysieke locatie"],
        "root": ["contactkanalen"],
    },
    "WHEN": {
        "instance": ["in het weekend", "tijdens piekuren"],
        "node": ["weekendperiode", "piekbelasting"],
        "semantic_category": ["identity", "state"],
        "category_label": ["weekpatroon", "capaciteitsdruk"],
        "root": ["tijdspatronen"],
    },
}


def create_taxonomy_enriched_model(axis: str, axis_data: dict, schema_data: dict = None, slot_type_map: dict = None):
    """Create axis-specific TaxonomyEnrichedIdeaResponse with tailored Field descriptions."""
    prompt_rules = axis_data.get("prompt_rules", {})
    schema_data = schema_data or {}
    slot_type_map = slot_type_map or {}

    # Build type-enriched slot description for the idea field
    slots = schema_data.get("slots", {})
    slot_names = " + ".join(slots.keys()) if slots else "axis-specific content"

    # Enrich node description with dimension type constraint
    node_desc = prompt_rules.get(
        "node_instruction",
        "Canonical, reusable concept label (noun phrase)"
    )
    dimension_type = slot_type_map.get("dimension", {})
    if dimension_type.get("is_alias"):
        node_desc += f" Must be a {dimension_type['type_name']}."

    # Type glossary for model-level description (definitions appear once, fields reference by name)
    glossary = _build_type_glossary()
    taxonomy_config = ConfigDict(json_schema_extra={"description": glossary}) if glossary else None

    # Axis-specific examples (these actually reach the LLM via JSON Schema)
    ex = _TAXONOMY_EXAMPLES.get(axis, {})

    # Create axis-specific SemanticTaxonomyResponse with tailored field descriptions
    # Using create_model() so $defs key and title match in JSON Schema
    config_kwargs = {"__config__": taxonomy_config} if taxonomy_config else {}
    AxisSemanticTaxonomyResponse = create_model(
        f"SemanticTaxonomyResponse_{axis}",
        __base__=SemanticTaxonomyResponse,
        **config_kwargs,
        instance=(str, Field(
            description=prompt_rules.get(
                "instance_instruction",
                "Contiguous verbatim span from the original response (no rewording)"
            ),
            examples=ex.get("instance", [])
        )),
        node=(str, Field(
            description=node_desc,
            examples=ex.get("node", [])
        )),
        semantic_category=(
            Literal["identity", "attribute", "function", "state", "evaluation", "relation"],
            Field(
                description=f"Semantic category for the {axis} dimension. One of: identity, attribute, function, state, evaluation, relation.",
                examples=ex.get("semantic_category", [])
            )
        ),
        category_label=(str, Field(
            description=prompt_rules.get(
                "category_label_instruction",
                "Concise descriptive label for the specific classification within the semantic category"
            ),
            examples=ex.get("category_label", [])
        )),
        root=(str, Field(
            default="",
            description=prompt_rules.get(
                "root_instruction",
                "Top-level domain framing implied by the survey question (optional)"
            ),
            examples=ex.get("root", [])
        )),
    )

    AxisTaxonomyEnrichedIdeaResponse = create_model(
        f"TaxonomyEnrichedIdeaResponse_{axis}",
        __base__=TaxonomyEnrichedIdeaResponse,
        idea=(str, Field(
            description=(
                f"Idea following the {axis} pattern: {axis_data['pattern']}. "
                f"Slot structure: {slot_names}. "
                f"Must begin with the canonical_phrasing template."
            )
        )),
        taxonomy=(Optional[AxisSemanticTaxonomyResponse], Field(
            default=None,
            description="Semantic taxonomy: instance → node → semantic_category (category_label) → root"
        )),
    )

    return AxisTaxonomyEnrichedIdeaResponse
