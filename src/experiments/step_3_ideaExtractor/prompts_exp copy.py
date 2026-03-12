from __future__ import annotations
from typing import ClassVar, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, create_model

try:
    from .facet_data import FacetDefinition, PromptRules, get_facets_in_decision_order
except ImportError:
    from experiments.step_3_ideaExtractor.facet_data import FacetDefinition, PromptRules, get_facets_in_decision_order


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
    """Build the primary facet decision tree prompt for a single chunk."""
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

Here is the survey question that was asked in {language}
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
- Defines the TYPE of content coders should extract, not examples of that content
- Must generalize to unseen responses — no specific attributes, examples, or verbatim response content

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
        description="Clear definition of the primary facet at proper abstraction level, specific to this survey question. Must NOT contain specific examples or attribute names from responses."
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

Here is the survey question that was asked in {language}:
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

Identify **mutually exclusive, collectively exhaustive thematic domains** that represent **distinct domains of relevance, impact, or meaning for {entity}**, as evidenced by the responses.

Use the fewest domains needed for full coverage — typically 5–15, but prefer fewer clean domains over more overlapping ones.

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
4. Aim for **fewest domains needed for full coverage** — enough to differentiate meaningfully, few enough to be analytically useful.
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
        description="fewest mutually exclusive thematic domains possible for full coverage from the responses"
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

Here is the survey question that was asked in {language}:
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

Here are the chunk-level concept type analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

## YOUR TASK

Consolidate these chunk-level thematic domain lists into the fewest mutually exclusive thematic domains needed for full coverage — typically 5–15, but prefer fewer clean domains over more overlapping ones.

## CONSOLIDATION RULES

1. **Merge semantically equivalent domains** — if multiple chunks produced similar domains (e.g. "access and logistics" and "service accessibility"), merge them into one
2. **Preserve robust distinctions** — if a domain appears consistently across chunks, it reflects a real pattern in the data; keep it
3. **Account for every chunk-level domain** — each domain from every chunk must be explicitly accounted for: kept as-is, merged into a semantically related domain, or flagged as a chunk-specific artifact (e.g., driven by an unusual cluster of responses rather than a genuine structural aspect of {entity}). Never silently discard a domain.
4. **Prefer precision over breadth, but respect boundaries** — create domains specific enough to be analytically useful. Only broaden a domain when two chunk-level domains genuinely describe the same aspect. Do not create domains so narrow that their boundaries with neighbors become ambiguous.
5. **Each domain must describe a thematic ASPECT of {entity}** — think "section headers in a research report about {entity}." Reject linguistic role labels like "moral attribute" or "functional trait."

Verify your final set is mutually exclusive and collectively exhaustive before submitting.

All output values (labels, definitions) must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""

class ConceptTypeConsolidatedResponse(BaseModel):
    """Consolidated thematic domains after merging all chunks."""
    concept_types: List[ConceptTypeItem] = Field(
        description="Fewest mutually exclusive thematic domains needed for full coverage, consolidated from all chunks"
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: Idea Extraction (dynamic model)
# ═══════════════════════════════════════════════════════════════════════

def _format_facet_examples(facet: FacetDefinition) -> str:
    """Format facet examples for the extraction prompt."""
    if not facet.examples:
        return ""
    lines = []
    for ex in facet.examples:
        lines.append(f"**{ex.survey_context}**")
        lines.append(f'Response: "{ex.response}"')
        lines.append(f"  instance: {ex.instance}")
        lines.append(f"  concept_type: {ex.concept_type}")
        lines.append(f"  rung_1: {ex.rung_1} (what does \"{ex.instance}\" mean in context?)")
        lines.append(f"  rung_2: {ex.rung_2} (what broader phenomenon does this point to?)")
        lines.append(f"  valence: {ex.valence}")
        lines.append("")
    return "\n".join(lines)


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
) -> str:
    """Build the taxonomy-enriched idea extraction prompt."""
    examples_block = _format_facet_examples(facet)

    return f"""You are an expert in extracting structured ideas from survey responses.

<survey_context>
Language: {language}
Question ({language}): "{var_lab}"
Respondent type: {perspective}
Domain: {domain}
Entity: {entity}
Topic: {topic}
Intent: {intent}
</survey_context>

<taxonomy_lens>
Lens: "{facet.noun_phrase_descriptor}"
Anchor: {facet.dimension_marker}
</taxonomy_lens>

<response>
Respondent ID: {respondent_id}
Response: {response}
</response>

---

## STEP 1 — SPLIT INTO ATOMIC IDEAS

Extract ALL distinct ideas from the response. When in doubt → SPLIT. Over-splitting is preferred.

{facet.instruction}

SPLITTING RULES (NON-NEGOTIABLE):
- Items joined by conjunctions ("and", "or", "en", "und", "et", "y", "ou") or commas that express DIFFERENT concepts → SPLIT into separate ideas
- Example: "faster and cheaper" → TWO ideas: (1) "faster", (2) "cheaper"
- Each idea gets its own canonical phrasing, classification, and valence

---

## STEP 2 — REFORMULATE EACH IDEA

For each idea, produce an idea statement using EXACTLY this pattern:

{canonical_phrasing}

- Do NOT alter the template prefix
- Replace {facet.dimension_marker} with the SHORTEST verbatim span from the response that expresses the idea
- Do NOT include the literal marker token in output
- Use respondent_id: {respondent_id}
- Use {language}, preserve original meaning

---

## STEP 3 — CLASSIFY AND LADDER

For each idea, perform these steps (all in {language}):

### A. CONCEPT TYPE (classify)
Assign the idea to the single best-fitting thematic domain from the table below. When it could fit multiple domains, choose the one that best captures the primary aspect of {entity} the idea relates to.
- {concept_type_table}

### B. ABSTRACTION LADDER (ladder UP from the instance — 2 rungs)
Build a 2-rung ladder of increasing abstraction from the instance:
- **rung_1**: What does this instance MEAN in context? Name the concrete phenomenon or interpretation.
- **rung_2**: What BROADER significance or higher-level theme does this point to?
Each rung must be more abstract than the previous. Do not repeat the instance or concept_type.

### C. VALENCE (direction of instance relative to concept type)
- "+" = the instance strengthens or reinforces this domain
- "-" = the instance weakens or undermines this domain
- "0" = no directional effect on this domain
- Valence is NOT sentiment or desirability

### Examples (study the BOTTOM-UP ladder — your output must be in {language})

{examples_block}Empty, irrelevant, or nonsensical response → return [].

Begin processing now and provide your output as valid JSON following the response schema provided.
"""


class SemanticTaxonomyResponse(BaseModel):
    """Abstraction ladder base class. Fields are overridden by create_extraction_model()."""

    instance: str = ""
    rung_1: str = ""
    rung_2: str = ""
    concept_type: str = ""

    @field_validator('instance', 'rung_1', 'rung_2', mode='before')
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
        description="Abstraction ladder: instance -> rung_1 -> rung_2 (bottom-up) + concept_type"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="Directional effect of the instance on the concept type: +, -, or 0"
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

    rung_1_desc = prompt_rules.rung_1_instruction
    rung_2_desc = prompt_rules.rung_2_instruction

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
        rung_1=(str, Field(
            description=rung_1_desc,
        )),
        rung_2=(str, Field(
            description=rung_2_desc,
        )),
        concept_type=concept_type_field,
    )

    # Add fuzzy-match validator for concept_type (runs before Literal validation)
    _key_map = {k.lower(): k for k in allowed_keys} if concept_types else {}
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
            return stripped

    FacetTaxonomy.__name__ = f"SemTax_{facet_key}"
    FacetTaxonomy.__qualname__ = f"SemTax_{facet_key}"

    # Create facet-specific extraction model with strict validators
    class FacetExtractionModel(TaxonomyEnrichedIdeaResponse):
        _template_prefix: ClassVar[str] = _prefix
        _dimension_marker: ClassVar[str] = _marker

        idea: str = Field(
            description="Complete idea statement beginning with the canonical_phrasing template"
        )
        abstraction_ladder: Optional[FacetTaxonomy] = Field(
            default=None,
            description="Abstraction ladder: instance -> rung_1 -> rung_2 (bottom-up) + concept_type"
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

