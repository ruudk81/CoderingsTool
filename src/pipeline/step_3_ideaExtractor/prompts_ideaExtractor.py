from __future__ import annotations
from typing import ClassVar, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, create_model

try:
    from .dimension_data import DimensionDefinition, PromptRules, get_dimensions_in_decision_order
except ImportError:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition, PromptRules, get_dimensions_in_decision_order


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


# --- 1b. Group 2: Subject matter (sector / topic / entity) ---

def build_context_specifier_group2_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
) -> str:
    """Build the Group 2 context specifier prompt (sector/topic/entity)."""
    return f"""You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 2 specifiers (subject matter):

1. **sector**: Industry or sector
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
    sector: str = Field(
        description="Industry/sector the survey concerns",
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
- For **sector**: Standardize to lowercase, single/hyphenated word
- For **topic**: Choose the most representative subject matter across all chunks
- For **entity**: Standardize format (lowercase_with_underscores)

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 2 specifiers as valid JSON following the response schema provided."""


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: Primary Dimension Selection (Decision Tree)
# ═══════════════════════════════════════════════════════════════════════

# All 11 dimension keys for Literal type validation
_ALL_DIMENSION_KEYS = (
    "PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS", "IDENTITY_DEFINITION",
    "ACTORS_TARGETS", "CONTEXT_CONDITIONS", "MOTIVATIONS_DRIVERS",
    "EXPERIENCE_PERCEPTION", "EVALUATION_PRIORITIZATION",
    "BEHAVIOR_FUNCTION", "ATTRIBUTES_ASSOCIATIONS", "RELATIONS_DEPENDENCIES",
    "GENERAL_OTHER",
)


def _build_decision_tree_block() -> str:

    dimensions = get_dimensions_in_decision_order()
    emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟", "1️⃣1️⃣"]
    lines = []
    for dimension, emoji in zip(dimensions, emoji_numbers):
        signals = "\n".join(f"  • {s}" for s in dimension.criterion_signals)
        exclusions = "\n".join(f"  ✗ {e}" for e in dimension.exclusions)
        block = (
            f"{emoji} {dimension.key}\n"
            f"  {dimension.criterion}\n"
            f"  **Criterion signals**\n"
            f"{signals}\n"
        )
        if dimension.clarification:
            clarification = "\n".join(f"  • {c}" for c in dimension.clarification)
            block += f"  **Clarification**\n{clarification}\n"
        block += (
            f"  **Exclusions**\n"
            f"{exclusions}\n"
            f"  ➡ If YES → select {dimension.key}"
        )
        lines.append(block)
    return "\n\n".join(lines)


# --- 3a. Per-chunk dimension selection (decision tree) ---

def build_primary_dimension_decision_tree_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
    perspective: str,
    intent: str,
    sector: str,
    entity: str,
    topic: str,
) -> str:
    """Build the primary dimension decision tree prompt for a single chunk."""
    decision_tree = _build_decision_tree_block()

    return f"""You are selecting the SINGLE best primary dimension for organizing a set of open-ended responses.
Your task is not to summarize responses or label each one. Your task is to identify the main semantic axis along which the responses DIFFER.

Here is the language you will be working in:
<language>
{language}
</language>

Here is contextual information about the survey question:
<context>
- Sector: {sector}
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
"If I had to organize these responses into groups, which dimension would best explain the biggest, most meaningful differences between them?"

Choose the dimension that:
* Explains variation across most responses
* Creates the clearest top-level separation
* Would naturally be used as the first folder when organizing insights

If multiple dimensions seem plausible:
1. Choose the dimension that applies to a larger share of responses
2. If still tied, choose the dimension earlier in the decision order

------------------------------
DECISION TREE (Apply in Order, Stop at First Fit)
------------------------------

{decision_tree}

------------------------------
RULES
------------------------------
* Select exactly one dimension.
* Apply the decision tree steps in order (1 through 11). Stop at the FIRST step where the answer is clearly YES for the dominant variation.
* Base your decision on dominant variation, not edge cases.
* Dimensions are organizational lenses, not labels for individual responses.
* Evidence snippets must be copied verbatim from <sample_responses>.
* Clarification must contrast the chosen dimension with at least one plausible alternative in a single sentence.
* Write a primary_dimension_description by completing: "Responses vary in [what differentiates them] regarding [survey topic]." State only the abstract principle of variation — do not list or illustrate specific response content.


All string values (including evidence snippets) must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class PrimaryDimensionChunkResponse(BaseModel):
    """LLM response for single chunk primary dimension selection (decision tree)."""
    primary_dimension: Literal[
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
    ] = Field(
        description="The single best primary dimension for organizing responses"
    )
    decision_tree_stop_position: int = Field(
        description="Which decision tree step (1-11) triggered the selection",
        ge=1, le=11,
    )
    evidence: List[str] = Field(
        description="2-3 verbatim snippets from sample_responses supporting the chosen dimension",
        examples=[["good service", "too expensive", "friendly staff"]]
    )
    clarification: str = Field(
        description="One sentence: why this dimension over the most plausible alternative"
    )
    primary_dimension_description: str = Field(
        description="Scope statement completing: 'Responses vary in [what differentiates them] regarding [survey topic].' Abstract principle of variation only, no listed content or examples"
    )



# --- 3b. Dimension consolidation ---

def build_primary_dimension_consolidation_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    chunk_results: str,
    chunk_responses: str = "",
) -> str:
    """Build the primary dimension consolidation prompt."""
    dimension_keys_str = ", ".join(_ALL_DIMENSION_KEYS)

    sample_block = ""
    if chunk_responses:
        sample_block = f"""
Here is a random sample of actual survey responses for grounding your decision:
<sample_responses>
{chunk_responses}
</sample_responses>
"""

    return f"""You are consolidating multiple chunk-level primary dimension analyses into a single global primary dimension for a survey question.

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
- Sector: {sector}
- Entity of interest: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

Here are the chunk-level analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>
{sample_block}

------------------------------
RULES
------------------------------
* Select the ONE dimension ({dimension_keys_str}) that provides the clearest partition boundaries for coding responses.
* If chunks converge on the same dimension, follow the consensus.
* If chunks diverge, consult <sample_responses> directly to determine which dimension best explains the dominant variation in the actual response data.
* Write a primary_dimension_description by completing: "Responses vary in [what differentiates them] regarding [survey topic]." State only the abstract principle of variation — do not list or illustrate specific response content.

All output values must be in {language}.

Begin processing now and provide your output as valid JSON following the response schema provided."""


class PrimaryDimensionConsolidatedResponse(BaseModel):
    """Consolidated primary dimension selection after merging all chunks."""
    primary_dimension: str = Field(
        description="The selected primary dimension",
        examples=list(_ALL_DIMENSION_KEYS),
    )
    primary_dimension_rationale: str = Field(
        description="1-2 sentences: why this dimension is the dominant organizing principle"
    )
    primary_dimension_description: str = Field(
        description="Scope statement completing: 'Responses vary in [what differentiates them] regarding [survey topic].' Abstract principle of variation only, no listed content or examples"
    )



def consolidate_primary_dimension_by_majority(
    chunk_results: List[PrimaryDimensionChunkResponse],
) -> Optional[PrimaryDimensionConsolidatedResponse]:
    """Return consolidated result if clear majority (>50%), else None.

    When a single dimension is selected by more than half the chunks, we skip
    the LLM consolidation call and construct the result programmatically.
    Returns None when there is no majority — the caller should then run
    the consolidation LLM with actual response data for tie-breaking.
    """
    from collections import Counter

    if not chunk_results:
        return None

    dimension_counts = Counter(r.primary_dimension for r in chunk_results)
    total = len(chunk_results)
    winner, winner_count = dimension_counts.most_common(1)[0]

    if winner_count <= total / 2:
        return None  # No majority — caller should run LLM consolidation

    # Pick description from winning chunk with earliest decision tree stop position
    winning_chunks = [r for r in chunk_results if r.primary_dimension == winner]
    best_chunk = min(winning_chunks, key=lambda r: r.decision_tree_stop_position)

    return PrimaryDimensionConsolidatedResponse(
        primary_dimension=winner,
        primary_dimension_rationale=f"{winner_count} out of {total} chunks selected {winner}.",
        primary_dimension_description=best_chunk.primary_dimension_description,
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: Domain Discovery
# ═══════════════════════════════════════════════════════════════════════

# --- 4a. Per-chunk discovery ---

def build_domain_discovery_prompt(
    *,
    language: str,
    survey_question: str,
    chunk_responses: str,
    chunk_size: int,
    perspective: str,
    intent: str,
    sector: str,
    entity: str,
    topic: str,
    primary_dimension: str,
    primary_dimension_description: str,
    domain_diagnostic: str,
    domain_instruction: str,
) -> str:
    """Build the domain discovery prompt for a single chunk."""
    return f"""You are a qualitative research methodologist specializing in taxonomy development for survey analysis.
Your task is to identify domains within a given dimension based on survey response data.

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
- Sector: {sector}
- Entity of interest: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

The primary dimension selected for this dataset is:
<primary_dimension>
{primary_dimension}: {primary_dimension_description}
</primary_dimension>

Here is a representative sample of {chunk_size} verbatim responses:
<sample_responses>
{chunk_responses}
</sample_responses>

## TAXONOMY HIERARCHY

You need to understand the following four-level taxonomy structure:

**1. Dimension (L1)**
- Definition: A dimension identifies the type of information conveyed by a response. 

**2. Domain (L2)**
{domain_instruction}

**3. Facet (L3)**
- Definition: A facet identifies the analytical lens through which the domain is being examined. 

**4. Attribute (L4)**
- Definition: An attribute identifies the specific observable property, feature, or signal that the response refers to. 

## YOUR TASK

Your task is to identify **domains** (level 2 in the hierarchy) for the given dimension based on the survey responses provided.

{domain_instruction}
{domain_diagnostic}

Your goal is to identify the **fewest domains possible** that provide **full coverage** of all the responses. Each domain must be:
- **Ontologically distinct** — no two domains may share conceptual space. A domain must not be a subset of another domain, and two domains must not be two lenses on the same phenomenon.
- **Semantically distant** — a coder assigning a response to a domain must not plausibly consider a neighboring domain. No "could go either way" situations.
- Focused on ONE specific aspect (not a compound list of multiple concerns)
- A natural grouping of related phenomena within the dimension
- Strictly within the boundaries and through the lens of the primary dimension above

## CRITICAL REQUIREMENTS

- All labels and definitions in your JSON output must be in the language specified in the <language> tags, which is {language}
- The "key" field should be in English for technical consistency
- Domain definitions must NOT contain examples or enumerations — no "such as", "like", "zoals"
- Domains must be ontologically distinct and semantically distant — no shared conceptual space, no coder hesitation

Begin processing now and provide your output as **valid JSON** following the response schema provided."""


class DomainItem(BaseModel):
    """A single domain discovered from the data."""
    key: str = Field(
        description="Short identifier",
        examples=["access_and_logistics", "value_proposition", "hospitality_and_interaction"]
    )
    label: str = Field(
        description="Human-readable label in {language} (1-4 words)",
        examples=["Toegang en logistiek", "Waardepropositie", "Gastvrijheid en interactie"]
    )
    definition: str = Field(
        description="Short clarification of domain label in {language} (1 sentence). CRITICAL: One focused boundary, not a compound list, no examples or enumerations"
    )


class DomainChunkResponse(BaseModel):
    """LLM response for single chunk domain discovery."""
    domains: List[DomainItem] = Field(
        description="fewest mutually exclusive domains possible for full coverage from the responses"
    )


# --- 4b. Consolidation ---

def build_domain_consolidation_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    primary_dimension: str,
    chunk_results: str,
    domain_diagnostic: str,
) -> str:
    """Build the domain consolidation prompt."""
    return f"""You are a taxonomy consolidation specialist.
Your task is to merge multiple chunk-level domain analyses into a single, coherent set of domains.

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
- Sector: {sector}
- Entity of interest: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Question intent: {intent}
</context>

The primary dimension selected for this dataset is:
<primary_dimension>
{primary_dimension}
</primary_dimension>

The diagnostic question for domains within this dimension is:
<domain_diagnostic>
{domain_diagnostic}
</domain_diagnostic>

Here are the chunk-level domain analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

## YOUR TASK

Your task is to consolidate these chunk-level domain lists into the fewest mutually exclusive domains needed for full coverage.

Important consolidation principles:
- MERGE domains that have conceptual overlap, near-equivalence, or represent subcategories of a broader concept
- MERGE domains that are two lenses on the same phenomenon (e.g., "merkidentiteit" and "merkperceptie" both describe brand characteristics)
- ENSURE ontological distinctness: no two domains may share conceptual space. A domain must not be a subset of another.
- ENSURE semantic distance: a coder assigning a response must not plausibly hesitate between two domains. No "could go either way" situations.
- MAINTAIN full coverage: the consolidated domains must collectively cover all concepts present in the chunk-level analyses
- MINIMIZE the total number of domains while preserving meaningful distinctions
- All domains must stay strictly within the boundaries and through the lens of the primary dimension
- Each domain definition must complete: "This domain covers responses about [single aspect]." Abstract boundary only, no examples or enumerations
- All domain labels and definitions must be in the language specified above

Before providing your final output, use the scratchpad to work through your consolidation logic:

<scratchpad>
In your scratchpad:
1. List all unique domains that appear across the chunk-level analyses
2. Identify groups of domains that have conceptual overlap or proximity
3. For each group, determine an appropriate consolidated domain label and definition
4. For each pair of surviving domains, ask: "Could a response plausibly belong to both?" If yes, merge them.
5. Verify that your consolidated domains provide complete coverage of the original set
</scratchpad>

After completing your analysis in the scratchpad, provide your consolidated taxonomy as valid JSON inside <output> tags.

Begin processing now and provide your output as valid JSON following the response schema provided."""

class DomainConsolidatedResponse(BaseModel):
    """Consolidated domains after merging all chunks."""
    domains: List[DomainItem] = Field(
        description="Fewest mutually exclusive domains needed for full coverage, consolidated from all chunks"
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: Idea Extraction (dynamic model)
# ═══════════════════════════════════════════════════════════════════════

def _format_dimension_examples(dimension: DimensionDefinition) -> str:
    """Format dimension examples for the extraction prompt."""
    if not dimension.examples:
        return ""
    lines = []
    for ex in dimension.examples:
        lines.append(f"**{ex.survey_context}**")
        lines.append(f'Response: "{ex.response}"')
        lines.append(f"  instance: {ex.instance}")
        lines.append(f"  interpretation: {ex.interpretation}")
        lines.append(f"  abstraction: {ex.abstraction}")
        lines.append(f"  domain: {ex.domain}")
        lines.append(f"  valence: {ex.valence}")
        lines.append("")
    return "\n".join(lines)


def build_taxonomy_enriched_extraction_prompt(
    *,
    language: str,
    var_lab: str,
    perspective: str,
    sector: str,
    entity: str,
    topic: str,
    intent: str,
    response: str,
    canonical_phrasing: str,
    dimension: DimensionDefinition,
    domain_table: str,
) -> str:
    """Build the taxonomy-enriched idea extraction prompt."""
    examples_block = _format_dimension_examples(dimension)

    return f"""You are an expert in extracting structured ideas from survey responses.

<survey_context>
Language: {language}
Question ({language}): "{var_lab}"
Respondent type: {perspective}
Sector: {sector}
Entity: {entity}
Topic: {topic}
Intent: {intent}
</survey_context>

<taxonomy_lens>
Lens: "{dimension.noun_phrase_descriptor}"
Anchor: {dimension.domain_marker}
</taxonomy_lens>

<response>
{response}
</response>

---

## STEP 1 — SPLIT INTO ATOMIC IDEAS

Extract ALL distinct ideas from the response. When in doubt → SPLIT. Over-splitting is preferred.

{dimension.instruction}

SPLITTING RULES (NON-NEGOTIABLE):
- Items joined by conjunctions ("and", "or", "en", "und", "et", "y", "ou") or commas that express DIFFERENT concepts → SPLIT into separate ideas
- Example: "faster and cheaper" → TWO ideas: (1) "faster", (2) "cheaper"
- Each idea gets its own canonical phrasing, classification, and valence

---

## STEP 2 — IDEA STATEMENT

For each idea, extract the SHORTEST span from the response that captures it.

Your output replaces {dimension.domain_marker} in: {canonical_phrasing}

Rules:
- Output ONLY the span — do NOT include the prefix or the marker token
- Use {language}, preserve the respondent's original wording


---

## STEP 2.5 — ABSTRACTION LADDER (all in {language})

For each idea, build a 2-rung ladder of increasing abstraction from the instance:

### A. INTERPRETATION (what is the respondent REALLY talking about?)
{dimension.prompt_rules.interpretation_instruction}
- This requires INTERPRETATION, not just normalization
- Different surface expressions pointing to the same meaning → same concept

### B. ABSTRACTION (what BROADER significance does this point to?)
{dimension.prompt_rules.abstraction_instruction}
- Must be more abstract than the interpretation
- Do not repeat the instance or domain

---

## STEP 3 — CLASSIFY

For each idea, perform these steps (all in {language}):

### A. DOMAIN ({dimension.prompt_rules.domain_diagnostic})
Assign the idea to the single best-fitting domain from the table below. {dimension.prompt_rules.domain_instruction}
- {domain_table}

### B. VALENCE (direction of instance relative to domain)
- "+" = the instance strengthens or reinforces this domain
- "-" = the instance weakens or undermines this domain
- "0" = no directional effect on this domain
- Valence is NOT sentiment or desirability

### Examples (your output must be in {language})

{examples_block}Empty, irrelevant, or nonsensical response → return [].

Begin processing now and provide your output as valid JSON following the response schema provided.
"""


class SemanticTaxonomyResponse(BaseModel):
    """Taxonomy response base class. Fields are overridden by create_extraction_model().

    Abstraction ladder: instance → interpretation → abstraction (survey language).
    Domain: L2 thematic domain.
    Facet (L3) is assigned later in step 5, NOT here.

    Note: ladder validation is tier-aware and applied in create_extraction_model(),
    not here. See _make_ladder_validator().
    """

    instance: str = ""
    interpretation: str = ""
    abstraction: str = ""
    domain: str = ""


class TaxonomyEnrichedIdeaResponse(BaseModel):
    """Base model for extraction. Use create_extraction_model() for dimension-specific versions.

    Note: respondent_id and idea_id are NOT in the response model — they are
    known by the caller and assigned programmatically. This reduces output tokens
    and eliminates a source of validation failures on nano models.
    """
    idea: str = Field(
        description="The extracted idea — shortest verbatim span expressing the concept"
    )
    abstraction_ladder: Optional[SemanticTaxonomyResponse] = Field(
        default=None,
        description="Abstraction ladder (instance → interpretation → abstraction) + domain classification"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="Directional effect of the instance on the domain: +, -, or 0"
    )


def create_extraction_model(
    *,
    dimension: DimensionDefinition,
    template_prefix: str,
    domains: list[DomainItem] | None = None,
    model: str = "",
) -> type[TaxonomyEnrichedIdeaResponse]:
    """Create dimension-specific extraction model with tier-aware validation.

    template_prefix and domain_marker are baked in via class closure.
    No ClassVar. Each call returns a fresh class. Safe for async.

    Ladder validation is tier-aware:
    - nano: lenient — coerce None/empty to "", never reject (retries are futile)
    - mini/default: strict — reject None/empty (retries will succeed)
    """
    _strict_ladder = "nano" not in model
    _prefix = template_prefix.strip() if template_prefix else ""
    _marker = dimension.domain_marker
    prompt_rules = dimension.prompt_rules
    dimension_key = dimension.key

    # Build domain field — use label (survey language) not key (English)
    if domains:
        allowed_labels = tuple(c.label for c in domains) + ("Other",)
        domain_field = (
            Literal[allowed_labels],
            Field(
                description=(
                    "Domain — which aspect of the entity does this concept belong to? One of: " +
                    ", ".join(f"{c.label} ({c.definition})" for c in domains) +
                    ", Other (does not fit any of the above)"
                ),
                examples=[c.label for c in domains[:3]]
            )
        )
    else:
        domain_field = (
            str,
            Field(
                description=(
                    f"Domain: which ASPECT of the entity does this concept belong to? "
                    f"Use a short label (1-4 words) suitable for organizing a codebook section. "
                    f"NOT a linguistic role ('moral attribute', 'functional trait') but a thematic category "
                    f"('products and services', 'marketing and communication', 'social responsibility')."
                ),
            )
        )

    # Create base dimension-specific SemanticTaxonomyResponse
    _BaseDimensionTaxonomy = create_model(
        f"SemTax_{dimension_key}_b",
        __base__=SemanticTaxonomyResponse,
        instance=(str, Field(
            description=prompt_rules.instance_instruction,
        )),
        interpretation=(str, Field(
            description=prompt_rules.interpretation_instruction,
        )),
        abstraction=(str, Field(
            description=prompt_rules.abstraction_instruction,
        )),
        domain=domain_field,
    )

    # Add fuzzy-match validator for domain (runs before Literal validation)
    _label_map = {k.lower(): k for k in allowed_labels} if domains else {}
    if domains:
        _label_map.update({k.lower().replace(' ', '_'): k for k in allowed_labels})
        # Also map English keys to labels for robustness
        for c in domains:
            _label_map[c.key.lower()] = c.label
            _label_map[c.key.lower().replace('_', ' ')] = c.label

    class DimensionTaxonomy(_BaseDimensionTaxonomy):
        @field_validator('instance', 'interpretation', 'abstraction', mode='before')
        @classmethod
        def validate_ladder_field(cls, v: object) -> str:
            """Tier-aware ladder validation (closure over _strict_ladder).

            strict (mini/default): reject None/empty — retry will succeed.
            lenient (nano): coerce to "" — retry is futile.
            """
            if v is None:
                if _strict_ladder:
                    raise ValueError("Field must not be None. Provide a non-empty string value.")
                return ""
            if not isinstance(v, str):
                if _strict_ladder:
                    raise TypeError(f"Expected str, got {type(v).__name__}: {v!r}")
                return str(v).strip().lower().rstrip('.,;:!?')
            stripped = v.strip()
            if not stripped:
                if _strict_ladder:
                    raise ValueError("Field must not be empty after stripping whitespace.")
                return ""
            return stripped.lower().rstrip('.,;:!?')

        @field_validator('domain', mode='before')
        @classmethod
        def normalize_domain(cls, v: object) -> str:
            if not isinstance(v, str) or not _label_map:
                return v
            stripped = v.strip()
            # Exact match (case-insensitive)
            if stripped.lower() in _label_map:
                return _label_map[stripped.lower()]
            # Normalize: _ → space, & → and, collapse whitespace, strip trailing punctuation
            normalized = stripped.lower().replace('_', ' ').replace('&', 'and').replace('  ', ' ').rstrip('.,;:')
            if normalized in _label_map:
                return _label_map[normalized]
            return stripped

    DimensionTaxonomy.__name__ = f"SemTax_{dimension_key}"
    DimensionTaxonomy.__qualname__ = f"SemTax_{dimension_key}"

    # Create dimension-specific extraction model with strict validators
    class DimensionExtractionModel(TaxonomyEnrichedIdeaResponse):
        _template_prefix: ClassVar[str] = _prefix
        _domain_marker: ClassVar[str] = _marker

        idea: str = Field(
            description="Complete idea statement beginning with the canonical_phrasing template"
        )
        abstraction_ladder: Optional[DimensionTaxonomy] = Field(
            default=None,
            description="Abstraction ladder (instance → interpretation → abstraction) + domain classification"
        )

        @field_validator('idea', mode='before')
        @classmethod
        def validate_idea(cls, v: str) -> str:
            """Basic validation — non-empty string only."""
            if not isinstance(v, str) or not v.strip():
                raise ValueError("idea must be a non-empty string.")
            return v.strip()

    DimensionExtractionModel.__name__ = f"IdeaExtr_{dimension_key}"
    DimensionExtractionModel.__qualname__ = f"IdeaExtr_{dimension_key}"
    return DimensionExtractionModel
