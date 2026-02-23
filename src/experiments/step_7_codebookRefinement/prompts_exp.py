"""
Experimental Prompts for Step 7: Codebook Refinement

This file contains the prompts used by codebookRefinement.py.
Modify these prompts to experiment with different codebook refinement approaches.

Original source: src/prompts.py (STEP 7: THEME ORGANIZATION section)
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# PRE-MECE PARTITION REVIEW (evaluates partition coherence before refinement)
# =============================================================================

PARTITION_REVIEW_PROMPT = """You are a qualitative research methodologist evaluating a group of survey codes.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
</survey_context>

<partition_to_review>
Partition name: "{partition_name}"
Number of codes: {n_codes}

Codes in this partition:
{codes_list}
</partition_to_review>

<peer_partitions>
Other partitions in this codebook (for context only):
{peer_partitions_list}
</peer_partitions>

<task>
Examine the {n_codes} codes above. Your task is to identify what structural domain(s) they belong to.

Step 1 — DESCRIBE: For each distinct structural domain you find among these codes, formulate:
  - A domain name: a concise noun phrase capturing the organizing concept
  - A domain description: 1-2 sentences explaining what this domain covers

Step 2 — COUNT: How many distinct structural domains did you identify?

Step 3 — DECIDE:
  - If ONE domain: action = "keep". Provide the domain_name and domain_description.
  - If TWO or more domains: action = "split". Assign each code (by ID) to its domain.

Guidelines:
- Ignore the partition name — it is a legacy label and may be misleading. Analyze the CODES.
- Different facets of the same domain are ONE domain, not multiple. A domain about "facilities" can include food, seating, and power outlets.
- Codes about the same topic but with opposite valence (e.g., "Short wait times (+)" vs "Long wait times (-)") belong to the SAME structural domain.
- The domain_name should clearly express the organizing concept. Do NOT simply reuse the partition name if a better name exists.

Example: Codes about cars, trains, and bicycles → ONE domain (domain_name: "transportation modes"). Codes about cars AND cooking recipes → TWO domains (transportation ≠ cooking).

CRITICAL RULES:
1. When splitting: every code ID must appear in exactly one sub-partition (no dropping, no duplicating)
2. When splitting: each sub-partition should have at least 2 codes when possible
3. All domain names must be concise noun phrases
4. All output text (domain names, descriptions, analysis) MUST be in {language}

Write all output in {language}. Provide output as valid JSON following the response schema provided.
</task>
"""


class PartitionSplitGroup(BaseModel):
    """A proposed sub-domain from splitting a partition with multiple domains."""
    new_partition_name: str = Field(
        ...,
        description="Concise noun phrase name for this structural domain"
    )
    domain_description: str = Field(
        ...,
        description="What this structural domain covers (1-2 sentences)"
    )
    code_ids: List[str] = Field(
        ...,
        description="List of code IDs that belong to this domain"
    )
    rationale: str = Field(
        ...,
        description="Why these codes form a coherent domain"
    )


class PartitionReviewResult(BaseModel):
    """Result of reviewing one partition for structural domain coherence."""
    partition_name: str = Field(
        ...,
        description="The original partition name being reviewed"
    )
    action: str = Field(
        ...,
        description="'keep' if one domain found, 'split' if multiple domains found"
    )
    domain_name: str = Field(
        default="",
        description="The identified structural domain name (when action='keep')"
    )
    domain_description: str = Field(
        default="",
        description="What this domain covers, 1-2 sentences (when action='keep')"
    )
    splits: List[PartitionSplitGroup] = Field(
        default_factory=list,
        description="Sub-domains with code assignments (when action='split')"
    )
    review_rationale: str = Field(
        ...,
        description="Analysis of structural domain(s) found in this partition"
    )


# =============================================================================
# PARTITION-FIRST REFINEMENT (replaces MAP-REDUCE + separate MECE)
# =============================================================================

PARTITION_REFINEMENT_PROMPT = """You are a codebook designer refining codes within a single structural domain for a survey codebook.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
</survey_context>

<partition_context>
Structural domain: "{partition_name}"
This domain contains {n_codes} codes.

Peer domains (codes that exist elsewhere — do NOT overlap with these):
{peer_partitions_list}
</partition_context>

<input_codes>
{codes_with_context}
</input_codes>

<task>
You have {n_codes} codes within "{partition_name}". Your tasks:

1. DOMAIN LABELING: Create a concise theme_label (<=10 words, noun phrase) and theme_description (1-2 sentences) that names this structural domain for the codebook.

2. CODE REFINEMENT:
   - Review each code's label and definition
   - Relabel codes so they read as proper survey codes (concise noun phrases, <=10 words)
   - Ensure definitions are clear and <=20 words
   - Only MERGE codes that are TRUE DUPLICATES (identical meaning, not just related)
   - When in doubt: KEEP SEPARATE

3. MECE ENFORCEMENT — for each code, define:
   - inclusion_examples: 3-5 concrete respondent ideas that belong here
   - exclusion_examples: 2-3 ideas that seem related but don't belong
   - near_neighbor_label: the most similar other code within this partition
   - tell_apart_rule: how to distinguish from the near neighbor
   - boundary_test: a self-contained yes/no question (no references to other codes)
   - diagnostic_signals: 3-5 concrete words/phrases that trigger assignment

4. PAIR VERIFICATION: For each pair of similar codes, construct one ambiguous example and show which code gets it and why.

5. Report any MECE issues that could not be resolved.
</task>

<design_rules>
- VALENCE-AWARE: code labels end with a valence suffix: (+) for positive, (-) for negative, or (0) for neutral/mixed. Preserve these suffixes exactly when relabeling. Never combine suffixes like (+,-).
- POSITIVE CRITERIA: define codes by what they ARE, not by what they're NOT
- INDEPENDENT: each boundary_test must work without knowing other codes exist
- PRESERVE: output should have approximately {n_codes} codes (merging should be rare)
- All output text (labels, definitions, examples, signals, descriptions) MUST be in {language}
</design_rules>

Write all output in {language}. Provide output as valid JSON following the response schema provided.
"""


CROSS_PARTITION_JUDGE_PROMPT = """You are a codebook quality judge verifying that codes across different structural domains do not overlap.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
</survey_context>

<codebook>
{codebook_summary}
</codebook>

<task>
Review the {total_codes} codes across {n_partitions} structural domains above. Your task:

1. Identify any code PAIRS from DIFFERENT partitions where a respondent's answer could plausibly be assigned to either code.
2. For each conflict found, specify:
   - Which two codes overlap and from which partitions
   - What kind of respondent answer would be ambiguous
   - Which code should "win" the ambiguous case and why
   - Whether this is minor (rare edge case) or major (systematic overlap)
3. Provide an overall assessment of the codebook's cross-partition MECE compliance.

IMPORTANT: When specifying partition names in partition_a and partition_b, use the EXACT partition key
from the "=== Partition: <key> ===" headers — not the theme label.

Focus on ACTIONABLE conflicts only — do not flag codes that are merely topically related but clearly distinguishable.

Write all output in {language}. Provide output as valid JSON following the response schema provided.
</task>
"""


# =============================================================================
# PARTITION-FIRST RESPONSE MODELS
# =============================================================================

class PartitionRefinementCodeEntry(BaseModel):
    """A single refined code with MECE assignment instructions."""
    code: str = Field(
        ...,
        description="Code label — concise noun phrase (<=10 words)"
    )
    definition: str = Field(
        ...,
        description="Code definition (<=20 words)"
    )
    inclusion_examples: List[str] = Field(
        ...,
        description="3-5 concrete examples of ideas that belong to this code"
    )
    exclusion_examples: List[str] = Field(
        ...,
        description="2-3 examples of ideas that do NOT belong but might seem like they do"
    )
    near_neighbor_label: str = Field(
        ...,
        description="The most similar other code that a coder might confuse with this one"
    )
    tell_apart_rule: str = Field(
        ...,
        description="How to distinguish this code from its near neighbor"
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a human coder asks to determine if an idea belongs here. "
            "Must be self-contained — no references to other codes."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description="3-5 concrete words, phrases, or framings that trigger assignment to this code"
    )
    source_code_ids: str = Field(
        ...,
        description="Original code ID(s) from input — comma-separated if merged"
    )


class PartitionPairVerification(BaseModel):
    """Verification that two similar codes are distinguishable."""
    code_a: str = Field(
        ...,
        description="First code in the pair"
    )
    code_b: str = Field(
        ...,
        description="Second code in the pair"
    )
    ambiguous_example: str = Field(
        ...,
        description="A constructed idea that could plausibly fit either code"
    )
    assigned_to: str = Field(
        ...,
        description="Which code the ambiguous example is assigned to"
    )
    reasoning: str = Field(
        ...,
        description="Why this assignment is correct, using only boundary_test and diagnostic_signals"
    )


class PartitionRefinementResult(BaseModel):
    """Complete result for one concept_type partition."""
    theme_label: str = Field(
        ...,
        description="Concise theme label for this structural domain (<=10 words, noun phrase)"
    )
    theme_description: str = Field(
        ...,
        description="What this domain covers (1-2 sentences)"
    )
    codes: List[PartitionRefinementCodeEntry] = Field(
        ...,
        description="Refined codes with MECE-verified assignment instructions"
    )
    verifications: List[PartitionPairVerification] = Field(
        ...,
        description="Self-verification tests: one per pair of similar codes"
    )
    refinement_analysis: str = Field(
        ...,
        description="Brief analysis of refinement decisions made"
    )
    mece_issues: List[str] = Field(
        default_factory=list,
        description="Any MECE violations or unresolvable overlaps found"
    )


class CrossPartitionConflict(BaseModel):
    """A detected cross-partition overlap."""
    code_a: str = Field(
        ...,
        description="Code from partition A"
    )
    partition_a: str = Field(
        ...,
        description="Partition key of code A — use the exact partition key from the '=== Partition: ... ===' header"
    )
    code_b: str = Field(
        ...,
        description="Code from partition B"
    )
    partition_b: str = Field(
        ...,
        description="Partition key of code B — use the exact partition key from the '=== Partition: ... ===' header"
    )
    overlap_description: str = Field(
        ...,
        description="What kind of respondent answer would be ambiguous between these two codes"
    )
    resolution: str = Field(
        ...,
        description="Which code should handle the overlap case, and why"
    )
    severity: str = Field(
        ...,
        description="'minor' (rare edge case) or 'major' (systematic overlap)"
    )


class CrossPartitionJudgeResult(BaseModel):
    """Result of cross-partition MECE verification."""
    conflicts: List[CrossPartitionConflict] = Field(
        ...,
        description="Cross-partition overlaps detected"
    )
    overall_assessment: str = Field(
        ...,
        description="Overall assessment of codebook's cross-partition MECE compliance"
    )
    is_mece_compliant: bool = Field(
        ...,
        description="Whether the codebook passes cross-partition MECE verification"
    )


# =============================================================================
# CROSS-PARTITION CONFLICT RESOLUTION
# =============================================================================

class ConflictResolutionAction(BaseModel):
    """Resolution for a single cross-partition conflict."""
    conflict_index: int = Field(
        ...,
        description="0-based index of the conflict being resolved"
    )
    action: str = Field(
        ...,
        description="'merge' for major overlaps (drop one code), 'sharpen' for minor overlaps (update boundaries)"
    )
    # For 'merge': which code survives
    surviving_code: Optional[str] = Field(
        None,
        description="Code label to keep (merge only)"
    )
    surviving_partition: Optional[str] = Field(
        None,
        description="Partition key of surviving code (merge only) — use exact partition key from conflict"
    )
    dropped_code: Optional[str] = Field(
        None,
        description="Code label to remove (merge only)"
    )
    dropped_partition: Optional[str] = Field(
        None,
        description="Partition key of dropped code (merge only) — use exact partition key from conflict"
    )
    merge_rationale: Optional[str] = Field(
        None,
        description="Why the surviving code wins (merge only)"
    )
    # For 'sharpen': updated fields on both codes
    code_a_updates: Optional[Dict[str, str]] = Field(
        None,
        description="Updated fields for code_a: boundary_test, tell_apart_rule (sharpen only)"
    )
    code_a_new_exclusions: Optional[List[str]] = Field(
        None,
        description="1-2 new exclusion examples to append to code_a (sharpen only)"
    )
    code_b_updates: Optional[Dict[str, str]] = Field(
        None,
        description="Updated fields for code_b: boundary_test, tell_apart_rule (sharpen only)"
    )
    code_b_new_exclusions: Optional[List[str]] = Field(
        None,
        description="1-2 new exclusion examples to append to code_b (sharpen only)"
    )
    sharpen_rationale: Optional[str] = Field(
        None,
        description="How the boundaries were clarified (sharpen only)"
    )


class ConflictResolutionResult(BaseModel):
    """Result of resolving all cross-partition conflicts."""
    resolutions: List[ConflictResolutionAction] = Field(
        ...,
        description="One resolution per conflict"
    )
    summary: str = Field(
        ...,
        description="Summary of all resolutions applied"
    )


CROSS_PARTITION_RESOLVE_PROMPT = """You are a codebook quality engineer resolving cross-partition overlaps in a qualitative survey codebook.

<context>
Survey question: "{survey_question}"
Language: {language}
</context>

The cross-partition MECE judge found {n_conflicts} conflicts between codes in different partitions. Your task is to resolve each one.

<conflicts>
{conflicts_formatted}
</conflicts>

<task>
For each conflict, choose one resolution strategy:

**MERGE** (for major overlaps / duplicates):
- Choose which code survives and which is dropped
- The surviving code's partition keeps it; the dropped code is removed entirely
- Use merge when both codes cover essentially the same concept

**SHARPEN** (for minor overlaps / fuzzy boundaries):
- Update boundary_test on BOTH codes to explicitly exclude the other's domain
- Add 1-2 concrete exclusion_examples to each code referencing the other's typical cases
- Update tell_apart_rule on both codes to reference the cross-partition neighbor

Rules:
- Major severity conflicts should typically use MERGE
- Minor severity conflicts should typically use SHARPEN
- All updated text must be in {language}
- boundary_test must remain a self-contained yes/no question
- exclusion_examples should be concrete respondent-like phrases (as if a survey respondent said them)
- tell_apart_rule should clearly state which partition handles which aspect

Provide output as valid JSON following the response schema provided.
</task>
"""


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

If ANY doubt exists -> KEEP SEPARATE

# Structure and Hierarchy
Organize codes into a 2-level or 3-level hierarchy:

**2-level (Theme -> Code)**: Use when themes are simple and codes don't need sub-grouping
**3-level (Theme -> Category -> Code)**: Use when a theme contains multiple sub-concepts that benefit from grouping

Guidelines:
- Every code must belong to exactly one theme
- Themes should be conceptually coherent (related codes grouped together)
- Use 3-level hierarchy when >=3 codes share a clear sub-concept within a theme
- Aim for 5-15 themes depending on codebook size

# Theme and Code Naming

**Theme Labels**
- <= 10 words, noun phrases preferred
- Describe the conceptual domain (e.g., "Duurzaamheid", "Klantenservice", "Prijsperceptie")
- No conjunctions or slashes

**Code Labels**
- Keep original code labels unless they violate naming rules
- <= 10 words, specific and atomic
- Preserve the valence indicator (+), (-), or (0) at the end of each code label. Never combine like (+,-)

**Code Descriptions**
- <= 20 words
- Define what belongs in this code
- Use patterns like: "Mentions of...", "References to..."

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
          "description": "<= 20 words explanation",
          "category": ""  // Empty for 2-level, category name for 3-level
        }}
      ]
    }}
  ]
}}

Notes:
- The number of codes in output should be close to the number of input codes (merging should be rare)
- No commentary before or after JSON
- All text (theme labels, code labels, descriptions, analysis) MUST be in {language}

Write all output in {language}. Begin organizing the codebook.
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

When in doubt -> KEEP SEPARATE

# Consolidation Steps
1. Identify TRUE duplicates across codebooks (codes with identical meaning)
2. Keep all unique codes
3. Organize all codes into a unified theme structure
4. Use 2-level or 3-level hierarchy as appropriate

# Theme Structure
**2-level (Theme -> Code)**: Simple organization
**3-level (Theme -> Category -> Code)**: Use when themes have clear sub-groupings

Guidelines:
- Merge similar THEMES across codebooks (organizational labels), but preserve the CODES within them
- Every code must appear exactly once in the final codebook
- Aim for 5-15 themes depending on total code count

# Label Rules
- Theme labels: <=10 words, noun phrases, no conjunctions/slashes
- Code labels: Keep original labels, <=10 words. Preserve the valence indicator (+), (-), or (0) at the end of each code label. Never combine like (+,-).
- Descriptions: <=30 words, define when to use the code

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
          "description": "Code definition (<=30 words)",
          "category": ""  // Empty for 2-level, category name for 3-level
        }}
      ]
    }}
  ]
}}

IMPORTANT: The total number of unique codes in your output should be close to the total unique codes across all input codebooks. Significant reduction indicates over-merging.

Write all output in {language}. Begin consolidating the codebooks.
"""


# =============================================================================
# STEP 7 MECE ENFORCEMENT (post-refinement)
# =============================================================================

CODEBOOK_MECE_ENFORCEMENT_PROMPT = """You are a codebook designer applying MECE (Mutually Exclusive, Collectively Exhaustive) constraints to a set of refined codes within a single concept-type partition.

Your output must be a DECISION SYSTEM — not a semantic description. Each code must be defined by criteria a human coder can independently apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
</survey_context>

<partition_context>
You are analyzing codes within the concept type: "{partition_name}"
{partition_description}

Peer partitions (do NOT include codes that belong to these):
{peer_partitions_list}

CRITICAL: Only retain codes that fall WITHIN "{partition_name}".
</partition_context>

<instruction>
You are given {n_codes} refined codes from the partition "{partition_name}". These codes have already been organized into themes. Your task is to ensure they are MECE:
- Mutually Exclusive: each survey response idea should clearly belong to exactly one code
- Collectively Exhaustive: every idea within this concept type should fit one code

You may suggest merges for codes that still overlap, but do NOT add or split codes. Focus on defining clear boundaries.

CRITICAL DESIGN RULE: Define each code using POSITIVE, INDEPENDENT criteria.
- DO NOT define a code by what it is NOT or by referencing other codes.
- Each code's boundary_test must work WITHOUT knowing what other codes exist.
- Use observable characteristics of the response text, not abstract semantic descriptions.
</instruction>

<principles>
1. The result must be a clean, non-overlapping set of codes WITHIN "{partition_name}".
2. Each code needs:
   - INCLUSION EXAMPLES: 3-5 concrete examples of ideas that belong to this code
   - EXCLUSION EXAMPLES: 2-3 examples of ideas that do NOT belong (but might seem like they do)
   - A BOUNDARY TEST: a yes/no question a coder can ask independently
   - DIAGNOSTIC SIGNALS: 3-5 concrete words/phrases that trigger assignment to this code
   - NEAR NEIGHBOR: the most similar other code that a coder might confuse with this one
   - TELL APART RULE: how to distinguish this code from its near neighbor
3. Codes are VALENCE-AWARE: code labels reflect the direction of ideas — (+) positive, (-) negative, (0) neutral/mixed. Preserve these suffixes exactly. Never combine like (+,-).
4. Every input code must appear in the output (no dropping codes).
</principles>

<label_constraints>
All output (labels, definitions, examples, signals) MUST be in {language}.
</label_constraints>

<codes>
{codes_list}
</codes>

<task>
1. Review the {n_codes} codes and their descriptions.
2. Check for overlaps: identify code pairs where a coder would hesitate.
3. For each code, define assignment instructions (inclusion_examples, exclusion_examples, boundary_test, diagnostic_signals, near_neighbor_label, tell_apart_rule).
4. VERIFY your codes are MECE by testing each pair of similar/adjacent codes:
   - Construct one AMBIGUOUS example (an idea that could plausibly fit either code)
   - Show which code gets it and WHY, using only your boundary_test and diagnostic_signals
   - If you cannot decide using your criteria alone, your codes are NOT MECE — flag this.
5. Report any MECE issues found.

Write all output in {language}. Provide output as valid JSON following the response schema provided.
</task>
"""


# =============================================================================
# MECE ENFORCEMENT RESPONSE MODELS
# =============================================================================

class MECECodeEntry(BaseModel):
    """A single code with MECE-verified assignment instructions."""
    code: str = Field(
        ...,
        description="Code label (must match a code from the input)"
    )
    theme: str = Field(
        ...,
        description="Theme this code belongs to"
    )
    inclusion_examples: List[str] = Field(
        ...,
        description="3-5 concrete examples of ideas that belong to this code"
    )
    exclusion_examples: List[str] = Field(
        ...,
        description="2-3 examples of ideas that do NOT belong but might seem like they do"
    )
    near_neighbor_label: str = Field(
        ...,
        description="The most similar other code that a coder might confuse with this one"
    )
    tell_apart_rule: str = Field(
        ...,
        description="How to distinguish this code from its near neighbor"
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a human coder asks to determine if an idea belongs here. "
            "Must be self-contained — no references to other codes."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description="3-5 concrete words, phrases, or framings that trigger assignment to this code"
    )


class MECEPairVerification(BaseModel):
    """Self-verification test for one pair of similar codes."""
    code_a: str = Field(
        ...,
        description="First code in the pair"
    )
    code_b: str = Field(
        ...,
        description="Second code in the pair"
    )
    ambiguous_example: str = Field(
        ...,
        description="A constructed idea that could plausibly fit either code"
    )
    assigned_to: str = Field(
        ...,
        description="Which code the ambiguous example is assigned to"
    )
    reasoning: str = Field(
        ...,
        description="Why this assignment is correct, using only boundary_test and diagnostic_signals"
    )


class MECEPartitionResult(BaseModel):
    """Complete MECE enforcement result for a single concept-type partition."""
    partition_name: str = Field(
        ...,
        description="The concept_type partition name"
    )
    codes: List[MECECodeEntry] = Field(
        ...,
        description="Codes with MECE-verified assignment instructions"
    )
    verifications: List[MECEPairVerification] = Field(
        ...,
        description=(
            "Self-verification tests: one per pair of similar codes. "
            "Proves the boundary criteria actually work."
        )
    )
    mece_issues: List[str] = Field(
        default_factory=list,
        description="Any MECE violations or unresolvable overlaps found"
    )

