"""
Experimental Prompts for Step 8: Code Assignment (Partition-Based, Ladder)

Two clean prompts for partition-based code assignment using the abstraction ladder:
- SINGLE_CODE_EVALUATION_PROMPT: Evaluate one code against an idea's abstraction ladder
- PARTITION_EVALUATION_PROMPT: Evaluate all codes from a partition against an idea's ladder

No fallback prompts — if no code matches, the idea gets unknown_label.

Post-assignment consolidation:
- AXIAL_CONSOLIDATION_PROMPT: Consolidate codes within a theme into higher-order abstractions
"""

# =============================================================================
# STEP 8: PARTITION-BASED CODE ASSIGNMENT (LADDER)
# =============================================================================

# Single code evaluation (partition has exactly 1 code)
SINGLE_CODE_EVALUATION_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses.
Your task is to determine if there is explicit or clearly paraphrased evidence that a specific code applies to the given response concept.

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response concept you need to analyze:
<response>
Idea ID: {idea_id}
Instance (verbatim respondent text): {instance}
Rung 1 (interpretation): {rung_1}
Concept Type: {concept_type}
</response>

Here is the code you need to evaluate:
<code_details>
Code: {code}
Definition: {definition}

Boundary test (primary focus check): {boundary_test}
Diagnostic signals (trigger words/phrases): {diagnostic_signals}

Inclusion Examples (valid references for this code):
    {inclusion_examples}

Routing redirects (ideas that belong to a neighboring code instead):
    {exclusion_examples}

Boundary: This code covers "{code}", which differs from "{near_neighbor_label}"
Routing rule: {tell_apart_rule}
</code_details>

**MANDATORY first: Identify the dominant perspective**
Before applying any decision rule, determine: what is the PRIMARY focus of this idea — is it describing a CAUSE/DRIVER (facility, stimulus, actor) or an EFFECT/OUTCOME (experience, feeling, result)? This determination constrains your evaluation.

Follow these DECISION RULES strictly:

1) Primary focus alignment
   * Apply the boundary test: "{boundary_test}"
   * Compare the answer to your dominant perspective determination above.
   * If the boundary test aligns with the dominant perspective → strong evidence for assignment.
   * If the answer is clearly NO (the primary focus is definitively elsewhere) → strong evidence against — continue to Rule 2 before final decision.

2) Diagnostic signals check
   * Scan the instance and concept for any of the diagnostic signals: {diagnostic_signals}
   * Presence of a signal is supporting evidence, not sufficient alone.

3) Evidence types
   * Explicit: the instance or concept uses terms that directly express the target concept.
   * Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   * Do NOT infer intent beyond the text. Do not rely on general world knowledge.

4) Dominance-based routing
   * Include if the target concept is explicit or an unambiguous paraphrase appears in the instance or concept, AND the primary focus of the idea aligns with this code's definition.
   * If a routing redirect matches the idea, note which neighboring code is suggested — but do NOT automatically exclude. Evaluate whether this code is still the PRIMARY fit for the idea's main focus.
   * **Causal hierarchy:** If this code captures the CAUSE/DRIVER and the neighbor captures the EFFECT/OUTCOME (or vice versa), assign to whichever code captures the cause. The thing that caused or enabled an experience takes priority over the experienced result.
   * Only exclude if:
       - The boundary test clearly fails AND no diagnostic signals are present;
       - The idea ONLY expresses the near-neighbor concept with no trace of this code's primary focus (per routing rule: {tell_apart_rule});
       - The concept is mentioned only in a negated or hypothetical way;
       - The response is too generic or off-topic.

5) Minimal supporting span
   * If Including, identify the shortest verbatim span from the instance that demonstrates the concept.
   * If Excluding, no supporting span is needed.
   * Preserve original casing and spelling; do not correct typos.

6) Confidence (0.00-1.00)
* 0.90-1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated; no interpretation needed. Another trained coder would definitely agree.
* 0.70-0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference. Another trained coder would likely agree.
* 0.50-0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment to justify. Reasonable coder disagreement is likely; discussion may be required.
* 0.00-0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response. Another trained coder would not assign this code.

7) **Confidence Threshold Rule (Critical)**
   * If the confidence score would be **below 0.70**, the decision **must be EXCLUDE**.
   * Borderline or partially implied concepts should **not** be coded as present.

IMPORTANT - RATIONALE STRUCTURE:
   * The rationale MUST begin with either "INCLUDE:" or "EXCLUDE:"
   * If INCLUDE: follow with the minimal supporting span in quotes, then a short explanation referencing the definition.
     Example: INCLUDE: "we krijgen geen begeleiding" -> explicitly expresses lack of support.
   * If EXCLUDE: briefly state the rule-based reason for exclusion.
     Example: EXCLUDE: No text expresses the target concept; content is generic.

Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "confidence": CONFIDENCE_SCORE,
  "rationale":  "INCLUDE: \\\\"...\\\\\" -> explanation in {language}" OR "EXCLUDE: brief explanation in {language}"
}}

Critical requirements:
- The confidence score must be a number between 0.00 and 1.00
- If the confidence score is below 0.70, the rationale MUST begin with "EXCLUDE:"
- The rationale must follow the INCLUDE:/EXCLUDE: format exactly
- Focus only on the specific concept defined by the code
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""

# All codes from partition (2+ codes)
PARTITION_EVALUATION_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses.
Your task is to evaluate ALL codes from a concept type partition and determine which one (if any) best fits the response concept.

Remember: codes are designed to be generalizable — they capture an underlying concept that respondents may express in many different ways. Match based on conceptual meaning, not surface-level keyword overlap. A code fits when the idea's underlying concept aligns with the code's definition, regardless of the specific words used.

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response concept you need to analyze:
<response>
Idea ID: {idea_id}
Instance (verbatim respondent text): {instance}
Rung 1 (interpretation): {rung_1}
Concept Type: {concept_type}
</response>
{dominance_axis_block}
Here are all codes available for this concept type. Codes may be organized into sub-partitions for clarity. Evaluate EACH code against the response:
<partition_codes>
{partition_codes_formatted}
</partition_codes>

**MANDATORY Step 0: Dominant Perspective Identification (complete BEFORE evaluating any code)**

Before evaluating ANY code, you MUST answer these three questions:

a) What is the PRIMARY experiential, causal, or functional focus of this idea? State it in one phrase.
b) Is the idea describing a CAUSE/DRIVER (what made something happen, a facility, a stimulus, an actor) or an EFFECT/OUTCOME (what was experienced, felt, or resulted)?
c) If a routing dimension is provided above, answer that routing question now.

Your answers to Step 0 CONSTRAIN all subsequent evaluation. You may NOT assign a code whose primary focus contradicts your Step 0 determination. Include your Step 0 answers in the best_match rationale.

**Evaluation Process (Steps 1-5):**
1. For each code, apply its boundary test — does it align with your Step 0 determination?
2. Check for diagnostic signals in the instance and concept
3. Use routing redirects to identify which code is the BEST fit — redirects suggest alternatives, they do not block assignment
4. If multiple codes match with confidence >= 0.70, apply the Tie-Breaking rules below
5. If no code matches with confidence >= 0.70, apply the NONE Guardrail below — do NOT default to NONE without completing the guardrail check

**Decision Rules (apply to EACH code):**

1) Conceptual relevance (informed by Step 0)
   * Apply the code's boundary test question to the idea.
   * If the answer is YES → strong evidence FOR assignment.
   * If the answer is NO but the idea is conceptually related to the code's definition → moderate evidence — continue evaluating with other rules before deciding.
   * The boundary test is a guide, not a gate. Codes are generalizable: an idea can match a code even if the boundary test is not a perfect fit, as long as the underlying concept aligns.

2) Diagnostic signals check
   * Scan the instance and concept for the code's diagnostic signals OR their synonyms, hypernyms, or close semantic equivalents. Codes are conceptual — respondents express the same concept using different vocabulary.
   * Presence of a signal or semantic equivalent is supporting evidence.
   * Absence of all listed signals does NOT disqualify the code — evaluate the idea's underlying meaning against the code's definition instead.

3) Evidence types
   * Explicit: the instance or concept uses terms that directly express the target concept.
   * Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   * Do NOT infer intent beyond the text. Do not rely on general world knowledge.

4) Dominance-based routing
   * Include if the target concept is explicit or an unambiguous paraphrase appears, AND the core meaning of the idea (from Step 0) aligns with this code.
   * If a routing redirect matches the idea, note which code is suggested as a better fit — but do NOT automatically exclude. Evaluate whether this code or the suggested code better captures the idea's core meaning as determined in Step 0.
   * Only exclude a code if:
       - The boundary test clearly fails AND no diagnostic signals or semantic equivalents are present;
       - The idea ONLY expresses a different code's concept with no trace of this code's core concept;
       - The concept is mentioned only in a negated or hypothetical way;
       - The response is too generic or off-topic.
   * When an idea touches multiple codes, assign to the code whose underlying concept most centrally captures the idea's meaning as determined in Step 0.

5) Minimal supporting span
   * If Including, identify the shortest verbatim span from the instance that demonstrates the concept.
   * Preserve original casing and spelling; do not correct typos.

**Confidence Anchors (0.00-1.00):**
* 0.90-1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated; no interpretation needed.
* 0.70-0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference.
* 0.50-0.69 (C: Moderate Match): The code's concept is present but expressed indirectly or with different vocabulary. Requires judgment but the conceptual link is defensible.
* 0.00-0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response.

**Confidence Threshold Rule (Critical):**
* Only codes with confidence >= 0.70 qualify as confident matches.
* If the best-fitting code has confidence < 0.70, set best_match.code to "NONE".

**Catch-all Deprioritization (Critical — codes marked [LAST RESORT]):**
* Codes containing "overig/anders" or "other/miscellaneous" are CATCH-ALL residual codes. They exist only for ideas that genuinely fit no specific code.
* You may ONLY assign a catch-all code if the idea does NOT reference:
  - a concrete object (e.g., toiletten, eten, schaduw, line-up)
  - a specific experience (e.g., lang wachten, goed geluid, lekker eten)
  - an identifiable actor (e.g., beveiliging, artiesten, personeel)
  - a describable condition (e.g., warm weer, drukte, modder)
* If ANY of these is present, a specific code MUST be assigned — even if the match is imperfect. Re-evaluate the most plausible specific code with lowered stringency rather than defaulting to a catch-all.
* When evaluating a catch-all code: cap its confidence at 0.50 unless no specific code reaches 0.40.

**NONE Guardrail (Critical — mandatory before setting NONE):**
* You may set best_match.code to "NONE" ONLY after completing this checklist:
  1. Does the idea express a concrete object, specific experience, identifiable actor, or describable condition? If YES → a specific code MUST be chosen. Re-evaluate the most plausible code with lowered stringency.
  2. Can the idea's Step 0 dominant focus be mapped to ANY code's definition, even loosely? If YES → assign that code with confidence reflecting the match strength.
  3. ONLY if BOTH checks are NO (the idea is genuinely abstract, generic, or off-topic) → set best_match.code to "NONE".

**Best Available Match Rule (Critical):**
* If no code reaches confidence >= 0.70 but at least one code reaches >= 0.55, assign the best-fitting code with its actual confidence score. A partial match is more informative than "NONE".
* Only set NONE when the best code confidence is below 0.55 AND the NONE Guardrail checklist is satisfied.
* Remember: codes are generalizable by design. If an idea clearly expresses a concept that a code captures, assign it — even if the phrasing doesn't match the examples exactly.

**Tie-Breaking (when multiple codes have confidence >= 0.70):**
1. **Causal hierarchy rule:** If one code captures the CAUSE/DRIVER and another captures the EFFECT/OUTCOME, assign to the CAUSE code. The thing that caused or enabled the experience takes priority over the experienced result. (Example: "Door de hitte was het fijn dat er schaduwplekken waren" → shade provision is the cause, comfort is the effect → assign to the shade/facility code.)
2. **Step 0 alignment:** Choose the code that best matches your Step 0 dominant perspective determination.
3. If still tied, choose the code with the highest confidence score.
4. If still tied, prefer the more specific code (narrower definition).

Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "evaluations": [
    {{"code": "CODE_NAME_1", "confidence": SCORE, "rationale": "INCLUDE: \\\\"span\\\\" -> explanation" or "EXCLUDE: reason"}},
    {{"code": "CODE_NAME_2", "confidence": SCORE, "rationale": "INCLUDE: \\\\"span\\\\" -> explanation" or "EXCLUDE: reason"}}
  ],
  "best_match": {{
    "code": "BEST_CODE_NAME or NONE",
    "confidence": SCORE,
    "rationale": "Step 0: [dominant focus] [cause/effect]. INCLUDE: \\\\"span\\\\" -> explanation in {language}" or "Step 0: [dominant focus]. EXCLUDE: brief explanation in {language}"
  }}
}}

Critical requirements:
- Complete Step 0 BEFORE evaluating any code
- Evaluate ALL codes in the evaluations array
- The best_match rationale MUST begin with "Step 0:" followed by your dominant perspective determination
- The best_match.code must be one of the evaluated codes OR "NONE" (only after completing the NONE Guardrail checklist)
- The best_match.confidence must match the confidence of the selected code (or 0.0 if NONE)
- All rationales must follow the INCLUDE:/EXCLUDE: format
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""


# =============================================================================
# STEP 8: SIMILARITY-BASED CODE ASSIGNMENT
# =============================================================================

# Codes selected by semantic similarity from the full codebook (not partition-restricted)
SIMILARITY_EVALUATION_PROMPT = """You assign codes to ideas expressed in survey responses. Pick the single best-fitting code from the candidates below. Match on conceptual meaning — respondents express the same concept in many different ways.

Survey question: {var_lab}
Language: {language}

<idea>
Idea ID: {idea_id}
Verbatim: "{instance}"
Interpretation: {rung_1}
Abstraction: {rung_2}
Concept type: {concept_type}
</idea>

<candidate_codes>
{candidate_codes_formatted}
</candidate_codes>

Instructions:
1. Read the idea (verbatim + interpretation + abstraction).
2. For each candidate code, check: does the idea express this code's concept? Use the code's definition, boundary test, diagnostic signals, and examples to decide.
3. Pick the single best-fitting code. If multiple codes fit, pick the one whose definition most centrally captures the idea's meaning.
4. Only return "NONE" if the idea is genuinely off-topic or too vague for any code. Prefer a specific code over "NONE" — an imperfect match is better than no match.
5. Codes marked [LAST RESORT] should only be chosen when no specific code fits at all.

Confidence guide:
* 0.85-1.00: Code meaning is explicitly stated in the verbatim text.
* 0.70-0.84: Code meaning is clearly present but phrased differently.
* 0.55-0.69: Code concept is present but expressed indirectly. The link is defensible.
* Below 0.55: No meaningful match — return "NONE".

Respond with JSON only:
{{
  "idea_id": "{idea_id}",
  "code": "BEST_CODE or NONE",
  "confidence": 0.00,
  "rationale": "brief explanation in {language}"
}}"""


# =============================================================================
# POST-ASSIGNMENT: AXIAL CODEBOOK CONSOLIDATION
# =============================================================================

AXIAL_CONSOLIDATION_PROMPT = """
You are a {language} qualitative research methodologist performing axial coding consolidation.

After initial open coding and code assignment, this theme contains more fine-grained codes than needed for clear reporting. Your task is to consolidate codes within this theme into a smaller set of higher-order codes, without losing conceptual clarity.

**Theme:** {theme}
**Current number of codes in this theme:** {n_codes}
**Target number of codes after consolidation:** {target_k}

Below are all codes in this theme, with their assignment frequencies and definitions:

<codes>
{codes_formatted}
</codes>

**Consolidation Rules:**

1. **Valence constraint:** Only merge codes that share the same valence (positive, negative, or neutral). NEVER merge a positive code with a negative code.

2. **Conceptual uniqueness protection:** For each code, ask:
   a. Is this code conceptually unique within the theme?
   b. Does it represent a divergent, critical, or corrective perspective?
   c. Would merging it change the interpretive narrative?
   If 2 or more answers are "yes" -> the code MUST be retained as standalone, regardless of frequency.

3. **Merging criteria (ALL must apply):**
   - The codes share substantial semantic overlap in their definitions
   - Their inclusion examples overlap significantly
   - They would be reported as a single finding in a research summary
   - Merging does not obscure a meaningful distinction

4. **Consolidated label requirements:**
   - Semantically neutral (not biased toward one merged code)
   - Thematically appropriate
   - Not overly broad — it should still be informative

5. **Coverage constraint:** Every original code must appear exactly once: either in `retained_codes` OR in exactly one entry of `consolidated_codes.original_codes`.

6. **Target guidance:** Aim for exactly {target_k} final codes (retained + consolidated combined), but conceptual quality takes precedence over hitting the exact number. If you cannot reduce to {target_k} without losing meaningful distinctions, produce more codes and explain why.

Provide your output as valid JSON following the response schema provided.
"""
