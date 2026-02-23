"""
Experimental Prompts for Step 8: Code Assignment (Partition-Based, Ladder)

Two clean prompts for partition-based code assignment using the abstraction ladder:
- SINGLE_CODE_EVALUATION_PROMPT: Evaluate one code against an idea's abstraction ladder
- PARTITION_EVALUATION_PROMPT: Evaluate all codes from a partition against an idea's ladder

No fallback prompts — if no code matches, the idea gets unknown_label.
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
Concept: {concept}
Concept Type: {concept_type}
</response>

Here is the code you need to evaluate:
<code_details>
Code: {code}
Definition: {definition}

Boundary test (yes/no gate): {boundary_test}
Diagnostic signals (trigger words/phrases): {diagnostic_signals}

Inclusion Examples (valid references for this code):
    {inclusion_examples}

Exclusion Examples (invalid references for this code):
    {exclusion_examples}

Boundary: This code covers "{code}", which differs from "{near_neighbor_label}"
How to tell them apart: {tell_apart_rule}
</code_details>

Follow these DECISION RULES strictly:

1) Boundary test gate
   * First apply the boundary test: "{boundary_test}"
   * If the answer is clearly NO, EXCLUDE immediately.

2) Diagnostic signals check
   * Scan the instance and concept for any of the diagnostic signals: {diagnostic_signals}
   * Presence of a signal is supporting evidence, not sufficient alone.

3) Evidence types
   * Explicit: the instance or concept uses terms that directly express the target concept.
   * Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   * Do NOT infer intent beyond the text. Do not rely on general world knowledge.

4) Include vs Exclude
   * Include if the target concept is explicit or an unambiguous paraphrase appears in the instance or concept.
   * Exclude if the response:
       - Only expresses the near neighbor concept (per {tell_apart_rule});
       - Matches any Exclusion Example pattern;
       - Mentions the concept only in a negated or hypothetical/conditional way (e.g., "would/if/might" without an asserted claim);
       - Is too generic or off-topic.
   * If both Inclusion-like and Exclusion-like signals appear, Exclusion takes precedence unless the Inclusion is explicit and clearly satisfies the Definition.

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

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response concept you need to analyze:
<response>
Idea ID: {idea_id}
Instance (verbatim respondent text): {instance}
Concept: {concept}
Concept Type: {concept_type}
</response>

Here are all codes available for this concept type. Codes may be organized into sub-partitions for clarity. Evaluate EACH code against the response:
<partition_codes>
{partition_codes_formatted}
</partition_codes>

**Evaluation Process:**
1. For each code, first apply its boundary test as a yes/no gate
2. Check for diagnostic signals in the instance and concept
3. Evaluate inclusion/exclusion examples
4. If multiple codes match with confidence >= 0.70, choose the MOST SPECIFIC one that fits the evidence
5. If no code matches with confidence >= 0.70, set best_match.code to "NONE"

**Decision Rules (apply to EACH code):**

1) Boundary test gate
   * Apply the code's boundary test question. If the answer is clearly NO, EXCLUDE that code.

2) Diagnostic signals check
   * Scan the instance and concept for the code's diagnostic signals. Presence is supporting evidence, not sufficient alone.

3) Evidence types
   * Explicit: the instance or concept uses terms that directly express the target concept.
   * Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   * Do NOT infer intent beyond the text. Do not rely on general world knowledge.

4) Include vs Exclude
   * Include if the target concept is explicit or an unambiguous paraphrase appears in the instance or concept.
   * Exclude if the response:
       - Only expresses a different code's concept;
       - Matches any Exclusion Example pattern;
       - Mentions the concept only in a negated or hypothetical/conditional way;
       - Is too generic or off-topic.

5) Minimal supporting span
   * If Including, identify the shortest verbatim span from the instance that demonstrates the concept.
   * Preserve original casing and spelling; do not correct typos.

**Confidence Anchors (0.00-1.00):**
* 0.90-1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated; no interpretation needed.
* 0.70-0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference.
* 0.50-0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment. EXCLUDE threshold.
* 0.00-0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response.

**Confidence Threshold Rule (Critical):**
* Only codes with confidence >= 0.70 qualify as matches.
* If the best-fitting code has confidence < 0.70, set best_match.code to "NONE".

**Tie-Breaking (when multiple codes have confidence >= 0.70):**
1. Choose the code with the highest confidence score.
2. If still tied, prefer the more specific code (narrower definition).
3. If still tied, prefer the code whose definition most closely matches the supporting span.

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
    "rationale": "INCLUDE: \\\\"span\\\\" -> explanation in {language}" or "EXCLUDE: brief explanation in {language}"
  }}
}}

Critical requirements:
- Evaluate ALL codes in the evaluations array
- The best_match.code must be one of the evaluated codes OR "NONE" if no code reaches 0.70 confidence
- The best_match.confidence must match the confidence of the selected code (or 0.0 if NONE)
- All rationales must follow the INCLUDE:/EXCLUDE: format
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""
