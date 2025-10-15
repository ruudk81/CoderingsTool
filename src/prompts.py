
# =============================================================================
# STEP 2: SPELL CHECKING
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

After processing all tasks, provide your output in the following JSON format:
{{
  "corrections": [
    {{
      "respondent_id": "ID_FROM_TASK",
      "corrected_response": "The fully corrected response"
    }},
    ...
  ]
}}

Ensure that your output is a valid JSON object with a single key "corrections", whose value is an array of objects. Each object in the "corrections" array must have exactly these fields:
- "respondent_id": "ID_FROM_TASK"
- "corrected_response": "The fully corrected response"

Additional guidelines:
- Pay close attention to the context and meaning of each response when making corrections.
- Ensure that your corrections maintain the original intent of the respondent.
- If a suggested correction doesn't fit the context, consider alternative corrections that preserve the meaning.
- Double-check that your JSON output is properly formatted and includes all corrected responses.

Begin processing the correction tasks now, and provide your output in the specified JSON format.
"""

# =============================================================================
# STEP 3: QUALITY FILTERING 
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses. 
Your task is to determine whether each response is meaningless and assign appropriate quality filter codes.

Task Description:
Analyze each response and classify it based on the following criteria:

Decision Criteria:
1. **Don't Know/Uncertainty (Code 99999997)**: Responses that express "don't know", "not applicable", or only express uncertainty
   - Examples: "I don't know", "N/A", "Not applicable", "No idea", "?"

2. **Nonsensical/Gibberish (Code 99999999)**: Responses that are meaningless, gibberish, or simply repeat the question
   - Examples: "asdfkj", "lorem ipsum", random characters, just repeating the question

3. **Meaningful Response (No Code)**: Responses that provide actual content, opinions, or information
   - These should have quality_filter = false and quality_filter_code = null

Input:
You will be provided with a survey question and a list of responses to evaluate.

Survey question:
<survey_question>
{var_lab}
</survey_question>

Here are the responses you need to evaluate:
<responses>
{responses}
</responses>

Your output should be a JSON array. Each object in the array must contain exactly:
- "respondent_id": (string or number) The respondent's ID
- "response": (string) The exact response text
- "quality_filter": (boolean) true if meaningless, false if meaningful
- "quality_filter_code": (number or null) 99999997 for uncertainty, 99999999 for gibberish, null for meaningful

Follow these steps for each response:
1. Read the response carefully.
2. Determine if the response expresses uncertainty/don't know (code 99999997)
3. If not uncertainty, determine if it's gibberish/nonsensical (code 99999999)
4. If neither, it's meaningful (quality_filter = false, quality_filter_code = null)
5. Create a JSON object with all required fields

After processing all responses, return the complete JSON array.

Remember to use the exact format specified. Here's an example of how entries in your output should look:
[
  {{
    "respondent_id": "1",
    "response": "I don't know",
    "quality_filter": true,
    "quality_filter_code": 99999997
  }},
  {{
    "respondent_id": "2",
    "response": "The product is easy to use and has great features.",
    "quality_filter": false,
    "quality_filter_code": null
  }},
  {{
    "respondent_id": "3",
    "response": "asdfghjkl",
    "quality_filter": true,
    "quality_filter_code": 99999999
  }}
]

Ensure that your entire output is a valid JSON array containing all evaluated responses.
"""

# =============================================================================
# STEP 4: IDEA EXTRACTION  
# =============================================================================

EXTRACT_SUBJECT = """
You are a {language} language expert in interpreting survey questions.
Your task is to extract the canonical focus entity — the single main product, brand, service, actor, or topic the question is about — and produce a reusable phrasing template.

<input>
Language: {language}
Survey question: {survey_question}
</input>

Definitions
- Canonical focus entity: the main noun phrase that answers “what or who is this question about?”
  It can be:
  - a product/brand/service being evaluated
  - a person/group/institution expected to act
  - a topic/concept/event being discussed

Constraints
- Choose exactly one canonical focus.
- Return a **concise, normalized noun phrase** in {language}.
- Do **not** include determiners (e.g., “the”, “de”, “het”), quotes, brackets, or trailing punctuation.
- Preserve capitalization for proper nouns/brands; otherwise use lowercase.
- If the question uses pronouns/deictics (e.g., “this brand”, “our app”), resolve to the most specific **named** entity present; if none, use the most specific generic noun (“app”, “klantenservice”, “bezorgproces”).
- If multiple entities appear, pick the one **most central** to what is being evaluated, judged, or requested.
- If the question refers to a sub-part/feature (e.g., “payment process” of a store), pick the **sub-part** if that is clearly the evaluation target.

Template requirement
1. Create a phrasing template that downstream steps can use to express evaluations or actions.
2. Use this structure:  
   **"[CANONICAL_TERM] [VERB/STATE] [ATTRIBUTE_OR_ACTION]"**
3. Insert:
   - The canonical term (noun phrase)
   - The most natural verb/state in {language} (e.g., “is”, “zijn”, “bieden”, “doet”, “heeft”)
   - Leave `[ATTRIBUTE_OR_ACTION]` as a placeholder
4. The result should sound natural and complete up to the placeholder.

Output format (return **only** this JSON object)
{
  "canonical_term": "canonical noun entity in {language}",
  "canonical_phrasing": "template with canonical term and verb/state inserted in {language}, e.g. 'Elektrische voertuigen zijn [EIGENSCHAP_OF_ACTIE]'"
}

Validation checklist before returning:
- Exactly two keys: `canonical_term`, `canonical_phrasing`
- Values are in {language}
- `canonical_term` is a noun phrase with no article or punctuation
- `canonical_phrasing` includes the correct verb/state (no placeholders for it) and ends with `[ATTRIBUTE_OR_ACTION]`

"""

IDEA_EXTRACTION_PROMPT = """
You are a {language} language expert in analyzing written responses to open-ended questions in surveys. 
Your task is to extract ALL distinct ideas expressed in a respondent's written answer.  

<inputs>
Survey question: {var_lab}
Canonical subject focus: {subject}
Respondent ID: {respondent_id}
Written response: {response}
</inputs>

<instructions>
1. Understand the context by reading the survey question and response carefully. 
        
2. Idea Identification
    - Extract all distinct ideas that directly answer or relate to the survey question. 
    - An “idea” is:
        - A single, complete thought or opinion
        - A specific action, behavior, or experience mentioned
        - A reason, cause, or explanation given
        - An emotion, attitude, or evaluation expressed

3. Atomicity
    - Keep each idea atomic — only one concept per idea.
    - Avoid merging ideas even if they are related.
    - Split compound statements connected by “and”, “but”, or similar connectors into separate ideas.
    - The idea you will return must not contain coordinating conjunctions or list markers in {language}.
    - Forbid list/coordination punctuation: "/", "&", ",", ";", ":", "-", "–" (hyphens allowed only inside a single lexicalized word, not to join ideas).

4. Phrasing template
    - Use this exact phrasing template in your output: {phrasing_template}
    - Do not change sentiment or tone during normalization.

5. Include Implicit Ideas
    - Capture both explicit statements and ideas that are clearly implied by the response.

6. Edge Cases
    - If the response is empty, irrelevant, or “N/A”: return an empty array [].
    - If the response is off-topic: extract ideas anyway but note they may be off-topic.
    - If only one idea is present: return it in a single-item array.
</instructions>

<output_format>
Return the extracted ideas as a JSON array. Each item should include:
- "respondent_id": exactly as provided
- "idea_id": a string number (numbering always starts at "1" and increments sequentially -e.g. "1", "2", etc.).
- "idea": the descriptive phrase in {language}, normalized and phrased using the provided phrasing template.
Always output in {language}.
</output_format>

Here's an example of the desired output example based on the input example:

<input_example>
Survey question: "What could the manufacturer of electric vehicles do better in your opinion?"
Respondent ID: 987654321
Response: "They should make the cars charge faster and improve battery life."
</input_example>

<output_example>
[
  {{
    "respondent_id": "987654321",
    "idea_id": "1", 
    "idea": "Electric vehicles should charge faster"
  }},
  {{
    "respondent_id": "987654321",
    "idea_id": "2",
    "idea": "Electric vehicles should have improved battery life"
  }}
]
</output_example>

Notice how:
- The original term “cars” was replaced with the canonical term “electric vehicles” from the survey question.
- All ideas use SUBJECT phrasing consistently.
- Each idea is separate and atomic.
- The meaning and sentiment of the original statement are preserved.
- Terms are consistent across all ideas, even if the respondent used different or less specific words.    
    
Return ONLY the JSON array. Keep field names in English; write values in {language}.
"""


# =============================================================================
# STEP 7:  CODEBOOK GENERATION : speculative codes
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

Your task is to generate {{n_codes}} diverse, hypothetical codes that might emerge from analyzing responses to the given survey question. Create codes that could apply to ANY survey topic. Do not assume the survey is about education, healthcare, or any specific domain. Let the survey question guide your code generation.

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
# STEP 7:  CODEBOOK GENERATION : 4 promt chain
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher applying Braun & Clarke’s (2006) thematic analysis method. 
Your task is to analyze a cluster of descriptive codes and construct one or more themes by identifying dominant patterns of shared meaning. 
The descriptive codes are derived from responses to the research question.

<inputs>
Language to use: {language}

Cluster ID: {cluster_id}

Research question:
"{survey_question}"

Cluster of descriptive codes to analyze:
{cluster_text}
</inputs>

<guidance>
- A central organizing concept (COC) is a unifying pattern of shared meaning, sentiment, or intention that brings together a group of codes into an **ATOMIC** theme in light of the research question.

- Atomic means:
  • One single idea, action, or expectation (not multiple at once).  
  • Belongs to one aspect, domain, category, or product attribute only.  
  • Expresses one consistent sentiment or intention (no mixing of positive/negative, or “keep” vs. “change”).  
  • Is concrete and directly actionable — cannot be meaningfully split further.  
  • If a code mentions multiple aspects (joined by “and,” “or,” commas, lists), crosses domains, or contains contradictory stances → split into multiple atomic concepts.  
</guidance>

Follow these steps exactly and in order. Do not skip or reorder any step. Use your analytical judgment and reflexivity throughout—remember that themes are not discovered in data but actively constructed.

<analysis_steps>
1. Interpret descriptive codes in light of the research question:
    - How does each code address the research question?
    - What patterns are meaningful for analyzing concrete, actionable answers?

2. Remove outliers. 
    - Eliminate codes that do not connect to any broader pattern across multiple codes.
    - Eliminate codes that do not represent a meaningful segment (too rare, irrelevant, or idiosyncratic).

3. Identify COC(s):
    - Can all codes be grouped around one central organizing ?
    - Practical test: 
        - Can I summarize all codes into one sentence that:
            a) alligns with <guidance>,
            b) captures exactly ONE ATOMIC theme,
            c) contains no coordinating conjunctions (e.g., "and," "or") or list punctuation (commas, slashes),
            d) preserves unity, consistency, and contrast?
        - If yes → single COC. If no → multiple COCs.
    - If one COC exists, continue with it; otherwise, work with multiple COCs.

4. Refine COCs:
    - Remove singletons (COCs based on only one code).
    - Remove COCs without a dominant shared pattern or conceptual overlap.
    - Remove vague or overly broad COCs (e.g., “positivity,” “challenges,” “general satisfaction”).

5. Construct theme(s):
    - Each theme must represent ONE ATOMIC concept only — see <guidance>.
    - If multiple atomic concepts exist, create multiple theme objects (one per concept).
    - Do not combine multiple aspects, domains, or contradictory stances into a single theme.
   
6. Document the analysis:
    - State how many COCs were identified.
    - If only one COC: explain why it is sufficient.
    - If multiple COCs: justify why not a single COC.

7. Write a theme label and clarification for that label:
    - Theme Label Template: "[≤ 10 words |  active/actionable formulation of ONE ATOMIC theme in relation to the research question - short, active, atomic]"
    - Clarification Template: "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label - tight, grounded, evidence-based]"

</analysis_steps>

Provide your response as a valid JSON dictionary using this exact structure:
{{
  "{cluster_id}": {{
    "analysis": "provide your analysis here in {language}",
    "extracted_themes": [
      {{
        "theme_id": 1,
        "theme_label": "[≤10-word label in {language}]",
        "theme_clarification": "[≤30-word clarification in {language}, with representative codes]"
      }}
      // Add additional theme objects here if more than one COC was found
    ]
  }}
}}

Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after.
- Keep field names in English; write values in {language}.
- Replace "cluster_id"  with the actual values provided.
- Conduct your analysis in the specified language.
"""

CODING_DECISION_PROMPT = """
You are a {language} qualitative research assistant helping to maintain a codebook for survey data analysis. 
Your task is to decide whether existing codes in a codebook are sufficient to describe a new theme, or whether modifications or new codes are needed.

<inputs>
Here is the survey question being analyzed:
"{survey_question}"

Here is the new theme:
- name: "{theme_name}"
- decription: "{theme_description}"    

Here is the list with existing codes in the codebook:
{code_text}
</inputs>

You have three possible decisions:
- **USE**: An existing code already captures the new theme's central meaning sufficiently
- **MODIFY**: An existing code is close but needs refinement for clarity, scope, or better alignment  
- **CREATE**: No existing code sufficiently captures the new theme

Follow these decision criteria:

A. **Coverage Assessment**
<coverage_assesment>
Ask yourself: “To what extent does one of the existing codes already express what this new theme is about—including its meaning, nuance, and boundaries?”

You’re estimating percentage overlap between:
- The new theme’s intended meaning, and
- The best-fitting existing code.

This is a judgment call—but must be justified.
</coverage_assesment>

B. **Decision Thresholds**
<decision_thresholds>
You base your final decision on how much coverage the existing code provides:

1. **USE** — if coverage ≥ 85%  
The existing code already captures the full meaning or is missing only minor phrasing differences that don’t change the conceptual scope.

2. **MODIFY** — if coverage is 70% to <85% AND mismatch can be resolved with a minor refinement:  
A small change to clarity, wording, or level of abstraction would make it fit perfectly, without introducing new meanings.

3. **CREATE** — if:  
- Coverage is <70%, OR  
- Coverage is 70–85% but mismatch cannot be fixed with simple refinement:  
The theme contains new concepts, facets, or intentions that don’t exist in the existing code, even if the topic looks related.
</decision_thresholds>

C. **Edge cases**
<edge_cases>
- For borderline cases (84–86% coverage): Choose **MODIFY** if a single wording/level adjustment achieves full coverage; otherwise **USE** if no material meaning is missing
- If the theme requires a composite name like "X and Y" or "X including Y, Z", then **CREATE** (the theme is not atomic)
- Do not modify existing codes in ways that would dilute their meaning for other uses
- When justifying your decision, you must explicitly reference the coverage percentage and which decision rule applies
</edge_cases>

Your response must be valid JSON only, following this exact format:

Output schema:
{{
  "coding_decision": {{
    "theme_number": {theme_id},
    "theme_name": "Exact name of the theme from <theme_to_code>",
    "matched_candidates": [
      {{"code": "Exact candidate code A", "definition": "Definition in light of the survey question"}},
      {{"code": "Exact candidate code B", "definition": "Definition in light of the survey question"}}
    ],
    "decision": "USE | MODIFY | CREATE",
    "source_code": "Exact candidate code name if use/modify, or null if create",
    "justification": "If abstraction level rules apply, justification must reference this explicitly in {language}. (e.g., ‘Theme is more general than code X but overlaps 95%, so USE.’)"
  }}
}}

Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation.
- Use theme_id provided.
- Keep field names in English; write values in {language}.
- Justification must clearly state the estimated overlap percentage and reference the decision rule applied (coverage thresholds and/or abstraction level).
"""

CODE_CREATION_PROMPT = """
You are a qualitative research assistant helping to maintain a codebook for survey data analysis. 
Your task is to CREATE a new code that captures a distinct theme emerging from responses to a specific survey question.

<inputs>
1) Language to use: {language}

2) Survey question:
"{survey_question}"

3) New theme that emerged:
- name: "{theme_name}"
- description: "{theme_description}"
</inputs>

Follow these critical coding principles:

- **ATOMIC**: The code must express ONE clear idea only.
    - Forbidden punctuation (unless lexicalized): "/", "&", "+", ",", ";", ":", "-", "–"
    - Use only ONE main verb if present.
- **PRECISE**: Code boundaries must be specific enough to enable consistent coding decisions.
- **CONCISE**: Code labels should be 5–10 words maximum. If longer, shorten while preserving meaning.
- **OPERATIONAL**: Code label and definition must align with the survey question.
- **GROUNDED**: Base your label and definition on the provided theme.    

Consider these possible **code types**:
- Attribute codes: qualities or characteristics
- Process codes: actions, procedures, or methods  
- Relational codes: interactions or relationships
- State codes: existing conditions or statuses
- Evaluative codes: assessments or judgments
- Brand/company names: named entities

Use these **strong patterns** for code definitions (≤25 words):
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these **weak patterns**:
- Compound: "References to [issue A] including [aspect 1], [aspect 2], [aspect 3]."
- Vague: "Mentions of various [things] related to [topic]."
- Interpretive: "Underlying [abstract concept] manifesting in different ways."

Output the result in this strict JSON schema (no commentary or explanation):
{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": "null",
    "code_label": "new or modified code label in {language}",
    "code_definition": "≤25-word operational definition in {language}"
  }}
}}

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

CODING_MODIFICATION_PROMPT = """
You are a qualitative research assistant helping to maintain a codebook for survey data analysis. 
Your task is to MODIFY an existing code so that it fits a new theme, while preserving the **core meaning** of the original code.

<inputs>
1) Language to use: {language}

2) Survey question:
"{survey_question}"

3) New theme that emerged:
- name: "{theme_name}"
- description: "{theme_description}"

4) Original code to modify:
{source_code}
</inputs>

Follow these critical coding principles:

- **ATOMIC**: The code must express ONE clear idea only.
    - Forbidden punctuation (unless lexicalized): "/", "&", "+", ",", ";", ":", "-", "–"
    - Use only ONE main verb if present.
- **PRECISE**: Code boundaries must be specific enough to enable consistent coding decisions.
- **CONCISE**: Code labels should be 5–10 words maximum. If longer, shorten while preserving meaning.
- **OPERATIONAL**: Code label and definition must align with the survey question.
- **GROUNDED**: Base your label and definition on the provided theme.    

Consider these possible **code types**:
- Attribute codes: qualities or characteristics
- Process codes: actions, procedures, or methods  
- Relational codes: interactions or relationships
- State codes: existing conditions or statuses
- Evaluative codes: assessments or judgments
- Brand/company names: named entities

Use these **strong patterns** for code definitions (≤25 words):
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these **weak patterns**:
- Compound: "References to [issue A] including [aspect 1], [aspect 2], [aspect 3]."
- Vague: "Mentions of various [things] related to [topic]."
- Interpretive: "Underlying [abstract concept] manifesting in different ways.

Output the result in this strict JSON schema (no commentary or explanation):
{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": {source_code},
    "code_label": "new or modified code label in {language}",
    "code_definition": "≤25-word operational definition in {language}"
  }}
}}

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

VALIDATION_PROMPT = """
You are a {language} curator of codebooks for thematic analysis following Braun & Clarke (2006) methodology. 
Your role is to maintain parsimonious codebooks with non-overlapping and non-redundant codes by reviewing and making final decisions on coding proposals.

<inputs>
Language to use: {language}

Existing codes in codebook:
{code_text}
    
Proposal background:
    
A new theme emerged from analyzing responses to this survey question: 
"{survey_question}" 

This is the new theme:
- name: "{theme_name}"
- description: "{theme_description}"    

The proposal to review:
{step3_recommendation}
</inputs>

Your task is to systematically evaluate this proposal and make a final coding decision. Use the scratchpad below to work through your evaluation before providing your final JSON response.

<scratchpad>
Work through your evaluation systematically:

Step 1: Evaluate the CREATE/MODIFY decision against these criteria:
a. Parsimony: Has sufficient effort been made to use existing codes or combinations before proposing new/modified codes?
b. Abstraction Level: Is any proposed new code at an appropriate abstraction level, consistent with existing codes?
c. Non-Redundancy: Does the proposal avoid creating codes that significantly overlap with existing ones?

Step 2: Evaluate code label and definition:
d. Atomicity:
   • Does it express only one idea (no merged/compound themes)?
   • Does it avoid forbidden punctuation: "/", "&", "+", ",", ";", ":", "-", "–" (unless the punctuation is lexicalized within a compound noun (e.g., ‘gebruiksklaar-product’ is allowed).")?
   • Does it contain at most ONE main action (verb)?
   • Does it avoid conjunctions ("and/or") unless lexicalized?

e. Form & Length:
   • Is the label ≤10 words with no canonical subject from survey question and no implied actor?
   • Does it follow allowed forms:
     – Noun phrase: <adjective(s)> <noun>
     – Imperative verb + object: <verb> <object>
     – Infinitive: <to/infinitive verb> <object>
   • Is the definition ≤25 words, operational/observable, grounded in responses, and non-vague?
   
Step 3: APPROVE/REJECT proposal
- If all criteria PASS → APPROVE (you may make minor refinements for full compliance)
- If any criterion FAILS → REJECT (identify issues and rewrite to comply)

Step 4: If rejected on the grounds of parsimony, non-redunancy or overlap, make a final decision about:  
- **USE**: An existing code already captures the new theme's central meaning sufficiently
- **MODIFY**: An existing code is close but needs refinement for clarity, scope, or better alignment  
- **CREATE**: No existing code sufficiently captures the new theme

Step 5: Determine your final components:
- validated_decision: USE, MODIFY, or CREATE code
- source_code: 
    - If USE, this exact code: {source_code}
    - If MODIFY, the exact code from the existing codebook you seek to modify
    - If CREATE, write "null"
- validated_code and validated_decision: final compliant label and definition
- decision_rationale: brief explanation of approval/rejection
</scratchpad>

Now provide your final evaluation as valid JSON in the specified language. Return ONLY the JSON response with no additional text, comments, or extra fields. Use this exact schema:

Output schema:
{{
  "code_validation": {{
    "theme_number": {theme_id},
    "theme_name": {cluster_summary},
    "original_recommendation": {{
        "code": "Exact recommended label",
        "definition": "Exact recommended definition"
      }},
    "verdict": "APPROVE" | "REJECT",
    "decision_rationale": "Brief explanation as to why the recommendation was approved or rejected",
    "validated_decision" : "USE | MODIFY | CREATE" , 
    "source_code": "If USE, this exact code: {source_code}; If MODIFY, the exact code from the existing codebook you seek to modify - or null, if CREATE",
    "validated_code": {{
      "code": "Final validated label (≤10 words, rule-compliant)",
      "definition": "Final validated definition (≤25 words, operational, grounded)"
    }}
  }}
}}

If USE, this exact code: {source_code}; If MODIFY, the exact code from the existing codebook you seek to modify - or null, if CREATE

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

# =============================================================================
# STEP 8 THEME ORGANIZATION WITH REASONING MODELS
# =============================================================================

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative researcher and codebook methodologist. 
Your task is to take a raw list of descriptive codes and transform it into a refined and structured codebook. 
The descriptive codes are derived from survey responses.

<inputs>
Language to use: {language}

survey_question: {survey_question}

Raw descriptive codes to refine:
{raw_codes}
</inputs>

<guidance>
A high-quality codebook must be:
- Non-redundant: No duplicate codes that restate the same idea in different words.
- Inherently distinct: codes within the same category differ in content, not just phrasing.
- Semantically differentiated: Wording for each code should be clearly different; minor overlap is acceptable.
- Parsimonious: Use the fewest codes needed to cover the data comprehensively.
- Well-structured: Organize in a clear hierarchy — Themes → Codes.
- Each code has exactly one parent (no multi-parenting).
- Consistently labeled: Use short, uniform, action-oriented labels that are meaningful for the survey question: “{survey_question}”.    
    
When codes overlap semantically:
- Merge near-duplicates into a single code and remove duplicates.
- Consolidate overlapping codes under one inclusive label that captures the shared idea.
- **Preserve contrasts**: Keep separate codes that reflect  distinctive, contrasting, or polar aspects/cateogries/product-attributes, even if they belong under the same broader theme.
- Refine wording of any vague code to make its scope precise.    

Labeling:
- Prefer active, specific labels (e.g., “Seeks clearer instructions” over “Clarity”).
- Include a brief definition/decision rule and 1–2 examples for each code.
</guidance>

<analysis_steps>
1. Review all raw codes.
   - Identify redundant codes (semantic duplicates, identical meaning).
   - Identify overlapping codes that can be merged into broader codes.

2. Construct main themes that represent broad domains.

3. Assign refined codes under these themes.
   - Each code must represent ONE distinct, actionable concept.
   - Remove vague or overly broad codes.

4. Ensure consistent naming:
   - Labels ≤ 8 words.
   - Active or descriptive phrasing.
   - No repetition across themes.

5. Document the restructuring in "analysis":
   - How many raw codes were merged.
   - Which semantic duplicates were consolidated.
   - Which themes were created and why.
</analysis_steps>

Provide your response as a valid JSON dictionary using this exact structure:
{{
  "analysis": "Provide your analysis here in {language} (describe main restructuring decisions, what was merged, how categories were formed).",
   "refined_codebook": [
      {{
        "theme": "Main theme label",
        "codes": [
          {{
            "id" : "original code_id(s) - only multiple if merged"
            "code": "Refined subcode label",
            "description": "≤ 20 words explanation of what this code means"
          }}
          // Add additional subcodes as needed
        ]
      }}
      // Add additional categories here
    ]
}}


Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after.  
- Replace "codebook_id" and "language" with the actual values provided.  
- Conduct your analysis in the specified language.  
"""

THEME_ORGANIZATION_REASONING_PROMPT = """
You are a qualitative research specialist with expertise in thematic analysis. Your task is to organize a set of codes into atomic themes based on semantic similarity in the context of a research question.

Here is the research question you should consider:
<research_question>
{research_question}
</research_question>

Here are the codes with their definitions that you need to organize:
<codes_and_definitions>
{codebook}
</codes_and_definitions>

You should provide all labels and descriptions in this language:
<language>
{language}
</language>

## CORE TASK
Organize the provided codes into **atomic themes** based on semantic similarity in light of the research question. An atomic theme label refers to one single idea that cannot be split into multiple, distinct concepts without losing clarity.

## KEY REQUIREMENTS

**Atomic Theme Names:**
- Must express only ONE clear concept
- Do NOT use compound labels, slashes (/), conjunctions (like "and"/"en"), or lists of terms
- ✅ Good: "Price", "Health", "Convenience" 
- ❌ Bad: "Price and Quality", "Health/Nutrition", "Time and Convenience"

**Grouping Rules:**
- Each theme must contain 2 or more codes that express related ideas
- Base groupings on meaning in the context of the research question
- Every unique idea should be represented once. If two or more codes express the same or highly similar concept, you may merge them or remove the duplicates to maintain atomicity and avoid redundancy. Each remaining code should appear only once.
- Group codes that address similar aspects or concerns related to the research question

## INSTRUCTIONS

<scratchpad>
Before providing your final answer, think through the following:

1. **Analyze the codes:** What are the main conceptual areas represented in the codes?
2. **Identify natural groupings:** Which codes seem to address similar aspects of the research question?
3. **Create atomic theme names:** For each grouping, what single concept best captures what those codes are about?
4. **Verify completeness:** Have you included every code exactly once?
5. **Check atomicity:** Are all your theme names truly atomic (single concepts)?
6. **Handle redundancy:** 
   - If you encounter codes that are semantically identical (e.g., different phrasings of the same idea) or highly overlapping (e.g., small wording differences but no substantial conceptual distinction), you may:
     - Keep only the clearest or most comprehensive version.
     - Remove or merge the others.
   
</scratchpad>

Now provide your analysis in the following JSON format. Return only valid JSON with no additional text:

```json
{{
  "themes": [
    {{
      "theme_name": "[Atomic theme name in {language}]",
      "theme_description": "[Detailed description of what this theme represents in {language}]",
      "codes": [
        {{
          "code": "[Original code name]",
          "definition": "[Original code definition]",
        }}
      ]
    }}
  ],
  "methodology": "Single-prompt hierarchical theme organization using reasoning model",
  "total_codes_organized": {codes_count},
  "language": "{language}"
}}    
```

## FINAL CHECKLIST
Before submitting, verify that:
- All theme names are atomic (no conjunctions, slashes, or multiple concepts)
- Each code appears exactly once
- Every theme contains at least 2 codes
- All labels and descriptions are in the specified language
- The JSON format is valid and complete
"""

# =============================================================================
# STEP 9: CODE ASSIGNMENT
# =============================================================================

CODE_ASSIGNMENT_PROMPT = """
You are a {language} language expert in qualitative data analysis, specializing in applying codebooks to open-ended survey responses. Your task is to assign the single most appropriate code from a focused list of 5 candidate codes to a specific response segment.

First, review the original survey question:
<survey_question>
{var_lab}
</survey_question>

Next, examine the response segment you need to analyze:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

Now, review the 5 candidate codes and their descriptions:
<candidate_codes>
{candidate_codes}

Note:
- Prioritize the CODE LABEL; Use the CODE DESCRIPTION only as supporting context to confirm scope and clarify meaning.
</candidate_codes>

Your goal is to select the single best fitting code for the response segment. Follow these steps:

1. Carefully read and understand each candidate code's definition.
2. Analyze the semantic meaning of the response segment, considering the context of the survey question.
3. Identify which code best captures the core concept expressed in the response.
4. Assign exactly one code, even if the fit isn't perfect. Choose the best available option based on semantic meaning.

When selecting the best fitting code:
- Prioritize exact conceptual matches based on meaning.
- Do not rely solely on surface keywords. Base your choice on semantic alignment with the code's definition.

After selecting the code, rate the strength of the fit using this scale:
- Excellent (0.90–1.00): Exact match — the idea uses the same language or concepts as the code definition, with no ambiguity or need for interpretation.
- Very Good (0.80–0.89): Very strong fit — conceptually aligns and clearly supports the code with minimal nuance.
- Good (0.60–0.79): Strong fit — covers the code’s core meaning, but may involve some rephrasing or interpretation.
- Moderate (0.50–0.59): Somewhat related — a plausible fit that requires noticeable interpretation or reframing to align with the code.
- Poor (0.3–0.49): Weak conceptual fit; a stretch, but arguably the closest option available.
- Very Poor (0.0–0.29): Barely relevant; applied only due to lack of better alternatives.

Provide your response in the following JSON format:
<output_format>
{{
  "idea_id": "{idea_id}",
  "idea": "{idea_text}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
}}
</output_format>

Critical requirements:
- Use exact code names as provided in the candidate codes list.
- Assign one and only one code per response.
- The confidence score must reflect conceptual fit, not how likely you feel about the assignment.
- The rationale must explain the semantic connection to the code definition.
- Return ONLY the JSON object in {language}.

Begin the code assignment now.
"""
