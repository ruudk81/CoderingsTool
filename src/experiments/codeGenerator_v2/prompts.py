"""
Prompts for codeGenerator_v2 experiment.

These prompts are copies from src/prompts.py for local experimentation.
Modify these prompts to test different codebook generation approaches.

This file contains only the Step 6 (codebook generation) prompts:
- CLUSTER_SUMMARY_PROMPT (Stage 1: Theme Extraction)
- CODING_DECISION_PROMPT (Stage 2: Code Decision)
- CODE_CREATION_PROMPT (Stage 3: Code Creation)
- CODING_MODIFICATION_PROMPT (Stage 3: Code Modification)
- VERTICAL_INSTRUCTIONS (placeholder for modification)
- HIERARCHICAL_INSTRUCTIONS (placeholder for modification)
- VALIDATION_PROMPT (Stage 4: Validation)
"""

# =============================================================================
# STEP 6: CODEBOOK GENERATION - 4 PROMPT CHAIN
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher applying Braun & Clarke's (2006) thematic analysis method. 
Your task is to analyze a cluster of descriptive codes and construct one or more ATOMIC themes (central organizing concepts) along a given taxonomy axis, then draft initial codebook entries for each theme.

You will be working in the following language:
<language>
{language}
</language>

Here is the cluster you will analyze:
<cluster_id>
{cluster_id}
</cluster_id>

<context>
- Domain: {domain}
- Topic: {topic}
</context>

<research_parameters>
Research question: "{survey_question}"

Response interpretation:
- Perspective: {perspective}
- Intent: {intent}
</research_parameters>

<cluster_data>
{cluster_text}
</cluster_data>

<taxonomy>
Coding dimension: {taxonomy_axis}
Taxommy axis: {taxonomy_axis_description}
Taxonomy focus: {taxonomy_actionable_type}
</taxonomy>

<key_definitions>
- ATOMIC THEME: A single idea, action, expectation, or motive relevant to the research question. No mixing of multiple concepts.
- TAXONOMY AXIS: The conceptual dimension along which the theme is defined.
- ABSTRACTION LEVEL: The conceptual "height" of the theme. Must be one of:
  • "Driver/Motive/Why" (highest abstraction - underlying reasons)
  • "Attribute/What" (mid-level - characteristics or qualities)
  • "Action/How" (concrete - specific behaviors or methods)
- NEAR-NEIGHBOR: An adjacent concept that could be confused with the theme.
</key_definitions>

<guidance>
- Atomic means:
  • Single idea only (no "and/or" combinations).
  • One aspect/domain/category/attribute only.
  • One consistent sentiment/intention (no polarity mixing).
  • Concrete and directly actionable.
- If a code mentions multiple aspects or mixed sentiment → split into separate atomic concepts.

Label constraints (strict):
- ≤ 10 words
- Active/actionable formulation of ONE ATOMIC theme in relation to the research question
- Atomic means "One domain or aspect only"
- If verb is used → one main verb (present tense).
- **Never** include reasons (no "to", "so that", "because").
- Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
- Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

Definition constraints (strict):
- ≤ 30 words
- grounded in theme description
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
</guidance>

Follow these steps exactly and in order. Do not skip or reorder any step. Use your analytical judgment and reflexivity throughout—remember that themes are not discovered in data but actively constructed.

<analysis_steps>
1. Interpret descriptive codes in light of the TAXONOMY FOCUS ("{taxonomy_actionable_type}") and RESEARCH QUESTION:
    - How does each code address the research question?
    - What patterns are meaningful for analyzing concrete, actionable answers?

2. Remove outliers.
    - Eliminate codes that do not connect to any broader pattern across multiple codes.
    - Eliminate codes that do not represent a meaningful segment (too rare, irrelevant, or idiosyncratic).

3. Identify COC(s):
    - Practical test: Can all codes be summarized in one sentence that:
     (a) aligns with <guidance>,
     (b) captures exactly ONE ATOMIC theme,
     (c) uses no coordinating conjunctions or list punctuation,
     (d) preserves unity, consistency, and contrast?
   - If yes → single COC. If no → multiple COCs.
   - Proceed with the resulting COC(s).

4. Refine COCs:
    - Remove singletons (COCs based on only one code).
    - Remove COCs without a dominant shared pattern or conceptual overlap.
    - Remove vague or overly broad COCs (e.g., "positivity," "challenges," "general satisfaction").

5. For each retained COC, produce a codebook-ready draft with:
   - theme_label: "[≤ 10 words | active/actionable formulation of ONE ATOMIC theme in relation to the research question]"
   - theme_clarification: "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label — tight, grounded, evidence-based]"
   - abstraction_level: Select one of: "Driver/Motive/Why" | "Attribute/What" | "Action/How"
   - assignment_examples (EXAMPLES, not rules):
       • inclusion: Provide 2–3 short, positive EXAMPLE assignment expressions (each starts with a verb; use observable cues).
       • exclusion: Provide 1–2 short EXAMPLE boundary examples to prevent overreach (what must NOT be included).
       • near_neighbor:
         — label: closest potentially-confusable theme or "Unknown"
         — tell_apart_rule: one sentence explaining how to distinguish the two (e.g., "This theme focuses on X (driver/what/how), whereas the neighbor focuses on Y.")

6) Document the analysis:
   - State how many COCs were identified and retained.
   - If only one COC: explain why it is sufficient.
   - If multiple COCs: justify why a single COC would violate atomicity or clarity.
</analysis_steps>

Output strictly as valid JSON using this exact structure (values in {language}, field names in English):
{{
  "cluster_id": "{cluster_id}",
  "analysis": "Provide your analysis here in {language}.",
  "extracted_themes": [
    {{
      "theme_id": 1,
      "theme_label": "[≤ 10 words | active/actionable formulation of ONE ATOMIC theme in relation to the research question]",
      "theme_clarification": "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label — tight, grounded, evidence-based]",
      "abstraction_level": "Driver/Motive/Why | Attribute/What | Action/How",
      "assignment_examples": {{
        "inclusion": [
          "[examples inclusion in {language}]"
        ],
        "exclusion": [
          "[examples exclusion in {language}]"
        ],
        "near_neighbor": {{
          "label": "[neighbor label in {language} or 'Unknown']",
          "tell_apart_rule": "[1-sentence distinction in {language}]"
        }}
      }}
    }}
  ]
}}


Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after.
- Keep field names in English; write values in {language}.
- The cluster_id value must be exactly "{cluster_id}" as provided in the inputs.
- Conduct your analysis in the specified language.
"""

CODING_DECISION_PROMPT = """
You are a {language} qualitative research assistant responsible for maintaining a structured codebook.

Your task is to classify a new theme by deciding whether it should:
- be assigned to an existing code (**USE**),
- extend an existing code (**MODIFY – Vertical / same motive**),
- consolidate into a parent code that groups multiple codes (**MODIFY – Hierarchical / different motive but same conceptual family**),
- or be added as a new code (**CREATE**).

You must base your reasoning on:
1) Cosine similarity (semantic proximity),
2) The underlying motive/intent (the "why" behind the theme),
3) The conceptual structure of the codebook (to maintain clear boundaries).

<inputs>
Survey Question:
"{survey_question}"

New Theme to Classify:
- name: "{theme_name}"
- description: "{theme_description}"
- abstraction level: "{abstraction_level}"
- what's included:
    {inclusion}
- what's excluded:
    {exclusion}
- boundary: {near_neighbor}

Existing Codes:
{code_text}
</inputs>

Follow these steps exactly and in order. Do not skip or reorder any step.

<analysis_steps>
1. Cosine Similarity Rules (EXTREMELY STRICT - no deviation allowed):

- **≥ 0.90** → Same meaning → **USE**
- **< 0.70** → Different meaning → **CREATE**
- **0.70–0.90** → Perform Motive Test

2. MOTIVE TEST:
Ask: *Does the new theme share the same underlying motive/driver as the highest-similarity code?*

- Same motive → **MODIFY (Vertical Expansion)**
  • Add new expressions to **inclusion rules**
  • Keep same abstraction level
  • Maintain atomicity

- Different motive but same conceptual family → **MODIFY (Hierarchical Expansion)**
  • Create or reference a **parent** theme at a higher abstraction level
  • Existing code and new theme become **sub-themes**

- Different motive and not in same family → **CREATE** new code

3. Structural Constraints (Always Enforce)
- Codes must remain **atomic**: one idea, one motive, one sentiment.
- Inclusion rules describe when to assign the theme.
- Exclusion rules describe common misfits to keep boundaries clear.

4. Update Assignment Examples
- If decision is **USE** → preserve original assignment_examples unchanged
- If decision is **MODIFY**:
  • inclusion: combine original + new expressions from the theme
  • exclusion: combine original + new boundary clarifications if needed
  • near_neighbor: update label if boundaries shifted due to modification
  • tell_apart_rule: update if the distinction from neighbor changed
- If decision is **CREATE** → use assignment_examples from the new theme as-is
</analysis_steps>

Respond with **valid JSON only** in the following structure:

{{
  "coding_decision": {{
    "theme_number": {theme_id},
    "theme_name": "Exact name of the theme from <theme_to_code>",
    "matched_candidates": [
        {{"code": "Exact candidate code A", "definition": "Definition in light of the survey question"}},
        // Add additional candidaties,if there are any
        ]
    "decision": "USE | MODIFY | CREATE",
    "source_code": "Exact candidate code name if use/modify, or null if create",
    "modify_parameters":{{
       "modify_instruction": "vertical_broaden_same_motive | hierarchical_parent_diff_motive_same_family | none",
       "motive_comparison": "same | different_same_family | different_not_related",
       "abstraction_level_action": "keep | broaden_to_parent | none",
       "inclusion_update": "null or concrete additions to inclusion rules",
       "exclusion_update": "null or concrete boundary clarifications",
       "parent_theme_label": "null or suggested parent label",
       "near_neighbor_label_update": "null or updated neighbor label if boundaries changed",
       "tell_apart_rule_update": "null or updated tell-apart rule if distinction changed"}},
    "justification": "Explain modification decision by referencing cosine score and whether motive was same or different, or null if use/create",
    "updated_assignment_examples": {{
      "inclusion": ["[updated or original inclusion examples in {language}]"],
      "exclusion": ["[updated or original exclusion examples in {language}]"],
      "near_neighbor": {{
        "label": "[updated or original neighbor label in {language}]",
        "tell_apart_rule": "[updated or original tell-apart rule in {language}]"
      }}
    }}.
  }}
}}

Examples (for Interpretation)

1. **Vertical MODIFY Example**
cosine = 0.81, same motive
→ "This theme expresses the same underlying reason as Code A but in a new form. MODIFY (vertical)."

2. **Hierarchical MODIFY Example**
cosine = 0.80, different motive but same conceptual family
→ "This theme shares the domain of Code A but with a different driving motive. MODIFY (hierarchical)."

3. **CREATE Example**
cosine = 0.64
→ "The meaning and motive differ significantly. CREATE new code."

REQUIREMENTS
- Output **must be valid JSON only** (no commentary).
- Keep field names in English; write values in {language}.
- Include **cosine score** in justification.
- Identify motive comparison explicitly.

"""

CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant.
Your task is to CREATE a new **atomic** code that captures the meaning of a newly identified theme from survey responses.

<inputs>
Survey question:
"{survey_question}"

New theme:
- name: "{theme_name}"
- description: "{theme_description}"
- abstraction level: "{abstraction_level}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
- Excluded expressions (these should NOT be covered by the code):
  {exclusion}
- boundary: {near_neighbor}
</inputs>

DEFINITION OF AN ATOMIC CODE:
- Expresses exactly **one** idea, action, attitude, or expectation.
- Cannot be split into two meaningful codes without losing clarity.
- Contains **no conjunctions** (and/or), **no lists**, and **no dual motives**.

LABEL RULES (strict):
- ≤ 10 words
- Active/actionable formulation of ONE ATOMIC theme in relation to the research question
- Atomic means "One domain or aspect only"
- If verb is used → one main verb (present tense).
- **Never** include reasons (no "to", "so that", "because").
- Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
- Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

DEFINITION RULES:
- ≤ 30 words
- grounded in theme description
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.

GOOD DEFINITION PATTERNS:
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"

AVOID:
- Broad summaries (e.g., "general dissatisfaction").
- Multi-part or layered meaning.
- Psychological interpretation not grounded in wording.

ASSIGNMENT EXAMPLES:
- Provide concrete, actionable assignment examples to guide future code assignment
- inclusion: 2-3 short examples of expressions that SHOULD be coded here
- exclusion: 1-2 short examples of what should NOT be included
- near_neighbor: Identify closest confusable concept and how to tell them apart

Output the result in this strict JSON schema (no commentary or explanation):
{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": "null",
    "code_label": "new or modified code label in {language}",
    "code_definition": "≤25-word operational definition in {language}",
    "assignment_examples": {{
      "inclusion": ["[2-3 concrete examples of what to include in {language}]"],
      "exclusion": ["[1-2 concrete examples of what to exclude in {language}]"],
      "near_neighbor": {{
        "label": "[closest confusable concept in {language} or 'Unknown']",
        "tell_apart_rule": "[1-sentence distinction in {language}]"
      }}
    }}
  }}
}}

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

# Placeholders for CODING_MODIFICATION_PROMPT
VERTICAL_INSTRUCTIONS = """
   - Keep the abstraction level of the original code.
   - Create a **single atomic shared concept** that:
        (a) captures the meaning of both original code and new theme,
        (b) is grounded in the shared intent (same motive),
        (c) remains expressible as **one idea** in the label.
   - The modified label must:
        • reflect the broadened meaning,
        • NOT introduce multiple aspects or motives,
        • NOT be more abstract than necessary.
   - The modified definition must:
        • describe the **shared meaning space**,
        • reflect: original inclusions + inclusion_update,
        • exclude: original exclusions + exclusion_update.
   - Do **not** modify assignment rules here."""

HIERARCHICAL_INSTRUCTIONS = """
   - Shared conceptual domain but different motives → create hierarchical structure.
   - Original code and new theme remain **atomic child codes**.
   - Parent code represents the shared **purpose/motive domain**.

   Parent label:
        - parent theme = {parent_theme_label}
        - If parent theme is not None or Null → use it as-is.
        - If null → generate a label at **Driver/Motive/Why** level.
        - Must:
            • express shared purpose/orientation,
            • NOT describe behaviors/outcomes,
            • NOT blend child labels,
            • be broader, not vaguer.

   Structure:
       - Parent = conceptual anchor (why level),
       - Children = distinct manifestations (how/what),
       - Child meanings **do not change**."""

CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new theme, while preserving **atomic meaning** and **clear conceptual boundaries**.

<inputs>
Survey question:
"{survey_question}"

New theme to integrate:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
- Excluded expressions (these should NOT be covered by the code):
  {exclusion}

Original code (to be modified):
- code_label: {source_code}
- code_definition: {source_definition}

Current assignment examples (before modification):
- Current inclusion examples:
  {current_inclusion}
- Current exclusion examples:
  {current_exclusion}
- Current near neighbor boundary:
  {current_near_neighbor}

Required modifications:
- inclusion_update (new expressions that must now be included in-scope):
  {inclusion_update}
- exclusion_update (boundaries to clarify so scope does not overextend):
  {exclusion_update}
</inputs>

Follow these instruction exactly and in order. Do not skip or reorder any instruction.

<coding_instructions>
MODIFICATION INSTRUCTIONS:
{modification_instructions}

LABEL INSTRUCTIONS:
    - ≤ 10 words
    - Active/actionable formulation of ONE ATOMIC theme in relation to the research question
    - Atomic means "One domain or aspect only"
    - If verb is used → one main verb (present tense).
    - **Never** include reasons (no "to", "so that", "because").
    - Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
    - Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

DEFINITION INSTRUCTIONS:
    - ≤ 30 words
    - grounded in theme description
    - Must describe **what belongs in this code**, not why it happens.
    - Must align directly with the survey question.
    - Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
    - Do not explain causes, conditions, or interpretations.
    - GOOD DEFINITION PATTERNS:
        • "References to…"
        • "Mentions of…"
        • "Expressions of…"
        • "Concerns about…"

ASSIGNMENT EXAMPLES INSTRUCTIONS:
    - Update assignment examples to reflect the modified code:
      • inclusion: Combine original + new expressions from inclusion_update
      • exclusion: Combine original + new boundaries from exclusion_update
      • near_neighbor: Update label/rule if boundaries changed due to modification
</coding_instructions>

OUTPUT FORMAT (valid JSON only, no commentary, in {language}):

{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": {source_code},
    "code_label": "yur new/modified code label in {language}",
    "code_definition": "your definition in {language}",
    "assignment_examples": {{
      "inclusion": ["[updated inclusion examples combining original + new in {language}]"],
      "exclusion": ["[updated exclusion examples combining original + new in {language}]"],
      "near_neighbor": {{
        "label": "[updated or original neighbor label in {language}]",
        "tell_apart_rule": "[updated or original tell-apart rule in {language}]"
      }}
    }}
  }}
}}

REQUIREMENTS:
- Output must be valid JSON only.
- No commentary outside JSON.
- If hierarchical_parent_diff_motive_same_family → ensure parent label is conceptual, not descriptive or repetitive.
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

Further information about the new code:
- The new code should cover these expressions:
  {inclusion_examples}
- The new code should NOT cover these expressions:
  {exclusion_examples}
- near_neighbor: {near_neighbor_label} (Tell apart: {tell_apart_rule})
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
   • Does it avoid forbidden punctuation: "/", "&", "+", ",", ";", ":", "-", "–" (unless the punctuation is lexicalized within a compound noun (e.g., 'gebruiksklaar-product' is allowed).")?
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

Step 5: Validate Assignment Examples:
- Ensure inclusion examples align with the refined code/definition
- Ensure exclusion examples maintain clear boundaries
- Verify near_neighbor and tell_apart_rule are still accurate
- Refine if needed to match validated code

Step 6: Determine your final components:
- validated_decision: USE, MODIFY, or CREATE code
- source_code:
    - If USE, this exact code: {source_code}
    - If MODIFY, the exact code from the existing codebook you seek to modify
    - If CREATE, write "null"
- validated_code and validated_decision: final compliant label, definition, and assignment_examples
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
      "definition": "Final validated definition (≤25 words, operational, grounded)",
      "assignment_examples": {{
        "inclusion": ["[validated/refined inclusion examples in {language}]"],
        "exclusion": ["[validated/refined exclusion examples in {language}]"],
        "near_neighbor": {{
          "label": "[validated neighbor label in {language}]",
          "tell_apart_rule": "[validated tell-apart rule in {language}]"
        }}
      }}
    }}
  }}
}}

If USE, this exact code: {source_code}; If MODIFY, the exact code from the existing codebook you seek to modify - or null, if CREATE

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""
