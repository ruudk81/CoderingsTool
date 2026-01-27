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

Before beginning your analysis, understand these key definitions: 
<key_definitions>
- TAXONOMY AXIS: This is the conceptual dimension along which all themes will be organized. It defines what kind of phenomenon we are theming. It sets the “coordinate system” for the analysis.
- CODING DIMENSION: This is the specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: This is a central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: This indicates how conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
- NEAR-NEIGHBOR: This is a conceptually adjacent theme that could be easily confused with the atomic theme and therefore needs to be distinguished in the codebook.
</key_definitions>

You will be working in the following language:
<language>
{language}
</language>

All your analytical work, theme labels, definitions, and examples must be written in this language. Only JSON field names should remain in English.

Here is the cluster identifier you are analyzing:
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

Here is the taxonomy framework guiding your analysis:
<taxonomy>
Taxonomy axis: {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy>

Some constraints and strict rules you need to know about:
<label_constraints>
Label constraints (strict):
- Short noun phrase (≤ 10 words) of an ATOMIC theme in light of the research question, taxonomy axis and coding dimension.
- Make the semantic core of the theme the head of the noun phrase.
- Adhere to the atomicity rules below.
- DO NOT repeat the actor, domain, topic or entity in the label ({perspective}, {domain}, {topic} and {entity}). 
</label_constraints>

<atomicity_rules>
- Atomicity rules (strict):
  • All naming and labelling of ATOMIC THEMES must be single-valued. 
  • Do not output multi-labels, spans, or hybrids. 
  • When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
  • NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
- If a code mentions multiple aspects or mixed sentiment → split into separate atomic concepts.
</atomicity_rules>

<definition_constraints>
Definition constraints (strict):
- ≤ 30 words 
- grounded in cluster data
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question, taxonomy axis, and coding dimension.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
- DO NOT repeat the actor, domain, topic or entity in the description ({perspective}, {domain}, {topic} and {entity}). 
</definition_constraints>

Now follow these analysis steps exactly and in order. Do not skip or reorder any step:

<analysis_steps>
Step 1. Interpret descriptive codes in light of the taxonomy focus and research question:
    - Review each code in the cluster data  
    - Ask yourself: How does each code address the research question, taxonomy axis, and coding dimension?  
    - Identify what patterns are meaningful for analyzing concrete, actionable answers

Step 2. Remove outliers.
    - Eliminate codes that do not connect to any broader pattern across multiple codes.
    - Eliminate codes that do not represent a meaningful segment (too rare, irrelevant, or idiosyncratic).

Step 3. Identify COC(s):
    - Practical test: Can all codes be summarized in one sentence that:
     (a) aligns with <guidance>,
     (b) captures exactly ONE ATOMIC theme,
     (c) uses no coordinating conjunctions or list punctuation,
     (d) preserves unity, consistency, and contrast?
   - If yes → single COC. If no → multiple COCs.
   - Proceed with the resulting COC(s).

Step 4. Refine COCs:
    - Remove singletons (COCs based on only one code or a neglectable number of codes in full sumple).

Step 5: Create codebook entries
For each retained COC, produce a codebook-ready draft with:
  - theme_label: Follow all label constraints above
  - theme_clarification: Follow all definition constraints above; include illustrative descriptive codes from the cluster data that clarify and support the label 
  - abstraction_level: Select exactly one of: "Driver/Motive/Why" | "Attribute/What" | "Action/How"
  - assignment_examples (these are EXAMPLES, not exhaustive rules):
    • inclusion: Provide 2–3 short, positive example assignment expressions (each starts with a verb; use observable cues)
    • exclusion: Provide 1–2 short example boundary cases to prevent overreach (what must NOT be included)
    • near_neighbor:
      — label: The closest potentially-confusable theme, or "Unknown" if none exists
      — tell_apart_rule: One sentence explaining how to distinguish this theme from the neighbor (e.g., "This theme focuses on X (driver/what/how), whereas the neighbor focuses on Y.")

Step 6) Document the analysis:
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
You are a qualitative research assistant responsible for maintaining a structured codebook for thematic analysis following Braun & Clarke (2006) methodology. 
Your task is to classify a newly identified theme by comparing it against existing codes in the codebook.
You must ensure the codebook remains MECE (Mutually Exclusive, Collectively Exhaustive) by strictly adhering to the specified taxonomy structure.

You will be working with the following parameters:

<language>
 {language}
</language>

<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

<context>
- Domain: {domain}
- Topic: {topic}
Survey Question: "{survey_question}"
</context>

<new_theme>
New Theme to Classify:
- name: "{theme_name}"
- description: "{theme_description}"
- what's included:
    {inclusion}
</new_theme>

<existing_codes>
Existing Codes:
{code_text}
</existing_codes>

Your task is to classify the new theme by deciding which of the following actions to take:

**Decision Options:**
- **USE** — The theme is semantically identical to an existing code (no changes needed)
- **MODIFY_VERTICAL** — Broaden an existing code at the same abstraction level to include this theme as an additional expression within the primary coding dimension
- **MODIFY_HIERARCHICAL** — Create or reference a parent code that groups multiple related codes across different abstraction levels within the same conceptual family
- **CREATE** — Add a new code because the theme represents a distinct concept not covered by existing codes

**Analysis Framework:**

Follow these steps in order:

**STEP 1: Apply Cosine Similarity Rules**
- If cosine similarity ≥ 0.95 → Same meaning → Decision: USE
- If cosine similarity < 0.55 → Different meanings → Decision: CREATE
- If 0.55 ≤ cosine similarity < 0.95 → Proceed to STEP 2

**STEP 2: Semantic Test (Only if cosine similarity is 0.55–0.95)**

Evaluate the new theme against the single highest-similarity code by answering:

Q1 (Conceptual Family): Do the new theme and the highest-similarity code belong to the same conceptual family, given the research question, taxonomy axis, and primary coding dimension?

Q2 (Abstraction Level): Are the new theme and the highest-similarity code at the same abstraction level (Driver/Attribute/Action) on the taxonomy axis/coding dimension?

Apply these decision rules:
- If Same Family AND Same Abstraction Level → Decision: MODIFY_VERTICAL
  - Broaden the existing code's inclusion rules to cover the new expression
  - Keep the abstraction level unchanged
  - Maintain atomicity (one core concept per code)

- If Same Family AND Different Abstraction Level → Decision: MODIFY_HIERARCHICAL
  - Create or reference a parent code at a higher abstraction level
  - Treat the existing code and new theme as sub-codes within the same family

- If Not Same Family → Decision: CREATE

**STEP 3: Enforce Structural Constraints**
- Codes must remain atomic: one idea, one motive, one sentiment
- Inclusion rules describe when to assign the theme
- Exclusion rules describe common misfits to maintain clear boundaries
- The codebook must remain MECE

**STEP 4: Determine Assignment Example Updates**

If decision is **USE**:
- Preserve original assignment_examples unchanged

If decision is **MODIFY_VERTICAL** or **MODIFY_HIERARCHICAL**:
- inclusion: Combine original + new expressions from the theme
- exclusion: Combine original + new boundary clarifications if needed
- near_neighbor: Update label if boundaries shifted due to modification
- tell_apart_rule: Update if the distinction from neighbor changed

If decision is **CREATE**:
- Use assignment_examples from the new theme as-is

Before providing your final answer, use <scratchpad> tags to work through your analysis systematically:

1. Identify the top candidate codes based on semantic similarity
2. Note the cosine similarity scores for top candidates
3. Apply the cosine similarity rules from STEP 1
4. If needed (cosine 0.55-0.95), perform the semantic test from STEP 2
5. Determine your decision (USE/MODIFY/CREATE) and provide justification referencing the cosine score and conceptual family/abstraction level analysis
6. Plan what updates are needed to assignment examples based on your decision

After completing your analysis in the scratchpad, provide your final answer as valid JSON only inside <json_output> tags.

The JSON must follow this exact structure:

{{
  "coding_decision": {{
    "theme_number": {theme_id},
    "theme_name": {theme_name},
    "matched_candidates": [
        {{"code": "Exact candidate code A", "definition": "Definition in light of the survey question"}},
        // Add additional candidaties,if there are any
        ]
    "decision": "USE | MODIFY_VERTICAL | MODIFY_HIERARCHICAL | CREATE",
    "source_code": "Exact candidate code name if use/modify, or null if create",
    "modify_parameters":{{
       "modify_instruction": "vertical_broaden_same_level | hierarchical_parent_diff_level | none",
       "conceptual_family": "same | different",
       "abstraction_level": "same | different",
       "abstraction_level_action": "keep | broaden_to_parent | none",
       "inclusion_update": "null or concrete additions to inclusion rules",
       "exclusion_update": "null or concrete boundary clarifications",
       "parent_theme_label": "null or suggested parent label",
       "near_neighbor_label_update": "null or updated neighbor label if boundaries changed",
       "tell_apart_rule_update": "null or updated tell-apart rule if distinction changed"}},
    "justification": "Explain modification decision by referencing cosine score and conceptual family/abstraction level comparison, or null if use/create",
    "updated_assignment_examples": {{
      "inclusion": ["[updated or original inclusion examples in {language}]"],
      "exclusion": ["[updated or original exclusion examples in {language}]"],
      "near_neighbor": {{
        "label": "[updated or original neighbor label in {language}]",
        "tell_apart_rule": "[updated or original tell-apart rule in {language}]"
      }} | null
    }}.
  }}
}}

**Interpretation Examples:**
1. **MODIFY_VERTICAL Example**: cosine = 0.81, same conceptual family, same abstraction level → "This theme expresses the same underlying concept as Code A but in a new form. MODIFY_VERTICAL to broaden inclusion rules."
2. **MODIFY_HIERARCHICAL Example**: cosine = 0.80, same conceptual family, different abstraction level → "This theme shares the conceptual domain of Code A but at a different abstraction level. MODIFY_HIERARCHICAL to create parent code."
3. **CREATE Example**: cosine = 0.64, different conceptual family → "The meaning and conceptual family differ significantly. CREATE new code."

**Requirements:**
- Output must be valid JSON only (no additional commentary outside the json_output tags)
- Keep field names in English; write values in the language specified in codebook_parameters
- Include cosine score in justification
- Identify conceptual family and abstraction level comparison explicitly in justification
- Ensure all updates maintain MECE principles and code atomicity
"""

CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant.
Your task is to CREATE a new code that captures the meaning of a newly identified atomic theme from survey responses.

Before beginning your analysis, understand these key definitions:
<key_definitions>
- TAXONOMY AXIS: This is the conceptual dimension along which all themes will be organized. It defines what kind of phenomenon we are theming. It sets the “coordinate system” for the analysis.
- CODING DIMENSION: This is the specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: This is a central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: This indicates how conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
</key_definitions>

You will be working with the following parameters:

<language>
 {language}
</language>

<context>
- Domain: {domain}
- Topic: {topic}
Survey Question: "{survey_question}"
</context>

<new_theme>
New theme:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
</new_theme>

Here is the taxonomy framework guiding your analysis:
<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

LABEL RULES (strict):
- Short noun phrase (≤ 10 words) of an ATOMIC theme in light of the research question, taxonomy axis and coding dimension.
- Make the semantic core of the theme the head of the noun phrase.
- Adhere to the atomicity rules below.
- DO NOT repeat the actor, domain, topic or entity in the label ({perspective}, {domain}, {topic} and {entity}). 

ATOMICITY  RULES (strict):
- All categorical codes, labels and names  must be single-valued. 
- Do not output multi-labels, spans, or hybrids. 
- When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
- NEVER include reasons (no "to", "so that", "because").
- NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).

DEFINITION RULES:
- ≤ 30 words 
- grounded in cluster data
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question, taxonomy axis, and coding dimension.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
- DO NOT repeat the actor, domain, topic or entity in the description ({perspective}, {domain}, {topic} and {entity}). 

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
        (b) is grounded in the shared conceptual family and abstraction level,
        (c) remains expressible as **one idea** in the label.
   - The modified label must:
        • reflect the broadened meaning,
        • NOT introduce multiple aspects or abstraction levels,
        • NOT be more abstract than necessary.
   - The modified definition must:
        • describe the **shared meaning space**,
        • reflect: original inclusions + inclusion_update,
        • exclude: original exclusions + exclusion_update.
   - Do **not** modify assignment rules here."""

HIERARCHICAL_INSTRUCTIONS = """
   - Shared conceptual family but different abstraction levels → create hierarchical structure.
   - Original code and new theme remain **atomic child codes**.
   - Parent code represents the shared **conceptual family**.

   Parent label:
        - parent theme = {parent_theme_label}
        - If parent theme is not None or Null → use it as-is.
        - If null → generate a label at a higher abstraction level (Driver/Why level).
        - Must:
            • express shared conceptual family,
            • NOT describe behaviors/outcomes,
            • NOT blend child labels,
            • be broader, not vaguer.

   Structure:
       - Parent = conceptual anchor (higher abstraction level),
       - Children = distinct manifestations (different abstraction levels),
       - Child meanings **do not change**."""

CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new theme, while preserving **atomic meaning** and **clear conceptual boundaries**.

Before beginning your analysis, understand these key definitions:

<key_definitions>
- TAXONOMY AXIS: This is the conceptual dimension along which all themes will be organized. It defines what kind of phenomenon we are theming. It sets the “coordinate system” for the analysis.
- CODING DIMENSION: This is the specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: This is a central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: This indicates how conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
</key_definitions>

You will be working with the following parameters:

<language>
 {language}
</language>

<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

<context>
- Domain: {domain}
- Topic: {topic}
Survey Question: "{survey_question}"
</context>

New theme to integrate:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}

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

Here is the taxonomy framework guiding your analysis:
<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

LABEL RULES (strict):
- Short noun phrase (≤ 10 words) of an ATOMIC theme in light of the research question, taxonomy axis and coding dimension.
- Make the semantic core of the theme the head of the noun phrase.
- Adhere to the atomicity rules below.
- DO NOT repeat the actor, domain, topic or entity in the label ({perspective}, {domain}, {topic} and {entity}). 

ATOMICITY  RULES (strict):
- All categorical codes, labels and names  must be single-valued. 
- Do not output multi-labels, spans, or hybrids. 
- When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
- NEVER include reasons (no "to", "so that", "because").
- NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).

DEFINITION RULES:
- ≤ 30 words 
- grounded in cluster data
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question, taxonomy axis, and coding dimension.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
- DO NOT repeat the actor, domain, topic or entity in the description ({perspective}, {domain}, {topic} and {entity}). 

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
- If hierarchical_parent_diff_level → ensure parent label is conceptual, not descriptive or repetitive.
"""

VALIDATION_PROMPT = """
You are a codebook curator for thematic analysis following Braun & Clarke (2006) methodology. 
Your role is to maintain parsimonious codebooks with non-overlapping and non-redundant codes by reviewing and making final decisions on coding proposals. 
MECE is your guiding principle - codes must be Mutually Exclusive and Collectively Exhaustive.

Before beginning your task, understand these key definitions:

<key_definitions>
- TAXONOMY AXIS: The conceptual dimension along which all themes are organized. It defines what kind of phenomenon we are theming and sets the "coordinate system" for the analysis.
- CODING DIMENSION: The specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: A central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: How conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
</key_definitions>

Here is the codebook context you will be working with:

<codebook_context>
- Domain: {domain}
- Topic: {topic}

Existing codes in codebook:
{code_text}
</codebook_context>

Here is the coding proposal you need to evaluate:

<coding_proposal>
A new theme emerged from analyzing responses to this survey question:
"{survey_question}"

This is a new theme to be evaluated:
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
</coding_proposal>

<validation_criteria>
a. Parsimony: Has sufficient effort been made to use existing codes or combinations before proposing new/modified codes?
b. Abstraction Level: Is any proposed new code at an appropriate abstraction level, consistent with existing codes?
c. Non-Redundancy: Does the proposal avoid creating codes that significantly overlap with existing ones?
d. Atomicity: do labels adhere to these atomicity rules:
- All categorical codes, labels and names  must be single-valued. 
- Do not output multi-labels, spans, or hybrids. 
- When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
- NEVER include reasons (no "to", "so that", "because").
- NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
e. DO NOT repeat the actor, domain, topic or entity in the label or names ({perspective}, {domain}, {topic} and {entity}). 
</validation_criteria>

Your task is to systematically evaluate the coding proposal against the validation criteria and make a final coding decision. Work through your evaluation step-by-step in a scratchpad before providing your final response.

<scratchpad>
Work through your evaluation systematically following these steps:

**Step 1: Evaluate the CREATE/MODIFY/USE decision**
- Parsimony: Has sufficient effort been made to use existing codes or combinations before proposing new/modified codes?
- Abstraction Level: Is any proposed new code at an appropriate abstraction level, consistent with existing codes?
- Non-Redundancy: Does the proposal avoid creating codes that significantly overlap with existing ones?

**Step 2: Evaluate code label and definition quality**
- Atomicity:
  • Does it express only one idea (no merged/compound themes)?
  • Does it avoid forbidden punctuation: "/", "&", "+", ",", ";", ":", "-", "–" (unless lexicalized within a compound noun)?
  • Does it contain at most ONE main action (verb)?
  • Does it avoid conjunctions ("and/or") unless lexicalized?
  • Does it avoid reasons (no "to", "so that", "because")?

- Form & Length:
  • Is the label ≤10 words with no canonical subject from survey question and no implied actor?
  • Does it follow allowed forms: noun phrase (<adjective(s)> <noun>), imperative verb + object (<verb> <object>), or infinitive (<to/infinitive verb> <object>)?
  • Is the definition ≤25 words, operational/observable, grounded in responses, and non-vague?

- Alignment:
  • Does the label and definition align directly with the survey question, taxonomy axis, and coding dimension?
  • Does the definition describe **what belongs in this code**, not why it happens?
  • Does the definition use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments)?
  • Does it avoid repeating the actor, domain, topic or entity in the label or definition ({perspective}, {domain}, {topic} and {entity})?

**Step 3: Determine APPROVE or REJECT**
- If all criteria PASS → APPROVE (you may make minor refinements for full compliance)
- If any criterion FAILS → REJECT (identify issues and rewrite to comply)

**Step 4: If rejected, make final decision**
- **USE**: An existing code already captures the new theme's central meaning sufficiently
- **MODIFY**: An existing code is close but needs refinement for clarity, scope, or better alignment
- **CREATE**: No existing code sufficiently captures the new theme

**Step 5: Validate assignment examples**
- Ensure inclusion examples align with the refined code/definition
- Ensure exclusion examples maintain clear boundaries
- Verify near_neighbor and tell_apart_rule are still accurate
- Refine if needed to match validated code

**Step 6: Determine final components**
- validated_decision: USE, MODIFY, or CREATE
- source_code: If USE, the exact code from the proposal; if MODIFY, the exact code from the existing codebook you seek to modify; if CREATE, write "null"
- validated_code: final compliant label, definition, and assignment_examples
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
