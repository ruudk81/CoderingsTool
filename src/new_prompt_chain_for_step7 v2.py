#prompt 1: current RESPONSE_SUMMARY_PROMPT 
CLUSTER_SUMMARY_PROMPT= """
You are a {language} qualitative analyst who treats survey data as stories waiting to be told.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<cluster_text>
{cluster_text}
</cluster_text>

────────────────────────────────────────
ANALYTIC GUIDANCE
────────────────────────────────────────
1. **Immerse yourself** in the cluster; read every response as if it were
   a line in a short story.

2. **Active theme construction**
   ▸ Ask: “What overarching narrative connects these voices?”  
   ▸ Look for latent meaning beneath surface wording.  
   ▸ Push for unity first; fragment only when unity truly fails.

3. **Theme criteria**
   ▸ A *theme* is one indivisible idea that helps answer the
     survey question and appears in ≥{MIN_THEME_PCT}% of responses.  
   ▸ Multiple themes are allowed only when they are clearly distinct
     *and* each meets the same coverage rule.

4. **Reflection checkpoint**  
   Before finalising, write a one-sentence *analyst_note* on how you
   decided whether the cluster is unified or fragmented.

5. **Theme IDs**  
   Number themes sequentially: 1, 2, 3 …

6. **No coherent theme?**  
   If nothing meets the ≥{MIN_THEME_PCT}% threshold, mark the cluster as
   **noise**.

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text)
────────────────────────────────────────
{
  "cluster_summary": "<≤25-word synopsis capturing the cluster’s gist>",
  "analyst_note":     "<your single-sentence reflection>",
  "themes": [
    {
      "theme_id":      1,
      "statement":     "<5-15-word concise label>",
      "rationale":     "<why this theme matters to the survey question>",
      "coverage_pct":  <integer 0-100>   // optional but encouraged
    }
    // ...repeat for each theme
  ]
}

/* If classified as noise, return: */
{
  "cluster_summary": "This cluster lacks a coherent theme.",
  "analyst_note":    "Explained why no concept met the coverage rule.",
  "themes": []
}
"""


CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative analyst tasked with mapping each theme onto
the best-fitting codes already present in the master codebook.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<cluster_summary>
{cluster_summary}
</cluster_summary>

<existing_codebook>
{code_text}
</existing_codebook>

────────────────────────────────────────
ANALYTIC GUIDANCE
────────────────────────────────────────
1. Work theme-by-theme  
   • For every theme_id in cluster_summary.themes, reread its statement and
     rationale to grasp the core concept.

2. Rate candidate codes  
   • strong    ≥ 80 % semantic overlap  → definitely use  
   • partial  40-79 % overlap         → consider if it fills a gap  
   • weak    < 40 % overlap           → ignore

3. Select codes  
   • Include all *strong* matches.  
   • Add *partial* matches only when they jointly cover uncoded aspects.  
   • Never include *weak* matches.

4. Identify coverage gaps  
   • If any slice of a theme remains uncoded, list it in 3-7 words under
     `uncovered_concepts`.

5. Reflection  
   • Add an `analyst_note` (≤ 25 words) per theme explaining key choices or
     trade-offs.

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text)
────────────────────────────────────────
{
  "code_selection": [
    {
      "theme_id": 1,
      "matches": [
        {
          "code": "<exact name from codebook>",
          "definition": "<exact definition from codebook>",
          "fit_level": "strong | partial",
          "fit_comment": "<why this code fits>"
        }
        // 0-many
      ],
      "uncovered_concepts": [
        "<facet of theme still uncovered>"
        // 0-many
      ],
      "analyst_note": "<single-sentence rationale>"
    }
    // …repeat for every theme_id
  ]
}
"""

CODE_GENERATION_PROMPT = """
You are a {language} coding-scheme designer whose goal is to ensure every
theme is represented by exactly one atomic, non-redundant code—either by
reusing an existing code, modifying one, or creating a new code.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<cluster_summary>                <!-- output of Prompt 1 -->
{cluster_summary}
</cluster_summary>

<code_selection>                 <!-- output of Prompt 2 -->
{code_selection}
</code_selection>

────────────────────────────────────────
ANALYTIC GUIDANCE
────────────────────────────────────────
1. Theme-by-theme review  
   • For each theme_id, inspect its `matches`, `fit_level`s and any
     `uncovered_concepts`.

2. Decide the coding action (per theme)  
   Use the table below as guidance—not a hard rule—documenting your rationale
   if you override the default threshold.

   ┌────────────────────┬──────────────────────────────────────────────┬───────────────┐
   │ decision           │ when to choose (guideline)                  │ key fields    │
   ├────────────────────┼──────────────────────────────────────────────┼───────────────┤
   │ use_existing       │ A single code covers ≳90 % of the theme      │ codes_to_use  │
   │ modify_existing    │ One code covers 60-90 % but needs a tweak    │ code_to_modify│
   │ create_new         │ No code covers ≥60 % of the theme            │ new_code_*    │
   └────────────────────┴──────────────────────────────────────────────┴───────────────┘

3. Atomicity & parsimony checks  
   • Each final code must express ONE idea—no “and/with”.  
   • Avoid duplicates; if two candidate codes overlap, pick or merge, but
     never leave both.

4. Reflection  
   • Add an `analyst_note` (≤ 25 words) per theme summarising tricky
     judgement calls.

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text)
────────────────────────────────────────
{
  "cluster_analysis": {
    "theme_count": <integer>,
    "theme_descriptions": [
      "<theme 1 statement>",
      "<theme 2 statement>",
      …
    ]
  },
  "coding_decisions": [
    {
      "theme_id": 1,
      "decision": "use_existing | modify_existing | create_new",
      "action_details": {
        "codes_to_use":       ["<code name>", …],       // for use_existing
        "code_to_modify":     "<code name or null>",    // for modify_existing
        "modified_code_name": "<new name or null>",
        "modified_code_definition": "<definition or null>",
        "new_code_name":      "<name or null>",         // for create_new
        "new_code_definition":"<definition or null>"
      },
      "justification": "<why this choice best covers the theme>",
      "analyst_note": "<≤25-word reflection>"
    }
    // …repeat for every theme_id
  ],
  "overall_justification": "<how the set of decisions keeps the codebook atomic, non-redundant and parsimonious>"
}
"""

VALIDATION_PROMPT = """
You are a {language} QA auditor.  
Your mission: rigorously vet each coding decision so that every theme is
covered by one atomic, semantically accurate, non-redundant code.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<cluster_summary>       
{cluster_summary}
</cluster_summary>

<code_selection>       
{code_selection}
</code_selection>

<coding_decisions>      
{coding_decisions}
</coding_decisions>

────────────────────────────────────────
EVALUATION RUBRIC  (apply per theme)
────────────────────────────────────────
a) **Theme separation**  – Are themes truly distinct?  
b) **Semantic fit**      – Does the proposed code capture the theme?  
c) **Atomicity**         – Is the code one indivisible idea?  
d) **Parsimony**         – Is the simplest adequate option chosen?  
e) **Redundancy**        – Does the code overlap others?

Allowed judgements per code  
    APPROVE · REVISE · REJECT · MERGE · SPLIT

────────────────────────────────────────
VALIDATION STEPS
────────────────────────────────────────
1. Work theme-by-theme, referencing the original theme statement.  
2. Evaluate the chosen action against the rubric.  
3. If changes are needed, provide the corrected `validated_code`.  
4. Add a concise `decision_rationale` (≤40 words).  
5. When you REVISE, supply an improved name/definition.  
6. When you MERGE or SPLIT, describe what to merge/split and why.

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text)
────────────────────────────────────────
{
  "theme_assessment": {
    "theme_count_identified": <integer>,   // from coding_decisions
    "theme_separation_valid": true | false,
    "theme_separation_reasoning": "<why themes should stay separate or merge>"
  },

  "code_validations": [
    {
      "theme_id": 1,
      "original_decision": { …exact object from coding_decisions… },
      "evaluation": {
        "semantic_fit":   "good | weak | poor",
        "atomicity":      "good | split_needed",
        "parsimony":      "good | excessive",
        "redundancy":     "none | overlaps_with_<code>"
      },
      "decision": "APPROVE | REVISE | REJECT | MERGE | SPLIT",
      "decision_rationale": "<≤40-word explanation>",
      "validated_code": {
        "code": "<final code name>",
        "definition": "<final definition>"
      },
      "analyst_note": "<optional reflection on tricky judgement (≤25 words)>"
    }
    // …repeat for every theme_id
  ],

  "overall_validation": {
    "all_themes_coded": true | false,
    "final_code_count": <integer>,
    "summary": "<≤40-word wrap-up of the entire audit>"
  }
}
"""
