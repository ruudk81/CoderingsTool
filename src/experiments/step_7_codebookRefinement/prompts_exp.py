"""
Experimental Prompts for Step 7: Codebook Refinement

This file contains the prompts used by codebookRefinement.py.
Modify these prompts to experiment with different codebook refinement approaches.

Original source: src/prompts.py (STEP 7: THEME ORGANIZATION section)
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
- All text in the specified output language

Begin organizing the codebook.
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
- Code labels: Keep original labels, <=10 words
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

Begin consolidating the codebooks.
"""
