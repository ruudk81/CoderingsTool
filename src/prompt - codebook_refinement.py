CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative researcher and codebook methodologist. 
Your task is to take a raw list of descriptive codes and transform it into a refined and structured codebook. 
The descriptive codes are derived from survey responses.

<inputs>
Language to use: {language}

Raw descriptive codes to refine:
{raw_codes}
</inputs>

<guidance>
A high-quality codebook must be:
- Non-redundant: no codes that repeat the same idea in different words.  
- Semantically differentiated: each code uses distinct wording; small overlap is acceptable.  
- Inherently distinct: codes within the same category are not identical in content.  
- Parsimonious: no unnecessary duplication; concise but comprehensive.  
- Structured: grouped under 6–8 main categories with consistent subcodes.  
- Consistently named: short, uniform, and meaningful labels.  

When codes overlap semantically or thematically:
- Merge them into a single code with a clear, inclusive label.  
- If codes are near-identical, consolidate into one code and discard duplicates.  
- If codes represent sub-aspects of the same domain, nest them as subcodes.  
- If codes are vague, reword them for clarity.  

</guidance>

<analysis_steps>
1. Review all raw codes.  
   - Identify redundant codes (semantic duplicates, identical meaning).  
   - Identify overlapping codes that can be merged into broader codes with subcodes.  

2. Construct main categories that represent broad domains.

3. Assign refined codes as subcodes under these main categories.  
   - Each subcode must represent ONE distinct, actionable concept.  
   - Remove vague or overly broad codes.  

4. Ensure consistent naming:  
   - Labels ≤ 8 words.  
   - Active or descriptive phrasing.  
   - No repetition across categories.  

5. Document the restructuring:  
   - How many raw codes were merged?  
   - Which semantic duplicates were consolidated?  
   - Which main categories were created?  

</analysis_steps>

Provide your response as a valid JSON dictionary using this exact structure:
{{
  "analysis": "Provide your analysis here in {language} (describe main restructuring decisions, what was merged, how categories were formed).",
   "refined_codebook": [
      {{
        "category": "Main category label",
        "subcodes": [
          {{
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
