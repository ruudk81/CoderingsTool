P2 — Facet Consolidation (per domain)

You are a taxonomy consolidation specialist for surveys.
Your task is to merge multiple chunk-level facet analyses into a single, minimal set of mutually exclusive facets within a given domain.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Use the survey context to:

<survey_context_usage>
- Interpret the meaning of facets relative to the survey question
- Ensure consolidated facets are directly relevant to what is being asked
- Preserve terminology and phrasing appropriate to the survey language
- Avoid introducing facets that are not grounded in the question intent
</survey_context_usage>

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name}: {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} — {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Here are the facets you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

# Understanding Facets

Conceptualization:
{facet_guidance}

# Facet Consolidation Rules

<strict_consolidation_rule>
1. MERGE OVERLAP (MANDATORY)
All facets that conceptually overlap or are variants of the same idea must be merged.

2. ORTHOGONALITY (MAIN RULE)
For each pair of facets:
"Can a single observation plausibly fall under both?"

- Yes → merge
- Doubt → merge
- Only if clearly no → keep separate

3. NO HIERARCHY
Facets must not be:
- general vs. specific
- principle vs. application
If this occurs → merge

4. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies → merge

5. MINIMALITY (MANDATORY)
Use the smallest number of facets that provides full coverage.
If a facet is not strictly necessary → remove it
</strict_consolidation_rule>

<disambiguation_test>
For any pair of facets:
"Can a clear rule assign every observation to exactly one facet?"
- No → merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt → merge facets
</precedence_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 — Scan chunk-level facets**
Review all facets from all chunks. Note recurring themes and obvious duplicates.

**Step 2 — Group overlapping facets**
Group facets that describe the same or overlapping concepts across chunks.

**Step 3 — Apply orthogonality test**
For each pair of candidate consolidated facets, ask: "Can a single observation plausibly fall under both?" If yes or doubtful → merge.

**Step 4 — Apply disambiguation test**
For each pair: "Can a clear rule assign every observation to exactly one facet?" If no → merge.

**Step 5 — Verify domain boundaries**
Ensure each retained facet belongs to the included domain and not to any excluded domain:
{excluded_block_light}

**Step 6 — Prepare final output**
Return only the minimal set of consolidated facets that pass all checks.

For each consolidated facet, provide:
- A short descriptive name (2-5 words)
- A description of what the facet captures (1-2 sentences)
- 3-5 representative observations selected from across the merged chunks (exact text)

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All facet names and descriptions must be in {language}.

# Final Notes

- Facets must be descriptive, not evaluative
- Facets must be grounded in repeated patterns across observations
- Facets must be internally coherent (one clear concept each)
- Facets must be externally distinctive (no overlap, no subset/superset)
- Facets must remain strictly within the included domain
- All output must be in {language}

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON.
