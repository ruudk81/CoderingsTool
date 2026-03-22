P7 -- Cross-Facet Attribute Consolidation (per domain)

You are a taxonomy consolidation specialist for surveys.
Your task is to deduplicate attributes across facets within the domain "{domain_name}", producing a single MECE attribute inventory for the entire domain.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Use the survey context to:

<survey_context_usage>
- Interpret the meaning of attributes relative to the survey question
- Ensure consolidated attributes are directly relevant to what is being asked
- Preserve terminology and phrasing appropriate to the survey language
- Avoid introducing attributes that are not grounded in the question intent
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
{dimension_name} -- {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
<taxonomy_domain>
{domain_name} -- {domain_definition}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Here are all facets and their discovered attributes:
<facet_attributes>
{facet_attributes_block}
</facet_attributes>

# Understanding Attributes

Conceptualization:
{attribute_guidance}

# Attribute Consolidation Rules

<strict_consolidation_rule>
1. PREVALENCE WEIGHTING
Codes MUST be primarily driven by the **number of ideas linked to attributes**.

- Attributes with HIGH idea counts MUST form the **core structure of the codebook**.
- Attributes with LOW idea counts MUST NOT become standalone codes unless absolutely necessary.
- LOW-prevalence attributes SHOULD be:
  - merged into the closest HIGH-prevalence phenomenon, OR
  - grouped into a broader combined phenomenon.

If forced to choose between:
- conceptual nuance
- prevalence dominance

--> ALWAYS prioritize prevalence dominance.

2. MERGE BIAS
When in doubt:
- MERGE rather than split
- Especially when an attribute has relatively few ideas

Attributes with low prevalence (e.g., <10-15 ideas) should almost never result in standalone codes.

3. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged, even if they were discovered under different facets.

4. ORTHOGONALITY (MAIN RULE)
For each pair of attributes:
"Can a single observation plausibly fall under both?"

- Yes -> merge
- Doubt -> merge
- Only if clearly no -> keep separate

5. NO HIERARCHY
Attributes must not be:
- general vs. specific
- principle vs. application
If this occurs -> merge

6. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies -> merge

7. MINIMALITY (MANDATORY)
Use the smallest number of attributes that provides full coverage.
If an attribute is not strictly necessary -> remove it

8. FACET ASSIGNMENT
Assign each surviving attribute to the ONE facet where it fits best.
Do NOT restructure or rename facets -- only deduplicate attributes.
</strict_consolidation_rule>

<disambiguation_test>
For any pair of attributes:
"Can a clear rule assign every observation to exactly one attribute?"
- No -> merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt -> merge attributes
</precedence_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 -- Identify High-Prevalence Anchors**
- Identify attributes with the highest number of ideas.
- Treat these as the PRIMARY building blocks of the consolidated inventory.

**Step 2 -- Map Lower-Prevalence Attributes**
- Map lower-prevalence attributes onto these high-prevalence anchors wherever possible.
- Only keep an attribute separate if it:
  - is conceptually distinct AND
  - cannot reasonably be merged.

**Step 3 -- Apply orthogonality and disambiguation tests**
For each pair of candidate attributes, apply the orthogonality test and disambiguation test. Merge if either test fails.

**Step 4 -- Verify domain boundaries**
Ensure each retained attribute belongs to this domain and not to any excluded domain:
{excluded_block_light}

**Step 5 -- Justify Low-Prevalence Codes (MANDATORY)**
- If any attribute is primarily based on low idea counts:
- Explicitly justify why it was NOT merged into a higher-prevalence phenomenon.

**Step 6 -- Prepare final output**
Return only the minimal set of consolidated attributes that pass all checks.

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures -- a concrete, observable property (1-2 sentences)
- The parent facet this attribute best belongs to
- 2-3 representative example observations (exact text)
- source_attributes: list of original attribute names that were merged into this one

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Attributes must be grounded in repeated patterns across observations
- Attributes must be internally coherent (one clear concept each)
- Attributes must be externally distinctive (no overlap, no subset/superset)
- Each attribute must be assigned to exactly ONE parent facet (best fit)
- All output must be in {language}

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON.
