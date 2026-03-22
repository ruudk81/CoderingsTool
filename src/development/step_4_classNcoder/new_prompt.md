# P4: Code Generation from Attributes — New Prompt

This file contains the complete specification: function, prompt template, and response schema.
Review all three before migrating to `prompts_exp.py`.

---

## 1. Prompt Template

```
You are tasked with deriving a PARSIMONIOUS codebook with MUTUALLY EXCLUSIVE and COLLECTIVELY EXHAUSTIVE codes that represent conceptually and semantically distinct PHENOMENA from a taxonomy inventory of attributes. These attributes were derived from written responses to a survey question.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

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

Here is the taxonomy inventory of attributes organized by facet:

<taxonomy_inventory>
The inventory below is organized by Facet > Attribute
{inventory_block}
</taxonomy_inventory>

# Understanding Phenomena vs Attributes

**Attributes** are specific observations or qualities mentioned in responses. They represent individual data points.

**Phenomena** are underlying conceptual patterns that multiple attributes may indicate. A phenomenon is the broader experience, perception, or association that manifests through various specific attributes.

Your task is to identify phenomena, NOT to create one code per attribute.

# Code Derivation Rules

## 1. Phenomenon Rule
Codes must represent underlying PHENOMENA rather than individual attributes. Multiple attributes describing different manifestations of the same underlying experience, perception, or association MUST be merged into a single code.

## 2. Specificity Rule
Do NOT create separate codes simply because attributes differ in specificity. General statements and specific examples should be treated as indicators of the same phenomenon.

Example: "The train was delayed by 20 minutes" and "public transport is often late" both indicate unreliable punctuality and should be coded under the same broader phenomenon.

## 3. Example-Level Rule
Do NOT create codes that represent specific items or examples. These should be treated as indicators of broader phenomena.

## 4. Attribute Mapping Rule
Do NOT create a separate code for each attribute. Attributes are observations that may belong to the same phenomenon.

## 5. Minimum Coverage Rule
Each code should ideally cover multiple attributes. Only create a single-attribute code if the phenomenon is clearly distinct and cannot be meaningfully merged with others.

## 6. Parsimony Rule
Prefer broader phenomenon-based codes over narrow attribute-based codes. Use the smallest number of codes that still capture all distinct phenomena present in the inventory.

## 7. Expected Code Range
The final codebook should normally contain 3–5 codes unless the attributes clearly describe more distinct phenomena.

## 8. Mutual Exclusivity Rule
Codes must represent clearly different phenomena so that responses can be coded consistently without ambiguity.

## 9. Valence Sensitivity Rule
Generate separate codes for positive and negative phenomena. Do NOT combine praise and criticism into a single code. If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.

## 10. Hierarchy Rule
Only use attribute content to derive codes. Do NOT create codes directly from domain or facet labels. Facets are organizational structures; your codes should emerge from the actual attribute patterns.

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 — Identify Underlying Phenomena**
Review all attributes across all facets. Look for patterns where multiple attributes describe different manifestations of the same underlying phenomenon. Group attributes that share the same conceptual core.

**Step 2 — Check for Valence Distinctions**
Within each phenomenon group, check whether positive and negative valences are present. If so, split into separate codes.

**Step 3 — Name Each Phenomenon**
Assign a descriptive name (3-5 word noun phrase in {language}) to each distinct phenomenon.

**Step 4 — Verify Parsimony and Coverage**
Ensure you have the minimum number of codes needed while covering all attributes. Aim for 3-5 codes unless the data clearly requires more.

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (code names, definitions, typical indicators, and evaluation) must be written in {language}.

# Final Notes

Remember: You are creating a PARSIMONIOUS codebook. Resist the temptation to create one code per attribute or per facet. Look for the deeper phenomena that connect multiple attributes together. Your goal is conceptual clarity with minimal redundancy.

Begin your analysis now.
```

---

## 2. Function

```python
def build_code_from_attributes_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional['DimensionDefinition'],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    domain_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
    attribute_assignments: Optional[Dict[str, str]] = None,
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Generate codebook codes from a structured attribute inventory.

    Args:
        dimension_def: DimensionDefinition for taxonomy structure lines (or None for fallback)
        domain_name: Name of the domain being processed
        domain_definition: Inclusion definition of the domain
        domain_attributes: {domain_name: {facet_name: [DiscoveredAttribute, ...]}}
        attribute_assignments: idea_id -> attribute_name, for frequency display
        excluded_domains: list of (name, definition) for other domains
    """
    # Dimension-specific taxonomy structure
    if dimension_def:
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(dimension_def.prompt_rules.domain_instruction)
        facet_key_idea = _extract_key_idea(dimension_def.prompt_rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(dimension_def.prompt_rules.attribute_instruction)
    else:
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        facet_key_idea = "the analytical lens applied to the subject"
        attribute_key_idea = "the specific observable property being described"

    # Excluded domains block
    excluded_block = ""
    if excluded_domains:
        excl_lines = [
            f"- {excl_name} — {excl_def}"
            for excl_name, excl_def in excluded_domains
        ]
        excluded_block = (
            "\nYou must NOT include categories that belong to these excluded domains:\n"
            "<excluded_domains>\n"
            + "\n\n".join(excl_lines)
            + "\n</excluded_domains>"
        )

    # Compute attribute frequencies
    attr_counts: Dict[str, int] = {}
    if attribute_assignments:
        for attr_name in attribute_assignments.values():
            attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

    # Build inventory: Facet > Attribute (single domain)
    facet_attrs = next(iter(domain_attributes.values()), {})
    inventory_lines = []
    for facet_name, attributes in sorted(facet_attrs.items()):
        inventory_lines.append(f"\n{facet_name}")
        for attr in attributes:
            examples = "; ".join(attr.example_observations[:2])
            count = attr_counts.get(attr.attribute_name, 0)
            freq_tag = f" [{count} ideas]" if attr_counts else ""
            line = f"- {attr.attribute_name}{freq_tag}: {attr.attribute_description}"
            if examples:
                line += f" (e.g., {examples})"
            inventory_lines.append(line)
    inventory_block = "\n".join(inventory_lines)

    return f"""<PROMPT TEMPLATE FROM SECTION 1>"""
```

---

## 3. Response Schema (Pydantic models)

```python
class CodeFromAttributes(BaseModel):
    """A formal qualitative code derived from attributes."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description="Clear definition of what this code covers (1-2 sentences)"
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Attribute names this code is derived from (exactly as they appear in the inventory)"
    )


class CodeGenerationFromAttributesResult(BaseModel):
    """P4 output: codes derived from attributes."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before deriving codes: "
            "(1) identify underlying phenomena by grouping attributes, "
            "(2) check for valence distinctions, "
            "(3) name each phenomenon, "
            "(4) verify parsimony and coverage"
        )
    )
    evaluation: str = Field(
        ..., description="Brief evaluation of how codes were derived from attributes — what was merged and why"
    )
    codes: List[CodeFromAttributes] = Field(
        ..., description="Formal codes derived from the attribute inventory"
    )
```

---

## 4. Caller Changes Required

**`_run_code_generation_from_attributes()`** — add parameters:
- `domain_name: str`
- `domain_definition: str`
- `excluded_domains: Optional[List[Tuple[str, str]]]`

**`_process_codebook_async()`** — at the call site, provide:
- `domain_name=domain_name`
- `domain_definition=partition_contexts[domain_name].partition_definition`
- `excluded_domains=[(n, partition_contexts[n].partition_definition) for n in partition_contexts if n != domain_name]`
- `dimension_def=prompt_context.dimension_def`
