# Step 3 ideaExtractor — Work To Be Done

---

## Job 1: Audit and sharpen dimension_data.py instructions + enforce MECE in discovery and consolidation prompts

### Context

Step 3 discovers domains, and step 4 discovers facets and attributes within those domains. If the domain instructions in `dimension_data.py` are poorly calibrated, the LLM produces overlapping domains — and that overlap cascades into facets, attributes, and codes downstream.

Known example: ATTRIBUTES_ASSOCIATIONS has `domain_diagnostic = "What entity has the trait?"` — this fails for single-entity brand surveys where the entity is always the same (e.g., ASN Bank). The LLM falls back to topic-based domains ("financiële aspecten", "marketing en communicatie") instead of orthogonal attribute types.

### Part A: Audit and sharpen the analytical lens

Review the `PromptRules` for all 11 dimensions in `dimension_data.py`. For each dimension, check whether `domain_instruction`, `domain_diagnostic`, `facet_instruction`, `facet_diagnostic`, `attribute_instruction`, and `attribute_diagnostic`:

1. Clearly define the **lens** through which that taxonomy level should be carved — what question does this level answer for this specific dimension?
2. Are calibrated for realistic survey contexts (single-entity, multi-entity, etc.)
3. Would produce categories that a human coder could assign without hesitation

Scope: 11 dimensions × 3 taxonomy levels (domain, facet, attribute) = 33 instruction sets to audit. Not all will need changes.

Output: updated `dimension_data.py` with sharpened instructions. No prompt builder changes needed — the builders format from these fields.

### Part B: Enforce MECE in discovery and consolidation prompts

Separately from Part A, strengthen the discovery and consolidation prompts to be explicit about what "mutually exclusive" means in operational terms. This applies to **all** discovery and consolidation prompts, not just dimensions that need lens fixes:

**Prompts to update:**
- `build_domain_discovery_prompt` (step 3)
- `build_domain_consolidation_prompt` (step 3)
- `build_facet_discovery_prompt` (step 4, Job 2 in step 4's work_to_be_done)
- `build_facet_consolidation_prompt` (step 4, Job 2 in step 4's work_to_be_done)

**MECE criteria to add explicitly:**
- **Ontological distinctness** — categories at each level must not share conceptual space. A domain should not be a subset of another domain, and two domains should not be two lenses on the same phenomenon.
- **Semantic distance** — categories must be different enough that a coder assigning a response to a category wouldn't plausibly consider a neighboring category. No "could go either way" situations.

### Relationship to step 4 work

This job is a prerequisite for step 4's Jobs 2 and 3 (facet and attribute MECE enforcement). Clean domains from step 3 → clean facets in step 4 → clean attributes in step 4. The cascade works top-down.

### Files to modify

| File | Change |
|------|--------|
| `src/development/step_3_ideaExtractor/dimension_data.py` | Part A: sharpen PromptRules for all 11 dimensions |
| `src/development/step_3_ideaExtractor/prompts_exp.py` | Part B: add MECE criteria to `build_domain_discovery_prompt` and `build_domain_consolidation_prompt` |

---

## Jobs completed

### Job 1 Part A — Completed 2026-03-17

**Changes made to `dimension_data.py`:**

**Domain instruction rewrites (7 dimensions):**
- ATTRIBUTES_ASSOCIATIONS: "What entity has the trait?" → "What dimension of the entity is being described?"
- IDENTITY_DEFINITION: "What is being defined?" → "Which dimension of identity is being described?"
- ACTORS_TARGETS: "In what situation are actors involved?" → "In what sphere of activity are actors involved?"
- CONTEXT_CONDITIONS: "What situation is being discussed?" → "What type of contextual environment is described?"
- MOTIVATIONS_DRIVERS: "What is the motivation about?" → "What area of life or concern is the motivation about?"
- EVALUATION_PRIORITIZATION: "What object is evaluated?" → "What aspect of the entity is being evaluated?"
- RELATIONS_DEPENDENCIES: "What entities are involved?" → "In what sphere does this relationship exist?"

**Facet diagnostic de-anchoring (all 11 dimensions):**
Removed illustrative examples from all facet diagnostics. Kept only the abstract question to prevent anchoring bias.

**No changes to:** attribute instructions (all clean), 4 well-calibrated domain instructions (PRESCRIPTIVE_CHANGE, EXPERIENCE_PERCEPTION, BEHAVIOR_FUNCTION, GENERAL_OTHER).

**Backup:** `dimension_data_OLD.py`

**Status:** Needs validation by re-running step 3 on ASN Bank dataset.
