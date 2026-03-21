# Step 5 classNcoder — Prompt Audit Checklist

Source: `prompts_exp.py`

Audit checks per prompt:
- **Formatting ↔ Prompt**: Function parameters/comments match what the prompt template actually uses
- **Schema ↔ Prompt**: Pydantic response model fields align with what the prompt instructs the LLM to produce
- **No dead code**: No unused parameters, variables, or schema fields

---

## §1 Dimension Context Block (shared helper)

**Function:** `build_dimension_context_block()`
**Parameters:** `dimension_def`, `dimension_name`, `dimension_description`, `domain_name`, `domain_definition`
**Returns:** XML `<taxonomy_context>` block (no response model — helper only)

Not a standalone prompt — used by §4, §7, §8.

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| No dead code | ☐ |

**Notes:** Also uses helpers `_extract_key_idea()`, `_build_exclusion_block()`.

---

## §2 Facet Discovery (P1)

**Function:** `build_facet_discovery_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `partition_name`, `partition_definition`, `observations`, `excluded_domains`
**Response model:** `FacetDiscoveryResult` → `List[DiscoveredFacet]`

Schema fields per `DiscoveredFacet`:
- `facet_name: str` — "Short descriptive name for the facet (2-5 words)"
- `facet_description: str` — "What this facet captures (1-2 sentences)"
- `example_observations: List[str]` — "3-5 representative observations from the input"

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Known issues:**
- `survey_question` — accepted but NOT interpolated in prompt
- `dataset_context_section` — accepted but NOT interpolated in prompt
- `partition_definition` — accepted but NOT interpolated in prompt (only `partition_name` is used)
- `dimension_name`, `dimension_description` — only used in fallback path when `dimension_def is None`

---

## §3 Facet Consolidation (P1.5)

**Function:** `build_facet_consolidation_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `partition_name`, `partition_definition`, `chunk_results`, `excluded_domains`
**Response model:** `FacetConsolidatedResponse` → `List[DiscoveredFacet]`

Schema: same `DiscoveredFacet` as §2.

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Known issues:**
- `survey_question` — accepted but NOT interpolated in prompt
- `dataset_context_section` — accepted but NOT interpolated in prompt

---

## §4 Facet Assignment (P2)

**Function:** `build_facet_assignment_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `domain_name`, `domain_definition`, `facets`, `other_label`, `ideas`
**Response model:** `FacetAssignmentBatch` → `List[FacetAssignment]`
**Helpers:** `_build_facet_codebook_block()`, `_build_ideas_block_for_facet_assignment()`, `_valence_display()`

Schema fields per `FacetAssignment`:
- `idea_id: str` — "The EXACT idea_id from the input. Do not modify."
- `idea: str` — "Echo back the EXACT idea text from the input for this idea_id."
- `assigned_facet_id: str` — "The facet ID from the [F#] prefix (e.g. 'F1', 'F3'). Return ONLY the ID, not the facet name."
- `confidence: float` — "Confidence in the assignment (0.0 to 1.0)"

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Notes:** Uses `build_dimension_context_block()` (§1). Prompt instructs "Return the facet ID from [F#] brackets" — matches schema `assigned_facet_id`.

---

## §5 Attribute Discovery (P3)

**Function:** `build_attribute_discovery_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `domain_name`, `domain_definition`, `facet_name`, `facet_description`, `observations`, `excluded_facets`
**Response model:** `AttributeDiscoveryResult` → `List[DiscoveredAttribute]`

Schema fields per `DiscoveredAttribute`:
- `attribute_name: str` — "Short descriptive name for the attribute (2-5 words)"
- `attribute_description: str` — "What this attribute captures — a concrete, observable property (1-2 sentences)"
- `parent_facet: str` — "The facet this attribute belongs to"
- `example_observations: List[str]` — "2-3 representative observations from the input"

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Known issues:**
- `survey_question` — accepted but NOT interpolated in prompt
- `dataset_context_section` — accepted but NOT interpolated in prompt
- `domain_name` — accepted but NOT interpolated in prompt (only `facet_name` is used)
- `domain_definition` — accepted but NOT interpolated in prompt
- `dimension_name`, `dimension_description` — only used in fallback path

---

## §6 Attribute Chunk Consolidation (P3.25)

**Function:** `build_attribute_chunk_consolidation_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `domain_name`, `facet_name`, `facet_description`, `chunk_results`, `excluded_facets`
**Response model:** `AttributeChunkConsolidatedResponse` → `List[DiscoveredAttribute]`

Schema: same `DiscoveredAttribute` as §5.

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Known issues:**
- `survey_question` — accepted but NOT interpolated in prompt
- `dataset_context_section` — accepted but NOT interpolated in prompt
- `domain_name` — accepted but NOT interpolated in prompt

---

## §7 Attribute Assignment (P4a)

**Function:** `build_attribute_assignment_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `domain_name`, `domain_definition`, `facet_name`, `facet_description`, `attributes`, `ideas`
**Response model:** `AttributeAssignmentBatch` → `List[AttributeAssignment]`
**Helpers:** `_build_attribute_codebook_block()`, `_build_ideas_block_for_facet_assignment()`

Schema fields per `AttributeAssignment`:
- `idea_id: str` — "The EXACT idea_id from the input. Do not modify."
- `idea: str` — "Echo back the EXACT idea text from the input for this idea_id."
- `assigned_attribute_id: str` — "The attribute ID from the [A#] prefix (e.g. 'A1', 'A3'). Return ONLY the ID, not the attribute name."
- `confidence: float` — "Confidence in the assignment (0.0 to 1.0)"

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Notes:** Uses `build_dimension_context_block()` (§1). Prompt instructs "Return the attribute ID from [A#] brackets" — matches schema `assigned_attribute_id`.

---

## §8 Attribute Consolidation (P3.5)

**Function:** `build_attribute_consolidation_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_def`, `dimension_name`, `dimension_description`, `domain_name`, `domain_definition`, `facet_attributes_block`
**Response model:** `AttributeConsolidatedResponse` → `List[ConsolidatedAttribute]`

Schema fields per `ConsolidatedAttribute`:
- `attribute_name: str` — "Short descriptive name for the attribute (2-5 words)"
- `attribute_description: str` — "What this attribute captures (1-2 sentences)"
- `parent_facet: str` — "The facet name this attribute best belongs to"
- `example_observations: List[str]` — "2-3 representative observations from the input"
- `source_attributes: List[str]` — "Original attribute names that were merged into this consolidated attribute"

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Known issues:**
- `survey_question` — accepted but NOT interpolated in prompt
- `dataset_context_section` — accepted but NOT interpolated in prompt

---

## §9 Code Generation from Attributes (P4)

**Function:** `build_code_from_attributes_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_name`, `dimension_description`, `domain_attributes`, `valence_label`, `attribute_assignments`
**Response model:** `CodeGenerationFromAttributesResult` → `evaluation: str` + `List[CodeFromAttributes]`

Schema fields per `CodeFromAttributes`:
- `code_name: str` — "Short code name (2-5 words)"
- `definition: str` — "Clear definition of what this code covers (1-2 sentences)"
- `typical_indicators: List[str]` — "Words or phrases that signal this code"
- `source_attributes: List[str]` — "Attribute names this code is derived from"

Also: `FormalCode = CodeFromAttributes` (backward-compat alias)

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Known issues:**
- `valence_label` — accepted as parameter, documented in docstring, but NOT interpolated in the prompt template

---

## §10 Codebook Consolidation (P4.5)

**Function:** `build_codebook_consolidation_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `dimension_name`, `dimension_description`, `raw_codes`, `code_provenance`, `code_frequencies`
**Response model:** `CodebookConsolidationResult` → `evaluation: str` + `List[ConsolidatedCode]`

Schema fields per `ConsolidatedCode`:
- `code_name: str` — "Short code name (3-5 word noun phrase)"
- `definition: str` — "A short interpretive claim that reads like an analyst conclusion..."
- `diagnostic_test: str` — "Completes the sentence: 'This is about whether ...'"
- `valence: str` — "One of: 'positive', 'negative', 'neutral'"
- `typical_indicators: List[str]` — "Words or phrases that signal this code"
- `source_attributes: List[str]` — "Attribute names this code is derived from (from all merged codes)"

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Notes:** This is the most complex prompt (~1400 words). Prompt includes 12 core principles, dual-layer code definition requirements, workflow steps.

---

## §11 Code Assignment (P5)

**Function:** `build_single_dual_assignment_prompt()`
**Parameters:** `survey_question`, `language`, `dataset_context_section`, `codes`, `other_label`, `idea`, `facet_lookup`
**Response model:** `CodeAttributeAssignment`
**Helpers:** `_build_codes_block()`

Schema fields:
- `assigned_code_id: str` — "The code ID from the [C#] prefix (e.g. 'C1', 'C7'). Return ONLY the ID."
- `confidence: float` — "Confidence in the assignment (0.0 to 1.0)"
- `rationale: str` — "Brief rationale for the code choice"

Also defined but used only downstream (not by this prompt's LLM call):
- `CodeAssignment` — internal wrapper with `idea_id`
- `CodeAssignmentBatch` — batch wrapper

| Check | Status |
|-------|--------|
| Formatting ↔ Prompt | ☐ |
| Schema ↔ Prompt | ☐ |
| No dead code | ☐ |

**Notes:** Function name `build_single_dual_assignment_prompt` is a legacy name — it only assigns codes (attributes are assigned in §7). Docstring says "assign a single idea to a code AND attribute" which is outdated.

---

## Summary of pre-identified issues

| § | Issue | Type |
|---|-------|------|
| §2 | `survey_question`, `dataset_context_section`, `partition_definition` accepted but not used in prompt | Dead params |
| §3 | `survey_question`, `dataset_context_section` accepted but not used in prompt | Dead params |
| §5 | `survey_question`, `dataset_context_section`, `domain_name`, `domain_definition` accepted but not used in prompt | Dead params |
| §6 | `survey_question`, `dataset_context_section`, `domain_name` accepted but not used in prompt | Dead params |
| §8 | `survey_question`, `dataset_context_section` accepted but not used in prompt | Dead params |
| §9 | `valence_label` accepted but not interpolated in prompt | Dead param |
| §9 | `FormalCode` alias kept for backward compat | Potential dead code |
| §11 | Function name `build_single_dual_assignment_prompt` is misleading (no dual output) | Naming |
| §11 | Docstring says "code AND attribute" but only assigns code | Stale docs |
| §11 | `CodeAssignment` and `CodeAssignmentBatch` defined but not used by this prompt | Potentially dead models |
