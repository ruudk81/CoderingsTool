# Step 5 classNcoder — Prompt Work Items

Source: `prompts_exp.py` (audit findings from PROMPT_AUDIT.md)

## Audit Summary

| § | Prompt | Dead params | Schema gaps | Other issues |
|---|--------|-------------|-------------|--------------|
| 2 | Facet Discovery | 4 | 2 nits | — |
| 3 | Facet Consolidation | 3 | 1 | 3 text errors |
| 4 | Facet Assignment | 0 ✓ | 1 nit | — |
| 5 | Attribute Discovery | 5 | 2 | — |
| 6 | Attr Chunk Consolidation | 4 | 2 | — |
| 7 | Attribute Assignment | 0 ✓ | 1 nit | — |
| 8 | Attr Consolidation | 3 | 1 | — |
| 9 | Code Generation | 1 | 2 | 2 dead code items |
| 10 | Codebook Consolidation | 0 ✓ | 3 | 2 hardcoded domain-specific principles |
| 11 | Code Assignment | 0 ✓ | 1 | 3 stale names, misplaced models |

### Cross-cutting themes

1. **Dead params pattern** — §2, §3, §5, §6, §8 all accept `survey_question` + `dataset_context_section` but never use them. §4 and §7 (assignment prompts) are the reference pattern that does it right.
2. **Consolidation instruction inconsistency** — §6 uses `<strict_consolidation_rule>` + `<disambiguation_test>` + `<precedence_rule>` (strong). §3 and §8 use `<scratchpad>` (weaker). All three do the same type of job.
3. **Schema echo-back fields** — §4 and §7 have `idea` echo-back in schema that prompt never mentions. Works via instructor, used for drift detection.
4. **Unprompted schema fields** — `evaluation` (§9, §10), `source_attributes` (§8), `typical_indicators` + `source_attributes` (§10) are required by schema but not in prompt output instructions.
5. **Valence tags unexplained** — §4 and §7 include valence in ideas block but prompt never says what to do with it. §11 does it right (instruction step 1 says "Read the idea text, domain, facet, and valence").

---

## §2 Facet Discovery (P1) — `build_facet_discovery_prompt()`

### 2.1 Dead parameters (4) — DONE

The following parameters are accepted by the function but never interpolated in the prompt template:

| Parameter | Line | Why it's likely needed | Action |
|-----------|------|----------------------|--------|
| `survey_question` | 140 | Gives the LLM context about what respondents were asked | Add to prompt (e.g. `<survey_context>` block, like §4 does) |
| `dataset_context_section` | 142 | Gives sector/entity/topic context | Add to prompt (e.g. `<survey_context>` block, like §4 does) |
| `partition_definition` | 147 | Tells the LLM what the domain means — currently only the domain name is shown | Add to `<domain>` block: `{partition_name} — {partition_definition}` |
| `dimension_description` | 145 | Describes what kind of variation the dimension captures | Add to taxonomy levels block or a `<dimension_context>` section |

**Reference pattern:** §4 Facet Assignment already uses all of these correctly:
```
<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_block}   ← includes dimension + domain via build_dimension_context_block()
```

### 2.2 Schema ↔ Prompt alignment (2 nits) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| Prompt asks for "observation numbers" but schema expects observation text | Prompt line ~237: "Which observation numbers support it"; Schema: `example_observations: List[str]` — "3-5 representative observations from the input" | The scratchpad step asks for numbers (cheap reasoning token), but the final output schema asks for actual text. This works in practice because instructor enforces the schema, but the prompt instruction and schema disagree on format. |
| Prompt never explicitly asks for a facet description | Prompt task instructions (Steps 1-5) never say "write a description for each facet" | Works because the schema's `Field(description=...)` tells the LLM to produce it. Low risk, but prompt could be more explicit. |

---

## §3 Facet Consolidation (P1.5) — `build_facet_consolidation_prompt()`

### 3.1 Dead parameters (3) — DONE

Same pattern as §2 — these are accepted but never interpolated:

| Parameter | Line | Why it's likely needed | Action |
|-----------|------|----------------------|--------|
| `survey_question` | 280 | Context about what respondents were asked | Add `<survey_context>` block |
| `dataset_context_section` | 282 | Sector/entity/topic context | Add `<survey_context>` block |
| `dimension_description` | 285 | What kind of variation the dimension captures | Add to taxonomy block |

Note: `partition_definition` IS used here (`{partition_name} — {partition_definition}` in `<domain>` block), unlike §2 where it's dead.

### 3.2 Schema ↔ Prompt alignment (1) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| Prompt never mentions example observations | Prompt consolidation instructions have no guidance about examples | Schema `DiscoveredFacet.example_observations` (3-5) is required but the prompt doesn't say to select or preserve examples when merging. The LLM must infer this purely from the schema. |

### 3.3 Text errors (3) — DONE

| Issue | Line | Fix |
|-------|------|-----|
| "an facet" (grammar) | 351 | → "a facet" |
| "Strictly within facet scope — the facet must stay fully within the conceptual boundaries of the defined facet" — says "facet" but should say "domain" | 353 | → "Strictly within domain scope — the facet must stay fully within the conceptual boundaries of the defined domain and not leak into adjacent domains." |
| "If an facet" (grammar) | 386 | → "If a facet" |

---

## §4 Facet Assignment (P2) — `build_facet_assignment_prompt()`

### 4.1 All parameters used ✓

This prompt is the **reference pattern** — all params are wired in correctly via `<survey_context>` + `build_dimension_context_block()` + helper formatters. No dead params.

### 4.2 Schema ↔ Prompt alignment (1 nit)

| Issue | Location | Detail |
|-------|----------|--------|
| Schema asks LLM to echo back idea text, prompt doesn't mention it | `FacetAssignment.idea` — "Echo back the EXACT idea text" | Prompt instructions (steps 1-5) never ask the LLM to return the idea text. Works via schema enforcement. This echo-back is used downstream for content-drift validation (`SequenceMatcher` in `qualitative_researcher.py`). |

### 4.3 Observations

- `other_label` is always passed as `None` by the orchestrator — the "Other" category feature exists but is never activated.
- Valence tags (`[+]`, `[-]`, `[0]`) appear in ideas block but the prompt never explains what they mean or how they should influence assignment.

---

## §5 Attribute Discovery (P3) — `build_attribute_discovery_prompt()`

### 5.1 Dead parameters (5) — DONE

| Parameter | Line | Why it's likely needed | Action |
|-----------|------|----------------------|--------|
| `survey_question` | 583 | Context about what respondents were asked | Add `<survey_context>` block (§4 pattern) |
| `dataset_context_section` | 585 | Sector/entity/topic context | Add `<survey_context>` block (§4 pattern) |
| `dimension_description` | 588 | What kind of variation the dimension captures | Add to taxonomy block |
| `domain_name` | 589 | LLM doesn't know which domain this facet belongs to | Add `<domain>` block or mention in `<facet>` context |
| `domain_definition` | 590 | What the domain means | Add alongside `domain_name` |

This is the worst case — 5 dead params. The LLM gets no survey context, no dataset context, and doesn't even know which domain it's working in.

### 5.2 Schema ↔ Prompt alignment (2) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| `parent_facet` required in schema but prompt never asks for it | `DiscoveredAttribute.parent_facet` | The LLM must echo back the facet name it's already told it's working within. Redundant — orchestrator already knows this. Wastes output tokens. |
| Prompt asks for "observation numbers" but schema expects observation text | Step 2 vs `example_observations: List[str]` | Same mismatch as §2 |

### 5.3 Structural observation

Unlike §2 Facet Discovery, this prompt has no dominant/minor distinction — all discovered attributes are returned. This is fine since facet-level frequency filtering already happened in P1.

---

## §6 Attribute Chunk Consolidation (P3.25) — `build_attribute_chunk_consolidation_prompt()`

### 6.1 Dead parameters (4) — DONE

| Parameter | Line | Why it's likely needed | Action |
|-----------|------|----------------------|--------|
| `survey_question` | 722 | Context about what respondents were asked | Add `<survey_context>` block (§4 pattern) |
| `dataset_context_section` | 724 | Sector/entity/topic context | Add `<survey_context>` block (§4 pattern) |
| `dimension_description` | 727 | What kind of variation the dimension captures | Add to taxonomy block |
| `domain_name` | 728 | LLM doesn't know which domain this facet belongs to | Add context (e.g. in `<facet>` block or separate `<domain>` block) |

Note: unlike §5, this function doesn't accept `domain_definition` at all.

### 6.2 Schema ↔ Prompt alignment (2) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| `parent_facet` required in schema but prompt never asks for it | `DiscoveredAttribute.parent_facet` | Same as §5 — redundant echo-back of the facet the LLM is already told it's working in |
| `example_observations` required but prompt never mentions selecting examples when merging | `DiscoveredAttribute.example_observations` (2-3) | Prompt consolidation rules focus on merge/overlap logic but give no guidance about preserving representative examples |

### 6.3 Cross-section observation: §3 vs §6 consolidation pattern mismatch

§3 (Facet Consolidation) and §6 (Attribute Chunk Consolidation) do the same job at different taxonomy levels — merge chunk-level discoveries. But they use different instruction patterns:

| Aspect | §3 Facet Consolidation | §6 Attribute Chunk Consolidation |
|--------|----------------------|-------------------------------|
| Consolidation rules | `<scratchpad>` with 5 informal steps | `<strict_consolidation_rule>` with 5 formal rules |
| Disambiguation test | None | `<disambiguation_test>` block |
| Precedence rules | None | `<precedence_rule>` block |
| Merge criteria | "conceptual overlap, near-equivalence" | "orthogonality test: can an observation fall under both?" |

§6's pattern is stronger. Consider aligning §3 to use the same structured approach.

---

## §7 Attribute Assignment (P4a) — `build_attribute_assignment_prompt()`

### 7.1 All parameters used ✓

Like §4, all params are wired in correctly via `<survey_context>` + `build_dimension_context_block()` + `<facet_context>` + helper formatters. No dead params.

### 7.2 Schema ↔ Prompt alignment (1 nit)

| Issue | Location | Detail |
|-------|----------|--------|
| Schema asks LLM to echo back idea text, prompt doesn't mention it | `AttributeAssignment.idea` — "Echo back the EXACT idea text" | Same as §4 — works via schema enforcement, used for content-drift validation downstream |

### 7.3 Observations

- No `other_label` / "Other" fallback (unlike §4) — deliberate: every idea must map to a discovered attribute.
- Valence tags (`[+]`, `[-]`, `[0]`) appear in ideas block but prompt doesn't explain them (same as §4).
- Structurally near-identical to §4, which is good — consistent pattern across assignment prompts.

---

## §8 Attribute Consolidation (P3.5) — `build_attribute_consolidation_prompt()`

### 8.1 Dead parameters (3) — DONE

| Parameter | Line | Why it's likely needed | Action |
|-----------|------|----------------------|--------|
| `survey_question` | 1003 | Context about what respondents were asked | Add `<survey_context>` block (§4 pattern) |
| `dataset_context_section` | 1005 | Sector/entity/topic context | Add `<survey_context>` block (§4 pattern) |
| `dimension_description` | 1008 | What kind of variation the dimension captures | Add to taxonomy block |

Note: `domain_name` and `domain_definition` ARE both used here (unlike §5/§6).

### 8.2 Schema ↔ Prompt alignment (1) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| `source_attributes` required in schema but prompt never asks for it | `ConsolidatedAttribute.source_attributes` — "Original attribute names that were merged" | Prompt says to merge overlapping attributes but never instructs the LLM to list which originals were merged. The orchestrator needs this for remapping assignments (line 762 in `qualitative_researcher.py`). |

### 8.3 Schema design: `parent_facet` is meaningful here ✓

Unlike §5/§6 where `parent_facet` was redundant (all attributes belong to the same facet), here it's correct — this prompt consolidates *across* facets, so the LLM must decide which facet each surviving attribute belongs to.

### 8.4 Cross-section observation: consolidation instruction pattern

§8 uses the `<scratchpad>` approach (same as §3), not the `<strict_consolidation_rule>` + `<disambiguation_test>` + `<precedence_rule>` pattern from §6. Since §8 is the harder task (cross-facet dedup vs within-facet chunk merge), it could benefit even more from the stricter rules.

| Prompt | Task | Pattern | Stronger? |
|--------|------|---------|-----------|
| §3 Facet Consolidation | Within-domain chunk merge | `<scratchpad>` | Weaker |
| §6 Attr Chunk Consolidation | Within-facet chunk merge | `<strict_consolidation_rule>` | Stronger |
| §8 Attr Consolidation | Cross-facet dedup (hardest) | `<scratchpad>` | Weaker |

Consider standardizing all three on the §6 pattern.

---

## §9 Code Generation from Attributes (P4) — `build_code_from_attributes_prompt()`

### 9.1 Dead parameter (1) — DONE

| Parameter | Line | Detail | Action |
|-----------|------|--------|--------|
| `valence_label` | 1154 | Accepted, documented in docstring ("scopes code generation by valence"), but never interpolated in the prompt. Orchestrator always passes `""`. | Remove param, or wire it in if valence-scoped generation is intended |

### 9.2 Schema ↔ Prompt alignment (2) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| Word count mismatch for code_name | Schema: "2-5 words"; Prompt: "3–5 word noun phrase" | Minor — pick one and align |
| `evaluation` field not prompted for | `CodeGenerationFromAttributesResult.evaluation` — "Brief evaluation of how codes were derived" | Prompt never asks the LLM for an evaluation or reflection. Works via schema enforcement but could be more explicit. |

### 9.3 Dead code (2) — DONE

| Item | Location | Detail |
|------|----------|--------|
| `FormalCode` alias | `prompts_exp.py:1132-1133` | `FormalCode = CodeFromAttributes` — imported by `qualitative_researcher.py:80` but never referenced in code body. Both alias and import are dead. |
| Unused import | `qualitative_researcher.py:80` | `FormalCode` imported but never used |

### 9.4 Observations

- This is one of the best-structured prompts. All meaningful params are used, it has clear rules, and explicitly lists output requirements including `source_attributes`.
- The prompt uses `{dimension_name}` 7 times throughout the instructions to keep rules dimension-aware — good pattern.
- The Valence Sensitivity Rule in the prompt says to "generate separate codes for positive and negative phenomena" — this is baked into the prompt text, making the `valence_label` parameter redundant (the prompt handles valence internally rather than via pre-filtering).

---

## §10 Codebook Consolidation (P4.5) — `build_codebook_consolidation_prompt()`

### 10.1 All parameters used ✓

All params wired in correctly. No dead params.

### 10.2 Schema ↔ Prompt alignment (3) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| `typical_indicators` not in output requirements | `ConsolidatedCode.typical_indicators` | Input codes show "Indicators:" but the `<code_definition_requirements>` section only lists `code_name`, `definition`, `diagnostic_test`. The LLM must infer from schema to produce indicators. |
| `source_attributes` not in output requirements | `ConsolidatedCode.source_attributes` | Same as above — input codes show "Source attributes:" but output requirements don't ask to track/merge these. |
| `evaluation` not prompted for | `CodebookConsolidationResult.evaluation` | Wrapper field not requested in prompt (same as §9). |

### 10.3 Hardcoded domain-specific content (2)

These principles contain examples that are specific to a banking/brand-association dataset and will not generalize:

| Principle | Lines | Hardcoded content |
|-----------|-------|-------------------|
| **12. DO-NOT-MERGE DISTINCTIONS** | 1445-1452 | "Ethics vs Trust", "Service quality vs Usability", "Brand image vs Brand awareness", "Financial attractiveness vs Product availability" — all banking/brand specific |
| **10. ATTRIBUTE TYPE SEPARATION** | 1431-1437 | "Values-based attributes (ethics, sustainability)", "Functional attributes (products, usability, pricing)", "Perceptual attributes (image, recognition, personality)" — brand association specific |

These should either be:
- Removed (let the general rules handle it)
- Made dynamic (derived from `dimension_def` or the actual attribute types present in the data)
- Moved to a dataset-specific config

### 10.4 Observations

- This is the most elaborate prompt (~1400 words, 12 principles). Well-structured with clear hierarchy.
- Good use of `{dimension_name}` and `{dimension_description}` throughout to keep rules dimension-aware.
- Workflow step 7 references `{survey_question}` for actionability check — good integration.
- The `<code_definition_requirements>` section is explicit about `code_name`, `definition`, and `diagnostic_test` but silent about `typical_indicators` and `source_attributes` — these should be added for completeness.

---

## §11 Code Assignment (P5) — `build_single_dual_assignment_prompt()`

### 11.1 All parameters used ✓

All params wired in correctly. No dead params.

### 11.2 Schema ↔ Prompt alignment (1) — DONE

| Issue | Location | Detail |
|-------|----------|--------|
| `rationale` required in schema but prompt doesn't ask for it | `CodeAttributeAssignment.rationale` — "Brief rationale for the code choice" | Instructions (steps 1-5) never mention providing a rationale. Works via schema enforcement. |

### 11.3 Stale naming (3) — PARTIAL (docstring fixed, function/class rename deferred)

| Item | Location | Issue | Fix |
|------|----------|-------|-----|
| Function name | `build_single_dual_assignment_prompt` | "dual" is legacy — only assigns codes (attributes assigned in §7) | Rename to `build_code_assignment_prompt` |
| Docstring | line 1574 | "assign a single idea to a code AND attribute" | Update to "assign a single idea to a code" |
| Class name | `CodeAttributeAssignment` | "Attribute" misleading — only handles code assignment | Rename to `CodeAssignmentResponse` or similar |

### 11.4 Misplaced models

`CodeAssignment` and `CodeAssignmentBatch` are defined here but are NOT LLM response models — they're post-hoc wrappers used by `code_assignment.py` (line 774) to add `idea_id` from the original task. They arguably belong in `models_exp.py` rather than `prompts_exp.py`.

### 11.5 Observations

- The prompt includes domain, facet, and valence per idea — and instruction step 1 explicitly says "Read the idea text, domain, facet, and valence." Good — unlike §4/§7 where valence appears but is never referenced in instructions.
- Confidence scale with qualitative labels (0.90+ = clear, etc.) is a nice touch not present in §4/§7.
