# Pipeline Overhaul Plan: 6-Step Discovery → Assignment Pipeline

## Overview

Restructure the step 4 classNcoder pipeline from the current flow into a clean 6-step pipeline where each taxonomy level follows a discover → assign pattern, and each step enriches ideas before the next step begins.

## Current Flow

```
P1/P1.5  → discover facets + consolidate per domain
P2       → assign facets to ideas
P3/P3.5  → discover attributes per facet + consolidate across facets
P4/P4.5  → generate codes from attribute inventory + consolidate
P5       → assign codes + attributes to ideas (dual assignment)
```

## New Flow

```
Step 1:  Discover + consolidate FACETS (per domain)
Step 2:  Assign FACETS to ideas
Step 3:  Discover ATTRIBUTES (per facet, chunked if needed)
Step 4:  Assign ATTRIBUTES to ideas (per facet)
         → then consolidate attributes (with frequency data from assignment)
Step 5:  Discover + consolidate CODES (from attribute inventory with frequencies)
Step 6:  Assign CODES to ideas (with embedding pre-filter, code-only)
```

## Key Design Principles

a) **Facets assigned before attributes discovered** — attribute discovery only sees ideas within one facet
b) **Attributes assigned per facet before consolidation** — consolidation uses assignment frequency data to make informed merge decisions
c) **Codes based on attribute inventory per domain** — with frequency of assignment visible; consolidation tracks merges to preserve frequency data
d) **Embeddings only for code assignment (step 6)** — facet and attribute assignment have small enough candidate sets to not need pre-filtering
e) **Building on existing logic** — the pipeline is deepened, not rewritten. Existing functions are reused/adapted
f) **Git checkpoint before starting** — revertable
g) **Architecture.md updated at end**

---

## Detailed Step Descriptions

### Step 1: Discover + Consolidate Facets (EXISTING — minor changes)

**Current**: `_discover_domain_facets()` + `_consolidate_domain_facets()`
**Input**: Ideas grouped by domain (partition), with formatted labels (instance → interpretation → abstraction)
**Output**: `DomainFacetSet` per domain — list of facets with names, descriptions, examples
**Chunking**: Hierarchical consolidation (max 6 chunks, max 150 items per consolidation call)
**Changes needed**: None — this works as-is

### Step 2: Assign Facets (EXISTING — no changes)

**Current**: `_assign_facets_to_ideas()`
**Input**: Ideas + discovered facets per domain
**Output**: Each idea gets a `facet` field
**Changes needed**: None — this works as-is

### Step 3: Discover Attributes (EXISTING — minor restructure)

**Current**: `_discover_facet_attributes()` runs per (domain, facet), with chunking
**Input**: Ideas assigned to a specific facet, formatted as labels
**Output**: List of `DiscoveredAttribute` per facet
**Chunking**: If >100 observations, chunk and consolidate within facet
**Changes needed**:
- Remove the cross-facet consolidation that currently happens here (move to after step 4)
- Keep per-facet chunk consolidation as-is

### Step 4: Assign Attributes + Consolidate (NEW + MOVED)

#### Step 4a: Assign attributes to ideas (NEW)

**Input**: Ideas with assigned facets + discovered attributes per facet
**Prompt**: For each idea, pick the best-matching attribute from the attributes belonging to its assigned facet. Small candidate set (~5-15 attributes per facet), so no embedding pre-filter needed.
**Concurrency**: Group ideas by facet → all ideas in same facet get same attribute list. Process all facets concurrently.
**Output**: Each idea gets an `assigned_attribute` field
**Response model**: `AttributeAssignment` — idea_id, attribute_name, confidence

**Prompt structure**:
```
<facet>
Duurzaamheid en milieubewustzijn — [description]
</facet>

<attributes>
[A1] Milieuvriendelijkheid — [description]
[A2] Klimaatbewustzijn — [description]
...
</attributes>

<idea>
instance: groene
interpretation: milieuvriendelijke en duurzame kenmerken
valence: +
</idea>

→ Pick one attribute
```

#### Step 4b: Consolidate attributes across facets (MOVED from current P3.5)

**Input**: Attributes per facet, now enriched with **assignment frequency** (how many ideas were assigned to each attribute)
**What changes vs current P3.5**: The consolidation prompt now includes frequency counts per attribute, e.g.:
```
Facet: "Milieubewustzijn"
  - Milieuvriendelijkheid (570 ideas)
  - Klimaatbewustzijn (46 ideas)
  - Natuurbehoud (32 ideas)
```
This helps the LLM make better merge decisions — a low-frequency attribute can be merged into a high-frequency neighbour, while high-frequency attributes should be preserved.

**Merge tracking**: When two attributes are merged, the consolidated result must track which original attributes were merged and their combined count. This is needed for step 5 (code generation) to know the true frequency.

**Output**: Consolidated attribute inventory per domain, with frequency data preserved

### Step 5: Discover + Consolidate Codes (MODIFIED)

**Input**: Consolidated attribute inventory per domain, with frequency data
**What changes vs current P4/P4.5**:
- Input is now a frequency-weighted attribute inventory, not raw idea labels
- Each attribute entry shows: name, description, frequency, parent facet
- The LLM generates codes by grouping related attributes, informed by their frequency
- Code consolidation (P4.5) already works well with our new prompt — keep as-is

**Frequency in code generation prompt**:
```
<attribute_inventory>
Domain: "duurzaamheid en maatschappelijke verantwoordelijkheid"

  Facet: "Milieubewustzijn en ecologische duurzaamheid"
    - Milieuvriendelijkheid — [description] — 570 ideas
    - Duurzaamheid (lange termijn) — [description] — 89 ideas
    - Klimaatbewustzijn — [description] — 46 ideas
    ...

  Facet: "Ethisch en verantwoord ondernemen"
    - Eerlijkheid en integriteit — [description] — 74 ideas
    - Verantwoord beleggen — [description] — 30 ideas
    ...
</attribute_inventory>
```

**Output**: Consolidated codebook (ConsolidatedCode list) — same model as now

### Step 6: Assign Codes (SIMPLIFIED)

**Input**: Ideas (now with facet + attribute already assigned) + codebook
**What changes vs current P5**:
- **Code-only assignment** — no dual assignment. Attribute is already set from step 4.
- Embedding pre-filter still applies (top-5 codes per idea)
- Simpler prompt (no "pick attribute within code")
- Simpler response model: `CodeAssignment` (code_id, confidence, rationale) — no attribute field

**Prompt structure**:
```
<codebook>
[C1] (+) Ethisch en duurzaam imago
    Definition: ...
    Diagnostic: ...
    Indicators: ...
</codebook>

<idea>
instance: groene
interpretation: milieuvriendelijke en duurzame kenmerken
attribute: Milieuvriendelijkheid     ← already assigned
valence: +
</idea>

→ Pick one code
```

---

## Implementation Tasks

### Phase 0: Preparation
- [ ] Create git checkpoint (tag or branch)
- [ ] Review existing functions to map what can be reused

### Phase 1: Step 4a — Attribute Assignment
- [ ] Create `build_attribute_assignment_prompt()` in prompts_exp.py
- [ ] Create `AttributeAssignment` response model in prompts_exp.py
- [ ] Add `_assign_attributes_to_ideas()` method in qualitative_researcher.py
  - Group ideas by facet
  - For each facet: build prompt with facet's attributes, assign each idea
  - Concurrent processing across facets
  - Rate limiting (reuse existing setup)
- [ ] Wire into pipeline in `run_experiment.py`
- [ ] Add verbose reporting for attribute assignment
- [ ] Test: verify every idea gets an attribute

### Phase 2: Step 4b — Move Attribute Consolidation After Assignment
- [ ] Move `_consolidate_domain_attributes()` call to AFTER attribute assignment
- [ ] Update consolidation prompt to include frequency data per attribute
- [ ] Add merge tracking: `ConsolidatedAttribute` gets `merged_from` field with original names + counts
- [ ] Update `build_attribute_consolidation_prompt()` to show frequencies
- [ ] Remap idea attributes: after consolidation, update ideas whose attribute was merged → point to consolidated name
- [ ] Test: verify frequency data is correct and ideas are remapped

### Phase 3: Step 5 — Code Generation from Frequency-Weighted Inventory
- [ ] Update code generation prompt input: attribute inventory with frequencies instead of raw labels
- [ ] Update `build_codebook_generation_prompt()` to format frequency-weighted attributes
- [ ] Verify P4.5 consolidation still works (should, since input format is similar)
- [ ] Test: verify codes are generated from the frequency-weighted inventory

### Phase 4: Step 6 — Simplify Code Assignment
- [ ] Remove attribute assignment from P5 prompt (code-only)
- [ ] Simplify `CodeAttributeAssignment` → `CodeAssignment` (remove attribute field)
- [ ] Update `_build_codes_block()` — remove attributes section under each code
- [ ] Update `CodeAssigner` to not handle attribute assignment (already done in step 4)
- [ ] Remove `_attribute_assignments` dict and related logic
- [ ] Embedding pre-filter unchanged
- [ ] Test: verify code assignment works without attribute dual-assignment

### Phase 5: Cleanup and Documentation
- [ ] Remove dead code from all files
- [ ] Update ARCHITECTURE.md
- [ ] Update work_to_be_done.md
- [ ] Final import verification
- [ ] Run full pipeline end-to-end
- [ ] Push to GitHub

---

## Files Affected

| File | Changes |
|------|---------|
| `prompts_exp.py` | New attribute assignment prompt + model; update code generation prompt for frequencies; simplify P5 prompt |
| `qualitative_researcher.py` | New `_assign_attributes_to_ideas()`; move consolidation after assignment; update pipeline flow |
| `code_assignment.py` | Simplify to code-only assignment; remove attribute handling |
| `run_experiment.py` | Wire new step 4a; update pipeline orchestration; update caching |
| `models_exp.py` | Possibly add `ConsolidatedAttribute` with merge tracking |
| `config_classNcoder_exp.py` | Attribute assignment config (batch size, model, etc.) |
| `ARCHITECTURE.md` | Update pipeline description |

---

## Risk Mitigation

- Git checkpoint before starting → can revert
- Each phase is independently testable
- Existing logic is adapted, not rewritten
- Attribute assignment is a simple constrained choice (small candidate set) → low LLM failure risk

---

## Status Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Preparation | DONE | Tag: pre-pipeline-overhaul |
| Phase 1: Attribute Assignment | DONE | Prompt, model, method implemented |
| Phase 2: Move Consolidation | DONE | Frequency data + remap logic |
| Phase 3: Code Generation Update | DONE | Frequency-weighted inventory |
| Phase 4: Simplify Code Assignment | DONE | Code-only, attributes pre-assigned |
| Phase 5: Cleanup & Documentation | DONE | ARCHITECTURE.md updated |
