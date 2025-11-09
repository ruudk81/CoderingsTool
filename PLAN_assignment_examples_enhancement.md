# Enhancement Plan: Assignment Examples Throughout Code Generation Pipeline

## Executive Summary

Currently, `inclusion_examples`, `exclusion_examples`, `near_neighbor_label`, and `tell_apart_rule` are:
- ✓ Generated in Chain 1 (CLUSTER_SUMMARY_PROMPT)
- ✓ Used as context in Chains 2, 3, 4
- ✓ Stored in `step1_summaries`
- ❌ **NOT updated** when codes are modified/merged
- ❌ **NOT included** in final codebook output
- ❌ **NOT available** to codeAssigner

**Goal:** Track and update these fields through all 4 chains, store them in the codebook, and make them available to codeAssigner for better assignment accuracy.

---

## Current State Analysis

### Chain 1: CLUSTER_SUMMARY_PROMPT ✓
**Status:** Working correctly
- **Generates:** assignment_examples (inclusion, exclusion, near_neighbor with tell_apart_rule)
- **Model:** `ClusterThemeItem.assignment_examples: AssignmentExamples`
- **Storage:** `step1_summaries[cluster_id]['themes'][0].assignment_examples`

### Chain 2: CODING_DECISION_PROMPT ⚠️
**Status:** Uses but doesn't update
- **Uses:** Receives inclusion/exclusion/near_neighbor as INPUT (lines 677-681)
- **Outputs:** `ModifyParameters` with `inclusion_update`, `exclusion_update` (lines 115-116)
- **Missing:**
  - No `near_neighbor_label_update`
  - No `tell_apart_rule_update`
  - No complete `assignment_examples` in output

### Chain 3: CODE_CREATION_PROMPT ❌
**Status:** Uses but doesn't output
- **Uses:** Receives inclusion/exclusion as INPUT (lines 872-875)
- **Outputs:** `GeneratedCode` with only `code_label`, `code_definition`
- **Missing:** No `assignment_examples` in output

### Chain 4: VALIDATION_PROMPT ❌
**Status:** Doesn't handle assignment_examples at all
- **Outputs:** `ValidatedCode` with only `code`, `definition`
- **Missing:** No validation or output of `assignment_examples`

### Final Codebook Models ❌
**Status:** No fields for assignment_examples
- `CodebookEntry`: Only has code, definition, source_cluster
- `Codebook`: Only has code, definition, source_cluster, theme, theme_description
- **Missing:** All assignment_examples fields

### codeAssigner ❌
**Status:** Cannot access assignment_examples
- Receives only basic codebook with code/definition
- Prompts don't show inclusion/exclusion/tell_apart context

---

## Detailed Implementation Plan

### PHASE 1: Update Pydantic Models in codeGenerator.py

#### 1.1 Update ModifyParameters (Line ~111)
```python
class ModifyParameters(BaseModel):
    modify_instruction: Literal["vertical_broaden_same_motive", "hierarchical_parent_diff_motive_same_family", "none"]
    motive_comparison: Literal["same", "different_same_family", "different_not_related"]
    abstraction_level_action: Literal["keep", "broaden_to_parent", "none"]
    inclusion_update: Optional[str] = None
    exclusion_update: Optional[str] = None
    parent_theme_label: Optional[str] = None
    # NEW FIELDS:
    near_neighbor_label_update: Optional[str] = None
    tell_apart_rule_update: Optional[str] = None
```

#### 1.2 Update CodingDecision (Line ~120)
```python
class CodingDecision(BaseModel):
    theme_number: int
    theme_name: str
    matched_candidates: List[MatchedCandidate]
    decision: str  # use | modify | create
    source_code: Optional[str] = None
    modify_parameters: ModifyParameters
    justification: str
    # NEW FIELD:
    updated_assignment_examples: Optional[AssignmentExamples] = None  # Reuse existing model
```

#### 1.3 Update GeneratedCode (Line ~136)
```python
class GeneratedCode(BaseModel):
    theme_number: int
    theme_name: str
    source_code: Optional[str] = None
    code_label: str
    code_definition: str
    # NEW FIELD:
    assignment_examples: AssignmentExamples
```

#### 1.4 Update ValidatedCode (Line ~149)
```python
class ValidatedCode(BaseModel):
    code: str
    definition: str
    # NEW FIELD:
    assignment_examples: AssignmentExamples
```

---

### PHASE 2: Update Prompts in prompts.py

#### 2.1 Update CODING_DECISION_PROMPT (Line ~653)

**Current INPUT section (lines 677-681):**
```
- what's included:
    {inclusion}
- what's excluded:
    {exclusion}
- boundary: {near_neighbor}
```
✓ Already good

**Update OUTPUT schema (lines 728-736) to include:**
```json
{
  "coding_decision": {
    // ... existing fields ...
    "modify_parameters": {
       // ... existing fields ...
       "near_neighbor_label_update": "null or updated neighbor label if boundaries changed",
       "tell_apart_rule_update": "null or updated tell-apart rule if boundaries changed"
    },
    "updated_assignment_examples": {
      "inclusion": ["[updated inclusion examples]"],
      "exclusion": ["[updated exclusion examples]"],
      "near_neighbor": {
        "label": "[updated or original neighbor label]",
        "tell_apart_rule": "[updated or original tell-apart rule]"
      }
    }
  }
}
```

**Add instructions in analysis_steps:**
```
4. Update Assignment Examples:
   - If decision is USE → preserve original assignment_examples
   - If decision is MODIFY:
     * inclusion: original + new expressions from theme
     * exclusion: original + new boundary clarifications
     * near_neighbor: update if boundaries shifted
     * tell_apart_rule: update if distinction changed
   - If decision is CREATE → use assignment_examples from new theme
```

#### 2.2 Update CODE_CREATION_PROMPT (Line ~754)

**Add to OUTPUT schema:**
```json
{
  "generated_code": {
    // ... existing fields ...
    "assignment_examples": {
      "inclusion": ["[2-3 concrete examples of what to include]"],
      "exclusion": ["[1-2 concrete examples of what to exclude]"],
      "near_neighbor": {
        "label": "[closest confusable concept or 'Unknown']",
        "tell_apart_rule": "[1-sentence distinction]"
      }
    }
  }
}
```

**Add instructions:**
```
Assignment Examples:
- Provide concrete, actionable assignment examples
- inclusion: 2-3 short examples of expressions that should be coded here
- exclusion: 1-2 short examples of what should NOT be included
- near_neighbor: Identify closest confusable concept and how to tell them apart
```

#### 2.3 Update CODING_MODIFICATION_PROMPT (Line ~861)

**Add to INPUT section (after line 885):**
```
Current assignment examples (to be updated):
- Current inclusion examples:
  {current_inclusion}
- Current exclusion examples:
  {current_exclusion}
- Current boundary (near neighbor):
  {current_near_neighbor}
```

**Add to OUTPUT schema:**
```json
{
  "generated_code": {
    // ... existing fields ...
    "assignment_examples": {
      "inclusion": ["[updated inclusion examples]"],
      "exclusion": ["[updated exclusion examples]"],
      "near_neighbor": {
        "label": "[updated neighbor label]",
        "tell_apart_rule": "[updated tell-apart rule]"
      }
    }
  }
}
```

**Add instructions:**
```
Update Assignment Examples:
- inclusion: Combine original + new expressions from inclusion_update
- exclusion: Combine original + new boundaries from exclusion_update
- near_neighbor: Update label/rule if boundaries changed due to modification
```

#### 2.4 Update VALIDATION_PROMPT (Line ~920)

**Add to INPUT section:**
```
Assignment examples to validate:
- inclusion: {inclusion_examples}
- exclusion: {exclusion_examples}
- near_neighbor: {near_neighbor_label} (Tell apart: {tell_apart_rule})
```

**Add to OUTPUT schema:**
```json
{
  "code_validation": {
    // ... existing fields ...
    "validated_code": {
      "code": "...",
      "definition": "...",
      "assignment_examples": {
        "inclusion": ["[validated/refined inclusion examples]"],
        "exclusion": ["[validated/refined exclusion examples]"],
        "near_neighbor": {
          "label": "[validated neighbor label]",
          "tell_apart_rule": "[validated tell-apart rule]"
        }
      }
    }
  }
}
```

**Add validation instructions:**
```
Validate Assignment Examples:
- Ensure inclusion examples align with refined code/definition
- Ensure exclusion examples maintain clear boundaries
- Verify near_neighbor and tell_apart_rule are still accurate
- Refine if needed to match validated code
```

---

### PHASE 3: Update Downstream Processing in codeGenerator.py

#### 3.1 Update _process_chain2_decision() (~line 2650)

**After extracting decision, also extract updated_assignment_examples:**
```python
# Extract decision
decision_obj = step2_result.coding_decision

# Extract updated assignment examples
if hasattr(decision_obj, 'updated_assignment_examples') and decision_obj.updated_assignment_examples:
    current_assignment_examples = decision_obj.updated_assignment_examples
else:
    # Fallback: use original from step1
    current_assignment_examples = theme_data_from_step1.assignment_examples
```

#### 3.2 Update _process_chain3_generation() (~line 2750)

**Pass assignment_examples to Chain 3:**
```python
# For MODIFY operations, pass current assignment examples
if decision == "MODIFY":
    prompt_params['current_inclusion'] = self._format_examples(current_assignment_examples.inclusion)
    prompt_params['current_exclusion'] = self._format_examples(current_assignment_examples.exclusion)
    prompt_params['current_near_neighbor'] = f"{current_assignment_examples.near_neighbor.label} (Tell apart: {current_assignment_examples.near_neighbor.tell_apart_rule})"
```

**Extract assignment_examples from Chain 3 output:**
```python
generated_code_obj = step3_result.generated_code

# Extract assignment examples
if hasattr(generated_code_obj, 'assignment_examples') and generated_code_obj.assignment_examples:
    final_assignment_examples = generated_code_obj.assignment_examples
else:
    # Fallback to current
    final_assignment_examples = current_assignment_examples
```

#### 3.3 Update _process_chain4_validation() (~line 2900)

**Pass assignment_examples to Chain 4:**
```python
prompt_params['inclusion_examples'] = self._format_examples(final_assignment_examples.inclusion)
prompt_params['exclusion_examples'] = self._format_examples(final_assignment_examples.exclusion)
prompt_params['near_neighbor_label'] = final_assignment_examples.near_neighbor.label
prompt_params['tell_apart_rule'] = final_assignment_examples.near_neighbor.tell_apart_rule
```

**Extract validated assignment_examples:**
```python
validated_code_obj = step4_result.code_validation.validated_code

# Extract validated assignment examples
if hasattr(validated_code_obj, 'assignment_examples') and validated_code_obj.assignment_examples:
    validated_assignment_examples = validated_code_obj.assignment_examples
else:
    # Fallback
    validated_assignment_examples = final_assignment_examples
```

#### 3.4 Store in final cluster_results

**Update cluster_results structure to include:**
```python
cluster_results.append({
    'cluster_id': cluster_id,
    'code': validated_code,
    'definition': validated_definition,
    'source_cluster_id': source_cluster_id,
    # NEW FIELDS:
    'inclusion_examples': validated_assignment_examples.inclusion,
    'exclusion_examples': validated_assignment_examples.exclusion,
    'near_neighbor_label': validated_assignment_examples.near_neighbor.label,
    'tell_apart_rule': validated_assignment_examples.near_neighbor.tell_apart_rule,
    # ... other fields ...
})
```

---

### PHASE 4: Update Codebook Models in models.py

#### 4.1 Update CodebookEntry (Line ~61)
```python
class CodebookEntry(BaseModel):
    code: str
    definition: str
    source_cluster: Optional[str] = None
    # NEW FIELDS:
    inclusion_examples: Optional[List[str]] = None
    exclusion_examples: Optional[List[str]] = None
    near_neighbor_label: Optional[str] = None
    tell_apart_rule: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

#### 4.2 Update Codebook (Line ~106)
```python
class Codebook(BaseModel):
    code: str
    definition: str
    source_cluster: Optional[str] = None
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    # NEW FIELDS:
    inclusion_examples: Optional[List[str]] = None
    exclusion_examples: Optional[List[str]] = None
    near_neighbor_label: Optional[str] = None
    tell_apart_rule: Optional[str] = None
```

#### 4.3 Update ThemeEnrichedCodebookEntry (Line ~114)
```python
class ThemeEnrichedCodebookEntry(CodebookEntry):
    code: Optional[str] = None
    definition: Optional[str] = None
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    category: str = ""
    category_description: str = ""
    source_cluster: Optional[Union[int, str]] = None
    # NEW FIELDS (inherited from CodebookEntry):
    inclusion_examples: Optional[List[str]] = None
    exclusion_examples: Optional[List[str]] = None
    near_neighbor_label: Optional[str] = None
    tell_apart_rule: Optional[str] = None
```

---

### PHASE 5: Update Pipeline Step 6 (pipeline.py)

#### 5.1 Extract assignment_examples when building codebook (~line 1118)

```python
# Build codebook from generation results
codebook_entries = []
for item in codebook_reasoning.cluster_results:
    codebook_entry = models.CodebookEntry(
        code=item['code'],
        definition=item['definition'],
        source_cluster=item['source_cluster_id'],
        # NEW FIELDS:
        inclusion_examples=item.get('inclusion_examples'),
        exclusion_examples=item.get('exclusion_examples'),
        near_neighbor_label=item.get('near_neighbor_label'),
        tell_apart_rule=item.get('tell_apart_rule')
    )
    codebook_entries.append(codebook_entry)
```

#### 5.2 Preserve through theme enrichment (~line 1270)

**When building theme_enriched_codebook, preserve assignment_examples:**
```python
theme_enriched_entry = models.ThemeEnrichedCodebookEntry(
    code=entry.code,
    definition=entry.definition,
    theme=theme_name,
    theme_description=theme_description,
    category=category_name,
    category_description=category_description,
    source_cluster=entry.source_cluster,
    # PRESERVE ASSIGNMENT EXAMPLES:
    inclusion_examples=entry.inclusion_examples,
    exclusion_examples=entry.exclusion_examples,
    near_neighbor_label=entry.near_neighbor_label,
    tell_apart_rule=entry.tell_apart_rule
)
```

---

### PHASE 6: Update Code Assignment Prompts (prompts.py)

#### 6.1 Update DEFAULT_CODE_EVALUATION_PROMPT (~line 1371)

**Add to prompt after showing code/definition:**
```
Assignment Guidance:
- Include when: {inclusion_examples}
- Exclude when: {exclusion_examples}
- Boundary: This code covers {default_code}, which differs from {near_neighbor_label}
  Tell apart: {tell_apart_rule}
```

**Update format call in codeAssigner.py:**
```python
prompt = DEFAULT_CODE_EVALUATION_PROMPT.format(
    language=self.language,
    var_lab=self.var_lab,
    idea_id=idea_id,
    idea_text=idea_text,
    default_code=default_code.code,
    default_definition=default_code.definition,
    # NEW:
    inclusion_examples=self._format_examples_list(default_code.inclusion_examples),
    exclusion_examples=self._format_examples_list(default_code.exclusion_examples),
    near_neighbor_label=default_code.near_neighbor_label or "Unknown",
    tell_apart_rule=default_code.tell_apart_rule or "N/A"
)
```

#### 6.2 Update FALLBACK_CODE_ASSIGNMENT_PROMPT (~line 1423)

**Update code listing format:**
```
Available Codes:
{for each code}
Code: {code.code}
Definition: {code.definition}
Include when: {code.inclusion_examples}
Exclude when: {code.exclusion_examples}
Boundary: Differs from {code.near_neighbor_label} - {code.tell_apart_rule}
---
{end for}
```

**Update format call in codeAssigner.py:**
```python
all_codes_text = "\n".join([
    f"Code: {code.code}\n"
    f"Definition: {code.definition}\n"
    f"Include when: {self._format_examples_list(code.inclusion_examples)}\n"
    f"Exclude when: {self._format_examples_list(code.exclusion_examples)}\n"
    f"Boundary: Differs from {code.near_neighbor_label or 'Unknown'} - {code.tell_apart_rule or 'N/A'}\n"
    for code in self.codebook
])
```

#### 6.3 Add helper method in codeAssigner.py

```python
def _format_examples_list(self, examples: Optional[List[str]]) -> str:
    """Format examples list for prompt display"""
    if not examples:
        return "No specific examples provided"
    return "\n".join([f"  • {ex}" for ex in examples])
```

---

### PHASE 7: Update Pipeline Step 8 (pipeline.py)

#### 7.1 Pass full codebook with assignment_examples (~line 1595)

```python
assigner = CodeAssigner(
    cluster_models=cluster_models,
    codebook=[models.Codebook(
        code=entry.code,
        definition=entry.definition,
        theme=entry.theme,
        theme_description=entry.theme_description,
        source_cluster=entry.source_cluster,
        # NEW - PRESERVE ASSIGNMENT EXAMPLES:
        inclusion_examples=entry.inclusion_examples,
        exclusion_examples=entry.exclusion_examples,
        near_neighbor_label=entry.near_neighbor_label,
        tell_apart_rule=entry.tell_apart_rule
    ) for entry in theme_enriched_codebook.codes],
    var_lab=var_lab,
    code_to_theme_mapping=code_to_theme_mapping,
    config=code_assignment_config,
    model_config=model_config,
    processing_config=processing_config,
    verbose=verbose,
    prompt_printer=prompt_printer
)
```

---

## Testing Plan

### Test 1: Quick Validation (sample_size=20)
```bash
# Set environment variable
export SAMPLE_SIZE=20

# Run pipeline
cd src
python pipeline.py

# Verify:
# 1. Check codebook_generation_reasoning cache has assignment_examples in cluster_results
# 2. Check codebook cache has assignment_examples fields populated
# 3. Check final Excel export includes assignment_examples columns
# 4. Inspect code assignment prompts (use verbose mode or prompt_printer)
```

### Test 2: Assignment Quality Comparison
```python
# Compare assignment confidence before/after enhancement
# Run same dataset twice:
# 1. With old version (code/definition only)
# 2. With new version (code/definition + assignment_examples)

# Metrics to compare:
# - Average assignment confidence
# - Number of high-confidence assignments (≥0.7)
# - Number of low-confidence assignments (<0.5)
# - Stage 1 success rate (default code accepted)
# - Stage 2 fallback rate
```

### Test 3: Verify Updates Flow Through Chains
```python
# For a MODIFY decision, verify:
# 1. Chain 2 outputs updated_assignment_examples
# 2. Chain 3 receives and uses them
# 3. Chain 4 validates and refines them
# 4. Final codebook contains updated versions

# Use verbose mode + codegenResults utility:
samples = codegenResults.get_cluster_analysis(codebook_reasoning, cluster_id="5")
print("Inclusion examples:", samples['inclusion_examples'])
print("Exclusion examples:", samples['exclusion_examples'])
print("Tell apart rule:", samples['tell_apart_rule'])
```

---

## Risk Assessment

### Low Risk
- Adding optional fields to Pydantic models (backward compatible)
- Extending prompts with additional context (LLM will adapt)
- Adding helper methods in codeAssigner

### Medium Risk
- Updating prompt output schemas (requires LLM to follow new format)
- Mitigation: Make all new fields Optional with fallbacks
- Mitigation: Test with cheap config first

### High Risk
- None identified - this is primarily additive functionality

---

## Rollback Plan

If issues arise:
1. All new fields are Optional - system works without them
2. Fallback logic preserves old behavior when fields are None
3. Can disable by reverting prompts while keeping model changes
4. Cache invalidation will reprocess with old logic if needed

---

## Success Criteria

1. ✓ Assignment examples generated in Chain 1
2. ✓ Assignment examples updated in Chains 2-4 when codes are modified
3. ✓ Assignment examples stored in codebook models
4. ✓ Assignment examples passed to codeAssigner
5. ✓ Assignment examples visible in code assignment prompts
6. ✓ Higher average assignment confidence scores
7. ✓ Better handling of boundary cases between similar codes

---

## Implementation Order

1. **Phase 1**: Update Pydantic models (codeGenerator.py)
2. **Phase 4**: Update Codebook models (models.py)
3. **Phase 2**: Update prompts (prompts.py)
4. **Phase 3**: Update downstream processing (codeGenerator.py)
5. **Phase 5**: Update pipeline Step 6 (pipeline.py)
6. **Phase 6**: Update code assignment prompts (prompts.py)
7. **Phase 7**: Update pipeline Step 8 (pipeline.py)
8. **Test**: Run with cheap config and validate

Total estimated files to modify: 4
- codeGenerator.py
- models.py
- prompts.py
- pipeline.py

Total estimated lines to add/modify: ~500-700 lines
