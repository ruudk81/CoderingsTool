# Phase 1: Prompt Change Checklist

## Overview
This document maps all dependencies when changing prompt outputs/parameters to ensure the prompt engineering workflow remains functional. A prompt change ripples through: **Prompt → codeGenerator internal flow → codebook_reasoning → displayResults/promptTester**.

## Critical Dependency Chain

### 1. Internal codeGenerator Dependencies (Step-to-Step Flow)

#### Step 1 → Step 2 Flow
**Location**: `_select_candidate_codes()` and `_select_candidate_codes_unlimited()`

**Current Data Transfer**:
```python
# Step1 Output (ClusterSummaryOutput) becomes Step2 Input
theme_data = themes[cluster_id]  # ClusterSummaryOutput object
cluster_summary = f"Theme: {theme_data.theme_name}\nDescription: {theme_data.theme_description}"
```

**Dependencies**:
- `theme_data.theme_name` → Used in cluster_summary parameter
- `theme_data.theme_description` → Used in cluster_summary parameter
- Survey question (`self.var_lab`) → Used in prompt formatting

#### Step 2 → Step 3 Flow
**Location**: `_generate_code()` and `_generate_code_unlimited()`

**Current Data Transfer**:
```python
# Step2 Output (List[CandidateCode]) becomes Step3 Input
candidate_codes_text = "\n".join([
    f"Code: {code.code}\nDefinition: {code.definition}" 
    for code in candidate_selection
])
```

**Dependencies**:
- `candidate_selection[].code` → Used in candidate_codes_text parameter
- `candidate_selection[].definition` → Used in candidate_codes_text parameter
- Theme data from Step 1 → Still used in Step 3 prompt

#### Step 3 → Step 4 Flow
**Location**: `_validate_and_update_codebook()` and `_validate_code_unlimited()`

**Current Data Transfer**:
```python
# Step3 Output (CodeRecommendation) becomes Step4 Input
decisions_text = []
for decision in code_generation.coding_decisions:
    decisions_text.append(f"Decision: {decision.decision}")
    decisions_text.append(f"Justification: {decision.justification}")
```

**Dependencies**:
- `code_generation.coding_decisions[].decision` → Used in validation prompt
- `code_generation.coding_decisions[].justification` → Used in validation prompt
- Original theme and cluster data → Still referenced in validation

### 2. codeGenerator → Pipeline Interface

#### CodeGeneratorReasoningResults Structure
**Location**: `generate_async()` method in codeGenerator.py

**Critical Fields** (used by downstream tools):
```python
CodeGeneratorReasoningResults(
    # Input capture for promptTester
    step1_inputs: Dict[int, Dict[str, Any]]      # ← promptTester reads this
    step2_inputs: Dict[int, Dict[str, Any]]      # ← promptTester reads this  
    step3_inputs: Dict[int, Dict[str, Any]]      # ← promptTester reads this
    step4_inputs: Dict[int, Dict[str, Any]]      # ← promptTester reads this
    
    # Output capture for displayResults
    step1_summaries: Dict[int, Dict[str, Any]]   # ← displayResults reads this
    step2_analysis: Dict[int, List[Dict[str, str]]]  # ← displayResults reads this
    step3_recommendations: Dict[int, Dict[str, Any]]  # ← displayResults reads this
    step4_validations: Dict[int, Dict[str, Any]]     # ← displayResults reads this
    
    # Other fields...
    cluster_assignments: Dict[int, Dict[str, Any]]   # ← displayResults reads this
)
```

### 3. Downstream Tool Dependencies

#### displayResults.py Dependencies
**Location**: `C:\Users\rkn\Python_apps\CoderingsTool\src\utils\codeGenerator_displayResults.py`

**Critical Field Access Patterns**:
```python
# Theme display - reads step1_summaries
themes = getattr(results, 'step1_summaries', {}).get(cluster_id, {}).get('themes', [])

# Cluster ideas display - reads step1_inputs  
cluster_ideas = getattr(results, 'step1_inputs', {}).get(cluster_id, {}).get('cluster_ideas', '')

# Candidate codes display - reads step2_analysis
candidate_codes = getattr(results, 'step2_analysis', {}).get(cluster_id, [])

# Recommendations display - reads step3_recommendations
recommendations = getattr(results, 'step3_recommendations', {}).get(cluster_id, {}).get('coding_decisions', [])

# Validation display - reads step4_validations
validations = getattr(results, 'step4_validations', {}).get(cluster_id, {}).get('code_validations', [])
```

#### promptTester.py Dependencies  
**Location**: `C:\Users\rkn\Python_apps\CoderingsTool\src\utils\promptTester.py`

**Critical Field Access Patterns**:
```python
# Prompt reconstruction - reads step1_inputs through step4_inputs
step1_params = getattr(results, 'step1_inputs', {}).get(cluster_id, {})
step2_params = getattr(results, 'step2_inputs', {}).get(cluster_id, {})
step3_params = getattr(results, 'step3_inputs', {}).get(cluster_id, {})
step4_params = getattr(results, 'step4_inputs', {}).get(cluster_id, {})

# Uses these to reconstruct original prompts exactly as sent to LLM
```

## Change Checklist

### When Modifying Prompt Outputs

#### ✅ Step 1: Update Pydantic Models
**File**: `src/models.py`
- [ ] Update `ClusterThemeItem` if changing theme structure
- [ ] Update `ClusterSummaryOutput` if changing step1 output
- [ ] Update `CandidateCode` if changing step2 output  
- [ ] Update `CodeRecommendation`/`CodingDecision` if changing step3 output
- [ ] Update `ValidationResult`/`CodeValidation` if changing step4 output

#### ✅ Step 2: Update Internal codeGenerator Flow
**File**: `src/utils/codeGenerator.py`

**For Step 1 → Step 2 changes**:
- [ ] Update `_select_candidate_codes()` around lines 1622-1650
- [ ] Update `_select_candidate_codes_unlimited()` around lines 2425-2465
- [ ] Search for `theme_data.theme_name` and `theme_data.theme_description` usage

**For Step 2 → Step 3 changes**:
- [ ] Update `_generate_code()` around lines 1684-1720  
- [ ] Update `_generate_code_unlimited()` around lines 2470-2510
- [ ] Search for `candidate_selection` and `code.code`/`code.definition` usage

**For Step 3 → Step 4 changes**:
- [ ] Update `_validate_and_update_codebook()` around lines 1771-1800
- [ ] Update `_validate_code_unlimited()` around lines 2534-2570
- [ ] Search for `coding_decisions` usage

#### ✅ Step 3: Update Data Capture
**File**: `src/utils/codeGenerator.py`

- [ ] Update input capture in `_capture_prompt_params()` method (lines 620-625)
- [ ] Update step1_summaries capture (around line 733)
- [ ] Update step2_analysis capture (around lines 1672, 1719, 2459)  
- [ ] Update step3_recommendations capture (around lines 1751, 2507)
- [ ] Update step4_validations capture (around lines 1904, 2571)

#### ✅ Step 4: Update displayResults.py
**File**: `src/utils/codeGenerator_displayResults.py`

- [ ] Update theme display logic (search for `'themes'`)
- [ ] Update cluster ideas display (search for `'cluster_ideas'`)  
- [ ] Update candidate codes display (search for `step2_analysis`)
- [ ] Update recommendations display (search for `'coding_decisions'`)
- [ ] Update validation display (search for `'code_validations'`)

#### ✅ Step 5: Update promptTester.py
**File**: `src/utils/promptTester.py`

- [ ] Update prompt reconstruction logic for affected steps
- [ ] Ensure parameter names match what's captured in step inputs
- [ ] Test prompt reconstruction with new output format

#### ✅ Step 6: Update Prompt Templates
**File**: `src/prompts.py`

- [ ] Update `CLUSTER_SUMMARY_PROMPT` if changing Step 1
- [ ] Update `CANDIDATE_CODE_SELECTION_PROMPT` if changing Step 2  
- [ ] Update `CODE_GENERATION_PROMPT` if changing Step 3
- [ ] Update `VALIDATION_PROMPT` if changing Step 4

### Validation Steps

#### ✅ Critical Validation (Must Pass)
1. **Run codeGenerator**: Ensure it completes without errors
2. **Test displayResults**: Verify all sections display correctly
3. **Test promptTester**: Ensure prompts can be reconstructed and tested
4. **Check data completeness**: Ensure all expected fields are captured

#### ✅ Validation Script Template
```python
# Quick validation script
def validate_prompt_changes(codebook_reasoning):
    """Validate that prompt changes didn't break downstream tools"""
    
    # Check displayResults dependencies
    for cluster_id in codebook_reasoning.step1_summaries.keys():
        assert 'themes' in codebook_reasoning.step1_summaries[cluster_id]
        assert cluster_id in codebook_reasoning.step2_analysis
        assert cluster_id in codebook_reasoning.step3_recommendations
        assert cluster_id in codebook_reasoning.step4_validations
    
    # Check promptTester dependencies  
    assert len(codebook_reasoning.step1_inputs) > 0
    assert len(codebook_reasoning.step2_inputs) > 0
    assert len(codebook_reasoning.step3_inputs) > 0
    assert len(codebook_reasoning.step4_inputs) > 0
    
    print("✅ All dependencies validated")
```

## Common Field Mappings

### Current Field Names to Track
| Component | Field Name | Usage |
|-----------|------------|-------|
| ClusterThemeItem | `theme_name` | Used in step2 prompt parameter |
| ClusterThemeItem | `summary` | Displayed by displayResults |
| CandidateCode | `code` | Used in step3 prompt parameter |
| CandidateCode | `definition` | Used in step3 prompt parameter |
| CodingDecision | `decision` | Used in step4 prompt parameter |
| CodingDecision | `justification` | Used in step4 prompt parameter |

### Naming Conventions
- Use consistent field names across all steps
- Prefer explicit names: `theme_name` not `name`
- Use plural for lists: `themes` not `theme`
- Keep parameter names descriptive: `cluster_ideas` not `ideas`

## Emergency Rollback
If changes break downstream tools:
1. Revert Pydantic models in `models.py`
2. Revert data capture changes in `codeGenerator.py`
3. Test that displayResults and promptTester work again
4. Investigate specific failure points before retry