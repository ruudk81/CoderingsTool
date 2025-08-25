# CodeGeneratorReasoningResults Requirements Analysis

## 1. Data Structure Requirements

Based on analysis of `displayResults.py`, `promptTester.py`, and `models.py`, here's what needs to be in the `codebook_reasoning` object:

### Core Required Fields (from models.py)

```python
class CodeGeneratorReasoningResults(BaseModel):
    # Raw data
    cluster_results: List[Dict[str, Any]]
    
    # INPUTS to each prompt (for promptTester transparency)
    step1_inputs: Dict[int, Dict[str, Any]] = {}  # Cluster summary inputs
    step2_inputs: Dict[int, Dict[str, Any]] = {}  # Candidate selection inputs  
    step3_inputs: Dict[int, Dict[str, Any]] = {}  # Code generation inputs
    step4_inputs: Dict[int, Dict[str, Any]] = {}  # Validation inputs
    
    # OUTPUTS from each prompt (for displayResults)
    step1_summaries: Dict[int, Dict[str, Any]]     # Theme extraction results
    step2_analysis: Dict[int, List[Dict[str, str]]]  # Candidate codes list
    step3_recommendations: Dict[int, Dict[str, Any]]  # Code generation results
    step4_validations: Dict[int, Dict[str, Any]]     # Validation results
    
    # Additional fields
    cluster_assignments: Dict[int, Dict[str, Any]]
    stats: Dict[str, Any]
    # ... other metadata fields
```

## 2. DisplayResults Expectations

From `codeGenerator_displayResults.py`:

### What displayResults reads:
- `step1_summaries[cluster_id]` → themes data
- `step1_inputs[cluster_id]['cluster_ideas']` → cluster ideas text 
- `step2_analysis[cluster_id]` → list of candidate codes `[{'code': str, 'definition': str}, ...]`
- `step3_recommendations[cluster_id]` → coding decisions
- `step4_validations[cluster_id]` → validation decisions  
- `cluster_assignments[cluster_id]` → final codes

### Key Data Formats:
```python
# step1_summaries[cluster_id]
{
    "themes": [{"theme_name": str, ...}, ...]
}

# step1_inputs[cluster_id] 
{
    "cluster_ideas": str  # Text with cluster ideas
}

# step2_analysis[cluster_id]
[
    {"code": str, "definition": str},
    {"code": str, "definition": str}
]

# step3_recommendations[cluster_id]
{
    "coding_decisions": [
        {
            "decision": str,  # 'create_new', 'modify_existing', 'use_existing'
            "justification": str,
            "action_details": {...}
        }
    ]
}

# step4_validations[cluster_id]
{
    "code_validations": [
        {
            "decision": str,  # 'APPROVE', 'REVISE', 'REJECT', 'SPLIT'  
            "decision_rationale": str,
            "validated_code": {"code": str, "definition": str}
        }
    ]
}
```

## 3. PromptTester Expectations

From `promptTester.py`:

### What promptTester reads:
- `step1_summaries`, `step2_analysis`, `step3_recommendations`, `step4_validations` → for cluster selection
- `step1_inputs[cluster_id]`, `step2_inputs[cluster_id]`, `step3_inputs[cluster_id]`, `step4_inputs[cluster_id]` → for prompt reconstruction

### Key Requirements:
- Must have cluster_id as dictionary key for all step data
- All 4 steps must have data for a cluster to be considered "complete"
- Input data must match what was actually sent to each prompt

## 4. Critical Alignment Requirements

1. **Cluster IDs must be consistent** across all dictionaries
2. **step2_analysis must contain actual candidate codes** that were used in step3 prompt
3. **step3_recommendations must contain LLM reasoning** that aligns with step2 candidate codes
4. **All data must be captured at the right moments**:
   - step1_inputs: At cluster summary prompt construction
   - step2_analysis: At candidate code selection (actual codes used)
   - step3_inputs: At code generation prompt construction  
   - step3_recommendations: At code generation prompt output
   - step4_inputs: At validation prompt construction
   - step4_validations: At validation prompt output

## 5. Data Capture Strategy Needed

Need to capture:
1. **Cluster ideas** = input to prompt 1 (cluster summary)
2. **Candidate codes** = output from prompt 2 / input to prompt 3  
3. **Recommendations** = output from prompt 3 / input to prompt 4
4. **Validations** = output from prompt 4

All with correct cluster ID alignment and captured at the moment of actual usage.