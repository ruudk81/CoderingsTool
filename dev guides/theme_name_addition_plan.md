# Impact Analysis: Adding `theme_name` to CLUSTER_SUMMARY_PROMPT Output

## Current Structure → New Structure
**From:**
```json
[{"theme_id": 1, "theme_statement": "<theme statement here>"}]
```

**To:**
```json
[{"theme_id": 1, "theme_name": "<name of theme here>", "theme_statement": "<theme statement here>"}]
```

## **1. MODELS.PY Changes**
**File:** `src/models.py`
**Target:** `ClusterThemeItem` class (lines 96-101)

**Required Changes:**
- Add `theme_name: str = Field(description="Short thematic name (2-5 words)")` 
- Keep existing `theme_id: int` and `theme_statement: str`
- Update field ordering if desired

## **2. CODEGENERATOR.PY Changes**
**File:** `src/utils/codeGenerator.py`

### **2.1 Data Access & Storage** (14 locations)
- **Line 372**: `theme_statements = [themes[cid].root[0].theme_statement for cid in cluster_ids]` 
  - Need parallel `theme_names` list for embedding labeling
- **Lines 830, 921**: Update `step1_summaries` storage to include theme_name
- **Line 918-922**: Capture both theme_name and theme_statement in results

### **2.2 Helper Methods** (2 locations)  
- **Line 744-748**: `_get_theme_statement()` - **No change needed**
- **Add new method**: `_get_theme_name()` to extract theme_name

### **2.3 Prompt Parameter Formatting** (6 locations)
- **Lines 1814, 1881, 1963**: `cluster_summary = self._get_theme_statement(theme_data)`
  - **Decision needed**: Use theme_name, theme_statement, or both in downstream prompts?
  - May need to update how cluster_summary is formatted for prompts

### **2.4 Display & Logging** (5 locations)
- **Lines 534-535, 1297, 1360, 1744**: Update verbose output to show theme_name + theme_statement
- **Line 1356**: Embedding generation - use theme_name, theme_statement, or both?

## **3. CODEGENRESULTS.PY Changes**  
**File:** `src/utils/codegenResults.py`

### **3.1 Theme Display Logic** (1 location)
- **Lines 38-48**: `themes = step1_data.get("themes", [])`
  - Currently uses `str(theme)` to display themes
  - **Update needed**: Extract and display both theme_name and theme_statement
  - Format as: `"[theme_name]: [theme_statement]"` or similar

## **4. CODEGENPROMPTTESTER.PY Changes**
**File:** `src/utils/codegenPromptTester.py`  
- **No direct changes needed** - uses actual pipeline inputs via `step1_inputs`, `step2_inputs`, etc.
- **Indirect impact**: Will display updated prompt formats automatically

## **5. PROMPTS.PY Considerations**
**Files:** `src/prompts.py`

### **5.1 Downstream Prompt Templates** (3 prompts affected)
- **CANDIDATE_CODE_SELECTION_PROMPT** (line 364): `{cluster_summary}`
- **CODE_GENERATION_PROMPT** (line 421): `{cluster_summary}`  
- **VALIDATION_PROMPT** (line 511): `{cluster_summary}`

### **5.2 Decision Needed:**
- Should `cluster_summary` contain:
  - Just `theme_statement` (current behavior) ✓
  - Just `theme_name` 
  - Both: `"Theme Name: [name]\nTheme Statement: [statement]"`
  - Combined: `"[name]: [statement]"`

## **Implementation Priority:**
1. **High**: Models.py, codeGenerator data storage/access  
2. **Medium**: codegenResults display logic
3. **Low**: Logging/display improvements, prompt formatting decisions