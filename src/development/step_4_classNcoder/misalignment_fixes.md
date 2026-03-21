# Misalignment Fixes: All Assignment Stages

## Principle

Every ID that goes into an LLM prompt must come back validated. No fallbacks, no silent acceptance of invalid IDs, no raw strings as substitutes. If validation fails, the assignment is rejected — not silently accepted with corrupted data.

## Reference: qualityFilter best practices

qualityFilter.py implements proven defensive patterns that classNcoder should adopt:
- **Always override with original input ID** (never trust LLM-returned ID)
- **Create distinguishable fallback responses** (never skip silently)
- **Iterate ALL original inputs on merge** (not just successes)
- **Count reconciliation** (input count == output count)
- **Validate response before accessing fields**
- **Retry timed-out tasks**

Each fix below references the specific best practice(s) it applies.

---

## Fix 1: Validate returned idea_ids in P2 facet assignment

**Problem**: LLM could return an idea_id not in the input batch. Currently accepted silently.

**Best practice applied**: BP1 (always use original input ID, log drift)

**Location**: `qualitative_researcher.py` ~line 1280-1286

**Current**:
```python
for assignment in assignments:
    facet_name = facet_id_to_name.get(assignment.assigned_facet_id, assignment.assigned_facet_id)
    all_assignments[assignment.idea_id] = facet_name
```

**Fix**: Don't trust LLM-returned idea_id. Match by position within batch, use original idea_id. Log drift if LLM returns different ID.
```python
# Match assignments to batch ideas by position (like qualityFilter)
for idx, assignment in enumerate(assignments):
    if idx >= len(batch_ideas):
        print(f"    WARNING: LLM returned more assignments than ideas — skipping extra")
        break
    original_idea = batch_ideas[idx]

    # Drift detection (BP1)
    if assignment.idea_id != original_idea.idea_id:
        print(f"    ID DRIFT: LLM returned '{assignment.idea_id}' but input was '{original_idea.idea_id}'")

    # Always use ORIGINAL idea_id (BP1)
    all_assignments[original_idea.idea_id] = facet_name
```

**Alternative (ID-based, not positional)**: Keep ID-based matching but validate:
```python
expected_ids = {idea.idea_id for idea in batch_ideas}
for assignment in assignments:
    if assignment.idea_id not in expected_ids:
        print(f"    WARNING: LLM returned unexpected idea_id '{assignment.idea_id}' — skipping")
        continue
    all_assignments[assignment.idea_id] = facet_name
```

**Decision needed**: positional override (qualityFilter pattern) vs ID-based with validation. Recommendation: ID-based with validation, since batches may have >1 idea and LLM order is not guaranteed.

**Also**: Add to Pydantic Field description: "Return the EXACT idea_id from the input. Do not modify or generate new IDs."

**Files**: `qualitative_researcher.py`, `prompts_exp.py` (FacetAssignment model)

---

## Fix 2: No invalid facet_id fallback in P2

**Problem**: Invalid facet_id (e.g., "sustainability" instead of "F3") falls back to raw string as facet name.

**Best practice applied**: BP6 (validate response before accessing fields)

**Location**: `qualitative_researcher.py` ~line 1282-1284

**Current**:
```python
facet_name = facet_id_to_name.get(
    assignment.assigned_facet_id,
    assignment.assigned_facet_id,  # FALLBACK: raw string used as name
)
```

**Fix**: Reject invalid IDs, don't use raw string as fallback:
```python
facet_name = facet_id_to_name.get(assignment.assigned_facet_id)
if facet_name is None:
    print(f"    WARNING: Invalid facet_id '{assignment.assigned_facet_id}' "
          f"for idea '{assignment.idea_id}' — skipping")
    continue
```

**Files**: `qualitative_researcher.py`, `prompts_exp.py` (FacetAssignment model — tighten Field description)

---

## Fix 3: Detect duplicate idea_id assignments

**Problem**: If same idea_id appears in multiple batch results, dict overwrites silently.

**Best practice applied**: BP4 (count reconciliation — detect unexpected duplicates)

**Location**: `qualitative_researcher.py` ~line 1286

**Fix**: Log duplicates (don't block — document why they shouldn't happen):
```python
if assignment.idea_id in all_assignments:
    print(f"    WARNING: Duplicate assignment for '{assignment.idea_id}' — "
          f"old='{all_assignments[assignment.idea_id]}', new='{facet_name}'")
all_assignments[assignment.idea_id] = facet_name
```

**Files**: `qualitative_researcher.py`

---

## Fix 4: Create fallback entries for unmatched ideas

**Problem**: Ideas with no assignment from LLM are silently dropped — no count, no log.

**Best practices applied**: BP2 (distinguishable fallback responses) + BP3 (iterate ALL originals) + BP4 (count reconciliation)

**Location**: `qualitative_researcher.py` — after all batches in `_run_facet_assignment()`

**Fix**: After all batches processed, iterate ALL input ideas and create fallback for missing:
```python
# BP3: iterate ALL originals
expected_ids = {idea.idea_id for idea in ideas}
assigned_ids = set(all_assignments.keys())
missing = expected_ids - assigned_ids

# BP4: count reconciliation
if missing:
    print(f"    WARNING: {len(missing)}/{len(ideas)} ideas received no facet assignment")

# BP2: create distinguishable fallback
for idea_id in missing:
    all_assignments[idea_id] = "__UNASSIGNED__"
```

**Files**: `qualitative_researcher.py`

---

## Fix 5: Validate returned idea_ids in P4a attribute assignment

**Problem**: Same as Fix 1 but for attribute assignment.

**Best practice applied**: BP1 (always use original input ID, log drift)

**Location**: `qualitative_researcher.py` ~line 1379-1385

**Fix**: Same pattern as Fix 1 — validate against expected_ids, log drift, use original ID:
```python
expected_ids = {idea.idea_id for idea in batch_ideas}
for assignment in assignments:
    if assignment.idea_id not in expected_ids:
        print(f"    WARNING: LLM returned unexpected idea_id '{assignment.idea_id}' — skipping")
        continue
    # Always use validated idea_id
    all_assignments[assignment.idea_id] = attr_name
```

**Files**: `qualitative_researcher.py`, `prompts_exp.py` (AttributeAssignment model)

---

## Fix 6: No invalid attribute_id fallback in P4a

**Problem**: Same as Fix 2 but for attribute assignment. Invalid attribute_id falls back to raw string.

**Best practice applied**: BP6 (validate response before accessing fields)

**Location**: `qualitative_researcher.py` ~line 1381-1384

**Fix**: Same pattern as Fix 2 — reject invalid IDs:
```python
attr_name = attr_id_to_name.get(assignment.assigned_attribute_id)
if attr_name is None:
    print(f"    WARNING: Invalid attribute_id '{assignment.assigned_attribute_id}' "
          f"for idea '{assignment.idea_id}' — skipping")
    continue
```

**Files**: `qualitative_researcher.py`, `prompts_exp.py` (AttributeAssignment model)

---

## Fix 7: Iron-clad idea_id generation in ideaExtractor

**Problem**: idea_id = `f"{respondent_id}_{counter}"` could collide across questions. respondent_id type is `Any`.

**Best practice applied**: BP1 (type safety on IDs)

**Location**: `ideaExtractor.py` ~line 1844-1846

**Fix**:
```python
# Ensure respondent_id is always string, stripped
resp_id = str(task['respondent_id']).strip()
idea_counter = i + 1
idea_id = f"{resp_id}_{idea_counter}"
```

**Additional safeguards**:
- Cast respondent_id to str consistently at entry point
- Add global collision detection: `if idea_id in seen_ids: raise ValueError(...)`
- Consider including variable name in ID if processing multiple variables

**Files**: `ideaExtractor.py` (later — not classNcoder, separate task)

---

## Fix 8: Validate code assignment in P5

### 8a: Log per-task code_id resolution failures

**Problem**: If LLM returns code_id not in scoped candidate list, falls back silently.

**Best practice applied**: BP6 (validate response before accessing fields)

**Location**: `code_assignment.py` ~line 556-564

**Fix**: Log every fallback:
```python
label = task_id_map.get(cat_id)
if label:
    self._per_task_resolutions[idea.idea_id] = label
else:
    print(f"    WARNING: Code ID '{cat_id}' not in scoped candidates for "
          f"idea '{idea.idea_id}' — falling back to global resolution")
```

### 8b: Block global fallback when embedding pre-filter is active

**Problem**: When pre-filter scopes codes to C1-C5, but global resolution maps C3 to a DIFFERENT code than the scoped C3. Wrong code assigned silently.

**Best practice applied**: BP1 (use original/scoped ID, not global fallback)

**Location**: `code_assignment.py` ~line 479-496

**Fix**: When embedding pre-filter is active, ONLY use per-task resolution:
```python
if idea_id in self._per_task_resolutions:
    id_resolution[idea_id] = self._per_task_resolutions[idea_id]
elif self._idea_code_candidates:
    # Pre-filter active — do NOT use global fallback (IDs are scoped)
    print(f"    WARNING: No scoped resolution for idea '{idea_id}' — unassigned")
else:
    # No pre-filter — global resolution is safe
    raw_id = getattr(assignment, 'assigned_code_id', '') or ''
    cat_id = self._normalize_id(raw_id)
    label = self._id_to_label.get(cat_id)
    if label:
        id_resolution[idea_id] = label
```

### 8c: Assert single assignment per task

**Problem**: P5 sends one idea per task but wraps result as a list. If list has >1 item, only first is used.

**Best practice applied**: BP6 (validate response before accessing fields)

**Location**: `code_assignment.py` ~line 712-721

**Fix**: Add assertion:
```python
if hasattr(result, 'assignments') and len(result.assignments) > 1:
    print(f"    WARNING: Multiple assignments returned for single idea — using first")
```

### 8d: Create fallback entries for unassigned ideas in P5

**Problem**: Ideas with no code assignment get None fields in output model. No distinguishable fallback.

**Best practices applied**: BP2 (fallback responses) + BP3 (iterate all originals) + BP4 (count reconciliation)

**Location**: `code_assignment.py` ~line 1061-1117 (`_build_output_models`)

**Fix**: Already iterates all originals. Add explicit fallback and count:
```python
# In _build_output_models, after building all new_ideas:
unassigned = [i for i in new_ideas if not i.assigned_code]
if unassigned:
    print(f"    WARNING: {len(unassigned)} ideas have no code assignment")
    for idea in unassigned:
        idea.assigned_code = "__UNASSIGNED__"
        idea.confidence = 0.0
```

### 8e: Retry timed-out tasks before fallback

**Problem**: classNcoder collects timed-out tasks but never retries them.

**Best practice applied**: BP5 (retry timed-out tasks — qualityFilter uses tenacity, ideaExtractor does batch retry)

**Location**: `code_assignment.py` after main processing loop

**Fix**: Retry with reduced concurrency:
```python
if timed_out:
    print(f"  Retrying {len(timed_out)} timed-out tasks...")
    for idx, task in timed_out:
        try:
            result = await self._process_task_with_retry(task)
            results[idx] = result
        except Exception as e:
            print(f"    Retry failed for idea {task['idea'].idea_id}: {e}")
```

---

## Implementation priority

| Fix | Severity | Best practice | Effort | Priority |
|-----|----------|---------------|--------|----------|
| Fix 1 | **CRITICAL** | BP1 (original ID) | Small | **P0** |
| Fix 5 | **CRITICAL** | BP1 (original ID) | Small | **P0** |
| Fix 4 | **HIGH** | BP2+BP3+BP4 (fallback+iterate+count) | Small | **P0** |
| Fix 2 | MEDIUM | BP6 (validate response) | Small | **P1** |
| Fix 6 | MEDIUM | BP6 (validate response) | Small | **P1** |
| Fix 8b | MEDIUM | BP1 (scoped ID only) | Medium | **P1** |
| Fix 8d | MEDIUM | BP2+BP3+BP4 (fallback+iterate+count) | Small | **P1** |
| BP4 count reconciliation | MEDIUM | BP4 | Small | **P1** |
| Fix 7 | MEDIUM | BP1 (type safety) | Medium | **P2** |
| Fix 3 | MEDIUM | BP4 (count) | Small | **P2** |
| Fix 8e | LOW | BP5 (retry) | Medium | **P2** |
| Fix 8a | LOW | BP6 (validate) | Small | **P2** |
| Fix 8c | LOW | BP6 (validate) | Small | **P3** |

## Files to modify

| File | Fixes |
|------|-------|
| `qualitative_researcher.py` | Fix 1, 2, 3, 4, 5, 6 |
| `prompts_exp.py` | Fix 1, 2, 5, 6 (model field descriptions) |
| `code_assignment.py` | Fix 8a, 8b, 8c, 8d, 8e |
| `ideaExtractor.py` | Fix 7 (later — not classNcoder) |

## Implementation order

1. **Round 1 (P0)**: Fix 1, 5, 4 — ID validation + fallbacks across P2 and P4a
2. **Round 2 (P1)**: Fix 2, 6, 8b, 8d + count reconciliation — reject invalid IDs, block global fallback, fallbacks in P5
3. **Round 3 (P2)**: Fix 3, 7, 8e, 8a — duplicate detection, iron-clad IDs, retry, logging
4. **Round 4 (later)**: Fix 7 in ideaExtractor (separate from classNcoder)
