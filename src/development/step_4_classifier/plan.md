# Step 4 Classifier — Issue Fix Plan

Source: analysis of verbose log `step4_taxonomy_20260325_130057.txt`

---

## Issue 1 — Config consolidation (two config files, one source of truth)

**Problem:** `src/config_steps/config_classifier.py` and `src/development/step_4_classifier/config_classifier.py` are out of sync. Steps 1-3 have a single canonical config in `config_steps/` — step 4 should too. Dev config is source of truth.

**Key difference:**
| Phase | config_steps (stale) | dev (source of truth) |
|---|---|---|
| P1 | nano | mini |
| P4 | nano | mini |
| P2/P5 | mini | default |

**Changes:**
1. Overwrite `src/config_steps/config_classifier.py` with full content of `src/development/step_4_classifier/config_classifier.py`
2. Update import in `src/development/step_4_classifier/classifier.py` line 69: `from .config_classifier` → `from config_steps.config_classifier`
3. Delete `src/development/step_4_classifier/config_classifier.py`


**Status** : RESOLVED
---

## Issue 2 — `veiligheid en personeel`: 0 facets, 103 ideas lost ✅ CLOSED

**Resolution:** Two-part fix, both delivered:

**Part A — 60s timeout floor** — done in Issue 1 config consolidation (`timeout_floor_seconds: 60.0`, `default_timeout_seconds: 60.0`). P1 log showed P95=31s with 4 timeouts at exactly 45s — 60s gives proper headroom.

**Part B — Fallback facet** — dropped. Synthetic facets corrupt downstream P4/P6 taxonomy structure. The timeout fix is the right solution; the persistent stats system (`data/model_perf_stats.json`) will further refine the floor empirically after real runs. If a domain still times out at 60s, it surfaces as a visible WARNING rather than silent data loss.

---

## Issue 3 — `organisatie en logistiek`: 20 attributes → 0 after P7 ✅ RESOLVED

**Problem:** P7 returned `ConsolidatedAttribute` objects with unrecognized `parent_facet` values. Rebuild loop produces `{}`. Line 783 overwrites domain with empty dict, destroying all 20 attributes.

**Fix:** Guard at line 798: if P7 returns 0 valid attributes but domain had some before, keep pre-P7 state.

---

## Issue 4 — Content drift warnings ✅ RESOLVED

### 4a — `canonical_phrasing:` literal in idea text
Some ideas have `idea = "Pinkpop Festival → canonical_phrasing: goede sfeer"` — the label leaks from step 3's field description. LLM strips it → similarity < 0.7 → assignment skipped.

**Fix:** Strip `canonical_phrasing:` pattern in `prompts_classifier.py` when building ideas block for P3 and P6.

### 4b — LLM strips template prefix in echo-back
Idea `"Pinkpop Festival → De vele schaduwplekken"` echoed back as `"De vele schaduwplekken"` — correct content, prefix stripped → similarity ~0.70, just below threshold.

**Fix:** Normalize both strings before SequenceMatcher by stripping template prefix from both sides before comparing (P3 ~line 1265, P6 ~line 1447).

---

## Issue 5 — Invalid `attribute_id`: `''`, `'A0'`, `'A?'`

**Problem:** P6 LLM returns invalid IDs — `''` (skipped), `'A0'` (off-by-one), `'A?'` (uncertainty placeholder). All cause skipped assignments at the `attr_id_to_name.get(...)` lookup → idea loses attribute classification permanently.

### 5a — Improve P6 prompt

**File:** `src/development/step_4_classifier/prompts_classifier.py`
**Location:** `build_attribute_assignment_prompt` — "Important requirements" block (~line 1196)

Add after the existing requirements:
```
- Attribute IDs start at A1 — A0 does not exist; never return 'A0'
- Never return an empty ID or a placeholder such as 'A?' — always pick the closest matching attribute
- When uncertain, choose the attribute whose description best matches the core meaning of the idea; do not leave the assignment blank
```

### 5b — Single-attribute fallback

**File:** `src/development/step_4_classifier/classifier.py`
**Location:** Invalid attribute_id guard (~line 1500)

Replace:
```python
attr_name = attr_id_to_name.get(assignment.assigned_attribute_id)
if attr_name is None:
    print(f"    WARNING: Invalid attribute_id ...")
    continue
```

With:
```python
attr_name = attr_id_to_name.get(assignment.assigned_attribute_id)
if attr_name is None:
    if len(attr_id_to_name) == 1:
        attr_name = next(iter(attr_id_to_name.values()))
    else:
        print(f"    WARNING: Invalid attribute_id '{assignment.assigned_attribute_id}' "
              f"for idea '{original_idea.idea_id}' — skipping")
        continue
```

**Logic:** if the facet has exactly one attribute there is no ambiguity — assign to it regardless of the invalid ID returned.

---

## Files changed

| File | Issues |
|---|---|
| `src/config_steps/config_classifier.py` | 1a (overwrite with dev content + 60s timeout) |
| `src/development/step_4_classifier/classifier.py` | 1b (import), 2b (fallback facet), 3 (P7 guard), 4b (similarity), 5b (attr fallback) |
| `src/development/step_4_classifier/prompts_classifier.py` | 4a (strip canonical_phrasing), 5a (prompt) |
| `src/development/step_4_classifier/config_classifier.py` | 1c (**DELETE**) |
