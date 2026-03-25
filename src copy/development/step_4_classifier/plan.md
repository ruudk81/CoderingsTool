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

## Issue 2 — `veiligheid en personeel`: 0 facets, 103 ideas lost

**Problem:** Domain had 1 P1 chunk (95 obs). Chunk timed out. 0 facets → domain skipped in P3/P4/P6. All 103 ideas permanently unclassified.

**Two-part fix:**

**Part A — Raise timeout floor to 60s** in `ClassifierRampConfig` (src of truth = dev config, being replaced in issue 1):
- `timeout_floor_seconds: float = 60.0`
- `default_timeout_seconds: float = 60.0`

P1 log showed P95=31s with 4 timeouts at exactly 45s. 60s gives proper headroom for large discovery prompts (~5k tokens).

**Part B — Fallback facet** in `src/development/step_4_classifier/classifier.py` after P1 results (~line 475). If a domain produced 0 facets but has ideas, inject a single generic facet so no ideas are silently lost.

---

## Issue 3 — `organisatie en logistiek`: 20 attributes → 0 after P7

**Problem:** P7 returned `ConsolidatedAttribute` objects with unrecognized `parent_facet` values. Rebuild loop produces `{}`. Line 783 overwrites domain with empty dict, destroying all 20 attributes.

**Fix:** Guard at line 783: if P7 returns 0 valid attributes but domain had some before, keep pre-P7 state.

---

## Issue 4 — Content drift warnings

### 4a — `canonical_phrasing:` literal in idea text
Some ideas have `idea = "Pinkpop Festival → canonical_phrasing: goede sfeer"` — the label leaks from step 3's field description. LLM strips it → similarity < 0.7 → assignment skipped.

**Fix:** Strip `canonical_phrasing:` pattern in `prompts_classifier.py` when building ideas block for P3 and P6.

### 4b — LLM strips template prefix in echo-back
Idea `"Pinkpop Festival → De vele schaduwplekken"` echoed back as `"De vele schaduwplekken"` — correct content, prefix stripped → similarity ~0.70, just below threshold.

**Fix:** Normalize both strings before SequenceMatcher by stripping template prefix from both sides before comparing (P3 ~line 1265, P6 ~line 1447).

---

## Issue 5 — Invalid `attribute_id`: `''`, `'A0'`, `'A?'`

**Problem:** P6 LLM returns invalid IDs — `''` (skipped), `'A0'` (off-by-one), `'A?'` (uncertainty placeholder). All cause skipped assignments.

### 5a — Improve P6 prompt
Add explicit guidance: A0 does not exist, IDs start at A1; never return empty or placeholder; always pick closest match.

### 5b — Single-attribute fallback
If attribute_id is invalid but facet has only one attribute, assign to it automatically.

---

## Files changed

| File | Issues |
|---|---|
| `src/config_steps/config_classifier.py` | 1a (overwrite with dev content + 60s timeout) |
| `src/development/step_4_classifier/classifier.py` | 1b (import), 2b (fallback facet), 3 (P7 guard), 4b (similarity), 5b (attr fallback) |
| `src/development/step_4_classifier/prompts_classifier.py` | 4a (strip canonical_phrasing), 5a (prompt) |
| `src/development/step_4_classifier/config_classifier.py` | 1c (**DELETE**) |
