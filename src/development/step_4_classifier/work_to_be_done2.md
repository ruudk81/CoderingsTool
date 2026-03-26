# Work To Be Done 2 — step_4_classifier (post-run analysis, 2026-03-26)

Issues identified from verbose log analysis of Q20/500 run.

---

## Issue 1 — Invalid `attribute_id ''` and `'?'` still occurring (P6)

**Severity:** High — each invalid ID silently drops an idea from attribute assignment permanently.

**Observed:**
- 10/73 ideas skipped with `attribute_id ''` in facet `Betere voedingsbalans` (3 attributes)
- 1/91 ideas skipped with `attribute_id '?'` in facet `Minder zout, suiker en (verzadigd) vet`
- Total: 11 ideas lost in this run (≈2% of assigned ideas)

**Root cause:**
The prompt fix (plan.md Issue 5a) and single-attribute fallback (5b) partially help, but the core problem remains: the LLM occasionally returns empty or placeholder IDs even for multi-attribute facets. The single-attribute fallback only recovers when a facet has exactly 1 attribute — it cannot help when 2+ attributes exist.

**Options to investigate:**
- **1a — Extend the fallback**: when `attribute_id` is `''` or `'?'`, attempt a similarity-based recovery (match idea text against attribute descriptions) rather than immediately skipping. This is safe because attribute descriptions are available at the point of assignment.
- **1b — Strengthen the prompt**: make the invalid-ID rules more prominent (e.g., move to top of requirements block, use ALL CAPS or "CRITICAL:" prefix).
- **1c — Retry on invalid ID**: treat an invalid `attribute_id` as a recoverable error and retry the single idea against its facet with a tightened prompt.

---

## Issue 2 — Idea ID drift in P3 (batch contamination)

**Severity:** Low-medium — 1 idea unassigned per run; could worsen at larger scale.

**Observed:**
```
ID DRIFT: LLM returned unexpected idea_id '399288995_1' in batch 0 — skipping
WARNING: 1/112 ideas received no facet assignment
```

**Root cause:** The LLM occasionally returns an idea ID from a different batch (cross-contamination in prompt context). The ID drift guard correctly catches and skips it, but the idea is then permanently unassigned.

**Options to investigate:**
- **2a — Retry on ID drift**: ideas that were skipped due to ID drift should be collected and retried as a standalone single-idea batch after the main pass.
- **2b — Prompt hardening**: make the idea-ID constraint more explicit (e.g., numbered list of valid IDs included in the prompt) to reduce LLM ID hallucination.

---

## Issue 3 — No cold-start stats for gpt-5-mini phases (P1, P4, P7)

**Severity:** Low — affects cold-start timeout floor accuracy only; warm-up corrects quickly.

**Observed:** `model_perf_stats.json` has entries for `gpt-5-nano` (P3, P6) but nothing for `gpt-5-mini` phases:
- `step4_p1_facet_discovery` — missing
- `step4_p4_attribute_discovery` — missing
- `step4_p7_consolidation` — missing

**Root cause:** P1, P4, P7 are small phases (8–18 tasks each) — the classifier may not be writing stats for them, or the MIN_SAMPLES threshold (10) is never reached in a single run.

**To investigate:**
- Check whether `update_phase_stats` is called for P1/P4/P7 in `classifier.py`
- If not, add stats tracking for those phases
- Consider lowering `MIN_SAMPLES` for phases that structurally never exceed 20 tasks (e.g., P1 with 7 domains will always have ≈7–10 tasks)

---

## Issue 4 — `other` domain passes through pipeline with 0 facets (silent no-op)

**Severity:** Low — currently harmless, but 31 ideas (5% of total) are permanently unclassified.

**Observed:**
```
Domain 'other': 23 observations, 1 chunk(s) of ~23 (overlap 20%)
Domain 'other' facets: 0 raw → 0 consolidated [6.6s]
other: (n=23, 1 chunk(s), 0 facets, 0 assigned, 0 attributes)
```

**Root cause:** The `other` domain receives a P1 call (costs tokens) but predictably produces 0 facets because the ideas are definitionally heterogeneous. The domain then propagates through the full pipeline doing nothing useful.

**Options to investigate:**
- **4a — Skip `other` in P1**: detect the `other` domain before P1 and exclude it from facet discovery entirely (saves 1 API call per run).
- **4b — Log `other` ideas separately**: surface the 31 unclassified ideas in the verbose report so they are visible rather than silently dropped.

---

## Summary table

| Issue | Severity | Quick fix available? |
|---|---|---|
| 1 — Invalid attribute_id '' / '?' | High | Partial (similarity fallback or retry) |
| 2 — ID drift → unassigned idea | Low-medium | Yes (retry deferred ideas) |
| 3 — No stats for mini phases P1/P4/P7 | Low | Yes (add stats write calls) |
| 4 — `other` domain wastes P1 call | Low | Yes (skip detection) |
