# Cross-Domain Attribute Consolidation (P8) — Status & Remaining Work

## Status

- **Code integration: DONE.** P8 runs after P7 in `run_classifier.py`, overwriting the `taxonomy` / `taxonomy_classified` cache in place.
- **Remap bugfixes: DONE** (A–F, see below).
- **End-to-end verification: DONE** (2026-06-07, Merk X 2000). P8 ran clean: 26 → 21 attributes, 1846 ideas preserved (zero loss), `P8 consistency: OK` (no violations).
- **Dev docs: DONE.** All six docs (CLAUDE, ARCHITECTURE, PROCESSING, CACHE_LOGIC, COST_TRACKING, CONSOLIDATION_LOGIC) reflect P8.
- **Tooling: handled via existing view scripts** (option C). No new `view_cross_domain_results.py` / `debug_cross_domain_prompts.py` were built — the post-P8 taxonomy is inspected with `view_taxonomy.py` and `view_assignments_attributes_consolidated.py` (both read the P8-overwritten cache), and the merge report is in the run's verbose log.

## Remap bugfixes (A–F)

The integrated P8 had several remap correctness bugs, all fixed and verified:

- **A — merge map keyed by `(domain, attribute_name)`.** Bare attribute names are not unique across domains; the old code keyed on bare name, so cache and growing model could diverge and the wrong/extra ideas got remapped. LLM source names are now resolved against the attributes present in each window.
- **B — valence/confidence carried.** Remap dropped `attribute_valence` / `attribute_confidence`; they now move with the idea.
- **C — `_verify_consistency()` self-check.** Verifies idea-count preservation, no dropped valence/confidence, and no orphan assignments; prints `P8 consistency: OK` or warnings; sets `stats["consistency_violations"]`.
- **D — same-domain facet move.** `facet_assignments` is now updated for same-domain merges that change facet (previously only cross-domain).
- **E — guard for <2 attributes.** P8 is skipped (no seriation crash, nothing to consolidate).
- **F — `idea.domain` kept in sync with `partition_name`.** P8 is the first phase to move ideas across domains; both domain fields are now updated together.

Plus: a `P8 merge report` is printed in the main `run_classifier.py` run (was only in `run_consolidator.py`).

## Remaining work

### Phase 2 — Fine-tune (open)
Cross-domain over-merge still occurs (e.g. heterogeneous `bankprofiel` attributes and the whole `other` domain pulled into `merkindruk`). Knobs: raise `p8_similarity_threshold`, shrink `p8_window_size`, soften prevalence weighting in the prompt. Iterate with `run_consolidator.py` (re-runs P8 on cached P7, fast).

**`__UNASSIGNED__` exclusion — DONE.** Sentinels (`__UNASSIGNED__`, `(no attribute)`) are excluded from P8 via `_SENTINEL_ATTRIBUTES`: `_collect_ideas_per_attribute` skips them (commit d723aa8) so they never get a centroid or enter a window, and `_build_inventory` skips them too (consistency guard). They cannot reach the LLM input block.

### Upstream taxonomy MECE (belongs to step 3, not P8)
A deeper quality issue surfaced during verification: domains are not fully MECE. `merkindruk` is defined so broadly ("the general evaluative impression") that it becomes a catch-all mixing aspects at different abstraction levels, and it overlaps with `merkherkenning` — so the same concept (e.g. "eekhoorn") lands inconsistently across domains. The root is the **domain definitions (step 3)** and **facet granularity (P1/P2)**, not P8. Track this in step 3 prompt work.

## Parking lot
- **Pre-P8 snapshot**: P8 overwrites P7 output destructively. Adding `raw_xdomain_*` fields (mirroring P7's `raw_attributes`) would enable a before/after diff view and easy revert.
- **Window param scaling**: defaults (window=10, overlap=2) tuned for ~30-attribute taxonomies; may need tuning for 50+.

## Reference
Algorithm: [CONSOLIDATION_LOGIC.md](CONSOLIDATION_LOGIC.md). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) §Cross-Domain Consolidation.
