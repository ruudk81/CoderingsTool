# Cross-Domain Attribute Consolidation — Core Logic

Reference document for the cross-domain consolidation algorithm (P8), as implemented in `cross_domain_consolidator.py`.

---

## Purpose

P7 consolidates attributes across facets *within* each domain. This phase (P8) consolidates attributes *across* domains. The problem: with 35-50+ attributes across 5-10 domains, a single LLM call can't handle the full taxonomy. The solution: use embeddings to find which attributes are semantically close, chunk them into LLM-digestible groups, consolidate within each group, then remap assignments.

## Pipeline Position

After P7 (cross-facet attribute consolidation). Operates on the complete post-P7 taxonomy.

```
P7: Cross-facet attribute consolidation (per domain)
  ↓
P8: Cross-domain attribute consolidation (global)
  ↓
Output: TaxonomyResultsCache + growing model (updated)
```

## Algorithm Overview

Five stages, executed sequentially:

```
Stage 1: EMBED        — embed all ideas, compute centroid per attribute
Stage 2: SIMILARITY   — pairwise cosine similarity between attribute centroids
Stage 3: ORDER        — seriate attributes into a 1D similarity ordering
Stage 4: CONSOLIDATE  — sliding window → concurrent LLM calls per group
Stage 5: REMAP        — apply merge decisions to cache + growing model
```

---

## Stage 1: Embedding

### Input
- `List[TaxonomyClassifiedModel]` (growing model from P7)
- `code_source` config (default: `instance_interpretation`)

### Logic
1. Group all ideas by `(partition_name, attribute)` from the growing model. Sentinel attributes (`__UNASSIGNED__`, `(no attribute)` — `_SENTINEL_ATTRIBUTES`) are skipped, so they never get a centroid, enter a window, or reach the LLM input block. `_build_inventory` applies the same filter, keeping the two consistent.
2. For each idea, format text via `format_idea_text(idea, code_source)` — configurable:
   - `instance` — raw verbatim span
   - `instance_interpretation` — `"{instance} | {interpretation}"` (default)
   - `full_abstraction_ladder` — `"{instance} | {interpretation} | {abstraction}"`
3. Embed all texts in one batched call via `SharedEmbedder.embed_texts()` (~1820 texts → 19 batches of 100)
4. Slice embeddings back per attribute
5. Compute **centroid** (mean of all idea embeddings) per attribute — represents the full semantic distribution
6. Compute **medoid** (closest real idea to center) for display only

### Output
- Per attribute: centroid vector (dim 3072), medoid text, idea count

### Design decision: centroid over medoid
Centroid captures the full spread of an attribute's ideas. Medoid is one point — if an attribute has sub-clusters, the medoid lands in the dominant one. For *comparing* attributes, centroid is more representative.

### Reuses
- `SharedEmbedder` from `utils/embedder.py` (batched async embedding)
- `format_idea_text()` from `utils/embedder.py` (text formatting)
- `compute_medoid()` from `utils/embedder.py` (for display label)

---

## Stage 2: Pairwise Similarity

### Logic
1. Stack all N centroids into `[N × 3072]` matrix
2. Compute `cosine_similarity()` → `[N × N]` matrix
3. Filter: only cross-domain pairs (same-domain already handled by P7)
4. Apply threshold floor (default 0.6) — pairs below this are too dissimilar

### Key finding
The similarity distribution is compressed for single-topic surveys (e.g., all about one bank). Merk X dataset: mean 0.80, range 0.635-0.959. The threshold is a noise floor, not a precision filter — the real work is done by the ordering/windowing in Stage 3.

---

## Stage 3: Seriation + Sliding Window

### Problem
N attributes (35-50+) is too many for one LLM call. Need to chunk into groups of ~10 where similar attributes are together.

### Logic
1. **Seriation**: agglomerative clustering (average linkage) on centroid distance matrix → dendrogram → `leaves_list()` gives a 1D ordering where similar attributes are adjacent
2. **Sliding window**: slide a window of `WINDOW_SIZE` (10) across the order with `WINDOW_OVERLAP` (2, ~20%)
3. **Edge case**: if last window adds ≤ `WINDOW_OVERLAP` new attributes, merge it into the previous window

### Why seriation, not clustering assignments
Clustering gives hard partitions. Seriation gives a continuous ordering that the sliding window can chunk with natural overlap. Attributes near a cluster boundary appear in two windows and get considered in both LLM calls.

### Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `WINDOW_SIZE` | 10 | Max attributes per LLM call |
| `WINDOW_OVERLAP` | 2 | Overlap between adjacent windows (~20%) |

### Output
- List of windows (each a list of attribute indices)
- Merk X dataset: 4 groups of [10, 10, 10, 11]

---

## Stage 4: LLM Consolidation

### Prompt
Adapted from P7's `build_attribute_consolidation_prompt()`. Same consolidation rules (prevalence weighting, merge bias, orthogonality, disambiguation test, minimality). Key differences from P7:

| Aspect | P7 | P8 (cross-domain) |
|--------|----|--------------------|
| Scope | Across facets within 1 domain | Across domains within 1 group |
| Input format | facet → attributes | domain → facet → attributes |
| Assignment | Best facet | Best domain AND facet |
| Tie-breaker | N/A (single domain) | Domain with more ideas |
| Excluded domains | Yes (other domains) | No (all in scope) |

### Input block format
```
Domain: "klantbenadering" — [domain definition]
  Excludes (belong to other domains): [this domain's exclusions, from step 3]
  Facet: "Vriendelijke omgang" — [facet description]
    - "Vriendelijke omgang" (33 ideas) — [attribute description]
  Facet: "Klantondersteuning" — [facet description]
    - "Ondersteunende service" (38 ideas) — [attribute description]

Domain: "reputatie en waardering" — [domain definition]
  Facet: "Menselijkheid en vriendelijkheid" — [facet description]
    - "Warme vriendelijke uitstraling" (44 ideas) — [attribute description]
```

### Response model
```
CrossDomainConsolidatedAttribute:
  attribute_name: str           # 2-5 words
  attribute_description: str    # 1-2 sentences
  parent_domain: str            # best-fit domain
  parent_facet: str             # best-fit facet within domain
  source_attributes: List[str]  # original names merged into this one

CrossDomainConsolidatedResponse:
  scratchpad: str               # step-by-step reasoning
  attributes: List[CrossDomainConsolidatedAttribute]
```

### Execution
- All groups dispatched concurrently via `SmoothRequester.process_all()`
- Model from `classifier_p8` config key (default tier)
- Temperature: 0.3 (same as P7)

### Consolidation rules (8 rules, inherited from P7)
1. **Prevalence weighting** — high idea counts form core structure
2. **Merge bias** — when in doubt, merge
3. **Merge overlap** — conceptually overlapping attributes must merge, even across domains
4. **Orthogonality** — "can one observation fall under both?" → merge
5. **No hierarchy** — general vs specific → merge
6. **No object splitting** — same principle, different object → merge
7. **Minimality** — smallest number of attributes for full coverage
8. **Domain & facet assignment** — assign to best domain+facet; tie-break by idea count
9. **Respect domain boundaries** — each domain's `Excludes` list (from step 3) is shown; do not merge an attribute into, or assign its parent_domain to, a domain that excludes its concept

---

## Stage 5: Remap

### Overlap conflict resolution
Attributes in overlapping windows may get different treatment across groups. Resolution: **"merge wins, first group takes precedence."**

- Process groups in seriation order (1 → N)
- If a source attribute was already mapped by an earlier group → skip
- If new → apply the merge/rename

Rationale: earlier groups have higher internal similarity (by seriation ordering), so their merge decisions are more confident.

### Merge map
Built per window. The LLM returns bare source names; each is resolved to a concrete `(domain, attribute_name)` against the attributes actually present in that window — names are **not** unique across domains, so a bare name alone is ambiguous. Unknown names (LLM hallucinations not in the window) resolve to nothing and are skipped. For each consolidated attribute, each resolved source:
- ≠ the target `(new_domain, attribute_name)` → merge/rename (map source → target)
- == the target → no change (skip)

Result: `Dict[(domain, old_name) → MergeTarget(new_name, new_domain, new_facet, new_description)]`

### TaxonomyResultsCache updates

For every remap (same-domain and cross-domain):
- Move the idea_ids in `attribute_assignments` from old name → new name, **carrying** their `attribute_valence` and `attribute_confidence`.
- Set `facet_assignments` to the target facet — also for same-domain merges that change facet.
- For cross-domain moves, additionally move `facet_valence` / `facet_confidence` from the source to the target domain.
- Remove the old attribute from the source `attributes[facet]` (dropping empty facets); ensure the target attribute exists under `target.new_facet`.

### Growing model updates
For each idea whose `(idea.partition_name, idea.attribute)` is in the merge map:
- `idea.attribute` → new attribute name
- `idea.facet` → new facet
- `idea.partition_name` → new domain
- `idea.domain` → new domain (`domain` is canonicalized to `partition_name` at growing-model build, so the two never diverge; P8 keeps both updated together on a cross-domain move)

### Invariant & self-check
Total ideas before == total ideas after. Zero idea loss. `_verify_consistency()` checks this plus: no idea lost its attribute/facet valence or confidence, and every assignment references an attribute that exists in the taxonomy (no orphans; `__UNASSIGNED__` excluded). It prints `P8 consistency: OK` or `⚠` warnings, and sets `stats["consistency_violations"]`. P8 is skipped entirely when fewer than 2 attributes exist.

---

## Configuration Summary

| Parameter | Default | Location |
|-----------|---------|----------|
| `code_source` | `"instance_interpretation"` | Embedding text format |
| `SIMILARITY_THRESHOLD` | 0.6 | Noise floor for pairwise similarity |
| `WINDOW_SIZE` | 10 | Max attributes per LLM call |
| `WINDOW_OVERLAP` | 2 | Overlap between adjacent windows |
| Model | `get_step_model("classifier_p8")` | Default tier (same as P7) |
| Temperature | 0.3 | Same as P7 |
| Max tokens | 16000 | Output token budget |

---

## Cache Strategy

P8 adds **no** cache keys. It reloads the just-cached `taxonomy` (metadata) and `taxonomy_classified` (growing model), applies the remap, and **overwrites the same keys in place**. Destructive — there is no pre-P8 snapshot (unlike P7's `raw_attributes` / `raw_attribute_assignments`). The merge report is emitted to the verbose log during the run, not persisted.
