"""
Step 5: Clusterer V4 — Object-Aware Map-Reduce MECE

Two discovery modes:
  "clustering" (default):
    Stage 1: Object Discovery (cluster categories → MECE objects)
    Stage 2: Map Objects to Ideas (MECE object → categories → ideas)
  "semantic_category":
    Partition ideas by semantic_category field (6 fixed groups, no clustering)

Both modes feed into:
  Stage 3: Object-Aware Map-Reduce MECE (per-object topic extraction)
"""
