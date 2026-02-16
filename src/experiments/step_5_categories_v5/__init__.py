"""
Step 5: Category Discovery V5 — Partition-Aware Map-Reduce MECE

Always partitions by semantic_category (6 fixed groups:
identity, attribute, function, state, evaluation, relation).

Two processing modes within each partition:
  Mode A ("direct"):    MAP/REDUCE/MECE on category_labels directly
  Mode B ("clustered"): Pre-cluster labels via UMAP+HDBSCAN,
                        then MAP/REDUCE/MECE with cluster hints as context

Usage:
    cd src && python -m experiments.step_5_categories_v5.run_experiment
"""
