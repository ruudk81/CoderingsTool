"""
Step 5: Category Discovery — Partition-Aware Map-Reduce MECE

Partitions by concept_type (data-driven groups from step 3).

Two processing modes within each partition:
  Mode A ("direct"):    MAP/REDUCE/MECE on concept labels directly
  Mode B ("clustered"): Pre-cluster labels via UMAP+HDBSCAN,
                        then MAP/REDUCE/MECE with cluster hints as context

Usage:
    cd src && python -m experiments.step_5_categories.run_experiment
"""
