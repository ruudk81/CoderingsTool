"""
Step 5: Category Discovery v2 — Qualitative Researcher Pipeline

Partitions ideas by domain (data-driven groups from step 3),
then runs a cross-partition pipeline:
  1. Theme Discovery (chunked, per partition) — identify descriptive themes
  2. Theme Consolidation (per partition) — LLM-based deduplication
  3. Reflexive Thematic Analysis (cross-partition) — analytical themes + subthemes
  4. Category Assignment — assign ideas to leaf subthemes

Usage:
    cd src && python -m development.step_4_classNcoder_v2.run_experiment
"""
