"""
Step 4: Taxonomy Classifier — the facet and attribute layers, seven phases

Discovers facets (L3) and attributes (L4) within domains (L2) from step 3,
building a complete taxonomy with per-idea assignments. The two layers are
found together in one discovery call and settled apart, one consolidation call
per level; see `classifier.TaxonomyClassifier` for the phase order.

Usage:
    cd src && python -m pipeline.step_4_classifier.run_classifier
"""
