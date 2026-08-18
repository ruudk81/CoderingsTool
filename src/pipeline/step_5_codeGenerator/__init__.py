"""
Step 5: Code Generator — van taxonomie naar codeboek.

    taxonomy_input -> concept_inventory -> v2.attribute_cards ->
    v2.consolidation (1 LLM-call: welke attributen vormen samen één code) ->
    v2.grouping (Python: valentiepolen, partitiebewaking) ->
    codebook_writer (1 LLM-call: naam, definitie, diagnostiek) ->
    drie deterministische bewakers -> Overig-sweep -> scorecard ->
    cache onder "mece_codes" voor step 6.

De v1-keten (relaties, consolidator, MECE-rondes) staat met pensioen in
`_quarantine_v1/` — zie de `__init__.py` daar.

Usage:
    cd src && python -m pipeline.step_5_codeGenerator.v2.run_codebook_v2
"""
