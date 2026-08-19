"""
Step 5: Code Generator — van taxonomie naar codeboek.

    taxonomy_input -> concept_inventory -> attribute_cards ->
    consolidation (1 LLM-call: welke attributen vormen samen één code) ->
    grouping (Python: valentiepolen, partitiebewaking) ->
    codebook_writer (1 LLM-call: naam, definitie, diagnostiek) ->
    drie deterministische bewakers -> Overig-sweep -> scorecard ->
    cache onder "mece_codes" voor step 6.

De v1-keten (relaties, consolidator, MECE-rondes) staat met pensioen in
`_quarantine_v1/` — zie de `__init__.py` daar.

Usage:
    cd src && python -m pipeline.step_5_codeGenerator.run_codebook
"""
