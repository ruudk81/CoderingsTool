"""De v1-keten van step 5, met pensioen sinds de v2-promotie (2026-08-18).

Niets hierin draait in productie. `run_pipeline.py` en `app_backend.py` roepen
`v2.run_codebook_v2` aan; step 6 en step 7 lezen de cache die v2 schrijft.

Bewaard, niet verwijderd: v2 is op één dataset gemeten. Breekt v2 op een andere
boomvorm, dan is dit de keten om tegen af te zetten. De modules importeren nog
en hun tests draaien mee, zodat "teruggrijpen" ook echt kan.

Volgorde van de keten en de reden voor het pensioen:
`.superpowers/specs/2026-08-18-step5-v2-promotienotitie.md`
"""
