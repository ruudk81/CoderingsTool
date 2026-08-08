Twee dingen voor als je klaar bent met de fasen-refactor.

  1. De fasenlijst in de app loopt achter. src/app/app_views.py:550 bouwt de fasen van stap 4 als tuple(f"classifier_p{i}" for i in range(1, 9)) — dus p1 tot en met p8. Sinds jouw commit 0387b56a kent config.STEP_MODEL_TIERS alleen nog p1 tot en met
  p7. Daardoor staan er nu twee tests rood (test_registry_phases_exist_in_config en test_models_line_resolves_for_all_steps) en loopt de stap-4-pagina van de app stuk op een KeyError. Trek die range() gelijk met de eindstand van je fasen.

  2. Belangrijker: laat die mismatch niet crashen. config.get_step_model() doet STEP_MODEL_TIERS[phase] zonder vangnet, en app_views.models_line() roept dat aan voor een read-only regeltje dat alleen toont welk model een stap draait. Puur cosmetisch —
  maar een onbekende fase neemt de hele pagina mee. Zolang jij fasen hernoemt lopen die twee lijsten per definitie tijdelijk uit de pas, en dan is de app onbruikbaar precies wanneer je eraan werkt.

  Laat een onbekende fase in dat regeltje verschijnen als "onbekend" in plaats van een KeyError te gooien. De test blijft de drift luid melden — dáár wil je het signaal — en de app blijft ondertussen werken. Twee regels.

  De fasenlijst staat nu op twee plekken en wordt met de hand synchroon gehouden; die test is het enige mechanisme dat dat bewaakt. Punt 2 zorgt dat de kosten van uit-de-pas-lopen een regel tekst zijn in plaats van een kapotte pagina.