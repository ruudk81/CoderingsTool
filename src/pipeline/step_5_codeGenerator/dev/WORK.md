# Step 5 — Work

Known gaps and planned fixes. System design belongs in
[ARCHITECTURE.md](ARCHITECTURE.md); caching contracts in [CACHE_LOGIC.md](CACHE_LOGIC.md);
runtime behavior in [PROCESSING.md](PROCESSING.md).

## Meetcontext van de cijfers hieronder (2026-08-20)

De instabiliteitscijfers verderop in dit bestand — 26/31/25/28 codes, 183-191
wisselende paren, 89-90% paar-overeenstemming, 0 van 84 groeperingen die in
alle vijf runs terugkwam — zijn gemeten op 2026-08-18, vóór twee dingen die
sindsdien zijn veranderd:

- **Het model.** Op 2026-08-18 draaide `codegen_relations` op `("5.6", 3)`
  (luna) bij `"medium"` reasoning effort. Sinds commit `342cde02`
  (2026-08-19) is dat `("5.4", 5)` (gpt-5.4) bij `"high"` — zie CLAUDE.md en
  ARCHITECTURE.md voor de huidige waarden.
- **De fixture.** De gecachte ASN-taxonomie waar step 5 nu tegen leest heeft
  **49 attributen, 2317 respondenten, 2728 geclassificeerde antwoorden, 5
  domeinen** — niet de 60 attributen / 1236 respondenten waarop onderstaande
  cijfers zijn gemeten (step 4 is sindsdien opnieuw gedraaid en heeft een
  andere boom opgeleverd).
- **"1236 respondenten" is hierboven vermoedelijk fout — onzeker, niet
  gecorrigeerd.** Het projectlogboek elders onderscheidt 1236 *responses* van
  1092 *respondenten* mét een idee. Welke van de twee hier bedoeld is, is niet
  nagekeken; behandel het getal als onzeker tot dat wel gebeurd is, ook waar
  het verderop in dit bestand terugkomt ("Eén dataset").

Deze cijfers blijven de enige meting die er is en worden hieronder niet
verwijderd. Maar vergelijk ze niet één-op-één met een nieuwe run: een verschil
kan van het model komen, van de boom, of van allebei — niet noodzakelijk van
wat er in de tussentijd aan step 5 zelf is veranderd.

## tau is een invoer, geen constante (open — 2026-08-20)

Zolang er op een tweede dataset niets gemeten is, blijft tau een expliciet
argument van het experiment en komt hij NIET als constante in `config.py`. De
reden is dat hij er neutraal uitziet en dat niet is: de keuze voor 0,5 werd
onderweg mede beargumenteerd met het feit dat `Kosten en prijsstelling` er dan
blijft staan — dat is afstellen op de inhoud van deze dataset, in een vorm die
de regel over use-case-agnostische prompts niet vangt. Zelfde categorie als
`DEGENERATION_FLOOR`/`CEILING`: beredeneerd, niet gemeten.

Sinds de facetpool weegt de keuze bovendien veel minder: met pool geven 0,5 en
0,7 vrijwel hetzelfde codeboek (`direction_loss` 277 tegen 282), terwijl het
verschil zonder pool 317 tegen 409 was.

## Consensus over N runs — gemeten, haalde de lat niet (2026-08-20)

Gebouwd en gemeten: deel 1 N keer draaien, per attribuutPAAR tellen hoe vaak
twee samen zaten, en die co-associatiematrix met volledige koppeling tot een
indeling snijden. Gereedschap in `consensus/`, verslag in
`exports/experiment_logs/consensus_verslag_20260820.md`.

Sinds 2026-08-21 is dat gereedschap geen los script meer maar een volwaardige
kandidaatketen (`consensus/`, getrackt naast `_quarantine_v1/` in plaats van
onder `dev/`): eigen runner, eigen consolidatiedispatch (N runs in één
`SmoothRequester`), eigen cache-write onder de gedeelde `mece_codes`-sleutel,
eigen kosten-/log-/prompt-identiteit. De meting hieronder verandert daardoor
niet — het is dezelfde 0,788 op dezelfde vraag. Wat wel verandert is wat een
promotiebesluit vraagt: geen herbouw meer, maar een verhuizing (modules
omhoog de stapmap in, de huidige productieketen naar `_quarantine_v1/`),
precies zoals `v2/` dat op 2026-08-19 deed.

Op 2026-08-22 is die keten in één draaibestand samengetrokken: `consensus/
run_codebook.py` draagt alle vijf de acties (`alles`, `verzamelen`, `codeboek`,
`analyse`, `vergelijk`) achter een instellingenblok, en `run_experiment.py`
bestaat niet meer. `SET = AUTO` telt zelf door naar het volgende vrije nummer,
zodat een ronde geen handmatige edit meer vraagt. Drie dingen die daarbij zijn
rechtgezet en die als open punt genoteerd stonden:

- **Promptexport en kostenpost werden binnen één ronde overschreven.** Elke
  actie maakte zijn eigen `PromptPrinter` en `CostTracker`; `save_prompts_to_json`
  opent in `'w'` en `record_phase` WIJST TOE op (stap, fase). Gemeten op de ronde
  van 2026-08-22: 0 van de 60 consolidatieprompts bewaard, en 30 van de 60 calls
  geboekt — gerapporteerd $0,19 waar het ≈ $0,31 was. `alles` bezit nu één
  printer en één teller voor de hele ronde.
- **De drempel telde in responses, niet in respondenten.** `build_shapes` toetst
  tegen `len(pool.resp_ids)` (unieke respondenten) terwijl `t_keep` rekende met
  `len(classified)` (responses): 1% van 2728 is 27, 1% van 2317 is 23. Omgezet
  naar respondenten. **Nog niet gemeten** — verwacht meer codes en een lagere
  `direction_loss`. Productie's `run_codebook.py` rekent nog met responses, dus
  de twee ketens liggen sindsdien NIET meer op dezelfde drempel.
- **`codeboek` had geen bewaking op een verschoven attribuutuniversum**, terwijl
  `vergelijk` die wel had — de actie die het artefact naar de gedeelde cache
  schrijft was dus permissiever dan de actie die alleen een getal print.
  Rechtgezet.

Hoofdmaat = ARI tussen twee onafhankelijke consensusindelingen. Op de vooraf
vastgelegde τ=0,5: **0,788** tegen de geeiste 0,90. Gefaald. Consensus
verdubbelt de reproduceerbaarheid wel (losse runs: mediaan ARI 0,42), maar niet
genoeg, en de uitkomst is niet monotoon in τ: 0,5 → 0,788, 0,7 → 0,900,
0,9 → 0,693. Daaronder ligt een ruil: de drempel die betekenisvol consolideert
reproduceert het slechtst.

**Op MERGE-niveau is het beeld gunstiger, en dat is de bruikbare vraag.** Niet
"reproduceert de hele indeling" maar "welke samenvoegingen komen terug". Op een
schone attribuutlijst (43, na uitsluiting van de vangnetten hieronder), luna,
2 x 30 runs, τ=0,7: **5 van de 5 groepen identiek** over twee onafhankelijke
sets. Ter vergelijking: groepen uit een enkele run kwamen historisch 0 van 84
keer terug. De ARI mat de verkeerde eenheid — die telt ook de honderden
beslissingen over attributen die toch alleen blijven.

Schaalcurve (luna, meerdere disjuncte splitsingen, τ=0,7): N=10 geeft 42-63%
paar-overeenstemming, N=15 53-65%, N=30 73%. Stijgt door tot 30. Uitwisselbaar
met modelkwaliteit: gpt-5.4 bij N=10 gaf 8 van 9 groepen terug, luna bij N=30
zeven van negen — en 30 luna-calls kosten ongeveer twee gpt-5.4-calls.

Die merge-meting staat sinds 2026-08-21 in het gereedschap in plaats van in een
losse berekening: `analysis.merge_recurrence` telt identieke samenvoegingen plus
de paarovereenstemming over ALLEEN samengevoegd materiaal (paren die in beide
indelingen apart zitten vallen uit de noemer — dat is wat de 89-90% hierboven
misleidend maakte), en `vergelijk` drukt het naast de ARI af. `vergelijk` neemt
daarvoor SET / SET_B, want de 30-runssets zijn 3 en 4 en luna heeft geen set 2.
Reproduceren: in het instellingenblok van `run_codebook.py` ACTIE = "vergelijk",
CONFIG = "luna", SET = 3, SET_B = 4, TAU = 0.7 zetten en op Run klikken — geeft
ARI 0,844, 7 van 9 samenvoegingen identiek, 73,3% paarovereenstemming.

**Nog niet reproduceerbaar: de 5-van-5 hierboven** (open). Dat cijfer is gemeten
op een vangnetvrije attribuutlijst. Het filter dat dat toen deed —
`_codebook_body`'s `_schoon` — bestaat nergens meer: het verdween met
`run_experiment.py` op 2026-08-21, en `vergelijk` heeft nooit een eigen
vangnetfilter gehad. De opgeslagen sets 3 en 4 zijn bovendien nog mét
vangnetten verzameld, van vóór `exclude_drains` bestond.

De enige uitsluiting die vandaag bestaat, zit vóór het verzamelen, niet erna:
`ConsensusConfig.exclude_drains` (het instellingenblok z'n `DRAINS`, standaard
`"uit"` = uitgesloten) stuurt `build_cards(exclude_drains=True)`, dus een
NIEUWE `verzamelen`-ronde onder de huidige standaardinstelling laat vangnetten
al buiten de kaarten. Niets filtert een al opgeslagen `RunSet` achteraf. Om
5-van-5 na te rekenen zijn er dus twee wegen, geen van beide gebouwd: (a) twee
verse sets van 30 runs verzamelen onder de huidige standaard en die door
`vergelijk` halen, of (b) een achteraf-filter aan `vergelijk`/`analyse`
toevoegen dat vangnet-attributen (via `is_drain` op de step-4-cache,
`load_material`) uit een geladen `RunSet` haalt vóór `together_from_runs` —
geen LLM-call, wel nieuwe code.

### Zekerheid en betekenis zijn niet hetzelfde (open)

De methode weegt alleen hoe vaak een paar samen zat, niet hoeveel materiaal
eronder ligt. Daardoor kreeg een merge van twee bakjes met elk een respondent
29-van-30-zekerheid, terwijl een thematische merge over drie attributen met
honderden respondenten op 24-van-30 stond. De vangnetten zijn nu uitgesloten,
maar het onderliggende punt blijft: een volgende versie hoort prevalentie mee
te wegen, niet alleen recurrentie.

### Vangnetten op de kaarten (mechanisme klaar, vlag UIT — 2026-08-21)

Step 4 markeert zijn `other`-attributen met `drain_key`; step 5 las die sleutel
nooit en legde ze als volwaardige onderwerpen voor. Het model merkte ze met
28-29 van 30 samen met hun naamgenoot. Mechanisme staat: `is_drain` op `AttributeRef` en `Concept`,
`build_cards(exclude_drains=True)` slaat ze over. Effect gemeten op de
opgeslagen partities: paar-overeenstemming 73% → 78%, groepen identiek
7-van-9 → 5-van-5. Kosten: 8 respondenten (0,3%) verschuiven naar Overig, ruim
binnen het 10%-plafond.

**De vlag staat UIT in productie** (2026-08-21, op verzoek): het experiment moet
eerst als geheel gepromoveerd of verworpen worden, en tot dat besluit valt hoort
er geen enkele gedragswijziging ongemerkt mee te liften. `run_codebook.py` geeft
de vlag niet mee; alleen `consensus/` zet hem aan.

Los daarvan blijft dit een gemiste contractregel en geen experimentele feature:
step 4 documenteert dat je vangnetten op `drain_key` herkent en step 5 keek er
nooit naar. Bij een verwerping van het experiment hoort deze vlag alsnog
afzonderlijk gewogen te worden.

#### De motivering klopt niet, en er is een betere derde weg (open — 2026-08-21)

De reden die overal bij deze uitsluiting staat — een vangnet zou "geen
beantwoordbare vorm" hebben omdat het per definitie restant is — **is onjuist**.
Een vangnet is niet onderwerploos maar FACET-GEBONDEN: `Overig — Politieke
richting` betekent "de rest binnen Politieke richting", en dat is een onderwerp,
alleen niet gespecificeerd. Het model dat zo'n vangnet bij de hoofdcode van zijn
eigen facet zet doet dus iets verdedigbaars, geen onzin.

Wat er wél mis is, is wat die merge met de METING doet. De 28-29 van 30 komt uit
een facet- en naamovereenkomst en is daarmee bijna automatisch, dus hij staat
bovenaan de co-associatiematrix terwijl er een of twee respondenten onder zitten.
Dat is exact het gat dat hierboven onder "Zekerheid en betekenis zijn niet
hetzelfde" staat — recurrentie zonder prevalentie — en niet een eigenschap van
vangnetten.

**Nagemeten kostenpost, groter dan eerder genoteerd.** `apply_overig_sweep`
maakt EEN globale `Overig`-code voor de hele dataset, niet een per facet. Met de
vlag aan verliezen die 8 respondenten dus hun facetcontext volledig: "de rest van
Politieke richting" wordt "de rest van alles". Het onderscheid dat een vangnet
draagt gaat weg, niet alleen het meetartefact.

**Derde weg, niet gebouwd:** haal vangnetten van de consolidatiekaarten (het
model hoeft er geen oordeel over te vellen) maar route ze deterministisch naar de
code van hun EIGEN FACET in plaats van naar de globale `Overig`. Die redenering
staat al in de code: `pool_thin_within_facet` behandelt het facet als legitieme
groepeereenheid en steekt nooit een facetgrens over, en een vangnet is per
definitie sub-drempelig materiaal binnen een facet. Dan verdwijnt het
meetartefact en blijft de facetbetekenis staan. Geen LLM-vraag, Python.

Gevolg voor het promotiebesluit: de keuze is niet "vlag aan of uit" maar
drie-weg. De vlag blijft UIT tot die derde weg gewogen is.

## De promotie is geoefend — hij loopt vast op `stability.py` (open — 2026-08-22)

De afspraak achter `consensus/` is dat je step 5's ketenmodules kunt verwijderen,
`consensus/` op hun plek kunt zetten, en `run_pipeline.py` step 5 gewoon blijft
draaien. Taken 1-3 maakten dat waar en legden het vast in drie bewakers
(`consensus/test_zelfstandigheid.py`). Op 2026-08-22 is de verhuizing daarnaast
één keer daadwerkelijk uitgevoerd en weer teruggedraaid — een repetitie, want een
bewaker toetst wat hij bevraagt en de repetitie toetst wat er overblijft.

**Uitkomst: de afspraak klopt nog niet.** De keten zelf verhuist schoon en
`run_codebook(force_recalc=False)` draait erna precies zoals `run_pipeline.py`
hem aanroept. Maar twee modules die NIET door `consensus/` worden vervangen
blijven achter met een import die na de verhuizing nergens meer op uitkomt.

### Blokkade: `stability.py` en `postmortem.py` overleven de verhuizing niet

`stability.py` doet `from .consolidation import resolve_consolidation`
(enkelvoud, één run). `consensus/consolidation.py` levert
`resolve_consolidations` (meervoud, N runs) plus `build_tasks`, en geen
`resolve_consolidation`. `postmortem.py` importeert `StabilityReport` uit
`stability.py` en valt in dezelfde fout mee.

Dat is geen falende test maar een ImportError tijdens collectie, dus pytest
breekt de hele suite af — 24 tests (10 + 14) komen niet eens aan de start. Beide
modules zijn het meetgereedschap achter "De consolidatiecall reproduceert niet"
en "Post-mortem" hieronder; ze horen dus niet zomaar met de oude keten mee weg.

**Besluit (2026-08-22): de promotie verwijdert `stability.py`, `postmortem.py`,
`prompts_postmortem.py`, `test_stability.py` en `test_postmortem.py`.** Geen
shim in `consensus/` ervoor, en ze verhuizen ook niet mee. Drie dingen dragen
dat besluit:

1. **Niets importeert ze behalve productie's eigen `run_codebook.py`** —
   precies het bestand dat `consensus/` vervangt. `postmortem.py` importeert
   `StabilityReport` uit `stability.py`, en verder importeert niemand van
   beide binnen step 5: `consensus/consensus.py` NOEMT `stability.py` alleen
   in commentaar (het legt uit waarom het er bewust NIET uit leent), en
   `consensus/analysis.py`/`test_consensus.py` lenen van `step_3` resp. een
   docstring-verwijzing, niet van deze twee modules. Na promotie zijn ze dus
   wezen.
2. **Ze zijn slapend, niet levend.** `stability_runs` staat overal op 0,
   inclusief `run_codebook.py`'s eigen `__main__`, met het commentaar erbij:
   "de post-mortem-splitser staat uit tot zijn vraagvorm herzien is."
3. **`stability.py`'s eigen docstring zegt al dat zijn kernbezwaar elders is
   beantwoord** — door `consensus/consensus.py`, dat volledige in plaats van
   enkelvoudige koppeling gebruikt. De paar-stabiliteitsmeting wordt dus NIET
   opgeheven; ze is al verhuisd naar `consensus/analysis.py` en is daar beter
   geworden (zie "Consensus over N runs" hierboven: complete linkage sluit
   precies de A-C-op-apart-situatie uit die `stability.py` als reden gaf om
   geen consensusindeling af te leiden).

**Kosten, en dit is het enige oordeel in dit besluit dat een afweging is in
plaats van een constatering**: de geparkeerde LLM-post-mortem-splitser
verlaat de boom. Hij is terug te halen uit de git-historie (commit `79a6843a`
en ervoor), maar er blijft geen code voor achter — wie de vraagvorm ooit
herziet ("welke twee onderwerpen horen het minst bij elkaar?", zie "Post-mortem"
hieronder) begint met een terugzet uit git, niet met een bestand in de boom.

### De verhuisrecept vraagt vier ingrepen, niet één

1. **24 productiebestanden weg**: de 5 die `consensus/` met een eigen versie
   vervangt (`run_codebook`, `consolidation`, `prompts_consolidation`,
   `view_codebook`, `view_prompts`), de 11 gekopieerde ketenmodules, en de 8
   testbestanden die `consensus/` ook heeft.
2. **`__init__.py` is een 25e botsing.** `git mv consensus/*.py .` valt er
   meteen over. Productie's `__init__.py` beschrijft de STAP en hoort te
   blijven; die van `consensus/` beschrijft een submap die na de verhuizing niet
   meer bestaat en gaat weg.
3. **De importsubstitutie is twee regels, en de volgorde doet ertoe.** Naast
   `pipeline.step_5_codeGenerator.consensus.<module>` bestaat de vorm
   `from pipeline.step_5_codeGenerator.consensus import <module>`. Pas die
   tweede EERST toe: `consensus/` bevat een module die óók `consensus.py` heet,
   en `...consensus.consensus import` collapst anders eerst naar
   `...consensus import` en daarna ten onrechte naar `... import` — waarna
   `run_codebook.py` `consensus_partition` uit het pakket probeert te halen.
4. **`codebook_io.py` heeft een tweede, niet-import-fix nodig.** Zijn
   `project_root` is een positieafhankelijke constante; de kopie draagt bewust
   één `.parent` extra omdat ze een map dieper ligt. Na de verhuizing moet die
   er weer af, anders wijst de repo-root een niveau te hoog en schrijven
   promptexport en logs naast de repo. Dit is niet langer een handmatige stap
   zonder toets erachter: `test_project_root_wijst_naar_de_repo_root`
   (`consensus/test_zelfstandigheid.py`, toegevoegd na deze repetitie, commit
   `d4f118bf`) bevraagt waar `project_root` uitkomt in plaats van hoeveel
   `.parent`-stappen ervoor staan, en klopt daarom op beide dieptes — vóór en
   ná de verhuizing. De stap blijft nodig; hij faalt nu zichtbaar als hij
   vergeten wordt.

### Wat de repetitie juist NIET nodig bleek te hebben

- **De quarantainetests hoeven niet weg.** Stap 3b van de taakbrief ging ervan
  uit dat `_quarantine_v1/` breekt zodra de geleende modules verdwijnen. Dat
  gebeurt niet: de 11 modules verdwijnen niet, ze worden vervangen door hun
  inhoudelijk identieke kopie op hetzelfde pad. Gemeten op de gepromoveerde
  boom: **125 tests groen**. De v1-keten blijft dus leesbaar én draaibaar.
- **`test_prompts_consolidation.py` overleeft en slaagt.** Dat bestand blijft
  achter zonder tegenhanger in `consensus/` en toetst na de verhuizing dus de
  ANDERE `prompts_consolidation.py`. Het gaat niet stuk: **11 groen**. Wat het
  daar meet is daarmee wel een open vraag — het is geschreven op productie's
  prompt en slaagt op die van de kandidaat.

### Steigerwerk dat bij de verhuizing hoort te sneuvelen

`consensus/test_zelfstandigheid.py` (14 tests) en
`consensus/test_consolidation.py::test_prompt_is_byte_identiek_aan_productie_op_import_en_docstring_na`
vergelijken de kopie met het origineel via `hier.parent`. Na de verhuizing is
`hier.parent` `pipeline/` en bestaat dat origineel niet meer. Dat is geen
regressie maar het einde van hun functie: ze bewaken het naast-elkaar-bestaan,
en dat is precies wat de promotie opheft. Ze horen in dezelfde commit weg als
de verhuizing.

### Telling

Basis 1047 groen. Op de gepromoveerde boom, met de blokkade uitgesloten:
**906 groen**. Het verschil sluit exact — 102 (de 8 vervangen
productietestbestanden) + 24 (geblokkeerd) + 14 (`test_zelfstandigheid`) + 1
(de byte-identiteitstest) = 141. Er is dus niets stil omgevallen.

Na terugdraaien staat de boom byte-identiek terug op 1047 groen.

## De consolidatiecall reproduceert niet (open — 2026-08-18)

Vier runs op identieke invoer (ASN, 60 attributen) gaven **26, 31, 25 en 28
codes**. Op paarniveau is het beeld rustiger: vijf consolidatieruns geven
**89-90% paar-overeenstemming** met 183-191 wisselende paren — de meeste
samen-of-apart-beslissingen liggen dus vast, en de spreiding in het aantal codes
komt van een minderheid die wiebelt.

Dit is minder erg dan het klinkt: een codeboek maak je één keer en hergebruik je
daarna. Maar het is niet opgelost, en het is de reden dat de post-mortem
hieronder is gebouwd.

Meetgereedschap: `stability.py`, `run_consolidation_repeatedly()`. Kost geen
extra call bovenop de run die je toch nodig had — de eerste run wordt het
codeboek, de rest dient de meting.

## Post-mortem: mechanisme staat, vraagvorm levert het null-antwoord (open — 2026-08-18)

`stability.py` en `postmortem.py` draaien. De meting werkt en reproduceert.

De post-mortem zelf niet. Twee live runs splitsten 9 van de 10 kandidaatgroepen
VOLLEDIG uit tot losse attributen (`Duurzaamheid en maatschappelijke
verantwoordelijkheid` -> 7 delen van elk 1 attribuut), waarmee 56 groepen op 60
attributen overbleven. Beide keren ving `check_degeneration` het op en bleef de
cache ongeschreven — de bewaking doet precies wat ze moet doen.

De oorzaak is de vraagvorm, niet de bedrading. "Is dit één ding of zijn het er
meerdere?" heeft een antwoord dat altijd verdedigbaar lijkt: meerdere. Elk
attribuut heeft een eigen naam en definitie, dus alles apart is nooit aantoonbaar
fout. Hetzelfde patroon dat ARCHITECTURE.md vier keer documenteert — een model
gevraagd naar een structurele eigenschap levert het null-antwoord — alleen ligt
het nulpunt hier op maximaal splitsen in plaats van op niet groeperen.

Wat een volgende poging moet beproeven: een geforceerde opzoekvraag in plaats van
een open oordeel. Bijvoorbeeld "welke twee onderwerpen in deze groep horen het
minst bij elkaar?" — dan is er altijd precies één antwoord, en beslist Python of
die afstand groot genoeg is om te knippen. Dat is de constructie die in dit
pakket wel werkt.

Eerder verworpen en niet opnieuw proberen: de trigger op "groep bevat een
attribuut dat ergens wiebelt". Met 183 wisselende paren over 60 attributen is dat
elke groep. De trigger staat nu op een wisselend paar BINNEN de groep
(`StabilityReport.has_unstable_pair_within`), wat 17 kandidaten terugbracht naar 10.

Tot dat is opgelost staat `stability_runs` overal op 0, inclusief het `__main__`
van `run_codebook.py`.

## Twee mislukte pogingen, één oorzaak: de vraag is nergens begrensd (2026-08-18)

| poging | vraag aan het model | resultaat |
|---|---|---|
| post-mortem splitser | "is dit één ding of zijn het er meerdere?" | alles apart — 56 groepen op 60 attributen |
| consolidatie over runs | "minimaliseer het aantal codes" | alles samen — 6 groepen, grootste 54,3% |

De tweede prompt is letterlijk step 4's werkende consolidatieprompt, met step 4's
regels en zijn recurrentiekolom. Hij faalde toch. **Het verschil met step 4 is niet
de vraag maar het bereik:** step 4 vraagt "minimaliseer" binnen één facet, dus de
boom zet de bovengrens. Step 5 vraagt hetzelfde over de hele inventaris, waar het
minimum 1 is.

Elke open vraag over structuur die niet begrensd is, loopt naar een uiterste. Dat
is dezelfde les die ARCHITECTURE.md vier keer noteert, nu met de oorzaak erbij:
niet "het model geeft het null-antwoord", maar "het null-antwoord is bereikbaar
omdat niets de vraag begrenst".

### Groepsrecurrentie is structureel bijna nul

Over vijf runs op dezelfde invoer: **0** van de 84 voorgestelde groeperingen kwam
in alle vijf runs terug, 70 kwamen precies één keer voor. Dat botst niet met de
89-90% paarovereenstemming, het volgt eruit: een groep van vijf attributen is pas
identiek als álle tien zijn paren identiek beslist zijn, en 0,9^10 is ~35%. Bij
grotere groepen zakt dat naar nul.

Gevolg, geldig op elke dataset: **een ontwerp dat op hele groepen bouwt heeft
vrijwel geen signaal.** Het bewijs zit op paarniveau.

### Waar de tekentafel staat

De vraag is niet welke prompt, maar wat de vraag begrenst. Drie kandidaten, geen
ervan uitgewerkt:

1. **step 4's boom** — facet als eenheid, zoals step 4 zelf. Zie "Nog niet
   uitgeprobeerd" verderop; we komen er nu voor de derde keer op uit.
2. **de paren** — bouwen op wat wél vastligt.
3. **niet groeperen** — misschien hoort step 5 deze vraag helemaal niet te stellen.

Het experiment staat in `dev/experiment_consolidatie_runs/run_experiment.py`
(read-only, raakt productie niet), verslag in `exports/experiment_logs/`. Het
blijft staan om zijn fase A+B, niet om zijn uitkomst: die meten de
groepsrecurrentie hierboven, en dat is het gereedschap dat "Eén dataset"
verderop nodig heeft zodra er een tweede is.

Het oudere `dev/experiment_consolidatie/` is op 2026-08-19 verwijderd. Dat
beproefde of een LLM deze consolidatie kán doen als hij de onderzoeksvraag, de
tellingen en de letterlijke antwoorden ziet; het antwoord was ja en staat
sindsdien in productie als `prompts_consolidation.py`. Verslag van die run:
`exports/experiment_logs/experiment_consolidatie_20260817_192226_run1.txt`.

## Geen bewaker voor MECE-zonder-excuses (open — 2026-08-18)

De MECE-fase is met v1 verdwenen omdat ze het verkeerde mat (zie "Separability is
not orthogonality" hieronder), en er kwam niets voor terug. Twee overlappende
codes in één consolidatievoorstel worden niet automatisch gevangen — zichtbaar in
de output, repareerbaar in step 4 of door een mens, niet door step 5 zelf.

Python garandeert de VORM: een hele partitie, zuivere valentie, niets onder `t`
dat alleen staat, gemelde degeneratie. Of een attribuut inhoudelijk juist is
gegroepeerd is een oordeel waar geen deterministische check over gaat.

## De dikke duurzaamheidscode (open — 2026-08-18)

Eén code dekt 6 attributen en 657 respondenten, inclusief natuurbeelden die daar
inhoudelijk niet bij horen. Dit is precies wat de post-mortem moest oplossen. Op
de run van 2026-08-18 (28 codes) haalde de scorecard PASS met 100% dekking, dus
geen enkele automatische check meldt dit.

## Eén dataset (open — 2026-08-18)

Alles hierboven is gemeten op de ASN-fixture (1236 respondenten, 60 attributen
— zie "Meetcontext" bovenaan voor hoe dat zich verhoudt tot de cache van
vandaag). Gedrag bij een andere boomvorm — 5 facetten en 20 attributen, of 50
respondenten — is ongemeten. `test_data.py` heeft zes datasets uitgecommentarieerd staan;
Pinkpop en de NAVO-flitspeiling toetsen de generaliteit het hardst omdat ze een
andere vraagsoort stellen.

Hangt hieraan vast: `DEGENERATION_FLOOR` / `DEGENERATION_CEILING`
(`grouping.py`, 0.05/0.90) en `SHARE_THRESHOLD` (`postmortem.py`, 0.20)
zijn beredeneerd, niet gemeten. Herijken zodra er runs op meer dan één dataset
zijn.

## Naming-mismatch guard is mostly false positives (open, low)

`find_naming_mismatches` (`codebook_writer.py`) is lexical: it flags a code
whose name shares no word with any of its member attributes. On the ASN
fixture this mostly fires on legitimate paraphrase rather than a real
mismatch — e.g. it flags `Betaalbare kostenindruk` against its member
attribute `Kosten en betaalbaarheid`.

## Overig share is unstable across runs (open, low)

Observed 0.7%–9.3% on the same fixture across repeated runs, against the 10%
cap `codebook_verifier.py`'s scorecard checks for.

## Temperatuurvelden zijn dode configuratie (open, low — 2026-08-20)

`CodebookConfig.temperature_relations` (`0.0`) en `temperature_writer` (`0.3`)
bereiken de API nooit: `utils/llm.py` voegt `temperature` alleen toe voor
niet-redenerende modellen, en beide step-5-fasen draaien op een redeneermodel
(`("5.4", 5)`, zie CLAUDE.md). Zolang dat zo blijft zijn deze twee velden dode
configuratie — ze suggereren een knop die niet bestaat. Opruimen (verwijderen,
of expliciet markeren als niet-verzonden) is nog niet gedaan.

## `typing.List[Literal[...]]`-cachingval in make_writer_model / make_postmortem_model (open — 2026-08-20)

`Literal`-gelijkheid is volgorde-onafhankelijk (PEP 586) en `typing.List`
cachet op gelijkheid van zijn argumenten, dus twee permutaties van dezelfde
tagset leveren hetzelfde geannoteerde type op — de tweede aanroep krijgt zonder
foutmelding de volgorde van de eerste terug. `make_consolidation_model`
(`prompts_consolidation.py`) is hierom naar het ingebouwde `list[Literal[...]]`
omgezet, dat niet op die manier cachet (zie de docstring daar).

`make_writer_model` (`prompts_writer.py`) en `make_postmortem_model`
(`prompts_postmortem.py`) gebruiken nog `typing.List` en hebben dezelfde val.
Ze zijn vandaag veilig omdat elk van de twee maar één keer per proces draait,
dus er is nooit een tweede permutatie van dezelfde sleutelverzameling om mee te
botsen. Zodra één van beide fasen binnen één proces herhaald wordt — bijvoorbeeld
voor een consensus-achtige meting over de writer-stap — wordt dit een echte
bug, en is dezelfde omzetting naar `list[...]` nodig.

## Two-pole valence: contract verruimd naar vier waarden (gesloten — 2026-08-22)

`ConsolidatedCode.valence` kent nu `non_negative` naast de drie oude waarden, en
`code_shape.stored_valence()` vertaalt niets meer.

**Waarom dit geen promotievraag bleek maar een bug.** De notitie hier zei tot
2026-08-22 dat de vertaling "vandaag onproblematisch" was omdat de
productieketen `two_pole` nooit aanzet. Dat klopte niet meer: de kandidaat in
`consensus/` draait er dagelijks op, en die schrijft onder dezelfde
`mece_codes`-sleutel. Het gevolg was dat step 6's richtingsbewaking uit stond in
het pad dat feitelijk gedraaid werd — `opposes()` laat `neutral` bewust buiten
zijn tabel (beschrijvend materiaal heeft geen tegenpool), dus een negatief idee
onder een als `neutral` opgeslagen niet-negatieve code botste nergens mee.

Wat er is veranderd: de `Literal` in `models.py`, het weghalen van de vertaling,
en `non_negative: "negative"` in `_OPPOSITE` (`step_6_codeAssigner/
valence_filter.py`). Step 7 leest `code.valence` niet, dus die bleef ongemoeid.

Verbreden van een `Literal` is achterwaarts compatibel voor lezen: elke
bestaande cache draagt een van de drie oude waarden en die blijven geldig.

Wat de bewaking nu doet:

| code | `+` | `-` | `0` |
|---|---|---|---|
| `non_negative` | — | **botst** | — |
| `neutral` | — | — | — |
| `negative` | **botst** | — | — |
| `positive` | — | **botst** | — |

Merk op dat er wordt vergeleken met de POOL van het idee en niet met de valentie
van een andere code: in een tweedelingscodeboek bestaat geen positieve code, en
toch botst een positief idee met een negatieve. Step 6 hoeft dus nooit te weten
met welke modus een codeboek gemaakt is.

## `CodeShape.resp_pos`/`resp_neg`/`resp_neu` zijn write-only (open, low — 2026-08-20)

Buiten `grouping.py` en de tests leest niets deze drie velden; alleen
`resp_ids` wordt gebruikt. Ze waren al write-only vóór deze branch — dit is
geen regressie van het consensus-experiment, alleen een constatering ervan.
De opruimregel van dit project verbiedt dode velden, maar verwijderen raakt
`grouping.py` en valt buiten deze branch.

## `or non_negative` in de schrijfprompt is dode tekst zodra de tweedeling sneuvelt (open — 2026-08-20)

Regel 1 van de schrijfprompt (`prompts_writer.py`) noemt `neutral or
non_negative` als de twee waarden waarbij een code beschrijvend blijft, niet
richtinggevend. `non_negative` bestaat alleen via `build_shapes(two_pole=True)`
— zie "Two-pole valence" hierboven. Promoveert `consensus/` de
tweedeling niet, dan noemt de prompt een waarde die nooit meer kan voorkomen.
Opruimen hangt af van die uitkomst; de promptregel zelf blijft voorlopig
ongewijzigd.

---

## Nog niet uitgeprobeerd: snijden door step 4's boom in plaats van hergroeperen

Een alternatief dat deze keten niet is: behandel het codeboek als een **snede door step
4's boom** — facet is de code, splits een blad dat de drempel alleen haalt,
klap een facet dicht waarvan alle attributen dun zijn — in plaats van een
groepering opnieuw af te leiden. Dat maakt de codeset een deterministische
functie van (boom, tellingen, gevraagd aantal), en geeft elke code een
structureel adres (domein → facet → attributen) waar HITL op kan handelen.

De knop die dit omdraait: **het gevraagde aantal codes is de input en de drempel
de output**, per domein naar rato van zijn aandeel in het materiaal.

Wat ervoor pleit: de consolidatiecall reproduceert niet (zie boven), een snede per
definitie wel.

**Bekende grens, geen blokkade**: een snede kan omhoog samenvoegen maar nooit
onder een blad splitsen, dus hij erft step 4's granulariteit exact. Die
afhankelijkheid is symmetrisch — de huidige keten produceert dezelfde dikke code uit hetzelfde
blad (zie "De dikke duurzaamheidscode"), dus step 4's gebreken onderscheiden de
twee ontwerpen niet. Een snede maakt ze zichtbaar in plaats van ze achter een
LLM-oordeel te verbergen.

**De echte voorwaarde**: de snijregels zijn met de hand op één dataset afgeleid.
Toets ze op een tweede vóór er iets gebouwd wordt.

Referentieprofiel, ASN 2026-08-14 — 6 domeinen, 30 facetten, 88 substantiële
attributen → **40 codes**, waarvan: 2 facetgrensoverschrijdingen
(`Publieksbekendheid` → Marktpositie; `Politiek-principiële kleur` +
`Niet-winstgerichte oriëntatie`), 1 MECE-oordeel (`Groen- en natuurbeeld` vs
`Algemene natuurassociatie`), 0 codes die op valentie gesplitst moesten worden.
Per domein: duurzaamheid 6, organisatie-identiteit 12, natuur/dieren 3,
klantrelatie 7, merkuitingen 5, bankproducten 7.

**Falsificatie** — het ontwerp gaat van tafel, niet in de tuning, als op de
tweede dataset óf meer dan **5 facetgrensoverschrijdingen** nodig zijn (de
facetlaag is geen groepering waarlangs je kunt snijden), óf meer dan **een kwart
van de facetten** geen bruikbare code oplevert (de facetlaag is niet de juiste
granulariteit om mee te beginnen).

---

## Historie — geldt voor de keten in `_quarantine_v1/`

Deze bevindingen gingen over v1 en beschrijven code die niet meer draait. Bewaard
omdat de meting bruikbaar blijft als v1 ooit tegen de huidige keten gezet wordt, en omdat de
tweede sectie een fout beschrijft die elke volgende MECE-poging moet vermijden.

### De umbrella-laag reproduceerde niet (gemeten 2026-08-14)

v1's stage 2 verzon een "umbrella" per attribuut in vrije tekst en gebruikte
toevallige naamgelijkheid als join-sleutel. Twee runs op `temperature=0.0`
deelden **geen enkele** umbrella-naam: 20 vs 28 distincte umbrellas over 92
concepten, ARI **0,648**, **0 van 92** attributen kreeg twee keer dezelfde naam.
`synonym_of` leverde in beide runs 0 paren op.

Volledige opbrengst van twee LLM-calls plus een reparatiefase: twee
cross-facet-merges en één MECE-oordeel op 99 attributen.

Instrument: `_quarantine_v1/measure_grouping_stability.py`.

### Separability is not orthogonality

v1's Pass B blind probe measured whether a model CAN sort real ideas between two
codes, not whether the two codes cover different dimensions. It scored 90-97%
accuracy even for codes that plainly cover the same underlying phenomenon,
because ideas from different attributes are lexically distinguishable even when
the dimension they describe isn't.

That is why `mece_separability_threshold` could not be tuned to a value that
fixed both symptoms: at 0.70 four sustainability codes stayed separate (51
codes), at 0.80 the same (47), at 0.90 one code covered 36% of the sample (36).
Tightening the threshold pushed on a signal that did not measure the thing that
was wrong.

**This is what a next MECE attempt has to beat**: a signal that responds to
shared dimension, not shared wording.

### Pass A/B name-space ambiguity

v1's Pass A/B lookup was keyed on code name by necessity — the LLM only ever saw
names, so two codes sharing a name were ambiguous to it, not just to the Python
bookkeeping around it. `apply_merges` was re-keyed on `shape.key` so an untouched
namesake was never swept into someone else's merge; the upstream ambiguity was
contained, not eliminated.
