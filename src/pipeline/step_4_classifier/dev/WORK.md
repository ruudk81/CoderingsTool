# Step 4 — Work

## Een attribuutnaam kan in twee domeinen bestaan (open — 2026-08-20)

`_merge_duplicate_names()` vouwt dubbele attribuutnamen binnen EEN domein samen.
Over domeinen heen gebeurt dat niet, en dat is verdedigbaar: twee domeinen mogen
onafhankelijk op dezelfde naam uitkomen zonder dat het hetzelfde ding is.

Zichtbaar wordt het pas in step 5. Op de ASN-set bestaat `Algemene indruk`
tweemaal, in twee domeinen, en het codeboek toont die naam daardoor als bron
onder twee verschillende codes. Voor een lezer leest dat als hetzelfde onderwerp
op twee plekken; het id onderscheidt ze wel, de naam niet.

Niet opgelost, en niet zomaar op te lossen door alsnog over domeinen heen te
vouwen — dat zou twee verschillende dingen samentrekken. Kandidaten: de naam bij
finalisatie disambigueren met zijn domein wanneer hij elders al voorkomt, of
step 5 laten disambigueren bij het renderen. Het eerste raakt de identiteit, het
tweede alleen de presentatie.


Known gaps and planned fixes. Mechanismen horen in
[ARCHITECTURE.md](ARCHITECTURE.md), runtime in [PROCESSING.md](PROCESSING.md).

## Poort en verhuizing zijn gebouwd, niet gemeten (2026-08-17)

Twee ingrepen, twee commits, zodat een vóór/ná-meting ze kan scheiden. De
vóór-stand is de run van 2026-08-17 14:09, die in `exports/experiment_logs/`
staat.

- **De poort op naslijpen zou op die run nul keer vuren.** Alle 18 calls
  verantwoordden 100% van hun attributen. Dat is één trekking; wat het
  betekent is dat de poort geen verbetering is maar een net onder een fase die
  het op dit moment niet nodig heeft.
- **De verhuizing is nooit gedraaid.** Of een groep van één ooit een andere
  home noemt, is onbekend. Kijk eerst of hij vuurt (`attribute_relocated_
  cross_domain`), daarna pas of de bestemmingen kloppen.
- **De prompttekst van de cross-domeinronde is veranderd en dus niet meer die
  van de vóór-stand.** Een vóór/ná op cross-domein vergelijkt daarmee twee
  prompts, niet één prompt met en zonder uitgang. Wie de uitgang apart wil
  beoordelen, draait de drie tekstwijzigingen terug en laat de logregel staan.

## `facet_assignment` en `facet_settle` zijn bedraad, niet gemeten (2026-08-16)

De twee fasen bestonden al en waren getest; deze taak zette ze in de keten
tussen `facet_consolidation` en `attribute_consolidation`, met een eigen
stoppunt voor allebei. Vier dingen blijven open:

- **De meting is niet gedaan.** `data/cache` is bewust niet gekopieerd en de
  pijplijn niet gedraaid — dat kost geld en hoort bij de volgende sessie, niet
  bij de bedrading zelf. Twee runs met `stop_after_phase="facet_settle"`,
  tegen de ruisvloer van taak 1 en tegen de beslisregels van het plan.
  Rapporteer het vangnetaandeel **per domein**: op 2026-08-14 zag 7,6% er
  gezond uit terwijl één domein op 44,9% stond, en dat is precies de fout die
  een totaal verbergt.
- **Facetten krijgen geen cross-domeinronde.** `cross_domain` vouwt hetzelfde
  begrip samen dat in meerdere domeinen apart is vastgezet, maar alleen op
  attribuutniveau — facetten zijn in elke fase domein-vast, met opzet, zie
  "De domein is fixed" in `classifier.py`'s moduledocstring. Of twee facetten
  in verschillende domeinen dezelfde vraag beantwoorden, wordt dus nergens
  getoetst. Geen bekend gemeten geval; vermeld zodat het niet als vergeten
  aanvoelt de volgende keer iemand ernaar zoekt.
- **De facettoewijzing is in dit plan alleen bewijsmateriaal.** `idea_id ->
  facet_id` uit `facet_assignment` voedt de tellingen van `facet_settle` en
  wordt daarna weggegooid: de uiteindelijke `Placement` van elk idee komt uit
  de latere `assignment`-fase, die opnieuw op de dan gezette structuur draait
  en niets weet van wat `facet_assignment` koos. Dat is geen bug — er is nog
  geen structuur om ideeën definitief op te zetten vóórdat `facet_settle` heeft
  gevouwen — maar het betekent dat deze taak geen facetlaag met echte,
  behouden toewijzingen oplevert. Zie ARCHITECTURE.md.
- **Het vervolgplan zet de attribuutkant om.** Dezelfde count-aware aanpak —
  eerst een echte toewijzing, dan pas de inventaris settelen op wat ideeën
  werkelijk droegen — bestaat nu alleen voor de facetlaag.
  `attribute_consolidation` settelt nog steeds op chunk-prevalentie, hetzelfde
  systematisch-verkeerde signaal dat de aanleiding was voor deze taak. Een
  vervolgplan zou `assignment` en `attribute_consolidation` in dezelfde volgorde
  zetten als hier is gedaan voor facetten.

## De toewijzing was op naam gesleuteld (2026-08-16, OPGELOST)

**Opgelost met `Placement`**: binnen step 4 draagt een idee `(domein, facet,
attribuut)` als één ding, en `flatten_placements()` splitst dat pas op de
stapgrens. `derive_facet_assignments()` is daarmee verdwenen — het facet is nu een
gegeven en geen afleiding. Vier consumenten zijn omgezet: de tellingen en
antwoordteksten in `_build_refinement_tasks`, `remap` in `_apply_refinement`, de
aantallen in `_run_cross_domain`, en de facetregisters bij het wegschrijven.

Eén restpost blijft: **twee facetten met dezelfde naam** binnen één domein. De
`Placement` draagt de facetnaam, niet zijn positie, en `_run_cross_domain` klapt
gelijknamige facetkaarten sowieso al samen (`rebuilt[domein][facetnaam]`). Dat is
een oudere, aparte samenvoeging — niet nieuw en niet door deze ingreep
verergerd — maar wel de reden dat de plaatsing niet volledig exact is.

Wat het probleem wás, als naslag:

`assignments` was `Dict[idea_id, attribute_name]`. Twee facetten van hetzelfde
domein mogen een attribuut met dezelfde naam dragen — attribuutconsolidatie doet
één facet per call en ziet de buren niet — en dan is niet meer te zeggen in welk
facet een idee zit.

**Aangetoond in de run van 2026-08-16 15:00**, uit de cross-domein-inventaris:

```
financiële dienstverlening | Online bankieren
    [A20] Bankhandelingen en kanalen        → 6 responses
    [A31] Financiële producten en diensten  → 6 responses
merkidentiteit en communicatie | Merkbekendheid
    [A59] Merknaam en identiteit    → 18 responses
    [A66] Visuele merkherkenning    → 18 responses
    [A79] Merkuitstraling           → 18 responses
```

Dat zijn niet drie keer achttien responsen maar één keer achttien, drie keer
geteld: `Counter(assignments.values())` telt op naam. 66 van 1946 regels in de
inventaris (3,4%) staan op een dubbele naam.

**Wat er stukgaat**, allemaal omdat ze (facet, attribuut) reconstrueren uit een
naam:

- `_build_refinement_tasks` — `counts` en `contents` op naam. Alle drie de
  `Merkbekendheid`-facetten krijgen hetzelfde aantal én dezelfde antwoordteksten
  te zien, inclusief responsen die in een ander facet zitten.
- `_apply_refinement` — `remap[(domein, naam)]`. Claimen twee facetten dezelfde
  naam, dan wint de laatste schrijver en gaan álle ideeën op die naam naar één
  doel.
- `derive_facet_assignments` — `home[_norm(naam)] = (domein, facet)`, dus de
  laatste facet in de lus wint voor élk idee op die naam.
- `_run_cross_domain` — de aantallen in de inventaris, zie hierboven.

**Waar het verloren gaat, en waarom de fix niet lokaal is.** De toewijzing wéét
het facet: `build_assignment_menu` bouwt `id_map[A17] = {facet_name,
attribute_name, is_drain}` en het model kiest een id, geen naam. De parse gooit
het facet weg door alleen `attribute_name` in `assignments` te schrijven. Aan de
naslijpkant is het dus niet te repareren — dáár is de informatie er al niet meer.

Twee routes zijn afgewezen. **Namen uniek maken per domein** lost het symptoom
op met een inhoudelijke ingreep: je legt twee facetten een naam op om een
sleutelprobleem te omzeilen, en de klant ziet die naam. **Op A#-ids werken**
vergt dat `identity.py` eerder stempelt dan bij het opslaan, en levert binnen de
stap niets op wat een tuple niet ook geeft.

Let op de wisselwerking met de facetsplitsing: de domeinbrede naslijpcall van
vóór 2026-08-16 zag de drie kopieën naast elkaar en kón ze samenvoegen. De
facetcalls zien elkaar niet. Cross-domein kan het nog repareren — die zoekt
expliciet duplicaten over facetten — maar pas nadat naslijpen er drie keer los op
heeft geoordeeld.

## Na het slopen van de misfit-uitgang (2026-08-16)

Vier ingrepen op één dag, alle vier op bewijs uit het actielog van de run van
16:24. Wat is gedaan:

- **De misfit-uitgang is weg** — `RefinementMisfitGroup`, `instance_texts`,
  `build_move_targets_block`, regel 5 en de hele routeerlus in
  `_apply_refinement`. Hij noemde zijn bestemmingen vóór de fase en resolvede ze
  erna; 83 van 118 teksten (70%) landden op een naam die hun buurcall net had
  opgeslokt, 31 hadden helemaal geen bestemming. Naslijpen kan nu alleen nog
  samenvoegen.
- **De `<10%`-drempel is weg** en de regels staan nog maar op één plek. Ze
  stonden dubbel — promptbody én `scratchpad`-description — en de twee kopieën
  droegen verschillende versies van regel 4.
- **Het facettotaal staat in de kopregel** van `build_contents_block`, zodat
  "klein" een ijkpunt heeft binnen een call die één facet ziet.
- **`_merge_duplicate_names()`** vouwt na naslijpen samen wat één naam in twee
  facetten van één domein draagt, deterministisch.
- **Vangnetten doen ook als bron niet mee**, in de prompt (regel 5) én in de
  code (`drain_norms`, actie `drain_source_ignored`).

Wat daarvan open blijft:

**Alle vier zijn ongemeten.** Er is één trekking vóór (de run van 16:24) en nul
erna. Draai vóór/ná op dezelfde step-3-invoer, twee runs per stand voor de
ruisvloer, met `stop_after_phase="refinement"`. Tegenmetriek: het grootste blad
én het vangnetaandeel **per domein** — een totaal middelt zijn eigen signaal weg.
De vier ingrepen zijn in één keer gedaan, dus de meting scheidt ze niet; wie een
van de vier apart wil beoordelen moet hem apart terugdraaien.

**De plaatsingstoets is er (2026-08-17), en is ongemeten.** Een groep van één
mag sindsdien een andere scope als home noemen — zie ARCHITECTURE.md,
"Cross-domein". Wat open blijft is of hij vuurt en of hij het goede doet: er is
nul runs mee gedraaid. De aanleiding stond op 2026-08-16 in dit bestand —
`Betrouwbaarheid en veiligheid` (66 ideeën) en `Deskundigheid` onder het facet
`Schaal en ontwikkeling` — en die run is de vóór-stand. Tegenmetriek:
`attribute_relocated_cross_domain` tellen naast het grootste blad en het
vangnetaandeel per domein, want churn is het risico van deze uitgang, niet
onjuistheid.

**Cross-domein kiest zijn home op omvang.** De prompt zegt *"Choose the scope
where most of these responses already sit"*. Op 2026-08-16 verhuisde
`Klantgerichte hulp en service` (33 ideeën) daardoor naar het facet
`Interpersoonlijke klantbenadering`, terwijl het facet `Klantondersteuning en
afhandeling` met 6 ideeën achterbleef — een facet dat bijna hetzelfde heet als
het attribuut dat eruit vertrok. Alternatief: de home is de scope wiens
facetvraag de overlevende beantwoordt. Dat is een promptwijziging en vraagt een
meting; niet doen als bijvangst.

**`cross_domain` heet niet meer wat het doet.** Het is de fase over álle scopes.
Hernoemen raakt het perf-model en de configsleutels (zie de regel over
fasesleutels in `CLAUDE.md`), dus het is een eigen ingreep.

**De vuilnisbakketen op één facet.** `Schaal en ontwikkeling` liep op 2026-08-16
door vier gedocumenteerde faalvormen tegelijk: tweede claimant van F13 én F14
(`divided_source_facet` ×2, dus zonder attributen), daarna
`attribute_consolidation_failed` ("kept every candidate"), daarna een naslijpcall
die 2 van de 7 attributen verantwoordde, daarna een cross-domein-instroom uit een
ander domein. De derde stap is sinds 2026-08-17 gedekt: naslijpen heeft dezelfde
poort als de consolidatiefasen. Op de run van 2026-08-17 zou hij nul keer
gevuurd hebben — alle 18 calls verantwoordden 100% — dus hij is verzekering en
geen verbetering, en de 13 `attribute_kept_unclaimed_in_refinement` van
2026-08-16 zijn niet reproduceerbaar gebleken op de nieuwe prompttekst.

## Na de splitsing van consolidatie (2026-08-15)

De gecombineerde consolidatiecall is gesplitst in `facet_consolidation` en
`attribute_consolidation`. Alles hieronder is open.

**Eén run gedaan, de vóór/ná-meting niet.** De run van 2026-08-16 is de eerste
sinds de splitsing en staat in `exports/experiment_logs/`, mét
`facet_consolidation`- en `attribute_consolidation`-regels. Ruwe uitkomst: 63
kandidaatfacetten → 26, en 222 kandidaatattributen → 126 over 25 calls. Dat is
één trekking zonder ruisvloer ernaast, dus het beoordeelt de ingreep niet. Draai
vóór/ná op dezelfde step-3-invoer, met twee runs per stand.

**Eén van de 25 attribuutcalls stortte in.** `Reclame en communicatie`
verantwoordde 1 van 12 kandidaten en gaf één attribuut terug dat `"Voorlopige
consolidatie"` heette; het net herbouwde de andere elf en het log schreef
`12 → 12`. Sinds 2026-08-16 weigert de parse zo'n antwoord en draait de
requester de call opnieuw (zie ARCHITECTURE, "Eerst een poort, dan pas een
net"). **Wat daarmee níét is opgelost is de aanleiding**: die pool bevatte vier
kandidaten die hetzelfde beschrijven (`Aansprekende reclame-uitwerking`,
`Communicatieve werking`, `Ongewone reclame-uitwerking`, `Reclame-uitstraling`,
twee ervan met een letterlijk identiek voorbeeld) en elf van de twaalf stonden
op `[1/4 passes]`, dus prevalentie gaf geen enkel onderscheid. Verstrengelde
pool zonder tiebreak; de poort zorgt alleen dat het niet meer stil doorgaat.

**Hoe vaak dit gebeurt is onbekend.** n = 1 run, 1 gebeurtenis. De meting die
het beantwoordt is goedkoop: attribuutconsolidatie 3× draaien met
`stop_after_phase="attribute_consolidation"` en per call `accounted_for /
candidates` uit het actielog tellen. Dat levert meteen de ruisvloer die de
vóór/ná-meting hierboven toch nodig heeft.

**Er is geen streefaantal.** Het aantal attributen is een *symptoom* waaraan te
zien is of een samenvoegcriterium werkt, nooit een doelvariabele om op te
sturen. Een getal dat op één dataset goed voelt, is een eigenschap van díé
dataset; sturen op zo'n getal is de dataset in de prompts fitten zonder dat er
ooit een getal in een prompt staat. Wat generiek moet zijn is het criterium —
de MERGE TEST, prevalentie binnen één vraag, de toets op de facetvraag — en de
toets daarop is of het op een tweede dataset óók werkt.

**Tegenmetriek verplicht.** Minder attributen is triviaal te bereiken door harder
samen te voegen; het aantal alleen kan de ingreep dus niet goedkeuren. De
tegenmetriek is tweeledig: het aantal ideeën in het grootste blad (nu 320 / 25%
van de respondenten) en het vangnetaandeel **per domein** (44,9% en 25,4% in twee
domeinen, tegen 0,3-1,6% in vier andere — een totaal van 7,6% middelt zijn eigen
signaal weg). Beide moeten meebewegen, anders is er alleen verplaatst.

**Het model van de attribuutfase is een aanname.** Beide fasen staan op
`("5.4", 5)`, omdat de gecombineerde voorganger daar op 2026-08-15 heen ging. De
attribuutfase heeft een veel smallere scope — één facet, geen domein — en is
daarmee de eerste kandidaat om naar `("5.6", 3)` te zakken. Pas op een meting.

**De prompt zwijgt over een kandidaat die werkelijk uiteenvalt.** De legacy-prompt
zei het expliciet: een kandidaat wiens inhoud echt over meerdere survivors
verdeeld hoort, mag door elk van die survivors genoemd worden — "that is the
honest record". Géén van de twee nieuwe prompts zegt dat nog. De code handelt het
geval wél af: de attributen gaan naar de eerste claimant en het krijgt een
`divided_source_facet`-regel. Op de run van 2026-08-15 (nog de gecombineerde
fase) vuurde dat 22 keer. **Behandel dat getal als een bovengrens:** die run
draaide vóór de fix in `88b29fa4`, waarin een survivor die hetzelfde id twee keer
citeert nog als twee claimanten telde en dus een splitsing meldde die niet
bestond. Hoeveel van de 22 echt gedeelde kandidaten waren, is pas na een nieuwe
run bekend — maar het geval bestaat en de code handelt het af. Openstaande
promptvraag:
moet de facetprompt dit weer zeggen, of is de eerste-claimant-regel stil het
gewenste gedrag? De tweede survivor mist die attributen tot naslijpen ze
verplaatst — kijk hiernaar na de eerste run.

## Eerste waarnemingen op de zesfasige opzet (2026-08-14)

Gemeten op de taxonomie die nu in de cache staat (ASN, 1236 respondenten, 2182
ideeën), vanaf de step-5-kant. Drie van de vier openstaande vragen hieronder zijn
daarmee beantwoord; de vierde (vuren de rondes) is van buitenaf niet zichtbaar.

- **Aantal.** 99 attributen (88 inhoudelijk + 11 vangnet) over 30 facetten.
  Zonder streefgetal is dat op zichzelf geen oordeel; wat het wél zegt staat
  hieronder, in het blad dat een kwart van de steekproef draagt.
- **Vangnetaandeel: 7,6% totaal — en dat cijfer verbergt het probleem.**
  Per domein: **44,9%** (merkuitingen en herkenning, 96/214 ideeën) en **25,4%**
  (bankproducten, 52/205), tegen 0,3-1,6% in de vier andere domeinen. In het
  merkendomein valt bijna de helft van wat mensen zeiden buiten elk facet.
  `taxonomy_health` aggregeert dit tot één percentage over de hele run, en op
  dat niveau ziet 7,6% er gezond uit. **De tegenmetriek moet per domein
  gerapporteerd worden, anders middelt hij zijn eigen signaal weg.**
- **`facet == attribuut`: twee gevallen**, waarvan één met gewicht —
  `Kleur- en groenbeelden` → `Groen- en natuurbeeld` (81 ideeën). Het andere is
  triviaal (`Ideële oriëntatie`, 5 ideeën). De toets in de discovery-prompt houdt
  dus grotendeels stand.

### Eén blad draagt een kwart van de steekproef (open, zwaar)

`Duurzaamheidsgerichte koers`: **320 ideeën / 308 respondenten** — 25% van alle
respondenten in één attribuut, naast `Natuur- en klimaatbescherming` (78) en
`Toekomstgerichte ecologie` (7) in hetzelfde facet.

Waarom dit step 4's probleem is en niet step 5's: een codeboek is een snede door
deze boom, en een snede kan omhoog samenvoegen maar nooit onder een blad
splitsen. Wat hier in één attribuut zit, kan stroomafwaarts nooit meer uit
elkaar. Hetzelfde geldt voor de twee vangnetten hierboven: een drain van 45%
wordt in step 5 onvermijdelijk één enorme Overig-code.

De prevalentieregel in de **attribuut**consolidatieprompt (regel 2) is de eerste
plek om te kijken — zoals de lens-notitie hieronder al voorspelde, maar dan de
andere kant op: niet te veel kleine facetten, maar één blad dat alles opzuigt.

Sinds 2026-08-16 hoeft dit niet meer los te worden opgezocht: `taxonomy_health`
rapporteert `largest_leaf` (met `largest_leaf_share`), `leaf_buckets` en
`largest_facet` (met `largest_facet_share`) op elke run, vangnetten uitgezonderd
via `is_drain_item`. Dat meet alleen — de onderliggende samenvoeging is er nog
niet door aangepast.

## De lens-framing kan smalle, talrijke items geven (open, verwacht)

De instructies komen weer uit `old_step_4_prompts.py`, en dat register leverde
bij de start van de vorige herbouw 49 facetten / 182 attributen. De dimensie-
vraag die dat toen terugbracht is er bewust uit. Consolidatie en naslijpen moeten
het nu in hun eentje doen.

Als het aantal te hoog uitvalt is de eerste plek om te kijken de
prevalentieregel in de consolidatieprompts — de facetprompt voor te veel
facetten, de attribuutprompt voor te veel attributen — en niet de
discovery-prompt: grover indelen hoort te gebeuren waar de aantallen zichtbaar
zijn.

## `verdict: "out"` heeft geen bestemming (open, oud)

Contentloze ideeën blijven staan waar ze zitten, geteld in het actielog. Met de
vangnetten is dit minder scherp dan het was — een kaal oordeel heeft nu tenminste
een eerlijke plek — maar `out` als uitgang doet nog steeds niets.

Oorzaak ligt op de step-3/step-4-grens: step 3 heeft `bare_evaluation` vervangen
door twee vangnetten die iets anders vangen, dus een oordeel zonder onderwerp
heeft geen domein en wordt over de inhoudelijke domeinen uitgesmeerd.

## Kleinere afwijkingen, gevonden bij de promptaudit van 2026-08-15

- **`prompts_valence.py` gebruikt `attribute_description`** waar de rest van
  step 4 sinds 2026-08-12 `attribute_definition` heet. Het lekt niet: regel 260
  van `valence_consolidator.py` leest het responsemodelveld en regel 282 schrijft
  de dict-sleutel, dus de vertaling gebeurt op de grens. Wel een naam die in zijn
  eentje uit de pas loopt.
(De derde bevinding — de toewijzingsprompt eindigt als enige niet op
`UNIVERSAL_RULES` — is opgelost door de docs te laten wijken, niet de code:
`prompts_shared.py` en `DESIGN_VALENCE_NEUTRALITY.md` benoemen die fase nu als
bewuste uitzondering. Hij kiest een id uit een menu en bedenkt geen namen.
Zie hieronder: die oplossing blijkt zelf één fase te missen — inmiddels twee,
sinds `facet_assignment` op 2026-08-16 op dezelfde grond is bijgeschreven in
`prompts_shared.py:15`. `prompts_valence.py` is de enige die nog niet genoemd
wordt, en om een andere reden: hij bedenkt wél een naam. Zie hieronder.)

## `valence_merge` valt buiten twee "enige"-claims (open, laag, 2026-08-15)

Beide gevonden bij de eindreview van de consolidatiesplitsing, en allebei bewust
geparkeerd: ze raken `valence_merge`, de negende fase die vanuit de runner
draait, en die is bij die splitsing niet aangeraakt. Hier vastgelegd zodat ze
niet met dat traject verdwijnen.

**1. `prompts_refinement.py:365` — "The only phase that sees more than one domain
at a time".** Dat is te ruim. `valence_consolidator._rename()` bouwt één payload
uit álle `merge_pairs` en doet daar één LLM-call over; `detect_valence_splits()`
groepeert op `(domein, facet)` over de hele studie, dus die ene call ziet paren
uit meerdere domeinen. De claim misleidt eerder dan dat hij liegt: de
valence-payload draagt alleen `pair_id`, `name_a/b`, `desc_a/b` en `samples` —
géén domeinveld — dus die call rendert nooit een domeinnaam en vergelijkt nooit
over een domeingrens. De bedoeling van de zin overleeft dus; de formulering niet.
Nauwkeurig is: de enige fase die attributen **over domeingrenzen heen
vergelijkt**. Lage prioriteit — meenemen wanneer die prompt toch openligt.

**2. `prompts_shared.py:15` telde vóór 2026-08-16 één te weinig, maar de kern is
niet de zin.** De regel noemde alleen toewijzing als uitzondering op
`UNIVERSAL_RULES`; sinds taak 7 van de facetlaag noemt hij de twee
toewijzingsfasen (facet- én attribuut-) en wijst voor `prompts_valence.py`
apart naar dit item. Dat lost het telfoutje op, niet de vraag eronder.
`prompts_valence.py` importeert alleen `INSTRUCTOR_HINT` en eindigt op
`Begin now and {INSTRUCTOR_HINT}`, dus het is een derde prompt zonder
`UNIVERSAL_RULES` — en om een andere reden dan de twee toewijzingsfasen. Zij
kiezen een id uit een menu en bedenken geen naam; de valence-merge bedenkt er
wél een — het neutrale samengevoegde label
(`ValenceNeutralAttribute.attribute_name`, hoogstens 5 woorden). De enige prompt
in step 4 die bestaat om valence-splitsingen te repareren, is dus ook de enige
naamgevende prompt zonder regel 2, de regel die splitsen op evaluatieve richting
verbiedt. Dat de moduledocstring van `prompts_valence.py` zélf naar die regel
verwijst als "the lever that actually works", maakt het scherper.

*Wat hier open staat is niet of het commentaar klopt, maar of
`prompts_valence.py` `UNIVERSAL_RULES` hóórt te dragen.* Dat is een
promptontwerpbeslissing en die wil een meting, geen doc-aanpassing. Wie het
oppakt: lees eerst wat die prompt al over neutraliteit zegt — hij verbiedt
expliciet het coderen van positief/negatief en vraagt om het onderliggende
onderwerp — en lees
[DESIGN_VALENCE_NEUTRALITY.md](DESIGN_VALENCE_NEUTRALITY.md), waar de naamgeving
van deze fase beschreven staat als een LLM-call met een deterministische
single-token fallback. Mogelijk zijn de vier universele regels hier overbodig of
zelfs contraproductief; mogelijk is regel 2 juist precies wat ontbreekt.

Let op bij het oppakken: dezelfde te enge formulering stond op twee plekken.
`prompts_shared.py:15` is bijgewerkt (zie hierboven); `DESIGN_VALENCE_NEUTRALITY.md`
zegt het nog op de oude manier — "Every step-4 prompt that names something ends
on `UNIVERSAL_RULES`. (Assignment is the exception…)" — en telt daar dus nog
steeds maar één uitzondering. Wie deze open vraag oppakt repareert die zin ook.

## Een observatie is ruwe tekst én interpretatie in één string (open, step-3-grens)

Step 3 levert zijn partitielabels als `"geen winst → niet op winst gericht"`: het
antwoord van de respondent en de normalisering die step 3 eraan gaf, samengeplakt.
Step 4 geeft die string ongewijzigd door als `example_observations`, dus het
codeboek toont hem ook zo.

Het nummer ervoor is opgelost (`_strip_enumeration`), de pijl niet. Splitsen op
`→` binnen step 4 zou het formaat van step 3 hardcoderen in een stap die er niets
van hoort te weten, en breekt op elke dataset waar die pijl niet staat. De echte
vraag is een contractvraag: moet step 4 ruwe observatie en interpretatie als twee
velden krijgen in plaats van als één string? Zie
[dimension_data is een gedeeld contract] in het geheugen — dezelfde grens.

Merk op dat de keuze consequenties heeft voor step 5: de ruwe tekst laat zien hoe
respondenten praten, de interpretatie waarom step 3 het idee daar plaatste.

## Uit de review van 2026-08-15 niet uitgevoerd, met reden

- **Domeingrens voor financiële producten.** De review wilde een inclusieregel in
  de consolidatieprompt — destijds nog één, inmiddels de facetprompt. Die grens
  komt per domein uit step 3 (`boundary_test`,
  `exclusions`); een regel in step 4 zou een tweede, concurrerende bron worden.
  Als de grens onduidelijk is, is dat step-3-werk.
- **Eén vaste facet-as, een domeinspecifiek merge-voorbeeld, prevalentiedrempels
  als "3-5 passen".** Alle drie afgelezen van de ASN-data. Zie het lekpad in
  CLAUDE.md en het getallenverbod hieronder. De generieke vervanging is de toets
  op de facetvraag: is hij te beantwoorden door een onderwerp te noemen, dan is
  het een domeinsplitsing.
- **De stappen 3/4/6 samenvatten tot verwijzingen naar één beslisboom.** De
  stappen zijn de werkinstructie op de plek waar het werk gebeurt; herhaling daar
  kan een anker zijn. Weghalen is een gedragswijziging en vraagt een meting.
  De létterlijke dubbeling (facet- en attribuutdefinitie tweemaal gerenderd) is
  wel weg — dat was dezelfde zin binnen een paar honderd woorden.

## Identieke tekst wordt als blok behandeld (geaccepteerde versimpeling)

Toewijzing doet één call per uniek label, dus elke instantie van een tekst volgt
dezelfde route. Fout waar één woord in twee contexten twee dingen betekent. De
uitgangen die dit ook op naslijpen lieten gelden (`split`, `move`) bestaan niet
meer, dus het speelt nu alleen nog op de toewijzingsgrens.

## Een greedy consolidatiegroep kan buren scheiden

Kandidaten worden op genormaliseerde facetnaam gesorteerd vóór het groeperen,
zodat bijna-identieke voorstellen meestal in dezelfde groep vallen. Meestal,
niet altijd: de groepsgrens kan precies tussen twee buren vallen. Aanvaard — de
volgende ronde zet de overlevenden alsnog bij elkaar.

## Verboden richting bij verder sleutelen

Geen getallen in prompts. Een drempel als "minstens 5% van zijn scope" is
afgelezen van één dataset en valt onder hetzelfde verbod als een
use-case-voorbeeld. Wat wél mag: een aantal dat tijdens de run uit de data komt.
Gemeten alternatief dat níét werkt: PCA op de embeddings; 99% van de variantie
vraagt 55 tot 245 componenten, want het eigenwaardespectrum van
zins-embeddings is vlak.

## Historie

- Het assen-traject (2026-08-01/02) en zijn metingen staan in git en in
  `exports/diagnostics/2026-08-01-*`.
- De tienfasige opzet (twee lagen, elk vier fasen, plus twee cross-scope rondes)
  liep tot 2026-08-13. Zijn hoofdprobleem — naslijpen blies op wat consolidatie
  samenvoegde, cross-scope haalde het er weer af, netto stilstand — is de
  aanleiding voor de herschreven splitsclausule.
