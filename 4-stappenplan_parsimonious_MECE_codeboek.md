# Specificatie: van survey-taxonomie naar een parsimonious, relevant en zoveel mogelijk MECE codeboek

## 0. Doel en uitgangspunten

### 0.1 Doel

Deze specificatie beschrijft een zelfstandig uitvoerbaar proces waarmee een agent uit open surveyantwoorden en een LLM-gegenereerde taxonomie een compact, empirisch onderbouwd codeboek maakt. Het codeboek moet:

- zo weinig mogelijk codes bevatten zonder relevante betekenis te verliezen;
- sterk door prevalentie worden gestuurd;
- bruikbaar en betrouwbaar zijn voor menselijke of automatische codering;
- op het niveau van afzonderlijke **idea units** zoveel mogelijk mutually exclusive en collectively exhaustive (MECE) zijn;
- valentie in codes mogen opnemen, terwijl `domain`, `facet` en `attribute` descriptief en neutraal blijven;
- beslissingen volledig herleidbaar maken tot brondata, tellingen en regels.

De taxonomie is een rijke beschrijving van de betekenisruimte. Het codeboek is geen één-op-één vertaling daarvan, maar een empirisch gewogen compressie van wat daadwerkelijk in de antwoorden voorkomt.

### 0.2 Exacte vier stappen

1. **Empirical Concept Inventory**
2. **Consolidation: MERGE / KEEP / SPLIT / DROP**
3. **Valence Resolution**
4. **Final Codebook**

De stappen moeten in deze volgorde worden uitgevoerd. Iteratie naar een eerdere stap is alleen toegestaan wanneer een kwaliteitscheck faalt; leg iedere iteratie vast.

### 0.3 Eenheid van analyse: idea unit

Een **idea unit** is het kleinste zelfstandig codeerbare tekstsegment dat één inhoudelijke bewering, waarneming, behoefte, voorkeur of evaluatie uitdrukt. Eén antwoord kan nul, één of meerdere idea units bevatten.

Voorbeeld:

> “Het product is lekker, maar veel te duur en de verpakking gaat moeilijk open.”

Bevat ten minste drie idea units:

1. product smaakt lekker;
2. prijs is te hoog;
3. verpakking is moeilijk te openen.

**MECE geldt primair op idea-unitniveau, niet op antwoordniveau.** Een antwoord mag daarom meerdere codes krijgen. Een idea unit krijgt in beginsel precies één inhoudelijke eindcode. Een tweede code is alleen toegestaan wanneer dezelfde tekst aantoonbaar twee niet-reduceerbare betekenissen bevat en verdere segmentatie linguïstisch onmogelijk of betekenisvervormend is. Leg zo'n uitzondering vast.

### 0.4 Taxonomiemodel

Iedere idea unit kan worden beschreven met:

- `domain`: breed inhoudelijk gebied;
- `facet`: onderdeel of perspectief binnen het domein;
- `attribute`: concrete descriptieve eigenschap of dimensie;
- `valence`: evaluatieve richting, pool of positie op de dimensie.

Alleen `valence` is evaluatief. Labels voor `domain`, `facet` en `attribute` moeten neutraal en descriptief zijn. Voorbeelden van valentie zijn positief/negatief, hoog/laag, te veel/te weinig, makkelijk/moeilijk, aanwezig/ontbrekend of een andere empirisch passende richting.

### 0.5 Belangrijke definities

- **Prevalentie:** aantal of aandeel unieke analyse-eenheden waarin een concept voorkomt. Rapporteer minimaal prevalentie onder idea units en onder respondenten/antwoorden.
- **Relevantie:** bijdrage aan het beantwoorden van de onderzoeksvraag of een vooraf vastgelegd beslisdoel.
- **Parsimonie:** minimale codeboekcomplexiteit die nog voldoende empirische en analytische informatiewaarde behoudt.
- **Mutually exclusive:** twee codes concurreren niet om dezelfde enkelvoudige idea unit.
- **Collectively exhaustive:** iedere relevante, inhoudelijke idea unit kan worden gecodeerd, eventueel tijdelijk met een gecontroleerde restcode.
- **Concept:** een nog valentie-neutrale, empirisch aangetroffen betekenis, gewoonlijk gebaseerd op `domain × facet × attribute` plus semantische formulering.
- **Code:** een operationele categorie in het eindcodeboek, eventueel met expliciete valentie.

### 0.6 Globale inputs

Vereist:

1. stabiele `response_id` en, indien beschikbaar, `respondent_id`;
2. originele antwoordtekst, ongewijzigd bewaard;
3. surveyvraag en relevante vraagcontext;
4. taxonomie met definities voor `domain`, `facet`, `attribute` en `valence`;
5. initiële taxonomietoewijzingen per antwoord of tekstsegment, indien aanwezig;
6. onderzoeksvraag, gebruiksdoel en eventuele vooraf vastgelegde kritieke thema's;
7. taal en populatiecontext.

Aanbevolen:

- steekproefgewichten;
- relevante respondentsegmenten;
- coderingshistorie of menselijke annotaties;
- lijst met veiligheids-, compliance- of bedrijfscritische signalen;
- gewenste maximale codeboekgrootte als zachte, niet als absolute grens.

### 0.7 Globale prevalentieregels

Gebruik standaard zowel absolute als relatieve prevalentie. Definieer vóór de analyse:

- `N_resp`: aantal geldige respondenten;
- `N_ans`: aantal niet-lege antwoorden;
- `N_iu`: aantal inhoudelijke idea units;
- `n_resp(c)`: unieke respondenten met concept/code `c`;
- `n_ans(c)`: unieke antwoorden met `c`;
- `n_iu(c)`: idea units met `c`;
- `p_resp(c) = n_resp(c) / N_resp`;
- `p_ans(c) = n_ans(c) / N_ans`;
- `p_iu(c) = n_iu(c) / N_iu`.

Voorkom dat herhaling binnen één lang antwoord prevalentie kunstmatig opblaast: voor respondent- en antwoordprevalentie telt een concept per respondent respectievelijk antwoord maximaal één keer. Gebruik surveygewichten aanvullend, niet ter vervanging van ongewogen tellingen.

Als de opdrachtgever geen drempels opgeeft, gebruikt de agent deze **adaptieve standaardwaarden**:

- `T_keep = max(3 respondenten, 1% van geldige respondenten)`;
- `T_split_child = max(3 respondenten, 1% van geldige respondenten)` per dochtercode;
- `T_rare = minder dan T_keep`;
- voor kleine steekproeven (`N_resp < 100`): gebruik primair absolute aantallen en inhoudelijke dekking;
- voor zeer grote steekproeven (`N_resp >= 10.000`): verhoog de relatieve keep-drempel zo nodig naar 0,5–1% of gebruik een vooraf afgesproken informatiewaardecriterium om code-explosie te voorkomen.

Deze waarden zijn startpunten. De agent mag ze aanpassen aan steekproefgrootte, antwoordlengte en onderzoeksdoel, maar moet de aanpassing motiveren en vóór consolidatie vastleggen.

### 0.8 Beschermde uitzonderingen op prevalentie

Een zeldzaam concept mag alleen zelfstandig blijven wanneer minstens één van deze redenen aantoonbaar geldt:

- veiligheids-, ethisch, juridisch of compliance-risico;
- expliciet strategisch of onderzoekskritiek thema;
- klein maar relevant respondentsegment zou anders systematisch worden gewist;
- unieke, actiegerichte betekenis zonder passende bovenliggende code;
- plausibel opkomend signaal waarvoor detectie belangrijker is dan frequentie.

Markeer dit als `protected_exception = true`, vermeld de reden en rapporteer het als zeldzaam. Gebruik uitzonderingen spaarzaam; zij mogen parsimonie niet ongemotiveerd ondermijnen.

---

## 1. Empirical Concept Inventory

### 1.1 Doel

Maak een volledige, brongebonden inventaris van de betekenissen die werkelijk in de antwoorden voorkomen. Deze inventaris is nog geen codeboek en bevat nog geen definitieve valentiecodes. De stap voorkomt dat theoretisch mogelijke taxonomiecombinaties zonder empirische steun als code worden aangemaakt.

### 1.2 Inputs

- originele antwoorden en identifiers;
- surveyvraag en context;
- taxonomie en definities;
- initiële taxonomietoewijzingen, indien beschikbaar;
- onderzoeksdoel, kritieke thema's en segmentvariabelen;
- vooraf vastgelegde segmentatie- en prevalentiedrempels.

### 1.3 Outputs

1. een idea-unitbestand met bronspans en taxonomietoewijzingen;
2. een conceptinventaris met één rij per empirisch concept;
3. prevalentie per concept, totaal en per relevant segment;
4. representatieve en grensgevallen per concept;
5. onzekerheden, taxonomiegaten en mogelijke duplicaten;
6. een auditlog van segmentatie- en normalisatiebeslissingen.

### 1.4 Procedure

#### 1.4.1 Data voorbereiden

1. Behoud de originele tekst exact in `response_text_raw`.
2. Maak alleen voor analyse een genormaliseerde versie; verwijder geen ontkenningen, intensiveringen of vergelijkingen.
3. Markeer lege, onbegrijpelijke, puur procedurele en buiten-scope antwoorden afzonderlijk.
4. Dedupliceer respondenten niet zonder expliciete regel. Markeer exacte duplicaten zodat gevoeligheidsanalyse mogelijk blijft.

#### 1.4.2 Antwoorden segmenteren in idea units

1. Splits wanneer een tekst meerdere zelfstandig codeerbare proposities bevat.
2. Splits niet enkel op leestekens; voeg fragmenten samen als zij alleen gezamenlijk betekenis hebben.
3. Bewaar per unit de exacte `source_span` en positie in het antwoord.
4. Laat contrastconstructies zoals “lekker maar duur” in aparte units uiteenvallen.
5. Bewaar context die nodig is voor pronomen, ellipsen, sarcasme of vergelijking.
6. Ken een `segmentation_confidence` toe en stuur lage zekerheid naar review.

#### 1.4.3 Taxonomie toewijzen

1. Ken aan iedere inhoudelijke idea unit de best passende `domain`, `facet` en `attribute` toe.
2. Gebruik alleen bestaande taxonomie-items wanneer de definitie werkelijk past.
3. Registreer een nieuw kandidaat-item als `taxonomy_gap` als geen item past; forceer geen toewijzing.
4. Registreer waargenomen valentie voorlopig als bewijskenmerk, niet als definitieve code.
5. Bewaar alternatieve toewijzingen en confidence bij ambiguïteit.

#### 1.4.4 Empirische concepten vormen

Groepeer idea units die hetzelfde descriptieve kernconcept uitdrukken. Start met `domain × facet × attribute`, maar controleer semantisch of:

- verschillende taxonomiecombinaties feitelijk hetzelfde concept benoemen;
- één combinatie meerdere inhoudelijk verschillende concepten maskeert;
- formuleringen synoniem, hiërarchisch of causaal gerelateerd zijn;
- het concept zonder valentie begrijpelijk en neutraal kan worden beschreven.

Maak geen concepten voor lege cellen in de taxonomie. Een concept vereist minstens één bron-unit.

#### 1.4.5 Prevalentie en bewijs berekenen

Bereken per concept:

- `n_iu`, `n_ans`, `n_resp` en bijbehorende percentages;
- gewogen prevalentie indien gewichten bestaan;
- prevalentie per vooraf relevant segment;
- verdeling van voorlopig waargenomen valentie;
- aantal en aandeel onzekere toewijzingen;
- minimaal drie representatieve voorbeelden indien beschikbaar;
- minimaal één grensgeval en één mogelijk tegenvoorbeeld indien beschikbaar.

Voor representatieve voorbeelden: selecteer semantisch centrale, korte, privacyveilige fragmenten; kies niet uitsluitend spectaculaire uitspraken.

### 1.5 Beslisregels

- Maak alleen empirisch aangetroffen concepten.
- Houd conceptlabels valentie-neutraal.
- Combineer nog niet uitsluitend vanwege lage prevalentie; consolidatie gebeurt in stap 2.
- Splits een conceptinventaris-item voorlopig als de onderliggende units verschillende operationele betekenissen hebben.
- Tel één unit niet dubbel binnen hetzelfde concept.
- Markeer concepten onder `T_keep` als `rare_candidate`, niet automatisch als DROP.
- Markeer een concept als `segment_concentrated` wanneer de totale prevalentie laag is maar de prevalentie in een vooraf relevant segment substantieel is.

### 1.6 Aanbevolen datastructuren

#### Idea unit

```yaml
idea_unit_id: IU-000001
response_id: R-0001
respondent_id: P-0001
source_span: "veel te duur"
response_text_raw: "Lekker, maar veel te duur."
span_start: 12
span_end: 25
domain_id: D-PRICE
facet_id: F-AFFORDABILITY
attribute_id: A-PRICE_LEVEL
observed_valence: high_negative
assignment_confidence: 0.96
alternative_assignment: null
segmentation_confidence: 0.99
taxonomy_gap: false
in_scope: true
weight: 1.0
```

#### Concept inventory-item

```yaml
concept_id: C-0042
neutral_label: "prijsniveau"
definition: "Waarnemingen of evaluaties over de hoogte van de prijs."
domain_id: D-PRICE
facet_id: F-AFFORDABILITY
attribute_id: A-PRICE_LEVEL
idea_unit_ids: [IU-000001, IU-000117]
n_iu: 28
n_answers: 27
n_respondents: 26
p_iu: 0.041
p_answers: 0.052
p_respondents: 0.050
weighted_p_respondents: 0.048
observed_valence_distribution:
  high_negative: 23
  low_positive: 3
  neutral: 2
segment_prevalence: {}
representative_examples: [IU-000001, IU-000117, IU-000203]
boundary_examples: [IU-000411]
rare_candidate: false
segment_concentrated: false
protected_exception: false
uncertainty_notes: null
```

### 1.7 Kwaliteitschecks

- **Traceerbaarheid:** 100% van conceptoccurrences verwijst naar een idea unit en bronspan.
- **Segmentatiedekking:** alle niet-lege antwoorden zijn verwerkt; uitgesloten antwoorden hebben een reden.
- **Neutraliteit:** conceptlabels en definities bevatten geen evaluatieve richting, behalve wanneer de eigenschap zelf lexicaal onvermijdelijk evaluatief is; herformuleer dan naar een neutrale dimensie.
- **Taxonomie-integriteit:** elk gebruikt ID bestaat of is expliciet als gap gemarkeerd.
- **Tellingen:** `n_resp ≤ n_ans ≤ n_iu` wanneer één antwoord per respondent geldt; afwijkingen worden verklaard.
- **Geen hallucinated coverage:** ieder concept heeft bronbewijs.
- **Stabiliteit:** heranalyseer een gestratificeerde steekproef; nieuwe relevante concepten mogen slechts marginaal optreden. Bij veel nieuwe concepten is de inventaris onvolledig.

### 1.8 Edge cases

- **Zeer korte antwoorden:** “duur” kan met vraagcontext een volledige unit zijn.
- **Ontkenning:** “niet duur” is niet gelijk aan “goedkoop”; behoud de precieze richting.
- **Gemengde valentie:** “duur maar het waard” bevat prijsniveau én prijs-waardeverhouding; segmenteer indien mogelijk.
- **Ironie/sarcasme:** markeer lage zekerheid en review.
- **Voorwaardelijke uitspraken:** behoud conditie, bijvoorbeeld “goed als de prijs daalt”.
- **Vergelijkingen:** leg referentiepunt vast: duurder dan concurrent, groter dan voorheen.
- **Meertaligheid:** groepeer op betekenis, bewaar originele tekst en vertaalversie afzonderlijk.
- **Rare segment signal:** behoud als consolidatiekandidaat met segmentbewijs.
- **Niet-inhoudelijke antwoorden:** gebruik statusvelden, geen inhoudelijke code.

---

## 2. Consolidation: MERGE / KEEP / SPLIT / DROP

### 2.1 Doel

Transformeer de conceptinventaris naar de kleinst mogelijke set inhoudelijke codekandidaten die relevante empirische variatie behoudt. Iedere conceptkandidaat krijgt precies één primaire beslissing: `MERGE`, `KEEP`, `SPLIT` of `DROP`.

### 2.2 Inputs

- gevalideerde Empirical Concept Inventory;
- prevalentietellingen en segmentverdelingen;
- onderzoeksdoel en beschermde uitzonderingen;
- representatieve voorbeelden en grensgevallen;
- vastgelegde drempels;
- eventueel een zachte doelrange voor het aantal codes.

### 2.3 Outputs

1. een consolidation decision table;
2. geconsolideerde, nog valentie-neutrale codekandidaten;
3. mappings van elk concept en iedere idea unit naar een kandidaat of uitsluitingsstatus;
4. merge-, split- en drop-rationales;
5. een overlap- en dekkingsrapport;
6. een lijst voor menselijke review bij lage zekerheid.

### 2.4 Beslisvolgorde

Pas onderstaande volgorde toe om codegroei te beperken:

1. **scope en bruikbaarheid controleren**;
2. **DROP beoordelen**;
3. **MERGE maximaal verantwoord toepassen**;
4. **KEEP alleen wanneer zelfstandigheid aantoonbaar waarde heeft**;
5. **SPLIT alleen wanneer noodzakelijk en empirisch gedragen**;
6. volledige set opnieuw testen op overlap en dekking.

Een zachte maximumomvang mag nooit leiden tot verlies van beschermde of duidelijk beslisrelevante betekenis. Als de doelomvang niet haalbaar is, rapporteer waarom.

### 2.5 Beslisregels per actie

#### DROP

DROP een concept als één of meer van onderstaande situaties geldt en geen beschermde uitzondering geldt:

- buiten scope of niet relevant voor de onderzoeksvraag;
- geen interpreteerbare inhoud, spam of louter procedurele tekst;
- artefact van verkeerde segmentatie of taxonomietoewijzing;
- semantisch redundant en beter volledig onder een ander concept te brengen — gebruik dan bij voorkeur MERGE voor traceerbaarheid;
- onder `T_keep`, zonder unieke actie-, segment- of informatiewaarde, en passend onder een brede rest- of bovenliggende code;
- uitsluitend theoretisch mogelijk en niet empirisch aangetroffen.

DROP betekent niet dat brondata wordt verwijderd. Bewaar mappings en `drop_reason`. Zeldzame betekenis die onder een bredere code past wordt doorgaans MERGE, niet DROP.

#### MERGE

MERGE concepten wanneer zij voor het coderingsdoel niet betrouwbaar of nuttig te onderscheiden zijn. Vereist:

- dezelfde of nagenoeg dezelfde operationele betekenis; **of**
- een ouder-kindrelatie waarbij het kind onvoldoende prevalent of beslisrelevant is voor een zelfstandige code; **of**
- verschillende formuleringen leiden tot dezelfde waarschijnlijke analytische actie;
- samengevoegde definitie blijft coherent en grensgevallen kunnen eenduidig worden toegewezen.

Voer een merge niet uit als daardoor tegengestelde inhoudelijke attributen, verschillende handelingsimplicaties of beschermde segmentverschillen verdwijnen. Valentieverschillen alleen blokkeren een merge in deze stap niet; die worden in stap 3 opgelost.

Bij twijfel hanteert de agent de **compressievoorkeur**: merge tenzij een aantoonbare meerwaarde van onderscheid bestaat.

#### KEEP

KEEP een concept zelfstandig wanneer:

- het minimaal `T_keep` haalt én een coherente operationele betekenis heeft; of
- het een beschermde uitzondering is; en
- het niet zonder relevant informatieverlies in een andere kandidaat past; en
- de grens met naburige concepten in woorden en voorbeelden uitlegbaar is.

Prevalentie boven de drempel is geen automatisch recht op een aparte code. Veelvoorkomende synoniemen of operationeel equivalente concepten moeten alsnog worden gemerged.

#### SPLIT

SPLIT alleen wanneer een kandidaat aantoonbaar meerdere niet-overlappende betekenissen bevat die afzonderlijk nuttig zijn. Vereist normaal gesproken:

- identificeerbare semantische subclusters met verschillende definities of handelingsimplicaties;
- elke dochter haalt `T_split_child`, of heeft een beschermde uitzondering;
- menselijke of modelmatige codering kan de dochters betrouwbaar onderscheiden;
- de split verlaagt ambiguïteit of verhoogt analytische relevantie substantieel;
- de dochters zijn gezamenlijk uitputtend voor de parent-occurrences, eventueel met een tijdelijke restcategorie.

Splits niet alleen omdat de taxonomie meer detail toestaat. Splits niet op valentie; valentie-afhandeling is stap 3.

### 2.6 Prevalentie zwaar laten meewegen

Bereken voor iedere kandidaat een beslissingsprofiel. Gebruik geen ondoorzichtige totaalscore als enige beslisser, maar eventueel deze transparante prioritering:

1. beschermde uitzondering;
2. scope/relevantie;
3. respondentprevalentie;
4. unieke informatiewaarde ten opzichte van naburige concepten;
5. segmentconcentratie;
6. codeerbaarheid en grenshelderheid;
7. complexiteitskosten van een extra code.

Voor elk voorgesteld onderscheid moet de agent expliciet beantwoorden:

> “Welke relevante analyse, beslissing of interpretatie wordt slechter als deze aparte code niet bestaat?”

Als daar geen concreet antwoord op is, MERGE of DROP. Voor elke split moet bovendien worden aangetoond dat beide of alle dochters empirisch voldoende voorkomen.

### 2.7 Procedure

1. Maak een semantische nabijheidsmatrix of kandidaatgroepen binnen en over taxonomietakken.
2. Identificeer exacte duplicaten, synoniemen, ouder-kindrelaties en nabije handelingscategorieën.
3. Beoordeel zeldzame concepten eerst op beschermde uitzonderingen en segmentconcentratie.
4. Ken iedere conceptkandidaat een voorlopige actie toe met rationale en bewijs.
5. Bouw mergegroepen en formuleer per groep één neutrale kandidaatdefinitie.
6. Test of kandidaatdefinities alle gemapte units dekken zonder incoherente betekenis.
7. Onderzoek alleen daarna brede of heterogene kandidaten op noodzakelijke splits.
8. Map alle idea units opnieuw naar de geconsolideerde kandidaten.
9. Bereken prevalentie opnieuw na merges/splits.
10. Voer een pairwise overlapcheck en een ongecodeerde-unitscheck uit.
11. Herhaal alleen waar checks aantoonbare problemen vinden.

### 2.8 Aanbevolen datastructuren

#### Consolidation decision

```yaml
decision_id: CD-0021
source_concept_ids: [C-0042, C-0108]
action: MERGE
target_candidate_ids: [K-0014]
rationale: "Beide concepten beschrijven prijsniveau; onderscheid levert geen andere analyse of actie op."
prevalence_before:
  C-0042: {n_respondents: 26, p_respondents: 0.050}
  C-0108: {n_respondents: 4, p_respondents: 0.008}
prevalence_after:
  K-0014: {n_respondents: 30, p_respondents: 0.058}
information_loss_assessment: low
protected_exception: false
review_status: approved
confidence: 0.94
```

#### Neutrale codekandidaat

```yaml
candidate_id: K-0014
neutral_label: "prijsniveau"
definition: "Uitspraken over de hoogte van de prijs."
included_concept_ids: [C-0042, C-0108]
included_idea_unit_ids: []
n_respondents: 30
p_respondents: 0.058
inclusion_rule: "Codeer wanneer de unit de prijs als hoog, laag of anderszins positioneert."
exclusion_rule: "Niet gebruiken voor waarde-voor-geld zonder uitspraak over de prijsdimensie."
nearest_neighbors: [K-0015]
boundary_note: "Prijsniveau versus prijs-kwaliteitverhouding."
protected_exception: false
```

### 2.9 Kwaliteitschecks

- **Volledige besluitvorming:** elk inventarisconcept heeft exact één primaire consolidatiebeslissing.
- **Herleidbaarheid:** elke kandidaat kan terug naar concepten, units en bronspans.
- **Parsimonietest:** iedere zelfstandige kandidaat heeft een expliciete bestaansreden.
- **Split-test:** alle niet-beschermde dochters halen `T_split_child`.
- **Merge-coherentie:** steekproef van gemergede units past zonder semantische rek in de nieuwe definitie.
- **Pairwise exclusiviteit:** voor elk naburig kandidatenpaar is een grensregel beschikbaar.
- **Dekking:** alle relevante idea units zijn gemapt of hebben een expliciete DROP/review-status.
- **Segmentfairness:** geen vooraf relevant segment verliest systematisch een betekenis door aggregatie.
- **Robuustheid:** vergelijk oplossingen bij iets hogere/l lagere drempel; grote instabiliteit vereist review.

### 2.10 Edge cases

- **Veel zeldzame concepten:** merge naar coherente bovenliggende codes; maak niet één betekenisloze “overig”-bak.
- **Dominant concept:** houd breed als detail geen andere beslissing ondersteunt; split alleen bij prevalente, bruikbare subtypen.
- **Kleine steekproef:** absolute aantallen, voorbeelden en dekkingsverlies wegen zwaarder dan percentages.
- **Scheve segmenten:** rapporteer totaal én binnen-segment; bescherm alleen vooraf relevante segmenten.
- **Causale ketens:** oorzaak en gevolg zijn verschillende betekenissen wanneer beide expliciet worden genoemd; segmenteer eerst.
- **Taxonomie-overlap:** de operationele codegrens heeft voorrang; documenteer taxonomiecrosswalk.
- **Overig groeit te groot:** als een restcode meer dan circa 5–10% van relevante units bevat of intern duidelijk heterogeen is, heropen inventaris/consolidatie.

---

## 3. Valence Resolution

### 3.1 Doel

Bepaal per geconsolideerde, neutrale kandidaat of en hoe valentie in de uiteindelijke code(s) wordt verwerkt. Voeg alleen empirisch ondersteunde en analytisch nuttige richtingen toe. Genereer nooit automatisch symmetrische positieve en negatieve codes.

### 3.2 Inputs

- geconsolideerde neutrale kandidaten en mappings;
- idea units met voorlopig waargenomen valentie;
- prevalentie per valentiepool, totaal en per segment;
- definities van relevante valentieschalen;
- onderzoeksvraag en beschermde uitzonderingen.

### 3.3 Outputs

1. een valence decision per neutrale kandidaat;
2. uiteindelijke valentie-ingevulde codekandidaten;
3. regels voor neutrale, gemengde, ambigue en contextafhankelijke gevallen;
4. mappings van iedere relevante idea unit naar een eindcodekandidaat;
5. prevalentietellingen per eindcode;
6. auditlog van niet-aangemaakte theoretische tegenpolen.

### 3.4 Toegestane valentiestrategieën

Kies per kandidaat exact één primaire strategie:

1. **VALENCE_SPLIT:** maak aparte codes voor empirisch sterke richtingen, bijvoorbeeld “prijs te hoog” en “prijs laag/aantrekkelijk”.
2. **DOMINANT_VALENCE_CODE:** maak alleen de empirisch relevante richting als code; maak geen lege of zeer zeldzame spiegelcode.
3. **VALENCE_INCLUSION:** gebruik één brede inhoudscode waarin richting via een apart veld wordt opgeslagen; geschikt wanneer valentie analytisch secundair is of afzonderlijke richtingen te klein zijn.
4. **NON_EVALUATIVE_CODE:** gebruik één code wanneer units feitelijk descriptief zijn en geen evaluatieve richting nodig is.
5. **POLARITY_AS_CONTENT:** gebruik richting-specifieke codes wanneer hoog/laag, aanwezig/ontbrekend of makkelijk/moeilijk de operationele betekenis bepaalt en voldoende voorkomt.

### 3.5 Beslisregels

#### Maak een zelfstandige valentiecode wanneer

- de richting `T_keep` haalt of beschermd is;
- de richting een andere relevante interpretatie of actie impliceert;
- units betrouwbaar aan die richting kunnen worden toegewezen;
- de code voldoende homogeen blijft.

#### Maak geen zelfstandige valentiecode wanneer

- de richting alleen theoretisch mogelijk is maar niet voorkomt;
- zij onder `T_keep` blijft, niet beschermd is en zonder relevant verlies kan worden geabsorbeerd;
- valentie vooral toon of intensiteit is en niet de inhoudelijke beslissing verandert;
- positieve en negatieve voorbeelden niet betrouwbaar van neutrale beschrijvingen zijn te scheiden.

#### Geen geforceerde symmetrie

Als “te duur” 18% en “goedkoop/aantrekkelijk geprijsd” 0,4% voorkomt, maak standaard alleen “prijs te hoog” als zelfstandige code. Behandel de zeldzame tegenpool via een bredere code, restprocedure of apart valentieveld, tenzij zij beschermd of analytisch essentieel is.

#### Neutraal en afwezig zijn verschillend

- `neutral`: de unit positioneert het attribuut zonder duidelijke evaluatie, bijvoorbeeld “de prijs is €10”.
- `not_mentioned`: het attribuut komt niet voor; dit is geen code en geen valentie.
- `ambiguous`: richting kan niet betrouwbaar worden vastgesteld.

#### Intensiteit

Maak geen aparte codes voor “duur” en “extreem duur” tenzij intensiteit prevalent, betrouwbaar codeerbaar en beslisrelevant is. Sla intensiteit anders op als optioneel veld.

### 3.6 Procedure

1. Definieer per attribuut de empirisch passende valentie-as; neem niet aan dat positief/negatief altijd volstaat.
2. Classificeer iedere gemapte idea unit als een concrete pool, neutraal, gemengd of ambigu.
3. Bereken per pool `n_iu`, `n_ans`, `n_resp`, percentages en segmentverdeling.
4. Controleer ontkenning, referentiepunt en intensiteit.
5. Kies één valentiestrategie per neutrale kandidaat.
6. Pas `T_keep` en uitzonderingsregels toe op iedere mogelijke eindcode.
7. Formuleer alleen empirisch ondersteunde codekandidaten.
8. Map units naar eindcodekandidaten en beoordeel ongecodeerde of dubbel gemapte units.
9. Test naburige richtingen op exclusiviteit met grensgevallen.
10. Leg expliciet vast welke theoretische polen niet als code zijn aangemaakt en waarom.

### 3.7 Aanbevolen datastructuren

#### Valence decision

```yaml
valence_decision_id: VD-0014
candidate_id: K-0014
valence_axis: "ervaren prijsniveau"
observed_poles:
  too_high_negative: {n_respondents: 23, p_respondents: 0.044}
  low_positive: {n_respondents: 3, p_respondents: 0.006}
  neutral: {n_respondents: 2, p_respondents: 0.004}
strategy: DOMINANT_VALENCE_CODE
resulting_code_ids: [CODE-012]
omitted_poles:
  - pole: low_positive
    reason: "Onder prevalentiedrempel, niet beschermd en geen zelfstandige besliswaarde."
ambiguous_handling: review_then_broader_code
confidence: 0.95
```

#### Eindcodekandidaat

```yaml
code_id: CODE-012
label: "Prijs te hoog"
neutral_parent_candidate_id: K-0014
domain_id: D-PRICE
facet_id: F-AFFORDABILITY
attribute_id: A-PRICE_LEVEL
valence: too_high_negative
definition: "De respondent beoordeelt of ervaart de prijs als hoog of te hoog."
n_respondents: 23
p_respondents: 0.044
protected_exception: false
```

### 3.8 Kwaliteitschecks

- **Empirische steun:** iedere eindcode heeft minimaal één occurrence en normaal gesproken `T_keep` of een beschermde uitzondering.
- **Geen lege spiegelcodes:** geen code uitsluitend voor taxonomische symmetrie.
- **Richtingstraceerbaarheid:** valentie blijkt uit de tekst/context, niet uit aannames over wat wenselijk is.
- **Exclusiviteit:** één enkelvoudige unit kan niet tegelijk twee tegengestelde polen krijgen.
- **Gemengde units:** zijn verder gesegmenteerd of expliciet als uitzondering behandeld.
- **Neutraliteitsbewaking:** `domain`, `facet` en `attribute` blijven neutraal; evaluatie staat in `valence`, codelabel en operationele definitie.
- **Prevalentie na resolutie:** tellingen zijn opnieuw berekend, niet afgeleid door simpele optelling met dubbeltellingen.
- **Omitted-pole audit:** iedere niet-aangemaakte waargenomen pool heeft een reden.

### 3.9 Edge cases

- **“Niet slecht”:** niet automatisch gelijk aan positief; codeer volgens context als zwak positief, neutraal of ambigu.
- **“Duur maar de moeite waard”:** segmenteer naar prijsniveau negatief en waarde-voor-geld positief als beide betekenissen zelfstandig zijn.
- **Optimale middenwaarde:** bij attributen waar zowel te hoog als te laag negatief kan zijn, gebruik een passende as; forceer geen lineaire positief-negatief schaal.
- **Contextuele richting:** “groot” kan positief of negatief zijn afhankelijk van product en behoefte. Leg evaluatie apart vast.
- **Impliciete valentie:** alleen toewijzen bij conventioneel en contextueel voldoende duidelijke formuleringen; anders ambigu.
- **Wensen en tekorten:** “ik wil meer keuze” impliceert mogelijk onvoldoende keuze; definieer een expliciete inferentieregel en pas die consistent toe.
- **Absentie:** “geen klantenservice” is aanwezigheid/ontbreken en vaak negatief; onderscheid inhoudelijke pool van evaluatieve toon.

---

## 4. Final Codebook

### 4.1 Doel

Produceer een compact, duidelijk, reproduceerbaar en versieerbaar eindcodeboek waarmee menselijke of automatische codeurs relevante idea units consistent kunnen coderen, plus een complete crosswalk en kwaliteitsrapportage.

### 4.2 Inputs

- gevalideerde eindcodekandidaten uit stap 3;
- idea-unit-to-code mappings;
- prevalentie en segmentresultaten;
- alle definities, voorbeelden en grensregels;
- consolidation- en valence-auditlogs;
- resultaten van pilotcodering of modelvalidatie, indien beschikbaar.

### 4.3 Outputs

Minimaal:

1. een machineleesbaar codeboek (`CSV`, `JSON` of `YAML`);
2. een mensleesbaar codeboek (`Markdown` of document);
3. code-instructies op idea-unitniveau;
4. taxonomie-naar-codeboek-crosswalk;
5. excluded/dropped register;
6. kwaliteits- en prevalentierapport;
7. versie, datum, datasetfingerprint en changelog.

### 4.4 Verplichte velden per code

- stabiele `code_id`;
- kort, uniek `label`;
- operationele `definition`;
- `domain_id`, `facet_id`, `attribute_id`, `valence`;
- inclusiecriteria;
- exclusiecriteria;
- dichtstbijzijnde concurrerende codes en beslisregels;
- minimaal twee positieve voorbeelden indien beschikbaar;
- minimaal één negatief voorbeeld of grensgeval indien beschikbaar;
- prevalentie: `n_iu`, `n_ans`, `n_resp` en percentages;
- eventueel gewogen en segmentprevalentie;
- `protected_exception` plus reden;
- confidence/reviewstatus;
- provenance naar kandidaat-, concept- en beslissing-ID's.

### 4.5 Codeerprocedure voor gebruik van het codeboek

1. Lees de surveyvraag en het volledige antwoord voor context.
2. Segmenteer het antwoord in idea units volgens de vastgelegde regels.
3. Bepaal voor iedere unit de inhoudelijke kern.
4. Vergelijk eerst met inclusiecriteria en daarna met exclusie- en grensregels.
5. Kies de meest specifieke bestaande code die empirisch in het codeboek is toegestaan; creëer tijdens productiecodering niet ad hoc een nieuwe code.
6. Ken in beginsel één code per idea unit toe.
7. Ken meerdere codes aan één antwoord toe wanneer het meerdere idea units bevat.
8. Gebruik `REVIEW` bij echte ambiguïteit en leg kandidaatcodes vast.
9. Gebruik een restcode alleen conform de gecontroleerde regels hieronder.

### 4.6 Rest- en statuscodes

Houd inhoudelijke codes gescheiden van workflow/statuscodes. Aanbevolen statuswaarden:

- `NO_ANSWER`: leeg of ontbrekend;
- `UNINTELLIGIBLE`: niet interpreteerbaar;
- `OUT_OF_SCOPE`: begrijpelijk maar buiten de vraag;
- `REVIEW`: tijdelijke menselijke beoordeling nodig;
- `OTHER_RELEVANT`: relevante betekenis zonder passende code.

`OTHER_RELEVANT` is een veiligheidsklep, geen standaardcode. Vereis een tekstnotitie. Monitor de inhoud en prevalentie. Heropen het codeboek wanneer:

- `OTHER_RELEVANT` circa 5% of meer van relevante idea units bereikt; of
- een coherent nieuw concept `T_keep` bereikt; of
- een beschermd signaal ontstaat.

### 4.7 Aanbevolen datastructuur

```yaml
codebook_metadata:
  codebook_id: CB-2026-001
  version: 1.0.0
  created_at: 2026-08-12
  language: nl
  unit_of_analysis: idea_unit
  multicoding_at_response_level: allowed
  default_codes_per_idea_unit: 1
  taxonomy_version: TAX-003
  dataset_fingerprint: "sha256:..."
  thresholds:
    keep: "max(3 respondents, 1%)"
    split_child: "max(3 respondents, 1%)"

codes:
  - code_id: CODE-012
    label: "Prijs te hoog"
    definition: "De respondent beoordeelt of ervaart de prijs als hoog of te hoog."
    taxonomy:
      domain_id: D-PRICE
      facet_id: F-AFFORDABILITY
      attribute_id: A-PRICE_LEVEL
      valence: too_high_negative
    include_when:
      - "De prijs expliciet hoog, duur of te hoog wordt genoemd."
      - "Een prijsverhoging als onwenselijk wordt ervaren."
    exclude_when:
      - "Alleen waarde-voor-geld wordt besproken zonder oordeel over prijsniveau."
      - "Alleen een feitelijk bedrag wordt genoemd zonder positionering."
    nearest_code_boundaries:
      - other_code_id: CODE-013
        rule: "Gebruik CODE-013 voor prijs-kwaliteitverhouding; CODE-012 voor prijsniveau."
    positive_examples:
      - "Veel te duur."
      - "De prijs ligt te hoog."
    negative_examples:
      - text: "Goede prijs voor wat je krijgt."
        expected_code: CODE-013
    prevalence:
      n_idea_units: 24
      n_answers: 23
      n_respondents: 23
      p_respondents: 0.044
    protected_exception: false
    provenance:
      candidate_ids: [K-0014]
      concept_ids: [C-0042, C-0108]
      decision_ids: [CD-0021, VD-0014]
    review_status: approved
```

### 4.8 Finale MECE-check

MECE wordt operationeel, niet absoluut, beoordeeld.

#### Mutually exclusive op idea-unitniveau

1. Vergelijk ieder paar semantisch nabije codes.
2. Test grensgevallen blind: onafhankelijke codeurs of runs moeten dezelfde code kiezen.
3. Voeg een expliciete prioriteitsregel toe waar definities raken.
4. Merge codes als onderscheid structureel onbetrouwbaar is en geen kritieke waarde heeft.
5. Splits de idea unit verder als ogenschijnlijke overlap door samengestelde tekst komt.

#### Collectively exhaustive op idea-unitniveau

1. Codeer een verse, niet voor constructie gebruikte steekproef.
2. Meet aandeel relevante units zonder code en aandeel `OTHER_RELEVANT`.
3. Analyseer ongecodeerde units op coherente clusters.
4. Voeg alleen een code toe als het cluster drempel/uitzonderingsregels haalt; anders houd gecontroleerd in de restcategorie.

#### Multi-code op antwoordniveau

Meerdere codes in één antwoord zijn verwacht en vormen geen MECE-schending wanneer zij aan verschillende idea units zijn toegekend. Rapporteer daarom zowel unit- als antwoordprevalentie.

### 4.9 Betrouwbaarheids- en validatiechecks

Voer, proportioneel aan risico en omvang, minimaal uit:

- **Blind pilot:** codeer een gestratificeerde hold-outsteekproef die veelvoorkomende, zeldzame en grensgevallen bevat.
- **Intercoder/model agreement:** rapporteer per code en totaal; gebruik bij voorkeur Cohen's kappa voor twee codeurs of Krippendorff's alpha bij passende opzet, naast raw agreement.
- **Per-code precision/recall:** wanneer een gouden standaard beschikbaar is.
- **Confusiematrix:** inspecteer vooral naburige codes.
- **Dekking:** percentage relevante idea units met een inhoudelijke code.
- **Restcodepercentage:** totaal en per segment.
- **Prevalentiestabiliteit:** bootstrap of split-halfvergelijking bij voldoende data.
- **Parsimonie-audit:** laat voor iedere code opnieuw de bestaansreden controleren.
- **Valentie-audit:** controleer dat er geen spiegelcodes zonder empirische steun bestaan.

Aanbevolen acceptatiecriteria, tenzij vooraf anders bepaald:

- minimaal 95% van relevante idea units krijgt een inhoudelijke code of een verklaarde `OTHER_RELEVANT` status;
- `OTHER_RELEVANT` blijft bij voorkeur onder 5%;
- geen onverklaarde overlap voor hoogfrequente codeparen;
- minimaal 80% raw agreement op hold-out en een passende chance-corrected maat die geen structureel probleem aanwijst;
- alle beschermde uitzonderingen zijn aantoonbaar vindbaar;
- 100% van codes heeft inclusie-, exclusie- en provenancevelden.

Zie drempels als signalen voor review, niet als garanties. Bij zeldzame codes kan kappa instabiel zijn; beoordeel dan voorbeelden, precision/recall en prevalentie mee.

### 4.10 Finale parsimoniecheck

Voor iedere code:

1. noteer prevalentie;
2. benoem unieke betekenis en verwachte analytische toepassing;
3. identificeer de beste mergekandidaat;
4. simuleer het informatieverlies van die merge;
5. verwijder of merge de code als het verlies gering en niet beschermd is.

Bereken aanvullend:

- totaal aantal inhoudelijke codes;
- codes onder `T_keep` en hun uitzonderingsredenen;
- aandeel units in de top 5, top 10 en top 20 codes;
- long-tail-aandeel;
- aantal codes dat samen de eerste 80%, 90% en 95% van occurrences dekt.

Een compact codeboek heeft niet noodzakelijk een vooraf bepaald aantal codes. Het optimale aantal is het minimum waarbij dekking, grenshelderheid en relevante informatiewaarde aanvaardbaar blijven.

### 4.11 Stopcriteria

Het codeboek is gereed wanneer:

- alle vier stappen en auditlogs compleet zijn;
- alle relevante units in de constructie- en hold-outdata codeerbaar zijn binnen de afgesproken dekking;
- resterende overlap alleen verklaarde uitzonderingen betreft;
- alle lage-prevalentiecodes beschermd of overtuigend gemotiveerd zijn;
- geen empirisch lege of puur symmetrische valentiecodes bestaan;
- pilotresultaten voldoen of afwijkingen expliciet zijn geaccepteerd;
- codeboek, crosswalk, mappings en changelog versieerbaar zijn opgeleverd.

### 4.12 Onderhoud en versiebeheer

- Gebruik semantic versioning: patch voor tekstuele verduidelijking, minor voor compatibele codewijzigingen, major voor structurele herziening.
- Wijzig bestaande `code_id`'s niet; retireer codes met status en opvolger.
- Bewaar een migratiematrix van oude naar nieuwe codes.
- Herkalibreer bij nieuwe datagolf, nieuwe populatie, gewijzigde vraag of groeiende `OTHER_RELEVANT`.
- Vergelijk prevalentie over versies alleen via een expliciete crosswalk.

---

## 5. Agentuitvoering: verplicht beslislog en opleverpakket

Hoewel het inhoudelijke proces exact uit de vier bovengenoemde stappen bestaat, gelden onderstaande uitvoeringsvereisten over alle stappen heen.

### 5.1 Verplicht beslislog

Iedere niet-triviale beslissing bevat:

- beslissing-ID;
- stapnummer;
- betrokken bron-, concept-, kandidaat- en/of code-ID's;
- actie;
- prevalentie vóór en na de actie;
- gebruikte regel;
- korte inhoudelijke rationale;
- geschat informatieverlies;
- beschermde-uitzonderingsstatus;
- confidence;
- reviewstatus en reviewer indien van toepassing;
- timestamp en proces/modelversie.

### 5.2 Verplicht opleverpakket

```text
deliverable/
├── 01_idea_units.csv
├── 02_empirical_concept_inventory.csv
├── 03_consolidation_decisions.csv
├── 04_valence_decisions.csv
├── 05_final_codebook.yaml
├── 05_final_codebook.md
├── 06_taxonomy_codebook_crosswalk.csv
├── 07_unit_code_mappings.csv
├── 08_exclusions_and_other.csv
├── 09_quality_report.md
├── 10_decision_log.jsonl
└── CHANGELOG.md
```

### 5.3 Verboden shortcuts

De agent mag niet:

- de taxonomie één-op-één als codeboek overnemen;
- codes genereren voor niet-waargenomen taxonomiecombinaties;
- automatisch positieve en negatieve spiegelcodes maken;
- alleen percentages gebruiken zonder absolute aantallen;
- prevalentie tellen zonder respondent- of antwoorddeduplicatie;
- zeldzame concepten automatisch verwijderen zonder uitzonderingscheck;
- een split uitvoeren zonder prevalentie per dochter;
- overlap oplossen door willekeurige toewijzing zonder grensregel;
- één code per antwoord afdwingen;
- bronspans, mappings of afgewezen concepten verwijderen;
- een grote heterogene restcategorie als bewijs van uitputtendheid beschouwen.

### 5.4 Compact algoritme

```text
INPUT: responses, survey context, taxonomy, research goal, thresholds

STEP 1 — EMPIRICAL CONCEPT INVENTORY
  preserve raw responses
  segment each response into idea units
  assign neutral domain/facet/attribute; record observed valence separately
  form only empirically evidenced neutral concepts
  calculate unit-, answer-, respondent- and segment prevalence
  attach examples, boundaries, uncertainty and provenance
  validate completeness and neutrality

STEP 2 — CONSOLIDATION
  for every concept:
    check scope and protected exceptions
    consider DROP
    compare with neighbors and prefer justified MERGE
    KEEP only with prevalence/unique relevance
    SPLIT only for distinct, codeable, prevalent children
  remap units and recalculate prevalence
  test overlap, coverage, segment loss and parsimony

STEP 3 — VALENCE RESOLUTION
  identify empirical valence axis and poles per candidate
  calculate prevalence per pole
  choose one strategy per candidate
  create only supported, useful valenced codes
  do not create theoretical mirror codes
  remap units; test direction boundaries and omitted poles

STEP 4 — FINAL CODEBOOK
  write operational definitions, inclusion/exclusion and boundary rules
  attach examples, prevalence and provenance
  pilot on held-out data
  assess idea-unit MECE, multi-code response behavior, reliability and coverage
  compress again where information loss is acceptably low
  version and export the complete deliverable package

OUTPUT: smallest defensible codebook with traceable empirical coverage
```

## 6. Eindprincipe

De leidende optimalisatie is:

> **Minimaliseer het aantal codes, onder de voorwaarden dat relevante empirische betekenis, betrouwbare codeerbaarheid, beschermde signalen en voldoende idea-unitdekking behouden blijven.**

Prevalentie bepaalt daarbij zwaar welke verschillen een zelfstandige code verdienen. De taxonomie biedt structuur, maar niet ieder taxonomie-item wordt een code. Valentie wordt pas na inhoudelijke consolidatie opgelost. MECE wordt primair beoordeeld per idea unit; multi-coding van een antwoord is correct wanneer het antwoord meerdere idea units bevat.
