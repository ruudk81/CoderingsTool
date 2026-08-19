# Adhoc SPSS-levering — experiment

**Status:** experiment, gedraaid en geverifieerd op `M260502 … met Qd1` / `Qd1` / `full`
(2026-08-19). Nog niet in de pijplijn, niet in de app, geen tests, geen dev-docs.
Raakt geen productiecode aan: leest step 7's output, schrijft naar `exports/adhoc/`.

## Waarom dit bestaat

Step 7 levert analysebestanden: één gedeelde nummering, leesbare kolomnamen, alles in
één bestand. Dat is goed om mee te werken, maar het is niet wat er in een tabellenboek
gaat. Daarvoor moet elk bestand de Motivaction-conventies volgen, moeten de koppen erin
zitten, en moet het per meting gesplitst zijn. Dit script doet dat zonder step 7 te
veranderen.

## Draaien

```bash
cd src && python -m pipeline.step_7_export.adhoc.adhoc_export
```

Welke run hij pakt volgt uit `test_data.py` (bestand + variabele + sample_size) — niet
uit "het nieuwste bestand op schijf", want dan pakt hij na een halve run het verkeerde.
Ontbreekt er een bestand van die run, dan stopt hij met de melding welk.

**Step 7 moet eerst gedraaid hebben**, en zijn output moet ongewijzigd zijn. Het script
leidt af welke kolom welke code is uit de catalogus in de cache; wijken de kolomnamen af
van wat step 7 schrijft, dan stopt hij (dat is één keer gebeurd, op een handmatig
hernoemd codeboek). Step 7 opnieuw draaien kost twee seconden en geen LLM-aanroepen.

## Wat er uit komt — `exports/adhoc/`

Negen `.sav`'s: per output het geheel plus de twee metingen.

| | rijen |
|---|---|
| `{base}_codeboek.sav` / `_2025` / `_2026` | 4586 / 2127 / 2459 |
| `{base}_taxonomie.sav` / `_2025` / `_2026` | 4586 / 2127 / 2459 |
| `{base}_gecombineerd.sav` / `_2025` / `_2026` | 6535 / 2976 / 3559 |

Plus vier bestanden die ongewijzigd worden gekopieerd: `{base}_codering.xlsx`,
`{base}_codeboek.xlsx` en twee CSV's. Die worden niet gesplitst — codeboek en legenda
zijn per definitie gedeeld over beide metingen. Eén taxonomie, twee metingen.

`exports/adhoc/` staat in `retention.py` al expliciet als niet-beheerde map, dus de
opruimer laat hem met rust.

## De conventies

Afgelezen aan de 149 variabelen in het bronbestand, niet aan een handboek. De prefix
zegt wat voor soort variabele het is:

```
m   dichotome set (meerkeuze)        n   enkelvoudig categorisch
x   tekst, id, ruwe variabelen       w   weegfactor
```

### m — codeboek en taxonomie

**Naam** `m<VraagID><LAAG>_<nummer>`; het volgnummer is altijd het laatste token.
Lagen: `COD` (24 codes + 3 filtercategorieën), `DOM` (8), `FAC` (18), `ATT` (49).
Filtercategorieën houden hun 8-cijferige code: `mQd1COD_99999997`.

**De laag zit vast aan de vraag-id, zonder underscore.** Dat is geen cosmetiek: de
tabelleringssoftware groepeert op de stam vóór het laatste `_<nummer>`, dus met
`mQd1_dom_1` en `mQd1_att_1` belanden alle 75 taxonomievariabelen in één tabel. Met
`mQd1DOM_1` / `mQd1FAC_1` / `mQd1ATT_1` zijn het er drie. Gemeten in SPSS op 2026-08-19,
nadat de eerste versie één tabel opleverde.

**Label** `[]<basis>[][]<VraagID><LAAG>_<n> <kwalificatie><vraagtekst> <antwoord>`.
Zonder basis blijft er één leeg bracket-paar over, zoals in het bronbestand.

**Waarden** `{0: 'Niet Genoemd', 1: 'Wel Genoemd', 99999999: 'Missing'}`, `F8.2`,
nominaal. Dichotoom, dus 0 is een echte waarneming en geen ontbrekende waarde.
Niet-gestelde vragen blijven system-missing; de waarde 99999999 wordt niet
weggeschreven (zo staat het ook in de bron — het label bestaat, de waarde niet).

### n — koppen en het lange bestand

De banner wordt hier gebouwd: acht bronvariabelen worden `nKOP1` … `nKOP8` met een
kopnaam als label (`A. METING`, `B. GESLACHT`, …). Waarden en waardelabels gaan
ongewijzigd mee en zijn categorisch en 1-based — anders dan de m-variabelen, die bij 0
beginnen omdat 0 daar "niet genoemd" betekent.

Het bronbestand had zelf maar één `nKOP`-variabele (`A. METING`); die wordt opnieuw
opgebouwd uit `nMeting`, zodat één regel geldt voor alle acht.

In `gecombineerd` zijn `code`, `domain`, `facet`, `attribute` en `valence` enkelvoudig
categorisch — één rij is één idee — dus `nQd1COD`, `nQd1DOM`, `nQd1FAC`, `nQd1ATT`,
`nQd1VAL`, met dezelfde vastgeplakte laag. Geen `m`: er is daar geen dichotome set. De waardelabels die step 7 al zette
(codenamen, domeinnamen, `negatief/neutraal/positief`) blijven staan.

### x en w

`xDLNMID` (step 7 laat de x vallen, hier komt hij terug), `xQd1` voor de open
antwoordtekst, en `xQd1INSTANCE` / `xQd1INTERPRETATIE` / `xQd1ABSTRACTIE` voor de
abstractieladder. `weegvar` houdt zijn eigen naam en prefix.

### Twee bewuste afwijkingen van het bronbestand

1. **Optie 1 houdt zijn volgnummer in het label.** In de bron laat de eerste optie het
   nummer weg (`mQa2_1` heeft label `…Qa2 Bij welke bank…`, niet `Qa2_1`). Dat is een
   artefact van de vragenlijsttool; hier is de reeks consistent.
2. **Het domeinnummer staat niet in de variabelenaam maar in het label**, als
   domeinnaam. Step 7 schrijft `Qd1attr_17_3` (attribuut 17, domein 3); onder de
   m-conventie zou dat lezen als optie 3. Het kan zonder verlies weg: facet- en
   attribuutnummers zijn doorgenummerd over alle domeinen heen en dus al uniek. Het
   label wint erbij — `Attribuut (domein 'klantinteractie en toegang') — … Algemene
   indruk` is leesbaarder dan `Algemene indruk_2` met de legenda ernaast. Zonder dat
   onderscheid zouden hier twee attributen identiek heten.

## Het `PROJECT_SPECIFIEK`-blok

Bovenin `adhoc_export.py` staat één blok met alles wat déze dataset kent: het id in de
bron, de splitvariabele en haar waarden, de acht koppen met hun kopnamen, de extra
bronvariabelen, en de basistekst. De rest van het bestand is mechaniek. Dat blok is de
naad: bij promotie wordt het vervangen, niet de rest.

## Naar productie

1. **De basistekst afleiden in plaats van invullen.** Nu een constante
   (`"Basis - Kent ASN Bank"`). De bron heeft hem zelf: het label van `xQd1_1` begint met
   `[]Basis - Kent ASN Bank[][]`. Generiek: zoek de bronvariabele die bij dezelfde vraag
   hoort en neem daar het eerste bracket-segment uit.
2. **De koppen laten kiezen.** Welke achtergrondvariabelen banner worden is een
   onderzoeksbeslissing, geen afleiding — dit hoort een keuze in de app te worden.
3. **Het afkappen van labels omkeren.** Nu kapt hij af op 256 tekens vanaf het eind — en
   dat eind is juist het antwoordlabel, het enige dat per variabele verschilt. Bij deze
   dataset haalt het langste label 235 tekens, dus het gebeurt nergens; bij een langere
   vraagtekst gaat het stil mis. Productie moet de vraagtekst inkorten, niet het antwoord.
4. **De naamgeving in `exportNaming.py` opnemen.** De meting-achtervoegsels
   (`_codeboek_2025.sav`) staan buiten de doctype-woordenlijst en zijn met de hand
   samengesteld; `parse_export_filename()` kan ze niet teruglezen. Zolang dit experiment
   is, is dat prima — `exports/adhoc/` wordt door niets gescand.
5. **Splitsen optioneel maken.** Niet elke dataset heeft metingen.

## Wat er is gecontroleerd

Teruggelezen uit de weggeschreven `.sav`'s:

- **Rijaantallen** — 2127 + 2459 = 4586 per respondentbestand, 2976 + 3559 = 6535 voor
  gecombineerd; gelijk aan step 7's eigen output, dus er is niets bijgekomen.
- **Kolomtotalen** per variabele identiek aan het originele codeboek en de originele
  taxonomie (27/27 en 75/75) — geen enkele waarde is verschoven.
- **Id** — `xDLNMID` uniek in codeboek en taxonomie (0 duplicaten in alle zes
  bestanden). In `gecombineerd` bewust niet: 6535 rijen over 4578 respondenten, want
  1133 respondenten gaven meer dan één associatie. Dat bestand is dus geen sleutelbestand.
- **Splitsing** — nul respondenten in beide metingen, en 2025 ∪ 2026 is exact de id-set
  van het origineel.
- **Metadata** — waardelabels, format en measure op alle 102 m-variabelen; alle acht
  koppen aanwezig, 1-based, met hun waardelabels uit de bron.
- **Labels** — geen boven 256 tekens (langste 235), geen twee variabelen met hetzelfde
  label.
- **Consistentie** — de 8 respondenten zonder rij in `gecombineerd` zijn exact de 8 die
  in het codeboek nul codes scoren.

Wat **niet** is gecontroleerd: of de bestanden in SPSS zelf schoon openen en of de
multiple-response sets daar naar wens gedefinieerd kunnen worden. Dat is de volgende
stap, en die bepaalt of dit experiment slaagt.
