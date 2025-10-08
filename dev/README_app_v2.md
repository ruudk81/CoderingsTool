# CoderingsTool v2 - Enhanced App

## Overzicht
Een verbeterde versie van de CoderingsTool app met Nederlandse taalondersteuning als standaard en een moderne single-page interface.

## Nieuwe Features

### ✅ Nederlandse Taalondersteuning
- **Standaard Nederlands**: App start in het Nederlands zoals de originele app
- **Vertaalde Interface**: Alle knoppen, labels en statusberichten in het Nederlands
- **Nederlandse Mock Data**: Voorbeelden van Nederlandse enquêteresponsen
- **Taalwisselaar**: Schakel tussen Nederlands en Engels met één klik

### ✅ Verbeterde Gebruikersinterface
- **Single-Page Design**: Alle stappen zichtbaar op één pagina
- **Uitklapbare Stap Kaarten**: Elke pipeline stap heeft eigen kaart
- **Realtime Voortgang**: Live voortgangsbalken en status indicatoren
- **Geïntegreerde Debug Info**: Voorbeelden en statistieken direct zichtbaar

### ✅ Bestaande Configuratie
- **Import ui_text.py**: Gebruikt bestaande Nederlandse vertalingen
- **Import config.py**: Gebruikt bestaande `DEFAULT_LANGUAGE = "Dutch"` configuratie
- **Respecteert Taalsysteem**: Nederlands voor UI, "Dutch" voor LLM verwerking

## Stap Namen (Nederlands)
1. **Data Upload** - SPSS bestand uploaden
2. **Data Laden** - Data uit bestand laden  
3. **Voorbewerking** - Tekst normaliseren en spellingcontrole
4. **Kwaliteitsfiltering** - Lage kwaliteit responsen filteren
5. **Idee Extractie** - Individuele ideeën extraheren
6. **Embeddings** - Embeddings genereren
7. **Clustering** - Vergelijkbare ideeën groeperen
8. **Code Generatie** - Codes voor clusters genereren
9. **Thema Identificatie** - Thema's in codeboek identificeren
10. **Code Toewijzing** - Codes aan ideeën toewijzen
11. **Resultaten Exporteren** - Exporteren naar Excel

## Mock Data Voorbeelden

### Nederlandse Enquête Responsen
- "De service was uitstekend en het personeel zeer vriendelijk"
- "Ik vond de kwaliteit van het eten erg goed, maar de prijs wat hoog"
- "Het restaurant heeft een mooie ambiance en het eten smaakte heerlijk"

### Nederlandse Spellingcorrecties
- "restauraunt" → "restaurant"
- "ik vindt" → "ik vind"
- "servies" → "service"

### Nederlandse Codeboek
- **Service Kwaliteit**: Opmerkingen over de kwaliteit van de ontvangen service
- **Eten Kwaliteit**: Feedback over smaak, presentatie en versheid van het eten
- **Sfeer**: Observaties over restaurant ambiance en omgeving

## Hoe te Gebruiken

```bash
cd src
streamlit run app_v2.py
```

## Verschillen met Originele App

### Verbeteringen
✅ **Single-page interface** - Geen navigatie verwarring  
✅ **Realtime debug informatie** - Zie resultaten direct per stap  
✅ **Nederlandse mock data** - Realistische Nederlandse voorbeelden  
✅ **Progressieve interface** - Elke stap bouwt voort op vorige  
✅ **Modulaire componenten** - Makkelijker te onderhouden  

### Beperkingen (Prototype)
⚠️ **Mock data alleen** - Geen echte pipeline verbinding  
⚠️ **Geen echte API calls** - Gesimuleerde verwerking  
⚠️ **Geen bestandsverwerking** - Alleen UI demonstratie  

## Volgende Stappen
1. **Echte Data Integratie** - Verbind met bestaande pipeline_runner
2. **LLM Configuratie** - Gebruik echte Nederlandse LLM context
3. **Bestandsverwerking** - Echte SPSS bestand ondersteuning
4. **Testing** - Test met echte Nederlandse enquêtedata