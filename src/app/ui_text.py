# -*- coding: utf-8 -*-
"""UI text constants for the CoderingsTool application with language support.

Only the two tables the app actually reads live here: STEP_NAMES (navigation /
progress labels) and STEP_INFO (the RUN-screen blurb shown before a step runs).
Everything else is inline `T("nl", "en")` at the call site in app.py / app_views.py.
"""

# Language configuration
DEFAULT_LANGUAGE = "nl"  # Dutch as default
AVAILABLE_LANGUAGES = ["nl", "en"]

# Texts organized by language
UI_TEXTS = {
    "nl": {
        # Step Names (for navigation and progress display)
        "STEP_NAMES": {
            0: "Upload Data",
            1: "Tekstbewerking",
            2: "Kwaliteitsfilter",
            3: "Descriptieve labels",
            4: "Taxonomie",
            5: "Codeboek",
            6: "Codes toewijzen",
            7: "Export"
        },

        # What each step does — shown on the RUN screen before the user starts it
        "STEP_INFO": {
            0: "Selecteer een eerder verwerkte dataset of upload een nieuw "
               "SPSS-bestand (.sav) en kies de tekstvariabele.",
            1: "Normaliseert de antwoorden en voert spellingscontrole uit "
               "(Hunspell + LLM). De enquêtevraag dient als context.",
            2: "Beoordeelt elk antwoord op betekenis: leeg, 'weet niet' en "
               "wartaal worden gemarkeerd en uitgesloten van verdere analyse.",
            3: "Bepaalt eerst de context (sector, onderwerp, perspectief en "
               "dominante dimensie) en extraheert daarna per antwoord discrete "
               "ideeën met een abstractieladder (uiting → interpretatie → "
               "abstractie).",
            4: "Classificeert alle ideeën in een taxonomie: facetten en "
               "attributen per domein, met consolidatie binnen en tussen "
               "domeinen.",
            5: "Genereert uit de taxonomie een codeboek: codes met definitie "
               "en valentie.",
            6: "Wijst aan elk idee een code uit het codeboek toe, met "
               "confidence en onderbouwing.",
            7: "Exporteert de resultaten: workbook met coderingen (+ .sav) en "
               "het codeboek als Excel."
        }
    },

    "en": {
        # Step Names (for navigation and progress display)
        "STEP_NAMES": {
            0: "Upload Data",
            1: "Preprocessing",
            2: "Quality Filter",
            3: "Descriptive Labels",
            4: "Taxonomy",
            5: "Codebook",
            6: "Code Assignment",
            7: "Export"
        },

        # What each step does — shown on the RUN screen before the user starts it
        "STEP_INFO": {
            0: "Pick a previously processed dataset or upload a new SPSS file "
               "(.sav) and select the text variable.",
            1: "Normalizes the responses and runs spell checking (Hunspell + "
               "LLM). The survey question serves as context.",
            2: "Grades every response for meaning: empty, 'don't know' and "
               "gibberish answers are flagged and excluded from further "
               "analysis.",
            3: "First establishes the context (sector, topic, perspective and "
               "dominant dimension), then extracts discrete ideas per response "
               "with an abstraction ladder (instance → interpretation → "
               "abstraction).",
            4: "Classifies all ideas into a taxonomy: facets and attributes "
               "per domain, with consolidation within and across domains.",
            5: "Generates a codebook from the taxonomy: codes with definition "
               "and valence.",
            6: "Assigns a codebook code to every idea, with confidence and "
               "rationale.",
            7: "Exports the results: codings workbook (+ .sav) and the "
               "codebook as Excel."
        }
    }
}


# Helper function to get text in current language
def get_text(key: str, language: str = DEFAULT_LANGUAGE):
    """Get text in specified language, fallback to default if not found."""
    if language not in UI_TEXTS:
        language = DEFAULT_LANGUAGE
    return UI_TEXTS[language].get(key, UI_TEXTS[DEFAULT_LANGUAGE].get(key, key))
