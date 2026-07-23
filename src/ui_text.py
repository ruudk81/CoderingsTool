# -*- coding: utf-8 -*-
"""UI text constants for the CoderingsTool application with language support"""

# Language configuration
DEFAULT_LANGUAGE = "nl"  # Dutch as default
AVAILABLE_LANGUAGES = ["nl", "en"]

# Texts organized by language
UI_TEXTS = {
    "nl": {
        # App Configuration
        "APP_TITLE": "CoderingsTool",
        "APP_DESCRIPTION": """
        AI-gestuurde analyse van open antwoorden.
        Verwerk, cluster en codeer kwalitatieve data uit enquêtes
        """,
        
        # Sidebar
        "SIDEBAR_HEADER": "Navigatie",
        "SIDEBAR_DESCRIPTION": "",
        
        # Upload Page
        "UPLOAD_HELP": "Upload een SPSS-bestand (.sav) met enquete antwoorden. Het bestand moet respondent-ID's en tekst antwoorden bevatten.",
        
        # Step Information
        "PREPROCESSING_INFO": """
        Deze stap zal:
        - Tekst antwoorden normaliseren
        - Spellingcontrole uitvoeren (Nederlands en Engels)
        - Data opschonen en standaardiseren
        """,
        
        "FILTERING_INFO": """
        Kwaliteitsfiltering zal:
        - Betekenisloze of lage kwaliteit antwoorden identificeren
        - Antwoorden beoordelen op inhouds kwaliteit
        - Antwoorden onder kwaliteitsdrempel wegfilteren
        """,
        
        "EXTRACTION_INFO": """
        Deze stap zal:
        - Responsies segmenteren in discrete ideeën
        - Elke idee een unieke ID geven
        - Voorbereiden voor embedding generatie
        """ ,
        
        "EMBEDDING_INFO": """
        Genereer embeddings om:
        - Tekst om te zetten naar numerieke representaties
        - Semantische embeddings voor codes en beschrijvingen te maken
        - Data voor te bereiden voor clustering
        """,
        
        "CLUSTERING_INFO": """
        Categorisering zal:
        - Ideeën partitioneren op basis van concept type
        - MECE categorieën ontdekken per partitie via MAP/REDUCE analyse
        - Elk idee toewijzen aan precies één categorie
        """,
        
        "LABELING_INFO": """
        Thematisch labelen zal:
        - Beschrijvende labels aan clusters toewijzen
        - Samenvattingen voor elk thema genereren
        - Interpreteerbare resultaten leveren
        """,

        "THEME_IDENTIFICATION_INFO": """
        Deze stap zal:
        - Codes groeperen in thema's
        - Hiërarchische thema structuur maken
        - Thema beschrijvingen genereren
        """,
        
        "RESULTS_INFO": """
        Uw analyse is compleet! U kunt:
        - De geclusterde en gelabelde antwoorden bekijken
        - Resultaten downloaden als CSV
        - De thematische structuur van uw data verkennen
        """,
        
        # Step Descriptions for Info Panel
        "STEP_DESCRIPTIONS": [
            "Upload uw SPSS databestand met enquete antwoorden.",
            "Preprocessing normaliseert en schoont de tekstdata voor analyse.",
            "Kwaliteitsfiltering verwijdert lage kwaliteit of betekenisloze antwoorden.",
            "Embeddings zetten tekst om in numerieke representaties voor clustering.",
            "Clustering groepeert vergelijkbare antwoorden in hierarchische thema's.",
            "Labeling wijst betekenisvolle beschrijvingen toe aan elke cluster.",
            "Bekijk en download uw geanalyseerde resultaten."
        ],
        
        # Error Messages
        "ERROR_FILE_TYPE": "Upload alstublieft een geldig SPSS-bestand (.sav)",
        "ERROR_FILE_SIZE": "Bestandsgrootte overschrijdt de maximale limiet van 50MB",
        "ERROR_PROCESSING": "Er is een fout opgetreden tijdens de verwerking. Probeer het opnieuw.",
        "ERROR_API_KEY": "OpenAI API-sleutel niet geconfigureerd. Stel OPENAI_API_KEY omgevingsvariabele in.",
        
        # Success Messages
        "SUCCESS_UPLOAD": "Bestand succesvol geupload!",
        "SUCCESS_PREPROCESSING": "Tekst preprocessing voltooid!",
        "SUCCESS_FILTERING": "Kwaliteitsfiltering voltooid!",
        "SUCCESS_EMBEDDING": "Embeddings succesvol gegenereerd!",
        "SUCCESS_CLUSTERING": "Clustering voltooid!",
        "SUCCESS_LABELING": "Thematisch labelen voltooid!",
        
        # Button Labels
        "BTN_UPLOAD": "Upload Bestand",
        "BTN_PREPROCESS": "Start Preprocessing",
        "BTN_FILTER": "Pas Filters Toe",
        "BTN_EMBED": "Genereer Embeddings",
        "BTN_CLUSTER": "Start Clustering",
        "BTN_LABEL": "Genereer Labels",
        "BTN_DOWNLOAD": "Download Resultaten",
        "BTN_RESTART": "Start Nieuwe Analyse",
        
        # Language Selector
        "LANGUAGE_LABEL": "Taal:",
        "CURRENT_STEP": "Huidige Stap:",

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
        # (current 8-step pipeline; app_v2)
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
               "abstractie). Dit is de zwaarste stap: enkele minuten en "
               "LLM-credits.",
            4: "Classificeert alle ideeën in een taxonomie: facetten en "
               "attributen per domein, met consolidatie binnen en tussen "
               "domeinen (fasen P1–P8). Duurt enkele minuten.",
            5: "Genereert uit de taxonomie een codeboek: codes met definitie "
               "en valentie (fasen P8–P9).",
            6: "Wijst aan elk idee een code uit het codeboek toe, met "
               "confidence en onderbouwing.",
            7: "Exporteert de resultaten: workbook met coderingen (+ .sav) en "
               "het codeboek als Excel."
        }
    },
    
    "en": {
        # App Configuration
        "APP_TITLE": "CodingTool",
        "APP_DESCRIPTION": """
        AI-driven analysis of open-ended responses.
        Process, cluster, and code qualitative survey data.
        """,
        
        # Sidebar
        "SIDEBAR_HEADER": "Navigation",
        "SIDEBAR_DESCRIPTION": "",
        
        # Upload Page
        "UPLOAD_HELP": "Upload an SPSS file (.sav) containing survey responses. The file should include respondent IDs and text responses.",
        
        # Step Information
        "PREPROCESSING_INFO": """
        This step will:
        - Normalize text responses
        - Perform spell checking (Dutch and English)
        - Clean and standardize the data
        """,
        
        "FILTERING_INFO": """
        Quality filtering will:
        - Identify meaningless or low-quality responses
        - Grade responses based on content quality
        - Filter out responses below quality threshold
        """,
        
        "EXTRACTION_INFO": """
        This step will:
        - Segment responses into discrete ideas  
        - Assign unique IDs to each idea
        - Prepare for embedding generation
        """,
        
        "EMBEDDING_INFO": """
        Generate embeddings to:
        - Convert text to numerical representations
        - Create semantic embeddings for codes and descriptions
        - Prepare data for clustering
        """,
        
        "CLUSTERING_INFO": """
        Category discovery will:
        - Partition ideas by concept type
        - Discover MECE categories per partition via MAP/REDUCE analysis
        - Assign each idea to exactly one category
        """,
        
        "LABELING_INFO": """
        Thematic labeling will:
        - Assign descriptive labels to clusters
        - Generate summaries for each theme
        - Provide interpretable results
        """,

        "THEME_IDENTIFICATION_INFO": """
        This step will:
        - Group codes into themes
        - Create hierarchical theme structure
        - Generate theme descriptions
        """,
        
        "RESULTS_INFO": """
        Your analysis is complete! You can:
        - View the clustered and labeled responses
        - Download results as CSV
        - Explore the thematic structure of your data
        """,
        
        # Step Descriptions for Info Panel
        "STEP_DESCRIPTIONS": [
            "Upload your SPSS data file containing survey responses.",
            "Preprocessing normalizes and cleans the text data for analysis.",
            "Quality filtering removes low-quality or meaningless responses.",
            "Embeddings convert text into numerical representations for clustering.",
            "Clustering groups similar responses into hierarchical themes.",
            "Labeling assigns meaningful descriptions to each cluster.",
            "Review and download your analyzed results."
        ],
        
        # Error Messages
        "ERROR_FILE_TYPE": "Please upload a valid SPSS file (.sav)",
        "ERROR_FILE_SIZE": "File size exceeds the maximum limit of 50MB",
        "ERROR_PROCESSING": "An error occurred during processing. Please try again.",
        "ERROR_API_KEY": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
        
        # Success Messages
        "SUCCESS_UPLOAD": "File uploaded successfully!",
        "SUCCESS_PREPROCESSING": "Text preprocessing completed!",
        "SUCCESS_FILTERING": "Quality filtering completed!",
        "SUCCESS_EMBEDDING": "Embeddings generated successfully!",
        "SUCCESS_CLUSTERING": "Clustering completed!",
        "SUCCESS_LABELING": "Thematic labeling completed!",
        
        # Button Labels
        "BTN_UPLOAD": "Upload File",
        "BTN_PREPROCESS": "Start Preprocessing",
        "BTN_FILTER": "Apply Filters",
        "BTN_EMBED": "Generate Embeddings",
        "BTN_CLUSTER": "Run Clustering",
        "BTN_LABEL": "Generate Labels",
        "BTN_DOWNLOAD": "Download Results",
        "BTN_RESTART": "Start New Analysis",
        
        # Language Selector
        "LANGUAGE_LABEL": "Language:",
        "CURRENT_STEP": "Current Step:",

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
        # (current 8-step pipeline; app_v2)
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
               "abstraction). The heaviest step: several minutes and LLM "
               "credits.",
            4: "Classifies all ideas into a taxonomy: facets and attributes "
               "per domain, with consolidation within and across domains "
               "(phases P1–P8). Takes several minutes.",
            5: "Generates a codebook from the taxonomy: codes with definition "
               "and valence (phases P8–P9).",
            6: "Assigns a codebook code to every idea, with confidence and "
               "rationale.",
            7: "Exports the results: codings workbook (+ .sav) and the "
               "codebook as Excel."
        }
    }
}

# Helper function to get text in current language
def get_text(key: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Get text in specified language, fallback to default if not found"""
    if language not in UI_TEXTS:
        language = DEFAULT_LANGUAGE
    
    return UI_TEXTS[language].get(key, UI_TEXTS[DEFAULT_LANGUAGE].get(key, key))

# For backward compatibility - expose default language texts as direct attributes
for key, value in UI_TEXTS[DEFAULT_LANGUAGE].items():
    globals()[key] = value