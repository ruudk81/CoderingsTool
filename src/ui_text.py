"""UI text constants for the CoderingsTool application with language support"""

# Language configuration
DEFAULT_LANGUAGE = "nl"  # Dutch as default
AVAILABLE_LANGUAGES = ["nl", "en"]

# Texts organized by language
UI_TEXTS = {
    "nl": {
        # App Configuration
        "APP_TITLE": "CoderingsTool - Enquête Respons Analyse",
        "APP_DESCRIPTION": """
        Analyseer open antwoorden uit enquêtes met AI-gestuurde tekstverwerking. 
        Deze tool helpt u kwalitatieve data uit SPSS-bestanden te verwerken, clusteren en categoriseren.
        """,
        
        # Sidebar
        "SIDEBAR_HEADER": "Analyse Pipeline",
        "SIDEBAR_DESCRIPTION": "Volg de stappen om uw enquêtedata te analyseren",
        
        # Upload Page
        "UPLOAD_HELP": "Upload een SPSS-bestand (.sav) met enquête antwoorden. Het bestand moet respondent-ID's en tekst antwoorden bevatten.",
        
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
        
        "EMBEDDING_INFO": """
        Genereer embeddings om:
        - Tekst om te zetten naar numerieke representaties
        - Semantische embeddings voor codes en beschrijvingen te maken
        - Data voor te bereiden voor clustering
        """,
        
        "CLUSTERING_INFO": """
        Hiërarchische clustering zal:
        - Vergelijkbare antwoorden groeperen
        - Meta, meso en micro clusters creëren
        - Thema's in de data identificeren
        """,
        
        "LABELING_INFO": """
        Thematisch labelen zal:
        - Beschrijvende labels aan clusters toewijzen
        - Samenvattingen voor elk thema genereren
        - Interpreteerbare resultaten leveren
        """,
        
        "RESULTS_INFO": """
        Uw analyse is compleet! U kunt:
        - De geclusterde en gelabelde antwoorden bekijken
        - Resultaten downloaden als CSV
        - De thematische structuur van uw data verkennen
        """,
        
        # Step Descriptions for Info Panel
        "STEP_DESCRIPTIONS": [
            "Upload uw SPSS databestand met enquête antwoorden.",
            "Preprocessing normaliseert en schoont de tekstdata voor analyse.",
            "Kwaliteitsfiltering verwijdert lage kwaliteit of betekenisloze antwoorden.",
            "Embeddings zetten tekst om in numerieke representaties voor clustering.",
            "Clustering groepeert vergelijkbare antwoorden in hiërarchische thema's.",
            "Labeling wijst betekenisvolle beschrijvingen toe aan elke cluster.",
            "Bekijk en download uw geanalyseerde resultaten."
        ],
        
        # Error Messages
        "ERROR_FILE_TYPE": "Upload alstublieft een geldig SPSS-bestand (.sav)",
        "ERROR_FILE_SIZE": "Bestandsgrootte overschrijdt de maximale limiet van 50MB",
        "ERROR_PROCESSING": "Er is een fout opgetreden tijdens de verwerking. Probeer het opnieuw.",
        "ERROR_API_KEY": "OpenAI API-sleutel niet geconfigureerd. Stel OPENAI_API_KEY omgevingsvariabele in.",
        
        # Success Messages
        "SUCCESS_UPLOAD": "Bestand succesvol geüpload!",
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
        "CURRENT_STEP": "Huidige Stap:"
    },
    
    "en": {
        # App Configuration
        "APP_TITLE": "CoderingsTool - Survey Response Analysis",
        "APP_DESCRIPTION": """
        Analyze open-ended survey responses with AI-powered text processing. 
        This tool helps you preprocess, cluster, and categorize qualitative data from SPSS files.
        """,
        
        # Sidebar
        "SIDEBAR_HEADER": "Analysis Pipeline",
        "SIDEBAR_DESCRIPTION": "Follow the steps to analyze your survey data",
        
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
        
        "EMBEDDING_INFO": """
        Generate embeddings to:
        - Convert text to numerical representations
        - Create semantic embeddings for codes and descriptions
        - Prepare data for clustering
        """,
        
        "CLUSTERING_INFO": """
        Hierarchical clustering will:
        - Group similar responses together
        - Create meta, meso, and micro clusters
        - Identify themes in the data
        """,
        
        "LABELING_INFO": """
        Thematic labeling will:
        - Assign descriptive labels to clusters
        - Generate summaries for each theme
        - Provide interpretable results
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
        "CURRENT_STEP": "Current Step:"
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