import os

# File handling
ALLOWED_EXTENSIONS = ['.sav']
MAX_FILE_SIZE_MB = 50

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CONTEXT_WINDOW = 4000
MAX_OUTPUT_TOKENS = 1000

# Preprocessing settings
BATCH_SIZE = 100  

current_dir =  os.getcwd()
if    os.path.basename(current_dir) == 'utils':
      hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'hunspell'))
elif  os.path.basename(current_dir) == 'modules':
      hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hunspell'))
elif  os.path.basename(current_dir) == 'src':
      hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', 'hunspell'))
elif  os.path.basename(current_dir) == 'Coderingstool':
      hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell')) 

# Hunspell settings
SUPPORTED_LANGUAGES = ["nl", "en_GB"]  # Dutch and English
HUNSPELL_PATH = os.path.join(hunspell_dir, "hunspell.exe")
DUTCH_DICT_PATH = os.path.join(hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(hunspell_dir,  "dict", "en_GB")
DEFAULT_LANGUAGE = "Dutch"  
#MAX_SPELL_CHECK_LENGTH = 200   

