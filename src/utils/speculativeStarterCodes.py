import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
# Standard library imports
import asyncio
from typing import List, Dict

# Third-party imports
import instructor
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from prompts import INITIAL_CODEBOOK_CREATION_PROMPT
from config import ModelConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


class SpeculativeStarterCodes:
    def __init__(self, 
                 var_lab: str, 
                 n_codes: int = 20,
                 verbose: bool = False, 
                 prompt_printer = None):
        self.var_lab = var_lab
        self.n_codes = n_codes
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        self.language = DEFAULT_LANGUAGE
        self.model_config = ModelConfig()
        self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
    async def _generate_codes_async(self) -> List[models.CodeDefinition]:
        code_template = "\n".join([f"{i+1}. Code {i+1}" for i in range(self.n_codes)])
        prompt = INITIAL_CODEBOOK_CREATION_PROMPT.format(
            n_codes = self.n_codes,
            language = self.language,
            survey_question = self.var_lab,
            code_template=code_template)
        
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="Codebook Creation",
                utility_name="SpeculativeStarterCodes",
                prompt_content=prompt,
                prompt_type="Initial Codebook Creation"
            )
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("speculative_codes"),
                messages=[{"role": "user", "content": prompt}],
                response_model=List[models.CodeDefinition],
                temperature=0.7,
                max_retries=3)
            
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error generating starter codes: {str(e)}")
            return []
    
    def generate(self) -> List[Dict[str, str]]:
        self.verbose_reporter.section_header("SPECULATIVE STARTER CODES GENERATION")
        starter_codes_models = asyncio.run(self._generate_codes_async())
        
        starter_codes = [
            {"code": code.code, "definition": code.definition} 
            for code in starter_codes_models]
        
        self.verbose_reporter.summary("STARTER CODES GENERATED", {
            "Requested codes": self.n_codes,
            "Generated codes": len(starter_codes) })
        
        if self.verbose and starter_codes:
            print("\nSample starter codes:")
            for i, code in enumerate(starter_codes[:3]):
                print(f"  {i+1}. {code['code']}: {code['definition']}")
            if len(starter_codes) > 3:
                print(f"  ... and {len(starter_codes) - 3} more codes")
        
        return starter_codes