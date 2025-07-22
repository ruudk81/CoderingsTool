import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
from typing import List, Dict, Optional
import instructor
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import ModelConfig
from utils.verboseReporter import VerboseReporter
from prompts import INITIAL_CODEBOOK_CREATION_PROMPT

# === UTILS ========================================================================================================
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# === CONSTANTS ========================================================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class SpeculativeStarterCodes:
    """
    Generates initial hypothetical codes for GATOS codebook generation.
    These starter codes help bootstrap the iterative codebook creation process.
    """
    
    def __init__(self, 
                 var_lab: str, 
                 n_codes: int = 20,
                 verbose: bool = False, 
                 prompt_printer = None):
        """
        Initialize the speculative starter codes generator.
        
        Args:
            var_lab: The survey question/variable label
            n_codes: Number of starter codes to generate (default: 20)
            verbose: Enable verbose output
            prompt_printer: Optional prompt capture utility
        """
        self.var_lab = var_lab
        self.n_codes = n_codes
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.prompt_printer = prompt_printer
        self.model_config = ModelConfig()
        self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
    async def _generate_codes_async(self) -> List[models.CodeDefinition]:
        """
        Generate starter codes using the LLM.
        
        Returns:
            List of CodeDefinition objects
        """
        # Create the code template
        code_template = "\n".join([f"{i+1}. Code {i+1}" for i in range(self.n_codes)])
        
        # Build the prompt
        prompt = INITIAL_CODEBOOK_CREATION_PROMPT.format(
            k_to_start=self.n_codes,
            data_type="survey response",
            data_collection_context=f"a survey asking: {self.var_lab}",
            code_template=code_template
        )
        
        # Parse the prompt to extract the expected format
        # Since the LLM needs to return a list format, we'll modify the approach
        # to ask for JSON directly
        json_prompt = f"""{prompt}

Please provide your response as a JSON array of objects, where each object has "code" and "definition" fields. For example:
[
  {{"code": "Technical difficulties", "definition": "Issues related to technology or system failures"}},
  {{"code": "Communication problems", "definition": "Challenges in exchanging information or understanding"}}
]"""
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="gatos_codebook",
                utility_name="SpeculativeStarterCodes",
                prompt_content=json_prompt,
                prompt_type="Initial Codebook Creation"
            )
            
        try:
            # Get structured response using instructor
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model("phase_1"),
                messages=[{"role": "user", "content": json_prompt}],
                response_model=List[models.CodeDefinition],
                temperature=0.7,
                max_retries=3
            )
            
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error generating starter codes: {str(e)}")
            # Return empty list on error
            return []
    
    def generate(self) -> List[Dict[str, str]]:
        """
        Main entry point - generates starter codes synchronously.
        
        Returns:
            List of dictionaries with 'code' and 'definition' keys
        """
        self.verbose_reporter.section_header("SPECULATIVE STARTER CODES GENERATION")
        start_time = time.time()
        
        # Run async generation
        starter_codes_models = asyncio.run(self._generate_codes_async())
        
        # Convert to dict format
        starter_codes = [
            {"code": code.code, "definition": code.definition} 
            for code in starter_codes_models
        ]
        
        elapsed_time = time.time() - start_time
        
        # Report results
        self.verbose_reporter.summary("STARTER CODES GENERATED", {
            "Requested codes": self.n_codes,
            "Generated codes": len(starter_codes),
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print first few codes if verbose
        if self.verbose and starter_codes:
            print("\nSample starter codes:")
            for i, code in enumerate(starter_codes[:3]):
                print(f"  {i+1}. {code['code']}: {code['definition']}")
            if len(starter_codes) > 3:
                print(f"  ... and {len(starter_codes) - 3} more codes")
        
        return starter_codes