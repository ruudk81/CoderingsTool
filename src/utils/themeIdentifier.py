import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
from typing import List, Dict, Any
import instructor
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
from utils.verboseReporter import VerboseReporter
from prompts import THEME_IDENTIFICATION_PROMPT

# === UTILS ========================================================================================================
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

class ThemeIdentifier:
    
    def __init__(self, 
                 codebook: List[Dict[str, str]], 
                 var_lab: str,
                 verbose: bool = False, 
                 prompt_printer = None):
 
        self.codebook = codebook
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.prompt_printer = prompt_printer
        self.model_config = ModelConfig()
        self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
    def _format_codes_for_prompt(self) -> str:
      
        if not self.codebook:
            return "No codes available"
        
        formatted_codes = []
        for i, code in enumerate(self.codebook): 
            code_text = code.code or f"Code {i+1}"
            definition = code.definition or "No definition available"
            formatted_codes.append(f"{i+1}. {code_text}: {definition}")
            
        return "\n".join(formatted_codes)
    
    async def _identify_themes_async(self) -> models.ThemeAnalysis:
      
        # Format codes for prompt
        codes_text = self._format_codes_for_prompt()
        
        # Build prompt using Braun & Clarke methodology
        prompt = THEME_IDENTIFICATION_PROMPT.format(
            language = DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            codes=codes_text
        )
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="theme_identification",
                utility_name="ThemeIdentifier",
                prompt_content=prompt,
                prompt_type="Theme Identification"
            )
        
        try:
            # Get structured response using instructor
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("hierarchical_organisation"),   
                messages=[{"role": "user", "content": prompt}],
                response_model=models.ThemeAnalysis,
                temperature=0.3,
                max_retries=3
            )
            
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error identifying themes: {str(e)}")
            
            # Return empty analysis on error
            return models.ThemeAnalysis(
                initial_observations=["Error occurred during theme identification"],
                suggested_themes=[],
                reflection={
                    "broad_or_narrow_themes": "Analysis failed due to error",
                    "contradictions_or_unexpected_patterns": "Could not analyze",
                    "potential_subthemes": "Analysis incomplete", 
                    "unclassified_codes": str([code.code for code in self.codebook])
                }
            )
    
    def identify_themes(self) -> Dict[str, Any]:
       
        self.verbose_reporter.section_header("THEME IDENTIFICATION")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'suggested_themes': [],
                'theme_analysis': models.ThemeAnalysis(
                    initial_observations=["No codes provided for analysis"],
                    suggested_themes=[],
                    reflection={
                        "broad_or_narrow_themes": "No analysis possible - no codes",
                        "contradictions_or_unexpected_patterns": "N/A",
                        "potential_subthemes": "N/A",
                        "unclassified_codes": "N/A"
                    }
                )
            }
        
        self.verbose_reporter.stat_line(f"Analyzing {len(self.codebook)} codes for theme patterns")
        
        # Run async theme identification
        theme_analysis = asyncio.run(self._identify_themes_async())
        
        elapsed_time = time.time() - start_time
        
        # Report results
        num_themes = len(theme_analysis.suggested_themes)
        self.verbose_reporter.summary("THEME IDENTIFICATION COMPLETE", {
            "Input codes": len(self.codebook),
            "Themes identified": num_themes,
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print themes if verbose
        if self.verbose and num_themes > 0:
            print("\nIdentified themes:")
            for i, theme in enumerate(theme_analysis.suggested_themes):
                print(f"  {i+1}. {theme.theme_name}")
                print(f"     Concept: {theme.concept}")
                print(f"     Codes ({len(theme.codes)}): {', '.join(theme.codes)}")
                
        # Check for unclassified codes
        all_theme_codes = set()
        for theme in theme_analysis.suggested_themes:
            all_theme_codes.update(theme.codes)
            
        codebook_codes = {code.code for code in self.codebook}
        unclassified = codebook_codes - all_theme_codes
        
        if unclassified and self.verbose:
            print(f"\nUnclassified codes ({len(unclassified)}): {', '.join(unclassified)}")
            
        return {
            'suggested_themes': theme_analysis.suggested_themes,
            'theme_analysis': theme_analysis,
            'stats': {
                'total_codes': len(self.codebook),
                'themes_identified': num_themes,
                'codes_in_themes': len(all_theme_codes),
                'unclassified_codes': len(unclassified)
            }
        }