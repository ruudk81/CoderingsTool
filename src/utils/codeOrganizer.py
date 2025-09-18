import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import time
import asyncio
import json
from typing import List, Dict, Any, Optional
#from dataclasses import dataclass

#import instructor
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

# === MODELS ========================================================================================================
from pydantic import BaseModel, Field #, model_validator
#import models

# === CONFIG ========================================================================================================
from prompts import THEME_ORGANIZATION_REASONING_PROMPT
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY 

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ============================================================================
# PYDANTIC MODELS FOR THEME ORGANIZATION RESPONSE
# ============================================================================

class OrganizedCode(BaseModel):
    """Individual code within a theme"""
    code: str = Field(description="The code name")
    definition: str = Field(description="The code definition") 
    source_cluster: Optional[str] = Field(description="Original cluster ID", default=None)

class ThemeHierarchyLevel(BaseModel):
    """A theme with its codes at a specific hierarchy level"""
    theme_name: str = Field(description="Atomic theme name")
    theme_description: str = Field(description="Detailed theme description")
    #level: int = Field(description="Hierarchy level (1=broad, 2=intermediate, 3=specific)")
    codes: List[OrganizedCode] = Field(description="Codes belonging to this theme")
    #is_miscellaneous: bool = Field(description="Whether this is a miscellaneous theme", default=False)

class OrganizedCodebook(BaseModel):
    """Complete hierarchical theme organization"""
    themes: List[ThemeHierarchyLevel] = Field(description="All themes organized hierarchically")
    methodology: str = Field(description="Method used for organization")
    total_codes_organized: int = Field(description="Total number of codes organized")
    language: str = Field(description="Language of the codebook")

class ThemeOrganizationResult(BaseModel):
    """Full response wrapper for theme organization"""
    result: OrganizedCodebook = Field(description="The organized codebook")
    processing_time: float = Field(description="Time taken to process")
    model_used: str = Field(description="OpenAI model used")

# ============================================================================
# THEME ORGANIZER CLASS
# ============================================================================

class RetryableError(Exception):
    """Exception for retryable errors in API calls"""
    pass

class CodeOrganizer:
    """Theme organizer using OpenAI reasoning models with single-prompt approach"""
    
    def __init__(self, 
                 codebook: List[Dict[str, str]], 
                 var_lab: str, 
                 language: str = DEFAULT_LANGUAGE, 
                 verbose: bool = True, 
                 model: str = "gpt-5-mini", 
                 reasoning_effort: str = "low", 
                 text_verbosity: str = "medium"):
        self.codebook = codebook
        self.var_lab = var_lab
        self.language = language
        self.verbose = verbose
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        
        # Initialize verbose reporter
        self.reporter = VerboseReporter(enabled=verbose)
        
        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        retry=retry_if_exception_type(RetryableError)
    )
    async def _make_api_call(self, prompt: str) -> OrganizedCodebook:
        """Make API call with retry logic"""
        try:
            # Import here to avoid circular imports
            from utils.codeGenerator import _sync_responses_create
            
            self.reporter.summary("API Request", {
                "Model": self.model,
                "Reasoning Effort": self.reasoning_effort,
                "Text Verbosity": self.text_verbosity,
                "Codes Count": len(self.codebook)
            })
            
            # Use the existing responses.create wrapper with asyncio.to_thread
            response = await asyncio.to_thread(
                _sync_responses_create,
                model=self.model,
                prompt=prompt,
                reasoning_effort=self.reasoning_effort,
                text_verbosity=self.text_verbosity,
                timeout=300.0  # 5 minutes for reasoning models
            )
            
            # Parse the JSON response
            try:
                response_content = response.output_text
                if not response_content:
                    raise RetryableError("Empty response content")
                    
            except AttributeError as e:
                self.reporter.error(f"Response object missing output_text attribute: {e}")
                self.reporter.debug(f"Response object type: {type(response)}")
                self.reporter.debug(f"Response attributes: {dir(response)}")
                raise RetryableError(f"Invalid response format: {e}")
            
            try:
                # Try to parse as JSON first
                json_data = json.loads(response_content)
                
                # Convert to our Pydantic model
                organized_codebook = OrganizedCodebook(**json_data)
                
                self.reporter.summary("API Response Parsed", {
                    "Themes Created": len(organized_codebook.themes),
                    "Total Codes": organized_codebook.total_codes_organized,
                    "Methodology": organized_codebook.methodology
                })
                
                return organized_codebook
                
            except json.JSONDecodeError as e:
                self.reporter.error(f"Failed to parse JSON response: {e}")
                self.reporter.debug(f"Raw response length: {len(response_content)} chars")
                self.reporter.debug(f"Raw response preview: {response_content[:500]}...")
                raise RetryableError(f"Invalid JSON response: {e}")
                
            except Exception as e:
                self.reporter.error(f"Failed to create OrganizedCodebook from JSON: {e}")
                self.reporter.debug(f"JSON data keys: {list(json_data.keys()) if 'json_data' in locals() else 'N/A'}")
                raise RetryableError(f"Invalid codebook structure: {e}")
                
        except Exception as e:
            error_str = str(e)
            # Map rate limits and server errors to retryable errors
            if "429" in error_str or any(x in error_str for x in ["5", "timeout", "connection"]):
                self.reporter.warning(f"Retryable error occurred: {error_str}")
                raise RetryableError(error_str) from e
            else:
                self.reporter.error(f"Non-retryable error: {error_str}")
                raise  # Re-raise non-retryable errors immediately
    
    async def organize_themes_reasoning(self) -> Dict[str, Any]:
        """
        Organize codebook themes using OpenAI reasoning models in a single prompt
        
        Returns:
            Dict with keys: themes, code_to_theme_mapping, themes_summary, methodology
        """
        self.reporter.section_header("THEME ORGANIZATION WITH REASONING MODEL")
        
        start_time = time.time()
        
        try:
            # Prepare codebook data for prompt
            codebook_text = ""
            for i, entry in enumerate(self.codebook, 1):
                codebook_text += f"{entry['code']}\n"
            
            # Build the prompt
            prompt = THEME_ORGANIZATION_REASONING_PROMPT.format(
                language=self.language,
                codebook=codebook_text.strip(),
                research_question=self.var_lab,
                codes_count=len(self.codebook)
            )
            
            self.reporter.summary("Input Preparation", {
                "Research Question": self.var_lab[:100] + "..." if len(self.var_lab) > 100 else self.var_lab,
                "Total Codes": len(self.codebook),
                "Language": self.language,
                "Prompt Length": len(prompt)
            })
            
            # Make the API call
            organized_result = await self._make_api_call(prompt)
            
            # Process results into format expected by pipeline
            themes_summary = []
            code_to_theme_mapping = {}
            
            for theme in organized_result.themes:
                # Add to themes summary (matching existing format)
                theme_entry = {
                    'theme_name': theme.theme_name,
                    'theme_description': theme.theme_description,
                    'cluster_id': f"reasoning_{len(themes_summary)+1}",
                    #'is_miscellaneous': theme.is_miscellaneous,
                    'codes': []
                }
                
                # Process codes in this theme
                for code in theme.codes:
                    code_entry = {
                        'code_name': code.code,
                        'definition': code.definition,
                        'source_cluster': code.source_cluster
                    }
                    theme_entry['codes'].append(code_entry)
                    
                    # Map code to theme
                    code_to_theme_mapping[code.code] = theme.theme_name
                
                themes_summary.append(theme_entry)
            
            processing_time = time.time() - start_time
            
            self.reporter.summary("Organization Complete", {
                "Total Themes": len(themes_summary),
                "Codes Mapped": len(code_to_theme_mapping),
                "Processing Time": f"{processing_time:.2f}s",
                "Average Codes per Theme": f"{len(code_to_theme_mapping) / max(len(themes_summary), 1):.1f}"
            })
            
            # Return in format expected by existing pipeline
            return {
                'themes': themes_summary,
                'code_to_theme_mapping': code_to_theme_mapping,
                'themes_summary': themes_summary,  # Duplicate for compatibility
                'methodology': f"Single-prompt reasoning with {self.model} (effort: {self.reasoning_effort})",
                'processing_time': processing_time,
                'model_used': self.model
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.reporter.error(f"Theme organization failed after {processing_time:.2f}s: {str(e)}")
            
            # Return minimal fallback structure
            return {
                'themes': [],
                'code_to_theme_mapping': {},
                'themes_summary': [],
                'methodology': f"Failed: {str(e)}",
                'processing_time': processing_time,
                'model_used': self.model
            }