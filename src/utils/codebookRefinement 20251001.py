import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Third-party imports
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Local imports
from config import ModelConfig, DEFAULT_MODEL_CONFIG, DEFAULT_LANGUAGE, OPENAI_API_KEY
from models import RefinedCodebookModel, CodeRefinementResults
from utils.codeGenerator import CodeGeneratorReasoningResults
from utils.verboseReporter import VerboseReporter

# Setup logging
logger = logging.getLogger(__name__)

# === PROMPT TEMPLATE ========================================================================================================

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative researcher and codebook methodologist. 
Your task is to take a raw list of descriptive codes and transform it into a refined and structured codebook. 
The descriptive codes are derived from survey responses.

<inputs>
Language to use: {language}

Raw descriptive codes to refine:
{raw_codes}
</inputs>

<guidance>
A high-quality codebook must be:
- Non-redundant: no codes that repeat the same idea in different words.  
- Semantically differentiated: each code uses distinct wording; small overlap is acceptable.  
- Inherently distinct: codes within the same category are not identical in content.  
- Parsimonious: no unnecessary duplication; concise but comprehensive.  
- Structured: grouped under 6–8 main categories with consistent subcodes.  
- Consistently named: short, uniform, and meaningful labels.  

When codes overlap semantically or thematically:
- Merge them into a single code with a clear, inclusive label.  
- If codes are near-identical, consolidate into one code and discard duplicates.  
- If codes represent sub-aspects of the same domain, nest them as subcodes.  
- If codes are vague, reword them for clarity.  

</guidance>

<analysis_steps>
1. Review all raw codes.  
   - Identify redundant codes (semantic duplicates, identical meaning).  
   - Identify overlapping codes that can be merged into broader codes with subcodes.  

2. Construct main categories that represent broad domains.

3. Assign refined codes as subcodes under these main categories.  
   - Each subcode must represent ONE distinct, actionable concept.  
   - Remove vague or overly broad codes.  

4. Ensure consistent naming:  
   - Labels ≤ 8 words.  
   - Active or descriptive phrasing.  
   - No repetition across categories.  

5. Document the restructuring:  
   - How many raw codes were merged?  
   - Which semantic duplicates were consolidated?  
   - Which main categories were created?  

</analysis_steps>

Provide your response as a valid JSON dictionary using this exact structure:
{{
  "analysis": "Provide your analysis here in {language} (describe main restructuring decisions, what was merged, how categories were formed).",
   "refined_codebook": [
      {{
        "category": "Main category label",
        "subcodes": [
          {{
            "code": "Refined subcode label",
            "description": "≤ 20 words explanation of what this code means"
          }}
          // Add additional subcodes as needed
        ]
      }}
      // Add additional categories here
    ]
}}


Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after.  
- Replace "codebook_id" and "language" with the actual values provided.  
- Conduct your analysis in the specified language.  
"""

# === DIRECT JSON PARSING ========================================================================================================
# No special models needed - we parse JSON directly and validate with our existing RefinedCodebookModel

# === CORE IMPLEMENTATION ========================================================================================================

@dataclass 
class CodebookRefinementConfig:
    """Configuration for codebook refinement"""
    model_config: ModelConfig
    language: str = DEFAULT_LANGUAGE
    verbose: bool = True

class CodebookRefinementProcessor:
    """
    Processes raw codebooks through GPT-5 refinement to create structured, hierarchical codebooks.
    Follows the existing qualityFilter.py patterns for LLM prompt processing.
    """
    
    def __init__(self, config: CodebookRefinementConfig):
        self.config = config
        self.model_config = config.model_config
        self.api_key = OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Setup verbose reporter
        self.reporter = VerboseReporter(enabled=config.verbose)
        
        logger.info(f"Initialized CodebookRefinementProcessor with model: {self.model_config.codebook_refinement_model}")
    
    async def refine_codebook(self, reasoning_results: CodeGeneratorReasoningResults) -> CodeRefinementResults:
        """
        Main method to refine a codebook from raw codes to structured hierarchy.
        
        Args:
            reasoning_results: Results from code generation step
            
        Returns:
            CodeRefinementResults with original and refined codebooks
        """
        start_time = datetime.now()
        
        self.reporter.step_start("Codebook Refinement (Step 7b)")
        
        try:
            # Extract raw codes
            raw_codes = self._extract_raw_codes(reasoning_results)
            
            if not raw_codes:
                self.reporter.warning("No codes found in reasoning results")
                return self._create_empty_results(reasoning_results, start_time)
            
            self.reporter.info(f"Processing {len(raw_codes)} raw codes for refinement")
            
            # Call GPT-5 for refinement
            refined_model = await self._call_refinement_llm(raw_codes)
            
            # Create results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results = CodeRefinementResults(
                original_codebook=[code.model_dump() if hasattr(code, 'model_dump') else code for code in reasoning_results.codebook],
                refined_codebook=refined_model,
                processing_stats={
                    'original_code_count': len(raw_codes),
                    'refined_category_count': len(refined_model.refined_codebook),
                    'total_refined_subcodes': sum(len(cat.subcodes) for cat in refined_model.refined_codebook),
                    'processing_time_seconds': processing_time,
                    'model_used': self.model_config.codebook_refinement_model,
                    'language': self.config.language,
                    'reasoning_effort': self.model_config.get_reasoning_effort_for_stage('codebook_refinement'),
                    'text_verbosity': self.model_config.get_text_verbosity_for_stage('codebook_refinement')
                },
                timestamp=start_time.isoformat()
            )
            
            self.reporter.step_complete(
                f"Refinement completed: {len(raw_codes)} codes → {len(refined_model.refined_codebook)} categories "
                f"with {sum(len(cat.subcodes) for cat in refined_model.refined_codebook)} subcodes "
                f"in {processing_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            self.reporter.error(f"Refinement failed: {str(e)}")
            logger.error(f"Codebook refinement error: {str(e)}", exc_info=True)
            return self._create_error_results(reasoning_results, start_time, str(e))
    
    def _extract_raw_codes(self, reasoning_results: CodeGeneratorReasoningResults) -> List[str]:
        """Extract raw code strings from reasoning results"""
        raw_codes = []
        
        # Extract from codebook if available
        if hasattr(reasoning_results, 'codebook') and reasoning_results.codebook:
            for code_entry in reasoning_results.codebook:
                if isinstance(code_entry, dict):
                    code_text = code_entry.get('code', '')
                    if code_text and code_text not in raw_codes:
                        raw_codes.append(code_text)
                elif hasattr(code_entry, 'code'):
                    if code_entry.code and code_entry.code not in raw_codes:
                        raw_codes.append(code_entry.code)
        
        # Alternative: extract from step4_validations
        if not raw_codes and hasattr(reasoning_results, 'step4_validations'):
            for cluster_id, validation_data in reasoning_results.step4_validations.items():
                if isinstance(validation_data, dict) and 'code_validation' in validation_data:
                    code_validation = validation_data['code_validation']
                    if 'validated_code' in code_validation:
                        validated_code = code_validation['validated_code']
                        code_text = validated_code.get('code', '')
                        if code_text and code_text not in raw_codes:
                            raw_codes.append(code_text)
        
        return raw_codes
    
    async def _call_refinement_llm(self, raw_codes: List[str]) -> RefinedCodebookModel:
        """Call GPT-5 for codebook refinement using direct responses.create"""
        
        # Format codes for prompt
        formatted_codes = '\n'.join([f"- {code}" for code in raw_codes])
        
        # Create prompt
        prompt = CODEBOOK_REFINEMENT_PROMPT.format(
            language=self.config.language,
            raw_codes=formatted_codes
        )
        
        self.reporter.info(f"Calling {self.model_config.codebook_refinement_model} for refinement")
        
        # Get model configuration
        model_name = self.model_config.get_model_for_stage('codebook_refinement')
        reasoning_effort = self.model_config.get_reasoning_effort_for_stage('codebook_refinement')
        text_verbosity = self.model_config.get_text_verbosity_for_stage('codebook_refinement')
        
        # Prepare request parameters for responses.create
        request_params = {
            "model": model_name,
            "input": [{"role": "user", "content": prompt}]
        }
        
        # Add GPT-5 specific parameters for reasoning models
        model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")
        if model_type == "reasoning":
            request_params["reasoning"] = {"effort": reasoning_effort}
            request_params["text"] = {"verbosity": text_verbosity}
        
        try:
            # Make the direct API call
            response = await self.client.responses.create(**request_params)
            
            # Extract text from response
            response_text = response.output_text
            
            # Parse JSON
            import json
            response_data = json.loads(response_text)
            
            # DEBUG: Print the actual JSON structure
            self.reporter.info("=== DEBUG: GPT-5 Response Structure ===")
            self.reporter.info(f"Response keys: {list(response_data.keys())}")
            if 'refined_codebook' in response_data:
                self.reporter.info(f"Refined codebook length: {len(response_data['refined_codebook'])}")
                if response_data['refined_codebook']:
                    first_cat = response_data['refined_codebook'][0]
                    self.reporter.info(f"First category keys: {list(first_cat.keys())}")
                    if 'subcodes' in first_cat and first_cat['subcodes']:
                        first_subcode = first_cat['subcodes'][0]
                        self.reporter.info(f"First subcode keys: {list(first_subcode.keys())}")
            
            # Manually convert nested structures to Pydantic objects with defensive programming
            from models import RefinedSubcode, RefinedCodebookCategory
            
            categories = []
            refined_codebook_data = response_data.get('refined_codebook', [])
            
            for cat_data in refined_codebook_data:
                try:
                    # Convert subcodes defensively
                    subcodes = []
                    subcodes_data = cat_data.get('subcodes', [])
                    
                    for subcode_data in subcodes_data:
                        if isinstance(subcode_data, dict) and 'code' in subcode_data and 'description' in subcode_data:
                            subcode = RefinedSubcode(
                                code=subcode_data['code'],
                                description=subcode_data['description']
                            )
                            subcodes.append(subcode)
                        else:
                            self.reporter.warning(f"Skipping malformed subcode: {subcode_data}")
                    
                    # Convert category defensively
                    if 'category' in cat_data:
                        category = RefinedCodebookCategory(
                            category=cat_data['category'],
                            subcodes=subcodes
                        )
                        categories.append(category)
                    else:
                        self.reporter.warning(f"Skipping category without 'category' field: {cat_data}")
                        
                except Exception as e:
                    self.reporter.error(f"Failed to convert category: {e}")
                    self.reporter.error(f"Category data: {cat_data}")
                    continue
            
            # Convert to our model with properly structured data
            refined_model = RefinedCodebookModel(
                analysis=response_data.get('analysis', 'No analysis provided'),
                refined_codebook=categories,
                generation_metadata={
                    'model': model_name,
                    'reasoning_effort': reasoning_effort,
                    'text_verbosity': text_verbosity,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.reporter.info(f"LLM call successful: {len(refined_model.refined_codebook)} categories generated")
            
            return refined_model
            
        except Exception as e:
            self.reporter.error(f"LLM call failed: {str(e)}")
            raise
    
    def _create_empty_results(self, reasoning_results: CodeGeneratorReasoningResults, start_time: datetime) -> CodeRefinementResults:
        """Create empty results when no codes found"""
        return CodeRefinementResults(
            original_codebook=[],
            refined_codebook=RefinedCodebookModel(
                analysis="No codes found for refinement",
                refined_codebook=[]
            ),
            processing_stats={'error': 'No codes found for refinement'},
            timestamp=start_time.isoformat()
        )
    
    def _create_error_results(self, reasoning_results: CodeGeneratorReasoningResults, start_time: datetime, error_msg: str) -> CodeRefinementResults:
        """Create error results when refinement fails"""
        return CodeRefinementResults(
            original_codebook=[code.model_dump() if hasattr(code, 'model_dump') else code for code in reasoning_results.codebook] if hasattr(reasoning_results, 'codebook') else [],
            refined_codebook=RefinedCodebookModel(
                analysis=f"Refinement failed: {error_msg}",
                refined_codebook=[]
            ),
            processing_stats={'error': error_msg},
            timestamp=start_time.isoformat()
        )

# === UTILITY FUNCTIONS ========================================================================================================

async def refine_codebook(
    reasoning_results: CodeGeneratorReasoningResults,
    model_config: Optional[ModelConfig] = None,
    language: str = DEFAULT_LANGUAGE,
    verbose: bool = True
) -> CodeRefinementResults:
    """
    Main entry point for codebook refinement.
    
    Args:
        reasoning_results: Results from code generation step
        model_config: Model configuration (uses default if None)
        language: Language for processing
        verbose: Enable verbose output
        
    Returns:
        CodeRefinementResults with original and refined codebooks
    """
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG
    
    config = CodebookRefinementConfig(
        model_config=model_config,
        language=language,
        verbose=verbose
    )
    
    processor = CodebookRefinementProcessor(config)
    return await processor.refine_codebook(reasoning_results)

def print_refinement_report(results: CodeRefinementResults):
    """Print a formatted report of refinement results"""
    stats = results.processing_stats
    
    print(f"\n{'='*60}")
    print("CODEBOOK REFINEMENT REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Model: {stats.get('model_used', 'unknown')}")
    print(f"Language: {stats.get('language', 'unknown')}")
    print(f"Reasoning Effort: {stats.get('reasoning_effort', 'unknown')}")
    print(f"Text Verbosity: {stats.get('text_verbosity', 'unknown')}")
    
    print(f"\nOriginal Codes: {stats.get('original_code_count', 0)}")
    print(f"Refined Categories: {stats.get('refined_category_count', 0)}")
    print(f"Total Subcodes: {stats.get('total_refined_subcodes', 0)}")
    
    if 'error' in stats:
        print(f"\n⚠️  ERROR: {stats['error']}")
    else:
        print(f"\nProcessing completed in {stats.get('processing_time_seconds', 0):.2f} seconds")
        
        if results.refined_codebook.refined_codebook:
            print(f"\nREFINED CATEGORIES:")
            print(f"{'-'*40}")
            for i, category in enumerate(results.refined_codebook.refined_codebook, 1):
                print(f"{i}. {category.category} ({len(category.subcodes)} subcodes)")
                for subcode in category.subcodes[:3]:  # Show first 3 subcodes
                    print(f"   - {subcode.code}")
                if len(category.subcodes) > 3:
                    print(f"   ... and {len(category.subcodes) - 3} more subcodes")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("Codebook Refinement Processor - GPT-5 based hierarchical refinement")
    print("Usage: from utils.codebookRefinement import refine_codebook")