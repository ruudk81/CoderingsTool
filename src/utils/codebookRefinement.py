import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import logging
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from openai import AsyncOpenAI

from config import ModelConfig, DEFAULT_MODEL_CONFIG, DEFAULT_LANGUAGE, OPENAI_API_KEY
from prompts import CODEBOOK_REFINEMENT_PROMPT
from models import RefinedCodebookModel, CodeRefinementResults
from utils.codeGenerator import CodeGeneratorReasoningResults
from utils.verboseReporter import VerboseReporter

# Setup logging
logger = logging.getLogger(__name__)

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
    
    def refine_codebook(self, survey_question: str, reasoning_results: CodeGeneratorReasoningResults) -> CodeRefinementResults:

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
            refined_model = self._call_refinement_llm(survey_question, raw_codes)
            
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
    
    def _extract_raw_codes(self, reasoning_results: CodeGeneratorReasoningResults) -> List[dict]:
        """Extract raw codes with IDs from reasoning results

        Returns:
            List of dicts with 'id', 'code', 'definition', and 'source_cluster_id' fields
        """
        raw_codes = []
        code_id_counter = 1

        # Extract from codebook if available (primary source)
        if hasattr(reasoning_results, 'codebook') and reasoning_results.codebook:
            for code_data in reasoning_results.codebook:
                if isinstance(code_data, dict) and 'code' in code_data:
                    raw_codes.append({
                        'id': str(code_id_counter),
                        'code': code_data['code'],
                        'definition': code_data.get('definition', ''),
                        'source_cluster_id': code_data.get('source_cluster_id', '')  # Preserve source cluster mapping
                    })
                    code_id_counter += 1

        # Alternative: extract from step4_validations (fallback)
        if not raw_codes and hasattr(reasoning_results, 'step4_validations'):
            for cluster_id, validation_data in reasoning_results.step4_validations.items():
                if isinstance(validation_data, dict) and 'code_validation' in validation_data:
                    code_validation = validation_data['code_validation']
                    if 'validated_code' in code_validation:
                        validated_code = code_validation['validated_code']
                        code_text = validated_code.get('code', '')
                        if code_text:
                            raw_codes.append({
                                'id': str(code_id_counter),
                                'code': code_text,
                                'definition': validated_code.get('definition', ''),
                                'source_cluster_id': str(cluster_id)  # Use cluster_id as source
                            })
                            code_id_counter += 1

        return raw_codes
    
    def _call_refinement_llm(self, survey_question: str, raw_codes: List[dict]) -> RefinedCodebookModel:
        """Call GPT-5 for codebook refinement using simple sync call"""
        from openai import OpenAI

        # Build mapping from sequential ID to source_cluster_id
        id_to_cluster_map = {code['id']: code.get('source_cluster_id', '') for code in raw_codes}

        # Format codes for prompt with IDs
        formatted_codes = '\n'.join([f"- [ID: {code['id']}] {code['code']}" for code in raw_codes])

        # Create prompt
        prompt = CODEBOOK_REFINEMENT_PROMPT.format(
            language=self.config.language,
            survey_question=survey_question,
            raw_codes=formatted_codes)

        self.reporter.info(f"Calling {self.model_config.codebook_refinement_model} for refinement")

        # Simple sync client
        client = OpenAI(api_key=self.api_key)
        
        try:
            # Ultra-simple API call
            response = client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"}
            )
            
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
                    # Convert codes defensively (changed from 'subcodes' to 'codes')
                    subcodes = []
                    subcodes_data = cat_data.get('codes', [])

                    for subcode_data in subcodes_data:
                        if isinstance(subcode_data, dict) and 'code' in subcode_data and 'description' in subcode_data:
                            # Map sequential ID back to source_cluster_id
                            sequential_id = subcode_data.get('id', '')
                            source_cluster = id_to_cluster_map.get(sequential_id, '')

                            subcode = RefinedSubcode(
                                id=sequential_id,  # Keep sequential ID for traceability
                                code=subcode_data['code'],
                                description=subcode_data['description'],
                                category=subcode_data.get('category', ''),  # Parse category field for 3-level hierarchy
                                source_cluster=source_cluster  # Map back to original cluster IDs
                            )
                            subcodes.append(subcode)
                        else:
                            self.reporter.warning(f"Skipping malformed subcode: {subcode_data}")
                    
                    # Convert category defensively (changed from 'category' to 'theme')
                    if 'theme' in cat_data:
                        category = RefinedCodebookCategory(
                            category=cat_data['theme'],
                            subcodes=subcodes
                        )
                        categories.append(category)
                    else:
                        self.reporter.warning(f"Skipping category without 'theme' field: {cat_data}")
                        
                except Exception as e:
                    self.reporter.error(f"Failed to convert category: {e}")
                    self.reporter.error(f"Category data: {cat_data}")
                    continue
            
            # Convert to our model with properly structured data
            # Use .model_dump() to convert Pydantic objects to dicts for proper validation
            refined_model = RefinedCodebookModel(
                analysis=response_data.get('analysis', 'No analysis provided'),
                refined_codebook=[cat.model_dump() for cat in categories],
                generation_metadata={
                    'model': "gpt-5",
                    'reasoning_effort': "minimal",
                    'text_verbosity': "low",
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

def refine_codebook(
    survey_question: str,    
    reasoning_results: CodeGeneratorReasoningResults,
    model_config: Optional[ModelConfig] = None,
    language: str = DEFAULT_LANGUAGE,
    verbose: bool = True
) -> CodeRefinementResults:
   
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG
    
    config = CodebookRefinementConfig(
        model_config=model_config,
        language=language,
        verbose=verbose
    )
    
    processor = CodebookRefinementProcessor(config)
    return processor.refine_codebook(survey_question, reasoning_results)

def get_refinement_report(results: CodeRefinementResults) -> dict:
    """Get refinement results as a structured dict for display in Streamlit

    Args:
        results: CodeRefinementResults object from refine_codebook()

    Returns:
        dict: Structured report with metadata, stats, analysis, and refined categories
    """
    stats = results.processing_stats

    # Build categories list with subcodes
    categories = []
    if results.refined_codebook.refined_codebook:
        for i, category in enumerate(results.refined_codebook.refined_codebook, 1):
            categories.append({
                'number': i,
                'category_name': category.category,
                'subcode_count': len(category.subcodes),
                'subcodes': [
                    {
                        'id': subcode.id,
                        'code': subcode.code,
                        'description': subcode.description,
                        'category': subcode.category  # Empty string for 2-level, category name for 3-level
                    }
                    for subcode in category.subcodes
                ]
            })

    # Structure the report
    report = {
        'metadata': {
            'timestamp': results.timestamp,
            'model_used': stats.get('model_used', 'unknown'),
            'language': stats.get('language', 'unknown'),
            'reasoning_effort': stats.get('reasoning_effort', 'unknown'),
            'text_verbosity': stats.get('text_verbosity', 'unknown')
        },
        'stats': {
            'original_code_count': stats.get('original_code_count', 0),
            'refined_category_count': stats.get('refined_category_count', 0),
            'total_refined_subcodes': stats.get('total_refined_subcodes', 0),
            'processing_time_seconds': stats.get('processing_time_seconds', 0)
        },
        'analysis': {
            'text': results.refined_codebook.analysis if results.refined_codebook.analysis else None
        },
        'categories': categories,
        'error': stats.get('error', None),
        'original_codebook': [
            {
                'code': code.get('code', ''),
                'definition': code.get('definition', ''),
                'cluster_id': code.get('cluster_id', '')
            }
            for code in results.original_codebook
        ] if results.original_codebook else []
    }

    return report


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
        print(f"\n[WARNING] ERROR: {stats['error']}")
    else:
        print(f"\nProcessing completed in {stats.get('processing_time_seconds', 0):.2f} seconds")
        
        if results.refined_codebook.analysis:
            print("\nLLM ANALYSIS:")
            print("-" * 40)
            print(results.refined_codebook.analysis)
            print()
        
        if results.refined_codebook.refined_codebook:
            print("\nREFINED HIERARCHY:")
            for i, theme in enumerate(results.refined_codebook.refined_codebook, 1):
                print(f"\n{i}. Theme: {theme.category} ({len(theme.subcodes)} codes)")

                # Group subcodes by category for 3-level hierarchy display
                direct_codes = []  # Codes directly under theme (category == "")
                categorized_codes = {}  # {category_name: [codes]}

                for subcode in theme.subcodes:
                    if subcode.category:
                        # 3-level: Code belongs to a category
                        if subcode.category not in categorized_codes:
                            categorized_codes[subcode.category] = []
                        categorized_codes[subcode.category].append(subcode)
                    else:
                        # 2-level: Code directly under theme
                        direct_codes.append(subcode)

                # Display direct codes (2-level)
                for code in direct_codes:
                    print(f"   - {code.code}")

                # Display categorized codes (3-level)
                for cat_name, codes in categorized_codes.items():
                    print(f"   Category: {cat_name} ({len(codes)} codes)")
                    for code in codes:
                        print(f"      - {code.code}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("Codebook Refinement Processor - GPT-5 based hierarchical refinement")
    print("Usage: from utils.codebookRefinement import refine_codebook")