import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import logging
import math
from datetime import datetime
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from openai import AsyncOpenAI, OpenAI

from config import ModelConfig, DEFAULT_MODEL_CONFIG, DEFAULT_LANGUAGE, OPENAI_API_KEY
from prompts import CODEBOOK_REFINEMENT_PROMPT, CODEBOOK_MERGE_PROMPT
from models import RefinedCodebookModel, CodeRefinementResults #RefinedCodebookCategory
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
    prompt_printer: Optional[Any] = None
    hierarchical_threshold: int = 20
    target_batch_size: int = 20
    overlap_percentage: float = 0.10

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

        # Setup prompt printer
        self.prompt_printer = config.prompt_printer

        logger.info(f"Initialized CodebookRefinementProcessor with model: {self.model_config.codebook_refinement_model}")
    
    def refine_codebook(self, survey_question: str, reasoning_results: CodeGeneratorReasoningResults) -> CodeRefinementResults:
        """Main entry point - decides between single-batch or MAP-REDUCE"""

        try:
            # Extract raw codes
            raw_codes = self._extract_raw_codes(reasoning_results)

            if not raw_codes:
                self.reporter.warning("No codes found in reasoning results")
                return self._create_empty_results(reasoning_results, datetime.now())

            self.reporter.stat_line(f"Processing {len(raw_codes)} codes")

            # Decide processing strategy
            if len(raw_codes) > self.config.hierarchical_threshold:
                self.reporter.stat_line(f"Using MAP-REDUCE (threshold: {self.config.hierarchical_threshold})")
                return self._refine_hierarchically(survey_question, raw_codes)
            else:
                self.reporter.stat_line("Using single-batch refinement")
                return self._refine_single_batch(survey_question, raw_codes)

        except Exception as e:
            self.reporter.error(f"Refinement failed: {str(e)}")
            logger.error(f"Codebook refinement error: {str(e)}", exc_info=True)
            return self._create_error_results(reasoning_results, datetime.now(), str(e))

    def _refine_single_batch(self, survey_question: str, raw_codes: List[dict]) -> CodeRefinementResults:
        """Single-batch refinement (for <= threshold codes)"""

        start_time = datetime.now()

        # Use MAP method for single batch (same prompt, no REDUCE needed)
        refined_model = self._refine_batch_map(survey_question, raw_codes, batch_id=0)

        # Build results
        processing_time = (datetime.now() - start_time).total_seconds()

        return CodeRefinementResults(
            original_codebook=[{k: v for k, v in code.items() if k not in ['is_boundary']} for code in raw_codes],
            refined_codebook=refined_model,
            processing_stats={
                'original_code_count': len(raw_codes),
                'refined_category_count': len(refined_model.refined_codebook),
                'total_refined_subcodes': sum(len(cat.subcodes) for cat in refined_model.refined_codebook),
                'processing_time_seconds': processing_time,
                'model_used': self.model_config.codebook_refinement_model,
                'language': self.config.language,
                'reasoning_effort': 'minimal',
                'text_verbosity': 'low'
            },
            timestamp=start_time.isoformat()
        )
    
    def _extract_raw_codes(self, reasoning_results: CodeGeneratorReasoningResults) -> List[dict]:
        """Extract raw codes with IDs and assignment_examples from reasoning results

        Returns:
            List of dicts with 'id', 'code', 'definition', 'source_cluster_id',
            'inclusion_examples', 'exclusion_examples', 'near_neighbor_label', 'tell_apart_rule' fields
        """
        import json
        raw_codes = []
        code_id_counter = 1

        # Extract from codebook if available (primary source)
        if hasattr(reasoning_results, 'codebook') and reasoning_results.codebook:
            for code_data in reasoning_results.codebook:
                if isinstance(code_data, dict) and 'code' in code_data:
                    # Parse inclusion_examples (stored as JSON string)
                    inclusion_examples = []
                    if 'inclusion_examples' in code_data and code_data['inclusion_examples']:
                        try:
                            inclusion_examples = json.loads(code_data['inclusion_examples'])
                        except (json.JSONDecodeError, TypeError):
                            inclusion_examples = []

                    # Parse exclusion_examples (stored as JSON string)
                    exclusion_examples = []
                    if 'exclusion_examples' in code_data and code_data['exclusion_examples']:
                        try:
                            exclusion_examples = json.loads(code_data['exclusion_examples'])
                        except (json.JSONDecodeError, TypeError):
                            exclusion_examples = []

                    raw_codes.append({
                        'id': str(code_id_counter),
                        'code': code_data['code'],
                        'definition': code_data.get('definition', ''),
                        'source_cluster_id': code_data.get('source_cluster_id', ''),  # Preserve source cluster mapping
                        'inclusion_examples': inclusion_examples,
                        'exclusion_examples': exclusion_examples,
                        'near_neighbor_label': code_data.get('near_neighbor_label', None),
                        'tell_apart_rule': code_data.get('tell_apart_rule', None)
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
                            # Extract assignment_examples from validated_code if available
                            inclusion_examples = []
                            exclusion_examples = []
                            near_neighbor_label = None
                            tell_apart_rule = None

                            if 'assignment_examples' in validated_code and validated_code['assignment_examples']:
                                assignment_ex = validated_code['assignment_examples']
                                if hasattr(assignment_ex, 'inclusion'):
                                    inclusion_examples = assignment_ex.inclusion
                                if hasattr(assignment_ex, 'exclusion'):
                                    exclusion_examples = assignment_ex.exclusion
                                if hasattr(assignment_ex, 'near_neighbor'):
                                    near_neighbor = assignment_ex.near_neighbor
                                    if hasattr(near_neighbor, 'label'):
                                        near_neighbor_label = near_neighbor.label
                                    if hasattr(near_neighbor, 'tell_apart_rule'):
                                        tell_apart_rule = near_neighbor.tell_apart_rule

                            raw_codes.append({
                                'id': str(code_id_counter),
                                'code': code_text,
                                'definition': validated_code.get('definition', ''),
                                'source_cluster_id': str(cluster_id),  # Use cluster_id as source
                                'inclusion_examples': inclusion_examples,
                                'exclusion_examples': exclusion_examples,
                                'near_neighbor_label': near_neighbor_label,
                                'tell_apart_rule': tell_apart_rule
                            })
                            code_id_counter += 1

        return raw_codes
    
    def _format_code_with_assignment_examples(self, code: dict) -> str:
        """Format a single code with all its details including assignment_examples"""
        parts = [f"- [ID: {code['id']}] {code['code']}"]
        parts.append(f"  Definition: {code.get('definition', 'Not specified')}")

        # Format inclusion examples
        inclusion_examples = code.get('inclusion_examples', [])
        if inclusion_examples:
            parts.append("  Inclusion examples:")
            for example in inclusion_examples:
                parts.append(f"    • {example}")
        else:
            parts.append("  Inclusion examples: Not specified")

        # Format exclusion examples
        exclusion_examples = code.get('exclusion_examples', [])
        if exclusion_examples:
            parts.append("  Exclusion examples:")
            for example in exclusion_examples:
                parts.append(f"    • {example}")
        else:
            parts.append("  Exclusion examples: Not specified")

        # Format near neighbor and tell-apart rule
        near_neighbor = code.get('near_neighbor_label')
        tell_apart = code.get('tell_apart_rule')
        if near_neighbor and tell_apart:
            parts.append(f"  Near neighbor: {near_neighbor} (Tell apart: {tell_apart})")
        elif near_neighbor:
            parts.append(f"  Near neighbor: {near_neighbor}")
        else:
            parts.append("  Near neighbor: Not specified")

        return '\n'.join(parts)

    def _parse_response_json(self, response) -> dict:
        """Extract and parse JSON from GPT-5 response"""
        import json
        response_text = response.output_text
        return json.loads(response_text)

    def _convert_to_refined_model(self, response_data: dict, id_to_cluster_map: dict) -> RefinedCodebookModel:
        """Convert parsed JSON to RefinedCodebookModel with ID mapping"""
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
                        # Handle merged IDs from GPT-5 (e.g., "2,3" → "8,12")
                        sequential_id = subcode_data.get('id', '')

                        if not sequential_id:
                            self.reporter.warning(f"    Code '{subcode_data.get('code')}' has no ID - cannot map to source_cluster")
                            source_cluster = ''
                        elif ',' in sequential_id:
                            # GPT-5 merged multiple codes - split and look up each cluster
                            id_parts = [id.strip() for id in sequential_id.split(',')]
                            cluster_parts = [id_to_cluster_map.get(id, '') for id in id_parts]
                            # Filter out empty values and join
                            source_cluster = ','.join([c for c in cluster_parts if c])
                            if False: #debug
                                self.reporter.debug(f"    Merged ID '{sequential_id}' → clusters '{source_cluster}'")
                            if not source_cluster:
                                self.reporter.warning(f"    Failed to map IDs '{sequential_id}' to any clusters")
                        else:
                            # Single ID, direct lookup
                            source_cluster = id_to_cluster_map.get(sequential_id, '')
                            if False: #debug 
                                self.reporter.debug(f"    Single ID '{sequential_id}' → cluster '{source_cluster}'")
                            if not source_cluster:
                                self.reporter.warning(f"    ID '{sequential_id}' not found in id_to_cluster_map")

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

        return refined_model

    def _call_refinement_llm(self, survey_question: str, raw_codes: List[dict]) -> RefinedCodebookModel:
        """Call GPT-5 for codebook refinement using simple sync call"""
        from openai import OpenAI

        # Build mapping from sequential ID to source_cluster_id
        id_to_cluster_map = {code['id']: code.get('source_cluster_id', '') for code in raw_codes}

        # Format codes for prompt with IDs and assignment_examples
        formatted_codes = '\n\n'.join([self._format_code_with_assignment_examples(code) for code in raw_codes])

        # Create prompt
        prompt = CODEBOOK_REFINEMENT_PROMPT.format(
            language=self.config.language,
            survey_question=survey_question,
            raw_codes=formatted_codes)

        self.reporter.info(f"Calling {self.model_config.codebook_refinement_model} for refinement")

        # Capture prompt if enabled
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="step_7_codebook_refinement",
                utility_name="codebookRefinement",
                prompt_content=prompt,
                prompt_type="gpt5_refinement",
                metadata={
                    'model': self.model_config.codebook_refinement_model,
                    'raw_code_count': len(raw_codes),
                    'language': self.config.language
                }
            )

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

            # Parse response and convert to model
            response_data = self._parse_response_json(response)
            refined_model = self._convert_to_refined_model(response_data, id_to_cluster_map)
            
            self.reporter.info(f"LLM call successful: {len(refined_model.refined_codebook)} categories generated")
            
            return refined_model
            
        except Exception as e:
            self.reporter.error(f"LLM call failed: {str(e)}")
            raise

    def _create_overlapping_batches(self, raw_codes: List[dict]) -> List[List[dict]]:
        """Split codes into sequential batches with 10% overlap at boundaries"""

        batch_size = self.config.target_batch_size
        overlap_size = math.ceil(batch_size * self.config.overlap_percentage)
        stride = batch_size - overlap_size

        batches = []
        start_idx = 0

        while start_idx < len(raw_codes):
            end_idx = min(start_idx + batch_size, len(raw_codes))
            batch = raw_codes[start_idx:end_idx].copy()

            # Mark overlap codes (codes that were in previous batch)
            if start_idx > 0:
                for i in range(min(overlap_size, len(batch))):
                    batch[i] = {**batch[i], 'is_boundary': True}

            batches.append(batch)
            start_idx += stride

        self.reporter.stat_line(f"Created {len(batches)} batches with {overlap_size}-code overlap")
        for i, b in enumerate(batches):
            boundary_count = sum(1 for c in b if c.get('is_boundary', False))
            self.reporter.stat_line(f"  Batch {i}: {len(b)} codes ({boundary_count} boundary)")

        return batches

    def _refine_batch_map(self, survey_question: str, batch: List[dict], batch_id: int) -> RefinedCodebookModel:
        """MAP: Refine single batch using existing CODEBOOK_REFINEMENT_PROMPT"""

        # Build ID mapping
        id_to_cluster_map = {code['id']: code.get('source_cluster_id', '') for code in batch}

        # Format codes
        formatted_codes = '\n\n'.join([
            self._format_code_with_assignment_examples(code)
            for code in batch
        ])

        # Use existing CODEBOOK_REFINEMENT_PROMPT with optional subset note
        prompt = CODEBOOK_REFINEMENT_PROMPT.format(
            language=self.config.language,
            survey_question=survey_question,
            raw_codes=formatted_codes
        )

        # Add subset context note
        subset_note = "\n\n**NOTE**: This is a subset of the full dataset. Some codes from adjacent subsets may relate to themes here."
        prompt = prompt.replace("Begin now.", subset_note + "\n\nBegin now.")

        self.reporter.stat_line(f"MAP: Refining batch {batch_id} ({len(batch)} codes)")

        # Capture prompt if enabled
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name=f"step_7_map_batch_{batch_id}",
                utility_name="codebookRefinement_MAP",
                prompt_content=prompt,
                prompt_type="map_refinement",
                metadata={'batch_id': batch_id, 'batch_size': len(batch)}
            )

        # Call GPT-5
        client = OpenAI(api_key=self.api_key)

        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"}
        )

        # Parse response (use existing parse logic)
        parsed_json = self._parse_response_json(response)
        return self._convert_to_refined_model(parsed_json, id_to_cluster_map)

    def _format_codebooks_for_reduce(self, map_results: List[Dict]) -> str:
        """Format multiple codebooks from MAP phase for REDUCE prompt"""

        formatted_codebooks = []

        for i, mr in enumerate(map_results):
            result = mr['result']
            batch_id = mr['batch_id']

            parts = [f"=== CODEBOOK {i+1} (from Batch {batch_id}) ==="]
            parts.append(f"Total themes: {len(result.refined_codebook)}")
            parts.append("")

            for theme in result.refined_codebook:
                parts.append(f"Theme: {theme.category}")
                parts.append("  Codes under this theme:")
                for subcode in theme.subcodes:
                    parts.append(f"    - [{subcode.id}] {subcode.code}: {subcode.description}")

                    # Mark boundary codes
                    code_ids = subcode.id.split(',')
                    matching_originals = [
                        c for c in mr['original_codes']
                        if c['id'] in code_ids and c.get('is_boundary', False)
                    ]
                    if matching_originals:
                        parts.append("      (⚠️ boundary code - may appear in adjacent codebooks)")

                parts.append("")

            formatted_codebooks.append('\n'.join(parts))

        return '\n\n'.join(formatted_codebooks)

    def _merge_codebooks_reduce(self, survey_question: str, map_results: List[Dict]) -> RefinedCodebookModel:
        """REDUCE: Merge multiple codebooks into final unified codebook"""

        # Format codebooks for prompt
        codebooks_summary = self._format_codebooks_for_reduce(map_results)

        # Create REDUCE prompt
        prompt = CODEBOOK_MERGE_PROMPT.format(
            language=self.config.language,
            survey_question=survey_question,
            codebooks_summary=codebooks_summary,
            n_codebooks=len(map_results)
        )

        self.reporter.stat_line(f"REDUCE: Merging {len(map_results)} codebooks")

        # Capture prompt if enabled
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="step_7_reduce",
                utility_name="codebookRefinement_REDUCE",
                prompt_content=prompt,
                prompt_type="reduce_codebook_merge",
                metadata={'codebook_count': len(map_results)}
            )

        # Call GPT-5
        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"}
        )

        # Build comprehensive ID mapping from all batches
        combined_id_map = {}
        for mr in map_results:
            for code in mr['original_codes']:
                seq_id = code['id']
                cluster_id = code.get('source_cluster_id', '')
                combined_id_map[seq_id] = cluster_id
                if False: #debug  
                    self.reporter.debug(f"  ID map: {seq_id} → {cluster_id}")

        if False: #debug  
             self.reporter.debug(f"Built combined_id_map with {len(combined_id_map)} entries")

        # Parse response
        parsed_json = self._parse_response_json(response)
        return self._convert_to_refined_model(parsed_json, combined_id_map)

    def _refine_hierarchically(self, survey_question: str, raw_codes: List[dict]) -> CodeRefinementResults:
        """Hierarchical MAP-REDUCE refinement orchestrator"""

        start_time = datetime.now()

        # STEP 1: Create overlapping batches
        self.reporter.stat_line("=== Creating overlapping batches ===")
        batches = self._create_overlapping_batches(raw_codes)

        # STEP 2: MAP Phase - Refine each batch
        self.reporter.stat_line(f"=== MAP Phase: Processing {len(batches)} batches ===")
        map_results = []
        for i, batch in enumerate(batches):
            batch_result = self._refine_batch_map(survey_question, batch, i)
            map_results.append({
                'batch_id': i,
                'result': batch_result,
                'original_codes': batch
            })

            theme_count = len(batch_result.refined_codebook)
            self.reporter.stat_line(f"  Batch {i} complete: {len(batch)} codes → {theme_count} themes")

        # Count total themes from MAP
        total_map_themes = sum(len(mr['result'].refined_codebook) for mr in map_results)
        self.reporter.stat_line(f"MAP Phase complete: {total_map_themes} total themes across {len(batches)} codebooks")

        # STEP 3: REDUCE Phase - Merge codebooks
        self.reporter.stat_line(f"=== REDUCE Phase: Merging {len(map_results)} codebooks ===")
        final_result = self._merge_codebooks_reduce(survey_question, map_results)

        final_theme_count = len(final_result.refined_codebook)
        final_code_count = sum(len(theme.subcodes) for theme in final_result.refined_codebook)

        self.reporter.stat_line(f"REDUCE Phase complete: {total_map_themes} themes → {final_theme_count} final themes")

        # STEP 4: Build results
        processing_time = (datetime.now() - start_time).total_seconds()

        processing_stats = {
            'original_code_count': len(raw_codes),
            'refined_category_count': final_theme_count,
            'total_refined_subcodes': final_code_count,
            'processing_time_seconds': processing_time,
            'model_used': self.model_config.codebook_refinement_model,
            'language': self.config.language,
            'reasoning_effort': 'minimal',
            'text_verbosity': 'low',
            'batch_count': len(batches),
            'map_theme_count': total_map_themes
        }

        return CodeRefinementResults(
            original_codebook=[
                {k: v for k, v in code.items() if k not in ['is_boundary']}
                for code in raw_codes
            ],
            refined_codebook=final_result,
            processing_stats=processing_stats,
            timestamp=start_time.isoformat()
        )

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
    verbose: bool = True,
    prompt_printer=None
) -> CodeRefinementResults:

    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG

    config = CodebookRefinementConfig(
        model_config=model_config,
        language=language,
        verbose=verbose,
        prompt_printer=prompt_printer
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