import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import json
import logging
import math
import numpy as np
from datetime import datetime
from typing import List, Optional, Any, Dict, Tuple
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

# === CONSTANTS ========================================================================================================
OPENAI_EMBEDDING_DIMENSION = 1536     # OpenAI embedding vector size

from config import ModelConfig, DEFAULT_MODEL_CONFIG, DEFAULT_LANGUAGE, OPENAI_API_KEY, API_PROVIDER, AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER
from utils.llm import create_client, llm_create_sync


# =============================================================================
# LLM Response Models (match exact prompt output format)
# =============================================================================
class LLMCodeItem(BaseModel):
    """Response model for a single code in LLM output (uses 'code' not 'subcode')"""
    id: str = Field(..., description="Original code ID or comma-separated IDs if merged")
    code: str = Field(..., description="Code label")
    description: str = Field(..., description="Code description (≤20 words)")
    category: str = Field(default="", description="Empty for 2-level, category name for 3-level")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMThemeItem(BaseModel):
    """Response model for a theme in LLM output (uses 'theme' and 'codes')"""
    theme: str = Field(..., description="Main theme label")
    codes: List[LLMCodeItem] = Field(..., description="List of codes under this theme")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMRefinementResponse(BaseModel):
    """Response model matching exact LLM output format for codebook refinement"""
    analysis: str = Field(..., description="Detailed analysis of refinement decisions")
    refined_codebook: List[LLMThemeItem] = Field(..., description="Refined codebook with themes and codes")
    model_config = ConfigDict(arbitrary_types_allowed=True)
try:
    from .prompts_exp import CODEBOOK_REFINEMENT_PROMPT, CODEBOOK_MERGE_PROMPT
except ImportError:
    from prompts_exp import CODEBOOK_REFINEMENT_PROMPT, CODEBOOK_MERGE_PROMPT
from experiments.models_exp import (
    RefinedCodebookModel, CodeRefinementResults, RefinedSubcode, RefinedCodebookCategory,
    CodeTransformation, BatchTransformationRecord, RefinementLineage
)
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
    target_batch_size: int = 10
    overlap_size: int = 1

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

        # Initialize centralized client for Azure/OpenAI abstraction
        self.client = create_client(
            model=self.model_config.codebook_refinement_model,
            async_mode=False,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER if API_PROVIDER == "azure" else None
        )

        # Setup verbose reporter
        self.reporter = VerboseReporter(enabled=config.verbose)

        # Setup prompt printer
        self.prompt_printer = config.prompt_printer

        logger.info(f"Initialized CodebookRefinementProcessor with model: {self.model_config.codebook_refinement_model}")
    
    def refine_codebook(self, survey_question: str, reasoning_results: CodeGeneratorReasoningResults) -> CodeRefinementResults:
        """Main entry point - decides between single-batch or MAP-REDUCE"""

        # Store reasoning_results for access in batching methods
        self.reasoning_results = reasoning_results

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

        # Build master ID map for lineage tracking
        master_id_map = {
            code['id']: code.get('source_cluster_id', '')
            for code in raw_codes
        }

        # Initialize lineage tracking
        lineage = RefinementLineage(
            original_codes=raw_codes,
            master_id_to_cluster_map=master_id_map,
            map_batches=[],
            timestamp=start_time.isoformat()
        )

        # Use MAP method for single batch (same prompt, no REDUCE needed)
        refined_model, batch_record = self._refine_batch_map(survey_question, raw_codes, batch_id=0)
        lineage.map_batches.append(batch_record)

        # Reconcile orphaned clusters (single batch has no REDUCE phase)
        self.reporter.stat_line("=== Cluster Reconciliation ===")
        try:
            lineage = self._reconcile_orphaned_clusters(lineage, refined_model)

            # Apply reconciliation to patch source_clusters
            if lineage.reconciled_mappings:
                refined_model = self._apply_reconciliation(refined_model, lineage)
        except Exception as e:
            self.reporter.warning(f"  Reconciliation failed: {str(e)} - continuing without reconciliation")
            logger.error(f"Reconciliation error: {str(e)}", exc_info=True)

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
                'text_verbosity': 'low',
                'orphaned_clusters': len(lineage.orphaned_clusters),
                'reconciled_clusters': len(lineage.reconciled_mappings)
            },
            timestamp=start_time.isoformat(),
            lineage=lineage
        )
    
    def _extract_raw_codes(self, reasoning_results: CodeGeneratorReasoningResults) -> List[dict]:
        """Extract raw codes with IDs and assignment_examples from reasoning results

        Returns:
            List of dicts with 'id', 'code', 'definition', 'source_cluster_id',
            'inclusion_examples', 'exclusion_examples', 'near_neighbor_label', 'tell_apart_rule' fields
        """
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

    def _extract_json_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks if present"""
        text = text.strip()

        # Check for ```json...``` or ```...``` wrappers
        if text.startswith('```'):
            lines = text.split('\n')
            if len(lines) >= 3 and lines[-1].strip() == '```':
                # Remove first and last line (code block markers)
                return '\n'.join(lines[1:-1])

        return text

    def _parse_response_json(self, response) -> dict:
        """Extract and parse JSON from response (handles both OpenAI and Azure formats)"""
        response_text = None

        # Try OpenAI Responses API format first (response.output array)
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == "message":
                    response_text = item.content[0].text
                    break

        # Try Azure Chat Completions format (response.choices[0].message.content)
        if response_text is None and hasattr(response, 'choices') and response.choices:
            response_text = response.choices[0].message.content

        if response_text is None:
            raise ValueError("No message content found in response")

        # Strip markdown wrappers
        clean_text = self._extract_json_from_markdown(response_text)

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            # Enhanced error logging for debugging
            self.reporter.error(f"JSON parse error: {str(e)}")
            self.reporter.error(f"Response length: {len(response_text)} characters")
            if len(response_text) > 1000:
                self.reporter.error(f"First 300 chars: {response_text[:300]}")
                self.reporter.error(f"Last 300 chars: {response_text[-300:]}")
            raise

    def _convert_to_refined_model(self, response_data: dict, id_to_cluster_map: dict) -> RefinedCodebookModel:
        """Convert parsed JSON to RefinedCodebookModel with ID mapping"""
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

                        # Sanitize ID: strip brackets, whitespace (defensive against GPT-5 formatting)
                        if sequential_id:
                            sequential_id = sequential_id.strip().strip('[]')

                        if not sequential_id:
                            self.reporter.warning(f"    Code '{subcode_data.get('code')}' has no ID - cannot map to source_cluster")
                            source_cluster = ''
                        elif ',' in sequential_id:
                            # GPT-5 merged multiple codes - split and look up each cluster
                            id_parts = [id.strip() for id in sequential_id.split(',')]
                            cluster_parts = [id_to_cluster_map.get(id, '') for id in id_parts]
                            # Filter out empty values and join
                            source_cluster = ','.join([c for c in cluster_parts if c])
                            logger.debug(f"Merged ID '{sequential_id}' → clusters '{source_cluster}'")
                            if not source_cluster:
                                self.reporter.warning(f"    Failed to map IDs '{sequential_id}' to any clusters")
                        else:
                            # Single ID, direct lookup
                            source_cluster = id_to_cluster_map.get(sequential_id, '')
                            logger.debug(f"Single ID '{sequential_id}' → cluster '{source_cluster}'")
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
                'model': self.model_config.codebook_refinement_model,
                'reasoning_effort': "minimal",
                'text_verbosity': "low",
                'timestamp': datetime.now().isoformat()
            }
        )

        return refined_model

    def _convert_llm_response_to_model(self, response: LLMRefinementResponse, id_to_cluster_map: dict) -> RefinedCodebookModel:
        """Convert LLMRefinementResponse (instructor output) to RefinedCodebookModel.

        Args:
            response: Validated LLMRefinementResponse from instructor
            id_to_cluster_map: Mapping from sequential code ID to source cluster ID

        Returns:
            RefinedCodebookModel with properly mapped cluster IDs
        """
        categories = []

        for theme_item in response.refined_codebook:
            try:
                subcodes = []
                for code_item in theme_item.codes:
                    # Map sequential ID back to source_cluster_id
                    sequential_id = code_item.id.strip().strip('[]') if code_item.id else ''

                    if not sequential_id:
                        self.reporter.warning(f"    Code '{code_item.code}' has no ID - cannot map to source_cluster")
                        source_cluster = ''
                    elif ',' in sequential_id:
                        # Merged codes - split and look up each cluster
                        id_parts = [id.strip() for id in sequential_id.split(',')]
                        cluster_parts = [id_to_cluster_map.get(id, '') for id in id_parts]
                        source_cluster = ','.join([c for c in cluster_parts if c])
                        if not source_cluster:
                            self.reporter.warning(f"    Failed to map IDs '{sequential_id}' to any clusters")
                    else:
                        source_cluster = id_to_cluster_map.get(sequential_id, '')
                        if not source_cluster:
                            self.reporter.warning(f"    ID '{sequential_id}' not found in id_to_cluster_map")

                    subcode = RefinedSubcode(
                        id=sequential_id,
                        code=code_item.code,
                        description=code_item.description,
                        category=code_item.category or '',
                        source_cluster=source_cluster
                    )
                    subcodes.append(subcode)

                category = RefinedCodebookCategory(
                    category=theme_item.theme,
                    subcodes=subcodes
                )
                categories.append(category)

            except Exception as e:
                self.reporter.error(f"Failed to convert theme: {e}")
                continue

        return RefinedCodebookModel(
            analysis=response.analysis,
            refined_codebook=[cat.model_dump() for cat in categories],
            generation_metadata={
                'model': self.model_config.codebook_refinement_model,
                'reasoning_effort': "minimal",
                'text_verbosity': "low",
                'timestamp': datetime.now().isoformat()
            }
        )

    def _call_refinement_llm(self, survey_question: str, raw_codes: List[dict]) -> RefinedCodebookModel:
        """Call GPT-5 for codebook refinement using simple sync call"""
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

        try:
            model_name = self.model_config.codebook_refinement_model
            model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")

            # Get temperature for chat models (reasoning models use defaults in llm.py)
            temperature = self.model_config.get_temperature_for_stage('refinement') if model_type == "chat" else 0.0

            # Use LLMRefinementResponse for structured output with instructor
            response = llm_create_sync(
                client=self.client,
                model=model_name,
                prompt=prompt,
                response_model=LLMRefinementResponse,
                temperature=temperature,
                max_tokens=self.model_config.default_max_tokens,
                track_usage=True
            )

            # Convert LLM response to internal model format
            refined_model = self._convert_llm_response_to_model(response, id_to_cluster_map)

            self.reporter.info(f"LLM call successful: {len(refined_model.refined_codebook)} categories generated")

            return refined_model

        except Exception as e:
            self.reporter.error(f"LLM call failed: {str(e)}")
            raise

    def _extract_code_embeddings(self, raw_codes: List[dict]) -> np.ndarray:
        """Extract and average embeddings for each code from their source clusters"""
        code_embeddings = []

        for code in raw_codes:
            cluster_id = code.get('source_cluster_id', '')
            if not cluster_id:
                # Fallback: zero vector if no cluster ID
                code_embeddings.append(np.zeros(OPENAI_EMBEDDING_DIMENSION))  # OpenAI embedding dim
                continue

            # Find cluster in reasoning_results (cluster_results is List[Dict])
            cluster_found = False
            for cluster_result in self.reasoning_results.cluster_results:
                # cluster_result is a dict with 'response_ideas' key
                response_ideas = cluster_result.get('response_ideas', [])
                if response_ideas:
                    first_idea = response_ideas[0]
                    # first_idea might be dict or ClusterSubmodel, handle both
                    if isinstance(first_idea, dict):
                        result_cluster_id = (first_idea.get('expanded_cluster')
                                           if first_idea.get('expanded_cluster')
                                           else str(first_idea.get('initial_cluster', '')))
                    else:
                        result_cluster_id = (first_idea.expanded_cluster
                                           if first_idea.expanded_cluster
                                           else str(first_idea.initial_cluster))

                    if str(result_cluster_id) == str(cluster_id):
                        # Extract all embeddings from this cluster
                        embeddings = []
                        for idea in response_ideas:
                            if isinstance(idea, dict):
                                emb = idea.get('idea_embedding')
                            else:
                                emb = idea.idea_embedding
                            if emb is not None:
                                embeddings.append(emb)

                        if embeddings:
                            # Average embeddings
                            avg_embedding = np.mean(embeddings, axis=0)
                            code_embeddings.append(avg_embedding)
                            cluster_found = True
                            break

            if not cluster_found:
                # Fallback: zero vector
                code_embeddings.append(np.zeros(OPENAI_EMBEDDING_DIMENSION))

        return np.array(code_embeddings)

    def _create_similarity_batches(self, raw_codes: List[dict]) -> List[List[dict]]:
        """Create batches by clustering codes using cosine similarity of embeddings"""

        batch_size = self.config.target_batch_size
        overlap_size = self.config.overlap_size

        # Extract embeddings for all codes
        embeddings = self._extract_code_embeddings(raw_codes)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms

        # Hierarchical clustering
        n_batches = math.ceil(len(raw_codes) / batch_size)
        linkage_matrix = linkage(normalized_embeddings, method='ward')
        cluster_labels = fcluster(linkage_matrix, n_batches, criterion='maxclust')

        # Group codes by cluster label
        clusters = {}
        for code, label in zip(raw_codes, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(code)

        # Split large clusters and create final batches with overlap
        batches = []
        for cluster_codes in clusters.values():
            # Split cluster if larger than batch_size
            for i in range(0, len(cluster_codes), batch_size):
                batch = cluster_codes[i:i + batch_size]
                batches.append(batch)

        # Add 1-code overlap between consecutive batches (left boundaries)
        final_batches = []
        for i, batch in enumerate(batches):
            if i > 0:
                # Add last code from previous batch as first code (left boundary)
                overlap_code = {**batches[i-1][-1], 'is_boundary': True, 'boundary_type': 'left'}
                final_batch = [overlap_code] + batch
            else:
                final_batch = batch
            final_batches.append(final_batch)

        # Add right boundaries
        for i in range(len(final_batches) - 1):
            # Add first non-boundary code from next batch as last code (right boundary)
            next_batch = final_batches[i+1]
            # Get first non-boundary code from next batch
            first_code_idx = 1 if i < len(final_batches) - 2 else 0
            if first_code_idx < len(next_batch):
                right_boundary = {**next_batch[first_code_idx], 'is_boundary': True, 'boundary_type': 'right'}
                final_batches[i].append(right_boundary)

        # Log batch creation
        self.reporter.stat_line(f"Created {len(final_batches)} similarity-based batches with {overlap_size}-code overlap (two-sided)")
        for i, b in enumerate(final_batches):
            left_boundaries = sum(1 for c in b if c.get('is_boundary', False) and c.get('boundary_type') == 'left')
            right_boundaries = sum(1 for c in b if c.get('is_boundary', False) and c.get('boundary_type') == 'right')
            cluster_ids = set(c.get('source_cluster_id', '') for c in b if c.get('source_cluster_id'))
            self.reporter.stat_line(f"  Batch {i}: {len(b)} codes ({left_boundaries}L+{right_boundaries}R boundary, {len(cluster_ids)} clusters)")

        return final_batches

    def _refine_batch_map(self, survey_question: str, batch: List[dict], batch_id: int) -> Tuple[RefinedCodebookModel, BatchTransformationRecord]:
        """MAP: Refine single batch using existing CODEBOOK_REFINEMENT_PROMPT

        Returns:
            Tuple of (refined codebook model, batch transformation record for lineage tracking)
        """

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
        subset_note = "\n\n**NOTE**: This batch contains semantically similar codes clustered by cosine similarity. Adjacent batches may contain related themes."
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

        # Call LLM with centralized client
        model_name = self.model_config.codebook_refinement_model
        model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")

        # Get temperature for chat models
        temperature = self.model_config.get_temperature_for_stage('refinement') if model_type == "chat" else 0.0

        response = llm_create_sync(
            client=self.client,
            model=model_name,
            prompt=prompt,
            response_model=LLMRefinementResponse,
            temperature=temperature,
            max_tokens=self.model_config.default_max_tokens,
            track_usage=True
        )

        # Convert LLM response to internal model format
        result = self._convert_llm_response_to_model(response, id_to_cluster_map)

        # Track transformations for lineage
        input_ids = {c['id'] for c in batch if not c.get('is_boundary', False)}
        transformations = []
        referenced_ids = set()

        for theme in result.refined_codebook:
            for subcode in theme.subcodes:
                id_parts = [p.strip() for p in subcode.id.split(',') if p.strip()]
                referenced_ids.update(id_parts)

                transformations.append(CodeTransformation(
                    phase="MAP",
                    batch_id=batch_id,
                    transformation_type="MERGED" if len(id_parts) > 1 else "PRESERVED",
                    input_ids=id_parts,
                    output_id=subcode.id,
                    source_cluster_ids=[id_to_cluster_map.get(i, '') for i in id_parts],
                    final_code_label=subcode.code
                ))

        # Track DROPPED codes
        dropped_ids = list(input_ids - referenced_ids)
        for dropped_id in dropped_ids:
            transformations.append(CodeTransformation(
                phase="MAP",
                batch_id=batch_id,
                transformation_type="DROPPED",
                input_ids=[dropped_id],
                output_id=None,
                source_cluster_ids=[id_to_cluster_map.get(dropped_id, '')]
            ))

        # Build batch record
        batch_record = BatchTransformationRecord(
            batch_id=batch_id,
            input_ids=list(input_ids),
            input_cluster_map={k: v for k, v in id_to_cluster_map.items() if k in input_ids},
            output_ids=[subcode.id for theme in result.refined_codebook for subcode in theme.subcodes],
            transformations=transformations,
            dropped_ids=dropped_ids
        )

        return result, batch_record

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

    def _merge_codebooks_reduce(self, survey_question: str, map_results: List[Dict], id_to_cluster_map: dict) -> Tuple[RefinedCodebookModel, BatchTransformationRecord]:
        """REDUCE: Merge multiple codebooks into final unified codebook

        Returns:
            Tuple of (refined codebook model, reduce transformation record for lineage tracking)
        """

        # Format codebooks for prompt
        codebooks_summary = self._format_codebooks_for_reduce(map_results)

        # Collect all input IDs from MAP phase outputs
        map_output_ids = set()
        map_id_to_cluster = {}
        for mr in map_results:
            for theme in mr['result'].refined_codebook:
                for subcode in theme.subcodes:
                    map_output_ids.add(subcode.id)
                    # Map the output ID to its source clusters
                    if subcode.source_cluster:
                        map_id_to_cluster[subcode.id] = subcode.source_cluster

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

        # Call LLM with centralized client
        model_name = self.model_config.codebook_refinement_model
        model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")

        # Get temperature for chat models
        temperature = self.model_config.get_temperature_for_stage('refinement') if model_type == "chat" else 0.0

        response = llm_create_sync(
            client=self.client,
            model=model_name,
            prompt=prompt,
            response_model=LLMRefinementResponse,
            temperature=temperature,
            max_tokens=self.model_config.default_max_tokens,
            track_usage=True
        )

        # Use id_to_cluster_map passed from caller (built from original raw_codes)
        # This preserves ALL cluster IDs, including those dropped/merged in MAP phase
        result = self._convert_llm_response_to_model(response, id_to_cluster_map)

        # Track REDUCE transformations
        transformations = []
        referenced_ids = set()

        for theme in result.refined_codebook:
            for subcode in theme.subcodes:
                id_parts = [p.strip() for p in subcode.id.split(',') if p.strip()]
                referenced_ids.update(id_parts)

                transformations.append(CodeTransformation(
                    phase="REDUCE",
                    batch_id=None,
                    transformation_type="MERGED" if len(id_parts) > 1 else "PRESERVED",
                    input_ids=id_parts,
                    output_id=subcode.id,
                    source_cluster_ids=[id_to_cluster_map.get(i, '') for i in id_parts],
                    final_code_label=subcode.code
                ))

        # Track DROPPED codes from REDUCE phase
        dropped_ids = list(map_output_ids - referenced_ids)
        for dropped_id in dropped_ids:
            transformations.append(CodeTransformation(
                phase="REDUCE",
                batch_id=None,
                transformation_type="DROPPED",
                input_ids=[dropped_id],
                output_id=None,
                source_cluster_ids=[map_id_to_cluster.get(dropped_id, '')]
            ))

        # Build reduce record
        reduce_record = BatchTransformationRecord(
            batch_id=-1,  # Sentinel for REDUCE phase
            input_ids=list(map_output_ids),
            input_cluster_map=map_id_to_cluster,
            output_ids=[subcode.id for theme in result.refined_codebook for subcode in theme.subcodes],
            transformations=transformations,
            dropped_ids=dropped_ids
        )

        return result, reduce_record

    def _reconcile_orphaned_clusters(self, lineage: RefinementLineage, final_codebook: RefinedCodebookModel) -> RefinementLineage:
        """Find orphaned clusters and map them to destination codes.

        Args:
            lineage: The transformation lineage with map_batches and reduce_record
            final_codebook: The final refined codebook

        Returns:
            Updated lineage with orphaned_clusters and reconciled_mappings populated
        """
        # Get expected clusters (all clusters from original codes)
        expected = set(lineage.master_id_to_cluster_map.values()) - {''}

        # Get actual clusters (all clusters in final codebook)
        actual = set()
        for theme in final_codebook.refined_codebook:
            for subcode in theme.subcodes:
                if subcode.source_cluster:
                    for cid in subcode.source_cluster.split(','):
                        actual.add(cid.strip())

        # Find orphaned clusters
        orphaned = expected - actual
        lineage.orphaned_clusters = list(orphaned)

        if not orphaned:
            self.reporter.stat_line("  ✓ All clusters preserved - no reconciliation needed")
            return lineage

        self.reporter.stat_line(f"  Orphaned clusters: {len(orphaned)}")

        # For each orphaned cluster, trace its transformation path
        for orphaned_cluster in orphaned:
            # Find original code with this cluster
            original_id = None
            original_code_label = None
            for code in lineage.original_codes:
                if code.get('source_cluster_id') == orphaned_cluster:
                    original_id = code['id']
                    original_code_label = code.get('code', 'unknown')
                    break

            if not original_id:
                self.reporter.warning(f"    Cluster {orphaned_cluster}: No original code found")
                continue

            # Find what happened to this code in MAP phase
            reconciled = False
            for batch in lineage.map_batches:
                for transform in batch.transformations:
                    if original_id in transform.input_ids:
                        if transform.transformation_type == "MERGED":
                            # Find which cluster survived the merge
                            for cid in transform.source_cluster_ids:
                                if cid != orphaned_cluster and cid in actual:
                                    lineage.reconciled_mappings[orphaned_cluster] = cid
                                    self.reporter.stat_line(
                                        f"    Cluster {orphaned_cluster} ('{original_code_label}'): "
                                        f"MERGED → reconciled to {cid}"
                                    )
                                    reconciled = True
                                    break
                        elif transform.transformation_type == "DROPPED":
                            # Use embedding similarity to find nearest code
                            nearest = self._find_nearest_by_embedding(orphaned_cluster, final_codebook, actual)
                            if nearest:
                                lineage.reconciled_mappings[orphaned_cluster] = nearest
                                self.reporter.stat_line(
                                    f"    Cluster {orphaned_cluster} ('{original_code_label}'): "
                                    f"DROPPED → reconciled to {nearest} (by embedding)"
                                )
                                reconciled = True
                            else:
                                self.reporter.warning(
                                    f"    Cluster {orphaned_cluster} ('{original_code_label}'): "
                                    f"DROPPED → no suitable match found"
                                )
                        break
                if reconciled:
                    break

        return lineage

    def _find_nearest_by_embedding(self, orphaned_cluster: str, final_codebook: RefinedCodebookModel, actual_clusters: set) -> Optional[str]:
        """Find the nearest surviving cluster by embedding similarity.

        Args:
            orphaned_cluster: The cluster ID that was lost
            final_codebook: The final refined codebook
            actual_clusters: Set of cluster IDs that exist in the final codebook

        Returns:
            The cluster ID of the best matching surviving code, or None if no good match
        """
        # Check if we have access to reasoning_results with cluster_data
        if not hasattr(self, 'reasoning_results') or not self.reasoning_results:
            return None

        if not hasattr(self.reasoning_results, 'cluster_data') or not self.reasoning_results.cluster_data:
            return None

        # Get embeddings for the orphaned cluster
        cluster_data = self.reasoning_results.cluster_data
        if orphaned_cluster not in cluster_data:
            return None

        orphaned_embeddings = cluster_data[orphaned_cluster].get('embeddings', [])
        if not orphaned_embeddings:
            return None

        # Compute centroid of orphaned cluster
        orphaned_centroid = np.mean(orphaned_embeddings, axis=0)

        # Find best matching cluster from actual clusters
        best_match = None
        best_similarity = 0.0
        similarity_threshold = 0.70

        for candidate_cluster in actual_clusters:
            if candidate_cluster not in cluster_data:
                continue

            candidate_embeddings = cluster_data[candidate_cluster].get('embeddings', [])
            if not candidate_embeddings:
                continue

            # Compute centroid of candidate cluster
            candidate_centroid = np.mean(candidate_embeddings, axis=0)

            # Compute cosine similarity
            similarity = np.dot(orphaned_centroid, candidate_centroid) / (
                np.linalg.norm(orphaned_centroid) * np.linalg.norm(candidate_centroid)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate_cluster

        if best_similarity >= similarity_threshold:
            return best_match
        return None

    def _apply_reconciliation(self, codebook: RefinedCodebookModel, lineage: RefinementLineage) -> RefinedCodebookModel:
        """Patch source_cluster fields with reconciled mappings.

        Args:
            codebook: The refined codebook to patch
            lineage: The lineage with reconciled_mappings

        Returns:
            The codebook with patched source_cluster fields
        """
        if not lineage.reconciled_mappings:
            return codebook

        # Build cluster -> subcode lookup
        cluster_to_subcode = {}
        for theme in codebook.refined_codebook:
            for subcode in theme.subcodes:
                if subcode.source_cluster:
                    for cid in subcode.source_cluster.split(','):
                        cluster_to_subcode[cid.strip()] = subcode

        # Append orphaned clusters to their target codes
        patched_count = 0
        for orphaned, target in lineage.reconciled_mappings.items():
            if target in cluster_to_subcode:
                subcode = cluster_to_subcode[target]
                old_source = subcode.source_cluster
                subcode.source_cluster = f"{subcode.source_cluster},{orphaned}"
                patched_count += 1
                self.reporter.stat_line(
                    f"    ✓ Patched: '{subcode.code[:40]}' source_cluster: "
                    f"'{old_source}' → '{subcode.source_cluster}'"
                )

        self.reporter.stat_line(f"  Reconciliation complete: {patched_count} clusters patched")
        return codebook

    def _refine_hierarchically(self, survey_question: str, raw_codes: List[dict]) -> CodeRefinementResults:
        """Hierarchical MAP-REDUCE refinement orchestrator"""

        start_time = datetime.now()

        # Build master ID map from original raw_codes (before batching)
        # This preserves ALL cluster IDs, including those that may be merged/dropped in MAP phase
        master_id_map = {
            code['id']: code.get('source_cluster_id', '')
            for code in raw_codes
        }

        # Initialize lineage tracking
        lineage = RefinementLineage(
            original_codes=raw_codes,
            master_id_to_cluster_map=master_id_map,
            map_batches=[],
            timestamp=start_time.isoformat()
        )

        # STEP 1: Create similarity-based batches
        self.reporter.stat_line("=== Creating similarity-based batches ===")
        batches = self._create_similarity_batches(raw_codes)

        # STEP 2: MAP Phase - Refine each batch with tracking
        self.reporter.stat_line(f"=== MAP Phase: Processing {len(batches)} batches ===")
        map_results = []
        for i, batch in enumerate(batches):
            batch_result, batch_record = self._refine_batch_map(survey_question, batch, i)
            lineage.map_batches.append(batch_record)
            map_results.append({
                'batch_id': i,
                'result': batch_result,
                'original_codes': batch
            })

            theme_count = len(batch_result.refined_codebook)
            merged = sum(1 for t in batch_record.transformations if t.transformation_type == "MERGED")
            dropped = len(batch_record.dropped_ids)
            self.reporter.stat_line(f"  Batch {i} complete: {len(batch)} codes → {theme_count} themes ({merged} merged, {dropped} dropped)")

        # Count total themes from MAP
        total_map_themes = sum(len(mr['result'].refined_codebook) for mr in map_results)
        self.reporter.stat_line(f"MAP Phase complete: {total_map_themes} total themes across {len(batches)} codebooks")

        # STEP 3: REDUCE Phase - Merge codebooks with tracking
        self.reporter.stat_line(f"=== REDUCE Phase: Merging {len(map_results)} codebooks ===")
        final_result, reduce_record = self._merge_codebooks_reduce(survey_question, map_results, master_id_map)
        lineage.reduce_record = reduce_record

        final_theme_count = len(final_result.refined_codebook)
        final_code_count = sum(len(theme.subcodes) for theme in final_result.refined_codebook)

        reduce_merged = sum(1 for t in reduce_record.transformations if t.transformation_type == "MERGED")
        reduce_dropped = len(reduce_record.dropped_ids)
        self.reporter.stat_line(f"REDUCE Phase complete: {total_map_themes} themes → {final_theme_count} final themes ({reduce_merged} merged, {reduce_dropped} dropped)")

        # STEP 4: Reconcile orphaned clusters
        self.reporter.stat_line("=== Cluster Reconciliation ===")
        try:
            lineage = self._reconcile_orphaned_clusters(lineage, final_result)

            # Apply reconciliation to patch source_clusters
            if lineage.reconciled_mappings:
                final_result = self._apply_reconciliation(final_result, lineage)
        except Exception as e:
            self.reporter.warning(f"  Reconciliation failed: {str(e)} - continuing without reconciliation")
            logger.error(f"Reconciliation error: {str(e)}", exc_info=True)

        # STEP 5: Build results
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
            'map_theme_count': total_map_themes,
            'orphaned_clusters': len(lineage.orphaned_clusters),
            'reconciled_clusters': len(lineage.reconciled_mappings)
        }

        return CodeRefinementResults(
            original_codebook=[
                {k: v for k, v in code.items() if k not in ['is_boundary']}
                for code in raw_codes
            ],
            refined_codebook=final_result,
            processing_stats=processing_stats,
            timestamp=start_time.isoformat(),
            lineage=lineage
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
    prompt_printer: Optional[Any] = None
) -> CodeRefinementResults:
    """Refine a raw codebook using LLM processing.

    Decides between single-batch refinement (for small codebooks) or
    MAP-REDUCE hierarchical refinement (for larger codebooks exceeding threshold).

    Args:
        survey_question: The survey question text for context
        reasoning_results: CodeGenerator results containing raw codes
        model_config: Model configuration (uses default if None)
        language: Output language code
        verbose: Enable verbose progress reporting
        prompt_printer: Optional prompt capture utility

    Returns:
        CodeRefinementResults with refined codebook structure
    """
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