import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import json
import logging
import math
import numpy as np
from collections import Counter
from datetime import datetime
from typing import List, Optional, Any, Dict, Tuple
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

# === CONSTANTS ========================================================================================================
OPENAI_EMBEDDING_DIMENSION = 1536     # OpenAI embedding vector size

from config import ModelConfig, DEFAULT_MODEL_CONFIG, DEFAULT_LANGUAGE, OPENAI_API_KEY, API_PROVIDER, AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER
import asyncio
import nest_asyncio
nest_asyncio.apply()

from aiolimiter import AsyncLimiter
from utils.llm import create_client, llm_create_sync, llm_create_async


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
    from .prompts_exp import (
        CODEBOOK_REFINEMENT_PROMPT, CODEBOOK_MERGE_PROMPT,
        CODEBOOK_MECE_ENFORCEMENT_PROMPT, MECEPartitionResult,
        PARTITION_REFINEMENT_PROMPT, CROSS_PARTITION_JUDGE_PROMPT,
        PartitionRefinementResult, CrossPartitionJudgeResult,
        PARTITION_REVIEW_PROMPT, PartitionReviewResult,
        CROSS_PARTITION_RESOLVE_PROMPT, ConflictResolutionResult,
    )
except ImportError:
    from prompts_exp import (
        CODEBOOK_REFINEMENT_PROMPT, CODEBOOK_MERGE_PROMPT,
        CODEBOOK_MECE_ENFORCEMENT_PROMPT, MECEPartitionResult,
        PARTITION_REFINEMENT_PROMPT, CROSS_PARTITION_JUDGE_PROMPT,
        PartitionRefinementResult, CrossPartitionJudgeResult,
        PARTITION_REVIEW_PROMPT, PartitionReviewResult,
        CROSS_PARTITION_RESOLVE_PROMPT, ConflictResolutionResult,
    )
from development.models_exp import (
    RefinedCodebookModel, CodeRefinementResults, RefinedSubcode, RefinedCodebookCategory,
    CodeTransformation, BatchTransformationRecord, RefinementLineage,
    ThemeEnrichedCodebookEntryExp, ThemeEnrichedCodebookModelExp,
)
from development.step_6_codeGenerator.codeGenerator_exp import CodeGeneratorReasoningResults, is_other_cluster
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

    # =========================================================================
    # MECE ENFORCEMENT (post-refinement)
    # =========================================================================

    @staticmethod
    def _base_cluster_id(cluster_id: str) -> str:
        """Extract the base cluster ID from a sub-cluster ID.

        Step 6 generates sub-cluster IDs like "9-3", "17-2" from base clusters
        "9", "17". This method strips the sub-cluster suffix.
        """
        return cluster_id.split('-')[0] if '-' in cluster_id else cluster_id

    def _extract_concept_type_mapping(self, reasoning_results: CodeGeneratorReasoningResults) -> Dict[str, str]:
        """Map source_cluster_id -> concept_type (partition_name) from cluster_data.

        Reads cluster_data -> ideas -> partition_name to determine the concept_type
        for each source cluster. Uses majority vote for robustness.

        Handles sub-cluster IDs (e.g., "9-3") by looking up the base cluster ID ("9")
        in cluster_data, then mapping both the base and sub-cluster IDs.

        Returns:
            Dict mapping source_cluster_id to concept_type name.
            Includes both base cluster IDs ("9") and sub-cluster IDs ("9-3").
            Empty dict if concept_type data is not available (graceful fallback).
        """
        # First pass: build base_cluster_id -> concept_type from cluster_data
        base_cluster_to_concept_type = {}

        if not hasattr(reasoning_results, 'cluster_data') or not reasoning_results.cluster_data:
            return {}

        for cluster_id, data in reasoning_results.cluster_data.items():
            ideas = data.get('ideas', [])
            if not ideas:
                continue

            # Extract partition_name (= concept_type) from ideas
            concept_types = []
            for idea in ideas:
                partition_name = None
                if isinstance(idea, dict):
                    partition_name = idea.get('partition_name') or idea.get('concept_type')
                elif hasattr(idea, 'partition_name') and idea.partition_name:
                    partition_name = idea.partition_name
                elif hasattr(idea, 'concept_type') and idea.concept_type:
                    partition_name = idea.concept_type

                if partition_name:
                    concept_types.append(partition_name)

            if concept_types:
                base_cluster_to_concept_type[str(cluster_id)] = Counter(concept_types).most_common(1)[0][0]

        if not base_cluster_to_concept_type:
            return {}

        # Second pass: also map sub-cluster IDs from codebook entries
        cluster_to_concept_type = dict(base_cluster_to_concept_type)

        if hasattr(reasoning_results, 'codebook') and reasoning_results.codebook:
            for entry in reasoning_results.codebook:
                src = entry.get('source_cluster_id', '')
                if not src:
                    continue
                for cid in src.split(','):
                    cid = cid.strip()
                    if cid and cid not in cluster_to_concept_type:
                        base = self._base_cluster_id(cid)
                        if base in base_cluster_to_concept_type:
                            cluster_to_concept_type[cid] = base_cluster_to_concept_type[base]

        return cluster_to_concept_type

    def _group_codes_by_concept_type(
        self,
        refined_model: RefinedCodebookModel,
        concept_type_map: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group refined codes by their source concept_type.

        Args:
            refined_model: The refined codebook with themes and subcodes.
            concept_type_map: Mapping from source_cluster_id to concept_type.

        Returns:
            Dict mapping concept_type name to list of code dicts
            (each with 'code', 'description', 'theme', 'source_cluster').
            Codes with unknown concept_type go into "Other".
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for theme in refined_model.refined_codebook:
            for subcode in theme.subcodes:
                # Determine concept_type from source_cluster
                concept_type = "Other"
                if subcode.source_cluster:
                    # Take first cluster for merged codes
                    first_cluster = subcode.source_cluster.split(',')[0].strip()
                    concept_type = concept_type_map.get(first_cluster, "Other")

                code_entry = {
                    'code': subcode.code,
                    'description': subcode.description,
                    'theme': theme.category,
                    'source_cluster': subcode.source_cluster or '',
                    'category': subcode.category or '',
                }

                if concept_type not in groups:
                    groups[concept_type] = []
                groups[concept_type].append(code_entry)

        return groups

    def _format_codes_for_mece(self, codes: List[Dict[str, Any]]) -> str:
        """Format codes for the MECE enforcement prompt."""
        lines = []
        for code in codes:
            lines.append(f"- Code: {code['code']}")
            lines.append(f"  Description: {code['description']}")
            lines.append(f"  Theme: {code['theme']}")
            if code.get('category'):
                lines.append(f"  Category: {code['category']}")
            lines.append("")
        return '\n'.join(lines)

    def enforce_mece(
        self,
        survey_question: str,
        refined_model: RefinedCodebookModel,
        reasoning_results: CodeGeneratorReasoningResults
    ) -> Dict[str, MECEPartitionResult]:
        """Run MECE enforcement per concept_type partition.

        1. Extracts concept_type mapping from step 6 data
        2. Groups refined codes by concept_type
        3. Calls LLM per partition with CODEBOOK_MECE_ENFORCEMENT_PROMPT
        4. Falls back to single global partition if no concept_types available

        Args:
            survey_question: The survey question text
            refined_model: The refined codebook from the refinement step
            reasoning_results: Step 6 results (for concept_type extraction)

        Returns:
            Dict mapping partition_name to MECEPartitionResult.
            Also stores concept_type_map on self for downstream use.
        """
        # Extract concept_type mapping
        self.concept_type_map = self._extract_concept_type_mapping(reasoning_results)

        if self.concept_type_map:
            self.reporter.stat_line(f"Concept types found: {len(set(self.concept_type_map.values()))} types across {len(self.concept_type_map)} clusters")
            for ct in sorted(set(self.concept_type_map.values())):
                count = sum(1 for v in self.concept_type_map.values() if v == ct)
                self.reporter.stat_line(f"  - {ct}: {count} clusters")
        else:
            self.reporter.stat_line("No concept_type data available — using global MECE enforcement")

        # Group codes by concept_type
        code_groups = self._group_codes_by_concept_type(refined_model, self.concept_type_map)

        if not code_groups:
            # Fallback: all codes in one group
            all_codes = []
            for theme in refined_model.refined_codebook:
                for subcode in theme.subcodes:
                    all_codes.append({
                        'code': subcode.code,
                        'description': subcode.description,
                        'theme': theme.category,
                        'source_cluster': subcode.source_cluster or '',
                        'category': subcode.category or '',
                    })
            code_groups = {"All codes": all_codes}

        self.reporter.stat_line(f"MECE enforcement: {len(code_groups)} partitions")

        # Run MECE enforcement per partition
        all_partition_names = sorted(code_groups.keys())
        mece_results: Dict[str, MECEPartitionResult] = {}

        for partition_name in all_partition_names:
            codes = code_groups[partition_name]
            self.reporter.stat_line(f"\n  Partition '{partition_name}': {len(codes)} codes")

            # Build peer partitions list
            peer_partitions = [p for p in all_partition_names if p != partition_name]
            peer_list = '\n'.join([f"- {p}" for p in peer_partitions]) if peer_partitions else "None"

            # Build partition description
            partition_desc = ""
            if self.concept_type_map:
                partition_desc = f"Concept type: {partition_name}"

            # Format codes
            codes_list = self._format_codes_for_mece(codes)

            # Build prompt
            prompt = CODEBOOK_MECE_ENFORCEMENT_PROMPT.format(
                survey_question=survey_question,
                language=self.config.language,
                partition_name=partition_name,
                partition_description=partition_desc,
                peer_partitions_list=peer_list,
                n_codes=len(codes),
                codes_list=codes_list,
            )

            # Capture prompt if enabled
            if self.prompt_printer:
                self.prompt_printer.capture_prompt(
                    step_name=f"step_7_mece_{partition_name}",
                    utility_name="codebookRefinement_MECE",
                    prompt_content=prompt,
                    prompt_type="mece_enforcement",
                    metadata={
                        'partition_name': partition_name,
                        'code_count': len(codes),
                        'model': self.model_config.codebook_refinement_model,
                    }
                )

            try:
                model_name = self.model_config.codebook_refinement_model
                model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")
                temperature = self.model_config.get_temperature_for_stage('refinement') if model_type == "chat" else 0.0

                response = llm_create_sync(
                    client=self.client,
                    model=model_name,
                    prompt=prompt,
                    response_model=MECEPartitionResult,
                    temperature=temperature,
                    max_tokens=self.model_config.default_max_tokens,
                    track_usage=True,
                )

                mece_results[partition_name] = response

                # Report results
                self.reporter.stat_line(f"    -> {len(response.codes)} codes processed, {len(response.verifications)} pair verifications")
                if response.mece_issues:
                    for issue in response.mece_issues:
                        self.reporter.warning(f"    MECE issue: {issue}")

            except Exception as e:
                self.reporter.error(f"    MECE enforcement failed for '{partition_name}': {str(e)}")
                logger.error(f"MECE enforcement error for {partition_name}: {str(e)}", exc_info=True)

        return mece_results

    # =========================================================================
    # PRE-MECE PARTITION REVIEW (evaluates partition coherence)
    # =========================================================================

    def _format_codes_for_partition_review(self, codes: List[dict]) -> str:
        """Format codes for the partition review prompt."""
        lines = []
        for code in codes:
            lines.append(f"Code ID: {code['id']}")
            lines.append(f"  Label: {code['code']}")
            lines.append(f"  Definition: {code.get('definition', '')}")
            lines.append("")
        return '\n'.join(lines)

    async def _review_partition_async(
        self,
        partition_name: str,
        codes: List[dict],
        peer_partitions: List[str],
        survey_question: str,
        async_client: Any,
        semaphore: asyncio.Semaphore,
        rate_limiter: AsyncLimiter,
    ) -> PartitionReviewResult:
        """Review one partition for conceptual coherence (async)."""

        codes_list = self._format_codes_for_partition_review(codes)
        peer_list = '\n'.join(f"- {p}" for p in peer_partitions) if peer_partitions else "None"

        prompt = PARTITION_REVIEW_PROMPT.format(
            survey_question=survey_question,
            language=self.config.language,
            partition_name=partition_name,
            n_codes=len(codes),
            codes_list=codes_list,
            peer_partitions_list=peer_list,
        )

        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name=f"step_7_partition_review_{partition_name}",
                utility_name="codebookRefinement_partition_review",
                prompt_content=prompt,
                prompt_type="partition_review",
                metadata={
                    'partition_name': partition_name,
                    'code_count': len(codes),
                    'model': self.model_config.codebook_refinement_model,
                }
            )

        model_name = self.model_config.codebook_refinement_model
        model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")
        temperature = 0.0  # Review should be deterministic

        async with semaphore:
            async with rate_limiter:
                response = await llm_create_async(
                    client=async_client,
                    model=model_name,
                    prompt=prompt,
                    response_model=PartitionReviewResult,
                    temperature=temperature,
                    max_tokens=self.model_config.default_max_tokens,
                )

        return response

    async def _review_all_partitions(
        self,
        survey_question: str,
        code_groups: Dict[str, List[dict]],
        async_client: Any,
    ) -> Dict[str, PartitionReviewResult]:
        """Review all partitions for coherence concurrently.

        Partitions with <=2 codes are auto-kept (too small to split meaningfully).
        """
        semaphore = asyncio.Semaphore(min(len(code_groups), 10))
        rate_limiter = AsyncLimiter(5, time_period=1.0)

        all_partition_names = sorted(code_groups.keys())

        tasks = {}
        auto_kept = {}
        for name in all_partition_names:
            if len(code_groups[name]) <= 2:
                # Auto-keep small partitions
                auto_kept[name] = PartitionReviewResult(
                    partition_name=name,
                    action="keep",
                    domain_name=name,
                    domain_description="",
                    splits=[],
                    review_rationale=f"Auto-kept: partition has only {len(code_groups[name])} code(s), too small to split"
                )
                continue

            peer_partitions = [p for p in all_partition_names if p != name]
            tasks[name] = self._review_partition_async(
                partition_name=name,
                codes=code_groups[name],
                peer_partitions=peer_partitions,
                survey_question=survey_question,
                async_client=async_client,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
            )

        results = dict(auto_kept)

        if tasks:
            results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, result in zip(tasks.keys(), results_list):
                if isinstance(result, Exception):
                    self.reporter.error(f"Partition review '{name}' FAILED: {type(result).__name__}: {result}")
                else:
                    results[name] = result

        return results

    def _apply_partition_reorganization(
        self,
        code_groups: Dict[str, List[dict]],
        review_results: Dict[str, PartitionReviewResult],
        concept_type_map: Dict[str, str],
    ) -> Tuple[Dict[str, List[dict]], Dict[str, str], Dict[str, str]]:
        """Apply partition splits based on review results.

        Returns:
            (new_code_groups, partition_remap, updated_concept_type_map) where:
            - new_code_groups: reorganized code groups
            - partition_remap: {old_partition_name: new_partition_name} for split partitions
            - updated_concept_type_map: {source_cluster_id: new_partition_name}
        """
        new_code_groups: Dict[str, List[dict]] = {}
        partition_remap: Dict[str, str] = {}
        updated_concept_type_map = dict(concept_type_map)

        for partition_name, codes in code_groups.items():
            review = review_results.get(partition_name)

            if not review or review.action == "keep":
                # Keep as-is
                new_code_groups[partition_name] = codes
                continue

            # Split: reorganize codes into new sub-partitions
            # Build code_id -> code lookup
            code_by_id = {code['id']: code for code in codes}
            assigned_ids = set()

            for split_group in review.splits:
                new_name = split_group.new_partition_name
                new_codes = []
                for code_id in split_group.code_ids:
                    if code_id in code_by_id:
                        new_codes.append(code_by_id[code_id])
                        assigned_ids.add(code_id)

                        # Update concept_type_map for this code's source clusters
                        src = code_by_id[code_id].get('source_cluster_id', '')
                        if src:
                            for cid in src.split(','):
                                cid = cid.strip()
                                if cid:
                                    updated_concept_type_map[cid] = new_name
                                    # Also update base cluster ID
                                    base = self._base_cluster_id(cid)
                                    if base != cid:
                                        updated_concept_type_map[base] = new_name
                    else:
                        self.reporter.warning(
                            f"Partition review: code ID '{code_id}' not found in '{partition_name}'"
                        )

                if new_codes:
                    new_code_groups[new_name] = new_codes

            # Safety: any codes not assigned by the split go into a fallback
            unassigned = [code for code in codes if code['id'] not in assigned_ids]
            if unassigned:
                self.reporter.warning(
                    f"Partition '{partition_name}': {len(unassigned)} codes not assigned by split, "
                    f"keeping in original partition"
                )
                new_code_groups.setdefault(partition_name, []).extend(unassigned)

            # Build remap: new split name → original partition name
            # (step 8 uses this to remap codebook concept_types back to match ideas)
            for split_group in review.splits:
                partition_remap[split_group.new_partition_name] = partition_name

        # Also remap concept_type_map entries for kept partitions (no change needed, they stay the same)

        return new_code_groups, partition_remap, updated_concept_type_map

    # =========================================================================
    # PARTITION-FIRST REFINEMENT (replaces MAP-REDUCE + separate MECE)
    # =========================================================================

    def _group_raw_codes_by_concept_type(
        self,
        reasoning_results: CodeGeneratorReasoningResults
    ) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
        """Group raw step 6 codes by concept_type partition.

        DIRECT_OTHER codes (source_cluster_id starting with "other_") are
        excluded from partition refinement and stored on self._other_codes
        for later injection as pass-through entries.

        Returns:
            (code_groups, concept_type_map) where:
            - code_groups: {concept_type_name: [raw_code_dicts]}
            - concept_type_map: {source_cluster_id: concept_type_name}
        """
        raw_codes = self._extract_raw_codes(reasoning_results)
        concept_type_map = self._extract_concept_type_mapping(reasoning_results)

        # Separate "other" codes that bypass refinement
        normal_codes = []
        self._other_codes = []
        for code in raw_codes:
            if is_other_cluster(code.get('source_cluster_id', '')):
                self._other_codes.append(code)
            else:
                normal_codes.append(code)

        if self._other_codes:
            self.reporter.stat_line(
                f"Excluded {len(self._other_codes)} 'other' codes from partition refinement "
                f"(will be re-injected as pass-through)"
            )

        groups: Dict[str, List[dict]] = {}
        for code in normal_codes:
            src = code.get('source_cluster_id', '')
            first_cluster = src.split(',')[0].strip() if src else ''
            concept_type = concept_type_map.get(first_cluster, "other")
            groups.setdefault(concept_type, []).append(code)

        return groups, concept_type_map

    def _format_codes_with_step6_context(self, codes: List[dict]) -> str:
        """Format codes with their step 6 context for the partition refinement prompt."""
        lines = []
        for code in codes:
            lines.append(f"Code ID: {code['id']}")
            lines.append(f"  Label: {code['code']}")
            lines.append(f"  Definition: {code.get('definition', '')}")
            lines.append(f"  Source cluster: {code.get('source_cluster_id', '')}")
            if code.get('inclusion_examples'):
                examples = code['inclusion_examples']
                if isinstance(examples, list):
                    lines.append(f"  Existing inclusion examples: {'; '.join(str(e) for e in examples)}")
                else:
                    lines.append(f"  Existing inclusion examples: {examples}")
            if code.get('exclusion_examples'):
                examples = code['exclusion_examples']
                if isinstance(examples, list):
                    lines.append(f"  Existing exclusion examples: {'; '.join(str(e) for e in examples)}")
                else:
                    lines.append(f"  Existing exclusion examples: {examples}")
            if code.get('near_neighbor_label'):
                lines.append(f"  Existing near neighbor: {code['near_neighbor_label']}")
            if code.get('tell_apart_rule'):
                lines.append(f"  Existing tell-apart rule: {code['tell_apart_rule']}")
            lines.append("")
        return '\n'.join(lines)

    def _build_cross_partition_summary(self, partition_results: Dict[str, PartitionRefinementResult]) -> str:
        """Build condensed summary for cross-partition judge prompt."""
        lines = []
        for partition_name, result in sorted(partition_results.items()):
            lines.append(f"=== Partition: {partition_name} (theme: {result.theme_label}) ===")
            lines.append(f"Description: {result.theme_description}")
            for code in result.codes:
                lines.append(f"  - {code.code}: {code.definition}")
                lines.append(f"    Boundary test: {code.boundary_test}")
            lines.append("")
        return '\n'.join(lines)

    async def _refine_partition_async(
        self,
        partition_name: str,
        codes: List[dict],
        peer_partitions: List[str],
        survey_question: str,
        async_client: Any,
        semaphore: asyncio.Semaphore,
        rate_limiter: AsyncLimiter,
        display_name: Optional[str] = None,
    ) -> PartitionRefinementResult:
        """Refine + MECE-enforce one partition (async)."""

        effective_name = display_name or partition_name
        codes_with_context = self._format_codes_with_step6_context(codes)
        peer_list = '\n'.join(f"- {p}" for p in peer_partitions) if peer_partitions else "None"

        prompt = PARTITION_REFINEMENT_PROMPT.format(
            survey_question=survey_question,
            language=self.config.language,
            partition_name=effective_name,
            n_codes=len(codes),
            peer_partitions_list=peer_list,
            codes_with_context=codes_with_context,
        )

        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name=f"step_7_partition_{partition_name}",
                utility_name="codebookRefinement_partition",
                prompt_content=prompt,
                prompt_type="partition_refinement",
                metadata={
                    'partition_name': partition_name,
                    'code_count': len(codes),
                    'model': self.model_config.codebook_refinement_model,
                }
            )

        model_name = self.model_config.codebook_refinement_model
        model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")
        temperature = self.model_config.get_temperature_for_stage('refinement') if model_type == "chat" else 0.0

        async with semaphore:
            async with rate_limiter:
                response = await llm_create_async(
                    client=async_client,
                    model=model_name,
                    prompt=prompt,
                    response_model=PartitionRefinementResult,
                    temperature=temperature,
                    max_tokens=self.model_config.default_max_tokens,
                )

        return response

    async def _cross_partition_judge_async(
        self,
        survey_question: str,
        partition_results: Dict[str, PartitionRefinementResult],
        async_client: Any,
    ) -> CrossPartitionJudgeResult:
        """Cross-partition MECE judge (single LLM call)."""

        codebook_summary = self._build_cross_partition_summary(partition_results)
        total_codes = sum(len(r.codes) for r in partition_results.values())

        prompt = CROSS_PARTITION_JUDGE_PROMPT.format(
            survey_question=survey_question,
            language=self.config.language,
            codebook_summary=codebook_summary,
            total_codes=total_codes,
            n_partitions=len(partition_results),
        )

        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="step_7_cross_partition_judge",
                utility_name="codebookRefinement_judge",
                prompt_content=prompt,
                prompt_type="cross_partition_judge",
                metadata={
                    'total_codes': total_codes,
                    'n_partitions': len(partition_results),
                    'model': self.model_config.codebook_refinement_model,
                }
            )

        model_name = self.model_config.codebook_refinement_model
        model_type = self.model_config.MODEL_TYPES.get(model_name, "chat")
        temperature = 0.0  # Judge should be deterministic

        response = await llm_create_async(
            client=async_client,
            model=model_name,
            prompt=prompt,
            response_model=CrossPartitionJudgeResult,
            temperature=temperature,
            max_tokens=self.model_config.default_max_tokens,
        )

        return response

    def _find_code_in_partition(self, code_name: str, partition_name: str,
                                partition_results: Dict[str, PartitionRefinementResult]):
        """Find a code object by name within a specific partition."""
        if partition_name in partition_results:
            for code in partition_results[partition_name].codes:
                if code.code == code_name:
                    return code
        return None

    def _format_code_details(self, code) -> str:
        """Format a single code's full details for the resolution prompt."""
        signals = ", ".join(code.diagnostic_signals) if code.diagnostic_signals else "(none)"
        inclusions = "\n    ".join(f"+ {ex}" for ex in (code.inclusion_examples or []))
        exclusions = "\n    ".join(f"- {ex}" for ex in (code.exclusion_examples or []))
        return (
            f"  Definition: {code.definition}\n"
            f"  Boundary test: {code.boundary_test}\n"
            f"  Diagnostic signals: {signals}\n"
            f"  Inclusion examples:\n    {inclusions}\n"
            f"  Exclusion examples:\n    {exclusions}\n"
            f"  Near neighbor: {code.near_neighbor_label or 'N/A'}\n"
            f"  Tell-apart rule: {code.tell_apart_rule or 'N/A'}"
        )

    def _format_conflicts_for_resolution(
        self,
        judge_result: CrossPartitionJudgeResult,
        partition_results: Dict[str, PartitionRefinementResult],
    ) -> str:
        """Build formatted conflict details for the resolution prompt."""
        sections = []
        for i, conflict in enumerate(judge_result.conflicts):
            code_a = self._find_code_in_partition(conflict.code_a, conflict.partition_a, partition_results)
            code_b = self._find_code_in_partition(conflict.code_b, conflict.partition_b, partition_results)

            section = f"--- Conflict {i} ({conflict.severity}) ---\n"
            section += f'Code A: "{conflict.code_a}" [partition: {conflict.partition_a}]\n'
            if code_a:
                section += self._format_code_details(code_a) + "\n\n"
            else:
                section += "  (code details not found)\n\n"

            section += f'Code B: "{conflict.code_b}" [partition: {conflict.partition_b}]\n'
            if code_b:
                section += self._format_code_details(code_b) + "\n\n"
            else:
                section += "  (code details not found)\n\n"

            section += f"Overlap: {conflict.overlap_description}\n"
            section += f"Judge's suggestion: {conflict.resolution}"
            sections.append(section)

        return "\n\n".join(sections)

    async def _resolve_cross_partition_conflicts_async(
        self,
        judge_result: CrossPartitionJudgeResult,
        partition_results: Dict[str, PartitionRefinementResult],
        survey_question: str,
        async_client: Any,
    ) -> ConflictResolutionResult:
        """Resolve cross-partition conflicts by merging duplicates or sharpening boundaries."""

        conflicts_formatted = self._format_conflicts_for_resolution(judge_result, partition_results)

        prompt = CROSS_PARTITION_RESOLVE_PROMPT.format(
            survey_question=survey_question,
            language=self.config.language,
            n_conflicts=len(judge_result.conflicts),
            conflicts_formatted=conflicts_formatted,
        )

        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="step_7_cross_partition_resolve",
                utility_name="codebookRefinement_resolve",
                prompt_content=prompt,
                prompt_type="cross_partition_resolve",
                metadata={
                    'n_conflicts': len(judge_result.conflicts),
                    'model': self.model_config.codebook_refinement_model,
                }
            )

        model_name = self.model_config.codebook_refinement_model
        temperature = 0.0  # Resolution should be deterministic

        resolution_result = await llm_create_async(
            client=async_client,
            model=model_name,
            prompt=prompt,
            response_model=ConflictResolutionResult,
            temperature=temperature,
            max_tokens=self.model_config.default_max_tokens,
        )

        # Apply resolutions to partition_results
        # Track codes dropped by earlier merges so later conflicts can skip them
        dropped_codes: set = set()  # (code_name, partition_key) tuples

        for res in resolution_result.resolutions:
            if res.conflict_index >= len(judge_result.conflicts):
                continue
            conflict = judge_result.conflicts[res.conflict_index]

            # Skip conflicts that reference a code already dropped by an earlier merge
            conflict_codes = {
                (conflict.code_a, conflict.partition_a),
                (conflict.code_b, conflict.partition_b),
            }
            if conflict_codes & dropped_codes:
                logger.info(
                    f"Skipping conflict {res.conflict_index}: references already-dropped code"
                )
                continue

            if res.action == "merge" and res.dropped_code and res.dropped_partition:
                # Remove dropped code from its partition
                if res.dropped_partition in partition_results:
                    codes = partition_results[res.dropped_partition].codes
                    partition_results[res.dropped_partition].codes = [
                        c for c in codes if c.code != res.dropped_code
                    ]
                    dropped_codes.add((res.dropped_code, res.dropped_partition))

            elif res.action == "sharpen":
                # Update code_a
                code_a = self._find_code_in_partition(conflict.code_a, conflict.partition_a, partition_results)
                if code_a and res.code_a_updates:
                    if 'boundary_test' in res.code_a_updates:
                        code_a.boundary_test = res.code_a_updates['boundary_test']
                    if 'tell_apart_rule' in res.code_a_updates:
                        code_a.tell_apart_rule = res.code_a_updates['tell_apart_rule']
                if code_a and res.code_a_new_exclusions:
                    if code_a.exclusion_examples is None:
                        code_a.exclusion_examples = []
                    code_a.exclusion_examples.extend(res.code_a_new_exclusions)

                # Update code_b
                code_b = self._find_code_in_partition(conflict.code_b, conflict.partition_b, partition_results)
                if code_b and res.code_b_updates:
                    if 'boundary_test' in res.code_b_updates:
                        code_b.boundary_test = res.code_b_updates['boundary_test']
                    if 'tell_apart_rule' in res.code_b_updates:
                        code_b.tell_apart_rule = res.code_b_updates['tell_apart_rule']
                if code_b and res.code_b_new_exclusions:
                    if code_b.exclusion_examples is None:
                        code_b.exclusion_examples = []
                    code_b.exclusion_examples.extend(res.code_b_new_exclusions)

        return resolution_result

    async def _run_all_partitions(
        self,
        survey_question: str,
        code_groups: Dict[str, List[dict]],
        async_client: Any,
        domain_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, PartitionRefinementResult]:
        """Run all partition refinements concurrently."""
        semaphore = asyncio.Semaphore(min(len(code_groups), 10))
        rate_limiter = AsyncLimiter(5, time_period=1.0)  # 5 requests/sec

        all_partition_names = sorted(code_groups.keys())

        tasks = {}
        for name in all_partition_names:
            peer_partitions = [p for p in all_partition_names if p != name]
            display_name = domain_names.get(name, name) if domain_names else name
            tasks[name] = self._refine_partition_async(
                partition_name=name,
                codes=code_groups[name],
                peer_partitions=peer_partitions,
                survey_question=survey_question,
                async_client=async_client,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
                display_name=display_name,
            )

        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

        results = {}
        for name, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                self.reporter.error(f"Partition '{name}' FAILED: {type(result).__name__}: {result}")
            else:
                self.reporter.stat_line(
                    f"  Partition '{name}': {len(code_groups[name])} codes → "
                    f"{len(result.codes)} refined, {len(result.verifications)} verifications"
                )
                if result.mece_issues:
                    for issue in result.mece_issues:
                        self.reporter.warning(f"    MECE issue ({name}): {issue}")
                results[name] = result

        return results

    def refine_codebook_partitioned(
        self,
        survey_question: str,
        reasoning_results: CodeGeneratorReasoningResults,
    ) -> Tuple[Dict[str, PartitionRefinementResult], Optional[CrossPartitionJudgeResult], Dict[str, str], Dict[str, str]]:
        """Main entry point: partition review + refinement + MECE + cross-partition judge.

        Returns:
            (partition_results, judge_result, concept_type_map, partition_remap)
        """
        # 1. Group raw codes by concept_type
        code_groups, concept_type_map = self._group_raw_codes_by_concept_type(reasoning_results)

        total_codes = sum(len(g) for g in code_groups.values())
        self.reporter.stat_line(f"Partitioned {total_codes} codes into {len(code_groups)} concept-type groups")
        for name, codes in sorted(code_groups.items()):
            self.reporter.stat_line(f"  - {name}: {len(codes)} codes")

        if not code_groups:
            self.reporter.warning("No code groups found — cannot proceed with partition refinement")
            return {}, None, concept_type_map, {}

        # 2. Create async client
        async_client = create_client(
            model=self.model_config.codebook_refinement_model,
            async_mode=True,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER if API_PROVIDER == "azure" else None
        )

        # 3. PARTITION REVIEW: evaluate coherence of each partition
        partition_remap: Dict[str, str] = {}
        self.reporter.section_header("PARTITION REVIEW")
        loop = asyncio.get_event_loop()
        review_results = loop.run_until_complete(
            self._review_all_partitions(survey_question, code_groups, async_client)
        )

        # Report review results
        splits_found = 0
        for name, review in sorted(review_results.items()):
            if review.action == "keep":
                rename = f" → '{review.domain_name}'" if review.domain_name and review.domain_name != name else ""
                self.reporter.stat_line(f"  KEEP '{name}'{rename}: {review.review_rationale[:80]}")
            else:
                splits_found += 1
                split_names = [s.new_partition_name for s in review.splits]
                self.reporter.stat_line(
                    f"  SPLIT '{name}' -> {split_names}: {review.review_rationale[:80]}"
                )

        # Apply reorganization if any splits
        if splits_found > 0:
            code_groups, partition_remap, concept_type_map = self._apply_partition_reorganization(
                code_groups, review_results, concept_type_map
            )
            total_codes_after = sum(len(g) for g in code_groups.values())
            self.reporter.stat_line(
                f"\nPartition review: {splits_found} split(s) applied. "
                f"{total_codes_after} codes across {len(code_groups)} partitions"
            )
            for name, codes in sorted(code_groups.items()):
                self.reporter.stat_line(f"  - {name}: {len(codes)} codes")
        else:
            self.reporter.stat_line(f"\nPartition review: all {len(code_groups)} partitions coherent, no splits needed")

        # Build domain_names: partition_key → reviewed domain name for refinement prompt
        domain_names: Dict[str, str] = {}
        for name, review in review_results.items():
            if review.action == "keep":
                domain_names[name] = review.domain_name if review.domain_name else name
            # For split partitions, new partition names are already domain names
        for name in code_groups:
            if name not in domain_names:
                domain_names[name] = name  # fallback for split-created partitions

        # 4. Run all partitions concurrently (MECE refinement)
        self.reporter.section_header("PARTITION REFINEMENT")
        partition_results = loop.run_until_complete(
            self._run_all_partitions(survey_question, code_groups, async_client, domain_names=domain_names)
        )

        refined_count = sum(len(r.codes) for r in partition_results.values())
        self.reporter.stat_line(f"\nPartition refinement complete: {refined_count} codes across {len(partition_results)} partitions")

        # 5. Cross-partition judge
        judge_result = None
        if len(partition_results) > 1:
            self.reporter.section_header("CROSS-PARTITION MECE JUDGE")
            try:
                judge_result = loop.run_until_complete(
                    self._cross_partition_judge_async(survey_question, partition_results, async_client)
                )

                if judge_result.conflicts:
                    for conflict in judge_result.conflicts:
                        self.reporter.warning(
                            f"Cross-partition overlap: '{conflict.code_a}' ({conflict.partition_a}) "
                            f"vs '{conflict.code_b}' ({conflict.partition_b}) — {conflict.severity}"
                        )
                self.reporter.stat_line(
                    f"Cross-partition judge: {'MECE compliant' if judge_result.is_mece_compliant else 'issues found'} "
                    f"({len(judge_result.conflicts)} conflicts)"
                )
            except Exception as e:
                self.reporter.error(f"Cross-partition judge failed: {type(e).__name__}: {e}")
                logger.error(f"Cross-partition judge error: {e}", exc_info=True)

        # 6. Resolve cross-partition conflicts
        if judge_result and judge_result.conflicts:
            self.reporter.section_header("CROSS-PARTITION CONFLICT RESOLUTION")
            try:
                resolution_result = loop.run_until_complete(
                    self._resolve_cross_partition_conflicts_async(
                        judge_result, partition_results, survey_question, async_client
                    )
                )
                merges = 0
                sharpens = 0
                for res in resolution_result.resolutions:
                    if res.action == "merge":
                        merges += 1
                        self.reporter.stat_line(
                            f"MERGED: dropped '{res.dropped_code}' from '{res.dropped_partition}', "
                            f"kept '{res.surviving_code}' in '{res.surviving_partition}'"
                        )
                    elif res.action == "sharpen":
                        conflict = judge_result.conflicts[res.conflict_index]
                        sharpens += 1
                        self.reporter.stat_line(
                            f"SHARPENED: '{conflict.code_a}' ({conflict.partition_a}) "
                            f"↔ '{conflict.code_b}' ({conflict.partition_b})"
                        )
                resolved_count = sum(len(r.codes) for r in partition_results.values())
                self.reporter.stat_line(
                    f"Resolution complete: {merges} merged, {sharpens} sharpened "
                    f"→ {resolved_count} codes across {len(partition_results)} partitions"
                )
            except Exception as e:
                self.reporter.error(f"Conflict resolution failed: {type(e).__name__}: {e}")
                logger.error(f"Conflict resolution error: {e}", exc_info=True)

        return partition_results, judge_result, concept_type_map, partition_remap

    def build_theme_enriched_codebook(
        self,
        partition_results: Dict[str, PartitionRefinementResult],
        judge_result: Optional[CrossPartitionJudgeResult],
        concept_type_map: Dict[str, str],
        source_variable: str,
        partition_remap: Optional[Dict[str, str]] = None,
        reasoning_results: Optional[CodeGeneratorReasoningResults] = None,
    ) -> ThemeEnrichedCodebookModelExp:
        """Build the final enriched codebook model from partition results.

        If reasoning_results is provided and DIRECT_OTHER codes were filtered
        during refinement (stored on self._other_codes), they are injected as
        pass-through codebook entries and an other_idea_assignments mapping
        is built for step 8 to pre-assign those ideas.
        """
        enriched_entries = []
        themes_summary = []
        code_to_theme_mapping = {}

        dominance_axes = {}
        for partition_name, result in sorted(partition_results.items()):
            themes_summary.append({
                'theme_name': result.theme_label,
                'theme_description': result.theme_description,
                'code_count': len(result.codes),
                'dominance_axis': getattr(result, 'dominance_axis', ''),
            })
            # Collect dominance axes for step 8 routing
            axis = getattr(result, 'dominance_axis', '')
            if axis:
                dominance_axes[partition_name] = axis

            for code in result.codes:
                entry = ThemeEnrichedCodebookEntryExp(
                    code=code.code,
                    definition=code.definition,
                    theme=result.theme_label,
                    theme_description=result.theme_description,
                    category="",
                    category_description="",
                    source_cluster=code.source_code_ids,
                    inclusion_examples=code.inclusion_examples,
                    exclusion_examples=code.exclusion_examples,
                    near_neighbor_label=code.near_neighbor_label,
                    tell_apart_rule=code.tell_apart_rule,
                    boundary_test=code.boundary_test,
                    diagnostic_signals=code.diagnostic_signals,
                    concept_type=partition_name,
                    mece_verified=True,
                )
                enriched_entries.append(entry)
                code_to_theme_mapping[code.code] = result.theme_label

        # Inject DIRECT_OTHER codes as pass-through entries
        other_idea_assignments: Optional[Dict[str, str]] = None
        other_codes = getattr(self, '_other_codes', [])
        if other_codes:
            other_idea_assignments = {}

            # Extract idea_id → code mapping from step 6 cluster_results
            if reasoning_results and hasattr(reasoning_results, 'cluster_results'):
                for cr in reasoning_results.cluster_results:
                    if cr.get('decision') == 'DIRECT_OTHER' and 'idea_ids' in cr:
                        code_label = cr.get('final_code', '')
                        for idea_id in cr['idea_ids']:
                            other_idea_assignments[idea_id] = code_label

            # Inject each "other" code as a pass-through codebook entry
            for other_code in other_codes:
                src = other_code.get('source_cluster_id', '')
                code_label = other_code.get('code', '')
                definition = other_code.get('definition', '')

                # Extract partition name from source_cluster_id: "other_{partition}_{valence}"
                partition_name = src
                if src.startswith("other_"):
                    stripped = src[len("other_"):]
                    last_underscore = stripped.rfind('_')
                    partition_name = stripped[:last_underscore] if last_underscore >= 0 else stripped

                # Use language-aware "other" label for theme
                from development.step_5_categories.config_categories_exp import get_other_category_label
                other_label = get_other_category_label(self.config.language)
                theme_label = f"{partition_name} — {other_label}"

                entry = ThemeEnrichedCodebookEntryExp(
                    code=code_label,
                    definition=definition,
                    theme=theme_label,
                    theme_description=f"Catch-all for ideas in '{partition_name}' not assigned a specific code",
                    category="",
                    category_description="",
                    source_cluster=src,
                    concept_type=partition_name,
                    mece_verified=False,
                )
                enriched_entries.append(entry)
                code_to_theme_mapping[code_label] = theme_label

            self.reporter.stat_line(
                f"Injected {len(other_codes)} 'other' codes as pass-through entries "
                f"({len(other_idea_assignments)} ideas pre-assigned)"
            )

        # Serialize judge results
        cross_partition_data = None
        if judge_result:
            cross_partition_data = judge_result.model_dump()

        return ThemeEnrichedCodebookModelExp(
            codes=enriched_entries,
            themes_summary=themes_summary,
            code_to_theme_mapping=code_to_theme_mapping,
            theme_methodology="Partition-first refinement with partition review + cross-partition MECE judge",
            source_variable=source_variable,
            concept_type_mapping=concept_type_map,
            cross_partition_results=cross_partition_data,
            partition_remap=partition_remap if partition_remap else None,
            dominance_axes=dominance_axes if dominance_axes else None,
            other_idea_assignments=other_idea_assignments if other_idea_assignments else None,
        )

    # =========================================================================
    # LEGACY METHODS (below)
    # =========================================================================

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

def enforce_mece(
    survey_question: str,
    refined_model: RefinedCodebookModel,
    reasoning_results: CodeGeneratorReasoningResults,
    model_config: Optional[ModelConfig] = None,
    language: str = DEFAULT_LANGUAGE,
    verbose: bool = True,
    prompt_printer: Optional[Any] = None
) -> Dict[str, 'MECEPartitionResult']:
    """Run MECE enforcement on a refined codebook, partitioned by concept_type.

    Args:
        survey_question: The survey question text for context
        refined_model: The refined codebook from refine_codebook()
        reasoning_results: Step 6 CodeGeneratorReasoningResults (for concept_type tracing)
        model_config: Model configuration (uses default if None)
        language: Output language code
        verbose: Enable verbose progress reporting
        prompt_printer: Optional prompt capture utility

    Returns:
        Dict mapping partition_name to MECEPartitionResult.
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
    processor.reasoning_results = reasoning_results
    return processor.enforce_mece(survey_question, refined_model, reasoning_results)


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