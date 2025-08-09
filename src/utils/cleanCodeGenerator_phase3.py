"""
Phase 3 Implementation with LangChain and Original SharedCodebook
Properly chains Step2→Step3→Step4 per cluster with batch processing
"""

import asyncio
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from asyncio_throttle import Throttler

# LangChain imports - using the same as original codeGenerator.py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

# Models and config
import models
from config import OPENAI_API_KEY, ModelConfig, DEFAULT_LANGUAGE
from pydantic import BaseModel, Field, RootModel
from typing import List
from prompts import (
    CANDIDATE_CODE_SELECTION_PROMPT,
    CODE_GENERATION_PROMPT,
    VALIDATION_PROMPT
)

# Pydantic model for Step 2 output (matching original codeGenerator)
class CodebookAnalysisOutput(RootModel[List[models.CandidateCode]]):
    """Output from Step 2 - Candidate Code Selection - Direct array of candidate codes"""
    root: List[models.CandidateCode] = Field(description="Array of selected relevant codes")

# Original SharedCodebook from codeGenerator.py
@dataclass
class SharedCodebook:
    """Thread-safe shared codebook with async lock and version tracking"""
    _codes: List[Dict[str, str]]
    _lock: asyncio.Lock
    _version: int = 0
    _update_log: List[Dict[str, Any]] = None
    
    def __init__(self, initial_codes: List[Dict[str, str]]):
        self._codes = initial_codes.copy() if initial_codes else []
        self._lock = asyncio.Lock()
        self._version = 0
        self._update_log = []
    
    async def get_current_snapshot(self) -> Tuple[List[Dict[str, str]], int]:
        """Get current codes and version atomically"""
        async with self._lock:
            return self._codes.copy(), self._version
    
    async def add_code_if_new(self, code: str, definition: str) -> Tuple[bool, int]:
        """Add a new code if it doesn't exist, return (added, new_version)"""
        async with self._lock:
            # Check if code already exists
            for existing in self._codes:
                if existing['code'].lower() == code.lower():
                    return False, self._version
            
            # Add new code
            self._codes.append({'code': code, 'definition': definition})
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add',
                'code': code,
                'timestamp': time.time()
            })
            return True, self._version
    
    async def replace_code(self, original_code: str, new_code: str, new_definition: str) -> Tuple[bool, int]:
        """Replace an existing code with a modified version, return (replaced, new_version)"""
        async with self._lock:
            # Find and replace the original code
            for i, existing in enumerate(self._codes):
                if existing['code'].lower() == original_code.lower():
                    self._codes[i] = {'code': new_code, 'definition': new_definition}
                    self._version += 1
                    self._update_log.append({
                        'version': self._version,
                        'action': 'replace',
                        'original_code': original_code,
                        'new_code': new_code,
                        'timestamp': time.time()
                    })
                    return True, self._version
            
            # If original code not found, add as new
            self._codes.append({'code': new_code, 'definition': new_definition})
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add_as_fallback',
                'code': new_code,
                'timestamp': time.time()
            })
            return True, self._version
    
    async def get_code_definition(self, code_name: str) -> Optional[str]:
        """Get the definition of a specific code"""
        async with self._lock:
            for existing in self._codes:
                if existing['code'].lower() == code_name.lower():
                    return existing['definition']
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get codebook statistics"""
        async with self._lock:
            return {
                'total_codes': len(self._codes),
                'version': self._version,
                'updates': len(self._update_log)
            }


class LangChainPhase3Processor:
    """Process clusters using LangChain SequentialChain with batch concurrency"""
    
    def __init__(
        self,
        cluster_data: List[models.ClusterModel],
        theme_map: Dict[str, Dict],
        embedding_book: Dict[str, Dict],
        shared_codebook: SharedCodebook,
        var_lab: str,
        model_config: ModelConfig,
        verbose_reporter,
        prompt_printer=None
    ):
        self.cluster_data = cluster_data
        self.theme_map = theme_map
        self.embedding_book = embedding_book
        self.shared_codebook = shared_codebook
        self.var_lab = var_lab
        self.model_config = model_config
        self.verbose_reporter = verbose_reporter
        self.prompt_printer = prompt_printer
        
        # LLM setup
        self.model_name = model_config.get_model_for_stage("cluster_analysis")
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=self.model_name,
            temperature=0.0,
            max_tokens=4000
        )
        
        # Create LangChain chains for each step
        self._setup_chains()
        
        # Results storage
        self.step2_results = {}
        self.step3_results = {}
        self.step4_results = {}
    
    def _setup_chains(self):
        """Setup LangChain chains for Step 2, 3, and 4 using LCEL like original codeGenerator"""
        
        # Step 2: Candidate Code Selection Chain
        step2_prompt = PromptTemplate(
            template=CANDIDATE_CODE_SELECTION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "code_text"]
        )
        self.step2_chain = (
            step2_prompt
            | self.llm
            | PydanticOutputParser(pydantic_object=CodebookAnalysisOutput)
        ).with_config({"max_concurrency": 10})
        
        # Step 3: Code Generation Chain
        step3_prompt = PromptTemplate(
            template=CODE_GENERATION_PROMPT,
            input_variables=["language", "survey_question", "candidate_codes", "cluster_summary"]
        )
        self.step3_chain = (
            step3_prompt
            | self.llm
            | PydanticOutputParser(pydantic_object=models.CodeRecommendation)
        ).with_config({"max_concurrency": 10})
        
        # Step 4: Validation Chain
        step4_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "step3_recommendation", "candidate_codes"]
        )
        self.step4_chain = (
            step4_prompt
            | self.llm
            | PydanticOutputParser(pydantic_object=models.ValidationResult)
        ).with_config({"max_concurrency": 10})
    
    async def process_clusters_in_batches(self, batch_size: int = 10) -> Dict[str, Any]:
        """Process clusters in batches with real-time codebook updates between batches"""
        
        # Group responses by actual HDBSCAN clusters
        cluster_groups = self._group_by_actual_clusters()
        actual_clusters = list(cluster_groups.keys())
        total_clusters = len(actual_clusters)
        
        self.verbose_reporter.stat_line(f"Processing {total_clusters} actual HDBSCAN clusters in batches of {batch_size}")
        
        successful_clusters = 0
        total_new_codes = 0
        total_replaced_codes = 0
        
        # Initialize throttler for optimal API usage (balance speed vs rate limits)
        # GPT-4 typically allows 500 RPM, so we use ~400 to stay safe
        throttler = Throttler(rate_limit=6.0, period=1.0)  # ~6 requests per second = 360 RPM
        
        # Process clusters in batches
        for batch_start in range(0, total_clusters, batch_size):
            batch_end = min(batch_start + batch_size, total_clusters)
            batch_clusters = actual_clusters[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            
            self.verbose_reporter.stat_line(f"Processing batch {batch_num}: clusters {batch_start+1}-{batch_end}")
            
            # Get current codebook state for this batch
            current_codes, version = await self.shared_codebook.get_current_snapshot()
            self.verbose_reporter.stat_line(f"Codebook has {len(current_codes)} codes (version {version})")
            
            # Process batch concurrently with throttling
            async def throttled_process(cluster_id, cluster_ideas, current_codes):
                async with throttler:
                    return await self._process_single_cluster(cluster_id, cluster_ideas, current_codes)
            
            batch_tasks = [
                throttled_process(cluster_id, cluster_groups[cluster_id], current_codes)
                for cluster_id in batch_clusters
            ]
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Apply codebook updates from this batch
            batch_new_codes = 0
            batch_replaced_codes = 0
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.verbose_reporter.error(f"Cluster processing failed: {result}")
                    continue
                
                if result and result.get('status') == 'completed':
                    successful_clusters += 1
                    
                    # Apply codebook updates from Step 4
                    step4_result = result.get('step4_result', {})
                    if step4_result.get('new_codes_added'):
                        for code_info in step4_result['new_codes_added']:
                            added, _ = await self.shared_codebook.add_code_if_new(
                                code_info['code'],
                                code_info['definition']
                            )
                            if added:
                                batch_new_codes += 1
                                total_new_codes += 1
                    
                    if step4_result.get('codes_replaced'):
                        for code_info in step4_result['codes_replaced']:
                            replaced, _ = await self.shared_codebook.replace_code(
                                code_info['original_code'],
                                code_info['new_code'],
                                code_info['new_definition']
                            )
                            if replaced:
                                batch_replaced_codes += 1
                                total_replaced_codes += 1
            
            self.verbose_reporter.stat_line(f"Batch {batch_num} added {batch_new_codes} codes, modified {batch_replaced_codes}")
        
        # Final statistics
        final_codes, final_version = await self.shared_codebook.get_current_snapshot()
        
        return {
            'clusters_processed': total_clusters,
            'successful_clusters': successful_clusters,
            'total_new_codes': total_new_codes,
            'total_replaced_codes': total_replaced_codes,
            'final_codebook_size': len(final_codes),
            'final_version': final_version
        }
    
    def _group_by_actual_clusters(self) -> Dict[str, List[str]]:
        """Group response ideas by actual HDBSCAN cluster IDs"""
        cluster_groups = {}
        
        for response in self.cluster_data:
            if hasattr(response, 'response_ideas') and response.response_ideas:
                for idea in response.response_ideas:
                    if hasattr(idea, 'initial_cluster') and idea.initial_cluster is not None:
                        cluster_id = str(idea.initial_cluster)
                        
                        # Skip noise cluster
                        if cluster_id == '-1':
                            continue
                        
                        if cluster_id not in cluster_groups:
                            cluster_groups[cluster_id] = []
                        
                        cluster_groups[cluster_id].append(idea.idea)
        
        return cluster_groups
    
    async def _process_single_cluster(
        self, 
        cluster_id: str, 
        cluster_ideas: List[str],
        current_codes: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Process a single cluster through the LangChain sequential chain"""
        
        try:
            # Get cluster themes and summary
            cluster_summary = self._get_cluster_summary(cluster_id, cluster_ideas)
            
            # Format current codes for prompt
            code_text = self._format_codes_for_prompt(current_codes)
            
            # Step 2: Get candidate codes
            step2_inputs = {
                'language': DEFAULT_LANGUAGE,
                'survey_question': self.var_lab,
                'cluster_summary': cluster_summary,
                'code_text': code_text
            }
            
            try:
                candidate_codes_output = await self.step2_chain.ainvoke(step2_inputs)
                # Extract list from RootModel
                candidate_codes_list = candidate_codes_output.root if candidate_codes_output else []
                step2_result = {
                    'candidate_codes': candidate_codes_list,
                    'status': 'completed'
                }
            except Exception as e:
                self.verbose_reporter.error(f"Step 2 failed for cluster {cluster_id}: {e}")
                step2_result = {'candidate_codes': [], 'status': 'failed', 'error': str(e)}
                return {'cluster_id': cluster_id, 'status': 'step2_failed', 'error': str(e)}
            
            # Step 3: Generate code recommendations
            if step2_result['candidate_codes']:
                candidate_codes_text = "\n".join([
                    f"- {code.code}: {code.definition}" 
                    for code in step2_result['candidate_codes']
                ])
                
                step3_inputs = {
                    'language': DEFAULT_LANGUAGE,
                    'survey_question': self.var_lab,
                    'candidate_codes': candidate_codes_text,
                    'cluster_summary': cluster_summary
                }
                
                try:
                    code_recommendation = await self.step3_chain.ainvoke(step3_inputs)
                    step3_result = {
                        'code_recommendation': code_recommendation,
                        'status': 'completed'
                    }
                except Exception as e:
                    self.verbose_reporter.error(f"Step 3 failed for cluster {cluster_id}: {e}")
                    step3_result = {'status': 'failed', 'error': str(e)}
                    return {'cluster_id': cluster_id, 'status': 'step3_failed', 'error': str(e)}
            else:
                step3_result = {'status': 'skipped', 'reason': 'no_candidate_codes'}
            
            # Step 4: Validate and update codebook
            if step3_result.get('code_recommendation'):
                step3_recommendation = self._format_step3_recommendation(step3_result['code_recommendation'])
                
                step4_inputs = {
                    'language': DEFAULT_LANGUAGE,
                    'survey_question': self.var_lab,
                    'cluster_summary': cluster_summary,
                    'step3_recommendation': step3_recommendation,
                    'candidate_codes': candidate_codes_text
                }
                
                try:
                    validation_result = await self.step4_chain.ainvoke(step4_inputs)
                    step4_result = self._extract_codebook_updates(validation_result, cluster_id)
                except Exception as e:
                    self.verbose_reporter.error(f"Step 4 failed for cluster {cluster_id}: {e}")
                    step4_result = {'status': 'failed', 'error': str(e)}
            else:
                step4_result = {'status': 'skipped', 'reason': 'no_code_recommendation'}
            
            # Store results
            self.step2_results[cluster_id] = step2_result
            self.step3_results[cluster_id] = step3_result
            self.step4_results[cluster_id] = step4_result
            
            return {
                'cluster_id': cluster_id,
                'status': 'completed',
                'step2_result': step2_result,
                'step3_result': step3_result,
                'step4_result': step4_result
            }
            
        except Exception as e:
            self.verbose_reporter.error(f"Failed to process cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _get_cluster_summary(self, cluster_id: str, cluster_ideas: List[str]) -> str:
        """Get formatted cluster summary from theme_map or create from ideas"""
        if cluster_id in self.theme_map:
            themes = self.theme_map[cluster_id].get('themes', [])
            if themes:
                return "\n".join([f"Theme {i+1}: {theme}" for i, theme in enumerate(themes)])
        
        # Fallback: format cluster ideas
        return "\n".join([f"- {idea}" for idea in cluster_ideas[:10]])  # Limit to first 10
    
    def _format_codes_for_prompt(self, codes: List[Dict[str, str]]) -> str:
        """Format codes list for inclusion in prompt"""
        if not codes:
            return "No existing codes yet."
        
        return "\n".join([f"- {code['code']}: {code['definition']}" for code in codes])
    
    def _format_step3_recommendation(self, code_recommendation: models.CodeRecommendation) -> str:
        """Format Step 3 recommendation for Step 4 prompt"""
        recommendations = []
        for decision in code_recommendation.coding_decisions:
            action_details = decision.action_details
            if decision.decision == "use_existing" and action_details.codes_to_use:
                recommendations.append(f"Use existing codes: {', '.join(action_details.codes_to_use)}")
            elif decision.decision == "modify_existing" and action_details.modified_code_name:
                recommendations.append(f"Modify '{action_details.codes_to_modify}' to '{action_details.modified_code_name}': {action_details.modified_code_definition}")
            elif decision.decision == "create_new" and action_details.new_code_name:
                recommendations.append(f"Create new code '{action_details.new_code_name}': {action_details.new_code_definition}")
        
        return "\n".join(recommendations)
    
    def _extract_codebook_updates(self, validation_result: models.ValidationResult, cluster_id: str) -> Dict[str, Any]:
        """Extract codebook updates from ValidationResult model"""
        new_codes_added = []
        codes_replaced = []
        
        # Parse code validations to extract updates
        for code_validation in validation_result.code_validations:
            decision = code_validation.decision.upper()
            validated_code = code_validation.validated_code
            
            if decision in ['APPROVE', 'REVISE'] and validated_code:
                # Check if this is a modification of an existing code
                original_rec = code_validation.original_recommendation
                if 'modify' in original_rec.lower():
                    # This is a code replacement - extract original code name from recommendation
                    # For now, we'll add it as new code (TODO: improve parsing)
                    new_codes_added.append({
                        'code': validated_code.code,
                        'definition': validated_code.definition
                    })
                else:
                    # This is a new code
                    new_codes_added.append({
                        'code': validated_code.code,
                        'definition': validated_code.definition
                    })
        
        return {
            'validation_result': validation_result.model_dump(),
            'new_codes_added': new_codes_added,
            'codes_replaced': codes_replaced,
            'status': 'completed'
        }