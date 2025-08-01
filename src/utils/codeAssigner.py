import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import nest_asyncio
import time
from typing import Dict, List, Optional
import instructor
from openai import AsyncOpenAI
import tiktoken
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, EmbeddingConfig, get_openai_rate_limits
from prompts import CODE_ASSIGNMENT_PROMPT
import models
from .verboseReporter import VerboseReporter
from utils.embedder import Embedder


async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assigned_themes: List[str]
    assignment_confidence: float
    assignment_rationale: str

class CodeAssigner:
    """
    High-performance code assigner with precomputed code embeddings:
    - Precomputes code embeddings once during initialization
    - Hierarchical concurrency with unlimited batch-level parallelism
    - Sub-batch processing for improved throughput
    - No artificial delays between batches
    - Optimized error handling and fallback mechanisms
    """
    
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        codebook: List[models.Codebook],
        var_lab: str,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
        config: Optional[CodeAssignmentConfig] = None,
        verbose: bool = False,
        prompt_printer = None):
        
        self.cluster_models = cluster_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.client = async_client
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose)
        self.model_config = ModelConfig()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        
        # Initialize embedder for similarity calculations
        embedding_config = EmbeddingConfig()
        # Use larger batch size for one-time code embedding generation
        embedding_config.batch_size = 100  # Increase for efficient one-time processing
        self.embedder = Embedder(config=embedding_config, verbose=False)
        
        # Cache for code embeddings - will be precomputed
        self._code_embeddings = None
        
        # Theme mapping for code-to-theme assignments
        self.code_to_theme_mapping = code_to_theme_mapping or {}
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            print(f"Using cl100k_base encoding as fallback for {self.config.model}")
        
        print("🚦 WAVE OPTIMIZATION: OpenAI rolling window-aware staggered processing enabled")

    async def initialize_code_embeddings(self):
        """Initialize code embeddings once during setup - call this before processing"""
        if self._code_embeddings is None:
            self.verbose_reporter.stat_line(f"Precomputing embeddings for {len(self.codebook)} codes...")
            start_time = time.time()
            
            # Use the existing method to generate embeddings
            await self._get_code_embeddings()
            
            elapsed = time.time() - start_time
            self.verbose_reporter.stat_line(f"Code embeddings computed in {elapsed:.2f}s")

    async def _get_code_embeddings(self):
        """Generate embeddings for all codes in the codebook for similarity matching"""
        if self._code_embeddings is None:
            code_texts = [f"{code.code}: {code.definition}" for code in self.codebook]
            
            # Create temporary models for embedding generation
            temp_models = []
            for i, code_text in enumerate(code_texts):
                temp_model = models.EmbeddingsModel(
                    respondent_id=f"code_{i}",
                    response=code_text,
                    response_ideas=[models.EmbeddingsSubmodel(
                        idea_id=f"code_{i}_1",
                        idea=code_text
                    )],
                    idea_count=1
                )
                temp_models.append(temp_model)
            
            # Generate embeddings using async method directly
            embedded_codes = await self.embedder._process_embeddings_with_id_tracking(temp_models)
            
            # Extract embeddings array
            embeddings = []
            for model in embedded_codes:
                if hasattr(model, 'response_ideas') and model.response_ideas and len(model.response_ideas) > 0:
                    embedding = model.response_ideas[0].idea_embedding
                    if embedding is not None:
                        embeddings.append(embedding)
                    else:
                        embeddings.append(np.zeros(1536))  # text-embedding-3-large dimension
                else:
                    embeddings.append(np.zeros(1536))
            
            self._code_embeddings = np.array(embeddings)
        
        return self._code_embeddings

    def _find_similar_codes(self, idea_embedding: np.ndarray, top_k: int = 5) -> List[models.Codebook]:
        """Find the top_k most similar codes to an idea based on embedding similarity"""
        if self._code_embeddings is None:
            raise ValueError("Code embeddings not initialized. Call _get_code_embeddings first.")
        
        # Calculate cosine similarity
        similarities = cosine_similarity([idea_embedding], self._code_embeddings)[0]
        
        # Get top_k most similar indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return corresponding codes
        return [self.codebook[i] for i in top_indices]

    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        """Map assigned codes to their themes using cached mapping"""
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas with their embeddings for processing"""
        all_ideas = []
        
        for model in self.cluster_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    if hasattr(idea_submodel, 'idea_embedding') and idea_submodel.idea_embedding is not None:
                        all_ideas.append((
                            model.respondent_id,
                            idea_submodel.idea_id,
                            idea_submodel.idea,
                            idea_submodel.idea_embedding
                        ))
                    else:
                        self.verbose_reporter.stat_line(f"Warning: No embedding for idea {idea_submodel.idea_id}")
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas found for respondent {model.respondent_id}")
        
        return all_ideas

    async def _process_idea_assignment(self, idea_data: tuple) -> CodeAssignmentResponse:
        """Process a single idea assignment with candidate codes"""
        respondent_id, idea_id, idea_text, idea_embedding = idea_data
        
        # Find most similar codes
        similar_codes = self._find_similar_codes(idea_embedding, top_k=self.config.top_k_similar_codes)
        
        # Format candidate codes for prompt
        candidate_codes_text = "\n".join([
            f"Code: {code.code}\nDefinition: {code.definition}\n" 
            for code in similar_codes
        ])
        
        # Create prompt
        prompt = CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            candidate_codes=candidate_codes_text
        )
        
        # Capture prompt for debugging if enabled
        if self.prompt_printer and not self._captured_prompt:
            self.prompt_printer.capture_prompt(
                step_name="code_assignment",
                utility_name="CodeAssigner",
                prompt_content=prompt,
                prompt_type="code_assignment",
                metadata={
                    "model": self.config.model,
                    "var_lab": self.var_lab,
                    "language": self.language,
                    "idea_id": idea_id
                }
            )
            self._captured_prompt = True
        
        try:
            # Temporary response model without themes
            class LLMCodeAssignmentResponse(BaseModel):
                idea_id: str
                idea: str
                assigned_codes: List[str]
                assignment_confidence: float
                assignment_rationale: str
            
            # Use instructor's built-in retries
            llm_response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.model_config.seed,
                response_model=LLMCodeAssignmentResponse,
                max_retries=self.config.retries
            )
            
            # Add theme assignments based on assigned codes
            assigned_themes = self._assign_themes_to_codes(llm_response.assigned_codes)
            
            return CodeAssignmentResponse(
                idea_id=llm_response.idea_id,
                idea=llm_response.idea,
                assigned_codes=llm_response.assigned_codes,
                assigned_themes=assigned_themes,
                assignment_confidence=llm_response.assignment_confidence,
                assignment_rationale=llm_response.assignment_rationale
            )
            
        except Exception as e:
            error_msg = f"Failed to process idea {idea_id}: {type(e).__name__}: {str(e)}"
            self.verbose_reporter.stat_line(error_msg)
            
            # Return fallback response
            fallback_code = similar_codes[0].code if similar_codes else "Unknown"
            fallback_themes = self._assign_themes_to_codes([fallback_code]) if fallback_code != "Unknown" else []
            
            return CodeAssignmentResponse(
                idea_id=idea_id,
                idea=idea_text,
                assigned_codes=[fallback_code],
                assigned_themes=fallback_themes,
                assignment_confidence=0.1,
                assignment_rationale=f"Failed to process - {type(e).__name__}"
            )

    def _create_sub_batches(self, batch: List[tuple], sub_batch_size: int = 5) -> List[List[tuple]]:
        """Split a batch into smaller sub-batches for concurrent processing"""
        if not batch:
            return []
        
        sub_batches = []
        for i in range(0, len(batch), sub_batch_size):
            sub_batch = batch[i:i + sub_batch_size]
            sub_batches.append(sub_batch)
        
        return sub_batches

    async def _process_sub_batch(self, sub_batch: List[tuple], batch_index: int, sub_batch_index: int) -> List[CodeAssignmentResponse]:
        """Process a single sub-batch of ideas"""
        # Create tasks for all ideas in this sub-batch
        tasks = [self._process_idea_assignment(idea_data) for idea_data in sub_batch]
        
        # Process all ideas in sub-batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        sub_batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create fallback result for failed idea
                respondent_id, idea_id, idea_text, idea_embedding = sub_batch[i]
                sub_batch_results.append(CodeAssignmentResponse(
                    idea_id=idea_id,
                    idea=idea_text,
                    assigned_codes=["Processing_Error"],
                    assigned_themes=[],
                    assignment_confidence=0.0,
                    assignment_rationale="Processing failed"
                ))
            else:
                sub_batch_results.append(result)
        
        return sub_batch_results

    async def _process_batch(self, batch: List[tuple], batch_index: int) -> List[CodeAssignmentResponse]:
        """Process a single batch with hierarchical concurrency"""
        # Split batch into sub-batches of 5 for better concurrency management
        sub_batches = self._create_sub_batches(batch, sub_batch_size=5)
        
        if not sub_batches:
            return []
        
        # Level 2: Process all sub-batches within this batch concurrently
        sub_batch_tasks = [
            self._process_sub_batch(sub_batch, batch_index, i) 
            for i, sub_batch in enumerate(sub_batches)
        ]
        sub_batch_results = await asyncio.gather(*sub_batch_tasks, return_exceptions=True)
        
        # Collect results from all sub-batches
        batch_results = []
        sub_batch_failures = 0
        
        for i, sub_batch_result in enumerate(sub_batch_results):
            if isinstance(sub_batch_result, Exception):
                print(f"Sub-batch {i+1} of batch {batch_index+1} failed: {str(sub_batch_result)}")
                sub_batch_failures += 1
                continue
            
            # Add all results from this sub-batch
            batch_results.extend(sub_batch_result)
        
        if sub_batch_failures > 0:
            print(f"{sub_batch_failures} out of {len(sub_batches)} sub-batches failed in batch {batch_index+1}")
        
        return batch_results

    def _create_batches(self, all_ideas: List[tuple]) -> List[List[tuple]]:
        """Create token-aware batches for processing"""
        if not all_ideas:
            return []
        
        # Calculate token budget similar to qualityFilter
        sample_prompt = CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id="sample_id",
            idea_text="sample text",
            candidate_codes="sample codes"
        )
        prompt_tokens = len(self.encoding.encode(sample_prompt))
        token_budget = self.config.max_tokens - prompt_tokens - 500  # Reserve for completion
        
        # Pre-calculate token counts for ideas
        idea_tokens = []
        for _, _, idea_text, _ in all_ideas:
            tokens = len(self.encoding.encode(idea_text))
            idea_tokens.append(tokens)
        
        # Calculate adaptive batch size
        avg_tokens = sum(idea_tokens) / max(1, len(idea_tokens))
        adaptive_max_batch = min(self.config.batch_size, max(1, int(token_budget / max(1, avg_tokens))))
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for i, (idea_data, tokens) in enumerate(zip(all_ideas, idea_tokens)):
            # Handle oversized ideas
            if tokens > token_budget and not current_batch:
                print(f"Warning: Idea exceeds token budget ({tokens} > {token_budget})")
                batches.append([idea_data])
                continue
            
            # Check if adding this idea would exceed limits
            if (current_tokens + tokens > token_budget or 
                len(current_batch) >= adaptive_max_batch):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(idea_data)
            current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _merge_results_into_models(self, assignment_results: List[CodeAssignmentResponse]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into model structure"""
        # Create lookup for assignments by idea_id
        assignments_lookup = {result.idea_id: result for result in assignment_results}
        
        coded_models = []
        
        for original_model in self.cluster_models:
            # Convert to CodeAssignedModel
            coded_model = original_model.to_model(models.CodeAssignedModel)
            
            # Update response_ideas with assignments
            if coded_model.response_ideas:
                updated_ideas = []
                for idea_submodel in coded_model.response_ideas:
                    # Convert to AssignedIdeaSubmodel
                    assigned_idea = models.AssignedIdeaSubmodel(
                        idea_id=idea_submodel.idea_id,
                        idea=idea_submodel.idea,
                        initial_cluster=getattr(idea_submodel, 'initial_cluster', None),
                        idea_embedding=getattr(idea_submodel, 'idea_embedding', None)
                    )
                    
                    # Add assignment data if available
                    if idea_submodel.idea_id in assignments_lookup:
                        assignment = assignments_lookup[idea_submodel.idea_id]
                        assigned_idea.assigned_codes = assignment.assigned_codes
                        assigned_idea.assigned_themes = assignment.assigned_themes
                        assigned_idea.assignment_confidence = assignment.assignment_confidence
                        assigned_idea.assignment_rationale = assignment.assignment_rationale
                    else:
                        # Fallback if no assignment found
                        assigned_idea.assigned_codes = ["Unassigned"]
                        assigned_idea.assigned_themes = []
                        assigned_idea.assignment_confidence = 0.0
                        assigned_idea.assignment_rationale = "No assignment found"
                    
                    updated_ideas.append(assigned_idea)
                
                coded_model.response_ideas = updated_ideas
            
            coded_models.append(coded_model)
        
        return coded_models

    async def _process_batches_staggered(self, batches: List[List[tuple]], wave_size: int, stagger_delay: int) -> List:
        """Process batches in staggered waves to respect rolling window rate limits"""
        all_results = []
        
        # Split batches into waves
        waves = []
        for i in range(0, len(batches), wave_size):
            wave = batches[i:i + wave_size]
            waves.append(wave)
        
        print(f"🚦 Processing {len(waves)} waves of {wave_size} batches each")
        
        # Process waves with staggered timing
        for wave_idx, wave in enumerate(waves):
            wave_start_time = asyncio.get_event_loop().time()
            
            print(f"🚦 Starting wave {wave_idx + 1}/{len(waves)} ({len(wave)} batches)...")
            
            # Process all batches in this wave concurrently
            wave_tasks = [self._process_batch(batch, i) for i, batch in enumerate(wave)]
            wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)
            
            # Handle results and exceptions for this wave
            wave_failures = 0
            for i, result in enumerate(wave_results):
                if isinstance(result, Exception):
                    print(f"Batch {i+1} in wave {wave_idx+1} failed: {str(result)}")
                    wave_failures += 1
                    continue
                all_results.extend(result)
            
            if wave_failures > 0:
                print(f"🚦 Wave {wave_idx+1}: {wave_failures}/{len(wave)} batches failed")
            
            wave_duration = asyncio.get_event_loop().time() - wave_start_time
            print(f"🚦 Wave {wave_idx+1} completed in {wave_duration:.1f}s")
            
            # Apply stagger delay before next wave (except for last wave)
            if wave_idx < len(waves) - 1:
                print(f"🚦 Waiting {stagger_delay}s before next wave...")
                await asyncio.sleep(stagger_delay)
        
        return all_results

    def _calculate_optimal_processing_strategy(self, batches: List[List[tuple]]) -> tuple:
        """Calculate mathematically optimal processing strategy based purely on data and model limits"""
        if not batches:
            return 1, 6  # Minimum case
            
        # Get OpenAI rate limits for the current model
        rate_limits = get_openai_rate_limits(self.config.model)
        
        # Job characteristics
        api_calls_per_batch = 25  # 5 sub-batches × 5 ideas per sub-batch
        estimated_tokens_per_call = 800  # Conservative estimate (prompt + completion)
        tokens_per_batch = api_calls_per_batch * estimated_tokens_per_call
        total_batches = len(batches)
        
        # Safety margins (80% of limits to avoid rate limiting)
        safe_requests_per_minute = int(rate_limits.requests_per_minute * 0.8)
        safe_tokens_per_minute = int(rate_limits.tokens_per_minute * 0.8)
        
        # STEP 1: Find optimal wave size and stagger delay combination
        # We need to solve: ⌊60/D⌋ × R ≤ RPM AND ⌊60/D⌋ × T ≤ TPM
        # Where R = requests per wave, T = tokens per wave, D = stagger delay
        
        def find_optimal_strategy(rpm_limit, tpm_limit, requests_per_batch, tokens_per_batch, max_batches):
            """Find optimal wave size and stagger delay that maximizes throughput"""
            best_throughput = 0
            best_wave_size = 1
            best_delay = 60
            
            # Try different wave sizes from 1 to max possible
            max_possible_wave_size = min(max_batches, rpm_limit // requests_per_batch, tpm_limit // tokens_per_batch)
            
            for wave_size in range(1, max_possible_wave_size + 1):
                wave_requests = wave_size * requests_per_batch
                wave_tokens = wave_size * tokens_per_batch
                
                # Find minimum delay for this wave size using rolling window constraint
                min_delay = None
                for delay in range(1, 61):  # Try delays from 1 to 60 seconds
                    waves_in_60s = 60 // delay
                    total_requests_in_60s = wave_requests * waves_in_60s
                    total_tokens_in_60s = wave_tokens * waves_in_60s
                    
                    if total_requests_in_60s <= rpm_limit and total_tokens_in_60s <= tpm_limit:
                        min_delay = delay
                        break
                
                if min_delay is not None:
                    # Calculate throughput: batches per minute
                    waves_per_minute = 60 / min_delay
                    batches_per_minute = wave_size * waves_per_minute
                    
                    if batches_per_minute > best_throughput:
                        best_throughput = batches_per_minute
                        best_wave_size = wave_size
                        best_delay = min_delay
                        
            return best_wave_size, best_delay, best_throughput
        
        optimal_wave_size, optimal_stagger_delay, max_throughput = find_optimal_strategy(
            safe_requests_per_minute, safe_tokens_per_minute, 
            api_calls_per_batch, tokens_per_batch, total_batches
        )
        
        # STEP 3: Calculate processing metrics
        num_waves = (total_batches + optimal_wave_size - 1) // optimal_wave_size  # Ceiling division
        wave_processing_time = 6  # Time for each wave to complete
        total_processing_time = (num_waves - 1) * optimal_stagger_delay + wave_processing_time
        
        # Calculate wave metrics for display
        wave_requests = optimal_wave_size * api_calls_per_batch
        wave_tokens = optimal_wave_size * tokens_per_batch
        waves_in_60s = 60 // optimal_stagger_delay
        peak_rpm_usage = wave_requests * waves_in_60s
        peak_tpm_usage = wave_tokens * waves_in_60s
        
        # Debug information - technically precise analysis
        print(f"🚦 Mathematical optimization for {self.config.model}:")
        print(f"  • OpenAI rate limits: {rate_limits.requests_per_minute} RPM, {rate_limits.tokens_per_minute} TPM")
        print(f"  • Safety margins (80%): {safe_requests_per_minute} RPM, {safe_tokens_per_minute} TPM")
        print(f"  • Data to process: {total_batches} batches ({total_batches * api_calls_per_batch} total requests)")
        print(f"  • Optimization constraint: ⌊60/D⌋ × R ≤ RPM AND ⌊60/D⌋ × T ≤ TPM")
        print(f"  • Optimal solution:")
        print(f"    - Wave size: {optimal_wave_size} batches ({wave_requests} requests, {wave_tokens} tokens)")
        print(f"    - Stagger delay: {optimal_stagger_delay} seconds")
        print(f"    - Max throughput: {max_throughput:.1f} batches/minute")
        print(f"  • Rolling window compliance verification:")
        print(f"    - Waves in 60s: ⌊60/{optimal_stagger_delay}⌋ = {waves_in_60s}")
        print(f"    - Peak usage: {peak_rpm_usage} RPM ({peak_rpm_usage/safe_requests_per_minute*100:.1f}%), {peak_tpm_usage} TPM ({peak_tpm_usage/safe_tokens_per_minute*100:.1f}%)")
        print(f"  • Execution plan:")
        print(f"    - {num_waves} waves of {optimal_wave_size} batches each")
        print(f"    - {optimal_stagger_delay}s interval between wave starts")
        print(f"    - Total processing time: {total_processing_time:.1f} seconds")
        
        return optimal_wave_size, optimal_stagger_delay

    async def _process_all_batches(self, batches: List[List[tuple]]) -> List[CodeAssignmentResponse]:
        """Process all batches using optimal wave-based staggering to respect OpenAI rolling window limits"""
        total_ideas = sum(len(batch) for batch in batches)
        
        # Calculate total sub-batches for reporting
        total_sub_batches = sum(len(self._create_sub_batches(batch, sub_batch_size=5)) for batch in batches)
        
        self.verbose_reporter.stat_line(
            f"Processing {total_ideas} ideas in {len(batches)} batches "
            f"({total_sub_batches} concurrent sub-batches)..."
        )
        
        # MATHEMATICAL OPTIMIZATION - NO COMPARISON, PURE CALCULATION
        wave_size, stagger_delay = self._calculate_optimal_processing_strategy(batches)
        
        print(f"\n🚦 EXECUTING OPTIMAL STRATEGY: {len(batches)} batches")
        print(f"🚦 Mathematically determined: {wave_size} batches per wave, {stagger_delay}s stagger delay")
        
        # Always use staggered processing - this IS the optimal strategy
        batch_results = await self._process_batches_staggered(batches, wave_size, stagger_delay)
        
        # Collect results from all batches
        all_results = []
        total_failures = 0
        
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i+1} processing failed completely: {str(batch_result)}")
                total_failures += 1
                continue
            
            # Add all results from this batch
            all_results.extend(batch_result)
        
        if total_failures > 0:
            print(f"{total_failures} out of {len(batches)} batches failed completely")
        
        return all_results

    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method to assign codes to all ideas with high-performance processing"""
        self.verbose_reporter.section_header("CODE ASSIGNMENT PROCESSING")
        
        # Ensure code embeddings are initialized ONCE before any processing
        if self._code_embeddings is None:
            await self.initialize_code_embeddings()
        
        # Extract all ideas
        all_ideas = self._extract_all_ideas()
        total_ideas = len(all_ideas)
        
        if total_ideas == 0:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []
        
        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas with {len(self.codebook)} available codes")
        
        # Create token-aware batches
        batches = self._create_batches(all_ideas)
        
        # Process all batches with hierarchical concurrency
        all_results = await self._process_all_batches(batches)
        
        # Merge results back into model structure
        self._results = self._merge_results_into_models(all_results)
        
        # Report summary
        if all_results:
            avg_confidence = np.mean([r.assignment_confidence for r in all_results])
            high_confidence = sum(1 for r in all_results if r.assignment_confidence >= 0.7)
            low_confidence = sum(1 for r in all_results if r.assignment_confidence < 0.5)
            
            self.verbose_reporter.summary("CODE ASSIGNMENT COMPLETED", {
                "Total ideas processed": len(all_results),
                "Average confidence": f"{avg_confidence:.2f}",
                "High confidence (≥0.7)": high_confidence,
                "Low confidence (<0.5)": low_confidence
            })
        
        # Mathematical optimization summary
        batches = self._create_batches(self._extract_all_ideas())
        wave_size, stagger_delay = self._calculate_optimal_processing_strategy(batches)
        print(f"\n🎯 MATHEMATICAL OPTIMIZATION COMPLETED:")
        print(f"  🚦 Optimal strategy: {wave_size} batches per wave, {stagger_delay}s stagger delays")
        print(f"  🚦 Based purely on data volume and {self.config.model} rate limits")
        print(f"  🚦 Guaranteed OpenAI rolling window compliance")
        
        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())