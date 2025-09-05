import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

import nest_asyncio
import instructor
from openai import AsyncOpenAI, RateLimitError
import tiktoken
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler

# === MODELS ========================================================================================================
from pydantic import BaseModel
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, get_openai_rate_limits
from prompts import CODE_ASSIGNMENT_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter

async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

@dataclass
class OptimalStrategy:
    """Evidence-based optimal processing strategy"""
    target_time_seconds: float
    launch_rate_per_second: float
    concurrent_limit: int
    bottleneck_type: str
    total_requests: int
    total_tokens: int
    safety_factor: float


class WorkloadAnalyzer:
    """Analyzes workload and calculates optimal processing strategy based on evidence"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def measure_token_usage(self, sample_prompts: List[str], num_samples: int = 10) -> float:
        """Measure actual token usage from real prompts"""
        if not sample_prompts:
            return 1500  # Conservative fallback
        
        # Sample random prompts if we have many
        sample_size = min(num_samples, len(sample_prompts))
        sampled_prompts = sample_prompts[:sample_size]
        
        token_counts = []
        for prompt in sampled_prompts:
            # Count prompt tokens
            prompt_tokens = len(self.encoding.encode(prompt))
            # Estimate completion tokens (typically 20-30% of prompt)
            completion_tokens = int(prompt_tokens * 0.25)
            total_tokens = prompt_tokens + completion_tokens
            token_counts.append(total_tokens)
        
        return statistics.mean(token_counts)
    
    def calculate_optimal_strategy(self, total_ideas: int, avg_tokens_per_request: float) -> OptimalStrategy:
        """Calculate mathematically optimal processing strategy"""
        # Get API limits from config
        rate_limits = get_openai_rate_limits(self.model_name)
        
        # Calculate total resource requirements
        total_requests = total_ideas
        total_tokens = total_ideas * avg_tokens_per_request
        
        # Calculate minimum time based on constraints
        time_by_requests = total_requests / rate_limits.requests_per_minute * 60
        time_by_tokens = total_tokens / rate_limits.tokens_per_minute * 60
        
        # Find bottleneck and minimum time
        bottleneck_time = max(time_by_requests, time_by_tokens)
        bottleneck_type = 'tokens' if time_by_tokens > time_by_requests else 'requests'
        
        # Apply safety factor (use 95% of capacity)
        safety_factor = 0.95
        target_time = bottleneck_time / safety_factor
        
        # Calculate optimal launch rate
        optimal_launch_rate = total_requests / target_time
        
        # Calculate concurrent request limit (3 seconds of buffer)
        concurrent_limit = int(optimal_launch_rate * 3)
        
        return OptimalStrategy(
            target_time_seconds=target_time,
            launch_rate_per_second=optimal_launch_rate,
            concurrent_limit=concurrent_limit,
            bottleneck_type=bottleneck_type,
            total_requests=total_requests,
            total_tokens=total_tokens,
            safety_factor=safety_factor
        )


class SlidingWindowMonitor:
    """Real-time monitoring of API usage with sliding windows"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: int = 60):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        
        # Sliding windows for tracking usage
        self.requests_window = deque()  # timestamps
        self.tokens_window = deque()    # (timestamp, token_count) tuples
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
    
    def _cleanup_windows(self):
        """Remove entries older than window_seconds"""
        cutoff_time = time.time() - self.window_seconds
        
        # Clean requests window
        while self.requests_window and self.requests_window[0] < cutoff_time:
            self.requests_window.popleft()
        
        # Clean tokens window
        while self.tokens_window and self.tokens_window[0][0] < cutoff_time:
            self.tokens_window.popleft()
    
    def record_request(self, tokens_used: int):
        """Record a completed API request"""
        now = time.time()
        self.requests_window.append(now)
        self.tokens_window.append((now, tokens_used))
        
        self.total_requests += 1
        self.total_tokens += tokens_used
        
        self._cleanup_windows()
    
    def get_current_utilization(self) -> Dict:
        """Get current resource utilization"""
        self._cleanup_windows()
        
        current_rpm = len(self.requests_window)
        current_tpm = sum(tokens for _, tokens in self.tokens_window)
        
        return {
            'current_rpm': current_rpm,
            'current_tpm': current_tpm,
            'rpm_utilization': current_rpm / self.rpm_limit,
            'tpm_utilization': current_tpm / self.tpm_limit,
            'rpm_remaining': self.rpm_limit - current_rpm,
            'tpm_remaining': self.tpm_limit - current_tpm,
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'elapsed_time': time.time() - self.start_time
        }
    
    def is_near_limit(self, threshold: float = 0.9) -> bool:
        """Check if we're approaching rate limits"""
        util = self.get_current_utilization()
        return (util['rpm_utilization'] > threshold or 
                util['tpm_utilization'] > threshold)


class SmartAPIClient:
    """API client with intelligent retry logic and precise rate limiting"""
    
    def __init__(self, throttler: Throttler, monitor: SlidingWindowMonitor, config: CodeAssignmentConfig, encoding, model_config: ModelConfig):
        self.throttler = throttler
        self.monitor = monitor
        self.config = config
        self.client = async_client
        self.model_config = model_config
        self.encoding = encoding
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def make_request(self, prompt: str, idea_id: str) -> Dict:
        """Make API request with intelligent retry and rate limiting"""
        
        # Apply precision rate limiting
        async with self.throttler:
            try:
                # Make the API call
                model_name = self.model_config.get_model_for_stage('code_assignment')
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed,
                    response_model=CodeAssignmentResponse,
                    max_retries=0  # Let tenacity handle retries
                )
                
                # Record successful request with accurate token count
                estimated_tokens = len(self.encoding.encode(prompt))
                self.monitor.record_request(estimated_tokens)
                
                return {
                    'idea_id': response.idea_id,
                    'idea': response.idea,
                    'assigned_codes': response.assigned_codes,
                    'assignment_confidence': response.assignment_confidence,
                    'assignment_rationale': response.assignment_rationale
                }
                
            except Exception as e:
                self.verbose_reporter.error(f"API request failed for idea {idea_id}: {str(e)}")
                raise


class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str
    assigned_themes: Optional[List[str]] = None


class EmbeddingLoader:
    """Utility class for loading and managing embeddings from cache"""
    
    @staticmethod
    def load_idea_embeddings_from_cache(cache_manager, filename):
        """Load idea embeddings from cache step 'embeddings'"""
        embeddings_results = cache_manager.load_from_cache(
            filename, "embeddings", models.EmbeddingsModel
        )
        
        if not embeddings_results:
            return []
        
        # Extract all ideas with their embeddings
        ideas_with_embeddings = []
        for result in embeddings_results:
            if result.response_ideas:
                for idea in result.response_ideas:
                    if idea.idea_embedding is not None:
                        ideas_with_embeddings.append({
                            'idea': idea.idea,
                            'idea_id': idea.idea_id,
                            'embedding': idea.idea_embedding,
                            'respondent_id': result.respondent_id
                        })
        
        return ideas_with_embeddings
    
    @staticmethod
    def format_codes_for_embedding(enriched_codebook):
        """Format enriched codebook entries for embedding generation"""
        # Use definitions only to match idea embedding format (just text)
        return [code.definition for code in enriched_codebook]


class CodeAssigner:
    """
    Simplified code assignment with direct LLM processing.
    LLM sees all codes in codebook instead of similarity-filtered subset.
    """
    
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        codebook: List[models.Codebook],
        var_lab: str,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
        cached_idea_embeddings: Optional[List[Dict]] = None,
        config: Optional[CodeAssignmentConfig] = None,
        model_config: Optional[ModelConfig] = None,
        verbose: bool = False,
        prompt_printer = None):
        
        self.cluster_models = cluster_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.model_config = model_config or ModelConfig()
        self.model = self.model_config.get_model_for_stage('code_assignment')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        
        # Cache for idea embeddings if provided (for compatibility)
        self._cached_idea_embeddings = cached_idea_embeddings
        
        # Theme mapping for code-to-theme assignments
        self.code_to_theme_mapping = code_to_theme_mapping or {}
        
        # Initialize components for optimal strategy
        self.workload_analyzer = WorkloadAnalyzer(self.model)
        
        # Initialize rate limits and monitoring
        rate_limits = get_openai_rate_limits(self.model)
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute
        
        self.verbose_reporter.stat_line(f"Model: {self.model}")
        self.verbose_reporter.stat_line(f"API Limits: {self.rpm_limit} RPM, {self.tpm_limit:,} TPM")


    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        """Map assigned codes to their themes using cached mapping"""
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas for processing (no embeddings needed)"""
        # Use cached embeddings if provided (for compatibility)
        if self._cached_idea_embeddings:
            all_ideas = []
            for cached_idea in self._cached_idea_embeddings:
                all_ideas.append((
                    cached_idea['respondent_id'],
                    cached_idea['idea_id'],
                    cached_idea['idea']
                ))
            self.verbose_reporter.stat_line(f"Using {len(all_ideas)} cached ideas")
            return all_ideas
        
        # Otherwise extract from cluster models
        all_ideas = []
        
        for model in self.cluster_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    all_ideas.append((
                        model.respondent_id,
                        idea_submodel.idea_id,
                        idea_submodel.idea
                    ))
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas found for respondent {model.respondent_id}")
        
        return all_ideas

    def _create_prompt(self, idea_id: str, idea_text: str) -> str:
        """Create prompt for a single idea with ALL codes from codebook"""
        # Format ALL codes for prompt
        candidate_codes_text = "\n".join([
            f"Code label: {code.code}\nCode description: {code.definition}\n" 
            for code in self.codebook
        ])
        
        # Create prompt
        prompt = CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            candidate_codes=candidate_codes_text
        )
        
        return prompt

    async def _process_single_idea(self, idea_data: tuple, api_client: SmartAPIClient) -> CodeAssignmentResponse:
        """Process a single idea assignment"""
        respondent_id, idea_id, idea_text = idea_data
        
        try:
            # Create prompt
            prompt = self._create_prompt(idea_id, idea_text)
            
            # Capture prompt for debugging if enabled
            if self.prompt_printer and not self._captured_prompt:
                self.prompt_printer.capture_prompt(
                    step_name="code_assignment",
                    utility_name="CodeAssigner",
                    prompt_content=prompt,
                    prompt_type="code_assignment",
                    metadata={
                        "model": self.model,
                        "var_lab": self.var_lab,
                        "language": self.language,
                        "idea_id": idea_id
                    }
                )
                self._captured_prompt = True
            
            # Make API call
            response_data = await api_client.make_request(prompt, idea_id)
            
            # Add theme assignments
            assigned_themes = self._assign_themes_to_codes(response_data['assigned_codes'])
            
            return CodeAssignmentResponse(
                idea_id=response_data['idea_id'],
                idea=response_data['idea'],
                assigned_codes=response_data['assigned_codes'],
                assigned_themes=assigned_themes,
                assignment_confidence=response_data['assignment_confidence'],
                assignment_rationale=response_data['assignment_rationale']
            )
            
        except Exception as e:
            # Return fallback response (first available code)
            fallback_code = self.codebook[0].code if self.codebook else "Unknown"
            fallback_themes = self._assign_themes_to_codes([fallback_code]) if fallback_code != "Unknown" else []
            
            return CodeAssignmentResponse(
                idea_id=idea_id,
                idea=idea_text,
                assigned_codes=[fallback_code],
                assigned_themes=fallback_themes,
                assignment_confidence=0.1,
                assignment_rationale=f"Processing failed: {str(e)}"
            )

    def _merge_results_into_models(self, assignment_results: List[CodeAssignmentResponse]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into model structure"""
        
        # If using cached embeddings, create simple models from assignments
        if self._cached_idea_embeddings and not self.cluster_models:
            coded_models = []
            
            # Group assignments by respondent_id
            respondent_assignments = {}
            for result in assignment_results:
                # Extract respondent_id from the cached data
                for cached_idea in self._cached_idea_embeddings:
                    if cached_idea['idea_id'] == result.idea_id:
                        resp_id = cached_idea['respondent_id']
                        if resp_id not in respondent_assignments:
                            respondent_assignments[resp_id] = []
                        respondent_assignments[resp_id].append(result)
                        break
            
            # Create CodeAssignedModel for each respondent
            for resp_id, assignments in respondent_assignments.items():
                assigned_ideas = []
                for assignment in assignments:
                    assigned_idea = models.AssignedIdeaSubmodel(
                        idea_id=assignment.idea_id,
                        idea=assignment.idea,
                        assigned_codes=assignment.assigned_codes,
                        assignment_confidence=assignment.assignment_confidence,
                        assignment_rationale=assignment.assignment_rationale,
                        assigned_themes=assignment.assigned_themes
                    )
                    assigned_ideas.append(assigned_idea)
                
                coded_model = models.CodeAssignedModel(
                    respondent_id=resp_id,
                    response='',  # We don't have the full response text
                    response_ideas=assigned_ideas
                )
                coded_models.append(coded_model)
            
            return coded_models
        
        # Original logic for cluster models
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

    async def _process_with_optimal_strategy(self, all_ideas: List[tuple]) -> List[CodeAssignmentResponse]:
        """Process all ideas using evidence-based optimal strategy"""
        
        # Step 1: Analyze workload and calculate optimal strategy
        sample_prompts = [self._create_prompt(idea[1], idea[2]) for idea in all_ideas[:10]]
        avg_tokens = self.workload_analyzer.measure_token_usage(sample_prompts)
        strategy = self.workload_analyzer.calculate_optimal_strategy(len(all_ideas), avg_tokens)
        
        #print("\n🎯 OPTIMAL STRATEGY CALCULATED:")
        #print(f"📊 Total requests: {strategy.total_requests}")
        #print(f"📊 Estimated tokens: {strategy.total_tokens:,} ({avg_tokens:.0f} per request)")
        #print(f"📊 Bottleneck: {strategy.bottleneck_type}")
        #print(f"📊 Target time: {strategy.target_time_seconds:.1f}s")
        #print(f"📊 Launch rate: {strategy.launch_rate_per_second:.1f} requests/second")
        #print(f"📊 Max concurrent: {strategy.concurrent_limit}")
        #print(f"📊 Capacity utilization: {strategy.safety_factor:.1%}")
        
        # Step 2: Initialize precision throttler and monitor
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second, period=1.0)
        monitor = SlidingWindowMonitor(self.rpm_limit, self.tpm_limit)
        api_client = SmartAPIClient(throttler, monitor, self.config, self.workload_analyzer.encoding, self.model_config)
        
        # Step 3: Launch all requests with precision timing
        #print(f"\n🚀 LAUNCHING {len(all_ideas)} REQUESTS AT OPTIMAL RATE")
        #start_time = time.time() # needed for debug reporting
        
        # Create all tasks - throttler handles the timing
        tasks = [
            asyncio.create_task(self._process_single_idea(idea_data, api_client))
            for idea_data in all_ideas
        ]
        
        # Monitor progress
        all_results = []
        completed = 0
        
        # Process results as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_results.append(result)
            completed += 1
            
            if completed % 100 == 0 or completed == len(all_ideas):
                self.verbose_reporter.progress_line(completed, len(all_ideas), "code assignments")
                
                #debug
                # elapsed = time.time() - start_time
                # current_rate = completed / elapsed if elapsed > 0 else 0
                # #util = monitor.get_current_utilization()
                #print(f"   📊 Rate: {current_rate:.1f} requests/second")
                #print(f"   📊 RPM utilization: {util['rpm_utilization']:.1%}")
                #print(f"   📊 TPM utilization: {util['tpm_utilization']:.1%}")
        
        # debub
        #total_time = time.time() - start_time
        #final_util = monitor.get_current_utilization()
        #print("\n🎯 OPTIMAL STRATEGY COMPLETED:")
        #print(f"   ✅ Target time: {strategy.target_time_seconds:.1f}s")
        #print(f"   ✅ Actual time: {total_time:.1f}s")
        #print(f"   ✅ Performance: {(strategy.target_time_seconds/total_time):.1%} of target")
        #print(f"   📊 Average rate: {len(all_ideas)/total_time:.1f} requests/second")
        #print(f"   📊 Peak RPM utilization: {final_util['rpm_utilization']:.1%}")
        #print(f"   📊 Peak TPM utilization: {final_util['tpm_utilization']:.1%}")
        #print(f"   📊 Total requests: {final_util['total_requests']}")
        #print(f"   📊 Total tokens: {final_util['total_tokens']:,}")
        
        return all_results

    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method to assign codes using direct LLM processing"""
        self.verbose_reporter.section_header("CODE ASSIGNMENT PROCESSING")
        
        # Extract all ideas
        all_ideas = self._extract_all_ideas()
        total_ideas = len(all_ideas)
        
        if total_ideas == 0:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []
        
        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas with {len(self.codebook)} available codes")
        
        # Process with optimal strategy
        all_results = await self._process_with_optimal_strategy(all_ideas)
        
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
        
        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())
