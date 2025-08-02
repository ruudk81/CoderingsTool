import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import nest_asyncio
import time
from typing import Dict, List, Optional, Tuple
import instructor
from openai import AsyncOpenAI
import tiktoken
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import deque
from aiolimiter import AsyncLimiter

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, EmbeddingConfig, get_openai_rate_limits
from prompts import CODE_ASSIGNMENT_PROMPT
import models
from .verboseReporter import VerboseReporter
from utils.embedder import Embedder


async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

@dataclass
class WavePerformanceRecord:
    """Record of a wave execution for performance profiling"""
    wave_size: int
    batch_count: int
    duration: float
    timestamp: float
    token_count: Optional[int] = None
    request_count: Optional[int] = None

@dataclass
class WaveProfiler:
    """Empirical wave performance measurement and prediction system"""
    measurements: List[WavePerformanceRecord] = field(default_factory=list)
    cache_file: Optional[Path] = None
    ema_alpha: float = 0.3  # Exponential moving average smoothing factor
    
    def __post_init__(self):
        """Load historical measurements if cache file exists"""
        if self.cache_file and self.cache_file.exists():
            self.load_measurements()
    
    def record_wave(self, wave_size: int, batch_count: int, duration: float, 
                   token_count: Optional[int] = None, request_count: Optional[int] = None):
        """Record a wave execution for future prediction"""
        record = WavePerformanceRecord(
            wave_size=wave_size,
            batch_count=batch_count,
            duration=duration,
            timestamp=time.time(),
            token_count=token_count,
            request_count=request_count
        )
        self.measurements.append(record)
        
        # Log for analysis
        print(f"WAVE_PROFILE: size={wave_size}, batches={batch_count}, "
              f"duration={duration:.2f}s, throughput={batch_count/duration:.1f} batches/s")
        
        # Persist if cache file configured
        if self.cache_file:
            self.save_measurements()
    
    def get_duration_estimate(self, wave_size: int) -> float:
        """Get empirical duration estimate for a wave size"""
        if not self.measurements:
            # Fallback to original linear model if no data
            return 4.0 + wave_size * 0.25
        
        # Find measurements for similar wave sizes (±20% tolerance)
        tolerance = max(1, int(wave_size * 0.2))
        similar_measurements = [
            m for m in self.measurements 
            if abs(m.wave_size - wave_size) <= tolerance
        ]
        
        if similar_measurements:
            # Use recent measurements with exponential moving average
            recent_measurements = sorted(similar_measurements, key=lambda x: x.timestamp)[-5:]
            if len(recent_measurements) == 1:
                return recent_measurements[0].duration
            
            # Apply EMA to recent measurements
            ema_duration = recent_measurements[0].duration
            for measurement in recent_measurements[1:]:
                ema_duration = self.ema_alpha * measurement.duration + (1 - self.ema_alpha) * ema_duration
            return ema_duration
        
        # Interpolate/extrapolate from existing data
        return self._interpolate_duration(wave_size)
    
    def _interpolate_duration(self, wave_size: int) -> float:
        """Interpolate duration estimate from existing measurements"""
        if len(self.measurements) < 2:
            return 4.0 + wave_size * 0.25  # Fallback
        
        # Simple linear interpolation between closest measurements
        sorted_measurements = sorted(self.measurements, key=lambda x: x.wave_size)
        
        # Find bounding measurements
        smaller = [m for m in sorted_measurements if m.wave_size <= wave_size]
        larger = [m for m in sorted_measurements if m.wave_size > wave_size]
        
        if smaller and larger:
            # Interpolate between bounds
            lower = smaller[-1]  # Largest smaller measurement
            upper = larger[0]   # Smallest larger measurement
            
            if lower.wave_size == upper.wave_size:
                return lower.duration
            
            # Linear interpolation
            ratio = (wave_size - lower.wave_size) / (upper.wave_size - lower.wave_size)
            return lower.duration + ratio * (upper.duration - lower.duration)
        
        elif smaller:
            # Extrapolate from trend of recent measurements
            recent = sorted_measurements[-3:]  # Use last 3 measurements
            if len(recent) >= 2:
                # Calculate trend
                size_diff = recent[-1].wave_size - recent[0].wave_size
                duration_diff = recent[-1].duration - recent[0].duration
                if size_diff > 0:
                    trend = duration_diff / size_diff
                    return recent[-1].duration + trend * (wave_size - recent[-1].wave_size)
            
            # Fallback: use most recent similar measurement
            return smaller[-1].duration
        
        else:
            # All measurements are larger, use smallest
            return larger[0].duration
    
    def get_optimal_wave_size(self, time_budget: float, max_wave_size: int) -> int:
        """Find optimal wave size within time budget based on empirical data"""
        best_size = 1
        best_efficiency = 0.0
        
        for size in range(1, min(max_wave_size + 1, 51)):  # Test up to reasonable limit
            estimated_duration = self.get_duration_estimate(size)
            if estimated_duration <= time_budget:
                efficiency = size / estimated_duration  # batches per second
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_size = size
            else:
                break  # Larger sizes will exceed budget
        
        return best_size
    
    def get_performance_summary(self) -> Dict:
        """Get summary statistics of wave performance"""
        if not self.measurements:
            return {"total_measurements": 0}
        
        durations = [m.duration for m in self.measurements]
        wave_sizes = [m.wave_size for m in self.measurements]
        
        return {
            "total_measurements": len(self.measurements),
            "wave_size_range": f"{min(wave_sizes)}-{max(wave_sizes)}",
            "duration_range": f"{min(durations):.1f}-{max(durations):.1f}s",
            "avg_duration": f"{np.mean(durations):.1f}s",
            "avg_throughput": f"{np.mean([m.batch_count/m.duration for m in self.measurements]):.1f} batches/s"
        }
    
    def save_measurements(self):
        """Save measurements to cache file"""
        if not self.cache_file:
            return
        
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "measurements": [
                {
                    "wave_size": m.wave_size,
                    "batch_count": m.batch_count,
                    "duration": m.duration,
                    "timestamp": m.timestamp,
                    "token_count": m.token_count,
                    "request_count": m.request_count
                }
                for m in self.measurements
            ]
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_measurements(self):
        """Load measurements from cache file"""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            self.measurements = [
                WavePerformanceRecord(
                    wave_size=m["wave_size"],
                    batch_count=m["batch_count"],
                    duration=m["duration"],
                    timestamp=m["timestamp"],
                    token_count=m.get("token_count"),
                    request_count=m.get("request_count")
                )
                for m in data.get("measurements", [])
            ]
            
            print(f"📊 Loaded {len(self.measurements)} wave performance measurements from cache")
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not load wave performance cache: {e}")
            self.measurements = []

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
        
        # Initialize wave performance profiler with cache persistence
        cache_dir = Path.home() / ".cache" / "coderingsTool" / "wave_performance"
        cache_file = cache_dir / f"wave_perf_{self.config.model.replace('/', '_')}.json"
        self.wave_profiler = WaveProfiler(cache_file=cache_file)
        
        # Initialize aiolimiter with sliding windows (6-second windows)
        rate_limits = get_openai_rate_limits(self.config.model)
        # Use 90% of limits for safety margin
        safe_rpm = int(rate_limits.requests_per_minute * 0.9)
        safe_tpm = int(rate_limits.tokens_per_minute * 0.9)
        
        # Create limiters with 6-second sliding windows (10 windows per minute)
        self.window_seconds = 6
        windows_per_minute = 60 / self.window_seconds
        
        # Distribute rate limits across windows
        self.request_limiter = AsyncLimiter(safe_rpm / windows_per_minute, self.window_seconds)
        self.token_limiter = AsyncLimiter(safe_tpm / windows_per_minute, self.window_seconds)
        
        # Show profiler status
        perf_summary = self.wave_profiler.get_performance_summary()
        if perf_summary["total_measurements"] > 0:
            print(f"📊 Performance history: Loaded {perf_summary['total_measurements']} measurements")
            print(f"📊 Average throughput: {perf_summary.get('avg_throughput', 'N/A')}")
        
        print(f"🔄 AIOLIMITER RATE LIMITING: {safe_rpm} RPM, {safe_tpm} TPM (90% of {rate_limits.requests_per_minute}/{rate_limits.tokens_per_minute})")
        print(f"🚦 Using {self.window_seconds}s sliding windows for smooth throughput")

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

    async def _process_batch_with_limiter(self, batch: List[tuple], batch_idx: int) -> List[CodeAssignmentResponse]:
        """Process a single batch with aiolimiter rate limiting"""
        # Calculate resource needs for this batch
        api_calls_per_batch = 25  # 5 sub-batches × 5 ideas per sub-batch
        estimated_tokens_per_call = 800
        tokens_per_batch = api_calls_per_batch * estimated_tokens_per_call
        
        # Acquire capacity from both limiters
        async with self.request_limiter.acquire(api_calls_per_batch):
            async with self.token_limiter.acquire(tokens_per_batch):
                # Process the batch
                batch_start_time = asyncio.get_event_loop().time()
                try:
                    results = await self._process_batch(batch, batch_idx)
                    batch_duration = asyncio.get_event_loop().time() - batch_start_time
                    
                    # Record performance for profiler
                    self.wave_profiler.record_wave(
                        wave_size=1,
                        batch_count=1,
                        duration=batch_duration,
                        request_count=api_calls_per_batch,
                        token_count=tokens_per_batch
                    )
                    
                    return results
                except Exception as e:
                    print(f"❌ Batch {batch_idx + 1} failed: {str(e)}")
                    return []


    async def _process_all_batches(self, batches: List[List[tuple]]) -> List[CodeAssignmentResponse]:
        """Process all batches using aiolimiter for optimal throughput"""
        total_ideas = sum(len(batch) for batch in batches)
        
        # Calculate total sub-batches for reporting
        total_sub_batches = sum(len(self._create_sub_batches(batch, sub_batch_size=5)) for batch in batches)
        
        self.verbose_reporter.stat_line(
            f"Processing {total_ideas} ideas in {len(batches)} batches "
            f"({total_sub_batches} concurrent sub-batches)..."
        )
        
        print(f"\n🔄 AIOLIMITER PROCESSING: {len(batches)} batches")
        print(f"🚦 Using {self.window_seconds}s sliding windows for smooth throughput")
        print(f"🚀 Launching all tasks immediately - aiolimiter handles scheduling")
        
        # Launch ALL batch tasks immediately - aiolimiter handles the rate limiting
        start_time = asyncio.get_event_loop().time()
        
        # Create all tasks at once
        tasks = [
            asyncio.create_task(self._process_batch_with_limiter(batch, i))
            for i, batch in enumerate(batches)
        ]
        
        # Wait for all tasks to complete
        all_results = []
        completed = 0
        
        # Use asyncio.as_completed to track progress
        for coro in asyncio.as_completed(tasks):
            batch_results = await coro
            all_results.extend(batch_results)
            completed += 1
            
            # Progress reporting every 10 batches
            if completed % 10 == 0 or completed == len(batches):
                elapsed = asyncio.get_event_loop().time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"🔄 Progress: {completed}/{len(batches)} batches completed ({rate:.1f} batches/s)")
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Final summary
        print(f"\n🔄 AIOLIMITER PROCESSING COMPLETED")
        print(f"   ✅ Total batches processed: {len(batches)}")
        print(f"   ⏱️  Total execution time: {total_time:.1f}s")
        print(f"   📊 Average throughput: {len(batches)/total_time:.1f} batches/second")
        print(f"   🚦 Event-driven scheduling with {self.window_seconds}s sliding windows")
        
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
        
        # Performance summary
        perf_summary = self.wave_profiler.get_performance_summary()
        print(f"\n🎯 AIOLIMITER OPTIMIZATION COMPLETED:")
        print(f"  📊 Performance measurements: {perf_summary.get('total_measurements', 0)} batch records")
        if perf_summary.get('total_measurements', 0) > 0:
            print(f"  📊 Historical performance: {perf_summary.get('avg_throughput', 'N/A')} avg throughput")
        print(f"  🚦 Event-driven rate limiting with {self.window_seconds}s sliding windows")
        print(f"  🚦 Zero polling overhead - optimal CPU usage")
        print(f"  🚀 All tasks launched immediately for maximum concurrency")
        
        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())