import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import re
import asyncio
import subprocess
import logging
import time
import statistics
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

import nest_asyncio
from pydantic import BaseModel
#from openai import RateLimitError
#from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler
#from openai import AsyncOpenAI
#import instructor
#import tiktoken
#import spacy

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, HUNSPELL_PATH, DUTCH_DICT_PATH, ENGLISH_DICT_PATH, SpellCheckConfig, DEFAULT_SPELLCHECK_CONFIG, ModelConfig, DEFAULT_MODEL_CONFIG, get_openai_rate_limits
from prompts import SPELLCHECK_INSTRUCTIONS

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding, get_spacy_nlp_conditional

logger = logging.getLogger(__name__)

# Nederlands or Engels
DICT_PATH = DUTCH_DICT_PATH if DEFAULT_LANGUAGE == "Dutch" else ENGLISH_DICT_PATH

class SpellCheckModel(BaseModel):
    respondent_id: Any
    original_response: str
    corrected_response: Optional[str] = None

class SpellCorrectionTask(BaseModel):
    respondent_id: Any 
    original_response: str 
    response_with_oov_placeholders: str 
    oov_words: str 
    suggestions: str

class SpellCorrectionBatch(BaseModel):
    tasks: List[SpellCorrectionTask] 

class CorrectionItem(BaseModel):
    respondent_id: Any 
    corrected_response: str 

class LLMCorrectionResponse(BaseModel):
    corrections: List[CorrectionItem] 

@dataclass
class SpellCheckOptimalStrategy:
    """Evidence-based optimal processing strategy for spell checking"""
    target_time_seconds: float
    launch_rate_per_second: float
    concurrent_limit: int
    bottleneck_type: str
    total_requests: int
    total_tokens: int
    safety_factor: float
    batch_size: int

class SpellCheckWorkloadAnalyzer:
    """Analyzes spell checking workload and calculates optimal processing strategy"""
    
    def __init__(self, model_name: str, encoding, config: SpellCheckConfig):
        self.model_name = model_name
        self.encoding = encoding
        self.config = config
    
    def measure_token_usage(self, sample_batches: List[List[Any]], base_prompt_template: str, var_lab: str) -> float:
        """Measure actual token usage from real spell checking prompts"""
        if not sample_batches:
            return 550  
        
        token_counts = []
        for batch in sample_batches[:3]:  # Sample first 3 batches
            tasks_string = ""
            for i, task in enumerate(batch):
                tasks_string += (
                    f"Task {i + 1}:\n"
                    f"Respondent ID: {task.respondent_id}\n"
                    f"Response: \"{task.response_with_oov_placeholders}\"\n"
                    f"Misspelled words: {task.oov_words}\n"
                    f"Suggested corrections: {task.suggestions}\n\n")
            
            prompt = base_prompt_template.format(
                language=DEFAULT_LANGUAGE,
                var_lab=var_lab,
                tasks=tasks_string
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            # Spell checking typically has shorter completions (structured output)
            completion_tokens = int(prompt_tokens * 0.15)
            total_tokens = prompt_tokens + completion_tokens
            token_counts.append(total_tokens)
        
        return statistics.mean(token_counts) if token_counts else 550
    
    def calculate_optimal_strategy(self, total_batches: int, avg_tokens_per_batch: float) -> SpellCheckOptimalStrategy:
        """Calculate evidence-based strategy for spell checking with rate smoothing"""
        # Get API limits from config
        rate_limits = get_openai_rate_limits(self.model_name)
        
        # Calculate total resource requirements
        total_requests = total_batches
        total_tokens = total_requests * avg_tokens_per_batch
        
        # Calculate optimal sustained rate (what we can maintain)
        optimal_sustained_rate = rate_limits.requests_per_minute / 60  # req/sec
        optimal_sustained_tokens = rate_limits.tokens_per_minute / 60  # tokens/sec
        
        # Find bottleneck for sustained rate
        time_by_requests = total_requests / optimal_sustained_rate
        time_by_tokens = total_tokens / optimal_sustained_tokens
        
        # Use evidence-based approach: plan for sustained rate
        bottleneck_time = max(time_by_requests, time_by_tokens)
        bottleneck_type = 'tokens' if time_by_tokens > time_by_requests else 'requests'
        
        # Apply configurable utilization for spell checking
        safety_factor = self.config.rate_limit_safety_factor
        target_time = bottleneck_time / safety_factor
        
        # Calculate launch rate
        optimal_launch_rate = total_requests / target_time
        
        # Aggressive concurrent limit for spell checking (configurable burst capacity)
        concurrent_limit = int(optimal_launch_rate * self.config.concurrent_burst_multiplier)
        
        return SpellCheckOptimalStrategy(
            target_time_seconds=target_time,
            launch_rate_per_second=optimal_launch_rate,
            concurrent_limit=concurrent_limit,
            bottleneck_type=bottleneck_type,
            total_requests=total_requests,
            total_tokens=total_tokens,
            safety_factor=safety_factor,
            batch_size=1
        )

class SpellCheckSlidingWindowMonitor:
    """Real-time monitoring of API usage for spell checking with sliding windows"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: int = 60, rate_limit_utilization: float = 0.98):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        self.rate_limit_utilization = rate_limit_utilization
        
        # Thread-safe tracking across all concurrent operations
        self._lock = asyncio.Lock()
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
    
    async def can_proceed(self, estimated_tokens: int = 0) -> bool:
        """Check if we can make request within 95% of 60-second limits"""
        async with self._lock:
            self._cleanup_windows()
            
            # Calculate current usage in 60-second window
            current_rpm = len(self.requests_window)
            current_tpm = sum(tokens for _, tokens in self.tokens_window)
            
            # Use configurable percentage of limits for spell checking
            would_exceed_rpm = (current_rpm + 1) > (self.rpm_limit * self.rate_limit_utilization)
            would_exceed_tpm = (current_tpm + estimated_tokens) > (self.tpm_limit * self.rate_limit_utilization)
            
            return not (would_exceed_rpm or would_exceed_tpm)
    
    async def record_request(self, tokens_used: int):
        """Record a completed API request"""
        async with self._lock:
            now = time.time()
            self.requests_window.append(now)
            self.tokens_window.append((now, tokens_used))
            
            self.total_requests += 1
            self.total_tokens += tokens_used
            
            self._cleanup_windows()
    
    async def get_current_utilization(self) -> Dict:
        """Get current resource utilization"""
        async with self._lock:
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
    
class HunspellSession:
    def __init__(self, hunspell_path, dict_path):
        self.process = subprocess.Popen(
            [hunspell_path, "-a", "-d", dict_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1)
        self.process.stdout.readline()
       
    def check_word(self, word):
        self.process.stdin.write(word + '\n')
        self.process.stdin.flush()
        result = self.process.stdout.readline().strip()
        while True:
            peek = self.process.stdout.readline().strip()
            if not peek:
                break
            result += "\n" + peek
            
        return result

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        self.process.terminate()

class SpellChecker:
    def __init__(self, config: SpellCheckConfig = None, model_config: ModelConfig = None, openai_api_key: Optional[str] = None, verbose: bool = False, prompt_printer = None, verbose_reporter: Optional['VerboseReporter'] = None):
        self.config = config or DEFAULT_SPELLCHECK_CONFIG
        self.model_config = model_config or ModelConfig()
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.model = self.model_config.get_model_for_stage('spell_check')
        
        # Initialize suggestion cache
        self.suggestion_cache = {} if self.config.enable_suggestion_caching else None
        self.suggestion_cache_hits = 0
        
        # Instructor-patched async OpenAI client for structured output (cached)
        self.client = get_openai_client(self.openai_api_key)
        
        self.hunspell_path = HUNSPELL_PATH
        self.dict_path = DICT_PATH
        self.prompt_printer = prompt_printer 
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        
        # Configuration reporting (verbose only)
        if self.verbose_reporter.enabled:
            self.verbose_reporter.empty_line()
            print("Spell checker configuration:")
            self.verbose_reporter.stat_line(f"Model: {self.model}", indent=1)
            self.verbose_reporter.stat_line(f"Language: {DEFAULT_LANGUAGE}", indent=1)
            self.verbose_reporter.stat_line(f"Dictionary: {self.dict_path}", indent=1)
            self.verbose_reporter.stat_line(f"Hunspell path: {self.hunspell_path}", indent=1)
            self.verbose_reporter.stat_line(f"Batch size: {self.config.batch_size}", indent=1)
            
        
        # Installation check with verbose error reporting
        if not self.check_hunspell_installation():
            if self.verbose_reporter.enabled:
                self.verbose_reporter.warning("Hunspell is not properly installed or configured - spell checking may fail")
                self.verbose_reporter.warning(f"Expected Hunspell at: {self.hunspell_path}")
                self.verbose_reporter.warning(f"Expected dictionary at: {self.dict_path}")
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("OK Hunspell installation verified")
        
        # Stats tracking
        self.stats = {
            'words_checked': 0,
            'oov_words_found': 0,
            'unique_oov_words': 0,
            'oov_words_in_tasks': 0,
            'responses_with_tasks': 0,
            'tasks_filtered_out': 0,
            'llm_calls_made': 0,
            'llm_calls_successful': 0,
            'llm_calls_failed': 0,
            'corrections_attempted': 0,
            'corrections_applied': 0,
            'corrections_rejected_validation': 0,
            'corrections_no_response': 0,
            'dictionary_verifications': 0,
            'processing_time': 0.0,
            'suggestion_cache_hits': 0,
            'suggestion_cache_size': 0
        }
    
    @staticmethod 
    def get_nlp(spell_check_enabled: bool = True):  
        """Load SpaCy language model conditionally with Streamlit caching"""
        return get_spacy_nlp_conditional(spell_check_enabled)
   
    @staticmethod
    @lru_cache(maxsize=1)
    def check_hunspell_installation() -> bool:
        try:
            result = subprocess.run(
                [HUNSPELL_PATH, "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error(f"Hunspell not found at {HUNSPELL_PATH}. Please check the path.")
            return False
        except Exception as e:
            logger.error(f"Error checking Hunspell installation: {str(e)}")
            return False
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def cached_levenshtein_distance(word1: str, word2: str) -> int:
        if word1 == word2:
            return 0
        
        # Create a matrix
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize the matrix
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if word1[i-1] == word2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return dp[m][n]
    
    async def run_hunspell_word_async(self, word: str) -> List[str]:
        """Simple subprocess approach for speed"""
        def run_hunspell():
            process = subprocess.Popen(
                [HUNSPELL_PATH, "-a", "-d", self.dict_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding="utf-8"
            )
            output, _ = process.communicate(input=f"{word}\n")
            return output

        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(None, run_hunspell)
        lines = [line for line in output.splitlines() if line and not line.startswith("@")]

        if lines and lines[0].startswith("&"):
            match = re.search(r": (.+)", lines[0])
            if match:
                suggestions = match.group(1).split(", ")
                return suggestions
        if lines and lines[0].startswith("*"):
            return [word]  # Word is correct
        return []
    
    async def verify_correction_with_dictionary(self, word: str) -> bool:
        """Verify LLM corrections against dictionary"""
        result = await self.run_hunspell_word_async(word)
        self.stats['dictionary_verifications'] += 1
        return bool(result and result[0] == word)
    
    async def pre_validate_suggestions(self, suggestions: List[str]) -> List[str]:
        """Pre-validate suggestions against dictionary to filter out invalid ones"""
        if not suggestions:
            return []
        
        # Validate all suggestions in parallel
        validation_tasks = [self.verify_correction_with_dictionary(sug.strip('.,!?;:"\'()[]{}')) 
                           for sug in suggestions if sug and sug != "OOV"]
        validation_results = await asyncio.gather(*validation_tasks)
        
        # Return only valid suggestions
        valid_suggestions = [sug for sug, is_valid in zip(suggestions, validation_results) if is_valid]
        return valid_suggestions
          
    async def find_best_split_for_spellcheck(self, oov_word: str) -> Tuple[str, str]:    
        excluded_tags = {"SYM", "PUNCT", "X", "SPACE", "NUM"}

        left_split_attempts = [(oov_word[:i], "left") for i in range(4, len(oov_word) + 1)]
        right_split_attempts = [(oov_word[i:], "right") for i in range(len(oov_word) - 3)]  

        all_splits = left_split_attempts + right_split_attempts
        processed_splits = list(self.get_nlp().pipe([split for split, _ in all_splits], batch_size=self.config.spacy_batch_size))

        valid_splits = [
            (split, tag) for (split, tag), doc in zip(all_splits, processed_splits)
            if len(split) > 2 and all(token.pos_ not in excluded_tags and token.vector_norm > 5 for token in doc) ]

        left_parts = [split for split, tag in valid_splits if tag == "left"]
        right_parts = [split for split, tag in valid_splits if tag == "right"]

        left_part = max(left_parts, key=len) if left_parts else ""
        right_part = max(right_parts, key=len) if right_parts else ""
        
        batch_candidates = []
        if right_part:
            left_remaining = oov_word[:-len(right_part)]
            right_remaining = right_part
            batch_candidates.extend([left_remaining, right_remaining])
        elif left_part:
            left_remaining = left_part
            right_remaining = oov_word[len(left_part):]
            batch_candidates.extend([left_remaining, right_remaining])
        else:
            batch_candidates.append(oov_word)

        hunspell_results = await asyncio.gather(
            *(self.run_hunspell_word_async(candidate) for candidate in batch_candidates))

        normalized_hunspell_results = {
            candidate: result if isinstance(result, list) else [result]
            for candidate, result in zip(batch_candidates, hunspell_results)}

        all_suggestions = [
            suggestion
            for suggestions in normalized_hunspell_results.values()
            for suggestion in suggestions]

        if not all(isinstance(s, str) for s in all_suggestions):
            raise TypeError("all_suggestions contains non-string values.")

        processed_suggestions = list(self.get_nlp().pipe(all_suggestions, batch_size=self.config.spacy_batch_size))

        filtered_suggestions = {
            candidate: [suggestion for suggestion, doc in zip(normalized_hunspell_results[candidate], processed_suggestions) if doc.vector_norm > 5]
            for candidate in batch_candidates}

        if left_part:
            right_remaining = oov_word[len(left_part):]
            right_part_suggestions = filtered_suggestions.get(right_remaining, [])
            right_part = (
                min(right_part_suggestions, key=lambda s: self.cached_levenshtein_distance(right_remaining, s))
                if right_part_suggestions else right_part)

        if right_part:
            left_remaining = oov_word[:-len(right_part)]
            left_part_suggestions = filtered_suggestions.get(left_remaining, [])
            left_part = (
                min(left_part_suggestions, key=lambda s: self.cached_levenshtein_distance(left_remaining, s))
                if left_part_suggestions else left_part)

        return left_part, right_part

    async def find_best_suggestions_batch_async(self, unique_oov_words: List[str]) -> Dict[str, List[Any]]:
        """Process unique OOV words with aggressive parallel processing and caching"""
        # Check cache first if enabled
        if self.suggestion_cache is not None:
            cached_suggestions = {}
            uncached_words = []
            
            for word in unique_oov_words:
                if word in self.suggestion_cache:
                    cached_suggestions[word] = self.suggestion_cache[word]
                    self.suggestion_cache_hits += 1
                else:
                    uncached_words.append(word)
            
            # If all words are cached, return immediately
            if not uncached_words:
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line(f"All {len(unique_oov_words)} words found in suggestion cache")
                return cached_suggestions
            
            # Process only uncached words
            unique_oov_words = uncached_words
            if self.verbose_reporter.enabled and cached_suggestions:
                self.verbose_reporter.stat_line(f"Found {len(cached_suggestions)} words in cache, processing {len(uncached_words)} new words")
        
        # Process uncached words
        if len(unique_oov_words) <= self.config.max_words_per_chunk:
            # Small dataset - use original method
            new_suggestions = await self._process_suggestions_single_batch(unique_oov_words)
        else:
            # Large dataset - use aggressive parallel processing
            new_suggestions = await self._process_suggestions_parallel_chunks(unique_oov_words)
        
        # Update cache if enabled
        if self.suggestion_cache is not None:
            self.suggestion_cache.update(new_suggestions)
            
            # Merge with cached suggestions
            if 'cached_suggestions' in locals():
                new_suggestions.update(cached_suggestions)
        
        return new_suggestions
    
    async def _process_suggestions_single_batch(self, unique_oov_words: List[str]) -> Dict[str, List[Any]]:
        """Original single-batch processing for small datasets"""
        sorted_oov_words = sorted(unique_oov_words)

        async def process_word(word):
            try:
                # Parallel Hunspell and splitting operations for better performance
                unsplit_task = self.run_hunspell_word_async(word)
                split_task = self.find_best_split_for_spellcheck(word)
                
                # Execute both operations concurrently
                unsplit_suggestions, (left_part, right_part) = await asyncio.gather(unsplit_task, split_task)
                
                split_suggestion = f"{left_part} {right_part}" if (left_part and right_part) else None
                unsplit_suggestion = (
                    min(unsplit_suggestions, key=lambda s: self.cached_levenshtein_distance(word, s))
                    if unsplit_suggestions else None)
                return word, unsplit_suggestion, split_suggestion
            except Exception as e:
                logger.error(f"Error processing word '{word}': {e}")
                return word, None, None
       
        results = await asyncio.gather(*(process_word(word) for word in sorted_oov_words))

        best_suggestions = defaultdict(list)
        for result in results:
            if result and len(result) == 3:
                best_suggestions[result[0]].append(result[1:])

        return best_suggestions
    
    async def _process_suggestions_parallel_chunks(self, unique_oov_words: List[str]) -> Dict[str, List[Any]]:
        """Quality Filter style aggressive parallel processing with workload analysis"""
        
        # WORKLOAD ANALYSIS (Quality Filter Style)
        print("[SUGGESTION GENERATION ANALYSIS]")
        sorted_oov_words = sorted(unique_oov_words)
        total_operations = len(sorted_oov_words)
        
        # Estimate processing rate based on system capabilities
        # Each word requires: Hunspell calls + SpaCy analysis + vector calculations
        estimated_ops_per_second = 8.0  # Conservative estimate based on your data (9.8 words/sec observed)
        estimated_time_sequential = total_operations / estimated_ops_per_second
        
        # Calculate optimal concurrent processing strategy
        max_concurrent = self.config.suggestion_processing_semaphore_limit
        optimal_batch_size = min(100, max(10, total_operations // max_concurrent))  # 10-100 words per batch
        total_batches = (total_operations + optimal_batch_size - 1) // optimal_batch_size
        
        # Aggressive parallelism calculation
        parallel_efficiency = min(max_concurrent, total_batches) * 0.75  # 75% efficiency due to coordination overhead
        estimated_time_parallel = estimated_time_sequential / parallel_efficiency
        
        print(f"- Words to process: {total_operations:,}")
        print(f"- Estimated sequential time: {estimated_time_sequential:.1f}s ({estimated_ops_per_second:.1f} words/sec)")
        print(f"- Optimal strategy: {max_concurrent} concurrent batches, {optimal_batch_size} words each") 
        print(f"- Total batches: {total_batches}")
        print(f"- Estimated parallel time: {estimated_time_parallel:.1f}s (speedup: {estimated_time_sequential/estimated_time_parallel:.1f}x)")
        print("Processing suggestion generation...")

        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create batches for optimal processing
        batches = []
        for i in range(0, len(sorted_oov_words), optimal_batch_size):
            batch = sorted_oov_words[i:i + optimal_batch_size]
            batches.append(batch)
        
        async def process_batch_aggressive(batch_words: List[str], batch_index: int):
            """Quality filter style batch processing with aggressive concurrency"""
            async with semaphore:
                batch_start_time = time.time()
                batch_results = defaultdict(list)
                
                # Process all words in batch concurrently (like quality filter processes items)
                async def process_single_word(word):
                    try:
                        # Parallel Hunspell and splitting operations
                        unsplit_task = self.run_hunspell_word_async(word)
                        split_task = self.find_best_split_for_spellcheck(word)
                        
                        # Execute both operations concurrently
                        unsplit_suggestions, (left_part, right_part) = await asyncio.gather(unsplit_task, split_task)
                        
                        split_suggestion = f"{left_part} {right_part}" if (left_part and right_part) else None
                        unsplit_suggestion = (
                            min(unsplit_suggestions, key=lambda s: self.cached_levenshtein_distance(word, s))
                            if unsplit_suggestions else None)
                        
                        return word, unsplit_suggestion, split_suggestion
                    except Exception as e:
                        logger.error(f"Error processing word '{word}': {e}")
                        return word, None, None
                
                # Process all words in this batch concurrently
                batch_tasks = [process_single_word(word) for word in batch_words]
                results = await asyncio.gather(*batch_tasks)
                
                # Collect results
                for result in results:
                    if result and len(result) == 3:
                        batch_results[result[0]].append(result[1:])
                
                batch_time = time.time() - batch_start_time
                batch_rate = len(batch_words) / max(batch_time, 0.1)
                
                return batch_results, batch_rate
        
        # Execute all batches with progress tracking (Quality Filter style)
        completed_batches = 0
        all_results = []
        total_rate_samples = []
        
        batch_tasks = [process_batch_aggressive(batch, idx) for idx, batch in enumerate(batches)]
        
        # Process batches and show progress like quality filter
        for completed_task in asyncio.as_completed(batch_tasks):
            batch_results, batch_rate = await completed_task
            all_results.append(batch_results)
            total_rate_samples.append(batch_rate)
            completed_batches += 1
            
            # Progress reporting (every 10% or every 10 batches)
            if completed_batches % max(1, total_batches // 10) == 0 or completed_batches % 10 == 0:
                progress_percent = (completed_batches / total_batches) * 100
                elapsed = time.time() - start_time
                current_rate = sum(len(batch) for batch in batches[:completed_batches]) / max(elapsed, 0.1)
                remaining_words = sum(len(batch) for batch in batches[completed_batches:])
                eta = remaining_words / max(current_rate, 0.1)
                print(f"Processing suggestion batches... {completed_batches}/{total_batches} ({progress_percent:.1f}%) [{current_rate:.1f} words/sec, ETA: {eta:.1f}s]")
        
        # Combine all results
        combined_results = defaultdict(list)
        for batch_result in all_results:
            for word, suggestions in batch_result.items():
                combined_results[word].extend(suggestions)
        
        # Final performance metrics
        total_time = time.time() - start_time
        total_rate = len(unique_oov_words) / max(total_time, 0.1)
        efficiency = (total_rate / estimated_ops_per_second) * 100 if estimated_ops_per_second > 0 else 100
        
        print(f"OK Completed suggestion generation: {len(unique_oov_words):,} words in {total_time:.1f}s ({total_rate:.1f} words/sec)")
        print(f"  Performance: {efficiency:.1f}% of estimated rate, {total_rate/estimated_ops_per_second:.1f}x speedup over sequential")
        
        return combined_results
        
    def create_correction_batches(self, tasks: List[Dict[str, Any]], prompt_header: str, max_tokens: int, completion_reserve: int) -> List[SpellCorrectionBatch]:
        tiktoken_model = DEFAULT_MODEL_CONFIG.get_model_for_stage('tiktoken_spellChecker')
        encoding = get_tiktoken_encoding(tiktoken_model)
            
        token_budget = max_tokens - len(encoding.encode(prompt_header)) - completion_reserve
        
        # Calculate optimal batch size based on rate limits and task count
        total_tasks = len(tasks)
        rate_limits = get_openai_rate_limits(self.model)
        
        # Target processing time of 60 seconds for optimal rate distribution
        target_time_seconds = 60
        optimal_requests = int(rate_limits.requests_per_minute * 0.9)  # Use 90% of RPM
        
        # Calculate optimal batch size
        if total_tasks <= optimal_requests:
            # Can process all in one minute - use larger batches
            optimal_batch_size = min(self.config.max_batch_size * 2, 20)
        else:
            # Need multiple minutes - optimize for steady flow
            optimal_batch_size = max(5, min(self.config.max_batch_size, total_tasks // optimal_requests))
        
        # Verbose reporting of batch optimization
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Batch optimization: {total_tasks} tasks, target batch size: {optimal_batch_size}", indent=1)
        
        batches = []
        current_batch_tasks = []
        current_batch_tokens = 0
        
        max_batch_size = optimal_batch_size
        
        for task in tasks:
            correction_task = SpellCorrectionTask(
                respondent_id=task['respondent_id'],
                original_response=task['response'],
                response_with_oov_placeholders=task['response_with_placeholders'],
                oov_words=task['oov_words'],
                suggestions=task['suggestions']
            )
            
            task_text = (
                f"Task:\n"
                f"Respondent ID: {task['respondent_id']}\n"
                f"Response: \"{task['response_with_placeholders']}\"\n"
                f"Misspelled words: {task['oov_words']}\n"
                f"Suggested corrections: {task['suggestions']}\n\n"
            )
            
            task_tokens = len(encoding.encode(task_text))
            
            if (current_batch_tokens + task_tokens > token_budget or 
                len(current_batch_tasks) >= max_batch_size):
                if current_batch_tasks:
                    batches.append(SpellCorrectionBatch(tasks=current_batch_tasks))
                current_batch_tasks = []
                current_batch_tokens = 0
            
            current_batch_tasks.append(correction_task)
            current_batch_tokens += task_tokens
        
        if current_batch_tasks:
            batches.append(SpellCorrectionBatch(tasks=current_batch_tasks))
        
        return batches

    async def get_best_corrections_with_ai(self, responses, best_suggestions_dict: Dict[str, List[Any]], var_lab: str, word_to_responses: Dict[str, List[int]] = None) -> Dict[str, str]:
        """Native async OpenAI client with validation - optimized with word-to-response mapping"""
        oov_words = list(best_suggestions_dict.keys())
        
        max_tokens = self.config.max_tokens  
        completion_reserve = self.config.completion_reserve
        
        corrected_sentences_dict = {}
        tasks = []
        prompt_header = SPELLCHECK_INSTRUCTIONS.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            tasks="")   
        
        responses_with_ids = [{'respondent_id': response.respondent_id, 'response': response.original_response} for response in responses]
    
        # Pre-validation tracking
        pre_validation_filtered = 0
        
        # Performance tracking for task creation
        task_creation_start = time.time()
        
        # Use inverted index if available, otherwise fall back to regex search
        if word_to_responses is not None:
            print(f"  • Creating correction tasks using optimized inverted index...")
            
            # Pre-compute suggestion strings for all OOV words to avoid redundant processing
            word_to_suggestion_str = {}
            validation_cache = {}  # Cache validation results
            
            # Batch pre-validation if enabled
            if self.config.enable_suggestion_pre_validation:
                print(f"  • Pre-validating suggestions for {len(oov_words)} OOV words...")
                all_suggestions_to_validate = set()
                
                # Collect all unique suggestions
                for word in oov_words:
                    if len(word) > 2 and word in best_suggestions_dict:
                        suggestions = best_suggestions_dict.get(word, ["OOV"])
                        for sug in suggestions:
                            if isinstance(sug, tuple):
                                all_suggestions_to_validate.update([s for s in sug if s and s != "OOV"])
                            else:
                                all_suggestions_to_validate.add(sug)
                
                # Validate all suggestions in one batch
                if all_suggestions_to_validate:
                    validation_tasks = [self.verify_correction_with_dictionary(sug.strip('.,!?;:"\'()[]{}')) 
                                       for sug in all_suggestions_to_validate if sug and sug != "OOV"]
                    validation_results = await asyncio.gather(*validation_tasks)
                    
                    # Build validation cache
                    for sug, is_valid in zip(all_suggestions_to_validate, validation_results):
                        validation_cache[sug] = is_valid
            
            # Process suggestions with cached validation results
            for word in oov_words:
                if len(word) > 2 and word in best_suggestions_dict:
                    suggestions = best_suggestions_dict.get(word, ["OOV"])
                    cleaned_suggestions = []
                    for sug in suggestions:
                        if isinstance(sug, tuple):
                            for s in sug:
                                if s and s != "OOV":
                                    # Use cached validation result if available
                                    if self.config.enable_suggestion_pre_validation:
                                        if validation_cache.get(s, False):
                                            cleaned_suggestions.append(s)
                                    else:
                                        cleaned_suggestions.append(s)
                        else:
                            if sug and sug != "OOV":
                                # Use cached validation result if available
                                if self.config.enable_suggestion_pre_validation:
                                    if validation_cache.get(sug, False):
                                        cleaned_suggestions.append(sug)
                                else:
                                    cleaned_suggestions.append(sug)
                    word_to_suggestion_str[word] = cleaned_suggestions
            
            # Create tasks using inverted index
            for idx, item in enumerate(responses_with_ids):
                response = item['response']
                response_oov_words = []
                
                # Get all OOV words for this response index from inverted mapping
                for word, response_indices in word_to_responses.items():
                    if idx in response_indices and len(word) > 2 and word in word_to_suggestion_str:
                        response_oov_words.append(word)
                
                if response_oov_words:
                    # FIXED: Replace ALL occurrences of each OOV word, not just first
                    response_with_placeholders = response
                    for word in response_oov_words:
                        pattern = rf'\b{re.escape(word)}\b'
                        # Remove count=1 to replace ALL occurrences
                        response_with_placeholders = re.sub(pattern, '<oov_word>', response_with_placeholders)
                    
                    # Get suggestions for all OOV words - use pre-computed suggestions
                    all_suggestions = []
                    has_valid_suggestions = False
                    
                    for word in response_oov_words:
                        # Get pre-validated suggestions (already validated in batch)
                        cleaned_suggestions = word_to_suggestion_str.get(word, [])
                        
                        if cleaned_suggestions:
                            has_valid_suggestions = True
                            all_suggestions.append(", ".join(cleaned_suggestions))
                        else:
                            all_suggestions.append("OOV")  # No valid suggestions
                    
                    # Only create task if we have at least one valid suggestion
                    if has_valid_suggestions or not self.config.enable_suggestion_pre_validation:
                        tasks.append({
                            "respondent_id": item['respondent_id'],
                            "response": response,
                            "response_with_placeholders": response_with_placeholders,
                            "oov_words": ", ".join(response_oov_words),
                            "suggestions": " | ".join(all_suggestions)
                        })
                    else:
                        pre_validation_filtered += 1
            
            # Report performance improvement
            task_creation_time = time.time() - task_creation_start
            print(f"  • Task creation completed in {task_creation_time:.1f}s using inverted index (optimized)")
            
        else:
            # Fallback to original implementation
            print(f"  • Creating correction tasks using standard regex search...")
            for item in responses_with_ids:
                response = item['response']
                response_oov_words = []
                for word in oov_words:
                    if len(word) > 2:
                        pattern = rf'\b{re.escape(word)}\b'
                        if re.search(pattern, response):
                            response_oov_words.append(word)
                      
                if response_oov_words:
                        # FIXED: Replace ALL occurrences of each OOV word, not just first
                        response_with_placeholders = response
                        for word in response_oov_words:
                            pattern = rf'\b{re.escape(word)}\b'
                            # Remove count=1 to replace ALL occurrences
                            response_with_placeholders = re.sub(pattern, '<oov_word>', response_with_placeholders)
                        
                        # Get suggestions for all OOV words with pre-validation
                        all_suggestions = []
                        has_valid_suggestions = False
                        
                        for word in response_oov_words:
                            suggestions = best_suggestions_dict.get(word, ["OOV"])
                            # Clean up suggestion format
                            cleaned_suggestions = []
                            for sug in suggestions:
                                if isinstance(sug, tuple):
                                    cleaned_suggestions.extend([s for s in sug if s and s != "OOV"])
                                else:
                                    cleaned_suggestions.append(sug)
                            
                            # Pre-validate suggestions if enabled
                            if self.config.enable_suggestion_pre_validation and cleaned_suggestions:
                                validated_suggestions = await self.pre_validate_suggestions(cleaned_suggestions)
                                if validated_suggestions:
                                    has_valid_suggestions = True
                                    all_suggestions.append(", ".join(validated_suggestions))
                                else:
                                    all_suggestions.append("OOV")  # No valid suggestions
                            else:
                                # Skip validation or no suggestions
                                if cleaned_suggestions and any(s != "OOV" for s in cleaned_suggestions):
                                    has_valid_suggestions = True
                                all_suggestions.append(", ".join(cleaned_suggestions))
                        
                        # Only create task if we have at least one valid suggestion
                        if has_valid_suggestions or not self.config.enable_suggestion_pre_validation:
                            tasks.append({
                                "respondent_id": item['respondent_id'],
                                "response": response,
                                "response_with_placeholders": response_with_placeholders,
                                "oov_words": ", ".join(response_oov_words),
                                "suggestions": " | ".join(all_suggestions)
                            })
                        else:
                            pre_validation_filtered += 1
            
            # Report performance
            task_creation_time = time.time() - task_creation_start
            print(f"  • Task creation completed in {task_creation_time:.1f}s using regex search (fallback)")
         
        repeated_char_pattern = re.compile(rf'^(.)\1{{{self.config.repeated_char_threshold-1},}}$')
        single_word_pattern = re.compile(r'^[A-Za-z]+$')
        filtered_tasks = [
            task for task in tasks
            if not (
                repeated_char_pattern.match(task['response']) or
                repeated_char_pattern.match(task['oov_words']) or
                (single_word_pattern.fullmatch(task['response']) and 'OOV' in task['suggestions'])) ]
        
        # Track task creation and filtering stats
        self.stats['responses_with_tasks'] = len(tasks)
        self.stats['tasks_filtered_out'] = len(tasks) - len(filtered_tasks)
        self.stats['pre_validation_filtered'] = pre_validation_filtered
        
        # Count unique OOV words that made it into tasks
        oov_words_in_tasks = set()
        for task in filtered_tasks:
            task_oov_words = [word.strip() for word in task['oov_words'].split(',')]
            oov_words_in_tasks.update(task_oov_words)
        self.stats['oov_words_in_tasks'] = len(oov_words_in_tasks)
        
        batches = self.create_correction_batches(filtered_tasks, prompt_header, max_tokens, completion_reserve)
        
        # === WORKLOAD ANALYSIS & RATE LIMITING SETUP ===
        if len(batches) > 1:  # Only use rate limiting for multiple batches
            # Initialize workload analyzer
            encoding = get_tiktoken_encoding(self.model)
            analyzer = SpellCheckWorkloadAnalyzer(self.model, encoding, self.config)
            
            # Measure actual token usage from sample batches
            sample_batches_for_analysis = []
            for batch in batches[:3]:  # Sample first 3 batches
                sample_batches_for_analysis.append(batch.tasks)
            
            avg_tokens_per_batch = analyzer.measure_token_usage(sample_batches_for_analysis, SPELLCHECK_INSTRUCTIONS, var_lab)
            
            # Calculate optimal strategy
            strategy = analyzer.calculate_optimal_strategy(len(batches), avg_tokens_per_batch)
            
            # Initialize rate monitoring with configurable utilization
            rate_limits = get_openai_rate_limits(self.model)
            monitor = SpellCheckSlidingWindowMonitor(
                rate_limits.requests_per_minute, 
                rate_limits.tokens_per_minute,
                rate_limit_utilization=self.config.rate_limit_utilization
            )
            
            # Initialize throttler for rate limiting
            throttler = Throttler(rate_limit=strategy.launch_rate_per_second)
            
            # Display strategy summary
            print(f"[SPELL CHECK ANALYSIS]")
            print(f"- Model: {self.model} (Limits: {rate_limits.requests_per_minute:,} RPM, {rate_limits.tokens_per_minute:,} TPM)")
            print(f"- Correction batches to process: {len(batches):,}")
            print(f"- Avg tokens per batch: {avg_tokens_per_batch:.0f}")
            print(f"- Optimal strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
            print(f"- Estimated time: {strategy.target_time_seconds:.1f}s ({strategy.bottleneck_type} bottleneck)")
            print(f"Processing correction batches...")
        
        async def process_batch_with_rate_limiting(batch: SpellCorrectionBatch, var_lab: str, batch_index: int, 
                                                use_rate_limiting: bool = False, throttler=None, monitor=None, 
                                                avg_tokens_per_batch: float = 0) -> Dict[str, str]:
            """Native async client with rate limiting and validation"""
            
            # Rate limiting: wait for permission to proceed
            if use_rate_limiting and throttler:
                async with throttler:
                    if monitor and not await monitor.can_proceed(int(avg_tokens_per_batch)):
                        # Back off if approaching limits
                        await asyncio.sleep(1)
            
            tasks_string = ""
            for i, task in enumerate(batch.tasks):
                tasks_string += (
                    f"Task {i + 1}:\n"
                    f"Respondent ID: {task.respondent_id}\n"
                    f"Response: \"{task.response_with_oov_placeholders}\"\n"
                    f"Misspelled words: {task.oov_words}\n"
                    f"Suggested corrections: {task.suggestions}\n\n")
            
            prompt = SPELLCHECK_INSTRUCTIONS.format(
                language=DEFAULT_LANGUAGE,
                var_lab=var_lab,
                tasks=tasks_string)
            
            # Calculate actual tokens for this request
            if use_rate_limiting and monitor:
                encoding = get_tiktoken_encoding(self.model)
                prompt_tokens = len(encoding.encode(prompt))
                estimated_tokens = int(prompt_tokens * 1.15)  # Include completion estimate
            else:
                estimated_tokens = 0
                
            # Capture prompt only for the first batch
            if self.prompt_printer and batch_index == 0:
                self.prompt_printer.capture_prompt(
                    step_name="preprocessing",
                    utility_name="SpellChecker",
                    prompt_content=prompt,
                    prompt_type="correction",
                    metadata={
                        "model": self.model,
                        "var_lab": var_lab,
                        "language": DEFAULT_LANGUAGE,
                        "batch_size": len(batch.tasks),
                        "total_batches": len(batches),
                        "batch_number": batch_index + 1,
                        "client_type": "instructor_async_rate_limited" if use_rate_limiting else "instructor_async"
                    }
                )
            
            try:
                self.stats['llm_calls_made'] += 1
                response = await self.client.chat.completions.create(
                    model=self.model,
                    response_model=LLMCorrectionResponse,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=completion_reserve,
                    temperature=self.config.temperature,
                    seed=self.config.seed,
                    max_retries=self.config.retries
                )
                
                corrections = response.corrections
                self.stats['llm_calls_successful'] += 1
                
                # Record API usage for rate monitoring
                if use_rate_limiting and monitor:
                    await monitor.record_request(estimated_tokens)
                
            except Exception as e:
                # Verbose error reporting for OpenAI issues
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.error(f"OpenAI API call failed for batch {batch_index}: {e}")
                else:
                    logger.error(f"LLM call failed for batch {batch_index}: {e}")
                corrections = []
                self.stats['llm_calls_failed'] += 1
            
            # Validation loop for corrections
            validated_corrections = {}
            for task in batch.tasks:
                # Find corresponding correction
                correction_text = task.original_response  # Default fallback
                correction_found = False
                
                for corr in corrections:
                    if str(corr.respondent_id) == str(task.respondent_id):
                        candidate_correction = corr.corrected_response
                        correction_found = True
                        self.stats['corrections_attempted'] += 1
                        
                        # Check for "[NO RESPONSE]" cases
                        if candidate_correction == "[NO RESPONSE]":
                            self.stats['corrections_no_response'] += 1
                            break
                        
                        # Validate correction quality
                        words_to_validate = task.oov_words.split(', ')
                        validation_passed = True
                        
                        for word in words_to_validate:
                            if word in candidate_correction:
                                # Check if the replacement word is valid
                                replaced_words = [w for w in candidate_correction.split() if w not in task.original_response.split()]
                                for replaced_word in replaced_words:
                                    is_valid = await self.verify_correction_with_dictionary(replaced_word.strip('.,!?;:"\'()[]{}'))
                                    if not is_valid:
                                        validation_passed = False
                                        break
                        
                        if validation_passed:
                            correction_text = candidate_correction
                            self.stats['corrections_applied'] += 1
                        else:
                            self.stats['corrections_rejected_validation'] += 1
                        break
                
                # Track cases where LLM didn't return a correction for this task
                if not correction_found:
                    self.stats['corrections_attempted'] += 1
                
                validated_corrections[task.original_response] = correction_text
            
            return validated_corrections
        
        # Sort batches for consistent processing order
        sorted_batches = sorted(batches, key=lambda b: str(b.tasks[0].respondent_id) if b.tasks else "")

        # Execute batches with appropriate strategy
        corrected_sentences_dict = {}
        
        if len(batches) > 1:
            # Use rate-limited processing for multiple batches
            semaphore = asyncio.Semaphore(strategy.concurrent_limit)
            
            async def controlled_batch_process(batch, batch_idx):
                async with semaphore:
                    return await process_batch_with_rate_limiting(
                        batch, var_lab, batch_idx, 
                        use_rate_limiting=True, 
                        throttler=throttler, 
                        monitor=monitor, 
                        avg_tokens_per_batch=avg_tokens_per_batch
                    )
            
            # Process with progress tracking
            completed_batches = 0
            tasks = [controlled_batch_process(batch, i) for i, batch in enumerate(sorted_batches)]
            
            # Process batches and show progress
            for completed_task in asyncio.as_completed(tasks):
                batch_result = await completed_task
                corrected_sentences_dict.update(batch_result)
                completed_batches += 1
                
                # Progress reporting
                progress_percent = (completed_batches / len(batches)) * 100
                print(f"Processing correction batches... {completed_batches}/{len(batches)} ({progress_percent:.1f}%)")
                
                # Optional: Show utilization stats periodically
                if completed_batches % 10 == 0 and len(batches) > 20:
                    utilization = await monitor.get_current_utilization()
                    rate_per_second = utilization['total_requests'] / max(utilization['elapsed_time'], 0.1)
                    print(f"  Current rate: {rate_per_second:.1f} req/s ({utilization['rpm_utilization']*100:.0f}% RPM, {utilization['tpm_utilization']*100:.0f}% TPM)")
        else:
            # Single batch - no need for rate limiting
            batch_result = await process_batch_with_rate_limiting(sorted_batches[0], var_lab, 0)
            corrected_sentences_dict.update(batch_result)
        
        return corrected_sentences_dict

    async def spell_check_async(self, responses: List[SpellCheckModel], var_lab: str) -> List[SpellCheckModel]:
        """Spell checking with improved performance and accuracy"""
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(responses)
        
        # Always show main progress
        self.verbose_reporter.empty_line()
        print(f"Processing {len(responses)} responses for spell checking...")
        
        sentences_list = [response.original_response for response in responses]
        
        # Calculate total words for metrics
        total_words = sum(len(sentence.split()) for sentence in sentences_list)
        
        # Verbose metrics
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Total words to analyze: {total_words:,}")
        else:
            print(f"  • Total words to analyze: {total_words:,}")
            
        # Initialize word frequency cache if enabled
        word_frequency_cache = {} if self.config.enable_word_frequency_cache else None
        
        oov_words = []
        docs_with_oov = 0
        
        # AGGRESSIVE BATCHED OOV IDENTIFICATION (Quality Filter Style)
        print("  • Extracting and batching words for OOV analysis...")
        
        # Extract all words first for efficient batching
        all_words_to_check = []
        word_to_responses = defaultdict(list)  # Track which responses contain each word
        
        for response_idx, doc in enumerate(self.get_nlp().pipe(sentences_list, batch_size=self.config.spacy_batch_size)):
            for token in doc:
                if token.is_alpha and token.ent_type_ == "" and len(token.text) > 2:
                    word_normalized = token.text.lower()
                    word_original = token.text
                    
                    # Check cache first
                    if word_frequency_cache is not None and word_normalized in word_frequency_cache:
                        is_oov = word_frequency_cache[word_normalized] 
                        if is_oov:
                            oov_words.append(word_original)
                            self.stats['oov_words_found'] += 1
                            word_to_responses[word_original].append(response_idx)
                    else:
                        # Add to batch for Hunspell checking
                        all_words_to_check.append((word_normalized, word_original, response_idx))
        
        print(f"  • Cached words processed, {len(all_words_to_check):,} words need Hunspell verification")
        
        if all_words_to_check:
            batch_size = self.config.hunspell_batch_size  # Configurable batch size
            total_batches = (len(all_words_to_check) + batch_size - 1) // batch_size
            
            print(f"  • Processing {len(all_words_to_check):,} words in {total_batches} Hunspell batches...")
            
            # Use multiple concurrent Hunspell sessions for parallel processing
            max_concurrent_sessions = min(self.config.hunspell_concurrent_sessions, total_batches)  # Configurable concurrent sessions
            semaphore = asyncio.Semaphore(max_concurrent_sessions)
            
            async def process_hunspell_batch(batch_words, batch_index):
                """Process a batch of words with dedicated Hunspell session"""
                async with semaphore:
                    session = HunspellSession(self.hunspell_path, self.dict_path)
                    batch_oov_words = []
                    
                    try:
                        for word_normalized, word_original, response_idx in batch_words:
                            self.stats['words_checked'] += 1
                            output = session.check_word(word_original)
                            is_oov = output and output.startswith(('&', '#'))
                            
                            # Cache the result
                            if word_frequency_cache is not None:
                                word_frequency_cache[word_normalized] = is_oov
                            
                            if is_oov:
                                batch_oov_words.append((word_original, response_idx))
                                self.stats['oov_words_found'] += 1
                        
                        # Progress reporting
                        progress = (batch_index + 1) / total_batches * 100
                        print(f"    Hunspell batch {batch_index + 1}/{total_batches} ({progress:.1f}%) - found {len(batch_oov_words)} OOV words")
                        
                        return batch_oov_words
                        
                    finally:
                        session.close()
            
            # Create batches and process concurrently
            batches = []
            for i in range(0, len(all_words_to_check), batch_size):
                batch = all_words_to_check[i:i + batch_size]
                batches.append(batch)
            
            # Process all batches concurrently
            start_time = time.time()
            batch_tasks = [process_hunspell_batch(batch, idx) for idx, batch in enumerate(batches)]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Combine results and track response flagging
            response_flagged = set()
            for batch_result in batch_results:
                for word_original, response_idx in batch_result:
                    oov_words.append(word_original)
                    word_to_responses[word_original].append(response_idx)
                    response_flagged.add(response_idx)
            
            docs_with_oov = len(response_flagged)
            processing_time = time.time() - start_time
            words_per_second = len(all_words_to_check) / max(processing_time, 0.1)
            
            print(f"  • Completed OOV identification: {len(all_words_to_check):,} words in {processing_time:.1f}s ({words_per_second:.1f} words/sec)")
            
        # FIXED: Process only unique OOV words to avoid duplicates
        unique_oov_words = list(set(oov_words))
        self.stats['unique_oov_words'] = len(unique_oov_words)
        
        # Limit unique OOV words processing to prevent excessive correction attempts
        if len(unique_oov_words) > self.config.max_unique_oov_words:
            print(f"⚠️  Too many unique OOV words found ({len(unique_oov_words):,})")
            print(f"⚠️  Limiting to first {self.config.max_unique_oov_words:,} most frequent OOV words")
            
            # Count frequency of each OOV word and keep most common ones
            from collections import Counter
            oov_counter = Counter(oov_words)
            most_common_oov = [word for word, count in oov_counter.most_common(self.config.max_unique_oov_words)]
            unique_oov_words = most_common_oov
            self.stats['unique_oov_words'] = len(unique_oov_words)
        
        # Verbose OOV analysis details
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Responses requiring correction: {docs_with_oov}")
            if word_frequency_cache:
                cache_hits = sum(1 for cached in word_frequency_cache.values() if not cached)
                self.verbose_reporter.stat_line(f"Word frequency cache hits: {cache_hits}")
        
        # Verbose progress indicators for large datasets
        if self.verbose_reporter.enabled and len(responses) > 1000:
            self.verbose_reporter.progress_line(len(responses), len(responses), "analyzing for OOV words")
    
        # Step 2: Get suggestions for unique OOV words only
        if unique_oov_words:
            best_suggestions_dict = await self.find_best_suggestions_batch_async(unique_oov_words)
            # Pass the word_to_responses mapping for optimized task creation
            corrected_sentences_dict = await self.get_best_corrections_with_ai(responses, best_suggestions_dict, var_lab, word_to_responses)
            corrected_sentences_dict = {k: v for k, v in corrected_sentences_dict.items() if v != '[NO RESPONSE]'}
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("No OOV words found - skipping correction step")
            corrected_sentences_dict = {}
        
        # Step 3: Update sentences with tracked respondent IDs
        corrections_made = 0
        correction_examples = []
        updated_responses = []
        
        for response in responses:
            corrected_response = corrected_sentences_dict.get(response.original_response, response.original_response)
            updated_response = SpellCheckModel(
                respondent_id=response.respondent_id,
                original_response = response.original_response,
                corrected_response = corrected_response)
            updated_responses.append(updated_response)
           
            # Track corrections for verbose output
            if response.original_response != corrected_response:
                original_normalized = ' '.join([word.lower().strip('.,!?;:"\'()[]{}') for word in response.original_response.split()])
                corrected_normalized = ' '.join([word.lower().strip('.,!?;:"\'()[]{}') for word in corrected_response.split()])
                
                if original_normalized != corrected_normalized:
                    corrections_made += 1
                    
                    # Store example for verbose output
                    if len(correction_examples) < self.config.max_correction_examples:
                        correction_examples.append((response.original_response, corrected_response))

        stats.end_timing()
        stats.output_count = len(updated_responses)
        self.stats['processing_time'] = stats.get_duration()
        self.stats['suggestion_cache_hits'] = self.suggestion_cache_hits
        self.stats['suggestion_cache_size'] = len(self.suggestion_cache) if self.suggestion_cache is not None else 0 
        
        # Performance summary for large datasets
        if total_words > 10000:
            processing_time = stats.get_duration()
            words_per_second = int(total_words / max(processing_time, 0.1))
            print(f"• Performance: {words_per_second:,} words/sec, {processing_time:.1f}s total")
        
        # Group all stats together
        if self.verbose_reporter.enabled: 
            self.verbose_reporter.stat_line(f"Corrections Failed (no correction): {self.stats['corrections_no_response']}")
            self.verbose_reporter.stat_line(f"Corrections rejected (validation): {self.stats['corrections_rejected_validation']}")
            
            # Word frequency cache statistics
            if word_frequency_cache:
                cache_size = len(word_frequency_cache)
                cache_efficiency = (cache_size / max(self.stats['words_checked'], 1)) * 100
                self.verbose_reporter.stat_line(f"Word cache efficiency: {cache_efficiency:.1f}% ({cache_size} unique words cached)")
            
            # Suggestion cache statistics  
            if self.suggestion_cache is not None and self.stats['suggestion_cache_hits'] > 0:
                self.verbose_reporter.stat_line(f"Suggestion cache hits: {self.stats['suggestion_cache_hits']} ({self.stats['suggestion_cache_size']} cached)")
          
        print(f"• Corrections applied: {self.stats['corrections_applied']}")
        
        # Store examples for end-of-phase summary (don't show here)
        self.correction_examples = correction_examples if correction_examples else []

        processed_responses = [models.PreprocessedModel(respondent_id=item.respondent_id, response=item.corrected_response) for item in updated_responses]
        
        return processed_responses
                  
    def spell_check(self, preprocess_responses: List[Dict], var_lab: str):
        """Enhanced synchronous wrapper with better error handling"""
        async def main():
            spellcheck_responses = [SpellCheckModel(
                    respondent_id=item.respondent_id, 
                    original_response=item.response) 
                    for item in preprocess_responses]
            
            try:
                return await self.spell_check_async(spellcheck_responses, var_lab)
            except Exception as e:
                logger.error(f"SpellChecker processing failed: {e}")
                # Fallback: return original responses
                return [models.PreprocessedModel(respondent_id=item.respondent_id, response=item.response) 
                       for item in preprocess_responses]
        
        nest_asyncio.apply()
        return asyncio.run(main())