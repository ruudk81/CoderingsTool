import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import re
import asyncio
import subprocess
import logging
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque, Counter
import itertools
from dataclasses import dataclass
import numpy as np
import nest_asyncio
from pydantic import BaseModel
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from aiolimiter import AsyncLimiter

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cached_resources import get_openai_client, get_tiktoken_encoding, get_spacy_nlp_conditional
from utils.llm import create_client, llm_create_async, ProbeResponse, RateLimits, extract_rate_limits_from_response
from config import get_reasoning_params

# === CONFIG — generic/universal ========================================================================================================
from config import (
    OPENAI_API_KEY, DEFAULT_LANGUAGE,
    ModelConfig, ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
    API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME,
)

# === CONFIG — step-specific ========================================================================================================
from pipeline.step_1_preProcessor.config_preProcessor import (
    HUNSPELL_PATH, DUTCH_DICT_PATH, ENGLISH_DICT_PATH,
    SpellCheckConfig, DEFAULT_SPELLCHECK_CONFIG,
    MAX_HUNSPELL_PROCESSES, MAX_SAFE_BATCH_SIZE,
    SUGGESTION_BATCH_SIZE, MAX_CONCURRENT_SUGGESTION_BATCHES,
    OUTPUT_TOKEN_RATIO, SPACY_VECTOR_NORM_THRESHOLD,
)

# === PROMPTS ========================================================================================================
from prompts_steps.prompts_preProcessor import (
    SPELLCHECK_INSTRUCTIONS,
    CorrectionItem,
    LLMCorrectionResponse,
)

logger = logging.getLogger(__name__)
DICT_PATH = DUTCH_DICT_PATH if DEFAULT_LANGUAGE == "Dutch" else ENGLISH_DICT_PATH

EXTRA_VERBOSE = False
if EXTRA_VERBOSE:  
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.CRITICAL)  # effectively "silent"    


# === STRUCTURED DATA MODELS ========================================================================================================
import models

class SpellCheckModel(BaseModel):
    respondent_id: Any
    original_response: str
    corrected_response: Optional[str] = None

# Note: CorrectionItem and LLMCorrectionResponse moved to prompts_exp.py
# (co-located with prompts following instructor schema pattern)

# === RATE LIMITING HELPER CLASSES ========================================================================================================
class TokenBucket:
    """Simple token bucket for TPM limiting (from qualityFilter.py)"""
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed):
        """Acquire tokens, returning wait time if not available"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            # Regenerate tokens based on time elapsed
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now
            
            if self.available >= tokens_needed:
                self.available -= tokens_needed
                return True
            else:
                # Calculate wait time
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                return wait_seconds
    
    async def wait_and_acquire(self, tokens_needed):
        """Wait if necessary and acquire tokens incrementally."""
        logger.debug(f"[TOKEN BUCKET] Requesting {tokens_needed} tokens")
        remaining = float(tokens_needed)
        while remaining > 0:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
                self.last_update = now

                take = min(self.available, remaining)
                if take > 0:
                    self.available -= take
                    remaining -= take

            if remaining > 0:
                # accrue more tokens; 1 token's worth (or a small floor)
                await asyncio.sleep(max(0.01, 60.0 / self.tpm))

        logger.debug(f"[TOKEN BUCKET] Acquired {tokens_needed} tokens, {self.available:.0f} remaining")
    
    async def reconcile(self, delta_tokens):
        """Reconcile actual vs estimated tokens"""
        # If we overestimated (delta < 0), return tokens
        # If we underestimated (delta > 0), we already used them
        if delta_tokens < 0:
            async with self.lock:
                old_available = self.available
                self.available = min(self.tpm, self.available - delta_tokens)
                logger.debug(f"[TOKEN BUCKET] Reconciled {-delta_tokens} tokens back, {old_available:.0f} → {self.available:.0f}")
        else:
            logger.debug(f"[TOKEN BUCKET] No reconciliation needed for +{delta_tokens} tokens (underestimated)")


class LatencyTracker:
    """Simple EMA tracker for latencies"""
    def __init__(self, processing_config: Optional[ProcessingConfig] = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)
    
    def add(self, value):
        """Add a latency measurement"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
    
    def get_timeout(self, est_tokens):
        """Calculate timeout based on EMA and token count with configurable bounds"""
        config = self.processing_config
        if not self.values:
            return max(config.adaptive_timeout_min_seconds, 180.0)  # Cold-start: generous for reasoning models

        # Use P95 latency as base
        p95 = np.percentile(list(self.values), 95)
        # Simple linear scaling with token count
        # Assume ~100ms per 1000 tokens as baseline
        token_factor = est_tokens / 1000
        timeout = p95 + (token_factor * 0.1)
        # Apply margin and configurable bounds
        return max(config.adaptive_timeout_min_seconds, min(config.adaptive_timeout_max_seconds, timeout * config.adaptive_timeout_margin))
    
    def get_avg_latency(self):
        """Get average latency for concurrency calculations"""
        if not self.values:
            return 2.0  # Default 2s
        return self.ema if self.ema is not None else 2.0


# === HUNSPELL ========================================================================================================
class HunspellSession:
    def __init__(self, hunspell_path, dict_path):
        self.process = subprocess.Popen(
            [hunspell_path, "-a", "-i", "utf-8", "-d", dict_path],
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

    def check_words_batch(self, words: List[str]) -> List[str]:
        """Check multiple words in a single efficient batch operation"""
        if not words:
            return []
        
        # Send all words at once
        for word in words:
            self.process.stdin.write(word + '\n')
        self.process.stdin.flush()  # Single flush for all words
        
        # Read all results at once
        results = []
        for _ in words:
            result = self.process.stdout.readline().strip()
            # Handle multi-line results
            while True:
                line = self.process.stdout.readline()
                if not line or line.strip() == '':
                    break
                result += "\n" + line.strip()
            results.append(result)
        
        return results

    def close(self):
        try:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.stderr.close()
            self.process.terminate()
            try:
                self.process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
            except subprocess.TimeoutExpired:
                # Force kill if process doesn't terminate gracefully
                self.process.kill()
                self.process.wait()  # Wait for forced termination
        except Exception as e:
            logger.error(f"Error closing Hunspell session: {e}")
            # Try force kill as last resort
            try:
                self.process.kill()
            except Exception:
                pass  # Ignore errors during force kill

class HunspellPool:
    """Pool of persistent Hunspell processes to avoid subprocess creation overhead (only for OOV identification)"""
    
    def __init__(self, hunspell_path: str, dict_path: str, pool_size: int = None):
        self.hunspell_path = hunspell_path
        self.dict_path = dict_path
        # Auto-tune pool size based on CPU count with more conservative limits
        if pool_size is None:
            pool_size = min(os.cpu_count(), MAX_HUNSPELL_PROCESSES)
        self.pool_size = pool_size
        self.sessions = []
        self.session_locks = []
        self.closed = False
        
        # Initialize the pool with persistent Hunspell sessions
        print("[IDENTIFYING OOV WORDS]")
        print(f"Initializing HunspellPool with {pool_size} persistent processes...")
        start_time = time.time()
        for i in range(pool_size):
            session = HunspellSession(hunspell_path, dict_path)
            self.sessions.append(session)
            self.session_locks.append(asyncio.Lock())
        
        init_time = time.time() - start_time
        print(f"HunspellPool initialized: {pool_size} processes ready in {init_time:.1f}s")
    
    async def check_word(self, word: str) -> str:
        """Check a single word using an available session from the pool"""
        if self.closed:
            raise RuntimeError("HunspellPool has been closed")
        
        # Try each session until we find an available one
        for i in range(self.pool_size):
            if self.session_locks[i].locked():
                continue
                
            async with self.session_locks[i]:
                try:
                    # Run the check in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, self.sessions[i].check_word, word)
                    return result
                except Exception as e:
                    logger.error(f"Error checking word '{word}' with session {i}: {e}")
                    # Recreate the session if it failed
                    self.sessions[i].close()
                    self.sessions[i] = HunspellSession(self.hunspell_path, self.dict_path)
                    raise
        
        # If all sessions are busy, wait for the first available
        async with self.session_locks[0]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.sessions[0].check_word, word)
    
    async def check_words_batch(self, words: List[str], batch_size: int = 100) -> List[str]:
        """Check multiple words efficiently using ALL available sessions in parallel"""
        if self.closed:
            raise RuntimeError("HunspellPool has been closed")
        
        if not words:
            return []
        
        optimal_batch_size = min(MAX_SAFE_BATCH_SIZE, max(100, batch_size))
        
        # Build batches with start indices to preserve order
        batches = [(i, words[i:i+optimal_batch_size]) for i in range(0, len(words), optimal_batch_size)]
        outputs = [None] * len(words)  # Preallocate output array
        
        print(f"    Distributing {len(words):,} words across {len(batches)} parallel batches ({self.pool_size} processes)")
        
        async def process_batch_parallel(idx: int, batch: List[str], session_idx: int) -> Tuple[int, List[str]]:
            """Process a batch using a specific session with error recovery - returns (start_idx, results)"""
            if not batch:
                return idx, []
                
            async with self.session_locks[session_idx]:
                try:
                    loop = asyncio.get_running_loop()
                    #start_time = time.time()
                    
                    # Run batch processing in executor
                    result = await loop.run_in_executor(None, self.sessions[session_idx].check_words_batch, batch)
                    
                    #batch_time = time.time() - start_time
                    #batch_rate = len(batch) / max(batch_time, 0.001)
                    
                    return idx, result
                    
                except Exception as e:
                    logger.error(f"Error processing batch of {len(batch)} words with session {session_idx}: {e}")
                    # Recreate the session if it failed
                    try:
                        self.sessions[session_idx].close()
                    except Exception:
                        pass  # Ignore close errors during recovery
                    self.sessions[session_idx] = HunspellSession(self.hunspell_path, self.dict_path)
                    # Return empty results for failed batch
                    return idx, [""] * len(batch)
        
        # Process ALL batches concurrently using ALL available sessions
        start_time = time.time()
        tasks = []
        for i, (idx, batch) in enumerate(batches):
            session_idx = i % self.pool_size  # Round-robin across sessions
            tasks.append(process_batch_parallel(idx, batch, session_idx))
        
        # Execute all batches in parallel with progress reporting
        completed_batches = 0
        words_processed = 0  # Track actual words processed for accurate progress
        failed_batches = 0
        
        # Process batches with progress updates
        for completed_task in asyncio.as_completed(tasks):
            try:
                idx, batch_result = await completed_task
                # Place results at correct position to preserve original order
                outputs[idx:idx+len(batch_result)] = batch_result
                completed_batches += 1
                words_processed += len(batch_result)
                
                # Progress reporting every 10% or every 50 batches
                if completed_batches % max(1, len(batches) // 10) == 0 or completed_batches % 50 == 0:
                    progress_percent = (completed_batches / len(batches)) * 100
                    elapsed = time.time() - start_time
                    rate = words_processed / max(elapsed, 0.001)
                    remaining_words = len(words) - words_processed
                    eta = remaining_words / max(rate, 0.001)
                    print(f"    Parallel batch progress: {completed_batches}/{len(batches)} ({progress_percent:.1f}%) [{rate:.0f} words/sec, ETA: {eta:.1f}s]")
                
            except Exception as e:
                # Count failed batches
                logger.error(f"Batch processing failed: {e}")
                failed_batches += 1
                completed_batches += 1
        
        total_time = time.time() - start_time
        total_rate = len(words) / max(total_time, 0.001)
        
        print(f"    Parallel processing completed: {len(words):,} words in {total_time:.1f}s ({total_rate:.0f} words/sec)")
        if failed_batches > 0:
            print(f"    Batch results: {len(batches) - failed_batches} successful, {failed_batches} failed")
        
        return outputs
    
    def close(self):
        """Close all Hunspell sessions in the pool"""
        if not self.closed:
            self.closed = True
            for session in self.sessions:
                try:
                    session.close()
                except Exception as e:
                    logger.error(f"Error closing Hunspell session: {e}")
            #logger.info("HunspellPool closed")

# === MAIN UTIL  ========================================================================================================

class SpellChecker:
    def __init__(self, config: SpellCheckConfig = None, model_config: ModelConfig = None, processing_config: ProcessingConfig = None, openai_api_key: Optional[str] = None, verbose: bool = False, prompt_printer = None, verbose_reporter: Optional['VerboseReporter'] = None):
        self.config = config or DEFAULT_SPELLCHECK_CONFIG
        self.model_config = model_config or ModelConfig()  # kept for backward compat
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.model = self.config.model
        
        self.suggestion_cache = {} if self.config.enable_suggestion_caching else None
        self.suggestion_cache_hits = 0

        self.client = create_client(self.model, async_mode=True)
        
        self.hunspell_path = HUNSPELL_PATH
        self.dict_path = DICT_PATH
        self.prompt_printer = prompt_printer 
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.empty_line()
            print("Spell checker configuration:")
            self.verbose_reporter.stat_line(f"Model: {self.model}", indent=1)
            self.verbose_reporter.stat_line(f"Language: {DEFAULT_LANGUAGE}", indent=1)
            self.verbose_reporter.stat_line(f"Dictionary: {self.dict_path}", indent=1)
            self.verbose_reporter.stat_line(f"Hunspell path: {self.hunspell_path}", indent=1)
            self.verbose_reporter.stat_line(f"Batch size: {self.config.batch_size}", indent=1)
            self.verbose_reporter.stat_line(f"Timeout range: {self.config.minimum_timeout_seconds}s - {self.config.maximum_timeout_seconds}s", indent=1)
            
        if not self.check_hunspell_installation():
            if self.verbose_reporter.enabled:
                self.verbose_reporter.warning("Hunspell is not properly installed or configured - spell checking may fail")
                self.verbose_reporter.warning(f"Expected Hunspell at: {self.hunspell_path}")
                self.verbose_reporter.warning(f"Expected dictionary at: {self.dict_path}")
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("OK Hunspell installation verified")

        self.hunspell_pool = None
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)
        self.failed_task_ids = set()  # Track respondent_ids of truly failed API calls
        self.stats = {
            'words_checked': 0,
            'oov_words_found': 0,
            'unique_oov_words': 0,
            'oov_words_in_tasks': 0,
            'responses_with_tasks': 0,
            'tasks_filtered_out': 0,
            'llm_calls_attempted': 0,
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
 
# --- HELPERS  ------------------------------------------------------------------------------------------------------------------ 
    
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
    
    def _init_hunspell_pool(self):
        """Initialize HunspellPool for efficient processing"""
        if self.hunspell_pool is None:
            pool_size = getattr(self.config, 'hunspell_pool_size', None)  # Use None to enable auto-tuning
            self.hunspell_pool = HunspellPool(self.hunspell_path, self.dict_path, pool_size)
            
           
    def _close_hunspell_pool(self):
        """Close the HunspellPool to free resources"""
        if self.hunspell_pool is not None:
            self.hunspell_pool.close()
            self.hunspell_pool = None
    
    
    @staticmethod
    @lru_cache(maxsize=1000000)
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
    
    def _estimate_avg_tokens_for_tasks(self, tasks: List[Dict]) -> int:
        """Estimate average tokens for spell correction tasks (following qualityFilter.py pattern)"""
        sample_size = min(10, len(tasks))
        if sample_size == 0:
            return 200  # Default estimate
        
        encoding = get_tiktoken_encoding(self.model)
        total_tokens = 0
        
        for i in range(sample_size):
            task = tasks[i]
            task_text = f"""Task:
Respondent ID: {task['respondent_id']}
Response: "{task['response_with_placeholders']}"
Misspelled words: {task['oov_words']}
Suggested corrections: {task['suggestions']}
"""
            full_prompt = SPELLCHECK_INSTRUCTIONS.format(
                language=DEFAULT_LANGUAGE,
                var_lab="sample_var",
                tasks=task_text
            )
            total_tokens += len(encoding.encode(full_prompt))
        
        avg_input = total_tokens / sample_size
        # Assume 15% output ratio for spell correction
        return int(avg_input * 1.15)
    
    
# --- PROCESSING PHASE 1 :: IDENTIFY OOV WORDS  ------------------------------------------------------------------------------------------------------------------ 
    """phase 1 is integrated in main pipeline"""

# --- PROCESSING PHASE 2 :: HUNSPELL SUGGESTIONS  ------------------------------------------------------------------------------------------------------------------ 
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
    
            if not uncached_words:
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line(f"All {len(unique_oov_words)} words found in suggestion cache")
                return cached_suggestions
            
            # Process only uncached words
            unique_oov_words = uncached_words
            if self.verbose_reporter.enabled and cached_suggestions:
                self.verbose_reporter.stat_line(f"Found {len(cached_suggestions)} words in cache, processing {len(uncached_words)} new words")
        
        try:
            new_suggestions = await self._process_suggestions_batched_spacy(unique_oov_words)
        
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            print("⚠️  Suggestion generation error - using fallback")
            new_suggestions = {word: [(None, None)] for word in unique_oov_words}
        
        # Update cache if enabled
        if self.suggestion_cache is not None:
            self.suggestion_cache.update(new_suggestions)
            
            # Merge with cached suggestions
            if 'cached_suggestions' in locals():
                new_suggestions.update(cached_suggestions)
        
        return new_suggestions
    
    
    async def _process_suggestions_batched_spacy(self, unique_oov_words: List[str]) -> Dict[str, List[Any]]:
        
        print("[HUNSPELL SUGGESTIONS]")
        start_time = time.time()
        sorted_oov_words = sorted(unique_oov_words)
        
        print(f"- Processing {len(sorted_oov_words)} words with optimized batched subprocess approach...")
        
        best_suggestions = defaultdict(list)
        
        # Batch configuration for optimal performance
        batch_size = SUGGESTION_BATCH_SIZE
        max_concurrent_batches = MAX_CONCURRENT_SUGGESTION_BATCHES
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Create batches
        batches = []
        for i in range(0, len(sorted_oov_words), batch_size):
            batch_words = sorted_oov_words[i:i + batch_size]
            batches.append((batch_words, i // batch_size))
        
        total_batches = len(batches)
        print(f"- Created {total_batches} batches of ~{batch_size} words each, max {max_concurrent_batches} concurrent batches")
        
        async def process_batch(batch_words, batch_index):
            """Process a batch of words with temporary HunspellSession"""
            async with semaphore:
                batch_results = {}
                
                try:
                    # Process words concurrently within batch using reliable subprocess approach
                    async def process_single_word(word):
                        """Process a single word and return results"""
                        try:
                            # Get unsplit suggestions using existing reliable method
                            unsplit_suggestions = await self.run_hunspell_word_async(word)
                            
                            # Get split suggestions (async operation)
                            split_result = await self.find_best_split_for_spellcheck(word)
                            left_part, right_part = split_result
                            split_suggestion = f"{left_part} {right_part}" if (left_part and right_part) else None
                            
                            # Select best unsplit suggestion
                            unsplit_suggestion = (
                                min(unsplit_suggestions, key=lambda s: self.cached_levenshtein_distance(word, s))
                                if unsplit_suggestions else None)
                            
                            return word, (unsplit_suggestion, split_suggestion)
                            
                        except Exception as e:
                            logger.error(f"Error processing word '{word}' in batch {batch_index}: {e}")
                            return word, (None, None)
                    
                    # Process all words in batch concurrently
                    word_tasks = [process_single_word(word) for word in batch_words]
                    word_results = await asyncio.gather(*word_tasks)
                    
                    # Collect results
                    for word, result in word_results:
                        batch_results[word] = result
                    
                    # Progress reporting
                    progress = (batch_index + 1) / total_batches * 100
                    elapsed = time.time() - start_time
                    rate = (batch_index + 1) * batch_size / max(elapsed, 0.1)
                    remaining_batches = total_batches - (batch_index + 1)
                    eta = remaining_batches * batch_size / max(rate, 0.1)
                    print(f"  Batch {batch_index + 1}/{total_batches} ({progress:.1f}%) - processed {len(batch_words)} words [{rate:.1f} words/sec, ETA: {eta:.1f}s]")
                    
                    return batch_results
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_index}: {e}")
                    # Return empty results for failed batch
                    return {word: (None, None) for word in batch_words}
        
        # Process all batches concurrently
        print("- Starting concurrent batch processing...")
        batch_tasks = [process_batch(batch_words, batch_idx) for batch_words, batch_idx in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Combine results from all batches
        for batch_result in batch_results:
            for word, (unsplit_suggestion, split_suggestion) in batch_result.items():
                best_suggestions[word].append((unsplit_suggestion, split_suggestion))
        
        total_time = time.time() - start_time
        rate = len(unique_oov_words) / max(total_time, 0.1)
        print(f"- Completed batched suggestion generation: {len(unique_oov_words):,} words in {total_time:.1f}s ({rate:.1f} words/sec)")
        
        return best_suggestions
    

    async def find_best_split_for_spellcheck(self, oov_word: str) -> Tuple[str, str]:    
        """Working split processing from old version - simple and reliable"""
        excluded_tags = {"SYM", "PUNCT", "X", "SPACE", "NUM"}

        left_split_attempts = [(oov_word[:i], "left") for i in range(4, len(oov_word) + 1)]
        right_split_attempts = [(oov_word[i:], "right") for i in range(len(oov_word) - 3)]  

        all_splits = left_split_attempts + right_split_attempts
        processed_splits = list(self.get_nlp().pipe([split for split, _ in all_splits], batch_size=self.config.spacy_batch_size))

        valid_splits = [
            (split, tag) for (split, tag), doc in zip(all_splits, processed_splits)
            if len(split) > 2 and all(token.pos_ not in excluded_tags and token.vector_norm > SPACY_VECTOR_NORM_THRESHOLD for token in doc) ]

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

        if not batch_candidates:
            return "", ""

        hunspell_results = await asyncio.gather(
            *(self.run_hunspell_word_async(candidate) for candidate in batch_candidates))

        normalized_hunspell_results = {
            candidate: result if isinstance(result, list) else [result]
            for candidate, result in zip(batch_candidates, hunspell_results)}

        all_suggestions = [
            suggestion
            for suggestions in normalized_hunspell_results.values()
            for suggestion in suggestions]

        if not all_suggestions or not all(isinstance(s, str) for s in all_suggestions):
            return left_part, right_part

        processed_suggestions = list(self.get_nlp().pipe(all_suggestions, batch_size=self.config.spacy_batch_size))

        filtered_suggestions = {
            candidate: [suggestion for suggestion, doc in zip(normalized_hunspell_results[candidate], processed_suggestions) if doc.vector_norm > SPACY_VECTOR_NORM_THRESHOLD]
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
    
    async def run_hunspell_word_async(self, word: str) -> List[str]:
        """Simple subprocess approach for reliable suggestion generation"""
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
    

# --- PROCESSING PHASE 3 :: CORRECTIONS BY AI  ------------------------------------------------------------------------------------------------------------------ 
    
    async def get_best_corrections_with_ai(self, responses, best_suggestions_dict: Dict[str, List[Any]], var_lab: str, word_to_responses: Dict[str, List[int]] = None) -> Dict[str, str]:
        """Native async OpenAI client with validation - optimized with word-to-response mapping"""
        
        self.var_lab = var_lab
        self.failed_task_ids.clear()  # Reset failure tracking for this run

        oov_words = list(best_suggestions_dict.keys())
        
        corrected_sentences_dict = {}
        tasks = []
        
        responses_with_ids = [{'respondent_id': response.respondent_id, 'response': response.original_response} for response in responses]
    
        # Pre-validation tracking
        pre_validation_filtered = 0
        enable_pre_validation = False  # Flag for pre-validation (currently disabled)

        # Performance tracking for task creation
        task_creation_start = time.time()

        # Use inverted index if available, otherwise fall back to regex search
        if word_to_responses is not None:
            print("  • Creating correction tasks using optimized inverted index...")
            
            # Pre-compute suggestion strings for all OOV words to avoid redundant processing
            word_to_suggestion_str = {}
            validation_cache = {}  # Cache validation results
         
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
                                    if enable_pre_validation:
                                        if validation_cache.get(s, False):
                                            cleaned_suggestions.append(s)
                                    else:
                                        cleaned_suggestions.append(s)
                        else:
                            if sug and sug != "OOV":
                                # Use cached validation result if available
                                if enable_pre_validation:
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
                    if has_valid_suggestions or not enable_pre_validation:
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
            logger.info(f"  • Task creation completed in {task_creation_time:.1f}s using inverted index (optimized)")
            
        else:
            # Fallback to original implementation
            print("  • Creating correction tasks using standard regex search...")
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
                                  
                            if cleaned_suggestions and any(s != "OOV" for s in cleaned_suggestions): 
                                has_valid_suggestions = True
                            all_suggestions.append(", ".join(cleaned_suggestions))

                        # Only create task if we have at least one valid suggestion
                        if has_valid_suggestions or not enable_pre_validation:
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

        # Log filtered task counts
        filtered_by_repeated_char = [t for t in tasks if repeated_char_pattern.match(t['response']) or repeated_char_pattern.match(t['oov_words'])]
        filtered_by_single_word_oov = [t for t in tasks if single_word_pattern.fullmatch(t['response']) and 'OOV' in t['suggestions']]

        if filtered_by_repeated_char or filtered_by_single_word_oov:
            print(f"  • Tasks filtered: {len(filtered_by_repeated_char)} repeated chars, {len(filtered_by_single_word_oov)} single-word without suggestions")

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

        ############################################################################################################################################
        # Prompt processing starts here
        ########################################################################################################################
        
        if filtered_tasks:

            nr_tasks = len(filtered_tasks)

            @dataclass
            class ApiLimits:
                tokens_per_minute: int
                requests_per_minute: int

            def compute_optimal_concurrency(limits: ApiLimits, latency_seconds: float, avg_tokens: float, processing_config: ProcessingConfig, cap: Optional[int] = None, min_conc: Optional[int] = None, headroom: Optional[float] = None) -> int:
                cap = cap if cap is not None else processing_config.concurrency_cap_default
                min_conc = min_conc if min_conc is not None else processing_config.concurrency_min_default
                headroom = headroom if headroom is not None else processing_config.rate_limit_headroom

                latency_seconds = max(float(latency_seconds or 0.5), 0.05)
                avg_tokens = max(float(avg_tokens or 1.0), 1.0)

                rpm_throughput = limits.requests_per_minute * headroom / 60
                tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
                candidates = [rpm_throughput, tpm_throughput]
                allowed_rps = max(min(candidates), 0.0)
                target = allowed_rps * latency_seconds   # Little's Law

                return int(max(min(target, cap), min_conc))
            
            async def bootstrap_measure_async(call_fn, n_probes: int = 3):
                """Run n_probes serial calls and return (avg_latency_s, avg_tokens). call_fn() -> usage dict."""
                latencies, tokens = [], []
                for _ in range(n_probes):
                    t0 = time.perf_counter()
                    usage = await call_fn()  # Let tenacity handle timeouts and retries
                    t1 = time.perf_counter()
                    latencies.append(max(t1 - t0, 0.001))
                    pt = int(usage.get("prompt_tokens", 0))
                    ct = int(usage.get("completion_tokens", 0))
                    tokens.append(max(pt + ct, 1))
                return sum(latencies)/len(latencies), sum(tokens)/len(tokens)

            
            async def probe_call_no_structured(self, task_dict):

                task_text = f"""Task:
Respondent ID: {task_dict['respondent_id']}
Response: "{task_dict['response_with_placeholders']}"
Misspelled words: {task_dict['oov_words']}
Suggested corrections: {task_dict['suggestions']}
"""

                prompt = SPELLCHECK_INSTRUCTIONS.format(
                    language=DEFAULT_LANGUAGE,
                    var_lab=task_dict.get('var_lab', self.var_lab),
                    tasks=task_text
                )

                # Use minimal ProbeResponse model for Azure compatibility (instructor requires response_model)
                resp = await llm_create_async(
                    client=self.client,
                    model=self.model,
                    prompt=prompt,
                    response_model=ProbeResponse,
                    temperature=self.config.temperature,
                    track_usage=False,  # Manual tracking for probes
                    **get_reasoning_params(self.model),
                )

                # Extract usage from instructor's _raw_response
                u = getattr(resp, "_raw_response", None)
                if u:
                    u = getattr(u, "usage", None)
                if not u:
                    u = getattr(resp, "usage", None)
                # Handle both Responses API (input_tokens) and Chat API (prompt_tokens)
                input_tokens = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
                output_tokens = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
                return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}

            async def fetch_rate_limits_from_api() -> RateLimits:
                """Make a minimal API call to fetch rate limits from response headers."""
                from openai import AsyncOpenAI
                from config import API_PROVIDER, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

                if API_PROVIDER == "azure":
                    client = AsyncOpenAI(
                        api_key=AZURE_OPENAI_API_KEY,
                        base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/",
                        default_query={"api-version": "2024-10-21"},
                    )
                    model = AZURE_OPENAI_DEPLOYMENT_NAME
                else:
                    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                    model = self.model

                # Make minimal API call with raw response to get headers
                if API_PROVIDER == "azure":
                    response = await client.chat.completions.with_raw_response.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_completion_tokens=5,
                    )
                else:
                    response = await client.responses.with_raw_response.create(
                        model=model,
                        input="Hi",
                    )

                return extract_rate_limits_from_response(response)

            # Fetch rate limits dynamically from API response headers
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("Fetching rate limits from API...")

            limits = await fetch_rate_limits_from_api()

            # Fallback if headers not available
            if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line(f"Warning: Using fallback rate limits (TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
                limits = RateLimits(
                    tokens_per_minute=FALLBACK_TPM,
                    requests_per_minute=FALLBACK_RPM,
                    tokens_per_day=FALLBACK_TPM * 60 * 24
                )
            else:
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line(f"Fetched from API: TPM={limits.tokens_per_minute:,}, RPM={limits.requests_per_minute:,}")

            # Store rate limits on self for use in diagnostics/reporting
            self.rate_limits = limits

            sample_tasks = filtered_tasks[:min(3, len(filtered_tasks))]
            if len(sample_tasks) < 3: 
                # Duplicate tasks if we have fewer than 3
                sample_tasks = sample_tasks * 3
                sample_tasks = sample_tasks[:3]

            if self.verbose_reporter.enabled:
                    print("[CORRECTION WITH AI]")
                    self.verbose_reporter.stat_line("Running bootstrap measurement (3 probe calls)...")
            
            start_time = time.time()
            task_cycle = itertools.cycle(sample_tasks)                                                                                                              
            
            async def probe_with_different_tasks():
                return await probe_call_no_structured(self, next(task_cycle))                                                                                       
            
            avg_latency_s, avg_tokens = await bootstrap_measure_async(probe_with_different_tasks, n_probes=3)
            
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Probe time: {time.time() - start_time:.3f}s")
                self.verbose_reporter.stat_line(f"Bootstrap results: {avg_latency_s:.3f}s avg latency, {avg_tokens:.0f} avg tokens")
                  
            for i in range(3):  # Add 3 samples to get started
                self.latency_tracker.add(avg_latency_s)

            Little = compute_optimal_concurrency(ApiLimits(limits.tokens_per_minute, limits.requests_per_minute), avg_latency_s, avg_tokens, self.processing_config)
            # Use ProcessingConfig for bounds instead of hardcoded constants
            min_concurrency = self.processing_config.concurrency_min_default
            optimal = max(Little, min_concurrency)
            semaphore = asyncio.Semaphore(min(nr_tasks, optimal))

            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Optimal by Little's law: {Little}")

            print("[RATE LIMITING SETUP]")
            print(f"- Model: {self.model}")
            print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
            print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")

            self.avg_tokens = self._estimate_avg_tokens_for_tasks(filtered_tasks)
            avg_tokens = self.avg_tokens  # Use the instance variable

            logger.info(f"- Calculated avg_tokens: {self.avg_tokens} (from {min(10, len(responses))} sample prompts)")

            # Create unified rate limiting system
            # Calculate arrival rate from throughput
            arrival_rate = min(
                limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
                limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60
            )

            if arrival_rate < 1:
                self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)  # one permit every N seconds
            else:
                self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)

            self.tpm_bucket = TokenBucket(limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

            # Use the bootstrap-measured optimal concurrency
            self.semaphore = semaphore  # Use the dynamically calculated semaphore from bootstrap
            self.optimal_concurrency = optimal

            rpm_throughput = limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
            tpm_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
            bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

            logger.info(f"- Expected throughput: {min(rpm_throughput, tpm_throughput):.1f}/s ({bottleneck} limited)")
            logger.info(f"- Optimal concurrency: {Little} (Little's Law: throughput × latency)")
            logger.info(f"- Constrained optimum: {optimal} min=100; max =300")

            expected_throughput = min(
                limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
                limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
            )
            
            # Use ProcessingConfig for worker bounds
            max_workers = self.processing_config.max_workers if hasattr(self.processing_config, 'max_workers') else 200
            min_workers = self.processing_config.min_workers if hasattr(self.processing_config, 'min_workers') else 50
            num_workers = min(max_workers, max(min_workers, int(expected_throughput * avg_latency_s * 2.0)))
           
            print(f"- Concurrent subroutines (workers): {num_workers}")
            print(f"- Concurrent ceiling (semaphore): {min(nr_tasks,optimal)}")
            
        else:
            print("No correction tasks to process")
            return {}
        
        # Process tasks using queue-and-workers pattern (following qualityFilter.py)
        corrected_sentences_dict = await self._process_all_tasks_async(filtered_tasks, var_lab, num_workers)
        
        # Calculate stats after main pass
        successful_tasks = self.stats.get('llm_calls_successful', 0)
        failed_tasks = self.stats.get('llm_calls_failed', 0)

        print(f"Processing individual tasks... {len(filtered_tasks)}/{len(filtered_tasks)} (100.0%)")
        print(f"- Successful: {successful_tasks}")
        print(f"- Failed: {failed_tasks}")

        # --- RETRY PASS for truly failed tasks ---
        if self.failed_task_ids:
            failed_task_list = [
                task for task in filtered_tasks
                if task['respondent_id'] in self.failed_task_ids
            ]

            if failed_task_list:
                print(f"\n[RETRY PASS] Retrying {len(failed_task_list)} failed tasks with reduced concurrency...")

                # Reset failure tracking for retry pass
                retry_failed_ids = set(self.failed_task_ids)  # Save for reporting
                self.failed_task_ids.clear()

                # Use conservative concurrency for retry (10% of original or min 5)
                retry_workers = max(5, min(len(failed_task_list), num_workers // 10))

                retry_results = await self._process_all_tasks_async(
                    failed_task_list, var_lab, retry_workers
                )

                # Merge retry results (overwrite fallback originals with actual corrections)
                corrected_sentences_dict.update(retry_results)

                recovered = len(retry_failed_ids) - len(self.failed_task_ids)
                still_failed = len(self.failed_task_ids)
                print(f"[RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")

                if still_failed > 0:
                    print(f"[RETRY PASS] Permanently failed respondent_ids: {sorted(self.failed_task_ids)[:20]}{'...' if still_failed > 20 else ''}")

        return corrected_sentences_dict

    async def _process_all_tasks_async(self, filtered_tasks: List[Dict], var_lab: str, num_workers: int) -> Dict[str, str]:
        """Process all tasks using queue + workers pattern (following qualityFilter.py)"""
        if not filtered_tasks:
            return {}
      
        # Create queue and results list (following qualityFilter.py pattern)
        queue = asyncio.Queue()
        results = [None] * len(filtered_tasks)
        
 
        # Add tasks to queue with result indices
        for i, task in enumerate(filtered_tasks):
            task['result_index'] = i
            task['var_lab'] = var_lab  
            await queue.put(task)
        
        print(f"Processing individual tasks... 0/{len(filtered_tasks)} (0.0%)")
        
        # Start workers + actual API call 
        workers = []
        #logger.info(f"[DEBUG QUEUE] Starting {num_workers} workers for {len(filtered_tasks)} tasks")
        for i in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results))
            workers.append(w)
        #logger.info(f"[DEBUG QUEUE] All {num_workers} workers started")
        
        # Progress monitoring with diagnostics (following qualityFilter.py pattern)
        start_time = time.time()
        last_report = start_time
        
    
        while not queue.empty():
                await asyncio.sleep(1)
                now = time.time()
                
                # Regular progress report every 5s
                if now - last_report >= 5:
                    completed = self.stats.get('llm_calls_attempted', 0)
                    remaining = queue.qsize()
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    successful = self.stats.get('llm_calls_successful', 0)
                    failed = self.stats.get('llm_calls_failed', 0)
                    
                    print(f"Progress: {completed}/{len(filtered_tasks)} ({completed/len(filtered_tasks)*100:.1f}%), "
                          f"Rate: {rate:.1f}/s, Queue: {remaining}, Success: {successful}, Failed: {failed}")
                    # logger.info(f"[DEBUG QUEUE] Progress: {completed}/{len(filtered_tasks)}, Queue: {remaining}, "
                    #           f"Success: {successful}, Failed: {failed}")
                    last_report = now
            
        await queue.join()
        
        # Stop workers
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        
        # Final performance stats
        total_processing_time = time.time() - start_time
        actual_rate = len(filtered_tasks) / max(total_processing_time, 0.1)
        print(f"\nCompleted {len(filtered_tasks)} tasks in {total_processing_time:.1f}s")
        print(f"- Average: {total_processing_time/len(filtered_tasks):.2f}s/task")
        print(f"- Processing rate: {actual_rate:.1f} tasks/sec")
        
        # Combine results from workers
        corrected_sentences_dict = {}
        for result in results:
            if result and isinstance(result, dict):
                corrected_sentences_dict.update(result)
        
        return corrected_sentences_dict    
    
    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue (following qualityFilter.py pattern)"""
        worker_id = id(asyncio.current_task())
        #logger.info(f"[DEBUG WORKER] Worker {worker_id} started")
        
        while True:
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    #logger.info(f"[DEBUG WORKER] Worker {worker_id} received sentinel, stopping")
                    break
                
                try:
                    #logger.info(f"[DEBUG WORKER] Worker {worker_id} processing task for {task['respondent_id']}")
                    result = await self._process_task_with_admission_controls(task)
                    results[task['result_index']] = result
                    #logger.info(f"[DEBUG WORKER] Worker {worker_id} completed task for {task['respondent_id']}")
                except Exception as e:
                    # After all retries failed
                    logger.error(f"[DEBUG WORKER] Worker {worker_id} - Task for '{task['respondent_id']}' failed: {e}")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(task['respondent_id'])
                    # Create fallback result (maps respondent_id to original text)
                    results[task['result_index']] = {task['respondent_id']: task['response']}
                finally:
                    self.stats['llm_calls_attempted'] += 1
                    queue.task_done()
                    
            except Exception as e:
                logger.error(f"[DEBUG WORKER] Worker {worker_id} critical error: {e}")
                break
        
        #logger.info(f"[DEBUG WORKER] Worker {worker_id} stopped")

    async def _process_task_with_admission_controls(self, task_dict: Dict[str, Any]) -> Dict[str, str]:
        """Process individual spell correction task with unified admission controls (from qualityFilter.py pattern)"""

        respondent_id = task_dict['respondent_id']
        #logger.info(f"[DEBUG API] Starting admission controls for {respondent_id}")

        # Count tokens needed for this task
        tokens_needed = self._count_task_tokens(task_dict)
        #logger.info(f"[DEBUG API] Acquiring {tokens_needed} tokens for {respondent_id}")

        # Calculate intelligent timeout BEFORE rate limiting to enable progressive learning
        timeout_seconds = self.latency_tracker.get_timeout(tokens_needed)
        #logger.info(f"[TIME OUT SECONDS] Task {task_dict['respondent_id']}: {len(self.latency_tracker.values)} samples → {timeout_seconds:.1f}s timeout")

        # FIX CONVOY EFFECT: Acquire semaphore FIRST to bound waiters,
        # then acquire token bucket and rate limiter
        async with self.semaphore:
            #logger.info(f"[DEBUG API] Entered semaphore for {respondent_id}")
            await self.tpm_bucket.wait_and_acquire(tokens_needed)
            #logger.info(f"[DEBUG API] Tokens acquired for {respondent_id}")
            async with self.rate_limiter:
                #logger.info(f"[DEBUG API] Entered rate limiter for {respondent_id}")
                
                # Initialize response and set fallback corrected_text
                response = None
                corrected_text = task_dict['response']  # Default fallback to original text
                             
                task_text = f"""Task:
Respondent ID: {task_dict['respondent_id']}
Response: "{task_dict['response_with_placeholders']}"
Misspelled words: {task_dict['oov_words']}
Suggested corrections: {task_dict['suggestions']}
"""
                
                full_prompt = SPELLCHECK_INSTRUCTIONS.format(
                        language=DEFAULT_LANGUAGE,
                        var_lab=task_dict.get('var_lab', self.var_lab),
                        tasks=task_text
                    )

                try:
                    #logger.info(f"[DEBUG API] Starting API call for {respondent_id}")
                    api_start_time = time.time() 
                    latency_start_time = time.perf_counter()  # Instead of time.time()
                    
                    # Make API call with adaptive timeout
                    response = await asyncio.wait_for(
                        llm_create_async(
                            client=self.client,
                            model=self.model,
                            prompt=full_prompt,
                            response_model=LLMCorrectionResponse,
                            temperature=self.config.temperature,
                            **get_reasoning_params(self.model),
                        ),
                        timeout=timeout_seconds
                    )
                    
                    # Record latency for future concurrency calculations
                    api_latency = max(time.perf_counter() - latency_start_time, 0.001)  # Match bootstrap 
                    self.latency_tracker.add(api_latency)

                    self.stats['llm_calls_successful'] += 1
                    logger.info(f"Success #{self.stats['llm_calls_successful']}")

                except asyncio.TimeoutError:
                    api_latency = time.time() - api_start_time
                    logger.error(f"[API TIMEOUT] Task {respondent_id} timed out after {timeout_seconds:.1f}s")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(respondent_id)
                    # Don't add timeout to latency tracker as it's not representative
                except RateLimitError as e:
                    logger.error(f"[API] RATE LIMIT (429): {respondent_id} - {str(e)}")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(respondent_id)
                except APITimeoutError as e:
                    logger.error(f"[API] API TIMEOUT: {respondent_id} - {str(e)}")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(respondent_id)
                except APIConnectionError as e:
                    logger.error(f"[API] CONNECTION ERROR: {respondent_id} - {str(e)}")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(respondent_id)
                except InternalServerError as e:
                    logger.error(f"[API] INTERNAL SERVER ERROR: {respondent_id} - {str(e)}")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(respondent_id)
                except Exception as e:
                    logger.error(f"[API] UNKNOWN ERROR: {respondent_id} - {type(e).__name__}: {str(e)}")
                    self.stats['llm_calls_failed'] += 1
                    self.failed_task_ids.add(respondent_id)
                    
                # Track actual token usage for reconciliation (only if response exists)
                if response and hasattr(response, '_raw_response'):
                    usage = response._raw_response.usage
                    if usage:
                        actual_total_tokens = usage.total_tokens
                        # Reconcile token difference with TokenBucket
                        delta = actual_total_tokens - tokens_needed
                        await self.tpm_bucket.reconcile(delta)
                        #logger.info(f"[DEBUG API] Token reconciliation for {respondent_id}: {delta} delta")
                
                # Parse structured response with validation (only if response exists)
                if response and response.corrections and len(response.corrections) > 0:
                    correction = response.corrections[0]  # Single task = single correction

                    # AUDIT: Log if LLM returned different ID (drift detection)
                    if str(correction.respondent_id) != str(task_dict['respondent_id']):
                        logger.warning(
                            f"ID drift detected: LLM returned '{correction.respondent_id}' "
                            f"but input was '{task_dict['respondent_id']}'"
                        )

                    corrected_text = correction.corrected_response

        return {task_dict['respondent_id']: corrected_text}

    def _count_task_tokens(self, task_dict: Dict[str, Any]) -> int:
        """Count tokens needed for individual task (input + estimated output)"""
        task_text = f"""Task:
Respondent ID: {task_dict['respondent_id']}
Response: "{task_dict['response_with_placeholders']}"
Misspelled words: {task_dict['oov_words']}
Suggested corrections: {task_dict['suggestions']}
"""
        
        full_prompt = SPELLCHECK_INSTRUCTIONS.format(
            language=DEFAULT_LANGUAGE,
            var_lab=task_dict.get('var_lab', self.var_lab),
            tasks=task_text
        )
        
        encoding = get_tiktoken_encoding(self.model)
        input_tokens = len(encoding.encode(full_prompt))
        estimated_output_tokens = int(input_tokens * OUTPUT_TOKEN_RATIO)
        
        return input_tokens + estimated_output_tokens

#--- RUN PIPELINE, Incl. PROCESSING PHASE 1: IDENTIFYING OOV WORDS -------------------------

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
        
        print("  • Extracting and batching words for OOV analysis...")

        # Extract all words first for efficient batching
        all_words_to_check = []
        word_to_responses = defaultdict(list)  # Track which responses contain each word

        # DIAGNOSTIC: Track what's being filtered at word identification level
        diag_total_tokens = 0
        diag_skipped_not_alpha = 0
        diag_skipped_named_entity = 0
        diag_skipped_too_short = 0

        for response_idx, doc in enumerate(self.get_nlp().pipe(sentences_list, batch_size=self.config.spacy_batch_size)):
            for token in doc:
                diag_total_tokens += 1

                if not token.is_alpha:
                    diag_skipped_not_alpha += 1
                    continue

                # Named entity filter REMOVED - was catching typos like "merkiglo", "merr"
                # Just track count for reporting
                if token.ent_type_ != "":
                    diag_skipped_named_entity += 1

                if len(token.text) <= 2:
                    diag_skipped_too_short += 1
                    continue

                # Word passed filters (is_alpha and len > 2)
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

        # DIAGNOSTIC: Print word identification filter stats
        # Note: Named entities are now INCLUDED (not skipped) - only tracking for info
        diag_passed_filters = diag_total_tokens - diag_skipped_not_alpha - diag_skipped_too_short
        print(f"  • Word filters: {diag_total_tokens:,} tokens → {diag_passed_filters:,} passed ({diag_passed_filters/max(diag_total_tokens,1)*100:.1f}%)")
        print(f"    (skipped: {diag_skipped_not_alpha:,} non-alpha, {diag_skipped_too_short:,} too short; {diag_skipped_named_entity:,} named entities now included)")
        print(f"  • Cached words processed, {len(all_words_to_check):,} words need Hunspell verification")
        
        if all_words_to_check:
            batch_size = self.config.hunspell_batch_size * 10  # 10,000 words per batch instead of 1,000
            
            print(f"  • Processing {len(all_words_to_check):,} words using HunspellPool with large batches...")
            
            # Initialize HunspellPool for efficient processing
            if self.hunspell_pool is None:
                self._init_hunspell_pool()
            
            start_time = time.time()
            
            # Extract just the original words for batch processing
            words_only = [item[1] for item in all_words_to_check]  # word_original
            
            # Process all words in efficient batches using HunspellPool
            batch_outputs = await self.hunspell_pool.check_words_batch(words_only, batch_size)

            # Count Hunspell results
            diag_oov_count = sum(1 for output in batch_outputs if output and output.startswith(('&', '#')))
            diag_correct_count = len(batch_outputs) - diag_oov_count

            print(f"\n  • Hunspell: {diag_correct_count:,} correct, {diag_oov_count:,} OOV (dictionary: {self.dict_path})")

            # Process results and update cache
            response_flagged = set()
            for i, (word_normalized, word_original, response_idx) in enumerate(all_words_to_check):
                self.stats['words_checked'] += 1
                output = batch_outputs[i]
                is_oov = output and output.startswith(('&', '#'))
                
                # Cache the result
                if word_frequency_cache is not None:
                    word_frequency_cache[word_normalized] = is_oov
                
                if is_oov:
                    oov_words.append(word_original)
                    word_to_responses[word_original].append(response_idx)
                    response_flagged.add(response_idx)
                    self.stats['oov_words_found'] += 1
                
                # Progress reporting for very large datasets
                if i > 0 and i % 20000 == 0:
                    progress = (i / len(all_words_to_check)) * 100
                    elapsed = time.time() - start_time
                    rate = i / max(elapsed, 0.1)
                    eta = (len(all_words_to_check) - i) / max(rate, 0.1)
                    print(f"    OOV analysis progress: {i:,}/{len(all_words_to_check):,} ({progress:.1f}%) [{rate:.0f} words/sec, ETA: {eta:.1f}s]")
            
            docs_with_oov = len(response_flagged)
            processing_time = time.time() - start_time
            words_per_second = len(all_words_to_check) / max(processing_time, 0.1)
            
            print(f"  • Completed OOV identification: {len(all_words_to_check):,} words in {processing_time:.1f}s ({words_per_second:.1f} words/sec)")
            print("    Performance improvement: HunspellPool eliminated subprocess creation overhead")
            
        # FIXED: Process only unique OOV words to avoid duplicates
        unique_oov_words = list(set(oov_words))
        self.stats['unique_oov_words'] = len(unique_oov_words)
        
        # Limit unique OOV words processing to prevent excessive correction attempts
        if len(unique_oov_words) > self.config.max_unique_oov_words:
            print(f"⚠️  Too many unique OOV words found ({len(unique_oov_words):,})")
            print(f"⚠️  Limiting to first {self.config.max_unique_oov_words:,} most frequent OOV words")

            # Count frequency of each OOV word and keep most common ones
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
    
        # Step 2-4: 
        if unique_oov_words:
            best_suggestions_dict = await self.find_best_suggestions_batch_async(unique_oov_words)
            # Pass the word_to_responses mapping for optimized task creation
            corrected_sentences_dict = await self.get_best_corrections_with_ai(responses, best_suggestions_dict, var_lab, word_to_responses)
            corrected_sentences_dict = {k: v for k, v in corrected_sentences_dict.items() if v != '[NO RESPONSE]'}
            # Note: corrected_sentences_dict is now keyed by respondent_id, not response text
         
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("No OOV words found - skipping correction step")
            corrected_sentences_dict = {}
        
        # Step 3: Update sentences with tracked respondent IDs
        corrections_made = 0
        correction_examples = []
        updated_responses = []
        
        #print(f"[COUNTING DEBUG] Starting correction counting for {len(responses)} responses")
        
        for i, response in enumerate(responses):
            corrected_response = corrected_sentences_dict.get(response.respondent_id, response.original_response)

            updated_response = SpellCheckModel(
                respondent_id=response.respondent_id,
                original_response = response.original_response,
                corrected_response = corrected_response)
            updated_responses.append(updated_response)
           
            # Track corrections for verbose output
            if response.original_response != corrected_response:
                # More robust normalization - keep hyphens and important punctuation for better comparison
                def normalize_for_comparison(text):
                    # Remove only peripheral punctuation, keep internal hyphens
                    words = text.lower().split()
                    normalized_words = [word.strip('.,!?;:"\'()[]{}') for word in words]
                    return ' '.join(normalized_words)
                
                original_normalized = normalize_for_comparison(response.original_response)
                corrected_normalized = normalize_for_comparison(corrected_response)
                
                if original_normalized != corrected_normalized:
                    corrections_made += 1
                    #logger.info(f"  -> COUNTED as correction #{corrections_made}")
                    
                    # Store example for verbose output
                    if len(correction_examples) < self.config.max_correction_examples:
                        correction_examples.append((response.original_response, corrected_response))
            
            else:
                # DEBUG: Log when no change detected
                logger.info(f"[COMPARISON DEBUG] No change detected for: '{response.original_response[:50]}...'")
                

        # IMPORTANT: Store response-level count BEFORE dictionary-based counting
        # corrections_made = how many actual responses were modified (the meaningful metric)
        self.stats['responses_corrected'] = corrections_made

        # Dictionary-based counting (for unique corrections)
        self.stats['corrections_attempted'] = 0
        self.stats['corrections_applied'] = 0
        self.stats['corrections_rejected_validation'] = 0
        self.stats['corrections_no_response'] = 0

        # Build a respondent_id -> original_response lookup for comparison
        respondent_originals = {r.respondent_id: r.original_response for r in responses}

        for resp_id, candidate_correction in corrected_sentences_dict.items():
            self.stats['corrections_attempted'] += 1

            original_response = respondent_originals.get(resp_id, "")
            # Check for "[NO RESPONSE]" cases or no change
            if candidate_correction == "[NO RESPONSE]" or candidate_correction == original_response:
                self.stats['corrections_no_response'] += 1
                continue

            self.stats['corrections_applied'] += 1

        # Summary showing BOTH metrics for clarity
        print("\nSUMMARY:")
        print(f"- Responses corrected: {self.stats['responses_corrected']} (actual responses modified)")
        print(f"- Unique corrections: {self.stats['corrections_applied']} (distinct strings in dictionary)")
        print(f"- Corrections attempted: {self.stats['corrections_attempted']}")
        print(f"- No correction/no change: {self.stats['corrections_no_response']}")
     
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
            
        if self.verbose_reporter.enabled: 
            # Word frequency cache statistics
            if 'word_frequency_cache' in locals() and word_frequency_cache:
                cache_size = len(word_frequency_cache)
                cache_efficiency = (cache_size / max(self.stats['words_checked'], 1)) * 100
                self.verbose_reporter.stat_line(f"Word cache efficiency: {cache_efficiency:.1f}% ({cache_size} unique words cached)")
            
            # Suggestion cache statistics  
            if self.suggestion_cache is not None and self.stats['suggestion_cache_hits'] > 0:
                self.verbose_reporter.stat_line(f"Suggestion cache hits: {self.stats['suggestion_cache_hits']} ({self.stats['suggestion_cache_size']} cached)")
          
        print(f"• Responses corrected: {self.stats['responses_corrected']} ({self.stats['corrections_applied']} unique)")
        
        # Store examples for end-of-phase summary (don't show here)
        self.correction_examples = correction_examples if correction_examples else []

        processed_responses = [models.PreprocessedModel(respondent_id=item.respondent_id, response=item.corrected_response) for item in updated_responses]
        
        # Clean up resources
        self._close_hunspell_pool()
        
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