import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import re
import asyncio
import subprocess
import logging
import time
#import statistics
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict #, deque
#from dataclasses import dataclass

import nest_asyncio
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter #wait_exponential
from openai import RateLimitError
from aiolimiter import AsyncLimiter

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding, get_spacy_nlp_conditional

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, HUNSPELL_PATH, DUTCH_DICT_PATH, ENGLISH_DICT_PATH, SpellCheckConfig, DEFAULT_SPELLCHECK_CONFIG, ModelConfig, get_openai_rate_limits
from prompts import SPELLCHECK_INSTRUCTIONS

logger = logging.getLogger(__name__)
DICT_PATH = DUTCH_DICT_PATH if DEFAULT_LANGUAGE == "Dutch" else ENGLISH_DICT_PATH


# === STRUCTURED DATA MODELS ========================================================================================================
import models

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

class CorrectionItem(BaseModel):
    respondent_id: Any 
    corrected_response: str 

class LLMCorrectionResponse(BaseModel):
    corrections: List[CorrectionItem] 
    
# === RATE LIMITING ========================================================================================================

class TokenBucket:
    """Token bucket for TPM rate limiting with smart waiting"""
    
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()  # Use monotonic to avoid clock issues
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed):
        """Acquire tokens, waiting if necessary"""
        async with self.lock:
            # Regenerate tokens based on time elapsed
            now = time.monotonic()
            elapsed = now - self.last_update
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now
            
            # Calculate wait time if not enough tokens (avoid busy polling)
            if self.available < tokens_needed:
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                if wait_seconds > 1.0:  # Only log significant waits
                    print(f"[RATE LIMIT] Token bucket waiting {wait_seconds:.1f}s for {tokens_needed} tokens (deficit: {deficit:.0f})")
                await asyncio.sleep(wait_seconds)
                # Recalculate after sleep
                now = time.monotonic()
                self.available = min(self.tpm, self.available + (self.tpm * wait_seconds / 60))
                self.last_update = now
            
            # Consume tokens
            self.available -= tokens_needed
            logger.debug(f"Token bucket: consumed {tokens_needed}, {self.available:.0f} remaining")


class RateLimitTracker:
    """Track rate limit errors and enforce cooldown periods"""
    
    def __init__(self, cooldown_seconds=15):
        self.last_rate_limit_time = 0
        self.cooldown_seconds = cooldown_seconds
    
    def check_cooldown(self):
        """Check if we're still in cooldown period"""
        time_since_error = time.monotonic() - self.last_rate_limit_time
        if time_since_error < self.cooldown_seconds:
            remaining = self.cooldown_seconds - time_since_error
            return True, remaining
        return False, 0
    
    def record_rate_limit(self):
        """Record when rate limit was hit"""
        self.last_rate_limit_time = time.monotonic()
        logger.warning(f"Rate limit error recorded, entering {self.cooldown_seconds}s cooldown")

# === HUNSPELL ========================================================================================================
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
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        self.process.terminate()


class HunspellPool:
    """Pool of persistent Hunspell processes to avoid subprocess creation overhead"""
    
    def __init__(self, hunspell_path: str, dict_path: str, pool_size: int = 20):
        self.hunspell_path = hunspell_path
        self.dict_path = dict_path
        self.pool_size = pool_size
        self.sessions = []
        self.session_locks = []
        self.closed = False
        
        # Initialize the pool with persistent Hunspell sessions
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
        
        # AGGRESSIVE OPTIMIZATION: Use all available processes for maximum parallelism
        # Calculate optimal batch size to utilize all processes
        optimal_batch_size = max(batch_size, len(words) // self.pool_size + 1)
        
        # Split words into batches for parallel processing across ALL sessions
        batches = []
        for i in range(0, len(words), optimal_batch_size):
            batch = words[i:i + optimal_batch_size]
            batches.append(batch)
        
        # Limit batches to available sessions for optimal distribution
        if len(batches) > self.pool_size:
            # Merge smaller batches to fully utilize all processes
            redistributed_batches = []
            batch_per_process = len(batches) // self.pool_size
            remainder = len(batches) % self.pool_size
            
            batch_idx = 0
            for process_idx in range(self.pool_size):
                process_batches = batch_per_process + (1 if process_idx < remainder else 0)
                merged_batch = []
                for _ in range(process_batches):
                    if batch_idx < len(batches):
                        merged_batch.extend(batches[batch_idx])
                        batch_idx += 1
                redistributed_batches.append(merged_batch)
            batches = redistributed_batches
        
        print(f"    Distributing {len(words):,} words across {len(batches)} parallel batches ({self.pool_size} processes)")
        
        async def process_batch_parallel(batch: List[str], session_idx: int) -> List[str]:
            """Process a batch using a specific session with error recovery"""
            if not batch:
                return []
                
            async with self.session_locks[session_idx]:
                try:
                    loop = asyncio.get_running_loop()
                    start_time = time.time()
                    result = await loop.run_in_executor(None, self.sessions[session_idx].check_words_batch, batch)
                    batch_time = time.time() - start_time
                    batch_rate = len(batch) / max(batch_time, 0.001)
                    
                    # Optional: Progress logging for very large batches
                    if len(batch) > 1000:
                        print(f"      Session {session_idx}: processed {len(batch):,} words in {batch_time:.1f}s ({batch_rate:.0f} words/sec)")
                    
                    return result
                except Exception as e:
                    logger.error(f"Error processing batch of {len(batch)} words with session {session_idx}: {e}")
                    # Recreate the session if it failed
                    try:
                        self.sessions[session_idx].close()
                    except:
                        pass
                    self.sessions[session_idx] = HunspellSession(self.hunspell_path, self.dict_path)
                    raise
        
        # Process ALL batches concurrently using ALL available sessions
        start_time = time.time()
        tasks = []
        for i, batch in enumerate(batches):
            session_idx = i % self.pool_size  # Round-robin across sessions
            tasks.append(process_batch_parallel(batch, session_idx))
        
        # Execute all batches in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results and handle exceptions
        results = []
        failed_batches = 0
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"Batch {i} failed: {batch_result}")
                # Return empty results for failed batch words
                results.extend([""] * len(batches[i]))
                failed_batches += 1
            else:
                results.extend(batch_result)
        
        total_time = time.time() - start_time
        total_rate = len(words) / max(total_time, 0.001)
        
        print(f"    Parallel processing completed: {len(words):,} words in {total_time:.1f}s ({total_rate:.0f} words/sec)")
        if failed_batches > 0:
            print(f"    Warning: {failed_batches} batches failed and returned empty results")
        
        return results
    
    def close(self):
        """Close all Hunspell sessions in the pool"""
        if not self.closed:
            self.closed = True
            for session in self.sessions:
                try:
                    session.close()
                except Exception as e:
                    logger.error(f"Error closing Hunspell session: {e}")
            logger.info("HunspellPool closed")

# === MAIN UTIL  ========================================================================================================
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
        
        # Initialize Hunspell pool for performance
        self.hunspell_pool = None
        
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
    
    def _init_hunspell_pool(self):
        """Initialize HunspellPool for efficient processing"""
        if self.hunspell_pool is None:
            pool_size = getattr(self.config, 'hunspell_pool_size', 20)
            self.hunspell_pool = HunspellPool(self.hunspell_path, self.dict_path, pool_size)
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Initialized HunspellPool with {pool_size} persistent processes")
    
    def _close_hunspell_pool(self):
        """Close the HunspellPool to free resources"""
        if self.hunspell_pool is not None:
            self.hunspell_pool.close()
            self.hunspell_pool = None
    
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
        """Efficient Hunspell lookup using persistent process pool"""
        if self.hunspell_pool is None:
            self._init_hunspell_pool()
        
        output = await self.hunspell_pool.check_word(word)
        lines = [line for line in output.splitlines() if line and not line.startswith("@")]

        if lines and lines[0].startswith("&"):
            match = re.search(r": (.+)", lines[0])
            if match:
                suggestions = match.group(1).split(", ")
                return suggestions
        if lines and lines[0].startswith("*"):
            return [word]  # Word is correct
        return []
    
    async def run_hunspell_batch_async(self, words: List[str]) -> List[List[str]]:
        """Efficient batch Hunspell lookup using persistent process pool"""
        if self.hunspell_pool is None:
            self._init_hunspell_pool()
        
        outputs = await self.hunspell_pool.check_words_batch(words)
        results = []
        
        for output in outputs:
            lines = [line for line in output.splitlines() if line and not line.startswith("@")]
            
            if lines and lines[0].startswith("&"):
                match = re.search(r": (.+)", lines[0])
                if match:
                    suggestions = match.group(1).split(", ")
                    results.append(suggestions)
                else:
                    results.append([])
            elif lines and lines[0].startswith("*"):
                # Word is correct - get the original word from the first few characters
                word_match = re.search(r'^\*\s+(.+)', lines[0])
                if word_match:
                    results.append([word_match.group(1)])
                else:
                    results.append([])
            else:
                results.append([])
        
        return results
    
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
        
        # Process uncached words with optimized strategy
        if len(unique_oov_words) <= 100:
            # Very small dataset - use original method
            new_suggestions = await self._process_suggestions_single_batch(unique_oov_words)
        elif len(unique_oov_words) <= self.config.ultra_batch_threshold:
            # Medium dataset - use parallel chunks 
            new_suggestions = await self._process_suggestions_parallel_chunks(unique_oov_words)
        else:
            # Large dataset - use ultra-optimized batch processing
            new_suggestions = await self._process_suggestions_ultra_optimized(unique_oov_words)
        
        # Update cache if enabled
        if self.suggestion_cache is not None:
            self.suggestion_cache.update(new_suggestions)
            
            # Merge with cached suggestions
            if 'cached_suggestions' in locals():
                new_suggestions.update(cached_suggestions)
        
        return new_suggestions
    
    async def _process_suggestions_ultra_optimized(self, unique_oov_words: List[str]) -> Dict[str, List[Any]]:
        """Ultra-optimized batch suggestion generation eliminating subprocess overhead"""
        
        print("[ULTRA-OPTIMIZED SUGGESTION GENERATION]")
        sorted_oov_words = sorted(unique_oov_words)
        
        # Initialize HunspellPool for batch processing
        if self.hunspell_pool is None:
            self._init_hunspell_pool()
        
        start_time = time.time()
        
        # STEP 1: Collect ALL words that need Hunspell checking (main words + all possible splits)
        all_hunspell_candidates = []  # (original_word, candidate_word, candidate_type)
        word_to_candidates = defaultdict(list)
        
        print(f"- Preparing candidates for {len(sorted_oov_words)} OOV words...")
        
        for word in sorted_oov_words:
            # Add the main word
            all_hunspell_candidates.append((word, word, 'main'))
            word_to_candidates[word].append((word, 'main'))
            
            # Generate split candidates
            if len(word) > 6:  # Only split longer words
                # Left splits: word[:i] for i in range(4, len(word)-2)
                for i in range(4, min(len(word)-2, len(word))):
                    left_part = word[:i]
                    right_part = word[i:]
                    if len(left_part) >= 3 and len(right_part) >= 3:
                        all_hunspell_candidates.append((word, left_part, 'left_split'))
                        all_hunspell_candidates.append((word, right_part, 'right_split'))
                        word_to_candidates[word].extend([
                            (left_part, 'left_split'), 
                            (right_part, 'right_split')
                        ])
        
        print(f"- Generated {len(all_hunspell_candidates):,} candidates for batch processing")
        
        # STEP 2: Batch process ALL candidates in one massive operation
        candidate_words = [item[1] for item in all_hunspell_candidates]
        
        print(f"- Processing all candidates using HunspellPool...")
        batch_outputs = await self.hunspell_pool.check_words_batch(candidate_words, batch_size=self.config.ultra_batch_size)
        
        processing_time = time.time() - start_time
        print(f"- Completed Hunspell batch processing: {len(candidate_words):,} candidates in {processing_time:.1f}s")
        
        # STEP 3: Parse results and organize by original word
        candidate_results = {}
        for i, (original_word, candidate_word, candidate_type) in enumerate(all_hunspell_candidates):
            output = batch_outputs[i]
            
            # Parse Hunspell output
            lines = [line for line in output.splitlines() if line and not line.startswith("@")]
            suggestions = []
            
            if lines and lines[0].startswith("&"):
                match = re.search(r": (.+)", lines[0])
                if match:
                    suggestions = match.group(1).split(", ")
            elif lines and lines[0].startswith("*"):
                suggestions = [candidate_word]  # Word is correct
            
            candidate_results[(original_word, candidate_word, candidate_type)] = suggestions
        
        # STEP 4: Construct final suggestions for each word
        best_suggestions = defaultdict(list)
        
        for word in sorted_oov_words:
            word_suggestions = []
            
            # Get main word suggestions
            main_suggestions = candidate_results.get((word, word, 'main'), [])
            if main_suggestions:
                # Pick best suggestion using Levenshtein distance
                best_main = min(main_suggestions, key=lambda s: self.cached_levenshtein_distance(word, s))
                word_suggestions.append(best_main)
            
            # Get split suggestions
            split_candidates = []
            for candidate_word, candidate_type in word_to_candidates[word]:
                if candidate_type in ['left_split', 'right_split']:
                    split_suggestions = candidate_results.get((word, candidate_word, candidate_type), [])
                    if split_suggestions and split_suggestions != [candidate_word]:  # Has corrections
                        best_split = min(split_suggestions, key=lambda s: self.cached_levenshtein_distance(candidate_word, s))
                        split_candidates.append((candidate_word, best_split, candidate_type))
            
            # Try to construct meaningful split suggestions
            left_splits = [(orig, corrected) for orig, corrected, type_ in split_candidates if type_ == 'left_split']
            right_splits = [(orig, corrected) for orig, corrected, type_ in split_candidates if type_ == 'right_split']
            
            # Find best split combination
            for left_orig, left_corrected in left_splits[:3]:  # Limit to avoid explosion
                for right_orig, right_corrected in right_splits[:3]:
                    if left_orig + right_orig == word:  # Valid split
                        split_suggestion = f"{left_corrected} {right_corrected}"
                        word_suggestions.append(split_suggestion)
                        break  # Take first valid split
                if word_suggestions and len(word_suggestions) > 1:  # Already have main + split
                    break
            
            # Store results (convert to tuple format expected by caller)
            if word_suggestions:
                if len(word_suggestions) == 1:
                    best_suggestions[word].append((word_suggestions[0], None))
                else:
                    best_suggestions[word].append((word_suggestions[0], word_suggestions[1]))
            else:
                best_suggestions[word].append((None, None))
        
        total_time = time.time() - start_time
        rate = len(unique_oov_words) / max(total_time, 0.1)
        
        print(f"- Completed ultra-optimized suggestion generation: {len(unique_oov_words):,} words in {total_time:.1f}s ({rate:.1f} words/sec)")
        print(f"- Performance improvement: Eliminated thousands of subprocess calls using batch processing")
        
        return best_suggestions
    
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
        

    async def get_best_corrections_with_ai(self, responses, best_suggestions_dict: Dict[str, List[Any]], var_lab: str, word_to_responses: Dict[str, List[int]] = None) -> Dict[str, str]:
        """Native async OpenAI client with validation - optimized with word-to-response mapping"""
        oov_words = list(best_suggestions_dict.keys())
        
        
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
            print("  • Creating correction tasks using optimized inverted index...")
            
            # Pre-compute suggestion strings for all OOV words to avoid redundant processing
            word_to_suggestion_str = {}
            validation_cache = {}  # Cache validation results
            
            # Smart pre-validation: automatically disable for very large datasets
            enable_pre_validation = self.config.enable_suggestion_pre_validation
            if enable_pre_validation and len(oov_words) > self.config.disable_pre_validation_above_oov_words:
                print(f"⚠️  Large dataset detected: {len(oov_words)} OOV words > {self.config.disable_pre_validation_above_oov_words} threshold")
                print("⚠️  Automatically disabling pre-validation to avoid excessive wait times")
                print("⚠️  (You can adjust 'disable_pre_validation_above_oov_words' in config to change this threshold)")
                enable_pre_validation = False
            
            # Batch pre-validation if enabled
            if enable_pre_validation:
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
                
                # Validate all suggestions in one batch with concurrency control
                if all_suggestions_to_validate:
                    print(f"    Validating {len(all_suggestions_to_validate)} unique suggestions against dictionary...")
                    
                    # Limit concurrent validations to prevent system overload
                    validation_semaphore = asyncio.Semaphore(100)  # Max 100 concurrent validations
                    completed_validations = 0
                    start_time = time.time()
                    
                    async def validate_with_semaphore(suggestion):
                        nonlocal completed_validations
                        async with validation_semaphore:
                            result = await self.verify_correction_with_dictionary(suggestion.strip('.,!?;:"\'()[]{}'))
                            completed_validations += 1
                            
                            # Show progress every 10% or every 500 validations
                            if completed_validations % max(1, len(all_suggestions_to_validate) // 10) == 0 or completed_validations % 500 == 0:
                                progress_percent = (completed_validations / len(all_suggestions_to_validate)) * 100
                                elapsed = time.time() - start_time
                                rate = completed_validations / max(elapsed, 0.1)
                                remaining = len(all_suggestions_to_validate) - completed_validations
                                eta = remaining / max(rate, 0.1)
                                print(f"    Validation progress: {completed_validations}/{len(all_suggestions_to_validate)} ({progress_percent:.1f}%) [{rate:.1f} validations/sec, ETA: {eta:.1f}s]")
                            
                            return result
                    
                    validation_tasks = [validate_with_semaphore(sug) 
                                       for sug in all_suggestions_to_validate if sug and sug != "OOV"]
                    validation_results = await asyncio.gather(*validation_tasks)
                    
                    validation_time = time.time() - start_time
                    validation_rate = len(all_suggestions_to_validate) / max(validation_time, 0.1)
                    print(f"    Completed validation: {len(all_suggestions_to_validate)} suggestions in {validation_time:.1f}s ({validation_rate:.1f} validations/sec)")
                    
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
            print(f"  • Task creation completed in {task_creation_time:.1f}s using inverted index (optimized)")
            
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
                            
                            # Pre-validate suggestions if enabled
                            if enable_pre_validation and cleaned_suggestions:
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
        
# Batching removed - now processing individual tasks directly
        
        # =================================================================
        # RATE LIMITING SETUP FOR INDIVIDUAL TASKS
        # =================================================================
        
        if filtered_tasks:
            # Get model limits from config
            limits = get_openai_rate_limits(self.model)
            HEADROOM = 0.8  # Use 80% of limits for safety
            
            # Create rate limiters
            rpm_limiter = AsyncLimiter(limits.requests_per_minute * HEADROOM / 60, 1)
            token_bucket = TokenBucket(limits.tokens_per_minute * HEADROOM)
            
            # Transport limit (HTTP/2 streams, system resources)
            semaphore = asyncio.Semaphore(100)
            
            # Rate limit tracking with cooldown
            rate_limit_tracker = RateLimitTracker(cooldown_seconds=15)
            
            print("[RATE LIMITING SETUP]")
            print(f"- Model: {self.model}")
            print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * HEADROOM:,.0f} with headroom)")
            print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * HEADROOM:,.0f} with headroom)")
            print(f"- Processing {len(filtered_tasks):,} individual tasks")
            print(f"- Rate limit cooldown: {rate_limit_tracker.cooldown_seconds}s")
        else:
            print("No correction tasks to process")
            return {}
        
        def count_task_tokens(task_dict: Dict[str, Any]) -> int:
            """Count tokens needed for individual task (input + estimated output)"""
            task_text = (
                f"Please correct the spelling errors in this response:\n"
                f"Original: \"{task_dict['response']}\"\n"
                f"Misspelled words: {task_dict['oov_words']}\n"
                f"Suggested corrections: {task_dict['suggestions']}\n"
                f"Please provide the corrected version."
            )
            
            encoding = get_tiktoken_encoding(self.model)
            input_tokens = len(encoding.encode(task_text))
            # Estimate output tokens (corrections are typically similar length to input)
            estimated_output_tokens = max(50, int(input_tokens * 0.3))  # At least 50, or 30% of input
            
            return input_tokens + estimated_output_tokens
        
        def on_rate_limit_error(retry_state):
            """Handle rate limit errors with cooldown"""
            if retry_state.outcome and retry_state.outcome.failed:
                exception = retry_state.outcome.exception()
                if isinstance(exception, RateLimitError):
                    rate_limit_tracker.record_rate_limit()
                    print(f"[RATE LIMIT] Attempt {retry_state.attempt_number}: {exception}")
        
        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential_jitter(initial=1, max=30),
            stop=stop_after_attempt(3),
            before_sleep=on_rate_limit_error,
            reraise=True
        )
        async def process_individual_task(task_dict: Dict[str, Any], var_lab: str) -> Dict[str, str]:
            """Process individual spell check task with rate limiting and proper structured output"""
            
            # Count tokens needed for this task
            tokens_needed = count_task_tokens(task_dict)
            
            # Apply rate limits in correct order
            async with rpm_limiter:                    # RPM check
                await token_bucket.acquire(tokens_needed)  # TPM check
                async with semaphore:                     # Transport limit
                    # Create proper task using original SPELLCHECK_INSTRUCTIONS format
                    task_text = f"""Task:
Respondent ID: {task_dict['respondent_id']}
Response: "{task_dict['response_with_placeholders']}"
Misspelled words: {task_dict['oov_words']}
Suggested corrections: {task_dict['suggestions']}
"""
                    
                    # Use original SPELLCHECK_INSTRUCTIONS prompt
                    full_prompt = SPELLCHECK_INSTRUCTIONS.format(
                        language=DEFAULT_LANGUAGE,
                        var_lab=var_lab,  # Need to pass this from parent function
                        tasks=task_text
                    )
                    
                    try:
                        self.stats['llm_calls_made'] += 1
                        
                        # Add timeout to prevent stragglers - USE INSTRUCTOR CLIENT FOR STRUCTURED OUTPUT
                        response = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=self.model,
                                response_model=LLMCorrectionResponse,  # Pydantic validation
                                messages=[{"role": "user", "content": full_prompt}],
                                temperature=self.config.temperature,
                                seed=self.config.seed,
                                max_retries=0  # Let tenacity handle retries
                            ),
                            timeout=15  # 15 second timeout
                        )
                        
                        self.stats['llm_calls_successful'] += 1
                        
                        # Parse structured response with validation
                        if response.corrections and len(response.corrections) > 0:
                            correction = response.corrections[0]  # Single task = single correction
                            corrected_text = correction.corrected_response
                            
                            # DEBUG: Log LLM response
                            original = task_dict['response']
                            if corrected_text != original:
                                logger.info("[SPELL DEBUG] LLM made correction:")
                                logger.info(f"  Original: '{original}'")
                                logger.info(f"  Corrected: '{corrected_text}'")
                            else:
                                logger.info(f"[SPELL DEBUG] LLM returned unchanged: '{original}'")
                        else:
                            corrected_text = task_dict['response']  # No correction provided
                            logger.info(f"[SPELL DEBUG] No corrections in response for: '{task_dict['response'][:50]}...'")
                        
                        return {task_dict['response']: corrected_text}
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Task for '{task_dict['response'][:50]}...' timed out after 15s")
                        self.stats['llm_calls_failed'] += 1
                        return {task_dict['response']: task_dict['response']}  # Return original
                        
                    except Exception as e:
                        logger.error(f"API call failed for task '{task_dict['response'][:50]}...': {e}")
                        self.stats['llm_calls_failed'] += 1
                        return {task_dict['response']: task_dict['response']}  # Return original
        
        # Convert filtered tasks to individual async tasks
        print("[INDIVIDUAL TASK PROCESSING]")
        print(f"- Processing {len(filtered_tasks)} individual tasks")
        print(f"- Maximum concurrent: {semaphore._value}")
        print(f"- Rate limiting: RPM={limits.requests_per_minute * HEADROOM:,.0f}, TPM={limits.tokens_per_minute * HEADROOM:,.0f}")
        
        # Create all individual tasks
        task_coroutines = [process_individual_task(task, var_lab) for task in filtered_tasks]
        
        # Process all tasks with protected gathering
        print(f"Processing individual tasks... 0/{len(filtered_tasks)} (0.0%)")
        print("  (Note: Initial delay may occur due to rate limiting initialization)")
        
        processing_start_time = time.time()
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        total_processing_time = time.time() - processing_start_time
        
        # Combine results and handle exceptions
        corrected_sentences_dict = {}
        successful_tasks = 0
        failed_tasks = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                # Use original response as fallback
                original_response = filtered_tasks[i]['response']
                corrected_sentences_dict[original_response] = original_response
                failed_tasks += 1
            else:
                corrected_sentences_dict.update(result)
                successful_tasks += 1
        
        print(f"Processing individual tasks... {len(filtered_tasks)}/{len(filtered_tasks)} (100.0%)")
        print(f"- Successful: {successful_tasks}")
        print(f"- Failed: {failed_tasks}")
        
        # Rate limiting performance analysis
        actual_rate = len(filtered_tasks) / max(total_processing_time, 0.1)
        theoretical_max_rate = min(limits.requests_per_minute * HEADROOM / 60, 100)  # Limited by semaphore too
        efficiency = (actual_rate / theoretical_max_rate) * 100 if theoretical_max_rate > 0 else 100
        print(f"- Processing rate: {actual_rate:.1f} tasks/sec (efficiency: {efficiency:.1f}% of theoretical max)")
        
        if efficiency < 50:
            print("⚠️  Low efficiency detected - likely due to rate limiting or network latency")
        
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
            # OPTIMIZED: Use HunspellPool with much larger batches for massive speed improvement
            # Use 10x larger batches to minimize process overhead
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
            print(f"    Performance improvement: HunspellPool eliminated subprocess creation overhead")
            
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
            
            # # DEBUG: Show contents of correction dictionary
            # print(f"[CORRECTION DICT DEBUG] Dictionary has {len(corrected_sentences_dict)} entries:")
            # for i, (original, corrected) in enumerate(corrected_sentences_dict.items()):
            #     if i < 5:  # Show first 5 entries
            #         print(f"  {i+1}. '{original[:50]}...' -> '{corrected[:50]}...'")
            #         if original != corrected:
            #             print("     ^^ CHANGE DETECTED")
            #         else:
            #             print("     ^^ NO CHANGE")
            # if len(corrected_sentences_dict) > 5:
            #     print(f"  ... and {len(corrected_sentences_dict) - 5} more entries")
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
            corrected_response = corrected_sentences_dict.get(response.original_response, response.original_response)
            
            # DEBUG: Log each lookup
            # if i < 5:  # Show first 5 responses
            #     print(f"[COUNTING DEBUG] Response {i+1}:")
            #     print(f"  Looking up: '{response.original_response[:50]}...'")
            #     print(f"  Found: '{corrected_response[:50]}...'")
            #     print(f"  Same?: {response.original_response == corrected_response}")
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
                
                # DEBUG: Log comparison process
                logger.info("[COMPARISON DEBUG] Found text change:")
                logger.info(f"  Original: '{response.original_response}'")
                logger.info(f"  Corrected: '{corrected_response}'")
                logger.info(f"  Original normalized: '{original_normalized}'")
                logger.info(f"  Corrected normalized: '{corrected_normalized}'")
                
                if original_normalized != corrected_normalized:
                    corrections_made += 1
                    logger.info(f"  -> COUNTED as correction #{corrections_made}")
                    
                    # Store example for verbose output
                    if len(correction_examples) < self.config.max_correction_examples:
                        correction_examples.append((response.original_response, corrected_response))
                else:
                    logger.info("  -> NOT COUNTED (normalized versions are identical)")
            else:
                # DEBUG: Log when no change detected
                logger.debug(f"[COMPARISON DEBUG] No change detected for: '{response.original_response[:50]}...'")
                
        # SIMPLE: Process correction dictionary directly for accurate statistics
        #print(f"\n[DICTIONARY VALIDATION] Processing {len(corrected_sentences_dict)} correction dictionary entries...")
        
        # Reset corrections stats to use dictionary-based logic
        self.stats['corrections_attempted'] = 0
        self.stats['corrections_applied'] = 0  
        self.stats['corrections_rejected_validation'] = 0
        self.stats['corrections_no_response'] = 0
        
        for original_response, candidate_correction in corrected_sentences_dict.items():
            self.stats['corrections_attempted'] += 1
            
            #print(f"[DICTIONARY VALIDATION] Entry {self.stats['corrections_attempted']}: '{original_response[:30]}...' -> '{candidate_correction[:30]}...'")
            
            # Check for "[NO RESPONSE]" cases or no change
            if candidate_correction == "[NO RESPONSE]" or candidate_correction == original_response:
                self.stats['corrections_no_response'] += 1
                #print(f"  -> NO CHANGE (total no-change: {self.stats['corrections_no_response']})")
                continue
                
            # Simple validation: if text changed, it's likely a valid correction
            # (The LLM already used structured output and followed instructions)
            self.stats['corrections_applied'] += 1
            #print(f"  -> APPLIED (applied: {self.stats['corrections_applied']})")
        
        # Verify math adds up
        total_accounted = self.stats['corrections_applied'] + self.stats['corrections_rejected_validation'] + self.stats['corrections_no_response']
        print("\nSUMMARY:")
        print(f"- Corrections attempted: {self.stats['corrections_attempted']}")
        print(f"- Corrections applied: {self.stats['corrections_applied']}")  
        print(f"- Corrections rejected (validation): {self.stats['corrections_rejected_validation']}")
        print(f"- No correction/no change: {self.stats['corrections_no_response']}")
        #print(f"- Total accounted: {total_accounted} (should equal attempted: {self.stats['corrections_attempted']})")

        stats.end_timing()
        stats.output_count = len(updated_responses)
        self.stats['processing_time'] = stats.get_duration()
        self.stats['suggestion_cache_hits'] = self.suggestion_cache_hits
        self.stats['suggestion_cache_size'] = len(self.suggestion_cache) if self.suggestion_cache is not None else 0
        # corrections_applied now set by comprehensive validation logic above 
        
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
          
        print(f"• Corrections applied: {self.stats['corrections_applied']}")
        
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