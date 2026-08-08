import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import re
import asyncio
import subprocess
import logging
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import nest_asyncio
from pydantic import BaseModel

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cachedResources import get_spacy_nlp_conditional
from utils.smoothRequester import SmoothRequester
from utils.llm import token_tracker
from config import get_reasoning_params

# === CONFIG — generic/universal ========================================================================================================
from config import (
    DEFAULT_LANGUAGE,
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
)

# === CONFIG — step-specific ========================================================================================================
from pipeline.step_1_preProcessor.config_preProcessor import (
    HUNSPELL_PATH, DUTCH_DICT_PATH, ENGLISH_DICT_PATH,
    SpellCheckConfig, DEFAULT_SPELLCHECK_CONFIG,
    MAX_HUNSPELL_PROCESSES, MAX_SAFE_BATCH_SIZE,
    SUGGESTION_BATCH_SIZE, MAX_CONCURRENT_SUGGESTION_BATCHES,
    SPACY_VECTOR_NORM_THRESHOLD,
    MAX_SUGGESTIONS_SHOWN,
    WORD_VOWELS,
    MAX_REPEATED_CHARS,
    MAX_CONSONANT_RUN,
    COMPOUND_FIRST_POS,
    COMPOUND_MIN_WORD_LENGTH,
)

# === PROMPTS ========================================================================================================
from pipeline.step_1_preProcessor.prompts_preProcessor import (
    SPELLCHECK_INSTRUCTIONS,
    LLMCorrectionResponse,
    SPLIT_COMPOUND_INSTRUCTIONS,
    SplitCompoundResponse,
)

logger = logging.getLogger(__name__)
DICT_PATH = DUTCH_DICT_PATH if DEFAULT_LANGUAGE == "Dutch" else ENGLISH_DICT_PATH

_REPEATED_CHARS = re.compile(rf"(.)\1{{{MAX_REPEATED_CHARS},}}")
_CONSONANT_RUN = re.compile(rf"[^{WORD_VOWELS}]{{{MAX_CONSONANT_RUN + 1},}}")

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
    def __init__(self, config: SpellCheckConfig = None, processing_config: ProcessingConfig = None, verbose: bool = False, prompt_printer = None, verbose_reporter: Optional['VerboseReporter'] = None, cost_tracker = None):
        self.config = config or DEFAULT_SPELLCHECK_CONFIG
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.config.model
        self.cost_tracker = cost_tracker

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_1_preprocessing", {
                "spell_check": self.model,
            })

        self.suggestion_cache = {} if self.config.enable_suggestion_caching else None
        self.suggestion_cache_hits = 0

        self.hunspell_path = HUNSPELL_PATH
        self.dict_path = DICT_PATH
        self.prompt_printer = prompt_printer
        self._prompt_captured = False  # capture one example, not one per respondent
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.empty_line()
            print("Spell checker configuration:")
            self.verbose_reporter.stat_line(f"Model: {self.model}", indent=1)
            self.verbose_reporter.stat_line(f"Language: {DEFAULT_LANGUAGE}", indent=1)
            self.verbose_reporter.stat_line(f"Dictionary: {self.dict_path}", indent=1)
            self.verbose_reporter.stat_line(f"Hunspell path: {self.hunspell_path}", indent=1)
            self.verbose_reporter.stat_line(f"Hunspell batch: {self.config.hunspell_batch_size * 10} words", indent=1)

        if not self.check_hunspell_installation():
            if self.verbose_reporter.enabled:
                self.verbose_reporter.warning("Hunspell is not properly installed or configured - spell checking may fail")
                self.verbose_reporter.warning(f"Expected Hunspell at: {self.hunspell_path}")
                self.verbose_reporter.warning(f"Expected dictionary at: {self.dict_path}")
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("OK Hunspell installation verified")

        self.hunspell_pool = None
        self.failed_task_ids = set()  # respondent_ids SmoothRequester could not resolve
        self.stats = {
            'words_checked': 0,
            'oov_words_found': 0,
            'unique_oov_words': 0,
            'dataset_vocabulary_words': 0,
            'unrepairable_words': 0,
            'split_compound_candidates': 0,
            'split_compounds_joined': 0,
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
    def is_checkable(text: str) -> bool:
        """Kan dit token een verkeerd gespeld woord zijn?

        Ruimer dan `is_alpha`, en precies één klasse ruimer: letters met een
        cijfer ertussen ("2eet", "Go4ed"). Dat is een typefout waarvan Hunspell
        het antwoord kent. Tokens met een leesteken blijven buiten beeld —
        "zzp-ers" en "i.o." zijn correct zoals ze staan, en binnenhalen zou ze
        aan een model aanbieden dat verplicht is iets te veranderen.
        """
        return text.isalnum() and any(c.isalpha() for c in text)

    @staticmethod
    def is_unrepairable(word: str) -> bool:
        """True for a token that is not language: no vowel, a hammered key, or a
        consonant run no word has. There is nothing to correct such a token to, so
        it must not become a correction task — asked anyway, the LLM returns a
        plausible word ("Xxx" -> "Mexx") and the noise becomes invisible.

        Measured on four datasets: catches all 18 known junk tokens, leaves 25 of
        26 known typos alone, and protects 14 of 15 acronyms as a side effect.
        """
        w = word.lower()
        if any(c.isdigit() for c in w):
            # Een cijfer staat voor een letter die we niet kunnen zien: "N8ks"
            # is "Niks". De klinkertoets zou hier "geen klinker" concluderen op
            # grond van een teken dat juist de klinker vervangt. Alleen de
            # hamerslag-toets bewijst dan nog iets.
            return bool(_REPEATED_CHARS.search(w))
        return (not any(c in WORD_VOWELS for c in w)
                or bool(_REPEATED_CHARS.search(w))
                or bool(_CONSONANT_RUN.search(w)))

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
            new_suggestions = {word: [] for word in unique_oov_words}
        
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
                        """Rank Hunspell's candidates; do not choose between them.

                        Edit distance cannot: for "Geeb" it puts Gees, Geer, Geel,
                        Geen and Geef all at 1, so an argmin here returns whichever
                        Hunspell listed first and the right answer never reaches the
                        prompt. Ranking is useful, deciding is not — the model has the
                        survey question and the sentence, and needs candidates to
                        apply them to.
                        """
                        try:
                            unsplit_suggestions = await self.run_hunspell_word_async(word)

                            left_part, right_part = await self.find_best_split_for_spellcheck(word)
                            # A "split" where either half is the whole word is not a
                            # split; it used to reach the prompt as "reklame reklame".
                            split_suggestion = (
                                f"{left_part} {right_part}"
                                if (left_part and right_part
                                    and left_part.lower() != word.lower()
                                    and right_part.lower() != word.lower())
                                else None)

                            ranked = sorted(
                                dict.fromkeys(s for s in unsplit_suggestions if s and s != word),
                                key=lambda s: self.cached_levenshtein_distance(word, s),
                            )[:MAX_SUGGESTIONS_SHOWN]
                            if split_suggestion:
                                ranked.append(split_suggestion)

                            return word, ranked

                        except Exception as e:
                            logger.error(f"Error processing word '{word}' in batch {batch_index}: {e}")
                            return word, []
                    
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
                    return {word: [] for word in batch_words}
        
        # Process all batches concurrently
        print("- Starting concurrent batch processing...")
        batch_tasks = [process_batch(batch_words, batch_idx) for batch_words, batch_idx in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Combine results from all batches: word -> ranked list of candidates
        for batch_result in batch_results:
            for word, ranked in batch_result.items():
                best_suggestions[word] = ranked
        
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
    
        # Performance tracking for task creation
        task_creation_start = time.time()

        # Use inverted index if available, otherwise fall back to regex search
        if word_to_responses is not None:
            print("  • Creating correction tasks using optimized inverted index...")
            
            # Pre-compute suggestion strings for all OOV words to avoid redundant processing
            word_to_suggestion_str = {}

            for word in oov_words:
                if len(word) > 2 and word in best_suggestions_dict:
                    word_to_suggestion_str[word] = [
                        s for s in best_suggestions_dict[word] if s and s != "OOV"]
            
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
                    for word in response_oov_words:
                        cleaned_suggestions = word_to_suggestion_str.get(word, [])
                        all_suggestions.append(", ".join(cleaned_suggestions) if cleaned_suggestions else "OOV")

                    tasks.append({
                        "respondent_id": item['respondent_id'],
                        "response": response,
                        "response_with_placeholders": response_with_placeholders,
                        "oov_words": ", ".join(response_oov_words),
                        "suggestions": " | ".join(all_suggestions)
                    })

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
                        
                        # Get suggestions for all OOV words
                        all_suggestions = []
                        for word in response_oov_words:
                            cleaned = [s for s in best_suggestions_dict.get(word, [])
                                       if s and s != "OOV"]
                            all_suggestions.append(", ".join(cleaned) if cleaned else "OOV")

                        tasks.append({
                            "respondent_id": item['respondent_id'],
                            "response": response,
                            "response_with_placeholders": response_with_placeholders,
                            "oov_words": ", ".join(response_oov_words),
                            "suggestions": " | ".join(all_suggestions)
                        })

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


        # Count unique OOV words that made it into tasks
        oov_words_in_tasks = set()
        for task in filtered_tasks:
            task_oov_words = [word.strip() for word in task['oov_words'].split(',')]
            oov_words_in_tasks.update(task_oov_words)
        self.stats['oov_words_in_tasks'] = len(oov_words_in_tasks)

        ############################################################################################################################################
        # Prompt processing starts here
        ########################################################################################################################
        
        if not filtered_tasks:
            print("No correction tasks to process")
            return {}

        # Dispatch through SmoothRequester: it owns workers, pacing, concurrency
        # control, adaptive timeouts, the retry pass and the warm start. This
        # module owns only prompt building and response parsing.
        requester = SmoothRequester(
            model=self.model,
            phase_key="step1_spell_check",
            num_tasks=len(filtered_tasks),
            verbose=self.verbose_reporter.enabled,
            processing_config=self.processing_config,
        )

        _snap_before = token_tracker.snapshot() if self.cost_tracker else None

        results = await requester.process_all(
            filtered_tasks,
            self._build_prepare_fn(),
            self._build_parse_fn(),
            self._build_fallback_fn(),
        )

        if self.cost_tracker and _snap_before is not None:
            self.cost_tracker.record_phase(
                "step_1_preprocessing", "spell_check",
                _snap_before, token_tracker.snapshot(), self.model)

        corrected_sentences_dict = {}
        for result in results:
            if result:
                corrected_sentences_dict.update(result)

        self.stats['llm_calls_successful'] = requester.stats['tasks_successful']
        self.stats['llm_calls_failed'] = requester.stats['tasks_failed']

        return corrected_sentences_dict

    # === SMOOTHREQUESTER CALLBACKS ==============================================

    def _build_prepare_fn(self):
        checker = self

        def prepare_fn(task: Dict) -> Dict:
            task_text = f"""Task:
Respondent ID: {task['respondent_id']}
Response: "{task['response_with_placeholders']}"
Misspelled words: {task['oov_words']}
Suggested corrections: {task['suggestions']}
"""
            prompt = SPELLCHECK_INSTRUCTIONS.format(
                language=DEFAULT_LANGUAGE,
                var_lab=task.get('var_lab', checker.var_lab),
                tasks=task_text,
            )

            if checker.prompt_printer is not None and not checker._prompt_captured:
                checker._prompt_captured = True
                checker.prompt_printer.capture_prompt(
                    step_name="preprocessor",
                    utility_name="SpellChecker",
                    prompt_content=prompt,
                    prompt_type="spell_correction",
                    metadata={
                        "model": checker.model,
                        "temperature": checker.config.temperature,
                        "language": DEFAULT_LANGUAGE,
                        "var_lab": task.get('var_lab', checker.var_lab),
                    },
                )

            return {
                'prompt': prompt,
                'response_model': LLMCorrectionResponse,
                'temperature': checker.config.temperature,
                'extra_kwargs': get_reasoning_params(checker.model, phase="spell_check"),
            }

        return prepare_fn

    def _build_parse_fn(self):
        def parse_fn(task: Dict, response) -> Optional[Dict[str, str]]:
            if not response or not response.corrections:
                return None

            correction = response.corrections[0]  # one task = one correction

            # AUDIT: the LLM echoing a different id means the prompt drifted
            if str(correction.respondent_id) != str(task['respondent_id']):
                logger.warning(
                    f"ID drift detected: LLM returned '{correction.respondent_id}' "
                    f"but input was '{task['respondent_id']}'"
                )

            return {task['respondent_id']: correction.corrected_response}

        return parse_fn

    def _build_fallback_fn(self):
        checker = self

        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            """Keep the original text when the call cannot be resolved."""
            checker.failed_task_ids.add(task['respondent_id'])
            return {task['respondent_id']: task['response']}

        return fallback_fn


    # === SPLIT COMPOUNDS ========================================================

    async def _confirm_split_compounds(self, candidates: Dict[int, List[Tuple[str, str]]]
                                       ) -> Dict[int, List[Tuple[str, str]]]:
        """Keep only the pairs whose glued form Hunspell recognises as a word."""
        if not candidates:
            return {}

        glued = sorted({(a + b).lower() for pairs in candidates.values() for a, b in pairs})
        if self.hunspell_pool is None:
            self._init_hunspell_pool()
        outputs = await self.hunspell_pool.check_words_batch(
            glued, self.config.hunspell_batch_size)
        is_word = {w: not (o and o.startswith(('&', '#')))
                   for w, o in zip(glued, outputs)}

        confirmed = {idx: [(a, b) for a, b in pairs if is_word.get((a + b).lower())]
                     for idx, pairs in candidates.items()}
        confirmed = {idx: pairs for idx, pairs in confirmed.items() if pairs}

        self.stats['split_compound_candidates'] = sum(len(p) for p in confirmed.values())
        if confirmed:
            print(f"  • Split-compound candidates: {self.stats['split_compound_candidates']} "
                  f"pairs in {len(confirmed)} responses")
        return confirmed

    async def _resolve_split_compounds(self, responses: List[SpellCheckModel],
                                       corrected: Dict[Any, str],
                                       candidates: Dict[int, List[Tuple[str, str]]],
                                       var_lab: str) -> Dict[Any, str]:
        """Ask which candidate pairs are one word, then glue those. The model returns
        verdicts, never text: the edit itself is a string join here, so this phase
        cannot reword, drop or reorder anything."""
        tasks = []
        for idx, pairs in candidates.items():
            response = responses[idx]
            text = corrected.get(response.respondent_id, response.original_response)
            # Only pairs still present after spell correction can be glued.
            live = [(a, b) for a, b in pairs
                    if re.search(rf'\b{re.escape(a)}\s+{re.escape(b)}\b', text)]
            if live:
                tasks.append({'respondent_id': response.respondent_id,
                              'response': text, 'pairs': live, 'var_lab': var_lab})

        if not tasks:
            return corrected

        requester = SmoothRequester(
            model=self.model,
            phase_key="step1_split_compound",
            num_tasks=len(tasks),
            verbose=self.verbose_reporter.enabled,
            processing_config=self.processing_config,
        )

        _snap_before = token_tracker.snapshot() if self.cost_tracker else None
        results = await requester.process_all(
            tasks,
            self._build_compound_prepare_fn(),
            self._build_compound_parse_fn(),
            lambda task, reason: {},          # unresolved: leave the text as it is
        )
        if self.cost_tracker and _snap_before is not None:
            self.cost_tracker.record_phase(
                "step_1_preprocessing", "split_compound",
                _snap_before, token_tracker.snapshot(), self.model)

        # One verdict per pair, not per response. Whether two words form a compound is
        # a property of the pair, not of the sentence it sits in — and the model does
        # not treat it that way: measured on ASN Qd1 it joined "milieu vriendelijk"
        # four times and refused three, in near-identical responses. Half a corpus
        # corrected is worse than none, because step 3 then sees two strings for one
        # idea. So the corpus votes, and every occurrence follows the majority.
        # A tie joins: in Dutch the closed compound is the more common correct form.
        votes = defaultdict(lambda: [0, 0])
        for result in results:
            for verdicts in (result or {}).values():
                for a, b, join in verdicts:
                    key = (a.lower(), b.lower())
                    votes[key][0] += int(join)
                    votes[key][1] += 1

        join_pairs = {key for key, (yes, total) in votes.items() if yes and yes * 2 >= total}
        split_votes = sum(1 for key in join_pairs if votes[key][0] < votes[key][1])

        merged = dict(corrected)
        joined = 0
        for task in tasks:
            text = task['response']
            for a, b in task['pairs']:
                if (a.lower(), b.lower()) in join_pairs:
                    text, n = re.subn(rf'\b{re.escape(a)}\s+{re.escape(b)}\b', a + b, text)
                    joined += n
            if text != task['response']:
                merged[task['respondent_id']] = text

        self.stats['split_compounds_joined'] = joined
        print(f"  • Split compounds joined: {joined} of "
              f"{self.stats['split_compound_candidates']} candidates "
              f"({len(join_pairs)} distinct pairs, {split_votes} decided by corpus majority)")
        return merged

    def _build_compound_prepare_fn(self):
        checker = self

        def prepare_fn(task: Dict) -> Dict:
            candidates = "\n".join(f'- "{a} {b}"' for a, b in task['pairs'])
            prompt = SPLIT_COMPOUND_INSTRUCTIONS.format(
                language=DEFAULT_LANGUAGE,
                var_lab=task['var_lab'],
                response=task['response'],
                candidates=candidates,
            )
            return {
                'prompt': prompt,
                'response_model': SplitCompoundResponse,
                'temperature': checker.config.temperature,
                'extra_kwargs': get_reasoning_params(checker.model, phase="spell_check"),
            }

        return prepare_fn

    def _build_compound_parse_fn(self):
        def parse_fn(task: Dict, response) -> Optional[Dict[Any, List[Tuple[str, str, bool]]]]:
            if not response or not response.verdicts:
                return None
            # Match verdicts back on the pair text, not on position: a model that
            # drops or reorders one must not shift the others onto wrong pairs.
            # Both answers are kept — the corpus vote needs the refusals too.
            wanted = {f"{a} {b}".lower(): (a, b) for a, b in task['pairs']}
            verdicts = [(*wanted[key], bool(v.join)) for v in response.verdicts
                        if (key := v.pair.strip().lower()) in wanted]
            return {task['respondent_id']: verdicts}

        return parse_fn

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
        diag_skipped_not_word = 0
        diag_skipped_named_entity = 0
        diag_skipped_too_short = 0

        # Candidate split compounds, collected in this same SpaCy pass. A pair only
        # qualifies once Hunspell confirms the glued form is a word — checked below,
        # together with the OOV batch.
        compound_candidates = defaultdict(list)

        for response_idx, doc in enumerate(self.get_nlp().pipe(sentences_list, batch_size=self.config.spacy_batch_size)):
            for first, second in zip(doc[:-1], doc[1:]):
                if (first.pos_ in COMPOUND_FIRST_POS
                        and first.is_alpha and second.is_alpha
                        and len(first.text) >= COMPOUND_MIN_WORD_LENGTH
                        and len(second.text) >= COMPOUND_MIN_WORD_LENGTH
                        and second.idx == first.idx + len(first.text) + 1):
                    compound_candidates[response_idx].append((first.text, second.text))

            for token in doc:
                diag_total_tokens += 1

                if not self.is_checkable(token.text):
                    diag_skipped_not_word += 1
                    continue

                # Named entity filter REMOVED - was catching typos like "merkiglo", "merr"
                # Just track count for reporting
                if token.ent_type_ != "":
                    diag_skipped_named_entity += 1

                if len(token.text) <= 2:
                    diag_skipped_too_short += 1
                    continue

                # Word passed filters (is_checkable and len > 2). The cache is keyed on
                # the word as written: Hunspell is case-sensitive, so "Nederlands"
                # and "nederlands" get different verdicts and cannot share an entry.
                word = token.text

                # Check cache first
                if word_frequency_cache is not None and word in word_frequency_cache:
                    is_oov = word_frequency_cache[word]
                    if is_oov:
                        oov_words.append(word)
                        self.stats['oov_words_found'] += 1
                        word_to_responses[word].append(response_idx)
                else:
                    # Add to batch for Hunspell checking
                    all_words_to_check.append((word, response_idx))

        # DIAGNOSTIC: Print word identification filter stats
        # Note: Named entities are now INCLUDED (not skipped) - only tracking for info
        diag_passed_filters = diag_total_tokens - diag_skipped_not_word - diag_skipped_too_short
        print(f"  • Word filters: {diag_total_tokens:,} tokens → {diag_passed_filters:,} passed ({diag_passed_filters/max(diag_total_tokens,1)*100:.1f}%)")
        print(f"    (skipped: {diag_skipped_not_word:,} geen woord, {diag_skipped_too_short:,} too short; {diag_skipped_named_entity:,} named entities now included)")
        print(f"  • Cached words processed, {len(all_words_to_check):,} words need Hunspell verification")
        
        if all_words_to_check:
            batch_size = self.config.hunspell_batch_size * 10  # 10,000 words per batch instead of 1,000
            
            print(f"  • Processing {len(all_words_to_check):,} words using HunspellPool with large batches...")
            
            # Initialize HunspellPool for efficient processing
            if self.hunspell_pool is None:
                self._init_hunspell_pool()
            
            start_time = time.time()
            
            # Hunspell sees the word as written, capitals included
            words_only = [word for word, _ in all_words_to_check]

            # Process all words in efficient batches using HunspellPool
            batch_outputs = await self.hunspell_pool.check_words_batch(words_only, batch_size)

            # Count Hunspell results
            diag_oov_count = sum(1 for output in batch_outputs if output and output.startswith(('&', '#')))
            diag_correct_count = len(batch_outputs) - diag_oov_count

            print(f"\n  • Hunspell: {diag_correct_count:,} correct, {diag_oov_count:,} OOV (dictionary: {self.dict_path})")

            # Process results and update cache
            response_flagged = set()
            for i, (word, response_idx) in enumerate(all_words_to_check):
                self.stats['words_checked'] += 1
                output = batch_outputs[i]
                is_oov = output and output.startswith(('&', '#'))

                # Cache the result
                if word_frequency_cache is not None:
                    word_frequency_cache[word] = is_oov

                if is_oov:
                    oov_words.append(word)
                    word_to_responses[word].append(response_idx)
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
            
        # --- Split compounds ---------------------------------------------------
        # A pair of words whose glued form is in the dictionary. Hunspell cannot see
        # this error class at all — both halves are valid words — so it never reaches
        # the OOV path. Keeping the pairs whose glued form exists turns the whole
        # class into a short list of yes/no questions.
        compound_candidates = await self._confirm_split_compounds(compound_candidates)

        # --- Dataset vocabulary ------------------------------------------------
        # An unknown word that recurs across many responses is this dataset's own
        # vocabulary — a brand, an abbreviation, a term of art — not a typo. Asking
        # the LLM to "correct" it is what damages the data, so it never becomes a
        # task. Case-insensitive, because respondents write the same name several
        # ways. This fails the safe way: a protected typo merely stays uncorrected,
        # while an unprotected name gets rewritten and the original is gone.
        response_count_per_word = defaultdict(set)
        for word, response_indices in word_to_responses.items():
            response_count_per_word[word.lower()].update(response_indices)

        vocab_threshold = max(
            self.config.dataset_vocab_min_responses,
            round(len(responses) * self.config.dataset_vocab_response_ratio))
        dataset_vocabulary = {
            word for word, indices in response_count_per_word.items()
            if len(indices) >= vocab_threshold}

        # Same treatment for tokens that are not language at all: nothing to correct
        # them to, so they must not become a task either.
        unrepairable = {w.lower() for w in word_to_responses if self.is_unrepairable(w)}

        leave_uncorrected = dataset_vocabulary | unrepairable
        if leave_uncorrected:
            oov_words = [w for w in oov_words if w.lower() not in leave_uncorrected]
            word_to_responses = defaultdict(list, {
                w: idx for w, idx in word_to_responses.items()
                if w.lower() not in leave_uncorrected})

        if dataset_vocabulary:
            preview = ", ".join(sorted(dataset_vocabulary)[:10])
            print(f"  • Dataset vocabulary left uncorrected "
                  f"(seen in >={vocab_threshold} responses): {preview}"
                  f"{', ...' if len(dataset_vocabulary) > 10 else ''}")
        if unrepairable:
            preview = ", ".join(sorted(unrepairable)[:10])
            print(f"  • Not language, left uncorrected: {preview}"
                  f"{', ...' if len(unrepairable) > 10 else ''}")

        self.stats['dataset_vocabulary_words'] = len(dataset_vocabulary)
        self.stats['unrepairable_words'] = len(unrepairable)

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
        
        # Verbose OOV analysis details. Count after the dataset-vocabulary filter:
        # before it, this number includes responses whose only unknown word is a
        # name that will never become a correction task.
        if self.verbose_reporter.enabled:
            responses_to_correct = len({idx for idxs in word_to_responses.values() for idx in idxs})
            self.verbose_reporter.stat_line(
                f"Responses requiring correction: {responses_to_correct} "
                f"(flagged before vocabulary filter: {docs_with_oov})")
        
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

        # Split compounds are invisible to Hunspell, so this phase runs on its own
        # candidate list — independent of whether the response had an OOV word.
        if compound_candidates:
            corrected_sentences_dict = await self._resolve_split_compounds(
                responses, corrected_sentences_dict, compound_candidates, var_lab)
        
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