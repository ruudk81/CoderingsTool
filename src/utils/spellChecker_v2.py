import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import re
import asyncio
import nest_asyncio
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple, AsyncContextManager
from openai import AsyncOpenAI
import tiktoken
import spacy
import subprocess
from collections import defaultdict
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json

from config import DEFAULT_MODEL, OPENAI_API_KEY, DEFAULT_LANGUAGE, HUNSPELL_PATH, DUTCH_DICT_PATH, ENGLISH_DICT_PATH, SpellCheckConfig, DEFAULT_SPELLCHECK_CONFIG, DEFAULT_MODEL_CONFIG
from prompts import SPELLCHECK_INSTRUCTIONS
import models
from .verboseReporter import VerboseReporter, ProcessingStats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nederlands of Engels (PRESERVED from v1)
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
class CorrectionConfidence:
    """Confidence scoring for corrections"""
    dictionary_verified: bool
    levenshtein_distance: int
    context_score: float
    hunspell_suggestion: bool
    total_score: float

# ============================================================================
# HUNSPELL CONNECTION POOL V2 - MAJOR IMPROVEMENT
# ============================================================================

class HunspellSessionV2:
    """Improved Hunspell session with better error handling and resource management"""
    
    def __init__(self, hunspell_path: str, dict_path: str):
        self.hunspell_path = hunspell_path
        self.dict_path = dict_path
        self.process = None
        self._lock = asyncio.Lock()
        self._is_closed = False
        self._initialize_process()
       
    def _initialize_process(self):
        """Initialize the Hunspell subprocess"""
        try:
            self.process = subprocess.Popen(
                [self.hunspell_path, "-a", "-d", self.dict_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1
            )
            # Read the initial Hunspell banner
            self.process.stdout.readline()
        except Exception as e:
            logger.error(f"Failed to initialize Hunspell process: {e}")
            raise
    
    async def check_word(self, word: str) -> str:
        """Check a word with thread safety"""
        async with self._lock:
            if self._is_closed or not self.process:
                raise RuntimeError("Hunspell session is closed")
            
            try:
                # Use executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self._check_word_sync, word)
                return result
            except Exception as e:
                logger.error(f"Error checking word '{word}': {e}")
                # Try to reinitialize process
                await self._reinitialize()
                return ""
    
    def _check_word_sync(self, word: str) -> str:
        """Synchronous word checking (called in executor)"""
        try:
            self.process.stdin.write(word + '\n')
            self.process.stdin.flush()
            result = self.process.stdout.readline().strip()
            
            # Read any additional lines
            while True:
                peek = self.process.stdout.readline().strip()
                if not peek:
                    break
                result += "\n" + peek
            
            return result
        except Exception as e:
            logger.error(f"Sync word check failed for '{word}': {e}")
            return ""
    
    async def _reinitialize(self):
        """Reinitialize the process if it fails"""
        try:
            if self.process:
                self.process.terminate()
            self._initialize_process()
            logger.info("Hunspell process reinitialized")
        except Exception as e:
            logger.error(f"Failed to reinitialize Hunspell: {e}")
    
    async def close(self):
        """Properly close the session"""
        async with self._lock:
            if not self._is_closed and self.process:
                try:
                    self.process.stdin.close()
                    self.process.stdout.close()
                    self.process.stderr.close()
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except Exception as e:
                    logger.error(f"Error closing Hunspell session: {e}")
                finally:
                    self._is_closed = True

class HunspellPoolV2:
    """Connection pool for Hunspell sessions - MAJOR V2 IMPROVEMENT"""
    
    def __init__(self, hunspell_path: str, dict_path: str, pool_size: int = 3):
        self.hunspell_path = hunspell_path
        self.dict_path = dict_path
        self.pool_size = pool_size
        self._pool: List[HunspellSessionV2] = []
        self._available = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection pool"""
        async with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.pool_size):
                try:
                    session = HunspellSessionV2(self.hunspell_path, self.dict_path)
                    self._pool.append(session)
                    await self._available.put(session)
                except Exception as e:
                    logger.error(f"Failed to create Hunspell session: {e}")
            
            self._initialized = True
            logger.info(f"Hunspell pool initialized with {len(self._pool)} sessions")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncContextManager[HunspellSessionV2]:
        """Get a session from the pool"""
        if not self._initialized:
            await self.initialize()
        
        session = await self._available.get()
        try:
            yield session
        finally:
            await self._available.put(session)
    
    async def close_all(self):
        """Close all sessions in the pool"""
        async with self._lock:
            for session in self._pool:
                await session.close()
            self._pool.clear()
            # Clear the queue
            while not self._available.empty():
                self._available.get_nowait()
            self._initialized = False

# ============================================================================
# SPELL CHECKER V2 - MAIN CLASS
# ============================================================================

class SpellCheckerV2:
    """V2 SpellChecker with async optimization, connection pooling, and improved accuracy"""
    
    def __init__(self, config: SpellCheckConfig = None, openai_api_key: Optional[str] = None, 
                 openai_model: str = None, verbose: bool = False, prompt_printer = None):
        self.config = config or DEFAULT_SPELLCHECK_CONFIG
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.openai_model = openai_model or DEFAULT_MODEL
        
        # V2 IMPROVEMENT: Native async OpenAI client instead of instructor
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        
        # PRESERVED: All Hunspell paths exactly as in v1
        self.hunspell_path = HUNSPELL_PATH
        self.dict_path = DICT_PATH  # V2 FIX: Uses language-aware path
        
        self.prompt_printer = prompt_printer 
        self.verbose_reporter = VerboseReporter(verbose)
        
        # V2 IMPROVEMENT: Connection pooling
        self.hunspell_pool = HunspellPoolV2(self.hunspell_path, self.dict_path, pool_size=3)
        
        # Check installation
        if not self.check_hunspell_installation():
            logger.warning("Hunspell is not properly installed or configured.")
        
        # V2 IMPROVEMENT: Enhanced stats tracking
        self.stats = {
            'words_checked': 0,
            'oov_words_found': 0,
            'corrections_applied': 0,
            'dictionary_verifications': 0,
            'llm_calls_made': 0,
            'processing_time': 0.0,
            'hunspell_pool_hits': 0
        }
    
    @staticmethod 
    @lru_cache(maxsize=1)  
    def get_nlp():  
        try:
            vocab = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
            nlp = spacy.load(vocab)
            return nlp
        except OSError:
            raise RuntimeError("SpaCy model not found. Please install it with: python -m spacy download")
   
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
    @lru_cache(maxsize=10000)  # PRESERVED: Same caching as v1
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
    
    async def check_word_with_pool(self, word: str) -> List[str]:
        """V2 IMPROVEMENT: Use connection pool for word checking"""
        async with self.hunspell_pool.get_session() as session:
            self.stats['hunspell_pool_hits'] += 1
            output = await session.check_word(word)
            
            if not output:
                return []
            
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
        """V2 IMPROVEMENT: Verify LLM corrections against dictionary"""
        result = await self.check_word_with_pool(word)
        self.stats['dictionary_verifications'] += 1
        return bool(result and result[0] == word)
    
    def calculate_correction_confidence(self, original: str, correction: str, 
                                      is_hunspell_suggestion: bool, dictionary_verified: bool) -> CorrectionConfidence:
        """V2 IMPROVEMENT: Confidence scoring for corrections"""
        levenshtein_dist = self.cached_levenshtein_distance(original, correction)
        
        # Simple context score based on length and character similarity
        context_score = 1.0 / (1.0 + levenshtein_dist * 0.1)
        
        # Calculate total confidence score
        total_score = 0.0
        if dictionary_verified:
            total_score += 0.4
        if is_hunspell_suggestion:
            total_score += 0.3
        if levenshtein_dist <= 2:
            total_score += 0.2
        total_score += context_score * 0.1
        
        return CorrectionConfidence(
            dictionary_verified=dictionary_verified,
            levenshtein_distance=levenshtein_dist,
            context_score=context_score,
            hunspell_suggestion=is_hunspell_suggestion,
            total_score=min(total_score, 1.0)
        )
    
    async def find_best_split_for_spellcheck_v2(self, oov_word: str) -> Tuple[str, str]:
        """V2 IMPROVED: Better compound word splitting with pool usage"""
        excluded_tags = {"SYM", "PUNCT", "X", "SPACE", "NUM"}

        left_split_attempts = [(oov_word[:i], "left") for i in range(4, len(oov_word) + 1)]
        right_split_attempts = [(oov_word[i:], "right") for i in range(len(oov_word) - 3)]  

        all_splits = left_split_attempts + right_split_attempts
        processed_splits = list(self.get_nlp().pipe([split for split, _ in all_splits], batch_size=self.config.spacy_batch_size))

        valid_splits = [
            (split, tag) for (split, tag), doc in zip(all_splits, processed_splits)
            if len(split) > 2 and all(token.pos_ not in excluded_tags and token.vector_norm > 5 for token in doc)
        ]

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

        # V2 IMPROVEMENT: Use connection pool instead of creating new processes
        hunspell_tasks = [self.check_word_with_pool(candidate) for candidate in batch_candidates]
        hunspell_results = await asyncio.gather(*hunspell_tasks)

        normalized_hunspell_results = {
            candidate: result if isinstance(result, list) else [result]
            for candidate, result in zip(batch_candidates, hunspell_results)
        }

        all_suggestions = [
            suggestion
            for suggestions in normalized_hunspell_results.values()
            for suggestion in suggestions
        ]

        if not all(isinstance(s, str) for s in all_suggestions):
            raise TypeError("all_suggestions contains non-string values.")

        processed_suggestions = list(self.get_nlp().pipe(all_suggestions, batch_size=self.config.spacy_batch_size))

        filtered_suggestions = {
            candidate: [suggestion for suggestion, doc in zip(normalized_hunspell_results[candidate], processed_suggestions) if doc.vector_norm > 5]
            for candidate in batch_candidates
        }

        if left_part:
            right_remaining = oov_word[len(left_part):]
            right_part_suggestions = filtered_suggestions.get(right_remaining, [])
            right_part = (
                min(right_part_suggestions, key=lambda s: self.cached_levenshtein_distance(right_remaining, s))
                if right_part_suggestions else right_part
            )

        if right_part:
            left_remaining = oov_word[:-len(right_part)]
            left_part_suggestions = filtered_suggestions.get(left_remaining, [])
            left_part = (
                min(left_part_suggestions, key=lambda s: self.cached_levenshtein_distance(left_remaining, s))
                if left_part_suggestions else left_part
            )

        return left_part, right_part
    
    async def find_best_suggestions_batch_async_v2(self, oov_words: List[str]) -> Dict[str, List[Any]]:
        """V2 IMPROVED: Parallel processing with better error handling"""
        # Sort oov_words to ensure consistent processing order for more stable LLM outcomes 
        sorted_oov_words = sorted(oov_words)

        async def process_word(word):
            try:
                # V2 IMPROVEMENT: Use connection pool
                unsplit_suggestions = await self.check_word_with_pool(word)
                left_part, right_part = await self.find_best_split_for_spellcheck_v2(word)
                split_suggestion = f"{left_part} {right_part}" if (left_part and right_part) else None
                unsplit_suggestion = (
                    min(unsplit_suggestions, key=lambda s: self.cached_levenshtein_distance(word, s))
                    if unsplit_suggestions else None
                )
                return word, unsplit_suggestion, split_suggestion
            except Exception as e:
                logger.error(f"Error processing word '{word}': {e}")
                return word, None, None
       
        # V2 IMPROVEMENT: Concurrent processing with semaphore for rate limiting
        semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        
        async def process_with_semaphore(word):
            async with semaphore:
                return await process_word(word)
        
        results = await asyncio.gather(*[process_with_semaphore(word) for word in sorted_oov_words], return_exceptions=True)

        best_suggestions = defaultdict(list)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Word processing exception: {result}")
                continue
            if result and len(result) == 3:
                best_suggestions[result[0]].append(result[1:])

        return best_suggestions
    
    def create_correction_batches_v2(self, tasks: List[Dict[str, Any]], prompt_header: str, max_tokens: int, completion_reserve: int) -> List[SpellCorrectionBatch]:
        """V2 IMPROVED: Better token calculation and larger batch sizes"""
        # V2 FIX: Use spell_check_model for token counting
        tiktoken_model = self.config.spell_check_model if hasattr(self.config, 'spell_check_model') else self.openai_model
        try:
            encoding = tiktoken.encoding_for_model(tiktoken_model)
        except KeyError:
            # Fallback to cl100k_base if model not found
            encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Using cl100k_base encoding as fallback for {tiktoken_model}")
        
        token_budget = max_tokens - len(encoding.encode(prompt_header)) - completion_reserve
        
        batches = []
        current_batch_tasks = []
        current_batch_tokens = 0
        
        # V2 IMPROVEMENT: Increase max batch size for better efficiency
        max_batch_size = min(self.config.max_batch_size * 2, 10)  # Double the batch size but cap at 10
        
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
    
    async def get_best_corrections_with_ai_v2(self, responses, best_suggestions_dict: Dict[str, List[Any]], var_lab: str) -> Dict[str, str]:
        """V2 MAJOR IMPROVEMENT: Native async OpenAI client with validation loop"""
        oov_words = list(best_suggestions_dict.keys())
        
        max_tokens = self.config.max_tokens  
        completion_reserve = self.config.completion_reserve
        
        corrected_sentences_dict = {}
        tasks = []
        prompt_header = SPELLCHECK_INSTRUCTIONS.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            tasks=""
        )   
        
        responses_with_ids = [{'respondent_id': response.respondent_id, 'response': response.original_response} for response in responses]
    
        # Create tasks for sentences with OOV words (PRESERVED logic from v1)
        for item in responses_with_ids:
            response = item['response']
            response_oov_words = []
            for word in oov_words:
                if len(word) > 2:
                    pattern = rf'\b{re.escape(word)}\b'
                    if re.search(pattern, response):
                        response_oov_words.append(word)
              
            if response_oov_words:
                # Create placeholder version of response
                response_with_placeholders = response
                for word in response_oov_words:
                    pattern = rf'\b{re.escape(word)}\b'
                    response_with_placeholders = re.sub(pattern, '<oov_word>', response_with_placeholders, count=1)
                
                # Get suggestions for all OOV words
                all_suggestions = []
                for word in response_oov_words:
                    suggestions = best_suggestions_dict.get(word, ["OOV"])
                    # Clean up suggestion format
                    cleaned_suggestions = []
                    for sug in suggestions:
                        if isinstance(sug, tuple):
                            cleaned_suggestions.extend([s for s in sug if s and s != "OOV"])
                        else:
                            cleaned_suggestions.append(sug)
                    all_suggestions.append(", ".join(cleaned_suggestions))
                
                tasks.append({
                    "respondent_id": item['respondent_id'],
                    "response": response,
                    "response_with_placeholders": response_with_placeholders,
                    "oov_words": ", ".join(response_oov_words),
                    "suggestions": " | ".join(all_suggestions)
                })
         
        # PRESERVED: Same filtering logic as v1
        repeated_char_pattern = re.compile(rf'^(.)\1{{{self.config.repeated_char_threshold-1},}}$')
        single_word_pattern = re.compile(r'^[A-Za-z]+$')
        filtered_tasks = [
            task for task in tasks
            if not (
                repeated_char_pattern.match(task['response']) or
                repeated_char_pattern.match(task['oov_words']) or
                (single_word_pattern.fullmatch(task['response']) and 'OOV' in task['suggestions'])
            )
        ]
        
        # V2 IMPROVEMENT: Better batching
        batches = self.create_correction_batches_v2(filtered_tasks, prompt_header, max_tokens, completion_reserve)
        
        async def process_batch_v2(batch: SpellCorrectionBatch, var_lab: str, batch_index: int) -> Dict[str, str]:
            """V2 MAJOR IMPROVEMENT: Native async client with validation"""
            tasks_string = ""
            for i, task in enumerate(batch.tasks):
                tasks_string += (
                    f"Task {i + 1}:\n"
                    f"Respondent ID: {task.respondent_id}\n"
                    f"Response: \"{task.response_with_oov_placeholders}\"\n"
                    f"Misspelled words: {task.oov_words}\n"
                    f"Suggested corrections: {task.suggestions}\n\n"
                )
            
            prompt = SPELLCHECK_INSTRUCTIONS.format(
                language=DEFAULT_LANGUAGE,
                var_lab=var_lab,
                tasks=tasks_string
            )
            
            # Capture prompt only for the first batch
            if self.prompt_printer and batch_index == 0:
                self.prompt_printer.capture_prompt(
                    step_name="preprocessing_v2",
                    utility_name="SpellCheckerV2",
                    prompt_content=prompt,
                    prompt_type="correction",
                    metadata={
                        "model": self.openai_model,
                        "var_lab": var_lab,
                        "language": DEFAULT_LANGUAGE,
                        "batch_size": len(batch.tasks),
                        "total_batches": len(batches),
                        "batch_number": batch_index + 1,
                        "version": "v2_native_async"
                    }
                )
            
            # V2 MAJOR IMPROVEMENT: Native async OpenAI client
            try:
                self.stats['llm_calls_made'] += 1
                response = await self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=completion_reserve,
                    temperature=self.config.temperature,
                    seed=self.config.seed
                )
                
                # Parse the JSON response manually (since we're not using instructor)
                response_content = response.choices[0].message.content
                
                try:
                    parsed_response = json.loads(response_content)
                    corrections = parsed_response.get('corrections', [])
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {response_content}")
                    corrections = []
                
            except Exception as e:
                logger.error(f"LLM call failed for batch {batch_index}: {e}")
                corrections = []
            
            # V2 IMPROVEMENT: Validation loop for corrections
            validated_corrections = {}
            for task in batch.tasks:
                # Find corresponding correction
                correction_text = task.original_response  # Default fallback
                
                for corr in corrections:
                    if str(corr.get('respondent_id', '')) == str(task.respondent_id):
                        candidate_correction = corr.get('corrected_response', task.original_response)
                        
                        # V2 IMPROVEMENT: Validate correction quality
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
                        break
                
                validated_corrections[task.original_response] = correction_text
            
            return validated_corrections
        
        # Sort batches to ensure consistent processing order (PRESERVED from v1)
        sorted_batches = sorted(batches, key=lambda b: str(b.tasks[0].respondent_id) if b.tasks else "")

        # V2 IMPROVEMENT: Process batches with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent batch processing
        
        async def process_batch_with_semaphore(batch, var_lab, i):
            async with semaphore:
                return await process_batch_v2(batch, var_lab, i)
        
        batch_results = await asyncio.gather(*[
            process_batch_with_semaphore(batch, var_lab, i) 
            for i, batch in enumerate(sorted_batches)
        ])
        
        # Combine results
        for result in batch_results:
            corrected_sentences_dict.update(result)
        
        return corrected_sentences_dict
    
    async def spell_check_async_v2(self, responses: List[SpellCheckModel], var_lab: str) -> List[SpellCheckModel]:
        """V2 MAIN METHOD: Enhanced spell checking with improved performance and accuracy"""
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(responses)
        
        self.verbose_reporter.step_start("Spell Checking V2 (Async + Connection Pool)")
        sentences_list = [response.original_response for response in responses]
    
        # Step 1: Identify OOV words with connection pool
        self.verbose_reporter.stat_line(f"Analyzing {len(responses)} responses for misspellings (V2)...")
        oov_words = []
        docs_with_oov = 0
        
        # V2 IMPROVEMENT: Use connection pool instead of single session
        await self.hunspell_pool.initialize()
        
        try:
            for doc in self.get_nlp().pipe(sentences_list, batch_size=self.config.spacy_batch_size):
                doc_flagged = False
                word_tasks = []
                doc_words = []
                
                # Collect words for batch processing
                for token in doc:
                    if token.is_alpha and token.ent_type_ == "" and len(token.text) > 2:
                        word = token.text
                        doc_words.append(word)
                        word_tasks.append(self.check_word_with_pool(word))
                
                # Process all words in this document concurrently
                if word_tasks:
                    word_results = await asyncio.gather(*word_tasks)
                    
                    for word, result in zip(doc_words, word_results):
                        self.stats['words_checked'] += 1
                        # Check if word is OOV (same logic as v1)
                        if result and not (result and result[0] == word):
                            # Check the raw output format
                            if isinstance(result, list) and len(result) > 0:
                                first_result = str(result[0]) if result[0] else ""
                                if first_result.startswith(('&', '#')) or not result:
                                    oov_words.append(word)
                                    doc_flagged = True
                                    self.stats['oov_words_found'] += 1
             
                if doc_flagged:
                    docs_with_oov += 1
                    
        except Exception as e:
            logger.error(f"Error during OOV identification: {e}")
        
        unique_oov_words = len(set(oov_words))
        self.verbose_reporter.stat_line(f"OOV words identified: {unique_oov_words} unique terms")
        self.verbose_reporter.stat_line(f"Responses requiring correction: {docs_with_oov}")
    
        # Step 2: Correct OOV words with V2 improvements
        if oov_words:
            best_suggestions_dict = await self.find_best_suggestions_batch_async_v2(oov_words)
            corrected_sentences_dict = await self.get_best_corrections_with_ai_v2(responses, best_suggestions_dict, var_lab)
            corrected_sentences_dict = {k: v for k, v in corrected_sentences_dict.items() if v != '[NO RESPONSE]'}
        else:
            corrected_sentences_dict = {}
        
        # Step 3: Update sentences with tracked respondent IDs (PRESERVED logic from v1)
        corrections_made = 0
        correction_examples = []
        updated_responses = []
        
        for response in responses:
            corrected_response = corrected_sentences_dict.get(response.original_response, response.original_response)
            updated_response = SpellCheckModel(
                respondent_id=response.respondent_id,
                original_response=response.original_response,
                corrected_response=corrected_response
            )
            updated_responses.append(updated_response)
           
            # Track corrections for verbose output (PRESERVED from v1)
            if response.original_response != corrected_response:
                original_normalized = ' '.join([word.lower().strip('.,!?;:"\'()[]{}') for word in response.original_response.split()])
                corrected_normalized = ' '.join([word.lower().strip('.,!?;:"\'()[]{}') for word in corrected_response.split()])
                
                if original_normalized != corrected_normalized:
                    corrections_made += 1
                    
                    # Store example for verbose output
                    if len(correction_examples) < self.config.max_correction_examples:
                        correction_examples.append((response.original_response, corrected_response))

        # Clean up connection pool
        await self.hunspell_pool.close_all()
        
        stats.end_timing()
        stats.output_count = len(updated_responses)
        self.stats['processing_time'] = stats.processing_time
        
        # V2 IMPROVEMENT: Enhanced reporting
        self.verbose_reporter.stat_line(f"Corrections applied: {corrections_made} changes")
        self.verbose_reporter.stat_line(f"Dictionary verifications: {self.stats['dictionary_verifications']}")
        self.verbose_reporter.stat_line(f"Hunspell pool hits: {self.stats['hunspell_pool_hits']}")
        self.verbose_reporter.stat_line(f"LLM calls made: {self.stats['llm_calls_made']}")
        
        # Show correction examples in verbose mode
        if correction_examples:
            self.verbose_reporter.correction_samples(correction_examples)
        
        self.verbose_reporter.step_complete("Spell checking V2 completed")

        processed_responses = [models.PreprocessedModel(respondent_id=item.respondent_id, response=item.corrected_response) for item in updated_responses]
        
        return processed_responses
    
    def spell_check(self, preprocess_responses: List[Dict], var_lab: str):
        """V2 IMPROVEMENT: Enhanced synchronous wrapper with better error handling"""
        async def main():
            spellcheck_responses = [SpellCheckModel(
                respondent_id=item.respondent_id, 
                original_response=item.response
            ) for item in preprocess_responses]
            
            try:
                return await self.spell_check_async_v2(spellcheck_responses, var_lab)
            except Exception as e:
                logger.error(f"SpellChecker V2 processing failed: {e}")
                # Fallback: return original responses
                return [models.PreprocessedModel(respondent_id=item.respondent_id, response=item.response) 
                       for item in preprocess_responses]
        
        nest_asyncio.apply()
        return asyncio.run(main())
    
    def __del__(self):
        """Cleanup method to ensure resources are released"""
        try:
            # Attempt to clean up the connection pool
            if hasattr(self, 'hunspell_pool'):
                # Note: This won't work in destructor for async cleanup
                # but it's here for documentation purposes
                pass
        except Exception:
            pass