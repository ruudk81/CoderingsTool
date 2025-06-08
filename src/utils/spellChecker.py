import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import re
import asyncio
import nest_asyncio
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import instructor
import tiktoken
import spacy
import subprocess
from collections import defaultdict

from config import (DEFAULT_MODEL, OPENAI_API_KEY, DEFAULT_LANGUAGE, HUNSPELL_PATH, 
                    DUTCH_DICT_PATH, ENGLISH_DICT_PATH, SpellCheckConfig, DEFAULT_SPELLCHECK_CONFIG)
from prompts import SPELLCHECK_INSTRUCTIONS
import models
from .verboseReporter import VerboseReporter, ProcessingStats

# Nederlands of Engels
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
    def __init__(self, config: SpellCheckConfig = None, openai_api_key: Optional[str] = None, 
                 openai_model: str = None, verbose: bool = False, prompt_printer = None):
        self.config = config or DEFAULT_SPELLCHECK_CONFIG
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.openai_model = openai_model or DEFAULT_MODEL
        self.client = instructor.patch(OpenAI(api_key=self.openai_api_key))
        self.hunspell_path = HUNSPELL_PATH
        self.dict_path = DICT_PATH
        self.prompt_printer = prompt_printer 
        self.verbose_reporter = VerboseReporter(verbose)
        if not self.check_hunspell_installation():
            print("Hunspell is not properly installed or configured.")
    
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
            print(f"Hunspell not found at {HUNSPELL_PATH}. Please check the path.")
            return False
        except Exception as e:
            print(f"Error checking Hunspell installation: {str(e)}")
            return False
    
    @staticmethod
    @lru_cache(maxsize=10000)  # TODO: Use config.cache_size
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
        
        def run_hunspell(): # because of a sync batchting seperate hunspell sessions
            process = subprocess.Popen(
                [HUNSPELL_PATH, "-a", "-d", DUTCH_DICT_PATH],
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

         
        lines = [line for line in output.splitlines() if line and not line.startswith("@")]
        if lines and lines[0].startswith("&"):
            match = re.search(r": (.+)", lines[0])
            if match:
                suggestions = match.group(1).split(", ")
                return suggestions
        if lines and lines[0].startswith("*"):
            return [word]  # Word is correct
        return []
          
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
    
    async def find_best_suggestions_batch_async(self, oov_words: List[str]) -> Dict[str, List[Any]]:

       # Sort oov_words to ensure consistent processing order for more stable LLM outcomes 
       sorted_oov_words = sorted(oov_words)

       async def process_word(word):
           unsplit_suggestions = await self.run_hunspell_word_async(word)
           left_part, right_part = await self.find_best_split_for_spellcheck(word)
           split_suggestion = f"{left_part} {right_part}" if (left_part and right_part) else None
           unsplit_suggestion = (
               min(unsplit_suggestions, key=lambda s: self.cached_levenshtein_distance(word, s))
               if unsplit_suggestions else None)
           return word, unsplit_suggestion, split_suggestion
       
       results = await asyncio.gather(*(process_word(word) for word in sorted_oov_words))

       best_suggestions = defaultdict(list)
       for result in results:
           best_suggestions[result[0]].append(result[1:])

       return best_suggestions
        
    
    def create_correction_batches(self, tasks: List[Dict[str, Any]], prompt_header: str, max_tokens: int, completion_reserve: int) -> List[SpellCorrectionBatch]:
        encoding = tiktoken.encoding_for_model(self.openai_model)
        token_budget = max_tokens - len(encoding.encode(prompt_header)) - completion_reserve
        
        batches = []
        current_batch_tasks = []
        current_batch_tokens = 0
        
        max_batch_size = self.config.max_batch_size
        
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
    
    async def get_best_corrections_with_ai(self, responses, best_suggestions_dict: Dict[str, List[Any]], var_lab: str) -> Dict[str, str]:
        oov_words = list(best_suggestions_dict.keys())
        
        max_tokens = self.config.max_tokens  
        completion_reserve = self.config.completion_reserve  # Reserve for completion
        
        corrected_sentences_dict = {}
        tasks = []
        prompt_header = SPELLCHECK_INSTRUCTIONS.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            tasks="")   
        
        responses_with_ids = [{'respondent_id': response.respondent_id, 'response': response.original_response} for response in responses]
    
        # Create tasks for sentences with OOV words
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
                    "suggestions": " | ".join(all_suggestions)  # Separate suggestions for each word
                    })
         
        repeated_char_pattern = re.compile(rf'^(.)\1{{{self.config.repeated_char_threshold-1},}}$')  # repeated N+ times
        single_word_pattern = re.compile(r'^[A-Za-z]+$')
        filtered_tasks = [
            task for task in tasks
            if not (
                repeated_char_pattern.match(task['response']) or
                repeated_char_pattern.match(task['oov_words']) or
                (single_word_pattern.fullmatch(task['response']) and 'OOV' in task['suggestions'])) ]
        
        # Create batches
        batches = self.create_correction_batches(filtered_tasks, prompt_header, max_tokens, completion_reserve)
        
        async def process_batch(batch: SpellCorrectionBatch, var_lab: str, batch_index: int) -> Dict[str, str]:
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
            
            # Capture prompt only for the first batch (or only batch if there's just one)
            if self.prompt_printer and batch_index == 0:
                self.prompt_printer.capture_prompt(
                    step_name="preprocessing",
                    utility_name="SpellChecker",
                    prompt_content=prompt,
                    prompt_type="correction",
                    metadata={
                        "model": self.openai_model,
                        "var_lab": var_lab,
                        "language": DEFAULT_LANGUAGE,
                        "batch_size": len(batch.tasks),
                        "total_batches": len(batches),
                        "batch_number": batch_index + 1
                    }
                )
            
            # Using instructor for structured output
            # with random seed, for more determined outcomes
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.openai_model,
                response_model=LLMCorrectionResponse,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=completion_reserve,
                max_retries=self.config.retries, 
                seed = self.config.seed)
            
            # Sorting corrections, to prevent variation by making processing consistent
            sorted_corrections = sorted(response.corrections, key=lambda x: str(x.respondent_id))
    
            corrections_by_id = {
                task.original_response: next((corr.corrected_response for corr in sorted_corrections if str(corr.respondent_id) == str(task.respondent_id)),
                task.original_response)  # Fall back to original if no match
                for task in batch.tasks}
            
            return corrections_by_id
        
        # Sort batches to ensure consistent processing order for stable llm results
        sorted_batches = sorted(batches, key=lambda b: str(b.tasks[0].respondent_id) if b.tasks else "")

        batch_results = await asyncio.gather(*(process_batch(batch, var_lab, i) for i, batch in enumerate(sorted_batches)))
        
        # Combine results
        for result in batch_results:
            corrected_sentences_dict.update(result)
        
        return corrected_sentences_dict    
    
    async def spell_check_async(self, responses: List[SpellCheckModel], var_lab: str) -> List[SpellCheckModel]:
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(responses)
        
        self.verbose_reporter.step_start("Spell Checking")
        sentences_list = [response.original_response for response in responses]
    
        # Step 1: Identify OOV words  
        self.verbose_reporter.stat_line(f"Analyzing {len(responses)} responses for misspellings...")
        oov_words = []
        docs_with_oov = 0
        
        # Single hunspell session to speed up process
        oov_identification_session = HunspellSession(self.hunspell_path, self.dict_path)
        
        try:
            for doc in self.get_nlp().pipe(sentences_list, batch_size=self.config.spacy_batch_size): #TODO process zonder spaCy
                doc_flagged = False
                for token in doc:
                    if token.is_alpha and token.ent_type_ == "" and len(token.text) > 2:
                        word = token.text
                        output = oov_identification_session.check_word(word)
                        if output and output.startswith(('&', '#')):
                            oov_words.append(word)
                            doc_flagged = True
             
                if doc_flagged:
                    docs_with_oov += 1
        finally:
            oov_identification_session.close()
            
        unique_oov_words = len(set(oov_words))
        self.verbose_reporter.stat_line(f"OOV words identified: {unique_oov_words} unique terms")
        self.verbose_reporter.stat_line(f"Responses requiring correction: {docs_with_oov}")
    
        # Step 2: Correct OOV words 
        if oov_words:
            best_suggestions_dict = await self.find_best_suggestions_batch_async(oov_words)
            corrected_sentences_dict = await self.get_best_corrections_with_ai(responses, best_suggestions_dict, var_lab)
            corrected_sentences_dict = {k: v for k, v in corrected_sentences_dict.items() if v != '[NO RESPONSE]'}
        else:
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
                    if len(correction_examples) < self.config.max_correction_examples:  # Collect examples for display
                        correction_examples.append((response.original_response, corrected_response))
                    
                    # Only show detailed output in non-verbose mode for debugging
                    if not self.verbose_reporter.enabled:
                        highlighted_original = []
                        for word in response.original_response.split():
                            clean_word = word.strip('.,!?;:"\'()[]{}')
                            if clean_word in oov_words:
                                highlighted_original.append(f"<{word}>")
                            else:
                                highlighted_original.append(word)
                       
                        original_words = response.original_response.split()
                        corrected_words = corrected_response.split()
                        
                        original_set = set(w.lower().strip('.,!?;:"\'()[]{}') for w in original_words)
                        highlighted_corrected = []
                        for word in corrected_words:
                            clean_word = word.lower().strip('.,!?;:"\'()[]{}')
                            if clean_word not in original_set:
                                highlighted_corrected.append(f"<{word}>")
                            else:
                                highlighted_corrected.append(word)
                        
                        #print(f"Original: {' '.join(highlighted_original)}")
                        #print(f"Corrected: {' '.join(highlighted_corrected)}\n")

        stats.end_timing()
        stats.output_count = len(updated_responses)
        
        # Report final statistics
        self.verbose_reporter.stat_line(f"Corrections applied: {corrections_made} changes")
        
        # Show correction examples in verbose mode
        if correction_examples:
            self.verbose_reporter.correction_samples(correction_examples)
        
        self.verbose_reporter.step_complete("Spell checking completed")
        
        #print(f"Total {corrections_made} responses corrected")

        processed_responses = [models.PreprocessModel(respondent_id=item.respondent_id, response=item.corrected_response) for item in updated_responses]
        
        return processed_responses
                  
    def spell_check(self, preprocess_responses: List[Dict], var_lab: str):
      
        async def main():
            spellcheck_responses = [SpellCheckModel(
                    respondent_id=item.respondent_id, 
                    original_response=item.response) 
                    for item in preprocess_responses]
            
            return await self.spell_check_async(spellcheck_responses, var_lab)
        
        nest_asyncio.apply()
        return asyncio.run(main())

# Example usage:
if __name__ == "__main__":
    
    from utils.testData import TestDataLoader
    loader = TestDataLoader()

    responses_dicts  = loader.get_test_data(
       filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
       variable_name = "Q20", 
       n_samples= 100,
       as_response_items=True)

    responses = [models.PreprocessModel(respondent_id=item['respondent_id'], response = item['sentence']) for item in responses_dicts]    
    
    var_label_dict = loader.list_variables(
      filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
      string_only = True)
    
    var_lab = var_label_dict['Q20']
        
    spell_checker = SpellChecker()
    processed_responses = spell_checker.spell_check(responses, var_lab)
    

    
    
    