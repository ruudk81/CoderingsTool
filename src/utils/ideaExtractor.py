import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import functools
import nest_asyncio
from typing import Dict, List, Optional, Union
import instructor
from openai import OpenAI
import tiktoken

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG
from prompts import IDEA_EXTRACTION_PROMPT
import models
from .verboseReporter import VerboseReporter, ProcessingStats

client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class IdeaExtractor:
    def __init__(
        self,
        responses: List[models.QualityFilteredModel],
        var_lab: str,
        config: Optional[SegmentationConfig] = None,
        verbose: bool = False,
        prompt_printer=None):
        
        self.responses = responses
        self.var_lab = var_lab
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.client = client
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.IdeaModel] = []
        self.verbose_reporter = VerboseReporter(verbose)
        self._stats = ProcessingStats()
        self.model_config = ModelConfig()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        
        # Initialize tokenizer for batch size calculation
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            print(f"Using cl100k_base encoding as fallback for {self.config.model}")

    def _calculate_token_budget(self) -> int:
        """Calculate available tokens for responses after accounting for prompt"""
        base_prompt = IDEA_EXTRACTION_PROMPT.format(
            var_lab=self.var_lab,
            language=self.language,
            respondent_id="",
            response=""
        )
        prompt_tokens = len(self.encoding.encode(base_prompt))
        return self.config.max_tokens - prompt_tokens - self.config.completion_reserve

    def _batch(self) -> List[List[tuple]]:
        """Create token-aware batches of responses"""
        token_budget = self._calculate_token_budget()
        
        if not self.responses:
            return []
        
        # Calculate adaptive batch size based on average response length
        avg_tokens = sum(len(self.encoding.encode(r.response)) for r in self.responses) / max(1, len(self.responses))
        adaptive_max_batch = min(self.config.max_batch_size, max(1, int(token_budget / max(1, avg_tokens))))
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for i, response in enumerate(self.responses):
            response_tokens = len(self.encoding.encode(response.response))
            
            # Handle oversized responses
            if response_tokens > token_budget and not current_batch:
                print(f"Warning: Response from {response.respondent_id} exceeds token budget ({response_tokens} > {token_budget})")
                batches.append([(i, response.respondent_id, response.response)])
                continue
            
            # Check if adding this response would exceed limits
            if (current_tokens + response_tokens > token_budget or 
                len(current_batch) >= adaptive_max_batch):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append((i, response.respondent_id, response.response))
            current_tokens += response_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _build_prompt(self, batch: List[tuple]) -> str:
        """Build prompt for a batch of responses"""
        # For batches, we need to format multiple responses
        responses_text = []
        for _, respondent_id, response in batch:
            responses_text.append(f"Respondent ID: {respondent_id}\nResponse: \"{response}\"")
        
        # If single response, use the original prompt format
        if len(batch) == 1:
            _, respondent_id, response = batch[0]
            return IDEA_EXTRACTION_PROMPT.format(
                var_lab=self.var_lab,
                language=self.language,
                respondent_id=respondent_id,
                response=response
            )
        
        # For multiple responses, we need a batch-friendly prompt
        batch_prompt = f"""Extract the main ideas from the following survey responses about "{self.var_lab}".

Language: {self.language}

For each response, identify and extract the distinct ideas or themes mentioned. Return a JSON array where each element represents a response and contains an array of ideas extracted from that response.

Responses:
{chr(10).join(responses_text)}

Return format:
[
  {{"respondent_id": "id1", "ideas": [{{"idea": "first idea"}}, {{"idea": "second idea"}}]}},
  {{"respondent_id": "id2", "ideas": [{{"idea": "idea text"}}]}},
  ...
]"""
        return batch_prompt

    async def _call_openai_api(self, prompt: str, batch_size: int) -> Union[List[Dict], Dict]:
        """Call OpenAI API with structured output"""
        tries = 0
        max_tries = self.config.max_retries
        
        while tries < max_tries:
            tries += 1
            try:
                loop = asyncio.get_running_loop()
                
                # Define response model based on batch size
                if batch_size == 1:
                    # Single response: expect array of ideas
                    response_model = List[Dict[str, str]]
                else:
                    # Multiple responses: expect array of responses with ideas
                    response_model = List[Dict[str, Union[str, List[Dict[str, str]]]]]
                
                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self.client.chat.completions.create,
                        model=self.config.model,
                        response_model=response_model,
                        max_retries=self.config.instructor_retries,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        seed=self.model_config.seed
                    )
                )
                return response
                
            except Exception as e:
                print(f"\nAPI call failed on attempt {tries}/{max_tries}:")
                print(f"Error: {str(e)}")
                
                if tries >= max_tries:
                    raise
                
                # Exponential backoff with delay from config
                await asyncio.sleep(self.config.retry_delay * tries)
                continue

    async def _extract_ideas_batch(self, batch: List[tuple], batch_index: int) -> List[models.IdeaModel]:
        """Process a batch and extract ideas"""
        prompt = self._build_prompt(batch)
        
        # Capture prompt only for the first batch
        if self.prompt_printer and not self._captured_prompt:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="idea_extraction",
                metadata={
                    "model": self.config.model,
                    "var_lab": self.var_lab,
                    "language": self.language,
                    "batch_size": len(batch),
                    "batch_number": batch_index + 1
                }
            )
            self._captured_prompt = True
        
        try:
            response_data = await self._call_openai_api(prompt, len(batch))
            
            # Process response based on batch size
            results = []
            if len(batch) == 1:
                # Single response format: just array of ideas
                idx, respondent_id, response_text = batch[0]
                ideas = []
                for i, idea_dict in enumerate(response_data):
                    idea_text = idea_dict.get('idea', '')
                    if idea_text:
                        ideas.append(models.IdeaSubmodel(
                            idea_id=f"{respondent_id}_{i+1}",
                            idea=idea_text
                        ))
                
                results.append(models.IdeaModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=self.responses[idx].quality_filter,
                    quality_filter_code=self.responses[idx].quality_filter_code,
                    response_ideas=ideas,
                    idea_count=len(ideas)
                ))
            else:
                # Multiple responses format
                batch_dict = {resp_id: (idx, resp_text) for idx, resp_id, resp_text in batch}
                
                for resp_data in response_data:
                    resp_id = resp_data.get('respondent_id')
                    if resp_id in batch_dict:
                        idx, response_text = batch_dict[resp_id]
                        ideas = []
                        
                        idea_list = resp_data.get('ideas', [])
                        for i, idea_dict in enumerate(idea_list):
                            idea_text = idea_dict.get('idea', '')
                            if idea_text:
                                ideas.append(models.IdeaSubmodel(
                                    idea_id=f"{resp_id}_{i+1}",
                                    idea=idea_text
                                ))
                        
                        results.append(models.IdeaModel(
                            respondent_id=resp_id,
                            response=response_text,
                            quality_filter=self.responses[idx].quality_filter,
                            quality_filter_code=self.responses[idx].quality_filter_code,
                            response_ideas=ideas,
                            idea_count=len(ideas)
                        ))
            
            return results
            
        except Exception as e:
            print(f"Batch {batch_index + 1} processing failed: {str(e)}")
            # Return error results for this batch
            results = []
            for idx, respondent_id, response_text in batch:
                results.append(models.IdeaModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=self.responses[idx].quality_filter,
                    quality_filter_code=self.responses[idx].quality_filter_code,
                    response_ideas=[
                        models.IdeaSubmodel(
                            idea_id=f"{respondent_id}_1",
                            idea="PROCESSING_ERROR"
                        )
                    ],
                    idea_count=1
                ))
            return results

    async def _process_all_batches(self):
        """Process all batches concurrently"""
        batches = self._batch()
        self.verbose_reporter.stat_line(f"Processing {len(self.responses)} responses in {len(batches)} batches...")
        
        tasks = [self._extract_ideas_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_failures = 0
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i+1} processing failed after all retries: {str(batch_result)}")
                total_failures += 1
                continue
            
            self._results.extend(batch_result)
        
        if total_failures > 0:
            print(f"{total_failures} out of {len(batches)} batches failed completely")

    def extract(self) -> List[models.IdeaModel]:
        """Main method to extract ideas from responses"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)
        
        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")
        self.verbose_reporter.stat_line(f"Processing {len(self.responses)} responses...")
        
        if not self.responses:
            self.verbose_reporter.stat_line("No responses to process")
            return []
        
        nest_asyncio.apply()
        asyncio.run(self._process_all_batches())
        
        # Ensure all responses are accounted for
        result_ids = {r.respondent_id for r in self._results}
        for response in self.responses:
            if response.respondent_id not in result_ids:
                # Add missing responses with error marker
                self._results.append(models.IdeaModel(
                    respondent_id=response.respondent_id,
                    response=response.response,
                    quality_filter=response.quality_filter,
                    quality_filter_code=response.quality_filter_code,
                    response_ideas=[
                        models.IdeaSubmodel(
                            idea_id=f"{response.respondent_id}_1",
                            idea="NOT_PROCESSED"
                        )
                    ],
                    idea_count=1
                ))
        
        self._stats.output_count = len(self._results)
        self._stats.end_timing()
        
        # Calculate statistics
        idea_examples = []
        unique_ideas = set()
        multi_idea_responses = 0
        total_idea_length = 0
        idea_count = 0
        
        for resp in self._results:
            if resp.response_ideas and len(resp.response_ideas) > 0:
                if len(resp.response_ideas) > 1:
                    multi_idea_responses += 1
                
                for idea in resp.response_ideas:
                    if idea.idea and idea.idea not in ["NA", "PROCESSING_ERROR", "NOT_PROCESSED"]:
                        unique_ideas.add(idea.idea)
                        idea_words = idea.idea.split()
                        total_idea_length += len(idea_words)
                        idea_count += 1
                        
                        # Collect examples
                        if len(idea_examples) < self.config.max_code_examples:
                            idea_examples.append(f'"{resp.response}" → "{idea.idea}"')
        
        avg_idea_length = total_idea_length / idea_count if idea_count > 0 else 0
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Total responses: {len(self._results)}")
        self.verbose_reporter.stat_line(f"Unique ideas identified: {len(unique_ideas)}")
        self.verbose_reporter.stat_line(f"Average idea length: {avg_idea_length:.1f} words")
        if multi_idea_responses > 0:
            self.verbose_reporter.stat_line(f"Responses with multiple ideas: {multi_idea_responses}")
        
        # Show idea examples
        if idea_examples:
            self.verbose_reporter.sample_list("Sample extracted ideas", idea_examples)
        
        self.verbose_reporter.step_complete("Idea extraction completed")
        
        return self._results

    def summary(self) -> Dict[str, Union[int, float]]:
        """Generate summary statistics"""
        total = len(self._results)
        processed = sum(1 for r in self._results 
                       if r.response_ideas and 
                       not any(idea.idea in ["PROCESSING_ERROR", "NOT_PROCESSED"] 
                              for idea in r.response_ideas))
        failed = total - processed
        
        total_ideas = sum(r.idea_count for r in self._results)
        unique_ideas = len(set(idea.idea for r in self._results 
                              for idea in r.response_ideas 
                              if idea.idea not in ["NA", "PROCESSING_ERROR", "NOT_PROCESSED"]))
        
        return {
            "total_responses": total,
            "processed_responses": processed,
            "failed_responses": failed,
            "success_rate": round((processed / total) * 100, 2) if total > 0 else 0,
            "total_ideas": total_ideas,
            "unique_ideas": unique_ideas,
            "avg_ideas_per_response": round(total_ideas / total, 2) if total > 0 else 0
        }