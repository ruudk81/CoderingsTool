import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import functools
import nest_asyncio
from typing import Dict, List, Optional, Union
import instructor
from openai import OpenAI
import tiktoken
from pydantic import BaseModel

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG
from prompts import IDEA_EXTRACTION_PROMPT
import models
from .verboseReporter import VerboseReporter, ProcessingStats

client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class IdeaResponse(BaseModel):
    """Response model for idea extraction matching the prompt format"""
    idea: str

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

    def _build_prompt(self, respondent_id: str, response: str) -> str:
        """Build prompt for a single response"""
        return IDEA_EXTRACTION_PROMPT.format(
            var_lab=self.var_lab,
            language=self.language,
            respondent_id=respondent_id,
            response=response
        )

    async def _call_openai_api(self, prompt: str) -> List[IdeaResponse]:
        """Call OpenAI API with structured output for single response"""
        tries = 0
        max_tries = self.config.max_retries
        
        while tries < max_tries:
            tries += 1
            try:
                loop = asyncio.get_running_loop()
                
                # Use IdeaResponse model for structured output
                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self.client.chat.completions.create,
                        model=self.config.model,
                        response_model=List[IdeaResponse],
                        max_retries=3,  # Default instructor retries
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

    async def _process_single_response(self, idx: int, respondent_id: str, response_text: str) -> models.IdeaModel:
        """Process a single response and extract ideas"""
        prompt = self._build_prompt(respondent_id, response_text)
        
        # Capture prompt only for the first response
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
                    "respondent_id": respondent_id
                }
            )
            self._captured_prompt = True
        
        try:
            response_data = await self._call_openai_api(prompt)
            
            # Process response - array of IdeaResponse objects
            ideas = []
            for i, idea_response in enumerate(response_data):
                if idea_response.idea:
                    ideas.append(models.IdeaSubmodel(
                        idea_id=f"{respondent_id}_{i+1}",
                        idea=idea_response.idea
                    ))
            
            return models.IdeaModel(
                respondent_id=respondent_id,
                response=response_text,
                quality_filter=self.responses[idx].quality_filter,
                quality_filter_code=self.responses[idx].quality_filter_code,
                response_ideas=ideas,
                idea_count=len(ideas)
            )
            
        except Exception as e:
            print(f"Processing failed for respondent {respondent_id}: {str(e)}")
            # Return error result
            return models.IdeaModel(
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
            )

    async def _process_all_responses(self):
        """Process all responses individually but concurrently"""
        self.verbose_reporter.stat_line(f"Processing {len(self.responses)} responses individually...")
        
        # Create tasks for each response
        tasks = []
        for idx, response in enumerate(self.responses):
            task = self._process_single_response(idx, response.respondent_id, response.response)
            tasks.append(task)
        
        # Process all responses concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_failures = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Response {i+1} processing failed: {str(result)}")
                total_failures += 1
                # Create error result for failed response
                self._results.append(models.IdeaModel(
                    respondent_id=self.responses[i].respondent_id,
                    response=self.responses[i].response,
                    quality_filter=self.responses[i].quality_filter,
                    quality_filter_code=self.responses[i].quality_filter_code,
                    response_ideas=[
                        models.IdeaSubmodel(
                            idea_id=f"{self.responses[i].respondent_id}_1",
                            idea="PROCESSING_ERROR"
                        )
                    ],
                    idea_count=1
                ))
            else:
                self._results.append(result)
        
        if total_failures > 0:
            print(f"{total_failures} out of {len(self.responses)} responses failed")

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
        asyncio.run(self._process_all_responses())
        
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
    
    def generate_codes(self, responses: List[models.QualityFilteredModel], var_lab: str, max_retries: int = 3) -> List[models.IdeaModel]:
        """Compatibility method for v1 interface"""
        # Update instance variables
        self.responses = responses
        self.var_lab = var_lab
        if max_retries != self.config.max_retries:
            self.config.max_retries = max_retries
        
        # Reset results for new run
        self._results = []
        self._captured_prompt = False
        
        # Call the main extract method
        return self.extract()