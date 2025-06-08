import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import functools
import nest_asyncio #for Spyder
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import instructor
from openai import OpenAI

from config import (DEFAULT_MODEL, OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig,
                    QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG)
from prompts import GRADER_INSTRUCTIONS
import models
from .verboseReporter import VerboseReporter, ProcessingStats

# Patch OpenAI client with instructor for structured output
client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY)) 

# GraderConfig is now replaced by QualityFilterConfig from config.py

class Grader:
    def __init__(
        self, 
        responses: List[models.DescriptiveModel], 
        var_lab: str,
        config: Optional[QualityFilterConfig] = None,
        verbose: bool = False,
        prompt_printer = None):
        
        self.responses = responses
        self.question = var_lab
        self.config = config or DEFAULT_QUALITY_FILTER_CONFIG
        self.client = client
        self.grader_instructions = GRADER_INSTRUCTIONS 
        self._results: List[models.DescriptiveModel] = []
        self.verbose_reporter = VerboseReporter(verbose)
        self._stats = ProcessingStats()
        self.model_config = ModelConfig()  # For accessing seed
        self.prompt_printer = prompt_printer 

    def _batch(self) -> List[List[tuple]]:
        indexed = [(i, r.respondent_id, r.response) for i, r in enumerate(self.responses)]
        return [indexed[i:i + self.config.batch_size] for i in range(0, len(indexed), self.config.batch_size)]

    def _build_prompt(self, var_lab: str, batch: List[tuple]) -> str:
        
        responses_text = "\n".join(f"respondent_id: {rid}, response: \"{response}\"" for _, rid, response in batch)
        return self.grader_instructions.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=responses_text)

    async def _call_openai_api(self, prompt: str) -> List[models.DescriptiveModel]:
        tries = 0
        max_tries = self.config.retries
        
        while tries < max_tries:
            tries += 1
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self.client.chat.completions.create,
                        model=self.config.model,
                        response_model=List[models.DescriptiveModel],
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
                
                # Exponential backoff
                await asyncio.sleep(2 ** tries)
                continue

    async def _grade_batch(self, batch: List[tuple], batch_index: int) -> List[models.DescriptiveModel]:
        prompt = self._build_prompt(self.question, batch)
        
        # Capture prompt only for the first batch
        if self.prompt_printer and batch_index == 0:
            self.prompt_printer.capture_prompt(
                step_name="segmentation",
                utility_name="QualityFilter",
                prompt_content=prompt,
                prompt_type="quality_assessment",
                metadata={
                    "model": self.config.model,
                    "var_lab": self.question,
                    "language": DEFAULT_LANGUAGE,
                    "batch_size": len(batch),
                    "batch_number": batch_index + 1
                }
            )
        
        response_data = await self._call_openai_api(prompt)
        return response_data

    async def _process_all_batches(self):
        batches = self._batch()
        self.verbose_reporter.stat_line(f"Processing {len(self.responses)} responses in {len(batches)} batches...")
        
        tasks = [self._grade_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        total_failures = 0
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i+1} processing failed after all retries: {str(batch_result)}")
                total_failures += 1
                continue

            self._results.extend(batch_result)
             
        if total_failures > 0 and not self.verbose_reporter.enabled:
            print(f"{total_failures} out of {len(batches)} batches failed completely after all retries")

    def grade(self) -> List[models.DescriptiveModel]:
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)
        
        self.verbose_reporter.step_start("Quality Assessment")
        
        nest_asyncio.apply()
        asyncio.run(self._process_all_batches())
        
        # Calculate quality statistics
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        filtered_examples = []
        
        for result in self._results:
            if hasattr(result, 'quality_score'):
                if result.quality_score >= self.config.high_quality_threshold:
                    quality_counts["high"] += 1
                elif result.quality_score >= self.config.medium_quality_threshold:
                    quality_counts["medium"] += 1
                else:
                    quality_counts["low"] += 1
            
            if result.quality_filter and len(filtered_examples) < self.config.max_filter_examples:
                filtered_examples.append(f'"{result.response}" (quality filter: meaningless)')
        
        self._stats.output_count = len([r for r in self._results if not r.quality_filter])
        self._stats.end_timing()
        
        # Report statistics
        total = len(self._results)
        filtered_count = sum(1 for r in self._results if r.quality_filter)
        
        if quality_counts["high"] > 0:
            self.verbose_reporter.stat_line(f"High quality: {quality_counts['high']} responses ({quality_counts['high']/total*100:.1f}%)")
        if quality_counts["medium"] > 0:
            self.verbose_reporter.stat_line(f"Medium quality: {quality_counts['medium']} responses ({quality_counts['medium']/total*100:.1f}%)")
        if quality_counts["low"] > 0:
            self.verbose_reporter.stat_line(f"Low quality: {quality_counts['low']} responses ({quality_counts['low']/total*100:.1f}%)")
        
        self.verbose_reporter.stat_line(f"Filtered out: {filtered_count} responses")
        
        # Show filtered examples
        if filtered_examples:
            self.verbose_reporter.sample_list("Sample filtered responses", filtered_examples)
        
        self.verbose_reporter.step_complete("Quality filtering completed")
        
        return self._results

    def filter(self) -> List[models.DescriptiveModel]:
        return [r for r in self._results if not r.quality_filter]

    def summary(self) -> Dict[str, Union[int, float]]:
        total = len(self._results)
        meaningless = sum(1 for r in self._results if r.quality_filter)
        meaningful = total - meaningless

        return {
            "total_responses": total,
            "meaningful_responses": meaningful,
            "meaningless_responses": meaningless,
            "meaningful_percentage": round((meaningful / total) * 100, 2) if total > 0 else 0}

# example/test section
if __name__ == "__main__":
    
    from utils import dataLoader, csvHandler
    
    filename     = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column    = "DLNMID"
    var_name     = "Q20"

    csv_handler          = csvHandler.CsvHandler()
    filepath             = csv_handler.get_filepath(filename, 'preprocessed')
    data_loader          = dataLoader.DataLoader()
    var_lab              = data_loader.get_varlab(filename = filename, var_name = var_name)

    preprocessed_text     = csv_handler.load_from_csv(filename, 'preprocessed', models.PreprocessModel)

    config = QualityFilterConfig(
        batch_size=20,
        retries=3 )

    grader = Grader(preprocessed_text, var_lab, config)
    all_results = grader.grade()
    filtered = grader.filter()

    summary = grader.summary()
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nAll results:")
    for r in all_results:
        if r.quality_filter:
        #print(r)
            print(f"{r.respondent_id}: {r.response} ")

    # print("\nMeaningful responses only:")
    # for r in filtered:
    #     print(f"{r.respondent_id}: {r.response}")