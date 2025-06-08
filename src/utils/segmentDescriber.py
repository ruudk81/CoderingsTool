import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field
import instructor
import openai
import tiktoken
import asyncio
import nest_asyncio
import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG
from prompts import SEGMENTATION_PROMPT, REFINEMENT_PROMPT, CODING_PROMPT
import models
from .verboseReporter import VerboseReporter, ProcessingStats


# Removed CodingBatch class - now using List[models.DescriptiveModel] directly

class SegmentDescriber:
    def __init__(
        self, 
        config: SegmentationConfig = None,
        provider: str = "openai", 
        api_key: str = None, 
        model: str = None, 
        base_url: str = None,
        var_lab : str = "",
        verbose: bool = False,
        model_config: ModelConfig = None,
        prompt_printer = None):
        
        # Use provided config or create default
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.model_config = model_config or ModelConfig()
        
        self.provider = provider.lower()
        self.openai_api_key = api_key or OPENAI_API_KEY
        # Use model from config for segmentation/description stage
        self.openai_model = model or self.config.model
        self.max_tokens = self.config.max_tokens
        self.completion_reserve = self.config.completion_reserve
        self.max_batch_size = self.config.max_batch_size
        self._debug_print_first_prompt = True
        self.varlab = var_lab
        self.verbose_reporter = VerboseReporter(verbose)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        
        # Initialize encoding for token calculations
        try:
            self.encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            print(f"Using cl100k_base encoding as fallback for {self.openai_model}")

        if not self.openai_api_key:
            raise ValueError("API key is required")
            
        if provider == "openai":
            self.client = instructor.from_openai(openai.AsyncOpenAI(api_key=self.openai_api_key))
        elif provider == "azure":
            if not base_url:
                raise ValueError("base_url is required for Azure provider")
            self.client = instructor.patch(openai.AsyncOpenAI(base_url=base_url))
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        #print(f"Initialized DescriptiveCoder with {provider} provider and {self.openai_model} model")
    
    def create_dynamic_batches(self, responses: List[models.DescriptiveModel], var_lab: str) -> List[List[models.DescriptiveModel]]:
        """Create batches optimized for multi-response processing"""
        try:
            encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            print(f"Using cl100k_base encoding as fallback for {self.openai_model}")
        
        # Calculate base prompt overhead (constant per batch)
        base_prompt_tokens = self._calculate_base_prompt_tokens(var_lab, encoding)
        
        # Available tokens for response content  
        available_tokens = int((self.config.max_tokens - base_prompt_tokens - self.config.completion_reserve) 
                              * self.config.target_token_utilization)
        
        if not responses:
            return []
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for response in responses:
            # Calculate tokens for this response in batch format
            response_tokens = len(encoding.encode(f"Response {len(current_batch) + 1} (ID: {response.respondent_id}): \"{response.response}\""))
            
            # Would this response exceed our budget?
            if (current_tokens + response_tokens > available_tokens and 
                len(current_batch) >= self.config.min_batch_size):
                
                # Finalize current batch
                batches.append(current_batch)
                current_batch = [response]
                current_tokens = response_tokens
            else:
                current_batch.append(response)
                current_tokens += response_tokens
            
            # Don't exceed max batch size
            if len(current_batch) >= self.config.max_batch_size:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _calculate_base_prompt_tokens(self, var_lab: str, encoding) -> int:
        """Calculate token overhead for base prompt templates"""
        # Sample base prompts without response content
        base_segmentation = SEGMENTATION_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=""  # Empty for calculation
        )
        base_refinement = REFINEMENT_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            segments=""  # Empty for calculation
        )
        base_coding = CODING_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            segments=""  # Empty for calculation
        )
        
        # Return the maximum (worst case) for any single stage
        return max(
            len(encoding.encode(base_segmentation)),
            len(encoding.encode(base_refinement)),
            len(encoding.encode(base_coding))
        )

    def _format_responses_for_prompt(self, responses: List[models.DescriptiveModel]) -> str:
        """Format multiple responses for batch processing"""
        formatted_responses = []
        for response in responses:
            formatted_responses.append(f"Response {len(formatted_responses) + 1} (ID: {response.respondent_id}): \"{response.response}\"")
        return "\n".join(formatted_responses)
    
    def _format_segments_for_refinement(self, segmented_results: List[Dict]) -> str:
        """Format segmented results for refinement stage"""
        import json
        return json.dumps(segmented_results, ensure_ascii=False, indent=2)
    
    def _format_segments_for_coding(self, refined_results: List[Dict]) -> str:
        """Format refined results for coding stage"""
        import json
        return json.dumps(refined_results, ensure_ascii=False, indent=2)

    async def process_batch_multi_response(self, batch: List[models.DescriptiveModel], var_lab: str, max_retries: int = 3) -> List[models.DescriptiveModel]:
        """Process a batch of responses using multi-response 3-stage approach"""
        try:
            # Stage 1: Segmentation - Process all responses at once
            responses_text = self._format_responses_for_prompt(batch)
            segmentation_result = await self._call_segmentation_stage(responses_text, var_lab, max_retries)
            
            # Stage 2: Refinement - Process all segmented results at once  
            segments_text = self._format_segments_for_refinement(segmentation_result)
            refinement_result = await self._call_refinement_stage(segments_text, var_lab, max_retries)
            
            # Stage 3: Coding - Process all refined results at once
            refined_text = self._format_segments_for_coding(refinement_result)
            coding_result = await self._call_coding_stage(refined_text, var_lab, max_retries)
            
            # Convert results back to DescriptiveModel objects
            processed_results = []
            for result_item in coding_result:
                respondent_id = result_item.get("respondent_id")
                segments = result_item.get("segments", [])
                
                # Find original response text
                original_response = next((r.response for r in batch if str(r.respondent_id) == str(respondent_id)), "")
                
                processed_results.append(models.DescriptiveModel(
                    respondent_id=respondent_id,
                    response=original_response,
                    quality_filter=None,
                    response_segment=[models.DescriptiveSubmodel(**seg) for seg in segments]
                ))
            
            return processed_results
            
        except Exception as e:
            print(f"Error in multi-response batch processing: {str(e)}")
            # Return fallback responses
            return [self._create_fallback_response(response) for response in batch]
    
    def _create_fallback_response(self, response: models.DescriptiveModel) -> models.DescriptiveModel:
        """Create fallback response for failed processing"""
        return models.DescriptiveModel(
            respondent_id=response.respondent_id,
            response=response.response,
            quality_filter=None,
            response_segment=[
                models.DescriptiveSubmodel(
                    segment_id="1",
                    segment_response=response.response,
                    segment_label="PROCESSING_ERROR",
                    segment_description="Er kon geen betekenisvolle analyse worden gegenereerd voor deze respons."
                )
            ]
        )
    
    async def _call_segmentation_stage(self, responses_text: str, var_lab: str, max_retries: int) -> List[Dict]:
        """Call segmentation stage with multiple responses"""
        prompt = SEGMENTATION_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=responses_text
        )
        
        # Capture prompt for first call
        if self.prompt_printer and not hasattr(self, '_captured_segmentation'):
            self.prompt_printer.capture_prompt(
                step_name="segmentation",
                utility_name="SegmentDescriber",
                prompt_content=prompt,
                prompt_type="segmentation_multi",
                metadata={
                    "model": self.openai_model,
                    "var_lab": var_lab,
                    "language": DEFAULT_LANGUAGE,
                    "stage": "1/3 - Multi-Response Segmentation"
                }
            )
            self._captured_segmentation = True
        
        return await self._call_llm_with_retries(prompt, max_retries)
    
    async def _call_refinement_stage(self, segments_text: str, var_lab: str, max_retries: int) -> List[Dict]:
        """Call refinement stage with multiple segmented responses"""
        prompt = REFINEMENT_PROMPT.format(
            var_lab=var_lab,
            segments=segments_text
        )
        
        return await self._call_llm_with_retries(prompt, max_retries)
    
    async def _call_coding_stage(self, refined_text: str, var_lab: str, max_retries: int) -> List[Dict]:
        """Call coding stage with multiple refined responses"""
        prompt = CODING_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            segments=refined_text
        )
        
        return await self._call_llm_with_retries(prompt, max_retries)
    
    async def _call_llm_with_retries(self, prompt: str, max_retries: int) -> List[Dict]:
        """Call LLM with retry logic"""
        retries = 0
        while retries <= max_retries:
            try:
                llm = ChatOpenAI(
                    temperature=self.config.temperature,
                    model=self.openai_model,
                    openai_api_key=self.openai_api_key,
                    seed=self.model_config.seed
                )
                
                parser = JsonOutputParser()
                result = await llm.ainvoke(prompt)
                return parser.parse(result.content)
                
            except Exception as e:
                print(f"LLM call failed on attempt {retries + 1}/{max_retries + 1}: {str(e)}")
                retries += 1
                if retries <= max_retries:
                    await asyncio.sleep(self.config.retry_delay * retries)
        
        raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts")
       
    
    async def generate_codes_async(self, responses: List[models.DescriptiveModel], var_lab: str, max_retries: int = 3) -> List[models.DescriptiveModel]:
        self._stats.start_timing()
        self._stats.input_count = len(responses)
        
        self.verbose_reporter.step_start("Descriptive Code Generation", emoji="🏷️")
        self.verbose_reporter.stat_line(f"Processing {len(responses)} responses...")
        
        batches = self.create_dynamic_batches(responses, var_lab)
        
        if not batches:
            print("No batches created. Returning original responses.")
            return responses
            
        self.verbose_reporter.stat_line(f"Processing {len(batches)} batches with multi-response approach...")
        
        # Process batches concurrently using new multi-response approach
        all_results = []
        
        # Track progress and failures
        successful_batches = 0
        failed_batches = 0
        responses_with_codes = 0
    
        batch_results = await asyncio.gather(
            *(self.process_batch_multi_response(batch, var_lab, max_retries) for batch in batches),
            return_exceptions=True
        )
        
        # Flatten results and handle any exceptions
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Batch processing error: {str(result)}")
                failed_batches += 1
                continue
                
            successful_batches += 1
            
            # Count how many responses have codes
            for resp in result:
                all_results.append(resp)
                if resp.response_segment and len(resp.response_segment) > 0:
                    responses_with_codes += 1
        
        self._stats.end_timing()
        self._stats.output_count = len(all_results)
        
        # Calculate statistics
        # total_responses = len(responses)
        # processed_responses = len(all_results)
        
        # Collect sample codes for verbose output
        code_examples = []
        unique_codes = set()
        multi_code_responses = 0
        total_code_length = 0
        code_count = 0
        
        for resp in all_results:
            if resp.response_segment and len(resp.response_segment) > 0:
                if len(resp.response_segment) > 1:
                    multi_code_responses += 1
                    
                for segment in resp.response_segment:
                    if segment.segment_label and segment.segment_label not in ["NA", "PROCESSING_ERROR"]:
                        unique_codes.add(segment.segment_label)
                        code_words = segment.segment_label.split()
                        total_code_length += len(code_words)
                        code_count += 1
                        
                        # Collect examples
                        if len(code_examples) < self.config.max_code_examples and segment.segment_response:
                            code_examples.append(f'"{segment.segment_response}" → "{segment.segment_label}"')
        
        avg_code_length = total_code_length / code_count if code_count > 0 else 0
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Unique themes identified: {len(unique_codes)}")
        self.verbose_reporter.stat_line(f"Average code length: {avg_code_length:.1f} words")
        if multi_code_responses > 0:
            self.verbose_reporter.stat_line(f"Responses with multiple codes: {multi_code_responses}")
        
        # Show code examples
        if code_examples:
            self.verbose_reporter.sample_list("Sample generated codes", code_examples)
        
        self.verbose_reporter.step_complete("Code generation completed")
        
        return all_results
    
    def generate_codes(self, responses: List[models.DescriptiveModel], var_lab: str, max_retries: int = 3) -> List[models.DescriptiveModel]:
        if not responses:
            print("No responses provided. Returning empty list.")
            return []
        
        self.var_lab = var_lab
        
        #print(f"\nThe survey question: {var_lab}\n")
    
        async def main():
            return await self.generate_codes_async(responses, var_lab, max_retries)
        
        # Apply nest_asyncio to allow running asyncio in interactive environments
        nest_asyncio.apply()
        return asyncio.run(main())


# Example usage
if __name__ == "__main__":
    
    from utils import dataLoader, csvHandler
    import random
    
    filename     = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column    = "DLNMID"
    var_name     = "Q20"

    csv_handler          = csvHandler.CsvHandler()
    filepath             = csv_handler.get_filepath(filename, 'preprocessed')
    data_loader          = dataLoader.DataLoader()
    var_lab              = data_loader.get_varlab(filename = filename, var_name = var_name)

    preprocessed_text     = csv_handler.load_from_csv(filename, 'preprocessed', models.PreprocessModel)
    input_data            = [item.to_model(models.DescriptiveModel) for item in preprocessed_text]

    coder = SegmentDescriber()
    results = coder.generate_codes(input_data, var_lab)
    
    random_results = random.sample(results, min(10, len(results))) 
    for result in random_results:
        print(f"\nRespondent ID: {result.respondent_id}")
        print(f"Response: {result.response}")
        print("Descriptive Codes:")
        codes = result.response_segment or []
        for code in codes:
            print(f"  - Segment: {code.segment_response}")
            print(f"    Code: {code.segment_label}")
            print(f"    Description: {code.segment_description}")
        print("\n")