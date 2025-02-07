import os
import sys
from typing import List, Any
from pydantic import BaseModel, Field
import instructor
import openai
import tiktoken
import asyncio
import nest_asyncio
from datetime import datetime

current_dir = os.getcwd()
if os.path.basename(current_dir) == 'src':
    modules_dir = os.path.abspath(os.path.join(current_dir, 'modules'))
elif os.path.basename(current_dir) == 'modules': 
   modules_dir=  current_dir
elif os.path.basename(current_dir) == 'utils': 
    modules_dir = os.path.abspath(os.path.join(current_dir, '..', 'modules'))

if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

from config import DEFAULT_MODEL, OPENAI_API_KEY, DEFAULT_LANGUAGE
from prompts import CODING_INSTRUCTIONS

# structured outputs
class PreprocessingSubmodel(BaseModel):
    segment_id: str
    segment_response: str
    descriptive_code: str
    code_description: str

class PreprocessingModel(BaseModel):
    respondent_id: Any
    response: str
    quality_filter: bool
    response_segment: List[PreprocessingSubmodel] 
class CodingBatch(BaseModel):
    tasks: List[PreprocessingModel] = Field(code_description="List of coding tasks in this batch")   
  
# Main function
class DescriptiveCoder:
    def __init__(
        self, 
        provider: str = "openai", 
        api_key: str = None, 
        model: str = None, 
        base_url: str = None,
        max_tokens: int = 16000,  
        completion_reserve: int = 1000,  
        max_batch_size: int = 5 
        ):
    
        self.provider = provider.lower(),
        self.openai_api_key = api_key or OPENAI_API_KEY
        self.openai_model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.completion_reserve = completion_reserve
        self.max_batch_size = max_batch_size
        self._debug_print_first_prompt = True
        
        if not self.openai_api_key:
            raise ValueError("API key is required")
            
        if provider == "openai":
            #self.client = instructor.patch(openai.AsyncOpenAI(api_key=self.openai_api_key))
            self.client = instructor.from_openai(openai.AsyncOpenAI(api_key=self.openai_api_key))

        elif provider == "azure":
            if not base_url:
                raise ValueError("base_url is required for Azure provider")
            self.client = instructor.patch(openai.AsyncOpenAI(base_url=base_url))
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        print(f"Initialized DescriptiveCoder with {provider} provider and {self.openai_model} model")
    
    def create_batches(self, responses: List[PreprocessingModel], var_lab: str) -> List[CodingBatch]:
        encoding = tiktoken.encoding_for_model(self.openai_model)
        
        # Calculate prompt length internally
        empty_prompt = CODING_INSTRUCTIONS['descriptive_code_prompt'].format(
            Language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses="")
        
        prompt_length = len(encoding.encode(empty_prompt))
        token_budget = self.max_tokens - prompt_length - self.completion_reserve
        
        # Skip calculation if no responses
        if not responses:
            return []
        
        # Calculate average tokens per response for adaptive batching
        avg_tokens_per_response = sum(len(encoding.encode(r.response)) for r in responses) / max(1, len(responses))
        adaptive_max_batch_size = min(self.max_batch_size, max(1, int(token_budget / max(1, avg_tokens_per_response))))
        
        batches = []
        current_batch_tasks = []
        current_batch_tokens = 0
        
        print(f"Creating batches with token budget: {token_budget}, adaptive max batch size: {adaptive_max_batch_size}")
        
        for response in responses:
            respondent_id = response.respondent_id
            response_text = response.response
            
            # Same format as used in process_batch function
            task_text = (
                f"Item:\n"
                f"Respondent ID: {respondent_id}\n"   
                f"Response: \"{response_text}\"\n")
                    
            task_tokens = len(encoding.encode(task_text))
            
            # Handle oversized individual responses
            if task_tokens > token_budget and not current_batch_tasks:
                print(f"Warning: Response from respondent {respondent_id} exceeds token budget ({task_tokens} > {token_budget}). Processing as single item batch.")
                batches.append(CodingBatch(tasks=[response]))
                continue
                
            # Start a new batch if current one would exceed limits
            if (current_batch_tokens + task_tokens > token_budget or 
                len(current_batch_tasks) >= adaptive_max_batch_size):
                if current_batch_tasks:  # Only add batch if it's not empty
                    batches.append(CodingBatch(tasks=current_batch_tasks))
                    current_batch_tasks = []
                    current_batch_tokens = 0
                
            # Add the response to the current batch
            current_batch_tasks.append(response)
            current_batch_tokens += task_tokens
        
        # Add the last batch if not empty
        if current_batch_tasks:
            batches.append(CodingBatch(tasks=current_batch_tasks))
        
        print(f"Created {len(batches)} batches from {len(responses)} responses")
        return batches
    
    async def process_batch(self, batch: CodingBatch, var_lab: str, max_retries: int = 3) -> List[PreprocessingModel]:
        tasks_string = ""
        for i, task in enumerate(batch.tasks):
            tasks_string += (
                f"Item {i + 1}:\n"
                f"Respondent ID: {task.respondent_id}\n"
                f"Response: \"{task.response}\"\n\n")
        
        prompt = CODING_INSTRUCTIONS['descriptive_code_prompt'].format(
            Language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=tasks_string)
       
        if hasattr(self, '_debug_print_first_prompt') and self._debug_print_first_prompt:
            print("\n" + "="*80)
            print("COMPLETE PROMPT (USER MESSAGE ONLY):")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
         
            print(f"Using model: {self.openai_model}")
            print("Response model: List[ResponseItem]")
            print("Note: Instructor will add system instructions for structured output\n")
         
            self._debug_print_first_prompt = False
        
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # Using instructor for structured output
                batch_results = await self.client.chat.completions.create(
                    model=self.openai_model,
                    response_model=List[PreprocessingModel],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    max_retries=2  # API-level retries for network issues
                )
                
                # Match results with input respondent IDs to preserve order
                result_map = {item.respondent_id: item for item in batch_results}
                ordered_results = []
                missing_ids = []
                
                for task in batch.tasks:
                    if task.respondent_id in result_map:
                        result = result_map[task.respondent_id]
                        
                        # Check if descriptive_codes exist and are not empty
                        if not result.descriptive_codes:
                            missing_ids.append(task.respondent_id)
                            # Keep track but still add to results for now
                        
                        ordered_results.append(result)
                    else:
                        # If no results for this ID, track it as missing
                        missing_ids.append(task.respondent_id)
                        ordered_results.append(task)  # Include original task
                
                # If we have missing codes, retry the batch if not at max retries
                if missing_ids and retries < max_retries:
                    #missing_str = ", ".join(str(id) for id in missing_ids)
                    #print(f"Missing codes for respondent IDs: {missing_str}. Retry {retries+1}/{max_retries}...")
                    retries += 1
                    # Exponential backoff with jitter
                    wait_time = (0.5 + asyncio.get_event_loop().time() % 1) * (2 ** retries)
                    await asyncio.sleep(wait_time)
                    continue
                
                # Special handling for any items still missing codes after all retries
                if missing_ids:
                    # Find specific responses that need special attention
                    for i, result in enumerate(ordered_results):
                        if result.respondent_id in missing_ids:
                            # Try to generate a code for just this one response
                            if retries >= max_retries:
                                print(f"Still missing codes for respondent ID {result.respondent_id} after {max_retries} retries.")
                                # Check if it's an empty/unclear response
                                response_text = result.response.strip()
                                if response_text in ["?", "geen idee", "weet niet"] or len(response_text) < 5:
                                    # Handle empty or unclear responses with a default code
                                    ordered_results[i].descriptive_codes = [
                                        PreprocessingSubmodel(
                                            response_segment=result.response,
                                            code="NO_MEANINGFUL_CONTENT",
                                            description="Response is empty, unclear, or lacks substantive content")
                                    ]
                                    print(f"Added default code for empty/unclear response from ID {result.respondent_id}")
                
                # If we get here, processing succeeded (or we reached max retries)
                if retries > 0 and not missing_ids:
                    print(f"Successfully processed batch after {retries} retries")
                return ordered_results
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= max_retries:
                    # Exponential backoff with jitter (0.5 to 1.5 seconds * 2^retry)
                    wait_time = (0.5 + asyncio.get_event_loop().time() % 1) * (2 ** retries)
                    print(f"Retry {retries}/{max_retries}: Error processing batch: {str(e)}")
                    print(f"Waiting {wait_time:.2f} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed to process batch after {max_retries} retries. Last error: {str(last_error)}")
        
        # If all retries failed, make sure each item has at least an empty codes list
        for i, task in enumerate(batch.tasks):
            if not hasattr(task, 'descriptive_codes') or task.descriptive_codes is None:
                batch.tasks[i].descriptive_codes = []
        
        return batch.tasks
    
    async def generate_codes_async(self, responses: List[PreprocessingModel], var_lab: str, max_retries: int = 3) -> List[PreprocessingModel]:
        start_time = datetime.now()
        print(f"Starting code generation for {len(responses)} responses")
        
        batches = self.create_batches(responses, var_lab)
        
        if not batches:
            print("No batches created. Returning original responses.")
            return responses
            
        total_batches = len(batches)
        print(f"Processing {total_batches} batches with max {max_retries} retries per batch if needed")
        
        # Process batches concurrently
        all_results = []
        
        # Track progress and failures
        successful_batches = 0
        failed_batches = 0
        responses_with_codes = 0
        
        batch_results = await asyncio.gather(
            *(self.process_batch(batch, var_lab, max_retries) for batch in batches),
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
                if resp.descriptive_codes and len(resp.descriptive_codes) > 0:
                    responses_with_codes += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        total_responses = len(responses)
        processed_responses = len(all_results)
        
        print(f"Completed code generation in {duration:.2f} seconds")
        print(f"Batches: {successful_batches} successful, {failed_batches} failed, {total_batches} total")
        print(f"Responses: {processed_responses} processed out of {total_responses}")
        print(f"Responses with codes: {responses_with_codes} out of {processed_responses} ({responses_with_codes/processed_responses*100:.1f}%)")
        
        # Check for responses without codes
        if responses_with_codes < processed_responses:
            #print(f"Warning: {processed_responses - responses_with_codes} responses did not receive descriptive codes")
            
            # Special handling for items that still don't have codes
            for resp in all_results:
                if not resp.descriptive_codes or len(resp.descriptive_codes) == 0:
                    # Handle empty/unclear responses
                    response_text = resp.response.strip()
                    if not response_text or len(response_text) < 5:
                        resp.descriptive_codes = [
                            PreprocessingSubmodel(
                                response_segment=resp.response,
                                code="NO_MEANINGFUL_CONTENT",
                                description="Response is empty, unclear, or lacks substantive content"
                            )
                        ]
                    else:
                        # For other responses, try to add a generic meaningful code
                        resp.descriptive_codes = [
                            PreprocessingSubmodel(
                                response_segment=resp.response,
                                code="UNCATEGORIZED_RESPONSE",
                                description="Response could not be automatically categorized"
                            )
                        ]
                
            print("Applied default codes to responses that didn't receive proper coding")
                
        return all_results
    
    def generate_codes(self, responses: List[PreprocessingModel], var_lab: str, max_retries: int = 3) -> List[PreprocessingModel]:
        if not responses:
            print("No responses provided. Returning empty list.")
            return []
            
        async def main():
            return await self.generate_codes_async(responses, var_lab, max_retries)
        
        # Apply nest_asyncio to allow running asyncio in interactive environments
        nest_asyncio.apply()
        return asyncio.run(main())
                
# Example usage:
if __name__ == "__main__":
    current_dir = os.getcwd()
    utils_dir = os.path.abspath(os.path.join(current_dir, "utils"))
    from utils.testData import TestDataLoader
    loader = TestDataLoader()

    responses_dicts  = loader.get_test_data(
       filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
       variable_name = "Q20", 
       n_samples= 100,
       as_response_items=True)

    responses = [PreprocessingModel(respondent_id=item['respondent_id'],response=item['sentence']  ) for item in responses_dicts]    
    
    var_label_dict = loader.list_variables(
      filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
      string_only = True)
    
    var_lab = var_label_dict['Q20']
    
    coder = DescriptiveCoder()
    results = coder.generate_codes(responses, var_lab)
    
    for result in results[:30]:
        print(f"\nRespondent ID: {result.respondent_id}")
        print(f"Response: {result.response}")
        print("Descriptive Codes:")
        codes = result.descriptive_codes or []
        for code in codes:
           print(f"  - Segment: {code.response_segment}")
           print(f"    Code: {code.code}")
           print(f"    Description: {code.description}")