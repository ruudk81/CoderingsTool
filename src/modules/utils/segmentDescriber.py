from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field
import instructor
import openai
import tiktoken
import asyncio
import nest_asyncio
from datetime import datetime
import json

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


from config import OPENAI_API_KEY, DEFAULT_LANGUAGE #,DEFAULT_MODEL
from prompts import SEGMENTATION_PROMPT, REFINEMENT_PROMPT, CODING_PROMPT
import models

DEFAULT_MODEL = "gpt-4.1-mini"

class LangChainPipeline :
    def __init__(self, model_name: str, api_key: str, language: str, var_lab: str):
        self.language = language
        self.var_lab = var_lab
        
        self.language = language
        self.llm = ChatOpenAI(
            temperature=0,
            model=model_name,
            openai_api_key=api_key)

        self.parser = JsonOutputParser()
        self.retry_delay = 2
        self.max_retries = 3
        
        self.chain = self.build_chain()

    def _safe_get(self, x, key):
        return x.get(key) if isinstance(x, dict) else None
    
    def _safe_extract_segments(self, inputs: Union[Dict, List, Any]) -> List[Dict]:
        if isinstance(inputs, dict):
            segments = inputs.get("segments", [])
            if isinstance(segments, list):
                return segments
            return [segments] if segments else []
        elif isinstance(inputs, list):
            return inputs
        return []

    def build_chain(self):
        
        def debug_log(label: str):
            return RunnableLambda(lambda x: print(f"\n--- {label} ---\n{json.dumps(x, indent=2, ensure_ascii=False)}\n") or x)
    
        segmentation_prompt = PromptTemplate.from_template(SEGMENTATION_PROMPT)
        refinement_prompt = PromptTemplate.from_template(REFINEMENT_PROMPT)
        coding_prompt = PromptTemplate.from_template(CODING_PROMPT)
    
        chain = (
           {
               "response": lambda x: x["response"],
               "var_lab": lambda x: x["var_lab"],
               "language": lambda x: x["language"]
           }
           | segmentation_prompt
           | self.llm
           | self.parser
           #| debug_log("🔍 After Segmentation")
           | RunnableLambda(lambda inputs: {
               "segments": self._safe_extract_segments(inputs),
               "var_lab": self._safe_get(inputs, "var_lab") if isinstance(inputs, dict) else self.var_lab,
               "language": self._safe_get(inputs, "language") if isinstance(inputs, dict) else self.language
           })
           | refinement_prompt
           | self.llm
           | self.parser
           #| debug_log("🔍 After Refinement")
           | RunnableLambda(lambda inputs: {
               "segments": [
                   {
                       "segment_id": segment.get("segment_id", "") if isinstance(segment, dict) else "",
                       "segment_response": segment.get("segment_response", "") if isinstance(segment, dict) else "",
                       "descriptive_code": segment.get("descriptive_code", "") if isinstance(segment, dict) else "",
                       "code_description": segment.get("code_description", "") if isinstance(segment, dict) else ""
                   } 
                   for segment in self._safe_extract_segments(inputs)
               ],
               "var_lab": self._safe_get(inputs, "var_lab") if isinstance(inputs, dict) else self.var_lab,
               "language": self._safe_get(inputs, "language") if isinstance(inputs, dict) else self.language
           })
           | coding_prompt
           | self.llm
           | self.parser
           #| debug_log("✅ Final Coding Output")
           )
   
        return chain

    async def invoke_with_retries(self, inputs: dict):
        retries = 0
        while retries <= self.max_retries:
            try:
                result = await self.chain.ainvoke(inputs)
                return result
            except Exception as e:
                print(f"Retry {retries + 1}: Error in LangChain chain execution: {str(e)}")
                retries += 1
                await asyncio.sleep(self.retry_delay * retries)

        raise RuntimeError("LangChain pipeline failed after max retries")

class CodingBatch(BaseModel):
    tasks: List[Dict] = Field(description="List of coding tasks in this batch")   

class SegmentDescriber:
    def __init__(
        self, 
        provider: str = "openai", 
        api_key: str = None, 
        model: str = None, 
        base_url: str = None,
        max_tokens: int = 16000,  
        completion_reserve: int = 1000,  
        max_batch_size: int = 5,
        var_lab : str = "" ):
        
        self.provider = provider.lower()
        self.openai_api_key = api_key or OPENAI_API_KEY
        self.openai_model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.completion_reserve = completion_reserve
        self.max_batch_size = max_batch_size
        self._debug_print_first_prompt = True
        self.varlab = var_lab
        
        self.langchain_pipeline = LangChainPipeline(
            model_name=self.openai_model,
            api_key=self.openai_api_key,
            language=DEFAULT_LANGUAGE,
            var_lab = "" ) 
            
        self.chain = self.langchain_pipeline.build_chain()

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
        
        print(f"Initialized DescriptiveCoder with {provider} provider and {self.openai_model} model")
    
    def create_batches(self, responses: List[models.DescriptiveModel], var_lab: str) -> List[CodingBatch]:
        #encoding = tiktoken.encoding_for_model(self.openai_model)
        
        try:
            encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            # Fall back to a known encoding that's similar to GPT-4 models
            encoding = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by GPT-4
            print(f"Using cl100k_base encoding as fallback for {self.openai_model}")
        
        # Calculate token budget
        segmentation_prompt = SEGMENTATION_PROMPT
        segmentation_prompt = segmentation_prompt.replace("{language}", DEFAULT_LANGUAGE)
        segmentation_prompt = segmentation_prompt.replace("{var_lab}", var_lab)
        segmentation_prompt = segmentation_prompt.replace("{response}", "")
        coding_prompt = CODING_PROMPT
        coding_prompt = coding_prompt.replace("{language}", DEFAULT_LANGUAGE)
        coding_prompt = coding_prompt.replace("{var_lab}", var_lab)
        coding_prompt = coding_prompt.replace("{segments}", "")
        prompt = segmentation_prompt + "\n" + coding_prompt
        
        prompt_length = len(encoding.encode(prompt))
        token_budget = self.max_tokens - prompt_length - self.completion_reserve
        
        # Skip calculation if no responses
        if not responses:
            return []
        
        # Calculate average tokens per response for adaptive batching
        avg_tokens_per_response = sum(len(encoding.encode(r.response)) for r in responses) / max(1, len(responses))
        adaptive_max_batch_size = min(self.max_batch_size, max(1, int(token_budget / max(1, avg_tokens_per_response))))
        
        print(f"estimated number of tokens= {prompt_length + avg_tokens_per_response}")
        
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
                batches.append(CodingBatch(tasks=[response.model_dump()]))  # Convert to dict for Pydantic v2
                continue
                
            # Start a new batch if current one would exceed limits
            if (current_batch_tokens + task_tokens > token_budget or 
                len(current_batch_tasks) >= adaptive_max_batch_size):
                if current_batch_tasks:  # Only add batch if it's not empty
                    batches.append(CodingBatch(tasks=current_batch_tasks))
                    current_batch_tasks = []
                    current_batch_tokens = 0
                
            # Add the response to the current batch
            current_batch_tasks.append(response.model_dump())  
            current_batch_tokens += task_tokens
        
        # Add the last batch if not empty
        if current_batch_tasks:
            batches.append(CodingBatch(tasks=current_batch_tasks))
        
        print(f"Created {len(batches)} batches from {len(responses)} responses")
        return batches

    async def process_response(self, respondent_id, response_text, var_lab, max_retries=3):
        """Process a single response with two-step approach"""
        retries = 0
        while retries <= max_retries:
            try:
                result = await self.langchain_pipeline.invoke_with_retries({
                    "response": response_text,
                    "var_lab": var_lab,
                    "language": DEFAULT_LANGUAGE
                })
         
                return models.DescriptiveModel(
                respondent_id=respondent_id,
                response=response_text,
                quality_filter=None,
                response_segment=[models.DescriptiveSubmodel(**seg) for seg in result]
                )
            
            except Exception as e:
                print(f"Error in LangChain pipeline: {str(e)}")
                return models.DescriptiveModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=None,
                    response_segment=[
                        models.DescriptiveSubmodel(
                            segment_id="1",
                            segment_response=response_text,
                            descriptive_code="NA",
                            code_description="NA"
                            )])
       
    async def process_batch(self, batch: CodingBatch, var_lab: str, max_retries: int = 3) -> List[models.DescriptiveModel]:
        """Process a batch of responses using two-step approach"""
        tasks = []
        
        # Create tasks for each response in the batch
        for task_dict in batch.tasks:
            # Convert dict back to DescriptiveModel if needed
            if isinstance(task_dict, dict):
                task = models.DescriptiveModel(**task_dict)
            else:
                task = task_dict
                
            # Add task to process list
            tasks.append(self.process_response(
                task.respondent_id, 
                task.response, 
                var_lab, 
                max_retries))
        
        # Process all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create fallback for failed task
                task_dict = batch.tasks[i]
                if isinstance(task_dict, dict):
                    respondent_id = task_dict.get("respondent_id")
                    response_text = task_dict.get("response", "")
                else:
                    respondent_id = task_dict.respondent_id
                    response_text = task_dict.response
                    
                processed_results.append(models.DescriptiveModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=None,
                    response_segment=[
                        models.DescriptiveSubmodel(
                            segment_id="1",
                            segment_response=response_text,
                            descriptive_code="PROCESSING_ERROR",
                            code_description="Er kon geen betekenisvolle analyse worden gegenereerd voor deze respons."
                        )
                    ]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def generate_codes_async(self, responses: List[models.DescriptiveModel], var_lab: str, max_retries: int = 3) -> List[models.DescriptiveModel]:
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
                if resp.response_segment and len(resp.response_segment) > 0:
                    responses_with_codes += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        total_responses = len(responses)
        processed_responses = len(all_results)
        
        print(f"Completed code generation in {duration:.2f} seconds")
        print(f"Batches: {successful_batches} successful, {failed_batches} failed, {total_batches} total")
        print(f"Responses: {processed_responses} processed out of {total_responses}")
        
        # Avoid division by zero when calculating percentage
        if processed_responses > 0:
            print(f"Responses with codes: {responses_with_codes} out of {processed_responses} ({responses_with_codes/processed_responses*100:.1f}%)")
        else:
            print("Responses with codes: 0 out of 0 (0%)")
        
        return all_results
    
    def generate_codes(self, responses: List[models.DescriptiveModel], var_lab: str, max_retries: int = 3) -> List[models.DescriptiveModel]:
        if not responses:
            print("No responses provided. Returning empty list.")
            return []
        
        self.var_lab = var_lab
        self.langchain_pipeline.var_lab = var_lab
        
        print(f"\nThe survey question: {var_lab}\n")
    
        async def main():
            return await self.generate_codes_async(responses, var_lab, max_retries)
        
        # Apply nest_asyncio to allow running asyncio in interactive environments
        nest_asyncio.apply()
        return asyncio.run(main())


# Example usage
if __name__ == "__main__":
    
    from modules.utils import data_io, csvHandler
    import random
    
    filename     = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column    = "DLNMID"
    var_name     = "Q20"

    csv_handler          = csvHandler.CsvHandler()
    filepath             = csv_handler.get_filepath(filename, 'preprocessed')
    data_loader          = data_io.DataLoader()
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
            print(f"    Code: {code.descriptive_code}")
            print(f"    Description: {code.code_description}")
        print("\n")