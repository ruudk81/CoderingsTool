import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field
import instructor
import openai
import tiktoken
import asyncio
import nest_asyncio

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG
from prompts import IDEA_EXTRACTION_PROMPT
import models
from .verboseReporter import VerboseReporter, ProcessingStats

class LangChainPipeline :
    def __init__(self, model_name: str, api_key: str, language: str, var_lab: str, 
                 temperature: float = 0.0, config: SegmentationConfig = None, prompt_printer = None):
        self.language = language
        self.var_lab = var_lab
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.prompt_printer = prompt_printer

        self.captured_prompt = False  #only print first prompt

        model_config = ModelConfig()
        
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            openai_api_key=api_key,
            seed=model_config.seed)

        self.parser = JsonOutputParser()
        self.retry_delay = self.config.retry_delay
        self.max_retries = self.config.max_retries
        
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
        
        idea_extraction_prompt = PromptTemplate.from_template(IDEA_EXTRACTION_PROMPT)
        def capture_idea_extraction_prompt(inputs):
            if self.prompt_printer and not self.captured_prompt:
                formatted_prompt = IDEA_EXTRACTION_PROMPT.format(
                    respondent_id=inputs.get("respondent_id", ""),
                    response=inputs.get("response", ""),
                    var_lab=inputs.get("var_lab", ""),
                    language=inputs.get("language", "")
                )
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction",
                    utility_name="IdeaExtractor", 
                    prompt_content=formatted_prompt,
                    prompt_type="gatos_idea_extraction",
                    metadata={
                        "model": self.llm.model_name,
                        "var_lab": inputs.get("var_lab", ""),
                        "stage": "Idea Extraction"
                    }
                )
                self.captured_prompt = True
            return inputs
    
        chain = (
             {
                "respondent_id": lambda x: x["respondent_id"],
                "response": lambda x: x["response"],
                "var_lab": lambda x: x["var_lab"],
                "language": lambda x: x["language"]
            }
            | RunnableLambda(capture_idea_extraction_prompt)
            | idea_extraction_prompt
            | self.llm
            | self.parser
        )

        return chain

    async def invoke_with_retries(self, inputs: dict):
        retries = 0
        while retries <= self.max_retries:
            try:
                result = await self.chain.ainvoke(inputs)
                return result
            except Exception as e:
                print(f"Retry {retries + 1}: Error in Enhanced LangChain chain execution: {str(e)}")
                retries += 1
                await asyncio.sleep(self.retry_delay * retries)

        raise RuntimeError("Enhanced LangChain pipeline failed after max retries")

class CodingBatch(BaseModel):
    tasks: List[Dict] = Field(description="List of coding tasks in this batch")   

class IdeaExtractor:
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
     
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.model_config = model_config or ModelConfig()
        
        self.provider = provider.lower()
        self.openai_api_key = api_key or OPENAI_API_KEY
        self.openai_model = model or self.config.model
        self.max_tokens = self.config.max_tokens
        self.completion_reserve = self.config.completion_reserve
        self.max_batch_size = self.config.max_batch_size
        self._debug_print_first_prompt = True
        self.varlab = var_lab
        self.language = DEFAULT_LANGUAGE  
        self.verbose_reporter = VerboseReporter(verbose)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        
        self.langchain_pipeline = LangChainPipeline(
            model_name=self.openai_model,
            api_key=self.openai_api_key,
            language=DEFAULT_LANGUAGE,
            var_lab = "",
            temperature=self.config.temperature,
            config=self.config,
            prompt_printer=self.prompt_printer)
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
        
    def create_batches(self, responses: List[models.QualityFilteredModel], var_lab: str) -> List[CodingBatch]:
        try:
            encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by GPT-4
            print(f"Using cl100k_base encoding as fallback for {self.openai_model}")
        
        idea_prompt = IDEA_EXTRACTION_PROMPT 
        idea_prompt = idea_prompt.replace("{var_lab}", var_lab)
        idea_prompt = idea_prompt.replace("{language}", self.language)
        idea_prompt = idea_prompt.replace("{response}", "")
        prompt = idea_prompt
        
        prompt_length = len(encoding.encode(prompt))
        token_budget = self.max_tokens - prompt_length - self.completion_reserve
     
        if not responses:
            return []
        
        avg_tokens_per_response = sum(len(encoding.encode(r.response)) for r in responses) / max(1, len(responses))
        adaptive_max_batch_size = min(self.max_batch_size, max(1, int(token_budget / max(1, avg_tokens_per_response))))
        
        batches = []
        current_batch_tasks = []
        current_batch_tokens = 0
        
        
        for response in responses:
            respondent_id = response.respondent_id
            response_text = response.response
            
            task_text = (
                f"Item:\n"
                f"Respondent ID: {respondent_id}\n"   
                f"Response: \"{response_text}\"\n")
                    
            task_tokens = len(encoding.encode(task_text))
            
            if task_tokens > token_budget and not current_batch_tasks:
                print(f"Warning: Response from respondent {respondent_id} exceeds token budget ({task_tokens} > {token_budget}). Processing as single item batch.")
                batches.append(CodingBatch(tasks=[response.model_dump()]))  # Convert to dict for Pydantic v2
                continue
                
            if (current_batch_tokens + task_tokens > token_budget or 
                len(current_batch_tasks) >= adaptive_max_batch_size):
                if current_batch_tasks:  # Only add batch if it's not empty
                    batches.append(CodingBatch(tasks=current_batch_tasks))
                    current_batch_tasks = []
                    current_batch_tokens = 0
                
            current_batch_tasks.append(response.model_dump())  
            current_batch_tokens += task_tokens
        
        if current_batch_tasks:
            batches.append(CodingBatch(tasks=current_batch_tasks))
        
        return batches

    async def process_response(self, respondent_id, response_text, var_lab, max_retries=3):
        """Process a single response with enhanced or legacy workflow"""
        retries = 0
        while retries <= max_retries:
            try:
                result = await self.langchain_pipeline.invoke_with_retries({
                    "respondent_id": respondent_id,
                    "response": response_text,
                    "var_lab": var_lab,
                    "language": DEFAULT_LANGUAGE
                    })    
         
                ideas_with_unique_ids = []
                for i, idea in enumerate(result):
                    # Create unique ID: respondent_id_ideaNumber (following your pattern)
                    idea_id = f"{respondent_id}_{i+1}"
                    ideas_with_unique_ids.append(models.IdeaSubmodel(
                        idea_id=idea_id,
                        idea=idea.get('idea', '')
                    ))
                
                return models.IdeaModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=None,
                    response_ideas=ideas_with_unique_ids,
                    idea_count=len(ideas_with_unique_ids)
                )
            
            except Exception as e:
                print(f"Error in GATOS idea extraction: {str(e)}")
                return models.IdeaModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=None,
                    response_ideas=[
                        models.IdeaSubmodel(
                            idea_id=f"{respondent_id}_1",  
                            idea="Error in idea extraction"
                        )],
                    idea_count=1
                )
       
    async def process_batch(self, batch: CodingBatch, var_lab: str, max_retries: int = 3) -> List[models.IdeaModel]:
        """Process a batch of responses using two-step approach"""
        tasks = []
        
        for task_dict in batch.tasks:
            if isinstance(task_dict, dict):
                task = models.QualityFilteredModel(**task_dict)
            else:
                task = task_dict
  
            tasks.append(self.process_response(
                task.respondent_id, 
                task.response, 
                var_lab, 
                max_retries))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
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
                    
                processed_results.append(models.IdeaModel(
                    respondent_id=respondent_id,
                    response=response_text,
                    quality_filter=None,
                    response_ideas=[
                        models.IdeaSubmodel(
                            idea_id=f"{respondent_id}_1",  
                            idea="PROCESSING_ERROR"
                        )
                    ],
                    idea_count=1
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def generate_codes_async(self, responses: List[models.QualityFilteredModel], var_lab: str, max_retries: int = 3) -> List[models.IdeaModel]:
        self._stats.start_timing()
        self._stats.input_count = len(responses)
        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")
        self.verbose_reporter.stat_line(f"Processing {len(responses)} responses...")
        
        batches = self.create_batches(responses, var_lab)
        
        if not batches:
            print("No batches created. Returning original responses.")
            return responses
         
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
            
            # Count how many responses have ideas extracted
            for resp in result:
                all_results.append(resp)
                if resp.response_ideas and len(resp.response_ideas) > 0:
                    responses_with_codes += 1
        
        self._stats.end_timing()
        self._stats.output_count = len(all_results)
          
        # Collect sample ideas for verbose output
        idea_examples = []
        unique_ideas = set()
        multi_idea_responses = 0
        total_idea_length = 0
        idea_count = 0
        
        for resp in all_results:
            if resp.response_ideas and len(resp.response_ideas) > 0:
                if len(resp.response_ideas) > 1:
                    multi_idea_responses += 1
                    
                for idea in resp.response_ideas:
                    if idea.idea and idea.idea not in ["NA", "PROCESSING_ERROR"]:
                        unique_ideas.add(idea.idea)
                        idea_words = idea.idea.split()
                        total_idea_length += len(idea_words)
                        idea_count += 1
                        
                        # Collect examples
                        if len(idea_examples) < self.config.max_code_examples:
                            idea_examples.append(f'"{resp.response}" → "{idea.idea}"')
        
        avg_idea_length = total_idea_length / idea_count if idea_count > 0 else 0
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Unique ideas identified: {len(unique_ideas)}")
        self.verbose_reporter.stat_line(f"Average idea length: {avg_idea_length:.1f} words")
        if multi_idea_responses > 0:
            self.verbose_reporter.stat_line(f"Responses with multiple ideas: {multi_idea_responses}")
        
        # Show idea examples
        if idea_examples:
            self.verbose_reporter.sample_list("Sample extracted ideas", idea_examples)
        
        self.verbose_reporter.step_complete("GATOS idea extraction completed")
        
        return all_results
    
    def generate_codes(self, responses: List[models.QualityFilteredModel], var_lab: str, max_retries: int = 3) -> List[models.IdeaModel]:
        if not responses:
            print("No responses provided. Returning empty list.")
            return []
        
        self.var_lab = var_lab
        self.langchain_pipeline.var_lab = var_lab
        
    
        async def main():
            return await self.generate_codes_async(responses, var_lab, max_retries)
        
        # Apply nest_asyncio to allow running asyncio in interactive environments
        nest_asyncio.apply()
        return asyncio.run(main())

