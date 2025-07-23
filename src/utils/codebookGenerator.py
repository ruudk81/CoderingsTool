import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np
import os

from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import tiktoken

from prompts import SYSTEM_MESSAGE_CODEBOOK, INITIAL_CODEBOOK_GENERATION, REVIEW_CODEBOOK_GENERATION
from config import EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY
import models
from utils.embedder import Embedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SurveyResponse:
    respondent_id: str
    response: str

@dataclass
class ProcessedResponse:
    respondent_id: str
    original_response: str
    code: Optional[str] = None
    definition: Optional[str] = None
    reasoning: Optional[str] = None
    needs_new_code: bool = False

@dataclass
class Batch:
    batch_id: int
    responses: List[SurveyResponse]
    token_count: int = 0

class CodeRecommendation(BaseModel):
    needs_new_code: bool = Field(description="Whether a new code is needed")
    code: Optional[str] = Field(default=None, description="The recommended code")
    definition: Optional[str] = Field(default=None, description="Code definition")
    reasoning: str = Field(description="Reasoning for the recommendation")


class ProcessingConfig:
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        max_batch_size: int = 10,
        token_buffer: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.token_buffer = token_buffer
        self.max_retries = max_retries
        self.retry_delay = retry_delay


class PromptInputData:
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 embedded_text: List[models.EmbeddingsModel],
                 starter_codes: List[Dict[str, str]], 
                 var_lab: str, 
                 k: int = 5):
        
        self.language = DEFAULT_LANGUAGE
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text  
        self.codebook = starter_codes.copy()  # Growing codebook
        self.var_lab = var_lab
        self.k = k
        self.embedder = Embedder(config=EmbeddingConfig(), verbose=False)
        
    def _prepare_cluster_text(self) -> Dict[int, Dict]:
        embedding_map = {}
        for result in self.embedded_text:
            if hasattr(result, 'idea_embeddings') and result.idea_embeddings:
                for idea in result.idea_embeddings:
                    embedding_map[idea.idea_id] = {
                        'idea': idea.idea,
                        'embedding': idea.idea_embedding}
        
        clusters = {}
        total_ideas = 0
        missing_embeddings = 0
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    if idea.idea_id in embedding_map:
                        embedding_data = embedding_map[idea.idea_id]
                        
                        if cluster_id not in clusters: 
                            clusters[cluster_id] = {'ideas': [], 'embeddings': []}
                        
                        clusters[cluster_id]['ideas'].append(embedding_data['idea'])
                        clusters[cluster_id]['embeddings'].append(embedding_data['embedding'])
                        total_ideas += 1
                    else:
                        missing_embeddings += 1
        
        clusters = {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
        
        return clusters        
            
    async def _embed_codebook_texts(self, code_texts: List[str]) -> List[np.ndarray]:
    
        try:
            embedding_config = EmbeddingConfig()
            client = AsyncOpenAI(api_key=os.environ.get(OPENAI_API_KEY))
            
            response = await client.embeddings.create(model=embedding_config.embedding_model, input=code_texts)
            
            embeddings = []
            for embedding_data in response.data:
                embeddings.append(np.array(embedding_data.embedding, dtype=np.float32))
                
            return embeddings
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error embedding codebook: {str(e)}")
            return []
    
    def _find_k_nearest_codes(self, cluster_embedding: np.ndarray, codebook_embeddings: List[np.ndarray]) -> List[Dict]:
    
        if not codebook_embeddings:
            return []
            
        codebook_array = np.array(codebook_embeddings)
        similarities = cosine_similarity(cluster_embedding.reshape(1, -1), codebook_array)[0]
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        
        seen = set()
        nearest_codes = []
        
        for idx in top_k_indices:
            if idx < len(self.codebook):
                code = self.codebook[idx]
                code_text = code.get('code', '')
                
                if code_text not in seen:
                    seen.add(code_text)
                    nearest_codes.append(code)
                    
                    if len(nearest_codes) >= self.k:
                        break
                
        return nearest_codes        

class PromptManager:
    def __init__(self, var_lab: str, code_text: str, cluster_text: str, code_recommendation: List[CodeRecommendation]):
        self.system_message = SYSTEM_MESSAGE_CODEBOOK.format(language = DEFAULT_LANGUAGE)
        self.initial_code_generation = INITIAL_CODEBOOK_GENERATION.format(language = DEFAULT_LANGUAGE, survey_question=var_lab, code_text=code_text, cluster_text=cluster_text)
        self.review_code_generation = REVIEW_CODEBOOK_GENERATION.format(language = DEFAULT_LANGUAGE, survey_question=var_lab, code_recommendation=code_recommendation, cluster_text=cluster_text)

        
    def get_initial_prompt(self) -> PromptTemplate:
        """Get prompt for initial code generation"""
        return PromptTemplate.from_template(
            self.system_message + "\n\n" + self.initial_code_generation
        )
    
    def get_review_prompt(self) -> PromptTemplate:
        """Get prompt for code review"""
        return PromptTemplate.from_template(
            self.system_message + "\n\n" + self.review_code_generation
        )


class BatchProcessor:
    """Handles batch creation and token management"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.encoding = self._get_encoding()
        
    def _get_encoding(self):
        """Get appropriate tokenizer for the model"""
        try:
            return tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            logger.warning(f"Using cl100k_base encoding as fallback for {self.config.model_name}")
            return tiktoken.get_encoding("cl100k_base")
    
    def create_batches(
        self, 
        responses: List[SurveyResponse], 
        existing_codebook: str,
        survey_question: str
    ) -> List[Batch]:
        """Create batches of responses considering token limits"""
        if not responses:
            return []
        
        # Calculate base prompt tokens
        base_tokens = len(self.encoding.encode(
            existing_codebook + survey_question
        ))
        
        available_tokens = self.config.max_tokens - base_tokens - self.config.token_buffer
        
        batches = []
        current_batch = []
        current_tokens = 0
        batch_id = 0
        
        for response in responses:
            response_tokens = len(self.encoding.encode(response.response))
            
            # Check if adding this response would exceed limits
            if (current_tokens + response_tokens > available_tokens or 
                len(current_batch) >= self.config.max_batch_size):
                
                if current_batch:
                    batches.append(Batch(
                        batch_id=batch_id,
                        responses=current_batch,
                        token_count=current_tokens
                    ))
                    batch_id += 1
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(response)
            current_tokens += response_tokens
        
        # Add final batch
        if current_batch:
            batches.append(Batch(
                batch_id=batch_id,
                responses=current_batch,
                token_count=current_tokens
            ))
        
        logger.info(f"Created {len(batches)} batches from {len(responses)} responses")
        return batches


class SurveyResponsePipeline:
    """Main pipeline for processing survey responses"""
    
    def __init__(
        self,
        api_key: str,
        prompts_config: Dict[str, str],
        config: Optional[ProcessingConfig] = None,
        verbose: bool = False
    ):
        self.config = config or ProcessingConfig()
        self.prompt_manager = PromptManager(prompts_config)
        self.batch_processor = BatchProcessor(self.config)
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        
        # Build the processing chain
        self.chain = self._build_chain()
        
    def _build_chain(self):
        """Build the three-stage processing chain"""
        
        # Stage 1: Initial code generation
        initial_prompt = self.prompt_manager.get_initial_prompt()
        
        # Stage 2: Review and refinement
        review_prompt = self.prompt_manager.get_review_prompt()
        
        # Output parser for final recommendation
        output_parser = JsonOutputParser(pydantic_object=CodeRecommendation)
        
        def extract_initial_suggestion(response: str) -> Dict[str, Any]:
            """Extract code and definition from initial response"""
            lines = response.strip().split('\n')
            code = None
            definition = None
            
            for i, line in enumerate(lines):
                if line.strip().startswith("Code:"):
                    code = line.replace("Code:", "").strip()
                elif line.strip().startswith("Definition:"):
                    definition = line.replace("Definition:", "").strip()
            
            return {
                "code": code or "No code suggested",
                "definition": definition or "No definition provided",
                "initial_response": response
            }
        
        def prepare_review_input(data: Dict[str, Any]) -> Dict[str, Any]:
            """Prepare input for review stage"""
            return {
                "existing_codebook": data["existing_codebook"],
                "survey_question": data["survey_question"],
                "clustered_responses": data["clustered_responses"],
                "code": data["code"],
                "definition": data["definition"]
            }
        
        # Complete chain
        chain = (
            # Stage 1: Initial code generation
            initial_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(extract_initial_suggestion)
            | RunnablePassthrough.assign(
                existing_codebook=lambda x: x.get("existing_codebook", ""),
                survey_question=lambda x: x.get("survey_question", ""),
                clustered_responses=lambda x: x.get("clustered_responses", "")
            )
            # Stage 2: Review and refinement
            | RunnableLambda(prepare_review_input)
            | review_prompt
            | self.llm
            | output_parser
        )
        
        return chain
    
    async def process_batch(
        self,
        batch: Batch,
        existing_codebook: str,
        survey_question: str
    ) -> List[ProcessedResponse]:
        """Process a single batch of responses"""
        
        # Combine responses for batch processing
        clustered_responses = "\n\n".join([
            f"Respondent {r.respondent_id}: {r.response}"
            for r in batch.responses
        ])
        
        try:
            # Run the chain
            result = await self.chain.ainvoke({
                "existing_codebook": existing_codebook,
                "survey_question": survey_question,
                "clustered_responses": clustered_responses
            })
            
            # Create processed responses
            processed = []
            for response in batch.responses:
                processed.append(ProcessedResponse(
                    respondent_id=response.respondent_id,
                    original_response=response.response,
                    code=result.code if result.needs_new_code else None,
                    definition=result.definition if result.needs_new_code else None,
                    reasoning=result.reasoning,
                    needs_new_code=result.needs_new_code,
                    processing_metadata={
                        "batch_id": batch.batch_id,
                        "timestamp": datetime.now().isoformat()
                    }
                ))
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {str(e)}")
            # Return error responses
            return [
                ProcessedResponse(
                    respondent_id=r.respondent_id,
                    original_response=r.response,
                    processing_metadata={
                        "batch_id": batch.batch_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                for r in batch.responses
            ]
    
    async def process_responses(
        self,
        responses: List[SurveyResponse],
        existing_codebook: str,
        survey_question: str
    ) -> List[ProcessedResponse]:
        """Process all survey responses"""
        
        # Create batches
        batches = self.batch_processor.create_batches(
            responses, 
            existing_codebook, 
            survey_question
        )
        
        if not batches:
            return []
        
        # Process batches concurrently
        tasks = [
            self.process_batch(batch, existing_codebook, survey_question)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing exception: {result}")
                continue
            all_results.extend(result)
        
        if self.verbose:
            self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[ProcessedResponse]):
        """Print processing summary"""
        total = len(results)
        needs_new_code = sum(1 for r in results if r.needs_new_code)
        errors = sum(1 for r in results if "error" in r.processing_metadata)
        
        print("\n--- Processing Summary ---")
        print(f"Total responses: {total}")
        print(f"Responses needing new codes: {needs_new_code}")
        print(f"Processing errors: {errors}")
        
        # Show unique codes
        unique_codes = set()
        for r in results:
            if r.code:
                unique_codes.add(r.code)
        
        if unique_codes:
            print(f"\nUnique codes suggested: {len(unique_codes)}")
            for code in list(unique_codes)[:5]:  # Show first 5
                print(f"  - {code}")
            if len(unique_codes) > 5:
                print(f"  ... and {len(unique_codes) - 5} more")


