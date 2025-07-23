import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

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
from config import EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
import models
from utils.embedder import Embedder
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

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
            logger.error(f"Error embedding codebook: {str(e)}")
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


class InductiveCodebookGenerator:
    """Main class for inductive codebook generation using LangChain pipeline"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel], 
        embedded_text: List[models.EmbeddingsModel],
        starter_codes: List[Dict[str, str]], 
        var_lab: str, 
        k: int = 5,
        verbose: bool = False, 
        prompt_printer = None
    ):
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        
        # Initialize data processor
        self.data_processor = PromptInputData(
            cluster_results=cluster_results,
            embedded_text=embedded_text, 
            starter_codes=starter_codes,
            var_lab=var_lab,
            k=k
        )
        
        # Processing config
        self.config = ProcessingConfig()
        self.model_config = ModelConfig()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_phase("phase1_descriptive"),
            temperature=self.config.temperature
        )
        
        # Track statistics
        self.stats = {
            'total_clusters': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'errors': 0
        }
        
    def _create_chain_for_cluster(self, code_text: str, cluster_text: str) -> Any:
        """Create LangChain processing chain for a specific cluster"""
        
        # Build the processing chain
        system_message = SYSTEM_MESSAGE_CODEBOOK.format(language=DEFAULT_LANGUAGE)
        
        # Stage 1: Initial code generation prompt
        initial_prompt_text = INITIAL_CODEBOOK_GENERATION.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            code_text=code_text,
            cluster_text=cluster_text
        )
        
        initial_prompt = PromptTemplate.from_template(system_message + "\n\n" + initial_prompt_text)
        
        def extract_initial_suggestion(response: str) -> Dict[str, Any]:
            """Extract code and definition from initial response"""
            lines = response.strip().split('\n')
            code = None
            definition = None
            
            for line in lines:
                if line.strip().startswith("Code:"):
                    code = line.replace("Code:", "").strip()
                elif line.strip().startswith("Definition:"):
                    definition = line.replace("Definition:", "").strip()
            
            return {
                "code": code or "No code suggested",
                "definition": definition or "No definition provided",
                "initial_response": response,
                "cluster_text": cluster_text,
                "original_code_text": code_text
            }
        
        def create_review_prompt(data: Dict[str, Any]) -> PromptTemplate:
            """Create review prompt with suggested code"""
            # Format the code suggestion for review
            if data.get("code") and data.get("definition"):
                review_code_text = f"Suggested new code:\n- {data['code']}: {data['definition']}\n\nExisting codes:\n{data['original_code_text']}"
            else:
                review_code_text = f"No new code suggested.\n\nExisting codes:\n{data['original_code_text']}"
            
            review_prompt_text = REVIEW_CODEBOOK_GENERATION.format(
                language=DEFAULT_LANGUAGE,
                survey_question=self.var_lab,
                code_text=review_code_text,
                cluster_text=data['cluster_text']
            )
            
            # Add JSON format instruction
            review_prompt_text += "\n\nReturn your final recommendation in the following JSON format:\n"
            review_prompt_text += "{\n"
            review_prompt_text += '  "needs_new_code": true/false,\n'
            review_prompt_text += '  "code": "The code name (if needs_new_code is true)",\n'
            review_prompt_text += '  "definition": "The code definition (if needs_new_code is true)",\n'
            review_prompt_text += '  "reasoning": "Your reasoning for this decision"\n'
            review_prompt_text += "}"
            
            return PromptTemplate.from_template(system_message + "\n\n" + review_prompt_text)
        
        # Complete chain: Initial generation -> Review -> Final decision  
        chain = (
            initial_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(extract_initial_suggestion)
            | RunnableLambda(create_review_prompt)
            | self.llm
            | RunnableLambda(lambda response: response.content if hasattr(response, 'content') else str(response))
            | RunnableLambda(self._parse_final_response)
        )
        
        return chain
    
    def _parse_final_response(self, response_text: str) -> CodeRecommendation:
        """Parse the final response into CodeRecommendation"""
        try:
            # Try to parse as JSON first
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    json_data = json.loads(json_str)
                    return CodeRecommendation(**json_data)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract decision from text
            needs_new_code = False
            code = None
            definition = None
            reasoning = response_text[:500]
            
            # Simple heuristics to determine if new code is needed
            if "needs_new_code" in response_text.lower():
                needs_new_code = "true" in response_text.lower()
            elif "new code" in response_text.lower() and "no" not in response_text.lower():
                needs_new_code = True
            
            # Try to extract code and definition
            lines = response_text.split('\n')
            for line in lines:
                if 'code' in line.lower() and ':' in line:
                    code = line.split(':', 1)[1].strip().strip('"')
                elif 'definition' in line.lower() and ':' in line:
                    definition = line.split(':', 1)[1].strip().strip('"')
            
            return CodeRecommendation(
                needs_new_code=needs_new_code,
                code=code,
                definition=definition,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return CodeRecommendation(
                needs_new_code=False,
                reasoning="Failed to parse response"
            )
    
    async def _process_cluster(self, cluster_id: int, cluster_data: Dict) -> Optional[CodeRecommendation]:
        """Process a single cluster through the LangChain pipeline"""
        try:
            # Calculate cluster embedding (mean of all ideas in cluster)
            cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
            
            # Get current codebook embeddings
            code_texts = [f"{code['code']}: {code['definition']}" for code in self.data_processor.codebook]
            codebook_embeddings = await self.data_processor._embed_codebook_texts(code_texts)
            
            if not codebook_embeddings:
                logger.warning(f"No codebook embeddings available for cluster {cluster_id}")
                return None
            
            # Find k nearest codes
            nearest_codes = self.data_processor._find_k_nearest_codes(cluster_embedding, codebook_embeddings)
            
            # Prepare texts for processing
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            code_text = "\n".join([f"- {code['code']}: {code['definition']}" for code in nearest_codes]) if nearest_codes else "No existing codes"
            
            # Capture prompt for debugging (first cluster only)
            if self.prompt_printer and cluster_id == 0:
                prompt_content = f"Existing codes:\n{code_text}\n\nCluster text:\n{cluster_text}"
                self.prompt_printer.capture_prompt(
                    step_name="gatos_codebook", 
                    utility_name="InductiveCodebookGenerator",
                    prompt_content=prompt_content,
                    prompt_type="LangChain Codebook Generation"
                )
            
            # Create and run chain for this cluster
            chain = self._create_chain_for_cluster(code_text, cluster_text)
            result = await chain.ainvoke({})
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
            self.stats['errors'] += 1
            return None
    
    def generate(self) -> Dict[str, Any]:
        """Generate inductive codebook using LangChain batch processing"""
        from utils.verboseReporter import VerboseReporter
        
        verbose_reporter = VerboseReporter(self.verbose)
        verbose_reporter.section_header("INDUCTIVE CODEBOOK GENERATION (LANGCHAIN)")
        
        # Prepare clusters from the data processor
        clusters = self.data_processor._prepare_cluster_text()
        
        if not clusters:
            verbose_reporter.stat_line("No valid clusters found with embeddings")
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': {},
                'stats': self.stats
            }
        
        self.stats['total_clusters'] = len(clusters)
        cluster_ids = sorted(clusters.keys())
        
        verbose_reporter.stat_line(f"Processing {len(cluster_ids)} clusters with batch processing")
        
        # Track cluster assignments
        cluster_to_code = {}
        
        # Process clusters iteratively (GATOS methodology)
        for i, cluster_id in enumerate(cluster_ids):
            if self.verbose and i % 10 == 0 and i > 0:
                verbose_reporter.stat_line(
                    f"Progress: {i}/{len(cluster_ids)} clusters processed "
                    f"({self.stats['new_codes_added']} new codes added)"
                )
            
            # Get current codebook embeddings for k-nearest neighbor search
            code_texts = [f"{code['code']}: {code['definition']}" for code in self.data_processor.codebook]
            codebook_embeddings = asyncio.run(self.data_processor._embed_codebook_texts(code_texts))
            
            if not codebook_embeddings:
                verbose_reporter.stat_line(f"Warning: Failed to embed codebook for cluster {cluster_id}")
                cluster_to_code[cluster_id] = "embedding_error"
                continue
            
            # Calculate cluster embedding (mean of all ideas in cluster)
            cluster_data = clusters[cluster_id]
            cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
            
            # Find k nearest codes
            nearest_codes = self.data_processor._find_k_nearest_codes(cluster_embedding, codebook_embeddings)
            
            # Prepare cluster data for processing
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            code_text = "\n".join([
                f"- {code['code']}: {code['definition']}" 
                for code in nearest_codes
            ]) if nearest_codes else "No existing codes in codebook"
            
            # Convert cluster to SurveyResponse format for batch processing
            survey_responses = [SurveyResponse(
                respondent_id=str(cluster_id),
                response=cluster_text
            )]
            
            # Create batch processor and pipeline
            batch_processor = BatchProcessor(self.config)
            
            # Create batches based on token estimates
            batches = batch_processor.create_batches(
                responses=survey_responses,
                existing_codebook=code_text,
                survey_question=self.var_lab
            )
            
            if not batches:
                cluster_to_code[cluster_id] = "batch_error"
                continue
            
            # Process the batch through LangChain pipeline
            try:
                # Create prompt manager for this cluster
                prompt_manager = PromptManager(
                    var_lab=self.var_lab,
                    code_text=code_text,
                    cluster_text=cluster_text,
                    code_recommendation=[]
                )
                
                # Create survey response pipeline
                pipeline = SurveyResponsePipeline(
                    api_key=OPENAI_API_KEY,
                    prompts_config={
                        'var_lab': self.var_lab,
                        'code_text': code_text,
                        'cluster_text': cluster_text
                    },
                    config=self.config,
                    verbose=False
                )
                
                # Process the batch
                processed_responses = asyncio.run(
                    pipeline.process_responses(
                        responses=survey_responses,
                        existing_codebook=code_text,
                        survey_question=self.var_lab
                    )
                )
                
                # Extract result
                if processed_responses and len(processed_responses) > 0:
                    result = processed_responses[0]
                    
                    if result.needs_new_code and result.code and result.definition:
                        # Add new code to codebook
                        new_code = {
                            'code': result.code,
                            'definition': result.definition,
                            'cluster_origin': str(cluster_id)
                        }
                        self.data_processor.codebook.append(new_code)
                        cluster_to_code[cluster_id] = result.code
                        self.stats['new_codes_added'] += 1
                        
                        if self.verbose:
                            print(f"  New code added for cluster {cluster_id}: '{result.code}'")
                    else:
                        # No new code needed
                        cluster_to_code[cluster_id] = "existing_code"
                        self.stats['no_new_codes_needed'] += 1
                else:
                    cluster_to_code[cluster_id] = "no_response"
                    
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
                cluster_to_code[cluster_id] = "processing_error"
                self.stats['errors'] += 1
        
        # Final summary
        verbose_reporter.summary("CODEBOOK GENERATION COMPLETE", {
            "Initial codes": len(self.starter_codes),
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.data_processor.codebook),
            "Clusters processed": len(cluster_ids),
            "Processing errors": self.stats['errors']
        })
        
        return {
            'codebook': self.data_processor.codebook,
            'cluster_assignments': cluster_to_code,
            'stats': {
                'initial_codes': len(self.starter_codes),
                'new_codes': self.stats['new_codes_added'],
                'total_codes': len(self.data_processor.codebook),
                'clusters_processed': len(cluster_ids),
                'no_new_codes_needed': self.stats['no_new_codes_needed'],
                'errors': self.stats['errors']
            }
        }


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
        self.prompts_config = prompts_config
        self.batch_processor = BatchProcessor(self.config)
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        
        # Build the processing chain will be done per batch
        # since we need different prompts for each cluster
        
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
            # Create prompt manager for this batch
            prompt_manager = PromptManager(
                var_lab=survey_question,
                code_text=existing_codebook,
                cluster_text=clustered_responses,
                code_recommendation=[]
            )
            
            # Build chain for this batch  
            chain = self._build_chain_for_batch(prompt_manager)
            
            # Run the chain
            result = await chain.ainvoke({})
            
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
    
    def _build_chain_for_batch(self, prompt_manager: PromptManager):
        """Build processing chain for a specific batch"""
        
        # Stage 1: Initial code generation
        initial_prompt = prompt_manager.get_initial_prompt()
        
        # Stage 2: Review and refinement  
        review_prompt = prompt_manager.get_review_prompt()
        
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
                "existing_codebook": data.get("existing_codebook", ""),
                "survey_question": data.get("survey_question", ""),
                "clustered_responses": data.get("clustered_responses", ""),
                "code": data.get("code"),
                "definition": data.get("definition")
            }
        
        # Complete chain
        chain = (
            # Stage 1: Initial code generation
            initial_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(extract_initial_suggestion)
            | RunnablePassthrough.assign(
                existing_codebook=lambda x: "",
                survey_question=lambda x: "",
                clustered_responses=lambda x: ""
            )
            # Stage 2: Review and refinement
            | RunnableLambda(prepare_review_input)
            | review_prompt
            | self.llm
            | RunnableLambda(lambda response: response.content if hasattr(response, 'content') else str(response))
            | RunnableLambda(self._parse_response_to_recommendation)
        )
        
        return chain
    
    def _parse_response_to_recommendation(self, response_text: str) -> CodeRecommendation:
        """Parse response to CodeRecommendation"""
        try:
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    json_data = json.loads(json_str)
                    return CodeRecommendation(**json_data)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract decision from text
            needs_new_code = "new code" in response_text.lower() and "no" not in response_text.lower()
            
            return CodeRecommendation(
                needs_new_code=needs_new_code,
                reasoning=response_text[:500]
            )
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return CodeRecommendation(
                needs_new_code=False,
                reasoning="Failed to parse response"
            )
    
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


