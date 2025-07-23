import os
import sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import time
from pydantic import BaseModel, Field
import hashlib
from enum import Enum

# Retry logic imports
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log,
    RetryError
)

from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from prompts import SYSTEM_MESSAGE_CODEBOOK, INITIAL_CODEBOOK_GENERATION, REVIEW_CODEBOOK_GENERATION
from config import EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
import models
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").disabled = True
logging.getLogger("tenacity").setLevel(logging.WARNING)  # Reduce tenacity noise

# ============================================================================
# ERROR HANDLING AND RETRY CONFIGURATION
# ============================================================================

class ErrorType(Enum):
    """Categorize different types of errors for appropriate handling"""
    API_RATE_LIMIT = "api_rate_limit"
    API_TIMEOUT = "api_timeout"
    API_SERVER_ERROR = "api_server_error"
    NETWORK_ERROR = "network_error"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"

class RetryableError(Exception):
    """Base class for retryable errors"""
    def __init__(self, message: str, error_type: ErrorType):
        super().__init__(message)
        self.error_type = error_type

class APIError(RetryableError):
    """API-related errors that should be retried"""
    pass

class ProcessingError(Exception):
    """Non-retryable processing errors"""
    def __init__(self, message: str, error_type: ErrorType):
        super().__init__(message)
        self.error_type = error_type

@dataclass
def classify_error(error: Exception) -> ErrorType:
    """Classify errors for appropriate retry behavior"""
    error_str = str(error).lower()
    
    if "rate limit" in error_str or "429" in error_str:
        return ErrorType.API_RATE_LIMIT
    elif "timeout" in error_str:
        return ErrorType.API_TIMEOUT
    elif "500" in error_str or "502" in error_str or "503" in error_str:
        return ErrorType.API_SERVER_ERROR
    elif "connection" in error_str or "network" in error_str:
        return ErrorType.NETWORK_ERROR
    elif "parsing" in error_str or "json" in error_str:
        return ErrorType.PARSING_ERROR
    elif "validation" in error_str:
        return ErrorType.VALIDATION_ERROR
    else:
        return ErrorType.UNKNOWN_ERROR

# Retry configurations for different error types
API_RETRY_CONFIG = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential(multiplier=2, min=1, max=30),
    "retry": retry_if_exception_type((APIError, asyncio.TimeoutError, ConnectionError)),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True
}

EMBEDDING_RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=2, max=10),
    "retry": retry_if_exception_type((APIError, asyncio.TimeoutError, ConnectionError)),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True
}

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class CodeSuggestion(BaseModel):
    """Structured output for initial code suggestion"""
    needs_new_code: bool = Field(description="Whether a new code is needed")
    code: Optional[str] = Field(default=None, description="The suggested code name")
    definition: Optional[str] = Field(default=None, description="The code definition")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the decision")

class CodeReview(BaseModel):
    """Structured output for code review"""
    approve_new_code: bool = Field(description="Whether to approve the new code")
    final_code: Optional[str] = Field(default=None, description="The final code name")
    final_definition: Optional[str] = Field(default=None, description="The final definition")
    revision_notes: Optional[str] = Field(default=None, description="Notes on any revisions made")

class BatchCodeSuggestions(BaseModel):
    """Batch processing multiple clusters for initial suggestions"""
    suggestions: List[CodeSuggestion] = Field(description="Code suggestions for multiple clusters")

class BatchCodeReviews(BaseModel):
    """Batch processing multiple clusters for reviews"""
    reviews: List[CodeReview] = Field(description="Code reviews for multiple clusters")

class ClusterInput(BaseModel):
    """Input data for a single cluster"""
    cluster_id: int
    cluster_text: str
    nearest_codes: str

# ============================================================================
# SHARED CODEBOOK WITH REAL-TIME UPDATES
# ============================================================================

@dataclass
class SharedCodebook:
    """Thread-safe shared codebook with async lock and version tracking"""
    _codes: List[Dict[str, str]]
    _lock: asyncio.Lock
    _version: int = 0
    _update_log: List[Dict[str, Any]] = None
    _embedding_cache: Dict[str, np.ndarray] = None
    
    def __init__(self, initial_codes: List[Dict[str, str]]):
        self._codes = initial_codes.copy()
        self._lock = asyncio.Lock()
        self._version = 0
        self._update_log = []
        self._embedding_cache = {}
    
    async def get_current_snapshot(self) -> Tuple[List[Dict[str, str]], int]:
        """Get current codes and version atomically"""
        async with self._lock:
            return self._codes.copy(), self._version
    
    async def add_code_if_new(self, code: str, definition: str) -> Tuple[bool, int]:
        """Add a new code if it doesn't exist, return (added, new_version)"""
        async with self._lock:
            # Check if code already exists
            for existing in self._codes:
                if existing['code'].lower() == code.lower():
                    return False, self._version
            
            # Add new code
            self._codes.append({'code': code, 'definition': definition})
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add',
                'code': code,
                'timestamp': time.time()
            })
            return True, self._version
    
    async def get_embeddings_for_version(self, version: int) -> Optional[List[np.ndarray]]:
        """Get cached embeddings for a specific version"""
        async with self._lock:
            return self._embedding_cache.get(version)
    
    async def cache_embeddings(self, version: int, embeddings: List[np.ndarray]):
        """Cache embeddings for a version"""
        async with self._lock:
            self._embedding_cache[version] = embeddings
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get codebook statistics"""
        async with self._lock:
            return {
                'total_codes': len(self._codes),
                'version': self._version,
                'updates': len(self._update_log),
                'cached_versions': len(self._embedding_cache)
            }

# ============================================================================
# OPTIMIZED EMBEDDING MANAGER
# ============================================================================

class OptimizedEmbeddingManager:
    """Manages embeddings with shared codebook integration"""
    
    def __init__(self, shared_codebook: SharedCodebook, verbose: bool = False):
        self.shared_codebook = shared_codebook
        self.embedding_config = EmbeddingConfig()
        self.verbose = verbose
        self._individual_cache: Dict[str, np.ndarray] = {}  # Individual text cache like v2
        self._code_text_cache: Dict[str, np.ndarray] = {}   # Specific cache for code texts
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_code_hash(self, code: str, definition: str) -> str:
        """Generate hash for code definition pair"""
        combined = f"{code}: {definition}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    async def get_embeddings_for_current_codebook(self) -> Tuple[List[Dict[str, str]], List[np.ndarray], int]:
        """Get embeddings for the current codebook state"""
        # Get current snapshot
        codes, version = await self.shared_codebook.get_current_snapshot()
        
        # Check if we have cached embeddings for this version
        cached_embeddings = await self.shared_codebook.get_embeddings_for_version(version)
        if cached_embeddings is not None:
            return codes, cached_embeddings, version
        
        # Generate embeddings for new version
        code_texts = [f"{code['code']}: {code['definition']}" for code in codes]
        embeddings = await self._embed_texts(code_texts)
        
        # Cache for this version
        await self.shared_codebook.cache_embeddings(version, embeddings)
        
        return codes, embeddings, version
    
    async def get_embeddings_for_codes_individually(self, codes: List[Dict[str, str]]) -> List[np.ndarray]:
        """Get embeddings for codes individually with aggressive caching (like v2)"""
        embeddings = []
        new_texts = []
        new_indices = []
        
        for i, code in enumerate(codes):
            code_hash = self._get_code_hash(code['code'], code['definition'])
            if code_hash in self._code_text_cache:
                embeddings.append((i, self._code_text_cache[code_hash]))
            else:
                code_text = f"{code['code']}: {code['definition']}"
                new_texts.append(code_text)
                new_indices.append((i, code_hash))
        
        # Embed new code texts if any
        if new_texts:
            try:
                new_embeddings = await self._embed_texts_with_retry(new_texts)
                
                # Cache new embeddings with both individual and code caches
                for j, embedding in enumerate(new_embeddings):
                    text = new_texts[j]
                    idx, code_hash = new_indices[j]
                    
                    # Cache in both caches
                    text_hash = self._get_text_hash(text)
                    self._individual_cache[text_hash] = embedding
                    self._code_text_cache[code_hash] = embedding
                    
                    embeddings.append((idx, embedding))
                    
            except (APIError, ProcessingError, RetryError) as e:
                logger.error(f"Failed to embed code texts: {str(e)}")
                return []
        
        # Sort by index and return
        embeddings.sort(key=lambda x: x[0])
        return [emb[1] for emb in embeddings]
    
    @retry(**EMBEDDING_RETRY_CONFIG)
    async def _embed_texts_with_retry(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts with retry logic for API failures"""
        try:
            client = AsyncOpenAI(api_key=os.environ.get(OPENAI_API_KEY))
            response = await client.embeddings.create(
                model=self.embedding_config.embedding_model,
                input=texts,
                timeout=30.0  # Add explicit timeout
            )
            
            embeddings = []
            for embedding_data in response.data:
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                raise APIError(f"Embedding API error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Embedding processing error: {str(e)}", error_type)
    
    async def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts with caching and retry logic"""
        # Check cache first
        embeddings = []
        new_texts = []
        new_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self._individual_cache:
                embeddings.append((i, self._individual_cache[text_hash]))
            else:
                new_texts.append(text)
                new_indices.append(i)
        
        # Embed new texts if any
        if new_texts:
            try:
                new_embeddings = await self._embed_texts_with_retry(new_texts)
                
                # Cache new embeddings
                for j, embedding in enumerate(new_embeddings):
                    text = new_texts[j]
                    text_hash = self._get_text_hash(text)
                    self._individual_cache[text_hash] = embedding
                    embeddings.append((new_indices[j], embedding))
                    
            except (APIError, ProcessingError) as e:
                logger.error(f"Failed to embed texts after retries: {str(e)}")
                return []
            except RetryError as e:
                logger.error(f"Retry exhausted for embedding: {str(e)}")
                return []
        
        # Sort by index and return
        embeddings.sort(key=lambda x: x[0])
        return [emb[1] for emb in embeddings]

# ============================================================================
# LANGCHAIN BATCH PROCESSOR
# ============================================================================

class LangChainBatchProcessor:
    """Processes clusters using LangChain's efficient batch capabilities"""
    
    def __init__(self, 
                 embedding_manager: OptimizedEmbeddingManager,
                 shared_codebook: SharedCodebook,
                 model_config: ModelConfig,
                 var_lab: str,
                 k: int = 5,
                 batch_size: int = 5,
                 max_concurrent_requests: int = 10,
                 verbose: bool = False,
                 prompt_printer = None):
        
        self.embedding_manager = embedding_manager
        self.shared_codebook = shared_codebook
        self.model_config = model_config
        self.var_lab = var_lab
        self.k = k
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        
        # Initialize LangChain components
        self._init_langchain_chain()
        
        # Stats tracking
        self.stats = {
            'clusters_processed': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'errors': 0,
            'retries': 0,
            'partial_failures': 0,
            'successful_recoveries': 0,
            'llm_time': 0.0,
            'embedding_time': 0.0
        }
    
    def _init_langchain_chain(self):
        """Initialize proper two-stage LangChain chains"""
        # Stage 1: Initial code generation
        self.initial_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("initial_codes"),
            temperature=0.0
        )
        
        # Stage 2: Review chain
        self.review_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("review_codes"),
            temperature=0.0
        )
        
        # Parsers for structured output
        self.suggestion_parser = PydanticOutputParser(pydantic_object=CodeSuggestion)
        self.review_parser = PydanticOutputParser(pydantic_object=CodeReview)
        
        # Initial suggestion chain - using PROPER prompts
        initial_prompt = PromptTemplate(
            template=SYSTEM_MESSAGE_CODEBOOK + "\n\n" + INITIAL_CODEBOOK_GENERATION + "\n\n{format_instructions}",
            input_variables=["language", "survey_question", "code_text", "cluster_text", "data type"],
            partial_variables={"format_instructions": self.suggestion_parser.get_format_instructions()}
        )
        
        self.initial_chain = (
            initial_prompt 
            | self.initial_llm 
            | self.suggestion_parser
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Review chain - using PROPER prompts  
        review_prompt = PromptTemplate(
            template=SYSTEM_MESSAGE_CODEBOOK + "\n\n" + REVIEW_CODEBOOK_GENERATION + "\n\n{format_instructions}",
            input_variables=["language", "survey_question", "code_text", "cluster_text"],
            partial_variables={"format_instructions": self.review_parser.get_format_instructions()}
        )
        
        self.review_chain = (
            review_prompt
            | self.review_llm
            | self.review_parser
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Capture prompt function (optional)
        def capture_initial_prompt(inputs):
            if self.prompt_printer and hasattr(self, '_capture_initial_count'):
                if self._capture_initial_count < 2:  # Only capture first few prompts
                    self.prompt_printer.capture_prompt(
                        step_name="codebook_generation_v3",
                        utility_name="LangChainBatchProcessor",
                        prompt_content=str(inputs),
                        prompt_type="initial_code_suggestion",
                        metadata={
                            "model": self.initial_llm.model_name,
                            "var_lab": self.var_lab,
                            "stage": "1/2 - Initial Code Suggestion"
                        }
                    )
                    self._capture_initial_count += 1
            return inputs
        
        def capture_review_prompt(inputs):
            if self.prompt_printer and hasattr(self, '_capture_review_count'):
                if self._capture_review_count < 2:  # Only capture first few prompts
                    self.prompt_printer.capture_prompt(
                        step_name="codebook_generation_v3",
                        utility_name="LangChainBatchProcessor",
                        prompt_content=str(inputs),
                        prompt_type="code_review",
                        metadata={
                            "model": self.review_llm.model_name,
                            "var_lab": self.var_lab,
                            "stage": "2/2 - Code Review"
                        }
                    )
                    self._capture_review_count += 1
            return inputs
        
        self._capture_initial_count = 0
        self._capture_review_count = 0
        
        # Add prompt capture to chains
        if self.prompt_printer:
            self.initial_chain = RunnableLambda(capture_initial_prompt) | self.initial_chain
            self.review_chain = RunnableLambda(capture_review_prompt) | self.review_chain
    
    async def _find_nearest_codes(self, cluster_embedding: np.ndarray, 
                                 codebook_embeddings: List[np.ndarray], 
                                 codes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Find k nearest codes to cluster embedding"""
        if not codebook_embeddings:
            return []
        
        # Calculate similarities
        codebook_array = np.array(codebook_embeddings)
        similarities = cosine_similarity(cluster_embedding.reshape(1, -1), codebook_array)[0]
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        
        # Get unique codes
        seen = set()
        nearest_codes = []
        
        for idx in top_k_indices:
            if idx < len(codes):
                code = codes[idx]
                code_text = code.get('code', '')
                
                if code_text not in seen:
                    seen.add(code_text)
                    nearest_codes.append(code)
                    
                    if len(nearest_codes) >= self.k:
                        break
        
        return nearest_codes
    
    @retry(**API_RETRY_CONFIG)
    async def _process_initial_batch_with_retry(self, batch_inputs: List[Dict]) -> List[Any]:
        """Process initial suggestions with retry logic"""
        try:
            results = await self.initial_chain.abatch(batch_inputs)
            return results
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                self.stats['retries'] += 1
                raise APIError(f"Initial batch processing error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Initial batch processing error: {str(e)}", error_type)
    
    @retry(**API_RETRY_CONFIG)
    async def _process_review_batch_with_retry(self, review_inputs: List[Dict]) -> List[Any]:
        """Process review suggestions with retry logic"""
        try:
            results = await self.review_chain.abatch(review_inputs)
            return results
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                self.stats['retries'] += 1 
                raise APIError(f"Review batch processing error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Review batch processing error: {str(e)}", error_type)
    
    async def _retry_individual_failures(self, failed_inputs: List[Tuple[int, Dict]], 
                                        is_review: bool = False) -> Dict[int, Any]:
        """Retry individual failed items from a batch"""
        recovery_results = {}
        
        for original_idx, input_data in failed_inputs:
            try:
                if is_review:
                    result = await self.review_chain.ainvoke(input_data)
                else:
                    result = await self.initial_chain.ainvoke(input_data)
                
                recovery_results[original_idx] = result
                self.stats['successful_recoveries'] += 1
                
                if self.verbose:
                    logger.info(f"Successfully recovered {'review' if is_review else 'initial'} processing for item {original_idx}")
                    
            except Exception as e:
                logger.error(f"Individual retry failed for item {original_idx}: {str(e)}")
                recovery_results[original_idx] = e
        
        return recovery_results
    
    async def _prepare_cluster_input_with_current_codebook(self, cluster_id: int, cluster_data: Dict) -> Tuple[Dict, Dict]:
        """Prepare input for a single cluster with current codebook state"""
        start_time = time.time()
        
        # Calculate cluster embedding
        cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
        
        # Get CURRENT codebook embeddings (this is key - fresh each time)
        embed_start = time.time()
        codes, codebook_embeddings, version = await self.embedding_manager.get_embeddings_for_current_codebook()
        self.stats['embedding_time'] += time.time() - embed_start
        
        # Find nearest codes
        nearest_codes = await self._find_nearest_codes(cluster_embedding, codebook_embeddings, codes)
        
        # Prepare inputs
        cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas'][:20]])
        code_text = "\n".join([
            f"- {code['code']}: {code['definition']}" 
            for code in nearest_codes
        ]) if nearest_codes else "No existing codes in codebook"
        
        batch_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "code_text": code_text,
            "cluster_text": cluster_text,
            "data type": "survey responses"
        }
        
        cluster_info = {
            'cluster_id': cluster_id,
            'start_time': start_time,
            'codebook_version': version
        }
        
        return batch_input, cluster_info
    
    async def process_batch_langchain(self, batch_clusters: List[Tuple[int, Dict]]) -> List[Dict[str, Any]]:
        """Process a batch of clusters using LangChain's batch capabilities with real-time codebook updates"""
        batch_results = []
        batch_inputs = []
        cluster_map = {}
        
        # Prepare inputs individually to see real-time codebook updates
        for i, (cluster_id, cluster_data) in enumerate(batch_clusters):
            try:
                batch_input, cluster_info = await self._prepare_cluster_input_with_current_codebook(cluster_id, cluster_data)
                batch_inputs.append(batch_input)
                cluster_map[i] = cluster_info
                
                if self.verbose:
                    logger.info(f"Prepared cluster {cluster_id} with codebook v{cluster_info['codebook_version']}")
                
            except Exception as e:
                logger.error(f"V3 PREPARATION ERROR for cluster {cluster_id}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                batch_results.append({
                    'cluster_id': cluster_id,
                    'status': 'v3_preparation_error',  # Clear V3 identifier
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'error_details': f"Failed during input preparation: {str(e)}"
                })
        
        if not batch_inputs:
            return batch_results
        
        # Process batch using proper two-stage LangChain approach with retry logic
        try:
            llm_start = time.time()
            
            # Stage 1: Initial suggestions using abatch with retry
            try:
                initial_results = await self._process_initial_batch_with_retry(batch_inputs)
            except (APIError, ProcessingError) as e:
                logger.warning(f"Initial batch failed, attempting individual recovery: {str(e)}")
                self.stats['partial_failures'] += 1
                
                # Attempt individual recovery
                failed_inputs = [(i, inp) for i, inp in enumerate(batch_inputs)]
                recovery_results = await self._retry_individual_failures(failed_inputs, is_review=False)
                
                # Rebuild results array
                initial_results = []
                for i in range(len(batch_inputs)):
                    if i in recovery_results:
                        initial_results.append(recovery_results[i])
                    else:
                        initial_results.append(Exception(f"Failed to process cluster after retries"))
            
            # Prepare review inputs for clusters that need new codes
            review_inputs = []
            review_cluster_map = {}
            
            for idx, initial_result in enumerate(initial_results):
                cluster_info = cluster_map[idx]
                cluster_id = cluster_info['cluster_id']
                
                # Handle failed initial results
                if isinstance(initial_result, Exception):
                    logger.error(f"Initial processing failed for cluster {cluster_id}: {str(initial_result)}")
                    batch_results.append({
                        'cluster_id': cluster_id,
                        'status': 'v3_initial_stage_error',  # Clear V3 identifier
                        'error': str(initial_result),
                        'error_type': type(initial_result).__name__,
                        'processing_time': time.time() - cluster_info['start_time']
                    })
                    self.stats['errors'] += 1
                    continue
                
                if not initial_result.needs_new_code:
                    # No new code needed
                    self.stats['no_new_codes_needed'] += 1
                    batch_results.append({
                        'cluster_id': cluster_id,
                        'status': 'no_new_code_needed',
                        'processing_time': time.time() - cluster_info['start_time']
                    })
                    self.stats['clusters_processed'] += 1
                else:
                    # Prepare for review stage
                    original_input = batch_inputs[idx]
                    review_code_text = f"Suggested new code:\n- {initial_result.code}: {initial_result.definition}\n\nExisting codes:\n{original_input['code_text']}"
                    
                    review_inputs.append({
                        "language": original_input["language"],
                        "survey_question": original_input["survey_question"],
                        "code_text": review_code_text,
                        "cluster_text": original_input["cluster_text"]
                    })
                    
                    review_cluster_map[len(review_inputs) - 1] = {
                        'cluster_id': cluster_id,
                        'cluster_info': cluster_info,
                        'initial_result': initial_result
                    }
            
            # Stage 2: Review suggestions using abatch with retry
            if review_inputs:
                try:
                    review_results = await self._process_review_batch_with_retry(review_inputs)
                except (APIError, ProcessingError) as e:
                    logger.warning(f"Review batch failed, attempting individual recovery: {str(e)}")
                    self.stats['partial_failures'] += 1
                    
                    # Attempt individual recovery
                    failed_inputs = [(i, inp) for i, inp in enumerate(review_inputs)]
                    recovery_results = await self._retry_individual_failures(failed_inputs, is_review=True)
                    
                    # Rebuild results array
                    review_results = []
                    for i in range(len(review_inputs)):
                        if i in recovery_results:
                            review_results.append(recovery_results[i])
                        else:
                            review_results.append(Exception(f"Failed to process review after retries"))
                
                # Process review results sequentially to maintain codebook consistency
                for idx, review_result in enumerate(review_results):
                    review_info = review_cluster_map[idx]
                    cluster_id = review_info['cluster_id']
                    cluster_info = review_info['cluster_info']
                    initial_result = review_info['initial_result']
                    
                    # Handle failed review results
                    if isinstance(review_result, Exception):
                        logger.error(f"Review failed for cluster {cluster_id}: {str(review_result)}")
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'v3_review_stage_error',  # Clear V3 identifier
                            'error': str(review_result),
                            'error_type': type(review_result).__name__,
                            'processing_time': time.time() - cluster_info['start_time']
                        })
                        self.stats['errors'] += 1
                        continue
                    
                    if review_result.approve_new_code and review_result.final_code:
                        # Add to shared codebook (this updates immediately for subsequent clusters)
                        added, new_version = await self.shared_codebook.add_code_if_new(
                            review_result.final_code,
                            review_result.final_definition or initial_result.definition
                        )
                        
                        if added:
                            self.stats['new_codes_added'] += 1
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: Added new code '{review_result.final_code}' (v{new_version}) - NOW AVAILABLE for subsequent clusters")
                        
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'new_code_added' if added else 'code_already_exists',
                            'code': review_result.final_code,
                            'definition': review_result.final_definition or initial_result.definition,
                            'processing_time': time.time() - cluster_info['start_time']
                        })
                    else:
                        self.stats['no_new_codes_needed'] += 1
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'no_new_code_after_review',
                            'processing_time': time.time() - cluster_info['start_time']
                        })
                    
                    self.stats['clusters_processed'] += 1
            
            self.stats['llm_time'] += time.time() - llm_start
                
        except Exception as e:
            logger.error(f"V3 BATCH PROCESSING ERROR: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Batch had {len(batch_clusters)} clusters")
            error_type = classify_error(e)
            
            # Add error results for all clusters in batch that weren't processed
            unprocessed_count = 0
            for idx in cluster_map:
                cluster_id = cluster_map[idx]['cluster_id']
                if not any(r['cluster_id'] == cluster_id for r in batch_results):
                    batch_results.append({
                        'cluster_id': cluster_id,
                        'status': 'v3_batch_processing_error',  # Clear V3 identifier
                        'error': str(e),
                        'error_type': error_type.value,
                        'error_details': f"Exception: {type(e).__name__}",
                        'processing_time': time.time() - cluster_map[idx]['start_time']
                    })
                    self.stats['errors'] += 1
                    unprocessed_count += 1
            
            logger.error(f"V3: {unprocessed_count} clusters marked as failed due to batch error")
            
            # Attempt individual recovery for failed clusters if possible
            if unprocessed_count > 0 and len(batch_inputs) > 0:
                logger.info(f"V3: Attempting individual recovery for {unprocessed_count} failed clusters")
                # This could be implemented as a fallback mechanism
        
        return batch_results
    
    async def _process_cluster_individually_as_fallback(self, cluster_id: int, cluster_data: Dict) -> Dict[str, Any]:
        """Process a single cluster individually as fallback when batch processing fails"""
        try:
            logger.info(f"V3: Attempting individual fallback for cluster {cluster_id}")
            
            # Process as a batch of 1
            single_cluster_batch = [(cluster_id, cluster_data)]
            results = await self.process_batch_langchain(single_cluster_batch)
            
            if results and len(results) > 0:
                return results[0]
            else:
                return {
                    'cluster_id': cluster_id,
                    'status': 'v3_individual_fallback_failed',
                    'error': 'No results from individual processing'
                }
                
        except Exception as e:
            logger.error(f"V3: Individual fallback failed for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'v3_individual_fallback_error',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def process_batch_with_sequential_codebook_updates(self, batch_clusters: List[Tuple[int, Dict]]) -> List[Dict[str, Any]]:
        """Alternative processing that handles clusters more sequentially for better codebook awareness"""
        batch_results = []
        
        # Process clusters in smaller sub-batches to balance performance and codebook awareness
        sub_batch_size = min(3, len(batch_clusters))  # Process 3 at a time max
        
        for i in range(0, len(batch_clusters), sub_batch_size):
            sub_batch = batch_clusters[i:i + sub_batch_size]
            
            try:
                sub_results = await self.process_batch_langchain(sub_batch)
                batch_results.extend(sub_results)
            except Exception as e:
                logger.error(f"V3: Sub-batch {i//sub_batch_size + 1} failed, attempting individual fallback: {e}")
                
                # Individual fallback for each cluster in the failed sub-batch
                for cluster_id, cluster_data in sub_batch:
                    fallback_result = await self._process_cluster_individually_as_fallback(cluster_id, cluster_data)
                    batch_results.append(fallback_result)
            
            # Small delay to allow codebook updates to propagate
            if i + sub_batch_size < len(batch_clusters):  # Not the last batch
                await asyncio.sleep(0.1)  # Brief pause for codebook updates
        
        return batch_results
    
    async def process_all_clusters_concurrent(self, clusters: Dict[int, Dict]) -> List[Dict[str, Any]]:
        """Process all clusters with true concurrent batch processing"""
        cluster_items = list(clusters.items())
        total_clusters = len(cluster_items)
        total_batches = (total_clusters + self.batch_size - 1) // self.batch_size
        
        verbose_reporter = VerboseReporter(self.verbose)
        
        # Create ALL batch tasks upfront
        batch_tasks = []
        
        for i in range(0, total_clusters, self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_clusters = cluster_items[i:i + self.batch_size]
            
            # Create async task for each batch - use sequential processing for better codebook awareness
            async def process_batch(batch_num=batch_num, batch_clusters=batch_clusters):
                """Process a single batch with sequential codebook updates"""
                if self.verbose:
                    logger.info(f"Batch {batch_num}/{total_batches} started")
                
                # Use sequential processing within batch for better codebook awareness
                results = await self.process_batch_with_sequential_codebook_updates(batch_clusters)
                
                if self.verbose:
                    new_codes = sum(1 for r in results if r['status'] == 'new_code_added')
                    logger.info(f"Batch {batch_num}/{total_batches} complete: {new_codes} new codes")
                
                return batch_num, results
            
            batch_tasks.append(process_batch())
        
        # Process ALL batches concurrently
        if self.verbose:
            verbose_reporter.step_start(
                f"Processing {total_clusters} clusters in {total_batches} concurrent batches"
            )
        
        all_batch_start = time.time()
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_batch_time = time.time() - all_batch_start
        
        # Collect all results
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch error: {result}")
            elif isinstance(result, tuple):
                batch_num, batch_cluster_results = result
                all_results.extend(batch_cluster_results)
        
        if self.verbose:
            verbose_reporter.step_complete(
                f"All {total_batches} batches completed in {all_batch_time:.1f}s "
                f"(true concurrent processing)"
            )
        
        return all_results

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class CodebookDataProcessor:
    """Handles data preparation for codebook generation"""
    
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 embedded_text: List[models.EmbeddingsModel],
                 k: int = 5):
        
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text
        self.k = k
        
    def prepare_cluster_text(self) -> Dict[int, Dict]:
        """Prepare cluster data with ideas and embeddings"""
        # Create embedding map
        embedding_map = {}
        for result in self.embedded_text:
            if hasattr(result, 'idea_embeddings') and result.idea_embeddings:
                for idea in result.idea_embeddings:
                    embedding_map[idea.idea_id] = {
                        'idea': idea.idea,
                        'embedding': idea.idea_embedding
                    }
        
        # Group by cluster
        clusters = {}
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
        
        # Filter out empty clusters
        return {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}

# ============================================================================
# MAIN GENERATOR
# ============================================================================

class InductiveCodebookGenerator:
    """V3 Generator using LangChain optimization and shared memory pattern"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel], 
        embedded_text: List[models.EmbeddingsModel],
        starter_codes: List[Dict[str, str]], 
        var_lab: str, 
        k: int = 5,
        verbose: bool = False, 
        prompt_printer = None,
        batch_size: int = 10,
        max_concurrent_requests: int = 5,
        config = None  # For compatibility
    ):
        logger.info("🚀 INITIALIZING CODEBOOK GENERATOR V3 (LangChain optimized)")
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize components
        self.model_config = ModelConfig()
        self.data_processor = CodebookDataProcessor(
            cluster_results=cluster_results,
            embedded_text=embedded_text,
            k=k
        )
        self.verbose_reporter = VerboseReporter(verbose)
    
    async def generate_async(self) -> Dict[str, Any]:
        """Generate codebook with LangChain optimization"""
        start_time = time.time()
        
        logger.info("🔥 STARTING V3 CODEBOOK GENERATION (NOT V2!)")
        self.verbose_reporter.section_header("CODEBOOK GENERATION V3 - LANGCHAIN OPTIMIZED", emoji="⚡")
        
        # Initialize shared codebook
        shared_codebook = SharedCodebook(self.starter_codes)
        
        # Initialize embedding manager
        embedding_manager = OptimizedEmbeddingManager(shared_codebook, self.verbose)
        
        # Prepare cluster data
        clusters = self.data_processor.prepare_cluster_text()
        if not clusters:
            return {
                'codebook': self.starter_codes,
                'cluster_assignments': {},
                'stats': {'error': 'No clusters to process'}
            }
        
        self.verbose_reporter.step_start(
            f"Processing {len(clusters)} clusters with LangChain batch optimization"
        )
        
        # Initialize batch processor
        batch_processor = LangChainBatchProcessor(
            embedding_manager=embedding_manager,
            shared_codebook=shared_codebook,
            model_config=self.model_config,
            var_lab=self.var_lab,
            k=self.k,
            batch_size=self.batch_size,
            max_concurrent_requests=self.max_concurrent_requests,
            verbose=self.verbose,
            prompt_printer=self.prompt_printer
        )
        
        # Process all clusters
        results = await batch_processor.process_all_clusters_concurrent(clusters)
        
        # Build cluster assignments
        cluster_to_code = {}
        for result in results:
            cluster_id = result['cluster_id']
            if result.get('code'):
                cluster_to_code[cluster_id] = result['code']
            else:
                cluster_to_code[cluster_id] = result['status']
        
        # Get final stats
        final_codes, final_version = await shared_codebook.get_current_snapshot()
        codebook_stats = await shared_codebook.get_stats()
        
        # Combine stats
        processing_time = time.time() - start_time
        final_stats = {
            **batch_processor.stats,
            **codebook_stats,
            'processing_time': processing_time,
            'initial_codes': len(self.starter_codes),
            'final_codes': len(final_codes),
            'new_codes': len(final_codes) - len(self.starter_codes),
            'avg_time_per_cluster': processing_time / len(clusters) if len(clusters) > 0 else 0
        }
        
        # Report results with comprehensive error statistics
        error_summary = {}
        if batch_processor.stats['errors'] > 0:
            error_summary[f"Errors"] = batch_processor.stats['errors']
        if batch_processor.stats['retries'] > 0:
            error_summary[f"Retries"] = batch_processor.stats['retries']
        if batch_processor.stats['partial_failures'] > 0:
            error_summary[f"Partial failures"] = batch_processor.stats['partial_failures']
        if batch_processor.stats['successful_recoveries'] > 0:
            error_summary[f"Successful recoveries"] = batch_processor.stats['successful_recoveries']
        
        summary_data = {
            "Initial codes": len(self.starter_codes),
            "New codes added": batch_processor.stats['new_codes_added'],
            "Final codebook size": len(final_codes),
            "Clusters processed": len(clusters),
            "Processing time": f"{processing_time:.2f}s",
            "Avg per cluster": f"{final_stats['avg_time_per_cluster']:.2f}s",
            "LLM time": f"{batch_processor.stats['llm_time']:.2f}s",
            "Embedding time": f"{batch_processor.stats['embedding_time']:.2f}s",
            "Method": "LangChain batch optimization with retry logic"
        }
        
        # Add error summary if there were any issues
        if error_summary:
            summary_data.update(error_summary)
            
        self.verbose_reporter.summary("V3 GENERATION COMPLETE", summary_data)
        
        # Log final results to confirm V3 was used
        logger.info(f"✅ V3 GENERATION COMPLETE: {len(final_codes)} total codes, {batch_processor.stats['new_codes_added']} new codes added")
        
        return {
            'codebook': final_codes,
            'cluster_assignments': cluster_to_code,
            'stats': final_stats,
            'generator_version': 'V3_LANGCHAIN_OPTIMIZED'  # Clear identifier
        }
    
    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for async generation"""
        return asyncio.run(self.generate_async())
    
    # Compatibility methods
    def generate_batch_concurrent(self) -> Dict[str, Any]:
        return self.generate()
    
    def generate_fully_concurrent(self) -> Dict[str, Any]:
        return self.generate()