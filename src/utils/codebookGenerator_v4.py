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

#from prompts import SYSTEM_MESSAGE_CODEBOOK, INITIAL_CODEBOOK_GENERATION, REVIEW_CODEBOOK_GENERATION
from prompts_v2 import SYSTEM_MESSAGE,  CODEBOOK_ANALYSIS_PROMPT, RESPONSE_SUMMARY_PROMPT, MATCH_AND_RECOMMEND_PROMPT, VALIDATION_PROMPT

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
logging.getLogger("tenacity").setLevel(logging.WARNING)   

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class CodebookAnalysis(BaseModel):
    """Output for codebook analysis step (Step 1)"""
    thematic_coverage: str = Field(description="Description of the main thematic areas these codes address")
    code_relationships: str = Field(description="How these codes connect and relate to each other thematically")

class ClusterAnalysis(BaseModel):
    """Output for cluster analysis step (Step 2)"""
    core_theme: str = Field(description="Specific aspect of the approach being discussed")
    sentiment_pattern: str = Field(description="Predominant sentiment: positive/negative/mixed")
    reasoning_focus: str = Field(description="Main justification provided for satisfaction/dissatisfaction")
    shared_terminology: List[str] = Field(description="Consistent concepts, phrases, or language patterns")
    cluster_coherence: str = Field(description="Explanation of what unites these ideas conceptually")

class MatchRecommendation(BaseModel):
    """Output for match and recommend step (Step 3)"""
    cluster_theme: str = Field(description="The core theme identified in cluster analysis")
    existing_code_matches: List[str] = Field(description="List of existing codes that match this theme")
    coverage: str = Field(description="How well existing codes cover this theme: full/partial/none")
    gap_analysis: str = Field(description="What's missing if coverage is partial or none")
    recommendation: str = Field(description="Recommendation: use existing/create new")
    new_code: Optional[str] = Field(description="New code name if creating new, null otherwise")
    new_definition: Optional[str] = Field(description="New code definition if creating new, null otherwise")
    justification: Optional[str] = Field(description="Justification for recommendation, null if not applicable")

class MatchRecommendationsResponse(BaseModel):
    """Container for multiple match recommendations from Step 3"""
    recommendations: List[MatchRecommendation] = Field(description="List of match recommendations for the cluster")

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

# Pydantic models for structured LLM outputs
# Only keeping models that are actively used - others will be created as needed

class CodebookAnalysis(BaseModel):
    """Output for codebook analysis step"""
    thematic_coverage: str = Field(description="Description of the main thematic areas these codes address")
    code_relationships: str = Field(description="How these codes connect and relate to each other thematically")

# Future Pydantic models will be added here as we enhance Steps 2, 3, and 4


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
# EMBEDDING MANAGER
# ============================================================================

class OptimizedEmbeddingManager:
    """Manages embeddings with shared codebook integration"""
    
    def __init__(self, shared_codebook: SharedCodebook, verbose: bool = False):
        self.shared_codebook = shared_codebook
        self.embedding_config = EmbeddingConfig()
        self.verbose = verbose
        self._individual_cache: Dict[str, np.ndarray] = {}
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    
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
    
    async def get_snapshot_embeddings(self, codes: List[Dict[str, str]], version: int) -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
        """Get embeddings for a codebook snapshot like v2 - always fresh, no version caching"""
        if not codes:
            return [], []
        
        # Generate embeddings fresh each time (like v2)
        code_texts = [f"{code['code']}: {code['definition']}" for code in codes]
        embeddings = await self._embed_texts_with_retry(code_texts)
        
        return codes, embeddings
    
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
    """Processes clusters using V2 multi-step prompts"""
    
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
        
        # Stats tracking (V3 enhanced)
        self.stats = {
            'clusters_processed': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'errors': 0,
            'retries': 0,  # V3 feature
            'partial_failures': 0,  # V3 feature
            'successful_recoveries': 0,  # V3 feature
            'llm_time': 0.0,
            'embedding_time': 0.0  # V3 feature (activated)
        }
    
    def _init_langchain_chain(self):
        """Initialize chains for V2 multi-step process with V3 4-stage model architecture"""
        
        # V3 Feature: Specialized LLMs for each of the 4 steps
        self.step1_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("initial_codes"),  # Codebook analysis
            temperature=0.0
        )
        
        self.step2_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("review_codes"),  # Response summary
            temperature=0.0
        )
        
        self.step3_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("initial_codes"),  # Match & recommend
            temperature=0.0
        )
        
        self.step4_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("review_codes"),  # Validation
            temperature=0.0
        )
        
        # Keep reference to primary LLM for compatibility (use step2_llm as default)
        self.llm = self.step2_llm
        
        # Use JsonOutputParser for flexible parsing
        from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
        
        # Step 1: Codebook Analysis Chain (uses step1_llm with Pydantic validation)
        codebook_prompt = PromptTemplate(
            template=CODEBOOK_ANALYSIS_PROMPT,
            input_variables=["system_message", "language", "survey_question", "code_text"]
        )
        
        self.codebook_chain = (
            codebook_prompt 
            | self.step1_llm 
            | PydanticOutputParser(pydantic_object=CodebookAnalysis)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 2: Response Summary Chain (uses step2_llm)
        summary_prompt = PromptTemplate(
            template=RESPONSE_SUMMARY_PROMPT,
            input_variables=["system_message", "language", "survey_question", "cluster_text"]
        )
        
        self.summary_chain = (
            summary_prompt
            | self.step2_llm
            | PydanticOutputParser(pydantic_object=ClusterAnalysis)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 3: Match and Recommend Chain (uses step3_llm)
        match_prompt = PromptTemplate(
            template=MATCH_AND_RECOMMEND_PROMPT,
            input_variables=["system_message", "existing_codes", "clustered_ideas", "codebook_analysis", "summaries"]
        )
        
        self.match_chain = (
            match_prompt
            | self.step3_llm
            | PydanticOutputParser(pydantic_object=MatchRecommendationsResponse)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 4: Validation Chain (uses step4_llm)
        validation_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["system_message", "recommendations", "redundancy_example"]
        )
        
        self.validation_chain = (
            validation_prompt
            | self.step4_llm
            | JsonOutputParser()
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Initialize capture counts and step diversity tracking
        self._capture_counts = {
            'codebook': 0,
            'summary': 0,
            'match': 0,
            'validation': 0
        }
        
        # Track which steps we've captured at least once (for guaranteed diversity)
        self._captured_steps = set()
        self._all_steps = {'codebook', 'summary', 'match', 'validation'}
        self._diversity_complete = False
    
    def _should_capture_prompt(self, step_type: str, max_per_step: int = 1) -> bool:
        """
        Determine if we should capture a prompt for this step type.
        Exactly-one approach: Capture 1 prompt from each step type, then stop.
        """
        if not self.prompt_printer:
            return False
            
        # Only capture if we haven't seen this step type yet
        return step_type not in self._captured_steps
    
    def _record_capture(self, step_type: str):
        """Record that we captured a prompt for this step type"""
        self._captured_steps.add(step_type)
        self._capture_counts[step_type] += 1
        
        # Debug logging for exactly-one progress
        if self.verbose:
            missing_steps = self._all_steps - self._captured_steps
            if missing_steps:
                logger.info(f"🎯 Prompt capture progress: {len(self._captured_steps)}/4 - captured {step_type}, still need: {', '.join(sorted(missing_steps))}")
            else:
                logger.info(f"✅ All 4 prompts captured! Pipeline structure complete.")
    
    @retry(**API_RETRY_CONFIG)
    async def _process_step1_with_retry(self, inputs: Dict) -> Dict:
        """Process Step 1 (Codebook Analysis) with retry logic"""
        try:
            return await self.codebook_chain.ainvoke(inputs)
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                self.stats['retries'] += 1
                raise APIError(f"Step 1 processing error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Step 1 processing error: {str(e)}", error_type)
    
    @retry(**API_RETRY_CONFIG)
    async def _process_step2_with_retry(self, inputs: Dict) -> Dict:
        """Process Step 2 (Response Summary) with retry logic"""
        try:
            return await self.summary_chain.ainvoke(inputs)
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                self.stats['retries'] += 1
                raise APIError(f"Step 2 processing error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Step 2 processing error: {str(e)}", error_type)
    
    @retry(**API_RETRY_CONFIG)
    async def _process_step3_with_retry(self, inputs: Dict) -> Dict:
        """Process Step 3 (Match & Recommend) with retry logic"""
        try:
            return await self.match_chain.ainvoke(inputs)
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                self.stats['retries'] += 1
                raise APIError(f"Step 3 processing error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Step 3 processing error: {str(e)}", error_type)
    
    @retry(**API_RETRY_CONFIG)
    async def _process_step4_with_retry(self, inputs: Dict) -> Dict:
        """Process Step 4 (Validation) with retry logic"""
        try:
            return await self.validation_chain.ainvoke(inputs)
        except Exception as e:
            error_type = classify_error(e)
            if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                            ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                self.stats['retries'] += 1
                raise APIError(f"Step 4 processing error: {str(e)}", error_type)
            else:
                raise ProcessingError(f"Step 4 processing error: {str(e)}", error_type)
    
    async def _retry_individual_failures(self, failed_clusters: List[Tuple[int, Dict]], 
                                        step_name: str = "unknown") -> Dict[int, Any]:
        """Retry individual failed clusters from a batch (V3 feature)"""
        recovery_results = {}
        
        for cluster_id, cluster_data in failed_clusters:
            try:
                # Process individual cluster as a single-item batch
                individual_results = await self.process_batch_langchain([(cluster_id, cluster_data)])
                
                if individual_results and len(individual_results) > 0:
                    recovery_results[cluster_id] = individual_results[0]
                    self.stats['successful_recoveries'] += 1
                    
                    if self.verbose:
                        logger.info(f"Successfully recovered processing for cluster {cluster_id} in {step_name}")
                else:
                    recovery_results[cluster_id] = {
                        'cluster_id': cluster_id,
                        'status': 'individual_recovery_no_results',
                        'error': 'No results from individual processing'
                    }
                    
            except Exception as e:
                logger.error(f"Individual retry failed for cluster {cluster_id}: {str(e)}")
                recovery_results[cluster_id] = {
                    'cluster_id': cluster_id,
                    'status': 'individual_recovery_failed', 
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        return recovery_results
    
    async def _process_cluster_individually_as_fallback(self, cluster_id: int, cluster_data: Dict) -> Dict[str, Any]:
        """Process a single cluster individually as fallback when batch processing fails (V3 feature)"""
        try:
            logger.info(f"V4: Attempting individual fallback for cluster {cluster_id}")
            
            # Process as a batch of 1
            single_cluster_batch = [(cluster_id, cluster_data)]
            results = await self.process_batch_langchain(single_cluster_batch)
            
            if results and len(results) > 0:
                return results[0]
            else:
                return {
                    'cluster_id': cluster_id,
                    'status': 'v4_individual_fallback_failed',
                    'error': 'No results from individual processing'
                }
                
        except Exception as e:
            logger.error(f"V4: Individual fallback failed for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'v4_individual_fallback_error',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def process_batch_langchain(self, batch_clusters: List[Tuple[int, Dict]]) -> List[Dict[str, Any]]:
        """Process a batch using V2 multi-step approach with V3 nearest codes logic"""
        batch_results = []
        
        # Process each cluster with nearest codes targeting (V3 logic)
        for cluster_id, cluster_data in batch_clusters:
            try:
                start_time = time.time()
                llm_start = time.time()  # Track LLM time separately
                
                # V3 Logic: Calculate cluster embedding and find nearest codes
                embed_start = time.time()
                cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
                nearest_codes = await self._find_nearest_codes(cluster_embedding)
                self.stats['embedding_time'] += time.time() - embed_start
                
                # Get current version for logging
                _, version = await self.shared_codebook.get_current_snapshot()
                
                # Build targeted code_text using nearest codes (V3 approach)
                if nearest_codes:
                    code_text = "\n".join([
                        f"- {code['code']}: {code['definition']}" 
                        for code in nearest_codes
                    ])
                else:
                    code_text = "No existing codes in codebook"
                
                # Step 1: Analyze nearest codes (not full codebook)
                codebook_input = {
                    "system_message": SYSTEM_MESSAGE.format(language=DEFAULT_LANGUAGE),
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "code_text": code_text
                }
                
                # Capture Step 1 prompt with diversity-first logic
                if self._should_capture_prompt('codebook'):
                    self.prompt_printer.capture_prompt(
                        step_name="codebook_generation_v4",
                        utility_name="LangChainBatchProcessor",
                        prompt_content=CODEBOOK_ANALYSIS_PROMPT.format(**codebook_input),
                        prompt_type="step1_codebook_analysis",
                        metadata={
                            "model": self.step1_llm.model_name,
                            "var_lab": self.var_lab,
                            "stage": "1/4 - Codebook Analysis",
                            "nearest_codes_count": len(nearest_codes),
                            "codebook_version": version
                        }
                    )
                    self._record_capture('codebook')
                
                # Execute Step 1: Codebook Analysis with retry
                try:
                    codebook_analysis = await self._process_step1_with_retry(codebook_input)
                except (APIError, ProcessingError) as e:
                    logger.error(f"Step 1 failed for cluster {cluster_id} after retries: {str(e)}")
                    # Fallback analysis
                    codebook_analysis = {
                        "thematic_coverage": "Analysis failed - could not analyze thematic areas",
                        "code_relationships": "Analysis failed - could not determine code relationships"
                    }
                except Exception as e:
                    logger.error(f"Step 1 unexpected error for cluster {cluster_id}: {str(e)}")
                    codebook_analysis = {
                        "thematic_coverage": "Analysis failed - unexpected error analyzing thematic areas",
                        "code_relationships": "Analysis failed - unexpected error determining code relationships"
                    }
                
                # Prepare cluster text for subsequent steps
                cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
                
                # Step 2: Summarize responses
                summary_input = {
                    "system_message": SYSTEM_MESSAGE.format(language=DEFAULT_LANGUAGE),
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "cluster_text": cluster_text
                }
                
                # Capture Step 2 prompt with diversity-first logic
                if self._should_capture_prompt('summary'):
                    self.prompt_printer.capture_prompt(
                        step_name="codebook_generation_v4",
                        utility_name="LangChainBatchProcessor",
                        prompt_content=RESPONSE_SUMMARY_PROMPT.format(**summary_input),
                        prompt_type="step2_response_summary",
                        metadata={
                            "model": self.step2_llm.model_name,
                            "var_lab": self.var_lab,
                            "stage": "2/4 - Response Summary",
                            "cluster_id": cluster_id,
                            "cluster_size": len(cluster_data['ideas'])
                        }
                    )
                    self._record_capture('summary')
                
                try:
                    summaries = await self._process_step2_with_retry(summary_input)
                except (APIError, ProcessingError) as e:
                    logger.error(f"Step 2 failed for cluster {cluster_id} after retries: {str(e)}")
                    summaries = {"theme": "Analysis failed", "tone": "unknown", "key_phrases": [], "unique": "Error in analysis"}
                except Exception as e:
                    logger.error(f"Step 2 unexpected error for cluster {cluster_id}: {str(e)}")
                    summaries = {"theme": "Analysis failed", "tone": "unknown", "key_phrases": [], "unique": "Error in analysis"}
                
                # Step 3: Match and recommend
                match_input = {
                    "system_message": SYSTEM_MESSAGE.format(language=DEFAULT_LANGUAGE),
                    "language": DEFAULT_LANGUAGE,
                    "existing_codes": code_text,
                    "clustered_ideas": cluster_text,
                    "codebook_analysis": codebook_analysis.model_dump_json(indent=2) if hasattr(codebook_analysis, 'model_dump_json') else str(codebook_analysis),
                    "summaries": summaries.model_dump_json(indent=2) if hasattr(summaries, 'model_dump_json') else str(summaries)
                }
                
                # Capture Step 3 prompt with diversity-first logic
                if self._should_capture_prompt('match'):
                    self.prompt_printer.capture_prompt(
                        step_name="codebook_generation_v4",
                        utility_name="LangChainBatchProcessor",
                        prompt_content=MATCH_AND_RECOMMEND_PROMPT.format(**match_input),
                        prompt_type="step3_match_recommend",
                        metadata={
                            "model": self.step3_llm.model_name,
                            "var_lab": self.var_lab,
                            "stage": "3/4 - Match & Recommend",
                            "cluster_id": cluster_id,
                            "codebook_analysis_present": bool(codebook_analysis),
                            "summaries_present": bool(summaries)
                        }
                    )
                    self._record_capture('match')
                
                try:
                    recommendations = await self._process_step3_with_retry(match_input)
                except (APIError, ProcessingError) as e:
                    logger.error(f"Step 3 failed for cluster {cluster_id} after retries: {str(e)}")
                    recommendations = []
                except Exception as e:
                    logger.error(f"Step 3 unexpected error for cluster {cluster_id}: {str(e)}")
                    recommendations = []
                
                # Extract new code recommendations from Pydantic container format
                new_codes_needed = False
                proposed_codes = []
                
                # Handle MatchRecommendationsResponse Pydantic object
                try:
                    # recommendations is now a MatchRecommendationsResponse object
                    if hasattr(recommendations, 'recommendations'):
                        recommendations_list = recommendations.recommendations
                        
                        for theme_analysis in recommendations_list:
                            # Handle Pydantic MatchRecommendation objects
                            if hasattr(theme_analysis, 'recommendation'):
                                recommendation = theme_analysis.recommendation.lower()
                                if 'create new' in recommendation:
                                    new_code = theme_analysis.new_code
                                    new_definition = theme_analysis.new_definition
                                    
                                    # Only add if we have actual code and definition (not null/empty)
                                    if new_code and new_definition and new_code.strip() and new_definition.strip():
                                        new_codes_needed = True
                                        proposed_codes.append({
                                            'code': new_code.strip(),
                                            'definition': new_definition.strip()
                                        })
                    # Fallback for dict format (backwards compatibility)
                    elif isinstance(recommendations, dict) and 'recommendations' in recommendations:
                        recommendations_list = recommendations['recommendations']
                        for theme_analysis in recommendations_list:
                            if isinstance(theme_analysis, dict):
                                recommendation = theme_analysis.get('recommendation', '').lower()
                                if 'create new' in recommendation:
                                    new_code = theme_analysis.get('new_code')
                                    new_definition = theme_analysis.get('new_definition')
                                    
                                    # Only add if we have actual code and definition (not null/empty)
                                    if new_code and new_definition and new_code.strip() and new_definition.strip():
                                        new_codes_needed = True
                                        proposed_codes.append({
                                            'code': new_code.strip(),
                                            'definition': new_definition.strip()
                                        })
                except Exception as e:
                    logger.error(f"Error parsing recommendations for cluster {cluster_id}: {e}")
                    logger.error(f"Recommendations content: {recommendations}")
                
                if not new_codes_needed:
                    self.stats['no_new_codes_needed'] += 1
                    batch_results.append({
                        'cluster_id': cluster_id,
                        'status': 'no_new_code_needed',
                        'processing_time': time.time() - start_time
                    })
                    self.stats['clusters_processed'] += 1
                    continue
                
                # Step 4: Validate if new codes are proposed
                if proposed_codes:
                    validation_input = {
                        "system_message": SYSTEM_MESSAGE.format(language=DEFAULT_LANGUAGE),
                        "language": DEFAULT_LANGUAGE,
                        "recommendations": str(proposed_codes),
                        "redundancy_example": "Example: 'student concerns' and 'learner worries' are redundant"
                    }
                    
                    # Capture Step 4 prompt with diversity-first logic
                    if self._should_capture_prompt('validation'):
                        self.prompt_printer.capture_prompt(
                            step_name="codebook_generation_v4",
                            utility_name="LangChainBatchProcessor",
                            prompt_content=VALIDATION_PROMPT.format(**validation_input),
                            prompt_type="step4_validation",
                            metadata={
                                "model": self.step4_llm.model_name,
                                "var_lab": self.var_lab,
                                "stage": "4/4 - Validation",
                                "cluster_id": cluster_id,
                                "proposed_codes_count": len(proposed_codes),
                                "proposed_codes": [c['code'] for c in proposed_codes]
                            }
                        )
                        self._record_capture('validation')
                    
                    try:
                        validation_results = await self._process_step4_with_retry(validation_input)
                    except (APIError, ProcessingError) as e:
                        logger.error(f"Step 4 failed for cluster {cluster_id} after retries: {str(e)}")
                        validation_results = {"validated_codes": []}
                    except Exception as e:
                        logger.error(f"Step 4 unexpected error for cluster {cluster_id}: {str(e)}")
                        validation_results = {"validated_codes": []}
                    
                    # Process validated codes from JSON format
                    validated_codes = []
                    try:
                        if isinstance(validation_results, dict):
                            validated_codes = validation_results.get('validated_codes', [])
                        else:
                            logger.error(f"Unexpected validation result format for cluster {cluster_id}: {validation_results}")
                    except Exception as e:
                        logger.error(f"Error parsing validation results for cluster {cluster_id}: {e}")
                        logger.error(f"Validation results content: {validation_results}")
                    
                    if validated_codes and len(validated_codes) > 0:
                        first_code = validated_codes[0]
                        added, new_version = await self.shared_codebook.add_code_if_new(
                            first_code.get('code', ''),
                            first_code.get('definition', '')
                        )
                        
                        if added:
                            self.stats['new_codes_added'] += 1
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: Added new code '{first_code['code']}' (v{new_version}) - NOW AVAILABLE for subsequent clusters")
                        
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'new_code_added' if added else 'code_already_exists',
                            'code': first_code.get('code', ''),
                            'definition': first_code.get('definition', ''),
                            'processing_time': time.time() - start_time
                        })
                    else:
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'no_codes_passed_validation',
                            'processing_time': time.time() - start_time
                        })
                
                self.stats['clusters_processed'] += 1
                
                # Track total LLM time for this cluster
                self.stats['llm_time'] += time.time() - llm_start
                
            except Exception as e:
                logger.error(f"V4 processing error for cluster {cluster_id}: {e}")
                batch_results.append({
                    'cluster_id': cluster_id,
                    'status': 'v4_processing_error',
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_time': time.time() - start_time if 'start_time' in locals() else 0
                })
                self.stats['errors'] += 1
        
        return batch_results
    
    async def _find_nearest_codes(self, cluster_embedding: np.ndarray) -> List[Dict[str, str]]:
        """Find k nearest codes using the current shared codebook (v2 logic)"""
        # Get CURRENT codebook state (like v2)
        current_codes, version = await self.shared_codebook.get_current_snapshot()
        
        if not current_codes:
            return []
        
        # Get fresh embeddings for current state (like v2)  
        codes, embeddings = await self.embedding_manager.get_snapshot_embeddings(
            current_codes, version
        )
        
        if not embeddings:
            return []
        
        # Calculate similarities (same logic as before)
        codebook_array = np.array(embeddings)
        similarities = cosine_similarity(cluster_embedding.reshape(1, -1), codebook_array)[0]
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        
        # Get unique codes (same logic as before)
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
            
            # Create async task for each batch - use concurrent processing
            async def process_batch(batch_num=batch_num, batch_clusters=batch_clusters):
                """Process a single batch with LangChain"""
                if self.verbose:
                    logger.info(f"Batch {batch_num}/{total_batches} started")
                
                results = await self.process_batch_langchain(batch_clusters)
                
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
    """V4 Generator using LangChain optimization and shared memory pattern"""
    
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
        
        logger.info("🚀 STARTING V4 CODEBOOK GENERATION (Multi-step prompts + V3 features)")
        self.verbose_reporter.section_header("CODEBOOK GENERATION V4 - MULTI-STEP PROMPTS + V3 FEATURES", emoji="🔥")
        
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
            error_summary["Errors"] = batch_processor.stats['errors']
        if batch_processor.stats['retries'] > 0:
            error_summary["Retries"] = batch_processor.stats['retries']
        if batch_processor.stats['partial_failures'] > 0:
            error_summary["Partial failures"] = batch_processor.stats['partial_failures']
        if batch_processor.stats['successful_recoveries'] > 0:
            error_summary["Successful recoveries"] = batch_processor.stats['successful_recoveries']
        
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
        
        # Log final results to confirm enhanced V4 was used
        logger.info(f"✅ V4 ENHANCED GENERATION COMPLETE: {len(final_codes)} total codes, {batch_processor.stats['new_codes_added']} new codes added")
        
        return {
            'codebook': final_codes,
            'cluster_assignments': cluster_to_code,
            'stats': final_stats,
            'generator_version': 'V4_MULTISTEP_PROMPTS_WITH_V3_FEATURES'  # Clear identifier
        }
    
    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for async generation"""
        return asyncio.run(self.generate_async())
   