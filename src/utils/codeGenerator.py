import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, RootModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from prompts import CODEBOOK_ANALYSIS_PROMPT, RESPONSE_SUMMARY_PROMPT, MATCH_AND_RECOMMEND_PROMPT, VALIDATION_PROMPT
from config import EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)   

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class CandidateCode(BaseModel):
    """Single candidate code from codebook analysis"""
    code: str = Field(description="Exact code name from existing codebook")
    definition: str = Field(description="Exact definition from existing codebook")

class CodebookAnalysisOutput(RootModel[List[CandidateCode]]):
    """Output from Step 1 - Codebook Analysis - Direct array of candidate codes"""
    root: List[CandidateCode] = Field(description="Array of selected relevant codes")

class ActionDetails(BaseModel):
    """Action details based on decision type"""
    codes_to_use: Optional[List[str]] = Field(default=None, description="List of codes if use_existing")
    codes_to_modify: Optional[str] = Field(default=None, description="Single code name if modify_existing")
    modified_code_name: Optional[str] = Field(default=None, description="Modified code name if create_new")
    modified_code_definition: Optional[str] = Field(default=None, description="Modified code definition if create_new")
    new_code_name: Optional[str] = Field(default=None, description="New code name if create_new")
    new_code_definition: Optional[str] = Field(default=None, description="New code definition if create_new")

class MatchRecommendation(BaseModel):
    """Output for match and recommend step (Step 3) - single recommendation per cluster"""
    cluster_core_theme: str = Field(description="The core theme identified from cluster analysis")
    decision: str = Field(description="Decision: use_existing|modify_existing|create_new")
    action_details: ActionDetails = Field(description="Action details based on decision")
    justification: str = Field(description="Explanation of why this is the most parsimonious choice")

class ValidationEvaluation(BaseModel):
    """Evaluation reasoning for Step 4"""
    parsimony_reasoning: str = Field(description="Assessment of whether existing options were exhausted")
    redundancy_reasoning: str = Field(description="Assessment of overlap with existing codes")
    justification_reasoning: str = Field(description="Assessment of decision alignment with reasoning")

class ValidatedCode(BaseModel):
    """Validated code output"""
    code: Optional[str] = Field(default=None, description="Final validated code name - always provide appropriate code even for REJECT")
    definition: Optional[str] = Field(default=None, description="Final validated definition - always provide appropriate definition even for REJECT")

class ValidationResponse(BaseModel):
    """Output for validation step (Step 4)"""
    evaluation: ValidationEvaluation = Field(description="Detailed evaluation scores and reasoning")
    decision: str = Field(description="Overall decision: APPROVE/REVISE/REJECT")
    decision_rationale: str = Field(description="Explanation for the overall decision")
    validated_code: ValidatedCode = Field(description="Final validated code - ALWAYS provide appropriate code/definition for any decision (APPROVE/REVISE/REJECT)")

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

# Fast retry configurations 
FAST_API_RETRY_CONFIG = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential(multiplier=1, min=0.5, max=30),  # Start with 0.5s instead of 1s
    "retry": retry_if_exception_type((APIError, asyncio.TimeoutError, ConnectionError)),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True
}

FAST_EMBEDDING_RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=0.5, min=0.5, max=10),  # Faster initial retry
    "retry": retry_if_exception_type((APIError, asyncio.TimeoutError, ConnectionError)),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True
}

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
    
    async def replace_code(self, original_code: str, new_code: str, new_definition: str) -> Tuple[bool, int]:
        """Replace an existing code with a modified version, return (replaced, new_version)"""
        async with self._lock:
            # Find and replace the original code
            for i, existing in enumerate(self._codes):
                if existing['code'].lower() == original_code.lower():
                    self._codes[i] = {'code': new_code, 'definition': new_definition}
                    self._version += 1
                    self._update_log.append({
                        'version': self._version,
                        'action': 'replace',
                        'original_code': original_code,
                        'new_code': new_code,
                        'timestamp': time.time()
                    })
                    return True, self._version
            
            # If original code not found, add as new
            self._codes.append({'code': new_code, 'definition': new_definition})
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add_as_fallback',
                'code': new_code,
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
    
    async def get_code_definition(self, code_name: str) -> Optional[str]:
        """Get the definition of a specific code"""
        async with self._lock:
            for existing in self._codes:
                if existing['code'].lower() == code_name.lower():
                    return existing['definition']
            return None

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
    
    async def get_snapshot_embeddings(self, codes: List[Dict[str, str]], version: int) -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
        """Get embeddings for a codebook snapshot - always fresh, no version caching"""
        if not codes:
            return [], []
        
        # Generate embeddings fresh each time 
        code_texts = [f"{code['code']}: {code['definition']}" for code in codes]
        embeddings = await self._embed_texts_with_retry(code_texts)
        
        return codes, embeddings
    
    @retry(**FAST_EMBEDDING_RETRY_CONFIG)
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
  
# ============================================================================
# LANGCHAIN BATCH PROCESSOR - HIERARCHICAL CONCURRENCY
# ============================================================================

class LangChainBatchProcessor:
    """Processes clusters using hierarchical concurrency and parallel step execution"""
    
    def __init__(self, 
                 embedding_manager: OptimizedEmbeddingManager,
                 shared_codebook: SharedCodebook,
                 model_config: ModelConfig,
                 var_lab: str,
                 k: int = 5,
                 batch_size: int = 10,
                 sub_batch_size: int = 5,
                 enable_step_parallelization: bool = True,
                 max_concurrent_steps: int = 2,
                 max_concurrent_requests: int = 10,
                 verbose: bool = False,
                 prompt_printer = None):
        
        self.embedding_manager = embedding_manager
        self.shared_codebook = shared_codebook
        self.model_config = model_config
        self.var_lab = var_lab
        self.k = k
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.enable_step_parallelization = enable_step_parallelization
        self.max_concurrent_steps = max_concurrent_steps
        self.max_concurrent_requests = max_concurrent_requests
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        
        # Initialize LangChain components
        self._init_langchain_chain()
        
        # Stats tracking  
        self.stats = {
            'clusters_processed': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'codes_modified': 0,
            'errors': 0,
            'retries': 0,  
            'partial_failures': 0,   
            'successful_recoveries': 0,   
            'llm_time': 0.0,
            'embedding_time': 0.0,
            'parallel_steps_executed': 0,
            'sub_batches_processed': 0,
            'concurrent_batches': 0,
            'decisions': {
                'use_existing': 0,
                'modify_existing': 0,
                'create_new': 0
            }
        }
        
        if self.verbose:
            logger.info("🚀 Initialized CodeGenerator with hierarchical concurrency:")
            logger.info(f"  - Batch size: {batch_size}, Sub-batch size: {sub_batch_size}")
            logger.info(f"  - Step parallelization: {enable_step_parallelization}")
            logger.info(f"  - Max concurrent steps: {max_concurrent_steps}")
    
    def _init_langchain_chain(self):
        """Initialize chains with optimized configurations"""
        
        self.step1_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("codes_analysis"),   
            temperature=0.0
        )
        
        self.step2_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("cluster_analysis"),   
            temperature=0.0
        )
        
        self.step3_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("recommend"),   
            temperature=0.0
        )
        
        self.step4_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("review"),   
            temperature=0.0
        )
        
        # Step 1: Codebook Analysis Chain
        codebook_prompt = PromptTemplate(
            template=CODEBOOK_ANALYSIS_PROMPT,
            input_variables=["language", "survey_question", "cluster_text", "code_text"]
        )
        
        self.codebook_chain = (
            codebook_prompt 
            | self.step1_llm 
            | PydanticOutputParser(pydantic_object=CodebookAnalysisOutput)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 2: Response Summary Chain
        summary_prompt = PromptTemplate(
            template=RESPONSE_SUMMARY_PROMPT,
            input_variables=["language", "survey_question", "cluster_text"]
        )
        
        self.summary_chain = (
            summary_prompt
            | self.step2_llm
            | StrOutputParser()
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 3: Match and Recommend Chain
        match_prompt = PromptTemplate(
            template=MATCH_AND_RECOMMEND_PROMPT,
            input_variables=["language", "survey_question", "candidate_codes", "clustered_survey_responses", "cluster_summary"]
        )
        
        self.match_chain = (
            match_prompt
            | self.step3_llm
            | PydanticOutputParser(pydantic_object=MatchRecommendation)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 4: Validation Chain
        validation_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["language", "survey_question", "candidate_codes", "clustered_ideas", "step3_recommendation"]
        )
        
        self.validation_chain = (
            validation_prompt
            | self.step4_llm
            | PydanticOutputParser(pydantic_object=ValidationResponse)
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

    def _split_into_sub_batches(self, batch_clusters: List[Tuple[int, Dict]]) -> List[List[Tuple[int, Dict]]]:
        """Split batch into sub-batches for hierarchical processing"""
        sub_batches = []
        for i in range(0, len(batch_clusters), self.sub_batch_size):
            sub_batch = batch_clusters[i:i + self.sub_batch_size]
            sub_batches.append(sub_batch)
        return sub_batches

    def _format_step3_recommendation(self, recommendation: MatchRecommendation) -> str:
        """Format prompt 3 for Step 4"""
        if not recommendation:
            return "No recommendation available"
            
        formatted = f"""
Cluster Theme: {recommendation.cluster_core_theme}
Recommendation: 
- {recommendation.decision.replace('_', ' ').title()}
"""
        if recommendation.action_details.codes_to_use:
            formatted += f"- Code(s) to use: {', '.join(recommendation.action_details.codes_to_use)}\n"
        if recommendation.action_details.codes_to_modify:
            formatted += f"- Code to modify: {recommendation.action_details.codes_to_modify}\n"
            formatted += f"- Modified code: {recommendation.action_details.modified_code_name}\n"
            formatted += f"- Modified definition: {recommendation.action_details.modified_code_definition}\n"
        if recommendation.action_details.new_code_name:
            formatted += f"- New code: {recommendation.action_details.new_code_name}\n"
            formatted += f"- Definition: {recommendation.action_details.new_code_definition}\n"

        formatted += f"\nJustification: {recommendation.justification}"
        
        return formatted.strip()

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
                logger.info("✅ All 4 prompts captured! Pipeline structure complete.")

    @retry(**FAST_API_RETRY_CONFIG)
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
    
    @retry(**FAST_API_RETRY_CONFIG)
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
    
    @retry(**FAST_API_RETRY_CONFIG)
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
    
    @retry(**FAST_API_RETRY_CONFIG)
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

    async def _process_parallel_steps(self, cluster_id: int, cluster_data: Dict, code_text: str, cluster_text: str) -> Tuple[Any, Any]:
        """Process Steps 1 and 2 in parallel (independent steps)"""
        if not self.enable_step_parallelization:
            # Fallback to sequential processing
            codebook_input = {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_text": cluster_text,
                "code_text": code_text
            }
            codebook_analysis = await self._process_step1_with_retry(codebook_input)
            
            summary_input = {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_text": cluster_text
            }
            summaries = await self._process_step2_with_retry(summary_input)
            
            return codebook_analysis, summaries
        
        # Parallel execution of Steps 1 & 2
        codebook_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_text": cluster_text,
            "code_text": code_text
        }
        
        summary_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_text": cluster_text
        }
        
        # Execute Steps 1 & 2 concurrently
        step1_task = self._process_step1_with_retry(codebook_input)
        step2_task = self._process_step2_with_retry(summary_input)
        
        try:
            codebook_analysis, summaries = await asyncio.gather(
                step1_task, step2_task, return_exceptions=True
            )
            
            # Handle individual step failures
            if isinstance(codebook_analysis, Exception):
                logger.error(f"Step 1 failed for cluster {cluster_id}: {str(codebook_analysis)}")
                codebook_analysis = "Analysis failed - could not analyze thematic areas"
            
            if isinstance(summaries, Exception):
                logger.error(f"Step 2 failed for cluster {cluster_id}: {str(summaries)}")
                summaries = "Analysis failed due to API error after retries."
            
            self.stats['parallel_steps_executed'] += 1
            return codebook_analysis, summaries
            
        except Exception as e:
            logger.error(f"Parallel steps processing error for cluster {cluster_id}: {e}")
            # Fallback results
            return ("Analysis failed - parallel processing error", 
                   "Analysis failed - parallel processing error")

    async def _process_cluster_optimized(self, cluster_id: int, cluster_data: Dict) -> Dict[str, Any]:
        """Process a single cluster with optimized parallel step execution"""
        try:
            start_time = time.time()
            llm_start = time.time()
            
            embed_start = time.time()
            cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
            nearest_codes = await self._find_nearest_codes(cluster_embedding)
            self.stats['embedding_time'] += time.time() - embed_start
            
            # Get current version for logging
            _, version = await self.shared_codebook.get_current_snapshot()
            
            # Build targeted code_text using nearest codes  
            if nearest_codes:
                code_text = "\n".join([
                    f"- {code['code']}: {code['definition']}" 
                    for code in nearest_codes
                ])
            else:
                code_text = "No existing codes in codebook"
            
            # Prepare cluster text
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            # Execute Steps 1 & 2 in parallel  
            codebook_analysis_result, summaries = await self._process_parallel_steps(
                cluster_id, cluster_data, code_text, cluster_text
            )
            
            # Extract candidate codes from Step 1 output
            candidate_codes = []
            if hasattr(codebook_analysis_result, 'root'):
                # RootModel structure
                candidate_codes = codebook_analysis_result.root
            elif isinstance(codebook_analysis_result, list):
                # Handle case where result is already a list
                candidate_codes = [CandidateCode(code=c['code'], definition=c['definition']) if isinstance(c, dict) else c for c in codebook_analysis_result]
            
            # Format candidate codes for Step 3 and validation
            if candidate_codes:
                candidate_codes_text = "\n".join([
                    f"- {code.code}: {code.definition}" if hasattr(code, 'code') else f"- {code['code']}: {code['definition']}"
                    for code in candidate_codes
                ])
            else:
                candidate_codes_text = "No candidate codes available"
            
            # Capture prompts if needed (diversity-first logic)
            if self._should_capture_prompt('codebook'):
                codebook_input = {
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "cluster_text": cluster_text,
                    "code_text": code_text
                }
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
                    utility_name="LangChainBatchProcessor",
                    prompt_content=CODEBOOK_ANALYSIS_PROMPT.format(**codebook_input),
                    prompt_type="step1_codebook_analysis",
                    metadata={
                        "model": self.step1_llm.model_name,
                        "var_lab": self.var_lab,
                        "stage": "1/4 - Codebook Analysis (Parallel)",
                        "nearest_codes_count": len(nearest_codes),
                        "codebook_version": version,
                        "parallel_execution": self.enable_step_parallelization
                    }
                )
                self._record_capture('codebook')
            
            if self._should_capture_prompt('summary'):
                summary_input = {
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "cluster_text": cluster_text
                }
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
                    utility_name="LangChainBatchProcessor",
                    prompt_content=RESPONSE_SUMMARY_PROMPT.format(**summary_input),
                    prompt_type="step2_response_summary",
                    metadata={
                        "model": self.step2_llm.model_name,
                        "var_lab": self.var_lab,
                        "stage": "2/4 - Response Summary (Parallel)",
                        "cluster_id": cluster_id,
                        "cluster_size": len(cluster_data['ideas']),
                        "parallel_execution": self.enable_step_parallelization
                    }
                )
                self._record_capture('summary')
            
            # Step 3: Match and recommend (uses candidate codes from Step 1)
            match_input = {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "candidate_codes": candidate_codes_text,
                "clustered_survey_responses": cluster_text,
                "cluster_summary": summaries if isinstance(summaries, str) else str(summaries)
            }
            
            # Capture Step 3 prompt
            if self._should_capture_prompt('match'):
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
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
            
            # Extract new code recommendations from single MatchRecommendation object
            new_codes_needed = False
            proposed_codes = []
            
            try:
                if hasattr(recommendations, 'decision'):
                    decision = recommendations.decision.lower()
                    
                    # Track decision statistics
                    if 'use_existing' in decision:
                        self.stats['decisions']['use_existing'] += 1
                    elif 'modify_existing' in decision:
                        self.stats['decisions']['modify_existing'] += 1
                    elif 'create_new' in decision:
                        self.stats['decisions']['create_new'] += 1
                    
                    if 'create_new' in decision:
                        # Access new code details from action_details
                        if hasattr(recommendations, 'action_details'):
                            new_code = recommendations.action_details.new_code_name
                            new_definition = recommendations.action_details.new_code_definition
                            
                            # Only add if we have actual code and definition (not null/empty)
                            if new_code and new_definition and new_code.strip() and new_definition.strip():
                                new_codes_needed = True
                                proposed_codes.append({
                                    'code': new_code.strip(),
                                    'definition': new_definition.strip()
                                })
                    elif 'modify_existing' in decision:
                        # Trigger Step 4 for modification validation
                        if hasattr(recommendations, 'action_details'):
                            original_code = recommendations.action_details.codes_to_modify
                            modified_code = recommendations.action_details.modified_code_name
                            modified_definition = recommendations.action_details.modified_code_definition
                            
                            # Only proceed if we have modification details
                            if original_code and modified_code and modified_definition and modified_code.strip() and modified_definition.strip():
                                new_codes_needed = True
                                proposed_codes.append({
                                    'original_code': original_code.strip(),
                                    'modified_code': modified_code.strip(),
                                    'modified_definition': modified_definition.strip()
                                })
                    
                    # Store the full recommendation for potential Step 4 use
                    cluster_recommendation = recommendations
                    
            except Exception as e:
                logger.error(f"Error parsing recommendations for cluster {cluster_id}: {e}")
                logger.error(f"Recommendations content: {recommendations}")
            
            if not new_codes_needed:
                self.stats['no_new_codes_needed'] += 1
                return {
                    'cluster_id': cluster_id,
                    'status': 'no_new_code_needed',
                    'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                    'step4_validated_code': None,
                    'processing_time': time.time() - start_time
                }
            
            # Step 4: Validate if new codes are proposed
            if proposed_codes:
                # Format Step 3 recommendation for readable context
                formatted_recommendation = self._format_step3_recommendation(recommendations) if hasattr(recommendations, 'cluster_core_theme') else str(recommendations)
                
                # For create_new: use definition-based codes. For modify_existing: use cluster-based codes
                is_create_new = any(pc.get('definition') for pc in proposed_codes)
                
                if is_create_new:
                    # Use definition-based nearest codes for new code redundancy detection
                    definition_text = await self._extract_definition_for_embedding(proposed_codes, recommendations)
                    if definition_text:
                        definition_nearest_codes = await self._find_nearest_codes_by_definition(definition_text)
                        if definition_nearest_codes:
                            definition_code_text = "\n".join([
                                f"- {code['code']}: {code['definition']}" 
                                for code in definition_nearest_codes
                            ])
                        else:
                            # Fallback to cluster-based codes if definition embedding fails
                            definition_code_text = code_text
                    else:
                        # Fallback to cluster-based codes if no definition extracted
                        definition_code_text = code_text
                else:
                    # For modify_existing: keep using cluster-based codes
                    definition_code_text = code_text
                
                validation_input = {
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "candidate_codes": definition_code_text,
                    "clustered_ideas": cluster_text,
                    "step3_recommendation": formatted_recommendation
                }
                
                # Capture Step 4 prompt
                if self._should_capture_prompt('validation'):
                    self.prompt_printer.capture_prompt(
                        step_name="codebook_generation",
                        utility_name="LangChainBatchProcessor",
                        prompt_content=VALIDATION_PROMPT.format(**validation_input),
                        prompt_type="step4_validation",
                        metadata={
                            "model": self.step4_llm.model_name,
                            "var_lab": self.var_lab,
                            "stage": "4/4 - Validation",
                            "cluster_id": cluster_id,
                            "proposed_codes_count": len(proposed_codes),
                            "proposed_codes": [c.get('code', c.get('modified_code', '')) for c in proposed_codes]
                        }
                    )
                    self._record_capture('validation')
                
                try:
                    validation_results = await self._process_step4_with_retry(validation_input)
                except (APIError, ProcessingError) as e:
                    logger.error(f"Step 4 failed for cluster {cluster_id} after retries: {str(e)}")
                    validation_results = None
                except Exception as e:
                    logger.error(f"Step 4 unexpected error for cluster {cluster_id}: {str(e)}")
                    validation_results = None
                
                # Process validated codes from ValidationResponse format
                validated_code = None
                validation_details = None
                
                try:
                    if validation_results and hasattr(validation_results, 'decision'):
                        # ValidationResponse object - store detailed validation info
                        validation_details = {
                            'decision': validation_results.decision,
                            'decision_rationale': validation_results.decision_rationale,
                            'reasoning': {
                                'parsimony': validation_results.evaluation.parsimony_reasoning,
                                'redundancy': validation_results.evaluation.redundancy_reasoning,
                                'justification': validation_results.evaluation.justification_reasoning
                            }
                        }
                        
                        # Extract validated code for ANY decision (APPROVE, REVISE, or REJECT)
                        if validation_results.validated_code and validation_results.validated_code.code:
                            validated_code = {
                                'code': validation_results.validated_code.code,
                                'definition': validation_results.validated_code.definition
                            }
                except Exception as e:
                    logger.error(f"Error parsing validation results for cluster {cluster_id}: {e}")
                    logger.error(f"Validation results content: {validation_results}")
                
                if validated_code:
                    # Check if this is a modification or new code
                    is_modification = any(pc.get('original_code') for pc in proposed_codes)
                    
                    if is_modification:
                        # Get original code name from proposed_codes
                        original_code_name = next(
                            (pc.get('original_code') for pc in proposed_codes if pc.get('original_code')),
                            None
                        )
                        
                        if original_code_name:
                            # Get original definition before modification
                            original_definition = await self.shared_codebook.get_code_definition(original_code_name)
                            
                            # Replace existing code with modified version
                            replaced, new_version = await self.shared_codebook.replace_code(
                                original_code_name,
                                validated_code.get('code', ''),
                                validated_code.get('definition', '')
                            )
                            
                            if replaced:
                                self.stats['codes_modified'] += 1
                                if self.verbose:
                                    new_code_name = validated_code.get('code', '')
                                    new_definition = validated_code.get('definition', '')
                                    
                                    name_changed = original_code_name != new_code_name
                                    definition_changed = original_definition != new_definition if original_definition else True
                                    
                                    if name_changed and definition_changed:
                                        logger.info(f"Cluster {cluster_id}: Modified code '{original_code_name}' -> '{new_code_name}' + definition updated (v{new_version})")
                                    elif name_changed:
                                        logger.info(f"Cluster {cluster_id}: Renamed code '{original_code_name}' -> '{new_code_name}' (v{new_version})")
                                    elif definition_changed:
                                        logger.info(f"Cluster {cluster_id}: Updated definition for '{original_code_name}' (v{new_version})")
                            
                            return {
                                'cluster_id': cluster_id,
                                'status': 'code_modified',
                                'original_code': original_code_name,
                                'code': validated_code.get('code', ''),
                                'definition': validated_code.get('definition', ''),
                                'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                                'step4_validated_code': validated_code,
                                'validation_details': validation_details,
                                'processing_time': time.time() - start_time
                            }
                    else:
                        # Standard new code addition (CRITICAL: Real-time update to shared memory)
                        added, new_version = await self.shared_codebook.add_code_if_new(
                            validated_code.get('code', ''),
                            validated_code.get('definition', '')
                        )
                        
                        if added:
                            self.stats['new_codes_added'] += 1
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: Added new code '{validated_code['code']}' (v{new_version}) - NOW AVAILABLE for subsequent clusters")
                        
                        return {
                            'cluster_id': cluster_id,
                            'status': 'new_code_added' if added else 'code_already_exists',
                            'code': validated_code.get('code', ''),
                            'definition': validated_code.get('definition', ''),
                            'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                            'step4_validated_code': validated_code,
                            'validation_details': validation_details,
                            'processing_time': time.time() - start_time
                        }
                else:
                    return {
                        'cluster_id': cluster_id,
                        'status': 'no_codes_passed_validation',
                        'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                        'step4_validated_code': None,
                        'validation_details': validation_details,
                        'processing_time': time.time() - start_time
                    }

            self.stats['clusters_processed'] += 1
            self.stats['llm_time'] += time.time() - llm_start
            
            return {
                'cluster_id': cluster_id,
                'status': 'processed_no_validation_needed',
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Processing error for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'Processing_error',
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }

    async def _process_sub_batch_langchain(self, sub_batch: List[Tuple[int, Dict]], sub_batch_idx: int) -> List[Dict[str, Any]]:
        """Process a sub-batch of clusters sequentially to preserve shared memory order"""
        sub_batch_results = []
        
        if self.verbose:
            logger.info(f"  Sub-batch {sub_batch_idx + 1}: Processing {len(sub_batch)} clusters sequentially")
        
        # Process clusters sequentially within sub-batch to preserve shared memory updates
        for cluster_id, cluster_data in sub_batch:
            try:
                result = await self._process_cluster_optimized(cluster_id, cluster_data)
                sub_batch_results.append(result)
            except Exception as e:
                logger.error(f"Sub-batch {sub_batch_idx + 1} cluster {cluster_id} error: {e}")
                sub_batch_results.append({
                    'cluster_id': cluster_id,
                    'status': 'sub_batch_processing_error',
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_time': 0
                })
        
        self.stats['sub_batches_processed'] += 1
        return sub_batch_results

    async def process_batch_langchain(self, batch_clusters: List[Tuple[int, Dict]], batch_idx: int = 0) -> List[Dict[str, Any]]:
        """Process a batch using hierarchical concurrency approach"""
        
        # Split batch into sub-batches
        sub_batches = self._split_into_sub_batches(batch_clusters)
        
        if self.verbose:
            logger.info(f"Batch {batch_idx + 1}: Split {len(batch_clusters)} clusters into {len(sub_batches)} sub-batches")
        
        # Process all sub-batches concurrently (Level 1 concurrency)
        sub_batch_tasks = [
            self._process_sub_batch_langchain(sub_batch, i) 
            for i, sub_batch in enumerate(sub_batches)
        ]
        
        sub_batch_results = await asyncio.gather(*sub_batch_tasks, return_exceptions=True)
        
        # Collect results from all sub-batches
        batch_results = []
        for i, sub_result in enumerate(sub_batch_results):
            if isinstance(sub_result, Exception):
                logger.error(f"Sub-batch {i+1} processing failed: {type(sub_result).__name__}: {str(sub_result)}")
                # Add error results for failed sub-batch
                if i < len(sub_batches):
                    for cluster_id, cluster_data in sub_batches[i]:
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'sub_batch_failed',
                            'error': str(sub_result),
                            'processing_time': 0
                        })
            else:
                batch_results.extend(sub_result)
        
        return batch_results

    async def process_all_clusters_concurrent(self, clusters: Dict[int, Dict]) -> List[Dict[str, Any]]:
        """Process all clusters with hierarchical concurrent processing"""
        cluster_items = list(clusters.items())
        total_clusters = len(cluster_items)
        total_batches = (total_clusters + self.batch_size - 1) // self.batch_size
        
        # Track progress
        self.completed_batches = 0
        self.completed_clusters = 0
        
        # Create ALL batch tasks upfront (Level 0 concurrency)
        batch_tasks = []
        
        for i in range(0, total_clusters, self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_clusters = cluster_items[i:i + self.batch_size]
            
            # Create async task for each batch using hierarchical processing
            async def process_batch(batch_num=batch_num, batch_clusters=batch_clusters, batch_start_idx=i):
                """Process a single batch with hierarchical concurrency"""
                # # Display batch start
                # print(f"Processing batch {batch_num}/{total_batches}... ", end="", flush=True)
                
                results = await self.process_batch_langchain(batch_clusters, batch_num - 1)
                
                # Collect codes and modifications from this batch
                new_codes_this_batch = []
                modified_codes_this_batch = []
                used_existing = 0
                
                for r in results:
                    if r['status'] == 'new_code_added' and r.get('code'):
                        new_codes_this_batch.append(r['code'])
                    elif r['status'] == 'code_modified' and r.get('code'):
                        # Try to get original code name for modification display
                        original = r.get('original_code', r.get('code', 'Unknown'))
                        new_code = r.get('code', '')
                        
                        # Determine type of modification
                        if original != new_code:
                            if original.lower() in new_code.lower() or new_code.lower() in original.lower():
                                mod_type = "refined"
                            else:
                                mod_type = "renamed"
                        else:
                            mod_type = "definition updated"
                        
                        modified_codes_this_batch.append((original, mod_type))
                    elif r['status'] in ['existing_codes_used', 'no_new_code_needed']:
                        used_existing += 1
                
                # Update global counters
                self.completed_batches += 1
                self.completed_clusters += len(batch_clusters)
                
                # Display codes added this batch
                if new_codes_this_batch:
                    for code in new_codes_this_batch:
                        self.verbose_reporter.stat_line(f'"{code}"', indent=1)
                
                # Display modifications this batch
                if modified_codes_this_batch:
                    self.verbose_reporter.stat_line("Codes modified this batch:")
                    for code_info in modified_codes_this_batch:
                        if isinstance(code_info, tuple):
                            original, mod_type = code_info
                            self.verbose_reporter.stat_line(f'"{original}" ({mod_type})', indent=1)
                        else:
                            # Fallback for old format
                            self.verbose_reporter.stat_line(f'"{code_info}" (definition refined)', indent=1)
                    
                return batch_num, results, len(new_codes_this_batch), len(modified_codes_this_batch), used_existing
            
            batch_tasks.append(process_batch())
        
        # Process ALL batches concurrently (Level 0 concurrency)
        all_batch_start = time.time()
        self.stats['concurrent_batches'] = total_batches
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_batch_time = time.time() - all_batch_start
        
        # Collect all results and track running totals
        all_results = []
        running_new_codes = 0
        running_modified_codes = 0
        running_existing_used = 0
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch error: {result}")
            elif isinstance(result, tuple) and len(result) >= 5:
                batch_num, batch_cluster_results, new_count, modified_count, existing_count = result
                all_results.extend(batch_cluster_results)
                
                # Update running totals
                running_new_codes += new_count
                running_modified_codes += modified_count
                running_existing_used += existing_count
                
            elif isinstance(result, tuple) and len(result) == 2:
                # Handle old format for compatibility
                batch_num, batch_cluster_results = result
                all_results.extend(batch_cluster_results)
        
        if self.verbose:
            self.verbose_reporter.step_complete(
                f"All {total_batches} batches completed in {all_batch_time:.1f}s "
                f"(Hierarchical: {self.stats['parallel_steps_executed']} parallel step executions)"
            )
        
        return all_results

    async def _extract_definition_for_embedding(self, proposed_codes: List[Dict], recommendations) -> Optional[str]:
        """Extract definition text for embedding from Step 3 recommendation"""
        if not proposed_codes:
            return None
            
        # Handle create_new case - use the proposed definition directly
        for pc in proposed_codes:
            if pc.get('definition'):
                return pc['definition']
        
        # Handle modify_existing case - construct definition from original + modification
        for pc in proposed_codes:
            if pc.get('original_code'):
                original_code = pc.get('original_code')
                modified_code = pc.get('modified_code')
                modified_definition = pc.get('modified_definition')
                
                if original_code and modified_code and modified_definition:
                    # Get the original definition from shared codebook (async)
                    original_definition = await self.shared_codebook.get_code_definition(original_code)
                    
                    if original_definition:
                        # Combine original definition with modification suggestion
                        return f"{original_definition}. Modified to include: {modified_code} - {modified_definition}"
                    else:
                        # Fallback to just the modification suggestion
                        return modified_definition
        
        return None

    async def _find_nearest_codes(self, cluster_embedding: np.ndarray) -> List[Dict[str, str]]:
        """Find k nearest codes using the current shared codebook"""
        # Get CURRENT codebook state 
        current_codes, version = await self.shared_codebook.get_current_snapshot()
        
        if not current_codes:
            return []
        
        # Get fresh embeddings for current state 
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

    async def _find_nearest_codes_by_definition(self, definition_text: str) -> List[Dict[str, str]]:
        """Find k nearest codes using embedding of definition text"""
        if not definition_text or not definition_text.strip():
            # Fallback to empty list if no definition
            return []
        
        # Get CURRENT codebook state
        current_codes, version = await self.shared_codebook.get_current_snapshot()
        
        if not current_codes:
            return []
        
        # Get fresh embeddings for current state
        codes, embeddings = await self.embedding_manager.get_snapshot_embeddings(
            current_codes, version
        )
        
        if not embeddings:
            return []
        
        # Generate embedding for the definition text
        try:
            definition_embeddings = await self.embedding_manager._embed_texts_with_retry([definition_text])
            if not definition_embeddings:
                return []
            definition_embedding = definition_embeddings[0]
        except Exception as e:
            logger.error(f"Failed to embed definition text: {e}")
            return []
        
        # Calculate similarities
        codebook_array = np.array(embeddings)
        similarities = cosine_similarity(definition_embedding.reshape(1, -1), codebook_array)[0]
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

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class CodebookDataProcessor:
    """Handles data preparation for codebook generation"""
    
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 k: int = 5,
                 embedded_text: List[models.EmbeddingsModel] = None):  # Deprecated
        
        self.cluster_results = cluster_results
        if embedded_text is not None:
            logger.warning("embedded_text parameter is deprecated - embeddings are now in cluster_results")
        self.k = k
        
    def prepare_cluster_text(self) -> Dict[int, Dict]:
        """Prepare cluster data with ideas and embeddings (simplified for clean model inheritance)"""
        clusters = {}
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    # ClusterModel now contains embeddings directly
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        if cluster_id not in clusters:
                            clusters[cluster_id] = {'ideas': [], 'embeddings': []}
                        
                        clusters[cluster_id]['ideas'].append(idea.idea)
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
        
        # Filter out empty clusters
        return {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}

# ============================================================================
# MAIN GENERATOR
# ============================================================================

class InductiveCodeGenerator:
    """Generator using hierarchical concurrency and parallel step execution"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel], 
        starter_codes: List[Dict[str, str]], 
        var_lab: str, 
        k: int = 5,
        verbose: bool = False, 
        prompt_printer = None,
        batch_size: int = 10,
        sub_batch_size: int = 5,
        enable_step_parallelization: bool = True,
        max_concurrent_steps: int = 2,
        max_concurrent_requests: int = 10,
        config = None,  # For compatibility
        embedded_text: List[models.EmbeddingsModel] = None  # Deprecated - for backward compatibility
    ):
        self.cluster_results = cluster_results
        # Note: embedded_text is no longer needed since ClusterModel contains embeddings
        if embedded_text is not None:
            logger.warning("embedded_text parameter is deprecated - embeddings are now in cluster_results")
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.enable_step_parallelization = enable_step_parallelization
        self.max_concurrent_steps = max_concurrent_steps
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize components
        self.model_config = ModelConfig()
        self.data_processor = CodebookDataProcessor(
            cluster_results=cluster_results,
            k=k
        )
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
    
    async def generate_async(self) -> Dict[str, Any]:
        """Generate codebook with hierarchical concurrency"""
        start_time = time.time()
        
        # Note: Section header will be handled by main pipeline, this is a sub-phase
        
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
        
        # Count total ideas
        total_ideas = sum(len(cluster_data['ideas']) for cluster_data in clusters.values())
        
        # Display configuration
        self.verbose_reporter.step_start("Inductive code generation from clusters", emoji="🔄")
        self.verbose_reporter.stat_line(f"Input: {len(clusters)} clusters with {total_ideas} ideas")
        self.verbose_reporter.stat_line(f"Starter codes: {len(self.starter_codes)}")
        self.verbose_reporter.stat_line("Configuration:")
        self.verbose_reporter.stat_line(f"Model (analysis): {self.model_config.get_model_for_stage('codebook_analysis')}", indent=1)
        self.verbose_reporter.stat_line(f"Model (validation): {self.model_config.get_model_for_stage('code_validation')}", indent=1)
        self.verbose_reporter.stat_line(f"Nearest neighbors (k): {self.k}", indent=1)
        self.verbose_reporter.stat_line(f"Batch size: {self.batch_size} clusters", indent=1)
        self.verbose_reporter.stat_line("Concurrency: 3-level hierarchical", indent=1)
        self.verbose_reporter.stat_line("Creating batches for concurrent processing and printing new codes ...")
        
        # Initialize  batch processor
        batch_processor = LangChainBatchProcessor(
            embedding_manager=embedding_manager,
            shared_codebook=shared_codebook,
            model_config=self.model_config,
            var_lab=self.var_lab,
            k=self.k,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
            enable_step_parallelization=self.enable_step_parallelization,
            max_concurrent_steps=self.max_concurrent_steps,
            max_concurrent_requests=self.max_concurrent_requests,
            verbose=self.verbose,
            prompt_printer=self.prompt_printer
        )
        
        # Process all clusters with hierarchical concurrency
        results = await batch_processor.process_all_clusters_concurrent(clusters)
        
        # Build cluster assignments and detailed results
        cluster_to_code = {}
        validation_details = {}
        step3_recommendations = {}
        step4_validated_codes = {}
        
        for result in results:
            cluster_id = result['cluster_id']
            if result.get('code'):
                cluster_to_code[cluster_id] = result['code']
            else:
                cluster_to_code[cluster_id] = result['status']
            
            # Store validation details if available
            if result.get('validation_details'):
                validation_details[cluster_id] = result['validation_details']
            
            # Store Step 3 recommendations if available
            if result.get('step3_recommendation'):
                step3_recommendations[cluster_id] = result['step3_recommendation']
            
            # Store Step 4 validated codes if available
            if result.get('step4_validated_code'):
                step4_validated_codes[cluster_id] = result['step4_validated_code']
        
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
            'avg_time_per_cluster': processing_time / len(clusters) if len(clusters) > 0 else 0,
            'performance_improvement_estimate': f"{((batch_processor.stats['parallel_steps_executed'] * 0.5) / processing_time * 100):.1f}% time saved from parallelization"
        }
        
        error_summary = {}
        if batch_processor.stats['errors'] > 0:
            error_summary["Errors"] = batch_processor.stats['errors']
        if batch_processor.stats['retries'] > 0:
            error_summary["Retries"] = batch_processor.stats['retries']
        if batch_processor.stats['partial_failures'] > 0:
            error_summary["Partial failures"] = batch_processor.stats['partial_failures']
        if batch_processor.stats['successful_recoveries'] > 0:
            error_summary["Successful recoveries"] = batch_processor.stats['successful_recoveries']
        
        # Complete processing
        self.verbose_reporter.step_complete("Cluster processing completed", emoji="✅")
        
        # Display final totals
        self.verbose_reporter.stat_line("Final totals:")
        self.verbose_reporter.stat_line(f"New codes added: {batch_processor.stats['new_codes_added']}", indent=1)
        self.verbose_reporter.stat_line(f"Codes modified: {batch_processor.stats['codes_modified']}", indent=1)
        self.verbose_reporter.stat_line(f"Existing codes used: {batch_processor.stats['no_new_codes_needed']}", indent=1)
        
        # Display codebook evolution
        self.verbose_reporter.stat_line(f"Codebook evolution: {len(self.starter_codes)} → {len(final_codes)} codes")
        
        # # Display sample new codes
        # if batch_processor.stats['new_codes_added'] > 0 and self.verbose:
        #     self._display_sample_new_codes(final_codes, self.starter_codes, step4_validated_codes)
        
        # # Display sample modified codes  
        # if batch_processor.stats['codes_modified'] > 0 and self.verbose:
        #     self._display_sample_modified_codes(step3_recommendations, step4_validated_codes)
        
        return {
            'codebook': final_codes,
            'cluster_assignments': cluster_to_code,
            'validation_details': validation_details,
            'step3_recommendations': step3_recommendations,
            'step4_validated_codes': step4_validated_codes,
            'stats': final_stats,
            'generator_version': 'HIERARCHICAL_CONCURRENCY_PARALLEL_STEPS'
        }
    
    # def _display_sample_new_codes(self, final_codes: List[Dict], starter_codes: List[Dict], validated_codes: Dict) -> None:
    #     """Display sample new codes that were added"""
    #     # Find codes that are in final_codes but not in starter_codes
    #     starter_code_names = {code['code'] for code in starter_codes}
    #     new_codes = [code for code in final_codes if code['code'] not in starter_code_names]
        
    #     if new_codes:
    #         self.verbose_reporter.empty_line()
    #         print("📋 Sample new codes added:")
            
    #         # Show up to 3 new codes
    #         num_samples = min(3, len(new_codes))
    #         for i, code in enumerate(new_codes[:num_samples]):
    #             definition = code['definition']
    #             if len(definition) > 80:
    #                 definition = definition[:77] + "..."
    #             print(f"  {i+1}. \"{code['code']}\" - {definition}")
            
    #         if len(new_codes) > num_samples:
    #             print(f"  ... and {len(new_codes) - num_samples} more new codes")
    
    # def _display_sample_modified_codes(self, step3_recommendations: Dict, validated_codes: Dict) -> None:
    #     """Display sample codes that were modified"""
    #     modifications = []
        
    #     # Find modifications from step3 recommendations
    #     for cluster_id, rec in step3_recommendations.items():
    #         if hasattr(rec, 'decision') and 'modify_existing' in rec.decision.lower():
    #             if hasattr(rec, 'action_details'):
    #                 original = getattr(rec.action_details, 'codes_to_modify', None)
    #                 modified = getattr(rec.action_details, 'modified_code_name', None)
    #                 if original and modified:
    #                     modifications.append((original, modified))
        
    #     if modifications:
    #         self.verbose_reporter.empty_line()
    #         print("📋 Sample code modifications:")
            
    #         # Show up to 3 modifications
    #         num_samples = min(3, len(modifications))
    #         for i, (original, modified) in enumerate(modifications[:num_samples]):
    #             print(f"  {i+1}. \"{original}\" → \"{modified}\"")
            
    #         if len(modifications) > num_samples:
    #             print(f"  ... and {len(modifications) - num_samples} more modifications")

    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for async generation"""
        return asyncio.run(self.generate_async())