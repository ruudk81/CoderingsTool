import os
import sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import time
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

import hashlib
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from prompts import SYSTEM_MESSAGE,  CODEBOOK_ANALYSIS_PROMPT, RESPONSE_SUMMARY_PROMPT, MATCH_AND_RECOMMEND_PROMPT, VALIDATION_PROMPT
from config import EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
from utils.verboseReporter import VerboseReporter

# === UTILS ========================================================================================================
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

class CoverageAssessment(BaseModel):
    """Coverage assessment details"""
    percentage: int = Field(description="Coverage percentage (0-100)", ge=0, le=100)
    rationale: str = Field(description="Explanation of what aspects are/aren't covered")

class ActionDetails(BaseModel):
    """Action details based on decision type"""
    codes_to_use: Optional[List[str]] = Field(default=None, description="List of codes if use_existing")
    code_to_modify: Optional[str] = Field(default=None, description="Code name if modify_existing")
    modification_suggestion: Optional[str] = Field(default=None, description="How to broaden if modify_existing")
    new_code_name: Optional[str] = Field(default=None, description="New code name if create_new")
    new_code_definition: Optional[str] = Field(default=None, description="New code definition if create_new")

class MatchRecommendation(BaseModel):
    """Output for match and recommend step (Step 3) - single recommendation per cluster"""
    cluster_core_theme: str = Field(description="The core theme identified from cluster analysis")
    best_matching_codes: List[str] = Field(description="Best matching existing codes")
    coverage_assessment: CoverageAssessment = Field(description="Coverage percentage and rationale")
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
    code: Optional[str] = Field(default=None, description="Final code name (approved/revised) or null if rejected")
    definition: Optional[str] = Field(default=None, description="Final definition (approved/revised) or null if rejected")

class ValidationResponse(BaseModel):
    """Output for validation step (Step 4)"""
    evaluation: ValidationEvaluation = Field(description="Detailed evaluation scores and reasoning")
    decision: str = Field(description="Overall decision: APPROVE/REVISE/REJECT")
    decision_rationale: str = Field(description="Explanation for the overall decision")
    validated_code: ValidatedCode = Field(description="Final validated code if approved")

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
        """Get embeddings for a codebook snapshot like v2 - always fresh, no version caching"""
        if not codes:
            return [], []
        
        # Generate embeddings fresh each time (like v2)
        #code_texts = [f"{code['code']}: {code['definition']}" for code in codes]
        code_texts = [f"{code['code']}" for code in codes]
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
        """Initialize chains"""
        
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
        
        #self.llm = self.step2_llm
        
        # Step 1: Codebook Analysis Chain (uses step1_llm with Pydantic validation)
        codebook_prompt = PromptTemplate(
            template=CODEBOOK_ANALYSIS_PROMPT,
            input_variables=["system_message", "language", "survey_question", "code_text"]
        )
        
        self.codebook_chain = (
            codebook_prompt 
            | self.step1_llm 
            | StrOutputParser()
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 2: Response Summary Chain (uses step2_llm)
        summary_prompt = PromptTemplate(
            template=RESPONSE_SUMMARY_PROMPT,
            input_variables=["system_message", "language", "survey_question", "cluster_text"]
        )
        
        self.summary_chain = (
            summary_prompt
            | self.step2_llm
            | StrOutputParser()
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 3: Match and Recommend Chain (uses step3_llm)
        match_prompt = PromptTemplate(
            template=MATCH_AND_RECOMMEND_PROMPT,
            input_variables=["system_message", "survey_question", "existing_codes", "clustered_ideas", "codebook_analysis", "summaries"]
        )
        
        self.match_chain = (
            match_prompt
            | self.step3_llm
            | PydanticOutputParser(pydantic_object=MatchRecommendation)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 4: Validation Chain (uses step4_llm)
        validation_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["system_message", "survey_question", "existing_codes", "clustered_ideas", "step3_recommendation"]
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
    
    def _format_step3_recommendation(self, recommendation: MatchRecommendation) -> str:
        """Format prompt 3 for Step 4"""
        if not recommendation:
            return "No recommendation available"
            
        formatted = f"""
Decision: {recommendation.decision.replace('_', ' ').title()}
"""
# Cluster Theme: {recommendation.cluster_core_theme}
# Coverage Assessment: {recommendation.coverage_assessment.rationale}
# Best Matching Codes: {', '.join(recommendation.best_matching_codes)}
#Action Details:

        
        if recommendation.action_details.codes_to_use:
            formatted += f"- Codes to use: {', '.join(recommendation.action_details.codes_to_use)}\n"
        if recommendation.action_details.code_to_modify:
            formatted += f"- Code to modify: {recommendation.action_details.code_to_modify}\n"
            formatted += f"- Modification: {recommendation.action_details.modification_suggestion}\n"
        if recommendation.action_details.new_code_name:
            formatted += f"- New code: {recommendation.action_details.new_code_name}\n"
            #formatted += f"- Definition: {recommendation.action_details.new_code_definition}\n"
            
        #formatted += f"\nJustification: {recommendation.justification}"
        
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
    
    async def process_batch_langchain(self, batch_clusters: List[Tuple[int, Dict]]) -> List[Dict[str, Any]]:
        """Process a batch using V2 multi-step approach with V3 nearest codes logic"""
        batch_results = []
        
        for cluster_id, cluster_data in batch_clusters:
            try:
                start_time = time.time()
                llm_start = time.time()  # Track LLM time separately
                
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
                    summaries = "Analysis failed due to API error after retries."
                except Exception as e:
                    logger.error(f"Step 2 unexpected error for cluster {cluster_id}: {str(e)}")
                    summaries = "Analysis failed due to processing error."
                
                # Step 3: Match and recommend
                match_input = {
                    "system_message": SYSTEM_MESSAGE.format(language=DEFAULT_LANGUAGE),
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "existing_codes": code_text,
                    "clustered_ideas": cluster_text,
                    "codebook_analysis": codebook_analysis if isinstance(codebook_analysis, str) else str(codebook_analysis),
                    "summaries": summaries if isinstance(summaries, str) else str(summaries)
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
                
                # Extract new code recommendations from single MatchRecommendation object
                new_codes_needed = False
                proposed_codes = []
                
                try:
                    if hasattr(recommendations, 'decision'):
                        decision = recommendations.decision.lower()
                        
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
                                original_code = recommendations.action_details.code_to_modify
                                modification = recommendations.action_details.modification_suggestion
                                
                                # Only proceed if we have modification details
                                if original_code and modification and original_code.strip() and modification.strip():
                                    new_codes_needed = True
                                    proposed_codes.append({
                                        'original_code': original_code.strip(),
                                        'modification_type': 'modify_existing',
                                        'modification_suggestion': modification.strip()
                                    })
                        
                        # Store the full recommendation for potential Step 4 use
                        cluster_recommendation = recommendations
                        
                    # Fallback for dict format (backwards compatibility)
                    elif isinstance(recommendations, dict) and 'decision' in recommendations:
                        decision = recommendations.get('decision', '').lower()
                        
                        if 'create_new' in decision:
                            action_details = recommendations.get('action_details', {})
                            new_code = action_details.get('new_code_name')
                            new_definition = action_details.get('new_code_definition')
                            
                            # Only add if we have actual code and definition (not null/empty)
                            if new_code and new_definition and new_code.strip() and new_definition.strip():
                                new_codes_needed = True
                                proposed_codes.append({
                                    'code': new_code.strip(),
                                    'definition': new_definition.strip()
                                })
                        elif 'modify_existing' in decision:
                            # Trigger Step 4 for modification validation
                            action_details = recommendations.get('action_details', {})
                            original_code = action_details.get('code_to_modify')
                            modification = action_details.get('modification_suggestion')
                            
                            # Only proceed if we have modification details
                            if original_code and modification and original_code.strip() and modification.strip():
                                new_codes_needed = True
                                proposed_codes.append({
                                    'original_code': original_code.strip(),
                                    'modification_type': 'modify_existing',
                                    'modification_suggestion': modification.strip()
                                })
                except Exception as e:
                    logger.error(f"Error parsing recommendations for cluster {cluster_id}: {e}")
                    logger.error(f"Recommendations content: {recommendations}")
                
                if not new_codes_needed:
                    self.stats['no_new_codes_needed'] += 1
                    batch_results.append({
                        'cluster_id': cluster_id,
                        'status': 'no_new_code_needed',
                        'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                        'step4_validated_code': None,  # No Step 4 for use_existing
                        'processing_time': time.time() - start_time
                    })
                    self.stats['clusters_processed'] += 1
                    continue
                
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
                        "system_message": SYSTEM_MESSAGE.format(language=DEFAULT_LANGUAGE),
                        "language": DEFAULT_LANGUAGE,
                        "survey_question": self.var_lab,
                        "existing_codes": definition_code_text,  # Now using definition-based codes
                        "clustered_ideas": cluster_text,
                        "step3_recommendation": formatted_recommendation
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
                            
                            if validation_results.decision in ['APPROVE', 'REVISE'] and validation_results.validated_code.code:
                                validated_code = {
                                    'code': validation_results.validated_code.code,
                                    'definition': validation_results.validated_code.definition
                                }
                        elif isinstance(validation_results, dict):
                            # Fallback for old format
                            validated_codes = validation_results.get('validated_codes', [])
                            if validated_codes:
                                validated_code = validated_codes[0]
                        else:
                            logger.error(f"Unexpected validation result format for cluster {cluster_id}: {validation_results}")
                    except Exception as e:
                        logger.error(f"Error parsing validation results for cluster {cluster_id}: {e}")
                        logger.error(f"Validation results content: {validation_results}")
                    
                    if validated_code:
                        # Check if this is a modification or new code
                        is_modification = any(pc.get('modification_type') == 'modify_existing' for pc in proposed_codes)
                        
                        if is_modification:
                            # Get original code name from proposed_codes
                            original_code_name = next(
                                (pc.get('original_code') for pc in proposed_codes if pc.get('modification_type') == 'modify_existing'),
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
                                    self.stats['new_codes_added'] += 1  # Count as modification
                                    if self.verbose:
                                        # Smart change detection logging
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
                                            logger.info(f"  Old: {original_definition[:100] if original_definition else 'N/A'}...")
                                            logger.info(f"  New: {new_definition[:100]}...")
                                        else:
                                            logger.info(f"Cluster {cluster_id}: Code '{original_code_name}' processed (no changes detected) (v{new_version})")
                                
                                batch_results.append({
                                    'cluster_id': cluster_id,
                                    'status': 'code_modified',
                                    'original_code': original_code_name,
                                    'code': validated_code.get('code', ''),
                                    'definition': validated_code.get('definition', ''),
                                    'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                                    'step4_validated_code': validated_code,
                                    'validation_details': validation_details,
                                    'processing_time': time.time() - start_time
                                })
                            else:
                                # Fallback to add as new if original not found
                                added, new_version = await self.shared_codebook.add_code_if_new(
                                    validated_code.get('code', ''),
                                    validated_code.get('definition', '')
                                )
                                
                                batch_results.append({
                                    'cluster_id': cluster_id,
                                    'status': 'modification_fallback_to_new',
                                    'code': validated_code.get('code', ''),
                                    'definition': validated_code.get('definition', ''),
                                    'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                                    'step4_validated_code': validated_code,
                                    'validation_details': validation_details,
                                    'processing_time': time.time() - start_time
                                })
                        else:
                            # Standard new code addition
                            added, new_version = await self.shared_codebook.add_code_if_new(
                                validated_code.get('code', ''),
                                validated_code.get('definition', '')
                            )
                            
                            if added:
                                self.stats['new_codes_added'] += 1
                                if self.verbose:
                                    logger.info(f"Cluster {cluster_id}: Added new code '{validated_code['code']}' (v{new_version}) - NOW AVAILABLE for subsequent clusters")
                            
                            batch_results.append({
                                'cluster_id': cluster_id,
                                'status': 'new_code_added' if added else 'code_already_exists',
                                'code': validated_code.get('code', ''),
                                'definition': validated_code.get('definition', ''),
                                'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                                'step4_validated_code': validated_code,
                                'validation_details': validation_details,
                                'processing_time': time.time() - start_time
                            })
                    else:
                        batch_results.append({
                            'cluster_id': cluster_id,
                            'status': 'no_codes_passed_validation',
                            'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                            'step4_validated_code': None,  # No validated code if none passed validation
                            'validation_details': validation_details,
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
                    'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
                    'step4_validated_code': None,  # No validated code if error occurred
                    'processing_time': time.time() - start_time if 'start_time' in locals() else 0
                })
                self.stats['errors'] += 1
        
        return batch_results
    
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
            if pc.get('modification_type') == 'modify_existing':
                original_code = pc.get('original_code')
                modification = pc.get('modification_suggestion')
                
                if original_code and modification:
                    # Get the original definition from shared codebook (async)
                    original_definition = await self.shared_codebook.get_code_definition(original_code)
                    
                    if original_definition:
                        # Combine original definition with modification suggestion
                        return f"{original_definition}. Modified to include: {modification}"
                    else:
                        # Fallback to just the modification suggestion
                        return modification
        
        return None

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
    """V4 Generator using LangChain optimization and shared memory pattern"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel], 
        starter_codes: List[Dict[str, str]], 
        var_lab: str, 
        k: int = 5,
        verbose: bool = False, 
        prompt_printer = None,
        batch_size: int = 10,
        max_concurrent_requests: int = 5,
        config = None,  # For compatibility
        embedded_text: List[models.EmbeddingsModel] = None  # Deprecated - for backward compatibility
    ):
        logger.info("🚀 INITIALIZING CODEBOOK GENERATOR V4 (Clean model inheritance)")
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
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize components
        self.model_config = ModelConfig()
        self.data_processor = CodebookDataProcessor(
            cluster_results=cluster_results,
            k=k
        )
        self.verbose_reporter = VerboseReporter(verbose)
    
    async def generate_async(self) -> Dict[str, Any]:
        """Generate codebook with LangChain optimization"""
        start_time = time.time()
        
        logger.info("🚀 STARTING V4 CODEBOOK GENERATION (Multi-step prompts)")
        self.verbose_reporter.section_header("CODEBOOK GENERATION - MULTI-STEP PROMPTS", emoji="🔥")
        
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
            'validation_details': validation_details,
            'step3_recommendations': step3_recommendations,
            'step4_validated_codes': step4_validated_codes,
            'stats': final_stats,
            'generator_version': 'V4_MULTISTEP_PROMPTS_WITH_V3_FEATURES'  # Clear identifier
        }
    
    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for async generation"""
        return asyncio.run(self.generate_async())
   