import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
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
from prompts import CLUSTER_SUMMARY_PROMPT, CANDIDATE_CODE_SELECTION_PROMPT, CODE_GENERATION_PROMPT, VALIDATION_PROMPT
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

class ThemeEntry(BaseModel):
    theme_id: int = Field(description="Sequential theme ID starting from 1")
    theme_name: str = Field(description="Short noun phrase for theme name")
    summary: str = Field(description="Theme summary in ≤25 words")

class ClusterSummaryOutput(RootModel[List[ThemeEntry]]):
    root: List[ThemeEntry] = Field(description="Array of themes with ID, name, and summary")

class CandidateCode(BaseModel):
    code: str = Field(description="Exact code name from existing codebook")
    definition: str = Field(description="Exact definition from existing codebook")

class CandidateCodeSelectionOutput(RootModel[List[CandidateCode]]):
    root: List[CandidateCode] = Field(description="Array of candidate codes for themes")

class ActionDetails(BaseModel):
    codes_to_use: Optional[List[str]] = Field(default=None, description="List of codes if use_existing")
    codes_to_modify: Optional[str] = Field(default=None, description="Single code name if modify_existing")
    modified_code_name: Optional[str] = Field(default=None, description="Modified code name if create_new")
    modified_code_definition: Optional[str] = Field(default=None, description="Modified code definition if create_new")
    new_code_name: Optional[str] = Field(default=None, description="New code name if create_new")
    new_code_definition: Optional[str] = Field(default=None, description="New code definition if create_new")

class ClusterAnalysis(BaseModel):
    number_of_themes: int = Field(description="Number of themes identified")
    theme_descriptions: List[str] = Field(description="Brief descriptions of each theme")

class CodingDecision(BaseModel):
    theme_number: int = Field(description="Theme number being processed")
    theme_description: str = Field(description="What this theme is about")
    decision: str = Field(description="use_existing|modify_existing|create_new")
    action_details: ActionDetails = Field(description="Action details based on decision")
    justification: str = Field(description="Why this action is appropriate for this theme")

class CodeGenerationOutput(BaseModel):
    cluster_analysis: ClusterAnalysis = Field(description="Analysis of themes in cluster")
    coding_decisions: List[CodingDecision] = Field(description="Coding decision for each theme")
    overall_justification: str = Field(description="Why treating these as separate themes improves codebook quality")

class ThemeAssessment(BaseModel):
    number_of_themes_identified: int = Field(description="Number of themes identified")
    theme_separation_valid: bool = Field(description="Whether theme separation is valid")
    theme_separation_reasoning: str = Field(description="Are themes distinct or should they be merged/split")

class CodeEvaluation(BaseModel):
    semantic_fit: Optional[str] = Field(default=None, description="Assessment of semantic fit")
    atomicity: Optional[str] = Field(default=None, description="Assessment of atomicity")
    parsimony: Optional[str] = Field(default=None, description="Assessment of parsimony")
    redundancy: Optional[str] = Field(default=None, description="Assessment of redundancy")

class ValidatedCode(BaseModel):
    code: str = Field(description="Final code name")
    definition: str = Field(description="Final definition")

class CodeValidation(BaseModel):
    theme_number: int = Field(description="Theme number being validated")
    theme_description: str = Field(description="What theme is being coded")
    original_recommendation: str = Field(description="What was proposed")
    evaluation: CodeEvaluation = Field(description="Evaluation of the recommendation")
    decision: str = Field(description="APPROVE|REVISE|REJECT|MERGE|SPLIT")
    decision_rationale: str = Field(description="Explanation")
    validated_code: Union[ValidatedCode, List[ValidatedCode]] = Field(description="Final code(s) - single for APPROVE/REVISE/REJECT, list for SPLIT")

class OverallValidation(BaseModel):
    all_themes_coded: bool = Field(description="Whether all themes were successfully coded")
    final_code_count: int = Field(description="Final number of codes")
    summary: str = Field(description="Brief summary of validation outcome")

class ValidationOutput(BaseModel):
    theme_assessment: ThemeAssessment = Field(description="Assessment of theme separation")
    code_validations: List[CodeValidation] = Field(description="Validation of each code")
    overall_validation: OverallValidation = Field(description="Overall validation summary")

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
# TOKEN ESTIMATION FOR DURATION PREDICTION
# ============================================================================

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    
    # Remove extra whitespace and count characters
    cleaned_text = ' '.join(text.split())
    char_count = len(cleaned_text)
    
    # Rough token estimation: 4 chars per token
    # Add small buffer for punctuation and special tokens
    estimated_tokens = int(char_count / 4) + 10
    return max(estimated_tokens, 1)

def estimate_code_list_tokens(codes: list) -> int:
    if not codes:
        return 10  # Empty list still has structure tokens
    
    total_chars = 0
    for code in codes:
        if isinstance(code, dict):
            total_chars += len(code.get('code', '')) + len(code.get('definition', ''))
        else:
            total_chars += len(str(code))
    
    # Add JSON structure overhead (brackets, quotes, commas)
    structure_overhead = len(codes) * 20  # ~20 chars per code for JSON structure
    return int((total_chars + structure_overhead) / 4) + 15

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_cluster_summary(themes, analyst_note: str = None) -> str:
    """Generate cluster summary from themes using new format with theme_name"""
    if not themes:
        return "This cluster lacks coherent themes."
    
    # Handle new format (list of ThemeEntry objects or dicts with theme_id, theme_name, summary)
    if isinstance(themes, list) and themes:
        first_theme = themes[0]
        if hasattr(first_theme, 'theme_name'):
            # ThemeEntry object format - use theme_name
            if len(themes) == 1:
                return first_theme.theme_name
            else:
                theme_parts = [f"Theme {theme.theme_id}: {theme.theme_name}" for theme in themes]
                return "\n".join(theme_parts)
        elif isinstance(first_theme, dict) and 'theme_name' in first_theme:
            # Dict format with theme_id, theme_name, summary - use theme_name
            if len(themes) == 1:
                return first_theme['theme_name']
            else:
                theme_parts = [f"Theme {theme['theme_id']}: {theme['theme_name']}" for theme in themes]
                return "\n".join(theme_parts)
    
    return "This cluster lacks coherent themes."

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
    _embedding_cache: Dict[int, List[np.ndarray]] = None
    _max_cached_versions: int = 5  # Keep only recent versions in cache
    
    def __init__(self, initial_codes: List[Dict[str, str]], max_cached_versions: int = 5):
        self._codes = initial_codes.copy()
        self._lock = asyncio.Lock()
        self._version = 0
        self._update_log = []
        self._embedding_cache = {}
        self._max_cached_versions = max_cached_versions
    
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
        """Cache embeddings for a version with memory management"""
        async with self._lock:
            self._embedding_cache[version] = embeddings
            
            # Memory management: keep only recent versions
            if len(self._embedding_cache) > self._max_cached_versions:
                # Find and remove oldest version
                oldest_version = min(self._embedding_cache.keys())
                del self._embedding_cache[oldest_version]
                logger.debug(f"Evicted embedding cache for version {oldest_version} (keeping last {self._max_cached_versions} versions)")
    
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
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'api_calls_saved': 0
        }
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def get_snapshot_embeddings(self, codes: List[Dict[str, str]], version: int) -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
        """Get embeddings for a codebook snapshot with version-based caching"""
        if not codes:
            return [], []
        
        # Check if we have cached embeddings for this version
        cached_embeddings = await self.shared_codebook.get_embeddings_for_version(version)
        
        if cached_embeddings is not None:
            # Cache hit - return cached embeddings
            self.cache_stats['hits'] += 1
            self.cache_stats['api_calls_saved'] += len(codes)
            if self.verbose:
                logger.info(f"Embedding cache HIT for version {version} ({len(codes)} codes)")
            return codes, cached_embeddings
        
        # Cache miss - generate new embeddings
        self.cache_stats['misses'] += 1
        if self.verbose:
            logger.info(f"Embedding cache MISS for version {version} - generating embeddings for {len(codes)} codes")
        
        code_texts = [f"{code['code']}: {code['definition']}" for code in codes]
        embeddings = await self._embed_texts_with_retry(code_texts)
        
        # Cache the embeddings for this version
        await self.shared_codebook.cache_embeddings(version, embeddings)
        
        return codes, embeddings
    
    @retry(**FAST_EMBEDDING_RETRY_CONFIG)
    async def _embed_texts_with_retry(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts with retry logic for API failures"""
        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
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

    async def generate_theme_embedding(self, theme_text: str) -> np.ndarray:
        """Generate embedding for a single theme text"""
        try:
            embeddings = await self._embed_texts_with_retry([theme_text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding for theme: {theme_text[:50]}... Error: {e}")
            # Return a zero vector as fallback
            return np.zeros(1536, dtype=np.float32)  # OpenAI embedding dimension
  
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
            'embedding_cache_hits': 0,
            'embedding_cache_misses': 0,
            'embedding_api_calls_saved': 0,
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
        
        # Step 1: Cluster Summary Chain
        self.step1_llm = ChatOpenAI(
            **self.model_config.get_langchain_config_for_stage("cluster_summarization")
        )
        
        # Step 2: Candidate Code Selection Chain
        self.step2_llm = ChatOpenAI(
            **self.model_config.get_langchain_config_for_stage("candidate_code_selection")
        )
        
        # Step 3: Code Generation Chain
        self.step3_llm = ChatOpenAI(
            **self.model_config.get_langchain_config_for_stage("code_generation_recommendation")
        )
        
        # Step 4: Validation Chain
        self.step4_llm = ChatOpenAI(
            **self.model_config.get_langchain_config_for_stage("recommendation_validation")
        )
        
        # Step 1: Cluster Summary Chain
        cluster_summary_prompt = PromptTemplate(
            template=CLUSTER_SUMMARY_PROMPT,
            input_variables=["language", "survey_question", "cluster_text"]
        )
        
        self.cluster_summary_chain = (
            cluster_summary_prompt 
            | self.step1_llm 
            | PydanticOutputParser(pydantic_object=ClusterSummaryOutput)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 2: Candidate Code Selection Chain
        candidate_code_prompt = PromptTemplate(
            template=CANDIDATE_CODE_SELECTION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "code_text"]
        )
        
        self.candidate_code_chain = (
            candidate_code_prompt
            | self.step2_llm
            | PydanticOutputParser(pydantic_object=CandidateCodeSelectionOutput)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 3: Code Generation Chain
        code_generation_prompt = PromptTemplate(
            template=CODE_GENERATION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "candidate_codes"]
        )
        
        self.code_generation_chain = (
            code_generation_prompt
            | self.step3_llm
            | PydanticOutputParser(pydantic_object=CodeGenerationOutput)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Step 4: Validation Chain
        validation_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "candidate_codes", "step3_recommendation"]
        )
        
        self.validation_chain = (
            validation_prompt
            | self.step4_llm
            | PydanticOutputParser(pydantic_object=ValidationOutput)
        ).with_config({"max_concurrency": self.max_concurrent_requests})
        
        # Initialize capture counts and step diversity tracking (NEW ARCHITECTURE)
        self._capture_counts = {
            'cluster_summary': 0,
            'candidate_codes': 0,
            'code_generation': 0,
            'validation': 0
        }
        
        # Track which steps we've captured at least once (for guaranteed diversity)
        self._captured_steps = set()
        self._all_steps = {'cluster_summary', 'candidate_codes', 'code_generation', 'validation'}
        self._diversity_complete = False

    def _split_into_sub_batches(self, batch_clusters: List[Tuple[int, Dict]]) -> List[List[Tuple[int, Dict]]]:
        """Split batch into sub-batches for hierarchical processing"""
        sub_batches = []
        for i in range(0, len(batch_clusters), self.sub_batch_size):
            sub_batch = batch_clusters[i:i + self.sub_batch_size]
            sub_batches.append(sub_batch)
        return sub_batches


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

    async def _process_step_with_retry(self, step_num: int, inputs: Dict) -> Dict:
        """Process any LLM step with retry logic
        
        Args:
            step_num: Step number (1-4)
            inputs: Input dictionary for the chain
            
        Returns:
            Chain output dictionary
        """
        chains = {
            1: self.cluster_summary_chain,
            2: self.candidate_code_chain,
            3: self.code_generation_chain,
            4: self.validation_chain
        }
        
        @retry(**FAST_API_RETRY_CONFIG)
        async def _inner():
            try:
                return await chains[step_num].ainvoke(inputs)
            except Exception as e:
                error_type = classify_error(e)
                if error_type in [ErrorType.API_RATE_LIMIT, ErrorType.API_TIMEOUT, 
                                ErrorType.API_SERVER_ERROR, ErrorType.NETWORK_ERROR]:
                    self.stats['retries'] += 1
                    raise APIError(f"Step {step_num} processing error: {str(e)}", error_type)
                else:
                    raise ProcessingError(f"Step {step_num} processing error: {str(e)}", error_type)
        
        return await _inner()

    def _format_code_generation_result(self, code_generation_result) -> str:
        """Format code generation result for validation prompt"""
        if not code_generation_result or not hasattr(code_generation_result, 'coding_decisions'):
            return "No code generation result available"
        
        formatted_parts = []
        formatted_parts.append(f"Number of themes: {code_generation_result.cluster_analysis.number_of_themes}")
        
        for decision in code_generation_result.coding_decisions:
            formatted_parts.append(f"\nTheme {decision.theme_number}: {decision.theme_description}")
            formatted_parts.append(f"Decision: {decision.decision}")
            
            if decision.decision == 'use_existing' and decision.action_details.codes_to_use:
                formatted_parts.append(f"Codes to use: {', '.join(decision.action_details.codes_to_use)}")
            elif decision.decision == 'modify_existing':
                formatted_parts.append(f"Code to modify: {decision.action_details.codes_to_modify}")
                formatted_parts.append(f"New name: {decision.action_details.modified_code_name}")
                formatted_parts.append(f"New definition: {decision.action_details.modified_code_definition}")
            elif decision.decision == 'create_new':
                formatted_parts.append(f"New code: {decision.action_details.new_code_name}")
                formatted_parts.append(f"Definition: {decision.action_details.new_code_definition}")
                
            formatted_parts.append(f"Justification: {decision.justification}")
        
        return "\n".join(formatted_parts)

    async def _find_candidate_codes_for_themes(self, themes, cluster_id: int) -> List[Dict[str, str]]:
        """Find candidate codes for themes using per-theme embeddings"""
        all_candidate_codes = []
        
        # Extract theme names 
        theme_names = []
        if isinstance(themes, list) and themes:
            first_theme = themes[0]
            if hasattr(first_theme, 'theme_name'):
                theme_names = [theme.theme_name for theme in themes]
            elif isinstance(first_theme, dict) and 'theme_name' in first_theme:
                theme_names = [theme['theme_name'] for theme in themes]
        
        total = len(theme_names)
        for theme_idx, theme_name in enumerate(theme_names):
            idx = theme_idx + 1  # Current theme number (1-based)
            try:
                # Generate embedding for this specific theme name
                theme_embedding = await self.embedding_manager.generate_theme_embedding(theme_name)
                
                # Find nearest codes for this theme
                nearest_codes = await self._find_nearest_codes(theme_embedding)
                
                if nearest_codes:
                    all_candidate_codes.extend(nearest_codes)
                    print(f"{idx}/{total} {len(nearest_codes)} candidate codes")
                else:
                    print(f"{idx}/{total} 0 candidate codes")
                    # if self.verbose:
                    #     logger.info(f"  Theme {theme_idx + 1}: Found {len(nearest_codes)} candidate codes")
                        
            except Exception as e:
                logger.error(f"Failed to find codes for theme {theme_idx + 1} in cluster {cluster_id}: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen_codes = set()
        unique_codes = []
        for code in all_candidate_codes:
            code_key = f"{code['code']}::{code['definition']}"
            if code_key not in seen_codes:
                seen_codes.add(code_key)
                unique_codes.append(code)
        
        return unique_codes

    async def _process_multi_theme_pipeline(self, cluster_id: int, cluster_data: Dict, cluster_text: str) -> Dict[str, Any]:
        """Process cluster through the new 4-step multi-theme pipeline"""
        start_time = time.time()
        
        # Step 1: Extract themes from cluster
        step1_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_text": cluster_text
        }
        
        try:
            # Capture cluster summary prompt if needed
            if self._should_capture_prompt('cluster_summary'):
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
                    utility_name="LangChainBatchProcessor", 
                    prompt_content=CLUSTER_SUMMARY_PROMPT.format(**step1_input),
                    prompt_type="step1_cluster_summary",
                    metadata={
                        "model": self.step1_llm.model_name,
                        "var_lab": self.var_lab,
                        "stage": "1/4 - Multi-theme Cluster Summary",
                        "cluster_id": cluster_id
                    }
                )
                self._record_capture('cluster_summary')
            
            cluster_summary_result = await self._process_step_with_retry(1, step1_input)
            if not cluster_summary_result or not cluster_summary_result.root:
                return {
                    'cluster_id': cluster_id,
                    'status': 'no_themes_found',
                    'themes': [],
                    'processing_time': time.time() - start_time
                }
        except Exception as e:
            logger.error(f"Step 1 (theme extraction) failed for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'theme_extraction_failed',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        # Step 2: Get candidate codes for all themes (using per-theme embedding)
        candidate_codes = await self._find_candidate_codes_for_themes(
            cluster_summary_result.root, cluster_id
        )
        
        if not candidate_codes:
            code_text = "No existing codes in codebook"
        else:
            code_text = "\n".join([
                f"- {code['code']}: {code['definition']}" 
                for code in candidate_codes
            ])
        
        step2_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
            "code_text": code_text
        }
        
        try:
            # Capture candidate code selection prompt if needed
            if self._should_capture_prompt('candidate_codes'):
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
                    utility_name="LangChainBatchProcessor",
                    prompt_content=CANDIDATE_CODE_SELECTION_PROMPT.format(**step2_input),
                    prompt_type="step2_candidate_codes",
                    metadata={
                        "model": self.step2_llm.model_name,
                        "var_lab": self.var_lab,
                        "stage": "2/4 - Multi-theme Candidate Selection",
                        "cluster_id": cluster_id,
                        "themes_count": len(cluster_summary_result.root)
                    }
                )
                self._record_capture('candidate_codes')
            
            candidate_code_result = await self._process_step_with_retry(2, step2_input)
            selected_codes = candidate_code_result.root if hasattr(candidate_code_result, 'root') else []
        except Exception as e:
            error_msg = str(e)
            if "validation errors" in error_msg and "Field required" in error_msg:
                logger.warning(f"Step 2 parsing error for cluster {cluster_id}: LLM returned malformed JSON with empty objects. Continuing with no selected codes.")
                if self.verbose:
                    print(f"  ⚠️  Cluster {cluster_id}: Step 2 LLM output parsing failed - continuing")
            else:
                logger.error(f"Step 2 (candidate code selection) failed for cluster {cluster_id}: {e}")
            selected_codes = []
        
        # Step 3: Generate code recommendations for all themes
        selected_codes_text = "\n".join([
            f"- {code.code}: {code.definition}" 
            for code in selected_codes
        ]) if selected_codes else "No codes selected"
        
        step3_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
            "candidate_codes": selected_codes_text
        }
        
        try:
            # Capture code generation prompt if needed
            if self._should_capture_prompt('code_generation'):
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
                    utility_name="LangChainBatchProcessor",
                    prompt_content=CODE_GENERATION_PROMPT.format(**step3_input),
                    prompt_type="step3_code_generation",
                    metadata={
                        "model": self.step3_llm.model_name,
                        "var_lab": self.var_lab,
                        "stage": "3/4 - Multi-theme Code Generation",
                        "cluster_id": cluster_id,
                        "themes_count": len(cluster_summary_result.root),
                        "selected_codes_count": len(selected_codes)
                    }
                )
                self._record_capture('code_generation')
            
            code_generation_result = await self._process_step_with_retry(3, step3_input)
        except Exception as e:
            logger.error(f"Step 3 (code generation) failed for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'code_generation_failed',
                'error': str(e),
                'cluster_summary': cluster_summary_result,
                'themes': cluster_summary_result.root,
                'processing_time': time.time() - start_time
            }
        
        # Check if any codes need validation (create_new or modify_existing)
        needs_validation = any(
            decision.decision in ['create_new', 'modify_existing']
            for decision in code_generation_result.coding_decisions
        )
        
        if not needs_validation:
            # All decisions are use_existing - no validation needed
            final_codes = self._extract_final_codes(code_generation_result, selected_codes)
            
            # REAL-TIME CODEBOOK UPDATE: Even for use_existing, update codebook if there are new codes
            codebook_updates = []
            if final_codes:
                for final_code in final_codes:
                    if final_code['decision'] == 'create_new':
                        # Add new code to shared codebook immediately
                        added, version = await self.shared_codebook.add_code_if_new(
                            final_code['code'], final_code['definition']
                        )
                        if added:
                            self.stats['new_codes_added'] += 1  # Track stats
                            codebook_updates.append({
                                'action': 'added',
                                'code': final_code['code'],
                                'version': version,
                                'theme_number': final_code.get('theme_number', 'unknown')
                            })
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: REAL-TIME added '{final_code['code']}' (v{version}) - NOW AVAILABLE to concurrent clusters")
            
            return {
                'cluster_id': cluster_id,
                'status': 'no_validation_needed',
                'cluster_summary': cluster_summary_result,
                'themes': cluster_summary_result.root,
                # Store ACTUAL prompt inputs (exactly what each prompt receives)
                'step1_input': {
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "cluster_text": cluster_text
                },
                'step2_input': {
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
                    "code_text": code_text
                },
                'step3_input': {
                    "language": DEFAULT_LANGUAGE,
                    "survey_question": self.var_lab,
                    "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
                    "candidate_codes": selected_codes_text
                },
                # Store outputs for analysis
                'step1_output': cluster_summary_result.root,
                'step2_output': selected_codes,
                'step3_output': code_generation_result,
                'code_generation_result': code_generation_result,
                'final_codes': final_codes,
                'codebook_updates': codebook_updates,
                'processing_time': time.time() - start_time
            }
        
        # Step 4: Validate proposed codes
        formatted_recommendation = self._format_code_generation_result(code_generation_result)
        
        step4_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
            "candidate_codes": selected_codes_text,
            "step3_recommendation": formatted_recommendation
        }
        
        try:
            # Capture validation prompt if needed
            if self._should_capture_prompt('validation'):
                self.prompt_printer.capture_prompt(
                    step_name="codebook_generation",
                    utility_name="LangChainBatchProcessor",
                    prompt_content=VALIDATION_PROMPT.format(**step4_input),
                    prompt_type="step4_validation",
                    metadata={
                        "model": self.step4_llm.model_name,
                        "var_lab": self.var_lab,
                        "stage": "4/4 - Multi-theme Validation",
                        "cluster_id": cluster_id,
                        "themes_count": len(cluster_summary_result.root)
                    }
                )
                self._record_capture('validation')
            
            validation_result = await self._process_step_with_retry(4, step4_input)
        except Exception as e:
            logger.error(f"Step 4 (validation) failed for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'validation_failed',
                'error': str(e),
                'cluster_summary': cluster_summary_result,
                'themes': cluster_summary_result.root,
                'code_generation_result': code_generation_result,
                'processing_time': time.time() - start_time
            }
        
        # Extract final validated codes
        validated_codes = []
        for validation in validation_result.code_validations:
            # Map Step 4 validation decision back to Step 3 decision type
            # Find the original Step 3 decision for this theme
            original_decision = 'use_existing'  # default
            for decision in code_generation_result.coding_decisions:
                if decision.theme_number == validation.theme_number:
                    original_decision = decision.decision
                    break
            
            # Only include codes that were APPROVED, REVISED, or SPLIT (not REJECTED)
            if validation.decision in ['APPROVE', 'REVISE', 'SPLIT']:
                # Determine the effective decision for codebook updates
                # Use Step 4 validation decision for new code creation, otherwise use original
                effective_decision = 'create_new' if validation.decision in ['REVISE', 'SPLIT'] else original_decision
                
                if validation.decision == 'SPLIT':
                    # Handle SPLIT case - multiple codes from one theme
                    if isinstance(validation.validated_code, list):
                        for split_code in validation.validated_code:
                            if split_code and hasattr(split_code, 'code') and split_code.code:
                                validated_codes.append({
                                    'theme_number': validation.theme_number,
                                    'code': split_code.code,
                                    'definition': split_code.definition,
                                    'decision': 'create_new'  # SPLIT always creates new codes
                                })
                else:
                    # Handle single code case (APPROVE/REVISE)
                    if validation.validated_code and hasattr(validation.validated_code, 'code') and validation.validated_code.code:
                        validated_codes.append({
                            'theme_number': validation.theme_number,
                            'code': validation.validated_code.code,
                            'definition': validation.validated_code.definition,
                            'decision': effective_decision
                        })
        
        # Validate Step 3 output against provided codes
        step3_validation_warnings = []
        if code_generation_result and 'coding_decisions' in code_generation_result:
            step3_validation_warnings = self._validate_step3_code_references(
                code_generation_result, selected_codes
            )

        # REAL-TIME CODEBOOK UPDATE: Update SharedCodebook immediately so other concurrent clusters can see new codes
        codebook_updates = []
        if validated_codes:
            for final_code in validated_codes:
                if final_code['decision'] == 'create_new':
                    # Add new code to shared codebook immediately
                    added, version = await self.shared_codebook.add_code_if_new(
                        final_code['code'], final_code['definition']
                    )
                    if added:
                        self.stats['new_codes_added'] += 1  # Track stats
                        codebook_updates.append({
                            'action': 'added',
                            'code': final_code['code'],
                            'version': version,
                            'theme_number': final_code['theme_number']
                        })
                        if self.verbose:
                            logger.info(f"Cluster {cluster_id}: REAL-TIME added '{final_code['code']}' (v{version}) - NOW AVAILABLE to concurrent clusters")
                
                elif final_code['decision'] == 'modify_existing':
                    # Find original code from code generation decision
                    original_code = None
                    for decision in code_generation_result.coding_decisions:
                        if decision.theme_number == final_code['theme_number']:
                            original_code = decision.action_details.codes_to_modify
                            break
                    
                    if original_code:
                        # Replace existing code immediately
                        replaced, version = await self.shared_codebook.replace_code(
                            original_code, final_code['code'], final_code['definition']
                        )
                        if replaced:
                            self.stats['codes_modified'] += 1  # Track stats
                            codebook_updates.append({
                                'action': 'modified',
                                'original_code': original_code,
                                'new_code': final_code['code'],
                                'version': version,
                                'theme_number': final_code['theme_number']
                            })
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: REAL-TIME modified '{original_code}' -> '{final_code['code']}' (v{version})")

        return {
            'cluster_id': cluster_id,
            'status': 'completed',
            'cluster_summary': cluster_summary_result,
            'themes': cluster_summary_result.root,
            # Store ACTUAL prompt inputs (exactly what each prompt receives)
            'step1_input': {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_text": cluster_text
            },
            'step2_input': {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
                "code_text": code_text
            },
            'step3_input': {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
                "candidate_codes": selected_codes_text
            },
            'step4_input': {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_summary": _generate_cluster_summary(cluster_summary_result.root),
                "candidate_codes": selected_codes_text,
                "step3_recommendation": formatted_recommendation
            },
            # Store outputs for analysis
            'step1_output': cluster_summary_result.root,
            'step2_output': selected_codes,
            'step3_output': code_generation_result,
            'step4_output': validation_result,
            'step3_validation_warnings': step3_validation_warnings,
            # Legacy fields for compatibility
            'candidate_codes': selected_codes,
            'selected_codes': selected_codes,
            'code_generation_result': code_generation_result,
            'validation_result': validation_result,
            'final_codes': validated_codes,
            'codebook_updates': codebook_updates,
            'processing_time': time.time() - start_time
        }

    def _extract_final_codes(self, code_generation_result, selected_codes: List) -> List[Dict]:
        """Extract final codes when no validation is needed (use_existing decisions only)"""
        final_codes = []
        
        for decision in code_generation_result.coding_decisions:
            if decision.decision == 'use_existing' and decision.action_details.codes_to_use:
                # Find the actual code definitions from selected_codes
                for code_name in decision.action_details.codes_to_use:
                    matching_code = next(
                        (code for code in selected_codes if code.code == code_name),
                        None
                    )
                    if matching_code:
                        final_codes.append({
                            'theme_number': decision.theme_number,
                            'code': matching_code.code,
                            'definition': matching_code.definition,
                            'decision': 'use_existing'
                        })
        
        return final_codes

    def _validate_step3_code_references(self, code_generation_result: Dict, selected_codes: List) -> List[Dict]:
        """
        Validate that Step 3 only references codes that were provided in its input.
        Returns list of validation warnings for codes referenced but not provided.
        """
        warnings = []
        if not code_generation_result or 'coding_decisions' not in code_generation_result:
            return warnings
        
        # Get list of code names that were provided to Step 3
        provided_code_names = {code.code for code in selected_codes} if selected_codes else set()
        
        # Check each coding decision
        for decision in code_generation_result['coding_decisions']:
            decision_type = decision.get('decision', '')
            action_details = decision.get('action_details', {})
            
            if decision_type == 'use_existing' and action_details.get('codes_to_use'):
                # Check if all referenced codes were provided
                for code_name in action_details['codes_to_use']:
                    if code_name not in provided_code_names:
                        warnings.append({
                            'type': 'hallucinated_code_reference',
                            'decision_type': decision_type,
                            'theme_number': decision.get('theme_number', '?'),
                            'referenced_code': code_name,
                            'available_codes': list(provided_code_names),
                            'message': f"Step 3 referenced code '{code_name}' which was not provided in candidate_codes input"
                        })
            
            elif decision_type == 'modify_existing' and action_details.get('codes_to_modify'):
                # Check if base code for modification was provided
                base_code = action_details['codes_to_modify']
                if base_code not in provided_code_names:
                    warnings.append({
                        'type': 'hallucinated_base_code',
                        'decision_type': decision_type,
                        'theme_number': decision.get('theme_number', '?'),
                        'referenced_code': base_code,
                        'available_codes': list(provided_code_names),
                        'message': f"Step 3 wants to modify code '{base_code}' which was not provided in candidate_codes input"
                    })
        
        return warnings

    async def _process_cluster_new_architecture(self, cluster_id: int, cluster_data: Dict) -> Dict[str, Any]:
        """Process a single cluster through the new multi-theme architecture"""
        try:
            start_time = time.time()
            
            # Prepare cluster text
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            # Execute the complete multi-theme pipeline
            pipeline_result = await self._process_multi_theme_pipeline(
                cluster_id, cluster_data, cluster_text
            )
            
            # If pipeline failed, return early
            if pipeline_result['status'] in ['theme_extraction_failed', 'code_generation_failed', 'validation_failed']:
                return pipeline_result
            
            # If no themes found, skip SharedCodebook updates
            if pipeline_result['status'] == 'no_themes_found':
                return pipeline_result
            
            # SharedCodebook is now updated in real-time within _process_multi_theme_pipeline
            # Get codebook updates from the pipeline result
            codebook_updates = pipeline_result.get('codebook_updates', [])
            
            # Track usage stats for clusters that use existing codes
            if pipeline_result['status'] == 'no_validation_needed':
                # All use_existing decisions - count them
                final_codes = pipeline_result.get('final_codes', [])
                use_existing_count = sum(1 for code in final_codes if code.get('decision') == 'use_existing')
                self.stats['no_new_codes_needed'] += use_existing_count
            elif pipeline_result['status'] == 'completed':
                # Count use_existing decisions in mixed scenarios
                final_codes = pipeline_result.get('final_codes', [])
                use_existing_count = sum(1 for code in final_codes if code.get('decision') == 'use_existing')
                self.stats['no_new_codes_needed'] += use_existing_count
            pipeline_result['processing_time'] = time.time() - start_time
            
            # Track cluster completion
            self.stats['clusters_processed'] += 1
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Multi-theme processing error for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'status': 'processing_error',
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }

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
            codebook_analysis = await self._process_step_with_retry(1, codebook_input)
            
            summary_input = {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_text": cluster_text
            }
            summaries = await self._process_step_with_retry(2, summary_input)
            
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
        step1_task = self._process_step_with_retry(1, codebook_input)
        step2_task = self._process_step_with_retry(2, summary_input)
        
        try:
            codebook_analysis, summaries = await asyncio.gather(
                step1_task, step2_task, return_exceptions=True
            )
            
            # Handle individual step failures
            if isinstance(codebook_analysis, Exception):
                logger.error(f"Step 1 failed for cluster {cluster_id}: {str(codebook_analysis)}")
                codebook_analysis = None  # Return None instead of string to avoid type confusion
            
            if isinstance(summaries, Exception):
                logger.error(f"Step 2 failed for cluster {cluster_id}: {str(summaries)}")
                summaries = "Analysis failed due to API error after retries."
            
            self.stats['parallel_steps_executed'] += 1
            return codebook_analysis, summaries
            
        except Exception as e:
            logger.error(f"Parallel steps processing error for cluster {cluster_id}: {e}")
            # Fallback results - return None for Step 1 to avoid type issues
            return (None, "Analysis failed - parallel processing error")

    # async def _process_cluster_optimized(self, cluster_id: int, cluster_data: Dict) -> Dict[str, Any]:
    #     """Process a single cluster with optimized parallel step execution"""
    #     try:
    #         start_time = time.time()
    #         llm_start = time.time()
            
    #         embed_start = time.time()
    #         cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
    #         nearest_codes = await self._find_nearest_codes(cluster_embedding)
    #         self.stats['embedding_time'] += time.time() - embed_start
            
    #         # Get current version for logging
    #         _, version = await self.shared_codebook.get_current_snapshot()
            
    #         # Build targeted code_text using nearest codes  
    #         if nearest_codes:
    #             code_text = "\n".join([
    #                 f"- {code['code']}: {code['definition']}" 
    #                 for code in nearest_codes
    #             ])
    #         else:
    #             code_text = "No existing codes in codebook"
            
    #         # Prepare cluster text
    #         cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
    #         # Execute Steps 1 & 2 in parallel  
    #         codebook_analysis_result, summaries = await self._process_parallel_steps(
    #             cluster_id, cluster_data, code_text, cluster_text
    #         )
            
    #         # Extract candidate codes from Step 1 output
    #         candidate_codes = []
    #         if codebook_analysis_result is None or isinstance(codebook_analysis_result, str):
    #             # Step 1 failed
    #             logger.warning(f"Step 1 failed for cluster {cluster_id}")
    #             candidate_codes = []
    #         elif hasattr(codebook_analysis_result, 'root'):
    #             # RootModel structure
    #             candidate_codes = codebook_analysis_result.root
    #         elif isinstance(codebook_analysis_result, list):
    #             # Handle case where result is already a list
    #             candidate_codes = [CandidateCode(code=c['code'], definition=c['definition']) if isinstance(c, dict) else c for c in codebook_analysis_result]
            
    #         # Format candidate codes for Step 3 and validation
    #         if candidate_codes:
    #             candidate_codes_text = "\n".join([
    #                 f"- {code.code}: {code.definition}" if hasattr(code, 'code') else f"- {code['code']}: {code['definition']}"
    #                 for code in candidate_codes
    #             ])
    #         else:
    #             candidate_codes_text = "No candidate codes available"
            
    #         # Capture prompts if needed (diversity-first logic)
    #         if self._should_capture_prompt('codebook'):
    #             codebook_input = {
    #                 "language": DEFAULT_LANGUAGE,
    #                 "survey_question": self.var_lab,
    #                 "cluster_text": cluster_text,
    #                 "code_text": code_text
    #             }
    #             self.prompt_printer.capture_prompt(
    #                 step_name="codebook_generation",
    #                 utility_name="LangChainBatchProcessor",
    #                 prompt_content=CODEBOOK_ANALYSIS_PROMPT.format(**codebook_input),
    #                 prompt_type="step1_codebook_analysis",
    #                 metadata={
    #                     "model": self.step1_llm.model_name,
    #                     "var_lab": self.var_lab,
    #                     "stage": "1/4 - Codebook Analysis (Parallel)",
    #                     "nearest_codes_count": len(nearest_codes),
    #                     "codebook_version": version,
    #                     "parallel_execution": self.enable_step_parallelization
    #                 }
    #             )
    #             self._record_capture('codebook')
            
    #         if self._should_capture_prompt('summary'):
    #             summary_input = {
    #                 "language": DEFAULT_LANGUAGE,
    #                 "survey_question": self.var_lab,
    #                 "cluster_text": cluster_text
    #             }
    #             self.prompt_printer.capture_prompt(
    #                 step_name="codebook_generation",
    #                 utility_name="LangChainBatchProcessor",
    #                 prompt_content=RESPONSE_SUMMARY_PROMPT.format(**summary_input),
    #                 prompt_type="step2_response_summary",
    #                 metadata={
    #                     "model": self.step2_llm.model_name,
    #                     "var_lab": self.var_lab,
    #                     "stage": "2/4 - Response Summary (Parallel)",
    #                     "cluster_id": cluster_id,
    #                     "cluster_size": len(cluster_data['ideas']),
    #                     "parallel_execution": self.enable_step_parallelization
    #                 }
    #             )
    #             self._record_capture('summary')
            
    #         # Step 3: Match and recommend (uses candidate codes from Step 1)
    #         match_input = {
    #             "language": DEFAULT_LANGUAGE,
    #             "survey_question": self.var_lab,
    #             "candidate_codes": candidate_codes_text,
    #             "clustered_survey_responses": cluster_text,
    #             "cluster_summary": summaries if isinstance(summaries, str) else str(summaries)
    #         }
            
    #         # Capture Step 3 prompt
    #         if self._should_capture_prompt('match'):
    #             self.prompt_printer.capture_prompt(
    #                 step_name="codebook_generation",
    #                 utility_name="LangChainBatchProcessor",
    #                 prompt_content=MATCH_AND_RECOMMEND_PROMPT.format(**match_input),
    #                 prompt_type="step3_match_recommend",
    #                 metadata={
    #                     "model": self.step3_llm.model_name,
    #                     "var_lab": self.var_lab,
    #                     "stage": "3/4 - Match & Recommend",
    #                     "cluster_id": cluster_id,
    #                     "codebook_analysis_present": bool(codebook_analysis_result),
    #                     "summaries_present": bool(summaries)
    #                 }
    #             )
    #             self._record_capture('match')
            
    #         try:
    #             recommendations = await self._process_step_with_retry(3, match_input)
    #         except (APIError, ProcessingError) as e:
    #             logger.error(f"Step 3 failed for cluster {cluster_id} after retries: {str(e)}")
    #             recommendations = []
    #         except Exception as e:
    #             logger.error(f"Step 3 unexpected error for cluster {cluster_id}: {str(e)}")
    #             recommendations = []
            
    #         # Extract new code recommendations from CodeGenerationOutput object
    #         new_codes_needed = False
    #         proposed_codes = []
            
    #         try:
    #             if hasattr(recommendations, 'decision'):
    #                 decision = recommendations.decision.lower()
                    
    #                 # Track decision statistics
    #                 if 'use_existing' in decision:
    #                     self.stats['decisions']['use_existing'] += 1
    #                 elif 'modify_existing' in decision:
    #                     self.stats['decisions']['modify_existing'] += 1
    #                 elif 'create_new' in decision:
    #                     self.stats['decisions']['create_new'] += 1
                    
    #                 if 'create_new' in decision:
    #                     # Access new code details from action_details
    #                     if hasattr(recommendations, 'action_details'):
    #                         new_code = recommendations.action_details.new_code_name
    #                         new_definition = recommendations.action_details.new_code_definition
                            
    #                         # Only add if we have actual code and definition (not null/empty)
    #                         if new_code and new_definition and new_code.strip() and new_definition.strip():
    #                             new_codes_needed = True
    #                             proposed_codes.append({
    #                                 'code': new_code.strip(),
    #                                 'definition': new_definition.strip()
    #                             })
    #                 elif 'modify_existing' in decision:
    #                     # Trigger Step 4 for modification validation
    #                     if hasattr(recommendations, 'action_details'):
    #                         original_code = recommendations.action_details.codes_to_modify
    #                         modified_code = recommendations.action_details.modified_code_name
    #                         modified_definition = recommendations.action_details.modified_code_definition
                            
    #                         # Only proceed if we have modification details
    #                         if original_code and modified_code and modified_definition and modified_code.strip() and modified_definition.strip():
    #                             new_codes_needed = True
    #                             proposed_codes.append({
    #                                 'original_code': original_code.strip(),
    #                                 'modified_code': modified_code.strip(),
    #                                 'modified_definition': modified_definition.strip()
    #                             })
                    
    #                 # Store the full recommendation for potential Step 4 use
    #                 cluster_recommendation = recommendations
                    
    #         except Exception as e:
    #             logger.error(f"Error parsing recommendations for cluster {cluster_id}: {e}")
    #             logger.error(f"Recommendations content: {recommendations}")
            
    #         if not new_codes_needed:
    #             self.stats['no_new_codes_needed'] += 1
    #             return {
    #                 'cluster_id': cluster_id,
    #                 'status': 'no_new_code_needed',
    #                 'step2_summary': summaries,
    #                 'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
    #                 'step4_validated_code': None,
    #                 'candidate_codes': [{'code': code.code, 'definition': code.definition} for code in candidate_codes] if candidate_codes else [],
    #                 'processing_time': time.time() - start_time
    #             }
            
    #         # Step 4: Validate if new codes are proposed
    #         if proposed_codes:
    #             # Format Step 3 recommendation for readable context
    #             formatted_recommendation = self._format_step3_recommendation(recommendations) if hasattr(recommendations, 'cluster_core_theme') else str(recommendations)
                
    #             validation_input = {
    #                 "language": DEFAULT_LANGUAGE,
    #                 "survey_question": self.var_lab,
    #                 "candidate_codes": candidate_codes_text,
    #                 "clustered_ideas": cluster_text,
    #                 "step3_recommendation": formatted_recommendation
    #             }
                
    #             # Capture Step 4 prompt
    #             if self._should_capture_prompt('validation'):
    #                 self.prompt_printer.capture_prompt(
    #                     step_name="codebook_generation",
    #                     utility_name="LangChainBatchProcessor",
    #                     prompt_content=VALIDATION_PROMPT.format(**validation_input),
    #                     prompt_type="step4_validation",
    #                     metadata={
    #                         "model": self.step4_llm.model_name,
    #                         "var_lab": self.var_lab,
    #                         "stage": "4/4 - Validation",
    #                         "cluster_id": cluster_id,
    #                         "proposed_codes_count": len(proposed_codes),
    #                         "proposed_codes": [c.get('code', c.get('modified_code', '')) for c in proposed_codes]
    #                     }
    #                 )
    #                 self._record_capture('validation')
                
    #             try:
    #                 validation_results = await self._process_step_with_retry(4, validation_input)
    #             except (APIError, ProcessingError) as e:
    #                 logger.error(f"Step 4 failed for cluster {cluster_id} after retries: {str(e)}")
    #                 validation_results = None
    #             except Exception as e:
    #                 logger.error(f"Step 4 unexpected error for cluster {cluster_id}: {str(e)}")
    #                 validation_results = None
                
    #             # Process validated codes from ValidationOutput format
    #             validated_code = None
    #             validation_details = None
                
    #             try:
    #                 if validation_results and hasattr(validation_results, 'decision'):
    #                     # ValidationOutput object - store detailed validation info
    #                     validation_details = {
    #                         'decision': validation_results.decision,
    #                         'decision_rationale': validation_results.decision_rationale,
    #                         'reasoning': {
    #                             'semantic_fit_reasoning': validation_results.evaluation.semantic_fit_reasoning,
    #                             'atomicity_reasoning': validation_results.evaluation.atomicity_reasoning,
    #                             'parsimony_reasoning': validation_results.evaluation.parsimony_reasoning,
    #                             'redundancy_reasoning': validation_results.evaluation.redundancy_reasoning,
    #                             'justification_reasoning': validation_results.evaluation.justification_reasoning
    #                         }
    #                     }
                        
    #                     # Extract validated code for ANY decision (APPROVE, REVISE, or REJECT)
    #                     if validation_results.validated_code and validation_results.validated_code.code:
    #                         validated_code = {
    #                             'code': validation_results.validated_code.code,
    #                             'definition': validation_results.validated_code.definition
    #                         }
    #             except Exception as e:
    #                 logger.error(f"Error parsing validation results for cluster {cluster_id}: {e}")
    #                 logger.error(f"Validation results content: {validation_results}")
                
    #             if validated_code:
    #                 # Check if this is a modification or new code
    #                 is_modification = any(pc.get('original_code') for pc in proposed_codes)
                    
    #                 if is_modification:
    #                     # Get original code name from proposed_codes
    #                     original_code_name = next(
    #                         (pc.get('original_code') for pc in proposed_codes if pc.get('original_code')),
    #                         None
    #                     )
                        
    #                     if original_code_name:
    #                         # Get original definition before modification
    #                         original_definition = await self.shared_codebook.get_code_definition(original_code_name)
                            
    #                         # Replace existing code with modified version
    #                         replaced, new_version = await self.shared_codebook.replace_code(
    #                             original_code_name,
    #                             validated_code.get('code', ''),
    #                             validated_code.get('definition', '')
    #                         )
                            
    #                         if replaced:
    #                             self.stats['codes_modified'] += 1
    #                             if self.verbose:
    #                                 new_code_name = validated_code.get('code', '')
    #                                 new_definition = validated_code.get('definition', '')
                                    
    #                                 name_changed = original_code_name != new_code_name
    #                                 definition_changed = original_definition != new_definition if original_definition else True
                                    
    #                                 if name_changed and definition_changed:
    #                                     logger.info(f"Cluster {cluster_id}: Modified code '{original_code_name}' -> '{new_code_name}' + definition updated (v{new_version})")
    #                                 elif name_changed:
    #                                     logger.info(f"Cluster {cluster_id}: Renamed code '{original_code_name}' -> '{new_code_name}' (v{new_version})")
    #                                 elif definition_changed:
    #                                     logger.info(f"Cluster {cluster_id}: Updated definition for '{original_code_name}' (v{new_version})")
                            
    #                         return {
    #                             'cluster_id': cluster_id,
    #                             'status': 'code_modified',
    #                             'original_code': original_code_name,
    #                             'code': validated_code.get('code', ''),
    #                             'definition': validated_code.get('definition', ''),
    #                             'step2_summary': summaries,
    #                             'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
    #                             'step4_validated_code': validated_code,
    #                             'validation_details': validation_details,
    #                             'candidate_codes': [{'code': code.code, 'definition': code.definition} for code in candidate_codes] if candidate_codes else [],
    #                             'processing_time': time.time() - start_time
    #                         }
    #                 else:
    #                     # Standard new code addition (CRITICAL: Real-time update to shared memory)
    #                     added, new_version = await self.shared_codebook.add_code_if_new(
    #                         validated_code.get('code', ''),
    #                         validated_code.get('definition', '')
    #                     )
                        
    #                     if added:
    #                         self.stats['new_codes_added'] += 1
    #                         if self.verbose:
    #                             logger.info(f"Cluster {cluster_id}: Added new code '{validated_code['code']}' (v{new_version}) - NOW AVAILABLE for subsequent clusters")
                        
    #                     return {
    #                         'cluster_id': cluster_id,
    #                         'status': 'new_code_added' if added else 'code_already_exists',
    #                         'code': validated_code.get('code', ''),
    #                         'definition': validated_code.get('definition', ''),
    #                         'step2_summary': summaries,
    #                         'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
    #                         'step4_validated_code': validated_code,
    #                         'validation_details': validation_details,
    #                         'candidate_codes': [{'code': code.code, 'definition': code.definition} for code in candidate_codes] if candidate_codes else [],
    #                         'processing_time': time.time() - start_time
    #                     }
    #             else:
    #                 return {
    #                     'cluster_id': cluster_id,
    #                     'status': 'no_codes_passed_validation',
    #                     'step2_summary': summaries,
    #                     'step3_recommendation': cluster_recommendation if 'cluster_recommendation' in locals() else None,
    #                     'step4_validated_code': None,
    #                     'validation_details': validation_details,
    #                     'candidate_codes': [{'code': code.code, 'definition': code.definition} for code in candidate_codes] if candidate_codes else [],
    #                     'processing_time': time.time() - start_time
    #                 }

    #         self.stats['clusters_processed'] += 1
    #         self.stats['llm_time'] += time.time() - llm_start
            
    #         return {
    #             'cluster_id': cluster_id,
    #             'status': 'processed_no_validation_needed',
    #             'step2_summary': summaries,
    #             'candidate_codes': [{'code': code.code, 'definition': code.definition} for code in candidate_codes] if candidate_codes else [],
    #             'processing_time': time.time() - start_time
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Processing error for cluster {cluster_id}: {e}")
    #         logger.error(f"Error type: {type(e).__name__}", exc_info=True)
    #         return {
    #             'cluster_id': cluster_id,
    #             'status': 'Processing_error',
    #             'error': str(e),
    #             'error_type': type(e).__name__,
    #             'step2_summary': summaries if 'summaries' in locals() else None,
    #             'candidate_codes': [],
    #             'processing_time': time.time() - start_time if 'start_time' in locals() else 0
    #         }

    async def _process_sub_batch_langchain(self, sub_batch: List[Tuple[int, Dict]], sub_batch_idx: int) -> List[Dict[str, Any]]:
        """Process a sub-batch of clusters sequentially to preserve shared memory order"""
        sub_batch_results = []
        
        if self.verbose:
            logger.info(f"  Sub-batch {sub_batch_idx + 1}: Processing {len(sub_batch)} clusters sequentially")
        
        # Process clusters sequentially within sub-batch to preserve shared memory updates
        for cluster_id, cluster_data in sub_batch:
            try:
                result = await self._process_cluster_new_architecture(cluster_id, cluster_data)
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
    
    def _estimate_batch_tokens(self, batch_clusters: List[Dict[str, Any]], nearest_codes: List[Dict[str, str]]) -> tuple[int, int]:
        """
        Estimate input and output tokens for a batch of clusters
        Returns: (input_tokens, output_tokens)
        """
        total_input_tokens = 0
        total_output_tokens = 0
        
        for cluster_data in batch_clusters:
            # Estimate cluster text tokens (all ideas joined)
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            cluster_text_tokens = estimate_tokens(cluster_text)
            
            # Estimate code text tokens (k nearest codes)
            code_text_tokens = estimate_code_list_tokens(nearest_codes[:self.k])
            
            # Step 1: Codebook Analysis
            step1_input = cluster_text_tokens + code_text_tokens + 200  # prompt template
            step1_output = 300  # JSON array of candidate codes
            
            # Step 2: Response Summary  
            step2_input = cluster_text_tokens + 150  # prompt template
            step2_output = 150  # summary text
            
            # Step 3: Match & Recommend (uses candidate codes from step 1)
            candidate_codes_tokens = 250  # estimated from step 1 output
            step3_input = candidate_codes_tokens + cluster_text_tokens + 150 + 250  # candidates + cluster + summary + prompt
            step3_output = 450  # recommendation JSON
            
            # Step 4: Validation
            step4_input = candidate_codes_tokens + cluster_text_tokens + 450 + 200  # candidates + cluster + recommendation + prompt
            step4_output = 300  # validation JSON
            
            total_input_tokens += step1_input + step2_input + step3_input + step4_input
            total_output_tokens += step1_output + step2_output + step3_output + step4_output
        
        return total_input_tokens, total_output_tokens
    
    def _predict_batch_duration(self, batch_clusters: List[Dict[str, Any]], nearest_codes: List[Dict[str, str]]) -> float:
        """
        Predict processing duration for a batch based on token count
        Returns estimated seconds
        """
        input_tokens, output_tokens = self._estimate_batch_tokens(batch_clusters, nearest_codes)
        total_tokens = input_tokens + output_tokens
        
        # Conservative estimates for gpt-4.1-mini with API overhead
        tokens_per_second = 6000  # Conservative estimate
        concurrency_efficiency = 0.6  # Account for API rate limits and parallel processing overhead
        
        base_duration = total_tokens / tokens_per_second
        adjusted_duration = base_duration / concurrency_efficiency
        
        # Add API overhead (network latency, processing time)
        api_overhead = 2.0  # seconds
        
        # Return realistic estimate with reasonable bounds
        estimated_duration = max(3.0, min(15.0, adjusted_duration + api_overhead))
        return estimated_duration
    
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
        
        # Predict processing duration for the first batch to give users realistic expectations
        if self.verbose:
            # Create sample batches to predict first batch duration
            cluster_list = [(cid, cdata) for cid, cdata in clusters.items()]
            first_batch_clusters = [cluster_list[i][1] for i in range(min(self.batch_size, len(cluster_list)))]
            
            # Get sample nearest codes for prediction (use starter codes as approximation)
            sample_nearest_codes = self.starter_codes[:self.k]
            
            # Predict duration for first batch
            estimated_duration = self._predict_batch_duration(first_batch_clusters, sample_nearest_codes)
            estimated_duration_range = 2*estimated_duration
            
            self.verbose_reporter.stat_line("Creating batches for concurrent processing...")
            self.verbose_reporter.stat_line(f"Waiting for first batch to complete (est. {estimated_duration:.0f}s - {estimated_duration_range:.0f}s)...", indent=1)
        else:
            self.verbose_reporter.stat_line("Creating batches for concurrent processing...")
        
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
        
        # Build cluster assignments and detailed results with multi-theme support
        cluster_assignments = {}
        cluster_themes = {}
        cluster_summaries = {}
        code_generation_results = {}
        validation_results = {}
        codebook_updates_all = {}
        
        # New: Store intermediate step data for codebook_reasoning
        step1_summaries = {}
        step2_analysis = {}  # Step 2 results
        step3_recommendations = {}
        validation_details = {}  # Step 4 detailed results
        
        # Storage for actual prompt inputs and outputs
        step1_inputs = {}  # Actual inputs to Prompt 1
        step2_inputs = {}  # Actual inputs to Prompt 2  
        step3_inputs = {}  # Actual inputs to Prompt 3
        step4_inputs = {}  # Actual inputs to Prompt 4
        step_outputs = {}   # All step outputs
        step3_validation_warnings = {}  # Validation warnings for Step 3 output
        
        for result in results:
            cluster_id = result['cluster_id']
            
            # Store themes and summary
            if result.get('themes'):
                cluster_themes[cluster_id] = result['themes']
            
            if result.get('cluster_summary') or result.get('themes'):
                # Store full Step 1 results for codebook_reasoning
                themes = result.get('themes', [])
                
                # Generate cluster summary from themes
                if hasattr(result.get('cluster_summary'), 'root'):
                    # ClusterSummaryOutput object with new format
                    cluster_obj = result['cluster_summary']
                    themes = cluster_obj.root
                    generated_summary = _generate_cluster_summary(themes)
                elif hasattr(result.get('cluster_summary'), 'themes'):
                    # ClusterSummaryOutput object with old format (backwards compatibility)
                    cluster_obj = result['cluster_summary']
                    themes = cluster_obj.themes
                    analyst_note = getattr(cluster_obj, 'analyst_note', None)
                    generated_summary = _generate_cluster_summary(themes, analyst_note)
                else:
                    # Already processed themes
                    generated_summary = _generate_cluster_summary(themes)
                
                # Convert themes to dict format for storage
                themes_for_storage = themes
                if isinstance(themes, list) and themes and hasattr(themes[0], 'theme_id'):
                    # Convert ThemeEntry objects to dict format
                    themes_for_storage = [
                        {
                            'theme_id': theme.theme_id,
                            'theme_name': theme.theme_name,
                            'summary': theme.summary
                        } for theme in themes
                    ]
                
                step1_data = {
                    'cluster_summary': generated_summary,
                    'themes': themes_for_storage
                }
                step1_summaries[cluster_id] = step1_data
                cluster_summaries[cluster_id] = generated_summary
            
            # Store actual prompt inputs (what each prompt received)
            if result.get('step1_input'):
                step1_inputs[cluster_id] = result['step1_input']
            if result.get('step2_input'):
                step2_inputs[cluster_id] = result['step2_input']
            if result.get('step3_input'):
                step3_inputs[cluster_id] = result['step3_input']
            if result.get('step4_input'):
                step4_inputs[cluster_id] = result['step4_input']
            
            # Store step outputs
            if result.get('step2_output'):
                # Convert to list of dicts for codebook_reasoning
                codes_list = []
                for code in result['step2_output']:
                    if hasattr(code, 'code') and hasattr(code, 'definition'):
                        codes_list.append({'code': code.code, 'definition': code.definition})
                    elif isinstance(code, dict):
                        codes_list.append({'code': code.get('code', ''), 'definition': code.get('definition', '')})
                    else:
                        codes_list.append({'code': str(code), 'definition': ''})
                step2_analysis[cluster_id] = codes_list
            
            # Store validation warnings
            if result.get('step3_validation_warnings'):
                step3_validation_warnings[cluster_id] = result['step3_validation_warnings']
            
            # Store detailed results
            if result.get('code_generation_result'):
                code_generation_results[cluster_id] = result['code_generation_result']
                # Store Step 3 recommendations in dict format for codebook_reasoning
                step3_recommendations[cluster_id] = result['code_generation_result'].dict() if hasattr(result['code_generation_result'], 'dict') else result['code_generation_result']
            
            if result.get('validation_result'):
                validation_results[cluster_id] = result['validation_result']
                # Store Step 4 validation details in dict format for codebook_reasoning
                validation_details[cluster_id] = result['validation_result'].dict() if hasattr(result['validation_result'], 'dict') else result['validation_result']
                
            if result.get('codebook_updates'):
                codebook_updates_all[cluster_id] = result['codebook_updates']
            
            # Build cluster assignments (multiple codes per cluster now possible)
            if result.get('final_codes'):
                cluster_assignments[cluster_id] = {
                    'status': result['status'],
                    'codes': result['final_codes'],
                    'theme_count': len(result.get('themes', []))
                }
            else:
                cluster_assignments[cluster_id] = {
                    'status': result['status'],
                    'codes': [],
                    'theme_count': len(result.get('themes', []))
                }
        
        # Get final stats
        final_codes, final_version = await shared_codebook.get_current_snapshot()
        codebook_stats = await shared_codebook.get_stats()
        
        # Combine stats
        processing_time = time.time() - start_time
        
        # Add embedding cache statistics
        embedding_cache_stats = {
            'embedding_cache_hits': embedding_manager.cache_stats['hits'],
            'embedding_cache_misses': embedding_manager.cache_stats['misses'],
            'embedding_api_calls_saved': embedding_manager.cache_stats['api_calls_saved'],
            'embedding_cache_hit_rate': f"{(embedding_manager.cache_stats['hits'] / (embedding_manager.cache_stats['hits'] + embedding_manager.cache_stats['misses']) * 100):.1f}%" if (embedding_manager.cache_stats['hits'] + embedding_manager.cache_stats['misses']) > 0 else "0%"
        }
        
        final_stats = {
            **batch_processor.stats,
            **codebook_stats,
            **embedding_cache_stats,
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
        
        # Display embedding cache statistics
        if embedding_manager.cache_stats['hits'] + embedding_manager.cache_stats['misses'] > 0:
            self.verbose_reporter.stat_line("Embedding cache performance:")
            self.verbose_reporter.stat_line(f"Cache hits: {embedding_manager.cache_stats['hits']}", indent=1)
            self.verbose_reporter.stat_line(f"Cache misses: {embedding_manager.cache_stats['misses']}", indent=1)
            self.verbose_reporter.stat_line(f"Hit rate: {embedding_cache_stats['embedding_cache_hit_rate']}", indent=1)
            self.verbose_reporter.stat_line(f"API calls saved: {embedding_manager.cache_stats['api_calls_saved']}", indent=1)
        
        # # Display sample new codes
        # if batch_processor.stats['new_codes_added'] > 0 and self.verbose:
        #     self._display_sample_new_codes(final_codes, self.starter_codes, step4_validated_codes)
        
        # # Display sample modified codes  
        # if batch_processor.stats['codes_modified'] > 0 and self.verbose:
        #     self._display_sample_modified_codes(step3_recommendations, step4_validated_codes)
        
        return {
            'codebook': final_codes,
            'cluster_assignments': cluster_assignments,
            'cluster_themes': cluster_themes,
            'cluster_summaries': cluster_summaries,
            'cluster_data': clusters,
            'code_generation_results': code_generation_results,
            'validation_results': validation_results,
            'codebook_updates': codebook_updates_all,
            'stats': final_stats,
            'generator_version': 'MULTI_THEME_ARCHITECTURE',
            # Store actual prompt inputs and outputs for transparency
            'step1_inputs': step1_inputs,   # Actual inputs to Prompt 1
            'step2_inputs': step2_inputs,   # Actual inputs to Prompt 2
            'step3_inputs': step3_inputs,   # Actual inputs to Prompt 3
            'step4_inputs': step4_inputs,   # Actual inputs to Prompt 4
            'step3_validation_warnings': step3_validation_warnings,  # Validation warnings
            # Legacy data for backward compatibility
            'step1_summaries': step1_summaries,
            'step2_analysis': step2_analysis,
            'step3_recommendations': step3_recommendations,
            'validation_details': validation_details
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