import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
# from dataclasses import dataclass
# from collections import deque

from openai import OpenAI, RateLimitError
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler
from sklearn.metrics.pairwise import cosine_similarity

# === MODELS ========================================================================================================
import models
from models import ClusterSummaryOutput, CandidateCodeSelectionOutput, CodeRecommendation, ValidationResult, CodeGeneratorReasoningResults, CandidateCode, ClusterThemeItem 

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, DEFAULT_CODEDESIGNER_CONFIG, get_openai_rate_limits
from prompts import CLUSTER_SUMMARY_PROMPT, CANDIDATE_CODE_SELECTION_PROMPT, CODE_GENERATION_PROMPT, VALIDATION_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter
from .qualityFilter import WorkloadAnalyzer, SlidingWindowMonitor

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Initialize client
client = OpenAI()

# ============================================================================
# ASYNC WRAPPERS FOR TRUE CONCURRENCY
# ============================================================================

class RetryableError(Exception):
    """Retryable API errors for tenacity"""
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=0.5, max=8),
    retry=retry_if_exception_type(RetryableError)
)
def _sync_responses_create(model: str, prompt: str, reasoning_effort: str = "minimal", text_verbosity: str = "low"):
    """Sync wrapper for responses.create with retry logic"""
    try:
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text={"verbosity": text_verbosity},
            reasoning={"effort": reasoning_effort},
        )
    except Exception as e:
        # Map rate limits and server errors to retryable errors
        if "429" in str(e) or "5" in str(e)[:1]:  # 5xx errors
            raise RetryableError(str(e)) from e
        raise  # Re-raise non-retryable errors immediately

async def async_responses_create(model: str, prompt: str, reasoning_effort: str = "minimal", text_verbosity: str = "low"):
    """Async wrapper using asyncio.to_thread for true concurrency"""
    return await asyncio.to_thread(_sync_responses_create, model, prompt, reasoning_effort, text_verbosity)

# Global semaphore for concurrency control (start conservative)
_concurrency_semaphore = asyncio.Semaphore(16)

async def async_responses_create_with_semaphore(model: str, prompt: str, reasoning_effort: str = "minimal", text_verbosity: str = "low"):
    """Async wrapper with semaphore-based concurrency control"""
    async with _concurrency_semaphore:
        return await async_responses_create(model, prompt, reasoning_effort, text_verbosity)


# ============================================================================
# CODEDESIGNER API CLIENT WITH RATE LIMITING
# ============================================================================

class CodeDesignerAPIClient:
    
    def __init__(self, throttler: Throttler, monitor: SlidingWindowMonitor, config, encoding, model_config: ModelConfig, verbose_reporter: VerboseReporter, async_client):
        self.throttler = throttler
        self.monitor = monitor
        self.config = config
        self.client = async_client
        self.model_config = model_config
        self.encoding = encoding
        self.verbose_reporter = verbose_reporter
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def make_request(self, task_coro, task_info: str, prompt: str = None):
        """Make API request with intelligent retry and accurate token tracking"""
        
        # Apply precision rate limiting
        async with self.throttler:
            try:
                # Execute the task coroutine
                result = await task_coro
                
                # Record successful request with accurate token count
                if prompt:
                    # Count actual tokens
                    estimated_tokens = len(self.encoding.encode(prompt))
                else:
                    # Fallback to conservative estimate
                    estimated_tokens = 800
                
                await self.monitor.record_request(estimated_tokens)
                
                return result
                
            except Exception as e:
                self.verbose_reporter.error(f"API request failed for {task_info}: {str(e)}")
                if hasattr(self.verbose_reporter, '_parent_stats'):
                    self.verbose_reporter._parent_stats['api_errors'] = self.verbose_reporter._parent_stats.get('api_errors', 0) + 1
                raise


# ============================================================================
# SHARED CODEBOOK 
# ============================================================================

class SharedCodebook:
    """Thread-safe shared codebook with async lock and version tracking"""
    
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
            
            self._codes.append({'code': new_code, 'definition': new_definition})
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add_as_fallback',
                'original_code': original_code,
                'new_code': new_code,
                'timestamp': time.time()
            })
            return True, self._version
    
    async def get_version_info(self) -> Dict[str, Any]:
        """Get codebook version information"""
        async with self._lock:
            return {
                'version': self._version,
                'total_codes': len(self._codes),
                'recent_updates': self._update_log[-5:] if self._update_log else []
            }
    
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
                # Remove oldest cached version
                oldest_version = min(self._embedding_cache.keys())
                del self._embedding_cache[oldest_version]
    
    async def batch_update(self, new_codes: List[Dict[str, str]], expected_base_version: int) -> bool:
        """Phase 3: Batch update multiple codes atomically"""
        async with self._lock:
            # Version conflict check (defensive programming)
            if self._version != expected_base_version:
                # Log version conflict but proceed (dissimilar batches should rarely conflict)
                self._update_log.append({
                    'version': self._version,
                    'action': 'version_conflict_resolved',
                    'expected_version': expected_base_version,
                    'actual_version': self._version,
                    'timestamp': time.time()
                })
            
            # Batch add all new codes
            added_count = 0
            for code_dict in new_codes:
                code = code_dict['code']
                definition = code_dict['definition']
                cluster_id = code_dict.get('cluster_id', 'unknown')
                
                # Check if code already exists
                exists = False
                for existing in self._codes:
                    if existing['code'].lower() == code.lower():
                        exists = True
                        break
                
                if not exists:
                    self._codes.append({'code': code, 'definition': definition})
                    added_count += 1
                    self._update_log.append({
                        'version': self._version + 1,
                        'action': 'batch_add',
                        'code': code,
                        'cluster_id': cluster_id,
                        'timestamp': time.time()
                    })
            
            # Update version once for the entire batch
            if added_count > 0:
                self._version += 1
            
            return added_count > 0


# ============================================================================
# THEME-BASED SIMILARITY ENGINE
# ============================================================================

class SimilarityEngine:
    """Handles theme-based similarity calculation and batch formation"""
    
    def __init__(self, similarity_threshold: float = 0.7, verbose_reporter: VerboseReporter = None):
        self.similarity_threshold = similarity_threshold
        self.verbose_reporter = verbose_reporter or VerboseReporter(False)
    
    async def embed_themes(self, themes: Dict[str, ClusterSummaryOutput]) -> Dict[str, np.ndarray]:
        """Generate embeddings for all theme names using efficient batch processing (like embedder.py)"""
        self.verbose_reporter.step_start("Theme Embedding")
        self.verbose_reporter.stat_line(f"Processing {len(themes)} theme names in batches")
        
        # Prepare data for batch processing
        cluster_ids = list(themes.keys())
        theme_statements = [themes[cid].root[0].theme_statement for cid in cluster_ids]  # Each sub-cluster has only one theme now
        
        # Process in batches (OpenAI supports up to 2048 inputs per call, but use smaller batches for reliability)
        batch_size = 100
        theme_embeddings = {}
        
        try:
            for i in range(0, len(theme_statements), batch_size):
                batch_statements = theme_statements[i:i + batch_size]
                batch_ids = cluster_ids[i:i + batch_size]
                
                self.verbose_reporter.stat_line(f"Processing batch {i//batch_size + 1}/{(len(theme_statements) + batch_size - 1)//batch_size} ({len(batch_statements)} themes)")
                
                # Use efficient batch embedding like embedder.py
                batch_embeddings = await self._embed_openai_batch(batch_statements)
                
                # Map embeddings back to cluster IDs
                for cluster_id, embedding in zip(batch_ids, batch_embeddings):
                    theme_embeddings[cluster_id] = embedding
                    
        except Exception as e:
            self.verbose_reporter.error(f"Batch embedding failed: {e}")
            # Fallback to individual processing if batch fails
            self.verbose_reporter.warning("Falling back to individual embedding processing")
            
            for cluster_id, theme in themes.items():
                try:
                    # Use first theme's statement
                    theme_statement = theme.root[0].theme_statement if theme.root else "Unknown"
                    embedding = await self._get_embedding(theme_statement)
                    theme_embeddings[cluster_id] = embedding
                except Exception as individual_error:
                    self.verbose_reporter.error(f"Failed to embed theme for cluster {cluster_id}: {individual_error}")
                    # Use zero vector as fallback
                    theme_embeddings[cluster_id] = np.zeros(1536)
        
        self.verbose_reporter.stat_line(f"Generated embeddings for {len(theme_embeddings)} themes")
        self.verbose_reporter.step_complete("Theme Embedding")
        return theme_embeddings
    
    async def _embed_openai_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Efficient batch embedding using OpenAI API (adapted from embedder.py) with retry logic"""
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Retry logic similar to embedder.py
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-small"
                )
                return [np.array(item.embedding, dtype=np.float32) for item in response.data]
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                # Exponential backoff
                wait_time = 0.8 * (2 ** attempt)
                self.verbose_reporter.warning(f"Embedding batch failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)[:100]}")
                await asyncio.sleep(wait_time)

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for single text using OpenAI embeddings API (fallback method)"""
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def create_dissimilarity_batches(self, theme_embeddings: Dict[str, np.ndarray], themes: Dict = None) -> List[List[str]]:
        """Create batches using Progressive Dispersion with Conflict Set Tracking"""
        self.verbose_reporter.step_start("Progressive Dispersion Batching")
        
        cluster_ids = list(theme_embeddings.keys())
        embeddings_matrix = np.array([theme_embeddings[cid] for cid in cluster_ids])
        
        # Calculate pairwise similarity matrix
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Report similarity distribution (keep for comparison)
        self._report_similarity_distribution(similarity_matrix)
        
        # Report hierarchical dissimilarity batching strategy  
        progressive_thresholds = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        self.verbose_reporter.stat_line(f"Using hierarchical dissimilarity batching with max 0.85 similarity: {' → '.join(map(str, progressive_thresholds))}")
        
        # Hierarchical Dissimilarity Batching
        # Create batches in order of increasing similarity tolerance
        batches = []
        unassigned_indices = set(range(len(cluster_ids)))
        
        for batch_num, threshold in enumerate(progressive_thresholds):
            if not unassigned_indices:
                break
                
            # Create batch with themes having max similarity < threshold
            batch_indices = self._extract_similarity_constrained_batch(
                similarity_matrix, list(unassigned_indices), threshold
            )
            
            if batch_indices:
                # Convert to cluster_ids and add to batches
                batch_cluster_ids = [cluster_ids[i] for i in batch_indices]
                batches.append(batch_cluster_ids)
                
                # Remove assigned themes
                unassigned_indices -= set(batch_indices)
                
                self.verbose_reporter.stat_line(f"Batch {batch_num + 1} (threshold < {threshold}): {len(batch_indices)} themes")
            else:
                self.verbose_reporter.stat_line(f"Batch {batch_num + 1} (threshold < {threshold}): 0 themes (skipped)")
        
        # Handle any remaining themes that couldn't be batched within 0.85 similarity constraint
        if unassigned_indices:
            # Try to create final batches still respecting max 0.85 similarity
            remaining_themes = list(unassigned_indices)
            while remaining_themes:
                # Extract one more batch at 0.85 threshold
                final_batch_indices = self._extract_similarity_constrained_batch(
                    similarity_matrix, remaining_themes, 0.85
                )
                if final_batch_indices:
                    final_batch_cluster_ids = [cluster_ids[i] for i in final_batch_indices]
                    batches.append(final_batch_cluster_ids)
                    # Remove assigned themes
                    for idx in final_batch_indices:
                        remaining_themes.remove(idx)
                    self.verbose_reporter.stat_line(f"Additional batch (max 0.85): {len(final_batch_indices)} themes")
                else:
                    # Force remaining themes into singletons if they can't be grouped at 0.85
                    for idx in remaining_themes:
                        singleton_cluster_id = [cluster_ids[idx]]
                        batches.append(singleton_cluster_id)
                        self.verbose_reporter.stat_line(f"Singleton batch: {cluster_ids[idx]} (couldn't group at 0.85)")
                    break
        
        # Report final batch statistics
        for batch_idx, batch_cluster_ids in enumerate(batches):
            # Convert cluster_ids back to indices for similarity calculation
            batch_indices = [cluster_ids.index(cid) for cid in batch_cluster_ids]
            
            # Calculate batch statistics
            batch_similarities = []
            for i, idx_i in enumerate(batch_indices):
                for j in range(i + 1, len(batch_indices)):
                    idx_j = batch_indices[j]
                    batch_similarities.append(similarity_matrix[idx_i, idx_j])
            
            avg_sim = np.mean(batch_similarities) if batch_similarities else 0
            max_sim = max(batch_similarities) if batch_similarities else 0
            
            self.verbose_reporter.stat_line(
                f"Final Batch {batch_idx + 1}: {len(batch_cluster_ids)} themes, "
                f"avg_sim={avg_sim:.3f}, max_sim={max_sim:.3f}"
            )
        
        # VERBOSE: Final batch assignments summary
        self.verbose_reporter.stat_line("\n=== FINAL BATCH ASSIGNMENTS ===")
        for batch_idx, batch_cluster_ids in enumerate(batches):
            theme_labels = []
            if themes:
                theme_labels = [f"C{cid}: '{themes[cid].root[0].theme_statement if themes[cid].root else 'unknown'}'" if cid in themes else f"C{cid}: unknown" 
                              for cid in batch_cluster_ids]
            else:
                theme_labels = [f"C{cid}" for cid in batch_cluster_ids]
            
            #self.verbose_reporter.stat_line(f"Batch {batch_idx + 1}: {', '.join(theme_labels)}")
            self.verbose_reporter.stat_line(f"Batch {batch_idx + 1}:\n" + "\n".join(theme_labels))
            
        
        # Final reporting
        self._report_batch_quality(batches, similarity_matrix, cluster_ids)
        self.verbose_reporter.step_complete("Progressive Dispersion Batching")
        return batches
    
    def _report_similarity_distribution(self, similarity_matrix: np.ndarray):
        """Report similarity distribution statistics"""
        n_clusters = similarity_matrix.shape[0]
        similarities = similarity_matrix[np.triu_indices(n_clusters, k=1)]
        
        self.verbose_reporter.stat_line(f"Analyzing similarity distribution for {n_clusters} themes:")
        
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for threshold in thresholds:
            count = np.sum(similarities < threshold)
            percentage = count / len(similarities) * 100 if len(similarities) > 0 else 0
            self.verbose_reporter.stat_line(
                f"  Similarity < {threshold}: {count} pairs ({percentage:.1f}%)"
            )
    
    def _report_batch_quality(self, batches, similarity_matrix, cluster_ids):
        """Report detailed batch quality metrics"""
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("=== Batch Quality Report ===")
        self.verbose_reporter.stat_line(f"Total batches created: {len(batches)}")
        
        # Check if any high-similarity pairs ended up in same batch
        violations = 0
        conflict_threshold = 0.9
        
        for batch_idx, batch in enumerate(batches):
            batch_indices = [cluster_ids.index(cid) for cid in batch]
            for i, idx_i in enumerate(batch_indices):
                for j in range(i + 1, len(batch_indices)):
                    idx_j = batch_indices[j]
                    if similarity_matrix[idx_i, idx_j] > conflict_threshold:
                        violations += 1
                        self.verbose_reporter.warning(
                            f"High similarity ({similarity_matrix[idx_i, idx_j]:.3f}) "
                            f"in batch {batch_idx + 1}"
                        )
        
        if violations == 0:
            self.verbose_reporter.stat_line("✓ No high-similarity conflicts in any batch")
        else:
            self.verbose_reporter.warning(f"⚠ Found {violations} high-similarity pairs in same batch")
        
        # Report batch size distribution
        batch_sizes = [len(batch) for batch in batches]
        self.verbose_reporter.stat_line(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={np.mean(batch_sizes):.1f}")
    
    def create_sub_batches(self, batch: List[int], max_size: int = 10) -> List[List[int]]:
        """Split large batch into sub-batches for rate limiting"""
        if len(batch) <= max_size:
            return [batch]
        
        sub_batches = []
        for i in range(0, len(batch), max_size):
            sub_batch = batch[i:i + max_size]
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def _extract_similarity_constrained_batch(self, similarity_matrix: np.ndarray, available_indices: List[int], max_similarity_threshold: float) -> List[int]:
        """Extract single largest batch from available indices where max similarity < threshold"""
        if not available_indices:
            return []
        
        # Greedy approach: start with first theme, add compatible themes
        batch_indices = [available_indices[0]]
        remaining = set(available_indices[1:])
        
        # Keep adding themes that don't violate similarity constraint
        added_theme = True
        while added_theme and remaining:
            added_theme = False
            
            # Try each remaining theme
            for candidate_idx in list(remaining):
                # Check if candidate is compatible with all themes in current batch
                can_add = True
                for batch_member_idx in batch_indices:
                    if similarity_matrix[candidate_idx, batch_member_idx] >= max_similarity_threshold:
                        can_add = False
                        break
                
                if can_add:
                    batch_indices.append(candidate_idx)
                    remaining.remove(candidate_idx)
                    added_theme = True
                    break  # Start over to find next compatible theme
        
        return batch_indices


# ============================================================================
# MAIN CODEDESIGNER CLASS
# ============================================================================

class InductiveCodeGenerator:
    """CodeDesigner: Theme-based similarity processing with 4-stage pipeline"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel],
        starter_codes: List[Dict[str, str]], 
        var_lab: str,
        verbose: bool = False,
        verbose_detailed: bool = False,
        prompt_printer = None,
        config = None,
        **kwargs  # For backward compatibility
    ):
        self.cluster_results = cluster_results
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_detailed = verbose_detailed
        self.prompt_printer = prompt_printer
        self.config = config or DEFAULT_CODEDESIGNER_CONFIG
        
        # Initialize components
        self.model_config = ModelConfig()
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        
        # Initialize async client and rate limiting
        self.async_client = client  # Use global client
        self.embedding_client = OpenAI()
        
        # Initialize tokenizer with proper model mapping
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            # Map newer model names to their tiktoken-compatible equivalents
            tiktoken_model_mapping = {
                'gpt-4.1-mini': 'gpt-4o-mini',
                'gpt-4.1': 'gpt-4o', 
                'gpt-4.1-turbo': 'gpt-4o'
            }
            
            tiktoken_model = tiktoken_model_mapping.get(self.config.model)
            if tiktoken_model:
                try:
                    self.encoding = tiktoken.encoding_for_model(tiktoken_model)
                    # This is expected behavior, not a fallback
                except KeyError:
                    # Only this is truly a fallback
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                    self.verbose_reporter.warning(f"Fallback to cl100k_base encoding for {self.config.model}")
            else:
                self.encoding = tiktoken.get_encoding("cl100k_base")
                self.verbose_reporter.warning(f"Fallback to cl100k_base encoding for {self.config.model}")
        
        # Initialize workload analyzer and rate limits
        rate_limits = get_openai_rate_limits(self.config.model)
        self.workload_analyzer = WorkloadAnalyzer(self.config.model, self.encoding)
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute
        
        # Initialize global rate limiting components
        self.global_monitor = SlidingWindowMonitor(self.rpm_limit, self.tpm_limit)
        
        # Initialize components
        self.similarity_engine = SimilarityEngine(
            similarity_threshold=self.config.similarity_threshold,
            verbose_reporter=self.verbose_reporter
        )
        self.shared_codebook = SharedCodebook(starter_codes)
        
        # Results storage
        self._results = []
        self._processing_stats = {}
        
        # Initialize prompt tracking for CodeGeneratorReasoningResults
        self.step1_inputs = {}   # Theme extraction inputs
        self.step2_inputs = {}   # Not used in current architecture  
        self.step3_inputs = {}   # Code generation inputs
        self.step4_inputs = {}   # Validation inputs
        self.step1_summaries = {}  # Theme extraction results
        self.step2_analysis = {}   # Not used in current architecture
        self.step3_recommendations = {}  # Code generation results
        self.step4_validations = {}  # Validation results
        self.step4_validated_codes = {}  # Final validated codes
        self.cluster_assignments = {}  # Cluster-to-code mappings
    
    def _capture_prompt_params(self, cluster_id: Union[int, str], step: str, **kwargs):
        """Capture exact parameters used in prompt.format() for debugging/testing"""
        # Convert cluster_id to string for consistent dict key format
        key = str(cluster_id)
        if step == "step1":
            self.step1_inputs[key] = kwargs
        elif step == "step2":
            self.step2_inputs[key] = kwargs
        elif step == "step3":
            self.step3_inputs[key] = kwargs
        elif step == "step4":
            self.step4_inputs[key] = kwargs
    
    def _get_theme_statement(self, theme_data) -> str:
        """Safely get theme statement from theme data"""
        if hasattr(theme_data, 'root') and theme_data.root:
            return theme_data.root[0].theme_statement
        return "Unknown theme"
    
    def _get_theme_description(self, theme_data) -> str:
        """Safely get theme description from theme data"""
        # For now, use theme statement as description since they're the same
        return self._get_theme_statement(theme_data)
    
    def extract_cluster_data(self) -> Dict[int, Dict[str, Any]]:
        """Extract cluster data from ClusterModel objects"""
        clusters = {}
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        if cluster_id not in clusters:
                            clusters[cluster_id] = {'ideas': [], 'embeddings': []}
                        
                        clusters[cluster_id]['ideas'].append(idea.idea)
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
        
        # Filter out empty clusters
        return {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
    
    async def extract_themes(self, clusters: Dict[int, Dict[str, Any]]) -> Dict[int, ClusterSummaryOutput]:
        """Stage 1: Extract themes from all clusters with rate limiting"""
        self.verbose_reporter.step_start("Theme Extraction")
        self.verbose_reporter.stat_line(f"Processing {len(clusters)} clusters")
        
        # Create tasks for all clusters
        tasks = []
        for cluster_id, cluster_data in clusters.items():
            task = self._extract_single_theme(cluster_id, cluster_data['ideas'])
            tasks.append(task)
        
        # Process with rate limiting using measured tokens
        results = await self._process_with_optimal_strategy(tasks, "theme_extraction", clusters)
        
        # Convert to dictionary
        theme_results = {}
        for result in results:
            if result is not None:
                theme_results[result.cluster_id] = result
        
        self.verbose_reporter.stat_line(f"Extracted {len(theme_results)} themes")
        self.verbose_reporter.step_complete("Theme Extraction")
        return theme_results
    
    def expand_multi_theme_clusters(self, themes: Dict[int, ClusterSummaryOutput], 
                                   clusters: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, ClusterSummaryOutput], Dict[str, Dict[str, Any]]]:
        """Expand multi-theme clusters into sub-clusters for independent processing"""
        self.verbose_reporter.step_start("Multi-Theme Cluster Expansion")
        
        expanded_themes = {}
        expanded_clusters = {}
        
        # Also expand step1_summaries to match the new sub-cluster structure
        expanded_step1_summaries = {}
        
        for cluster_id, theme_data in themes.items():
            if len(theme_data.root) > 1:
                # Multi-theme cluster: create sub-clusters
                self.verbose_reporter.stat_line(f"Expanding cluster {cluster_id} into {len(theme_data.root)} sub-clusters")
                
                for i, theme_item in enumerate(theme_data.root, 1):
                    sub_cluster_id = f"{cluster_id}-{i}"
                    
                    # Create single-theme ClusterSummaryOutput for sub-cluster
                    single_theme_data = ClusterSummaryOutput([theme_item])
                    expanded_themes[sub_cluster_id] = single_theme_data
                    
                    # Duplicate cluster data for sub-cluster
                    if cluster_id in clusters:
                        expanded_clusters[sub_cluster_id] = clusters[cluster_id].copy()
                    
                    # Create step1_summary for this sub-cluster with only its single theme
                    if cluster_id in self.step1_summaries:
                        expanded_step1_summaries[sub_cluster_id] = {
                            'cluster_summary': theme_item.theme_statement,
                            'themes': [theme_item.theme_statement]  # Single theme only
                        }
                    
            else:
                # Single-theme cluster: keep as-is but convert to string ID for consistency
                string_cluster_id = str(cluster_id)
                expanded_themes[string_cluster_id] = theme_data
                
                if cluster_id in clusters:
                    expanded_clusters[string_cluster_id] = clusters[cluster_id].copy()
                
                # Also convert step1_summaries to string ID
                if cluster_id in self.step1_summaries:
                    expanded_step1_summaries[string_cluster_id] = self.step1_summaries[cluster_id]
        
        # Replace the original step1_summaries with the expanded version
        self.step1_summaries = expanded_step1_summaries
        
        self.verbose_reporter.stat_line(f"Expanded {len(themes)} clusters into {len(expanded_themes)} processing units")
        self.verbose_reporter.step_complete("Multi-Theme Cluster Expansion")
        
        return expanded_themes, expanded_clusters
    
    async def _extract_single_theme(self, cluster_id: Union[int, str], ideas: List[str]):
        """Extract theme for single cluster using instructor"""
        ideas_text = "\n".join([f"- {idea}" for idea in ideas])
        
        # Prepare exact parameters for prompt
        params = {
            'survey_question': self.var_lab,
            'language': DEFAULT_LANGUAGE,
            'cluster_text': ideas_text
        }
        
        prompt = CLUSTER_SUMMARY_PROMPT.format(**params)
        
        # Capture exact parameters used in prompt construction
        self._capture_prompt_params(cluster_id, "step1", **params)
        
        # Capture prompt with prompt_printer if available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="theme_extraction",
                utility_name="codeGenerator",
                prompt_content=prompt,
                prompt_type="cluster_summary",
                metadata={
                    "cluster_id": cluster_id,
                    "model": self.config.model,
                    "ideas_count": len(ideas)
                }
            )
        
        try:
            # Use async wrapper for true concurrency with GPT-5 reasoning parameters
            resp = await async_responses_create_with_semaphore(
                model="gpt-5-nano",
                prompt=prompt,
                reasoning_effort="minimal",
                text_verbosity="low"
            )
            response = ClusterSummaryOutput.model_validate_json(resp.output_text).root
            
            # Handle ClusterSummaryOutput response from CLUSTER_SUMMARY_PROMPT
            # Debug: Check response type
            if hasattr(response, '__await__'):
                self.verbose_reporter.error(f"Response is still a coroutine for cluster {cluster_id}: {type(response)}")
                return None
            
            if response and len(response) > 0:
                # Take the first theme item (most common case)
                theme_item = response[0]
                
                # Create a proper result object with required attributes
                class ThemeExtractionResult:
                    def __init__(self, themes_list: List[ClusterThemeItem], cluster_id: int):
                        self.themes = themes_list
                        self.cluster_id = cluster_id
                    
                    @property
                    def root(self):
                        """Backward compatibility for code that expects .root"""
                        return self.themes
                
                result = ThemeExtractionResult(response, cluster_id)
                
                # Capture theme extraction results for transparency
                self.step1_summaries[cluster_id] = {
                    'cluster_summary': f"{theme_item.theme_statement}",
                    'themes': [item.theme_statement for item in response],  # All themes from array
                }
                
                return result
            else:
                return None
            
        except Exception as e:
            self.verbose_reporter.error(f"Theme extraction failed for cluster {cluster_id}: {e}")
            return None
    
    async def _process_with_optimal_strategy(self, tasks: List, stage_name: str, clusters: Dict[int, Dict[str, Any]] = None):
        """Process tasks with rate limiting and optimal strategy"""
        if not tasks:
            return []
        
        # Calculate optimal strategy using measured tokens
        if clusters is not None:
            avg_tokens = self._measure_stage_tokens(stage_name, clusters)
        else:
            avg_tokens = self._measure_stage_tokens(stage_name, {})
            
        strategy = self.workload_analyzer.calculate_optimal_strategy(
            total_batches=len(tasks),
            avg_tokens_per_batch=avg_tokens,
            sub_batches_per_batch=1
        )
        
        self.verbose_reporter.stat_line(f"Optimal strategy for {stage_name}: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
        
        # Initialize throttling and monitoring
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second, period=1.0)
        api_client = CodeDesignerAPIClient(
            throttler, self.global_monitor, self.config, self.encoding, 
            self.model_config, self.verbose_reporter, self.async_client
        )
        
        # Process all tasks with sophisticated rate limiting
        results = []
        completed = 0
        
        # Create all tasks with prompts for accurate token tracking
        api_tasks = []
        for i, task in enumerate(tasks):
            # For theme extraction, we can reconstruct the prompt to track tokens accurately
            if stage_name == "theme_extraction" and clusters:
                cluster_items = list(clusters.items())
                if i < len(cluster_items):
                    cluster_id, cluster_data = cluster_items[i]
                    ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
                    prompt = CLUSTER_SUMMARY_PROMPT.format(
                        survey_question=self.var_lab,
                        language=DEFAULT_LANGUAGE,
                        cluster_text=ideas_text
                    )
                    api_task = asyncio.create_task(api_client.make_request(task, f"{stage_name}_task_{i}", prompt))
                else:
                    api_task = asyncio.create_task(api_client.make_request(task, f"{stage_name}_task_{i}"))
            else:
                api_task = asyncio.create_task(api_client.make_request(task, f"{stage_name}_task_{i}"))
            api_tasks.append(api_task)
        
        # Process results as they complete
        for coro in asyncio.as_completed(api_tasks):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
                completed += 1
                
                # Progress reporting
                if completed % 5 == 0 or completed == len(api_tasks):
                    self.verbose_reporter.progress_line(completed, len(api_tasks), f"{stage_name} tasks")
            except Exception as e:
                self.verbose_reporter.error(f"Task failed in {stage_name}: {e}")
                completed += 1
        
        # Final statistics
        final_stats = await self.global_monitor.get_current_utilization()
        self.verbose_reporter.stat_line(f"Stage {stage_name} completed: {final_stats['total_requests']} requests in {final_stats['elapsed_time']:.1f}s")
        
        return results
    
    def _measure_stage_tokens(self, stage_name: str, clusters: Dict[int, Dict[str, Any]]) -> float:
        """Measure actual token usage for pipeline stages using sample data (like qualityFilter)"""
        if stage_name == "theme_extraction":
            return self._measure_theme_extraction_tokens(clusters)
        elif stage_name == "code_generation":
            return 1200  # Will be replaced by sub-batch specific measurement
        elif stage_name == "validation":
            return 900   # Will be replaced by sub-batch specific measurement
        else:
            return 1000

    def _measure_theme_extraction_tokens(self, clusters: Dict[int, Dict[str, Any]]) -> float:
        """Measure theme extraction token usage from sample data"""
        # Build actual theme extraction prompts from first 10 clusters for measurement
        sample_prompts = []
        cluster_items = list(clusters.items())[:min(10, len(clusters))]
        
        for cluster_id, cluster_data in cluster_items:
            ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            prompt = CLUSTER_SUMMARY_PROMPT.format(
                survey_question=self.var_lab,
                language=DEFAULT_LANGUAGE,
                cluster_text=ideas_text
            )
            sample_prompts.append(prompt)
        
        # Measure real token usage using workload analyzer pattern
        if sample_prompts:
            token_counts = []
            for prompt in sample_prompts:
                # Count input tokens
                prompt_tokens = len(self.encoding.encode(prompt))
                # Estimate completion tokens (theme name + description)
                completion_tokens = 50  # Conservative estimate for short output
                total_tokens = prompt_tokens + completion_tokens
                token_counts.append(total_tokens)
            
            import statistics
            measured_tokens = statistics.mean(token_counts)
            self.verbose_reporter.stat_line(f"Measured theme_extraction token usage: {measured_tokens:.0f} tokens/request (from {len(sample_prompts)} samples)")
            return measured_tokens
        else:
            return 800  # Fallback only if no samples available

    async def _measure_code_generation_tokens(self, clusters: Dict[int, Dict[str, Any]], themes: Dict[int, ClusterSummaryOutput]) -> Dict[str, float]:
        """Measure real token usage for all 3 code generation steps (like qualityFilter approach)"""
        self.verbose_reporter.step_start("Code Generation Token Measurement")
        
        # Sample first 5-10 clusters for measurement (balance accuracy vs speed)
        sample_size = min(8, len(clusters))
        sample_items = list(clusters.items())[:sample_size]
        
        token_measurements = {
            'candidate_selection': [],
            'code_generation': [], 
            'validation': []
        }
        
        # Get current codebook state for realistic measurements
        current_codes, version = await self.shared_codebook.get_current_snapshot()
        
        for cluster_id, cluster_data in sample_items:
            if cluster_id not in themes:
                continue
                
            theme_data = themes[cluster_id]
            #ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            # Step 4a: Candidate Selection prompt
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in current_codes[:20]])  # Limit like the real implementation
            
            candidate_prompt = CANDIDATE_CODE_SELECTION_PROMPT.format(
                survey_question=self.var_lab,
                language=DEFAULT_LANGUAGE,
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}",
                code_text=codes_text
            )
            candidate_tokens = len(self.encoding.encode(candidate_prompt)) + 200  # + completion estimate
            token_measurements['candidate_selection'].append(candidate_tokens)
            
            # Step 4b: Code Generation prompt (simulate with 3 candidate codes)
            candidate_codes_text = "Code: EXAMPLE_CODE_1\nDefinition: Example definition 1\n\nCode: EXAMPLE_CODE_2\nDefinition: Example definition 2\n"
            
            code_gen_prompt = CODE_GENERATION_PROMPT.format(
                language=DEFAULT_LANGUAGE,
                survey_question=self.var_lab,
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}",
                candidate_codes=candidate_codes_text
            )
            code_gen_tokens = len(self.encoding.encode(code_gen_prompt)) + 150  # + completion estimate
            token_measurements['code_generation'].append(code_gen_tokens)
            
            # Step 4c: Validation prompt
            validation_codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                              for code in current_codes[:15]])  # Limited like real implementation
            
            validation_prompt = VALIDATION_PROMPT.format(
                language=DEFAULT_LANGUAGE,
                survey_question=self.var_lab,
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}",
                candidate_codes=validation_codes_text,
                step3_recommendation='{"coding_decisions": [{"decision": "create_new", "justification": "Example reasoning"}]}'
            )
            validation_tokens = len(self.encoding.encode(validation_prompt)) + 100  # + completion estimate
            token_measurements['validation'].append(validation_tokens)
        
        # Calculate averages
        import statistics
        measured_averages = {}
        for step, token_list in token_measurements.items():
            if token_list:
                avg_tokens = statistics.mean(token_list)
                measured_averages[step] = avg_tokens
                self.verbose_reporter.stat_line(f"Measured {step} token usage: {avg_tokens:.0f} tokens/request (from {len(token_list)} samples)")
            else:
                # Fallback estimates
                fallback_tokens = {'candidate_selection': 1200, 'code_generation': 1000, 'validation': 900}
                measured_averages[step] = fallback_tokens[step]
        
        self.verbose_reporter.step_complete("Code Generation Token Measurement")
        return measured_averages
    
    async def process_batches_sequentially(self, dissimilarity_batches: List[List[str]], 
                                         clusters: Dict, themes: Dict, 
                                         theme_embeddings: Dict) -> List[Dict[str, Any]]:
        """Process Level 1 batches sequentially, Level 2 batches concurrently with staggering"""
        self.verbose_reporter.step_start("Two-Level Batch Processing")
        self.verbose_reporter.stat_line(f"Processing {len(dissimilarity_batches)} Level 1 batches sequentially")
        
        # Store theme embeddings for use in candidate selection
        self._theme_embeddings_cache = theme_embeddings
        
        # Measure token usage for code generation steps
        self.code_gen_token_measurements = await self._measure_code_generation_tokens(clusters, themes)
        
        # Calculate global rate limiting strategy
        composite_tokens = (
            self.code_gen_token_measurements.get('candidate_selection', 1200) +
            self.code_gen_token_measurements.get('code_generation', 1000) +
            self.code_gen_token_measurements.get('validation', 900)
        )
        
        total_clusters = sum(len(batch) for batch in dissimilarity_batches)
        self.verbose_reporter.stat_line(f"Total clusters to process: {total_clusters}")
        self.verbose_reporter.stat_line(f"Composite tokens per cluster: {composite_tokens}")
        
        all_results = []
        total_batches = len(dissimilarity_batches)
        
        for batch_idx, dissimilarity_batch in enumerate(dissimilarity_batches):
            self.verbose_reporter.step_start(f"Level 1 Batch {batch_idx + 1}/{total_batches}")
            self.verbose_reporter.stat_line(f"Processing {len(dissimilarity_batch)} clusters")
            
            # Create Level 2 sub-batches
            if len(dissimilarity_batch) > self.config.max_sub_batch_size:
                sub_batches = self.similarity_engine.create_sub_batches(dissimilarity_batch, self.config.max_sub_batch_size)
            else:
                sub_batches = [dissimilarity_batch]  # Single sub-batch
             
            # Process Level 2 sub-batches concurrently with staggering
            sub_batch_tasks = []
            for i, sub_batch in enumerate(sub_batches):
                stagger_delay = i * 0.5  # 500ms between sub-batch starts for smooth distribution
                task = self._process_sub_batch_with_stagger(
                    sub_batch, clusters, themes, stagger_delay
                )
                sub_batch_tasks.append(task)
            
            # Wait for ALL sub-batches to complete before next Level 1 batch
            batch_results = await asyncio.gather(*sub_batch_tasks)
            
            # Flatten and collect results
            for sub_batch_result in batch_results:
                all_results.extend(sub_batch_result)
            
            # Level 1 batch complete - codebook is updated
            version_info = await self.shared_codebook.get_version_info()
            self.verbose_reporter.stat_line(f"Codebook version: {version_info['version']}, total codes: {version_info['total_codes']}")
            self.verbose_reporter.step_complete(f"Level 1 Batch {batch_idx + 1} completed")
            # Next Level 1 batch starts immediately (if API limits allow)
        
        self.verbose_reporter.step_complete("Two-Level Batch Processing")
        return all_results
    
    async def _process_sub_batch_with_stagger(self, sub_batch: List[str], clusters: Dict, themes: Dict, 
                                            stagger_delay: float) -> List[Dict[str, Any]]:
        """Process sub-batch with initial stagger delay and rate limiting"""
        
        # Apply stagger delay for smooth distribution
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)
      
        # Process all clusters in sub-batch with unlimited concurrency
        # Get current codebook snapshot for consistent processing
        codebook_snapshot, base_version = await self.shared_codebook.get_current_snapshot()
        
        # PRE-COMPUTE: Ensure codebook embeddings exist for this version before cluster processing
        # This prevents multiple clusters from generating the same embeddings simultaneously
        cached_embeddings = await self.shared_codebook.get_embeddings_for_version(base_version)
        if cached_embeddings is None and codebook_snapshot:
            code_texts = [f"{code['code']}: {code['definition']}" for code in codebook_snapshot]
            try:
                code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                await self.shared_codebook.cache_embeddings(base_version, code_embeddings)
                self.verbose_reporter.stat_line(f"Cached embeddings for version {base_version}")
            except Exception as e:
                self.verbose_reporter.error(f"Failed to pre-compute embeddings for version {base_version}: {e}")
        
        cluster_tasks = []
        for cluster_id in sub_batch:
            task = self._process_single_cluster(
                cluster_id, clusters, themes, codebook_snapshot, base_version
            )
            cluster_tasks.append(task)
        
        # Process with unlimited concurrency  
        results = []
        for coro in asyncio.as_completed(cluster_tasks):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
            except Exception as e:
                self.verbose_reporter.error(f"Cluster processing failed: {e}")
        
        # Update SharedCodebook with any new codes from this sub-batch
        if results:
            await self._merge_codebook_updates(results, base_version)
            
            # POST-PROCESS: Generate embeddings for any new codes added during batch processing
            updated_codes, new_version = await self.shared_codebook.get_current_snapshot()
            if new_version > base_version:
                # Codebook was updated during processing, ensure embeddings exist for new version
                cached_embeddings = await self.shared_codebook.get_embeddings_for_version(new_version)
                if cached_embeddings is None:
                    # Generate embeddings for all codes in the updated codebook
                    if self.verbose_detailed: 
                        self.verbose_reporter.stat_line(f"Post-processing: Generating embeddings for updated codebook (version {new_version})")
                    code_texts = [f"{code['code']}: {code['definition']}" for code in updated_codes]
                    try:
                        code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                        await self.shared_codebook.cache_embeddings(new_version, code_embeddings)
                        self.verbose_reporter.stat_line(f"Cached embeddings for updated version {new_version}")
                    except Exception as e:
                        self.verbose_reporter.error(f"Failed to generate embeddings for updated codebook (version {new_version}): {e}")
        
        return results
    
    # async def _process_single_cluster_pipeline(self, cluster_id: int, 
    #                                          clusters: Dict, themes: Dict) -> Optional[Dict[str, Any]]:
    #     """3-step pipeline for single cluster"""
        
    #     if cluster_id not in themes or cluster_id not in clusters:
    #         return None
        
    #     cluster_data = clusters[cluster_id]
    #     theme_data = themes[cluster_id]
        
    #     try:
    #         # Step 1: Get current codebook for candidate selection (ensures latest codes are visible)
    #         current_codes, _ = await self.shared_codebook.get_current_snapshot()
    #         nearest_codes = await self._find_nearest_codes_by_theme(
    #             cluster_id, theme_data, current_codes, k=5
    #         )
            
    #         # Step 1: Candidate selection - pure unlimited call
    #         candidate_selection = await self._select_candidate_codes(
    #             cluster_id, cluster_data, theme_data, nearest_codes
    #         )
            
    #         # Step 4b: Code Generation - unlimited
    #         code_generation = await self._generate_code(
    #             cluster_id, cluster_data, theme_data, candidate_selection
    #         )
            
    #         # Step 4c: Validation
    #         validation = await self._validate_code(
    #             cluster_id, cluster_data, theme_data, code_generation, candidate_selection
    #         )
            
    #         # Extract final code/definition from complex validation structure
    #         final_code = None
    #         final_definition = None
    #         if validation and hasattr(validation, 'code_validations') and validation.code_validations:
    #             # Get the first validated code (for now - could aggregate multiple)
    #             first_validation = validation.code_validations[0]
    #             if first_validation.validated_code:
    #                 final_code = first_validation.validated_code.code
    #                 final_definition = first_validation.validated_code.definition
            
    #         return {
    #             'cluster_id': cluster_id,
    #             'theme_name': self._get_theme_statement(theme_data),
    #             'theme_description': self._get_theme_description(theme_data),
    #             'ideas_count': len(cluster_data['ideas']),
    #             'candidate_selection': candidate_selection,
    #             'code_generation': code_generation,
    #             'validation': validation,
    #             'final_code': final_code,
    #             'final_definition': final_definition
    #         }
            
    #     except Exception as e:
    #         self.verbose_reporter.error(f"Pipeline failed for cluster {cluster_id}: {e}")
    #         return None
    
    async def _find_nearest_codes_by_theme(self, cluster_id: Union[int, str], theme_data, 
                                          current_codes: List[Dict[str, str]], k: int = 5) -> List[Dict[str, str]]:
        """Find k nearest codes to themes using cosine similarity - handles multiple themes per cluster"""
        if not current_codes:
            return []
        
        # Handle multiple themes per cluster (aggregate approach)
        all_nearest_codes = []
        
        # Check if theme_data has multiple themes
        if hasattr(theme_data, 'root') and isinstance(theme_data.root, list) and len(theme_data.root) > 1:
            # Multiple themes: get k codes for each theme and aggregate
            for theme_item in theme_data.root:
                theme_embedding = await self._get_theme_embedding_for_item(cluster_id, theme_item)
                if theme_embedding is not None:
                    nearest_codes = await self._get_nearest_codes_by_embedding(theme_embedding, current_codes, k)
                    all_nearest_codes.extend(nearest_codes)
        else:
            # Single theme: use cached embedding (backward compatibility)
            if not hasattr(self, '_theme_embeddings_cache'):
                self.verbose_reporter.error(f"No theme embeddings found for cluster {cluster_id}")
                return []
                
            theme_embedding = self._theme_embeddings_cache.get(cluster_id)
            if theme_embedding is None:
                self.verbose_reporter.error(f"No theme embedding found for cluster {cluster_id}")
                return []
            
            all_nearest_codes = await self._get_nearest_codes_by_embedding(theme_embedding, current_codes, k)
        
        # Deduplicate by code name while preserving order
        seen_codes = set()
        deduplicated_codes = []
        for code in all_nearest_codes:
            code_key = code['code']
            if code_key not in seen_codes:
                seen_codes.add(code_key)
                deduplicated_codes.append(code)
        
        return deduplicated_codes
    
    async def _get_theme_embedding_for_item(self, cluster_id: Union[int, str], theme_item) -> Optional[np.ndarray]:
        """Get embedding for a specific theme item"""
        try:
            # Generate embedding for this specific theme
            theme_text = theme_item.theme_statement
            embedding = await self.similarity_engine._get_embedding(theme_text)
            return embedding
        except Exception as e:
            self.verbose_reporter.error(f"Failed to embed theme '{theme_item.theme_statement}' for cluster {cluster_id}: {e}")
            return None
    
    async def _get_nearest_codes_by_embedding(self, theme_embedding: np.ndarray, 
                                            current_codes: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
        """Get k nearest codes to a theme embedding using cosine similarity"""
        # Get codebook version
        _, version = await self.shared_codebook.get_current_snapshot()
        
        # Check for cached code embeddings
        code_embeddings = await self.shared_codebook.get_embeddings_for_version(version)
        
        if code_embeddings is None:
            # Generate embeddings for all codes
            #self.verbose_reporter.stat_line(f"Generating embeddings for {len(current_codes)} codes (version {version})")
            
            # Format codes for embedding (same format as old codeGenerator)
            code_texts = [f"{code['code']}: {code['definition']}" for code in current_codes]
            
            # Batch embed all codes
            try:
                code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                # Cache the embeddings
                await self.shared_codebook.cache_embeddings(version, code_embeddings)
            except Exception as e:
                self.verbose_reporter.error(f"Failed to generate code embeddings: {e}")
                return []
        
        # Calculate cosine similarities
        code_embeddings_array = np.array(code_embeddings)
        theme_embedding_array = theme_embedding.reshape(1, -1)
        similarities = cosine_similarity(theme_embedding_array, code_embeddings_array)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Filter by similarity threshold and return the nearest codes
        nearest_codes = []
        min_similarity_threshold = 0.3  # Only consider codes with at least 30% similarity
        for idx in top_k_indices:
            if idx < len(current_codes) and similarities[idx] >= min_similarity_threshold:
                nearest_codes.append(current_codes[idx])
        
        # Detailed verbodse: similarity score nearest codes to theme embedding
        if self.verbose_detailed and nearest_codes:
            similarity_values = [round(float(similarities[idx]), 3) for idx in top_k_indices]
            codes_with_scores = [f"{code['code']} ({score})" for code, score in zip(nearest_codes, similarity_values)]
            self.verbose_reporter.stat_line(f"Found {len(nearest_codes)} nearest codes with similarities: {codes_with_scores}")
        
        
        return nearest_codes

    
    async def design(self) -> List[Dict[str, Any]]:
        """Main method: Run complete 4-stage CodeDesigner pipeline with comprehensive error handling"""
        start_time = time.time()
        
        try:
            self.verbose_reporter.step_start("CodeDesigner Pipeline")
            self.verbose_reporter.stat_line(f"Model: {self.config.model}")
            self.verbose_reporter.stat_line(f"Similarity threshold: {self.config.similarity_threshold}")
            
            # Initialize processing statistics
            self._processing_stats = {
                'start_time': start_time,
                'clusters_found': 0,
                'themes_extracted': 0,
                'themes_embedded': 0,
                'batches_created': 0,
                'clusters_processed': 0,
                'codes_added': 0,
                'codes_modified': 0,
                'validation_failures': 0,
                'api_errors': 0,
                'stage_times': {}
            }
            
            # Stage 0: Extract cluster data with error recovery
            stage_start = time.time()
            try:
                clusters = self.extract_cluster_data()
                self._processing_stats['clusters_found'] = len(clusters)
                self.verbose_reporter.stat_line(f"Extracted data from {len(clusters)} clusters")
                
                if not clusters:
                    self.verbose_reporter.warning("No valid clusters found - check input data")
                    return []
                    
            except Exception as e:
                self.verbose_reporter.error(f"Failed to extract cluster data: {e}")
                return []
                
            self._processing_stats['stage_times']['data_extraction'] = time.time() - stage_start
            
            # Stage 1: Theme Extraction with comprehensive error handling
            stage_start = time.time()
            try:
                themes = await self.extract_themes(clusters)
                self._processing_stats['themes_extracted'] = len(themes)
                
                if not themes:
                    self.verbose_reporter.warning("No themes extracted - check cluster content or API connectivity")
                    return []
                    
            except Exception as e:
                self.verbose_reporter.error(f"Critical failure in theme extraction: {e}")
                # Attempt graceful degradation
                self.verbose_reporter.warning("Attempting to continue with partial results...")
                themes = {}
                
            self._processing_stats['stage_times']['theme_extraction'] = time.time() - stage_start
            
            # Stage 1.5: Expand multi-theme clusters into sub-clusters
            themes, clusters = self.expand_multi_theme_clusters(themes, clusters)
            
            # Stage 2: Theme Embedding with fallback handling
            stage_start = time.time()
            try:
                theme_embeddings = await self.similarity_engine.embed_themes(themes)
                self._processing_stats['themes_embedded'] = len(theme_embeddings)
                
                if not theme_embeddings:
                    self.verbose_reporter.warning("No theme embeddings generated - check embedding API")
                    return []
                    
            except Exception as e:
                self.verbose_reporter.error(f"Critical failure in theme embedding: {e}")
                return []
                
            self._processing_stats['stage_times']['theme_embedding'] = time.time() - stage_start
            
            # Stage 3: Similarity-Based Batching with validation
            stage_start = time.time()
            try:
                dissimilarity_batches = self.similarity_engine.create_dissimilarity_batches(theme_embeddings, themes)
                self._processing_stats['batches_created'] = len(dissimilarity_batches)
                
                if not dissimilarity_batches:
                    self.verbose_reporter.warning("No batches created - all themes may be too similar")
                    # Create single large batch as fallback
                    dissimilarity_batches = [list(theme_embeddings.keys())]
                    
            except Exception as e:
                self.verbose_reporter.error(f"Failure in batch creation: {e}")
                # Fallback: process all clusters individually
                dissimilarity_batches = [[cid] for cid in theme_embeddings.keys()]
                
            self._processing_stats['stage_times']['batch_creation'] = time.time() - stage_start
            
            # Stage 4: Sequential Batch Processing with error recovery
            stage_start = time.time()
            try:
                all_results = await self.process_batches_sequentially(
                    dissimilarity_batches, clusters, themes, theme_embeddings
                )
                self._processing_stats['clusters_processed'] = len(all_results)
                
            except Exception as e:
                # Enhanced error logging to identify exact failure point
                error_msg = str(e).strip()
                self.verbose_reporter.error(f"Critical failure in batch processing: {repr(error_msg)}")
                self.verbose_reporter.error(f"Error type: {type(e).__name__}")
                self.verbose_reporter.error(f"Error length: {len(error_msg)}")
                
                # Print the full stack trace to understand where exactly this is failing
                import traceback
                self.verbose_reporter.error("Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        self.verbose_reporter.error(f"  {line}")
                        
                all_results = []
                
            self._processing_stats['stage_times']['batch_processing'] = time.time() - stage_start
            
            # Final statistics and validation
            processing_time = time.time() - start_time
            self._processing_stats['total_time'] = processing_time
            
            # Final codebook statistics
            try:
                final_version_info = await self.shared_codebook.get_version_info()
                self._processing_stats['final_codebook_version'] = final_version_info['version']
                self._processing_stats['final_codebook_size'] = final_version_info['total_codes']
                self.verbose_reporter.stat_line(f"Final codebook: version {final_version_info['version']}, {final_version_info['total_codes']} codes")
            except Exception as e:
                self.verbose_reporter.error(f"Failed to get final codebook stats: {e}")
            
            # Comprehensive final reporting
            self._generate_final_report(processing_time, len(all_results))
            
            self.verbose_reporter.step_complete("CodeDesigner Pipeline")
            
            self._results = all_results
            return all_results
            
        except Exception as e:
            # Ultimate fallback error handling
            self.verbose_reporter.error(f"Critical pipeline failure: {e}")
            processing_time = time.time() - start_time
            self._processing_stats['total_time'] = processing_time
            self._processing_stats['critical_failure'] = str(e)
            
            self.verbose_reporter.step_complete("CodeDesigner Pipeline (FAILED)")
            return []
    
    def _generate_final_report(self, processing_time: float, clusters_processed: int):
        """Generate comprehensive final processing report"""
        self.verbose_reporter.step_start("Final Processing Report")
        
        # Performance metrics
        self.verbose_reporter.stat_line(f"Total processing time: {processing_time:.1f}s")
        self.verbose_reporter.stat_line(f"Clusters processed: {clusters_processed}")
        
        if 'stage_times' in self._processing_stats:
            self.verbose_reporter.stat_line("Stage breakdown:")
            for stage, duration in self._processing_stats['stage_times'].items():
                percentage = (duration / processing_time) * 100 if processing_time > 0 else 0
                self.verbose_reporter.stat_line(f"  {stage}: {duration:.1f}s ({percentage:.1f}%)")
        
        # Processing efficiency
        if processing_time > 0:
            clusters_per_second = clusters_processed / processing_time
            self.verbose_reporter.stat_line(f"Processing rate: {clusters_per_second:.2f} clusters/second")
        
        # Success rates
        clusters_found = self._processing_stats.get('clusters_found', 0)
        themes_extracted = self._processing_stats.get('themes_extracted', 0)
        
        if clusters_found > 0:
            theme_success_rate = (themes_extracted / clusters_found) * 100
            processing_success_rate = (clusters_processed / clusters_found) * 100
            
            self.verbose_reporter.stat_line(f"Theme extraction success: {themes_extracted}/{clusters_found} ({theme_success_rate:.1f}%)")
            self.verbose_reporter.stat_line(f"Overall processing success: {clusters_processed}/{clusters_found} ({processing_success_rate:.1f}%)")
        
        # Codebook growth
        initial_codes = len(self.starter_codes)
        final_codes = self._processing_stats.get('final_codebook_size', initial_codes)
        codes_added = final_codes - initial_codes
        
        self.verbose_reporter.stat_line(f"Codebook growth: {initial_codes} → {final_codes} (+{codes_added} codes)")
        
        # Quality indicators
        validation_failures = self._processing_stats.get('validation_failures', 0)
        api_errors = self._processing_stats.get('api_errors', 0)
        
        if validation_failures > 0 or api_errors > 0:
            self.verbose_reporter.stat_line(f"Issues encountered: {validation_failures} validation failures, {api_errors} API errors")
        else:
            self.verbose_reporter.stat_line("Processing completed without major issues")
        
        self.verbose_reporter.step_complete("Final Processing Report")
    
    def generate(self) -> CodeGeneratorReasoningResults:
        """Generate codes and return complete reasoning results"""
        return asyncio.run(self.generate_async())
    
    async def generate_async(self) -> CodeGeneratorReasoningResults:
        """Async method for code generation - returns proper CodeGeneratorReasoningResults"""
        # Run the design pipeline
        results = await self.design()
        
        # Extract deduplicated codebook from SharedCodebook
        final_codes, version = await self.shared_codebook.get_current_snapshot()
        
        # Get raw cluster data for stats calculations
        cluster_data = self._prepare_cluster_data_for_results()
        
        
        # Convert to CodeGeneratorReasoningResults format
        return CodeGeneratorReasoningResults(
            # Raw cluster results
            cluster_results=results,
            
            step1_inputs=self.step1_inputs,
            step2_inputs=self.step2_inputs,   
            step3_inputs=self.step3_inputs,
            step4_inputs=self.step4_inputs,
            
            step1_summaries=self.step1_summaries,
            step2_analysis=self.step2_analysis,  
            step3_recommendations=self.step3_recommendations,
            step4_validations=self.step4_validations,
            step4_validated_codes=self.step4_validated_codes,
            
            # Processing metadata
            stats=self.summary(),
            generator_version="codeDesigner_v3_complex_prompts",
            var_lab=self.var_lab,
            total_clusters=len(self.cluster_assignments),
            total_ideas=sum(len(cluster_data.get('ideas', [])) for cluster_data in cluster_data.values()) if cluster_data else 0,
            processing_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # Cluster assignments for cross-reference
            cluster_assignments=self.cluster_assignments,
            
            # New fields for alignment with old codeGenerator
            codebook=final_codes,  # Final deduplicated codebook from SharedCodebook
            cluster_data=cluster_data,  # Raw cluster data for stats calculations
            validation_details=self.step4_validations  # Detailed validation results
        )
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get processing results"""
        return self._results
    
    def _prepare_cluster_data_for_results(self) -> Dict[int, Dict[str, Any]]:
        """Prepare cluster data from cluster_results for backward compatibility with old codeGenerator format"""
        clusters = {}
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            'cluster_id': cluster_id,
                            'ideas': [],
                            'embeddings': [],
                            'respondent_ids': []
                        }
                    
                    # Add idea data
                    clusters[cluster_id]['ideas'].append(idea.idea)
                    clusters[cluster_id]['respondent_ids'].append(idea.idea_id)  # Using idea_id as respondent identifier
                    
                    # Add embedding if available
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
        
        return clusters
    
    # ============================================================================
    # PHASE 3: UNLIMITED PARALLELISM WITH VERSION-BASED CODEBOOK UPDATES
    async def _process_single_cluster(self, cluster_id: str, clusters: Dict, themes: Dict,
                                               codebook_snapshot: List[Dict], base_version: int) -> Optional[Dict[str, Any]]:
        """Process single cluster with unlimited concurrency - no artificial limits (Phase 3)"""
        
        if cluster_id not in themes or cluster_id not in clusters:
            return None
        
        cluster_data = clusters[cluster_id]
        theme_data = themes[cluster_id]
        
        try:
            # Step 1: Get current codebook for candidate selection (ensures latest codes are visible)
            current_codes, _ = await self.shared_codebook.get_current_snapshot()
            nearest_codes = await self._find_nearest_codes_by_theme(
                cluster_id, theme_data, current_codes, k=5
            )
            
            # Step 1: Candidate selection - pure unlimited call
            step1_start = time.time()
            candidate_selection = await self._select_candidate_codes(
                cluster_id, cluster_data, theme_data, nearest_codes
            )
            step1_duration = time.time() - step1_start
            
            # Step 2: Code generation - pure unlimited call
            step2_start = time.time()
            code_generation = await self._generate_code(
                cluster_id, cluster_data, theme_data, candidate_selection
            )
            step2_duration = time.time() - step2_start
            
            # Step 3: Validation - pure unlimited call
            step3_start = time.time()
            validation = await self._validate_code(
                cluster_id, cluster_data, theme_data, code_generation, candidate_selection
            )
            step3_duration = time.time() - step3_start
            
            # Extract final code/definition from validation
            final_code = None
            final_definition = None
            if validation and hasattr(validation, 'code_validations') and validation.code_validations:
                first_validation = validation.code_validations[0]
                if first_validation.validated_code:
                    final_code = first_validation.validated_code.code
                    final_definition = first_validation.validated_code.definition
            
            
            return {
                'cluster_id': cluster_id,
                'theme_name': self._get_theme_statement(theme_data),
                'theme_description': self._get_theme_description(theme_data),
                'ideas_count': len(cluster_data['ideas']),
                'candidate_selection': candidate_selection,
                'code_generation': code_generation,
                'validation': validation,
                'final_code': final_code,
                'final_definition': final_definition,
                'base_version': base_version,
                'timing': {
                    'step1_duration': step1_duration,
                    'step2_duration': step2_duration,
                    'step3_duration': step3_duration
                }
            }
            
        except Exception as e:
            import traceback
            self.verbose_reporter.error(f"Pipeline failed for cluster {cluster_id}: {e}")
            self.verbose_reporter.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    async def _merge_codebook_updates(self, results: List[Dict[str, Any]], base_version: int):
        """Merge all codebook updates from sub-batch atomically (Phase 3)"""
        
        # Collect all new codes from the sub-batch results
        all_new_codes = []
        for result in results:
            cluster_id = result.get('cluster_id', 'unknown')
            
            # Debug: Check result structure
            if not result:
                self.verbose_reporter.error(f"C{cluster_id}: Empty result in merge_codebook_updates")
                continue
                
            if not result.get('validation'):
                self.verbose_reporter.error(f"C{cluster_id}: No validation in result")
                continue
                
            if not hasattr(result['validation'], 'code_validations'):
                self.verbose_reporter.error(f"C{cluster_id}: No code_validations in validation")
                continue
            
            # Process each code validation
            for i, code_validation in enumerate(result['validation'].code_validations):
                if code_validation.validated_code:
                    all_new_codes.append({
                        'code': code_validation.validated_code.code,
                        'definition': code_validation.validated_code.definition,
                        'cluster_id': cluster_id
                    })
                    self.verbose_reporter.stat_line(f"C{cluster_id}: Validated_code '{code_validation.validated_code.code}' - codebook updated")
                else:
                    self.verbose_reporter.error(f"C{cluster_id}: code_validation[{i}] has no validated_code")
        
        # Single atomic update to SharedCodebook if there are new codes
        if all_new_codes:
            self.verbose_reporter.stat_line(f"Batch updating SharedCodebook with {len(all_new_codes)} new codes")
            await self.shared_codebook.batch_update(all_new_codes, base_version)
        else:
            self.verbose_reporter.stat_line("No new codes to add to SharedCodebook from this sub-batch")
    
    
    async def _select_candidate_codes(self, cluster_id: Union[int, str], cluster_data: Dict, theme_data, nearest_codes: List[Dict]):
        """Select candidate codes with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in nearest_codes[:20]])
            
            cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}"
            
            # Prepare exact parameters for prompt
            params = {
                "survey_question": self.var_lab,
                "language": DEFAULT_LANGUAGE,
                "cluster_summary": cluster_summary,
                "code_text": codes_text
            }
            
            prompt = CANDIDATE_CODE_SELECTION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step2", **params)
            
            if self.verbose_detailed: 
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Starting candidate selection API call")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Prompt length: {len(prompt)} chars")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Available codes: {len(nearest_codes)}")
            
            # Use async wrapper for true concurrency with GPT-5 reasoning parameters
            resp = await async_responses_create_with_semaphore(
                model="gpt-5-nano",
                prompt=prompt,
                reasoning_effort="minimal",
                text_verbosity="low"
            )
            response = CandidateCodeSelectionOutput.model_validate_json(resp.output_text).root
            
            if self.verbose_detailed: 
                if response is None:
                    self.verbose_reporter.error(f"C{cluster_id}: STEP1 - API returned None response")
                    return []
                elif len(response) == 0:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - API returned empty list (valid)")
                else:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - API returned {len(response)} candidates")
            
            # # Capture step2_analysis - the actual candidate codes used in pipeline
            if response:
                self.step2_analysis[cluster_id] = [
                    {"code": code.code, "definition": code.definition} 
                    for code in response
                ]
            
            return response
            
        except Exception as e:
            # Error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Candidate selection failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP1 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return []
    
    async def _generate_code(self, cluster_id: Union[int, str], cluster_data: Dict, theme_data, candidate_selection):
        """Generate code with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            candidate_codes_text = ""
            if candidate_selection and len(candidate_selection) > 0:
                candidate_codes_text = "\n\n".join([f"Code: {code.code}\nDefinition: {code.definition}" 
                                                   for code in candidate_selection])
                
            cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}"
            
            # Prepare exact parameters for prompt
            params = {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "cluster_summary": cluster_summary,
                "candidate_codes": candidate_codes_text
            }
            
            prompt = CODE_GENERATION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step3", **params)
            
            if self.verbose_detailed: 
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Starting code generation API call")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Prompt length: {len(prompt)} chars")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Candidate codes: {len(candidate_selection) if candidate_selection else 0}")
            
            # Use async wrapper for true concurrency with GPT-5 reasoning parameters
            resp = await async_responses_create_with_semaphore(
                model="gpt-5-nano",
                prompt=prompt,
                reasoning_effort="minimal",
                text_verbosity="low"
            )
            response = CodeRecommendation.model_validate_json(resp.output_text)
            
            if self.verbose_detailed: 
                if response is None:
                    self.verbose_reporter.error(f"C{cluster_id}: STEP2 - API returned None response")
                    return None
                elif not hasattr(response, 'coding_decisions') or not response.coding_decisions:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - API returned response with no coding decisions")
                else:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - API returned {len(response.coding_decisions)} coding decisions")
            
            # Capture step3_recommendations (code generation results)
            if response and hasattr(response, 'coding_decisions'):
                self.step3_recommendations[cluster_id] = {
                    'coding_decisions': [
                        {
                            'theme_number': decision.theme_number,
                            'decision': decision.decision,
                            'final_code_label': decision.final_code_label,
                            'final_code_definition': decision.final_code_definition,
                            'source_code': decision.source_code,
                            'justification': decision.justification
                        } for decision in response.coding_decisions
                    ]
                }
            
            return response
            
        except Exception as e:
            # Error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Code generation failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP2 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return None
    
    # Removed _make_instructor_call_with_cleanup - using responses.create directly now

    async def _validate_code(self, cluster_id: Union[int, str], cluster_data: Dict, theme_data, 
                                       code_generation, candidate_selection):
        """Validate code with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            # Use candidate_selection instead of full codebook (consistent with rate-limited version)
            # if candidate_selection and len(candidate_selection) > 0:
            #     validation_codes_text = "\n".join([
            #         f"Code: {code.code}\nDefinition: {code.definition}\n" 
            #         for code in candidate_selection
            #     ])
            # else:
            #     validation_codes_text = "No existing codes available."
            
            cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}"
            step3_recommendation_text = str(code_generation.model_dump_json(indent=2)) if code_generation else "No recommendations"
            
            # Prepare exact parameters for prompt
            params = {
                'language': DEFAULT_LANGUAGE,
                'survey_question': self.var_lab,
                'cluster_summary': cluster_summary,
                #'candidate_codes': validation_codes_text,
                'step3_recommendation': step3_recommendation_text
            }
            
            prompt = VALIDATION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step4", **params)
            
            if self.verbose_detailed: 
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Starting validation API call")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Prompt length: {len(prompt)} chars")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Candidate codes: {len(candidate_selection) if candidate_selection else 0}")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Has code_generation: {code_generation is not None}")
            
            # Use async wrapper for true concurrency with GPT-5 reasoning parameters
            resp = await async_responses_create_with_semaphore(
                model="gpt-5-nano",
                prompt=prompt,
                reasoning_effort="minimal",
                text_verbosity="low"
            )
            response = ValidationResult.model_validate_json(resp.output_text)
            
            if self.verbose_detailed: 
                if response is None:
                    self.verbose_reporter.error(f"C{cluster_id}: STEP3 - API returned None response")
                    return None
                elif not hasattr(response, 'code_validations') or not response.code_validations:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - API returned response with no validations")
                else:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - API returned {len(response.code_validations)} validations")
            
            # Capture step4_validations
            if response and hasattr(response, 'code_validations'):
                self.step4_validations[cluster_id] = {
                    'code_validations': [
                        {
                            'theme_number': validation.theme_number,
                            'original_recommendation': {
                                'code': validation.original_recommendation.code,
                                'definition': validation.original_recommendation.definition
                            },
                            'decision': validation.decision,
                            'decision_rationale': validation.decision_rationale,
                            'validated_code': {
                                'code': validation.validated_code.code,
                                'definition': validation.validated_code.definition
                            }
                        } for validation in response.code_validations
                    ]
                }
            
            # Update SharedCodebook based on validation decisions (same logic as rate-limited version)
            if response and response.code_validations:
                codebook_updated = False
                new_version = None
                
                # Process each validation decision
                for i, validation in enumerate(response.code_validations):
                    # Get corresponding coding decision
                    if (hasattr(code_generation, 'coding_decisions') and 
                        code_generation.coding_decisions and 
                        i < len(code_generation.coding_decisions)):
                        coding_decision = code_generation.coding_decisions[i]
                        
                        # Complete decision matrix: Both APPROVE and REJECT are actionable
                        if validation.decision in ["APPROVE", "REJECT"] and validation.validated_code:
                            #action_source = "ORIGINAL" if validation.decision == "APPROVE" else "VALIDATED"
                            
                            # Handle create decisions (both APPROVE and REJECT add new codes)
                            if coding_decision.decision == "create":
                                added, new_version = await self.shared_codebook.add_code_if_new(
                                    validation.validated_code.code, validation.validated_code.definition
                                )
                                if added:
                                    self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                    codebook_updated = True
                                    #self.verbose_reporter.stat_line(f"C{cluster_id}: CREATE+{validation.decision} - Added new code '{validation.validated_code.code}' ({action_source})")
                            
                            # Handle modify decisions (both APPROVE and REJECT replace original)
                            elif coding_decision.decision == "modify" and coding_decision.source_code:
                                replaced, new_version = await self.shared_codebook.replace_code(
                                    coding_decision.source_code, 
                                    validation.validated_code.code, validation.validated_code.definition
                                )
                                if replaced:
                                    self._processing_stats['codes_modified'] = self._processing_stats.get('codes_modified', 0) + 1
                                    codebook_updated = True
                                    #self.verbose_reporter.stat_line(f"C{cluster_id}: MODIFY+{validation.decision} - Replaced '{coding_decision.source_code}' with '{validation.validated_code.code}' ({action_source})")
                            
                            # Handle use decisions
                            elif coding_decision.decision == "use":
                                #if validation.decision == "APPROVE":
                                    # use + APPROVE = no update (existing code stays as-is)
                                    # self.verbose_reporter.stat_line(f"C{cluster_id}: USE+APPROVE - No codebook update needed")
                                if validation.decision == "REJECT":
                                    # use + REJECT = replace existing with validated_code
                                    # Need to get the original code name that was being used
                                    if hasattr(coding_decision, 'source_code') and coding_decision.source_code:
                                        replaced, new_version = await self.shared_codebook.replace_code(
                                            coding_decision.source_code,
                                            validation.validated_code.code, validation.validated_code.definition
                                        )
                                        if replaced:
                                            self._processing_stats['codes_modified'] = self._processing_stats.get('codes_modified', 0) + 1
                                            codebook_updated = True
                                            #self.verbose_reporter.stat_line(f"C{cluster_id}: USE+REJECT - Replaced '{coding_decision.source_code}' with '{validation.validated_code.code}' (VALIDATED)")
                                    else:
                                        # Fallback: add as new code if we can't identify original
                                        added, new_version = await self.shared_codebook.add_code_if_new(
                                            validation.validated_code.code, validation.validated_code.definition
                                        )
                                        if added:
                                            self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                            codebook_updated = True
                                            #self.verbose_reporter.stat_line(f"C{cluster_id}: USE+REJECT - Added validated code '{validation.validated_code.code}' (no original identified)")
                        
                        elif validation.decision == "REVISE" and validation.validated_code:
                            # Validation revised the decision - use the revised code
                            added, new_version = await self.shared_codebook.add_code_if_new(
                                validation.validated_code.code, validation.validated_code.definition
                            )
                            if added:
                                self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                codebook_updated = True
                                #self.verbose_reporter.stat_line(f"C{cluster_id}: REVISE - Added revised code '{validation.validated_code.code}'")
                        
                        else:
                            self.verbose_reporter.error(f"C{cluster_id}: UNHANDLED validation decision '{validation.decision}' or missing validated_code")
                
                # Generate embeddings for new/modified codes
                if codebook_updated and new_version is not None:
                    # Get updated codebook
                    updated_codes, _ = await self.shared_codebook.get_current_snapshot()
                    
                    # Check if we need to regenerate embeddings (cache invalidated)
                    cached_embeddings = await self.shared_codebook.get_embeddings_for_version(new_version)
                    if cached_embeddings is None:
                        # Generate embeddings for all codes
                        if self.verbose_detailed: 
                            self.verbose_reporter.stat_line(f"C{cluster_id}: Generating embeddings for updated codebook (version {new_version})")
                        code_texts = [f"{code['code']}: {code['definition']}" for code in updated_codes]
                        try:
                            code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                            await self.shared_codebook.cache_embeddings(new_version, code_embeddings)
                            #self.verbose_reporter.stat_line(f"C{cluster_id}: Updated embeddings cache for version {new_version}")
                        except Exception as e:
                            self.verbose_reporter.error(f"C{cluster_id}: Failed to generate embeddings for new codes: {e}")
            
            return response
            
        except Exception as e:
            # Enhanced error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP3 - Code validation failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP3 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP3 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP3 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return None
    
    # ============================================================================
    # PHASE 3: PERFORMANCE TRACKING AND BOTTLENECK IDENTIFICATION
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary statistics"""
        clusters_found = self._processing_stats.get('clusters_found', 0)
        themes_extracted = self._processing_stats.get('themes_extracted', 0)
        clusters_processed = len(self._results)
        
        # Calculate success rates
        theme_success_rate = (themes_extracted / clusters_found * 100) if clusters_found > 0 else 0
        processing_success_rate = (clusters_processed / clusters_found * 100) if clusters_found > 0 else 0
        
        # Calculate processing efficiency
        total_time = self._processing_stats.get('total_time', 0)
        processing_rate = (clusters_processed / total_time) if total_time > 0 else 0
        
        # Codebook statistics
        initial_codes = len(self.starter_codes)
        final_codes = self._processing_stats.get('final_codebook_size', initial_codes)
        
        return {
            # Core metrics
            'total_clusters_processed': clusters_processed,
            'clusters_found': clusters_found,
            'themes_extracted': themes_extracted,
            'processing_success_rate': round(processing_success_rate, 2),
            'theme_extraction_success_rate': round(theme_success_rate, 2),
            
            # Performance metrics
            'total_processing_time_seconds': round(total_time, 2),
            'processing_rate_clusters_per_second': round(processing_rate, 3),
            'stage_times': self._processing_stats.get('stage_times', {}),
            
            # Configuration
            'similarity_threshold': self.config.similarity_threshold,
            'model_used': self.config.model,
            'max_sub_batch_size': self.config.max_sub_batch_size,
            
            # Codebook evolution
            'initial_codebook_size': initial_codes,
            'final_codebook_size': final_codes,
            'codebook_growth': final_codes - initial_codes,
            'final_codebook_version': self._processing_stats.get('final_codebook_version', 0),
            
            # Quality indicators
            'batches_created': self._processing_stats.get('batches_created', 0),
            'themes_embedded': self._processing_stats.get('themes_embedded', 0),
            'validation_failures': self._processing_stats.get('validation_failures', 0),
            'api_errors': self._processing_stats.get('api_errors', 0),
            
            # Pipeline health
            'pipeline_completed_successfully': 'critical_failure' not in self._processing_stats,
            'critical_failure': self._processing_stats.get('critical_failure'),
            
            # Raw processing stats
            'raw_processing_stats': self._processing_stats
        }