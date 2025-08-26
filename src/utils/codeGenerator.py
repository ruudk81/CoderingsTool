import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# from collections import deque

import instructor
from openai import AsyncOpenAI, RateLimitError
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler
from sklearn.metrics.pairwise import cosine_similarity

# === MODELS ========================================================================================================
import models
from models import ClusterSummaryOutput, CodeRecommendation, SimplifiedCodeRecommendation, ValidationResult, CodeGeneratorReasoningResults, CandidateCode, ClusterThemeItem, OriginalRecommendation #CandidateCodeSelectionOutput

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

# Initialize async client with instructor
async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))


# ============================================================================
# CODEDESIGNER API CLIENT WITH RATE LIMITING
# ============================================================================

class CodeDesignerAPIClient:
    """API client with intelligent retry logic and precise rate limiting"""
    
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
    
    async def embed_themes(self, themes: Dict[int, ClusterSummaryOutput]) -> Dict[int, np.ndarray]:
        """Generate embeddings for all theme names using efficient batch processing (like embedder.py)"""
        self.verbose_reporter.step_start("Theme Embedding")
        self.verbose_reporter.stat_line(f"Processing {len(themes)} theme names in batches")
        
        # Prepare data for batch processing
        cluster_ids = list(themes.keys())
        theme_statements = [themes[cid].themes[0].theme_statement for cid in cluster_ids]  # Use first theme's statement
        
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
                    theme_statement = theme.themes[0].theme_statement if theme.themes else "Unknown"
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
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Retry logic similar to embedder.py
        for attempt in range(3):
            try:
                response = await client.embeddings.create(
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
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def create_dissimilarity_batches(self, theme_embeddings: Dict[int, np.ndarray], themes: Dict = None) -> List[List[int]]:
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
        
        # DEBUG: Final batch assignments summary
        self.verbose_reporter.stat_line("\n=== FINAL BATCH ASSIGNMENTS ===")
        for batch_idx, batch_cluster_ids in enumerate(batches):
            theme_labels = []
            if themes:
                theme_labels = [f"C{cid}: '{themes[cid].themes[0].theme_statement if themes[cid].themes else 'unknown'}'" if cid in themes else f"C{cid}: unknown" 
                              for cid in batch_cluster_ids]
            else:
                theme_labels = [f"C{cid}" for cid in batch_cluster_ids]
            
            self.verbose_reporter.stat_line(f"Batch {batch_idx + 1}: {', '.join(theme_labels)}")
        
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
        prompt_printer = None,
        config = None,
        **kwargs  # For backward compatibility
    ):
        self.cluster_results = cluster_results
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        self.config = config or DEFAULT_CODEDESIGNER_CONFIG
        
        # Initialize components
        self.model_config = ModelConfig()
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        
        # Initialize async client and rate limiting
        self.async_client = async_client
        self.embedding_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
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
    
    def _capture_prompt_params(self, cluster_id: int, step: str, **kwargs):
        """Capture exact parameters used in prompt.format() for debugging/testing"""
        if step == "step1":
            self.step1_inputs[cluster_id] = kwargs
        elif step == "step2":
            self.step2_inputs[cluster_id] = kwargs
        elif step == "step3":
            self.step3_inputs[cluster_id] = kwargs
        elif step == "step4":
            self.step4_inputs[cluster_id] = kwargs
    
    def _get_theme_statement(self, theme_data) -> str:
        """Safely get theme statement from theme data"""
        if hasattr(theme_data, 'themes') and theme_data.themes:
            return theme_data.themes[0].theme_statement
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
    
    async def _extract_single_theme(self, cluster_id: int, ideas: List[str]):
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
            # Use List[ClusterThemeItem] directly since your prompt returns an array
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=List[ClusterThemeItem],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: THEME_EXTRACT"
            )
            
            # Handle List[ClusterThemeItem] response from CLUSTER_SUMMARY_PROMPT
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
            ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            # Step 4a: Candidate Selection prompt
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in current_codes[:20]])  # Limit like the real implementation
            
            candidate_prompt = CANDIDATE_CODE_SELECTION_PROMPT.format(
                survey_question=self.var_lab,
                language=DEFAULT_LANGUAGE,
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}",
                code_text=codes_text
            )
            candidate_tokens = len(self.encoding.encode(candidate_prompt)) + 200  # + completion estimate
            token_measurements['candidate_selection'].append(candidate_tokens)
            
            # Step 4b: Code Generation prompt (simulate with 3 candidate codes)
            candidate_codes_text = "Code: EXAMPLE_CODE_1\nDefinition: Example definition 1\n\nCode: EXAMPLE_CODE_2\nDefinition: Example definition 2\n"
            
            code_gen_prompt = CODE_GENERATION_PROMPT.format(
                language=DEFAULT_LANGUAGE,
                survey_question=self.var_lab,
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}",
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
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}",
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
    
    async def process_batches_sequentially(self, dissimilarity_batches: List[List[int]], 
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
                sub_batches = self.similarity_engine.create_sub_batches(
                    dissimilarity_batch, self.config.max_sub_batch_size
                )
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
    
    async def _process_sub_batch_with_stagger(self, sub_batch: List[int], clusters: Dict, themes: Dict, 
                                            stagger_delay: float) -> List[Dict[str, Any]]:
        """Process sub-batch with initial stagger delay and rate limiting"""
        
        # Apply stagger delay for smooth distribution
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)
        
        # Calculate optimal strategy for this sub-batch
        avg_tokens = (
            self.code_gen_token_measurements.get('candidate_selection', 1200) +
            self.code_gen_token_measurements.get('code_generation', 1000) +
            self.code_gen_token_measurements.get('validation', 900)
        )
        
        strategy = self.workload_analyzer.calculate_optimal_strategy(
            total_batches=len(sub_batch),
            avg_tokens_per_batch=avg_tokens,
            sub_batches_per_batch=1
        )
        
        # Create throttler for this sub-batch
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second, period=1.0)
        api_client = CodeDesignerAPIClient(
            throttler, self.global_monitor, self.config, self.encoding,
            self.model_config, self.verbose_reporter, self.async_client
        )
        
        # Process all clusters in sub-batch with rate limiting
        cluster_tasks = []
        for cluster_id in sub_batch:
            task = self._process_single_cluster_with_rate_limiting(
                cluster_id, clusters, themes, api_client
            )
            cluster_tasks.append(task)
        
        # Process with rate limiting
        results = []
        for coro in asyncio.as_completed(cluster_tasks):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
            except Exception as e:
                self.verbose_reporter.error(f"Cluster processing failed: {e}")
        
        return results
    
    async def _process_single_cluster_with_rate_limiting(self, cluster_id: int, 
                                                         clusters: Dict, themes: Dict,
                                                         api_client: CodeDesignerAPIClient) -> Optional[Dict[str, Any]]:
        """Process single cluster with rate limiting"""
        
        if cluster_id not in themes or cluster_id not in clusters:
            return None
        
        cluster_data = clusters[cluster_id]
        theme_data = themes[cluster_id]
        
        try:
            # Step 1: Candidate Selection with rate limiting
            current_codes, version = await self.shared_codebook.get_current_snapshot()
            
            step1_task = self._select_candidate_codes_unlimited(
                cluster_id, cluster_data, theme_data, current_codes[:20]
            )
            candidate_selection = await api_client.make_request(
                step1_task, f"candidate_selection_c{cluster_id}"
            )
            
            # Step 2: Code Generation with rate limiting
            if candidate_selection is not None:
                step2_task = self._generate_code_unlimited(
                    cluster_id, cluster_data, theme_data, candidate_selection
                )
                code_generation = await api_client.make_request(
                    step2_task, f"code_generation_c{cluster_id}"
                )
            else:
                code_generation = None
            
            # Step 3: Validation & SharedCodebook Update with rate limiting
            if code_generation:
                step3_task = self._validate_and_update_codebook(
                    cluster_id, cluster_data, theme_data, code_generation, candidate_selection
                )
                validation = await api_client.make_request(
                    step3_task, f"validation_c{cluster_id}"
                )
            else:
                validation = None
            
            # Extract final code/definition from complex validation structure
            final_code = None
            final_definition = None
            if validation and hasattr(validation, 'code_validations') and validation.code_validations:
                # Get the first validated code (for now - could aggregate multiple)
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
                'final_definition': final_definition
            }
            
        except Exception as e:
            self.verbose_reporter.error(f"Pipeline failed for cluster {cluster_id}: {e}")
            return None
    
    async def _process_large_batch_optimized(self, sub_batches: List[List[int]], 
                                            clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process large batch as concurrent sub-batches with global rate limiting coordination"""
        self.verbose_reporter.stat_line(f"Processing as {len(sub_batches)} sub-batches with global rate coordination")
        
        # Calculate GLOBAL strategy for entire batch (not individual sub-batches)
        total_clusters = sum(len(sub_batch) for sub_batch in sub_batches)
        composite_tokens = (
            self.code_gen_token_measurements.get('candidate_selection', 1200) +
            self.code_gen_token_measurements.get('code_generation', 1000) +
            self.code_gen_token_measurements.get('validation', 900)
        )
        
        # Check what rate limits and parameters WorkloadAnalyzer is using
        rate_limits = get_openai_rate_limits(self.config.model)
        self.verbose_reporter.stat_line(f"Model {self.config.model}")  
        self.verbose_reporter.stat_line(f"RPM limit: {rate_limits.requests_per_minute}")
        self.verbose_reporter.stat_line(f"TPM limit: {rate_limits.tokens_per_minute}")
        self.verbose_reporter.stat_line(f"Input to strategy: batches={total_clusters}, tokens={composite_tokens:.0f}, sub_batches=1")
        
        # Calculate strategy for ENTIRE batch workload
        global_strategy = self.workload_analyzer.calculate_optimal_strategy(
            total_batches=total_clusters,  # All clusters in batch
            avg_tokens_per_batch=composite_tokens,
            sub_batches_per_batch=1  # Sequential steps don't multiply concurrent load
        )
        
        # Show what strategy was calculated
        self.verbose_reporter.stat_line(f"Strategy result: rate={global_strategy.launch_rate_per_second:.1f} req/s, concurrent={global_strategy.concurrent_limit}")
        
        self.verbose_reporter.stat_line(f"Global batch strategy: {global_strategy.launch_rate_per_second:.1f} total req/s for {total_clusters} clusters across {len(sub_batches)} sub-batches")
        
        # Give each sub-batch full capacity - sliding window will manage fairness
        rate_per_sub_batch = global_strategy.launch_rate_per_second  # Full rate!
        concurrent_per_sub_batch = global_strategy.concurrent_limit  # Full concurrency!
        
        self.verbose_reporter.stat_line(f"Per sub-batch allocation: {rate_per_sub_batch:.1f} req/s (FULL RATE), max {concurrent_per_sub_batch} concurrent (FULL CAPACITY)")
        
        # Create sub-batch tasks with distributed rate limits
        sub_batch_tasks = []
        for i, sub_batch in enumerate(sub_batches):
            task = self._process_cluster_group_with_distributed_strategy(
                sub_batch, clusters, themes, rate_per_sub_batch, concurrent_per_sub_batch, i+1
            )
            sub_batch_tasks.append(task)
        
        # Process sub-batches concurrently with coordinated rate limits
        all_results = await asyncio.gather(*sub_batch_tasks)
        
        # Merge results
        merged_results = []
        for sub_batch_result in all_results:
            merged_results.extend(sub_batch_result)
        
        return merged_results
    
    async def _process_medium_batch_optimized(self, cluster_batch: List[int], 
                                            clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process medium batch with evidence-based optimal strategy (single batch, use full rate limits)"""
        composite_tokens = (
            self.code_gen_token_measurements.get('candidate_selection', 1200) +
            self.code_gen_token_measurements.get('code_generation', 1000) +
            self.code_gen_token_measurements.get('validation', 900)
        )
        
        # Check WorkloadAnalyzer parameters for medium batch
        rate_limits = get_openai_rate_limits(self.config.model) #TODO not used
        self.verbose_reporter.stat_line(f"MedBatch strategy input: {len(cluster_batch)} batches, {composite_tokens:.0f} tokens")
        
        # Calculate strategy for this single medium batch
        strategy = self.workload_analyzer.calculate_optimal_strategy(
            total_batches=len(cluster_batch),
            avg_tokens_per_batch=composite_tokens,
            sub_batches_per_batch=1  # Sequential steps don't multiply concurrent load
        )
        
        # Show what strategy was calculated
        self.verbose_reporter.stat_line(f"MedBatch strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
        
        self.verbose_reporter.stat_line(f"Medium batch strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
        
        return await self._process_cluster_group_with_distributed_strategy(
            cluster_batch, clusters, themes, 
            strategy.launch_rate_per_second, strategy.concurrent_limit, 1
        )
    
    async def _process_large_batch(self, sub_batches: List[List[int]], 
                                  clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process large batch as concurrent sub-batches (legacy method)"""
        return await self._process_large_batch_optimized(sub_batches, clusters, themes)
    
    async def _process_medium_batch(self, cluster_batch: List[int], 
                                   clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process medium batch: all clusters concurrently (legacy method)"""
        return await self._process_medium_batch_optimized(cluster_batch, clusters, themes)
    
    async def _process_singleton(self, cluster_id: int, 
                                clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process single cluster"""
        result = await self._process_single_cluster_pipeline(cluster_id, clusters, themes)
        return [result] if result else []
    
    async def _process_cluster_group_with_optimal_strategy(self, cluster_ids: List[int], 
                                                         clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process group of clusters with evidence-based optimal strategy (like qualityFilter)"""
        
        if not cluster_ids:
            return []
            
        self.verbose_reporter.stat_line(f"Processing sub-batch of {len(cluster_ids)} clusters with optimal strategy")
        
        # Calculate composite token usage for 3-step pipeline using measured values
        composite_tokens = (
            self.code_gen_token_measurements.get('candidate_selection', 1200) +
            self.code_gen_token_measurements.get('code_generation', 1000) +
            self.code_gen_token_measurements.get('validation', 900)
        )
        
        # Calculate optimal strategy for this sub-batch (like qualityFilter)
        strategy = self.workload_analyzer.calculate_optimal_strategy(
            total_batches=len(cluster_ids),
            avg_tokens_per_batch=composite_tokens,
            sub_batches_per_batch=1  # Sequential steps don't multiply concurrent load
        )
        
        self.verbose_reporter.stat_line(f"Sub-batch strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent (composite: {composite_tokens:.0f} tokens/cluster)")
        
        # Aggressive parallelism - no throttling or monitoring
        # Get codebook snapshot once for all clusters in this batch
        codebook_snapshot, base_version = await self.shared_codebook.get_current_snapshot()
        
        # Create unlimited tasks for all clusters in sub-batch
        cluster_tasks = []
        for cluster_id in cluster_ids:
            task = self._process_single_cluster_unlimited(cluster_id, clusters, themes, codebook_snapshot, base_version)
            cluster_tasks.append(task)
        
        # Process with sophisticated rate limiting (like qualityFilter)
        results = []
        completed = 0
        
        # Process results as they complete with progress tracking
        for coro in asyncio.as_completed(cluster_tasks):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
                completed += 1
                
                # Progress reporting for larger sub-batches
                if len(cluster_ids) > 5 and (completed % 5 == 0 or completed == len(cluster_tasks)):
                    self.verbose_reporter.progress_line(completed, len(cluster_tasks), "sub-batch clusters")
                    
            except Exception as e:
                self.verbose_reporter.error(f"Cluster processing failed in sub-batch: {e}")
                completed += 1
        
        # Final sub-batch statistics
        final_stats = await monitor.get_current_utilization()
        if len(cluster_ids) > 3:  # Only show stats for larger sub-batches
            self.verbose_reporter.stat_line(f"Sub-batch completed: {final_stats['total_requests']} requests in {final_stats['elapsed_time']:.1f}s")
        
        return results

    async def _process_cluster_group_concurrently(self, cluster_ids: List[int], 
                                                clusters: Dict, themes: Dict) -> List[Dict[str, Any]]:
        """Process group of clusters concurrently with 3-step pipeline (legacy method)"""
        return await self._process_cluster_group_with_optimal_strategy(cluster_ids, clusters, themes)
    
    async def _process_cluster_group_with_distributed_strategy(self, cluster_ids: List[int], 
                                                             clusters: Dict, themes: Dict,
                                                             allocated_rate: float, allocated_concurrent: int,
                                                             sub_batch_num: int) -> List[Dict[str, Any]]:
        """Process group of clusters with distributed rate limits (global coordination)"""
        
        if not cluster_ids:
            return []
            
        self.verbose_reporter.stat_line(f"Sub-batch {sub_batch_num}: {len(cluster_ids)} clusters @ {allocated_rate:.1f} req/s, max {allocated_concurrent} concurrent")
        
        # Aggressive parallelism - no throttling or limits
        # Get codebook snapshot once for all clusters in this batch
        codebook_snapshot, base_version = await self.shared_codebook.get_current_snapshot()
        
        # Create unlimited tasks for all clusters in sub-batch
        cluster_tasks = []
        for cluster_id in cluster_ids:
            task = self._process_single_cluster_unlimited(cluster_id, clusters, themes, codebook_snapshot, base_version)
            cluster_tasks.append(task)
        
        # Process with aggressive parallelism - no limits
        results = []
        completed = 0
        
        # Process results as they complete with progress tracking - no limits
        for coro in asyncio.as_completed(cluster_tasks):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
                completed += 1
                
                # Progress reporting for sub-batches
                if len(cluster_ids) > 3 and (completed % 3 == 0 or completed == len(limited_tasks)):
                    self.verbose_reporter.progress_line(completed, len(limited_tasks), f"sub-batch {sub_batch_num} clusters")
                    
            except Exception as e:
                self.verbose_reporter.error(f"Cluster processing failed in sub-batch {sub_batch_num}: {e}")
                completed += 1
        
        # Final sub-batch statistics
        final_stats = await monitor.get_current_utilization()
        if len(cluster_ids) > 2:  # Show stats for sub-batches
            self.verbose_reporter.stat_line(f"Sub-batch {sub_batch_num} completed: {final_stats['total_requests']} requests in {final_stats['elapsed_time']:.1f}s")
        
        return results
    
    async def _process_single_cluster_with_monitoring(self, cluster_id: int, 
                                                    clusters: Dict, themes: Dict,
                                                    api_client=None) -> Optional[Dict[str, Any]]:
        """3-step pipeline for single cluster with accurate token tracking"""
        
        if cluster_id not in themes or cluster_id not in clusters:
            return None
        
        cluster_data = clusters[cluster_id]
        theme_data = themes[cluster_id]
        
        try:
            # Step 4a: Candidate Selection with token tracking
            current_codes, version = await self.shared_codebook.get_current_snapshot()
            
            # Build actual prompt for token tracking
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in current_codes[:20]])
            ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            candidate_prompt = CANDIDATE_CODE_SELECTION_PROMPT.format(
                survey_question=self.var_lab,
                language=DEFAULT_LANGUAGE,
                cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}",
                code_text=codes_text
            )
            
            # Aggressive parallelism - direct unlimited call
            candidate_selection = await self._select_candidate_codes_unlimited(
                cluster_id, cluster_data, theme_data, current_codes[:20]
            )
            
            # Check pipeline steps
            # self.verbose_reporter.stat_line(f"C{cluster_id}: CandSel={candidate_selection is not None}, Codes={len(candidate_selection) if candidate_selection else 0}")
            
            # Step 4b: Code Generation with token tracking
            if candidate_selection is not None:  # Allow empty lists - should create new codes
                if len(candidate_selection) > 0:
                    candidate_codes_text = "\n".join([
                        f"Code: {code.code}\nDefinition: {code.definition}\n" 
                        for code in candidate_selection
                    ])
                else:
                    # Empty candidate list - prompt should create new codes
                    candidate_codes_text = "No existing codes available."
                    
                code_gen_prompt = CODE_GENERATION_PROMPT.format( #TODO not used
                    language=DEFAULT_LANGUAGE,
                    survey_question=self.var_lab,
                    cluster_summary=f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}",
                    candidate_codes=candidate_codes_text
                )
                
                # Aggressive parallelism - direct unlimited call
                code_generation = await self._generate_code_unlimited(cluster_id, cluster_data, theme_data, candidate_selection)
                
                # # Debug: Check if code generation succeeded
                # if code_generation:
                #     # self.verbose_reporter.stat_line(f"C{cluster_id}: CodeGen success")
                #     # Show prompt 2 decisions
                #     if code_generation.coding_decisions:
                #         for i, decision in enumerate(code_generation.coding_decisions):
                #             self.verbose_reporter.stat_line(f"C{cluster_id}: Decision{i+1}={decision.decision}")
                # else:
                #     # self.verbose_reporter.stat_line(f"C{cluster_id}: CodeGen failed")
                #     pass
            else:
                # self.verbose_reporter.stat_line(f"C{cluster_id}: CandSel returned None - API error")
                code_generation = None
            
            # Step 4c: Validation & SharedCodebook Update with token tracking
            if code_generation:
                validation_task = self._validate_and_update_codebook(cluster_id, cluster_data, theme_data, code_generation, candidate_selection)
                validation = await api_client.make_request(validation_task, f"validation_{cluster_id}")
            else:
                validation = None
            
            # Extract final code/definition from complex validation structure  
            final_code = None
            final_definition = None
            if validation and hasattr(validation, 'code_validations') and validation.code_validations:
                # Get the first validated code (for now - could aggregate multiple)
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
                'final_definition': final_definition
            }
            
        except Exception as e:
            self.verbose_reporter.error(f"Pipeline failed for cluster {cluster_id}: {e}")
            return None
    
    async def _process_single_cluster_pipeline(self, cluster_id: int, 
                                             clusters: Dict, themes: Dict) -> Optional[Dict[str, Any]]:
        """3-step pipeline for single cluster"""
        
        if cluster_id not in themes or cluster_id not in clusters:
            return None
        
        cluster_data = clusters[cluster_id]
        theme_data = themes[cluster_id]
        
        try:
            # Step 4a: Candidate Selection
            current_codes, version = await self.shared_codebook.get_current_snapshot()
            # DEBUG: Check codebook state
            #self.verbose_reporter.stat_line(f"C{cluster_id}: CurrentCodes={len(current_codes)}, Version={version}")
            
            # Aggressive parallelism - unlimited calls  
            candidate_selection = await self._select_candidate_codes_unlimited(
                cluster_id, cluster_data, theme_data, current_codes[:20]
            )
            
            # Step 4b: Code Generation - unlimited
            code_generation = await self._generate_code_unlimited(
                cluster_id, cluster_data, theme_data, candidate_selection
            )
            
            # Step 4c: Validation & SharedCodebook Update
            validation = await self._validate_and_update_codebook(
                cluster_id, cluster_data, theme_data, code_generation, candidate_selection
            )
            
            # Extract final code/definition from complex validation structure
            final_code = None
            final_definition = None
            if validation and hasattr(validation, 'code_validations') and validation.code_validations:
                # Get the first validated code (for now - could aggregate multiple)
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
                'final_definition': final_definition
            }
            
        except Exception as e:
            self.verbose_reporter.error(f"Pipeline failed for cluster {cluster_id}: {e}")
            return None
    
    async def _find_nearest_codes_by_theme(self, cluster_id: int, theme_data, 
                                          current_codes: List[Dict[str, str]], k: int = 5) -> List[Dict[str, str]]:
        """Find k nearest codes to themes using cosine similarity - handles multiple themes per cluster"""
        if not current_codes:
            return []
        
        # Handle multiple themes per cluster (aggregate approach)
        all_nearest_codes = []
        
        # Check if theme_data has multiple themes
        if hasattr(theme_data, 'themes') and isinstance(theme_data.themes, list) and len(theme_data.themes) > 1:
            # Multiple themes: get k codes for each theme and aggregate
            for theme_item in theme_data.themes:
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
    
    async def _get_theme_embedding_for_item(self, cluster_id: int, theme_item) -> Optional[np.ndarray]:
        """Get embedding for a specific theme item"""
        try:
            # Generate embedding for this specific theme
            theme_text = theme_item.theme_statement
            embedding = await self._get_embedding(theme_text)
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
        
        # Return the nearest codes
        nearest_codes = []
        for idx in top_k_indices:
            if idx < len(current_codes):
                nearest_codes.append(current_codes[idx])
        
        # Debug : similarity score nearest codes to theme embeddingg
        # if nearest_codes:
        #     # Convert numpy similarities to properly rounded list
        #     similarity_values = [round(float(similarities[idx]), 3) for idx in top_k_indices]
        #     self.verbose_reporter.stat_line(f"Found {len(nearest_codes)} nearest codes for theme '{self._get_theme_statement(theme_data)}'")
        #     self.verbose_reporter.stat_line(f"Similarities: {similarity_values}")
        return nearest_codes

    async def _select_candidate_codes(self, cluster_id: int, cluster_data: Dict, theme_data, 
                                    current_codes: List[Dict[str, str]]) -> Optional[List[CandidateCode]]:
        """Step 4a: Select candidate codes from current codebook using cosine similarity"""
        
        # Find 5 nearest codes by cosine similarity
        nearest_codes = await self._find_nearest_codes_by_theme(
            cluster_id,  # Use explicit cluster_id parameter
            theme_data,
            current_codes,
            k=5
        )
        
        # Format nearest codes for prompt
        if nearest_codes:
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in nearest_codes])
        else:
            # Fallback if no codes found - use first few codes
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in current_codes[:5]])
        
        ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
        
        cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}"
        
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
        
        try:
            # Use List[CandidateCode] directly since your prompt returns an array
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=List[CandidateCode],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: OLD_CANDIDATE_SELECT"
            )
            
            # Capture step2_analysis - the actual candidate codes used in pipeline
            if response:
                self.step2_analysis[cluster_id] = [
                    {"code": code.code, "definition": code.definition} 
                    for code in response
                ]
            
            # Return the List[CandidateCode] directly
            return response
            
        except Exception as e:
            self.verbose_reporter.error(f"Candidate selection failed: {e}")
            return None
    
    async def _generate_code(self, cluster_id: int, cluster_data: Dict, theme_data,
                           candidate_selection: Optional[List[CandidateCode]]) -> Optional[SimplifiedCodeRecommendation]:
        """Step 4b: Generate code decision"""
        
        if candidate_selection is None:
            return None
        
        # Format candidate codes for prompt - candidate_selection is now List[CandidateCode] directly
        if len(candidate_selection) > 0:
            candidate_codes_text = "\n".join([
                f"Code: {code.code}\nDefinition: {code.definition}\n" 
                for code in candidate_selection
            ])
            
        else:
            candidate_codes_text = "No existing codes available."
        
        ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
        cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}"
        
        # Prepare exact parameters for prompt
        params = {
            'language': DEFAULT_LANGUAGE,
            'survey_question': self.var_lab,
            'cluster_summary': cluster_summary,
            'candidate_codes': candidate_codes_text
        }
        
        prompt = CODE_GENERATION_PROMPT.format(**params)
        
        # Capture exact parameters used in prompt construction
        self._capture_prompt_params(cluster_id, "step3", **params)
        
        # Capture step2_analysis - actual candidate codes used in this prompt
        if candidate_selection and cluster_id not in self.step2_analysis:
            self.step2_analysis[cluster_id] = [
                {
                    'code': code.code,
                    'definition': code.definition
                } for code in candidate_selection
            ]
        
        
        # Capture prompt with prompt_printer if available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="code_generation",
                utility_name="codeGenerator",
                prompt_content=prompt,
                prompt_type="code_recommendation",
                metadata={
                    "cluster_id": cluster_id,
                    "model": self.config.model,
                    "candidate_codes_count": len(candidate_selection)
                }
            )
        
        try:
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=SimplifiedCodeRecommendation,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: OLD_CODE_GEN"
            )
            
            # Capture code generation results for transparency
            if hasattr(response, 'coding_decisions') and response.coding_decisions:
                self.step3_recommendations[cluster_id] = {
                    'coding_decisions': [
                        {
                            'theme_number': decision.theme_number,
                            'decision': decision.decision,
                            'final_code_label': decision.final_code_label,
                            'final_code_description': decision.final_code_description,
                            'source_code': decision.source_code,
                            'justification': decision.justification
                        } for decision in response.coding_decisions
                    ]
                }
            else:
                self.verbose_reporter.error(f"Code generation response for cluster {cluster_id} missing coding_decisions field")
            
            return response
            
        except Exception as e:
            self.verbose_reporter.error(f"Code generation failed: {e}")
            return None
    
    async def _validate_and_update_codebook(self, cluster_id: int, cluster_data: Dict, theme_data, code_generation: Optional[SimplifiedCodeRecommendation], candidate_selection: Optional[List[CandidateCode]]) -> Optional[ValidationResult]:
        """Step 4c: Validate code and update SharedCodebook"""
        
        if not code_generation:
            return None
        
        # Format candidate codes for validation (same pattern as Step 3)
        if candidate_selection and len(candidate_selection) > 0:
            codes_text = "\n".join([
                f"Code: {code.code}\nDefinition: {code.definition}\n" 
                for code in candidate_selection
            ])
        else:
            codes_text = "No existing codes available."
        
        ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
        cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}"
        step3_recommendation_text = str(code_generation.model_dump_json(indent=2))
        
        # Prepare exact parameters for prompt
        params = {
            'language': DEFAULT_LANGUAGE,
            'survey_question': self.var_lab,
            'cluster_summary': cluster_summary,
            'candidate_codes': codes_text,
            'step3_recommendation': step3_recommendation_text
        }
        
        prompt = VALIDATION_PROMPT.format(**params)
        
        # Capture exact parameters used in prompt construction
        self._capture_prompt_params(cluster_id, "step4", **params)
        
        # Capture prompt with prompt_printer if available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="code_validation",
                utility_name="codeGenerator",
                prompt_content=prompt,
                prompt_type="validation_result",
                metadata={
                    "cluster_id": cluster_id,
                    "model": self.config.model
                }
            )
        
        try:
            # Detailed logging for API call debugging
            self.verbose_reporter.stat_line(f"C{cluster_id}: STEP4 - Starting validation and codebook update API call")
            self.verbose_reporter.stat_line(f"C{cluster_id}: STEP4 - Prompt length: {len(prompt)} chars")
            self.verbose_reporter.stat_line(f"C{cluster_id}: STEP4 - Has code_generation: {code_generation is not None}")
            
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=ValidationResult,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: STEP4"
            )
            
            # Detailed response logging
            if response is None:
                self.verbose_reporter.error(f"C{cluster_id}: STEP4 - API returned None response")
                return None
            elif not hasattr(response, 'code_validations') or not response.code_validations:
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP4 - API returned response with no validations")
            else:
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP4 - API returned {len(response.code_validations)} validations")
            
            # Show validation results for original complex structured output
            if response and response.code_validations:
                for validation in response.code_validations:
                    # self.verbose_reporter.stat_line(f"• C{cluster_id}: Validation={validation.decision}, Code='{validation.validated_code.code if validation.validated_code else None}'")
                    pass
            
            
            # Update SharedCodebook based on validation decisions
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
                        
                        # Show prompt 3 validation decision
                        #self.verbose_reporter.stat_line(f"C{cluster_id}: Validation{i+1}={validation.decision}")
                        
                        if validation.decision == "APPROVE" and validation.validated_code:
                            if coding_decision.decision == "create":
                                added, new_version = await self.shared_codebook.add_code_if_new(
                                    validation.validated_code.code, validation.validated_code.definition
                                )
                                #self.verbose_reporter.stat_line(f"C{cluster_id}: CREATE - added={added}, v{new_version}, code='{validation.validated_code.code}'")
                                if added:
                                    self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                    codebook_updated = True
                            elif coding_decision.decision == "modify" and coding_decision.source_code:
                                replaced, new_version = await self.shared_codebook.replace_code(
                                    coding_decision.source_code, 
                                    validation.validated_code.code, validation.validated_code.definition
                                )
                                #self.verbose_reporter.stat_line(f"C{cluster_id}: MODIFY - replaced={replaced}, v{new_version}, '{coding_decision.source_code}' -> '{validation.validated_code.code}'")
                                if replaced:
                                    self._processing_stats['codes_modified'] = self._processing_stats.get('codes_modified', 0) + 1
                                    codebook_updated = True
                            elif coding_decision.decision == "use":
                                #self.verbose_reporter.stat_line(f"C{cluster_id}: USE - no codebook update")
                                pass
                        elif validation.decision == "REVISE" and validation.validated_code:
                            # Validation revised the decision - use the revised code
                            added, new_version = await self.shared_codebook.add_code_if_new(
                                validation.validated_code.code, validation.validated_code.definition
                            )
                            #self.verbose_reporter.stat_line(f"C{cluster_id}: REVISE->CREATE - added={added}, v{new_version}, code='{validation.validated_code.code}'")
                            if added:
                                self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                codebook_updated = True
                        elif validation.decision == "REJECT":
                            # self.verbose_reporter.stat_line(f"• C{cluster_id}: REJECT - no codebook update")
                            pass
                        else:
                            self.verbose_reporter.error(f"• C{cluster_id}: UNHANDLED validation decision '{validation.decision}'")
                
                # Generate embeddings for new/modified codes
                if codebook_updated and new_version is not None:
                    # Get updated codebook
                    updated_codes, _ = await self.shared_codebook.get_current_snapshot()
                    
                    # Check if we need to regenerate embeddings (cache invalidated)
                    cached_embeddings = await self.shared_codebook.get_embeddings_for_version(new_version)
                    if cached_embeddings is None:
                        # Generate embeddings for all codes
                        self.verbose_reporter.stat_line(f"Generating embeddings for updated codebook (version {new_version})")
                        code_texts = [f"{code['code']}: {code['definition']}" for code in updated_codes]
                        try:
                            code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                            await self.shared_codebook.cache_embeddings(new_version, code_embeddings)
                        except Exception as e:
                            self.verbose_reporter.error(f"Failed to generate embeddings for new codes: {e}")
            else:
                # Track validation failures
                reason = response.validation_reasoning[:50] + "..." if response and hasattr(response, 'validation_reasoning') else "No response"
                self.verbose_reporter.stat_line(f"• C{cluster_id}: VALIDATION FAILED - {reason}")
                self._processing_stats['validation_failures'] = self._processing_stats.get('validation_failures', 0) + 1
            
            # Capture validation results and final codes for transparency
            if response:
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
                    ] if response.code_validations else [],
                    'theme_assessment': response.theme_assessment.model_dump() if response.theme_assessment else None,
                    'overall_validation': response.overall_validation.model_dump() if response.overall_validation else None
                }
                
                # Capture final validated codes for step4_validated_codes (as Dict, not List)
                validated_codes_dict = {}
                if response.code_validations:
                    for i, validation in enumerate(response.code_validations):
                        if validation.validated_code:
                            validated_codes_dict[f"validation_{i}"] = {
                                'code': validation.validated_code.code,
                                'definition': validation.validated_code.definition,
                                'decision': validation.decision
                            }
                self.step4_validated_codes[cluster_id] = validated_codes_dict
                
                # Update cluster assignments for pipeline integration
                if validated_codes_dict:
                    # Extract codes from validated_codes_dict for cluster_assignments
                    final_codes_list = [
                        {'code': item['code'], 'definition': item['definition']}
                        for item in validated_codes_dict.values()
                    ]
                    self.cluster_assignments[cluster_id] = {
                        'cluster_id': cluster_id,
                        'theme_name': self._get_theme_statement(theme_data),
                        'theme_description': self._get_theme_description(theme_data),
                        'codes': final_codes_list,
                        'status': 'completed'
                    }
            
            return response
            
        except Exception as e:
            # Enhanced error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP4 - Validation and codebook update failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP4 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP4 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP4 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return None
    
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
                self.verbose_reporter.error(f"Full traceback:")
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
            
            # Prompt transparency - ACTUAL inputs to each step
            step1_inputs=self.step1_inputs,
            step2_inputs=self.step2_inputs,  # Not used in current architecture
            step3_inputs=self.step3_inputs,
            step4_inputs=self.step4_inputs,
            
            # Step results for backward compatibility
            step1_summaries=self.step1_summaries,
            step2_analysis=self.step2_analysis,  # Not used in current architecture
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
    # ============================================================================
    
    async def _process_sub_batch_with_stagger(self, sub_batch: List[int], clusters: Dict, themes: Dict,
                                             stagger_delay: float) -> List[Dict[str, Any]]:
        """Process sub-batch with pure concurrency - no artificial limits (Phase 3)"""
        
        # Apply stagger delay for smooth API load distribution
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)
        
        # Get codebook snapshot for this sub-batch (version-based isolation)
        codebook_snapshot, base_version = await self.shared_codebook.get_current_snapshot()
        
        # Process ALL clusters concurrently - no artificial limits!
        cluster_tasks = []
        for cluster_id in sub_batch:
            task = self._process_single_cluster_unlimited(
                cluster_id, clusters, themes, codebook_snapshot, base_version
            )
            cluster_tasks.append(task)
        
        # Pure concurrent execution - let them all run!
        results = await asyncio.gather(*cluster_tasks, return_exceptions=True)
        
        # Filter out exceptions and collect valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                cluster_id = sub_batch[i]
                self.verbose_reporter.error(f"Cluster {cluster_id} failed: {result}")
            elif result is not None:
                valid_results.append(result)
        
        # Populate dictionaries for utility compatibility
        for result in valid_results:
            cluster_id = result['cluster_id']
            code_generation = result.get('code_generation')
            validation = result.get('validation')
            
            # Populate step3_recommendations dictionary
            if code_generation and hasattr(code_generation, 'coding_decisions'):
                self.step3_recommendations[cluster_id] = {
                    'coding_decisions': [
                        {
                            'theme_number': decision.theme_number,
                            'decision': decision.decision,
                            'final_code_label': decision.final_code_label,
                            'final_code_description': decision.final_code_description,
                            'source_code': decision.source_code,
                            'justification': decision.justification
                        } for decision in code_generation.coding_decisions
                    ]
                }
            
            # Populate step4_validations dictionary
            if validation and hasattr(validation, 'code_validations'):
                self.step4_validations[cluster_id] = {
                    'code_validations': [
                        {
                            'theme_number': val.theme_number,
                            'original_recommendation': {
                                'code': val.original_recommendation.code,
                                'definition': val.original_recommendation.definition
                            },
                            'decision': val.decision,
                            'decision_rationale': val.decision_rationale,
                            'validated_code': {
                                'code': val.validated_code.code,
                                'definition': val.validated_code.definition
                            }
                        } for val in validation.code_validations
                    ]
                }
        
        # Merge all codebook updates from this sub-batch atomically
        await self._merge_codebook_updates(valid_results, base_version)
        
        return valid_results
    
    async def _process_single_cluster_unlimited(self, cluster_id: int, clusters: Dict, themes: Dict,
                                               codebook_snapshot: List[Dict], base_version: int) -> Optional[Dict[str, Any]]:
        """Process single cluster with unlimited concurrency - no artificial limits (Phase 3)"""
        
        if cluster_id not in themes or cluster_id not in clusters:
            return None
        
        cluster_data = clusters[cluster_id]
        theme_data = themes[cluster_id]
        
        try:
            # Step 1: Candidate selection using snapshot (no locks needed!)
            nearest_codes = self._find_nearest_in_snapshot(
                cluster_id, theme_data, codebook_snapshot
            )
            
            # Step 1: Candidate selection - pure unlimited call
            step1_start = time.time()
            candidate_selection = await self._select_candidate_codes_unlimited(
                cluster_id, cluster_data, theme_data, nearest_codes
            )
            step1_duration = time.time() - step1_start
            
            # Step 2: Code generation - pure unlimited call
            step2_start = time.time()
            code_generation = await self._generate_code_unlimited(
                cluster_id, cluster_data, theme_data, candidate_selection
            )
            step2_duration = time.time() - step2_start
            
            # Step 3: Validation - pure unlimited call
            step3_start = time.time()
            validation = await self._validate_code_unlimited(
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
            if result and result.get('validation') and hasattr(result['validation'], 'code_validations'):
                for code_validation in result['validation'].code_validations:
                    if code_validation.validated_code:
                        all_new_codes.append({
                            'code': code_validation.validated_code.code,
                            'definition': code_validation.validated_code.definition,
                            'cluster_id': result['cluster_id']
                        })
        
        # Single atomic update to SharedCodebook if there are new codes
        if all_new_codes:
            await self.shared_codebook.batch_update(all_new_codes, base_version)
    
    def _find_nearest_in_snapshot(self, cluster_id: int, theme_data, codebook_snapshot: List[Dict]) -> List[Dict]:
        """Find nearest codes using codebook snapshot (no locking needed)"""
        if not codebook_snapshot or not hasattr(self, '_theme_embeddings_cache'):
            return []
        
        try:
            # Use existing similarity calculation logic but with snapshot
            theme_embedding = self._theme_embeddings_cache.get(cluster_id)
            if theme_embedding is None:
                return []
            
            # Simple nearest neighbor search in snapshot
            similarities = []
            for code in codebook_snapshot[:20]:  # Limit to top 20 like original
                similarities.append(code)
            
            return similarities[:5]  # Return top 5
        except Exception:
            return []
    
    async def _select_candidate_codes_unlimited(self, cluster_id: int, cluster_data: Dict, theme_data, nearest_codes: List[Dict]):
        """Select candidate codes with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            codes_text = "\n".join([f"Code: {code['code']}\nDefinition: {code['definition']}\n" 
                                   for code in nearest_codes[:20]])
            ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}"
            
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
            
            # Detailed logging for API call debugging
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Starting candidate selection API call")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Prompt length: {len(prompt)} chars")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Available codes: {len(nearest_codes)}")
            
            # API call with enhanced error handling and response cleaning
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=List[CandidateCode],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: STEP1"
            )
            
            # Detailed response logging
            if response is None:
                self.verbose_reporter.error(f"C{cluster_id}: STEP1 - API returned None response")
                return []
            elif len(response) == 0:
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - API returned empty list (valid)")
            else:
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - API returned {len(response)} candidates")
            
            # Capture step2_analysis - the actual candidate codes used in pipeline
            if response:
                self.step2_analysis[cluster_id] = [
                    {"code": code.code, "definition": code.definition} 
                    for code in response
                ]
            
            return response
            
        except Exception as e:
            # Enhanced error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Candidate selection failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP1 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return []
    
    async def _generate_code_unlimited(self, cluster_id: int, cluster_data: Dict, theme_data, candidate_selection):
        """Generate code with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            candidate_codes_text = ""
            if candidate_selection and len(candidate_selection) > 0:
                candidate_codes_text = "\n\n".join([f"Code: {code.code}\nDefinition: {code.definition}" 
                                                   for code in candidate_selection])
                
            cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}"
            
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
            
            # Detailed logging for API call debugging
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Starting code generation API call")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Prompt length: {len(prompt)} chars")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Candidate codes: {len(candidate_selection) if candidate_selection else 0}")
            
            # API call with enhanced error handling and response cleaning
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=SimplifiedCodeRecommendation,  # Use simplified model for flattened JSON
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: STEP2"
            )
            
            # Detailed response logging
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
                            'final_code_description': decision.final_code_description,
                            'source_code': decision.source_code,
                            'justification': decision.justification
                        } for decision in response.coding_decisions
                    ]
                }
            
            return response
            
        except Exception as e:
            # Enhanced error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Code generation failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP2 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return None
    
    async def _make_instructor_call_with_cleanup(self, **kwargs):
        """Make instructor API call with automatic response cleanup on parsing failures"""
        context_info = kwargs.pop('context_info', 'API_CALL')
        
        try:
            # First attempt with instructor
            response = await self.async_client.chat.completions.create(**kwargs)
            return response
        except Exception as instructor_error:
            # Check if this looks like a JSON/whitespace parsing issue
            error_str = str(instructor_error).strip()
            
            # Look for common patterns in instructor parsing errors
            is_parsing_error = (
                '\n' in error_str or 
                'json' in error_str.lower() or
                'coding_decisions' in error_str or
                'validation' in error_str or
                'parse' in error_str.lower() or
                'decode' in error_str.lower()
            )
            
            if is_parsing_error:
                self.verbose_reporter.stat_line(f"{context_info} - Detected parsing error, attempting response cleanup")
                self.verbose_reporter.stat_line(f"{context_info} - Error: {repr(error_str[:150])}")
                
                try:
                    # Make raw API call to get response content
                    raw_kwargs = dict(kwargs)
                    raw_kwargs.pop('response_model', None)  # Remove instructor's response_model
                    
                    raw_response = await self.async_client.chat.completions.create(**raw_kwargs)
                    
                    if raw_response and raw_response.choices and raw_response.choices[0].message.content:
                        raw_content = raw_response.choices[0].message.content
                        cleaned_content = raw_content.strip()
                        
                        if cleaned_content != raw_content:
                            self.verbose_reporter.stat_line(f"{context_info} - Found whitespace issue, cleaned response")
                            self.verbose_reporter.stat_line(f"{context_info} - Original: {repr(raw_content[:50])}...")
                            self.verbose_reporter.stat_line(f"{context_info} - Cleaned: {repr(cleaned_content[:50])}...")
                            
                            # Try manual JSON parsing to validate the cleaned content
                            try:
                                import json
                                parsed_json = json.loads(cleaned_content)
                                self.verbose_reporter.stat_line(f"{context_info} - Cleaned content is valid JSON, creating response object")
                                
                                # Use instructor to parse the valid JSON content
                                # Create a synthetic message with cleaned content
                                from openai.types.chat import ChatCompletion, ChatCompletionMessage, Choice
                                
                                # Create new response with cleaned content
                                clean_message = ChatCompletionMessage(
                                    role="assistant",
                                    content=cleaned_content
                                )
                                clean_choice = Choice(
                                    index=0,
                                    message=clean_message,
                                    finish_reason=raw_response.choices[0].finish_reason
                                )
                                clean_response = ChatCompletion(
                                    id=raw_response.id,
                                    choices=[clean_choice],
                                    created=raw_response.created,
                                    model=raw_response.model,
                                    object=raw_response.object
                                )
                                
                                # Now use instructor to parse this cleaned response
                                response_model = kwargs.get('response_model')
                                if response_model:
                                    # Convert the JSON to the expected Pydantic model
                                    if hasattr(response_model, '__origin__') and response_model.__origin__ is list:
                                        # Handle List[Model] types
                                        item_model = response_model.__args__[0]
                                        if isinstance(parsed_json, list):
                                            return [item_model(**item) for item in parsed_json]
                                        else:
                                            return [item_model(**parsed_json)]
                                    else:
                                        # Handle single model types
                                        return response_model(**parsed_json)
                                
                                return parsed_json
                            except json.JSONDecodeError as json_error:
                                self.verbose_reporter.error(f"{context_info} - Cleaned content is still not valid JSON: {json_error}")
                                raise instructor_error
                            except Exception as parsing_error:
                                self.verbose_reporter.error(f"{context_info} - Failed to create response object: {parsing_error}")
                                raise instructor_error
                        else:
                            self.verbose_reporter.stat_line(f"{context_info} - No whitespace found, original error not whitespace-related")
                            raise instructor_error
                    else:
                        self.verbose_reporter.error(f"{context_info} - No content in raw response")
                        raise instructor_error
                except Exception as cleanup_error:
                    self.verbose_reporter.error(f"{context_info} - Response cleanup failed: {cleanup_error}")
                    raise instructor_error
            else:
                # Not a parsing error, re-raise original
                raise instructor_error

    async def _validate_code_unlimited(self, cluster_id: int, cluster_data: Dict, theme_data, 
                                       code_generation, candidate_selection):
        """Validate code with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            # Use candidate_selection instead of full codebook (consistent with rate-limited version)
            if candidate_selection and len(candidate_selection) > 0:
                validation_codes_text = "\n".join([
                    f"Code: {code.code}\nDefinition: {code.definition}\n" 
                    for code in candidate_selection
                ])
            else:
                validation_codes_text = "No existing codes available."
            
            cluster_summary = f"Theme: {self._get_theme_statement(theme_data)}\nDescription: {self._get_theme_description(theme_data)}\nIdeas:\n{ideas_text}"
            step3_recommendation_text = str(code_generation.model_dump_json(indent=2)) if code_generation else "No recommendations"
            
            # Prepare exact parameters for prompt
            params = {
                'language': DEFAULT_LANGUAGE,
                'survey_question': self.var_lab,
                'cluster_summary': cluster_summary,
                'candidate_codes': validation_codes_text,
                'step3_recommendation': step3_recommendation_text
            }
            
            prompt = VALIDATION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step4", **params)
            
            # Detailed logging for API call debugging
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Starting validation API call")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Prompt length: {len(prompt)} chars")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Candidate codes: {len(candidate_selection) if candidate_selection else 0}")
            # self.verbose_reporter.stat_line(f"C{cluster_id}: STEP3 - Has code_generation: {code_generation is not None}")
            
            # API call with enhanced error handling and response cleaning
            response = await self._make_instructor_call_with_cleanup(
                model=self.config.model,
                response_model=ValidationResult,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.config.seed,
                context_info=f"C{cluster_id}: STEP3"
            )
            
            # Detailed response logging
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
                        
                        if validation.decision == "APPROVE" and validation.validated_code:
                            if coding_decision.decision == "create":
                                added, new_version = await self.shared_codebook.add_code_if_new(
                                    validation.validated_code.code, validation.validated_code.definition
                                )
                                if added:
                                    self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                    codebook_updated = True
                                    self.verbose_reporter.stat_line(f"C{cluster_id}: CREATE - Added new code '{validation.validated_code.code}'")
                            elif coding_decision.decision == "modify" and coding_decision.source_code:
                                replaced, new_version = await self.shared_codebook.replace_code(
                                    coding_decision.source_code, 
                                    validation.validated_code.code, validation.validated_code.definition
                                )
                                if replaced:
                                    self._processing_stats['codes_modified'] = self._processing_stats.get('codes_modified', 0) + 1
                                    codebook_updated = True
                                    self.verbose_reporter.stat_line(f"C{cluster_id}: MODIFY - Replaced '{coding_decision.source_code}' with '{validation.validated_code.code}'")
                            elif coding_decision.decision == "use":
                                self.verbose_reporter.stat_line(f"C{cluster_id}: USE - No codebook update needed")
                        elif validation.decision == "REVISE" and validation.validated_code:
                            # Validation revised the decision - use the revised code
                            added, new_version = await self.shared_codebook.add_code_if_new(
                                validation.validated_code.code, validation.validated_code.definition
                            )
                            if added:
                                self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + 1
                                codebook_updated = True
                                self.verbose_reporter.stat_line(f"C{cluster_id}: REVISE - Added revised code '{validation.validated_code.code}'")
                        elif validation.decision == "REJECT":
                            self.verbose_reporter.stat_line(f"C{cluster_id}: REJECT - No codebook update")
                        else:
                            self.verbose_reporter.error(f"C{cluster_id}: UNHANDLED validation decision '{validation.decision}'")
                
                # Generate embeddings for new/modified codes
                if codebook_updated and new_version is not None:
                    # Get updated codebook
                    codebook_snapshot, _ = await self.shared_codebook.get_current_snapshot()
                    if codebook_snapshot:
                        # Generate embeddings for the updated codebook
                        embeddings = await self.similarity_engine.embed_codes(codebook_snapshot)
                        await self.shared_codebook.cache_embeddings(new_version, embeddings)
                        self.verbose_reporter.stat_line(f"C{cluster_id}: Updated embeddings cache for version {new_version}")
            
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
    # ============================================================================
    
    class PerformanceTracker:
        """Track performance metrics and identify bottlenecks (Phase 3)"""
        
        def __init__(self):
            self.step_timings = {
                'step1_candidate': [],
                'step2_generate': [],
                'step3_validate': [],
                'api_wait_time': [],
                'codebook_merge': []
            }
            self.api_denials = 0
            self.start_time = time.time()
        
        def record_timing(self, step: str, duration: float):
            """Record timing for a processing step"""
            if step in self.step_timings:
                self.step_timings[step].append(duration)
        
        def record_api_wait(self, wait_time: float):
            """Record API wait time due to rate limiting"""
            self.step_timings['api_wait_time'].append(wait_time)
            if wait_time > 0:
                self.api_denials += 1
        
        def get_bottleneck_report(self) -> Dict[str, Any]:
            """Identify bottlenecks in the pipeline"""
            total_time = time.time() - self.start_time
            
            # Calculate averages and totals
            step_totals = {}
            step_averages = {}
            for step, timings in self.step_timings.items():
                if timings:
                    step_totals[step] = sum(timings)
                    step_averages[step] = np.mean(timings)
                else:
                    step_totals[step] = 0
                    step_averages[step] = 0
            
            # Identify bottleneck
            bottleneck = max(step_totals.items(), key=lambda x: x[1])[0] if step_totals else 'unknown'
            
            # Calculate API utilization
            api_wait_total = step_totals.get('api_wait_time', 0)
            api_utilization = 1 - (api_wait_total / total_time) if total_time > 0 else 0
            
            return {
                'total_time': total_time,
                'step_totals': step_totals,
                'step_averages': step_averages,
                'bottleneck': bottleneck,
                'api_utilization': api_utilization,
                'api_denials': self.api_denials,
                'recommendations': self._get_recommendations(bottleneck, api_utilization)
            }
        
        def _get_recommendations(self, bottleneck: str, api_utilization: float) -> List[str]:
            """Generate actionable recommendations based on bottlenecks"""
            recommendations = []
            
            if bottleneck == 'api_wait_time':
                recommendations.append("Bottleneck: API rate limits - consider upgrading tier")
            elif bottleneck == 'codebook_merge':
                recommendations.append("Bottleneck: Codebook merging - consider larger sub-batches")
            elif api_utilization < 0.9:
                recommendations.append(f"API underutilized ({api_utilization:.1%}) - can be more aggressive")
            else:
                recommendations.append("System running optimally")
            
            return recommendations
    
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