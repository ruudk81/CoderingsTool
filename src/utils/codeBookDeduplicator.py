import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import statistics
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

import instructor
from openai import AsyncOpenAI, RateLimitError
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import umap

# === MODELS ========================================================================================================
import models
from models import MergeDecision, DeduplicationResult

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, DeduplicationConfig, DEFAULT_DEDUPLICATION_CONFIG, get_openai_rate_limits
from prompts import DEDUPLICATION_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats

try:
    import nest_asyncio #for Spyder
    nest_asyncio.apply()
except ImportError:
    pass

async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
logger = logging.getLogger(__name__)

@dataclass
class OptimalStrategy:
    """Evidence-based optimal processing strategy for deduplication"""
    target_time_seconds: float
    launch_rate_per_second: float
    concurrent_limit: int
    bottleneck_type: str
    total_requests: int
    total_tokens: int
    safety_factor: float
    sub_batch_size: int


class WorkloadAnalyzer:
    """Analyzes workload and calculates optimal processing strategy for deduplication"""
    
    def __init__(self, model_name: str, encoding):
        self.model_name = model_name
        self.encoding = encoding
    
    def measure_token_usage(self, sample_batches: List[List[models.Codebook]], var_lab: str) -> float:
        """Measure actual token usage from real deduplication batches"""
        if not sample_batches:
            return 2000  # Conservative fallback for deduplication
        
        token_counts = []
        for batch in sample_batches[:3]:  # Sample first 3 batches
            codes_text = self._format_codes_batch(batch)
            prompt = DEDUPLICATION_PROMPT.format(
                language=DEFAULT_LANGUAGE,
                survey_question=var_lab,
                codes_batch=codes_text
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            # Estimate completion tokens (typically 30-40% of prompt for deduplication)
            completion_tokens = int(prompt_tokens * 0.35)
            total_tokens = prompt_tokens + completion_tokens
            token_counts.append(total_tokens)
        
        return statistics.mean(token_counts) if token_counts else 2000
    
    def _format_codes_batch(self, batch: List[models.Codebook]) -> str:
        """Format codes batch for prompt"""
        formatted_codes = []
        for i, code in enumerate(batch, 1):
            formatted_codes.append(f"{i}. {code.code}: {code.definition}")
        return "\n".join(formatted_codes)
    
    def calculate_optimal_strategy(self, total_batches: int, avg_tokens_per_batch: float) -> OptimalStrategy:
        """Calculate mathematically optimal processing strategy"""
        # Get API limits from config
        rate_limits = get_openai_rate_limits(self.model_name)
        
        # Calculate total resource requirements
        total_requests = total_batches
        total_tokens = total_batches * avg_tokens_per_batch
        
        # Calculate minimum time based on constraints
        time_by_requests = total_requests / rate_limits.requests_per_minute * 60
        time_by_tokens = total_tokens / rate_limits.tokens_per_minute * 60
        
        # Find bottleneck and minimum time
        if time_by_requests > time_by_tokens:
            bottleneck = "requests"
            min_time = time_by_requests
        else:
            bottleneck = "tokens"
            min_time = time_by_tokens
        
        # Add safety factor
        safety_factor = 1.2  # 20% safety margin
        target_time = min_time * safety_factor
        
        # Calculate optimal launch rate (requests per second)
        launch_rate = total_requests / target_time
        
        # Calculate concurrent limit (conservative)
        concurrent_limit = min(10, max(3, int(launch_rate * 2)))
        
        return OptimalStrategy(
            target_time_seconds=target_time,
            launch_rate_per_second=launch_rate,
            concurrent_limit=concurrent_limit,
            bottleneck_type=bottleneck,
            total_requests=total_requests,
            total_tokens=total_tokens,
            safety_factor=safety_factor,
            sub_batch_size=1  # Deduplication processes one batch at a time
        )


class SlidingWindowMonitor:
    """Monitor API usage with sliding windows for precise rate limiting"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests_window = deque()
        self.tokens_window = deque()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens = 0
    
    def _cleanup_windows(self):
        """Remove entries older than 1 minute"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        while self.requests_window and self.requests_window[0] < minute_ago:
            self.requests_window.popleft()
        
        while self.tokens_window and self.tokens_window[0][0] < minute_ago:
            self.tokens_window.popleft()
    
    def record_request(self, tokens_used: int):
        """Record a completed request with token usage"""
        current_time = time.time()
        self.requests_window.append(current_time)
        self.tokens_window.append((current_time, tokens_used))
        
        self.total_requests += 1
        self.total_tokens += tokens_used
        
        self._cleanup_windows()
    
    def get_current_utilization(self) -> Dict:
        """Get current resource utilization"""
        self._cleanup_windows()
        
        current_rpm = len(self.requests_window)
        current_tpm = sum(tokens for _, tokens in self.tokens_window)
        
        return {
            'current_rpm': current_rpm,
            'current_tpm': current_tpm,
            'rpm_utilization': current_rpm / self.rpm_limit,
            'tpm_utilization': current_tpm / self.tpm_limit,
            'rpm_remaining': self.rpm_limit - current_rpm,
            'tpm_remaining': self.tpm_limit - current_tpm,
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'elapsed_time': time.time() - self.start_time
        }


class SmartAPIClient:
    """API client with intelligent retry logic and precise rate limiting"""
    
    def __init__(self, throttler: Throttler, monitor: SlidingWindowMonitor, config: DeduplicationConfig, 
                 encoding, model_config: ModelConfig, verbose_reporter: VerboseReporter):
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
    async def make_request(self, batch: List[models.Codebook], var_lab: str) -> DeduplicationResult:
        """Make API request for deduplication with intelligent retry and rate limiting"""
        
        # Format codes batch
        codes_text = self._format_codes_batch(batch)
        
        prompt = DEDUPLICATION_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            survey_question=var_lab,
            codes_batch=codes_text
        )
        
        # Apply precision rate limiting
        async with self.throttler:
            try:
                # Make the API call
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    response_model=DeduplicationResult,
                    max_retries=0,  # Let tenacity handle retries
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed
                )
                
                # Record successful request with accurate token count
                estimated_tokens = len(self.encoding.encode(prompt))
                self.monitor.record_request(estimated_tokens)
                
                return response
                
            except Exception as e:
                self.verbose_reporter.error(f"API request failed for batch of {len(batch)} codes: {str(e)}")
                raise
    
    def _format_codes_batch(self, batch: List[models.Codebook]) -> str:
        """Format codes batch for prompt"""
        formatted_codes = []
        for i, code in enumerate(batch, 1):
            formatted_codes.append(f"{i}. {code.code}: {code.definition}")
        return "\n".join(formatted_codes)


class CodeBookDeduplicator:
    """Main class for codebook deduplication using sliding window overlap detection"""
    
    def __init__(
        self, 
        codebook: List[models.Codebook], 
        embedding_manager,  # OptimizedEmbeddingManager from codeGenerator
        var_lab: str,
        config: Optional[DeduplicationConfig] = None,
        verbose: bool = False):
        
        self.codebook = codebook
        self.embedding_manager = embedding_manager
        self.var_lab = var_lab
        self.config = config or DEFAULT_DEDUPLICATION_CONFIG
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.model_config = ModelConfig()
        
        # Initialize tokenizer for batch size calculation
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.verbose_reporter.warning(f"Using cl100k_base encoding as fallback for {self.config.model}")
        
        # Initialize workload analyzer
        self.workload_analyzer = WorkloadAnalyzer(self.config.model, self.encoding)
        
        # Initialize rate limits and monitoring
        rate_limits = get_openai_rate_limits(self.config.model)
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute
        
        self.verbose_reporter.stat_line(f"Model: {self.config.model}")
        self.verbose_reporter.stat_line(f"API Limits: {self.rpm_limit} RPM, {self.tpm_limit:,} TPM")
    
    async def deduplicate(self) -> List[models.Codebook]:
        """Main method to deduplicate codebook using sliding window analysis"""
        self.verbose_reporter.section_header("CODEBOOK DEDUPLICATION")
        
        if len(self.codebook) < self.config.min_codes_for_deduplication:
            self.verbose_reporter.stat_line(f"Skipping deduplication: only {len(self.codebook)} codes (minimum: {self.config.min_codes_for_deduplication})")
            return self.codebook
        
        self.verbose_reporter.stat_line(f"Processing {len(self.codebook)} codes for semantic duplicates")
        
        start_time = time.time()
        
        # Step 1: Generate embeddings for all codes
        code_embeddings = await self._generate_code_embeddings()
        
        # Step 2: Create similarity-based sliding window batches
        similarity_batches = await self._create_similarity_batches(code_embeddings)
        
        if not similarity_batches:
            self.verbose_reporter.stat_line("No similarity batches created - returning original codebook")
            return self.codebook
        
        # Step 3: Analyze optimal processing strategy
        strategy = await self._analyze_processing_strategy(similarity_batches)
        
        # Step 4: Process batches with optimal concurrency
        merge_decisions = await self._process_batches_concurrently(similarity_batches, strategy)
        
        # Step 5: Apply merge decisions to create final deduplicated codebook
        deduplicated_codebook = self._apply_merge_decisions(merge_decisions)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        original_count = len(self.codebook)
        final_count = len(deduplicated_codebook)
        duplicates_removed = original_count - final_count
        
        self.verbose_reporter.stat_line(f"Deduplication complete in {elapsed_time:.2f}s")
        self.verbose_reporter.stat_line(f"Original codes: {original_count}")
        self.verbose_reporter.stat_line(f"Final codes: {final_count}")
        self.verbose_reporter.stat_line(f"Duplicates merged: {duplicates_removed}")
        
        return deduplicated_codebook
    
    async def _generate_code_embeddings(self) -> List[Tuple[models.Codebook, np.ndarray]]:
        """Generate embeddings for all codes using the provided embedding manager"""
        self.verbose_reporter.stat_line("Generating embeddings for similarity analysis...")
        
        # Convert codebook to the format expected by embedding manager
        codes_dict = [{'code': code.code, 'definition': code.definition} for code in self.codebook]
        
        # Generate embeddings using the shared embedding manager
        codes, embeddings = await self.embedding_manager.get_snapshot_embeddings(codes_dict, version=1)
        
        # Combine codes with their embeddings
        code_embeddings = []
        for i, code in enumerate(self.codebook):
            if i < len(embeddings):
                code_embeddings.append((code, embeddings[i]))
        
        self.verbose_reporter.stat_line(f"Generated {len(code_embeddings)} code embeddings")
        return code_embeddings
    
    async def _create_similarity_batches(self, code_embeddings: List[Tuple[models.Codebook, np.ndarray]]) -> List[List[models.Codebook]]:
        """Create exactly 10-code batches using HDBSCAN with similarity-based padding"""
        if not code_embeddings:
            return []
        
        self.verbose_reporter.stat_line("Creating HDBSCAN-based clustering batches...")
        
        # Extract embeddings and codes
        embeddings_matrix = np.array([embedding for _, embedding in code_embeddings])
        codes = [code for code, _ in code_embeddings]
        
        # Calculate pairwise similarities for diagnostics
        similarity_matrix = cosine_similarity(embeddings_matrix)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        self.verbose_reporter.stat_line(f"Similarity stats: min={upper_triangle.min():.3f}, max={upper_triangle.max():.3f}, mean={upper_triangle.mean():.3f}")
        
        # Count pairs above threshold
        high_sim_pairs = np.sum(upper_triangle >= self.config.similarity_threshold)
        total_pairs = len(upper_triangle)
        self.verbose_reporter.stat_line(f"High similarity pairs (>={self.config.similarity_threshold}): {high_sim_pairs}/{total_pairs} ({100*high_sim_pairs/total_pairs:.1f}%)")
        
        # First, reduce dimensionality with UMAP for better clustering
        self.verbose_reporter.stat_line(f"Reducing embeddings to 10D with UMAP...")
        
        umap_reducer = umap.UMAP(
            n_components=10,
            metric='cosine',
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        
        reduced_embeddings = umap_reducer.fit_transform(embeddings_matrix)
        self.verbose_reporter.stat_line(f"Reduced embeddings from {embeddings_matrix.shape[1]}D to {reduced_embeddings.shape[1]}D")
        
        # Use HDBSCAN with minimum cluster size constraint on reduced embeddings
        self.verbose_reporter.stat_line(f"Running HDBSCAN with min_cluster_size={self.config.batch_size}")
        
        # HDBSCAN clustering - ensures each cluster has at least batch_size codes
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.batch_size,
            metric='euclidean',  # Use euclidean on reduced embeddings
            cluster_selection_method='eom'  # Excess of Mass for better semantic clustering
        )
        
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        # Group codes by cluster (excluding noise points labeled as -1)
        clusters = {}
        noise_codes = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:
                noise_codes.append((codes[i], embeddings_matrix[i]))
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((codes[i], embeddings_matrix[i]))
        
        self.verbose_reporter.stat_line(f"HDBSCAN found {len(clusters)} natural clusters, {len(noise_codes)} noise points")
        
        # Create batches from clusters
        batches = []
        
        # Process each cluster - each should be at least batch_size due to min_cluster_size
        for cluster_id, cluster_items in clusters.items():
            cluster_codes = [code for code, _ in cluster_items]
            cluster_embeddings = [emb for _, emb in cluster_items]
            
            if len(cluster_codes) <= self.config.batch_size:
                # Perfect size or small - use as single batch
                batch = cluster_codes[:]
                
                # Pad to exactly batch_size if needed
                if len(batch) < self.config.batch_size:
                    batch = self._pad_batch_with_similar_codes(
                        batch, cluster_embeddings, code_embeddings, similarity_matrix
                    )
                
                # Add cross-cluster padding for better duplicate detection
                batch = self._add_cross_cluster_padding(batch, code_embeddings, similarity_matrix)
                batches.append(batch[:self.config.batch_size])  # Ensure exactly batch_size
                
            else:
                # Large cluster: split into overlapping batches
                step_size = max(1, self.config.batch_size - self.config.overlap_size)
                
                for i in range(0, len(cluster_codes), step_size):
                    batch = cluster_codes[i:i + self.config.batch_size]
                    
                    if len(batch) >= self.config.batch_size:
                        # Add cross-cluster padding for better duplicate detection
                        batch = self._add_cross_cluster_padding(batch, code_embeddings, similarity_matrix)
                        batches.append(batch[:self.config.batch_size])
                    elif len(batch) >= self.config.batch_size // 2:
                        # Pad smaller end batches to full size
                        batch = self._pad_batch_with_similar_codes(
                            batch, [cluster_embeddings[j] for j in range(i, min(i + len(batch), len(cluster_embeddings)))],
                            code_embeddings, similarity_matrix
                        )
                        # Add cross-cluster padding for better duplicate detection
                        batch = self._add_cross_cluster_padding(batch, code_embeddings, similarity_matrix)
                        batches.append(batch[:self.config.batch_size])
        
        # Handle noise points by adding them to most similar existing batches
        if noise_codes:
            self.verbose_reporter.stat_line(f"Assigning {len(noise_codes)} noise points to similar batches")
            batches = self._assign_noise_to_batches(noise_codes, batches, similarity_matrix, code_embeddings)
        
        # Ensure we have enough batches - create additional if needed
        expected_batches = max(1, len(codes) // self.config.batch_size)
        if len(batches) < expected_batches:
            self.verbose_reporter.stat_line(f"Creating additional batches to reach target of {expected_batches}")
            batches = self._create_additional_batches(batches, code_embeddings, expected_batches)
        
        # Calculate overlap statistics
        all_codes_in_batches = set()
        total_code_instances = 0
        for batch in batches:
            total_code_instances += len(batch)
            all_codes_in_batches.update(code.code for code in batch)
        
        avg_overlap = (total_code_instances - len(all_codes_in_batches)) / max(1, len(batches) - 1) if len(batches) > 1 else 0
        
        self.verbose_reporter.stat_line(f"Created {len(batches)} HDBSCAN batches (avg size: {total_code_instances/len(batches) if batches else 0:.1f}, avg overlap: {avg_overlap:.1f})")
        return batches
    
    def _pad_batch_with_similar_codes(self, batch: List[models.Codebook], batch_embeddings: List[np.ndarray], 
                                     all_code_embeddings: List[Tuple[models.Codebook, np.ndarray]], 
                                     similarity_matrix: np.ndarray) -> List[models.Codebook]:
        """Pad a batch to target size using most similar codes"""
        if len(batch) >= self.config.batch_size:
            return batch
        
        # Calculate mean embedding of current batch
        mean_batch_embedding = np.mean(batch_embeddings, axis=0)
        
        # Find most similar codes not in current batch
        batch_code_names = {code.code for code in batch}
        similarities = []
        
        for code, embedding in all_code_embeddings:
            if code.code not in batch_code_names:
                sim = cosine_similarity([mean_batch_embedding], [embedding])[0][0]
                similarities.append((sim, code))
        
        # Sort by similarity and add best matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        needed = self.config.batch_size - len(batch)
        batch.extend([code for _, code in similarities[:needed]])
        
        return batch
    
    def _add_cross_cluster_padding(self, batch: List[models.Codebook], 
                                  all_code_embeddings: List[Tuple[models.Codebook, np.ndarray]], 
                                  similarity_matrix: np.ndarray) -> List[models.Codebook]:
        """Add cross-cluster similar codes to catch duplicates across semantic clusters"""
        if len(batch) >= self.config.batch_size:
            # For full batches, replace least similar codes with most similar cross-cluster codes
            cross_cluster_count = min(7, self.config.overlap_size + 2)  # Replace up to 7 codes
            
            # Calculate batch centroid
            batch_code_names = {code.code for code in batch}
            batch_embeddings = []
            for code in batch:
                for orig_code, embedding in all_code_embeddings:
                    if orig_code.code == code.code:
                        batch_embeddings.append(embedding)
                        break
            
            if not batch_embeddings:
                return batch
                
            batch_centroid = np.mean(batch_embeddings, axis=0)
            
            # Find most similar codes NOT in current batch
            cross_cluster_similarities = []
            for code, embedding in all_code_embeddings:
                if code.code not in batch_code_names:
                    sim = cosine_similarity([batch_centroid], [embedding])[0][0]
                    cross_cluster_similarities.append((sim, code))
            
            # Get top cross-cluster matches
            cross_cluster_similarities.sort(key=lambda x: x[0], reverse=True)
            top_cross_cluster = [code for _, code in cross_cluster_similarities[:cross_cluster_count]]
            
            # Replace least similar codes in batch with top cross-cluster matches
            batch_similarities = []
            for code in batch:
                for orig_code, embedding in all_code_embeddings:
                    if orig_code.code == code.code:
                        sim = cosine_similarity([batch_centroid], [embedding])[0][0]
                        batch_similarities.append((sim, code))
                        break
            
            # Sort batch codes by similarity (least similar first)
            batch_similarities.sort(key=lambda x: x[0])
            
            # Replace least similar with cross-cluster codes
            final_batch = [code for _, code in batch_similarities[cross_cluster_count:]]
            final_batch.extend(top_cross_cluster)
            
            return final_batch
        else:
            # For smaller batches, just add cross-cluster padding
            return self._pad_batch_with_similar_codes(batch, [], all_code_embeddings, similarity_matrix)
    
    def _assign_noise_to_batches(self, noise_codes: List[Tuple[models.Codebook, np.ndarray]], 
                                batches: List[List[models.Codebook]], 
                                similarity_matrix: np.ndarray,
                                all_code_embeddings: List[Tuple[models.Codebook, np.ndarray]]) -> List[List[models.Codebook]]:
        """Assign noise points to most similar existing batches"""
        
        for noise_code, noise_embedding in noise_codes:
            best_batch_idx = 0
            best_similarity = -1
            
            # Find most similar batch
            for batch_idx, batch in enumerate(batches):
                # Calculate similarity to batch centroid
                batch_embeddings = []
                for code in batch:
                    for orig_code, embedding in all_code_embeddings:
                        if orig_code.code == code.code:
                            batch_embeddings.append(embedding)
                            break
                
                if batch_embeddings:
                    batch_centroid = np.mean(batch_embeddings, axis=0)
                    similarity = cosine_similarity([noise_embedding], [batch_centroid])[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_batch_idx = batch_idx
            
            # Add to most similar batch (allow slight size increase for noise points)
            if best_batch_idx < len(batches):
                batches[best_batch_idx].append(noise_code)
        
        return batches
    
    def _create_additional_batches(self, existing_batches: List[List[models.Codebook]], 
                                  all_code_embeddings: List[Tuple[models.Codebook, np.ndarray]], 
                                  target_batch_count: int) -> List[List[models.Codebook]]:
        """Create additional batches if needed to reach target count"""
        
        # Find codes not yet in any batch
        used_codes = set()
        for batch in existing_batches:
            used_codes.update(code.code for code in batch)
        
        unused_codes = [code for code, _ in all_code_embeddings if code.code not in used_codes]
        
        # Create additional batches from unused codes
        while len(existing_batches) < target_batch_count and len(unused_codes) >= self.config.batch_size // 2:
            batch = unused_codes[:self.config.batch_size]
            unused_codes = unused_codes[self.config.batch_size:]
            
            # Pad if needed
            if len(batch) < self.config.batch_size:
                batch_embeddings = []
                for code in batch:
                    for orig_code, embedding in all_code_embeddings:
                        if orig_code.code == code.code:
                            batch_embeddings.append(embedding)
                            break
                
                if batch_embeddings:
                    batch = self._pad_batch_with_similar_codes(batch, batch_embeddings, all_code_embeddings, None)
            
            existing_batches.append(batch[:self.config.batch_size])
        
        return existing_batches
    
    async def _analyze_processing_strategy(self, batches: List[List[models.Codebook]]) -> OptimalStrategy:
        """Analyze workload and calculate optimal processing strategy"""
        self.verbose_reporter.stat_line("Analyzing optimal processing strategy...")
        
        # Measure token usage from sample batches
        avg_tokens_per_batch = self.workload_analyzer.measure_token_usage(batches, self.var_lab)
        
        # Calculate optimal strategy
        strategy = self.workload_analyzer.calculate_optimal_strategy(len(batches), avg_tokens_per_batch)
        
        self.verbose_reporter.stat_line(f"Optimal strategy: {strategy.concurrent_limit} concurrent, {strategy.launch_rate_per_second:.2f} req/sec")
        self.verbose_reporter.stat_line(f"Estimated time: {strategy.target_time_seconds:.1f}s ({strategy.bottleneck_type} bottleneck)")
        
        return strategy
    
    async def _process_batches_concurrently(self, batches: List[List[models.Codebook]], strategy: OptimalStrategy) -> List[DeduplicationResult]:
        """Process similarity batches concurrently with optimal strategy"""
        
        # Initialize throttling and monitoring
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second)
        monitor = SlidingWindowMonitor(self.rpm_limit, self.tpm_limit)
        api_client = SmartAPIClient(throttler, monitor, self.config, self.encoding, self.model_config, self.verbose_reporter)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(strategy.concurrent_limit)
        
        async def process_single_batch(batch_idx: int, batch: List[models.Codebook]) -> Optional[DeduplicationResult]:
            async with semaphore:
                try:
                    result = await api_client.make_request(batch, self.var_lab)
                    self.verbose_reporter.stat_line(f"Processed batch {batch_idx + 1}/{len(batches)}: {len(result.merge_decisions)} merges found")
                    return result
                except Exception as e:
                    self.verbose_reporter.error(f"Failed to process batch {batch_idx + 1}: {str(e)}")
                    return None
        
        # Process all batches
        self.verbose_reporter.stat_line(f"Processing {len(batches)} batches with {strategy.concurrent_limit} concurrent requests...")
        
        tasks = [process_single_batch(i, batch) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [result for result in results if isinstance(result, DeduplicationResult)]
        
        self.verbose_reporter.stat_line(f"Successfully processed {len(successful_results)}/{len(batches)} batches")
        
        # Show final utilization
        utilization = monitor.get_current_utilization()
        self.verbose_reporter.stat_line(f"API utilization: {utilization['total_requests']} requests, {utilization['total_tokens']:,} tokens")
        
        return successful_results
    
    def _apply_merge_decisions(self, dedup_results: List[DeduplicationResult]) -> List[models.Codebook]:
        """Apply merge decisions to create final deduplicated codebook"""
        
        # Collect all merge decisions
        all_merges = []
        codes_to_remove = set()
        
        for result in dedup_results:
            for merge in result.merge_decisions:
                all_merges.append(merge)
                # Mark all codes in merge except the final one for removal
                codes_to_remove.update(merge.codes_to_merge)
        
        # Create final codebook
        final_codebook = []
        
        # Add codes that weren't merged
        for code in self.codebook:
            if code.code not in codes_to_remove:
                final_codebook.append(code)
        
        # Add merged codes
        for merge in all_merges:
            merged_code = models.Codebook(
                code=merge.final_code_name,
                definition=merge.final_definition
            )
            final_codebook.append(merged_code)
        
        # Remove duplicates (in case same code appears multiple times)
        seen_codes = set()
        unique_codebook = []
        for code in final_codebook:
            if code.code not in seen_codes:
                unique_codebook.append(code)
                seen_codes.add(code.code)
        
        return unique_codebook


# === INTEGRATION FUNCTION ====================================================================================================

async def deduplicate_codebook(
    codebook: List[models.Codebook],
    embedding_manager,  # OptimizedEmbeddingManager from codeGenerator
    var_lab: str,
    config: Optional[DeduplicationConfig] = None,
    verbose: bool = False,
    two_pass: bool = True
) -> List[models.Codebook]:
    """
    Main entry point for codebook deduplication
    
    Args:
        codebook: List of codes to deduplicate
        embedding_manager: OptimizedEmbeddingManager instance from codeGenerator
        var_lab: Survey question for context
        config: Optional deduplication configuration
        verbose: Enable verbose output
        two_pass: Run deduplication twice to catch missed duplicates
    
    Returns:
        Deduplicated list of codes
    """
    
    # First pass
    if verbose:
        print("🔄 Running first deduplication pass...")
    
    deduplicator = CodeBookDeduplicator(
        codebook=codebook,
        embedding_manager=embedding_manager,
        var_lab=var_lab,
        config=config,
        verbose=verbose
    )
    
    first_pass_result = await deduplicator.deduplicate()
    
    if not two_pass or len(first_pass_result) == len(codebook):
        # No duplicates found in first pass, or two_pass disabled
        return first_pass_result
    
    # Second pass - run on first pass results
    if verbose:
        print(f"\n🔄 Running second deduplication pass on {len(first_pass_result)} codes...")
    
    # Create new embedding manager for second pass
    from utils.codeGenerator import SharedCodebook, OptimizedEmbeddingManager
    second_pass_shared_codebook = SharedCodebook([{'code': c.code, 'definition': c.definition} for c in first_pass_result])
    second_pass_embedding_manager = OptimizedEmbeddingManager(second_pass_shared_codebook, verbose=False)  # Less verbose for second pass
    
    second_pass_deduplicator = CodeBookDeduplicator(
        codebook=first_pass_result,
        embedding_manager=second_pass_embedding_manager,
        var_lab=var_lab,
        config=config,
        verbose=verbose
    )
    
    final_result = await second_pass_deduplicator.deduplicate()
    
    if verbose:
        total_removed = len(codebook) - len(final_result)
        pass1_removed = len(codebook) - len(first_pass_result)
        pass2_removed = len(first_pass_result) - len(final_result)
        print(f"📊 Two-pass summary: {len(codebook)} → {len(first_pass_result)} → {len(final_result)} codes")
        print(f"🎯 Pass 1 removed: {pass1_removed}, Pass 2 removed: {pass2_removed}, Total: {total_removed}")
    
    return final_result


if __name__ == "__main__":
    print("CodeBookDeduplicator - Semantic Duplicate Detection Complete! 🚀")