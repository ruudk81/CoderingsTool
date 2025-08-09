# Clean Code Generator - Rebuilt with simple 3-phase architecture
# Applies Phase 1's proven efficient patterns to all phases

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import json
import statistics
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from collections import deque

# Third-party imports
import instructor
import numpy as np
from openai import AsyncOpenAI, RateLimitError
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import (
    OPENAI_API_KEY, 
    ModelConfig, 
    DEFAULT_EMBEDDING_CONFIG,
    DEFAULT_LANGUAGE,
    get_openai_rate_limits,
    get_embedding_dimensions
)

# === PROMPTS ========================================================================================================
from prompts import (
    CLUSTER_SUMMARY_PROMPT,
    CANDIDATE_CODE_SELECTION_PROMPT, 
    CODE_GENERATION_PROMPT,
    VALIDATION_PROMPT
)

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .embedder import Embedder

# Initialize instructor client
async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


@dataclass
class OptimalStrategy:
    """Evidence-based optimal processing strategy (ported from qualityFilter.py)"""
    target_time_seconds: float
    launch_rate_per_second: float
    concurrent_limit: int
    bottleneck_type: str
    total_requests: int
    total_tokens: int
    safety_factor: float


class SlidingWindowMonitor:
    """Real-time monitoring of API usage with sliding windows (ported from qualityFilter.py)"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: int = 60):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        
        # Sliding windows for tracking usage
        self.requests_window = deque()  # timestamps
        self.tokens_window = deque()    # (timestamp, token_count) tuples
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
    
    def _cleanup_windows(self):
        """Remove entries older than window_seconds"""
        cutoff_time = time.time() - self.window_seconds
        
        # Clean requests window
        while self.requests_window and self.requests_window[0] < cutoff_time:
            self.requests_window.popleft()
        
        # Clean tokens window
        while self.tokens_window and self.tokens_window[0][0] < cutoff_time:
            self.tokens_window.popleft()
    
    def record_request(self, tokens_used: int):
        """Record a completed API request"""
        now = time.time()
        self.requests_window.append(now)
        self.tokens_window.append((now, tokens_used))
        
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


class WorkloadAnalyzer:
    """Analyzes workload and calculates optimal processing strategy (ported from qualityFilter.py)"""
    
    def __init__(self, model_name: str, encoding):
        self.model_name = model_name
        self.encoding = encoding
    
    def calculate_optimal_strategy(self, total_requests: int, avg_tokens_per_request: float) -> OptimalStrategy:
        """Calculate mathematically optimal processing strategy"""
        # Get API limits from config
        rate_limits = get_openai_rate_limits(self.model_name)
        
        # Calculate total resource requirements
        total_tokens = total_requests * avg_tokens_per_request
        
        # Calculate minimum time based on constraints
        time_by_requests = total_requests / rate_limits.requests_per_minute * 60
        time_by_tokens = total_tokens / rate_limits.tokens_per_minute * 60
        
        # Find bottleneck and minimum time
        bottleneck_time = max(time_by_requests, time_by_tokens)
        bottleneck_type = 'tokens' if time_by_tokens > time_by_requests else 'requests'
        
        # Apply safety factor (use 90% of capacity for cleaner architecture)
        safety_factor = 0.90
        target_time = bottleneck_time / safety_factor
        
        # Calculate optimal launch rate
        optimal_launch_rate = total_requests / target_time
        
        # Calculate concurrent request limit (3 seconds of buffer)
        concurrent_limit = int(optimal_launch_rate * 3)
        
        return OptimalStrategy(
            target_time_seconds=target_time,
            launch_rate_per_second=optimal_launch_rate,
            concurrent_limit=concurrent_limit,
            bottleneck_type=bottleneck_type,
            total_requests=total_requests,
            total_tokens=total_tokens,
            safety_factor=safety_factor
        )


# Phase 3 is implemented in cleanCodeGenerator_phase3.py with LangChain


class CleanCodeGenerator:
    """Clean 3-phase code generator using proven efficient patterns"""
    
    def __init__(
        self,
        cluster_data: List[models.ClusterModel],
        var_lab: str = "Survey question",
        config: Optional[ModelConfig] = None,
        verbose: bool = False,
        prompt_printer = None
    ):
        self.cluster_data = cluster_data
        self.var_lab = var_lab
        self.config = config or ModelConfig()
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        
        # Initialize embedder
        self.embedder = Embedder(
            config=DEFAULT_EMBEDDING_CONFIG,
            verbose=verbose
        )
        
        # TODO: Initialize shared codebook with LangChain implementation in Phase 3
        self.shared_codebook = None  # Will be replaced with original SharedCodebook
        
        # Get the model name for cluster analysis
        self.model_name = self.config.get_model_for_stage("cluster_analysis")
        
        # Initialize tokenizer for optimal strategy calculation
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.verbose_reporter.warning(f"Using cl100k_base encoding as fallback for {self.model_name}")
        
        # Initialize workload analyzer
        self.workload_analyzer = WorkloadAnalyzer(self.model_name, self.encoding)
        
        # Get rate limits
        rate_limits = get_openai_rate_limits(self.model_name)
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute
        
        # Processing results storage
        self.theme_map: Dict[str, Dict] = {}
        self.embedding_book: Dict[str, Dict] = {}
        self.step2_results: Dict[str, Any] = {}
        self.step3_results: Dict[str, Any] = {}
        self.step4_results: Dict[str, Any] = {}
    
    def _prepare_cluster_text(self) -> Dict[int, Dict]:
        """Prepare cluster data by grouping individual responses by their actual HDBSCAN cluster_id"""
        clusters = {}
        
        for result in self.cluster_data:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    # Initialize cluster if not seen before
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {'ideas': [], 'embeddings': []}
                    
                    # Add idea to the actual cluster
                    clusters[cluster_id]['ideas'].append(idea.idea)
                    
                    # Add embedding if available
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
        
        # Filter out empty clusters and log statistics
        valid_clusters = {cid: cdata for cid, cdata in clusters.items() if len(cdata['ideas']) > 0}
        
        total_ideas = sum(len(cluster_data['ideas']) for cluster_data in valid_clusters.values())
        self.verbose_reporter.stat_line(f"Grouped {len(self.cluster_data)} responses into {len(valid_clusters)} actual HDBSCAN clusters")
        self.verbose_reporter.stat_line(f"Total ideas across all clusters: {total_ideas}")
        
        return valid_clusters
    
    def _format_cluster_summary_for_prompts(self, cluster_summary_json: str) -> str:
        """Format cluster summary for readable prompt display (remove raw JSON)"""
        try:
            parsed = json.loads(cluster_summary_json)
            themes = parsed.get('themes', [])
            
            if len(themes) == 1:
                return f"Theme: {themes[0]}"
            else:
                formatted_themes = []
                for i, theme in enumerate(themes, 1):
                    formatted_themes.append(f"Theme {i}: {theme}")
                return "\n".join(formatted_themes)
        except (json.JSONDecodeError, KeyError):
            return cluster_summary_json
    
    async def generate(self) -> Dict[str, Any]:
        """Main entry point: Clean 3-phase sequential execution"""
        self.verbose_reporter.step_start("Clean Code Generator - 3-Phase Architecture")
        
        start_time = time.time()
        
        # Phase 1: Extract themes from clusters
        self.verbose_reporter.stat_line("Phase 1: Extracting themes...")
        await self.phase1_extract_themes()
        phase1_time = time.time() - start_time
        self.verbose_reporter.stat_line(f"Phase 1 completed in {phase1_time:.1f}s")
        
        # Phase 2: Embed themes
        phase2_start = time.time()
        self.verbose_reporter.stat_line("Phase 2: Embedding themes...")
        await self.phase2_embed_themes()
        phase2_time = time.time() - phase2_start
        self.verbose_reporter.stat_line(f"Phase 2 completed in {phase2_time:.1f}s")
        
        # Phase 3: Process codes (Steps 2-4 concurrently)
        phase3_start = time.time()
        self.verbose_reporter.stat_line("Phase 3: Processing codes...")
        await self.phase3_process_codes()
        phase3_time = time.time() - phase3_start
        self.verbose_reporter.stat_line(f"Phase 3 completed in {phase3_time:.1f}s")
        
        total_time = time.time() - start_time
        self.verbose_reporter.stat_line(f"Total processing time: {total_time:.1f}s")
        
        # Return results in compatible format
        return {
            'phase1_results': self.theme_map,
            'phase2_results': self.embedding_book,
            'phase3_results': {
                'step2': self.step2_results,
                'step3': self.step3_results,
                'step4': self.step4_results
            },
            'processing_metadata': {
                'total_time': total_time,
                'phase1_time': phase1_time,
                'phase2_time': phase2_time,
                'phase3_time': phase3_time,
                'clusters_processed': len(self.cluster_data)
            }
        }
    
    async def phase1_extract_themes(self):
        """Phase 1: Extract themes from clusters using concurrent instructor calls"""
        start_time = time.time()
        
        # First, group individual responses by actual HDBSCAN cluster_id (like original codeGenerator)
        actual_clusters = self._prepare_cluster_text()
        if not actual_clusters:
            self.verbose_reporter.error("No valid clusters found to process")
            return
        
        # Calculate optimal strategy for concurrent processing
        total_clusters = len(actual_clusters)  # Use actual cluster count, not individual responses
        avg_tokens_per_request = 1500  # Estimated based on cluster analysis
        strategy = self.workload_analyzer.calculate_optimal_strategy(total_clusters, avg_tokens_per_request)
        
        self.verbose_reporter.stat_line(f"Processing {total_clusters} clusters")
        self.verbose_reporter.stat_line(f"Strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
        
        # Initialize monitoring and throttling
        monitor = SlidingWindowMonitor(self.rpm_limit, self.tpm_limit)
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second, period=1.0)
        
        async def process_single_cluster(cluster_id: int, cluster_data: Dict) -> Tuple[str, Dict[str, Any]]:
            """Process single actual cluster to extract themes using instructor"""
            async with throttler:
                try:
                    # Get ideas from the grouped cluster data
                    cluster_ideas = cluster_data['ideas']
                    cluster_text = "\n".join([f"- {idea}" for idea in cluster_ideas])
                    
                    # Create prompt following existing pattern
                    prompt = CLUSTER_SUMMARY_PROMPT.format(
                        language=DEFAULT_LANGUAGE,
                        survey_question=self.var_lab,
                        cluster_text=cluster_text
                    )
                    
                    # Capture prompt for first cluster if prompt_printer available  
                    is_first_cluster = (cluster_id == min(actual_clusters.keys())) if actual_clusters else False
                    
                    if self.prompt_printer and is_first_cluster:
                        self.prompt_printer.capture_prompt(
                            step_name="phase1_extract_themes",
                            utility_name="CleanCodeGenerator",
                            prompt_content=prompt,
                            prompt_type="cluster_analysis",
                            metadata={
                                "model": self.model_name,
                                "cluster_id": cluster_id,
                                "ideas_count": len(cluster_ideas)
                            }
                        )
                    
                    # Make instructor API call with ClusterThemeAnalysis model
                    response = await async_client.chat.completions.create(
                        model=self.model_name,
                        response_model=models.ClusterThemeAnalysis,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=4000,
                        seed=42
                    )
                    
                    # Record successful request
                    estimated_tokens = len(self.encoding.encode(prompt)) + 500  # +500 for completion
                    monitor.record_request(estimated_tokens)
                    
                    # Convert to format expected by other phases
                    themes = response.themes if response.themes else []
                    summary_json = response.model_dump_json()
                    
                    return cluster_id, {
                        'themes': themes,
                        'summary_json': summary_json,
                        'cluster_text': cluster_text,
                        'cluster_data': cluster_data  # Store the grouped cluster data
                    }
                    
                except Exception as e:
                    self.verbose_reporter.error(f"Failed to process cluster {cluster_id}: {str(e)}")
                    # Return empty result to continue processing
                    return cluster_id, {
                        'themes': [],
                        'summary_json': '{"themes": []}',
                        'cluster_text': '',
                        'cluster_data': cluster_data  # Store the grouped cluster data even on error
                    }
        
        # Process all actual clusters concurrently using asyncio.as_completed for progress
        tasks = [process_single_cluster(cluster_id, cluster_data) for cluster_id, cluster_data in actual_clusters.items()]
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            cluster_id, result = await coro
            self.theme_map[str(cluster_id)] = result  # Convert cluster_id to string for consistency
            completed += 1
            
            # Progress reporting every 5 clusters or at end
            if completed % 5 == 0 or completed == len(tasks):
                self.verbose_reporter.progress_line(completed, len(tasks), "actual clusters")
        
        # Final stats
        phase1_time = time.time() - start_time
        total_themes = sum(len(data['themes']) for data in self.theme_map.values())
        final_stats = monitor.get_current_utilization()
        
        self.verbose_reporter.stat_line(f"Extracted {total_themes} themes from {len(self.theme_map)} clusters")
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Phase 1 completed in {phase1_time:.1f}s " +
                                          f"(RPM: {final_stats['rpm_utilization']:.0%}, " +
                                          f"TPM: {final_stats['tpm_utilization']:.0%} utilization)")
    
    async def phase2_embed_themes(self):
        """Phase 2: Batch embed all themes using OpenAI embeddings API"""
        start_time = time.time()
        
        # Collect all themes with tracking info
        all_theme_texts = []
        theme_tracking = []  # (cluster_id, theme_idx, text)
        
        for cluster_id, cluster_info in self.theme_map.items():
            themes = cluster_info.get('themes', [])
            for theme_idx, theme_text in enumerate(themes):
                # Clean theme text (remove "Theme N:" prefix if present)
                import re
                cleaned_theme = re.sub(r'^Theme \d+:\s*', '', theme_text.strip())
                
                all_theme_texts.append(cleaned_theme)
                theme_tracking.append((cluster_id, theme_idx, cleaned_theme))
        
        total_themes = len(all_theme_texts)
        self.verbose_reporter.stat_line(f"Embedding {total_themes} themes")
        
        if total_themes == 0:
            self.verbose_reporter.stat_line("No themes to embed")
            return
        
        # Initialize OpenAI client for embeddings
        embedding_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Process themes in large batches (embeddings API can handle ~2048 at once)
        batch_size = DEFAULT_EMBEDDING_CONFIG.batch_size  # Usually 2048
        embedding_results = []
        
        self.verbose_reporter.stat_line(f"Processing {len(all_theme_texts)} themes in batches of {batch_size}")
        
        for i in range(0, len(all_theme_texts), batch_size):
            batch_texts = all_theme_texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                self.verbose_reporter.stat_line(f"Embedding batch {batch_num}: {len(batch_texts)} themes")
                
                # Make embedding API call
                response = await embedding_client.embeddings.create(
                    input=batch_texts,
                    model=DEFAULT_EMBEDDING_CONFIG.embedding_model
                )
                
                # Extract embeddings as numpy arrays
                batch_embeddings = [
                    np.array(embedding_data.embedding, dtype=np.float32)
                    for embedding_data in response.data
                ]
                embedding_results.extend(batch_embeddings)
                
            except Exception as e:
                self.verbose_reporter.error(f"Embedding batch {batch_num} failed: {e}")
                # Add None for failed embeddings
                embedding_results.extend([None] * len(batch_texts))
        
        # Build embedding book with cluster_id -> theme_idx -> embedding mapping
        themes_embedded = 0
        for i, (cluster_id, theme_idx, theme_text) in enumerate(theme_tracking):
            if cluster_id not in self.embedding_book:
                self.embedding_book[cluster_id] = {}
            
            if i < len(embedding_results) and embedding_results[i] is not None:
                self.embedding_book[cluster_id][theme_idx] = {
                    'embedding': embedding_results[i],
                    'text': theme_text
                }
                themes_embedded += 1
        
        # Final stats
        phase2_time = time.time() - start_time
        self.verbose_reporter.stat_line(f"Embedded {themes_embedded} themes in {phase2_time:.1f}s")
        
        await embedding_client.close()  # Cleanup client
    
# Step 2, 3, 4 methods will be replaced with LangChain implementation
    
# Step 3 and 4 methods removed - will be replaced with LangChain implementation
    
    async def phase3_process_codes(self):
        """Phase 3: Process codes using LangChain SequentialChain with batch processing"""
        from .cleanCodeGenerator_phase3 import LangChainPhase3Processor, SharedCodebook
        
        start_time = time.time()
        
        # Initialize shared codebook with starter codes if available
        starter_codes = []  # TODO: Get from config or parameter
        self.shared_codebook = SharedCodebook(initial_codes=starter_codes)
        
        # Create Phase 3 processor with LangChain
        phase3_processor = LangChainPhase3Processor(
            cluster_data=self.cluster_data,
            theme_map=self.theme_map,
            embedding_book=self.embedding_book,
            shared_codebook=self.shared_codebook,
            var_lab=self.var_lab,
            model_config=self.config,
            verbose_reporter=self.verbose_reporter,
            prompt_printer=self.prompt_printer
        )
        
        # Process clusters in batches (10-15 clusters per batch for balance)
        batch_size = 12  # Configurable batch size
        results = await phase3_processor.process_clusters_in_batches(batch_size=batch_size)
        
        # Get results for compatibility
        self.step2_results = phase3_processor.step2_results
        self.step3_results = phase3_processor.step3_results
        self.step4_results = phase3_processor.step4_results
        
        phase3_time = time.time() - start_time
        
        self.verbose_reporter.stat_line(f"Phase 3 completed in {phase3_time:.1f}s")
        self.verbose_reporter.stat_line(f"Processed {results['clusters_processed']} clusters ({results['successful_clusters']} successful)")
        self.verbose_reporter.stat_line(f"Added {results['total_new_codes']} new codes, modified {results['total_replaced_codes']} codes")
        self.verbose_reporter.stat_line(f"Final codebook: {results['final_codebook_size']} codes")