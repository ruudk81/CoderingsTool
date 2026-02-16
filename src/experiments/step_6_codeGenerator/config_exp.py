"""
Experimental Configuration for Step 6: Code Generator

This file contains experimental configuration parameters for codeGenerator_exp.py.
Modify these settings freely to experiment without affecting production.

Original source: src/config.py (CodeDesignerConfig section) + codeGenerator_exp.py constants
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Literal

# =============================================================================
# RE-EXPORT PRODUCTION CONFIG (items that don't need modification)
# =============================================================================
from config import (
    OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, ProcessingConfig,
    DEFAULT_PROCESSING_CONFIG, API_PROVIDER,
    AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER, FALLBACK_TPM, FALLBACK_RPM,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL
)

# =============================================================================
# EXPERIMENTAL CONSTANTS (moved from codeGenerator_exp.py)
# =============================================================================

# Verbose/debug settings
EXTRA_VERBOSE = False

# Stage 1 text source: "idea" (full response text)
STAGE1_TEXT_SOURCE: Literal["idea"] = "idea"

# Stage 1 input source: "ideas" (sample raw ideas from clusters) or "mece_topics" (use MECE Phase A output)
# When "mece_topics", pass mece_topics= to InductiveCodeGenerator; STAGE1_TEXT_SOURCE is ignored.
STAGE1_INPUT_SOURCE: Literal["ideas", "mece_topics"] = "mece_topics"

# Timeout and latency
DEFAULT_TIMEOUT_SECONDS = 30.0        # Default timeout when no latency data
DEFAULT_LATENCY_SECONDS = 0.5         # Default latency estimate
MIN_LATENCY_SECONDS = 0.05            # Minimum latency bound
TOKENS_FOR_BASELINE_LATENCY = 1000    # Token count for baseline latency calculation
TIMEOUT_PER_1000_TOKENS = 0.1         # Additional timeout per 1000 tokens

# Token estimation
OUTPUT_TOKEN_ESTIMATE_MARGIN = 1.2    # 20% margin for output token estimation
FALLBACK_TOKEN_ESTIMATE = 400         # Fallback when tiktoken fails
INPUT_TOKEN_ESTIMATE_MARGIN = 1.15    # 15% margin for input token estimation
OUTPUT_ESTIMATE_PCT_OF_INPUT = 0.20   # Output estimate as % of input
FIRST_PROMPT_ALLOCATION_PCT = 0.85    # First prompt token allocation

# Similarity thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.7    # Default similarity threshold for theme embeddings
FINAL_SIMILARITY_THRESHOLD = 0.85     # Final fallback similarity threshold
CONFLICT_THRESHOLD = 0.9              # Threshold for batch conflict detection
PROGRESSIVE_SIMILARITY_THRESHOLDS = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]  # Progressive batching thresholds
SIMILARITY_ANALYSIS_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # Distribution analysis

# Batch and concurrency
EMBEDDING_BATCH_SIZE = 100            # Batch size for embedding generation
DEFAULT_SUB_BATCH_MAX_SIZE = 10       # Default max size for sub-batches
MIN_WORKER_CONCURRENCY = 50           # Minimum worker concurrency
MAX_WORKER_CONCURRENCY = 200          # Maximum worker concurrency

# Embedding
OPENAI_EMBEDDING_DIMENSION = 1536     # OpenAI embedding vector size

# Retry and backoff
EXPONENTIAL_BACKOFF_BASE = 0.8        # Base for exponential backoff in retries
ASSUMED_LATENCY_PER_REQUEST = 2.0     # Assumed latency for rate calculation

# Monitoring
BUCKET_STATUS_RECENT_SAMPLES = 10     # Samples for recent average calculation
LOW_TOKEN_THRESHOLD_PCT = 0.2         # Low token threshold (20% of capacity)
PROGRESS_REPORT_INTERVAL = 5          # Seconds between progress reports
DIAGNOSTIC_REPORT_INTERVAL = 30       # Seconds between diagnostic reports


# =============================================================================
# EXPERIMENTAL CODEDESIGNER CONFIG (copy from production for modification)
# =============================================================================
@dataclass
class CodeDesignerConfigExp:
    """Experimental CodeDesigner configuration - modify freely for experiments"""

    # Model configuration
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 4000
    seed: Optional[int] = 42

    # Theme-based similarity batching
    similarity_threshold: float = 0.7  # Cosine similarity threshold for dissimilarity batching
    max_sub_batch_size: int = 10  # Maximum clusters per sub-batch

    # Rate limiting and performance
    batch_size: int = 20  # Base batch size for API calls
    max_concurrent_requests: int = 15  # Maximum concurrent API requests
    async_concurrency_limit: int = 16  # Async concurrency limit for codeGenerator
    enable_aggressive_parallelism: bool = True  # Enable concurrent processing within batches

    # Processing strategy
    enable_sequential_batch_processing: bool = True  # Process dissimilarity batches sequentially
    enable_sub_batch_processing: bool = True  # Split large batches into sub-batches

    # Monitoring and reporting
    enable_similarity_distribution_analysis: bool = True  # Report similarity statistics
    enable_batch_analytics: bool = True  # Report batch formation statistics
    enable_performance_monitoring: bool = True  # Monitor processing performance

    # Idea sampling settings
    max_ideas_per_cluster: int = 30  # Maximum ideas to include per cluster for LLM processing

    # SharedCodebook settings
    enable_version_tracking: bool = True  # Track codebook versions
    enable_embedding_cache: bool = True  # Cache code embeddings per version
    max_cached_versions: int = 5  # Maximum cached codebook versions

    # Modification leak recovery settings
    enable_concurrent_leak_recovery: bool = True  # Use concurrent batch processing for modification leak recovery
    modification_leak_batch_size: int = 10  # Batch size for concurrent leak recovery

    # Theme extraction probability band settings
    probability_threshold: float = 0.8  # Only include ideas with cluster_probability < this value
    total_sample_budget: int = 30  # Total ideas to sample across all probability bands
    probability_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'inner':  (0.6, 0.8),   # High-ish probability: 0.6 <= prob < 0.8
        'border': (0.4, 0.6),   # Medium probability: 0.4 <= prob < 0.6
        'fringe': (0.0, 0.4),   # Low probability: prob < 0.4
    })
    band_labels: Dict[str, str] = field(default_factory=lambda: {
        'inner':  'inner members',
        'border': 'border members',
        'fringe': 'fringe members',
    })


# Default experimental instance
DEFAULT_CODEDESIGNER_CONFIG_EXP = CodeDesignerConfigExp()
