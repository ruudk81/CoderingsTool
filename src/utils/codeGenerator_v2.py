
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tenacity import retry
from pydantic import BaseModel, Field, ConfigDict

# Project imports
from config import DEFAULT_LANGUAGE #, ModelConfig as ConfigModelConfig
from utils.codeGenerator import FAST_API_RETRY_CONFIG, SharedCodebook, OptimizedEmbeddingManager #FAST_EMBEDDING_RETRY_CONFIG
from prompts import (
    CLUSTER_SUMMARY_PROMPT,
    CANDIDATE_CODE_SELECTION_PROMPT, 
    CODE_GENERATION_PROMPT,
    VALIDATION_PROMPT
)
from models import (
    #ClusterThemeAnalysis,
    CandidateCode,
    CodeRecommendation, 
    ValidationResult,
    CodeGeneratorReasoningResults
)
from pydantic import RootModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure INFO level messages are processed

# ============================================================================
# LANGCHAIN COMPATIBILITY MODELS
# ============================================================================

class ThemeEntry(BaseModel):
    """Theme entry for structured prompt output"""
    theme_id: int = Field(description="Sequential theme ID starting from 1")
    theme_name: str = Field(description="Short noun phrase for theme name")
    summary: str = Field(description="Theme summary in ≤25 words")

class ClusterSummaryOutput(RootModel[List[ThemeEntry]]):
    """Step 1 LangChain output model"""
    root: List[ThemeEntry] = Field(description="Array of themes with ID, name, and summary")

class CandidateCodeSelectionOutput(RootModel[List[CandidateCode]]):
    """Step 2 LangChain output model"""
    root: List[CandidateCode] = Field(description="Selected candidate codes")

# ============================================================================
# INTER-STAGE DATA MODELS
# ============================================================================

class ThemeCompact(BaseModel):
    """Compact theme representation to reduce token load"""
    theme_id: int
    theme_name: str  # Only name, no full summary for Phase 2
    embedding_key: str  # Reference to cached embedding

class ClusterStageAResult(BaseModel):
    """Results from Stage A (Phases 1-2) per cluster"""
    cluster_id: int
    original_cluster_text: str  # Keep for Phase 3+ if needed
    
    # Phase 1 results (compact)
    themes: List[ThemeCompact]
    theme_count: int
    
    # Phase 2 results (IDs only, not full definitions)
    nearest_code_ids: Dict[str, List[str]]  # theme_name -> [code_id1, code_id2, ...]
    embedding_cache_version: int  # Track which codebook version was used
    
    # Tracking data
    phase1_input: Dict[str, Any]
    phase1_processing_time: float
    phase2_processing_time: float
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class InterStageCache(BaseModel):
    """Structured storage between Stage A and Stage B"""
    stage_a_results: Dict[int, ClusterStageAResult]
    initial_codebook_version: int
    total_stage_a_time: float
    processing_metadata: Dict[str, Any]
    
    # Global theme embedding cache
    theme_embeddings: Dict[str, np.ndarray] = {}  # theme_name -> embedding
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class StageBProgress(BaseModel):
    """Track Stage B sliding window progress"""
    completed_clusters: List[int] = []
    active_window: List[int] = []
    pending_clusters: List[int] = []
    codebook_updates_log: List[Dict[str, Any]] = []
    current_batch_start_time: float = 0
    total_batches_processed: int = 0
    
class StageBClusterResult(BaseModel):
    """Results from Stage B processing for one cluster"""
    cluster_id: int
    
    # Phase 3-5 inputs (for perfect tracking)
    phase3_input: Dict[str, Any]  # Candidate selection input
    phase4_input: Dict[str, Any]  # Code generation input  
    phase5_input: Dict[str, Any]  # Validation input
    
    # Phase 3-5 outputs
    selected_candidates: List[CandidateCode]
    code_recommendations: CodeRecommendation
    validation_results: Optional[ValidationResult] = None
    
    # Final results
    final_codes: List[Dict[str, Any]]
    codebook_updates: List[Dict[str, Any]] = []
    
    processing_times: Dict[str, float]
    status: str

# ============================================================================
# OPTIMIZED MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Smart model selection per phase for cost/performance optimization"""
    
    # Phase 1: Complex theme extraction - needs nuanced understanding
    phase1_model: str = "gpt-4.1-mini"  
    phase1_temperature: float = 0.1
    
    # Phase 2: No LLM needed - pure embedding cosine similarity
    
    # Phase 3: Pattern matching for candidate selection
    phase3_model: str = "gpt-4.1-mini"
    phase3_temperature: float = 0.0
    
    # Phase 4: Complex reasoning for recommendations
    phase4_model: str = "gpt-4.1"
    phase4_temperature: float = 0.1
    
    # Phase 5: Focused validation
    phase5_model: str = "gpt-4.1-mini" 
    phase5_temperature: float = 0.0

# ============================================================================
# SLIDING WINDOW THROTTLE MANAGER
# ============================================================================

class SlidingWindowThrottle:
    
    def __init__(self, window_size: int = 3, throttle_interval: float = 60.0):
        self.window_size = window_size
        self.throttle_interval = throttle_interval  # seconds between batches
        self.last_batch_start_time = 0
        self.batch_count = 0
        
    async def get_next_batch(self, pending_clusters: List[int]) -> List[int]:
        """Get next batch of clusters, applying throttle if needed"""
        if not pending_clusters:
            return []
            
        # Apply throttle (except for first batch)
        if self.batch_count > 0:
            elapsed = time.time() - self.last_batch_start_time
            if elapsed < self.throttle_interval:
                wait_time = self.throttle_interval - elapsed
                print(f"⏱️  Throttling: waiting {wait_time:.1f}s before next batch...")  # Force print
                logger.info(f"⏱️  Throttling: waiting {wait_time:.1f}s before next batch...")
                await asyncio.sleep(wait_time)
        
        # Get next batch
        next_batch = pending_clusters[:self.window_size]
        self.last_batch_start_time = time.time()
        self.batch_count += 1
        
        print(f"🔄 Starting batch {self.batch_count}: clusters {next_batch}")  # Force print
        logger.info(f"🔄 Starting batch {self.batch_count}: clusters {next_batch}")
        return next_batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throttling statistics"""
        return {
            'total_batches': self.batch_count,
            'window_size': self.window_size,
            'throttle_interval': self.throttle_interval,
            'total_throttle_time': (self.batch_count - 1) * self.throttle_interval if self.batch_count > 0 else 0
        }

# ============================================================================
# STAGE A PROCESSOR - MASSIVE PARALLEL PROCESSING
# ============================================================================

class StageAProcessor:
    
    def __init__(self, 
                 shared_codebook: SharedCodebook,
                 embedding_manager: OptimizedEmbeddingManager,
                 model_config: ModelConfig,
                 batch_size: int = 20,
                 sub_batch_size: int = 10,
                 var_lab: str = "",
                 verbose: bool = False):
        self.shared_codebook = shared_codebook
        self.embedding_manager = embedding_manager
        self.model_config = model_config
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.var_lab = var_lab
        self.verbose = verbose
        
        # Initialize Phase 1 LangChain components
        self.phase1_llm = ChatOpenAI(
            model=model_config.phase1_model,
            temperature=model_config.phase1_temperature
        )
        
        # Phase 1 chain
        phase1_prompt = PromptTemplate(
            template=CLUSTER_SUMMARY_PROMPT,
            input_variables=["language", "survey_question", "cluster_text"]
        )
        
        self.phase1_chain = (
            phase1_prompt 
            | self.phase1_llm 
            | PydanticOutputParser(pydantic_object=ClusterSummaryOutput)
        )
        
        # Stats tracking
        self.stats = {
            'clusters_processed': 0,
            'themes_extracted': 0,
            'embeddings_generated': 0,
            'cache_hits': 0,
            'api_calls': 0
        }
    
    async def process_stage_a(self, clusters: Dict[int, Dict]) -> InterStageCache:
        """Main Stage A processing: Phases 1-2 for all clusters"""
        start_time = time.time()
        
        logger.info(f"🚀 Stage A: Processing {len(clusters)} clusters (Phases 1-2)")
        
        # Phase 1: Extract themes from all clusters (massive parallel)
        phase1_results = await self._process_phase1_all_clusters(clusters)
        
        # Phase 2: Generate embeddings and find nearest codes (parallel)
        stage_a_results = await self._process_phase2_all_clusters(phase1_results)
        
        total_time = time.time() - start_time
        
        # Create inter-stage cache
        current_codes, codebook_version = await self.shared_codebook.get_current_snapshot()
        
        inter_stage_cache = InterStageCache(
            stage_a_results=stage_a_results,
            initial_codebook_version=codebook_version,
            total_stage_a_time=total_time,
            processing_metadata={
                'phase1_model': self.model_config.phase1_model,
                'batch_size': self.batch_size,
                'sub_batch_size': self.sub_batch_size,
                'stats': self.stats
            }
        )
        
        logger.info(f"✅ Stage A complete: {total_time:.2f}s, {self.stats['themes_extracted']} themes extracted")
        return inter_stage_cache
    
    async def _process_phase1_all_clusters(self, clusters: Dict[int, Dict]) -> Dict[int, ClusterSummaryOutput]:
        """Phase 1: Extract themes from all clusters using hierarchical concurrency"""
        
        # Prepare inputs for batch processing
        cluster_items = [(cluster_id, cluster_data) for cluster_id, cluster_data in clusters.items()]
        
        # Use hierarchical batching for optimal API utilization
        results = {}
        total_batches = (len(cluster_items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(cluster_items), self.batch_size):
            batch_num = batch_idx // self.batch_size + 1
            batch_clusters = cluster_items[batch_idx:batch_idx + self.batch_size]
            
            logger.info(f"Phase 1 - Processing batch {batch_num}/{total_batches} ({len(batch_clusters)} clusters)")
            
            # Process batch with sub-batching
            batch_results = await self._process_phase1_batch(batch_clusters)
            results.update(batch_results)
        
        return results
    
    async def _process_phase1_batch(self, batch_clusters: List[Tuple[int, Dict]]) -> Dict[int, ClusterSummaryOutput]:
        """Process a batch of clusters through Phase 1 with sub-batching"""
        
        # Split into sub-batches for hierarchical processing
        sub_batches = []
        for i in range(0, len(batch_clusters), self.sub_batch_size):
            sub_batch = batch_clusters[i:i + self.sub_batch_size]
            sub_batches.append(sub_batch)
        
        # Process sub-batches concurrently
        sub_batch_tasks = [
            self._process_phase1_sub_batch(sub_batch, i)
            for i, sub_batch in enumerate(sub_batches)
        ]
        
        sub_batch_results = await asyncio.gather(*sub_batch_tasks, return_exceptions=True)
        
        # Collect results
        batch_results = {}
        for sub_result in sub_batch_results:
            if isinstance(sub_result, Exception):
                logger.error(f"Phase 1 sub-batch failed: {sub_result}")
                continue
            batch_results.update(sub_result)
        
        return batch_results
    
    async def _process_phase1_sub_batch(self, sub_batch: List[Tuple[int, Dict]], sub_batch_idx: int) -> Dict[int, ClusterSummaryOutput]:
        """Process a sub-batch of clusters through Phase 1"""
        
        # Process clusters in sub-batch concurrently
        tasks = [
            self._process_single_cluster_phase1(cluster_id, cluster_data)
            for cluster_id, cluster_data in sub_batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        sub_batch_results = {}
        for i, (cluster_id, _) in enumerate(sub_batch):
            if isinstance(results[i], Exception):
                logger.error(f"Phase 1 failed for cluster {cluster_id}: {results[i]}")
                continue
            sub_batch_results[cluster_id] = results[i]
        
        return sub_batch_results
    
    @retry(**FAST_API_RETRY_CONFIG)
    async def _process_single_cluster_phase1(self, cluster_id: int, cluster_data: Dict) -> ClusterSummaryOutput:
        """Process single cluster through Phase 1: Theme extraction"""
        
        # Prepare cluster text with reduced token load
        cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
        
        phase1_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_text": cluster_text
        }
        
        # Call Phase 1 chain
        try:
            self.stats['api_calls'] += 1
            theme_result = await self.phase1_chain.ainvoke(phase1_input)
            
            self.stats['themes_extracted'] += len(theme_result.root) if theme_result.root else 0
            return theme_result
            
        except Exception as e:
            logger.error(f"Phase 1 failed for cluster {cluster_id}: {e}")
            # Return empty result to continue processing
            return ClusterSummaryOutput(root=[])
    
    async def _process_phase2_all_clusters(self, phase1_results: Dict[int, ClusterSummaryOutput]) -> Dict[int, ClusterStageAResult]:
        """Phase 2: Generate embeddings and find nearest codes for all clusters"""
        
        logger.info(f"Phase 2 - Processing embeddings and cosine similarity for {len(phase1_results)} clusters")
        
        # Extract all unique themes for batch embedding generation
        all_themes = set()
        for result in phase1_results.values():
            if result.root:
                all_themes.update([theme.theme_name for theme in result.root if theme.theme_name.strip()])
        
        # Generate embeddings for all unique themes (batch optimization)
        theme_embeddings = {}
        if all_themes:
            theme_list = list(all_themes)
            logger.info(f"Generating embeddings for {len(theme_list)} unique themes...")
            
            embeddings = await self.embedding_manager._embed_texts_with_retry(theme_list)
            theme_embeddings = dict(zip(theme_list, embeddings))
            self.stats['embeddings_generated'] += len(embeddings)
        
        # Process each cluster to find nearest codes (concurrent)
        tasks = [
            self._process_single_cluster_phase2(cluster_id, phase1_result, theme_embeddings)
            for cluster_id, phase1_result in phase1_results.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        stage_a_results = {}
        for i, cluster_id in enumerate(phase1_results.keys()):
            if isinstance(results[i], Exception):
                logger.error(f"Phase 2 failed for cluster {cluster_id}: {results[i]}")
                continue
            stage_a_results[cluster_id] = results[i]
        
        return stage_a_results
    
    async def _process_single_cluster_phase2(self, 
                                           cluster_id: int, 
                                           phase1_result: ClusterSummaryOutput, 
                                           theme_embeddings: Dict[str, np.ndarray]) -> ClusterStageAResult:
        """Process single cluster through Phase 2: Find nearest codes via cosine similarity"""
        start_time = time.time()
        
        if not phase1_result.root:
            # No themes found, return minimal result
            return ClusterStageAResult(
                cluster_id=cluster_id,
                original_cluster_text="",
                themes=[],
                theme_count=0,
                nearest_code_ids={},
                embedding_cache_version=0,
                phase1_input={},
                phase1_processing_time=0,
                phase2_processing_time=time.time() - start_time
            )
        
        # Create compact theme representation
        themes_compact = []
        nearest_code_ids = {}
        
        current_codes, codebook_version = await self.shared_codebook.get_current_snapshot()
        
        for i, theme_entry in enumerate(phase1_result.root):
            theme_name = theme_entry.theme_name.strip()
            if not theme_name:
                continue
                
            theme_compact = ThemeCompact(
                theme_id=i + 1,
                theme_name=theme_name,
                embedding_key=theme_name
            )
            themes_compact.append(theme_compact)
            
            # Find nearest codes using pre-computed embeddings
            if theme_name in theme_embeddings:
                theme_embedding = theme_embeddings[theme_name]
                nearest_codes = await self._find_nearest_codes_for_theme(theme_embedding, current_codes)
                
                # Store only code IDs (not full definitions) for token optimization
                nearest_code_ids[theme_name] = [code['code'] for code in nearest_codes[:5]]  # Top 5
        
        phase2_time = time.time() - start_time
        
        return ClusterStageAResult(
            cluster_id=cluster_id,
            original_cluster_text="",  # Will be filled if needed
            themes=themes_compact,
            theme_count=len(themes_compact),
            nearest_code_ids=nearest_code_ids,
            embedding_cache_version=codebook_version,
            phase1_input={},  # Will be filled from tracking
            phase1_processing_time=0,  # Will be filled from tracking
            phase2_processing_time=phase2_time
        )
    
    async def _find_nearest_codes_for_theme(self, theme_embedding: np.ndarray, current_codes: List[Dict]) -> List[Dict]:
        """Find nearest codes for a theme using cosine similarity"""
        if not current_codes:
            return []
            
        try:
            # Get embeddings for current codes
            codes, embeddings = await self.embedding_manager.get_snapshot_embeddings(
                current_codes, 
                await self.shared_codebook.get_version()
            )
            
            if not embeddings:
                return []
            
            # Calculate cosine similarities
            codebook_array = np.array(embeddings)
            similarities = cosine_similarity(theme_embedding.reshape(1, -1), codebook_array)[0]
            
            # Get top K nearest codes
            k = 10
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            nearest_codes = []
            for idx in top_k_indices:
                if idx < len(codes):
                    code = codes[idx]
                    nearest_codes.append({
                        'code': code['code'],
                        'definition': code['definition'],
                        'similarity': float(similarities[idx])
                    })
            
            return nearest_codes
            
        except Exception as e:
            logger.error(f"Failed to find nearest codes for theme: {e}")
            return []

# ============================================================================
# STAGE B PROCESSOR - THROTTLED SLIDING WINDOW
# ============================================================================

class StageBProcessor:
    """Stage B: Throttled sliding window processing for Phases 3-5 (Decision Making)"""
    
    def __init__(self,
                 shared_codebook: SharedCodebook,
                 embedding_manager: OptimizedEmbeddingManager,
                 model_config: ModelConfig,
                 window_size: int = 3,
                 throttle_interval: float = 60.0,
                 var_lab: str = "",
                 verbose: bool = False):
        self.shared_codebook = shared_codebook
        self.embedding_manager = embedding_manager
        self.model_config = model_config
        self.var_lab = var_lab
        self.verbose = verbose
        
        # Initialize sliding window throttle
        self.throttle = SlidingWindowThrottle(window_size, throttle_interval)
        
        # Initialize LLMs and chains for Phases 3-5 with optimal models
        self.phase3_llm = ChatOpenAI(
            model=model_config.phase3_model,
            temperature=model_config.phase3_temperature
        )
        
        self.phase4_llm = ChatOpenAI(
            model=model_config.phase4_model,
            temperature=model_config.phase4_temperature
        )
        
        self.phase5_llm = ChatOpenAI(
            model=model_config.phase5_model,
            temperature=model_config.phase5_temperature
        )
        
        # Phase 3 chain (Candidate Code Selection)
        phase3_prompt = PromptTemplate(
            template=CANDIDATE_CODE_SELECTION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "code_text"]
        )
        
        self.phase3_chain = (
            phase3_prompt
            | self.phase3_llm
            | PydanticOutputParser(pydantic_object=CandidateCodeSelectionOutput)
        )
        
        # Phase 4 chain (Code Generation)
        phase4_prompt = PromptTemplate(
            template=CODE_GENERATION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "candidate_codes"]
        )
        
        self.phase4_chain = (
            phase4_prompt
            | self.phase4_llm
            | PydanticOutputParser(pydantic_object=CodeRecommendation)
        )
        
        # Phase 5 chain (Validation)
        phase5_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["language", "survey_question", "cluster_summary", "candidate_codes", "step3_recommendation"]
        )
        
        self.phase5_chain = (
            phase5_prompt
            | self.phase5_llm
            | PydanticOutputParser(pydantic_object=ValidationResult)
        )
        
        # Stats tracking
        self.stats = {
            'clusters_processed': 0,
            'codes_created': 0,
            'codes_modified': 0,
            'codes_reused': 0,
            'api_calls_phase3': 0,
            'api_calls_phase4': 0,
            'api_calls_phase5': 0,
            'throttle_time_total': 0
        }
    
    async def process_stage_b(self, inter_stage_cache: InterStageCache) -> Dict[int, StageBClusterResult]:
        """Main Stage B processing: Phases 3-5 with throttled sliding window"""
        start_time = time.time()
        
        pending_clusters = list(inter_stage_cache.stage_a_results.keys())
        total_clusters = len(pending_clusters)
        
        logger.info(f"⏱️  Stage B: Processing {total_clusters} clusters (Phases 3-5) with sliding window")
        logger.info(f"Window size: {self.throttle.window_size}, Throttle: {self.throttle.throttle_interval}s")
        
        results = {}
        progress = StageBProgress(pending_clusters=pending_clusters.copy())
        
        while progress.pending_clusters:
            # Get next throttled batch
            current_batch = await self.throttle.get_next_batch(progress.pending_clusters)
            if not current_batch:
                break
                
            progress.active_window = current_batch
            progress.current_batch_start_time = time.time()
            
            # Process current batch through Phases 3-5
            batch_results = await self._process_batch_phases_3_to_5(current_batch, inter_stage_cache)
            
            # Update results and progress
            results.update(batch_results)
            progress.completed_clusters.extend(current_batch)
            progress.pending_clusters = [c for c in progress.pending_clusters if c not in current_batch]
            progress.total_batches_processed += 1
            
            # Log progress
            completed = len(progress.completed_clusters)
            print(f"✅ Batch {progress.total_batches_processed} complete: {completed}/{total_clusters} clusters processed")  # Force print
            logger.info(f"✅ Batch {progress.total_batches_processed} complete: {completed}/{total_clusters} clusters processed")
        
        total_time = time.time() - start_time
        throttle_stats = self.throttle.get_stats()
        
        logger.info(f"🎯 Stage B complete: {total_time:.2f}s total, {throttle_stats['total_throttle_time']:.2f}s throttling")
        return results
    
    async def _process_batch_phases_3_to_5(self, 
                                         batch_clusters: List[int], 
                                         inter_stage_cache: InterStageCache) -> Dict[int, StageBClusterResult]:
        """Process a batch of clusters through Phases 3-5 concurrently"""
        
        # Process clusters in batch concurrently (within sliding window)
        tasks = [
            self._process_single_cluster_phases_3_to_5(cluster_id, inter_stage_cache)
            for cluster_id in batch_clusters
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results and update SharedCodebook
        batch_results = {}
        codebook_updates = []
        
        for i, cluster_id in enumerate(batch_clusters):
            if isinstance(results[i], Exception):
                logger.error(f"Phases 3-5 failed for cluster {cluster_id}: {results[i]}")
                continue
                
            cluster_result = results[i]
            batch_results[cluster_id] = cluster_result
            
            # Immediately update SharedCodebook with this cluster's results
            if cluster_result.codebook_updates:
                await self._apply_codebook_updates(cluster_result.codebook_updates)
                codebook_updates.extend(cluster_result.codebook_updates)
        
        if codebook_updates and self.verbose:
            logger.info(f"📝 Updated SharedCodebook: {len(codebook_updates)} changes from batch")
        
        return batch_results
    
    async def _process_single_cluster_phases_3_to_5(self, 
                                                   cluster_id: int, 
                                                   inter_stage_cache: InterStageCache) -> StageBClusterResult:
        """Process single cluster through Phases 3-5"""
        start_time = time.time()
        
        stage_a_result = inter_stage_cache.stage_a_results[cluster_id]
        processing_times = {}
        
        try:
            # Phase 3: Candidate code selection
            phase3_start = time.time()
            phase3_input, selected_candidates = await self._process_phase3(cluster_id, stage_a_result)
            processing_times['phase3'] = time.time() - phase3_start
            
            # Phase 4: Code generation recommendations  
            phase4_start = time.time()
            phase4_input, code_recommendations = await self._process_phase4(cluster_id, stage_a_result, selected_candidates)
            processing_times['phase4'] = time.time() - phase4_start
            
            # Phase 5: Validation and codebook update
            phase5_start = time.time()
            phase5_input, validation_results, final_codes, codebook_updates = await self._process_phase5(
                cluster_id, stage_a_result, selected_candidates, code_recommendations
            )
            processing_times['phase5'] = time.time() - phase5_start
            processing_times['total'] = time.time() - start_time
            
            return StageBClusterResult(
                cluster_id=cluster_id,
                phase3_input=phase3_input,
                phase4_input=phase4_input,
                phase5_input=phase5_input,
                selected_candidates=selected_candidates,
                code_recommendations=code_recommendations,
                validation_results=validation_results,
                final_codes=final_codes,
                codebook_updates=codebook_updates,
                processing_times=processing_times,
                status='completed'
            )
            
        except Exception as e:
            logger.error(f"Cluster {cluster_id} failed in Phases 3-5: {e}")
            return StageBClusterResult(
                cluster_id=cluster_id,
                phase3_input={},
                phase4_input={},
                phase5_input={},
                selected_candidates=[],
                code_recommendations=None,
                validation_results=None,
                final_codes=[],
                codebook_updates=[],
                processing_times={'total': time.time() - start_time},
                status='failed'
            )
    
    async def _process_phase3(self, 
                             cluster_id: int, 
                             stage_a_result: ClusterStageAResult) -> Tuple[Dict[str, Any], List[CandidateCode]]:
        """Phase 3: Candidate code selection with fresh codebook"""
        
        # Get CURRENT codebook (includes updates from previous clusters)
        current_codes, codebook_version = await self.shared_codebook.get_current_snapshot()
        
        # Build candidate codes text from Stage A nearest code IDs
        candidate_codes_text = ""
        if stage_a_result.nearest_code_ids:
            candidate_lines = []
            for theme_name, code_ids in stage_a_result.nearest_code_ids.items():
                candidate_lines.append(f"\n# Candidates for theme: {theme_name}")
                for code_id in code_ids:
                    # Find full definition from current codebook
                    for code in current_codes:
                        if code['code'] == code_id:
                            candidate_lines.append(f"- {code['code']}: {code['definition']}")
                            break
            candidate_codes_text = "\n".join(candidate_lines)
        
        if not candidate_codes_text:
            candidate_codes_text = "No candidate codes found"
        
        # Prepare Phase 3 input (token-optimized)
        phase3_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_summary": f"Themes: {', '.join([t.theme_name for t in stage_a_result.themes])}",
            "code_text": candidate_codes_text
        }
        
        # Call Phase 3 chain (candidate selection)
        try:
            self.stats['api_calls_phase3'] += 1
            selected_codes_result = await self.phase3_chain.ainvoke(phase3_input)
            
            return phase3_input, selected_codes_result.root if selected_codes_result.root else []
            
        except Exception as e:
            logger.error(f"Phase 3 failed for cluster {cluster_id}: {e}")
            return phase3_input, []
    
    async def _process_phase4(self, 
                             cluster_id: int, 
                             stage_a_result: ClusterStageAResult,
                             selected_candidates: List[CandidateCode]) -> Tuple[Dict[str, Any], CodeRecommendation]:
        """Phase 4: Code generation recommendations"""
        
        # Build selected codes text (token-optimized)
        selected_codes_text = "\n".join([
            f"- {code.code}: {code.definition}"
            for code in selected_candidates
        ]) if selected_candidates else "No codes selected"
        
        # Prepare Phase 4 input
        phase4_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_summary": f"Themes: {', '.join([t.theme_name for t in stage_a_result.themes])}",
            "candidate_codes": selected_codes_text
        }
        
        # Call Phase 4 chain (code generation)
        try:
            self.stats['api_calls_phase4'] += 1
            code_recommendations = await self.phase4_chain.ainvoke(phase4_input)
            
            return phase4_input, code_recommendations
            
        except Exception as e:
            logger.error(f"Phase 4 failed for cluster {cluster_id}: {e}")
            return phase4_input, None
    
    async def _process_phase5(self, 
                             cluster_id: int,
                             stage_a_result: ClusterStageAResult,
                             selected_candidates: List[CandidateCode], 
                             code_recommendations: CodeRecommendation) -> Tuple[Dict[str, Any], ValidationResult, List[Dict], List[Dict]]:
        """Phase 5: Validation and immediate codebook update"""
        
        if not code_recommendations:
            return {}, None, [], []
        
        # Check if validation is needed
        needs_validation = any(
            decision.decision in ['create_new', 'modify_existing']
            for decision in code_recommendations.coding_decisions
        )
        
        if not needs_validation:
            # All use_existing - no validation needed
            final_codes = self._extract_final_codes_from_recommendations(code_recommendations)
            return {}, None, final_codes, []
        
        # Build recommendation summary for validation (token-optimized)
        recommendation_summary = f"Recommendations for {len(code_recommendations.coding_decisions)} themes:\n"
        for decision in code_recommendations.coding_decisions:
            recommendation_summary += f"- Theme {decision.theme_number}: {decision.decision}\n"
        
        # Build candidate codes text
        candidate_codes_text = "\n".join([
            f"- {code.code}: {code.definition}"
            for code in selected_candidates
        ]) if selected_candidates else "No codes selected"
        
        phase5_input = {
            "language": DEFAULT_LANGUAGE,
            "survey_question": self.var_lab,
            "cluster_summary": f"Themes: {', '.join([t.theme_name for t in stage_a_result.themes])}",
            "candidate_codes": candidate_codes_text,
            "step3_recommendation": recommendation_summary
        }
        
        # Call Phase 5 chain (validation)
        try:
            self.stats['api_calls_phase5'] += 1
            validation_results = await self.phase5_chain.ainvoke(phase5_input)
            
            # Extract final validated codes and prepare codebook updates
            final_codes, codebook_updates = await self._extract_validated_codes_and_updates(
                validation_results, code_recommendations
            )
            
            return phase5_input, validation_results, final_codes, codebook_updates
            
        except Exception as e:
            logger.error(f"Phase 5 failed for cluster {cluster_id}: {e}")
            return phase5_input, None, [], []
    
    def _extract_final_codes_from_recommendations(self, code_recommendations: CodeRecommendation) -> List[Dict]:
        """Extract final codes from recommendations when no validation needed"""
        final_codes = []
        
        for decision in code_recommendations.coding_decisions:
            if decision.decision == 'use_existing' and decision.action_details.codes_to_use:
                for code_name in decision.action_details.codes_to_use:
                    final_codes.append({
                        'theme_number': decision.theme_number,
                        'code': code_name,
                        'definition': '',  # Will be filled from codebook
                        'decision': 'use_existing'
                    })
        
        return final_codes
    
    async def _extract_validated_codes_and_updates(self, 
                                                   validation_results: ValidationResult,
                                                   code_recommendations: CodeRecommendation) -> Tuple[List[Dict], List[Dict]]:
        """Extract validated codes and prepare immediate codebook updates"""
        final_codes = []
        codebook_updates = []
        
        for validation in validation_results.code_validations:
            # Only include codes that were APPROVED, REVISED, or SPLIT
            if validation.decision in ['APPROVE', 'REVISE', 'SPLIT']:
                # Find original recommendation for this theme
                original_decision = 'use_existing'
                for decision in code_recommendations.coding_decisions:
                    if decision.theme_number == validation.theme_number:
                        original_decision = decision.decision
                        break
                
                if validation.decision == 'SPLIT':
                    # Handle SPLIT case - multiple codes
                    if isinstance(validation.validated_code, list):
                        for split_code in validation.validated_code:
                            if split_code and hasattr(split_code, 'code') and split_code.code:
                                final_code = {
                                    'theme_number': validation.theme_number,
                                    'code': split_code.code,
                                    'definition': split_code.definition,
                                    'decision': 'create_new'
                                }
                                final_codes.append(final_code)
                                
                                # Prepare codebook update
                                codebook_updates.append({
                                    'action': 'add',
                                    'code': split_code.code,
                                    'definition': split_code.definition,
                                    'theme_number': validation.theme_number
                                })
                else:
                    # Handle single code case (APPROVE/REVISE)  
                    if validation.validated_code and hasattr(validation.validated_code, 'code'):
                        effective_decision = 'create_new' if validation.decision == 'REVISE' else original_decision
                        
                        final_code = {
                            'theme_number': validation.theme_number,
                            'code': validation.validated_code.code,
                            'definition': validation.validated_code.definition,
                            'decision': effective_decision
                        }
                        final_codes.append(final_code)
                        
                        # Prepare codebook update
                        if effective_decision == 'create_new':
                            codebook_updates.append({
                                'action': 'add',
                                'code': validation.validated_code.code,
                                'definition': validation.validated_code.definition,
                                'theme_number': validation.theme_number
                            })
                        elif effective_decision == 'modify_existing':
                            # Find original code to modify
                            original_code = None
                            for decision in code_recommendations.coding_decisions:
                                if decision.theme_number == validation.theme_number:
                                    original_code = decision.action_details.codes_to_modify
                                    break
                            
                            if original_code:
                                codebook_updates.append({
                                    'action': 'modify',
                                    'original_code': original_code,
                                    'new_code': validation.validated_code.code,
                                    'definition': validation.validated_code.definition,
                                    'theme_number': validation.theme_number
                                })
        
        return final_codes, codebook_updates
    
    async def _apply_codebook_updates(self, codebook_updates: List[Dict]) -> None:
        """Apply codebook updates immediately to SharedCodebook"""
        for update in codebook_updates:
            try:
                if update['action'] == 'add':
                    added, version = await self.shared_codebook.add_code_if_new(
                        update['code'], update['definition']
                    )
                    if added:
                        self.stats['codes_created'] += 1
                        if self.verbose:
                            logger.info(f"🆕 REAL-TIME added '{update['code']}' (v{version}) - available to next clusters")
                
                elif update['action'] == 'modify':
                    replaced, version = await self.shared_codebook.replace_code(
                        update['original_code'], update['new_code'], update['definition']
                    )
                    if replaced:
                        self.stats['codes_modified'] += 1
                        if self.verbose:
                            logger.info(f"🔄 REAL-TIME modified '{update['original_code']}' -> '{update['new_code']}' (v{version})")
                            
            except Exception as e:
                logger.error(f"Failed to apply codebook update: {update}, error: {e}")

# ============================================================================
# HYBRID CODE GENERATOR - MAIN ORCHESTRATOR
# ============================================================================

class HybridCodeGenerator:
    """Main orchestrator for hybrid two-stage code generation"""
    
    def __init__(self, 
                 codebook: List[Dict[str, str]],
                 var_lab: str = "",
                 batch_size: int = 20,
                 sub_batch_size: int = 10,
                 window_size: int = 3,
                 throttle_interval: float = 60.0,
                 verbose: bool = False,
                 model_config: Optional[ModelConfig] = None):
        
        # Initialize core components
        self.shared_codebook = SharedCodebook(codebook)
        self.embedding_manager = OptimizedEmbeddingManager(self.shared_codebook, verbose=verbose)
        self.model_config = model_config or ModelConfig()
        self.var_lab = var_lab
        self.verbose = verbose
        
        # Initialize stage processors
        self.stage_a_processor = StageAProcessor(
            shared_codebook=self.shared_codebook,
            embedding_manager=self.embedding_manager,
            model_config=self.model_config,
            batch_size=batch_size,
            sub_batch_size=sub_batch_size,
            var_lab=var_lab,
            verbose=verbose
        )
        
        self.stage_b_processor = StageBProcessor(
            shared_codebook=self.shared_codebook,
            embedding_manager=self.embedding_manager,
            model_config=self.model_config,
            window_size=window_size,
            throttle_interval=throttle_interval,
            var_lab=var_lab,
            verbose=verbose
        )
        
        # Global stats
        self.stats = {
            'total_clusters': 0,
            'stage_a_time': 0,
            'stage_b_time': 0,
            'total_time': 0
        }
    
    async def process_all_clusters(self, clusters: Dict[int, Dict]) -> CodeGeneratorReasoningResults:
        """Main entry point: Process all clusters through hybrid two-stage architecture"""
        total_start_time = time.time()
        self.stats['total_clusters'] = len(clusters)
        
        print(f"🚀 STARTING Hybrid Code Generation v2 for {len(clusters)} clusters")  # Force print
        print(f"Stage A: Batch size {self.stage_a_processor.batch_size}, Sub-batch {self.stage_a_processor.sub_batch_size}")  # Force print
        print(f"Stage B: Window size {self.stage_b_processor.throttle.window_size}, Throttle {self.stage_b_processor.throttle.throttle_interval}s")  # Force print
        
        logger.info(f"🚀 Starting Hybrid Code Generation v2 for {len(clusters)} clusters")
        logger.info(f"Stage A: Batch size {self.stage_a_processor.batch_size}, Sub-batch {self.stage_a_processor.sub_batch_size}")
        logger.info(f"Stage B: Window size {self.stage_b_processor.throttle.window_size}, Throttle {self.stage_b_processor.throttle.throttle_interval}s")
        
        # STAGE A: Massive parallel processing (Phases 1-2)
        print("🔄 Starting Stage A (Phases 1-2)...")  # Force print
        stage_a_start = time.time()
        inter_stage_cache = await self.stage_a_processor.process_stage_a(clusters)
        self.stats['stage_a_time'] = time.time() - stage_a_start
        print(f"✅ Stage A complete: {self.stats['stage_a_time']:.2f}s")  # Force print
        
        # STAGE B: Throttled sliding window (Phases 3-5)
        print("🔄 Starting Stage B (Phases 3-5)...")  # Force print
        stage_b_start = time.time()
        stage_b_results = await self.stage_b_processor.process_stage_b(inter_stage_cache)
        self.stats['stage_b_time'] = time.time() - stage_b_start
        print(f"✅ Stage B complete: {self.stats['stage_b_time']:.2f}s")  # Force print
        
        self.stats['total_time'] = time.time() - total_start_time
        
        # Build final results compatible with existing pipeline
        final_results = await self._build_final_results(inter_stage_cache, stage_b_results)
        
        logger.info("🎯 Generation")
        logger.info(f"Total time: {self.stats['total_time']:.2f}s (Stage A: {self.stats['stage_a_time']:.2f}s, Stage B: {self.stats['stage_b_time']:.2f}s)")
        
        return final_results
    
    async def _build_final_results(self, 
                                  inter_stage_cache: InterStageCache,
                                  stage_b_results: Dict[int, StageBClusterResult]) -> CodeGeneratorReasoningResults:
        """Build final results compatible with existing pipeline format"""
        
        # Collect all results for CodeGeneratorReasoningResults format
        step1_summaries = {}
        step2_analysis = {}
        step3_recommendations = {}
        step4_validations = {}
        cluster_assignments = {}
        
        # Input tracking for perfect alignment
        step1_inputs = {}
        step2_inputs = {}  # Phase 2 is embedding, no LLM input
        step3_inputs = {}
        step4_inputs = {}
        
        for cluster_id, stage_a_result in inter_stage_cache.stage_a_results.items():
            # Stage A results
            step1_summaries[cluster_id] = {
                'themes': [{'theme_name': t.theme_name} for t in stage_a_result.themes],
                'cluster_summary': f"{stage_a_result.theme_count} themes identified"
            }
            
            step1_inputs[cluster_id] = stage_a_result.phase1_input
            
            # Stage B results (if available)
            if cluster_id in stage_b_results:
                stage_b_result = stage_b_results[cluster_id]
                
                # Step 2 analysis (from Stage A nearest codes)
                step2_analysis[cluster_id] = []
                for theme_name, code_ids in stage_a_result.nearest_code_ids.items():
                    for code_id in code_ids:
                        step2_analysis[cluster_id].append({'code': code_id, 'definition': ''})
                
                # Step 3 recommendations
                if stage_b_result.code_recommendations:
                    step3_recommendations[cluster_id] = {
                        'coding_decisions': [
                            {
                                'theme_number': d.theme_number,
                                'theme_description': d.theme_description,
                                'decision': d.decision,
                                'action_details': d.action_details.dict() if d.action_details else {},
                                'justification': d.justification
                            }
                            for d in stage_b_result.code_recommendations.coding_decisions
                        ]
                    }
                
                # Step 4 validations
                if stage_b_result.validation_results:
                    step4_validations[cluster_id] = {
                        'code_validations': [
                            {
                                'theme_number': v.theme_number,
                                'theme_description': v.theme_description,
                                'decision': v.decision,
                                'decision_rationale': v.decision_rationale,
                                'validated_code': v.validated_code.dict() if hasattr(v.validated_code, 'dict') else v.validated_code
                            }
                            for v in stage_b_result.validation_results.code_validations
                        ]
                    }
                
                # Cluster assignments
                cluster_assignments[cluster_id] = {
                    'codes': stage_b_result.final_codes,
                    'status': stage_b_result.status
                }
                
                # Input tracking
                step3_inputs[cluster_id] = stage_b_result.phase3_input
                step4_inputs[cluster_id] = stage_b_result.phase4_input
        
        # Get final codebook state
        final_codes, final_version = await self.shared_codebook.get_current_snapshot()
        
        # Combine stats
        combined_stats = {
            **self.stats,
            'stage_a_stats': self.stage_a_processor.stats,
            'stage_b_stats': self.stage_b_processor.stats,
            'final_codebook_version': final_version,
            'initial_codes': len(inter_stage_cache.stage_a_results),
            'final_codes': len(final_codes)
        }
        
        # Use the total_ideas_count stored during initialization
        total_ideas = getattr(self, 'total_ideas_count', self.stats['total_clusters'] * 5)
        
        return CodeGeneratorReasoningResults(
            cluster_results=[],  # Not used in v2
            step1_inputs=step1_inputs,
            step2_inputs=step2_inputs,
            step3_inputs=step3_inputs,
            step4_inputs=step4_inputs,
            step1_summaries=step1_summaries,
            step2_analysis=step2_analysis,
            step3_recommendations=step3_recommendations,
            step4_validations=step4_validations,
            cluster_assignments=cluster_assignments,
            stats=combined_stats,
            generator_version="v2_hybrid",
            var_lab=self.var_lab,
            total_clusters=self.stats['total_clusters'],
            total_ideas=total_ideas,
            processing_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

# ============================================================================
# INTEGRATION FUNCTION FOR EXISTING PIPELINE
# ============================================================================

async def generate_codes_with_reasoning_v2(
    initial_clusters_results,  # The structured cluster data from clustering phase
    var_lab: str = "",
    batch_size: int = 20,
    sub_batch_size: int = 10,
    window_size: int = 3,
    throttle_interval: float = 60.0,
    verbose: bool = False,
    initial_codes: Optional[List[Dict[str, str]]] = None  # Optional speculative starter codes
) -> CodeGeneratorReasoningResults:
    """
    Main entry point for Hybrid Code Generation v2
    
    Compatible with existing pipeline interface
    Args:
        initial_clusters_results: Structured cluster data from clustering phase
        initial_codes: Optional speculative starter codes (default: empty)
    """
    
    print("🎯 ENTRY POINT: generate_codes_with_reasoning_v2 called")  # Force print
    print(f"Input: {len(initial_clusters_results) if initial_clusters_results else 0} cluster results")  # Force print
    
    # Extract clusters and initial codebook from structured data
    clusters = {}
    total_ideas_count = 0  # Track total ideas for final results
    
    for cluster_model in initial_clusters_results:
        if hasattr(cluster_model, 'response_ideas') and cluster_model.response_ideas:
            cluster_id = cluster_model.response_ideas[0].initial_cluster
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = {'ideas': []}
                # Extract ideas from response_ideas
                for idea in cluster_model.response_ideas:
                    if idea.initial_cluster == cluster_id:
                        clusters[cluster_id]['ideas'].append(idea.idea)
                        total_ideas_count += 1
    
    # Use initial_codes or empty codebook
    codebook = initial_codes or []
    
    # Initialize hybrid code generator
    generator = HybridCodeGenerator(
        codebook=codebook,
        var_lab=var_lab,
        batch_size=batch_size,
        sub_batch_size=sub_batch_size,
        window_size=window_size,
        throttle_interval=throttle_interval,
        verbose=verbose
    )
    
    # Store total_ideas_count for final results
    generator.total_ideas_count = total_ideas_count
    
    # Process all clusters
    return await generator.process_all_clusters(clusters)

if __name__ == "__main__":
    print("CodeGenerator v2 - Hybrid Two-Stage Architecture Complete! 🚀")