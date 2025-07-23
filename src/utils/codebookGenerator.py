"""
CODEBOOK GENERATOR - Real-Time Concurrent Architecture
=====================================================

This module implements an automated codebook generation system for survey analysis
with real-time dynamic codebook updates during concurrent processing.

## Processing Architecture

### REAL-TIME CONCURRENT PROCESSING (Default)
   - Clusters processed concurrently within batches
   - **Real-time codebook updates** - new codes immediately available to other clusters
   - Thread-safe synchronization using asyncio.Lock
   - Each cluster sees the most current codebook state when processing

### REAL-TIME PROCESSING FLOW
   ```
   Initial Codebook
        ↓
   [Batch 1] → Cluster A, B, C process concurrently
             → Cluster A adds new code → immediately available to B, C
             → Cluster B adds new code → immediately available to remaining clusters
        ↓
   [Batch 2] → All clusters use updated codebook with codes from Batch 1
   ```

### BATCH CONCURRENT PROCESSING
   - Updates codebook only between batches (not during concurrent processing)
   - Available via generate_batch_concurrent() method

### FULLY CONCURRENT PROCESSING (Legacy)
   - All clusters processed simultaneously with static initial codebook
   - Available via generate_fully_concurrent() method

### KEY COMPONENTS
   - InductiveCodebookGenerator: Main orchestrator with real-time synchronization
   - CodebookEmbeddingManager: Dynamic embedding caching with hash-based snapshots
   - CodebookDataProcessor: Handles data preparation
   - VerboseReporter: Progress reporting

### PERFORMANCE OPTIMIZATIONS
   - Embeddings computed once and cached with dynamic snapshot IDs
   - Thread-safe concurrent processing with semaphore-based rate limiting
   - Efficient similarity-based code selection using embeddings
   - Smart logging to avoid repetitive cache messages
   - Hash-based snapshot caching for dynamic codebook states
"""

import os
import sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import hashlib
import time

from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts import SYSTEM_MESSAGE_CODEBOOK, INITIAL_CODEBOOK_GENERATION, REVIEW_CODEBOOK_GENERATION
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

import logging
logging.getLogger("httpx").disabled = True



@dataclass
class EnhancedCodebookConfig:
    """Configuration for concurrent codebook generation"""
    max_concurrent_requests: int = 10  # Process clusters concurrently




@dataclass
class CodeEmbeddingCache:
    """Cache for code embeddings to avoid re-computing"""
    _cache: Dict[str, np.ndarray]
    _embedding_config: EmbeddingConfig
    
    def __init__(self):
        self._cache = {}
        self._embedding_config = EmbeddingConfig()
    
    def _get_code_hash(self, code_text: str) -> str:
        """Generate hash for a code text"""
        return hashlib.md5(code_text.encode('utf-8')).hexdigest()
    
    def get_cached_embedding(self, code_text: str) -> Optional[np.ndarray]:
        """Get cached embedding for a code text"""
        code_hash = self._get_code_hash(code_text)
        return self._cache.get(code_hash)
    
    def cache_embedding(self, code_text: str, embedding: np.ndarray):
        """Cache an embedding for a code text"""
        code_hash = self._get_code_hash(code_text)
        self._cache[code_hash] = embedding.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_codes': len(self._cache),
            'cache_size_mb': sum(emb.nbytes for emb in self._cache.values()) / 1024 / 1024
        }
    
    async def get_embeddings_with_cache(self, code_texts: List[str]) -> List[np.ndarray]:
        """Get embeddings, using cache when possible and batch-embedding new ones"""
        cached_embeddings = []
        new_texts = []
        new_indices = []
        
        # Check cache for each code text
        for i, code_text in enumerate(code_texts):
            cached_emb = self.get_cached_embedding(code_text)
            if cached_emb is not None:
                cached_embeddings.append((i, cached_emb))
            else:
                new_texts.append(code_text)
                new_indices.append(i)
        
        # Embed new texts in batch if any
        new_embeddings = []
        if new_texts:
            try:
                client = AsyncOpenAI(api_key=os.environ.get(OPENAI_API_KEY))
                response = await client.embeddings.create(
                    model=self._embedding_config.embedding_model, 
                    input=new_texts
                )
                
                # Process and cache new embeddings
                for j, embedding_data in enumerate(response.data):
                    embedding = np.array(embedding_data.embedding, dtype=np.float32)
                    code_text = new_texts[j]
                    
                    # Cache the embedding
                    self.cache_embedding(code_text, embedding)
                    new_embeddings.append((new_indices[j], embedding))
                    
            except Exception as e:
                logger.error(f"Error embedding new codes: {str(e)}")
                # Return partial results with cached embeddings only
                if not cached_embeddings:
                    return []
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(code_texts)
        
        for idx, emb in cached_embeddings:
            all_embeddings[idx] = emb
        
        for idx, emb in new_embeddings:
            all_embeddings[idx] = emb
        
        # Filter out None values (failed embeddings)
        result = [emb for emb in all_embeddings if emb is not None]
        
        return result


@dataclass
class CodebookSnapshot:
    """Represents a snapshot of the codebook at a specific point"""
    codes: List[Dict[str, str]]
    embeddings: List[np.ndarray]
    snapshot_id: int
    
    def get_code_texts(self) -> List[str]:
        """Get formatted code texts for this snapshot"""
        return [f"{code['code']}: {code['definition']}" for code in self.codes]


class CodebookEmbeddingManager:
    """Manages embeddings for codebook snapshots with efficient caching and batching"""
    
    def __init__(self, verbose: bool = False):
        self.cache = CodeEmbeddingCache()
        self.snapshots: Dict[int, CodebookSnapshot] = {}
        self.verbose = verbose
        self._logged_snapshots: set = set()  # Track which snapshots we've logged about
    
    async def get_snapshot_embeddings(self, codebook: List[Dict[str, str]], snapshot_id: int) -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
        """Get embeddings for a codebook snapshot, using cache when possible"""
        
        # For real-time updates, use codebook size as snapshot identifier to handle dynamic changes
        codebook_hash = self._get_codebook_hash(codebook)
        effective_snapshot_id = f"{snapshot_id}_{codebook_hash}"
        
        # Check if we already have this exact snapshot
        if effective_snapshot_id in self.snapshots:
            snapshot = self.snapshots[effective_snapshot_id]
            if self.verbose and effective_snapshot_id not in self._logged_snapshots:
                logger.info(f"Using cached snapshot {effective_snapshot_id} with {len(snapshot.codes)} codes")
                self._logged_snapshots.add(effective_snapshot_id)
            return snapshot.codes, snapshot.embeddings
        
        # Get code texts
        code_texts = [f"{code['code']}: {code['definition']}" for code in codebook]
        
        # Suppress repetitive embedding messages unless explicitly debugging
        if self.verbose and os.environ.get('CODEBOOK_DEBUG_EMBEDDINGS', '').lower() == 'true':
            cache_stats = self.cache.get_cache_stats()
            logger.info(f"Getting embeddings for {len(code_texts)} codes. Cache: {cache_stats['cached_codes']} codes")
        
        # Get embeddings (cached + new)
        embeddings = await self.cache.get_embeddings_with_cache(code_texts)
        
        if len(embeddings) != len(code_texts):
            logger.warning(f"Only got {len(embeddings)} embeddings for {len(code_texts)} codes")
            return codebook, embeddings
        
        # Cache the snapshot with effective ID
        snapshot = CodebookSnapshot(
            codes=codebook.copy(),
            embeddings=embeddings,
            snapshot_id=snapshot_id
        )
        self.snapshots[effective_snapshot_id] = snapshot
        
        if self.verbose and os.environ.get('CODEBOOK_DEBUG_EMBEDDINGS', '').lower() == 'true':
            cache_stats = self.cache.get_cache_stats()
            logger.info(f"Cached snapshot {effective_snapshot_id}. Total cache: {cache_stats['cached_codes']} codes")
        
        return codebook, embeddings
    
    def _get_codebook_hash(self, codebook: List[Dict[str, str]]) -> str:
        """Generate a hash for the current codebook state to handle dynamic changes"""
        codebook_str = "|".join([f"{code['code']}:{code['definition']}" for code in codebook])
        return str(hash(codebook_str))
    
    async def batch_embed_snapshots(self, snapshots_to_embed: List[Tuple[List[Dict[str, str]], int]]) -> Dict[int, Tuple[List[Dict[str, str]], List[np.ndarray]]]:
        """Batch embed multiple codebook snapshots efficiently"""
        results = {}
        
        if self.verbose:
            logger.info(f"Batch embedding {len(snapshots_to_embed)} snapshots")
        
        # Process all snapshots
        for codebook, snapshot_id in snapshots_to_embed:
            codes, embeddings = await self.get_snapshot_embeddings(codebook, snapshot_id)
            results[snapshot_id] = (codes, embeddings)
        
        return results
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding manager"""
        cache_stats = self.cache.get_cache_stats()
        return {
            'cached_snapshots': len(self.snapshots),
            'cache_stats': cache_stats,
            'total_memory_mb': cache_stats['cache_size_mb']
        }






    


class CodebookDataProcessor:
    """Handles data preparation for codebook generation"""
    
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 embedded_text: List[models.EmbeddingsModel],
                 starter_codes: List[Dict[str, str]], 
                 var_lab: str, 
                 k: int = 5):
        
        self.language = DEFAULT_LANGUAGE
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text  
        self.codebook = starter_codes.copy()  # Growing codebook
        self.var_lab = var_lab
        self.k = k
        
    def prepare_cluster_text(self) -> Dict[int, Dict]:
        """Prepare cluster data with ideas and embeddings"""
        # Create embedding map from embedded_text
        embedding_map = {}
        for result in self.embedded_text:
            if hasattr(result, 'idea_embeddings') and result.idea_embeddings:
                for idea in result.idea_embeddings:
                    embedding_map[idea.idea_id] = {
                        'idea': idea.idea,
                        'embedding': idea.idea_embedding
                    }
        
        # Group ideas by cluster
        clusters = {}
        total_ideas = 0
        missing_embeddings = 0
        
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
                        total_ideas += 1
                    else:
                        missing_embeddings += 1
        
        # Filter out empty clusters
        clusters = {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
        
        if missing_embeddings > 0:
            logger.warning(f"Missing embeddings for {missing_embeddings} ideas")
        
        return clusters
            
    async def embed_codebook_texts(self, code_texts: List[str]) -> List[np.ndarray]:
        """Embed codebook texts for similarity comparison"""
        try:
            embedding_config = EmbeddingConfig()
            client = AsyncOpenAI(api_key=os.environ.get(OPENAI_API_KEY))
            
            response = await client.embeddings.create(
                model=embedding_config.embedding_model, 
                input=code_texts
            )
            
            embeddings = []
            for embedding_data in response.data:
                embeddings.append(np.array(embedding_data.embedding, dtype=np.float32))
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding codebook: {str(e)}")
            return []
    
    def find_k_nearest_codes(self, cluster_embedding: np.ndarray, codebook_embeddings: List[np.ndarray]) -> List[Dict]:
        """Find k nearest codes to a cluster embedding"""
        if not codebook_embeddings:
            return []
            
        codebook_array = np.array(codebook_embeddings)
        similarities = cosine_similarity(cluster_embedding.reshape(1, -1), codebook_array)[0]
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        
        seen = set()
        nearest_codes = []
        
        for idx in top_k_indices:
            if idx < len(self.codebook):
                code = self.codebook[idx]
                code_text = code.get('code', '')
                
                if code_text not in seen:
                    seen.add(code_text)
                    nearest_codes.append(code)
                    
                    if len(nearest_codes) >= self.k:
                        break
                
        return nearest_codes


class InductiveCodebookGenerator:
    """Main class for inductive codebook generation using iterative GATOS methodology"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel], 
        embedded_text: List[models.EmbeddingsModel],
        starter_codes: List[Dict[str, str]], 
        var_lab: str, 
        k: int = 5,
        verbose: bool = False, 
        prompt_printer = None,
        batch_size: int = 5,  # Process clusters in batches
        max_concurrent_requests: int = 3,  # Limit concurrent API calls
        config: Optional[EnhancedCodebookConfig] = None  # Enhanced configuration
    ):
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize enhanced configuration
        self.config = config or EnhancedCodebookConfig()
        # Override with legacy parameters if provided
        if max_concurrent_requests != 3:
            self.config.max_concurrent_requests = max_concurrent_requests
        
        # Initialize model config
        self.model_config = ModelConfig()
        
        # Initialize data processor
        self.data_processor = CodebookDataProcessor(
            cluster_results=cluster_results,
            embedded_text=embedded_text, 
            starter_codes=starter_codes,
            var_lab=var_lab,
            k=k
        )
        
        # Initialize embedding manager for efficient caching
        self.embedding_manager = CodebookEmbeddingManager(verbose=verbose)
        
        
        # Pre-embed the initial codebook for better performance
        self._pre_embed_initial_codebook = True
        
        # Track statistics
        self.stats = {
            'total_clusters': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'errors': 0,
            'embedding_cache_hits': 0,
            'embedding_api_calls': 0,
            'retries': 0,
            'batch_failures': 0
        }
        
        
    async def _call_llm_for_code_generation(self, code_text: str, cluster_text: str) -> Dict[str, Any]:
        """Call LLM to determine if new code is needed"""
        # Initialize LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("initial_codes"),
            temperature=0.0
        )
        
        # Create prompts
        system_message = SYSTEM_MESSAGE_CODEBOOK.format(language=DEFAULT_LANGUAGE)
        
        # Stage 1: Initial code generation
        initial_prompt_text = INITIAL_CODEBOOK_GENERATION.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            code_text=code_text,
            cluster_text=cluster_text,
            **{"data type": "survey responses"}
        )
        
        initial_prompt = PromptTemplate.from_template(system_message + "\n\n" + initial_prompt_text)
        
        # Get initial suggestion
        initial_chain = initial_prompt | llm | StrOutputParser()
        initial_response = await initial_chain.ainvoke({})
        
        # Parse initial response
        code = None
        definition = None
        lines = initial_response.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith("Code:"):
                code = line.replace("Code:", "").strip()
            elif line.strip().startswith("Definition:"):
                definition = line.replace("Definition:", "").strip()
        
        # Stage 2: Review the suggestion
        if code and definition:
            review_code_text = f"Suggested new code:\n- {code}: {definition}\n\nExisting codes:\n{code_text}"
        else:
            review_code_text = f"No new code suggested.\n\nExisting codes:\n{code_text}"
        
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_stage("review_codes"),
            temperature=0.0
        )
        
        review_prompt_text = REVIEW_CODEBOOK_GENERATION.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            code_text=review_code_text,
            cluster_text=cluster_text
        )
        
        review_prompt = PromptTemplate.from_template(system_message + "\n\n" + review_prompt_text)
        
        # Get final decision
        review_chain = review_prompt | llm | StrOutputParser()
        final_response = await review_chain.ainvoke({})
        
        # Parse final response
        needs_new_code = False
        final_code = None
        final_definition = None
        
        # Look for the logical recommendation
        if "no new codes needed" in final_response.lower():
            needs_new_code = False
        else:
            # Extract code and definition from final response
            lines = final_response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("Code:"):
                    final_code = line.replace("Code:", "").strip()
                    needs_new_code = True
                elif line.strip().startswith("Definition:"):
                    final_definition = line.replace("Definition:", "").strip()
        
        return {
            'needs_new_code': needs_new_code,
            'code': final_code,
            'definition': final_definition,
            'initial_response': initial_response,
            'final_response': final_response
        }
    
    
    
    
    async def _process_cluster_with_semaphore(self, cluster_id: int, cluster_data: Dict, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process cluster with semaphore for rate limiting"""
        async with semaphore:
            return await self._process_cluster_legacy(cluster_id, cluster_data)
    
    async def _process_cluster_with_current_codebook(self, cluster_id: int, cluster_data: Dict, snapshot_id: int, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process cluster with current dynamic codebook and semaphore for rate limiting"""
        async with semaphore:
            try:
                # Use current codebook snapshot (includes new codes from previous batches)
                codes, codebook_embeddings = await self.embedding_manager.get_snapshot_embeddings(
                    self.data_processor.codebook, snapshot_id
                )
                
                if not codebook_embeddings:
                    return {
                        'cluster_id': cluster_id,
                        'status': 'embedding_error',
                        'needs_new_code': False
                    }
                
                # Process with current dynamic codebook
                cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
                nearest_codes = self.data_processor.find_k_nearest_codes(cluster_embedding, codebook_embeddings)
                
                cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
                code_text = "\n".join([
                    f"- {code['code']}: {code['definition']}" 
                    for code in nearest_codes
                ]) if nearest_codes else "No existing codes in codebook"
                
                result = await self._call_llm_for_code_generation(code_text, cluster_text)
                
                return {
                    'cluster_id': cluster_id,
                    'status': 'success',
                    'needs_new_code': result['needs_new_code'],
                    'code': result.get('code'),
                    'definition': result.get('definition')
                }
                
            except Exception as e:
                logger.error(f"Error in dynamic cluster processing {cluster_id}: {str(e)}")
                return {
                    'cluster_id': cluster_id,
                    'status': 'error',
                    'needs_new_code': False,
                    'error': str(e)
                }
    
    async def _process_cluster_with_realtime_updates(self, cluster_id: int, cluster_data: Dict, codebook_lock: asyncio.Lock, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process cluster with real-time codebook updates and thread-safe synchronization"""
        async with semaphore:
            try:
                # Get current codebook state (thread-safe)
                async with codebook_lock:
                    current_codebook = self.data_processor.codebook.copy()
                    # Generate a unique snapshot ID based on current codebook size
                    snapshot_id = len(current_codebook)
                
                # Get embeddings for current codebook
                codes, codebook_embeddings = await self.embedding_manager.get_snapshot_embeddings(
                    current_codebook, snapshot_id
                )
                
                if not codebook_embeddings:
                    return {
                        'cluster_id': cluster_id,
                        'status': 'embedding_error',
                        'added_new_code': False
                    }
                
                # Process with current codebook
                cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
                nearest_codes = self.data_processor.find_k_nearest_codes(cluster_embedding, codebook_embeddings)
                
                cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
                code_text = "\n".join([
                    f"- {code['code']}: {code['definition']}" 
                    for code in nearest_codes
                ]) if nearest_codes else "No existing codes in codebook"
                
                result = await self._call_llm_for_code_generation(code_text, cluster_text)
                
                # Handle real-time codebook update if new code is needed
                added_new_code = False
                assigned_code = "existing_code"
                
                if result['needs_new_code'] and result.get('code'):
                    # Thread-safe codebook update
                    async with codebook_lock:
                        new_code = {
                            'code': result['code'],
                            'definition': result['definition']
                        }
                        
                        # Check if code already exists (avoid duplicates)
                        existing_codes = [c['code'] for c in self.data_processor.codebook]
                        if result['code'] not in existing_codes:
                            self.data_processor.codebook.append(new_code)
                            added_new_code = True
                            assigned_code = result['code']
                            
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: Added new code '{result['code']}' (codebook now has {len(self.data_processor.codebook)} codes)")
                        else:
                            assigned_code = result['code']
                            if self.verbose:
                                logger.info(f"Cluster {cluster_id}: Code '{result['code']}' already exists, not adding duplicate")
                
                return {
                    'cluster_id': cluster_id,
                    'status': 'success',
                    'added_new_code': added_new_code,
                    'assigned_code': assigned_code
                }
                
            except Exception as e:
                logger.error(f"Error in real-time cluster processing {cluster_id}: {str(e)}")
                return {
                    'cluster_id': cluster_id,
                    'status': 'error',
                    'added_new_code': False,
                    'error': str(e)
                }
    
    async def _process_cluster_legacy(self, cluster_id: int, cluster_data: Dict) -> Dict[str, Any]:
        """Legacy cluster processing without shared state"""
        try:
            # Use initial codebook snapshot
            codes, codebook_embeddings = await self.embedding_manager.get_snapshot_embeddings(
                self.data_processor.codebook, 0
            )
            
            if not codebook_embeddings:
                return {
                    'cluster_id': cluster_id,
                    'status': 'embedding_error',
                    'needs_new_code': False
                }
            
            # Process with static codebook
            cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
            nearest_codes = self.data_processor.find_k_nearest_codes(cluster_embedding, codebook_embeddings)
            
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            code_text = "\n".join([
                f"- {code['code']}: {code['definition']}" 
                for code in nearest_codes
            ]) if nearest_codes else "No existing codes in codebook"
            
            result = await self._call_llm_for_code_generation(code_text, cluster_text)
            
            return {
                'cluster_id': cluster_id,
                'status': 'success',
                'needs_new_code': result['needs_new_code'],
                'code': result.get('code'),
                'definition': result.get('definition')
            }
            
        except Exception as e:
            logger.error(f"Error in legacy cluster processing {cluster_id}: {str(e)}")
            return {
                'cluster_id': cluster_id,
                'status': 'error',
                'needs_new_code': False,
                'error': str(e)
            }
    
    
    
    
    
    
    
    
    
    async def generate_async_realtime_concurrent(self) -> Dict[str, Any]:
        """Process clusters with real-time codebook updates during concurrent processing"""
        verbose_reporter = VerboseReporter(self.verbose)
        verbose_reporter.header("REAL-TIME CONCURRENT CODEBOOK GENERATION")
        verbose_reporter.info("Processing clusters concurrently with real-time codebook updates")
        
        # Get cluster data  
        clusters = self.data_processor.prepare_cluster_text()
        if not clusters:
            return {'codebook': self.data_processor.codebook, 'cluster_assignments': {}}
        
        self.stats['total_clusters'] = len(clusters)
        start_time = time.time()
        
        # Convert clusters dict to list for batching
        cluster_items = list(clusters.items())
        total_clusters = len(cluster_items)
        
        # Shared state for real-time updates
        codebook_lock = asyncio.Lock()
        cluster_to_code = {}
        
        # Process in batches with real-time updates
        batch_num = 0
        
        for i in range(0, total_clusters, self.batch_size):
            batch_num += 1
            batch_clusters = cluster_items[i:i + self.batch_size]
            batch_start_time = time.time()
            
            verbose_reporter.info(f"Processing batch {batch_num}: clusters {i+1}-{min(i+self.batch_size, total_clusters)} of {total_clusters} (real-time updates)")
            
            # Process batch concurrently with shared state
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            tasks = []
            
            for cluster_id, cluster_data in batch_clusters:
                task = self._process_cluster_with_realtime_updates(cluster_id, cluster_data, codebook_lock, semaphore)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results
            batch_new_codes = 0
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.stats['errors'] += 1
                    continue
                    
                cluster_id = result['cluster_id']
                if result['status'] == 'success':
                    cluster_to_code[cluster_id] = result.get('assigned_code', 'existing_code')
                    if result.get('added_new_code', False):
                        batch_new_codes += 1
                        self.stats['new_codes_added'] += 1
                    else:
                        self.stats['no_new_codes_needed'] += 1
                else:
                    cluster_to_code[cluster_id] = result['status']
                    self.stats['errors'] += 1
            
            batch_time = time.time() - batch_start_time
            
            if batch_new_codes > 0:
                verbose_reporter.info(f"Batch {batch_num} complete: {batch_new_codes} new codes added during processing. Codebook now has {len(self.data_processor.codebook)} codes. Time: {batch_time:.2f}s")
            else:
                verbose_reporter.info(f"Batch {batch_num} complete: No new codes needed. Time: {batch_time:.2f}s")
        
        processing_time = time.time() - start_time
        
        verbose_reporter.summary("REAL-TIME CONCURRENT GENERATION COMPLETE", {
            "Initial codes": len(self.starter_codes),
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.data_processor.codebook),
            "Batches processed": batch_num,
            "Clusters processed": len(clusters),
            "Processing time (s)": f"{processing_time:.2f}"
        })
        
        return {
            'codebook': self.data_processor.codebook,
            'cluster_assignments': cluster_to_code,
            'stats': self.stats
        }

    async def generate_async_batch_concurrent(self) -> Dict[str, Any]:
        """Process clusters in batches with concurrent processing within batches and dynamic codebook updates between batches"""
        verbose_reporter = VerboseReporter(self.verbose)
        verbose_reporter.header("BATCH CONCURRENT CODEBOOK GENERATION")
        verbose_reporter.info("Processing clusters in batches with dynamic codebook updates between batches")
        
        # Get cluster data  
        clusters = self.data_processor.prepare_cluster_text()
        if not clusters:
            return {'codebook': self.data_processor.codebook, 'cluster_assignments': {}}
        
        self.stats['total_clusters'] = len(clusters)
        start_time = time.time()
        
        # Convert clusters dict to list for batching
        cluster_items = list(clusters.items())
        total_clusters = len(cluster_items)
        
        # Process in batches
        cluster_to_code = {}
        batch_num = 0
        current_snapshot_id = 0
        
        for i in range(0, total_clusters, self.batch_size):
            batch_num += 1
            batch_clusters = cluster_items[i:i + self.batch_size]
            batch_start_time = time.time()
            
            verbose_reporter.info(f"Processing batch {batch_num}: clusters {i+1}-{min(i+self.batch_size, total_clusters)} of {total_clusters}")
            
            # Process batch concurrently
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            tasks = []
            
            for cluster_id, cluster_data in batch_clusters:
                task = self._process_cluster_with_current_codebook(cluster_id, cluster_data, current_snapshot_id, semaphore)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results and update codebook
            batch_new_codes = []
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.stats['errors'] += 1
                    continue
                    
                cluster_id = result['cluster_id']
                if result['status'] == 'success':
                    if result['needs_new_code'] and result.get('code'):
                        cluster_to_code[cluster_id] = result['code']
                        self.stats['new_codes_added'] += 1
                        # Add new code to codebook after batch completes
                        new_code = {
                            'code': result['code'],
                            'definition': result['definition']
                        }
                        batch_new_codes.append(new_code)
                        self.data_processor.codebook.append(new_code)
                    else:
                        cluster_to_code[cluster_id] = "existing_code"
                        self.stats['no_new_codes_needed'] += 1
                else:
                    cluster_to_code[cluster_id] = result['status']
                    self.stats['errors'] += 1
            
            batch_time = time.time() - batch_start_time
            
            # Update snapshot ID for next batch if codebook changed
            if batch_new_codes:
                current_snapshot_id += 1
                verbose_reporter.info(f"Batch {batch_num} complete: {len(batch_new_codes)} new codes added. Codebook now has {len(self.data_processor.codebook)} codes. Time: {batch_time:.2f}s")
            else:
                verbose_reporter.info(f"Batch {batch_num} complete: No new codes needed. Time: {batch_time:.2f}s")
        
        processing_time = time.time() - start_time
        
        verbose_reporter.summary("BATCH CONCURRENT GENERATION COMPLETE", {
            "Initial codes": len(self.starter_codes),
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.data_processor.codebook),
            "Batches processed": batch_num,
            "Clusters processed": len(clusters),
            "Processing time (s)": f"{processing_time:.2f}"
        })
        
        return {
            'codebook': self.data_processor.codebook,
            'cluster_assignments': cluster_to_code,
            'stats': self.stats
        }

    async def generate_async_fully_concurrent(self) -> Dict[str, Any]:
        """Process all clusters concurrently (fastest but no dynamic code addition)"""
        verbose_reporter = VerboseReporter(self.verbose)
        verbose_reporter.header("FULLY CONCURRENT CODEBOOK GENERATION")
        verbose_reporter.warning("Note: No dynamic code addition between clusters in this mode")
        
        # Get cluster data  
        clusters = self.data_processor.prepare_cluster_text()
        if not clusters:
            return {'codebook': self.data_processor.codebook, 'cluster_assignments': {}}
        
        self.stats['total_clusters'] = len(clusters)
        start_time = time.time()
        
        # Use static codebook (no dynamic updates)
        static_codebook = self.data_processor.codebook.copy()
        
        # Process all clusters concurrently with rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        tasks = []
        for cluster_id, cluster_data in clusters.items():
            task = self._process_cluster_with_semaphore(cluster_id, cluster_data, semaphore)
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        cluster_to_code = {}
        new_codes = []
        
        for result in results:
            if isinstance(result, Exception):
                self.stats['errors'] += 1
                continue
                
            cluster_id = result['cluster_id']
            if result['status'] == 'success':
                if result['needs_new_code'] and result.get('code'):
                    cluster_to_code[cluster_id] = result['code']
                    self.stats['new_codes_added'] += 1
                    # Collect new codes (but don't use for other clusters)
                    new_codes.append({
                        'code': result['code'],
                        'definition': result['definition']
                    })
                else:
                    cluster_to_code[cluster_id] = "existing_code"
                    self.stats['no_new_codes_needed'] += 1
            else:
                cluster_to_code[cluster_id] = result['status']
                self.stats['errors'] += 1
        
        # Update final codebook with all new codes
        final_codebook = static_codebook + new_codes
        self.data_processor.codebook = final_codebook
        
        processing_time = time.time() - start_time
        
        verbose_reporter.summary("FULLY CONCURRENT GENERATION COMPLETE", {
            "Initial codes": len(static_codebook),
            "New codes added": len(new_codes),
            "Final codebook size": len(final_codebook),
            "Clusters processed": len(clusters),
            "Processing time (s)": f"{processing_time:.2f}"
        })
        
        return {
            'codebook': final_codebook,
            'cluster_assignments': cluster_to_code,
            'stats': self.stats
        }
    
    def generate(self) -> Dict[str, Any]:
        """Generate codebook using real-time concurrent processing with dynamic updates"""
        async def run_generation():
            result = await self.generate_async_realtime_concurrent()
            
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': result.get('cluster_assignments', {}),
                'stats': {
                    'initial_codes': len(self.starter_codes),
                    'new_codes': self.stats['new_codes_added'],
                    'total_codes': len(self.data_processor.codebook),
                    'clusters_processed': self.stats['total_clusters'],
                    'no_new_codes_needed': self.stats['no_new_codes_needed'],
                    'errors': self.stats['errors']
                }
            }
        
        return asyncio.run(run_generation())
    
    def generate_batch_concurrent(self) -> Dict[str, Any]:
        """Generate codebook using batch concurrent processing (updates between batches only)"""
        async def run_generation():
            result = await self.generate_async_batch_concurrent()
            
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': result.get('cluster_assignments', {}),
                'stats': {
                    'initial_codes': len(self.starter_codes),
                    'new_codes': self.stats['new_codes_added'],
                    'total_codes': len(self.data_processor.codebook),
                    'clusters_processed': self.stats['total_clusters'],
                    'no_new_codes_needed': self.stats['no_new_codes_needed'],
                    'errors': self.stats['errors']
                }
            }
        
        return asyncio.run(run_generation())
    
    def generate_fully_concurrent(self) -> Dict[str, Any]:
        """Generate codebook using fully concurrent processing (legacy mode - no dynamic updates)"""
        async def run_generation():
            result = await self.generate_async_fully_concurrent()
            
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': result.get('cluster_assignments', {}),
                'stats': {
                    'initial_codes': len(self.starter_codes),
                    'new_codes': self.stats['new_codes_added'],
                    'total_codes': len(self.data_processor.codebook),
                    'clusters_processed': self.stats['total_clusters'],
                    'no_new_codes_needed': self.stats['no_new_codes_needed'],
                    'errors': self.stats['errors']
                }
            }
        
        return asyncio.run(run_generation())