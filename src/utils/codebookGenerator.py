import os
import sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import hashlib
import tiktoken
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
    """Enhanced configuration for efficient codebook generation"""
    max_tokens: int = 4000
    completion_reserve: int = 1000  # Reserve tokens for LLM response
    max_batch_size: int = 10
    min_batch_size: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_exponential_base: float = 2.0
    max_concurrent_batches: int = 5  # Process batches concurrently
    rate_limit_buffer: float = 0.1  # 10% buffer for rate limiting
    
    def __post_init__(self):
        # Ensure sensible defaults
        self.max_batch_size = max(self.min_batch_size, self.max_batch_size)
        self.max_retries = max(0, self.max_retries)


@dataclass
class ClusterBatch:
    """Represents a batch of clusters to process together with token awareness"""
    batch_id: int
    cluster_ids: List[int]
    cluster_data: Dict[int, Dict]  # cluster_id -> {'ideas': [], 'embeddings': []}
    estimated_tokens: int = 0
    codebook_snapshot: Optional[List[Dict]] = None
    snapshot_id: int = 0


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
    
    async def get_snapshot_embeddings(self, codebook: List[Dict[str, str]], snapshot_id: int) -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
        """Get embeddings for a codebook snapshot, using cache when possible"""
        
        # Check if we already have this exact snapshot
        if snapshot_id in self.snapshots:
            snapshot = self.snapshots[snapshot_id]
            if self.verbose:
                logger.info(f"Using cached snapshot {snapshot_id} with {len(snapshot.codes)} codes")
            return snapshot.codes, snapshot.embeddings
        
        # Get code texts
        code_texts = [f"{code['code']}: {code['definition']}" for code in codebook]
        
        if self.verbose:
            cache_stats = self.cache.get_cache_stats()
            logger.info(f"Getting embeddings for {len(code_texts)} codes. Cache: {cache_stats['cached_codes']} codes")
        
        # Get embeddings (cached + new)
        embeddings = await self.cache.get_embeddings_with_cache(code_texts)
        
        if len(embeddings) != len(code_texts):
            logger.warning(f"Only got {len(embeddings)} embeddings for {len(code_texts)} codes")
            return codebook, embeddings
        
        # Cache the snapshot
        snapshot = CodebookSnapshot(
            codes=codebook.copy(),
            embeddings=embeddings,
            snapshot_id=snapshot_id
        )
        self.snapshots[snapshot_id] = snapshot
        
        if self.verbose:
            cache_stats = self.cache.get_cache_stats()
            logger.info(f"Cached snapshot {snapshot_id}. Total cache: {cache_stats['cached_codes']} codes")
        
        return codebook, embeddings
    
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


class TokenAwareBatchProcessor:
    """Creates optimized batches based on token usage and complexity"""
    
    def __init__(self, config: EnhancedCodebookConfig, model_name: str, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.encoding = self._get_encoding(model_name)
    
    def _get_encoding(self, model_name: str):
        """Get appropriate tokenizer for the model"""
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Using cl100k_base encoding as fallback for {model_name}")
            return tiktoken.get_encoding("cl100k_base")
    
    def _estimate_base_prompt_tokens(self, var_lab: str, sample_code_text: str) -> int:
        """Estimate base prompt tokens for codebook generation"""
        # Estimate based on template structure
        base_prompt = f"""
        Survey question: {var_lab}
        Existing codes: {sample_code_text}
        Analysis steps and instructions...
        """
        return len(self.encoding.encode(base_prompt))
    
    def _estimate_cluster_tokens(self, cluster_data: Dict) -> int:
        """Estimate tokens for a cluster's ideas"""
        cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
        return len(self.encoding.encode(cluster_text))
    
    def create_adaptive_batches(
        self, 
        clusters: Dict[int, Dict], 
        codebook: List[Dict[str, str]], 
        var_lab: str
    ) -> List[ClusterBatch]:
        """Create token-aware adaptive batches"""
        
        if not clusters:
            return []
        
        # Calculate base prompt tokens
        sample_code_text = "\n".join([
            f"- {code['code']}: {code['definition']}" 
            for code in codebook[:5]  # Sample for estimation
        ]) if codebook else "No existing codes"
        
        base_prompt_tokens = self._estimate_base_prompt_tokens(var_lab, sample_code_text)
        available_tokens = self.config.max_tokens - base_prompt_tokens - self.config.completion_reserve
        
        if self.verbose:
            logger.info(f"Token budget: {available_tokens} (base: {base_prompt_tokens}, reserve: {self.config.completion_reserve})")
        
        # Calculate cluster token estimates
        cluster_items = []
        total_cluster_tokens = 0
        
        for cluster_id, cluster_data in clusters.items():
            cluster_tokens = self._estimate_cluster_tokens(cluster_data)
            cluster_items.append((cluster_id, cluster_data, cluster_tokens))
            total_cluster_tokens += cluster_tokens
        
        # Sort by complexity (token count) for better batching
        cluster_items.sort(key=lambda x: x[2])  # Sort by token count
        
        # Calculate adaptive batch size
        avg_tokens_per_cluster = total_cluster_tokens / len(cluster_items) if cluster_items else 0
        adaptive_max_batch_size = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, int(available_tokens / max(1, avg_tokens_per_cluster)))
        )
        
        if self.verbose:
            logger.info(f"Adaptive batch size: {adaptive_max_batch_size} (avg tokens per cluster: {avg_tokens_per_cluster:.1f})")
        
        # Create batches
        batches = []
        current_batch_clusters = {}
        current_batch_ids = []
        current_batch_tokens = 0
        batch_id = 0
        
        for cluster_id, cluster_data, cluster_tokens in cluster_items:
            # Check if adding this cluster would exceed limits
            if (current_batch_tokens + cluster_tokens > available_tokens or 
                len(current_batch_ids) >= adaptive_max_batch_size):
                
                # Create batch if not empty
                if current_batch_ids:
                    batches.append(ClusterBatch(
                        batch_id=batch_id,
                        cluster_ids=current_batch_ids,
                        cluster_data=current_batch_clusters,
                        estimated_tokens=current_batch_tokens,
                        codebook_snapshot=codebook.copy(),
                        snapshot_id=batch_id
                    ))
                    batch_id += 1
                    current_batch_clusters = {}
                    current_batch_ids = []
                    current_batch_tokens = 0
            
            # Handle oversized individual clusters
            if cluster_tokens > available_tokens and not current_batch_ids:
                logger.warning(f"Cluster {cluster_id} exceeds token budget ({cluster_tokens} > {available_tokens}). Processing as single item batch.")
                batches.append(ClusterBatch(
                    batch_id=batch_id,
                    cluster_ids=[cluster_id],
                    cluster_data={cluster_id: cluster_data},
                    estimated_tokens=cluster_tokens,
                    codebook_snapshot=codebook.copy(),
                    snapshot_id=batch_id
                ))
                batch_id += 1
                continue
            
            # Add cluster to current batch
            current_batch_ids.append(cluster_id)
            current_batch_clusters[cluster_id] = cluster_data
            current_batch_tokens += cluster_tokens
        
        # Add final batch if not empty
        if current_batch_ids:
            batches.append(ClusterBatch(
                batch_id=batch_id,
                cluster_ids=current_batch_ids,
                cluster_data=current_batch_clusters,
                estimated_tokens=current_batch_tokens,
                codebook_snapshot=codebook.copy(),
                snapshot_id=batch_id
            ))
        
        if self.verbose:
            total_batches = len(batches)
            avg_batch_size = sum(len(b.cluster_ids) for b in batches) / total_batches if total_batches > 0 else 0
            logger.info(f"Created {total_batches} adaptive batches, avg size: {avg_batch_size:.1f} clusters")
        
        return batches


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
        if batch_size != 5:
            self.config.max_batch_size = batch_size
        if max_concurrent_requests != 3:
            self.config.max_concurrent_batches = max_concurrent_requests
        
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
        
        # Initialize token-aware batch processor
        self.batch_processor = TokenAwareBatchProcessor(
            config=self.config,
            model_name=self.model_config.get_model_for_phase("phase1_descriptive"),
            verbose=verbose
        )
        
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
            model=self.model_config.get_model_for_phase("phase1_descriptive"),
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
    
    async def _process_cluster_with_retries(self, cluster_id: int, cluster_data: Dict, 
                                          codebook_snapshot: List[Dict], snapshot_id: int, 
                                          semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process cluster with retry logic and exponential backoff"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._process_cluster_single_attempt(
                    cluster_id, cluster_data, codebook_snapshot, snapshot_id, semaphore
                )
                
                if attempt > 0:  # Log successful retry
                    logger.info(f"Cluster {cluster_id} succeeded on attempt {attempt + 1}")
                    self.stats['retries'] += attempt
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    # Exponential backoff
                    delay = self.config.retry_delay * (self.config.retry_exponential_base ** attempt)
                    
                    if self.verbose:
                        logger.warning(f"Cluster {cluster_id} attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s")
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Cluster {cluster_id} failed after {self.config.max_retries + 1} attempts: {str(e)}")
                    self.stats['retries'] += attempt
        
        # All retries failed
        return {
            'cluster_id': cluster_id,
            'status': 'retry_exhausted',
            'needs_new_code': False,
            'error': str(last_exception),
            'attempts': self.config.max_retries + 1
        }
    
    async def _process_cluster_single_attempt(self, cluster_id: int, cluster_data: Dict, 
                                            codebook_snapshot: List[Dict], snapshot_id: int, 
                                            semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process a single cluster asynchronously with rate limiting"""
        async with semaphore:  # Limit concurrent requests
            try:
                # Get codebook embeddings using the caching system
                codes, codebook_embeddings = await self.embedding_manager.get_snapshot_embeddings(
                    codebook_snapshot, snapshot_id
                )
                
                if not codebook_embeddings:
                    return {
                        'cluster_id': cluster_id,
                        'status': 'embedding_error',
                        'needs_new_code': False
                    }
                
                # Calculate cluster embedding
                cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
                
                # Find k nearest codes
                nearest_codes = self.data_processor.find_k_nearest_codes(cluster_embedding, codebook_embeddings)
                
                # Prepare texts for LLM
                cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
                code_text = "\n".join([
                    f"- {code['code']}: {code['definition']}" 
                    for code in nearest_codes
                ]) if nearest_codes else "No existing codes in codebook"
                
                # Capture prompt for first cluster only
                if self.prompt_printer and cluster_id == 0:
                    prompt_content = f"Existing codes:\n{code_text}\n\nCluster text:\n{cluster_text}"
                    self.prompt_printer.capture_prompt(
                        step_name="gatos_codebook", 
                        utility_name="InductiveCodebookGenerator",
                        prompt_content=prompt_content,
                        prompt_type="GATOS Codebook Generation"
                    )
                
                # Call LLM
                result = await self._call_llm_for_code_generation(code_text, cluster_text)
                
                return {
                    'cluster_id': cluster_id,
                    'status': 'success',
                    'needs_new_code': result['needs_new_code'],
                    'code': result.get('code'),
                    'definition': result.get('definition')
                }
                
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
                return {
                    'cluster_id': cluster_id,
                    'status': 'error',
                    'needs_new_code': False,
                    'error': str(e)
                }
    
    async def process_batch_async(self, batch: ClusterBatch) -> List[Dict[str, Any]]:
        """Process a batch of clusters asynchronously with retry logic"""
        # Create dynamic semaphore based on configuration
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        # Use the pre-stored codebook snapshot and snapshot_id from batch
        current_codebook = batch.codebook_snapshot or self.data_processor.codebook.copy()
        snapshot_id = batch.snapshot_id
        
        # Create tasks for all clusters in batch with retry wrapper
        tasks = []
        for cluster_id in batch.cluster_ids:
            cluster_data = batch.cluster_data[cluster_id]
            task = self._process_cluster_with_retries(
                cluster_id, cluster_data, current_codebook, snapshot_id, semaphore
            )
            tasks.append(task)
        
        # Process all clusters in batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught by retry logic
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                cluster_id = batch.cluster_ids[i]
                logger.error(f"Unhandled exception in cluster {cluster_id}: {str(result)}")
                processed_results.append({
                    'cluster_id': cluster_id,
                    'status': 'unhandled_exception',
                    'needs_new_code': False,
                    'error': str(result)
                })
                self.stats['batch_failures'] += 1
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def generate_async(self) -> Dict[str, Any]:
        """Generate inductive codebook using concurrent batch processing with async operations"""
        verbose_reporter = VerboseReporter(self.verbose)
        verbose_reporter.section_header("ENHANCED INDUCTIVE CODEBOOK GENERATION (CONCURRENT + ADAPTIVE)")
        
        # Prepare clusters
        clusters = self.data_processor.prepare_cluster_text()
        
        if not clusters:
            verbose_reporter.stat_line("No valid clusters found with embeddings")
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': {},
                'stats': self.stats
            }
        
        self.stats['total_clusters'] = len(clusters)
        start_time = time.time()
        
        # Pre-embed initial codebook for efficiency
        if self._pre_embed_initial_codebook and self.data_processor.codebook:
            if self.verbose:
                verbose_reporter.stat_line(f"Pre-embedding initial codebook ({len(self.data_processor.codebook)} codes)")
            
            # Pre-embed the initial codebook (snapshot 0)
            await self.embedding_manager.get_snapshot_embeddings(
                self.data_processor.codebook.copy(), 0
            )
        
        # Create token-aware adaptive batches
        batches = self.batch_processor.create_adaptive_batches(
            clusters, self.data_processor.codebook, self.var_lab
        )
        
        verbose_reporter.stat_line(
            f"Processing {len(clusters)} clusters in {len(batches)} concurrent batches "
            f"(adaptive sizing, max concurrent: {self.config.max_concurrent_batches})"
        )
        
        # Process all batches concurrently - THIS IS THE KEY IMPROVEMENT
        batch_tasks = [self.process_batch_async(batch) for batch in batches]
        
        if self.verbose:
            verbose_reporter.stat_line("Starting concurrent batch processing...")
        
        # Execute all batches concurrently
        all_batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Track cluster assignments and process results
        cluster_to_code = {}
        
        for batch_idx, batch_results in enumerate(all_batch_results):
            if isinstance(batch_results, Exception):
                logger.error(f"Batch {batch_idx} failed completely: {str(batch_results)}")
                batch = batches[batch_idx]
                for cluster_id in batch.cluster_ids:
                    cluster_to_code[cluster_id] = "batch_failed"
                    self.stats['errors'] += 1
                continue
            
            # Process individual cluster results
            for result in batch_results:
                cluster_id = result['cluster_id']
                
                if result['status'] == 'success':
                    if result['needs_new_code'] and result.get('code') and result.get('definition'):
                        # Add new code to codebook
                        new_code = {
                            'code': result['code'],
                            'definition': result['definition'],
                            'cluster_origin': str(cluster_id)
                        }
                        self.data_processor.codebook.append(new_code)
                        cluster_to_code[cluster_id] = result['code']
                        self.stats['new_codes_added'] += 1
                        
                        if self.verbose:
                            logger.info(f"New code added for cluster {cluster_id}: '{result['code']}'")
                    else:
                        cluster_to_code[cluster_id] = "existing_code"
                        self.stats['no_new_codes_needed'] += 1
                elif result['status'] in ['retry_exhausted', 'unhandled_exception']:
                    cluster_to_code[cluster_id] = result['status']
                    self.stats['errors'] += 1
                else:
                    cluster_to_code[cluster_id] = result['status']
                    self.stats['errors'] += 1
        
        processing_time = time.time() - start_time
        
        # Get comprehensive statistics
        cache_stats = self.embedding_manager.get_manager_stats()
        
        # Final summary with performance metrics
        verbose_reporter.summary("ENHANCED CODEBOOK GENERATION COMPLETE", {
            "Initial codes": len(self.starter_codes),
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.data_processor.codebook),
            "Clusters processed": len(clusters),
            "Batches processed": len(batches),
            "Processing errors": self.stats['errors'],
            "Retries performed": self.stats['retries'],
            "Processing time (s)": f"{processing_time:.2f}",
            "Cached embeddings": cache_stats['cache_stats']['cached_codes'],
            "Cache memory (MB)": f"{cache_stats['total_memory_mb']:.2f}"
        })
        
        return {
            'codebook': self.data_processor.codebook,
            'cluster_assignments': cluster_to_code,
            'batches': batches,
            'stats': self.stats
        }
    
    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for the async generate method"""
        async def run_generation():
            result = await self.generate_async()
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': result.get('cluster_assignments', {}),
                'stats': {
                    'initial_codes': len(self.starter_codes),
                    'new_codes': self.stats['new_codes_added'],
                    'total_codes': len(self.data_processor.codebook),
                    'clusters_processed': self.stats['total_clusters'],
                    'batches_processed': len(result.get('batches', [])),
                    'no_new_codes_needed': self.stats['no_new_codes_needed'],
                    'errors': self.stats['errors'],
                    'retries': self.stats['retries'],
                    'batch_failures': self.stats['batch_failures']
                }
            }
        
        return asyncio.run(run_generation())