import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

# Third-party imports
import numpy as np
from openai import AsyncOpenAI

_genai = None
def _ensure_gemini():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        _genai = genai
    return _genai

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, GEMINI_API_KEY, EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG, ModelConfig, get_embedding_dimensions
from utils.llm import create_embedding_client

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats

# === CONSTANTS ========================================================================================================
GEMINI_REQUEST_STAGGER_DELAY = 0.05   # Seconds between Gemini requests to avoid rate limits
RETRY_BACKOFF_BASE = 0.8              # Base multiplier for exponential backoff
DEFAULT_RETRIES = 3                   # Default retry attempts for API calls


@dataclass
class ResponseData:
    respondent_id: str
    segment_id: str
    text_to_embed: str
    array_index: int  

class Embedder:
    """Generate embeddings for survey response ideas with ID tracking.

    Supports OpenAI and Gemini embedding providers with:
    - Batch processing with configurable concurrency
    - Text deduplication for efficiency
    - Optional question-aware embedding transformation
    - Full ID tracking through async operations

    Args:
        config: EmbeddingConfig with batch sizes and concurrency settings
        model_config: ModelConfig specifying embedding model
        provider: "openai" or "gemini"
        client: Optional pre-configured API client
        var_lab: Survey question label for question-aware embeddings
        verbose: Enable verbose progress reporting
    """

    def __init__(
        self, 
        config: EmbeddingConfig = None,
        model_config: ModelConfig = None,
        provider: str = "openai",
        client: Any = None, 
        var_lab: str = None,
        verbose: bool = False):
        
        self.config = config or DEFAULT_EMBEDDING_CONFIG
        self.model_config = model_config or ModelConfig()
        self.provider = provider.lower()

        if self.provider == "openai":
            self.client = client or create_embedding_client(async_mode=True)
        elif self.provider == "gemini":
            genai = _ensure_gemini()
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = None  # gemini calls go via the module
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self.embedding_model = self.model_config.get_model_for_stage('embedding')
        
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.stats = ProcessingStats()
        self.embedding_dimensions = get_embedding_dimensions(self.embedding_model)
        
        self.verbose_reporter.stat_line(f"Model: {self.embedding_model} ({self.embedding_dimensions} dimensions)")
    
    def _get_ResponseData(self, data: List[models.IdeasExtractedModel]) -> List[ResponseData]:
        """Create segment identifiers for tracking"""
        response_data = []
        
        for respondent_data in data:
            if respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    response_item = ResponseData(
                        respondent_id=str(respondent_data.respondent_id),
                        segment_id=str(response_idea.idea_id),
                        text_to_embed=response_idea.idea ,
                        array_index=len(response_data)
                    )
                    response_data.append(response_item)
        
        return response_data
    
    async def _generate_question_aware_embeddings(self, response_embeddings: np.ndarray, question: str) -> np.ndarray:
        """Generate question-aware embeddings by combining response and question embeddings"""
        # Generate question embedding
        question_embedding = await self._embed_single(question)
        
        # Create domain anchor (average of all response embeddings)
        domain_anchor = np.mean(response_embeddings, axis=0)
        
        # Combine embeddings using configured weights
        question_aware_embeddings = []
        for response_emb in response_embeddings:
            combined = (self.config.response_weight * response_emb + 
                       self.config.question_weight * question_embedding + 
                       self.config.domain_anchor_weight * domain_anchor)
            question_aware_embeddings.append(combined)
        
        return np.array(question_aware_embeddings)
    
    
    async def _embed_openai_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts using OpenAI API.

        Args:
            batch_texts: List of text strings to embed

        Returns:
            List of numpy arrays containing embeddings
        """
        response = await self.client.embeddings.create(
            input=batch_texts,
            model=self.embedding_model
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]
    
    
    def _gemini_values(self, response: Any) -> np.ndarray:
        """Normalize google-generativeai embed_content response to a float32 numpy array."""
        # Case 1: dict form: {"embedding": {"values": [...]}} or {"embedding": [...]}
        if isinstance(response, dict):
            emb = response.get("embedding")
            if isinstance(emb, dict) and "values" in emb:
                return np.array(emb["values"], dtype=np.float32)
            elif isinstance(emb, (list, tuple)):
                # Handle direct embedding array in dict: {"embedding": [...]}
                return np.array(emb, dtype=np.float32)
            # Sometimes a list under "data"
            data = response.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                emb = data[0].get("embedding")
                if isinstance(emb, dict) and "values" in emb:
                    return np.array(emb["values"], dtype=np.float32)

        # Case 2: object with .embedding.values
        emb = getattr(response, "embedding", None)
        if emb is not None:
            vals = getattr(emb, "values", None)
            if isinstance(vals, (list, tuple)):
                return np.array(vals, dtype=np.float32)

        # Case 3: list of one
        if isinstance(response, list) and response:
            first = response[0]
            # Recurse once to handle dict/object inside
            arr = self._gemini_values(first)
            if arr is not None:
                return arr

        # If we reach here, shape is unexpected
        raise TypeError(f"Unexpected Gemini embed_content return type: {type(response)} -> {response!r}")
        
    async def _embed_gemini_batch(self, batch_texts: List[str]):
        """
        GEMINI: Use concurrent processing approach (no true batch API available)
        
        PERFORMANCE OPTIMIZATION:
        - Concurrent processing with rate limiting (~21 seconds for 772 embeddings)
        - Uses semaphore to control concurrency and respect API rate limits
        """
        return await self._embed_gemini_concurrent(batch_texts)
    
    
    async def _embed_gemini_concurrent(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Optimized concurrent Gemini embeddings processing"""
        genai = _ensure_gemini()

        # Use provider-specific concurrency from config
        gemini_concurrency = min(self.config.gemini_max_concurrent, len(batch_texts))
        semaphore = asyncio.Semaphore(gemini_concurrency)

        async def embed_single_text(text: str, index: int):
            async with semaphore:
                # Stagger requests slightly to avoid rate limit spikes
                if index > 0:
                    await asyncio.sleep(GEMINI_REQUEST_STAGGER_DELAY)

                def _embed_single():
                    return genai.embed_content(
                        model=self.embedding_model,
                        content=text
                    )

                try:
                    result = await asyncio.to_thread(_embed_single)
                    return self._gemini_values(result)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to embed text at index {index}: {str(e)[:100]}...") from e

        tasks = [embed_single_text(text, i) for i, text in enumerate(batch_texts)]
        embeddings = await asyncio.gather(*tasks)
        return embeddings
   
    async def _embed_batch(self, batch_texts: List[str]):
        if self.provider == "openai":
            return await self._embed_openai_batch(batch_texts)
        elif self.provider == "gemini":
            return await self._embed_gemini_batch(batch_texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    async def _embed_single(self, text_to_embed: str) -> np.ndarray:
        """Generate embedding for a single text using configured provider.

        Args:
            text_to_embed: Text string to embed

        Returns:
            numpy array of embedding values (float32)
        """
        if self.provider == "openai":
            response = await self.client.embeddings.create(input=[text_to_embed], model=self.embedding_model)
            return np.array(response.data[0].embedding, dtype=np.float32)
        else:
            genai = _ensure_gemini()
            def _call():
                return genai.embed_content(model=self.embedding_model, content=text_to_embed)
            response = await asyncio.to_thread(_call)
            return self._gemini_values(response)

    async def _with_retries(self, fn, *, retries: int = DEFAULT_RETRIES, base: float = RETRY_BACKOFF_BASE):
        """Retry async function with exponential backoff.

        Args:
            fn: Async callable to retry
            retries: Maximum retry attempts
            base: Base multiplier for exponential backoff
        """
        for i in range(retries):
            try:
                return await fn()
            except asyncio.CancelledError:
                # Don't retry on cancellation - propagate immediately
                raise
            except Exception:
                if i == retries - 1:
                    raise
                await asyncio.sleep(base * (2 ** i))

    def _deduplicate_texts(self, response_data: List[ResponseData]) -> Tuple[List[str], Dict[str, List[int]], float]:
        """Extract unique texts and create replication mapping"""
        unique_texts_dict = {}
        text_to_indices = {}

        for item in response_data:
            text = item.text_to_embed
            if text not in unique_texts_dict:
                unique_texts_dict[text] = item.array_index
                text_to_indices[text] = [item.array_index]
            else:
                text_to_indices[text].append(item.array_index)

        unique_texts = list(unique_texts_dict.keys())
        compression_ratio = len(unique_texts) / len(response_data)

        return unique_texts, text_to_indices, compression_ratio

    def _replicate_embeddings(self, response_data: List[ResponseData],
                             unique_texts: List[str], unique_embeddings: List[np.ndarray],
                             text_to_indices: Dict[str, List[int]]) -> List[np.ndarray]:
        """Replicate unique embeddings to all instances"""
        all_embeddings = [None] * len(response_data)

        for text, embedding in zip(unique_texts, unique_embeddings):
            for idx in text_to_indices[text]:
                all_embeddings[idx] = embedding

        return all_embeddings

    async def _process_embeddings_with_id_tracking(self, data: List[models.EmbeddingsModel]) -> List[models.EmbeddingsModel]:
        """Process embeddings with explicit ID tracking"""
        
        # Generate segment identifiers
        response_data = self._get_ResponseData(data)

        if not response_data:
            return data

        # Deduplicate texts
        unique_texts, text_to_indices, compression_ratio = self._deduplicate_texts(response_data)

        self.verbose_reporter.stat_line(
            f"Processing {len(response_data)} ideas: {len(unique_texts)} unique texts "
            f"({compression_ratio:.1%} compression)"
        )

        # Extract unique texts for embedding
        texts_to_embed = unique_texts
        
        # Create batches of unique texts only
        if self.provider == "gemini":
            batch_size = getattr(self.config, 'gemini_batch_size', 20)
        elif self.provider == "openai":
            batch_size = getattr(self.config, 'openai_batch_size', 100)
        else:
            batch_size = self.config.batch_size

        batches = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            batches.append(batch_texts)
        
        # Process batches concurrently with provider-specific limits
        if self.provider == "gemini":
            max_concurrent = getattr(self.config, 'gemini_max_concurrent', 3)
        elif self.provider == "openai":
            max_concurrent = getattr(self.config, 'openai_max_concurrent', 5)
        else:
            max_concurrent = self.config.max_concurrent_requests
        
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch_texts: List[str]) -> List[np.ndarray]:
            async with semaphore:
                vectors = await self._with_retries(lambda: self._embed_batch(batch_texts))
                if len(vectors) != len(batch_texts):
                    raise RuntimeError(f"Provider returned {len(vectors)} vectors for {len(batch_texts)} texts")
                return vectors

        # Process all batches
        tasks = [process_batch(batch_texts) for batch_texts in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten unique embeddings
        unique_embeddings = []
        for batch_result in batch_results:
            unique_embeddings.extend(batch_result)

        # Replicate embeddings to all instances
        all_embeddings = self._replicate_embeddings(
            response_data, unique_texts, unique_embeddings, text_to_indices
        )

        # Create identifiers list for compatibility with downstream code
        all_identifiers = response_data
        
        # Apply question-aware processing if enabled and processing descriptions
        if (self.config.use_question_aware and 
            self.var_lab):
            
            self.verbose_reporter.stat_line("Applying question-aware embedding transformation...")
            all_embeddings = await self._generate_question_aware_embeddings(
                np.array(all_embeddings), 
                self.var_lab
            )
        
        # Create lookup
        embedding_lookup = {}
        for identifier, embedding in zip(all_identifiers, all_embeddings):
            key = (identifier.respondent_id, identifier.segment_id)
            embedding_lookup[key] = embedding
        
        # Apply embeddings back to data using ID lookup and convert to EmbeddingsModel
        updated_count = 0
        result = []
        
        for respondent_data in data:
            # Create EmbeddingsSubmodel objects with embeddings
            embeddings_submodels = []
            
            if hasattr(respondent_data, 'response_ideas') and respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    key = (str(respondent_data.respondent_id), str(response_idea.idea_id))
                    embedding_data = {
                        'idea_id': response_idea.idea_id,
                        'idea': response_idea.idea
                    }
                    if key in embedding_lookup:
                        embedding_data['idea_embedding'] = embedding_lookup[key]
                        updated_count += 1
                    
                    embeddings_submodels.append(models.EmbeddingsSubmodel(**embedding_data))
            
            # Create EmbeddingsModel with proper structure
            embeddings_model = models.EmbeddingsModel(
                respondent_id=respondent_data.respondent_id,
                response=respondent_data.response,
                response_type=getattr(respondent_data, 'response_type', None),
                quality_filter=getattr(respondent_data, 'quality_filter', None),
                quality_filter_code=getattr(respondent_data, 'quality_filter_code', None),
                response_ideas=embeddings_submodels,
                idea_count=len(embeddings_submodels)
            )
            result.append(embeddings_model)
        
        self.verbose_reporter.stat_line(f"Successfully applied {updated_count} embeddings")
        
        return result
   
    def get_embeddings_with_tracking(self, data: List[models.EmbeddingsModel], var_lab: str = None) -> List[models.EmbeddingsModel]:
        """Generate both code and description embeddings with ID tracking"""
        if var_lab is not None:
            self.var_lab = var_lab
 
        result = asyncio.run(self._process_embeddings_with_id_tracking(data)) 

        return result

