import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

# Third-party imports
import numpy as np
from openai import AsyncOpenAI

# Gemini client is now instantiated per-Embedder instance (no module-level state)

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, GEMINI_API_KEY, EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG, ModelConfig, get_embedding_dimensions
from utils.llm import create_embedding_client

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats

# === CONSTANTS ========================================================================================================
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
            self.genai_client = None
        elif self.provider == "gemini":
            from google import genai
            self.client = None
            self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self.embedding_model = self.model_config.get_model_for_stage('embedding')
        
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.stats = ProcessingStats()
        self.embedding_dimensions = get_embedding_dimensions(self.embedding_model)
        
        self.verbose_reporter.stat_line(f"Model: {self.embedding_model} ({self.embedding_dimensions} dimensions)")

    def _parse_specifiers(self, idea_text: str) -> dict:
        """Extract specifier values from bracketed format in idea text.

        Parses tags like [domain=retail][topic=groceries] from the idea text
        and returns them as a dictionary.

        Args:
            idea_text: Full idea text with specifier brackets

        Returns:
            Dict mapping specifier names to their values
        """
        import re
        specifiers = {}
        pattern = r'\[(\w+)=([^\]]*)\]'
        for match in re.finditer(pattern, idea_text):
            specifiers[match.group(1)] = match.group(2)
        return specifiers

    def _get_text_for_embedding(self, idea_text: str, template_prefix: str = None) -> str:
        """Extract text for embedding based on EMBEDDING_TEXT_FORMAT config.

        Args:
            idea_text: Full formatted idea with specifiers and template prefix
            template_prefix: The canonical phrasing prefix (e.g., "ASN Bank has the association")

        Returns:
            Text to embed based on config mode:
            - "with_context": Full text with all specifiers
            - "unique_only": Just the idea text, stripped of specifiers and prefix
            - "taxonomy_with_context": domain + topic + intent + taxonomy_phrase (space-concatenated)
            - "taxonomy_unique_only": Just the taxonomy_phrase value
        """
        from config import EMBEDDING_TEXT_FORMAT

        if EMBEDDING_TEXT_FORMAT == "with_context":
            return idea_text

        # Handle taxonomy modes (require specifiers from experimental ideaExtractor_v2)
        if EMBEDDING_TEXT_FORMAT == "taxonomy_unique_only":
            specifiers = self._parse_specifiers(idea_text)
            taxonomy_phrase = specifiers.get('taxonomy_phrase', '')
            if taxonomy_phrase:
                return taxonomy_phrase
            # Fallback to unique_only if no taxonomy_phrase

        if EMBEDDING_TEXT_FORMAT == "taxonomy_with_context":
            specifiers = self._parse_specifiers(idea_text)
            parts = []
            for key in ['domain', 'topic', 'intent', 'taxonomy_phrase']:
                if specifiers.get(key):
                    parts.append(specifiers[key])
            if parts:
                return ' '.join(parts)
            # Fallback to unique_only if no taxonomy fields

        # "unique_only" mode (default/fallback): strip context specifiers and template prefix
        lines = idea_text.split('\n')

        # Get the actual idea text (last line, after specifier lines)
        # Format is: [specifiers]\n[sentiment/sense]\n<actual idea text>
        idea_line = lines[-1] if len(lines) >= 1 else idea_text

        # Strip template prefix if provided
        if template_prefix and idea_line.startswith(template_prefix):
            unique_content = idea_line[len(template_prefix):].strip()
            # Return unique content if non-empty, otherwise return the idea line
            return unique_content if unique_content else idea_line

        return idea_line

    def _get_ResponseData(self, data: List[models.EmbeddingsModel]) -> List[ResponseData]:
        """Create segment identifiers for tracking with optional text extraction for embedding."""
        response_data = []

        # Extract template_prefix from the first item with data (used for all items)
        template_prefix = None
        for item in data:
            if hasattr(item, 'template_prefix') and item.template_prefix:
                template_prefix = item.template_prefix
                break

        # Log which embedding format is being used
        from config import EMBEDDING_TEXT_FORMAT
        if EMBEDDING_TEXT_FORMAT == "taxonomy_unique_only":
            self.verbose_reporter.stat_line("Embedding format: taxonomy_unique_only (taxonomy_phrase only)")
        elif EMBEDDING_TEXT_FORMAT == "taxonomy_with_context":
            self.verbose_reporter.stat_line("Embedding format: taxonomy_with_context (domain + topic + intent + taxonomy_phrase)")
        elif template_prefix and EMBEDDING_TEXT_FORMAT == "unique_only":
            self.verbose_reporter.stat_line(f"Embedding format: unique_only (stripping prefix: '{template_prefix[:50]}...')" if len(template_prefix) > 50 else f"Embedding format: unique_only (stripping prefix: '{template_prefix}')")
        elif EMBEDDING_TEXT_FORMAT == "unique_only":
            self.verbose_reporter.stat_line("Embedding format: unique_only (no template_prefix available, stripping specifiers only)")
        else:
            self.verbose_reporter.stat_line("Embedding format: with_context (full text)")

        for respondent_data in data:
            if respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    # Apply text extraction based on config
                    text_to_embed = self._get_text_for_embedding(response_idea.idea, template_prefix)

                    response_item = ResponseData(
                        respondent_id=str(respondent_data.respondent_id),
                        segment_id=str(response_idea.idea_id),
                        text_to_embed=text_to_embed,
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
    
    
    def _gemini_values(self, embeddings: List[Any]) -> List[np.ndarray]:
        """Extract embedding values from google.genai response.

        The new google.genai SDK returns consistent Pydantic models with .values attribute.

        Args:
            embeddings: List of ContentEmbedding objects from genai response

        Returns:
            List of numpy arrays containing embedding values
        """
        return [np.array(emb.values, dtype=np.float32) for emb in embeddings]
        
    async def _embed_gemini_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using google.genai SDK with native batch support.

        The new SDK supports multiple texts in a single API call, which is
        much faster than individual calls with staggering.

        Args:
            batch_texts: List of text strings to embed

        Returns:
            List of numpy arrays containing embeddings
        """
        from google.genai import types

        # The new SDK supports batch embedding natively
        def _embed_batch_sync():
            return self.genai_client.models.embed_content(
                model=self.embedding_model,
                contents=batch_texts,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY"
                )
            )

        try:
            result = await asyncio.to_thread(_embed_batch_sync)
            return self._gemini_values(result.embeddings)
        except Exception as e:
            raise RuntimeError(f"Failed to embed batch of {len(batch_texts)} texts: {str(e)[:200]}") from e
   
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
            from google.genai import types

            def _call():
                return self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=[text_to_embed],
                    config=types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY"
                    )
                )
            response = await asyncio.to_thread(_call)
            return self._gemini_values(response.embeddings)[0]

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
                idea_count=len(embeddings_submodels),
                template_prefix=getattr(respondent_data, 'template_prefix', None)
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

    def close(self):
        """Close the Gemini client to release resources."""
        if self.genai_client is not None:
            try:
                self.genai_client.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self.genai_client = None

    def __del__(self):
        """Cleanup Gemini client on garbage collection."""
        self.close()

