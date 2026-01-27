"""
Embedder V2 - Production Implementation

Generate embeddings for survey response ideas with configurable text formats
and quality analysis.

Supports:
- Configurable text formats: "idea", "taxonomy_phrase", "idea_without_template_prefix", "both"
- OpenAI and Gemini embedding providers
- Batch processing with configurable concurrency
- Text deduplication for efficiency
- Optional question-aware embedding transformation
- Embedding quality analysis (norms, pairwise similarity)
- Full ID tracking through async operations

NOTE: This is the production embedder (migrated from experiments/embedder_v2).
The old embedder is archived at src/utils/old/embedder_v1.py.
"""

# === MODULES ========================================================================================================
import asyncio
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

# Third-party imports
import numpy as np
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, GEMINI_API_KEY, ModelConfig, get_embedding_dimensions
from utils.llm import create_embedding_client

# === STEP-SPECIFIC CONFIG ==========================================================================================
from config_embedder import (
    EmbedderConfig,
    DEFAULT_EMBEDDER_CONFIG,
    BOTH_MODE_IDEA_FORMAT,
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats


@dataclass
class ResponseData:
    """Tracks embedding request with full ID context."""
    respondent_id: str
    segment_id: str
    text_to_embed: str
    array_index: int


@dataclass
class EmbeddingAnalysis:
    """Results from embedding quality analysis."""
    n_embeddings: int
    embedding_dim: int
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float
    mean_pairwise_similarity: Optional[float] = None
    std_pairwise_similarity: Optional[float] = None
    min_pairwise_similarity: Optional[float] = None
    max_pairwise_similarity: Optional[float] = None


class Embedder:
    """Generate embeddings for survey response ideas with configurable text formats.

    Supports OpenAI and Gemini embedding providers with:
    - Configurable text format: "idea", "taxonomy_phrase", "idea_without_template_prefix", "both"
    - Batch processing with configurable concurrency
    - Text deduplication for efficiency
    - Optional question-aware embedding transformation
    - Embedding quality analysis
    - Full ID tracking through async operations

    Args:
        config: EmbedderConfig with all embedder settings
        model_config: ModelConfig specifying embedding model (optional)
        client: Optional pre-configured API client
        var_lab: Survey question label for question-aware embeddings
    """

    def __init__(
        self,
        config: EmbedderConfig = None,
        model_config: ModelConfig = None,
        client: Any = None,
        var_lab: str = None
    ):
        self.config = config or DEFAULT_EMBEDDER_CONFIG
        self.model_config = model_config or ModelConfig()

        self.provider = self.config.provider.lower()
        self.verbose = self.config.verbose

        # Initialize provider-specific client
        if self.provider == "openai":
            self.client = client or create_embedding_client(async_mode=True)
            self.genai_client = None
        elif self.provider == "gemini":
            from google import genai
            self.client = None
            self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Get embedding model
        self.embedding_model = self.model_config.get_model_for_stage('embedding')
        self.var_lab = var_lab
        self.verbose_reporter = VerboseReporter(self.verbose, capture_logging=True)
        self.stats = ProcessingStats()
        self.embedding_dimensions = get_embedding_dimensions(self.embedding_model)

        # Store embedding text format from config
        self.embedding_text_format = self.config.embedding_text_format

        # Analysis results
        self.analysis: Optional[EmbeddingAnalysis] = None

        # Extraction metadata (optional, for template_prefix)
        self.extraction_metadata: Optional[models.ExtractionMetadata] = None

        self.verbose_reporter.stat_line(f"Model: {self.embedding_model} ({self.embedding_dimensions} dimensions)")
        self.verbose_reporter.stat_line(f"Provider: {self.provider}")
        self.verbose_reporter.stat_line(f"Text format: {self.embedding_text_format}")

    def set_extraction_metadata(self, metadata: models.ExtractionMetadata):
        """Set extraction metadata for template_prefix access.

        Args:
            metadata: ExtractionMetadata from ideaExtractor (contains template_prefix)
        """
        self.extraction_metadata = metadata
        if metadata and metadata.template_prefix:
            self.verbose_reporter.stat_line(f"Template prefix loaded: '{metadata.template_prefix[:50]}...' " if len(metadata.template_prefix) > 50 else f"Template prefix loaded: '{metadata.template_prefix}'")

    def _get_template_prefix(self) -> Optional[str]:
        """Get template_prefix from extraction_metadata.

        Returns:
            The template_prefix string or None if not available.
        """
        if self.extraction_metadata and self.extraction_metadata.template_prefix:
            return self.extraction_metadata.template_prefix
        return None

    def _get_text_for_embedding(self, idea) -> str:
        """Extract text for embedding based on config.

        Args:
            idea: Idea object with idea text and taxonomy_phrase fields

        Returns:
            Text to embed based on config mode:
            - "idea": The clean idea text (idea.idea)
            - "taxonomy_phrase": The taxonomy phrase (idea.taxonomy_phrase)
            - "idea_without_template_prefix": The idea text with template_prefix stripped
        """
        if self.embedding_text_format == "taxonomy_phrase":
            taxonomy_phrase = getattr(idea, 'taxonomy_phrase', '') or ''
            # Fallback to idea text if no taxonomy_phrase
            return taxonomy_phrase if taxonomy_phrase else idea.idea

        if self.embedding_text_format == "idea_without_template_prefix":
            idea_text = idea.idea
            template_prefix = self._get_template_prefix()
            if template_prefix and idea_text.startswith(template_prefix):
                unique_content = idea_text[len(template_prefix):].strip()
                # Return unique content if non-empty, otherwise return the idea text
                return unique_content if unique_content else idea_text
            # No template_prefix available or idea doesn't start with it
            return idea_text

        # Default: "idea" mode - return the idea text directly
        return idea.idea

    def _get_ResponseData(self, data: List[models.EmbeddingsModel]) -> List[ResponseData]:
        """Create segment identifiers for tracking with text extraction for embedding."""
        response_data = []

        # Log which embedding format is being used
        if self.embedding_text_format == "taxonomy_phrase":
            self.verbose_reporter.stat_line("Embedding format: taxonomy_phrase")
        elif self.embedding_text_format == "idea_without_template_prefix":
            prefix = self._get_template_prefix()
            if prefix:
                prefix_display = prefix[:50] + "..." if len(prefix) > 50 else prefix
                self.verbose_reporter.stat_line(f"Embedding format: idea_without_template_prefix (stripping: '{prefix_display}')")
            else:
                self.verbose_reporter.stat_line("Embedding format: idea_without_template_prefix (no prefix available, using full idea)")
        else:
            self.verbose_reporter.stat_line("Embedding format: idea")

        for respondent_data in data:
            if respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    text_to_embed = self._get_text_for_embedding(response_idea)

                    response_item = ResponseData(
                        respondent_id=str(respondent_data.respondent_id),
                        segment_id=str(response_idea.idea_id),
                        text_to_embed=text_to_embed,
                        array_index=len(response_data)
                    )
                    response_data.append(response_item)

        return response_data

    async def _generate_question_aware_embeddings(self, response_embeddings: np.ndarray, question: str) -> np.ndarray:
        """Generate question-aware embeddings by combining response and question embeddings."""
        # Generate question embedding
        question_embedding = await self._embed_single(question)

        # Create domain anchor (average of all response embeddings)
        domain_anchor = np.mean(response_embeddings, axis=0)

        # Combine embeddings using configured weights
        question_aware_embeddings = []
        for response_emb in response_embeddings:
            combined = (
                self.config.response_weight * response_emb +
                self.config.question_weight * question_embedding +
                self.config.domain_anchor_weight * domain_anchor
            )
            question_aware_embeddings.append(combined)

        return np.array(question_aware_embeddings)

    async def _embed_openai_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts using OpenAI API."""
        response = await self.client.embeddings.create(
            input=batch_texts,
            model=self.embedding_model
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def _gemini_values(self, embeddings: List[Any]) -> List[np.ndarray]:
        """Extract embedding values from google.genai response."""
        return [np.array(emb.values, dtype=np.float32) for emb in embeddings]

    async def _embed_gemini_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using google.genai SDK with native batch support."""
        from google.genai import types

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
        """Embed a batch of texts using the configured provider."""
        if self.provider == "openai":
            return await self._embed_openai_batch(batch_texts)
        elif self.provider == "gemini":
            return await self._embed_gemini_batch(batch_texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _embed_single(self, text_to_embed: str) -> np.ndarray:
        """Generate embedding for a single text using configured provider."""
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

    async def _with_retries(self, fn, *, retries: int = None, base: float = None):
        """Retry async function with exponential backoff."""
        retries = retries or self.config.default_retries
        base = base or self.config.retry_backoff_base

        for i in range(retries):
            try:
                return await fn()
            except asyncio.CancelledError:
                raise
            except Exception:
                if i == retries - 1:
                    raise
                await asyncio.sleep(base * (2 ** i))

    def _deduplicate_texts(self, response_data: List[ResponseData]) -> Tuple[List[str], Dict[str, List[int]], float]:
        """Extract unique texts and create replication mapping."""
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
        compression_ratio = len(unique_texts) / len(response_data) if response_data else 1.0

        return unique_texts, text_to_indices, compression_ratio

    def _replicate_embeddings(
        self,
        response_data: List[ResponseData],
        unique_texts: List[str],
        unique_embeddings: List[np.ndarray],
        text_to_indices: Dict[str, List[int]]
    ) -> List[np.ndarray]:
        """Replicate unique embeddings to all instances."""
        all_embeddings = [None] * len(response_data)

        for text, embedding in zip(unique_texts, unique_embeddings):
            for idx in text_to_indices[text]:
                all_embeddings[idx] = embedding

        return all_embeddings

    def analyze_embeddings(self, embeddings: List[np.ndarray]) -> EmbeddingAnalysis:
        """Analyze embedding quality and statistics.

        Args:
            embeddings: List of embedding vectors

        Returns:
            EmbeddingAnalysis with quality metrics
        """
        if not embeddings:
            return EmbeddingAnalysis(
                n_embeddings=0,
                embedding_dim=0,
                mean_norm=0.0,
                std_norm=0.0,
                min_norm=0.0,
                max_norm=0.0
            )

        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)

        analysis = EmbeddingAnalysis(
            n_embeddings=len(embeddings),
            embedding_dim=embeddings_array.shape[1],
            mean_norm=float(np.mean(norms)),
            std_norm=float(np.std(norms)),
            min_norm=float(np.min(norms)),
            max_norm=float(np.max(norms))
        )

        # Compute pairwise similarity if enabled and not too many embeddings
        max_for_similarity = self.config.max_embeddings_for_similarity
        if self.config.compute_similarity_stats and len(embeddings) <= max_for_similarity:
            # Normalize embeddings for cosine similarity
            normalized = embeddings_array / norms[:, np.newaxis]

            # Compute pairwise cosine similarities (upper triangle only)
            n = len(embeddings)
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = np.dot(normalized[i], normalized[j])
                    similarities.append(sim)

            if similarities:
                analysis.mean_pairwise_similarity = float(np.mean(similarities))
                analysis.std_pairwise_similarity = float(np.std(similarities))
                analysis.min_pairwise_similarity = float(np.min(similarities))
                analysis.max_pairwise_similarity = float(np.max(similarities))

        return analysis

    async def _process_embeddings_with_id_tracking(self, data: List[models.EmbeddingsModel]) -> List[models.EmbeddingsModel]:
        """Process embeddings with explicit ID tracking."""

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
            batch_size = self.config.gemini_batch_size
        elif self.provider == "openai":
            batch_size = self.config.openai_batch_size
        else:
            batch_size = 100

        batches = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            batches.append(batch_texts)

        # Process batches concurrently with provider-specific limits
        if self.provider == "gemini":
            max_concurrent = self.config.gemini_max_concurrent
        elif self.provider == "openai":
            max_concurrent = self.config.openai_max_concurrent
        else:
            max_concurrent = 5

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

        # Create identifiers list for compatibility
        all_identifiers = response_data

        # Apply question-aware processing if enabled
        if self.config.use_question_aware and self.var_lab:
            self.verbose_reporter.stat_line("Applying question-aware embedding transformation...")
            all_embeddings = await self._generate_question_aware_embeddings(
                np.array(all_embeddings),
                self.var_lab
            )

        # Run embedding analysis if enabled
        if self.config.analyze_embeddings:
            self.analysis = self.analyze_embeddings(all_embeddings)
            self.verbose_reporter.stat_line(f"Embedding analysis:")
            self.verbose_reporter.stat_line(f"  Dimensions: {self.analysis.embedding_dim}")
            self.verbose_reporter.stat_line(f"  Norm: mean={self.analysis.mean_norm:.4f}, std={self.analysis.std_norm:.4f}")
            if self.analysis.mean_pairwise_similarity is not None:
                self.verbose_reporter.stat_line(
                    f"  Pairwise similarity: mean={self.analysis.mean_pairwise_similarity:.4f}, "
                    f"std={self.analysis.std_pairwise_similarity:.4f}"
                )

        # Create lookup
        embedding_lookup = {}
        for identifier, embedding in zip(all_identifiers, all_embeddings):
            key = (identifier.respondent_id, identifier.segment_id)
            embedding_lookup[key] = embedding

        # Apply embeddings back to data using ID lookup
        updated_count = 0
        result = []

        for respondent_data in data:
            embeddings_submodels = []

            if hasattr(respondent_data, 'response_ideas') and respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    key = (str(respondent_data.respondent_id), str(response_idea.idea_id))
                    embedding_data = {
                        'idea_id': response_idea.idea_id,
                        'idea': response_idea.idea,
                        # Pass through clean fields from input
                        'taxonomy_phrase': getattr(response_idea, 'taxonomy_phrase', '') or '',
                        'parent_category': getattr(response_idea, 'parent_category', '') or '',
                        'sentiment': getattr(response_idea, 'sentiment', 'neutral') or 'neutral',
                        'sense': getattr(response_idea, 'sense', 'factual') or 'factual',
                    }
                    if key in embedding_lookup:
                        embedding_data['idea_embedding'] = embedding_lookup[key]
                        updated_count += 1

                    embeddings_submodels.append(models.EmbeddingsSubmodel(**embedding_data))

            embeddings_model = models.EmbeddingsModel(
                respondent_id=respondent_data.respondent_id,
                response=respondent_data.response,
                response_type=getattr(respondent_data, 'response_type', None),
                quality_filter=getattr(respondent_data, 'quality_filter', None),
                quality_filter_code=getattr(respondent_data, 'quality_filter_code', None),
                response_ideas=embeddings_submodels,
                idea_count=len(embeddings_submodels),
                template_prefix=getattr(respondent_data, 'template_prefix', None),
                embedding_text_format=self.embedding_text_format  # Store format for downstream alignment
            )
            result.append(embeddings_model)

        self.verbose_reporter.stat_line(f"Successfully applied {updated_count} embeddings")

        return result

    def get_embeddings_with_tracking(self, data: List[models.EmbeddingsModel], var_lab: str = None) -> List[models.EmbeddingsModel]:
        """Generate embeddings with ID tracking.

        Args:
            data: List of EmbeddingsModel (or IdeasExtractedModel) instances
            var_lab: Survey question label for question-aware embeddings

        Returns:
            List of EmbeddingsModel with embeddings applied
        """
        if var_lab is not None:
            self.var_lab = var_lab

        # Handle "both" mode: run two passes
        if self.embedding_text_format == "both":
            return self._process_both_embeddings(data)

        result = asyncio.run(self._process_embeddings_with_id_tracking(data))

        return result

    def _process_both_embeddings(self, data: List[models.EmbeddingsModel]) -> List[models.EmbeddingsModel]:
        """Process dual embeddings for 'both' mode.

        Runs two embedding passes:
        1. First pass: embed idea text (using BOTH_MODE_IDEA_FORMAT) -> idea_embedding
        2. Second pass: embed taxonomy_phrase -> taxonomy_embedding

        Args:
            data: List of EmbeddingsModel instances

        Returns:
            List of EmbeddingsModel with both embedding fields populated
        """
        original_format = self.embedding_text_format

        # === PASS 1: Embed idea text ===
        self.verbose_reporter.stat_line(f"\n--- PASS 1: Embedding idea text (format: {BOTH_MODE_IDEA_FORMAT}) ---")
        self.embedding_text_format = BOTH_MODE_IDEA_FORMAT
        result = asyncio.run(self._process_embeddings_with_id_tracking(data))

        # === PASS 2: Embed taxonomy_phrase ===
        self.verbose_reporter.stat_line(f"\n--- PASS 2: Embedding taxonomy_phrase ---")
        self.embedding_text_format = "taxonomy_phrase"
        taxonomy_result = asyncio.run(self._process_embeddings_with_id_tracking(data))

        # === MERGE: Copy taxonomy embeddings into result's taxonomy_embedding field ===
        self.verbose_reporter.stat_line(f"\n--- Merging embeddings ---")
        merged_count = 0
        for resp_idx, (result_resp, taxonomy_resp) in enumerate(zip(result, taxonomy_result)):
            if result_resp.response_ideas and taxonomy_resp.response_ideas:
                for idea_idx, (result_idea, taxonomy_idea) in enumerate(zip(result_resp.response_ideas, taxonomy_resp.response_ideas)):
                    # The taxonomy_resp has the taxonomy embedding in idea_embedding (since that's what was embedded)
                    if taxonomy_idea.idea_embedding is not None:
                        result_idea.taxonomy_embedding = taxonomy_idea.idea_embedding
                        merged_count += 1

        self.verbose_reporter.stat_line(f"Merged {merged_count} taxonomy embeddings")

        # Restore original format and set it on result models
        self.embedding_text_format = original_format
        for resp in result:
            resp.embedding_text_format = "both"

        return result

    def get_analysis(self) -> Optional[EmbeddingAnalysis]:
        """Get embedding analysis results."""
        return self.analysis

    def close(self):
        """Close the Gemini client to release resources."""
        if self.genai_client is not None:
            try:
                self.genai_client.close()
            except Exception:
                pass
            self.genai_client = None

    def __del__(self):
        """Cleanup Gemini client on garbage collection."""
        self.close()
