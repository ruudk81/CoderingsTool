"""
Embedder — v5-aligned embedding generation with configurable text formats

Generate embeddings for survey response ideas with configurable text formats
and quality analysis.

Supports:
- 8 single-pass text formats: "idea", "idea_bare", "concept", "concept_type",
  "concept_defined", "concept_typed", "idea_concept_defined", "ladder"
- Multi-pass modes: "default" (4 passes), "all" (4 passes) via MULTI_PASS_SPECS
- Batch processing with configurable concurrency
- Text deduplication for efficiency
- Embedding quality analysis (norms, pairwise similarity)
- Full ID tracking through async operations
"""

# === MODULES ========================================================================================================
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import ModelConfig, get_embedding_dimensions
from utils.llm import create_embedding_client

# === STEP-SPECIFIC CONFIG ==========================================================================================
from config_steps.config_embedder import (
    EmbedderConfig,
    DEFAULT_EMBEDDER_CONFIG,
    MULTI_PASS_SPECS,
)

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats


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

    Single-pass formats (stored in idea_embedding):
        "idea"            — idea text as-is (natural sentence incl. template_prefix)
        "idea_bare"       — idea with template_prefix stripped
        "concept"         — canonical concept noun phrase
        "concept_type"    — discovered concept type
        "concept_defined"      — concept → concept_type_definition
        "concept_typed"        — concept (concept_type)
        "idea_concept_defined" — idea → concept → concept_type_definition
        "ladder"               — instance → concept → concept_type → concept_type_definition

    Multi-pass formats (each pass stored in its own field):
        "default"         — 4 passes: idea, ladder, concept_defined, idea_concept_defined
        "all"             — 4 passes: idea, concept, concept_type, ladder

    Args:
        config: EmbedderConfig with all embedder settings
        model_config: ModelConfig specifying embedding model (optional)
        client: Optional pre-configured async OpenAI client
        var_lab: Survey question label
    """

    def __init__(
        self,
        config: EmbedderConfig = None,
        model_config: ModelConfig = None,
        client=None,
        var_lab: str = None
    ):
        self.config = config or DEFAULT_EMBEDDER_CONFIG
        self.model_config = model_config or ModelConfig()
        self.verbose = self.config.verbose

        # OpenAI embedding client
        self.client = client or create_embedding_client(async_mode=True)

        # Embedding model
        self.embedding_model = self.model_config.get_model_for_stage('embedding')
        self.var_lab = var_lab
        self.verbose_reporter = VerboseReporter(self.verbose, capture_logging=True)
        self.stats = ProcessingStats()
        self.embedding_dimensions = get_embedding_dimensions(self.embedding_model)

        # Store embedding text format from config
        self.embedding_text_format = self.config.embedding_text_format

        # Analysis results
        self.analysis: Optional[EmbeddingAnalysis] = None

        # Extraction metadata (optional, for template_prefix stripping)
        self.extraction_metadata: Optional[models.ExtractionMetadata] = None

        self.verbose_reporter.stat_line(f"Model: {self.embedding_model} ({self.embedding_dimensions} dimensions)")
        self.verbose_reporter.stat_line(f"Text format: {self.embedding_text_format}")

    def set_extraction_metadata(self, metadata: models.ExtractionMetadata):
        """Set extraction metadata for template_prefix access."""
        self.extraction_metadata = metadata
        if metadata and metadata.template_prefix:
            prefix = metadata.template_prefix
            display = f"'{prefix[:50]}...'" if len(prefix) > 50 else f"'{prefix}'"
            self.verbose_reporter.stat_line(f"Template prefix loaded: {display}")

    # =========================================================================
    # TEXT FORMAT DISPATCH
    # =========================================================================

    def _format_ladder_text(self, idea) -> str:
        """Format abstraction ladder: instance → concept → concept_type → concept_type_definition.

        Falls back to idea.idea when all ladder fields are empty.
        """
        parts = []
        for field in ('instance', 'concept', 'concept_type', 'concept_type_definition'):
            val = (getattr(idea, field, '') or '').strip()
            if val:
                parts.append(val)
        return " → ".join(parts) if parts else idea.idea

    def _get_text_for_embedding(self, idea) -> str:
        """Extract text for embedding based on current format setting.

        All multi-field formats use ' → ' (unicode arrow) as separator.
        """
        fmt = self.embedding_text_format

        if fmt == "idea_bare":
            # Strip template_prefix from idea text
            if self.extraction_metadata and self.extraction_metadata.template_prefix:
                prefix = self.extraction_metadata.template_prefix
                if idea.idea.startswith(prefix):
                    stripped = idea.idea[len(prefix):].strip()
                    return stripped if stripped else idea.idea
            return idea.idea

        if fmt == "concept":
            val = (getattr(idea, 'concept', '') or '').strip()
            return val if val else idea.idea

        if fmt == "concept_type":
            val = (getattr(idea, 'concept_type', '') or '').strip()
            return val if val else idea.idea

        if fmt == "concept_typed":
            concept = (getattr(idea, 'concept', '') or '').strip()
            concept_type = (getattr(idea, 'concept_type', '') or '').strip()
            if concept and concept_type:
                return f"{concept} ({concept_type})"
            return concept or idea.idea

        if fmt == "concept_defined":
            concept = (getattr(idea, 'concept', '') or '').strip()
            definition = (getattr(idea, 'concept_type_definition', '') or '').strip()
            if concept and definition:
                return f"{concept} → {definition}"
            return concept or idea.idea

        if fmt == "idea_concept_defined":
            concept = (getattr(idea, 'concept', '') or '').strip()
            definition = (getattr(idea, 'concept_type_definition', '') or '').strip()
            parts = [idea.idea]
            if concept:
                parts.append(concept)
            if definition:
                parts.append(definition)
            return " → ".join(parts)

        if fmt == "ladder":
            return self._format_ladder_text(idea)

        # Default: "idea" — full idea text (natural sentence incl. template_prefix)
        return idea.idea

    # =========================================================================
    # RESPONSE DATA EXTRACTION
    # =========================================================================

    def _get_ResponseData(self, data: List[models.EmbeddingsModel]) -> List[ResponseData]:
        """Create segment identifiers for tracking with text extraction for embedding."""
        response_data = []

        format_labels = {
            "idea":            "idea (natural sentence incl. template_prefix)",
            "idea_bare":       "idea (template_prefix stripped)",
            "concept":         "concept (canonical noun phrase)",
            "concept_type":    "concept_type (discovered type)",
            "concept_defined":      "concept → concept_type_definition",
            "concept_typed":        "concept (concept_type)",
            "idea_concept_defined": "idea → concept → concept_type_definition",
            "ladder":               "abstraction ladder (instance → concept → concept_type → concept_type_definition)",
        }
        label = format_labels.get(self.embedding_text_format, self.embedding_text_format)
        self.verbose_reporter.stat_line(f"Embedding format: {label}")

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

    # =========================================================================
    # OPENAI EMBEDDING API
    # =========================================================================

    async def _embed_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts using OpenAI API."""
        response = await self.client.embeddings.create(
            input=batch_texts,
            model=self.embedding_model
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

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

    # =========================================================================
    # DEDUPLICATION
    # =========================================================================

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

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def analyze_embeddings(self, embeddings: List[np.ndarray]) -> EmbeddingAnalysis:
        """Analyze embedding quality and statistics."""
        if not embeddings:
            return EmbeddingAnalysis(
                n_embeddings=0, embedding_dim=0,
                mean_norm=0.0, std_norm=0.0, min_norm=0.0, max_norm=0.0
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

        max_for_similarity = self.config.max_embeddings_for_similarity
        if self.config.compute_similarity_stats and len(embeddings) <= max_for_similarity:
            normalized = embeddings_array / norms[:, np.newaxis]
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

    # =========================================================================
    # CORE EMBEDDING PIPELINE
    # =========================================================================

    async def _process_embeddings_with_id_tracking(self, data: List[models.EmbeddingsModel]) -> List[models.EmbeddingsModel]:
        """Process embeddings with explicit ID tracking."""

        response_data = self._get_ResponseData(data)

        if not response_data:
            return data

        # Deduplicate texts
        unique_texts, text_to_indices, compression_ratio = self._deduplicate_texts(response_data)

        self.verbose_reporter.stat_line(
            f"Processing {len(response_data)} ideas: {len(unique_texts)} unique texts "
            f"({compression_ratio:.1%} compression)"
        )

        # Create batches
        batch_size = self.config.openai_batch_size
        batches = []
        for i in range(0, len(unique_texts), batch_size):
            batches.append(unique_texts[i:i+batch_size])

        # Process batches concurrently
        max_concurrent = self.config.openai_max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch_texts: List[str]) -> List[np.ndarray]:
            async with semaphore:
                vectors = await self._with_retries(lambda: self._embed_batch(batch_texts))
                if len(vectors) != len(batch_texts):
                    raise RuntimeError(f"API returned {len(vectors)} vectors for {len(batch_texts)} texts")
                return vectors

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

        all_identifiers = response_data

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
                        'instance': getattr(response_idea, 'instance', '') or '',
                        'concept': getattr(response_idea, 'concept', '') or '',
                        'concept_type': getattr(response_idea, 'concept_type', '') or '',
                        'concept_type_definition': getattr(response_idea, 'concept_type_definition', '') or '',
                        'valence': getattr(response_idea, 'valence', '') or '',
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
                embedding_text_format=self.embedding_text_format
            )
            result.append(embeddings_model)

        self.verbose_reporter.stat_line(f"Successfully applied {updated_count} embeddings")

        return result

    # =========================================================================
    # MULTI-PASS EMBEDDING
    # =========================================================================

    def _merge_pass_embeddings(
        self,
        result: List[models.EmbeddingsModel],
        pass_result: List[models.EmbeddingsModel],
        target_field: str
    ) -> int:
        """Merge embeddings from a pass result into a specific field on the base result.

        Each pass produces embeddings in idea_embedding. This method copies those
        into the correct target field (e.g. ladder_embedding).

        Returns:
            Number of embeddings merged.
        """
        merged_count = 0
        for result_resp, pass_resp in zip(result, pass_result):
            if result_resp.response_ideas and pass_resp.response_ideas:
                for result_idea, pass_idea in zip(result_resp.response_ideas, pass_resp.response_ideas):
                    if pass_idea.idea_embedding is not None:
                        setattr(result_idea, target_field, pass_idea.idea_embedding)
                        merged_count += 1
        return merged_count

    def _process_multi_pass_embeddings(
        self,
        data: List[models.EmbeddingsModel],
        pass_specs: list
    ) -> List[models.EmbeddingsModel]:
        """Generic multi-pass embedding processor.

        Runs N embedding passes based on pass_specs, merging each pass's
        embeddings into the correct target field on the result.
        """
        original_format = self.embedding_text_format
        result = None

        for pass_idx, pass_spec in enumerate(pass_specs):
            self.verbose_reporter.stat_line(
                f"\n--- PASS {pass_idx + 1}/{len(pass_specs)}: "
                f"Embedding {pass_spec.label} (format: {pass_spec.text_format}) ---"
            )
            self.embedding_text_format = pass_spec.text_format
            pass_result = asyncio.run(self._process_embeddings_with_id_tracking(data))

            if result is None:
                # First pass: use as base result (idea_embedding is already in place)
                result = pass_result
                if pass_spec.target_field != "idea_embedding":
                    # First pass targets a non-default field; move embeddings
                    merged = self._merge_pass_embeddings(result, pass_result, pass_spec.target_field)
                    for resp in result:
                        if resp.response_ideas:
                            for idea in resp.response_ideas:
                                if getattr(idea, pass_spec.target_field, None) is not None:
                                    idea.idea_embedding = None
                    self.verbose_reporter.stat_line(f"Stored {merged} {pass_spec.target_field} embeddings")
            else:
                # Subsequent passes: merge into target field
                merged = self._merge_pass_embeddings(result, pass_result, pass_spec.target_field)
                self.verbose_reporter.stat_line(f"Merged {merged} {pass_spec.target_field} embeddings")

        # Restore original format and set it on result models
        self.embedding_text_format = original_format
        for resp in result:
            resp.embedding_text_format = original_format

        return result

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_embeddings_with_tracking(self, data: List[models.EmbeddingsModel], var_lab: str = None) -> List[models.EmbeddingsModel]:
        """Generate embeddings with ID tracking.

        Args:
            data: List of EmbeddingsModel (or IdeasExtractedModel) instances
            var_lab: Survey question label

        Returns:
            List of EmbeddingsModel with embeddings applied
        """
        if var_lab is not None:
            self.var_lab = var_lab

        # Handle multi-pass modes via MULTI_PASS_SPECS
        if self.embedding_text_format in MULTI_PASS_SPECS:
            return self._process_multi_pass_embeddings(
                data, MULTI_PASS_SPECS[self.embedding_text_format]
            )

        result = asyncio.run(self._process_embeddings_with_id_tracking(data))

        return result

    def get_analysis(self) -> Optional[EmbeddingAnalysis]:
        """Get embedding analysis results."""
        return self.analysis
