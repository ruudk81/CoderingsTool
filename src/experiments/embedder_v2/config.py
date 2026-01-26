"""
Embedder V2 Experiment Configuration

Experimental settings for embedding generation that can be modified
without affecting production config.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


# =============================================================================
# HARDCODED SETTINGS FOR "BOTH" MODE
# =============================================================================
# Which idea format to use when embedding_text_format="both"
# Options: "idea" or "idea_without_template_prefix"
BOTH_MODE_IDEA_FORMAT = "idea_without_template_prefix"


@dataclass
class EmbedderExperimentConfig:
    """Configuration for the embedder experiment."""

    # ==========================================================================
    # DATASET SETTINGS (matching pipeline.py selection)
    # ==========================================================================
    filename: str = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column: str = "DLNMID"
    var_name: str = "Q20"
    sample_size: int = 50

    # ==========================================================================
    # EMBEDDING PROVIDER SETTINGS
    # ==========================================================================
    # Provider: "openai" or "gemini"
    provider: str = "openai"

    # Model selection (provider-specific)
    # OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    # Gemini: "models/text-embedding-004", "models/embedding-001"
    embedding_model: Optional[str] = None  # None = use default from ModelConfig

    # ==========================================================================
    # TEXT FORMAT SETTINGS
    # ==========================================================================
    # What text to embed:
    # - "idea": Embed the clean idea text (idea.idea field)
    # - "taxonomy_phrase": Embed the taxonomy phrase (idea.taxonomy_phrase field)
    # - "idea_without_template_prefix": Embed idea text with template_prefix stripped
    # - "both": Embed BOTH idea text AND taxonomy_phrase (stores in idea_embedding + taxonomy_embedding)
    embedding_text_format: Literal["idea", "taxonomy_phrase", "idea_without_template_prefix", "both"] = "idea"

    # ==========================================================================
    # BATCH PROCESSING SETTINGS
    # ==========================================================================
    # OpenAI settings
    openai_batch_size: int = 100
    openai_max_concurrent: int = 5

    # Gemini settings
    gemini_batch_size: int = 20
    gemini_max_concurrent: int = 10

    # ==========================================================================
    # QUESTION-AWARE EMBEDDING SETTINGS
    # ==========================================================================
    # Enable question-aware embedding transformation
    use_question_aware: bool = False

    # Weights for combining embeddings (must sum to 1.0)
    response_weight: float = 0.6      # Weight for response embeddings
    question_weight: float = 0.3      # Weight for question embeddings
    domain_anchor_weight: float = 0.1  # Weight for domain-relative positioning

    # ==========================================================================
    # EXPERIMENTAL FEATURES
    # ==========================================================================
    # Use experimental embedder implementation
    use_experimental_embedder: bool = True

    # Enable embedding quality analysis
    analyze_embeddings: bool = True

    # Compute pairwise similarity statistics
    compute_similarity_stats: bool = True

    # Number of sample embeddings to display
    sample_output_count: int = 10

    # ==========================================================================
    # OUTPUT SETTINGS
    # ==========================================================================
    verbose: bool = True
    save_results_to_file: bool = True

    def to_embedding_config(self):
        """Convert to production EmbeddingConfig for compatibility."""
        from config import EmbeddingConfig

        return EmbeddingConfig(
            batch_size=self.openai_batch_size,
            max_concurrent_requests=self.openai_max_concurrent,
            gemini_batch_size=self.gemini_batch_size,
            gemini_max_concurrent=self.gemini_max_concurrent,
            use_question_aware=self.use_question_aware,
            response_weight=self.response_weight,
            question_weight=self.question_weight,
            domain_anchor_weight=self.domain_anchor_weight,
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Default configuration for quick experiments
DEFAULT_EXPERIMENT_CONFIG = EmbedderExperimentConfig()

# Configuration for taxonomy phrase embedding
TAXONOMY_PHRASE_CONFIG = EmbedderExperimentConfig(
    embedding_text_format="taxonomy_phrase",
    analyze_embeddings=True,
    compute_similarity_stats=True,
)

# Configuration for idea embedding without template prefix
IDEA_WITHOUT_PREFIX_CONFIG = EmbedderExperimentConfig(
    embedding_text_format="idea_without_template_prefix",
    analyze_embeddings=True,
    compute_similarity_stats=True,
)

# Configuration for dual embedding (both idea and taxonomy_phrase)
BOTH_EMBEDDINGS_CONFIG = EmbedderExperimentConfig(
    embedding_text_format="both",
    analyze_embeddings=True,
    compute_similarity_stats=True,
)

# Configuration for question-aware embedding experiments
QUESTION_AWARE_CONFIG = EmbedderExperimentConfig(
    use_question_aware=True,
    response_weight=0.6,
    question_weight=0.3,
    domain_anchor_weight=0.1,
    analyze_embeddings=True,
)

# Configuration for Gemini provider experiments
GEMINI_CONFIG = EmbedderExperimentConfig(
    provider="gemini",
    embedding_model="models/text-embedding-004",
    gemini_batch_size=20,
    gemini_max_concurrent=10,
)

# Large dataset configuration (Merk X)
ASN_BANK_CONFIG = EmbedderExperimentConfig(
    filename="M000000 Associatiemonitor Merk X net databestand.sav",
    id_column="DLNMID",
    var_name="Qd1_combined",
    sample_size=2000,
)

# Pinkpop dataset configuration
PINKPOP_CONFIG = EmbedderExperimentConfig(
    filename="M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav",
    id_column="DLNMID",
    var_name="Q15",
    sample_size=2000,
)
