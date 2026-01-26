"""
Clusterer Preprocessing Module

Handles:
- L2 normalization of embeddings
- PCA dimensionality reduction for large datasets
- Embedding extraction from EmbeddingsModel
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import models
from .config import ClustererV2Config


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings.

    Args:
        embeddings: Array of shape (n_samples, n_features)

    Returns:
        L2-normalized embeddings (unit vectors)
    """
    return normalize(embeddings, norm='l2', axis=1)


def apply_pca(
    embeddings: np.ndarray,
    n_components: float = 0.99,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Variance to retain (0.99 = 99%)
        random_state: Random seed

    Returns:
        (reduced_embeddings, fitted_pca_model)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca


def extract_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererV2Config
) -> Tuple[np.ndarray, List[str], List[str], List[Tuple[int, int]], Optional[str], Optional[str]]:
    """
    Extract embeddings from EmbeddingsModel list.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererV2Config

    Returns:
        embeddings: Array of shape (n_ideas, embedding_dim)
        idea_texts: List of idea text strings (idea.idea)
        taxonomy_phrases: List of taxonomy phrase strings (idea.taxonomy_phrase)
        idea_indices: List of (response_idx, idea_idx) tuples for result mapping
        template_prefix: The canonical phrasing prefix (if available)
        embedding_text_format: The text format used for embedding (if available)
    """
    embeddings_list = []
    idea_texts = []
    taxonomy_phrases = []
    idea_indices = []
    template_prefix = None
    embedding_text_format = None

    for resp_idx, response in enumerate(input_list):
        # Extract template_prefix from first response that has it
        if template_prefix is None and hasattr(response, 'template_prefix') and response.template_prefix:
            template_prefix = response.template_prefix

        # Extract embedding_text_format from first response that has it
        if embedding_text_format is None and hasattr(response, 'embedding_text_format') and response.embedding_text_format:
            embedding_text_format = response.embedding_text_format

        if response.response_ideas:
            for idea_idx, idea in enumerate(response.response_ideas):
                if idea.idea_embedding is not None:
                    embeddings_list.append(idea.idea_embedding)
                    idea_texts.append(idea.idea if hasattr(idea, 'idea') else str(idea))
                    taxonomy_phrases.append(
                        idea.taxonomy_phrase if hasattr(idea, 'taxonomy_phrase') and idea.taxonomy_phrase else ""
                    )
                    idea_indices.append((resp_idx, idea_idx))

    if not embeddings_list:
        raise ValueError("No embeddings found in input data")

    embeddings = np.vstack(embeddings_list)

    if config.verbose:
        print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        if template_prefix:
            prefix_display = template_prefix[:50] + "..." if len(template_prefix) > 50 else template_prefix
            print(f"Template prefix: '{prefix_display}'")
        if embedding_text_format:
            print(f"Embedding text format: {embedding_text_format}")

    return embeddings, idea_texts, taxonomy_phrases, idea_indices, template_prefix, embedding_text_format


def preprocess_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererV2Config
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[Tuple[int, int]], Optional[PCA], Optional[str], Optional[str]]:
    """
    Full preprocessing pipeline: extract, normalize, optionally PCA.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererV2Config

    Returns:
        embeddings_normalized: L2-normalized original embeddings
        embeddings_processed: Processed embeddings (may be PCA-reduced)
        idea_texts: List of idea text strings (idea.idea)
        taxonomy_phrases: List of taxonomy phrase strings (idea.taxonomy_phrase)
        idea_indices: List of (response_idx, idea_idx) tuples
        pca_model: Fitted PCA model (or None if not applied)
        template_prefix: The canonical phrasing prefix (if available)
        embedding_text_format: The text format used for embedding (if available)
    """
    # Extract embeddings
    embeddings, idea_texts, taxonomy_phrases, idea_indices, template_prefix, embedding_text_format = extract_embeddings(input_list, config)
    n_samples = len(embeddings)

    # L2 normalize
    embeddings_normalized = l2_normalize(embeddings)

    if config.verbose:
        print(f"L2-normalized {n_samples} embeddings")

    # Apply PCA for large datasets
    pca_model = None
    if n_samples > config.pca_threshold:
        if config.verbose:
            print(f"Applying PCA (n > {config.pca_threshold})...")
        embeddings_processed, pca_model = apply_pca(
            embeddings_normalized,
            n_components=config.pca_variance_retained,
            random_state=config.umap_random_state
        )
        # Re-normalize after PCA
        embeddings_processed = l2_normalize(embeddings_processed)
        if config.verbose:
            print(f"PCA reduced to {embeddings_processed.shape[1]} components")
    else:
        embeddings_processed = embeddings_normalized

    return embeddings_normalized, embeddings_processed, idea_texts, taxonomy_phrases, idea_indices, pca_model, template_prefix, embedding_text_format
