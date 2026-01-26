"""
Clusterer Representation Engine Module

Wraps the existing c-TF-IDF implementation for cluster keyword extraction.
Supports spaCy-based lemmatization with POS filtering.
"""

import re
from typing import Dict, List, Tuple, Optional

import numpy as np

from .config import ClustererV2Config


# Lazy-loaded spaCy model
_SPACY_NLP = None


def get_spacy_nlp(model_name: str = "nl_core_news_lg"):
    """
    Get or load spaCy NLP model (lazy initialization).

    Args:
        model_name: Name of spaCy model to load

    Returns:
        spaCy Language model
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load(model_name, disable=["ner", "parser"])
        except OSError:
            # Model not found, try downloading
            from spacy.cli import download
            download(model_name)
            _SPACY_NLP = spacy.load(model_name, disable=["ner", "parser"])
    return _SPACY_NLP


def extract_noun_phrases_lemmatized(
    texts: List[str],
    nlp=None,
    model_name: str = "nl_core_news_lg"
) -> List[str]:
    """
    Extract lemmatized content words: ADJ, NOUN, PROPN (standalone or in phrases).

    This extracts meaningful content words while filtering out verbs, adverbs,
    and other parts of speech that add noise to c-TF-IDF.

    Pattern: (ADJ | NOUN | PROPN)+
    - Standalone adjectives like "betrouwbaar", "duurzaam" are kept
    - Noun phrases like "groen beleggen" are kept as phrases
    - Verbs, adverbs, etc. are filtered out

    Args:
        texts: List of text strings to process
        nlp: Pre-loaded spaCy model (will load if None)
        model_name: spaCy model name if nlp is None

    Returns:
        List of processed texts with only lemmatized content words
    """
    if nlp is None:
        nlp = get_spacy_nlp(model_name)

    processed = []

    # Process in batches for efficiency
    for doc in nlp.pipe(texts, batch_size=100):
        phrases = []
        current_phrase = []

        for token in doc:
            # Skip punctuation, spaces, and stopwords
            if token.is_punct or token.is_space:
                # End current phrase if we have one
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []
                continue

            # Check POS tag - include ADJ, NOUN, PROPN
            if token.pos_ in ('ADJ', 'NOUN', 'PROPN'):
                current_phrase.append(token.lemma_.lower())
            else:
                # Other POS: end current phrase
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []

        # Handle end of document
        if current_phrase:
            phrases.append(' '.join(current_phrase))

        # Join all phrases for this document
        processed.append(' '.join(phrases))

    return processed


def extract_text_for_display(idea_text: str, template_prefix: Optional[str] = None) -> str:
    """
    Extract clean text for display (strip template prefix if present).

    Args:
        idea_text: Clean idea text (new format - no embedded specifiers)
        template_prefix: The canonical phrasing prefix to strip

    Returns:
        Text for display
    """
    if template_prefix and idea_text.startswith(template_prefix):
        unique_content = idea_text[len(template_prefix):].strip()
        return unique_content if unique_content else idea_text
    return idea_text


def extract_text_for_format(
    idea_text: str,
    taxonomy_phrase: str,
    embedding_text_format: Optional[str]
) -> str:
    """
    Extract text matching what was actually embedded based on embedding_text_format.

    Args:
        idea_text: The idea text (idea.idea)
        taxonomy_phrase: The taxonomy phrase (idea.taxonomy_phrase)
        embedding_text_format: Format used for embedding ("idea" or "taxonomy_phrase")

    Returns:
        Text that matches what was embedded
    """
    if embedding_text_format == "taxonomy_phrase":
        # Use taxonomy_phrase, fallback to idea text
        return taxonomy_phrase if taxonomy_phrase else idea_text

    # Default: "idea" mode - return the idea text directly
    return idea_text


class RepresentationEngine:
    """
    Keyword extraction for clusters using multiple representation methods.

    Supports c-TF-IDF (primary), MMR (diversity-aware), and basic TF-IDF.
    Wraps representation modules from experiments/representation/.

    Usage:
        engine = RepresentationEngine(config)
        all_keywords = engine.extract_all_keywords(cluster_texts)
        # Returns: {"ctfidf": {...}, "mmr": {...}, "tfidf": {...}}
    """

    def __init__(self, config: ClustererV2Config):
        self.config = config
        self._ctfidf = None
        self._mmr = None
        self._tfidf = None
        # Track current ngram_range (may change based on embedding format)
        self._current_ngram_range: Optional[Tuple[int, int]] = None

    def _get_effective_ngram_range(self, embedding_text_format: Optional[str]) -> Tuple[int, int]:
        """
        Determine n-gram range based on embedding text format.

        For taxonomy_phrase format (single semantic units), use unigrams only.
        For regular text (idea format), use configured range (default: unigrams + bigrams).

        Args:
            embedding_text_format: The text format used for embedding ("idea" or "taxonomy_phrase")

        Returns:
            Tuple of (min_n, max_n) for n-gram range
        """
        # Taxonomy phrases are single semantic units - bigrams don't make sense
        if embedding_text_format == "taxonomy_phrase":
            return (1, 1)

        # Regular text (idea format): use configured range
        return self.config.ctfidf_ngram_range

    def _ensure_ctfidf(self, ngram_range: Optional[Tuple[int, int]] = None):
        """
        Lazy initialization of c-TF-IDF model.

        Args:
            ngram_range: Override n-gram range. If different from current, recreates model.
        """
        effective_range = ngram_range or self.config.ctfidf_ngram_range

        # Recreate if ngram_range changed
        if self._ctfidf is not None and self._current_ngram_range != effective_range:
            self._ctfidf = None

        if self._ctfidf is None:
            try:
                # Import from existing representation module
                import sys
                import os
                # Add parent path for imports
                parent_path = os.path.dirname(os.path.dirname(__file__))
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)

                from representation.ctfidf_representation import CTfidfRepresentation

                self._ctfidf = CTfidfRepresentation(
                    top_k=self.config.ctfidf_top_k,
                    bm25_weighting=self.config.ctfidf_bm25_weighting,
                    reduce_frequent_words=self.config.ctfidf_reduce_frequent_words,
                    ngram_range=effective_range,
                    min_df=self.config.ctfidf_min_df,
                    max_df=0.95,
                    language="nl"
                )
                self._current_ngram_range = effective_range
            except ImportError as e:
                raise ImportError(
                    f"Could not import CTfidfRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def _ensure_mmr(self):
        """Lazy initialization of MMR model."""
        if self._mmr is None:
            try:
                import sys
                import os
                parent_path = os.path.dirname(os.path.dirname(__file__))
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)

                from representation.mmr_representation import MMRRepresentation

                self._mmr = MMRRepresentation(
                    diversity=self.config.mmr_diversity,
                    top_k=self.config.ctfidf_top_k,
                    candidate_multiplier=self.config.mmr_candidate_multiplier
                )
            except ImportError as e:
                raise ImportError(
                    f"Could not import MMRRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def _ensure_tfidf(self, ngram_range: Optional[Tuple[int, int]] = None):
        """
        Lazy initialization of basic TF-IDF model.

        Args:
            ngram_range: Override n-gram range. If different from current, recreates model.
        """
        effective_range = ngram_range or self.config.ctfidf_ngram_range

        # Recreate if ngram_range changed
        if self._tfidf is not None and self._current_ngram_range != effective_range:
            self._tfidf = None

        if self._tfidf is None:
            try:
                import sys
                import os
                parent_path = os.path.dirname(os.path.dirname(__file__))
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)

                from representation.tfidf_representation import TfidfRepresentation

                self._tfidf = TfidfRepresentation(
                    top_k=self.config.ctfidf_top_k,
                    ngram_range=effective_range,
                    min_df=self.config.ctfidf_min_df
                )
            except ImportError as e:
                raise ImportError(
                    f"Could not import TfidfRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def extract_keywords(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_taxonomy_phrases: Optional[Dict[int, List[str]]] = None,
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top keywords for each cluster using c-TF-IDF.

        If lemmatization is enabled (config.ctfidf_use_lemmatization), texts are
        processed with spaCy to extract lemmatized noun phrases (ADJ* + NOUN+).

        Args:
            cluster_texts: Dict mapping cluster_id to list of idea texts (idea.idea)
            cluster_taxonomy_phrases: Dict mapping cluster_id to list of taxonomy phrases
            embedding_text_format: The text format used for embedding ("idea" or "taxonomy_phrase")
            verbose: Print progress

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples
        """
        if not self.config.generate_ctfidf:
            return {}

        self._ensure_ctfidf()

        # Clean texts using format-aware extraction
        cleaned_clusters = {}
        for cluster_id, texts in cluster_texts.items():
            taxonomy_list = cluster_taxonomy_phrases.get(cluster_id, []) if cluster_taxonomy_phrases else []
            cleaned_texts = []
            for i, text in enumerate(texts):
                taxonomy = taxonomy_list[i] if i < len(taxonomy_list) else ""
                cleaned_texts.append(
                    extract_text_for_format(text, taxonomy, embedding_text_format)
                )
            cleaned_clusters[cluster_id] = cleaned_texts

        if verbose:
            format_display = embedding_text_format or "idea (default)"
            print(f"  Text format: {format_display}")

        # Apply lemmatization if enabled
        if self.config.ctfidf_use_lemmatization:
            if verbose:
                print("  Applying spaCy lemmatization (ADJ | NOUN | PROPN)...")

            # Flatten all texts for batch processing
            all_texts = []
            cluster_offsets = {}
            offset = 0
            for cluster_id, texts in cleaned_clusters.items():
                cluster_offsets[cluster_id] = (offset, offset + len(texts))
                all_texts.extend(texts)
                offset += len(texts)

            # Process all texts at once
            lemmatized_all = extract_noun_phrases_lemmatized(
                all_texts,
                model_name=self.config.ctfidf_spacy_model
            )

            # Reconstruct clusters with lemmatized texts
            lemmatized_clusters = {}
            for cluster_id, (start, end) in cluster_offsets.items():
                lemmatized_clusters[cluster_id] = lemmatized_all[start:end]

            cleaned_clusters = lemmatized_clusters

        # Extract keywords using c-TF-IDF
        keywords = self._ctfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        return keywords

    def extract_keywords_from_labels(
        self,
        labels: np.ndarray,
        idea_texts: List[str],
        taxonomy_phrases: Optional[List[str]] = None,
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract keywords given cluster labels and idea texts.

        Args:
            labels: Cluster labels for each idea
            idea_texts: List of idea text strings (idea.idea)
            taxonomy_phrases: List of taxonomy phrase strings (idea.taxonomy_phrase)
            embedding_text_format: The text format used for embedding ("idea" or "taxonomy_phrase")
            verbose: Print progress

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples
        """
        # Build cluster_texts and cluster_taxonomy_phrases dicts
        cluster_texts = {}
        cluster_taxonomy_phrases = {}
        for i, label in enumerate(labels):
            if label >= 0:  # Exclude noise
                if label not in cluster_texts:
                    cluster_texts[label] = []
                    cluster_taxonomy_phrases[label] = []
                cluster_texts[label].append(idea_texts[i])
                taxonomy = taxonomy_phrases[i] if taxonomy_phrases and i < len(taxonomy_phrases) else ""
                cluster_taxonomy_phrases[label].append(taxonomy)

        return self.extract_keywords(
            cluster_texts,
            cluster_taxonomy_phrases=cluster_taxonomy_phrases,
            embedding_text_format=embedding_text_format,
            verbose=verbose
        )

    def get_cluster_summary(
        self,
        cluster_id: int,
        keywords: List[Tuple[str, float]],
        max_keywords: int = 10
    ) -> str:
        """
        Generate formatted text summary for a cluster.

        Args:
            cluster_id: Cluster ID
            keywords: List of (keyword, score) tuples
            max_keywords: Maximum keywords to include

        Returns:
            Formatted string summary
        """
        if not keywords:
            return f"Cluster {cluster_id}: (no keywords)"

        kw_strs = [f"{kw} ({score:.3f})" for kw, score in keywords[:max_keywords]]
        return f"Cluster {cluster_id}: {', '.join(kw_strs)}"

    def extract_all_keywords(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_taxonomy_phrases: Optional[Dict[int, List[str]]] = None,
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """
        Extract keywords using all enabled representation methods.

        Returns a dict keyed by method name with keyword dicts as values.
        c-TF-IDF is always run if generate_ctfidf=True.
        MMR and TF-IDF are added based on config flags.

        Args:
            cluster_texts: Dict mapping cluster_id to list of idea texts
            cluster_taxonomy_phrases: Dict mapping cluster_id to taxonomy phrases
            embedding_text_format: The text format used for embedding ("idea" or "taxonomy_phrase")
            verbose: Print progress

        Returns:
            Dict: {"ctfidf": {...}, "mmr": {...}, "tfidf": {...}}
        """
        results = {}

        # Determine effective n-gram range based on embedding format
        effective_ngram_range = self._get_effective_ngram_range(embedding_text_format)

        if verbose:
            if effective_ngram_range != self.config.ctfidf_ngram_range:
                print(f"  N-gram range: {effective_ngram_range} (auto-detected for {embedding_text_format})")
            else:
                print(f"  N-gram range: {effective_ngram_range}")

        # Clean texts using format-aware extraction (shared preprocessing)
        cleaned_clusters = self._preprocess_texts(
            cluster_texts,
            cluster_taxonomy_phrases,
            embedding_text_format,
            verbose
        )

        # Always run c-TF-IDF if enabled (primary)
        if self.config.generate_ctfidf:
            self._ensure_ctfidf(ngram_range=effective_ngram_range)
            if verbose:
                print("\n[c-TF-IDF] Extracting keywords...")
            results["ctfidf"] = self._ctfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        # MMR (uses c-TF-IDF internally, adds diversity)
        if self.config.generate_mmr_keywords:
            self._ensure_ctfidf(ngram_range=effective_ngram_range)  # Need c-TF-IDF for MMR
            self._ensure_mmr()
            if verbose:
                print(f"\n[MMR] Extracting keywords (diversity={self.config.mmr_diversity})...")
            results["mmr"] = self._extract_mmr_keywords(cleaned_clusters, effective_ngram_range, verbose)

        # Basic TF-IDF (independent per-cluster)
        if self.config.generate_tfidf_keywords:
            self._ensure_tfidf(ngram_range=effective_ngram_range)
            if verbose:
                print("\n[TF-IDF] Extracting keywords (per-cluster)...")
            results["tfidf"] = self._tfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        return results

    def _preprocess_texts(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_taxonomy_phrases: Optional[Dict[int, List[str]]],
        embedding_text_format: Optional[str],
        verbose: bool
    ) -> Dict[int, List[str]]:
        """
        Preprocess texts: format extraction and optional lemmatization.

        Returns cleaned cluster texts ready for keyword extraction.
        """
        # Clean texts using format-aware extraction
        cleaned_clusters = {}
        for cluster_id, texts in cluster_texts.items():
            taxonomy_list = cluster_taxonomy_phrases.get(cluster_id, []) if cluster_taxonomy_phrases else []
            cleaned_texts = []
            for i, text in enumerate(texts):
                taxonomy = taxonomy_list[i] if i < len(taxonomy_list) else ""
                cleaned_texts.append(
                    extract_text_for_format(text, taxonomy, embedding_text_format)
                )
            cleaned_clusters[cluster_id] = cleaned_texts

        if verbose:
            format_display = embedding_text_format or "idea (default)"
            print(f"  Text format: {format_display}")

        # Apply lemmatization if enabled
        if self.config.ctfidf_use_lemmatization:
            if verbose:
                print("  Applying spaCy lemmatization (ADJ | NOUN | PROPN)...")

            # Flatten all texts for batch processing
            all_texts = []
            cluster_offsets = {}
            offset = 0
            for cluster_id, texts in cleaned_clusters.items():
                cluster_offsets[cluster_id] = (offset, offset + len(texts))
                all_texts.extend(texts)
                offset += len(texts)

            # Process all texts at once
            lemmatized_all = extract_noun_phrases_lemmatized(
                all_texts,
                model_name=self.config.ctfidf_spacy_model
            )

            # Reconstruct clusters with lemmatized texts
            lemmatized_clusters = {}
            for cluster_id, (start, end) in cluster_offsets.items():
                lemmatized_clusters[cluster_id] = lemmatized_all[start:end]

            cleaned_clusters = lemmatized_clusters

        return cleaned_clusters

    def _extract_mmr_keywords(
        self,
        cleaned_clusters: Dict[int, List[str]],
        ngram_range: Tuple[int, int],
        verbose: bool
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract keywords using MMR (diversity-aware selection).

        MMR works on top of c-TF-IDF: it uses c-TF-IDF scores as relevance
        and adds diversity by penalizing keywords similar to already-selected ones.
        """
        # First compute c-TF-IDF (if not already done)
        cluster_ids = sorted(cleaned_clusters.keys())
        cluster_docs = [" ".join(cleaned_clusters[cid]) for cid in cluster_ids]

        # Build c-TF-IDF matrix
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=self.config.ctfidf_min_df,
            max_df=0.95,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"
        )

        try:
            count_matrix = vectorizer.fit_transform(cluster_docs)
            vocabulary = list(vectorizer.get_feature_names_out())
        except ValueError:
            if verbose:
                print("[MMR] Warning: Vectorization failed")
            return {}

        # Apply c-TF-IDF transformation
        self._ensure_ctfidf()
        ctfidf_matrix = self._ctfidf.transformer.fit_transform(count_matrix)

        # Apply MMR to each cluster
        mmr_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            ctfidf_scores = ctfidf_matrix[idx].toarray()[0]
            cluster_texts = cleaned_clusters[cluster_id]

            keywords = self._mmr.extract_topics(
                cluster_id=cluster_id,
                ctfidf_scores=ctfidf_scores,
                vocabulary=vocabulary,
                cluster_texts=cluster_texts
            )
            mmr_keywords[cluster_id] = keywords

        if verbose:
            print(f"[MMR] Extracted keywords for {len(mmr_keywords)} clusters")

        return mmr_keywords

    def extract_all_keywords_from_labels(
        self,
        labels: np.ndarray,
        idea_texts: List[str],
        taxonomy_phrases: Optional[List[str]] = None,
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """
        Extract all keywords given cluster labels and idea texts.

        Convenience method that builds cluster dicts from labels.

        Returns:
            Dict: {"ctfidf": {...}, "mmr": {...}, "tfidf": {...}}
        """
        # Build cluster_texts and cluster_taxonomy_phrases dicts
        cluster_texts = {}
        cluster_taxonomy_phrases = {}
        for i, label in enumerate(labels):
            if label >= 0:  # Exclude noise
                if label not in cluster_texts:
                    cluster_texts[label] = []
                    cluster_taxonomy_phrases[label] = []
                cluster_texts[label].append(idea_texts[i])
                taxonomy = taxonomy_phrases[i] if taxonomy_phrases and i < len(taxonomy_phrases) else ""
                cluster_taxonomy_phrases[label].append(taxonomy)

        return self.extract_all_keywords(
            cluster_texts,
            cluster_taxonomy_phrases=cluster_taxonomy_phrases,
            embedding_text_format=embedding_text_format,
            verbose=verbose
        )
