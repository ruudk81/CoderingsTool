"""
LLM-enhanced keyword representation

Uses GPT models to refine and enhance statistical keyword extraction.
Combines c-TF-IDF baseline with LLM semantic understanding and consolidation.

The algorithm:
1. Extract baseline keywords using c-TF-IDF
2. Sample representative ideas from cluster
3. Send keywords + ideas to LLM for refinement
4. LLM consolidates synonyms, rephrases for clarity, selects most representative
5. Returns refined keyword list with LLM-assigned relevance scores

This provides highest quality keywords but adds cost and latency.

Usage:
    from experiments.representation.llm_representation import LLMRepresentation

    llm = LLMRepresentation(model="gpt-4.1-mini", top_k=10)
    keywords = llm.extract_topics(
        cluster_id, ctfidf_scores, vocabulary, cluster_texts
    )
"""
import numpy as np
from typing import List, Tuple, Optional
import random
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.llm import create_client, llm_create_sync
from .base import BaseRepresentation


class KeywordItem(BaseModel):
    """A single keyword with justification"""
    keyword: str = Field(description="The keyword (1-3 words)")
    justification: str = Field(description="Brief justification (1 sentence) explaining why this keyword is representative")


class RefinedKeywords(BaseModel):
    """Pydantic model for LLM-refined keywords"""
    keywords: List[KeywordItem] = Field(
        description="List of refined keywords with justifications, ordered by importance"
    )


class LLMRepresentation(BaseRepresentation):
    """
    LLM-based keyword refinement using GPT models

    Args:
        model: LLM model to use (gpt-4.1-mini, gpt-4.1, gpt-5, etc.)
        top_k: Number of refined keywords to return
        candidate_multiplier: How many c-TF-IDF keywords to send to LLM (top_k * multiplier)
        max_ideas_sample: Maximum idea texts to send to LLM for context
        use_reasoning: Use reasoning model (gpt-5, o1) instead of chat model
        verbose: Print LLM calls and token usage
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        top_k: int = 10,
        candidate_multiplier: int = 2,
        max_ideas_sample: int = 10,
        use_reasoning: bool = False,
        verbose: bool = False
    ):
        self.model = model
        self.top_k = top_k
        self.candidate_multiplier = candidate_multiplier
        self.max_ideas_sample = max_ideas_sample
        self.use_reasoning = use_reasoning
        self.verbose = verbose

    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using LLM refinement of c-TF-IDF baseline

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: c-TF-IDF scores for this cluster (1D array)
            vocabulary: Feature names from vectorizer
            cluster_texts: Original idea texts
            embeddings: Not used in this implementation

        Returns:
            List of (keyword, llm_score) tuples
        """
        # Step 1: Get candidate keywords by c-TF-IDF
        n_candidates = min(
            self.top_k * self.candidate_multiplier,
            len([s for s in ctfidf_scores if s > 0])
        )

        if n_candidates == 0:
            return []

        candidate_indices = np.argsort(ctfidf_scores)[-n_candidates:][::-1]
        candidate_keywords = [vocabulary[i] for i in candidate_indices]
        candidate_scores = ctfidf_scores[candidate_indices]

        # Step 2: Sample representative ideas
        if len(cluster_texts) > self.max_ideas_sample:
            sampled_ideas = random.sample(cluster_texts, self.max_ideas_sample)
        else:
            sampled_ideas = cluster_texts

        # Step 3: Build LLM prompt
        prompt = self._build_refinement_prompt(
            cluster_id=cluster_id,
            candidate_keywords=list(zip(candidate_keywords, candidate_scores)),
            sample_ideas=sampled_ideas,
            target_k=self.top_k
        )

        # Step 4: Call LLM for refinement
        if self.verbose:
            print(f"  [LLM-enhanced] Calling {self.model} for keyword refinement...")

        client = create_client(model=self.model, async_mode=False)

        try:
            refined = llm_create_sync(
                client=client,
                prompt=prompt,
                response_model=RefinedKeywords,
                model=self.model
            )

            # Step 5: Convert LLM response to (keyword, score) format
            # Assign scores based on order (first = highest score)
            keywords = []
            for idx, kw_item in enumerate(refined.keywords[:self.top_k]):
                # Score: 1.0 for first, decreasing to ~0.5 for last
                score = 1.0 - (idx / (2 * self.top_k))
                keywords.append((kw_item.keyword, float(score)))

            if self.verbose:
                print(f"  [LLM-enhanced] Refined {len(candidate_keywords)} → {len(keywords)} keywords")

            return keywords

        except Exception as e:
            if self.verbose:
                print(f"  [LLM-enhanced] Error: {e}")
                print(f"  [LLM-enhanced] Falling back to c-TF-IDF baseline")

            # Fallback: return c-TF-IDF baseline
            return [
                (candidate_keywords[i], float(candidate_scores[i]))
                for i in range(min(self.top_k, len(candidate_keywords)))
            ]

    def _build_refinement_prompt(
        self,
        cluster_id: int,
        candidate_keywords: List[Tuple[str, float]],
        sample_ideas: List[str],
        target_k: int
    ) -> str:
        """
        Build LLM prompt for keyword refinement

        Args:
            cluster_id: Cluster identifier
            candidate_keywords: List of (keyword, score) tuples from c-TF-IDF
            sample_ideas: Representative idea texts
            target_k: Target number of refined keywords

        Returns:
            Prompt string
        """
        # Format candidate keywords
        keyword_lines = [
            f"  • {kw} (c-TF-IDF: {score:.3f})"
            for kw, score in candidate_keywords
        ]
        keywords_section = "\n".join(keyword_lines)

        # Format sample ideas
        ideas_section = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(sample_ideas)])

        prompt = f"""You are an expert in qualitative data analysis and keyword extraction.

<context>
Cluster ID: {cluster_id}
Number of ideas: {len(sample_ideas)}
</context>

<statistical_keywords>
Statistical analysis (c-TF-IDF) identified these candidate keywords:

{keywords_section}

These keywords were selected based on their statistical importance in distinguishing this cluster from others.
</statistical_keywords>

<cluster_ideas>
Representative ideas from this cluster:

{ideas_section}
</cluster_ideas>

<task>
Your task is to refine and consolidate these statistical keywords to create the {target_k} most representative keywords for this cluster.

**Instructions:**
1. Review the statistical keywords and cluster ideas
2. Consolidate synonyms or near-duplicates (e.g., "price" and "pricing" → "price")
3. Rephrase keywords for clarity if needed (prefer concrete nouns/verbs)
4. Select the {target_k} keywords that best capture the cluster's essence
5. Order keywords by representativeness (most important first)
6. For each keyword, provide a brief justification (1 sentence)

**Requirements:**
- Each keyword should be 1-3 words
- Keywords should be distinct (no redundancy)
- Keywords should be grounded in the actual ideas
- Prefer concrete terms over abstract concepts
- Focus on what makes this cluster distinctive
</task>

<output_format>
Return exactly {target_k} refined keywords, ordered by importance.
Each keyword should have a justification explaining why it's representative.
</output_format>"""

        return prompt

    def get_refinement_details(
        self,
        cluster_id: int,
        ctfidf_keywords: List[Tuple[str, float]],
        llm_keywords: List[Tuple[str, float]]
    ) -> dict:
        """
        Compare c-TF-IDF baseline with LLM refinement

        Args:
            cluster_id: Cluster identifier
            ctfidf_keywords: Original c-TF-IDF keywords
            llm_keywords: LLM-refined keywords

        Returns:
            Dict with comparison analysis
        """
        ctfidf_set = set(kw for kw, _ in ctfidf_keywords)
        llm_set = set(kw for kw, _ in llm_keywords)

        return {
            "cluster_id": cluster_id,
            "ctfidf_keywords": len(ctfidf_keywords),
            "llm_keywords": len(llm_keywords),
            "retained": list(ctfidf_set & llm_set),
            "removed": list(ctfidf_set - llm_set),
            "added": list(llm_set - ctfidf_set),
            "retention_rate": len(ctfidf_set & llm_set) / len(ctfidf_set) if ctfidf_set else 0.0
        }
