"""
MECE Topic Consolidator - Phase C

Takes candidate themes from Phase B (per-cluster theme generation) and
consolidates them into a MECE (Mutually Exclusive, Collectively Exhaustive)
topic set by merging overlapping themes.

This is an isolated experimental module for step_5_clusterer_v2.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from .config_clusterer_exp import ClustererConfig
    from .prompts_exp import MECE_CONSOLIDATION_PROMPT, MECETopicSet
    from .clusterer_helpers_exp import ClusterTheme
except ImportError:
    from config_clusterer_exp import ClustererConfig
    from prompts_exp import MECE_CONSOLIDATION_PROMPT, MECETopicSet
    from clusterer_helpers_exp import ClusterTheme

from utils.llm import create_client, llm_create_sync


class MECEConsolidator:
    """
    Consolidates per-cluster candidate themes into a MECE topic set.

    Phase C of the V2 pipeline. Takes all candidate themes from Phase B
    and uses an LLM to merge overlapping themes into a clean, non-overlapping
    set of topics with inclusion/exclusion definitions.

    For typical use (15-30 clusters), a single LLM call suffices.

    Usage:
        consolidator = MECEConsolidator(config)
        mece_topics = consolidator.consolidate(
            candidate_themes=themes,
            cluster_keywords=keywords,
            survey_question="What do you associate with X?",
        )
    """

    def __init__(self, config: ClustererConfig, prompt_template=None, response_model=None):
        """Initialize MECEConsolidator.

        Args:
            config: ClustererConfig
            prompt_template: Override prompt template (default: MECE_CONSOLIDATION_PROMPT)
            response_model: Override Pydantic response model (default: MECETopicSet)
        """
        self.config = config
        self._model = getattr(config, 'mece_model', config.llm_labels_model)
        self._temperature = getattr(config, 'mece_temperature', 0.3)
        self._max_tokens = getattr(config, 'mece_max_tokens', 4000)
        self._prompt_template = prompt_template or MECE_CONSOLIDATION_PROMPT
        self._response_model = response_model or MECETopicSet

    def _format_candidate_themes(
        self,
        candidate_themes: Dict[int, ClusterTheme],
        cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    ) -> str:
        """Format all candidate themes for the MECE consolidation prompt."""
        lines = []
        for cluster_id in sorted(candidate_themes.keys()):
            theme = candidate_themes[cluster_id]
            lines.append(f"Cluster {cluster_id} (n={theme.n_ideas}):")
            lines.append(f"  Topic: {theme.theme}")
            lines.append(f"  Definition: {theme.inclusion_definition}")
            if theme.key_concepts:
                lines.append(f"  Key concepts: {', '.join(theme.key_concepts)}")

            # Add top MMR keywords if available
            if cluster_keywords and cluster_id in cluster_keywords:
                kw_list = cluster_keywords[cluster_id]
                kw_str = ", ".join(kw for kw, _ in kw_list[:5])
                lines.append(f"  Keywords: {kw_str}")

            lines.append("")  # Blank line between clusters

        return "\n".join(lines)

    def _build_dataset_context_section(self, dataset_context: Optional[Dict[str, str]]) -> str:
        """Build dataset context section for prompt."""
        if not dataset_context:
            return ""
        parts = []
        if dataset_context.get('domain'):
            parts.append(f"Domain: {dataset_context['domain']}")
        if dataset_context.get('entity'):
            parts.append(f"Entity: {dataset_context['entity']}")
        if dataset_context.get('topic'):
            parts.append(f"Topic: {dataset_context['topic']}")
        if dataset_context.get('perspective'):
            parts.append(f"Perspective: {dataset_context['perspective']}")
        if dataset_context.get('intent'):
            parts.append(f"Intent: {dataset_context['intent']}")
        if not parts:
            return ""
        return "\n" + "\n".join(parts)

    def consolidate(
        self,
        candidate_themes: Dict[int, ClusterTheme],
        cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        taxonomy_axis: Optional[str] = None,
        taxonomy_description: Optional[str] = None,
        verbose: bool = False
    ) -> MECETopicSet:
        """
        Consolidate candidate themes into MECE topics.

        Args:
            candidate_themes: Dict mapping cluster_id to ClusterTheme (Phase B output)
            cluster_keywords: Optional MMR keywords per cluster
            survey_question: The survey question for context
            language: Output language
            dataset_context: Optional dataset metadata (domain, entity, etc.)
            taxonomy_axis: Optional taxonomy dimension constraint
            taxonomy_description: Optional taxonomy description
            verbose: Print progress

        Returns:
            MECETopicSet with consolidated topics
        """
        if verbose:
            print(f"\n[MECE Consolidation (Phase C)]")
            print(f"  Model: {self._model}")
            print(f"  Candidate themes: {len(candidate_themes)}")

        # Build taxonomy context
        if taxonomy_axis:
            taxonomy_context = f"""
<taxonomy_context>
Primary coding dimension: {taxonomy_axis}
Definition: {taxonomy_description or 'Not specified'}
Topics MUST describe content within this dimension ONLY.
</taxonomy_context>
"""
        else:
            taxonomy_context = ""

        # Format all candidate themes
        candidate_themes_list = self._format_candidate_themes(
            candidate_themes, cluster_keywords
        )

        # Build dataset context
        dataset_context_section = self._build_dataset_context_section(dataset_context)

        # Build prompt
        prompt = self._prompt_template.format(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            n_candidate_themes=len(candidate_themes),
            taxonomy_context=taxonomy_context,
            candidate_themes_list=candidate_themes_list,
        )

        if verbose:
            print(f"  Prompt length: ~{len(prompt.split())} words")

        try:
            client = create_client(model=self._model, async_mode=False)
            mece_topics = llm_create_sync(
                client=client,
                model=self._model,
                prompt=prompt,
                response_model=self._response_model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            if verbose:
                print(f"  Result: {len(mece_topics.topics)} MECE topics from {len(candidate_themes)} clusters")
                for topic in mece_topics.topics:
                    cluster_str = ", ".join(str(c) for c in topic.source_cluster_ids)
                    print(f"    - {topic.topic_label} (clusters: {cluster_str})")

            # Validate: check all clusters are accounted for
            all_assigned = set()
            for topic in mece_topics.topics:
                all_assigned.update(topic.source_cluster_ids)
            all_clusters = set(candidate_themes.keys())
            missing = all_clusters - all_assigned
            if missing and verbose:
                print(f"  WARNING: {len(missing)} clusters not assigned to any topic: {sorted(missing)}")

            return mece_topics

        except Exception as e:
            if verbose:
                print(f"  MECE consolidation failed: {type(e).__name__}: {e}")
            raise
