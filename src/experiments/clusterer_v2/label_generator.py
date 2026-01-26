"""
Clusterer Label Generator Module

LLM-based cluster label generation using the existing generate_cluster_description()
functionality from experiments/cluster_analysis.py.
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.llm import llm_create_sync, create_client
from .config import ClustererV2Config


# Old prompt (V3) - kept for reference
CLUSTER_DESCRIPTION_PROMPT_V3 = """You are a qualitative researcher analyzing survey response clusters.

<context>
Survey question: "{survey_question}"
Language: {language}{dataset_context_section}
</context>

<cluster_info>
Cluster ID: {cluster_id}
Number of {sample_type}: {num_ideas}
</cluster_info>
{taxonomy_context}{cluster_profile_section}{keywords_section}
<most_representative_{samples_tag}>
The following {sample_type} are most representative of this cluster:
{ideas_list}
</most_representative_{samples_tag}>

<task>
Analyze this cluster and provide a thematic summary:
1. Review the most representative {sample_type} carefully
2. Consider the cluster profile (sentiment/sense distributions) as additional context
3. Consider the statistical keywords as indicators of cluster-specific content
4. Identify the common atomic theme that unifies these {sample_type}{taxonomy_task_guidance}
5. Extract 3-5 key concepts present in this cluster
</task>

<output_format>
Provide your analysis in {language}:
- theme: Short atomic thematic label{taxonomy_output_constraint} (3-10 words)
- description: Clear description (1-2 sentences)
- key_concepts: List of 3-5 key concepts/themes
</output_format>"""

# New grounded MECE prompt (V4) - active
CLUSTER_DESCRIPTION_PROMPT = CLUSTER_DESCRIPTION_PROMPT_V4 = """You are a qualitative researcher labeling survey-response clusters.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>
<instruction>The theme label must read as a natural-language answer category to the survey question.</instruction>
{taxonomy_context}
<cluster_evidence>
Cluster ID: {cluster_id}
Number of {sample_type}: {num_ideas}

<representative_{samples_tag}>
These {sample_type} are representative of the cluster:
{ideas_list}
</representative_{samples_tag}>
{keywords_section}{cluster_profile_section}
</cluster_evidence>

<task>
1. Review the representative {sample_type} to identify common meaning.
2. Use the statistical keywords to sharpen what makes this cluster distinct.
3. Identify the common atomic theme expressed directly in the data.
4. Do not introduce concepts not supported by the {sample_type} or keywords.
5. Ensure the theme stays strictly within the taxonomy dimension{taxonomy_task_guidance}.
6. Ensure the theme reads as a short, noun-phrased natural-language answer to the survey question. Use the essence as the head noun, avoid generic language, clutter and verbs.
</task>

<output_format>
Provide your analysis in {language}:
- theme: Short noun-phrased label{taxonomy_output_constraint} (3-10 words)
- description: 1-2 sentence explanation of what respondents associate with the entity
- key_concepts: 3-5 concrete concepts grounded in data (from keywords or representative samples)
</output_format>"""


class ClusterDescription(BaseModel):
    """LLM-generated cluster description."""
    theme: str = Field(..., description="Short noun-phrased thematic label (3-10 words), reads as answer to survey question")
    description: str = Field(..., description="1-2 sentence explanation of what respondents associate with the entity")
    key_concepts: List[str] = Field(..., description="3-5 concrete concepts grounded in data (from keywords or samples)")


@dataclass
class ClusterLabel:
    """Container for cluster label information."""
    cluster_id: int
    theme: str
    description: str
    key_concepts: List[str]
    n_ideas: int


class LabelGenerator:
    """
    LLM-based cluster label generator.

    Uses the existing generate_cluster_description pattern from cluster_analysis.py
    to create thematic labels for clusters.

    Usage:
        generator = LabelGenerator(config)
        labels = generator.generate_all_labels(cluster_texts, cluster_keywords)
    """

    def __init__(self, config: ClustererV2Config):
        """
        Initialize LabelGenerator.

        Args:
            config: ClustererV2Config with LLM settings
        """
        self.config = config
        self._model = config.llm_labels_model
        self._max_ideas = config.llm_max_ideas_per_cluster

    def _get_sample_terminology(
        self,
        embedding_text_format: Optional[str]
    ) -> Tuple[str, str]:
        """
        Get terminology for samples based on embedding text format.

        Args:
            embedding_text_format: The text format used for embedding ("idea" or "taxonomy_phrase")

        Returns:
            Tuple of (tag_name, display_name):
            - tag_name: XML tag name (e.g., "taxonomy_phrases", "response_ideas")
            - display_name: Human-readable name (e.g., "taxonomy phrases", "response ideas")
        """
        if embedding_text_format == "taxonomy_phrase":
            return ("taxonomy_phrases", "taxonomy phrases")
        return ("response_ideas", "response ideas")

    def _build_dataset_context_section(self, dataset_context: Optional[Dict[str, str]]) -> str:
        """
        Build dataset context section for prompt (only if fields are populated).

        Args:
            dataset_context: Dict with keys: domain, topic, entity, perspective, intent

        Returns:
            Formatted string section or empty string if no context
        """
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

    def _build_cluster_profile_section(self, distributions: Optional[Dict[str, Dict[str, float]]]) -> str:
        """
        Build cluster profile section showing sentiment/sense distributions.

        Args:
            distributions: Dict with keys: sentiment, sense, taxonomy_phrases
                          Each value is a dict of {value: percentage}

        Returns:
            Formatted XML section or empty string if no meaningful distributions
        """
        if not distributions:
            return ""
        parts = []

        # Sentiment (skip if all neutral or empty)
        sent = distributions.get('sentiment', {})
        if sent and not (len(sent) == 1 and 'neutral' in sent):
            sent_str = ", ".join(f"{int(v*100)}% {k}" for k, v in sent.items())
            parts.append(f"Sentiment: {sent_str}")

        # Sense (skip if all factual or empty)
        sense = distributions.get('sense', {})
        if sense and not (len(sense) == 1 and 'factual' in sense):
            sense_str = ", ".join(f"{int(v*100)}% {k}" for k, v in sense.items())
            parts.append(f"Sense: {sense_str}")

        if not parts:
            return ""
        return f"""
<cluster_profile>
{chr(10).join(parts)}
Note: Do NOT encode sentiment or evaluation into the theme label.
</cluster_profile>
"""

    def generate_label(
        self,
        cluster_id: int,
        ideas: List[str],
        representative_samples: Optional[List[Tuple[str, float]]] = None,
        keywords: Optional[List[Tuple[str, float]]] = None,
        taxonomy_axis: Optional[str] = None,
        taxonomy_description: Optional[str] = None,
        taxonomy_actionable_type: Optional[str] = None,
        embedding_text_format: Optional[str] = None,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        cluster_distributions: Optional[Dict[str, Dict[str, float]]] = None,
        verbose: bool = False,
        return_prompt: bool = False
    ) -> ClusterLabel:
        """
        Generate LLM-based label for a single cluster.

        Args:
            cluster_id: Cluster identifier
            ideas: List of idea texts in the cluster (used for counting)
            representative_samples: Optional pre-selected representative samples
                                   as list of (text, similarity_score) tuples.
                                   If provided, these are used instead of random sampling.
            keywords: Optional list of (keyword, score) tuples from MMR/c-TF-IDF
            taxonomy_axis: Optional taxonomy axis (e.g., "WHAT", "WHO", "HOW")
            taxonomy_description: Optional description of the taxonomy axis
            embedding_text_format: Text format used for embeddings (for dynamic terminology)
            survey_question: Research question for context
            language: Language of the responses
            dataset_context: Optional dict with domain, topic, entity, perspective, intent
            cluster_distributions: Optional dict with sentiment, sense, taxonomy_phrases distributions
            verbose: Print progress
            return_prompt: If True, return (label, prompt) tuple

        Returns:
            ClusterLabel with theme, description, and key concepts
            (or tuple of (ClusterLabel, prompt_str) if return_prompt=True)
        """
        # Use representative samples if provided, otherwise fall back to random sampling
        if representative_samples:
            sample_ideas = [text for text, _ in representative_samples]
        else:
            sample_ideas = ideas
            if len(ideas) > self._max_ideas:
                sample_ideas = random.sample(ideas, self._max_ideas)

        # Deduplicate samples (preserve order, keep first occurrence)
        seen = set()
        unique_samples = []
        for idea in sample_ideas:
            if idea not in seen:
                seen.add(idea)
                unique_samples.append(idea)
        sample_ideas = unique_samples

        # Get dynamic terminology based on embedding format
        samples_tag, sample_type = self._get_sample_terminology(embedding_text_format)

        # Format ideas list
        ideas_formatted = "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(sample_ideas))

        # Build taxonomy context section (only if taxonomy info provided)
        if taxonomy_axis:
            actionable_type = taxonomy_actionable_type or "concepts"
            taxonomy_context = f"""
<taxonomy_context>
Primary coding dimension: {taxonomy_axis}
Definition: {taxonomy_description or 'Not specified'}
Actionable type: {actionable_type}
Labels MUST describe content within this dimension ONLY.
Do NOT include sentiment, evaluation, tone, or respondent intent in the label.
</taxonomy_context>
"""
            taxonomy_task_guidance = f" ({taxonomy_axis}: {actionable_type})"
            taxonomy_output_constraint = f" within the {taxonomy_axis} dimension"
        else:
            taxonomy_context = ""
            taxonomy_task_guidance = ""
            taxonomy_output_constraint = ""

        # Build keywords section (only if keywords provided)
        if keywords:
            kw_formatted = "\n".join(f"{i+1}. {kw}" for i, (kw, score) in enumerate(keywords[:10]))
            keywords_section = f"""
<statistical_keywords>
These terms statistically differentiate this cluster from others (c-TF-IDF).
Use to refine — but not override — the representative {sample_type}:
{kw_formatted}
</statistical_keywords>
"""
        else:
            keywords_section = ""

        # Build dataset context section (domain, entity, topic, etc.)
        dataset_context_section = self._build_dataset_context_section(dataset_context)

        # Build cluster profile section (sentiment/sense distributions)
        cluster_profile_section = self._build_cluster_profile_section(cluster_distributions)

        # Build prompt
        prompt = CLUSTER_DESCRIPTION_PROMPT.format(
            language=language,
            survey_question=survey_question,
            cluster_id=cluster_id,
            num_ideas=len(ideas),
            samples_tag=samples_tag,
            sample_type=sample_type,
            taxonomy_context=taxonomy_context,
            taxonomy_task_guidance=taxonomy_task_guidance,
            taxonomy_output_constraint=taxonomy_output_constraint,
            keywords_section=keywords_section,
            dataset_context_section=dataset_context_section,
            cluster_profile_section=cluster_profile_section,
            ideas_list=ideas_formatted
        )

        # Call LLM
        try:
            client = create_client(model=self._model, async_mode=False)
            description = llm_create_sync(
                client=client,
                model=self._model,
                prompt=prompt,
                response_model=ClusterDescription,
                temperature=0.3,
                max_tokens=1000
            )

            label = ClusterLabel(
                cluster_id=cluster_id,
                theme=description.theme,
                description=description.description,
                key_concepts=description.key_concepts,
                n_ideas=len(ideas)
            )
            return (label, prompt) if return_prompt else label

        except Exception as e:
            if verbose:
                print(f"  LLM Error for cluster {cluster_id}: {type(e).__name__}: {e}")

            # Return fallback
            label = ClusterLabel(
                cluster_id=cluster_id,
                theme=f"Cluster {cluster_id}",
                description="LLM label generation failed",
                key_concepts=[],
                n_ideas=len(ideas)
            )
            return (label, prompt) if return_prompt else label

    def generate_all_labels(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        representative_samples: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        extraction_metadata=None,
        embedding_text_format: Optional[str] = None,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        cluster_distributions: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
        verbose: bool = False
    ) -> Dict[int, ClusterLabel]:
        """
        Generate LLM-based labels for all clusters.

        Args:
            cluster_texts: Dict mapping cluster_id to list of idea texts
            cluster_keywords: Optional dict mapping cluster_id to keywords
            representative_samples: Optional dict mapping cluster_id to list of
                                   (text, similarity_score) tuples selected by
                                   centroid similarity
            extraction_metadata: Optional ExtractionMetadata for taxonomy context
            embedding_text_format: Text format used for embeddings (for terminology)
            survey_question: Research question for context
            language: Language of the responses
            dataset_context: Optional dict with domain, topic, entity, perspective, intent
            cluster_distributions: Optional dict mapping cluster_id to sentiment/sense/taxonomy distributions
            verbose: Print progress

        Returns:
            Dict mapping cluster_id to ClusterLabel
        """
        if not self.config.generate_llm_labels:
            return {}

        # Extract taxonomy info from metadata if available
        taxonomy_axis = None
        taxonomy_description = None
        taxonomy_actionable_type = None
        if extraction_metadata:
            taxonomy_axis = getattr(extraction_metadata, 'taxonomy_primary_axis', None)
            taxonomy_description = getattr(extraction_metadata, 'taxonomy_axis_description', None)
            taxonomy_actionable_type = getattr(extraction_metadata, 'taxonomy_actionable_type', None)

        if verbose:
            print(f"\n[LLM Label Generation]")
            print(f"  Model: {self._model}")
            print(f"  Clusters to label: {len(cluster_texts)}")
            if taxonomy_axis:
                print(f"  Taxonomy axis: {taxonomy_axis}")
            if dataset_context:
                context_parts = [f"{k}={v}" for k, v in dataset_context.items() if v]
                if context_parts:
                    print(f"  Dataset context: {', '.join(context_parts)}")
            if cluster_distributions:
                print(f"  Including cluster profiles (sentiment/sense distributions)")

        labels = {}
        sample_prompt = None  # Capture first prompt for verbose display

        for i, (cluster_id, ideas) in enumerate(sorted(cluster_texts.items())):
            keywords = cluster_keywords.get(cluster_id) if cluster_keywords else None
            samples = representative_samples.get(cluster_id) if representative_samples else None
            distributions = cluster_distributions.get(cluster_id) if cluster_distributions else None
            is_first = (i == 0)

            if verbose:
                sample_info = f", {len(samples)} representative" if samples else ""
                print(f"  Generating label for cluster {cluster_id} ({len(ideas)} ideas{sample_info})...", end=" ")

            result = self.generate_label(
                cluster_id=cluster_id,
                ideas=ideas,
                representative_samples=samples,
                keywords=keywords,
                taxonomy_axis=taxonomy_axis,
                taxonomy_description=taxonomy_description,
                taxonomy_actionable_type=taxonomy_actionable_type,
                embedding_text_format=embedding_text_format,
                survey_question=survey_question,
                language=language,
                dataset_context=dataset_context,
                cluster_distributions=distributions,
                verbose=verbose,
                return_prompt=is_first  # Get prompt for first cluster only
            )

            # Unpack result based on whether we requested the prompt
            if is_first:
                label, sample_prompt = result
            else:
                label = result
            labels[cluster_id] = label

            if verbose:
                # Truncate theme if too long
                theme_display = label.theme[:50] + "..." if len(label.theme) > 50 else label.theme
                print(f"'{theme_display}'")

        # Display sample prompt at end of verbose output
        if verbose and sample_prompt:
            print(f"\n  [Sample LLM Prompt (cluster 0)]")
            print("  " + "-" * 70)
            for line in sample_prompt.split('\n'):
                print(f"  {line}")
            print("  " + "-" * 70)

        return labels
