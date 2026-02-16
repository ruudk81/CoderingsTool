"""
Stage 1: Object Discovery — V4

Discovers MECE objects by clustering unique ontology items (typically categories)
using the same clustering pipeline (Phases 1-5), then generating per-cluster
themes and consolidating into MECE objects.

Adapted from V2's object_discovery.py (notebook-style) into a class-based module.

Usage:
    from .object_discovery import ObjectDiscoverer
    from .config_clusterer_exp import ClustererConfig

    discoverer = ObjectDiscoverer(config, extraction_metadata)
    mece_objects, clusterer, cat_names, cat_labels = discoverer.discover(embeddings_models)
"""

import asyncio
from typing import List, Dict, Optional, Tuple

from collections import Counter

import numpy as np

from experiments import models_exp as models
from utils.llm import create_client, llm_create_sync, create_embedding_client
from config import get_embedding_model_for_api

from .clusterer_exp import Clusterer
from .config_clusterer_exp import ClustererConfig
from .clusterer_helpers_exp import ClusterTheme, ThemeGenerator
from .prompts_exp import (
    CLUSTER_OBJECT_PROMPT, ClusterThemeDescription,
    FRAMING_QUESTION_PROMPT, FramingQuestionResult,
    MECE_OBJECT_CONSOLIDATION_PROMPT, MECEObjectDescription, MECEObjectSet,
)

# Import template lookup for taxonomy-driven framing
try:
    from experiments.step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
except ImportError:
    try:
        from step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
    except ImportError:
        TEMPLATE_LOOKUP = None


# Map ontology level → pre-computed embedding field (None = embed on the fly)
LEVEL_TO_EMBEDDING_FIELD = {
    "instance": "idea_embedding",
    "node": "node_embedding",
    "category": None,
    "root": None,
}


class ObjectDiscoverer:
    """
    Discovers MECE objects from clustered ontology items.

    Three-step process:
      1. Extract unique items at configured ontology level (e.g. category)
      2. Embed + cluster them via the V4 Clusterer (phases 1-5)
      3. Generate per-cluster themes → consolidate into MECE objects
    """

    def __init__(
        self,
        config: ClustererConfig,
        extraction_metadata=None,
    ):
        self.config = config
        self._extraction_metadata = extraction_metadata
        self._level = config.object_discovery_level

        # Pre-compute enriched taxonomy context from Step 3 metadata + template lookup
        self._taxonomy_context_str = self._build_enriched_taxonomy_context()

    def _build_enriched_taxonomy_context(self) -> str:
        """
        Build an enriched taxonomy_context block from ExtractionMetadata + template_lookup.

        Uses the full axis definition (allowed_concepts, excluded_concepts,
        dimension_description) from template_lookup.py rather than the thin
        3-line version that only passes axis name + description.

        Falls back to the thin version when template_lookup is unavailable.
        """
        meta = self._extraction_metadata
        if not meta:
            return ""

        taxonomy_axis = getattr(meta, 'taxonomy_axis', '') or ''
        taxonomy_description = getattr(meta, 'taxonomy_axis_description', '') or ''
        actionable_type = getattr(meta, 'taxonomy_actionable_type', '') or ''

        if not taxonomy_axis:
            return ""

        # Try enriched lookup from template_lookup.py
        if TEMPLATE_LOOKUP and taxonomy_axis in TEMPLATE_LOOKUP.get("axes", {}):
            axis_info = TEMPLATE_LOOKUP["axes"][taxonomy_axis]
            noun_desc = axis_info.get("noun_phrase_descriptor", taxonomy_axis)
            dimension_desc = axis_info.get("dimension_description", taxonomy_description)
            allowed = axis_info.get("allowed_concepts", [])
            excluded = axis_info.get("excluded_concepts", [])

            lines = [
                "<taxonomy_context>",
                f"Primary coding dimension: {noun_desc}",
                f"Dimension: {dimension_desc}",
            ]
            if allowed:
                lines.append(f"Allowed concept types: {', '.join(allowed)}")
            if excluded:
                lines.append(f"Excluded concept types: {', '.join(excluded)}")
            if actionable_type:
                lines.append(f"Actionable type: {actionable_type}")
            lines.append("")
            lines.append("Topics MUST describe content within this dimension ONLY.")
            if excluded:
                lines.append(f"Do NOT create topics/objects about: {', '.join(excluded)}")
            lines.append("</taxonomy_context>")
            return "\n".join(lines)

        # Fallback: thin taxonomy context
        return (
            f"<taxonomy_context>\n"
            f"Primary coding dimension: {taxonomy_axis}\n"
            f"Definition: {taxonomy_description or 'Not specified'}\n"
            f"Topics MUST describe content within this dimension ONLY.\n"
            f"</taxonomy_context>"
        )

    def get_taxonomy_context(self) -> str:
        """Public accessor for the enriched taxonomy context string."""
        return self._taxonomy_context_str

    def get_taxonomy_axis_info(self) -> Optional[Dict]:
        """Return the template_lookup axis info dict, or None if unavailable."""
        meta = self._extraction_metadata
        axis = getattr(meta, 'taxonomy_axis', '') if meta else ''
        if axis and TEMPLATE_LOOKUP and axis in TEMPLATE_LOOKUP.get("axes", {}):
            return TEMPLATE_LOOKUP["axes"][axis]
        return None

    def discover(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Tuple[MECEObjectSet, Clusterer, List[str], np.ndarray]:
        """
        Run full object discovery pipeline.

        Returns:
            (mece_objects, clusterer, category_names, cluster_labels)
            - mece_objects: MECEObjectSet with consolidated MECE objects
            - clusterer: Clusterer instance (phases 1-5 complete)
            - category_names: List of unique category names (aligned with labels)
            - cluster_labels: np.ndarray of cluster labels per unique category
        """
        print(f"\n{'='*70}")
        print(f"STAGE 1: Object Discovery (level={self._level})")
        print(f"{'='*70}")

        # Step 1: Extract unique items
        names, embeddings, metadata = self._extract_unique_items(embeddings_models)

        # Step 2: Wrap + cluster
        synthetic_models = self._wrap_as_embeddings_models(names, embeddings)
        clusterer = self._run_clustering(synthetic_models)
        labels = clusterer.get_labels()

        # Print detailed cluster info (probability histograms, samples)
        clusterer.print_all_clusters(n_samples=10)

        # Step 3: Generate themes per cluster
        themes = self._generate_object_themes(clusterer, names, metadata)

        # Step 3.5: Select framing question for consistent object level
        framing_result = self._select_framing_question(themes)

        # Step 4: Consolidate into MECE objects (with framing constraint)
        mece_objects = self._consolidate_objects(themes, framing_result)

        # Print nodes-per-object listing
        self._print_objects_with_nodes(mece_objects, names, labels, metadata)

        # Summary
        n_noise = int(np.sum(labels == -1))
        print(f"\n  Object Discovery Summary:")
        print(f"  {len(names)} unique {self._level}s → "
              f"{len(themes)} clusters → "
              f"{len(mece_objects.topics)} MECE objects")
        print(f"  Noise: {n_noise} items ({n_noise/len(names):.1%})")

        return mece_objects, clusterer, names, labels

    # =========================================================================
    # Step 1: Extract unique ontology items
    # =========================================================================

    def _extract_unique_items(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Tuple[List[str], np.ndarray, Dict[str, dict]]:
        """
        Extract unique ontology items at the configured level with embeddings.

        Uses pre-computed embeddings when available (instance, node).
        Generates embeddings on the fly for levels without a dedicated field (category, root).
        """
        level = self._level
        embedding_field = LEVEL_TO_EMBEDDING_FIELD.get(level)

        item_ideas: Dict[str, List] = {}
        item_embeddings: Dict[str, List[np.ndarray]] = {}
        n_ideas_total = 0
        n_empty = 0
        n_missing_emb = 0

        for resp in embeddings_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                n_ideas_total += 1
                # Map legacy level name to new model field
                attr_name = "semantic_category" if level == "category" else level
                item_name = (getattr(idea, attr_name, "") or "").strip()
                if not item_name:
                    n_empty += 1
                    continue

                if item_name not in item_ideas:
                    item_ideas[item_name] = []
                    if embedding_field:
                        item_embeddings[item_name] = []
                item_ideas[item_name].append(idea)

                if embedding_field:
                    emb = getattr(idea, embedding_field, None)
                    if emb is not None:
                        item_embeddings[item_name].append(np.array(emb, dtype=np.float32))
                    else:
                        n_missing_emb += 1

        if not item_ideas:
            raise ValueError(f"No valid '{level}' texts found in the data.")

        if n_empty > 0:
            print(f"  Note: {n_empty}/{n_ideas_total} ideas have empty '{level}' field")

        names = sorted(item_ideas.keys())

        if embedding_field:
            if n_missing_emb > 0:
                print(f"  WARNING: {n_missing_emb}/{n_ideas_total} ideas have no {embedding_field}")
            names = [n for n in names if item_embeddings.get(n)]
            if not names:
                raise ValueError(
                    f"All ideas have {embedding_field}=None. "
                    f"Run step 4 with embedding_text_format that includes '{level}'."
                )
            averaged = [np.stack(item_embeddings[n]).mean(axis=0) for n in names]
            embeddings_matrix = np.stack(averaged)
            print(f"  Using pre-computed embeddings from '{embedding_field}' "
                  f"(averaged per unique {level})")
        else:
            print(f"  No pre-computed embedding for '{level}' level — generating on the fly...")
            text_to_emb = asyncio.run(self._embed_texts(names))
            embeddings_matrix = np.stack([text_to_emb[n] for n in names])
            print(f"  Embedded {len(names)} unique {level} texts")

        # Build metadata per unique item
        metadata = {}
        for name in names:
            ideas = item_ideas[name]
            cat_c = Counter(i.semantic_category for i in ideas if i.semantic_category)
            root_c = Counter(i.root for i in ideas if i.root)
            metadata[name] = {
                "count": len(ideas),
                "category": cat_c.most_common(1)[0][0] if cat_c else "",
                "root": root_c.most_common(1)[0][0] if root_c else "",
            }

        print(f"\n  Item extraction ({level} level):")
        print(f"    Total ideas: {n_ideas_total}")
        print(f"    Unique {level}s: {len(names)}")
        print(f"    Embedding shape: {embeddings_matrix.shape}")

        return names, embeddings_matrix, metadata

    async def _embed_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Embed a list of unique text strings using the configured embedding provider."""
        client = create_embedding_client(async_mode=True)
        model = get_embedding_model_for_api()
        response = await client.embeddings.create(input=texts, model=model)
        return {
            text: np.array(item.embedding, dtype=np.float32)
            for text, item in zip(texts, response.data)
        }

    # =========================================================================
    # Step 2: Wrap + cluster
    # =========================================================================

    def _wrap_as_embeddings_models(
        self, names: List[str], embeddings: np.ndarray
    ) -> List[models.EmbeddingsModel]:
        """Wrap unique items into EmbeddingsModel format for the Clusterer."""
        wrapped = []
        for idx, (name, emb) in enumerate(zip(names, embeddings)):
            idea = models.EmbeddingsSubmodel(
                idea_id=f"obj_{idx}_0",
                idea=name,
                node=name,
                idea_embedding=emb,
            )
            resp = models.EmbeddingsModel(
                respondent_id=f"obj_{idx}",
                response=name,
                response_ideas=[idea],
                embedding_text_format="idea",
            )
            wrapped.append(resp)
        return wrapped

    def _run_clustering(
        self, synthetic_models: List[models.EmbeddingsModel]
    ) -> Clusterer:
        """Run V4 Clusterer (phases 1-5) on the synthetic models."""
        cluster_config = ClustererConfig(
            embedding_source="idea_embedding",
            generate_mece_topics=False,
            verbose=self.config.verbose,
        )
        clusterer = Clusterer(
            synthetic_models,
            config=cluster_config,
            extraction_metadata=self._extraction_metadata,
        )
        clusterer.run()
        return clusterer

    # =========================================================================
    # Step 3: Generate per-cluster themes
    # =========================================================================

    def _generate_object_themes(
        self,
        clusterer: Clusterer,
        names: List[str],
        metadata: Dict[str, dict],
    ) -> Dict[int, ClusterTheme]:
        """Generate per-cluster object themes using LLM."""
        labels = clusterer.get_labels()
        unique_labels = sorted(set(labels) - {-1})

        # Build cluster_texts: cluster_id → list of item names
        cluster_texts: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            label_int = int(label)
            if label_int not in cluster_texts:
                cluster_texts[label_int] = []
            cluster_texts[label_int].append(names[idx])

        # Build dataset context from extraction metadata
        survey_question = ""
        language = "Dutch"
        dataset_context = None
        if self._extraction_metadata:
            meta = self._extraction_metadata
            survey_question = getattr(meta, 'var_lab', '') or ''
            language = getattr(meta, 'lang', 'Dutch') or 'Dutch'
            dataset_context = {}
            for field in ('domain', 'entity', 'topic', 'perspective', 'intent'):
                val = getattr(meta, field, None)
                if val:
                    dataset_context[field] = val

        # Create ThemeGenerator with object prompt
        generator = ThemeGenerator(
            self.config,
            prompt_template=CLUSTER_OBJECT_PROMPT,
            response_model=ClusterThemeDescription,
        )

        print(f"\n  Generating object themes for {len(cluster_texts)} clusters...")
        themes = generator.generate_all_themes(
            cluster_texts=cluster_texts,
            extraction_metadata=self._extraction_metadata,
            survey_question=survey_question,
            language=language,
            dataset_context=dataset_context,
            verbose=self.config.verbose,
        )

        for cid, theme in sorted(themes.items()):
            print(f"    Cluster {cid}: {theme.theme}")

        return themes

    # =========================================================================
    # Step 3.5: Select framing question (taxonomy-driven, with LLM fallback)
    # =========================================================================

    def _select_framing_question(
        self, candidate_themes: Dict[int, ClusterTheme]
    ) -> FramingQuestionResult:
        """
        Select an analytic framing question that all MECE objects must answer.

        Primary path: deterministic lookup from ExtractionMetadata + template_lookup.py.
        Fallback: LLM-based derivation via FRAMING_QUESTION_PROMPT (when taxonomy info missing).
        """
        meta = self._extraction_metadata
        axis = getattr(meta, 'taxonomy_axis', '') if meta else ''

        # Try deterministic taxonomy lookup first
        if axis and TEMPLATE_LOOKUP and axis in TEMPLATE_LOOKUP.get("axes", {}):
            return self._framing_from_taxonomy(meta, axis)

        # Fallback: LLM-based derivation
        print(f"\n  No taxonomy axis info available — falling back to LLM-based framing question...")
        return self._framing_from_llm(candidate_themes)

    def _framing_from_taxonomy(
        self, meta: models.ExtractionMetadata, axis: str
    ) -> FramingQuestionResult:
        """Build framing question deterministically from cached taxonomy info."""
        axis_info = TEMPLATE_LOOKUP["axes"][axis]
        allowed = axis_info["allowed_concepts"]
        dimension_desc = axis_info["dimension_description"]
        noun_desc = axis_info.get("noun_phrase_descriptor", axis)
        entity = getattr(meta, 'entity', '') or 'the entity'
        actionable_type = getattr(meta, 'taxonomy_actionable_type', '') or ''

        # Construct framing question from allowed concepts + entity
        if len(allowed) > 1:
            concepts_str = ", ".join(allowed[:-1]) + f", or {allowed[-1]}"
        else:
            concepts_str = allowed[0] if allowed else "concept"
        framing_question = f"What {concepts_str} of {entity} is the respondent describing?"

        # Use the context-specific axis description if available, else the generic one
        analytic_question = getattr(meta, 'taxonomy_axis_description', '') or dimension_desc

        # Level description combines dimension + actionable type
        level_parts = [dimension_desc]
        if actionable_type:
            level_parts.append(f"The actionable concept type is: {actionable_type}.")
        level_description = "\n".join(level_parts)

        print(f"\n  Framing question derived from taxonomy (axis={axis}, no LLM call):")
        print(f"  Analytic question: {analytic_question}")
        print(f"  Framing question:  {framing_question}")
        print(f"  Allowed concepts:  {', '.join(allowed)}")
        excluded = axis_info.get("excluded_concepts", [])
        if excluded:
            print(f"  Excluded concepts: {', '.join(excluded)}")

        return FramingQuestionResult(
            analytic_question=analytic_question,
            framing_question=framing_question,
            level_description=level_description,
        )

    def _framing_from_llm(
        self, candidate_themes: Dict[int, ClusterTheme]
    ) -> FramingQuestionResult:
        """Fallback: use LLM to derive framing question when taxonomy info is missing."""
        survey_question = ""
        language = "Dutch"
        dataset_context_section = ""

        if self._extraction_metadata:
            meta = self._extraction_metadata
            survey_question = getattr(meta, 'var_lab', '') or ''
            language = getattr(meta, 'lang', 'Dutch') or 'Dutch'
            dataset_context = {}
            for field_name in ('domain', 'entity', 'topic', 'perspective', 'intent'):
                val = getattr(meta, field_name, None)
                if val:
                    dataset_context[field_name] = val
            if dataset_context:
                parts = [f"{k.title()}: {v}" for k, v in dataset_context.items()]
                dataset_context_section = "\n".join(parts)

        # Use enriched taxonomy context (may be empty if no taxonomy info)
        taxonomy_context = self._taxonomy_context_str

        candidate_themes_list = self._format_candidate_themes(candidate_themes)

        prompt = FRAMING_QUESTION_PROMPT.format(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            taxonomy_context=taxonomy_context,
            n_candidate_themes=len(candidate_themes),
            candidate_themes_list=candidate_themes_list,
        )

        print(f"\n  Selecting framing question via LLM from {len(candidate_themes)} candidate themes...")

        client = create_client(
            model=self.config.object_mece_model, async_mode=False
        )
        framing_result = llm_create_sync(
            client=client,
            model=self.config.object_mece_model,
            prompt=prompt,
            response_model=FramingQuestionResult,
            temperature=self.config.object_mece_temperature,
            max_tokens=1000,
        )

        print(f"  Analytic question: {framing_result.analytic_question}")
        print(f"  Framing question:  {framing_result.framing_question}")
        print(f"  Interpretation:    {framing_result.level_description}")

        return framing_result

    # =========================================================================
    # Step 4: Consolidate into MECE objects
    # =========================================================================

    def _consolidate_objects(
        self,
        candidate_themes: Dict[int, ClusterTheme],
        framing_result: Optional[FramingQuestionResult] = None,
    ) -> MECEObjectSet:
        """Consolidate per-cluster themes into MECE objects."""
        # Build dataset context
        survey_question = ""
        language = "Dutch"
        dataset_context = None

        if self._extraction_metadata:
            meta = self._extraction_metadata
            survey_question = getattr(meta, 'var_lab', '') or ''
            language = getattr(meta, 'lang', 'Dutch') or 'Dutch'
            dataset_context = {}
            for field_name in ('domain', 'entity', 'topic', 'perspective', 'intent'):
                val = getattr(meta, field_name, None)
                if val:
                    dataset_context[field_name] = val

        # Use enriched taxonomy context (pre-computed from template_lookup)
        taxonomy_context = self._taxonomy_context_str

        # Format candidate themes
        candidate_themes_list = self._format_candidate_themes(candidate_themes)

        # Build dataset context section
        dataset_context_section = ""
        if dataset_context:
            parts = []
            for key in ('domain', 'entity', 'topic', 'perspective', 'intent'):
                if dataset_context.get(key):
                    parts.append(f"{key.title()}: {dataset_context[key]}")
            if parts:
                dataset_context_section = "\n" + "\n".join(parts)

        # Build framing constraint context
        framing_question = ""
        level_description = ""
        if framing_result:
            framing_question = framing_result.framing_question
            level_description = framing_result.level_description

        # Build prompt
        prompt = MECE_OBJECT_CONSOLIDATION_PROMPT.format(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            n_candidate_themes=len(candidate_themes),
            taxonomy_context=taxonomy_context,
            candidate_themes_list=candidate_themes_list,
            framing_question=framing_question,
            level_description=level_description,
        )

        print(f"\n  Consolidating {len(candidate_themes)} cluster themes into MECE objects...")

        client = create_client(
            model=self.config.object_mece_model, async_mode=False
        )
        mece_objects = llm_create_sync(
            client=client,
            model=self.config.object_mece_model,
            prompt=prompt,
            response_model=MECEObjectSet,
            temperature=self.config.object_mece_temperature,
            max_tokens=self.config.object_mece_max_tokens,
        )

        # Validate all clusters assigned
        all_assigned = set()
        for obj in mece_objects.topics:
            all_assigned.update(obj.source_cluster_ids)
        all_clusters = set(candidate_themes.keys())
        missing = all_clusters - all_assigned
        if missing:
            print(f"  WARNING: {len(missing)} clusters not assigned: {sorted(missing)}")

        print(f"  Result: {len(mece_objects.topics)} MECE objects")
        for obj in mece_objects.topics:
            cluster_str = ", ".join(str(c) for c in obj.source_cluster_ids)
            print(f"    - {obj.topic_label} (clusters: {cluster_str})")

        return mece_objects

    def _format_candidate_themes(
        self, candidate_themes: Dict[int, ClusterTheme]
    ) -> str:
        """Format all candidate themes for the consolidation prompt."""
        lines = []
        for cluster_id in sorted(candidate_themes.keys()):
            theme = candidate_themes[cluster_id]
            lines.append(f"Cluster {cluster_id} (n={theme.n_ideas}):")
            lines.append(f"  Topic: {theme.theme}")
            lines.append(f"  Definition: {theme.inclusion_definition}")
            if theme.key_concepts:
                lines.append(f"  Key concepts: {', '.join(theme.key_concepts)}")
            lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Display: Nodes-per-object listing
    # =========================================================================

    def _print_objects_with_nodes(
        self,
        mece_objects: MECEObjectSet,
        names: List[str],
        labels: np.ndarray,
        metadata: Dict[str, dict],
    ):
        """Print which nodes belong to each MECE object with occurrence counts."""
        print(f"\n{'─'*70}")
        print(f"MECE OBJECTS → NODES MAPPING")
        print(f"{'─'*70}")

        for i, obj in enumerate(mece_objects.topics):
            source_nodes = []
            for cid in obj.source_cluster_ids:
                mask = labels == cid
                for idx in range(len(names)):
                    if mask[idx]:
                        source_nodes.append((names[idx], metadata[names[idx]]["count"]))
            source_nodes.sort(key=lambda x: -x[1])
            node_strs = [f"{name} ({count}x)" for name, count in source_nodes[:10]]
            print(f"\n  [{i+1}] {obj.topic_label}")
            print(f"      Inclusion: {obj.inclusion_definition}")
            print(f"      Boundary test: {obj.boundary_test}")
            signals_str = ", ".join(obj.diagnostic_signals[:5]) if obj.diagnostic_signals else "(none)"
            print(f"      Diagnostic signals: {signals_str}")
            print(f"      Nodes ({len(source_nodes)}): {', '.join(node_strs)}")
            if len(source_nodes) > 10:
                print(f"             ... and {len(source_nodes) - 10} more")
