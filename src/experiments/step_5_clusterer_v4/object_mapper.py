"""
Stage 2: Object Mapper — V4

Maps MECE objects (from Stage 1) back to all idea instances via the chain:
  MECE object → source_cluster_ids → category labels → ideas

This enables Stage 3 to run Map-Reduce MECE on the ideas belonging to each object.

Usage:
    from .object_mapper import ObjectMapper

    mapper = ObjectMapper(mece_objects, cat_names, cat_labels, embeddings_models)
    object_mappings = mapper.map_objects_to_ideas(text_source="idea")
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

from experiments import models_exp as models
from .prompts_exp import MECEObjectSet, MECEObjectDescription
from .clusterer_helpers_exp import get_idea_field_text


@dataclass
class ObjectIdeaMapping:
    """Mapping of a single MECE object to its ideas."""
    object_label: str
    mece_object: MECEObjectDescription
    idea_texts: List[str]
    category_names: List[str]
    idea_count: int


class ObjectMapper:
    """
    Maps MECE objects to their constituent ideas.

    Mapping chain:
      1. MECEObject.source_cluster_ids → cluster labels
      2. Cluster labels → unique category names (from Stage 1 clustering)
      3. Category names → all ideas with matching idea.semantic_category
      4. Extract text from ideas using configured text source
    """

    def __init__(
        self,
        mece_objects: MECEObjectSet,
        category_names: List[str],
        category_labels: np.ndarray,
        embeddings_models: List[models.EmbeddingsModel],
    ):
        """
        Args:
            mece_objects: MECE object set from Stage 1
            category_names: List of unique category names (aligned with category_labels)
            category_labels: np.ndarray of cluster labels per unique category
            embeddings_models: Original Step 4 embeddings (full idea data)
        """
        self._mece_objects = mece_objects
        self._category_names = category_names
        self._category_labels = category_labels
        self._embeddings_models = embeddings_models

    def map_objects_to_ideas(
        self, text_source: str = "idea"
    ) -> Dict[str, ObjectIdeaMapping]:
        """
        Map each MECE object to its ideas.

        Args:
            text_source: Which text field to extract from ideas.
                "idea", "node", "category", "root", "ontology", "instance"

        Returns:
            Dict mapping object_label → ObjectIdeaMapping
        """
        print(f"\n{'='*70}")
        print(f"STAGE 2: Map Objects to Ideas (text_source={text_source})")
        print(f"{'='*70}")

        # Step 1: Build cluster_id → category names mapping
        cluster_to_categories = self._build_cluster_to_categories()

        # Step 2: Build category → ideas mapping (from original embeddings)
        category_to_ideas = self._build_category_to_ideas()

        # Step 3: Map each MECE object to its ideas
        mappings: Dict[str, ObjectIdeaMapping] = {}
        total_mapped = 0

        for obj in self._mece_objects.topics:
            # Get all categories for this object via source_cluster_ids
            obj_categories = []
            for cid in obj.source_cluster_ids:
                obj_categories.extend(cluster_to_categories.get(cid, []))
            obj_categories = sorted(set(obj_categories))

            # Get all ideas for these categories
            idea_texts = []
            for cat_name in obj_categories:
                ideas = category_to_ideas.get(cat_name, [])
                for idea in ideas:
                    text = get_idea_field_text(idea, text_source)
                    if text:
                        idea_texts.append(text)

            mapping = ObjectIdeaMapping(
                object_label=obj.topic_label,
                mece_object=obj,
                idea_texts=idea_texts,
                category_names=obj_categories,
                idea_count=len(idea_texts),
            )
            mappings[obj.topic_label] = mapping
            total_mapped += len(idea_texts)

            print(f"  {obj.topic_label}: "
                  f"{len(obj_categories)} categories → "
                  f"{len(idea_texts)} ideas")

        # Check for unmapped ideas
        total_ideas = sum(
            len(resp.response_ideas or [])
            for resp in self._embeddings_models
        )
        print(f"\n  Total ideas mapped: {total_mapped} / {total_ideas}")
        if total_mapped < total_ideas:
            print(f"  Unmapped: {total_ideas - total_mapped} ideas "
                  f"(noise categories or empty fields)")

        return mappings

    def _build_cluster_to_categories(self) -> Dict[int, List[str]]:
        """Build mapping: cluster_id → list of category names."""
        cluster_to_cats: Dict[int, List[str]] = {}
        n_noise = 0

        for idx, label in enumerate(self._category_labels):
            if label == -1:
                n_noise += 1
                continue
            label_int = int(label)
            if label_int not in cluster_to_cats:
                cluster_to_cats[label_int] = []
            cluster_to_cats[label_int].append(self._category_names[idx])

        if n_noise > 0:
            print(f"  Note: {n_noise} categories are noise (label=-1), skipped")

        return cluster_to_cats

    def _build_category_to_ideas(self) -> Dict[str, list]:
        """Build mapping: category_name → list of idea objects (all instances)."""
        category_to_ideas: Dict[str, list] = {}

        for resp in self._embeddings_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                cat = (getattr(idea, 'semantic_category', '') or '').strip()
                if not cat:
                    continue
                if cat not in category_to_ideas:
                    category_to_ideas[cat] = []
                category_to_ideas[cat].append(idea)

        return category_to_ideas
