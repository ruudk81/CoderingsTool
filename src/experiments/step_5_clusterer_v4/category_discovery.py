"""
Semantic-Category-Based Discovery — V4

Alternative to ObjectDiscoverer + ObjectMapper that skips clustering entirely.
Partitions ideas by their `semantic_category` field (6 fixed groups: identity,
attribute, function, state, evaluation, relation) and constructs MECEObjectSet +
ObjectIdeaMapping from the dimension taxonomy.

Each semantic category becomes a MECE "object", and the existing Stage 3
MapReduce MECE pipeline runs per category to discover topics within it.

Usage:
    from .category_discovery import CategoryBasedDiscoverer

    discoverer = CategoryBasedDiscoverer(config, extraction_metadata)
    mece_objects, object_mappings = discoverer.discover(embeddings_models)
"""

from typing import List, Dict, Optional, Tuple

from experiments import models_exp as models

from .config_clusterer_exp import ClustererConfig
from .object_mapper import ObjectIdeaMapping
from .clusterer_helpers_exp import get_idea_field_text
from .prompts_exp import MECEObjectDescription, MECEObjectSet

# Import template lookup for taxonomy-driven context
try:
    from experiments.step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
except ImportError:
    try:
        from step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
    except ImportError:
        TEMPLATE_LOOKUP = None


# The 6 fixed semantic categories from the taxonomy
SEMANTIC_CATEGORIES = [
    "identity", "attribute", "function", "state", "evaluation", "relation"
]

# Base diagnostic signals per category (axis-independent fallback)
BASE_DIAGNOSTIC_SIGNALS = {
    "identity": ["is een", "type", "soort", "categorie", "geclassificeerd als"],
    "attribute": ["heeft", "eigenschap", "kwaliteit", "kenmerk", "feature"],
    "function": ["doet", "dient", "doel", "actie", "rol", "functie"],
    "state": ["momenteel", "tijdelijk", "situatie", "conditie", "ervaring"],
    "evaluation": ["goed", "slecht", "mening", "oordeel", "vind", "beoordeling"],
    "relation": ["verbindt", "tussen", "relatie", "afhankelijkheid", "vergeleken met"],
}

# Grouping axis instructions per category — tells the LLM to find sub-types
# of the semantic category, not subject-matter topics that happen to be in it.
CATEGORY_GROUPING_INSTRUCTIONS = {
    "identity": (
        "Coding categories must represent DIFFERENT TYPES OF IDENTITY "
        "(e.g., organizational type, sector classification, mission type) — "
        "not different entities or subjects being identified."
    ),
    "attribute": (
        "Coding categories must represent DIFFERENT TYPES OF ATTRIBUTES "
        "(e.g., financial attribute, service attribute, ethical attribute) — "
        "not different evaluative stances toward the same attribute."
    ),
    "function": (
        "Coding categories must represent DIFFERENT TYPES OF FUNCTIONS "
        "(e.g., financial function, advisory function, social function) — "
        "not different subjects performing a function."
    ),
    "state": (
        "Coding categories must represent DIFFERENT TYPES OF CONDITIONS "
        "(e.g., transitional state, stable state, emerging state, declining state) — "
        "not different subjects that happen to be in a condition."
    ),
    "evaluation": (
        "Coding categories must represent DIFFERENT TYPES OF EVALUATIONS "
        "(e.g., quality judgment, trust judgment, value-for-money judgment) — "
        "not different subjects being evaluated."
    ),
    "relation": (
        "Coding categories must represent DIFFERENT TYPES OF RELATIONS "
        "(e.g., competitive relation, partnership, dependency, comparison) — "
        "not different entities involved in a relation."
    ),
}

# Generic fallback descriptions (when no taxonomy axis is available)
GENERIC_DESCRIPTIONS = {
    "identity": {
        "inclusion": "Ideas describing what something IS — its type, class, or essential nature.",
        "boundary": "Does this statement classify or name what the entity fundamentally is?",
    },
    "attribute": {
        "inclusion": "Ideas describing what something HAS — inherent properties, qualities, or characteristics.",
        "boundary": "Does this statement describe an inherent property or quality of the entity?",
    },
    "function": {
        "inclusion": "Ideas describing what something DOES — its purpose, actions, role, or capabilities.",
        "boundary": "Does this statement describe an action, purpose, or function of the entity?",
    },
    "state": {
        "inclusion": "Ideas describing the CONDITION something is in — time-bound or situational states.",
        "boundary": "Does this statement describe a time-bound or situational condition?",
    },
    "evaluation": {
        "inclusion": "Ideas expressing a JUDGMENT about something — subjective assessments or value judgments.",
        "boundary": "Does this statement express a subjective judgment or evaluation?",
    },
    "relation": {
        "inclusion": "Ideas describing how something CONNECTS to other things — dependencies, trade-offs, or associations.",
        "boundary": "Does this statement describe a relationship, dependency, or connection between entities?",
    },
}


class CategoryBasedDiscoverer:
    """
    Alternative to ObjectDiscoverer + ObjectMapper.

    Partitions ideas by semantic_category field (6 fixed groups)
    and constructs MECEObjectSet + ObjectIdeaMapping from the taxonomy,
    bypassing clustering entirely.
    """

    def __init__(
        self,
        config: ClustererConfig,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
    ):
        self.config = config
        self._extraction_metadata = extraction_metadata
        self._taxonomy_axis = (
            getattr(extraction_metadata, 'taxonomy_axis', '')
            if extraction_metadata else ''
        )
        self._taxonomy_axis_info = self._load_axis_info()
        self._dimension_taxonomy = self._load_dimension_taxonomy()
        self._taxonomy_context_str = self._build_enriched_taxonomy_context()

    # =========================================================================
    # TAXONOMY LOOKUPS
    # =========================================================================

    def _load_axis_info(self) -> Optional[Dict]:
        """Load axis info from TEMPLATE_LOOKUP['axes'][taxonomy_axis]."""
        if (
            self._taxonomy_axis
            and TEMPLATE_LOOKUP
            and self._taxonomy_axis in TEMPLATE_LOOKUP.get("axes", {})
        ):
            return TEMPLATE_LOOKUP["axes"][self._taxonomy_axis]
        return None

    def _load_dimension_taxonomy(self) -> Optional[Dict]:
        """Load the dimension-specific taxonomy interpretations for the active axis."""
        if self._taxonomy_axis and TEMPLATE_LOOKUP:
            dim_tax = TEMPLATE_LOOKUP.get("dimension_taxonomy", {})
            dimensions = dim_tax.get("dimensions", {})
            return dimensions.get(self._taxonomy_axis)
        return None

    def get_taxonomy_axis_info(self) -> Optional[Dict]:
        """Return the template_lookup axis info dict, or None if unavailable."""
        return self._taxonomy_axis_info

    def get_taxonomy_context(self) -> str:
        """Public accessor for the enriched taxonomy context string."""
        return self._taxonomy_context_str

    def get_grouping_instructions(self) -> Dict[str, str]:
        """Return grouping instructions for each populated semantic category.

        These are injected into the <instruction> section of MAP/REDUCE/MECE
        prompts to ensure the LLM finds sub-types of the semantic category
        rather than subject-matter topics.

        Must be called after discover().
        """
        return {
            cat: CATEGORY_GROUPING_INSTRUCTIONS[cat]
            for cat in self._populated_categories
            if cat in CATEGORY_GROUPING_INSTRUCTIONS
        }

    def _build_enriched_taxonomy_context(self) -> str:
        """Build enriched taxonomy_context block (mirrors ObjectDiscoverer)."""
        meta = self._extraction_metadata
        if not meta:
            return ""

        taxonomy_axis = getattr(meta, 'taxonomy_axis', '') or ''
        taxonomy_description = getattr(meta, 'taxonomy_axis_description', '') or ''
        actionable_type = getattr(meta, 'taxonomy_actionable_type', '') or ''

        if not taxonomy_axis:
            return ""

        if self._taxonomy_axis_info:
            axis_info = self._taxonomy_axis_info
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

        return (
            f"<taxonomy_context>\n"
            f"Primary coding dimension: {taxonomy_axis}\n"
            f"Definition: {taxonomy_description or 'Not specified'}\n"
            f"Topics MUST describe content within this dimension ONLY.\n"
            f"</taxonomy_context>"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def discover(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Tuple[MECEObjectSet, Dict[str, ObjectIdeaMapping]]:
        """
        Partition ideas by semantic_category and build MECE structures.

        Returns:
            (mece_objects, object_mappings) — ready for Stage 3
        """
        print(f"\n{'='*70}")
        print(f"CATEGORY-BASED DISCOVERY (semantic_category partition)")
        print(f"{'='*70}")

        if self._taxonomy_axis:
            print(f"  Taxonomy axis: {self._taxonomy_axis}")
        if self._dimension_taxonomy:
            print(f"  Axis interpretations available: yes")

        # Step 1: Partition ideas by semantic_category
        text_source = self.config.mapreduce_text_source
        category_ideas = self._partition_ideas(embeddings_models, text_source)

        # Filter empty categories
        populated = {k: v for k, v in category_ideas.items() if v}
        self._populated_categories = populated

        print(f"\n  Semantic categories found: {sorted(populated.keys())}")
        for cat in sorted(populated.keys()):
            print(f"    {cat}: {len(populated[cat])} ideas")

        total_ideas = sum(len(v) for v in populated.values())
        all_ideas_count = sum(
            len(resp.response_ideas or []) for resp in embeddings_models
        )
        if total_ideas < all_ideas_count:
            print(f"  Note: {all_ideas_count - total_ideas} ideas have empty semantic_category")

        # Check for unknown categories
        unknown = set(populated.keys()) - set(SEMANTIC_CATEGORIES)
        if unknown:
            print(f"  WARNING: Unknown semantic_category values: {unknown}")

        # Step 2: Build MECE objects from taxonomy
        mece_objects = self._build_mece_object_set(populated)

        print(f"\n  MECE Objects ({len(mece_objects.topics)}):")
        for obj in mece_objects.topics:
            print(f"    - {obj.topic_label}: {obj.inclusion_definition[:80]}...")

        # Step 3: Build object mappings
        object_mappings = self._build_object_mappings(mece_objects, populated)

        skipped = set(SEMANTIC_CATEGORIES) - set(populated.keys())
        print(f"\n  Category Discovery Summary:")
        print(f"    {len(populated)} populated categories → {len(mece_objects.topics)} MECE objects")
        print(f"    {total_ideas} total ideas mapped")
        if skipped:
            print(f"    Empty categories (skipped): {sorted(skipped)}")

        return mece_objects, object_mappings

    # =========================================================================
    # IDEA PARTITIONING
    # =========================================================================

    def _partition_ideas(
        self,
        embeddings_models: List[models.EmbeddingsModel],
        text_source: str = "idea",
    ) -> Dict[str, List[str]]:
        """
        Partition all idea texts by semantic_category.

        Returns:
            Dict mapping category_name → list of idea texts
        """
        category_ideas: Dict[str, List[str]] = {}

        for resp in embeddings_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                cat = (getattr(idea, 'semantic_category', '') or '').strip().lower()
                if not cat:
                    continue
                if cat not in category_ideas:
                    category_ideas[cat] = []
                text = get_idea_field_text(idea, text_source)
                if text:
                    category_ideas[cat].append(text)

        return category_ideas

    # =========================================================================
    # MECE OBJECT CONSTRUCTION
    # =========================================================================

    def _build_mece_object_set(
        self,
        populated_categories: Dict[str, List[str]],
    ) -> MECEObjectSet:
        """Build MECEObjectSet from all populated semantic categories."""
        objects = []
        sorted_cats = sorted(populated_categories.keys())
        for idx, category in enumerate(sorted_cats):
            obj = self._build_mece_object_for_category(
                category, len(populated_categories[category]), idx
            )
            objects.append(obj)
        return MECEObjectSet(topics=objects)

    def _build_mece_object_for_category(
        self,
        category: str,
        n_ideas: int,
        category_idx: int,
    ) -> MECEObjectDescription:
        """
        Build MECEObjectDescription for a single semantic category.

        Uses dimension_taxonomy axis_interpretation when available,
        falls back to generic descriptions otherwise.
        """
        inclusion = self._build_inclusion_definition(category)
        boundary = self._build_boundary_test(category)
        signals = self._build_diagnostic_signals(category)

        return MECEObjectDescription(
            topic_label=category,
            inclusion_definition=inclusion,
            boundary_test=boundary,
            diagnostic_signals=signals,
            source_cluster_ids=[category_idx],
            merge_rationale="semantic_category partition (no clustering)",
        )

    def _build_inclusion_definition(self, category: str) -> str:
        """Build inclusion_definition from dimension_taxonomy or fallback.

        NOTE: Grouping axis instructions are NOT included here — they are
        injected separately via get_grouping_instructions() into the
        <instruction> section of the MAP/REDUCE/MECE prompts, where
        the LLM pays more attention to them.
        """
        if self._dimension_taxonomy:
            axis_interp = self._dimension_taxonomy.get("axis_interpretation", {})
            interp = axis_interp.get(category, "")
            if interp:
                axis_label = self._taxonomy_axis or "the coding dimension"
                return (
                    f'Ideas about "{interp}" within the {axis_label} dimension. '
                    f'This category captures statements where the respondent '
                    f'describes something that can be characterized as: {interp}.'
                )

        # Fallback to generic descriptions
        generic = GENERIC_DESCRIPTIONS.get(category, {})
        return generic.get("inclusion", f"Ideas related to the semantic category '{category}'.")

    def _build_boundary_test(self, category: str) -> str:
        """Derive boundary_test from decision_reminder rules or fallback."""
        if self._dimension_taxonomy:
            reminders = self._dimension_taxonomy.get("decision_reminder", [])
            for rule in reminders:
                # Rules follow pattern "If <condition> → <category>"
                if f"→ {category}" in rule.lower() or f"-> {category}" in rule.lower():
                    condition = rule.split("→")[0].replace("If ", "").replace("if ", "").strip()
                    if not condition:
                        parts = rule.split("->")
                        if len(parts) > 1:
                            condition = parts[0].replace("If ", "").replace("if ", "").strip()
                    if condition:
                        return f"Does this statement describe {condition}?"

            # Try axis_interpretation as fallback
            axis_interp = self._dimension_taxonomy.get("axis_interpretation", {})
            interp = axis_interp.get(category, "")
            if interp:
                return f"Does this statement describe '{interp}'?"

        # Fallback to generic
        generic = GENERIC_DESCRIPTIONS.get(category, {})
        return generic.get("boundary", f"Does this statement relate to {category}?")

    def _build_diagnostic_signals(self, category: str) -> List[str]:
        """Build diagnostic signals from base signals enriched with axis context."""
        signals = list(BASE_DIAGNOSTIC_SIGNALS.get(category, [category]))

        # Enrich with axis_interpretation keywords if available
        if self._dimension_taxonomy:
            axis_interp = self._dimension_taxonomy.get("axis_interpretation", {})
            interp = axis_interp.get(category, "")
            if interp:
                # Extract meaningful words from the interpretation
                words = [
                    w.strip().lower()
                    for w in interp.replace("'", "").split()
                    if len(w.strip()) > 2 and w.strip().lower() not in {"the", "how", "what", "its"}
                ]
                for w in words:
                    if w not in signals:
                        signals.append(w)

        return signals[:5]

    # =========================================================================
    # OBJECT MAPPINGS
    # =========================================================================

    def _build_object_mappings(
        self,
        mece_objects: MECEObjectSet,
        category_ideas: Dict[str, List[str]],
    ) -> Dict[str, ObjectIdeaMapping]:
        """Build ObjectIdeaMapping for each MECE object."""
        mappings = {}
        for obj in mece_objects.topics:
            # topic_label is the category key (lowercase)
            cat_key = obj.topic_label.lower()
            ideas = category_ideas.get(cat_key, [])
            mappings[obj.topic_label] = ObjectIdeaMapping(
                object_label=obj.topic_label,
                mece_object=obj,
                idea_texts=ideas,
                category_names=[cat_key],
                idea_count=len(ideas),
            )
        return mappings
