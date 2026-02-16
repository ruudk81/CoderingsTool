"""
Partition Discoverer for Category Discovery V5.

Partitions ideas by semantic_category (6 fixed groups), collects
unique category_labels per partition, and optionally pre-clusters
them via UMAP+HDBSCAN for Mode B.

Adapted from V4's category_discovery.py with:
- Cleaner terminology (partition instead of object)
- Configurable label_source and label_prefix
- Optional pre-clustering for Mode B
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from experiments import models_exp as models

from .config_categories_exp import CategoriesConfig
from .partition_labels import (
    collect_unique_labels, build_cluster_hints, PreclusterResult,
)
from .prompts_exp import PartitionDescription, PartitionSet

# Import template lookup for taxonomy-driven context
try:
    from experiments.step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
except ImportError:
    try:
        from step_3_ideaExtractor.template_lookup import TEMPLATE_LOOKUP
    except ImportError:
        TEMPLATE_LOOKUP = None


# =============================================================================
# CONSTANTS
# =============================================================================

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

# Grouping axis instructions per category — injected into <instruction> section
# of MAP/REDUCE/MECE prompts to ensure the LLM finds sub-types of the semantic
# category, not subject-matter topics.
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
        "inclusion": "Labels describing what something IS — its type, class, or essential nature.",
        "boundary": "Does this label classify or name what the entity fundamentally is?",
    },
    "attribute": {
        "inclusion": "Labels describing what something HAS — inherent properties, qualities, or characteristics.",
        "boundary": "Does this label describe an inherent property or quality of the entity?",
    },
    "function": {
        "inclusion": "Labels describing what something DOES — its purpose, actions, role, or capabilities.",
        "boundary": "Does this label describe an action, purpose, or function of the entity?",
    },
    "state": {
        "inclusion": "Labels describing the CONDITION something is in — time-bound or situational states.",
        "boundary": "Does this label describe a time-bound or situational condition?",
    },
    "evaluation": {
        "inclusion": "Labels expressing a JUDGMENT about something — subjective assessments or value judgments.",
        "boundary": "Does this label express a subjective judgment or evaluation?",
    },
    "relation": {
        "inclusion": "Labels describing how something CONNECTS to other things — dependencies, trade-offs, or associations.",
        "boundary": "Does this label describe a relationship, dependency, or connection between entities?",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PartitionLabelMapping:
    """Mapping of a partition to its unique labels."""
    partition_name: str
    partition: PartitionDescription
    labels: List[str]
    label_count: int


# =============================================================================
# MAIN CLASS
# =============================================================================

class PartitionDiscoverer:
    """
    Partitions ideas by semantic_category and collects unique labels
    per partition. Optionally pre-clusters labels for Mode B.
    """

    def __init__(
        self,
        config: CategoriesConfig,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
    ):
        self._config = config
        self._extraction_metadata = extraction_metadata
        self._taxonomy_axis = (
            getattr(extraction_metadata, 'taxonomy_axis', '')
            if extraction_metadata else ''
        )
        self._taxonomy_axis_info = self._load_axis_info()
        self._dimension_taxonomy = self._load_dimension_taxonomy()
        self._taxonomy_context_str = self._build_enriched_taxonomy_context()
        self._populated_categories: Dict[str, List] = {}

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
        """Return grouping instructions for each populated partition.

        Injected into the <instruction> section of MAP/REDUCE/MECE prompts.
        Must be called after discover().
        """
        return {
            cat: CATEGORY_GROUPING_INSTRUCTIONS[cat]
            for cat in self._populated_categories
            if cat in CATEGORY_GROUPING_INSTRUCTIONS
        }

    def _build_enriched_taxonomy_context(self) -> str:
        """Build enriched taxonomy_context block from extraction metadata."""
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
            lines.append("Categories MUST describe content within this dimension ONLY.")
            if excluded:
                lines.append(f"Do NOT create categories about: {', '.join(excluded)}")
            lines.append("</taxonomy_context>")
            return "\n".join(lines)

        return (
            f"<taxonomy_context>\n"
            f"Primary coding dimension: {taxonomy_axis}\n"
            f"Definition: {taxonomy_description or 'Not specified'}\n"
            f"Categories MUST describe content within this dimension ONLY.\n"
            f"</taxonomy_context>"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def discover(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Tuple[PartitionSet, Dict[str, PartitionLabelMapping], Optional[Dict[str, PreclusterResult]]]:
        """
        Partition ideas by semantic_category and collect labels.

        Returns:
            (partition_set, label_mappings, precluster_results)
            precluster_results is None for Mode A, populated for Mode B.
        """
        print(f"\n{'='*70}")
        print(f"PARTITION DISCOVERY (semantic_category → category_labels)")
        print(f"{'='*70}")
        print(f"  Processing mode: {self._config.processing_mode}")
        print(f"  Label source: {self._config.label_source}")
        if self._config.label_prefix:
            print(f"  Label prefix: \"{self._config.label_prefix}\"")
        if self._taxonomy_axis:
            print(f"  Taxonomy axis: {self._taxonomy_axis}")

        # Step 1: Partition ideas by semantic_category
        partitioned_ideas = self._partition_by_semantic_category(embeddings_models)

        # Filter empty partitions
        populated = {k: v for k, v in partitioned_ideas.items() if v}
        self._populated_categories = populated

        print(f"\n  Partitions found: {sorted(populated.keys())}")
        for cat in sorted(populated.keys()):
            print(f"    {cat}: {len(populated[cat])} ideas")

        # Check for unknown categories
        unknown = set(populated.keys()) - set(SEMANTIC_CATEGORIES)
        if unknown:
            print(f"  WARNING: Unknown semantic_category values: {unknown}")

        # Step 2: Collect unique labels per partition
        label_lists = {}
        empty_count = 0
        for cat, ideas in populated.items():
            labels = collect_unique_labels(
                ideas,
                label_source=self._config.label_source,
                label_prefix=self._config.label_prefix,
            )
            label_lists[cat] = labels
            n_empty = len(ideas) - sum(1 for idea in ideas if getattr(idea, self._config.label_source, ''))
            empty_count += n_empty

        if empty_count > 0:
            print(f"  WARNING: {empty_count} ideas had empty {self._config.label_source} field")

        print(f"\n  Unique labels per partition:")
        for cat in sorted(label_lists.keys()):
            print(f"    {cat}: {len(label_lists[cat])} unique labels")

        # Step 3: Optional pre-clustering (Mode B)
        precluster_results = None
        if self._config.processing_mode == "clustered":
            precluster_results = self._precluster_labels(label_lists)

        # Step 4: Build partition set and mappings
        partition_set = self._build_partition_set(label_lists)
        label_mappings = self._build_partition_mappings(partition_set, label_lists)

        print(f"\n  Partitions ({len(partition_set.partitions)}):")
        for p in partition_set.partitions:
            print(f"    - {p.partition_name}: {p.inclusion_definition[:80]}...")

        total_labels = sum(len(v) for v in label_lists.values())
        print(f"\n  Discovery Summary:")
        print(f"    {len(populated)} partitions, {total_labels} total unique labels")

        return partition_set, label_mappings, precluster_results

    # =========================================================================
    # PARTITIONING
    # =========================================================================

    def _partition_by_semantic_category(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Dict[str, List]:
        """Group idea objects by their semantic_category field.

        Returns:
            Dict mapping category_name → list of idea objects
        """
        partitions: Dict[str, List] = {}

        for resp in embeddings_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                cat = (getattr(idea, 'semantic_category', '') or '').strip().lower()
                if not cat:
                    continue
                if cat not in partitions:
                    partitions[cat] = []
                partitions[cat].append(idea)

        return partitions

    # =========================================================================
    # PRE-CLUSTERING (Mode B)
    # =========================================================================

    def _precluster_labels(
        self,
        label_lists: Dict[str, List[str]],
    ) -> Dict[str, PreclusterResult]:
        """Pre-cluster unique labels within each partition via UMAP+HDBSCAN.

        Partitions with fewer than precluster_min_labels are skipped.
        """
        import umap
        import hdbscan
        from utils.llm import create_client

        print(f"\n  Pre-clustering (Mode B):")
        results = {}

        for cat in sorted(label_lists.keys()):
            labels = label_lists[cat]
            if len(labels) < self._config.precluster_min_labels:
                print(f"    {cat}: {len(labels)} labels (below threshold {self._config.precluster_min_labels}, skipped)")
                continue

            # Embed labels on-the-fly
            print(f"    {cat}: embedding {len(labels)} labels...", end="")
            embeddings = self._embed_labels_sync(labels)
            print(f" done ({embeddings.shape})")

            # UMAP reduce
            n_neighbors = min(
                self._config.precluster_umap_n_neighbors,
                len(labels) - 1,
            )
            reducer = umap.UMAP(
                n_components=min(self._config.precluster_umap_n_components, len(labels) - 1),
                n_neighbors=n_neighbors,
                min_dist=self._config.precluster_umap_min_dist,
                metric=self._config.precluster_umap_metric,
                random_state=self._config.precluster_umap_random_state,
            )
            reduced = reducer.fit_transform(embeddings)

            # HDBSCAN cluster
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self._config.precluster_min_cluster_size,
                min_samples=self._config.precluster_min_samples,
                cluster_selection_method=self._config.precluster_cluster_selection_method,
            )
            cluster_labels_arr = clusterer.fit_predict(reduced)

            # Build result
            labels_by_cluster: Dict[int, List[str]] = {}
            noise_labels: List[str] = []
            for label, cluster_id in zip(labels, cluster_labels_arr):
                if cluster_id == -1:
                    noise_labels.append(label)
                else:
                    if cluster_id not in labels_by_cluster:
                        labels_by_cluster[cluster_id] = []
                    labels_by_cluster[cluster_id].append(label)

            n_clusters = len(labels_by_cluster)
            result = PreclusterResult(
                labels_by_cluster=labels_by_cluster,
                noise_labels=noise_labels,
                n_clusters=n_clusters,
                n_noise=len(noise_labels),
            )
            results[cat] = result
            print(f"    {cat}: {n_clusters} clusters, {len(noise_labels)} noise")

        return results

    def _embed_labels_sync(self, labels: List[str]) -> np.ndarray:
        """Embed labels synchronously via OpenAI embedding API."""
        from openai import OpenAI
        from config import OPENAI_API_KEY, API_PROVIDER

        if API_PROVIDER == "azure":
            from config import (
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VERSION,
                AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
            )
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
            )
            response = client.embeddings.create(
                input=labels,
                model=AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
            )
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.embeddings.create(
                input=labels,
                model="text-embedding-3-large",
            )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    # =========================================================================
    # PARTITION SET CONSTRUCTION
    # =========================================================================

    def _build_partition_set(
        self,
        label_lists: Dict[str, List[str]],
    ) -> PartitionSet:
        """Build PartitionSet from populated partitions."""
        partitions = []
        for category in sorted(label_lists.keys()):
            partition = self._build_partition_description(category)
            partitions.append(partition)
        return PartitionSet(partitions=partitions)

    def _build_partition_description(self, category: str) -> PartitionDescription:
        """Build PartitionDescription for a single semantic category."""
        return PartitionDescription(
            partition_name=category,
            inclusion_definition=self._build_inclusion_definition(category),
            boundary_test=self._build_boundary_test(category),
            diagnostic_signals=self._build_diagnostic_signals(category),
        )

    def _build_inclusion_definition(self, category: str) -> str:
        """Build inclusion_definition from dimension_taxonomy or fallback."""
        if self._dimension_taxonomy:
            axis_interp = self._dimension_taxonomy.get("axis_interpretation", {})
            interp = axis_interp.get(category, "")
            if interp:
                axis_label = self._taxonomy_axis or "the coding dimension"
                return (
                    f'Labels about "{interp}" within the {axis_label} dimension. '
                    f'This partition captures labels where the respondent '
                    f'describes something that can be characterized as: {interp}.'
                )

        generic = GENERIC_DESCRIPTIONS.get(category, {})
        return generic.get("inclusion", f"Labels related to the semantic category '{category}'.")

    def _build_boundary_test(self, category: str) -> str:
        """Derive boundary_test from decision_reminder rules or fallback."""
        if self._dimension_taxonomy:
            reminders = self._dimension_taxonomy.get("decision_reminder", [])
            for rule in reminders:
                if f"→ {category}" in rule.lower() or f"-> {category}" in rule.lower():
                    condition = rule.split("→")[0].replace("If ", "").replace("if ", "").strip()
                    if not condition:
                        parts = rule.split("->")
                        if len(parts) > 1:
                            condition = parts[0].replace("If ", "").replace("if ", "").strip()
                    if condition:
                        return f"Does this label describe {condition}?"

            axis_interp = self._dimension_taxonomy.get("axis_interpretation", {})
            interp = axis_interp.get(category, "")
            if interp:
                return f"Does this label describe '{interp}'?"

        generic = GENERIC_DESCRIPTIONS.get(category, {})
        return generic.get("boundary", f"Does this label relate to {category}?")

    def _build_diagnostic_signals(self, category: str) -> List[str]:
        """Build diagnostic signals from base signals enriched with axis context."""
        signals = list(BASE_DIAGNOSTIC_SIGNALS.get(category, [category]))

        if self._dimension_taxonomy:
            axis_interp = self._dimension_taxonomy.get("axis_interpretation", {})
            interp = axis_interp.get(category, "")
            if interp:
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
    # PARTITION MAPPINGS
    # =========================================================================

    def _build_partition_mappings(
        self,
        partition_set: PartitionSet,
        label_lists: Dict[str, List[str]],
    ) -> Dict[str, PartitionLabelMapping]:
        """Build PartitionLabelMapping for each partition."""
        mappings = {}
        for p in partition_set.partitions:
            labels = label_lists.get(p.partition_name, [])
            mappings[p.partition_name] = PartitionLabelMapping(
                partition_name=p.partition_name,
                partition=p,
                labels=labels,
                label_count=len(labels),
            )
        return mappings
