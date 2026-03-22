"""
Partition Discoverer for Category Discovery.

Partitions ideas by domain (data-driven groups from step 3),
collects unique concepts per partition, and optionally pre-clusters
them via UMAP+HDBSCAN for Mode B.

- Dynamic partitions from domain
- Partition descriptions from domains metadata in ExtractionMetadata
- Configurable label_source and label_prefix
- Optional pre-clustering for Mode B
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

import models

from config_steps.config_categories import CategoriesConfig
from .partition_labels import (
    collect_unique_labels, build_cluster_hints, PreclusterResult,
    format_label,
)
from prompts import DomainDescription, DomainSet


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PartitionLabelMapping:
    """Mapping of a partition to its unique labels."""
    partition_name: str
    partition: DomainDescription
    labels: List[str]
    label_count: int

# =============================================================================
# MAIN CLASS
# =============================================================================

class DomainDiscoverer:
    """
    Partitions ideas by domain and collects unique labels
    per partition. Optionally pre-clusters labels for Mode B.
    """

    def __init__(
        self,
        config: CategoriesConfig,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
    ):
        self._config = config
        self._extraction_metadata = extraction_metadata
        self._primary_dimension = (
            getattr(extraction_metadata, 'primary_dimension', '')
            if extraction_metadata else ''
        )
        # Build domains lookup: {key: {label, definition}} from metadata
        self._domains_lookup: Dict[str, Dict[str, str]] = {}
        if extraction_metadata and hasattr(extraction_metadata, 'domains'):
            for d in extraction_metadata.domains:
                key = d.get('key', '')
                if key:
                    self._domains_lookup[key] = d

        self._populated_partitions: Dict[str, List] = {}

    # =========================================================================
    # PARTITION CONTEXT
    # =========================================================================

    def get_grouping_instructions(self) -> Dict[str, str]:
        """Return grouping instructions for each populated partition.

        Injected into the <instruction> section of MAP/REDUCE/MECE prompts.
        Must be called after discover().
        """
        instructions = {}
        for ct_key in self._populated_partitions:
            ct_info = self._domains_lookup.get(ct_key, {})
            definition = ct_info.get('definition', '')
            label = ct_info.get('label', ct_key)
            if definition:
                instructions[ct_key] = (
                    f"Coding categories must represent DIFFERENT SUB-TYPES of "
                    f"\"{label}\" ({definition}) — "
                    f"not different subjects or topics."
                )
        return instructions

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def discover(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Tuple[DomainSet, Dict[str, PartitionLabelMapping], Optional[Dict[str, PreclusterResult]]]:
        """
        Partition ideas by domain and collect labels.

        Returns:
            (partition_set, label_mappings, precluster_results)
            precluster_results is None for Mode A, populated for Mode B.
        """
        print(f"\n{'='*70}")
        print(f"PARTITION DISCOVERY (concept_type \u2192 concepts)")
        print(f"{'='*70}")
        print(f"  Processing mode: {self._config.processing_mode}")
        print(f"  Label source: {self._config.label_source}")
        if self._config.label_prefix:
            print(f"  Label prefix: \"{self._config.label_prefix}\"")
        if self._primary_dimension:
            print(f"  Primary dimension: {self._primary_dimension}")

        # Step 1: Partition ideas by domain
        partitioned_ideas = self._partition_by_domain(embeddings_models)

        # Filter empty partitions
        populated = {k: v for k, v in partitioned_ideas.items() if v}
        self._populated_partitions = populated

        print(f"\n  Partitions found: {sorted(populated.keys())}")
        for ct in sorted(populated.keys()):
            print(f"    {ct}: {len(populated[ct])} ideas")

        # Step 2: Collect unique labels per partition
        label_lists = {}
        empty_count = 0
        for ct, ideas in populated.items():
            labels = collect_unique_labels(
                ideas,
                label_source=self._config.label_source,
                label_prefix=self._config.label_prefix,
            )
            label_lists[ct] = labels
            n_empty = sum(
                1 for idea in ideas
                if not format_label(idea, self._config.label_source)
            )
            empty_count += n_empty

        if empty_count > 0:
            print(f"  WARNING: {empty_count} ideas had empty {self._config.label_source} field")

        print(f"\n  Unique labels per partition:")
        for ct in sorted(label_lists.keys()):
            print(f"    {ct}: {len(label_lists[ct])} unique labels")

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

    def _partition_by_domain(
        self,
        embeddings_models: List[models.EmbeddingsModel],
    ) -> Dict[str, List]:
        """Group idea objects by their domain field.

        Returns:
            Dict mapping concept_type \u2192 list of idea objects
        """
        partitions: Dict[str, List] = {}

        for resp in embeddings_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                ct = (getattr(idea, self._config.PARTITION_SOURCE, '') or '').strip().lower()
                if not ct:
                    continue
                if ct not in partitions:
                    partitions[ct] = []
                partitions[ct].append(idea)

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

        for ct in sorted(label_lists.keys()):
            labels = label_lists[ct]
            if len(labels) < self._config.precluster_min_labels:
                print(f"    {ct}: {len(labels)} labels (below threshold {self._config.precluster_min_labels}, skipped)")
                continue

            # Embed labels on-the-fly
            print(f"    {ct}: embedding {len(labels)} labels...", end="")
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
            results[ct] = result
            print(f"    {ct}: {n_clusters} clusters, {len(noise_labels)} noise")

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
    ) -> DomainSet:
        """Build DomainSet from populated partitions."""
        partitions = []
        for ct_key in sorted(label_lists.keys()):
            partition = self._build_partition_description(ct_key)
            partitions.append(partition)
        return DomainSet(partitions=partitions)

    def _build_partition_description(self, ct_key: str) -> DomainDescription:
        """Build DomainDescription for a domain partition.

        Uses domains metadata from ExtractionMetadata when available,
        falls back to a generic description based on the partition name.
        """
        ct_info = self._domains_lookup.get(ct_key, {})
        label = ct_info.get('label', ct_key)
        definition = ct_info.get('definition', '')

        return DomainDescription(
            partition_name=ct_key,
            inclusion_definition=self._build_inclusion_definition(label, definition),
            boundary_test=self._build_boundary_test(label, definition),
            diagnostic_signals=self._build_diagnostic_signals(label, definition),
        )

    def _build_inclusion_definition(self, label: str, definition: str) -> str:
        """Build inclusion_definition from domain metadata."""
        if definition:
            return (
                f'Labels of type "{label}": {definition}. '
                f'This partition captures concepts that fall under this concept type.'
            )
        return f"Labels related to the concept type '{label}'."

    def _build_boundary_test(self, label: str, definition: str) -> str:
        """Derive boundary_test from domain metadata."""
        if definition:
            return f"Does this label describe a '{label}' concept ({definition})?"
        return f"Does this label relate to '{label}'?"

    def _build_diagnostic_signals(self, label: str, definition: str) -> List[str]:
        """Build diagnostic signals from domain label and definition."""
        signals = [label.lower()]
        if definition:
            words = [
                w.strip().lower()
                for w in definition.replace("'", "").replace(",", " ").split()
                if len(w.strip()) > 3 and w.strip().lower() not in {
                    "this", "that", "with", "from", "about", "which", "their", "they", "have",
                }
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
        partition_set: DomainSet,
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
