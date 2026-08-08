"""
Partition Discoverer for Taxonomy Classifier.

Partitions ideas by domain (data-driven groups from step 3),
collects unique labels per partition.

- Dynamic partitions from domain field
- Partition descriptions from domains metadata in ExtractionMetadata
- Configurable label_source and label_prefix
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import models

from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from .partition_labels import collect_unique_labels_with_domains, format_label
from models import DomainDescription, DomainSet


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PartitionLabelMapping:
    """Mapping of a partition to its unique labels and idea objects."""
    partition_name: str
    partition: DomainDescription
    labels: List[str]
    label_count: int
    label_domains: List[Optional[str]] = field(default_factory=list)
    ideas: List = field(default_factory=list)  # IdeasExtractedSubmodel objects

# =============================================================================
# MAIN CLASS
# =============================================================================

class DomainDiscoverer:
    """
    Partitions ideas by domain and collects unique labels
    per partition.
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
        # Index by both key ("eten_en_drinken") and lowercased label ("eten en drinken")
        # because ideas store the label, not the key, in their domain field.
        self._domains_lookup: Dict[str, Dict[str, str]] = {}
        if extraction_metadata and hasattr(extraction_metadata, 'domains'):
            for d in extraction_metadata.domains:
                key = d.get('key', '')
                if key:
                    self._domains_lookup[key] = d
                label = d.get('label', '')
                if label:
                    self._domains_lookup[label.lower()] = d

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
        ideas_models: List[models.IdeasExtractedModel],
    ) -> Tuple[DomainSet, Dict[str, PartitionLabelMapping]]:
        """
        Partition ideas by domain and collect labels.

        Returns:
            (partition_set, label_mappings)
        """
        print(f"\n{'='*70}")
        print(f"PARTITION DISCOVERY (domain → concepts)")
        print(f"{'='*70}")
        print(f"  Label source: {self._config.label_source}")
        if self._config.label_prefix:
            print(f"  Label prefix: \"{self._config.label_prefix}\"")
        if self._primary_dimension:
            print(f"  Primary dimension: {self._primary_dimension}")

        # Step 1: Partition ideas by domain
        partitioned_ideas = self._partition_by_domain(ideas_models)

        # Filter empty partitions
        populated = {k: v for k, v in partitioned_ideas.items() if v}
        self._populated_partitions = populated

        print(f"\n  Partitions found: {sorted(populated.keys())}")
        for ct in sorted(populated.keys()):
            print(f"    {ct}: {len(populated[ct])} ideas")

        # Step 2: Collect unique labels per partition
        label_lists = {}
        domain_lists = {}
        empty_count = 0
        for ct, ideas in populated.items():
            labels, domains = collect_unique_labels_with_domains(
                ideas,
                label_source=self._config.label_source,
                label_prefix=self._config.label_prefix,
            )
            label_lists[ct] = labels
            domain_lists[ct] = domains
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

        # Step 3: Build partition set and mappings
        partition_set = self._build_partition_set(label_lists)
        label_mappings = self._build_partition_mappings(partition_set, label_lists, domain_lists)

        print(f"\n  Partitions ({len(partition_set.partitions)}):")
        for p in partition_set.partitions:
            print(f"    - {p.partition_name}: {p.inclusion_definition[:80]}...")

        total_labels = sum(len(v) for v in label_lists.values())
        print(f"\n  Discovery Summary:")
        print(f"    {len(populated)} partitions, {total_labels} total unique labels")

        return partition_set, label_mappings

    # =========================================================================
    # PARTITIONING
    # =========================================================================

    def _partition_by_domain(
        self,
        ideas_models: List[models.IdeasExtractedModel],
    ) -> Dict[str, List]:
        """Group idea objects by their domain field.

        Returns:
            Dict mapping domain → list of idea objects
        """
        partitions: Dict[str, List] = {}

        for resp in ideas_models:
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
            # Prefer the boundary_test/exclusions persisted by step 3; fall back to
            # the locally-derived boundary_test for old caches that lack them.
            boundary_test=ct_info.get('boundary_test') or self._build_boundary_test(label, definition),
            diagnostic_signals=self._build_diagnostic_signals(label, definition),
            exclusions=ct_info.get('exclusions', []),
        )

    def _build_inclusion_definition(self, label: str, definition: str) -> str:
        """Build inclusion_definition from domain metadata."""
        if definition:
            return definition
        return f"Labels related to the domain '{label}'."

    def _build_boundary_test(self, label: str, definition: str) -> str:
        """Derive boundary_test from domain metadata."""
        if definition:
            return f"Does this label belong to the '{label}' domain ({definition})?"
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
        domain_lists: Optional[Dict[str, List[Optional[str]]]] = None,
    ) -> Dict[str, PartitionLabelMapping]:
        """Build PartitionLabelMapping for each partition."""
        mappings = {}
        for p in partition_set.partitions:
            labels = label_lists.get(p.partition_name, [])
            domains = (domain_lists or {}).get(p.partition_name, [])
            ideas = self._populated_partitions.get(p.partition_name, [])
            mappings[p.partition_name] = PartitionLabelMapping(
                partition_name=p.partition_name,
                partition=p,
                labels=labels,
                label_count=len(labels),
                label_domains=domains,
                ideas=ideas,
            )
        return mappings
