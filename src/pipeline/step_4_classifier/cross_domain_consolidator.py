"""
Cross-Domain Attribute Consolidator for Taxonomy Classifier.

P8: After P7 consolidates attributes across facets within each domain,
this module consolidates overlapping attributes across domain boundaries.

Algorithm:
  1. Embed all ideas, compute centroid per attribute
  2. Seriate attributes into a 1D similarity ordering (agglomerative clustering)
  3. Sliding window → overlapping groups of ~10 attributes
  4. SmoothRequester dispatch for LLM consolidation per group
  5. Remap assignments in TaxonomyResultsCache + growing model

See dev/CONSOLIDATION_LOGIC.md for full algorithm documentation.
"""

import copy
import time
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Set

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from utils.embedder import SharedEmbedder, format_idea_text, compute_medoid
from utils.llm import token_tracker
from utils.smoothRequester import SmoothRequester
from config import get_reasoning_params

from pipeline.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.models_classifier import (
    TaxonomyResultsCache,
    TaxonomyClassifiedModel,
    TaxonomyClassifiedSubmodel,
)
from pipeline.step_4_classifier.prompts_classifier import (
    build_cross_domain_consolidation_prompt,
    CrossDomainConsolidatedResponse,
    CrossDomainConsolidatedAttribute,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AttributeEntry(NamedTuple):
    """One attribute in the global inventory."""
    domain_name: str
    facet_name: str
    attribute_name: str
    attribute_description: str
    idea_count: int


class AttributeEmbedding(NamedTuple):
    """Centroid embedding result for one attribute."""
    domain_name: str
    attribute_name: str
    centroid: np.ndarray
    medoid_text: str
    idea_count: int


class MergeTarget(NamedTuple):
    """Where a merged/renamed attribute ends up."""
    new_attribute_name: str
    new_domain: str
    new_facet: str
    new_description: str


# =============================================================================
# CROSS-DOMAIN CONSOLIDATOR
# =============================================================================

class CrossDomainConsolidator:
    """Cross-domain attribute consolidation (P8).

    Finds overlapping attributes across domains using embedding centroids,
    clusters them into LLM-digestible groups via seriation + sliding window,
    consolidates via LLM, and remaps assignments.
    """

    def __init__(
        self,
        config: CategoriesConfig,
        prompt_printer=None,
        dataset_key: str = "",
        cost_tracker=None,
        fetched_limits=None,
    ):
        self._model = config.qr_model_p8
        self._temperature = config.qr_temperature
        self._max_tokens = config.qr_max_tokens_cross_domain

        # P8-specific config
        self._code_source = config.p8_code_source
        self._embedding_model = config.p8_embedding_model
        self._window_size = config.p8_window_size
        self._window_overlap = config.p8_window_overlap
        self._similarity_threshold = config.p8_similarity_threshold

        # Shared resources
        self._dataset_key = dataset_key
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()
        self._cost_tracker = cost_tracker
        self._fetched_limits = fetched_limits

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def consolidate(
        self,
        taxonomy_cache: TaxonomyResultsCache,
        classified: List[TaxonomyClassifiedModel],
        extraction_meta,
        verbose: bool = True,
    ) -> tuple:
        """Run cross-domain consolidation (P8).

        Returns:
            (new_taxonomy_cache, new_classified, merge_map, stats)
        """
        t_start = time.time()

        # Cost tracking snapshot
        _snap_p8 = token_tracker.snapshot() if self._cost_tracker else None

        # Step 1: Build attribute inventory
        inventory = self._build_inventory(taxonomy_cache)
        ideas_per_attr = self._collect_ideas_per_attribute(classified)

        if verbose:
            n_attrs = len(inventory)
            n_ideas = sum(e.idea_count for e in inventory)
            print(f"\n  Phase 6: Cross-domain Attribute Consolidation")
            print(f"    Input: {n_attrs} attributes, {n_ideas} ideas")

        # Step 2: Embed and compute centroids
        t_embed = time.time()
        attr_embeddings = await self._embed_and_compute_centroids(ideas_per_attr, verbose)
        t_embed = time.time() - t_embed

        if verbose:
            print(f"    P8 embedding: {sum(a.idea_count for a in attr_embeddings)} ideas, "
                  f"{len(attr_embeddings)} centroids ({t_embed:.1f}s)")

        # Step 3: Seriate and build sliding windows
        ordered = self._compute_attribute_order(attr_embeddings)
        windows = self._build_sliding_windows(ordered)

        if verbose:
            print(f"    P8 grouping: {len(windows)} groups "
                  f"(window={self._window_size}, overlap={self._window_overlap})")

        # Step 4: LLM consolidation via SmoothRequester
        results = await self._run_consolidation(
            windows, attr_embeddings, inventory,
            taxonomy_cache, extraction_meta, verbose,
        )

        # Step 5: Build merge map and remap
        merge_map = self._build_merge_map(results, windows, attr_embeddings)
        new_taxonomy = self._apply_remapping_to_cache(taxonomy_cache, merge_map, inventory)
        new_classified = self._apply_remapping_to_growing_model(classified, merge_map)

        t_total = time.time() - t_start

        # Count results
        attrs_before = len(inventory)
        attrs_after = sum(
            len(a) for r in new_taxonomy.partition_results.values()
            for a in r.attributes.values()
        )
        ideas_before = sum(
            len(r.attribute_assignments)
            for r in taxonomy_cache.partition_results.values()
        )
        ideas_after = sum(
            len(r.attribute_assignments)
            for r in new_taxonomy.partition_results.values()
        )

        if verbose:
            print(f"    Results ({t_total:.1f}s → {attrs_after} consolidated attributes):")
            # Show per-domain changes
            for domain_name in sorted(taxonomy_cache.partition_results.keys()):
                old_result = taxonomy_cache.partition_results[domain_name]
                new_result = new_taxonomy.partition_results[domain_name]
                old_n = sum(len(a) for a in old_result.attributes.values())
                new_n = sum(len(a) for a in new_result.attributes.values())
                old_ideas = len(old_result.attribute_assignments)
                new_ideas = len(new_result.attribute_assignments)
                if old_n != new_n or old_ideas != new_ideas:
                    remap_count = abs(new_ideas - old_ideas)
                    remap_msg = f", {remap_count} remapped" if remap_count else ""
                    print(f"      {domain_name}: {old_n} → {new_n} attributes{remap_msg}")

            if ideas_before != ideas_after:
                print(f"    WARNING: idea count changed {ideas_before} → {ideas_after}")

        # Cost tracking
        if self._cost_tracker and _snap_p8 is not None:
            self._cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p8_cross_domain_consolidation",
                _snap_p8, token_tracker.snapshot(), self._model)

        stats = {
            "attrs_before": attrs_before,
            "attrs_after": attrs_after,
            "ideas_before": ideas_before,
            "ideas_after": ideas_after,
            "merges": len(merge_map),
            "groups": len(windows),
            "wall_time": t_total,
        }

        return new_taxonomy, new_classified, merge_map, stats

    # =========================================================================
    # STEP 1: ATTRIBUTE INVENTORY
    # =========================================================================

    def _build_inventory(
        self,
        taxonomy_cache: TaxonomyResultsCache,
    ) -> List[AttributeEntry]:
        """Build a flat list of all attributes across all domains."""
        inventory = []
        for domain_name, result in taxonomy_cache.partition_results.items():
            attr_counts: Dict[str, int] = defaultdict(int)
            for attr_name in result.attribute_assignments.values():
                attr_counts[attr_name] += 1

            for facet_name, attrs in result.attributes.items():
                for attr_dict in attrs:
                    name = attr_dict.get("attribute_name", "?")
                    desc = attr_dict.get("attribute_description", "")
                    count = attr_counts.get(name, 0)
                    inventory.append(AttributeEntry(
                        domain_name=domain_name,
                        facet_name=facet_name,
                        attribute_name=name,
                        attribute_description=desc,
                        idea_count=count,
                    ))
        return inventory

    def _collect_ideas_per_attribute(
        self,
        classified: List[TaxonomyClassifiedModel],
    ) -> Dict[tuple, List[TaxonomyClassifiedSubmodel]]:
        """Group ideas by (domain, attribute) from the growing model."""
        groups: Dict[tuple, List[TaxonomyClassifiedSubmodel]] = defaultdict(list)
        for resp in classified:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                domain = idea.partition_name or "(unknown)"
                attr = idea.attribute or "(no attribute)"
                groups[(domain, attr)].append(idea)
        return groups

    # =========================================================================
    # STEP 2: EMBEDDING
    # =========================================================================

    async def _embed_and_compute_centroids(
        self,
        ideas_per_attr: Dict[tuple, List[TaxonomyClassifiedSubmodel]],
        verbose: bool = True,
    ) -> List[AttributeEmbedding]:
        """Embed all ideas, compute centroid per attribute."""
        attr_keys = []
        attr_texts = []
        attr_counts = []

        for (domain, attr_name), ideas in sorted(ideas_per_attr.items()):
            if attr_name == "(no attribute)":
                continue
            texts = [format_idea_text(idea, self._code_source) for idea in ideas]
            attr_keys.append((domain, attr_name))
            attr_texts.append(texts)
            attr_counts.append(len(texts))

        all_texts = []
        for texts in attr_texts:
            all_texts.extend(texts)

        embedder = SharedEmbedder(model=self._embedding_model)
        all_embeddings = await embedder.embed_texts(all_texts)

        results = []
        offset = 0
        for i, (domain, attr_name) in enumerate(attr_keys):
            count = attr_counts[i]
            attr_embeddings = all_embeddings[offset:offset + count]
            centroid = attr_embeddings.mean(axis=0)
            medoid_idx = compute_medoid(attr_embeddings)
            medoid_text = attr_texts[i][medoid_idx]
            results.append(AttributeEmbedding(
                domain_name=domain,
                attribute_name=attr_name,
                centroid=centroid,
                medoid_text=medoid_text,
                idea_count=count,
            ))
            offset += count

        return results

    # =========================================================================
    # STEP 3: SERIATION + SLIDING WINDOW
    # =========================================================================

    def _compute_attribute_order(
        self,
        attr_embeddings: List[AttributeEmbedding],
    ) -> List[int]:
        """Order attributes so that similar ones are adjacent."""
        centroids = np.array([a.centroid for a in attr_embeddings])
        sim_matrix = cosine_similarity(centroids)
        dist_matrix = 1 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="average")
        return leaves_list(Z).tolist()

    def _build_sliding_windows(
        self,
        ordered_indices: List[int],
    ) -> List[List[int]]:
        """Slide a window across ordered indices to produce overlapping groups."""
        step = self._window_size - self._window_overlap
        n = len(ordered_indices)
        windows = []

        for start in range(0, n, step):
            window = ordered_indices[start:start + self._window_size]
            windows.append(window)
            if start + self._window_size >= n:
                break

        # Merge last window if it adds too few new attributes
        if len(windows) >= 2:
            prev = set(windows[-2])
            new_in_last = [i for i in windows[-1] if i not in prev]
            if len(new_in_last) <= self._window_overlap:
                windows[-2] = windows[-2] + new_in_last
                windows.pop()

        return windows

    # =========================================================================
    # STEP 4: LLM CONSOLIDATION (SmoothRequester)
    # =========================================================================

    async def _run_consolidation(
        self,
        windows: List[List[int]],
        attr_embeddings: List[AttributeEmbedding],
        inventory: List[AttributeEntry],
        taxonomy_cache: TaxonomyResultsCache,
        extraction_meta,
        verbose: bool = True,
    ) -> List[Optional[CrossDomainConsolidatedResponse]]:
        """Run LLM consolidation for all groups via SmoothRequester."""
        # Resolve dimension
        dimension_def = None
        if extraction_meta and extraction_meta.primary_dimension:
            dimension_def = get_dimension(extraction_meta.primary_dimension)

        # Build dataset context section
        dataset_context_section = ""
        if extraction_meta:
            parts = []
            for key in ["sector", "entity", "topic", "perspective", "intent"]:
                value = getattr(extraction_meta, key, "")
                if value:
                    parts.append(f"{key.capitalize()}: {value}")
            if parts:
                dataset_context_section = (
                    "<dataset_context>\n" + "\n".join(parts) + "\n</dataset_context>"
                )

        # Build task list — one per window/group
        tasks = []
        for g, window in enumerate(windows, 1):
            tasks.append({
                "group_num": g,
                "window": window,
                "attr_embeddings": attr_embeddings,
                "inventory": inventory,
                "taxonomy_cache": taxonomy_cache,
                "extraction_meta": extraction_meta,
                "dimension_def": dimension_def,
                "dataset_context_section": dataset_context_section,
            })

        # SmoothRequester dispatch
        requester = SmoothRequester(
            model=self._model,
            dataset_key=self._dataset_key,
            phase_key="step4_p8_cross_domain_consolidation",
            num_tasks=len(tasks),
            verbose=verbose,
            known_limits=self._fetched_limits,
            show_setup=False,
            quiet=True,
            default_timeout=60.0,
        )

        results = await requester.process_all(
            tasks,
            self._p8_prepare_fn(),
            self._p8_parse_fn(),
            self._p8_fallback_fn(),
        )

        if verbose:
            s = requester.stats
            t_sr = s.get('wall_time', 0)
            print(f"    P8 consolidation: {len(tasks)} tasks, {t_sr:.1f}s "
                  f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        return results

    def _format_domain_attributes_block(
        self,
        window: List[int],
        attr_embeddings: List[AttributeEmbedding],
        inventory: List[AttributeEntry],
        taxonomy_cache: TaxonomyResultsCache,
    ) -> str:
        """Format attributes in a window as domain → facet → attribute block."""
        entry_lookup = {}
        for entry in inventory:
            entry_lookup[(entry.domain_name, entry.attribute_name)] = entry

        domain_defs = {}
        for part in taxonomy_cache.partition_set.partitions:
            domain_defs[part.partition_name] = part.inclusion_definition

        facet_descs: Dict[str, Dict[str, str]] = {}
        for domain_name, result in taxonomy_cache.partition_results.items():
            facet_descs[domain_name] = {}
            for facet_dict in result.facets:
                facet_descs[domain_name][facet_dict["facet_name"]] = facet_dict.get(
                    "facet_description", ""
                )

        by_domain_facet: Dict[str, Dict[str, List[AttributeEmbedding]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx in window:
            a = attr_embeddings[idx]
            entry = entry_lookup.get((a.domain_name, a.attribute_name))
            facet = entry.facet_name if entry else "(unknown)"
            by_domain_facet[a.domain_name][facet].append(a)

        lines = []
        for domain_name in sorted(by_domain_facet.keys()):
            domain_def = domain_defs.get(domain_name, "")
            lines.append(f'Domain: "{domain_name}" — {domain_def}')
            for facet_name in sorted(by_domain_facet[domain_name].keys()):
                facet_desc = facet_descs.get(domain_name, {}).get(facet_name, "")
                lines.append(f'  Facet: "{facet_name}" — {facet_desc}')
                for a in by_domain_facet[domain_name][facet_name]:
                    entry = entry_lookup.get((a.domain_name, a.attribute_name))
                    desc = entry.attribute_description if entry else ""
                    lines.append(
                        f'    - "{a.attribute_name}" ({a.idea_count} ideas) — {desc}'
                    )
            lines.append("")

        return "\n".join(lines)

    # -- SmoothRequester callbacks --

    def _p8_prepare_fn(self):
        """Return prepare_fn closure for P8 cross-domain consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            domain_attributes_block = self._format_domain_attributes_block(
                task["window"],
                task["attr_embeddings"],
                task["inventory"],
                task["taxonomy_cache"],
            )

            meta = task["extraction_meta"]
            prompt = build_cross_domain_consolidation_prompt(
                survey_question=getattr(meta, "var_lab", "") or "",
                language=getattr(meta, "lang", "Dutch") or "Dutch",
                dataset_context_section=task["dataset_context_section"],
                dimension_def=task["dimension_def"],
                dimension_name=getattr(meta, "primary_dimension", "") or "",
                dimension_description=getattr(meta, "primary_dimension_description", "") or "",
                domain_attributes_block=domain_attributes_block,
            )

            # Prompt capture
            gate_key = f"qr_cross_domain_consolidation_{task['group_num']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="cross_domain_consolidation",
                    metadata={
                        "model": self._model,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                        "group_num": task["group_num"],
                        "n_attributes": len(task["window"]),
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': CrossDomainConsolidatedResponse,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model, phase="classifier_p8"),
            }
        return prepare_fn

    def _p8_parse_fn(self):
        """Return parse_fn closure for P8 cross-domain consolidation."""
        def parse_fn(task: Dict, response) -> Optional[CrossDomainConsolidatedResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p8_fallback_fn():
        """Return fallback_fn closure for P8 cross-domain consolidation."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    # =========================================================================
    # STEP 5: MERGE MAP + REMAPPING
    # =========================================================================

    def _build_merge_map(
        self,
        results: List[Optional[CrossDomainConsolidatedResponse]],
        windows: List[List[int]],
        attr_embeddings: List[AttributeEmbedding],
    ) -> Dict[str, MergeTarget]:
        """Build merge map from all group results.

        "Merge wins, first group takes precedence."
        """
        merge_map: Dict[str, MergeTarget] = {}
        already_processed: set = set()

        for g, (window, response) in enumerate(zip(windows, results), 1):
            if not response:
                continue
            for consolidated in response.attributes:
                target = MergeTarget(
                    new_attribute_name=consolidated.attribute_name,
                    new_domain=consolidated.parent_domain,
                    new_facet=consolidated.parent_facet,
                    new_description=consolidated.attribute_description,
                )
                for source_name in consolidated.source_attributes:
                    if source_name in already_processed:
                        continue
                    already_processed.add(source_name)
                    if source_name != consolidated.attribute_name:
                        merge_map[source_name] = target

        return merge_map

    def _apply_remapping_to_cache(
        self,
        taxonomy_cache: TaxonomyResultsCache,
        merge_map: Dict[str, MergeTarget],
        inventory: List[AttributeEntry],
    ) -> TaxonomyResultsCache:
        """Apply merge map to taxonomy cache, returning a new copy."""
        new_cache = TaxonomyResultsCache.model_validate(
            copy.deepcopy(taxonomy_cache.model_dump())
        )

        attr_source_domain: Dict[str, str] = {}
        for entry in inventory:
            attr_source_domain[entry.attribute_name] = entry.domain_name

        for old_name, target in merge_map.items():
            source_domain = attr_source_domain.get(old_name)
            if not source_domain:
                continue

            source_result = new_cache.partition_results.get(source_domain)
            target_result = new_cache.partition_results.get(target.new_domain)
            if not source_result or not target_result:
                continue

            idea_ids_to_move = [
                iid for iid, aname in source_result.attribute_assignments.items()
                if aname == old_name
            ]
            if not idea_ids_to_move:
                continue

            # Remove from source
            for iid in idea_ids_to_move:
                del source_result.attribute_assignments[iid]
                source_result.attribute_valence.pop(iid, None)
                source_result.attribute_confidence.pop(iid, None)

            # Cross-domain: also move facet assignments
            if source_domain != target.new_domain:
                for iid in idea_ids_to_move:
                    old_facet_valence = source_result.facet_valence.pop(iid, None)
                    old_facet_conf = source_result.facet_confidence.pop(iid, None)
                    source_result.facet_assignments.pop(iid, None)

                    target_result.facet_assignments[iid] = target.new_facet
                    if old_facet_valence:
                        target_result.facet_valence[iid] = old_facet_valence
                    if old_facet_conf:
                        target_result.facet_confidence[iid] = old_facet_conf

            # Add to target
            for iid in idea_ids_to_move:
                target_result.attribute_assignments[iid] = target.new_attribute_name

            # Remove old attribute from source attributes dict
            for facet_name, attrs_list in list(source_result.attributes.items()):
                source_result.attributes[facet_name] = [
                    a for a in attrs_list if a.get("attribute_name") != old_name
                ]
                if not source_result.attributes[facet_name]:
                    del source_result.attributes[facet_name]

            # Ensure target attribute exists
            if target.new_facet not in target_result.attributes:
                target_result.attributes[target.new_facet] = []
            existing = [
                a for a in target_result.attributes[target.new_facet]
                if a.get("attribute_name") == target.new_attribute_name
            ]
            if not existing:
                target_result.attributes[target.new_facet].append({
                    "attribute_name": target.new_attribute_name,
                    "attribute_description": target.new_description,
                })

        return new_cache

    def _apply_remapping_to_growing_model(
        self,
        classified: List[TaxonomyClassifiedModel],
        merge_map: Dict[str, MergeTarget],
    ) -> List[TaxonomyClassifiedModel]:
        """Apply merge map to growing model, returning a new copy."""
        new_classified = []
        for resp in classified:
            resp_dict = copy.deepcopy(resp.model_dump())
            new_resp = TaxonomyClassifiedModel.model_validate(resp_dict)
            if new_resp.response_ideas:
                for idea in new_resp.response_ideas:
                    if idea.attribute and idea.attribute in merge_map:
                        target = merge_map[idea.attribute]
                        idea.attribute = target.new_attribute_name
                        idea.facet = target.new_facet
                        idea.partition_name = target.new_domain
            new_classified.append(new_resp)
        return new_classified
