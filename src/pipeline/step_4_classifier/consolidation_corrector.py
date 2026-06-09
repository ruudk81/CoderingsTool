"""
Post-hoc over-merge correction (the "P9 separability split").

P7/P8 are merge-biased and over-merge: distinct, substantial attributes collapse
into one catch-all bucket. This pass detects those buckets (threshold-free
within-bucket `own > sibling`, see consolidation_balance_core) and splits the
separable sources back apart ALONG PROVENANCE SEAMS — deterministically, reusing
each source's pre-P7 raw name/description. No LLM.

It writes only to `corrected_*` fields (the consolidated `attributes` /
`attribute_assignments` and the `raw_*` snapshots are left untouched), and returns
a corrected copy of the taxonomy + growing model. Wired into step 5 (default on),
which caches the result under separate `taxonomy_corrected` keys.

v1 scope: WITHIN-DOMAIN over-merges only (cross-domain moves by P8 are reported but
left merged). Restored attributes inherit the bucket's current facet (facets are
not changed).
"""

import copy
import time
from collections import defaultdict

from utils.llm import token_tracker
from pipeline.step_4_classifier.models_classifier import (
    TaxonomyResultsCache,
    TaxonomyClassifiedModel,
)
from pipeline.step_4_classifier import consolidation_balance_core as core

_RESTORE_SUFFIX = " (hersteld)"


class ConsolidationCorrector:
    """Splits over-merged catch-all buckets back apart along provenance seams."""

    def __init__(self, config, prompt_printer=None, dataset_key="", cost_tracker=None):
        self._code_source = config.correction_code_source
        self._embedding_model = config.correction_embedding_model
        self._k_min = config.correction_k_min
        self._k_band = config.correction_k_band
        self._min_split = config.correction_min_split_sources
        self._residual_dominance = config.correction_residual_dominance
        self._dataset_key = dataset_key
        self._cost_tracker = cost_tracker

    async def consolidate(self, taxonomy_cache, classified, extraction_meta, verbose=True):
        """Returns (corrected_taxonomy, corrected_classified, correction_map, stats, decision)."""
        t_start = time.time()
        snap_before = token_tracker.snapshot() if self._cost_tracker else None

        # --- measurement state (one embed pass) ---
        idea_index = core.index_growing_model(classified, self._code_source)
        raw_groups, raw_meta = core.collect_raw_groups(taxonomy_cache)
        final_groups = core.collect_final_groups(idea_index)
        final_sources, _raw_dominant_final = core.join_raw_to_final(raw_groups, idea_index)

        needed = {i for ids in raw_groups.values() for i in ids} | \
                 {i for ids in final_groups.values() for i in ids}
        _ids, matrix, gidx = await core.embed_ideas(idea_index, needed, self._embedding_model)
        n = len(_ids)
        raw_labels = core.label_list(raw_groups, gidx, n)
        fin_records = core.build_final_records(final_groups, gidx, final_sources, self._min_split)
        decision = core.over_merge_decision(
            fin_records, final_groups, gidx, matrix, raw_labels,
            k_min=self._k_min, k_band=self._k_band,
            min_split_sources=self._min_split, residual_dominance=self._residual_dominance,
        )

        # --- corrected copies, seeded to identity ---
        corrected_taxonomy = TaxonomyResultsCache.model_validate(
            copy.deepcopy(taxonomy_cache.model_dump()))
        corrected_classified = [
            TaxonomyClassifiedModel.model_validate(copy.deepcopy(m.model_dump()))
            for m in classified
        ]
        idea_by_id = {}
        for m in corrected_classified:
            for idea in (m.response_ideas or []):
                idea.corrected_attribute = idea.attribute
                idea.corrected_facet = idea.facet
                idea_by_id[idea.idea_id] = idea
        for res in corrected_taxonomy.partition_results.values():
            res.corrected_attributes = copy.deepcopy(res.attributes)
            res.corrected_attribute_assignments = dict(res.attribute_assignments)

        # --- apply splits ---
        correction_map = {}
        stats = {
            "buckets_examined": sum(1 for b in decision),
            "buckets_split": 0, "attrs_restored": 0, "ideas_moved": 0,
            "cross_domain_skipped": 0, "consistency_violations": 0,
        }
        for bucket in decision:
            if bucket.get("verdict") != "SPLIT":
                continue
            restored = self._apply_split(corrected_taxonomy, idea_by_id, bucket,
                                         raw_groups, raw_meta, taxonomy_cache, idea_index)
            if restored:
                stats["buckets_split"] += 1
                stats["attrs_restored"] += len(restored)
                stats["ideas_moved"] += sum(r[1] for r in restored)
                correction_map[(bucket["domain"], bucket["attribute"])] = [r[0] for r in restored]
        stats["cross_domain_skipped"] = sum(
            1 for b in decision if b.get("verdict") == "SPLIT"
            for s in b["sources"] if s["cross_domain"] and s["own_cluster"] and s["stable"] and not s["residual"]
        )

        violations = self._verify_consistency(taxonomy_cache, corrected_taxonomy)
        stats["consistency_violations"] = len(violations)
        stats["wall_time"] = time.time() - t_start

        if self._cost_tracker and snap_before is not None:
            self._cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p9_overmerge_correction",
                snap_before, token_tracker.snapshot(), self._embedding_model)

        if verbose:
            self._print_summary(decision, correction_map, stats, violations)

        return corrected_taxonomy, corrected_classified, correction_map, stats, decision

    # =========================================================================
    # SPLIT-BACK
    # =========================================================================

    def _apply_split(self, corrected_taxonomy, idea_by_id, bucket,
                     raw_groups, raw_meta, taxonomy_cache, idea_index):
        """Split a bucket's separable sources back into standalone corrected attributes.

        Returns list of (restored_name, n_ideas_moved). Within-domain only.
        """
        fk = (bucket["domain"], bucket["attribute"])
        d = bucket["domain"]
        res = corrected_taxonomy.partition_results.get(d)
        if res is None:
            return []
        b_facet = self._facet_of(res.attributes, fk[1])
        existing_names = {a.get("attribute_name")
                          for attrs in res.corrected_attributes.values() for a in attrs}

        restored = []
        for s in core.split_sources(bucket):
            S = (s["domain"], s["attribute"])
            ideas_to_move = [i for i in raw_groups.get(S, [])
                             if idea_index.get(i, {}).get("final") == fk]
            if not ideas_to_move:
                continue

            name = S[1]
            if name in existing_names:
                name = name + _RESTORE_SUFFIX
            existing_names.add(name)

            attr_dict = self._raw_attr_dict(taxonomy_cache, d, raw_meta.get(S, {}), S[1])
            attr_dict["attribute_name"] = name
            attr_dict["parent_facet"] = b_facet
            res.corrected_attributes.setdefault(b_facet, []).append(attr_dict)

            for i in ideas_to_move:
                res.corrected_attribute_assignments[i] = name
                idea = idea_by_id.get(i)
                if idea is not None:
                    idea.corrected_attribute = name
                    idea.corrected_facet = b_facet
            restored.append((name, len(ideas_to_move)))

        # Drop the bucket attribute if it has no ideas left after the splits.
        if not any(v == fk[1] for v in res.corrected_attribute_assignments.values()):
            for facet, attrs in list(res.corrected_attributes.items()):
                res.corrected_attributes[facet] = [a for a in attrs
                                                   if a.get("attribute_name") != fk[1]]
                if not res.corrected_attributes[facet]:
                    del res.corrected_attributes[facet]
        return restored

    @staticmethod
    def _facet_of(attributes, attr_name):
        """The facet under which attr_name lives in an attributes dict (fallback: first)."""
        for facet, attrs in attributes.items():
            if any(a.get("attribute_name") == attr_name for a in attrs):
                return facet
        return next(iter(attributes), "(unknown)")

    @staticmethod
    def _raw_attr_dict(taxonomy_cache, domain, raw_meta_entry, raw_name):
        """A copy of the pre-P7 attribute dict for raw_name (fallback to a minimal dict)."""
        res = taxonomy_cache.partition_results.get(domain)
        raw_facet = raw_meta_entry.get("facet")
        if res is not None and raw_facet in (res.raw_attributes or {}):
            for a in res.raw_attributes[raw_facet]:
                if a.get("attribute_name") == raw_name:
                    return copy.deepcopy(a)
        return {"attribute_name": raw_name,
                "attribute_description": raw_meta_entry.get("description", ""),
                "example_observations": []}

    # =========================================================================
    # CONSISTENCY
    # =========================================================================

    @staticmethod
    def _verify_consistency(taxonomy_cache, corrected_taxonomy):
        """Idea count preserved, no orphan corrected assignments."""
        violations = []
        base = sum(len(r.attribute_assignments) for r in taxonomy_cache.partition_results.values())
        corr = sum(len(r.corrected_attribute_assignments)
                   for r in corrected_taxonomy.partition_results.values())
        if base != corr:
            violations.append(f"idea count changed: {base} -> {corr}")
        for d, res in corrected_taxonomy.partition_results.items():
            names = {a.get("attribute_name")
                     for attrs in res.corrected_attributes.values() for a in attrs}
            orphans = {v for v in res.corrected_attribute_assignments.values()
                       if v not in names and v not in core.SENTINEL_ATTRIBUTES}
            if orphans:
                violations.append(f"[{d}] orphan corrected assignments: {sorted(orphans)[:5]}")
        return violations

    @staticmethod
    def _print_summary(decision, correction_map, stats, violations):
        print("\n  P9 over-merge correction:")
        for b in sorted(decision, key=lambda x: -x["count"]):
            if b.get("verdict") != "SPLIT":
                continue
            names = correction_map.get((b["domain"], b["attribute"]), [])
            print(f"    [{b['domain']}] {b['attribute']} (n={b['count']}) -> split into "
                  f"{len(names)}: {', '.join(names)}")
        print(f"    {stats['buckets_split']} bucket(s) split, {stats['attrs_restored']} attribute(s) "
              f"restored, {stats['ideas_moved']} idea(s) relabeled "
              f"({stats['cross_domain_skipped']} cross-domain source(s) left merged)")
        if violations:
            print(f"    correction CONSISTENCY: {len(violations)} issue(s):")
            for v in violations:
                print(f"      ⚠ {v}")
        else:
            print("    correction consistency: OK (idea count preserved, no orphans)")
