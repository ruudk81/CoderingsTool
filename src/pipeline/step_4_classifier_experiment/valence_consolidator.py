"""
Valence-neutral attribute consolidation (post-P7 cleanup).

Detects attribute PAIRS within a facet that differ only in evaluative direction
(a valence artifact baked into the taxonomy, e.g. "Algemene positieve waardering"
vs "Algemene niet-positieve waardering") and merges the safe ones into a single
descriptive attribute. Valence is carried by the per-idea `valence` field, not by
two attributes.

Split into a deterministic part and a small LLM part:
  - detection (which pairs)            -> deterministic (shared with view_valence_split.py)
  - the merged name/description        -> LLM (the genuinely semantic bit), with a
                                          deterministic single-token fallback
  - idea reassignment + cache update   -> deterministic; valence/confidence preserved

See dev/DESIGN_VALENCE_NEUTRALITY.md.
"""

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from config import get_reasoning_params
from utils.llm import create_client, llm_create_async, token_tracker

from pipeline.step_4_classifier_experiment.config_classifier import CategoriesConfig
from models import (
    TaxonomyResultsCache,
    TaxonomyClassifiedModel,
)
from pipeline.step_4_classifier_experiment.prompts_classifier import (
    build_valence_neutral_rename_prompt,
    ValenceNeutralRenameResponse,
)


# =============================================================================
# DETECTION (deterministic, shared with view_valence_split.py)
# =============================================================================

@dataclass
class ValenceSplitPair:
    domain: str
    facet: str
    name_a: str
    name_b: str
    sim: float
    val_a: Counter
    val_b: Counter
    total_a: int
    total_b: int
    samples_a: List[str] = field(default_factory=list)
    samples_b: List[str] = field(default_factory=list)
    auto_safe: bool = False
    fallback_name: Optional[str] = None  # deterministic merged name if single-token diff, else None


def _tokens(s: str) -> set:
    return {t for t in re.split(r"[^\w]+", (s or "").lower()) if t}


def label_similarity(a: str, b: str) -> float:
    """Max of token-set Jaccard (word-level) and char ratio (catches prefixes
    like Betrouwbaar/Onbetrouwbaar that share no tokens)."""
    ta, tb = _tokens(a), _tokens(b)
    jacc = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    ratio = SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
    return max(jacc, ratio)


def single_token_merge_name(a: str, b: str) -> Optional[str]:
    """If a and b differ in exactly one whitespace-token position, return the
    deterministic merged name (the differing token dropped). Else None."""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return None
    diff = [i for i in range(len(ta)) if ta[i].lower() != tb[i].lower()]
    if len(diff) != 1:
        return None
    merged = ta[: diff[0]] + ta[diff[0] + 1:]
    return " ".join(merged).strip() or None


def detect_valence_splits(
    classified: List[TaxonomyClassifiedModel],
    label_sim_threshold: float = 0.6,
    min_skew: float = 0.7,
    min_count: int = 5,
    auto_safe_sim: float = 0.8,
) -> List[ValenceSplitPair]:
    """Find attribute pairs (within a facet) that differ only in evaluative
    direction: near-identical labels AND opposite valence skew."""
    # (domain, facet, attribute) -> {valence Counter, samples}
    stats: Dict[tuple, dict] = defaultdict(lambda: {"val": Counter(), "samples": []})
    for resp in classified:
        for idea in (resp.response_ideas or []):
            if not idea.attribute:
                continue
            key = (idea.partition_name or "(unknown)", idea.facet or "(unknown)", idea.attribute)
            s = stats[key]
            s["val"][idea.valence or "?"] += 1
            if len(s["samples"]) < 3 and idea.instance:
                s["samples"].append(idea.instance)

    by_facet: Dict[tuple, list] = defaultdict(list)
    for (dom, fac, attr), s in stats.items():
        total = sum(s["val"].values())
        if total >= min_count:
            by_facet[(dom, fac)].append((attr, s, total))

    pairs: List[ValenceSplitPair] = []
    for (dom, fac), attrs in by_facet.items():
        if len(attrs) < 2:
            continue
        for (a1, s1, t1), (a2, s2, t2) in combinations(attrs, 2):
            sim = label_similarity(a1, a2)
            if sim < label_sim_threshold:
                continue
            p1 = s1["val"].get("+", 0) / t1
            p2 = s2["val"].get("+", 0) / t2
            complementary = (
                (p1 >= min_skew and p2 <= 1 - min_skew)
                or (p2 >= min_skew and p1 <= 1 - min_skew)
            )
            if not complementary:
                continue
            pairs.append(ValenceSplitPair(
                domain=dom, facet=fac, name_a=a1, name_b=a2, sim=sim,
                val_a=s1["val"], val_b=s2["val"], total_a=t1, total_b=t2,
                samples_a=list(s1["samples"]), samples_b=list(s2["samples"]),
                auto_safe=sim >= auto_safe_sim,
                fallback_name=single_token_merge_name(a1, a2),
            ))
    return pairs


# =============================================================================
# CONSOLIDATOR (deterministic merge + LLM rename of the merged attribute)
# =============================================================================

class ValenceConsolidator:
    """Collapses safe valence-split attribute pairs into one descriptive attribute."""

    def __init__(self, config: CategoriesConfig, cost_tracker=None):
        self._model = config.qr_model_p7_5
        self._temperature = config.qr_temperature
        self._cost_tracker = cost_tracker
        self._label_sim_threshold = 0.6
        self._min_skew = 0.7
        self._min_count = 5
        self._auto_safe_sim = 0.8

    async def consolidate(
        self,
        taxonomy_cache: TaxonomyResultsCache,
        classified: List[TaxonomyClassifiedModel],
        extraction_meta,
        verbose: bool = True,
    ) -> tuple:
        """Returns (new_taxonomy, new_classified, report, stats)."""
        pairs = detect_valence_splits(
            classified, self._label_sim_threshold, self._min_skew,
            self._min_count, self._auto_safe_sim,
        )
        # Safe scope: auto-merge-safe AND a clean single-token diff (so the
        # deterministic fallback name is unambiguous).
        merge_pairs = [p for p in pairs if p.auto_safe and p.fallback_name]

        stats = {"candidates": len(pairs), "merges": 0}
        if not merge_pairs:
            if verbose:
                print(f"\n  P7.5 valence merge: 0 pairs (of {len(pairs)} candidate(s))")
            return taxonomy_cache, classified, [], stats

        # Descriptions live in the taxonomy cache (not on the growing model)
        desc_lookup: Dict[tuple, str] = {}
        for dom, res in taxonomy_cache.partition_results.items():
            for attrs in res.attributes.values():
                for a in attrs:
                    desc_lookup[(dom, a.get("attribute_name"))] = a.get("attribute_description", "")

        language = getattr(extraction_meta, "lang", "Dutch") or "Dutch"
        _snap = token_tracker.snapshot() if self._cost_tracker else None
        names = await self._rename(merge_pairs, desc_lookup, language)
        if self._cost_tracker and _snap is not None:
            self._cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p7_5_valence_merge",
                _snap, token_tracker.snapshot(), self._model,
            )

        # Build merge map: (domain, old_name) -> (new_name, facet, new_desc)
        merge_map: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
        report = []
        for i, p in enumerate(merge_pairs):
            if i in names:
                new_name, new_desc = names[i]
                source = "llm"
            else:
                new_name = p.fallback_name
                new_desc = desc_lookup.get((p.domain, p.name_a), "")
                source = "fallback"
            merge_map[(p.domain, p.name_a)] = (new_name, p.facet, new_desc)
            merge_map[(p.domain, p.name_b)] = (new_name, p.facet, new_desc)
            report.append({
                "domain": p.domain, "facet": p.facet,
                "name_a": p.name_a, "name_b": p.name_b,
                "new_name": new_name, "source": source,
            })

        new_taxonomy = self._apply_to_cache(taxonomy_cache, merge_map)
        new_classified = self._apply_to_growing_model(classified, merge_map)
        stats["merges"] = len(merge_pairs)

        if verbose:
            print(f"\n  P7.5 valence merge: {len(merge_pairs)} pair(s) collapsed")
            for r in report:
                print(f"    \"{r['name_a']}\" + \"{r['name_b']}\" -> "
                      f"\"{r['new_name']}\"  [{r['source']}]  ({r['domain']})")

        return new_taxonomy, new_classified, report, stats

    async def _rename(self, merge_pairs, desc_lookup, language) -> Dict[int, Tuple[str, str]]:
        """LLM call producing one neutral name+description per pair. Empty dict on failure."""
        payload = [{
            "pair_id": i,
            "name_a": p.name_a, "desc_a": desc_lookup.get((p.domain, p.name_a), ""),
            "name_b": p.name_b, "desc_b": desc_lookup.get((p.domain, p.name_b), ""),
            "samples": (p.samples_a + p.samples_b)[:6],
        } for i, p in enumerate(merge_pairs)]
        try:
            client = create_client(self._model)
            response = await llm_create_async(
                client, self._model,
                build_valence_neutral_rename_prompt(payload, language),
                response_model=ValenceNeutralRenameResponse,
                temperature=self._temperature, max_tokens=2000,
                **get_reasoning_params(self._model, phase="classifier_p7"),
            )
            return {a.pair_id: (a.attribute_name, a.attribute_description) for a in response.attributes}
        except Exception as e:
            print(f"    P7.5 rename LLM call failed ({e}); using deterministic fallback names")
            return {}

    def _apply_to_cache(self, taxonomy_cache, merge_map):
        new_cache = TaxonomyResultsCache.model_validate(
            copy.deepcopy(taxonomy_cache.model_dump())
        )
        for (domain, old_name), (new_name, facet, new_desc) in merge_map.items():
            res = new_cache.partition_results.get(domain)
            if not res:
                continue
            for iid, val in list(res.attribute_assignments.items()):
                if val == old_name:
                    res.attribute_assignments[iid] = new_name
            if facet in res.attributes:
                res.attributes[facet] = [
                    a for a in res.attributes[facet] if a.get("attribute_name") != old_name
                ]
            fac_list = res.attributes.setdefault(facet, [])
            if not any(a.get("attribute_name") == new_name for a in fac_list):
                fac_list.append({"attribute_name": new_name, "attribute_description": new_desc})
        return new_cache

    def _apply_to_growing_model(self, classified, merge_map):
        new_classified = []
        for resp in classified:
            new_resp = TaxonomyClassifiedModel.model_validate(copy.deepcopy(resp.model_dump()))
            if new_resp.response_ideas:
                for idea in new_resp.response_ideas:
                    target = merge_map.get((idea.partition_name, idea.attribute))
                    if target:
                        new_name, facet, _ = target
                        idea.attribute = new_name
                        idea.facet = facet
            new_classified.append(new_resp)
        return new_classified
