#%%
"""Standalone replay of the attribute refinement phase, on cached data.

Feeds the phase the EXACT state discovery and consolidation left behind
(`raw_attributes` + `raw_attribute_assignments` from a cached taxonomy), so one
phase can be repeated without paying for the whole chain. A stop point is not a
resumption point, so `stop_after_phase` does not replace this: it stops the run
before the phase, this one starts it.

It calls the production `_run_attribute_refinement` and `_apply_attribute_refinement`
rather than a copy of them. A replay that reimplements the routing measures the
copy, not the phase.

READ-ONLY: never writes to any cache. Metrics go to exports/experiment_logs/.

Usage:
    cd src && python -m pipeline.step_4_classifier.replay_attribute_refinement
"""
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import models
from test_data import TEST_DATA
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.llm import token_tracker

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from pipeline.step_4_classifier.classifier import PromptContext, TaxonomyClassifier
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.prompts_attribute import ConsolidatedAttribute
from pipeline.step_4_classifier.prompts_facet import ConsolidatedFacet
from pipeline.step_4_classifier.taxonomy_health import drain_domains

# =============================================================================
# CONFIGURATION
# =============================================================================

# Which cached taxonomy to read the pre-refinement state from. The production
# key is fine: this module never writes, so reading it cannot damage a
# delivered taxonomy.
SOURCE_STEP = "taxonomy"
CLASSIFIED_STEP = "taxonomy_classified"

DOMAIN: Optional[str] = None   # None = every domain
FACET: Optional[str] = None    # None = every facet in scope

CONFIG = CategoriesConfig()


# =============================================================================
# LOADING
# =============================================================================

def load_state():
    vk = generate_enhanced_variable_key(
        [TEST_DATA.var_name], False, TEST_DATA.sample_size)
    cm = CacheManager()
    tax = cm.load_metadata_from_cache(
        filename=TEST_DATA.filename, step=SOURCE_STEP,
        variable_key=vk, model_cls=models.TaxonomyResultsCache)
    if tax is None:
        raise FileNotFoundError(f"No cached taxonomy at step '{SOURCE_STEP}'.")
    classified = cm.load_from_cache(
        TEST_DATA.filename, CLASSIFIED_STEP, vk, models.TaxonomyClassifiedModel)
    ideas = [i for r in (classified or []) if r.response_ideas for i in r.response_ideas]
    meta = cm.load_metadata_from_cache(
        filename=TEST_DATA.filename, step="extracted_ideas",
        variable_key=vk, model_cls=models.ExtractionMetadata)
    return tax, ideas, meta, vk


def _require_current_shape(card: Dict, kind: str) -> None:
    """Fail with a readable message on a cache written before the rewrite."""
    if "attribute_definition" in card or "facet_definition" in card:
        return
    raise ValueError(
        f"This cached taxonomy carries the pre-rewrite {kind} shape "
        f"({sorted(card)[:4]}...). The replay runs the current phase, which "
        f"reads facet_definition / attribute_definition. Rerun step 4 first."
    )


def build_inputs(tax, ideas, meta):
    """Rebuild exactly what `_run_attribute_refinement` takes, from the cache."""
    ideas_by_id = {i.idea_id: i for i in ideas}

    facets: Dict[str, List[ConsolidatedFacet]] = {}
    attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]] = {}
    facet_assignments: Dict[str, Dict[str, str]] = {}
    assignments: Dict[str, str] = {}
    labels: Dict[Tuple[str, str], Dict[str, str]] = {}

    for domain, res in tax.partition_results.items():
        if DOMAIN and domain != DOMAIN:
            continue
        for card in (res.facets or []):
            _require_current_shape(card, "facet")
            facets.setdefault(domain, []).append(ConsolidatedFacet(**card))

        raw_assign = res.raw_attribute_assignments or {}
        for facet_name, cards in (res.raw_attributes or {}).items():
            if (FACET and facet_name != FACET) or not cards:
                continue
            for card in cards:
                _require_current_shape(card, "attribute")
            attributes.setdefault(domain, {})[facet_name] = [
                ConsolidatedAttribute(**card) for card in cards]

        for idea_id, facet_name in (res.facet_assignments or {}).items():
            if FACET and facet_name != FACET:
                continue
            attribute_name = raw_assign.get(idea_id)
            if not attribute_name or idea_id not in ideas_by_id:
                continue
            facet_assignments.setdefault(domain, {})[idea_id] = facet_name
            assignments[idea_id] = attribute_name
            idea = ideas_by_id[idea_id]
            labels.setdefault((domain, facet_name), {})[idea_id] = (
                getattr(idea, "instance", "") or "")

    context = {f: (getattr(meta, f, None) or "") if meta else ""
               for f in ("sector", "entity", "topic", "perspective", "intent")}
    dimension_name = (getattr(meta, "primary_dimension", "") or "") if meta else ""
    ctx = PromptContext(
        language=(getattr(meta, "lang", "Dutch") or "Dutch") if meta else "Dutch",
        survey_question=(getattr(meta, "var_lab", "") or "") if meta else "",
        **context,
        dimension=get_dimension(dimension_name),
        dimension_name=dimension_name,
        dimension_description=(
            (getattr(meta, "primary_dimension_description", "") or "") if meta else ""),
        domains={
            part.partition_name: {
                "label": part.partition_name,
                "definition": part.inclusion_definition,
                "boundary_test": part.boundary_test,
                "exclusions": part.exclusions,
                "observations": [],
            }
            for part in tax.partition_set.partitions
        },
        drain_labels=drain_domains(meta),
    )
    return ctx, facets, attributes, assignments, facet_assignments, labels


# =============================================================================
# METRICS
# =============================================================================

def report(before, after, action_log) -> Dict:
    """Counter-metrics, read the whole table before calling a run a success.

    Under-merging is as real a failure as over-merging, so both directions are
    counted: solo facets and duplicate names on one side, moved ideas and
    unroutable claims on the other.
    """
    def flat(structure):
        return [(d, f, [a.attribute_name for a in attrs])
                for d, items in structure.items() for f, attrs in items.items()]

    rows_before, rows_after = flat(before), flat(after)
    homes = defaultdict(set)
    for domain, facet, names in rows_after:
        for name in names:
            homes[name.strip().lower()].add((domain, facet))
    duplicates = {n: sorted(v) for n, v in homes.items() if len(v) > 1}

    totals = next((e for e in action_log if e.get("action") == "_totals"), {})
    actions = Counter(e["action"] for e in action_log if not e["action"].startswith("_"))

    solo_before = sum(1 for _, _, names in rows_before if len(names) == 1)
    solo_after = sum(1 for _, _, names in rows_after if len(names) == 1)
    facet_eq_attr = sum(1 for _, facet, names in rows_after
                        if len(names) == 1
                        and names[0].strip().lower() == facet.strip().lower())

    return {
        "facets_in_scope": len(rows_after),
        "attributes_before": sum(len(n) for _, _, n in rows_before),
        "attributes_after": sum(len(n) for _, _, n in rows_after),
        "solo_facets_before": solo_before,
        "solo_facets_after": solo_after,
        "solo_share_after_pct": (round(100 * solo_after / len(rows_after))
                                 if rows_after else 0),
        "facet_eq_attribute_after": facet_eq_attr,
        "duplicate_attribute_names_after": len(duplicates),
        "duplicate_examples": dict(list(duplicates.items())[:8]),
        "ideas_remapped": totals.get("ideas_remapped", 0),
        "ideas_split": totals.get("ideas_split", 0),
        "ideas_moved": totals.get("ideas_moved", 0),
        "ideas_flagged_out_left_in_place": totals.get(
            "flagged_contentless_left_in_place", 0),
        "moves_with_unresolvable_target": totals.get(
            "moves_with_unresolvable_target", 0),
        "moves_whose_target_was_itself_split": totals.get(
            "moves_whose_target_was_itself_split", 0),
        "actions": dict(sorted(actions.items())),
    }


async def main():
    tax, ideas, meta, vk = load_state()
    ctx, facets, attributes, assignments, facet_assignments, labels = build_inputs(
        tax, ideas, meta)

    before = {d: {f: list(a) for f, a in items.items()}
              for d, items in attributes.items()}
    n_facets = sum(len(items) for items in attributes.values())
    print(f"source={SOURCE_STEP} | {len(assignments)} ideas | "
          f"model {CONFIG.model_attribute_refinement} | {n_facets} facets\n")

    classifier = TaxonomyClassifier(CONFIG)
    await classifier._initialize_async_resources(verbose=True)
    attributes, assignments, facet_assignments = (
        await classifier._run_attribute_refinement(
            ctx, attributes, assignments, facet_assignments, facets, labels,
            verbose=True))

    metrics = report(before, attributes, classifier._action_log)
    print("\n" + "=" * 78)
    print("COUNTER-METRICS")
    print("=" * 78)
    for key, value in metrics.items():
        if key != "duplicate_examples":
            print(f"  {key:42s} {value}")
    if metrics["duplicate_examples"]:
        print("\n  duplicate attribute names across facets (under-merge signal):")
        for name, homes in metrics["duplicate_examples"].items():
            print(f"    {name!r}: {homes}")

    out_dir = Path(__file__).resolve().parents[3] / "exports" / "experiment_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"{Path(TEST_DATA.filename).stem}_{vk}"
                      f"_attribute_refinement_replay.json")
    path.write_text(json.dumps(
        {"metrics": metrics, "actions": classifier._action_log},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {path}")
    print(f"total cost: ${token_tracker.total_cost_usd:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

# %%
