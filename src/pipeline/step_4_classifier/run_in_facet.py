#%%
"""Standalone P7 runner — test in-facet consolidation on cached data.

Feeds P7 the EXACT state P6 produced (`raw_attributes` + `raw_attribute_assignments`
from a cached taxonomy), which is also exactly what the old cross-facet consolidation
was given. So the two designs can be compared on identical input without rerunning P1-P6.

READ-ONLY: never writes to any cache. Results and metrics go to
exports/experiment_logs/.

Usage:
    cd src && python -m pipeline.step_4_classifier.run_in_facet
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
from utils.llm import create_client, llm_create_async, token_tracker
from config import get_reasoning_params

from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.prompts_classifier import (
    build_in_facet_consolidation_prompt, build_neighbour_block,
    InFacetConsolidatedResponse,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Which cached taxonomy to read the P6 state from. The production key is fine here:
# this module never writes, so reading it cannot damage a delivered taxonomy.
SOURCE_STEP = "taxonomy"
CLASSIFIED_STEP = "taxonomy_classified"

DOMAIN: Optional[str] = None   # None = every domain
FACET: Optional[str] = None    # None = every facet in scope
CONCURRENCY = 6
VERBOSE_PER_FACET = False      # True = print each facet's full before/after

CONFIG = CategoriesConfig()


# =============================================================================
# LOADING
# =============================================================================

def norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def load_state():
    vk = generate_enhanced_variable_key([TEST_DATA.var_name], False, TEST_DATA.sample_size)
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


def prompt_context_from(meta) -> Dict:
    dc = {}
    if meta:
        for f in ("sector", "entity", "topic", "perspective", "intent"):
            v = getattr(meta, f, None)
            if v:
                dc[f] = v
    parts = [f"{k.capitalize()}: {dc[k]}"
             for k in ("domain", "entity", "topic", "perspective", "intent") if dc.get(k)]
    return {
        "survey_question": (getattr(meta, "var_lab", "") or "") if meta else "",
        "language": (getattr(meta, "lang", "Dutch") or "Dutch") if meta else "Dutch",
        "dataset_context_section": (
            "<dataset_context>\n" + "\n".join(parts) + "\n</dataset_context>" if parts else ""),
        "dimension_name": (getattr(meta, "primary_dimension", "") or "") if meta else "",
        "dimension_description": (
            getattr(meta, "primary_dimension_description", "") or "") if meta else "",
    }


def contents_block(attrs, mine_by_attr, top_n) -> str:
    total = sum(len(v) for v in mine_by_attr.values())
    lines = []
    for a in attrs:
        name = a["attribute_name"]
        mine = mine_by_attr.get(name, [])
        pct = round(100 * len(mine) / total) if total else 0
        texts = Counter((i.instance or "").strip() for i in mine if (i.instance or "").strip())
        shown = " · ".join(f'"{t}" x{c}' for t, c in texts.most_common(top_n))
        more = (f" · ... {len(texts) - top_n} further distinct texts"
                if len(texts) > top_n else "")
        lines.append(f'- "{name}" — {len(mine)} ideas, {pct}% of this facet — '
                     f'{a["attribute_description"]}')
        lines.append(f'    actually contains: {shown}{more}' if shown
                     else '    actually contains: (no ideas assigned)')
    return "\n".join(lines)


# =============================================================================
# RUN ONE FACET
# =============================================================================

async def run_facet(client, sem, pc, tax, ideas_by_id, domain, facet) -> Optional[Dict]:
    res = tax.partition_results[domain]
    raw_attrs = (res.raw_attributes or {}).get(facet, [])
    if not raw_attrs:
        return None
    raw_assign = res.raw_attribute_assignments or {}
    names = {a["attribute_name"] for a in raw_attrs}

    mine_by_attr: Dict[str, List] = {a["attribute_name"]: [] for a in raw_attrs}
    for iid, aname in raw_assign.items():
        if aname in names and iid in ideas_by_id:
            mine_by_attr[aname].append(ideas_by_id[iid])
    n_ideas = sum(len(v) for v in mine_by_attr.values())
    if not n_ideas:
        return None

    facet_desc = next((f.get("facet_description", "") for f in res.facets
                       if f.get("facet_name") == facet), "")
    domain_def = next(p.inclusion_definition for p in tax.partition_set.partitions
                      if p.partition_name == domain)

    neighbours = []
    for other, oattrs in (res.raw_attributes or {}).items():
        if other == facet or not oattrs:
            continue
        counts = Counter(raw_assign.values())
        neighbours.append((other, [(a["attribute_name"], counts.get(a["attribute_name"], 0))
                                   for a in oattrs]))

    block = contents_block(raw_attrs, mine_by_attr, CONFIG.p7_contents_top_n)
    prompt = build_in_facet_consolidation_prompt(
        **pc, dimension_def=None,
        domain_name=domain, domain_definition=domain_def,
        facet_name=facet, facet_description=facet_desc,
        attributes_block=block, neighbour_block=build_neighbour_block(neighbours),
    )

    async with sem:
        try:
            out = await llm_create_async(
                client=client, model=CONFIG.qr_model_p7, prompt=prompt,
                response_model=InFacetConsolidatedResponse,
                temperature=0.0, max_tokens=CONFIG.qr_max_tokens_consolidation,
                **get_reasoning_params(CONFIG.qr_model_p7, phase="classifier_p7"),
            )
        except Exception as e:
            print(f"  FAILED {domain} > {facet}: {str(e)[:90]}")
            return {"domain": domain, "facet": facet, "failed": True,
                    "attrs_before": len(raw_attrs), "n_ideas": n_ideas}

    # ---- route ideas deterministically, exactly as _apply_p7_results would ----
    claims: Dict[str, int] = {}
    for it in out.attributes:
        for s in (it.source_attributes or []):
            if s in names:
                claims[s] = claims.get(s, 0) + 1
    contested = {s for s, n in claims.items() if n > 1
                 and not any(s in (it.source_attributes or []) and it.instance_texts
                             for it in out.attributes)}

    remap: Dict[str, str] = {}
    splits: Dict[Tuple[str, str], str] = {}
    for it in out.attributes:
        srcs = [s for s in (it.source_attributes or []) if s in names]
        if it.action == "split" and it.instance_texts:
            for s in (srcs or list(names)):
                for t in it.instance_texts:
                    splits[(s, norm(t))] = it.attribute_name
        else:
            for s in srcs:
                if s != it.attribute_name and s not in contested:
                    remap[s] = it.attribute_name

    moves: Dict[Tuple[str, str], Optional[str]] = {}
    for m in (out.misfits or []):
        for t in (m.instance_texts or []):
            moves[(m.from_attribute, norm(t))] = (
                m.target_attribute if m.verdict == "move" else None)

    n_moved = n_out = n_split = n_remapped = n_stuck = 0
    for aname, ideas in mine_by_attr.items():
        for i in ideas:
            t = norm(getattr(i, "instance", ""))
            if (aname, t) in moves:
                if moves[(aname, t)] is None:
                    n_out += 1
                else:
                    n_moved += 1
            elif (aname, t) in splits:
                n_split += 1
            elif aname in remap:
                n_remapped += 1
            elif aname in contested:
                n_stuck += 1

    after_names = [a.attribute_name for a in out.attributes]
    rec = {
        "domain": domain, "facet": facet, "n_ideas": n_ideas,
        "attrs_before": len(raw_attrs), "attrs_after": len(after_names),
        "names_before": sorted(names), "names_after": after_names,
        "returned": [{"name": a.attribute_name, "action": a.action,
                      "sources": list(a.source_attributes or []),
                      "instance_texts": list(a.instance_texts or [])}
                     for a in out.attributes],
        "actions": Counter(a.action for a in out.attributes),
        "ideas_moved": n_moved, "ideas_flagged_out": n_out,
        "ideas_split": n_split, "ideas_remapped": n_remapped,
        "ideas_stuck_unroutable": n_stuck,
        "unroutable_claims": sorted(contested),
        "move_targets": [m.target_attribute for m in (out.misfits or [])
                         if m.verdict == "move"],
        "misfits": [{"from": m.from_attribute, "verdict": m.verdict,
                     "target": m.target_attribute, "texts": m.instance_texts,
                     "reason": m.reason} for m in (out.misfits or [])],
    }

    if VERBOSE_PER_FACET:
        print("\n" + "=" * 78)
        print(f"{domain} > {facet}   ({n_ideas} ideas)")
        print(block)
        for a in out.attributes:
            print(f'  [{a.action:6s}] "{a.attribute_name}"  from {a.source_attributes}')
    print(f"  {domain[:28]:30s} {facet[:34]:36s} "
          f"{len(raw_attrs):2d}->{len(after_names):2d} attrs, "
          f"{n_moved:3d} moved, {n_out:3d} out")
    return rec


# =============================================================================
# METRICS
# =============================================================================

def report(tax, records, ideas_by_id) -> Dict:
    ok = [r for r in records if r and not r.get("failed")]
    failed = [r for r in records if r and r.get("failed")]

    before_total = sum(r["attrs_before"] for r in ok)
    after_total = sum(r["attrs_after"] for r in ok)
    solo_before = sum(1 for r in ok if r["attrs_before"] == 1)
    solo_after = sum(1 for r in ok if r["attrs_after"] == 1)
    eq_before = sum(1 for r in ok if r["attrs_before"] == 1
                    and norm(r["names_before"][0]) == norm(r["facet"]))
    eq_after = sum(1 for r in ok if r["attrs_after"] == 1
                   and norm(r["names_after"][0]) == norm(r["facet"]))

    # duplicate attribute names across facets (deterministic under-merge signal)
    name_homes = defaultdict(set)
    for r in ok:
        for n in r["names_after"]:
            name_homes[norm(n)].add((r["domain"], r["facet"]))
    dup_after = {n: sorted(v) for n, v in name_homes.items() if len(v) > 1}

    moved = sum(r["ideas_moved"] for r in ok)
    flagged_out = sum(r["ideas_flagged_out"] for r in ok)
    split = sum(r["ideas_split"] for r in ok)
    remapped = sum(r["ideas_remapped"] for r in ok)
    stuck = sum(r["ideas_stuck_unroutable"] for r in ok)
    n_ideas = sum(r["n_ideas"] for r in ok)

    # did any move target a facet other than its own? (allowed; structure never follows)
    home_of_attr = {}
    for r in ok:
        for n in r["names_after"]:
            home_of_attr[norm(n)] = (r["domain"], r["facet"])
    # global old-name -> new-name trail: facets consolidate concurrently, so a move
    # may name a target by the name the neighbour block showed while its own facet
    # was renaming it.
    renamed_to = {}
    for r in ok:
        for it in r["returned"]:
            if it["action"] == "split" and it["instance_texts"]:
                continue
            for src in it["sources"]:
                if norm(src) != norm(it["name"]):
                    renamed_to[norm(src)] = it["name"]
    cross_facet_moves = unresolved_targets = recovered_by_rename = 0
    for r in ok:
        for m in r["misfits"]:
            if m["verdict"] != "move" or not m["target"]:
                continue
            t = norm(m["target"])
            if t not in home_of_attr and t in renamed_to:
                t = norm(renamed_to[t])
                recovered_by_rename += 1
            h = home_of_attr.get(t)
            if h is None:
                unresolved_targets += 1
            elif h != (r["domain"], r["facet"]):
                cross_facet_moves += 1

    m = {
        "facets_processed": len(ok), "facets_failed": len(failed),
        "ideas_in_scope": n_ideas,
        "attributes_before": before_total, "attributes_after": after_total,
        "solo_facets_before": solo_before, "solo_facets_after": solo_after,
        "solo_share_before_pct": round(100 * solo_before / len(ok)) if ok else 0,
        "solo_share_after_pct": round(100 * solo_after / len(ok)) if ok else 0,
        "facet_eq_attribute_before": eq_before, "facet_eq_attribute_after": eq_after,
        "duplicate_attribute_names_after": len(dup_after),
        "duplicate_examples": dict(list(dup_after.items())[:8]),
        "ideas_remapped": remapped, "ideas_split": split,
        "ideas_moved": moved, "ideas_flagged_out_left_in_place": flagged_out,
        "ideas_stuck_unroutable": stuck,
        "move_groups_targeting_another_facet": cross_facet_moves,
        "move_groups_recovered_via_rename_trail": recovered_by_rename,
        "move_groups_with_unresolvable_target": unresolved_targets,
        "ideas_relocated_by_a_MERGE": 0,
    }
    return m


async def main():
    tax, ideas, meta, vk = load_state()
    ideas_by_id = {i.idea_id: i for i in ideas}
    pc = prompt_context_from(meta)

    pairs = []
    for dom, res in tax.partition_results.items():
        if DOMAIN and dom != DOMAIN:
            continue
        for fac in (res.raw_attributes or {}):
            if FACET and fac != FACET:
                continue
            pairs.append((dom, fac))

    print(f"source={SOURCE_STEP} | {len(ideas)} ideas | model {CONFIG.qr_model_p7} "
          f"| {len(pairs)} facets\n")

    client = create_client(CONFIG.qr_model_p7, async_mode=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    records = await asyncio.gather(*[
        run_facet(client, sem, pc, tax, ideas_by_id, d, f) for d, f in pairs])

    metrics = report(tax, records, ideas_by_id)
    print("\n" + "=" * 78)
    print("COUNTER-METRICS")
    print("=" * 78)
    for k, v in metrics.items():
        if k in ("duplicate_examples",):
            continue
        print(f"  {k:42s} {v}")
    if metrics["duplicate_examples"]:
        print("\n  duplicate attribute names across facets (under-merge signal):")
        for n, homes in metrics["duplicate_examples"].items():
            print(f"    {n!r}: {homes}")

    out_dir = Path(__file__).resolve().parents[3] / "exports" / "experiment_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(TEST_DATA.filename).stem}_{vk}_p7_sweep.json"
    path.write_text(json.dumps(
        {"metrics": metrics,
         "facets": [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items()}
                    for r in records if r]},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {path}")
    print(f"total cost: ${token_tracker.total_cost_usd:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

# %%
