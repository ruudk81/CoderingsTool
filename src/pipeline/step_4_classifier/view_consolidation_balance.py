#%%

"""
View consolidation balance: read-only diagnostic for over-merge (catch-all) in the
step-4 taxonomy. Thin wrapper around `consolidation_balance_core` — it loads the
cached taxonomy + growing model, runs the shared measurement + the threshold-free
over-merge decision, and prints two sections:

  RAW SPINE       — every pre-P7 raw attribute on two axes (SIZE via pooled
                    quartiles; SEPARABILITY via kNN own_purity), joined to the
                    post-P8 final bucket its ideas ended up in. Diagnostic only.
  OVER-MERGE      — per catch-all bucket (>= 2 source attributes), the threshold-
  DECISION          free verdict the corrector uses: a source is "own cluster" iff
                    its within-bucket neighbours are more itself than its co-merged
                    siblings; SPLIT iff >= MIN_SPLIT_SOURCES non-residual,
                    within-domain own-clusters. No magic threshold.

Read-only: no merges, no cache writes, no prompt changes. The actual correction is
`consolidation_corrector.py`, wired into step 5.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_consolidation_balance
"""

import sys
import asyncio
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_4_classifier.models_classifier import (
    TaxonomyClassifiedModel,
    TaxonomyResultsCache,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier import consolidation_balance_core as core

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

KNN_K = 10                  # neighbours per idea point (spine purity)

SIZE_LARGE_Q = 0.75         # count >= this quantile -> LARGE (spine)
SIZE_SMALL_Q = 0.25         # count <= this quantile -> SMALL (spine)
PURITY_DISTINCT_Q = 0.75    # own_purity >= this quantile -> DISTINCT (spine)
PURITY_FUSED_Q = 0.25       # own_purity <= this quantile -> FUSED (spine)

# Over-merge decision (threshold-free; passed straight to core.over_merge_decision).
PROD_K_MIN = 5
PROD_K_BAND = 2
MIN_SPLIT_SOURCES = 2       # >= this many non-residual own-clusters to split a bucket
RESIDUAL_DOMINANCE = 0.60   # share (or name == bucket name) above which a source is the residual

CODE_SOURCE = CategoriesConfig.p8_code_source
EMBEDDING_MODEL = CategoriesConfig.p8_embedding_model

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# SPINE (diagnostic, view-only)
# =============================================================================

def _classify(count, purity, thr):
    """(size, separability) -> action label, for the diagnostic spine."""
    size = "LARGE" if count >= thr["size_large"] else "SMALL" if count <= thr["size_small"] else "MID"
    sep = "DISTINCT" if purity >= thr["purity_distinct"] else "FUSED" if purity <= thr["purity_fused"] else "MID"
    if size == "SMALL" and sep == "FUSED":
        action = "MERGE (eagerness)"
    elif size == "SMALL" and sep == "DISTINCT":
        action = "PROTECT (override eagerness)"
    elif size == "LARGE" and sep == "FUSED":
        action = "RESISTANCE: merge-met-buur kandidaat"
    elif size == "LARGE" and sep == "DISTINCT":
        action = "RESISTANCE: keep"
    else:
        action = "-"
    return size, sep, action


def _build_raw_records(raw_keys, raw_meta, raw_counts, raw_dom_total, raw_purity, raw_nn,
                       raw_dominant_final, final_sources, fin_records, thr):
    records = []
    for k in raw_keys:
        d, a = k
        size, sep, action = _classify(raw_counts[k], raw_purity[k], thr)
        df = raw_dominant_final.get(k)
        merged = df is not None and len(final_sources.get(df, set())) > 1
        fr = fin_records.get(df)
        records.append({
            "domain": d, "facet": raw_meta.get(k, {}).get("facet", "(unknown)"), "attribute": a,
            "count": raw_counts[k], "share_of_domain": round(raw_counts[k] / raw_dom_total[d], 3),
            "own_purity": round(raw_purity[k], 3),
            "nearest_attr": (f"{raw_nn[k]['attr'][0]} / {raw_nn[k]['attr'][1]}" if raw_nn[k]["attr"] else None),
            "mixing": round(raw_nn[k]["mixing"], 3), "nn_cross_domain": raw_nn[k]["cross_domain"],
            "size_class": size, "separability": sep, "action": action,
            "merged_away": merged,
            "final_domain": df[0] if df else None, "final_attribute": df[1] if df else None,
            "final_count": fr["count"] if fr else None,
            "final_cross_domain": (df is not None and df[0] != d),
            "over_merge_flag": merged and sep == "DISTINCT",
        })
    return records


# =============================================================================
# MAIN
# =============================================================================

async def _run():
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cm = CacheManager()
    classified = cm.load_from_cache(FILENAME, "taxonomy_classified", variable_key, TaxonomyClassifiedModel)
    if not classified:
        raise FileNotFoundError("No taxonomy_classified cache — run step 4 first.")
    taxonomy_cache = cm.load_metadata_from_cache(FILENAME, "taxonomy", variable_key, TaxonomyResultsCache)
    if taxonomy_cache is None:
        raise FileNotFoundError("No taxonomy metadata cache — run step 4 first.")

    idea_index = core.index_growing_model(classified, CODE_SOURCE)
    raw_groups, raw_meta = core.collect_raw_groups(taxonomy_cache)
    final_groups = core.collect_final_groups(idea_index)
    if len(raw_groups) < 2:
        raise ValueError(f"Only {len(raw_groups)} raw attribute(s) — nothing to diagnose.")
    final_sources, raw_dominant_final = core.join_raw_to_final(raw_groups, idea_index)

    needed = {i for ids in raw_groups.values() for i in ids} | \
             {i for ids in final_groups.values() for i in ids}
    ids_order, matrix, gidx = await core.embed_ideas(idea_index, needed, EMBEDDING_MODEL)
    n = len(ids_order)

    nbr_idx = core.knn_indices(matrix, KNN_K)
    raw_labels = core.label_list(raw_groups, gidx, n)
    raw_purity, raw_nn = core.neighbour_stats(raw_groups, gidx, nbr_idx, raw_labels)

    raw_counts = {k: len([i for i in raw_groups[k] if i in gidx]) for k in raw_purity}
    raw_dom_total = defaultdict(int)
    for (d, _a), c in raw_counts.items():
        raw_dom_total[d] += c
    raw_keys = list(raw_counts)
    thr = {
        "size_large": float(np.percentile(list(raw_counts.values()), SIZE_LARGE_Q * 100)),
        "size_small": float(np.percentile(list(raw_counts.values()), SIZE_SMALL_Q * 100)),
        "purity_distinct": float(np.percentile([raw_purity[k] for k in raw_keys], PURITY_DISTINCT_Q * 100)),
        "purity_fused": float(np.percentile([raw_purity[k] for k in raw_keys], PURITY_FUSED_Q * 100)),
    }

    fin_records = core.build_final_records(final_groups, gidx, final_sources, MIN_SPLIT_SOURCES)
    decision = core.over_merge_decision(
        fin_records, final_groups, gidx, matrix, raw_labels,
        k_min=PROD_K_MIN, k_band=PROD_K_BAND,
        min_split_sources=MIN_SPLIT_SOURCES, residual_dominance=RESIDUAL_DOMINANCE,
    )

    raw_records = _build_raw_records(raw_keys, raw_meta, raw_counts, raw_dom_total,
                                     raw_purity, raw_nn, raw_dominant_final,
                                     final_sources, fin_records, thr)
    raw_q = {"size": core.quantiles(list(raw_counts.values())),
             "own_purity": core.quantiles([raw_purity[k] for k in raw_keys])}

    _print_report(raw_records, fin_records, raw_q, thr)
    _print_decision(decision)


def _print_decision(decision):
    print(f"\n{'=' * 96}")
    print("OVER-MERGE DECISION — within-bucket own vs sibling (threshold-free)")
    print("  own = S-members' within-B neighbours that are S ; sibling = that are a co-merged source")
    print("  own_cluster = own > sibling ; SPLIT iff >= "
          f"{MIN_SPLIT_SOURCES} non-residual within-domain own-clusters")
    print(f"{'=' * 96}")
    n_split = 0
    for b in sorted(decision, key=lambda x: -x["count"]):
        head = (f"\n[{b['domain']}] {b['attribute']}  n={b['count']} ({b['share']*100:.0f}%)  "
                f"floor={b['floor']}  eligible={b['n_eligible']}")
        if not b["measurable"]:
            print(head + f"  -> NOT MEASURABLE ({b.get('note', '')})")
            continue
        n_split += b["verdict"] == "SPLIT"
        print(head + f"  =>  {b['verdict']}  ({b['n_split']} source(s) to split out)")
        for s in b["sources"]:
            if s["residual"]:
                tag = "RESIDUAL"
            elif s["cross_domain"]:
                tag = "cross-domain (v1: keep)"
            elif s["own_cluster"]:
                tag = "OWN-CLUSTER -> split"
            else:
                tag = "interleaved"
            unstable = "" if s["stable"] else "  UNSTABLE"
            print(f"    {s['count']:>4} {s['share']*100:>3.0f}%  own={s['own']:.3f} sib={s['sibling']:.3f}  "
                  f"sep={s['separability']:.3f} k={s['k']}  -> {tag:<24}{unstable} | {s['attribute']}")
    measurable = sum(1 for b in decision if b["measurable"])
    print(f"\n  buckets to SPLIT: {n_split} / {measurable} measurable multi-source buckets")
    print(f"{'=' * 96}")


def _print_report(raw_records, fin_records, raw_q, thr):
    print(f"\n{'=' * 96}")
    print("CONSOLIDATION BALANCE  (raw pre-P7 spine -> post-P8, size x separability via kNN)")
    print(f"{FILENAME}  |  {VARIABLE}  |  n={SAMPLE_SIZE}  |  "
          f"{len(raw_records)} raw -> {len(fin_records)} final  |  k={KNN_K}")
    print(f"{'=' * 96}")
    print("\nRAW (pre-P7) distributions:")
    print(f"  size:        Q1={raw_q['size']['q1']:.0f}  med={raw_q['size']['median']:.0f}  "
          f"Q3={raw_q['size']['q3']:.0f}  [{raw_q['size']['min']:.0f}..{raw_q['size']['max']:.0f}]")
    print(f"  own_purity:  Q1={raw_q['own_purity']['q1']:.3f}  med={raw_q['own_purity']['median']:.3f}  "
          f"Q3={raw_q['own_purity']['q3']:.3f}  [{raw_q['own_purity']['min']:.3f}..{raw_q['own_purity']['max']:.3f}]")
    print(f"  THRESHOLDS:  LARGE>={thr['size_large']:.0f}  SMALL<={thr['size_small']:.0f}  "
          f"DISTINCT>={thr['purity_distinct']:.3f}  FUSED<={thr['purity_fused']:.3f}")

    print(f"\n{'=' * 96}\nRAW SPINE (size x separability -> steering action):")
    by_domain = defaultdict(list)
    for r in raw_records:
        by_domain[r["domain"]].append(r)
    for dom in sorted(by_domain):
        print(f"\n  {dom}:")
        for r in sorted(by_domain[dom], key=lambda r: -r["count"]):
            flag = "  <<< OVER-MERGE" if r["over_merge_flag"] else ""
            join = "MERGED" if r["merged_away"] else "kept/renamed"
            print(f"    {r['count']:>4} {r['share_of_domain']*100:>3.0f}%  "
                  f"{r['size_class']:<5} {r['separability']:<8} pur={r['own_purity']:.3f}  "
                  f"{join}->{r['final_attribute']}  | {r['attribute']}{flag}")

    actions = Counter(r["action"] for r in raw_records)
    n_over = sum(1 for r in raw_records if r["over_merge_flag"])
    print(f"\n{'=' * 96}\nSUMMARY:")
    for a, cnt in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>4}  {a}")
    print(f"\n  OVER-MERGE raw flags (spine heuristic): {n_over}")
    print(f"{'=' * 96}")


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()

# %%
