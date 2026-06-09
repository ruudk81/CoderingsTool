#%%

"""
View consolidation balance: per-attribute diagnostic for over-merge (catch-all),
on TWO axes — SIZE and SEPARABILITY — so we can steer P7/P8 consolidation
deterministically instead of via the blunt absolute "<10-15 ideas -> almost
never standalone" rule in the prompts.

Read-only, deterministic. No merges, no cache writes, no prompt changes.

Spine = the PRE-P7 raw attribute inventory (raw_attributes /
raw_attribute_assignments from the metadata cache) — where the lever fires. Each
raw attribute is measured, given the action label the steering WOULD assign, and
joined (by idea_id) to the POST-P8 final attribute its ideas ended up in.

AXIS 1 — SIZE: idea count per attribute (moment-based: LARGE >= Q3, SMALL <= Q1).

AXIS 2 — SEPARABILITY via a kNN graph over idea points (NOT centroids).
  Centroid cosine compresses to noise on single-topic surveys (everything 0.78-
  0.97) and regresses large attributes to the global mean. A neighbour test is
  RANK-based, so it keeps full 0-1 contrast under that compression.
    own_purity  = of a member's k nearest neighbours, fraction in the SAME attr.
                    high -> own neighbourhood -> DISTINCT (separable) -> protect
                    low  -> points sit among others -> FUSED -> merging is honest
    nearest/mix = the other attribute A's points most neighbour, and how strongly
                    -> "are they neighbours anyway?"
  Caveat: own_purity is mildly size-biased upward (a large attr fills more of the
  space). The catch-all verdict therefore uses SIBLING-MIXING within a bucket,
  which is symmetric and size-fair.

Cross table -> action label:
                  FUSED (low purity)        DISTINCT (high purity)
    SMALL    MERGE (eagerness)         PROTECT (override eagerness)
    LARGE    RESISTANCE: merge-w-buur  RESISTANCE: keep

Over-merge flag: a raw attr MERGED away while it was DISTINCT (a separable
cluster) — a merge the steering would have resisted.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_consolidation_balance
"""

import sys
import json
import asyncio
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.embedder import SharedEmbedder, format_idea_text
from pipeline.step_4_classifier.models_classifier import (
    TaxonomyClassifiedModel,
    TaxonomyResultsCache,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

KNN_K = 10                  # neighbours per idea point

SIZE_LARGE_Q = 0.75         # count >= this quantile -> LARGE
SIZE_SMALL_Q = 0.25         # count <= this quantile -> SMALL
PURITY_DISTINCT_Q = 0.75    # own_purity >= this quantile -> DISTINCT (separable)
PURITY_FUSED_Q = 0.25       # own_purity <= this quantile -> FUSED

# Catch-all final bucket: holds >= this share of its domain, OR fused >= this many
# DISTINCT (separable) raw sources.
CATCHALL_SHARE = 0.50
CATCHALL_MIN_DISTINCT_SOURCES = 2

CODE_SOURCE = CategoriesConfig.p8_code_source
EMBEDDING_MODEL = CategoriesConfig.p8_embedding_model

_SENTINEL_ATTRIBUTES = {"__UNASSIGNED__", "(no attribute)", ""}

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

EXPORT_DIR = project_root / "exports" / "consolidation_balance"


# =============================================================================
# DATA COLLECTION
# =============================================================================

def _index_growing_model(classified):
    """idea_id -> {text, final (domain, attr), is_sentinel}. Post-P8 state."""
    idx = {}
    for resp in classified:
        for idea in (resp.response_ideas or []):
            attr = (idea.attribute or "").strip()
            idx[idea.idea_id] = {
                "text": format_idea_text(idea, CODE_SOURCE),
                "final": (idea.partition_name or "(unknown)", attr),
                "is_sentinel": attr in _SENTINEL_ATTRIBUTES,
            }
    return idx


def _collect_raw_groups(taxonomy_cache):
    """(domain, raw_attr) -> [idea_id] and -> facet. Pre-P7 inventory."""
    groups = defaultdict(list)
    raw_facet = {}
    for domain, res in taxonomy_cache.partition_results.items():
        raw_assign = getattr(res, "raw_attribute_assignments", None) or {}
        attr_to_facet = {}
        for facet, attrs in (getattr(res, "raw_attributes", None) or {}).items():
            for a in attrs:
                attr_to_facet[a.get("attribute_name")] = facet
        for idea_id, attr in raw_assign.items():
            attr = (attr or "").strip()
            if attr in _SENTINEL_ATTRIBUTES:
                continue
            key = (domain, attr)
            groups[key].append(idea_id)
            raw_facet.setdefault(key, attr_to_facet.get(attr, "(unknown)"))
    return groups, raw_facet


def _collect_final_groups(idea_index):
    """(domain, final_attr) -> [idea_id]. Post-P8."""
    groups = defaultdict(list)
    for idea_id, info in idea_index.items():
        if not info["is_sentinel"]:
            groups[info["final"]].append(idea_id)
    return groups


# =============================================================================
# EMBEDDING + kNN GRAPH
# =============================================================================

async def _embed_ideas(idea_index, needed_ids):
    """Embed each needed idea once. Returns (ids_order, matrix, gidx)."""
    ids = [i for i in sorted(needed_ids) if i in idea_index]
    texts = [idea_index[i]["text"] for i in ids]
    embedder = SharedEmbedder(model=EMBEDDING_MODEL)
    matrix = await embedder.embed_texts(texts)
    gidx = {i: pos for pos, i in enumerate(ids)}
    return ids, matrix, gidx


def _knn_indices(matrix, k):
    """k nearest neighbour global indices per point (self excluded)."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(matrix)
    _dist, idx = nbrs.kneighbors(matrix)
    return [[j for j in row if j != i][:k] for i, row in enumerate(idx)]


def _label_list(groups, gidx, n):
    """Global-index -> attr_key (or None) for one grouping."""
    labels = [None] * n
    for key, ids in groups.items():
        for i in ids:
            pos = gidx.get(i)
            if pos is not None:
                labels[pos] = key
    return labels


def _neighbour_stats(groups, gidx, nbr_idx, labels):
    """Per attr: own_purity, nearest foreign attr + mixing fraction."""
    purity, nn = {}, {}
    for key, ids in groups.items():
        members = [gidx[i] for i in ids if i in gidx]
        if not members:
            continue
        same = total = 0
        foreign = Counter()
        for m in members:
            for j in nbr_idx[m]:
                total += 1
                lab = labels[j]
                if lab == key:
                    same += 1
                elif lab is not None:
                    foreign[lab] += 1
        purity[key] = same / total if total else 0.0
        if foreign:
            nb, cnt = foreign.most_common(1)[0]
            nn[key] = {"attr": nb, "mixing": cnt / total, "cross_domain": nb[0] != key[0]}
        else:
            nn[key] = {"attr": None, "mixing": 0.0, "cross_domain": False}
    return purity, nn


def _sibling_mixing(sources, groups, gidx, nbr_idx, labels):
    """Per source in a final bucket: fraction of neighbours that are {own, sibling, outside}."""
    src_set = set(sources)
    out = {}
    for key in sources:
        members = [gidx[i] for i in groups[key] if i in gidx]
        if not members:
            continue
        own = sib = outside = total = 0
        for m in members:
            for j in nbr_idx[m]:
                total += 1
                lab = labels[j]
                if lab == key:
                    own += 1
                elif lab in src_set:
                    sib += 1
                else:
                    outside += 1
        if total:
            out[key] = {"own": own / total, "sibling": sib / total, "outside": outside / total}
    return out


def _quantiles(values):
    arr = np.array(values, dtype=float)
    return {"q1": float(np.percentile(arr, 25)), "median": float(np.percentile(arr, 50)),
            "q3": float(np.percentile(arr, 75)), "min": float(arr.min()), "max": float(arr.max())}


def _classify(count, purity, thr):
    """(size, separability) -> action label."""
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

    idea_index = _index_growing_model(classified)
    raw_groups, raw_facet = _collect_raw_groups(taxonomy_cache)
    final_groups = _collect_final_groups(idea_index)
    if len(raw_groups) < 2:
        raise ValueError(f"Only {len(raw_groups)} raw attribute(s) — nothing to diagnose.")

    needed = {i for ids in raw_groups.values() for i in ids} | \
             {i for ids in final_groups.values() for i in ids}
    ids_order, matrix, gidx = await _embed_ideas(idea_index, needed)
    nbr_idx = _knn_indices(matrix, KNN_K)
    n = len(ids_order)

    raw_labels = _label_list(raw_groups, gidx, n)
    fin_labels = _label_list(final_groups, gidx, n)
    raw_purity, raw_nn = _neighbour_stats(raw_groups, gidx, nbr_idx, raw_labels)
    fin_purity, _fin_nn = _neighbour_stats(final_groups, gidx, nbr_idx, fin_labels)

    raw_counts = {k: len([i for i in raw_groups[k] if i in gidx]) for k in raw_purity}
    fin_counts = {k: len([i for i in final_groups[k] if i in gidx]) for k in fin_purity}
    raw_dom_total = defaultdict(int)
    for (d, _a), c in raw_counts.items():
        raw_dom_total[d] += c
    fin_dom_total = defaultdict(int)
    for (d, _a), c in fin_counts.items():
        fin_dom_total[d] += c

    raw_keys = list(raw_counts)
    thr = {
        "size_large": float(np.percentile(list(raw_counts.values()), SIZE_LARGE_Q * 100)),
        "size_small": float(np.percentile(list(raw_counts.values()), SIZE_SMALL_Q * 100)),
        "purity_distinct": float(np.percentile([raw_purity[k] for k in raw_keys], PURITY_DISTINCT_Q * 100)),
        "purity_fused": float(np.percentile([raw_purity[k] for k in raw_keys], PURITY_FUSED_Q * 100)),
    }

    # raw -> final join; which raw sources feed each final
    final_sources = defaultdict(set)
    raw_dominant_final = {}
    for k in raw_keys:
        finals = Counter(idea_index[i]["final"] for i in raw_groups[k]
                         if i in idea_index and not idea_index[i]["is_sentinel"])
        if not finals:
            raw_dominant_final[k] = None
            continue
        raw_dominant_final[k] = finals.most_common(1)[0][0]
        for fk in finals:
            final_sources[fk].add(k)

    # raw records
    raw_records = []
    for k in raw_keys:
        d, a = k
        size, sep, action = _classify(raw_counts[k], raw_purity[k], thr)
        df = raw_dominant_final.get(k)
        merged = df is not None and len(final_sources.get(df, set())) > 1
        over_merge = merged and sep == "DISTINCT"
        raw_records.append({
            "domain": d, "facet": raw_facet.get(k, "(unknown)"), "attribute": a,
            "count": raw_counts[k], "share_of_domain": round(raw_counts[k] / raw_dom_total[d], 3),
            "own_purity": round(raw_purity[k], 3),
            "nearest_attr": (f"{raw_nn[k]['attr'][0]} / {raw_nn[k]['attr'][1]}" if raw_nn[k]["attr"] else None),
            "mixing": round(raw_nn[k]["mixing"], 3), "nn_cross_domain": raw_nn[k]["cross_domain"],
            "size_class": size, "separability": sep, "action": action,
            "merged_away": merged,
            "final_domain": df[0] if df else None, "final_attribute": df[1] if df else None,
            "final_count": fin_counts.get(df) if df else None,
            "final_cross_domain": (df is not None and df[0] != d),
            "over_merge_flag": over_merge,
        })

    # final records + catch-all detection
    fin_records = {}
    for k in fin_purity:
        d, a = k
        share = fin_counts[k] / fin_dom_total[d]
        sources = final_sources.get(k, set())
        n_distinct_src = sum(
            1 for s in sources
            if raw_counts.get(s, 0) >= thr["size_small"] and raw_purity.get(s, 0) >= thr["purity_distinct"]
        )
        is_catchall = share >= CATCHALL_SHARE or n_distinct_src >= CATCHALL_MIN_DISTINCT_SOURCES
        fin_records[k] = {
            "domain": d, "attribute": a, "count": fin_counts[k], "share": round(share, 3),
            "own_purity": round(fin_purity[k], 3), "n_sources": len(sources),
            "n_distinct_sources": n_distinct_src, "catch_all": bool(is_catchall),
        }

    catchall_detail = _catchall_breakdown(fin_records, final_sources, raw_groups, gidx,
                                          nbr_idx, raw_labels, raw_counts, raw_purity)

    raw_q = {"size": _quantiles(list(raw_counts.values())),
             "own_purity": _quantiles([raw_purity[k] for k in raw_keys])}
    fin_q = {"size": _quantiles(list(fin_counts.values())),
             "own_purity": _quantiles(list(fin_purity.values()))}

    _print_report(raw_records, list(fin_records.values()), catchall_detail, raw_q, fin_q, thr)
    _write_json(raw_records, list(fin_records.values()), catchall_detail, variable_key, raw_q, fin_q, thr)


def _catchall_breakdown(fin_records, final_sources, raw_groups, gidx, nbr_idx,
                        raw_labels, raw_counts, raw_purity):
    """For each catch-all bucket, per-source own/sibling/outside neighbour mix."""
    detail = []
    for fk, rec in fin_records.items():
        if not rec["catch_all"]:
            continue
        sources = sorted(final_sources.get(fk, set()), key=lambda s: -raw_counts.get(s, 0))
        mix = _sibling_mixing(sources, raw_groups, gidx, nbr_idx, raw_labels)
        src_rows = []
        for s in sources:
            m = mix.get(s, {"own": 0, "sibling": 0, "outside": 0})
            verdict = "interleaves w/ siblings" if m["sibling"] >= m["own"] else "own cluster"
            src_rows.append({
                "domain": s[0], "attribute": s[1], "count": raw_counts.get(s, 0),
                "own_purity": round(raw_purity.get(s, 0), 3),
                "sibling_mix": round(m["sibling"], 3), "outside_mix": round(m["outside"], 3),
                "verdict": verdict,
            })
        n_own = sum(1 for r in src_rows if r["verdict"] == "own cluster")
        detail.append({
            "domain": fk[0], "attribute": fk[1], "count": rec["count"], "share": rec["share"],
            "bucket_verdict": ("OVER-MERGE (separable sources)" if n_own >= 2
                               else "interleaved (merge defensible)"),
            "sources": src_rows,
        })
    return detail


def _print_report(raw_records, fin_records, catchall_detail, raw_q, fin_q, thr):
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

    # ---- THE STAR: catch-all deep dive (are the fused sources really neighbours?) ----
    print(f"\n{'=' * 96}\nCATCH-ALL DEEP DIVE — were the fused sources actually neighbours?")
    for c in sorted(catchall_detail, key=lambda x: -x["count"]):
        print(f"\n  [{c['domain']}] {c['attribute']}  n={c['count']} ({c['share']*100:.0f}% v domein)"
              f"  => {c['bucket_verdict']}")
        for s in c["sources"]:
            print(f"      {s['count']:>4}  purity={s['own_purity']:.3f}  "
                  f"sibling-mix={s['sibling_mix']:.3f}  outside={s['outside_mix']:.3f}  "
                  f"-> {s['verdict']:<24} | {s['attribute']}")

    # ---- spine (compact) ----
    print(f"\n{'=' * 96}\nRAW SPINE (size x separability -> steering action):")
    by_domain = defaultdict(list)
    for r in raw_records:
        by_domain[r["domain"]].append(r)
    for dom in sorted(by_domain):
        rows = sorted(by_domain[dom], key=lambda r: -r["count"])
        print(f"\n  {dom}:")
        for r in rows:
            flag = "  <<< OVER-MERGE" if r["over_merge_flag"] else ""
            join = "MERGED" if r["merged_away"] else "kept/renamed"
            print(f"    {r['count']:>4} {r['share_of_domain']*100:>3.0f}%  "
                  f"{r['size_class']:<5} {r['separability']:<8} pur={r['own_purity']:.3f}  "
                  f"{join}->{r['final_attribute']}  | {r['attribute']}{flag}")

    actions = Counter(r["action"] for r in raw_records)
    n_over = sum(1 for r in raw_records if r["over_merge_flag"])
    n_catch = sum(1 for c in catchall_detail if c["bucket_verdict"].startswith("OVER-MERGE"))
    print(f"\n{'=' * 96}\nSUMMARY:")
    for a, nn_ in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"  {nn_:>4}  {a}")
    print(f"\n  OVER-MERGE raw flags: {n_over}")
    print(f"  CATCH-ALL buckets: {len(catchall_detail)}  "
          f"(verdict OVER-MERGE / separable: {n_catch})")
    print(f"{'=' * 96}")


def _write_json(raw_records, fin_records, catchall_detail, variable_key, raw_q, fin_q, thr):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": FILENAME, "variable_key": variable_key,
        "n_raw": len(raw_records), "n_final": len(fin_records), "knn_k": KNN_K,
        "config": {
            "code_source": CODE_SOURCE, "size_large_q": SIZE_LARGE_Q, "size_small_q": SIZE_SMALL_Q,
            "purity_distinct_q": PURITY_DISTINCT_Q, "purity_fused_q": PURITY_FUSED_Q,
            "catchall_share": CATCHALL_SHARE, "catchall_min_distinct_sources": CATCHALL_MIN_DISTINCT_SOURCES,
        },
        "raw_distributions": raw_q, "final_distributions": fin_q, "thresholds": thr,
        "catch_all_buckets": catchall_detail,
        "raw_attributes": raw_records, "final_attributes": fin_records,
    }
    path = EXPORT_DIR / f"{Path(FILENAME).stem}_{variable_key}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()

# %%
