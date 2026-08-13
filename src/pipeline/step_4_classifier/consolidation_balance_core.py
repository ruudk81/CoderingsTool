"""
Consolidation balance — shared stateless core.

Pure measurement + the THRESHOLD-FREE over-merge decision: a counter-metric that
tracks catch-all over-merge independently of the consolidation phases, used by the
read-only diagnostic (`view_consolidation_balance.py`). No I/O, no printing, no
cache, no TEST_DATA — safe to import from step 5.

Over-merge decision (per catch-all bucket B, per provenance source S, within B):
    own     = mean fraction of S-members' k neighbours that are S
    sibling = mean fraction that are a DIFFERENT eligible source of B
    own_cluster = own > sibling                       (no magic threshold)
The RESIDUAL (name == bucket name, or share >= residual_dominance) is never a split
candidate. Cross-domain sources are excluded (v1 = within-domain only). Bucket
verdict = SPLIT iff >= min_split_sources NON-RESIDUAL, within-domain, stable
own-clusters clear the substantial floor `floor(log|B|)`.
"""

from collections import defaultdict, Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils.embedder import SharedEmbedder, format_idea_text

SENTINEL_ATTRIBUTES = {"__UNASSIGNED__", "(no attribute)", ""}


# =============================================================================
# DATA COLLECTION (from the cached taxonomy + growing model)
# =============================================================================

def index_growing_model(classified, code_source):
    """idea_id -> {text, final (domain, attr), is_sentinel}. Final state, after refinement and cross-scope."""
    idx = {}
    for resp in classified:
        for idea in (resp.response_ideas or []):
            attr = (idea.attribute or "").strip()
            idx[idea.idea_id] = {
                "text": format_idea_text(idea, code_source),
                "final": (idea.partition_name or "(unknown)", attr),
                "is_sentinel": attr in SENTINEL_ATTRIBUTES,
            }
    return idx


def collect_raw_groups(taxonomy_cache):
    """(domain, raw_attr) -> [idea_id], plus raw_meta[(domain,attr)] = {facet, description}.

    The pre-refinement provenance inventory from raw_attributes / raw_attribute_assignments.
    """
    groups = defaultdict(list)
    raw_meta = {}
    for domain, res in taxonomy_cache.partition_results.items():
        raw_assign = getattr(res, "raw_attribute_assignments", None) or {}
        meta = {}
        for facet, attrs in (getattr(res, "raw_attributes", None) or {}).items():
            for a in attrs:
                meta[a.get("attribute_name")] = {
                    "facet": facet, "description": a.get("attribute_definition", ""),
                }
        for idea_id, attr in raw_assign.items():
            attr = (attr or "").strip()
            if attr in SENTINEL_ATTRIBUTES:
                continue
            key = (domain, attr)
            groups[key].append(idea_id)
            raw_meta.setdefault(key, meta.get(attr, {"facet": "(unknown)", "description": ""}))
    return groups, raw_meta


def collect_final_groups(idea_index):
    """(domain, final_attr) -> [idea_id]. Final state."""
    groups = defaultdict(list)
    for idea_id, info in idea_index.items():
        if not info["is_sentinel"]:
            groups[info["final"]].append(idea_id)
    return groups


def join_raw_to_final(raw_groups, idea_index):
    """Returns (final_sources, raw_dominant_final): which raw sources feed each
    final bucket (by idea_id), and each raw source's dominant final."""
    final_sources = defaultdict(set)
    raw_dominant_final = {}
    for k, ids in raw_groups.items():
        finals = Counter(idea_index[i]["final"] for i in ids
                         if i in idea_index and not idea_index[i]["is_sentinel"])
        if not finals:
            raw_dominant_final[k] = None
            continue
        raw_dominant_final[k] = finals.most_common(1)[0][0]
        for fk in finals:
            final_sources[fk].add(k)
    return final_sources, raw_dominant_final


def build_final_records(final_groups, gidx, final_sources, min_sources=2):
    """fin_records[(domain,attr)] = {domain, attribute, count, share, n_sources, catch_all}.

    A bucket is a correction candidate ("catch_all") iff it was formed from
    >= min_sources raw sources (a potential over-merge). Threshold-free — no
    quartile or share dependency in the decision path.
    """
    fin_counts = {k: len([i for i in ids if i in gidx]) for k, ids in final_groups.items()}
    dom_total = defaultdict(int)
    for (d, _a), c in fin_counts.items():
        dom_total[d] += c
    recs = {}
    for k, c in fin_counts.items():
        n_src = len(final_sources.get(k, set()))
        recs[k] = {
            "domain": k[0], "attribute": k[1], "count": c,
            "share": round(c / dom_total[k[0]], 3) if dom_total[k[0]] else 0.0,
            "n_sources": n_src, "catch_all": n_src >= min_sources,
        }
    return recs


# =============================================================================
# EMBEDDING + kNN
# =============================================================================

async def embed_ideas(idea_index, needed_ids, embedding_model):
    """Embed each needed idea once. Returns (ids_order, matrix, gidx)."""
    ids = [i for i in sorted(needed_ids) if i in idea_index]
    texts = [idea_index[i]["text"] for i in ids]
    matrix = await SharedEmbedder(model=embedding_model).embed_texts(texts)
    gidx = {i: pos for pos, i in enumerate(ids)}
    return ids, matrix, gidx


def knn_indices(matrix, k):
    """k nearest neighbour global indices per point (self excluded)."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(matrix)
    _dist, idx = nbrs.kneighbors(matrix)
    return [[j for j in row if j != i][:k] for i, row in enumerate(idx)]


def label_list(groups, gidx, n):
    """Global-index -> attr_key (or None) for one grouping."""
    labels = [None] * n
    for key, ids in groups.items():
        for i in ids:
            pos = gidx.get(i)
            if pos is not None:
                labels[pos] = key
    return labels


def neighbour_stats(groups, gidx, nbr_idx, labels):
    """Per attr: own_purity + nearest foreign attr + mixing fraction (diagnostic spine)."""
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


def quantiles(values):
    arr = np.array(values, dtype=float)
    return {"q1": float(np.percentile(arr, 25)), "median": float(np.percentile(arr, 50)),
            "q3": float(np.percentile(arr, 75)), "min": float(arr.min()), "max": float(arr.max())}


# =============================================================================
# THE DECISION (threshold-free; used identically by gate + corrector)
# =============================================================================

def over_merge_decision(fin_records, final_groups, gidx, matrix, raw_labels, *,
                        k_min=5, k_band=2, min_split_sources=2, residual_dominance=0.60):
    """Per catch-all bucket: own vs sibling within-B, residual + cross-domain guards.

    Returns a list of bucket dicts: {domain, attribute, count, share, floor,
    n_eligible, measurable, verdict (SPLIT/KEEP), n_split, sources:[...]} where each
    source carries own/sibling/separability/own_cluster/residual/cross_domain/stable/k.
    `split_out` for a SPLIT bucket = sources with own_cluster & stable & not residual
    & not cross_domain.
    """
    out = []
    for fk, rec in fin_records.items():
        if not rec["catch_all"]:
            continue
        member_pos = [gidx[i] for i in final_groups[fk] if i in gidx]
        b = len(member_pos)
        if b < 3:
            continue
        sub = matrix[member_pos]
        sublabels = [raw_labels[p] for p in member_pos]   # each member's RAW source key
        floor = int(np.log(b)) if b > 1 else 0
        src_count = Counter(s for s in sublabels if s is not None)
        eligible = {s: c for s, c in src_count.items() if c >= floor}
        elig_set = set(eligible)
        bucket = {
            "domain": fk[0], "attribute": fk[1], "count": rec["count"], "share": rec["share"],
            "floor": floor, "n_eligible": len(eligible), "measurable": True,
            "verdict": "KEEP", "n_split": 0, "sources": [],
        }
        if len(eligible) < 2:
            bucket.update(measurable=False, note="fewer than 2 sources clear the substantial floor")
            out.append(bucket)
            continue
        smallest = min(eligible.values())
        k_cap = min(smallest - 1, b - 1)
        if k_cap < k_min:
            bucket.update(measurable=False,
                          note=f"smallest eligible source ({smallest}) too thin for k>={k_min}")
            out.append(bucket)
            continue

        max_k = min(k_cap + k_band, b - 1)
        nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine").fit(sub)
        _d, idx = nbrs.kneighbors(sub)
        nbr_local = [[j for j in row if j != i][:max_k] for i, row in enumerate(idx)]

        def _own_sib(member_idxs, S, k):
            owns, sibs = [], []
            for m in member_idxs:
                nb = nbr_local[m][:k]
                if not nb:
                    continue
                owns.append(sum(1 for j in nb if sublabels[j] == S) / len(nb))
                sibs.append(sum(1 for j in nb if sublabels[j] in elig_set and sublabels[j] != S) / len(nb))
            return (float(np.mean(owns)) if owns else 0.0, float(np.mean(sibs)) if sibs else 0.0)

        for S, c in eligible.items():
            members = [i for i, lab in enumerate(sublabels) if lab == S]
            chance = c / b
            k = int(min(max(round(np.sqrt(c)), k_min), k_cap))
            verdicts = []
            for kk in {max(1, k - k_band), k, min(max_k, k + k_band)}:
                o, s = _own_sib(members, S, kk)
                verdicts.append(o > s)
            own, sib = _own_sib(members, S, k)
            sep = max(0.0, min(1.0, (own - chance) / (1 - chance))) if chance < 1 else 0.0
            is_residual = (S[1] == fk[1]) or (chance >= residual_dominance)
            bucket["sources"].append({
                "domain": S[0], "attribute": S[1], "count": c, "share": round(chance, 3),
                "own": round(own, 3), "sibling": round(sib, 3), "separability": round(sep, 3),
                "own_cluster": bool(own > sib), "stable": all(verdicts) == any(verdicts),
                "residual": bool(is_residual), "cross_domain": S[0] != fk[0], "k": k,
            })
        bucket["sources"].sort(key=lambda r: (r["residual"], -r["own"]))
        split = [s for s in bucket["sources"] if _is_split_source(s)]
        bucket["n_split"] = len(split)
        bucket["verdict"] = "SPLIT" if len(split) >= min_split_sources else "KEEP"
        out.append(bucket)
    return out


def _is_split_source(s):
    """A source eligible to be split out of its bucket (v1: within-domain only)."""
    return s["own_cluster"] and s["stable"] and not s["residual"] and not s["cross_domain"]


def split_sources(bucket):
    """The sources that would be split out of a SPLIT bucket."""
    return [s for s in bucket["sources"] if _is_split_source(s)]
