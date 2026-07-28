#%%

"""
View contamination: read-only diagnostic for ideas sitting in the wrong attribute.

Contamination is distinct from fragmentation. Fragmentation is two labels that
should have been one; contamination is one idea filed under a label that does not
describe it. Both inflate any measure of "overlap" between attributes, but only
fragmentation is fixable by consolidation — so they have to be told apart before
either is worked on.

MEASUREMENT (threshold-free, embeddings only — no LLM calls)
    Every idea is compared to its own attribute's centroid, computed LEAVE-ONE-OUT
    so a small attribute cannot look pure merely by containing the idea, and to the
    centroid of every other attribute. An idea is MISPLACED iff some rival centroid
    is nearer than its own. Same shape as `over_merge_decision`: a comparison, not a
    cut-off.

WHERE IT WAS CREATED — the question this diagnostic exists to answer
    P6      rival sits in the SAME facet — the attribute pick was wrong, and P6
            could in principle have got it right.
    P3      rival sits in ANOTHER FACET of the same domain — the facet pick was
            wrong, and no attribute-level step can rescue it: P6 only ever chooses
            within the facet P3 handed it.
    STEP 3  rival sits in ANOTHER DOMAIN — domain assignment put it there.

CAUSE SIGNALS (relative to this dataset, no absolute thresholds)
    NEAR-SYNONYM  the pair's label similarity is above the median among NEAREST-RIVAL
                  pairs. This is the one cause that belongs to consolidation: merging
                  the pair removes it. The baseline is nearest-rival pairs and not all
                  pairs, because the rival is selected as the nearest centroid and
                  label similarity correlates with that — against an all-pairs median
                  almost every rival scores "near-synonym" by selection alone (77% vs
                  the 43% the correct baseline gives).
    LOW-DISCRIM   the idea separates the centroids poorly (max minus mean similarity
                  below the dataset's first quartile) — the signature of contentless
                  answers that have no attribute they belong to.
    OTHER         genuinely misplaced. Embeddings CANNOT split this into
                  "multi-aspect" and "misread" — both present as an idea nearer to a
                  rival whose label is not a synonym. Separating them needs judgment
                  on a sample; this diagnostic deliberately does not guess.

Read-only: no cache writes, no merges, no reassignment.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_contamination
"""

import sys
import asyncio
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.embedder import SharedEmbedder
from models import TaxonomyClassifiedModel, TaxonomyResultsCache
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.cross_domain_consolidator import CrossDomainConsolidator
from pipeline.step_4_classifier import consolidation_balance_core as core

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

CONFIG = CategoriesConfig()
CODE_SOURCE = CONFIG.p8_code_source
EMBEDDING_MODEL = CONFIG.p8_embedding_model

TOP_ATTRIBUTES = 15         # worst-contaminated attributes listed
EXAMPLES_PER_ATTRIBUTE = 4  # misplaced ideas shown per attribute


# =============================================================================
# HELPERS
# =============================================================================

def _normalize(m):
    """Row-wise L2 normalisation; zero rows stay zero."""
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1.0, n)


def collect_attribute_meta(taxonomy_cache):
    """(domain, attribute) -> {facet, description} from the taxonomy STRUCTURE."""
    meta = {}
    for domain, res in taxonomy_cache.partition_results.items():
        for facet, attrs in (getattr(res, "attributes", None) or {}).items():
            for a in attrs:
                name = a.get("attribute_name") if isinstance(a, dict) else None
                if name:
                    meta[(domain, name)] = {
                        "facet": facet,
                        "description": (a.get("attribute_description") or ""),
                    }
    return meta


def where_created(own_key, rival_key, meta, home):
    """Which phase put this idea here: P6, P3 or STEP 3."""
    if own_key[0] != rival_key[0]:
        return "STEP 3"
    own_facet = meta.get(own_key, {}).get("facet") or home.get(own_key[1], (None, None))[1]
    rival_facet = meta.get(rival_key, {}).get("facet") or home.get(rival_key[1], (None, None))[1]
    return "P6" if own_facet == rival_facet else "P3"


# =============================================================================
# MEASUREMENT
# =============================================================================

async def measure(idea_index, final_groups, meta):
    """Per idea: own vs nearest-rival centroid, leave-one-out on the own side."""
    keys = [k for k, ids in final_groups.items()
            if len([i for i in ids if i in idea_index]) >= 2]
    needed = {i for k in keys for i in final_groups[k]}
    ids, matrix, gidx = await core.embed_ideas(idea_index, needed, EMBEDDING_MODEL)
    X = _normalize(np.asarray(matrix, dtype=float))

    members = {k: [gidx[i] for i in final_groups[k] if i in gidx] for k in keys}
    members = {k: v for k, v in members.items() if len(v) >= 2}
    keys = list(members)

    sums = np.vstack([X[members[k]].sum(axis=0) for k in keys])
    C = _normalize(sums)                      # attribute centroids
    S = X @ C.T                               # idea x attribute similarity

    # Own-side similarity must exclude the idea itself, or a 2-member attribute
    # scores near 1.0 for free and never registers as contaminated.
    own_col = np.full(len(X), np.nan)
    for c, k in enumerate(keys):
        pos = members[k]
        loo = _normalize(sums[c] - X[pos])
        own_col[pos] = np.einsum("ij,ij->i", X[pos], loo)

    label_texts = [f"{k[1]} — {meta.get(k, {}).get('description', '')}".strip(" —")
                   for k in keys]
    L = _normalize(np.asarray(
        await SharedEmbedder(model=EMBEDDING_MODEL).embed_texts(label_texts), dtype=float))
    LS = L @ L.T
    np.fill_diagonal(LS, np.nan)

    col_of = {k: c for c, k in enumerate(keys)}
    rows = []
    for k in keys:
        c = col_of[k]
        for pos in members[k]:
            sims = S[pos].copy()
            sims[c] = -np.inf
            r = int(np.argmax(sims))
            rows.append({
                "idea_id": ids[pos],
                "text": idea_index[ids[pos]]["text"],
                "own": k,
                "rival": keys[r],
                "own_sim": float(own_col[pos]),
                "rival_sim": float(sims[r]),
                "label_sim": float(LS[c, r]),
                "discrim": float(S[pos].max() - S[pos].mean()),
            })
    return rows, members


def classify(rows, meta, home):
    """Annotate each misplaced idea with where it was created and its cause signal.

    The near-synonym baseline is the median label similarity among NEAREST-RIVAL
    pairs, not among all pairs. The rival is picked as the nearest centroid, and
    label similarity correlates with that — so an all-pairs baseline would put
    almost every rival above it and the signal would be a selection effect rather
    than a finding.
    """
    misplaced = [r for r in rows if r["rival_sim"] > r["own_sim"]]
    if not misplaced:
        return misplaced
    q1 = float(np.percentile([r["discrim"] for r in rows], 25))
    rival_baseline = float(np.nanmedian([r["label_sim"] for r in rows]))
    for r in misplaced:
        r["where"] = where_created(r["own"], r["rival"], meta, home)
        if r["label_sim"] > rival_baseline:
            r["cause"] = "NEAR-SYNONYM"
        elif r["discrim"] <= q1:
            r["cause"] = "LOW-DISCRIM"
        else:
            r["cause"] = "OTHER"
    return misplaced, rival_baseline


# =============================================================================
# REPORTING
# =============================================================================

def report(rows, misplaced, members, meta, rival_baseline):
    n = len(rows)
    print(f"\n{'=' * 78}\nCONTAMINATION — {n} ideas across {len(members)} attributes")
    print(f"{'=' * 78}")
    print(f"\nMisplaced (a rival centroid is nearer than its own): "
          f"{len(misplaced)} / {n} ({len(misplaced) / n:.1%})")
    print(f"Near-synonym baseline (median label similarity among nearest-rival "
          f"pairs): {rival_baseline:.3f}")

    print("\nWHERE IT WAS CREATED")
    where = Counter(r["where"] for r in misplaced)
    for w, label in (("P6", "P6 — wrong attribute, same facet (recoverable)"),
                     ("P3", "P3 — wrong facet, same domain (unrecoverable at P6)"),
                     ("STEP 3", "STEP 3 — wrong domain")):
        c = where.get(w, 0)
        print(f"  {label:<52} {c:>5}  {c / len(misplaced):>6.1%}")

    print("\nCAUSE SIGNAL")
    cause = Counter(r["cause"] for r in misplaced)
    for c_, label in (("NEAR-SYNONYM", "NEAR-SYNONYM — rival label is a near-synonym (consolidation)"),
                      ("LOW-DISCRIM", "LOW-DISCRIM  — idea separates nothing (contentless)"),
                      ("OTHER", "OTHER        — multi-aspect or misread (needs judgment)")):
        c = cause.get(c_, 0)
        print(f"  {label:<62} {c:>5}  {c / len(misplaced):>6.1%}")

    per_attr = defaultdict(list)
    for r in misplaced:
        per_attr[r["own"]].append(r)
    ranked = sorted(per_attr.items(),
                    key=lambda kv: len(kv[1]) / len(members[kv[0]]), reverse=True)

    print(f"\n{'=' * 78}\nWORST-CONTAMINATED ATTRIBUTES (top {TOP_ATTRIBUTES} by share)\n{'=' * 78}")
    for key, rs in ranked[:TOP_ATTRIBUTES]:
        total = len(members[key])
        facet = meta.get(key, {}).get("facet", "(unknown)")
        print(f"\n{key[1]}  [{key[0]} / {facet}]")
        print(f"  {len(rs)}/{total} misplaced ({len(rs) / total:.0%})")
        top_rival, cnt = Counter(r["rival"] for r in rs).most_common(1)[0]
        to_top = [r for r in rs if r["rival"] == top_rival]
        print(f"  dominant rival: {top_rival[1]} [{top_rival[0]}] — {cnt} ideas "
              f"({cnt / len(rs):.0%} of the misplaced), signal "
              f"{Counter(r['cause'] for r in to_top).most_common(1)[0][0]}")
        # Show the ideas that go to the DOMINANT rival: that is where the finding
        # sits. Printing the first N misplaced instead surfaces the long tail.
        for r in to_top[:EXAMPLES_PER_ATTRIBUTE]:
            print(f"    {r['cause']:<13} {r['where']:<7} | {r['text'][:56]}")


# =============================================================================
# MAIN
# =============================================================================

async def _run():
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cm = CacheManager()
    classified = cm.load_from_cache(FILENAME, "taxonomy_classified", variable_key,
                                    TaxonomyClassifiedModel)
    if not classified:
        raise FileNotFoundError("No taxonomy_classified cache — run step 4 first.")
    taxonomy_cache = cm.load_metadata_from_cache(FILENAME, "taxonomy", variable_key,
                                                 TaxonomyResultsCache)
    if taxonomy_cache is None:
        raise FileNotFoundError("No taxonomy metadata cache — run step 4 first.")

    idea_index = core.index_growing_model(classified, CODE_SOURCE)
    final_groups = core.collect_final_groups(idea_index)
    meta = collect_attribute_meta(taxonomy_cache)
    home = CrossDomainConsolidator.attr_structure_home(taxonomy_cache)

    rows, members = await measure(idea_index, final_groups, meta)
    misplaced, rival_baseline = classify(rows, meta, home)
    report(rows, misplaced, members, meta, rival_baseline)


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
