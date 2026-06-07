#%%

"""
Evaluate the step 3 domain partition with embedding-based separability metrics.

Read-only. Embeds each idea's interpretation (the rung the domain decision is
conditioned on) and quantifies what the discovery prompt only asserts in prose
("ontologically distinct, semantically distant"):

  1. Inter-domain centroid similarity  → which domain PAIRS overlap (non-orthogonal)
  2. Intra-domain cohesion             → how tight each domain is internally
  3. Silhouette (cosine)               → per-idea fit to own domain vs nearest other
  4. Ambiguity margin                  → ideas sitting on a boundary (the eekhoorn case)

This doubles as the RC-2 distinctness check. Cost = embeddings only (cheap).

Usage:
    cd src && python -m pipeline.step_3_ideaExtractor.evaluate_domains
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import silhouette_samples

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.embedder import SharedEmbedder
from pipeline.step_3_ideaExtractor import models

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

EXCLUDE_DOMAINS = {"Other"}     # heterogeneous overflow — not a semantic cluster
AMBIGUITY_MARGIN = 0.05         # margin ≤ this → boundary/ambiguous idea
TOP_N_AMBIGUOUS = 25            # how many ambiguous ideas to list
TOP_N_PAIRS = 6                 # how many most-similar domain pairs to list


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ideas() -> List[models.IdeasExtractedSubmodel]:
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE
    )
    data = CacheManager().load_from_cache(
        FILENAME, "extracted_ideas", variable_key, models.IdeasExtractedModel
    )
    if not data:
        raise FileNotFoundError(
            f"No cached results for variable_key '{variable_key}'. Run step 3 first."
        )
    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)
    return ideas


def idea_text(idea: models.IdeasExtractedSubmodel) -> str:
    """Text the domain decision is conditioned on: interpretation, with fallbacks."""
    return (idea.interpretation or idea.instance or idea.idea or "").strip()


# =============================================================================
# METRICS
# =============================================================================

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def evaluate(ideas: List[models.IdeasExtractedSubmodel], emb: np.ndarray):
    domains = np.array([(i.domain or "(none)").strip() for i in ideas])
    keep = np.array([d not in EXCLUDE_DOMAINS for d in domains])

    emb_k = l2_normalize(emb[keep])
    dom_k = domains[keep]
    ideas_k = [i for i, k in zip(ideas, keep) if k]
    labels = sorted(set(dom_k))

    # --- centroids (normalized) ---
    centroids = {d: l2_normalize(emb_k[dom_k == d].mean(axis=0, keepdims=True))[0] for d in labels}
    C = np.stack([centroids[d] for d in labels])

    # --- 1. inter-domain centroid similarity ---
    sim = C @ C.T
    pairs = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            pairs.append((sim[a, b], labels[a], labels[b]))
    pairs.sort(reverse=True)

    # --- 2. intra-domain cohesion (mean cos to own centroid) ---
    cohesion = {}
    for d in labels:
        m = emb_k[dom_k == d]
        cohesion[d] = float((m @ centroids[d]).mean())

    # --- 3. silhouette (cosine) ---
    sil = silhouette_samples(emb_k, dom_k, metric="cosine") if len(labels) > 1 else np.zeros(len(dom_k))
    sil_by_dom = {d: float(sil[dom_k == d].mean()) for d in labels}

    # --- 4. ambiguity margin: sim(own centroid) - max sim(other centroid) ---
    sims_all = emb_k @ C.T  # [N x K]
    lab_idx = {d: j for j, d in enumerate(labels)}
    own = np.array([sims_all[n, lab_idx[dom_k[n]]] for n in range(len(dom_k))])
    other = sims_all.copy()
    for n in range(len(dom_k)):
        other[n, lab_idx[dom_k[n]]] = -np.inf
    nearest_other_idx = other.argmax(axis=1)
    nearest_other = other.max(axis=1)
    margin = own - nearest_other

    return {
        "labels": labels, "dom_k": dom_k, "ideas_k": ideas_k,
        "pairs": pairs, "cohesion": cohesion, "sil_overall": float(sil.mean()),
        "sil_by_dom": sil_by_dom, "margin": margin,
        "nearest_other": [labels[j] for j in nearest_other_idx],
    }


# =============================================================================
# REPORT
# =============================================================================

def report(ideas, R):
    labels, dom_k = R["labels"], R["dom_k"]
    counts = {d: int((dom_k == d).sum()) for d in labels}
    n_excl = len(ideas) - len(dom_k)

    print(f"\n{'='*78}")
    print(f"DOMAIN PARTITION EVALUATION — {len(dom_k)} ideas, {len(labels)} domains"
          f"  (excluded: {n_excl} {EXCLUDE_DOMAINS})")
    print(f"{'='*78}")
    print(f"Overall mean silhouette (cosine): {R['sil_overall']:+.3f}   "
          f"(>0.25 strong · 0.1-0.25 weak · <0.1 overlap-heavy)")

    print(f"\n{'─'*78}\nPER-DOMAIN  (size · cohesion · silhouette)\n{'─'*78}")
    for d in sorted(labels, key=lambda x: -counts[x]):
        print(f"  {counts[d]:5d}  coh {R['cohesion'][d]:.3f}  sil {R['sil_by_dom'][d]:+.3f}   {d}")

    print(f"\n{'─'*78}\nMOST-SIMILAR DOMAIN PAIRS  (high = overlap / non-orthogonal)\n{'─'*78}")
    for s, a, b in R["pairs"][:TOP_N_PAIRS]:
        flag = "  ⚠ overlap" if s > 0.5 else ""
        print(f"  cos {s:.3f}   {a}  ↔  {b}{flag}")

    margin = R["margin"]
    n_amb = int((margin <= AMBIGUITY_MARGIN).sum())
    n_mis = int((margin < 0).sum())
    print(f"\n{'─'*78}")
    print(f"AMBIGUITY  (margin = sim_own − sim_nearest_other)")
    print(f"  {n_amb} ideas ({100*n_amb/len(margin):.1f}%) on a boundary (margin ≤ {AMBIGUITY_MARGIN})")
    print(f"  {n_mis} ideas ({100*n_mis/len(margin):.1f}%) closer to ANOTHER domain (margin < 0 → likely misplaced)")
    print(f"{'─'*78}")
    order = np.argsort(margin)
    for n in order[:TOP_N_AMBIGUOUS]:
        idea = R["ideas_k"][n]
        print(f"  m {margin[n]:+.3f}  [{R['dom_k'][n]} → {R['nearest_other'][n]}]")
        print(f"            \"{(idea.instance or '')[:30]}\" → {(idea.interpretation or '')[:80]}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    ideas = load_ideas()
    ideas = [i for i in ideas if idea_text(i)]
    print(f"Embedding {len(ideas)} idea interpretations...")
    emb = await SharedEmbedder().embed_texts([idea_text(i) for i in ideas])
    R = evaluate(ideas, emb)
    report(ideas, R)


if __name__ == "__main__":
    asyncio.run(main())
