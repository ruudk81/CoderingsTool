#%%

"""
Evaluate the step 3 domain partition by checking fidelity to the TAXONOMY's own
artifacts — not by clustering ideas (semantics ≠ taxonomy).

The reference is the domain DEFINITION, and the check is RELATIVE (argmax / margin),
never an absolute cosine threshold (those are inflated for same-topic text).

ASSIGNMENT CONSISTENCY (assignment-level)
    Is each idea closer to its OWN domain's definition than to any other domain's
    definition? Per-domain agreement = % of its ideas whose nearest definition is
    itself. Low agreement → its definition does not describe its contents (vague
    boundary / dumping ground). Per-idea mismatches surface the misfits / genuine
    ambiguities (the RC-8 flag, done right).

(A cross-dimension-leak check — domain definition vs dimension descriptors — was
considered and dropped: the dimension is the single chosen lens, so a domain whose
SUBJECT resembles another dimension is universal and expected, not an error. See
dev/WORK_TO_BE_DONE.md.)

Read-only. Cost = embeddings only.

Usage:
    cd src && python -m pipeline.step_3_ideaExtractor.evaluate_assignments
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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

TOP_N_MISMATCH = 25   # how many worst assignment mismatches to list


# =============================================================================
# DATA LOADING
# =============================================================================

def _variable_key() -> str:
    return generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE
    )


def load_ideas() -> List[models.IdeasExtractedSubmodel]:
    data = CacheManager().load_from_cache(
        FILENAME, "extracted_ideas", _variable_key(), models.IdeasExtractedModel
    )
    if not data:
        raise FileNotFoundError(f"No cached results for '{_variable_key()}'. Run step 3 first.")
    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)
    return ideas


def load_metadata() -> models.ExtractionMetadata:
    return CacheManager().load_metadata_from_cache(
        FILENAME, "extracted_ideas", _variable_key(), models.ExtractionMetadata
    )


def idea_text(idea: models.IdeasExtractedSubmodel) -> str:
    """Text the domain decision is conditioned on: interpretation, with fallbacks."""
    return (idea.interpretation or idea.instance or idea.idea or "").strip()


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# =============================================================================
# ASSIGNMENT CONSISTENCY (idea ↔ domain definition)
# =============================================================================

def check_assignments(ideas, idea_emb, labels, anchor_emb):
    """Each idea vs every domain definition. Relative: is its own domain the nearest?"""
    lab_idx = {l: j for j, l in enumerate(labels)}
    keep = [i for i in range(len(ideas)) if (ideas[i].domain or "").strip() in lab_idx]
    idea_n = l2_normalize(idea_emb[keep])
    anc_n = l2_normalize(anchor_emb)
    sims = idea_n @ anc_n.T                       # [N x D]
    assigned = np.array([lab_idx[(ideas[i].domain or "").strip()] for i in keep])
    nearest = sims.argmax(axis=1)
    own = sims[np.arange(len(keep)), assigned]
    other = sims.copy()
    other[np.arange(len(keep)), assigned] = -np.inf
    margin = own - other.max(axis=1)

    agree = nearest == assigned
    per_dom = {}
    for l, j in lab_idx.items():
        m = assigned == j
        per_dom[l] = (int(m.sum()), float(agree[m].mean()) if m.any() else 0.0)

    return {
        "keep": keep, "labels": labels, "assigned": assigned, "nearest": nearest,
        "margin": margin, "agree": agree, "per_dom": per_dom,
    }


def report_assignments(ideas, A):
    labels = A["labels"]
    print(f"\n{'='*78}")
    print(f"ASSIGNMENT CONSISTENCY — idea vs its domain DEFINITION  (n={len(A['keep'])})")
    print(f"{'='*78}")
    print(f"Overall agreement (idea nearest its OWN definition): {100*A['agree'].mean():.1f}%")
    print(f"  (relative check — high = ideas land where the definition says they should)")

    print(f"\n{'─'*78}\nPER-DOMAIN agreement (low → definition doesn't describe its contents)\n{'─'*78}")
    for l in sorted(labels, key=lambda x: A["per_dom"][x][1]):
        n, ag = A["per_dom"][l]
        flag = "  ⚠ weak boundary" if ag < 0.5 else ""
        print(f"  {n:5d}  agree {100*ag:5.1f}%   {l}{flag}")

    order = np.argsort(A["margin"])
    print(f"\n{'─'*78}\nWORST MISMATCHES (idea sits nearer ANOTHER definition)\n{'─'*78}")
    for k in order[:TOP_N_MISMATCH]:
        i = ideas[A["keep"][k]]
        assigned_l = labels[A["assigned"][k]]
        nearest_l = labels[A["nearest"][k]]
        if assigned_l == nearest_l:
            continue
        print(f"  m {A['margin'][k]:+.3f}  [{assigned_l} → {nearest_l}]"
              f"  \"{(i.instance or '')[:22]}\" → {(i.interpretation or '')[:60]}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    ideas = [i for i in load_ideas() if idea_text(i)]
    meta = load_metadata()
    domains = meta.domains or []
    labels = [d.get("label", "") for d in domains]
    anchor_texts = [f"{d.get('label','')}: {d.get('definition','')}" for d in domains]

    print(f"Ideas: {len(ideas)} · domains: {len(labels)} · dimension: {meta.primary_dimension}")
    embedder = SharedEmbedder()
    idea_emb = await embedder.embed_texts([idea_text(i) for i in ideas])
    anchor_emb = await embedder.embed_texts(anchor_texts)

    A = check_assignments(ideas, idea_emb, labels, anchor_emb)
    report_assignments(ideas, A)


if __name__ == "__main__":
    asyncio.run(main())
