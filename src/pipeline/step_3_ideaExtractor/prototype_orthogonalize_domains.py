#%%

"""
PROTOTYPE (standalone, read-mostly) — sharpen domain boundaries for maximal orthogonality.

Idea: embeddings only SELECT representative exemplars (medoid) per domain; the LLM then
re-describes ALL domains jointly to be maximally orthogonal — WITHOUT reassigning any idea.
"embeddings select, the LLM decides." No pipeline change; nothing is reassigned.

Flow:
  1. load cached step-3 ideas (LLM-assigned domains) + metadata domains
  2. per domain: medoid → top-N representative ideas (instance → interpretation → abstraction)
  3. ONE LLM call: all domains + their exemplars → reformulated label/definition/
     boundary_test/exclusions (same keys, same count — only sharper wording)
  4. measure idea→own-definition agreement BEFORE vs AFTER (the thermometer), same
     assignments throughout. Rising agreement = sharper definitions describe the contents
     better. (Caveat: this sharpens DESCRIPTIONS; it cannot beat the data's separability.)

Usage:
    cd src && python -m pipeline.step_3_ideaExtractor.prototype_orthogonalize_domains
"""

import asyncio
import sys
from pathlib import Path
from typing import List

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.embedder import SharedEmbedder, find_representative_samples
from utils.llm import create_client, llm_create_async
from config import get_step_model, get_reasoning_params
from pipeline.step_3_ideaExtractor import models
from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
    DomainItem, build_orthogonalize_domains_prompt, ReformulatedDomains,
)
from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from test_data import TEST_DATA

FILENAME, VARIABLE, SAMPLE_SIZE = TEST_DATA.filename, TEST_DATA.var_name, TEST_DATA.sample_size
TOP_N = 8  # representative exemplars per domain


def _vk():
    return generate_enhanced_variable_key(selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE)


def _l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True); n[n == 0] = 1.0; return m / n


def _text(idea):
    return (idea.interpretation or idea.instance or "").strip()


def _agreement(idea_emb_norm, assigned_anchor_idx, anchors_norm, n_domains):
    """Per-domain + overall: is each idea nearest its OWN domain's anchor (argmax)?"""
    sims = idea_emb_norm @ anchors_norm.T
    nearest = sims.argmax(axis=1)
    agree = nearest == assigned_anchor_idx
    per = {}
    for j in range(n_domains):
        m = assigned_anchor_idx == j
        if m.any():
            per[j] = (int(m.sum()), float(agree[m].mean()))
    return float(agree.mean()), per


async def main():
    cm = CacheManager()
    data = cm.load_from_cache(FILENAME, "extracted_ideas", _vk(), models.IdeasExtractedModel)
    meta = cm.load_metadata_from_cache(FILENAME, "extracted_ideas", _vk(), models.ExtractionMetadata)
    domains = meta.domains or []
    keys = [d.get("key", "") for d in domains]
    labels = [d.get("label", "") for d in domains]
    key_of_label = {d["label"]: d["key"] for d in domains}
    old_def = {d["key"]: d.get("definition", "") for d in domains}

    # collect assignable ideas grouped by domain key
    ideas, idx_of_key = [], {k: j for j, k in enumerate(keys)}
    assigned = []
    per_domain_ideas = {k: [] for k in keys}
    for resp in data:
        for idea in (resp.response_ideas or []):
            lab = (idea.domain or "").strip()
            if lab in key_of_label and _text(idea):
                k = key_of_label[lab]
                ideas.append(idea); assigned.append(idx_of_key[k]); per_domain_ideas[k].append(idea)
    assigned = np.array(assigned)
    print(f"Ideas: {len(ideas)} · domains: {len(keys)} · dimension: {meta.primary_dimension}")

    emb = SharedEmbedder()
    idea_emb = _l2(await emb.embed_texts([_text(i) for i in ideas]))

    # medoid → representative exemplars per domain
    exemplars = {}
    for k in keys:
        di = [n for n in range(len(ideas)) if assigned[n] == idx_of_key[k]]
        if not di:
            exemplars[k] = []; continue
        sub = idea_emb[di]
        rep = find_representative_samples(sub, n=min(TOP_N, len(di)))
        exemplars[k] = [ideas[di[r]] for r in rep]

    # build reformulation prompt
    diag = get_dimension(meta.primary_dimension).prompt_rules.domain_diagnostic
    blocks = []
    for d in domains:
        k = d["key"]
        ex = "\n".join(
            f"      • {(i.instance or '')[:40]} → {(i.interpretation or '')[:70]} → {(i.abstraction or '')[:60]}"
            for i in exemplars.get(k, [])
        ) or "      (none)"
        block = f"  [{k}] {d['label']}: {d.get('definition','')}"
        if d.get("boundary_test"):
            block += f"\n    current boundary_test: {d['boundary_test']}"
        if d.get("exclusions"):
            block += f"\n    current exclusions: {', '.join(d['exclusions'])}"
        block += f"\n    representative ideas:\n{ex}"
        blocks.append(block)
    domains_block = "\n\n".join(blocks)

    prompt = build_orthogonalize_domains_prompt(
        language=meta.lang, survey_question=meta.var_lab, sector=meta.sector,
        entity=meta.entity, topic=meta.topic, perspective=meta.perspective, intent=meta.intent,
        primary_dimension=meta.primary_dimension, domain_diagnostic=diag, domains_block=domains_block,
    )

    model = get_step_model("idea_extraction_taxonomy")
    client = create_client(model=model, async_mode=True)
    res = await llm_create_async(client=client, model=model, prompt=prompt,
                                 response_model=ReformulatedDomains, temperature=0.0,
                                 **get_reasoning_params(model))
    # map by ORDER (key is no longer produced): res.domains[j] ↔ keys[j]
    new_by_key = {keys[j]: res.domains[j] for j in range(min(len(keys), len(res.domains)))}

    # anchors: old (label+definition) vs new (label+def+boundary+excl) vs new (def only)
    old_anchors = _l2(await emb.embed_texts([f"{labels[j]}: {old_def[keys[j]]}" for j in range(len(keys))]))
    new_rich = _l2(await emb.embed_texts([
        " ".join(filter(None, [f"{new_by_key[k].label}: {new_by_key[k].definition}", new_by_key[k].boundary_test, " ".join(new_by_key[k].exclusions)])) if k in new_by_key else f"{labels[j]}: {old_def[k]}"
        for j, k in enumerate(keys)
    ]))
    new_defonly = _l2(await emb.embed_texts([
        f"{new_by_key[k].label}: {new_by_key[k].definition}" if k in new_by_key else f"{labels[j]}: {old_def[k]}"
        for j, k in enumerate(keys)
    ]))

    ov_old, per_old = _agreement(idea_emb, assigned, old_anchors, len(keys))
    ov_def, _ = _agreement(idea_emb, assigned, new_defonly, len(keys))
    ov_rich, per_rich = _agreement(idea_emb, assigned, new_rich, len(keys))

    print("\n" + "=" * 80)
    print("AGREEMENT (same assignments; only the domain ANCHORS change)")
    print("=" * 80)
    print(f"  BEFORE (old label+definition)        : {100*ov_old:.1f}%")
    print(f"  AFTER  (new label+definition only)   : {100*ov_def:.1f}%   (isolates re-wording)")
    print(f"  AFTER  (new + boundary_test+exclusions): {100*ov_rich:.1f}%   (full new anchor)")

    print("\n" + "=" * 80)
    print("PER-DOMAIN agreement  (before → after-rich)  +  before/after definition")
    print("=" * 80)
    for j, k in enumerate(keys):
        nb, ab = per_old.get(j, (0, 0.0)), per_rich.get(j, (0, 0.0))
        nd = new_by_key.get(k)
        print(f"\n[{k}]  n={nb[0]}   {100*nb[1]:.0f}% → {100*ab[1]:.0f}%")
        print(f"   OLD label: {labels[j]}")
        print(f"   OLD def  : {old_def[k]}")
        if nd:
            print(f"   NEW label: {nd.label}")
            print(f"   NEW def  : {nd.definition}")
            print(f"   NEW ✓    : {nd.boundary_test}")
            print(f"   NEW ✗    : {', '.join(nd.exclusions)}")


if __name__ == "__main__":
    asyncio.run(main())
