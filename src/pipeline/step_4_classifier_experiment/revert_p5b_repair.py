#%%
"""Undo repair_p5b_remap.py — put the cache back to its honest, unrepaired state.

The repair sent every idea whose text a split did not list to the FIRST child of
that split. That rule was invented, not derived: 214 ideas from one attribute were
moved as a block on no evidence. An idea with no attribute is visible; an idea with
an arbitrary attribute is not. The second is worse.

This rebuilds the per-idea attributes from `raw_attribute_assignments` (the P6 state,
still in the cache) plus the P5b log, applying ONLY what the original run applied:
  - merge / widen / keep : source -> result
  - split                : only the response texts the model actually listed
  - misfit move          : only the texts listed, target resolved against the structure
Everything else keeps its pre-P5b name and is therefore visibly outside the taxonomy.

NO LLM CALLS. Writes taxonomy_exp / taxonomy_classified_exp only.

Usage:
    cd src && python -m pipeline.step_4_classifier_experiment.revert_p5b_repair
"""
import json
import sys
from collections import Counter
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import models
from test_data import TEST_DATA
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from identity import ensure_taxonomy_ids, restamp_assignment_ids

norm = lambda s: (s or "").strip().lower()


def main():
    vk = generate_enhanced_variable_key([TEST_DATA.var_name], False, TEST_DATA.sample_size)
    cm = CacheManager()
    tax = cm.load_metadata_from_cache(filename=TEST_DATA.filename, step="taxonomy_exp",
                                      variable_key=vk, model_cls=models.TaxonomyResultsCache)
    resp = cm.load_from_cache(TEST_DATA.filename, "taxonomy_classified_exp", vk,
                              models.TaxonomyClassifiedModel)
    log_path = (Path(__file__).resolve().parents[3] / "exports" / "experiment_logs" /
                f"{Path(TEST_DATA.filename).stem}_{vk}_p5b_log.json")
    actions = json.load(open(log_path, encoding="utf-8"))["actions"]

    home = {}
    for dom, res in tax.partition_results.items():
        for fac, lst in res.attributes.items():
            for a in lst:
                home[a["attribute_name"]] = (dom, fac)
    struct = set(home)

    remap, splits, moves = {}, {}, {}
    for a in actions:
        if a["action"] in ("merge", "widen", "keep"):
            for s in a.get("sources") or []:
                remap[(a["domain"], a["facet"], s)] = a["result"]
        elif a["action"] == "split":
            for s in a.get("sources") or []:
                for t in a.get("texts") or []:
                    splits[(a["domain"], a["facet"], s, norm(t))] = a["into"]
        elif a["action"] == "misfit_move" and a.get("target"):
            for t in a.get("texts") or []:
                moves[(a["domain"], a["facet"], a["from_attribute"], norm(t))] = a["target"]

    text_of = {i.idea_id: norm(getattr(i, "instance", ""))
               for r in (resp or []) if r.response_ideas for i in r.response_ideas}

    # rebuild from the P6 state, applying only what the original run applied
    rebuilt = {}
    for dom, res in tax.partition_results.items():
        facet_of = res.facet_assignments or {}
        for iid, raw_attr in (res.raw_attribute_assignments or {}).items():
            fac = facet_of.get(iid)
            txt = text_of.get(iid, "")
            tgt = moves.get((dom, fac, raw_attr, txt))
            if tgt and tgt in struct:
                rebuilt[iid] = tgt
                continue
            child = splits.get((dom, fac, raw_attr, txt))
            if child:
                rebuilt[iid] = child
                continue
            rebuilt[iid] = remap.get((dom, fac, raw_attr), raw_attr)

    for dom, res in tax.partition_results.items():
        for iid in list((res.attribute_assignments or {})):
            if iid in rebuilt:
                res.attribute_assignments[iid] = rebuilt[iid]

    changed = 0
    for r in (resp or []):
        for idea in (r.response_ideas or []):
            new = rebuilt.get(idea.idea_id)
            if new and new != idea.attribute:
                idea.attribute = new
                changed += 1
            place = home.get(idea.attribute)
            if place:
                idea.domain, idea.facet, idea.partition_name = place[0], place[1], place[0]

    ideas = [i for r in (resp or []) if r.response_ideas for i in r.response_ideas]
    orphans = [i for i in ideas if (i.attribute or "") not in struct]
    print(f"ideeën aangepast          : {changed}")
    print(f"zonder geldig attribuut   : {len(orphans)}  ({100*len(orphans)/len(ideas):.1f}%)")
    print("\ngrootste groepen zonder attribuut:")
    for a, c in Counter(i.attribute for i in orphans).most_common(8):
        print(f"   {c:4d}  {a!r}")

    ensure_taxonomy_ids(tax)
    restamp_assignment_ids(resp, tax)
    cm.save_metadata_to_cache(metadata=tax, filename=TEST_DATA.filename,
                              step="taxonomy_exp", variable_key=vk)
    cm.save_to_cache(data=resp, filename=TEST_DATA.filename,
                     step="taxonomy_classified_exp", variable_key=vk)
    print("\nteruggezet — de niet-passende ideeën zijn weer zichtbaar")


if __name__ == "__main__":
    main()

# %%
