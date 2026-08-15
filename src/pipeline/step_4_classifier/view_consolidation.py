#%%
"""
View the structure decisions of the last step-4 run, human-readable.

Renders the action log (exports/experiment_logs/<dataset>_<vk>_step4_log.json)
per domain: first what consolidation settled, then what assignment had to drain,
then what refinement judged on real counts. Cross-domain and run totals follow,
since neither is domain-scoped. Read-only — the log is the source, nothing is
recomputed.

Every action name the classifier writes is routed below. An action this renderer
does not know lands in UNKNOWN and is printed as such: the writer's names have
drifted before, and a renderer that silently drops what it does not recognise
reports "no decisions" for a run full of them.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_consolidation
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key
from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

RULE = "─" * 80

# Action names as classifier.py writes them, grouped by the phase that emits
# them. Domain-scoped phases carry a "domain" field; the last two do not.
CONSOLIDATION_ACTIONS = {
    "facet_exact_dedup", "chunk_consolidation", "facet_kept_unclaimed",
    "consolidation_failed", "consolidation_rounds_exhausted",
}
ASSIGNMENT_ACTIONS = {"assignment_failed_to_drain"}
REFINEMENT_ACTIONS = {
    "keep", "widen", "merge", "split", "misfit_move", "misfit_out",
    "unknown_source_name", "unroutable_claim",
    "attribute_kept_unclaimed_in_refinement", "refinement_failed",
}
CROSS_DOMAIN_ACTIONS = {
    "cross_domain_merge", "cross_domain_failed",
    "attribute_kept_unclaimed_cross_domain",
}
TOTAL_ACTIONS = {"ideas_moved"}


def _sources(e):
    """Sources as ' ← [a, b]', empty when a decision only names itself."""
    sources = e.get("sources") or []
    result = e.get("result")
    if not sources or sources == [result]:
        return ""
    return "  ← " + ", ".join(f"'{s}'" for s in sources)


def _print_consolidation(e):
    a = e["action"]
    if a == "facet_exact_dedup":
        print(f"    DEDUP  {e.get('before', 0)} → {e.get('after', 0)} facetten "
              f"(byte-identieke chunk-herhalingen)")
    elif a == "chunk_consolidation":
        print(f"    SAMEN  facetten {e.get('facets_before', 0)} → {e.get('facets_after', 0)}"
              f"  ·  attributen {e.get('attributes_before', 0)} → {e.get('attributes_after', 0)}")
    elif a == "facet_kept_unclaimed":
        print(f"    BLEEF  '{e.get('facet')}'  (door geen enkele output geclaimd)")
    elif a == "consolidation_failed":
        print(f"    FAAL   {e.get('note', 'geen resultaat')} "
              f"({len(e.get('candidates') or [])} kandidaten)")
    elif a == "consolidation_rounds_exhausted":
        print(f"    RONDEN op na {e.get('rounds', 0)} rondes — "
              f"{e.get('remaining', 0)} kandidaten niet samengevoegd")


def _print_assignment(e):
    print(f"    VANGNET {e.get('n_ideas', 0)} ideeën zonder antwoord "
          f"→ '{e.get('target')}'")


def _print_refinement(e):
    a = e["action"]
    if a in ("keep", "widen", "merge"):
        print(f"    {a.upper():6s} '{e.get('result')}'{_sources(e)}")
    elif a == "split":
        srcs = ", ".join(f"'{s}'" for s in (e.get("sources") or []))
        print(f"    SPLIT  → '{e.get('into')}'  uit: {srcs}  "
              f"({e.get('n_texts', 0)} teksten)")
    elif a == "misfit_move":
        print(f"    MOVE   {e.get('n_texts', 0)} teksten → '{e.get('target')}'")
    elif a == "misfit_out":
        print(f"    OUT    {e.get('n_texts', 0)} teksten "
              f"(geen bestemming: blijven staan waar ze zitten)")
    elif a == "unknown_source_name":
        print(f"    ?BRON  {e.get('sources')} — {e.get('note', '')}")
    elif a == "unroutable_claim":
        print(f"    !CLAIM {e.get('sources')} — {e.get('note', '')}")
    elif a == "attribute_kept_unclaimed_in_refinement":
        print(f"    BLEEF  '{e.get('attribute')}'  (door geen enkele output geclaimd)")
    elif a == "refinement_failed":
        print(f"    FAAL   {e.get('note', 'geen resultaat')} "
              f"({e.get('attributes_before', 0)} attributen onaangeroerd)")


def main():
    stem = Path(FILENAME).stem
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    log_path = (project_root / "exports" / "experiment_logs"
                / f"{stem}_{variable_key}_step4_log.json")
    if not log_path.exists():
        print(f"Geen actielog op: {log_path}")
        return
    data = json.loads(log_path.read_text(encoding="utf-8"))
    actions = data.get("actions", [])

    print("=" * 80)
    print("STRUCTUURBESLISSINGEN STEP 4")
    print("=" * 80)
    print(f"Dataset:  {data.get('dataset', FILENAME)}")
    print(f"Variable: {data.get('variable_key', variable_key)}")
    print(f"Log:      {log_path.name} ({len(actions)} acties)")

    # -- route every action, keeping what we do not recognise ------------------
    consolidation = defaultdict(list)
    assignment = defaultdict(list)
    refinement = defaultdict(list)
    cross_domain = []
    totals = []
    unknown = []
    for e in actions:
        a = e.get("action", "")
        if a in CONSOLIDATION_ACTIONS:
            consolidation[e.get("domain", "?")].append(e)
        elif a in ASSIGNMENT_ACTIONS:
            assignment[e.get("domain", "?")].append(e)
        elif a in REFINEMENT_ACTIONS:
            refinement[e.get("domain", "?")].append(e)
        elif a in CROSS_DOMAIN_ACTIONS:
            cross_domain.append(e)
        elif a in TOTAL_ACTIONS:
            totals.append(e)
        else:
            unknown.append(e)

    domains = sorted(set(consolidation) | set(assignment) | set(refinement))
    for dom in domains:
        print(f"\n{RULE}\nDOMEIN: {dom}\n{RULE}")

        if consolidation.get(dom):
            print("\n  consolidatie — de inventaris vastzetten:")
            for e in consolidation[dom]:
                _print_consolidation(e)

        if assignment.get(dom):
            print("\n  toewijzing:")
            for e in assignment[dom]:
                _print_assignment(e)

        if refinement.get(dom):
            print("\n  naslijpen — oordeel op echte aantallen:")
            for e in refinement[dom]:
                _print_refinement(e)

    if cross_domain:
        print(f"\n{RULE}\nCROSS-DOMEIN\n{RULE}")
        kept = [e for e in cross_domain
                if e["action"] == "attribute_kept_unclaimed_cross_domain"]
        for e in cross_domain:
            a = e["action"]
            if a == "cross_domain_merge":
                print(f"    MERGE  '{e.get('result')}'{_sources(e)}")
                print(f"           home: {e.get('home')}")
            elif a == "cross_domain_failed":
                print(f"    FAAL   {e.get('note', 'geen resultaat')}")
        if kept:
            print(f"\n    {len(kept)} attributen door geen enkele groep geclaimd "
                  f"— bleven staan waar ze stonden")

    if totals:
        print(f"\n{RULE}\nTOTALEN\n{RULE}")
        for e in totals:
            if e["action"] == "ideas_moved":
                print(f"  {e.get('n_ideas', 0)} ideeën verplaatst door naslijpen")

    if unknown:
        print(f"\n{RULE}\nONBEKENDE ACTIES\n{RULE}")
        print("  Deze viewer kent onderstaande namen niet. De schrijfkant in")
        print("  classifier.py is veranderd zonder dat dit bestand meeging.")
        for name, n in sorted(Counter(e.get("action", "") for e in unknown).items()):
            example = next(e for e in unknown if e.get("action") == name)
            fields = ", ".join(k for k in example if k != "action")
            print(f"    {name}  (n={n})  velden: {fields}")


if __name__ == "__main__":
    main()
