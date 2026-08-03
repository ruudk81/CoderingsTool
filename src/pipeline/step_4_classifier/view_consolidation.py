#%%
"""
View the consolidation decisions of the last step-4 run, human-readable.

Renders the action log (exports/experiment_logs/<dataset>_<vk>_p9_log.json)
per domain: first the P5 facet decisions (grouped by axis), then the P8
attribute decisions (grouped by facet). One line per decision, with the
response texts that moved. Read-only — the log is the source, nothing is
recomputed.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_consolidation
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

MAX_TEXTS = 6          # response texts shown per decision
RULE = "─" * 80

# Facet-level (P5) and attribute-level (P8) action names as the log writes them.
FACET_ACTIONS = {
    "facet_keep", "facet_widen", "facet_merge", "facet_split",
    "facet_misfit_move", "facet_misfit_out", "facet_kept_unclaimed",
    "unknown_source_facet", "unroutable_facet_claim",
    "facet_consolidation_failed",
}
ATTR_ACTIONS = {
    "keep", "widen", "merge", "split", "misfit_move", "misfit_out",
    "unknown_source_name", "unroutable_claim", "failed",
}


def _texts(entry, prefix="        "):
    texts = entry.get("texts") or []
    shown = [f'"{t}"' for t in texts[:MAX_TEXTS]]
    more = f" … +{len(texts) - MAX_TEXTS} meer" if len(texts) > MAX_TEXTS else ""
    return f"{prefix}{' · '.join(shown)}{more}" if shown else None


def _print_structure_line(e, kind):
    """One structure decision: keep/widen/merge/split."""
    action = e["action"].replace("facet_", "")
    result = e.get("result") or e.get("into") or ""
    sources = e.get("sources") or []
    if action == "split":
        print(f"    SPLIT  → '{result}'  uit: {sources}  ({e.get('n_texts', 0)} teksten)")
        line = _texts(e)
        if line:
            print(line)
    elif sources == [result] or not sources:
        print(f"    {action.upper():6s} '{result}'")
    else:
        print(f"    {action.upper():6s} '{result}'  ← {sources}")


def _print_misfit_line(e):
    verdict = "OUT " if e["action"].endswith("out") else "MOVE"
    src = e.get("from_facet") or e.get("from_attribute") or "?"
    tgt = e.get("target") or e.get("target_attribute") or ""
    arrow = f" → '{tgt}'" if tgt else " → (geen doel: contentloos, blijft staan)"
    print(f"    {verdict}   {e.get('n_texts', 0)} teksten uit '{src}'{arrow}")
    line = _texts(e)
    if line:
        print(line)
    reason = e.get("reason")
    if reason:
        print(f"        reden: {reason}")


def _print_guard_line(e):
    action = e["action"]
    if action in ("facet_kept_unclaimed",):
        print(f"    BLEEF  '{e.get('facet')}'  (door geen enkele output geclaimd)")
    elif action in ("unknown_source_facet", "unknown_source_name"):
        print(f"    ?BRON  onbekende bronnen genoemd: {e.get('sources')}")
    elif action in ("unroutable_facet_claim", "unroutable_claim"):
        print(f"    !CLAIM {e.get('sources')} door meerdere outputs geclaimd "
              f"zonder teksten — ideeën bleven op de bron")
    elif action in ("facet_consolidation_failed", "failed"):
        scope = e.get("axis") or e.get("facet") or ""
        print(f"    FAAL   {scope}: {e.get('note', 'geen resultaat')}")


def main():
    stem = Path(FILENAME).stem
    variable_key = f"{VAR_NAME}_{SAMPLE_SIZE}"
    log_path = project_root / "exports" / "experiment_logs" / f"{stem}_{variable_key}_p9_log.json"
    if not log_path.exists():
        print(f"Geen actielog op: {log_path}")
        return
    data = json.loads(log_path.read_text(encoding="utf-8"))
    actions = data.get("actions", [])

    print("=" * 80)
    print("CONSOLIDATIEBESLISSINGEN (P5 facetten, P8 attributen)")
    print("=" * 80)
    print(f"Dataset:  {data.get('dataset', FILENAME)}")
    print(f"Variable: {data.get('variable_key', variable_key)}")
    print(f"Log:      {log_path.name} ({len(actions)} acties)")

    # -- bucket by domain --------------------------------------------------
    facet_by_domain = defaultdict(list)
    attr_by_domain = defaultdict(list)
    totals = []
    for e in actions:
        a = e.get("action", "")
        if a in ("_facet_totals", "_totals", "orphaned_facet_assignment",
                 "orphaned_assignment"):
            totals.append(e)
        elif a in FACET_ACTIONS:
            facet_by_domain[e.get("domain", "?")].append(e)
        elif a in ATTR_ACTIONS:
            attr_by_domain[e.get("domain", "?")].append(e)
        # axis_system_* entries are P1 provenance; view_taxonomy covers them

    domains = sorted(set(facet_by_domain) | set(attr_by_domain))
    for dom in domains:
        print(f"\n{RULE}\nDOMEIN: {dom}\n{RULE}")

        f_entries = facet_by_domain.get(dom, [])
        if f_entries:
            print("\n  P5 — facetconsolidatie:")
            by_axis = defaultdict(list)
            for e in f_entries:
                by_axis[e.get("axis", "")].append(e)
            for axis in sorted(by_axis):
                if axis:
                    print(f"\n  as «{axis}»")
                for e in by_axis[axis]:
                    a = e["action"]
                    if a in ("facet_keep", "facet_widen", "facet_merge", "facet_split"):
                        _print_structure_line(e, "facet")
                    elif a.startswith("facet_misfit"):
                        _print_misfit_line(e)
                    else:
                        _print_guard_line(e)

        a_entries = attr_by_domain.get(dom, [])
        if a_entries:
            print("\n  P8 — attribuutconsolidatie:")
            by_facet = defaultdict(list)
            for e in a_entries:
                by_facet[e.get("facet", "")].append(e)
            for facet in sorted(by_facet):
                print(f"\n  facet «{facet}»")
                for e in by_facet[facet]:
                    a = e["action"]
                    if a in ("keep", "widen", "merge", "split"):
                        _print_structure_line(e, "attr")
                    elif a.startswith("misfit"):
                        _print_misfit_line(e)
                    else:
                        _print_guard_line(e)

    if totals:
        print(f"\n{RULE}\nTOTALEN\n{RULE}")
        for e in totals:
            a = e["action"]
            if a in ("_facet_totals", "_totals"):
                level = "facetten (P5)" if a == "_facet_totals" else "attributen (P8)"
                print(f"  {level}: {e.get('ideas_remapped', 0)} geremapt, "
                      f"{e.get('ideas_split', 0)} gesplitst, "
                      f"{e.get('ideas_moved', 0)} verplaatst, "
                      f"{e.get('flagged_contentless_left_in_place', 0)} contentloos blijven staan, "
                      f"{e.get('moves_with_unresolvable_target', 0)} moves zonder oplosbaar doel")
                unresolved = e.get("unresolved_target_names") or {}
                if unresolved:
                    print(f"      onoplosbare doelen: {unresolved}")
            else:
                level = "facet" if "facet" in a else "attribuut"
                print(f"  self-check ({level}): {e.get('ideas_affected', 0)} ideeën wezen naar "
                      f"verdwenen nodes — {e.get('restored_nodes', 0)} node(s) teruggezet: "
                      f"{e.get('facets') or e.get('attributes')}")


if __name__ == "__main__":
    main()
