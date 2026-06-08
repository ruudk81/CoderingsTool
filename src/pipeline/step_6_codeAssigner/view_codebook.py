#%%

"""
View codebook: codes/domains + attributes with assignment counts.

Read-only readout of the step-6 assignments in three lenses (each printed + saved
to its own CSV):
  1. codes only            (suffix _codes)
  2. codes + attributes     (suffix _codes_attrs)   — the codebook view
  3. domains + attributes   (suffix _domains_attrs)  — the taxonomy view

Per group (code or domain) and per attribute within it:
  - n ideas and % of the rows' idea base
  - % of RESPONSES — unique non-filtered respondents who mention it
  - valence balance: x% (+) / y% (-), where (+) = positive+neutral, (-) = negative
The smallest attributes per group (together ≤ OVERIG_TAIL_PCT, plus any unused
codebook attributes) fold into one "overig (k attrs)" row.

Code views exclude the __UNASSIGNED__ sentinel from the % base (reported
separately); the domain view covers every idea (each idea has a domain).

Usage:
    cd src && python -m pipeline.step_6_codeAssigner.view_codebook
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_6_codeAssigner.models_codeAssigner import CodeAssignedModel
from models import CodingResultsCache

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

OVERIG_TAIL_PCT = 0.10        # smallest attributes summing to ≤ this share of a group → "overig"
SAVE_CSV = True

# (title, group_by, show_attrs, csv_suffix)
VERSIONS = [
    ("CODES ONLY",          "code",   False, "codes"),
    ("CODES + ATTRIBUTES",   "code",   True,  "codes_attrs"),
    ("DOMAINS + ATTRIBUTES", "domain", True,  "domains_attrs"),
]

_UNASSIGNED = "__UNASSIGNED__"
_NO_ATTR = "(geen attribuut)"
_NO_GROUP = "(geen)"
_NEG_VALENCES = {"-", "-1", "neg", "negative"}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load step-6 response models + the step-5 codebook from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cm = CacheManager()
    results = cm.load_from_cache(FILENAME, "taxonomy_codes", variable_key, CodeAssignedModel)
    if not results:
        raise FileNotFoundError("No taxonomy_codes cache — run step 6 first.")
    codebook = cm.load_metadata_from_cache(FILENAME, "mece_codes", variable_key, CodingResultsCache)
    if not codebook:
        raise FileNotFoundError("No mece_codes cache — run step 5 first.")
    return results, codebook


# =============================================================================
# ANALYSIS
# =============================================================================

def _vsign(valence: str) -> str:
    v = (valence or "").lower()
    if v.startswith("pos") or v == "+":
        return "+"
    if v.startswith("neg") or v == "-":
        return "-"
    return "~"


def _derived_sign(pct_neg: float) -> str:
    if pct_neg >= 55:
        return "-"
    if pct_neg <= 45:
        return "+"
    return "~"


def _is_neg(valence: str) -> bool:
    return (valence or "").strip().lower() in _NEG_VALENCES


def build_rows(responses: List, codebook, group_by: str, show_attrs: bool) -> tuple:
    """Return (rows, base_n, n_responses, n_unassigned) grouped by code or domain."""
    is_code = (group_by == "code")
    group_n: Counter = Counter()
    group_neg: Counter = Counter()
    group_resp: Dict[str, set] = defaultdict(set)
    cell_n: Dict[str, Counter] = defaultdict(Counter)
    cell_neg: Dict[str, Counter] = defaultdict(Counter)
    cell_resp: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    resp_with_ideas: set = set()
    n_unassigned = 0

    for resp in responses:
        rid = str(resp.respondent_id)
        for idea in (resp.response_ideas or []):
            resp_with_ideas.add(rid)
            if is_code:
                key = (idea.assigned_code or "").strip()
                if not key or key == _UNASSIGNED:
                    n_unassigned += 1
                    continue
            else:
                key = (getattr(idea, "partition_name", "") or idea.domain or "").strip() or _NO_GROUP
            attr = (idea.assigned_attribute or "").strip() or _NO_ATTR
            neg = _is_neg(idea.valence)
            group_n[key] += 1
            group_neg[key] += neg
            group_resp[key].add(rid)
            cell_n[key][attr] += 1
            cell_neg[key][attr] += neg
            cell_resp[key][attr].add(rid)

    base_n = sum(group_n.values())
    n_responses = len(resp_with_ideas)
    pct_i = lambda n: (100.0 * n / base_n) if base_n else 0.0
    pct_r = lambda k: (100.0 * k / n_responses) if n_responses else 0.0

    def balance(n, neg):
        return (100.0 * (n - neg) / n, 100.0 * neg / n) if n else (0.0, 0.0)

    # Codebook lookup (code view only): code_name → (valence, source_attributes)
    cb = {}
    if is_code:
        for c in codebook.raw_codes:
            d = c if isinstance(c, dict) else c.__dict__
            cb[d["code_name"]] = (d.get("valence", ""), d.get("source_attributes", []) or [])

    keys = set(cb) | set(group_n) if is_code else set(group_n)
    ordered = sorted(keys, key=lambda k: (-group_n.get(k, 0), k.lower()))

    rows = []
    for key in ordered:
        gn = group_n.get(key, 0)
        gpos, gneg = balance(gn, group_neg.get(key, 0))
        if is_code:
            valence = _vsign(cb.get(key, ("", []))[0])
            source_attrs = cb.get(key, ("", []))[1]
        else:
            valence = _derived_sign(gneg)
            source_attrs = []
        rows.append({"level": "group", "group": key, "attribute": "", "valence": valence,
                     "n": gn, "pct_ideas": pct_i(gn), "pct_resp": pct_r(len(group_resp.get(key, ()))),
                     "pct_pos": gpos, "pct_neg": gneg})

        if not show_attrs:
            continue

        attrs = dict(cell_n.get(key, {}))
        for a in source_attrs:
            attrs.setdefault(a, 0)

        # Bottom-tail merge: smallest attributes summing to ≤ OVERIG_TAIL_PCT of gn,
        # plus all unused (n=0), folded into one "overig" row.
        threshold = OVERIG_TAIL_PCT * gn
        tail, cum = [], 0
        for a, an in sorted(attrs.items(), key=lambda kv: kv[1]):
            if an == 0 or cum + an <= threshold:
                tail.append(a)
                cum += an
            else:
                break
        tail_set = set(tail)

        def attr_row(a, an):
            apos, aneg = balance(an, cell_neg[key].get(a, 0))
            return {"level": "attr", "group": key, "attribute": a, "valence": "",
                    "n": an, "pct_ideas": pct_i(an),
                    "pct_resp": pct_r(len(cell_resp[key].get(a, ()))),
                    "pct_pos": apos, "pct_neg": aneg}

        for a, an in sorted(((a, an) for a, an in attrs.items() if a not in tail_set),
                            key=lambda kv: -kv[1]):
            rows.append(attr_row(a, an))

        if len(tail_set) >= 2:
            tn = sum(attrs[a] for a in tail_set)
            tneg = sum(cell_neg[key].get(a, 0) for a in tail_set)
            tresp = set().union(*(cell_resp[key].get(a, set()) for a in tail_set))
            tpos, tnegpct = balance(tn, tneg)
            rows.append({"level": "attr", "group": key, "attribute": f"overig ({len(tail_set)} attrs)",
                         "valence": "", "n": tn, "pct_ideas": pct_i(tn), "pct_resp": pct_r(len(tresp)),
                         "pct_pos": tpos, "pct_neg": tnegpct})
        elif tail_set:
            a = next(iter(tail_set))
            rows.append(attr_row(a, attrs[a]))

    return rows, base_n, n_responses, n_unassigned


# =============================================================================
# DISPLAY
# =============================================================================

def _bal(r) -> str:
    return f"{r['pct_pos']:.0f}% (+) / {r['pct_neg']:.0f}% (-)" if r["n"] else ""


def print_readout(title, group_label, rows, base_n, n_responses, n_unassigned):
    print(f"\n\n{'=' * 84}")
    print(f"[{title}]  {FILENAME}")
    print(f"{VARIABLE}  |  {base_n} ideas (base)"
          + (f"; {n_unassigned} unassigned" if n_unassigned else "")
          + f"  |  {n_responses} responses")
    print(f"{'=' * 84}")
    print(f"{group_label + ' / attribute':46}{'n':>5}{'%idea':>7}{'%resp':>7}   balans (+/-)")
    print(f"{'-' * 84}")
    for r in rows:
        if r["level"] == "group":
            print(f"\n{'[' + r['valence'] + '] ' + r['group']:46}"
                  f"{r['n']:>5}{r['pct_ideas']:>6.1f}%{r['pct_resp']:>6.1f}%   {_bal(r)}")
        else:
            tag = "" if r["n"] else "  (unused)"
            print(f"{'    ' + r['attribute']:46}"
                  f"{r['n']:>5}{r['pct_ideas']:>6.1f}%{r['pct_resp']:>6.1f}%   {_bal(r)}{tag}")
    print(f"\n{'-' * 84}")
    print(f"{'TOTAAL':46}{base_n:>5}{100.0:>6.1f}%")
    if n_unassigned:
        print(f"{'__UNASSIGNED__ (excl. van %-basis)':46}{n_unassigned:>5}")


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_csv(suffix, group_label, rows, base_n, n_responses, n_unassigned):
    exports_dir = project_root / "exports"
    exports_dir.mkdir(exist_ok=True)
    base = Path(FILENAME).stem.replace(" ", "_")
    csv_path = exports_dir / f"codebook_{base}_{VARIABLE}_{SAMPLE_SIZE}_{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["level", group_label, "attribute", "valence", "n",
                    "pct_of_base", "pct_of_responses", "pct_pos_neutral", "pct_neg"])
        for r in rows:
            w.writerow([r["level"], r["group"], r["attribute"], r["valence"], r["n"],
                        f"{r['pct_ideas']:.1f}", f"{r['pct_resp']:.1f}",
                        f"{r['pct_pos']:.1f}" if r["n"] else "",
                        f"{r['pct_neg']:.1f}" if r["n"] else ""])
        w.writerow(["total", "TOTAAL", "", "", base_n, "100.0", "", "", ""])
        w.writerow(["unassigned", _UNASSIGNED, "", "", n_unassigned, "", "", "", ""])
        w.writerow(["meta", "responses", "", "", n_responses, "", "", "", ""])
    print(f"  CSV → {csv_path.name}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    responses, codebook = load_data()
    for title, group_by, show_attrs, suffix in VERSIONS:
        group_label = "code" if group_by == "code" else "domain"
        rows, base_n, n_responses, n_unassigned = build_rows(
            responses, codebook, group_by, show_attrs)
        print_readout(title, group_label, rows, base_n, n_responses, n_unassigned)
        if SAVE_CSV:
            save_csv(suffix, group_label, rows, base_n, n_responses, n_unassigned)

# %%
