#%%

"""
View codebook: codes + attributes with assignment counts.

Read-only readout of the step-5 codebook annotated with the actual step-6
assignment counts. Per code (and per attribute within it):
  - n ideas and % of ASSIGNED ideas
  - % of RESPONSES — unique non-filtered respondents who mention this code/attribute
  - valence balance: x% (+) / y% (-), where (+) = positive+neutral, (-) = negative
The smallest attributes per code (together ≤ OVERIG_TAIL_PCT of the code's ideas,
plus any defined-but-unused ones) are folded into one "overig (k attrs)" row.
`__UNASSIGNED__` is excluded from the % base and reported separately.

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

OVERIG_TAIL_PCT = 0.10        # smallest attributes summing to ≤ this share of a code → "overig"
SAVE_CSV = True

_UNASSIGNED = "__UNASSIGNED__"
_NO_ATTR = "(geen attribuut)"
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


def _is_neg(valence: str) -> bool:
    return (valence or "").strip().lower() in _NEG_VALENCES


def build_rows(responses: List, codebook) -> tuple:
    """Return (rows, n_assigned, n_responses, n_unassigned).

    rows: ordered dicts {level, code, attribute, valence, n, pct_ideas, pct_resp,
    pct_pos, pct_neg}. pct_pos = positive+neutral share, pct_neg = negative share.
    """
    code_n: Counter = Counter()
    code_neg: Counter = Counter()
    code_resp: Dict[str, set] = defaultdict(set)
    cell_n: Dict[str, Counter] = defaultdict(Counter)
    cell_neg: Dict[str, Counter] = defaultdict(Counter)
    cell_resp: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    resp_with_ideas: set = set()
    n_unassigned = 0

    for resp in responses:
        rid = str(resp.respondent_id)
        for idea in (resp.response_ideas or []):
            resp_with_ideas.add(rid)
            code = (idea.assigned_code or "").strip()
            if not code or code == _UNASSIGNED:
                n_unassigned += 1
                continue
            attr = (idea.assigned_attribute or "").strip() or _NO_ATTR
            neg = _is_neg(idea.valence)
            code_n[code] += 1
            code_neg[code] += neg
            code_resp[code].add(rid)
            cell_n[code][attr] += 1
            cell_neg[code][attr] += neg
            cell_resp[code][attr].add(rid)

    n_assigned = sum(code_n.values())
    n_responses = len(resp_with_ideas)
    pct_i = lambda n: (100.0 * n / n_assigned) if n_assigned else 0.0
    pct_r = lambda k: (100.0 * k / n_responses) if n_responses else 0.0

    def balance(n, neg):
        return (100.0 * (n - neg) / n, 100.0 * neg / n) if n else (0.0, 0.0)

    # Codebook lookup: code_name → (valence, source_attributes)
    cb = {}
    for c in codebook.raw_codes:
        d = c if isinstance(c, dict) else c.__dict__
        cb[d["code_name"]] = (d.get("valence", ""), d.get("source_attributes", []) or [])

    all_codes = set(cb) | set(code_n)
    ordered = sorted(all_codes, key=lambda c: (-code_n.get(c, 0), c.lower()))

    rows = []
    for code in ordered:
        cn = code_n.get(code, 0)
        valence, source_attrs = cb.get(code, ("", []))
        cpos, cneg = balance(cn, code_neg.get(code, 0))
        rows.append({"level": "code", "code": code, "attribute": "",
                     "valence": _vsign(valence), "n": cn,
                     "pct_ideas": pct_i(cn), "pct_resp": pct_r(len(code_resp.get(code, ()))),
                     "pct_pos": cpos, "pct_neg": cneg})

        # All attributes for this code: assigned (n>0) + defined-but-unused (n=0)
        attrs = dict(cell_n.get(code, {}))
        for a in source_attrs:
            attrs.setdefault(a, 0)

        # Bottom-tail merge: smallest attributes summing to ≤ OVERIG_TAIL_PCT of cn,
        # plus all unused (n=0), folded into one "overig" row.
        threshold = OVERIG_TAIL_PCT * cn
        ascending = sorted(attrs.items(), key=lambda kv: kv[1])
        tail, cum = [], 0
        for a, an in ascending:
            if an == 0 or cum + an <= threshold:
                tail.append(a)
                cum += an
            else:
                break
        tail_set = set(tail)
        kept = sorted(((a, an) for a, an in attrs.items() if a not in tail_set),
                      key=lambda kv: -kv[1])

        for a, an in kept:
            apos, aneg = balance(an, cell_neg[code].get(a, 0))
            rows.append({"level": "attr", "code": code, "attribute": a, "valence": "",
                         "n": an, "pct_ideas": pct_i(an),
                         "pct_resp": pct_r(len(cell_resp[code].get(a, ()))),
                         "pct_pos": apos, "pct_neg": aneg})

        if len(tail_set) >= 2:
            tn = sum(attrs[a] for a in tail_set)
            tneg = sum(cell_neg[code].get(a, 0) for a in tail_set)
            tresp = set().union(*(cell_resp[code].get(a, set()) for a in tail_set)) if tail_set else set()
            tpos, tnegpct = balance(tn, tneg)
            rows.append({"level": "attr", "code": code,
                         "attribute": f"overig ({len(tail_set)} attrs)", "valence": "",
                         "n": tn, "pct_ideas": pct_i(tn), "pct_resp": pct_r(len(tresp)),
                         "pct_pos": tpos, "pct_neg": tnegpct})
        elif tail_set:  # single tail attr → show it individually
            a = next(iter(tail_set))
            an = attrs[a]
            apos, aneg = balance(an, cell_neg[code].get(a, 0))
            rows.append({"level": "attr", "code": code, "attribute": a, "valence": "",
                         "n": an, "pct_ideas": pct_i(an),
                         "pct_resp": pct_r(len(cell_resp[code].get(a, ()))),
                         "pct_pos": apos, "pct_neg": aneg})

    return rows, n_assigned, n_responses, n_unassigned


# =============================================================================
# DISPLAY
# =============================================================================

def _bal(r) -> str:
    if not r["n"]:
        return ""
    return f"{r['pct_pos']:.0f}% (+) / {r['pct_neg']:.0f}% (-)"


def print_readout(rows, n_assigned, n_responses, n_unassigned):
    n_total = n_assigned + n_unassigned
    print(f"\n{'=' * 84}")
    print(f"CODEBOOK — {FILENAME}")
    print(f"{VARIABLE}  |  {n_assigned} assigned ideas (of {n_total}; {n_unassigned} unassigned)"
          f"  |  {n_responses} responses")
    print(f"{'=' * 84}")
    print(f"{'code / attribute':46}{'n':>5}{'%idea':>7}{'%resp':>7}   {'balans (+/-)'}")
    print(f"{'-' * 84}")

    for r in rows:
        if r["level"] == "code":
            label = f"[{r['valence']}] {r['code']}"
            print(f"\n{label:46}{r['n']:>5}{r['pct_ideas']:>6.1f}%{r['pct_resp']:>6.1f}%   {_bal(r)}")
        else:
            label = f"    {r['attribute']}"
            tag = "" if r["n"] else "  (unused)"
            print(f"{label:46}{r['n']:>5}{r['pct_ideas']:>6.1f}%{r['pct_resp']:>6.1f}%   {_bal(r)}{tag}")

    print(f"\n{'-' * 84}")
    print(f"{'TOTAAL (som codes)':46}{n_assigned:>5}{100.0:>6.1f}%")
    if n_unassigned:
        upct = 100.0 * n_unassigned / n_total if n_total else 0.0
        print(f"{'__UNASSIGNED__ (excl. van %-basis)':46}{n_unassigned:>5}{upct:>6.1f}%  of {n_total} ideas")


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_csv(rows, n_assigned, n_responses, n_unassigned):
    exports_dir = project_root / "exports"
    exports_dir.mkdir(exist_ok=True)
    base = Path(FILENAME).stem.replace(" ", "_")
    csv_path = exports_dir / f"codebook_{base}_{VARIABLE}_{SAMPLE_SIZE}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["level", "code", "attribute", "valence", "n",
                    "pct_of_assigned", "pct_of_responses", "pct_pos_neutral", "pct_neg"])
        for r in rows:
            w.writerow([r["level"], r["code"], r["attribute"], r["valence"], r["n"],
                        f"{r['pct_ideas']:.1f}", f"{r['pct_resp']:.1f}",
                        f"{r['pct_pos']:.1f}" if r["n"] else "",
                        f"{r['pct_neg']:.1f}" if r["n"] else ""])
        w.writerow(["total", "TOTAAL", "", "", n_assigned, "100.0", "", "", ""])
        w.writerow(["unassigned", _UNASSIGNED, "", "", n_unassigned, "", "", "", ""])
        w.writerow(["meta", "responses", "", "", n_responses, "", "", "", ""])
    print(f"\nCSV saved to: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    responses, codebook = load_data()
    rows, n_assigned, n_responses, n_unassigned = build_rows(responses, codebook)
    print_readout(rows, n_assigned, n_responses, n_unassigned)
    if SAVE_CSV:
        save_csv(rows, n_assigned, n_responses, n_unassigned)

# %%
