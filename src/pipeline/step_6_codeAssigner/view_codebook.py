#%%

"""
View codebook: codes/domains + attributes (+ codes) with assignment counts.

Read-only readout of the step-6 assignments in four lenses (each printed + saved
to its own CSV):
  1. codes only               (_codes)
  2. domains + attributes      (_domains_attrs)    — taxonomy view
  3. codes + attributes        (_codes_attrs)      — codebook view
  4. domain → attribute → code (_domain_attr_code) — full chain (3 levels)

Per row: n ideas + % of the lens' idea base, % of RESPONSES (unique non-filtered
respondents), and valence balance x% (+) / y% (-) where (+) = positive+neutral,
(-) = negative. The smallest children per parent (together ≤ OVERIG_TAIL_PCT) fold
into one "overig (k …)" row.

Code lenses exclude the __UNASSIGNED__ sentinel from the % base (reported
separately); the domain lenses cover every idea (each idea has a domain).

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

OVERIG_TAIL_PCT = 0.10        # smallest children summing to ≤ this share of a parent → "overig"
SAVE_CSV = True

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
# SHARED HELPERS
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


def _code_valence_lookup(codebook) -> Dict[str, str]:
    out = {}
    for c in codebook.raw_codes:
        d = c if isinstance(c, dict) else c.__dict__
        out[d["code_name"]] = _vsign(d.get("valence", ""))
    return out


class _Cell:
    """Accumulator for one (n, negatives, respondent set) bucket."""
    __slots__ = ("n", "neg", "resp")

    def __init__(self):
        self.n = 0
        self.neg = 0
        self.resp = set()

    def add(self, rid, neg):
        self.n += 1
        self.neg += neg
        self.resp.add(rid)


def _balance(cell: _Cell):
    if not cell.n:
        return 0.0, 0.0
    return 100.0 * (cell.n - cell.neg) / cell.n, 100.0 * cell.neg / cell.n


def _fold_tail(items, parent_total, fold):
    """items: list of (label, _Cell) sorted however. Return (kept, tail) where
    tail is the smallest children summing to ≤ OVERIG_TAIL_PCT of parent_total."""
    if not fold:
        return list(items), []
    threshold = OVERIG_TAIL_PCT * parent_total
    tail, cum = [], 0
    for label, cell in sorted(items, key=lambda kv: kv[1].n):
        if cell.n == 0 or cum + cell.n <= threshold:
            tail.append((label, cell))
            cum += cell.n
        else:
            break
    tail_labels = {lbl for lbl, _ in tail}
    kept = [(lbl, c) for lbl, c in items if lbl not in tail_labels]
    return kept, tail


def _merge_cells(cells):
    m = _Cell()
    for c in cells:
        m.n += c.n
        m.neg += c.neg
        m.resp |= c.resp
    return m


def _row(depth, label, valence, cell, pct_i, pct_r):
    pos, neg = _balance(cell)
    return {"depth": depth, "label": label, "valence": valence, "n": cell.n,
            "pct_ideas": pct_i(cell.n), "pct_resp": pct_r(len(cell.resp)),
            "pct_pos": pos, "pct_neg": neg}


# =============================================================================
# BUILDERS
# =============================================================================

def build_groups(responses, codebook, group_by, show_attrs, fold_tail):
    """Two-level: group (code|domain) → attribute."""
    is_code = (group_by == "code")
    grp: Dict[str, _Cell] = defaultdict(_Cell)
    cell: Dict[str, Dict[str, _Cell]] = defaultdict(lambda: defaultdict(_Cell))
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
            grp[key].add(rid, neg)
            cell[key][attr].add(rid, neg)

    base_n = sum(c.n for c in grp.values())
    n_responses = len(resp_with_ideas)
    pct_i = lambda n: (100.0 * n / base_n) if base_n else 0.0
    pct_r = lambda k: (100.0 * k / n_responses) if n_responses else 0.0

    # code view: include defined-but-unused codes + source_attributes
    src_attrs: Dict[str, list] = {}
    if is_code:
        cv = {}
        for c in codebook.raw_codes:
            d = c if isinstance(c, dict) else c.__dict__
            cv[d["code_name"]] = _vsign(d.get("valence", ""))
            src_attrs[d["code_name"]] = d.get("source_attributes", []) or []
        for name in cv:
            grp.setdefault(name, _Cell())

    rows = []
    for key in sorted(grp, key=lambda k: (-grp[k].n, k.lower())):
        gc = grp[key]
        valence = cv.get(key, "~") if is_code else _derived_sign(_balance(gc)[1])
        rows.append(_row(0, key, valence, gc, pct_i, pct_r))
        if not show_attrs:
            continue
        items = {a: c for a, c in cell.get(key, {}).items()}
        for a in src_attrs.get(key, []):
            items.setdefault(a, _Cell())
        kept, tail = _fold_tail(list(items.items()), gc.n, fold_tail)
        for a, c in sorted(kept, key=lambda kv: -kv[1].n):
            rows.append(_row(1, a, "", c, pct_i, pct_r))
        if len(tail) >= 2:
            rows.append(_row(1, f"overig ({len(tail)} attrs)", "", _merge_cells([c for _, c in tail]), pct_i, pct_r))
        elif tail:
            a, c = tail[0]
            rows.append(_row(1, a, "", c, pct_i, pct_r))

    return rows, base_n, n_responses, n_unassigned


def build_domain_attr_code(responses, codebook, fold_tail):
    """Three-level: domain → attribute → code."""
    code_sign = _code_valence_lookup(codebook)
    dom: Dict[str, _Cell] = defaultdict(_Cell)
    da: Dict[str, Dict[str, _Cell]] = defaultdict(lambda: defaultdict(_Cell))
    dac: Dict[str, Dict[str, Dict[str, _Cell]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(_Cell)))
    resp_with_ideas: set = set()

    for resp in responses:
        rid = str(resp.respondent_id)
        for idea in (resp.response_ideas or []):
            resp_with_ideas.add(rid)
            d = (getattr(idea, "partition_name", "") or idea.domain or "").strip() or _NO_GROUP
            a = (idea.assigned_attribute or "").strip() or _NO_ATTR
            c = (idea.assigned_code or "").strip() or _UNASSIGNED
            neg = _is_neg(idea.valence)
            dom[d].add(rid, neg)
            da[d][a].add(rid, neg)
            dac[d][a][c].add(rid, neg)

    base_n = sum(c.n for c in dom.values())
    n_responses = len(resp_with_ideas)
    pct_i = lambda n: (100.0 * n / base_n) if base_n else 0.0
    pct_r = lambda k: (100.0 * k / n_responses) if n_responses else 0.0

    rows = []
    for d in sorted(dom, key=lambda k: (-dom[k].n, k.lower())):
        rows.append(_row(0, d, _derived_sign(_balance(dom[d])[1]), dom[d], pct_i, pct_r))
        for a in sorted(da[d], key=lambda k: -da[d][k].n):
            rows.append(_row(1, a, "", da[d][a], pct_i, pct_r))
            codes = list(dac[d][a].items())
            kept, tail = _fold_tail(codes, da[d][a].n, fold_tail)
            for c, cell in sorted(kept, key=lambda kv: -kv[1].n):
                rows.append(_row(2, c, code_sign.get(c, "~"), cell, pct_i, pct_r))
            if len(tail) >= 2:
                rows.append(_row(2, f"overig ({len(tail)} codes)", "",
                                 _merge_cells([c for _, c in tail]), pct_i, pct_r))
            elif tail:
                c, cell = tail[0]
                rows.append(_row(2, c, code_sign.get(c, "~"), cell, pct_i, pct_r))

    return rows, base_n, n_responses, 0


# =============================================================================
# DISPLAY
# =============================================================================

def _bal(r) -> str:
    return f"{r['pct_pos']:.0f}% (+) / {r['pct_neg']:.0f}% (-)" if r["n"] else ""


def print_readout(title, header_label, rows, base_n, n_responses, n_unassigned, compact=False):
    print(f"\n\n{'=' * 86}")
    print(f"[{title}]  {FILENAME}")
    print(f"{VARIABLE}  |  {base_n} ideas (base)"
          + (f"; {n_unassigned} unassigned" if n_unassigned else "")
          + f"  |  {n_responses} responses")
    print(f"{'=' * 86}")
    print(f"{header_label:50}{'n':>5}{'%idea':>7}{'%resp':>7}   balans (+/-)")
    print(f"{'-' * 86}")
    for r in rows:
        indent = "    " * r["depth"]
        label = f"{indent}[{r['valence']}] {r['label']}" if r["valence"] else f"{indent}{r['label']}"
        sep = "\n" if (r["depth"] == 0 and not compact) else ""
        tag = "" if r["n"] else "  (unused)"
        print(f"{sep}{label:50}{r['n']:>5}{r['pct_ideas']:>6.1f}%{r['pct_resp']:>6.1f}%   {_bal(r)}{tag}")
    print(f"\n{'-' * 86}")
    print(f"{'TOTAAL':50}{base_n:>5}{100.0:>6.1f}%")
    if n_unassigned:
        print(f"{'__UNASSIGNED__ (excl. van %-basis)':50}{n_unassigned:>5}")


def save_csv(suffix, header_cols, rows, base_n, n_responses, n_unassigned):
    exports_dir = project_root / "exports"
    exports_dir.mkdir(exist_ok=True)
    base = Path(FILENAME).stem.replace(" ", "_")
    csv_path = exports_dir / f"codebook_{base}_{VARIABLE}_{SAMPLE_SIZE}_{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["depth", "label", "valence", "n",
                    "pct_of_base", "pct_of_responses", "pct_pos_neutral", "pct_neg"])
        for r in rows:
            w.writerow([r["depth"], r["label"], r["valence"], r["n"],
                        f"{r['pct_ideas']:.1f}", f"{r['pct_resp']:.1f}",
                        f"{r['pct_pos']:.1f}" if r["n"] else "",
                        f"{r['pct_neg']:.1f}" if r["n"] else ""])
        w.writerow(["", "TOTAAL", "", base_n, "100.0", "", "", ""])
        w.writerow(["", _UNASSIGNED, "", n_unassigned, "", "", "", ""])
        w.writerow(["", "responses", "", n_responses, "", "", "", ""])
    print(f"  CSV → {csv_path.name}")


# =============================================================================
# MAIN
# =============================================================================

# (title, header_label, builder spec, csv_suffix)
VERSIONS = [
    ("CODES ONLY",             "code",                     ("groups", "code",   False, False), "codes"),
    ("DOMAINS + ATTRIBUTES",    "domain / attribute",       ("groups", "domain", True,  False), "domains_attrs"),
    ("CODES + ATTRIBUTES",      "code / attribute",         ("groups", "code",   True,  True),  "codes_attrs"),
    ("DOMAIN -> ATTRIBUTE -> CODE", "domain / attribute / code", ("dac",), "domain_attr_code"),
]

if __name__ == "__main__":
    responses, codebook = load_data()
    for title, header, spec, suffix in VERSIONS:
        if spec[0] == "groups":
            _, group_by, show_attrs, fold = spec
            rows, base_n, n_resp, n_una = build_groups(
                responses, codebook, group_by, show_attrs, fold)
            compact = not show_attrs
        else:
            rows, base_n, n_resp, n_una = build_domain_attr_code(
                responses, codebook, fold_tail=True)
            compact = False
        print_readout(title, header, rows, base_n, n_resp, n_una, compact)
        if SAVE_CSV:
            save_csv(suffix, header, rows, base_n, n_resp, n_una)

# %%
