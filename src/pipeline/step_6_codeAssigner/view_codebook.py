#%%

"""
View codebook: codes + attributes with assignment counts.

Read-only readout of the step-5 codebook annotated with the actual step-6
assignment counts. Per code (and per attribute within it): n ideas and the
percentage of ASSIGNED ideas (the __UNASSIGNED__ sentinel is excluded from the
percentage base and reported separately). Codes/attributes the codebook defines
but no idea received are shown with n=0.

Usage:
    cd src && python -m pipeline.step_6_codeAssigner.view_codebook
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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

SHOW_ZERO_CODES = True        # show codebook codes that received no ideas
SHOW_ZERO_ATTRIBUTES = True   # show a code's source_attributes that received no ideas
SAVE_CSV = True

_UNASSIGNED = "__UNASSIGNED__"
_NO_ATTR = "(geen attribuut)"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load step-6 assignments + the step-5 codebook from cache."""
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
    ideas = [i for r in results for i in (r.response_ideas or [])]
    return ideas, codebook


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


def build_rows(ideas: List, codebook) -> tuple:
    """Return (rows, n_assigned, n_unassigned).

    rows: ordered list of dicts {level, code, attribute, valence, n, pct}.
    """
    # Count ideas per (code, attribute) from step-6 assignments
    code_n: Counter = Counter()
    code_attr_n: Dict[str, Counter] = defaultdict(Counter)
    n_unassigned = 0
    for idea in ideas:
        code = (idea.assigned_code or "").strip()
        if not code or code == _UNASSIGNED:
            n_unassigned += 1
            continue
        attr = (idea.assigned_attribute or "").strip() or _NO_ATTR
        code_n[code] += 1
        code_attr_n[code][attr] += 1

    n_assigned = sum(code_n.values())
    pct = lambda n: (100.0 * n / n_assigned) if n_assigned else 0.0

    # Codebook definition lookup (code_name → (valence, source_attributes))
    cb = {}
    for c in codebook.raw_codes:
        d = c if isinstance(c, dict) else c.__dict__
        cb[d["code_name"]] = (d.get("valence", ""), d.get("source_attributes", []) or [])

    # Code order: every codebook code + any assigned code not in the codebook,
    # sorted by assignment count descending.
    all_codes = set(cb) | set(code_n)
    ordered = sorted(all_codes, key=lambda c: (-code_n.get(c, 0), c.lower()))

    rows = []
    for code in ordered:
        n = code_n.get(code, 0)
        if n == 0 and not SHOW_ZERO_CODES:
            continue
        valence, source_attrs = cb.get(code, ("", []))
        rows.append({"level": "code", "code": code, "attribute": "",
                     "valence": _vsign(valence), "n": n, "pct": pct(n)})

        # Attributes: assigned ones (by count) + defined-but-unused source_attrs
        attr_counts = code_attr_n.get(code, Counter())
        seen = set()
        for attr, an in attr_counts.most_common():
            seen.add(attr)
            rows.append({"level": "attr", "code": code, "attribute": attr,
                         "valence": "", "n": an, "pct": pct(an)})
        if SHOW_ZERO_ATTRIBUTES:
            for attr in source_attrs:
                if attr not in seen:
                    rows.append({"level": "attr", "code": code, "attribute": attr,
                                 "valence": "", "n": 0, "pct": 0.0})

    return rows, n_assigned, n_unassigned


# =============================================================================
# DISPLAY
# =============================================================================

def print_readout(rows, n_assigned, n_unassigned):
    n_total = n_assigned + n_unassigned
    print(f"\n{'=' * 72}")
    print(f"CODEBOOK — {FILENAME}")
    print(f"{VARIABLE}  |  N = {n_assigned} assigned ideas "
          f"(of {n_total} total; {n_unassigned} unassigned)")
    print(f"{'=' * 72}")
    print(f"{'code / attribute':52}{'n':>6}{'%':>8}")
    print(f"{'-' * 72}")

    for r in rows:
        if r["level"] == "code":
            label = f"[{r['valence']}] {r['code']}"
            print(f"\n{label:52}{r['n']:>6}{r['pct']:>7.1f}%")
        else:
            label = f"      {r['attribute']}"
            zero = "" if r["n"] else "  (unused)"
            print(f"{label:52}{r['n']:>6}{r['pct']:>7.1f}%{zero}")

    print(f"\n{'-' * 72}")
    print(f"{'TOTAAL (som codes)':52}{n_assigned:>6}{100.0:>7.1f}%")
    if n_unassigned:
        upct = 100.0 * n_unassigned / n_total if n_total else 0.0
        print(f"{'__UNASSIGNED__ (excl. van %-basis)':52}{n_unassigned:>6}{upct:>7.1f}%  of {n_total}")


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_csv(rows, n_assigned, n_unassigned):
    exports_dir = project_root / "exports"
    exports_dir.mkdir(exist_ok=True)
    base = Path(FILENAME).stem.replace(" ", "_")
    csv_path = exports_dir / f"codebook_{base}_{VARIABLE}_{SAMPLE_SIZE}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["level", "code", "attribute", "valence", "n", "pct_of_assigned"])
        for r in rows:
            w.writerow([r["level"], r["code"], r["attribute"], r["valence"],
                        r["n"], f"{r['pct']:.1f}"])
        w.writerow(["total", "TOTAAL", "", "", n_assigned, "100.0"])
        w.writerow(["unassigned", _UNASSIGNED, "", "", n_unassigned, ""])
    print(f"\nCSV saved to: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ideas, codebook = load_data()
    rows, n_assigned, n_unassigned = build_rows(ideas, codebook)
    print_readout(rows, n_assigned, n_unassigned)
    if SAVE_CSV:
        save_csv(rows, n_assigned, n_unassigned)

# %%
