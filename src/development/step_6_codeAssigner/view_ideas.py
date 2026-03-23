#%%

"""
View all ideas grouped by assigned code, with full ladder + taxonomy + code details.

Output:
  - Console: grouped by code, each idea on 4 bullet lines
  - CSV: flat export to exports/ for Excel inspection

Usage:
    cd src && python -m development.step_6_codeAssigner.view_ideas
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "development"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from development.step_6_codeAssigner.models_codeAssigner import CodeAssignedModel, CodeAssignedSubmodel

try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

MAX_PER_CODE = None  # None for all, or N to limit ideas shown per code
SAVE_CSV = True


# =============================================================================
# DATA LOADING
# =============================================================================

def load_assignments(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> List[CodeAssignedModel]:
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )
    cache_manager = CacheManager()
    data = cache_manager.load_from_cache(
        filename, "taxonomy_codes", variable_key, CodeAssignedModel
    )
    if not data:
        raise FileNotFoundError(
            f"No cached assignment results for 'code_assignment' / '{variable_key}'."
        )
    return data


def flatten_ideas(data: List[CodeAssignedModel]) -> List[CodeAssignedSubmodel]:
    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)
    return ideas


# =============================================================================
# DISPLAY
# =============================================================================

def print_ideas_by_code(ideas: List[CodeAssignedSubmodel], max_per_code: Optional[int] = None):
    # Group by code
    code_groups: Dict[str, List[CodeAssignedSubmodel]] = defaultdict(list)
    for idea in ideas:
        code = idea.assigned_code or "(unassigned)"
        code_groups[code].append(idea)

    # Sort codes by count descending
    sorted_codes = sorted(code_groups.items(), key=lambda x: -len(x[1]))

    total = len(ideas)
    assigned = sum(1 for i in ideas if i.assigned_code)
    print(f"\n{'='*80}")
    print(f"ALL IDEAS ({total} total, {assigned} assigned)")
    print(f"{'='*80}")

    for code_name, code_ideas in sorted_codes:
        confs = [i.confidence or 0.0 for i in code_ideas]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        print(f"\n{'='*80}")
        print(f"CODE: {code_name} — {len(code_ideas)} ideas (avg conf {avg_conf:.2f})")
        print(f"{'='*80}")

        display_ideas = code_ideas[:max_per_code] if max_per_code else code_ideas
        for idea in display_ideas:
            instance = idea.instance or ""
            interpretation = idea.interpretation or ""
            abstraction = idea.abstraction or ""
            valence = idea.valence or "0"
            domain = idea.domain or ""
            facet = idea.facet or ""
            attribute = idea.assigned_attribute or "(none)"
            conf = idea.confidence or 0.0

            print(f"\n  • Idea: {idea.idea_id} — \"{instance}\"")
            print(f"    Ladder: {instance} → {interpretation} → {abstraction} [{valence}]")
            print(f"    Taxonomy: {domain} > {facet} > {attribute}")
            print(f"    Code: {code_name} [conf: {conf:.2f}]")

        if max_per_code and len(code_ideas) > max_per_code:
            print(f"\n    ... ({len(code_ideas) - max_per_code} more ideas)")


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_csv(ideas: List[CodeAssignedSubmodel], filename: str = FILENAME, variable: str = VARIABLE):
    exports_dir = project_root / "exports"
    exports_dir.mkdir(exist_ok=True)

    base = Path(filename).stem.replace(" ", "_")
    csv_path = exports_dir / f"ideas_{base}_{variable}_{SAMPLE_SIZE}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "idea_id", "instance", "interpretation", "abstraction", "valence",
            "domain", "facet", "attribute",
            "code", "confidence",
        ])
        for idea in ideas:
            writer.writerow([
                idea.idea_id or "",
                idea.instance or "",
                idea.interpretation or "",
                idea.abstraction or "",
                idea.valence or "",
                idea.domain or "",
                idea.facet or "",
                idea.assigned_attribute or "",
                idea.assigned_code or "",
                f"{idea.confidence:.2f}" if idea.confidence else "",
            ])

    print(f"\nCSV saved to: {csv_path}")
    return csv_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    data = load_assignments()
    ideas = flatten_ideas(data)
    print_ideas_by_code(ideas, max_per_code=MAX_PER_CODE)

    if SAVE_CSV:
        save_csv(ideas)
