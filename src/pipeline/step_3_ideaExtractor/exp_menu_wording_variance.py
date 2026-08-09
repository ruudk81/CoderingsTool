#%%

"""Does the wording of the domain menu move the assignments?

The last open link in the chain. What the other two experiments established:

  exp_consolidation_variance   ten consolidations of one identical input found the
                               same six themes every time — and named them
                               differently every time, with their own definitions,
                               boundary tests and exclusions
  exp_assignment_variance      with one fixed menu, two extraction passes agreed on
                               97% of responses (ARI 0.927)

Both stages are stable given fixed input, yet a full rerun reproduces the grouping at
only ARI 0.38-0.82. The difference has to enter through what passes between them: the
menu, which keeps its themes but is re-described each run. The extraction prompt shows
the model each domain's definition, its ✓ membership test and its ✗ exclusions — so a
borderline idea can fall inside under one phrasing and outside under another.

This runs assignment twice over the same responses, once per menu, using two
consolidations of the *same* frozen chunk proposals. Same themes, different wording,
nothing else changed.

  ARI near 0.93  ->  wording is cosmetic; the instability comes from discovery
                     proposing genuinely different themes between runs
  ARI well below ->  wording is the mechanism, and the target is not which domains
                     exist but how sharply they are described

Only the ARI can answer this: the two menus use different labels, so no exact match
is possible. That is what it was built for.

Usage, from src/:
    python -m pipeline.step_3_ideaExtractor.exp_menu_wording_variance
    python -m pipeline.step_3_ideaExtractor.exp_menu_wording_variance --n 500
"""

import asyncio
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from .dimension_data import get_dimension
from .ideaExtractor import IdeaExtractor
from .measure_stability import adjusted_rand_index
from .exp_consolidation_variance import (
    load_inputs as load_all_responses,
    frozen_proposals,
    consolidate_n_times,
)
from .exp_assignment_variance import load_inputs as load_sample, extract_pass


# =============================================================================
# CONFIGURATION
# =============================================================================

N_RESPONSES = 300          # override with --n


# =============================================================================
# MENUS
# =============================================================================

def full_menu(consolidated, dimension) -> List:
    """Discovered domains plus the two standing ones, exactly as production assembles it."""
    standing = IdeaExtractor._resolve_standing_domains(consolidated, dimension)
    return list(consolidated.domains) + standing


def print_menu(name: str, domains) -> None:
    print(f"\n  menu {name}:")
    for d in domains:
        print(f"    • {d.label}")
        print(f"        {d.definition[:110]}")
        if d.boundary_test:
            print(f"        ✓ {d.boundary_test[:110]}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    n = N_RESPONSES
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    all_responses, meta = load_all_responses()
    sample, _, _ = load_sample(n)
    dimension = get_dimension(meta.primary_dimension)

    chunk_domains = frozen_proposals(all_responses, meta, refresh=False)
    consolidations = asyncio.run(
        consolidate_n_times(chunk_domains, all_responses, meta, n=2))

    menu_a, menu_b = (full_menu(c, dimension) for c in consolidations)
    print(f"\n{'=' * 72}\nTWO MENUS FROM IDENTICAL CHUNK PROPOSALS\n{'=' * 72}")
    print_menu("A", menu_a)
    print_menu("B", menu_b)

    a = asyncio.run(extract_pass(sample, meta, menu_a, "A"))
    b = asyncio.run(extract_pass(sample, meta, menu_b, "B"))

    shared = sorted(set(a) & set(b))
    single = [r for r in shared if len(a[r]) == 1 and len(b[r]) == 1]

    print(f"\n{'=' * 72}\nASSIGNMENT UNDER THE TWO MENUS\n{'=' * 72}")
    print(f"{len(shared)} responses in both passes | {len(single)} with a single idea")
    if not single:
        print("Not enough single-idea responses to compare.")
        return

    ari = adjusted_rand_index({r: a[r][0] for r in single},
                              {r: b[r][0] for r in single})
    print(f"\nARI between the two menus : {ari:.3f}")
    print(f"  same menu, two passes   : 0.927   (exp_assignment_variance)")
    print(f"  full rerun, everything  : 0.38 - 0.82")

    print("\nwhere they land differently (label pairs, most frequent first)")
    from collections import Counter
    pairs = Counter((a[r][0], b[r][0]) for r in single)
    for (x, y), count in pairs.most_common(12):
        print(f"  {count:>3}x  {x}  ->  {y}")


if __name__ == "__main__":
    main()
