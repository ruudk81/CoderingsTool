#%%

"""Read-only A/B harness: current vs new consolidation rules on ONE cached
domain (optionally scoped to a facet). Makes 2 LLM calls via utils/llm.py;
touches no cache. Dev tool for validating the consolidation prompt rewrite.

Run (from src/):
    python -m pipeline.step_4_classifier_experiment.view_consolidation_ab

Edit DOMAIN / FACET / PHASE below per run. Compare the two printed attribute
sets by hand: NEW should keep different dimensions distinct, group thin
same-dimension values, and use plain-language labels (no dimension-name
containers), while still consolidating where concepts genuinely coincide.
"""
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field

from config import get_step_model, get_reasoning_params
from utils.llm import create_client, llm_create_async, token_tracker
from pipeline.step_4_classifier_experiment.view_assignments_attributes_raw import load_ideas_with_raw_attributes


# =============================================================================
# CONFIGURATION
# =============================================================================
DOMAIN = "institutionele identiteit en marktpositie"
FACET = None            # None = whole domain; or a facet string to scope
PHASE = "classifier_p7"  # model tier to use for both A and B


# =============================================================================
# RESPONSE SCHEMA (throwaway; top-level is a BaseModel, not a bare List)
# =============================================================================
class Attr(BaseModel):
    name: str = Field(description="Plain-language attribute label")
    facet: str = Field(description="Best-fit facet for this attribute")
    dimension: str = Field(description="Underlying dimension it measures")
    n_covered: int = Field(description="How many input ideas it covers")


class Inventory(BaseModel):
    attributes: list[Attr]
    off_scope: str = Field(description="Ideas set aside as out of scope, with reason")


# =============================================================================
# RULE BLOCKS — A (current P7) vs B (new skeleton)
# =============================================================================
PREAMBLE = (
    'You are a taxonomy consolidation specialist for surveys. Deduplicate '
    'attributes across facets within the domain "{domain}", producing a MECE '
    'attribute inventory. Assign each survivor to the one facet where it fits '
    'best; do not restructure facets.'
)

RULES_CURRENT = """# Attribute Consolidation Rules
<strict_consolidation_rule>
1. PREVALENCE WEIGHTING
Attributes MUST be primarily driven by the number of ideas linked to attributes.
- HIGH idea counts MUST form the core structure of the codebook.
- LOW idea counts MUST NOT become standalone unless absolutely necessary.
- LOW-prevalence attributes SHOULD be merged into the closest HIGH-prevalence phenomenon, OR grouped into a broader combined phenomenon.
If forced to choose between conceptual nuance and prevalence dominance --> ALWAYS prioritize prevalence dominance.

2. MERGE BIAS
When in doubt: MERGE rather than split, especially when an attribute has relatively few ideas.
Attributes with low prevalence (e.g., <10-15 ideas) should almost never result in standalone attributes.

3. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged, even across facets. Variants that differ only in evaluative direction ("positive X"/"negative X") -> one attribute "X"; valence recorded separately.

4. ORTHOGONALITY (MAIN RULE)
"Can a single observation plausibly fall under both?" Yes -> merge; Doubt -> merge; Only if clearly no -> keep separate.

5. NO HIERARCHY
general vs specific -> merge.

6. NO OBJECT SPLITTING
Do not split based on object (humans vs animals) if the same principle applies -> merge.

7. MINIMALITY (MANDATORY)
Smallest number of attributes that provides full coverage.

8. FACET ASSIGNMENT
Assign each surviving attribute to the ONE facet where it fits best. Do NOT restructure or rename facets.
</strict_consolidation_rule>
When in doubt -> merge attributes."""

RULES_NEW = """# Attribute Consolidation Rules
<strict_consolidation_rule>
Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping by these rules, in order.

1. DIMENSION FIRST (orthogonality — the guardrail).
   For each concept, determine WHICH underlying dimension it answers.
   - Concepts on DIFFERENT dimensions are orthogonal: NEVER merge them into one attribute (e.g. socio-economic class vs political orientation vs age are different dimensions).
   - Mutually-exclusive VALUES/POLES of the SAME dimension are also kept apart (e.g. "young" vs "old"); merging opposite poles creates an empty container.
   - Do NOT create separate attributes based only on the object discussed (e.g. "humans" vs "animals") when the same underlying value applies — an object is not a dimension.

2. PREVALENCE SETS GRANULARITY (within a dimension only).
   - A high-count value keeps its own attribute — never dissolve a well-supported concept.
   - Several thin, same-dimension values are GROUPED into one attribute that still names the shared value/contrast in plain language.
   - Variants that differ only in evaluative direction ("positive X" and "negative X") collapse to ONE attribute "X"; the direction is recorded separately as valence, not as separate attributes.
   Prevalence decides how finely to split WITHIN a dimension; it NEVER licenses merging ACROSS dimensions.

3. LIFT, DON'T FLATTEN.
   When grouping is needed, raise concepts to a shared higher-abstraction label that still carries their meaning — NOT a label that merely names the axis.
   FORBIDDEN: empty containers that only name the dimension ("target-group positioning", "institutional trust", "size").
   REQUIRED: a stateable value a reader can picture ("a bank for ordinary people", "a trustworthy bank", "a small bank").

4. PLAIN, MEANINGFUL LABELS.
   Name every surviving attribute in everyday language. Test: reading the label alone, a layperson knows which distinction is meant, given the survey question. No jargon, no nominalizations, no dimension-names.

5. FACET ASSIGNMENT.
   Assign each surviving attribute to the ONE facet where it fits best. Do NOT restructure or rename facets -- only deduplicate attributes.

Precedence when rules conflict: 1 (orthogonality) > 2 (prevalence grouping) > 4 (label clarity).
</strict_consolidation_rule>"""

HINT = "\n\nProvide your output as valid JSON following the response schema provided."


# =============================================================================
# INPUT + RUN
# =============================================================================
def build_input(ideas) -> str:
    tree = defaultdict(lambda: defaultdict(list))
    for i in ideas:
        tree[i.facet or "(none)"][i.attribute or "(none)"].append(i)
    lines = []
    for facet in sorted(tree, key=lambda f: -sum(len(v) for v in tree[f].values())):
        lines.append(f"FACET: {facet}")
        for attr, grp in sorted(tree[facet].items(), key=lambda kv: -len(kv[1])):
            ex = "; ".join(f'"{(g.instance or "").strip()}"' for g in grp[:4])
            lines.append(f"  - {attr} (n={len(grp)})  ex: {ex}")
    return "\n".join(lines)


async def run(label: str, rules: str, data_input: str):
    model = get_step_model(PHASE)
    client = create_client(model, async_mode=True)
    prompt = f"{PREAMBLE.format(domain=DOMAIN)}\n\n{rules}\n\n# Input\n{data_input}{HINT}"
    res = await llm_create_async(
        client=client, model=model, prompt=prompt,
        response_model=Inventory, temperature=0.3, max_tokens=3500,
        **get_reasoning_params(model, PHASE),
    )
    byf = defaultdict(list)
    for a in res.attributes:
        byf[a.facet].append(a)
    print(f"\n{'='*84}\n{label}   (model={model})   -> {len(res.attributes)} attributes\n{'='*84}")
    for facet in byf:
        print(f"\n  FACET: {facet}")
        for a in byf[facet]:
            print(f"    • {a.name}  (n={a.n_covered})  [dim: {a.dimension}]")
    if res.off_scope.strip():
        print(f"\n  OFF-SCOPE: {res.off_scope}")


async def main():
    ideas = [i for i in load_ideas_with_raw_attributes()
             if (i.domain or "") == DOMAIN and (FACET is None or (i.facet or "") == FACET)]
    scope = f"{DOMAIN}" + (f" / {FACET}" if FACET else "")
    print(f"Scope: {scope}  ({len(ideas)} ideas)")
    data_input = build_input(ideas)
    await run("A — CURRENT rules", RULES_CURRENT, data_input)
    await run("B — NEW rules", RULES_NEW, data_input)
    print("\n" + token_tracker.get_summary())


if __name__ == "__main__":
    asyncio.run(main())
