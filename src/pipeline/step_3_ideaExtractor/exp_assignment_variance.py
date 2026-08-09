#%%

"""Is the unstable partition the taxonomy's fault, or the assignment's?

`exp_consolidation_variance.py` showed the taxonomy stages are steadier than they
look: ten consolidations of one identical input produced the same six themes every
time, only ever renamed. Yet a full rerun reproduces the respondent grouping at
ARI ~0.72. Something else moves it.

The remaining candidate is bulk extraction — one LLM call per response, each picking
a domain from the menu. That stage has never been measured on its own, because every
rerun so far changed the menu at the same time.

This holds the menu fixed (straight from the step 3 metadata cache) and runs
extraction twice over the same responses. Nothing differs between the two passes but
the model.

  high agreement  ->  assignment is reliable; the run-to-run spread comes from the
                      taxonomy after all, and the domain prompts are the place to work
  low agreement   ->  assignment is the source. Then no domain prompt can fix it, and
                      every before/after on the taxonomy has been measuring assignment
                      noise

Two numbers, because with a fixed menu both are available and they answer different
questions: exact agreement (did the same response get the same domain label?) and the
ARI (did respondents that were grouped together stay together?). Exact agreement is
the one to read; the ARI is there to compare against the full-run figure.

Usage, from src/:
    python -m pipeline.step_3_ideaExtractor.exp_assignment_variance
    python -m pipeline.step_3_ideaExtractor.exp_assignment_variance --n 500
"""

import asyncio
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import get_step_model, get_reasoning_params
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.llm import create_client, llm_create_async
import models

from test_data import TEST_DATA

from .config_ideaExtractor import DEFAULT_IDEA_EXTRACTION_CONFIG as CFG
from .dimension_data import get_dimension
from .ideaExtractor import IdeaExtractor, SAMPLING_SEED
from .measure_stability import adjusted_rand_index
from .prompts_ideaExtractor import (
    build_taxonomy_enriched_extraction_prompt,
    create_extraction_model,
    DomainItem,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_RESPONSES = 300          # override with --n
MAX_CONCURRENT = 12

MODEL = get_step_model("idea_extraction_abstraction_ladder")


# =============================================================================
# INPUT
# =============================================================================

def load_inputs(n: int):
    """Responses from step 2; the domain menu exactly as step 3 last cached it."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[TEST_DATA.var_name], is_merged=False,
        sample_size=TEST_DATA.sample_size)
    cm = CacheManager()

    filtered = cm.load_from_cache(TEST_DATA.filename, "quality_filter", variable_key,
                                  models.QualityFilteredModel)
    meta = cm.load_metadata_from_cache(TEST_DATA.filename, "extracted_ideas", variable_key,
                                       models.ExtractionMetadata)
    if not filtered or not meta:
        raise SystemExit("Need a step 2 cache and a step 3 metadata cache. Run both first.")

    meaningful = [r for r in filtered if not r.quality_filter]
    rng = random.Random(SAMPLING_SEED)
    sample = rng.sample(meaningful, min(n, len(meaningful)))

    domains = [
        DomainItem(
            key=d.get("key", "") or d.get("label", ""),
            label=d.get("label", ""),
            definition=d.get("definition", ""),
            boundary_test=d.get("boundary_test", "") or "",
            exclusions=list(d.get("exclusions") or []),
        )
        for d in (meta.domains or [])
    ]
    if not domains:
        raise SystemExit("The cached metadata carries no domains.")

    return sample, meta, domains


# =============================================================================
# ONE EXTRACTION PASS
# =============================================================================

async def extract_pass(sample, meta, domains, label: str) -> Dict[str, List[str]]:
    """Run extraction once over the sample. Returns respondent_id -> [domain, ...]."""
    dimension = get_dimension(meta.primary_dimension)
    domain_table = IdeaExtractor.build_domain_table(domains)
    model_cls = create_extraction_model(dimension=dimension, domains=domains)
    client = create_client(MODEL)
    gate = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(resp):
        prompt = build_taxonomy_enriched_extraction_prompt(
            language=meta.lang, var_lab=meta.var_lab, perspective=meta.perspective,
            sector=meta.sector, entity=meta.entity, topic=meta.topic, intent=meta.intent,
            response=resp.response, dimension=dimension, domain_table=domain_table)
        async with gate:
            try:
                out = await llm_create_async(
                    client=client, model=MODEL, response_model=List[model_cls],
                    prompt=prompt, temperature=CFG.temperature, max_tokens=CFG.max_tokens,
                    max_retries=3,
                    **get_reasoning_params(MODEL, phase="idea_extraction_abstraction_ladder"))
            except Exception as exc:                     # noqa: BLE001 — diagnostic script
                print(f"  [{label}] {resp.respondent_id}: {type(exc).__name__}")
                return resp.respondent_id, None
        return resp.respondent_id, [i.domain for i in out]

    print(f"Pass {label}: {len(sample)} responses...")
    pairs = await asyncio.gather(*(one(r) for r in sample))
    return {rid: doms for rid, doms in pairs if doms is not None}


# =============================================================================
# REPORTING
# =============================================================================

def report(a: Dict[str, List[str]], b: Dict[str, List[str]], domains) -> None:
    shared = sorted(set(a) & set(b))
    print(f"\n{'=' * 72}\nASSIGNMENT STABILITY, IDENTICAL MENU\n{'=' * 72}")
    print(f"{len(domains)} domains on the menu | {len(shared)} responses in both passes")

    same_count = [r for r in shared if len(a[r]) == len(b[r])]
    print(f"same number of ideas extracted: {len(same_count)}/{len(shared)} "
          f"({100 * len(same_count) / len(shared):.1f}%)")

    # Single-idea responses: the domain is a function of the whole text, so the two
    # passes are directly comparable without having to align idea order.
    single = [r for r in shared if len(a[r]) == 1 and len(b[r]) == 1]
    agree = [r for r in single if a[r][0] == b[r][0]]
    print(f"\nsingle-idea responses: {len(single)}")
    print(f"  identical domain     : {len(agree)} "
          f"({100 * len(agree) / len(single):.1f}%)" if single else "  none")

    if single:
        ari = adjusted_rand_index({r: a[r][0] for r in single},
                                  {r: b[r][0] for r in single})
        print(f"  ARI between passes   : {ari:.3f}")
        print(f"  (full reruns, menu changing too, sit at 0.38-0.82)")

        disagreements = Counter((a[r][0], b[r][0]) for r in single if a[r][0] != b[r][0])
        if disagreements:
            print("\n  where they disagree, most frequent first")
            for (x, y), n in disagreements.most_common(10):
                print(f"    {n:>3}x  {x}  ->  {y}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    n = N_RESPONSES
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    sample, meta, domains = load_inputs(n)
    print(f"dimension {meta.primary_dimension} | menu: {[d.label for d in domains]}")

    a = asyncio.run(extract_pass(sample, meta, domains, "A"))
    b = asyncio.run(extract_pass(sample, meta, domains, "B"))
    report(a, b, domains)


if __name__ == "__main__":
    main()
