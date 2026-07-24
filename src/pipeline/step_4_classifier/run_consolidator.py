#%%
"""
Step 4: Cross-Domain Consolidator runner (P8 only)

Runs P8 independently on cached P7 output. For testing, debugging,
and fine-tuning without re-running P1-P7.

Usage:
    cd src && python -m pipeline.step_4_classifier.run_consolidator
"""

import asyncio
import sys
import io
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import nest_asyncio
nest_asyncio.apply()

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from test_data import TEST_DATA
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker, fetch_rate_limits
from utils.costTracker import CostTracker

from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.cross_domain_consolidator import CrossDomainConsolidator
from models import (
    TaxonomyResultsCache,
    TaxonomyClassifiedModel,
)
from models import ExtractionMetadata


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

PRINT_PROMPTS = False

# Override P8 config here for experimentation
CONFIG = CategoriesConfig(
    # p8_code_source="instance_interpretation",
    # p8_window_size=10,
    # p8_window_overlap=2,
    # p8_similarity_threshold=0.6,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_p7_output(variable_key: str):
    """Load P7's cached taxonomy + growing model + extraction metadata."""
    cache_manager = CacheManager()

    taxonomy_cache = cache_manager.load_metadata_from_cache(
        filename=FILENAME, step="taxonomy",
        variable_key=variable_key, model_cls=TaxonomyResultsCache,
    )
    if not taxonomy_cache:
        raise FileNotFoundError(
            f"No taxonomy cache found. Run P1-P7 first.\n"
            f"  cd src && python -m pipeline.step_4_classifier.run_classifier"
        )

    classified = cache_manager.load_from_cache(
        filename=FILENAME, step="taxonomy_classified",
        variable_key=variable_key, model_cls=TaxonomyClassifiedModel,
    )
    if not classified:
        raise FileNotFoundError(
            f"No classified ideas found. Run P1-P7 first."
        )

    extraction_meta = cache_manager.load_metadata_from_cache(
        filename=FILENAME, step="extracted_ideas",
        variable_key=variable_key, model_cls=ExtractionMetadata,
    )

    n_domains = len(taxonomy_cache.partition_results)
    n_attrs = sum(
        len(a) for r in taxonomy_cache.partition_results.values()
        for a in r.attributes.values()
    )
    n_ideas = sum(
        len(r.attribute_assignments)
        for r in taxonomy_cache.partition_results.values()
    )
    print(f"Loaded P7 output: {n_domains} domains, {n_attrs} attributes, {n_ideas} ideas")

    return taxonomy_cache, classified, extraction_meta


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_consolidated_taxonomy(taxonomy_cache: TaxonomyResultsCache):
    """Print the full consolidated taxonomy."""
    from collections import defaultdict

    print(f"\n{'=' * 80}")
    print(f"CONSOLIDATED TAXONOMY ({len(taxonomy_cache.partition_results)} domains)")
    print(f"{'=' * 80}")

    for domain_name in sorted(taxonomy_cache.partition_results.keys()):
        result = taxonomy_cache.partition_results[domain_name]
        n_facets = len(result.facets)
        n_attrs = sum(len(a) for a in result.attributes.values())
        n_assigned = len(result.attribute_assignments)

        attr_counts: dict = defaultdict(int)
        for aname in result.attribute_assignments.values():
            attr_counts[aname] += 1

        print(f"\n{'─' * 80}")
        print(f"DOMAIN: {domain_name} "
              f"({n_facets} facets, {n_attrs} attributes, {n_assigned} ideas)")
        print(f"{'─' * 80}")

        for facet_name in sorted(result.attributes.keys()):
            attrs = result.attributes[facet_name]
            print(f"  Facet: {facet_name} ({len(attrs)} attributes)")
            for attr_dict in sorted(
                attrs, key=lambda a: -attr_counts.get(a.get("attribute_name", ""), 0)
            ):
                aname = attr_dict.get("attribute_name", "?")
                count = attr_counts.get(aname, 0)
                print(f"    - {aname} [{count} ideas]")

    total_attrs = sum(
        len(a) for r in taxonomy_cache.partition_results.values()
        for a in r.attributes.values()
    )
    total_ideas = sum(
        len(r.attribute_assignments)
        for r in taxonomy_cache.partition_results.values()
    )
    print(f"\n{'=' * 80}")
    print(f"TOTALS: {total_attrs} attributes, {total_ideas} ideas")
    print(f"{'=' * 80}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CROSS-DOMAIN CONSOLIDATOR (P8 only)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Embedding: {CONFIG.p8_code_source}")
    print(f"Window: {CONFIG.p8_window_size} (overlap {CONFIG.p8_window_overlap})")
    print(f"Model: {CONFIG.qr_model_p8}")
    print()

    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )

    # Load P7 output
    taxonomy_cache, classified, extraction_meta = load_p7_output(variable_key)

    # Set up prompt capture and cost tracking
    prompt_printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)
    cost_tracker = CostTracker(filename=FILENAME, variable_key=variable_key)

    # Fetch rate limits for SmoothRequester
    print("Fetching rate limits...")
    fetched_limits, _ = asyncio.run(fetch_rate_limits(CONFIG.qr_model_p8))

    # Run P8
    consolidator = CrossDomainConsolidator(
        config=CONFIG,
        prompt_printer=prompt_printer,
        dataset_key=variable_key,
        cost_tracker=cost_tracker,
        fetched_limits=fetched_limits,
    )

    new_taxonomy, new_classified, merge_map, stats = asyncio.run(
        consolidator.consolidate(
            taxonomy_cache=taxonomy_cache,
            classified=classified,
            extraction_meta=extraction_meta,
            verbose=True,
        )
    )

    cost_tracker.finalize_step("step_4_taxonomy_classifier")

    # Print merge report
    if merge_map:
        print(f"\n{'─' * 80}")
        print(f"MERGE MAP ({len(merge_map)} remappings)")
        print(f"{'─' * 80}")
        for (src_domain, old_name), target in sorted(merge_map.items()):
            print(f"  \"{old_name}\" ({src_domain}) → \"{target.new_attribute_name}\" "
                  f"({target.new_domain} > {target.new_facet})")

    # Print consolidated taxonomy
    print_consolidated_taxonomy(new_taxonomy)

    # Save to cache (overwrite P7 output)
    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=new_taxonomy, filename=FILENAME,
        step="taxonomy", variable_key=variable_key,
    )
    cache_manager.save_to_cache(
        data=new_classified, filename=FILENAME,
        step="taxonomy_classified", variable_key=variable_key,
    )
    print(f"\nSaved to cache: taxonomy + taxonomy_classified (P8 consolidated)")

    # Save prompts
    if prompt_printer and prompt_printer.prompts:
        prompts_dir = project_root / "exports" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        p8_prompts = [
            p for p in prompt_printer.prompts
            if p.get("prompt_type") == "cross_domain_consolidation"
        ]
        if p8_prompts:
            pp = PromptPrinter(enabled=True)
            pp.prompts = p8_prompts
            pp.save_prompts(str(prompts_dir / f"step4_p8_{variable_key}.json"))
            print(f"Prompts saved: {len(p8_prompts)} P8 prompts")

    # Token summary
    if token_tracker.call_count > 0:
        print(token_tracker.get_summary())


if __name__ == "__main__":
    token_tracker.reset()
    main()

# %%
