#%%
"""Dry retrieval-signal measurement for embedding-based facet shortlisting.

Read-only. For every cached per-idea facet assignment (the full-menu
baseline), compute where that facet ranks when the domain's facet cards are
ordered by embedding similarity to the idea's ladder label. No LLM calls;
only embedding calls (~$0.02 on the ASN chain).

Known limitation: the cached taxonomy is post-consolidation (few cards per
domain), while shortlisting matters for raw menus of dozens of cards. This
measures the strength of the retrieval signal (does the chosen facet rank
on top?), not final recall@k on a raw menu. Strong signal here justifies a
shortlist arm in the batch experiment; weak signal kills it early.

Run from src/:  python -m pipeline.step_4_classifier.view_shortlist_recall
"""
import asyncio
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

import models
from test_data import TEST_DATA
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.embedder import SharedEmbedder
from pipeline.step_4_classifier.assignment_batching import facet_card_text
from pipeline.step_4_classifier.partition_labels import format_label

RANK_BUCKETS = (1, 3, 10)


async def main() -> None:
    filename = TEST_DATA.filename
    variable_key = generate_enhanced_variable_key(
        selected_variables=[TEST_DATA.var_name], is_merged=False,
        sample_size=TEST_DATA.sample_size,
    )
    cache_manager = CacheManager()
    taxonomy = cache_manager.load_metadata_from_cache(
        filename=filename, step="taxonomy", variable_key=variable_key,
        model_cls=models.TaxonomyResultsCache,
    )
    ideas_models = cache_manager.load_from_cache(
        filename, "extracted_ideas", variable_key, models.IdeasExtractedModel,
    )
    if taxonomy is None or not ideas_models:
        raise SystemExit(
            "taxonomy of extracted_ideas cache ontbreekt — draai eerst steps 3+4")

    idea_by_id = {
        idea.idea_id: idea
        for m in ideas_models for idea in (m.response_ideas or [])
    }

    embedder = SharedEmbedder()
    total_ranks: list = []
    print(f"{'domain':<42} {'ideas':>6} {'cards':>6} "
          + " ".join(f"rank<={k:>2}" for k in RANK_BUCKETS) + "  mean")

    for domain_name in sorted(taxonomy.partition_results):
        result = taxonomy.partition_results[domain_name]
        facets = [f if isinstance(f, dict) else f.model_dump() for f in result.facets]
        if len(facets) < 2 or not result.facet_assignments:
            continue
        facet_names = [f.get("facet_name", "") for f in facets]
        name_to_index = {n: i for i, n in enumerate(facet_names)}

        pairs = [
            (idea_id, facet_name)
            for idea_id, facet_name in result.facet_assignments.items()
            if facet_name in name_to_index and idea_id in idea_by_id
        ]
        if not pairs:
            continue
        labels = [format_label(idea_by_id[i], "ladder", "") for i, _ in pairs]

        card_emb = await embedder.embed_texts([facet_card_text(f) for f in facets])
        idea_emb = await embedder.embed_texts(labels)
        card_n = card_emb / np.linalg.norm(card_emb, axis=1, keepdims=True)
        idea_n = idea_emb / np.linalg.norm(idea_emb, axis=1, keepdims=True)
        similarities = idea_n @ card_n.T  # [n_ideas x n_cards]

        ranks = []
        for row_index, (_, assigned_name) in enumerate(pairs):
            order = np.argsort(-similarities[row_index])
            rank = int(np.where(order == name_to_index[assigned_name])[0][0]) + 1
            ranks.append(rank)
        total_ranks.extend(ranks)

        ranks_arr = np.array(ranks)
        print(f"{domain_name:<42} {len(pairs):>6} {len(facets):>6} "
              + " ".join(f"{(ranks_arr <= k).mean() * 100:>7.1f}%" for k in RANK_BUCKETS)
              + f"  {ranks_arr.mean():.2f}")

    if total_ranks:
        arr = np.array(total_ranks)
        print("-" * 100)
        print(f"{'TOTAL':<42} {len(arr):>6} {'':>6} "
              + " ".join(f"{(arr <= k).mean() * 100:>7.1f}%" for k in RANK_BUCKETS)
              + f"  {arr.mean():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
