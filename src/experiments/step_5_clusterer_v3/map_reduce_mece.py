"""
Map-Reduce MECE — Per-cluster topic extraction via 3-step LLM pipeline.

All clusters are processed CONCURRENTLY. All LLM calls (map batches,
reduce calls, MECE calls) across all clusters share a single semaphore
and rate limiter for efficient throughput control.

Pipeline for each cluster:
  1. MAP:    Batch all ideas into groups of max batch_size.
             For each batch, identify Central Organizing Concepts (COCs).
  2. REDUCE: Consolidate COCs across batches into unified list.
             Skipped for single-batch clusters.
  3. MECE:   Apply MECE constraints with inclusion/exclusion boundaries.

Usage:
    from .map_reduce_mece import MapReduceMECE
    from .config_clusterer_exp import ClustererConfig

    config = ClustererConfig()
    mece = MapReduceMECE(config)
    results = mece.process_all_clusters(
        cluster_texts={0: ["idea1", "idea2", ...], 1: [...]},
        survey_question="What do you think about X?",
        language="Dutch",
    )

This is an isolated copy for experimentation in step_5_clusterer_v3.
Changes here do NOT affect the production pipeline.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import create_client, llm_create_async

from .config_clusterer_exp import ClustererConfig
from .prompts_exp import (
    MAP_THEMES_PROMPT,
    REDUCE_THEMES_PROMPT,
    MECE_BOUNDARIES_PROMPT,
    MapBatchCOCs,
    ReducedThemeList,
    ClusterMECETopicSet,
    MECETopic,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


@dataclass
class PromptContext:
    """Shared context passed to all prompt formatting methods."""
    survey_question: str
    language: str
    dataset_context_section: str
    taxonomy_context: str


@dataclass
class ClusterMECEResult:
    """Complete map-reduce MECE result for a single cluster."""
    cluster_id: int
    n_ideas: int
    n_batches: int
    reduce_skipped: bool
    topics: List[MECETopic]


class MapReduceMECE:
    """
    Per-cluster Map-Reduce MECE topic extractor.

    All clusters are processed concurrently. All LLM calls share a single
    async client, semaphore, and rate limiter for efficient throughput.

    Three-step pipeline for each cluster:
    1. MAP:    Batch all ideas, identify COCs per batch (async)
    2. REDUCE: Consolidate COCs across batches (async, single call)
    3. MECE:   Apply inclusion/exclusion boundaries (async, single call)
    """

    def __init__(self, config: ClustererConfig):
        self._model = config.mapreduce_model
        self._temperature = config.mapreduce_temperature
        self._max_tokens_map = config.mapreduce_max_tokens_map
        self._max_tokens_reduce = config.mapreduce_max_tokens_reduce
        self._max_tokens_mece = config.mapreduce_max_tokens_mece
        self._batch_size = config.mapreduce_batch_size
        self._concurrency = config.mapreduce_concurrency
        self._rpm_limit = config.mapreduce_rpm_limit

        # Shared async resources — initialized in _process_all_async()
        self._client = None
        self._semaphore = None
        self._rate_limiter = None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def process_all_clusters(
        self,
        cluster_texts: Dict[int, List[str]],
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        taxonomy_axis: Optional[str] = None,
        taxonomy_description: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[int, ClusterMECEResult]:
        """
        Process all clusters through the map-reduce-MECE pipeline.

        All clusters are processed concurrently. All LLM calls share
        one semaphore + rate limiter for throughput control.

        Args:
            cluster_texts: Dict mapping cluster_id to list of idea texts
            survey_question: The survey question for prompt context
            language: Language for output (default: Dutch)
            dataset_context: Optional dict with domain, entity, topic, etc.
            taxonomy_axis: Optional taxonomy primary axis
            taxonomy_description: Optional taxonomy axis description
            verbose: Print progress info

        Returns:
            Dict mapping cluster_id to ClusterMECEResult
        """
        # Build shared context
        dataset_context_section = self._build_dataset_context_section(dataset_context)
        taxonomy_context = self._build_taxonomy_context(taxonomy_axis, taxonomy_description)

        context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            taxonomy_context=taxonomy_context,
        )

        # Filter empty clusters
        active_clusters = {
            cid: ideas for cid, ideas in cluster_texts.items() if ideas
        }

        if verbose:
            total_ideas = sum(len(ideas) for ideas in active_clusters.values())
            total_batches = sum(
                len(self._create_batches(list(dict.fromkeys(ideas))))
                for ideas in active_clusters.values()
            )
            print(f"  Processing {len(active_clusters)} clusters concurrently "
                  f"({total_ideas} ideas, ~{total_batches} map batches)")
            print(f"  Concurrency: {self._concurrency} in-flight, {self._rpm_limit} RPM")

        # Run everything in a single event loop
        return asyncio.run(self._process_all_async(active_clusters, context, verbose))

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _process_all_async(
        self,
        cluster_texts: Dict[int, List[str]],
        context: PromptContext,
        verbose: bool,
    ) -> Dict[int, ClusterMECEResult]:
        """Main async entry: create shared resources, gather all clusters."""
        # Create shared resources (one client, one semaphore, one rate limiter)
        self._client = create_client(model=self._model, async_mode=True)
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._rate_limiter = AsyncLimiter(self._rpm_limit, time_period=60)

        start_time = time.time()

        # Launch all clusters concurrently
        tasks = {
            cluster_id: self._process_single_cluster(cluster_id, ideas, context, verbose)
            for cluster_id, ideas in sorted(cluster_texts.items())
        }

        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Collect results, handling any exceptions
        results = {}
        for cluster_id, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                print(f"  Cluster {cluster_id} FAILED: {type(result).__name__}: {result}")
            else:
                results[cluster_id] = result

        elapsed = time.time() - start_time
        if verbose:
            total_topics = sum(len(r.topics) for r in results.values())
            print(f"\n  All clusters done in {elapsed:.1f}s → "
                  f"{total_topics} MECE topics across {len(results)} clusters")

        return results

    # =========================================================================
    # SINGLE CLUSTER PIPELINE (async)
    # =========================================================================

    async def _process_single_cluster(
        self,
        cluster_id: int,
        ideas: List[str],
        context: PromptContext,
        verbose: bool = False,
    ) -> ClusterMECEResult:
        """Run the 3-step pipeline for a single cluster (async)."""

        # Step 0: Deduplicate and batch
        unique_ideas = list(dict.fromkeys(ideas))  # preserve order, remove dupes
        batches = self._create_batches(unique_ideas)
        n_batches = len(batches)

        if verbose:
            dedup_note = f" (dedup {len(ideas)}→{len(unique_ideas)})" if len(unique_ideas) < len(ideas) else ""
            print(f"    Cluster {cluster_id}: {len(unique_ideas)} ideas{dedup_note}, "
                  f"{n_batches} batch(es)")

        # Step 1: MAP — identify COCs per batch (all batches concurrent)
        map_results = await self._run_map_step(cluster_id, batches, context)

        total_themes = sum(len(r.themes) for r in map_results)

        # Step 2: REDUCE — consolidate COCs (skip if single batch)
        reduce_skipped = (n_batches == 1)

        if reduce_skipped:
            # Convert single map result to ReducedThemeList format
            consolidated = ReducedThemeList(
                themes=[
                    {
                        "theme_label": t.theme_label,
                        "description": t.description,
                        "merged_from": [t.theme_label],
                    }
                    for t in map_results[0].themes
                ]
            )
        else:
            consolidated = await self._run_reduce_step(cluster_id, map_results, context)

        # Step 3: MECE — apply boundaries
        mece_result = await self._run_mece_step(
            cluster_id, consolidated, len(unique_ideas), context
        )

        if verbose:
            print(f"    Cluster {cluster_id} → {len(mece_result.topics)} MECE topics "
                  f"(map: {total_themes} themes, "
                  f"{'reduce skipped' if reduce_skipped else f'reduced to {len(consolidated.themes)}'})")

        return ClusterMECEResult(
            cluster_id=cluster_id,
            n_ideas=len(unique_ideas),
            n_batches=n_batches,
            reduce_skipped=reduce_skipped,
            topics=mece_result.topics,
        )

    # =========================================================================
    # SHARED LLM CALL (all calls flow through here)
    # =========================================================================

    async def _llm_call(self, prompt: str, response_model, max_tokens: int):
        """Make a rate-limited LLM call through the shared semaphore."""
        async with self._semaphore:
            async with self._rate_limiter:
                return await llm_create_async(
                    client=self._client,
                    model=self._model,
                    prompt=prompt,
                    response_model=response_model,
                    temperature=self._temperature,
                    max_tokens=max_tokens,
                )

    # =========================================================================
    # MAP STEP (async parallel batches)
    # =========================================================================

    async def _run_map_step(
        self,
        cluster_id: int,
        batches: List[List[str]],
        context: PromptContext,
    ) -> List[MapBatchCOCs]:
        """Send each batch to LLM concurrently through shared rate limiter."""
        results = [None] * len(batches)

        async def process_batch(batch_idx: int, ideas: List[str]):
            prompt = self._build_map_prompt(
                cluster_id, ideas, batch_idx, len(batches), context
            )
            try:
                result = await self._llm_call(prompt, MapBatchCOCs, self._max_tokens_map)
                results[batch_idx] = result
            except Exception as e:
                print(f"    MAP cluster {cluster_id} batch {batch_idx + 1}/{len(batches)} "
                      f"FAILED: {type(e).__name__}: {e}")
                results[batch_idx] = MapBatchCOCs(themes=[])

        await asyncio.gather(*(
            process_batch(i, batch) for i, batch in enumerate(batches)
        ))

        return [r for r in results if r is not None]

    # =========================================================================
    # REDUCE STEP (async)
    # =========================================================================

    async def _run_reduce_step(
        self,
        cluster_id: int,
        map_results: List[MapBatchCOCs],
        context: PromptContext,
    ) -> ReducedThemeList:
        """Consolidate themes from all batches into one unified list."""
        prompt = self._build_reduce_prompt(cluster_id, map_results, context)

        try:
            return await self._llm_call(prompt, ReducedThemeList, self._max_tokens_reduce)
        except Exception as e:
            print(f"    REDUCE cluster {cluster_id} FAILED: "
                  f"{type(e).__name__}: {e}")
            # Fallback: flatten all map themes without deduplication
            return ReducedThemeList(
                themes=[
                    {
                        "theme_label": t.theme_label,
                        "description": t.description,
                        "merged_from": [t.theme_label],
                    }
                    for mr in map_results
                    for t in mr.themes
                ]
            )

    # =========================================================================
    # MECE STEP (async)
    # =========================================================================

    async def _run_mece_step(
        self,
        cluster_id: int,
        themes: ReducedThemeList,
        n_ideas: int,
        context: PromptContext,
    ) -> ClusterMECETopicSet:
        """Apply MECE constraints with inclusion/exclusion boundaries."""
        prompt = self._build_mece_prompt(cluster_id, themes, n_ideas, context)

        try:
            return await self._llm_call(prompt, ClusterMECETopicSet, self._max_tokens_mece)
        except Exception as e:
            print(f"    MECE cluster {cluster_id} FAILED: "
                  f"{type(e).__name__}: {e}")
            # Fallback: convert consolidated themes to MECE topics without boundaries
            return ClusterMECETopicSet(
                topics=[
                    MECETopic(
                        topic_label=t.theme_label,
                        inclusion_definition=t.description,
                        exclusion_definition="",
                        key_expressions=[],
                    )
                    for t in themes.themes
                ]
            )

    # =========================================================================
    # PROMPT BUILDERS
    # =========================================================================

    def _build_map_prompt(
        self,
        cluster_id: int,
        ideas: List[str],
        batch_idx: int,
        total_batches: int,
        context: PromptContext,
    ) -> str:
        """Build prompt for the MAP step."""
        ideas_list = "\n".join(f"- {idea}" for idea in ideas)

        return MAP_THEMES_PROMPT.format(
            survey_question=context.survey_question,
            language=context.language,
            dataset_context_section=context.dataset_context_section,
            taxonomy_context=context.taxonomy_context,
            cluster_id=cluster_id,
            batch_number=batch_idx + 1,
            total_batches=total_batches,
            n_ideas=len(ideas),
            ideas_list=ideas_list,
        )

    def _build_reduce_prompt(
        self,
        cluster_id: int,
        map_results: List[MapBatchCOCs],
        context: PromptContext,
    ) -> str:
        """Build prompt for the REDUCE step."""
        # Format all batch themes into readable blocks
        sections = []
        for batch_idx, batch_result in enumerate(map_results):
            section_lines = [f"Batch {batch_idx + 1}:"]
            for theme in batch_result.themes:
                section_lines.append(f"  - {theme.theme_label}: {theme.description}")
            sections.append("\n".join(section_lines))

        batch_themes_list = "\n\n".join(sections)
        n_total_themes = sum(len(r.themes) for r in map_results)

        return REDUCE_THEMES_PROMPT.format(
            survey_question=context.survey_question,
            language=context.language,
            dataset_context_section=context.dataset_context_section,
            taxonomy_context=context.taxonomy_context,
            n_batches=len(map_results),
            n_total_themes=n_total_themes,
            batch_themes_list=batch_themes_list,
        )

    def _build_mece_prompt(
        self,
        cluster_id: int,
        themes: ReducedThemeList,
        n_ideas: int,
        context: PromptContext,
    ) -> str:
        """Build prompt for the MECE step."""
        themes_lines = []
        for i, theme in enumerate(themes.themes, 1):
            themes_lines.append(
                f"{i}. {theme.theme_label}\n"
                f"   Description: {theme.description}\n"
                f"   Merged from: {', '.join(theme.merged_from)}"
            )
        themes_list = "\n\n".join(themes_lines)

        return MECE_BOUNDARIES_PROMPT.format(
            survey_question=context.survey_question,
            language=context.language,
            dataset_context_section=context.dataset_context_section,
            taxonomy_context=context.taxonomy_context,
            cluster_id=cluster_id,
            n_ideas=n_ideas,
            n_themes=len(themes.themes),
            themes_list=themes_list,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _create_batches(self, ideas: List[str]) -> List[List[str]]:
        """Split ideas into batches of max batch_size."""
        return [
            ideas[i:i + self._batch_size]
            for i in range(0, len(ideas), self._batch_size)
        ]

    @staticmethod
    def _build_dataset_context_section(
        dataset_context: Optional[Dict[str, str]],
    ) -> str:
        """Build dataset context block for prompts."""
        if not dataset_context:
            return ""

        parts = []
        for key in ["domain", "entity", "topic", "perspective", "intent"]:
            value = dataset_context.get(key, "")
            if value:
                parts.append(f"{key.capitalize()}: {value}")

        if not parts:
            return ""

        return "<dataset_context>\n" + "\n".join(parts) + "\n</dataset_context>"

    @staticmethod
    def _build_taxonomy_context(
        taxonomy_axis: Optional[str],
        taxonomy_description: Optional[str],
    ) -> str:
        """Build taxonomy context block for prompts."""
        if not taxonomy_axis:
            return ""

        desc = taxonomy_description or "Not specified"
        return (
            f"<taxonomy_context>\n"
            f"Primary coding dimension: {taxonomy_axis}\n"
            f"Definition: {desc}\n"
            f"Topics MUST describe content within this dimension ONLY.\n"
            f"</taxonomy_context>"
        )
