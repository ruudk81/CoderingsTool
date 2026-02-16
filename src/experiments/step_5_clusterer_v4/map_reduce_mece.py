"""
Stage 3: Object-Aware Map-Reduce MECE — V4

Per-object topic extraction via 3-step LLM pipeline with object context injection.

All objects are processed CONCURRENTLY. All LLM calls share a single semaphore
and rate limiter for efficient throughput control.

Pipeline for each MECE object:
  1. MAP:    Batch ideas, identify COCs per batch (with object context)
  2. REDUCE: Consolidate COCs across batches (with object context)
             Skipped for single-batch objects.
  3. MECE:   Apply MECE constraints with inclusion/exclusion boundaries (with object context)

Key difference from V3: All prompts include an {object_context} block with the
object's label, inclusion/exclusion definitions, and peer objects to prevent
topic bleed across object boundaries.

Usage:
    from .map_reduce_mece import ObjectAwareMapReduceMECE
    from .config_clusterer_exp import ClustererConfig

    processor = ObjectAwareMapReduceMECE(config)
    results = processor.process_all_objects(
        object_mappings={"obj1": mapping1, ...},
        mece_objects=mece_objects,
        survey_question="What do you think about X?",
    )

Adapted from V3's MapReduceMECE.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import create_client, llm_create_async

from .config_clusterer_exp import ClustererConfig
from .object_mapper import ObjectIdeaMapping
from .prompts_exp import (
    OBJECT_AWARE_MAP_THEMES_PROMPT,
    OBJECT_AWARE_REDUCE_THEMES_PROMPT,
    OBJECT_AWARE_MECE_BOUNDARIES_PROMPT,
    MapBatchCOCs,
    ReducedThemeList,
    ClusterMECETopicSet,
    MECETopic,
    MECEVerification,
    MECEObjectSet,
    MECEObjectDescription,
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
class ObjectContext:
    """Object-specific context injected into all prompts."""
    object_label: str
    object_inclusion: str
    object_boundary_test: str
    peer_objects_list: str
    grouping_instruction: str = ""


@dataclass
class ObjectMECEResult:
    """Complete map-reduce MECE result for a single object."""
    object_label: str
    n_ideas: int
    n_batches: int
    reduce_skipped: bool
    topics: List[MECETopic]
    mece_verifications: List[MECEVerification] = field(default_factory=list)


class ObjectAwareMapReduceMECE:
    """
    Per-object Map-Reduce MECE topic extractor with object context injection.

    All objects are processed concurrently. All LLM calls share a single
    async client, semaphore, and rate limiter for efficient throughput.

    Three-step pipeline for each object:
    1. MAP:    Batch all ideas, identify COCs per batch (async, with object context)
    2. REDUCE: Consolidate COCs across batches (async, with object context)
    3. MECE:   Apply inclusion/exclusion boundaries (async, with object context)
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

    def process_all_objects(
        self,
        object_mappings: Dict[str, ObjectIdeaMapping],
        mece_objects: MECEObjectSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        taxonomy_axis: Optional[str] = None,
        taxonomy_description: Optional[str] = None,
        taxonomy_axis_info: Optional[Dict] = None,
        taxonomy_actionable_type: Optional[str] = None,
        grouping_instructions: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ) -> Dict[str, ObjectMECEResult]:
        """
        Process all MECE objects through the map-reduce-MECE pipeline.

        All objects are processed concurrently. All LLM calls share
        one semaphore + rate limiter for throughput control.

        Args:
            object_mappings: Dict mapping object_label → ObjectIdeaMapping
            mece_objects: Full MECEObjectSet (for building peer object context)
            survey_question: The survey question for prompt context
            language: Language for output (default: Dutch)
            dataset_context: Optional dict with domain, entity, topic, etc.
            taxonomy_axis: Optional taxonomy primary axis
            taxonomy_description: Optional taxonomy axis description
            taxonomy_axis_info: Optional dict from template_lookup (allowed_concepts, etc.)
            taxonomy_actionable_type: Optional actionable type from ExtractionMetadata
            verbose: Print progress info

        Returns:
            Dict mapping object_label → ObjectMECEResult
        """
        print(f"\n{'='*70}")
        print(f"STAGE 3: Object-Aware Map-Reduce MECE")
        print(f"{'='*70}")

        # Build shared prompt context
        dataset_context_section = self._build_dataset_context_section(dataset_context)
        taxonomy_context = self._build_taxonomy_context(
            taxonomy_axis, taxonomy_description,
            taxonomy_axis_info=taxonomy_axis_info,
            taxonomy_actionable_type=taxonomy_actionable_type,
        )

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            taxonomy_context=taxonomy_context,
        )

        # Build per-object context (with peer objects + grouping instructions)
        object_contexts = self._build_all_object_contexts(
            mece_objects, grouping_instructions=grouping_instructions
        )

        # Filter empty mappings
        active_objects = {
            label: mapping for label, mapping in object_mappings.items()
            if mapping.idea_texts
        }

        if verbose:
            total_ideas = sum(m.idea_count for m in active_objects.values())
            total_batches = sum(
                len(self._create_batches(list(dict.fromkeys(m.idea_texts))))
                for m in active_objects.values()
            )
            print(f"  Processing {len(active_objects)} objects concurrently "
                  f"({total_ideas} ideas, ~{total_batches} map batches)")
            print(f"  Concurrency: {self._concurrency} in-flight, {self._rpm_limit} RPM")

        return asyncio.run(
            self._process_all_async(active_objects, object_contexts, prompt_context, verbose)
        )

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _process_all_async(
        self,
        object_mappings: Dict[str, ObjectIdeaMapping],
        object_contexts: Dict[str, ObjectContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> Dict[str, ObjectMECEResult]:
        """Main async entry: create shared resources, gather all objects."""
        self._client = create_client(model=self._model, async_mode=True)
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._rate_limiter = AsyncLimiter(self._rpm_limit, time_period=60)

        start_time = time.time()

        tasks = {
            label: self._process_single_object(
                label, mapping.idea_texts, object_contexts[label], prompt_context, verbose
            )
            for label, mapping in sorted(object_mappings.items())
        }

        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

        results = {}
        for label, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                print(f"  Object '{label}' FAILED: {type(result).__name__}: {result}")
            else:
                results[label] = result

        elapsed = time.time() - start_time
        if verbose:
            total_topics = sum(len(r.topics) for r in results.values())
            print(f"\n  All objects done in {elapsed:.1f}s → "
                  f"{total_topics} MECE topics across {len(results)} objects")

        return results

    # =========================================================================
    # SINGLE OBJECT PIPELINE (async)
    # =========================================================================

    async def _process_single_object(
        self,
        object_label: str,
        ideas: List[str],
        obj_context: ObjectContext,
        prompt_context: PromptContext,
        verbose: bool = False,
    ) -> ObjectMECEResult:
        """Run the 3-step pipeline for a single object (async)."""

        # Step 0: Deduplicate and batch
        unique_ideas = list(dict.fromkeys(ideas))
        batches = self._create_batches(unique_ideas)
        n_batches = len(batches)

        if verbose:
            dedup_note = f" (dedup {len(ideas)}→{len(unique_ideas)})" if len(unique_ideas) < len(ideas) else ""
            print(f"    Object '{object_label}': {len(unique_ideas)} ideas{dedup_note}, "
                  f"{n_batches} batch(es)")

        # Step 1: MAP
        map_results = await self._run_map_step(
            object_label, batches, obj_context, prompt_context
        )
        total_themes = sum(len(r.themes) for r in map_results)

        # Step 2: REDUCE (skip for single batch)
        reduce_skipped = (n_batches == 1)

        if reduce_skipped:
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
            consolidated = await self._run_reduce_step(
                object_label, map_results, obj_context, prompt_context
            )

        # Step 3: MECE
        mece_result = await self._run_mece_step(
            object_label, consolidated, len(unique_ideas), obj_context, prompt_context
        )

        if verbose:
            print(f"    Object '{object_label}' → {len(mece_result.topics)} MECE topics "
                  f"(map: {total_themes} themes, "
                  f"{'reduce skipped' if reduce_skipped else f'reduced to {len(consolidated.themes)}'})")

        return ObjectMECEResult(
            object_label=object_label,
            n_ideas=len(unique_ideas),
            n_batches=n_batches,
            reduce_skipped=reduce_skipped,
            topics=mece_result.topics,
            mece_verifications=mece_result.mece_verifications if hasattr(mece_result, 'mece_verifications') else [],
        )

    # =========================================================================
    # SHARED LLM CALL
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
    # MAP STEP
    # =========================================================================

    async def _run_map_step(
        self,
        object_label: str,
        batches: List[List[str]],
        obj_context: ObjectContext,
        prompt_context: PromptContext,
    ) -> List[MapBatchCOCs]:
        """Send each batch to LLM concurrently through shared rate limiter."""
        results = [None] * len(batches)

        async def process_batch(batch_idx: int, ideas: List[str]):
            prompt = self._build_map_prompt(
                object_label, ideas, batch_idx, len(batches),
                obj_context, prompt_context
            )
            try:
                result = await self._llm_call(prompt, MapBatchCOCs, self._max_tokens_map)
                results[batch_idx] = result
            except Exception as e:
                print(f"    MAP object '{object_label}' batch {batch_idx + 1}/{len(batches)} "
                      f"FAILED: {type(e).__name__}: {e}")
                results[batch_idx] = MapBatchCOCs(themes=[])

        await asyncio.gather(*(
            process_batch(i, batch) for i, batch in enumerate(batches)
        ))

        return [r for r in results if r is not None]

    # =========================================================================
    # REDUCE STEP
    # =========================================================================

    async def _run_reduce_step(
        self,
        object_label: str,
        map_results: List[MapBatchCOCs],
        obj_context: ObjectContext,
        prompt_context: PromptContext,
    ) -> ReducedThemeList:
        """Consolidate themes from all batches into one unified list."""
        prompt = self._build_reduce_prompt(
            object_label, map_results, obj_context, prompt_context
        )

        try:
            return await self._llm_call(prompt, ReducedThemeList, self._max_tokens_reduce)
        except Exception as e:
            print(f"    REDUCE object '{object_label}' FAILED: "
                  f"{type(e).__name__}: {e}")
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
    # MECE STEP
    # =========================================================================

    async def _run_mece_step(
        self,
        object_label: str,
        themes: ReducedThemeList,
        n_ideas: int,
        obj_context: ObjectContext,
        prompt_context: PromptContext,
    ) -> ClusterMECETopicSet:
        """Apply MECE constraints with inclusion/exclusion boundaries."""
        prompt = self._build_mece_prompt(
            object_label, themes, n_ideas, obj_context, prompt_context
        )

        try:
            return await self._llm_call(prompt, ClusterMECETopicSet, self._max_tokens_mece)
        except Exception as e:
            print(f"    MECE object '{object_label}' FAILED: "
                  f"{type(e).__name__}: {e}")
            return ClusterMECETopicSet(
                topics=[
                    MECETopic(
                        topic_label=t.theme_label,
                        inclusion_definition=t.description,
                        boundary_test="",
                        diagnostic_signals=[],
                        key_expressions=[],
                        tiebreaker_rules=[],
                    )
                    for t in themes.themes
                ],
                mece_verifications=[],
            )

    # =========================================================================
    # PROMPT BUILDERS
    # =========================================================================

    def _build_map_prompt(
        self,
        object_label: str,
        ideas: List[str],
        batch_idx: int,
        total_batches: int,
        obj_context: ObjectContext,
        prompt_context: PromptContext,
    ) -> str:
        """Build prompt for the MAP step with object context."""
        ideas_list = "\n".join(f"- {idea}" for idea in ideas)

        # Build grouping instruction block (non-empty only for semantic_category mode)
        gi = obj_context.grouping_instruction
        grouping_block = f"\n{gi}" if gi else ""

        return OBJECT_AWARE_MAP_THEMES_PROMPT.format(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            taxonomy_context=prompt_context.taxonomy_context,
            object_label=obj_context.object_label,
            object_inclusion=obj_context.object_inclusion,
            object_boundary_test=obj_context.object_boundary_test,
            peer_objects_list=obj_context.peer_objects_list,
            grouping_instruction=grouping_block,
            batch_number=batch_idx + 1,
            total_batches=total_batches,
            n_ideas=len(ideas),
            ideas_list=ideas_list,
        )

    def _build_reduce_prompt(
        self,
        object_label: str,
        map_results: List[MapBatchCOCs],
        obj_context: ObjectContext,
        prompt_context: PromptContext,
    ) -> str:
        """Build prompt for the REDUCE step with object context."""
        sections = []
        for batch_idx, batch_result in enumerate(map_results):
            section_lines = [f"Batch {batch_idx + 1}:"]
            for theme in batch_result.themes:
                section_lines.append(f"  - {theme.theme_label}: {theme.description}")
                if hasattr(theme, 'recognition_cue') and theme.recognition_cue:
                    section_lines.append(f"    Recognition: A coder assigns here when it {theme.recognition_cue}")
            sections.append("\n".join(section_lines))

        batch_themes_list = "\n\n".join(sections)
        n_total_themes = sum(len(r.themes) for r in map_results)

        gi = obj_context.grouping_instruction
        grouping_block = f"\n{gi}" if gi else ""

        return OBJECT_AWARE_REDUCE_THEMES_PROMPT.format(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            taxonomy_context=prompt_context.taxonomy_context,
            object_label=obj_context.object_label,
            object_inclusion=obj_context.object_inclusion,
            object_boundary_test=obj_context.object_boundary_test,
            peer_objects_list=obj_context.peer_objects_list,
            grouping_instruction=grouping_block,
            n_batches=len(map_results),
            n_total_themes=n_total_themes,
            batch_themes_list=batch_themes_list,
        )

    def _build_mece_prompt(
        self,
        object_label: str,
        themes: ReducedThemeList,
        n_ideas: int,
        obj_context: ObjectContext,
        prompt_context: PromptContext,
    ) -> str:
        """Build prompt for the MECE step with object context."""
        themes_lines = []
        for i, theme in enumerate(themes.themes, 1):
            themes_lines.append(
                f"{i}. {theme.theme_label}\n"
                f"   Description: {theme.description}\n"
                f"   Merged from: {', '.join(theme.merged_from)}"
            )
        themes_list = "\n\n".join(themes_lines)

        gi = obj_context.grouping_instruction
        grouping_block = f"\n{gi}" if gi else ""

        return OBJECT_AWARE_MECE_BOUNDARIES_PROMPT.format(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            taxonomy_context=prompt_context.taxonomy_context,
            object_label=obj_context.object_label,
            object_inclusion=obj_context.object_inclusion,
            object_boundary_test=obj_context.object_boundary_test,
            peer_objects_list=obj_context.peer_objects_list,
            grouping_instruction=grouping_block,
            n_ideas=n_ideas,
            n_themes=len(themes.themes),
            themes_list=themes_list,
        )

    # =========================================================================
    # OBJECT CONTEXT BUILDERS
    # =========================================================================

    def _build_all_object_contexts(
        self,
        mece_objects: MECEObjectSet,
        grouping_instructions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ObjectContext]:
        """Build ObjectContext for each MECE object with peer objects listed."""
        contexts = {}
        all_objects = mece_objects.topics

        for obj in all_objects:
            # Build peer objects list (all objects except this one)
            peer_lines = []
            for peer in all_objects:
                if peer.topic_label == obj.topic_label:
                    continue
                peer_lines.append(
                    f"- {peer.topic_label}: {peer.inclusion_definition}"
                )

            # Look up grouping instruction for this object (semantic_category mode)
            gi = ""
            if grouping_instructions:
                gi = grouping_instructions.get(obj.topic_label, "")

            contexts[obj.topic_label] = ObjectContext(
                object_label=obj.topic_label,
                object_inclusion=obj.inclusion_definition,
                object_boundary_test=obj.boundary_test,
                peer_objects_list="\n".join(peer_lines) if peer_lines else "(no peer objects)",
                grouping_instruction=gi,
            )

        return contexts

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
        taxonomy_axis_info: Optional[Dict] = None,
        taxonomy_actionable_type: Optional[str] = None,
    ) -> str:
        """
        Build taxonomy context block for prompts.

        When taxonomy_axis_info (from template_lookup.py) is available, builds an
        enriched block with allowed_concepts, excluded_concepts, and full
        dimension_description. Falls back to the thin 3-line version otherwise.
        """
        if not taxonomy_axis:
            return ""

        # Enriched path: use template_lookup data
        if taxonomy_axis_info:
            noun_desc = taxonomy_axis_info.get("noun_phrase_descriptor", taxonomy_axis)
            dimension_desc = taxonomy_axis_info.get("dimension_description", taxonomy_description or "Not specified")
            allowed = taxonomy_axis_info.get("allowed_concepts", [])
            excluded = taxonomy_axis_info.get("excluded_concepts", [])

            lines = [
                "<taxonomy_context>",
                f"Primary coding dimension: {noun_desc}",
                f"Dimension: {dimension_desc}",
            ]
            if allowed:
                lines.append(f"Allowed concept types: {', '.join(allowed)}")
            if excluded:
                lines.append(f"Excluded concept types: {', '.join(excluded)}")
            if taxonomy_actionable_type:
                lines.append(f"Actionable type: {taxonomy_actionable_type}")
            lines.append("")
            lines.append("Topics MUST describe content within this dimension ONLY.")
            if excluded:
                lines.append(f"Do NOT create topics/objects about: {', '.join(excluded)}")
            lines.append("</taxonomy_context>")
            return "\n".join(lines)

        # Fallback: thin taxonomy context
        desc = taxonomy_description or "Not specified"
        return (
            f"<taxonomy_context>\n"
            f"Primary coding dimension: {taxonomy_axis}\n"
            f"Definition: {desc}\n"
            f"Topics MUST describe content within this dimension ONLY.\n"
            f"</taxonomy_context>"
        )
