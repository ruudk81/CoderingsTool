"""
Category Assignment for MECE Categories.

Assigns each idea to exactly one MECE category within its domain
partition. All partitions are processed concurrently with shared rate limiting.

Pipeline:
  1. Group ideas by partition (domain)
  2. Bootstrap: fetch rate limits + probe calls for latency/token measurement
  3. Little's Law → optimal concurrency
  4. Batch ideas per partition, submit all batches concurrently
  5. Collect assignments, build CodeAssignedModel list

Usage:
    from .code_assignment import CodeAssigner
    from config_steps.config_categories import AssignmentConfig

    assigner = CodeAssigner(
        config=AssignmentConfig(),
        embeddings_models=embeddings_models,
        mece_results=mece_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
    )
    results = assigner.assign_all()
"""

import asyncio
import time
from typing import Dict, List, Optional

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import (
    create_client, llm_create_async, ProbeResponse,
    RateLimits, extract_rate_limits_from_response,
)
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
    OPENAI_API_KEY, API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
)

import models
from config_steps.config_categories import AssignmentConfig, get_other_category_label
from prompts import (
    CATEGORY_ASSIGNMENT_PROMPT,
    CodeAssignmentBatch,
    MECECode,
    DomainSet,
)

# Reuse bootstrap utilities from map_reduce_mece
from .map_reduce_mece import (
    ApiLimits, compute_optimal_concurrency, bootstrap_measure_async,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


class CodeAssigner:
    """
    Assigns each idea to exactly one MECE category within its domain
    partition. All partitions are processed concurrently through a shared
    semaphore and rate limiter.
    """

    def __init__(
        self,
        config: AssignmentConfig,
        embeddings_models: List[models.EmbeddingsModel],
        mece_results: Dict[str, models.DomainResultModel],
        partition_set: DomainSet,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
        prompt_printer=None,
    ):
        self._config = config
        self._embeddings_models = embeddings_models
        self._mece_results = mece_results
        self._partition_set = partition_set
        self._extraction_metadata = extraction_metadata

        # Prompt capture (optional — pass PromptPrinter to enable)
        self._prompt_printer = prompt_printer
        self._captured_assignment = False

        # Shared async resources — initialized in _assign_all_async()
        self._client = None
        self._semaphore = None
        self._rate_limiter = None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def assign_all(self) -> List[models.CodeAssignedModel]:
        """Sync entry point. Returns list of CodeAssignedModel."""
        return asyncio.run(self._assign_all_async())

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _assign_all_async(self) -> List[models.CodeAssignedModel]:
        """Main async orchestration: bootstrap, batch, gather, collect."""
        verbose = self._config.verbose

        # 1. Create async instructor client
        self._client = create_client(
            model=self._config.assignment_model, async_mode=True
        )

        # 2. Group all ideas by partition (domain)
        partition_ideas = self._group_ideas_by_partition()
        total_ideas = sum(len(ideas) for ideas in partition_ideas.values())

        if verbose:
            print(f"\n{'='*70}")
            print(f"CATEGORY ASSIGNMENT")
            print(f"{'='*70}")
            print(f"  Ideas: {total_ideas} across {len(partition_ideas)} partitions")
            print(f"  Model: {self._config.assignment_model}")
            print(f"  Batch size: {self._config.assignment_batch_size}")

        if total_ideas == 0:
            print("  WARNING: No ideas to assign")
            return self._build_output_models({})

        # 3. Fetch rate limits from API
        if verbose:
            print("  Fetching rate limits from API...")
        limits = await self._fetch_rate_limits()

        if verbose:
            print(f"  Rate limits: TPM={limits.tokens_per_minute:,}, "
                  f"RPM={limits.requests_per_minute:,}")

        # 4. Bootstrap measurement (3 probe calls)
        first_partition = next(iter(sorted(partition_ideas.keys())))
        first_ideas = partition_ideas[first_partition][
            :min(self._config.assignment_batch_size, 3)
        ]
        first_cats = self._mece_results[first_partition].categories
        probe_prompt = self._build_assignment_prompt(
            first_ideas, first_cats, first_partition
        )

        if verbose:
            print("  Running bootstrap measurement (3 probe calls)...")

        probe_start = time.time()

        async def probe_fn():
            return await self._probe_call(probe_prompt)

        avg_latency, avg_tokens = await bootstrap_measure_async(
            probe_fn, n_probes=3
        )

        if verbose:
            print(f"  Probe time: {time.time() - probe_start:.1f}s")
            print(f"  Bootstrap: {avg_latency:.2f}s avg latency, "
                  f"{avg_tokens:.0f} avg tokens")

        # 5. Little's Law → optimal concurrency
        processing_config = DEFAULT_PROCESSING_CONFIG
        headroom = processing_config.rate_limit_headroom

        api_limits = ApiLimits(
            limits.tokens_per_minute, limits.requests_per_minute
        )
        little_law_conc = compute_optimal_concurrency(
            api_limits, avg_latency, avg_tokens,
            processing_config=processing_config,
            cap=processing_config.concurrency_cap_permissive,
            min_conc=processing_config.concurrency_min_permissive,
        )

        # Adaptive: 3x Little's Law for burst headroom, floor of 5
        max_concurrency = processing_config.concurrency_cap_default
        adaptive_min = min(
            processing_config.concurrency_min_default,
            max(little_law_conc * 3, 5),
        )
        optimal = min(max_concurrency, max(little_law_conc, adaptive_min))

        # Arrival rate for rate limiter
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

        # Count total batches
        total_batches = sum(
            len(self._create_batches(ideas))
            for ideas in partition_ideas.values()
            if partition_ideas  # always true, just for clarity
        )

        # 6. Initialize shared rate-limiting resources
        self._semaphore = asyncio.Semaphore(min(total_batches, optimal))
        self._rate_limiter = AsyncLimiter(
            1, time_period=1.0 / max(arrival_rate, 0.01)
        )

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Expected throughput: {arrival_rate:.1f}/s "
                  f"({bottleneck} limited)")
            print(f"  Optimal by Little's Law: {little_law_conc}")
            print(f"  Concurrency (semaphore): {min(total_batches, optimal)}")
            print(f"  Total batches: {total_batches}")

        # 7. Build flat task list across all partitions
        all_tasks = []
        task_metadata = []

        for partition_name in sorted(partition_ideas.keys()):
            ideas = partition_ideas[partition_name]
            mece_result = self._mece_results.get(partition_name)

            if not mece_result or not mece_result.categories:
                if verbose:
                    print(f"  WARNING: No MECE categories for "
                          f"'{partition_name}', skipping {len(ideas)} ideas")
                continue

            batches = self._create_batches(ideas)
            for batch_idx, batch in enumerate(batches):
                task = self._assign_batch(
                    batch, mece_result.categories, partition_name
                )
                all_tasks.append(task)
                task_metadata.append(
                    (partition_name, batch_idx, len(batches))
                )

        # 8. Execute all batches concurrently
        if verbose:
            print(f"\n  Processing {len(all_tasks)} batches across "
                  f"{len(partition_ideas)} partitions...")

        start_time = time.time()
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # 9. Collect assignments into lookup: idea_id → CodeAssignment
        assignment_lookup = {}
        failed_count = 0
        mismatched_ids = 0

        for i, result in enumerate(results):
            partition_name, batch_idx, n_total = task_metadata[i]
            if isinstance(result, Exception):
                failed_count += 1
                if verbose:
                    print(f"    FAILED: {partition_name} batch "
                          f"{batch_idx+1}/{n_total}: "
                          f"{type(result).__name__}: {result}")
            else:
                for assignment in result.assignments:
                    assignment_lookup[assignment.idea_id] = assignment

        assigned_count = len(assignment_lookup)
        if verbose:
            print(f"\n  Assignment complete in {elapsed:.1f}s")
            print(f"  Assigned: {assigned_count}/{total_ideas} ideas")
            if failed_count:
                print(f"  Failed batches: {failed_count}")

        # 10. Build output models
        output = self._build_output_models(assignment_lookup)

        # Print per-partition summary
        if verbose:
            self._print_assignment_summary(output)

        return output

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    async def _assign_batch(
        self,
        ideas: List,
        categories: List[MECECode],
        partition_name: str,
    ) -> CodeAssignmentBatch:
        """Rate-limited LLM call for one batch of ideas."""
        prompt = self._build_assignment_prompt(
            ideas, categories, partition_name
        )

        # Prompt capture (first batch only)
        if self._prompt_printer is not None and not self._captured_assignment:
            self._prompt_printer.capture_prompt(
                step_name="code_assignment",
                utility_name="CodeAssigner",
                prompt_content=prompt,
                prompt_type="code_assignment",
                metadata={
                    "model": self._config.assignment_model,
                    "language": (
                        self._extraction_metadata.lang
                        if self._extraction_metadata else "Dutch"
                    ),
                    "partition_name": partition_name,
                    "n_ideas": len(ideas),
                    "n_categories": len(categories),
                }
            )
            self._captured_assignment = True

        async with self._semaphore:
            async with self._rate_limiter:
                return await llm_create_async(
                    client=self._client,
                    model=self._config.assignment_model,
                    prompt=prompt,
                    response_model=CodeAssignmentBatch,
                    temperature=self._config.assignment_temperature,
                    max_tokens=self._config.assignment_max_tokens,
                )

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_assignment_prompt(
        self,
        ideas: List,
        categories: List[MECECode],
        partition_name: str,
    ) -> str:
        """Build the full assignment prompt for a batch of ideas."""
        # Survey context from extraction metadata
        survey_question = ""
        language = "Dutch"
        dataset_context_section = ""

        if self._extraction_metadata:
            survey_question = self._extraction_metadata.var_lab or ""
            language = self._extraction_metadata.lang or "Dutch"
            parts = []
            for f in ('domain', 'entity', 'topic', 'perspective', 'intent'):
                val = getattr(self._extraction_metadata, f, None)
                if val:
                    parts.append(f"{f.capitalize()}: {val}")
            if parts:
                dataset_context_section = "\n".join(parts)

        # Find partition inclusion definition
        partition_inclusion = ""
        for p in self._partition_set.partitions:
            if p.partition_name == partition_name:
                partition_inclusion = p.inclusion_definition
                break

        # Resolve Other category label from language
        other_label = get_other_category_label(language)

        categories_block = self._build_categories_block(
            categories, other_label if self._config.include_other_category else None
        )
        ideas_block = self._build_ideas_block(ideas)

        return CATEGORY_ASSIGNMENT_PROMPT.format(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            partition_name=partition_name,
            partition_inclusion=partition_inclusion,
            categories_block=categories_block,
            ideas_block=ideas_block,
            other_category_label=other_label,
        )

    @staticmethod
    def _build_categories_block(
        categories: List[MECECode],
        other_label: Optional[str] = None,
    ) -> str:
        """Format MECE categories for the prompt.

        If other_label is provided, appends a fallback "Other" category.
        """
        lines = []
        for i, cat in enumerate(categories, 1):
            signals = (
                ", ".join(cat.diagnostic_signals)
                if cat.diagnostic_signals else "(none)"
            )
            block = (
                f"[{i}] {cat.category_label}\n"
                f"    Inclusion: {cat.inclusion_definition}\n"
                f"    Boundary test: {cat.boundary_test}\n"
                f"    Diagnostic signals: {signals}"
            )
            if cat.tiebreaker_rules:
                tb_lines = "\n".join(
                    f"      - {r}" for r in cat.tiebreaker_rules
                )
                block += f"\n    Tiebreaker rules:\n{tb_lines}"
            lines.append(block)

        if other_label:
            n = len(categories) + 1
            lines.append(
                f"[{n}] {other_label}\n"
                f"    Inclusion: Ideas that do not clearly fit any of the "
                f"above categories after applying all boundary tests.\n"
                f"    Boundary test: Do all other categories' boundary tests "
                f"fail for this idea?\n"
                f"    Diagnostic signals: no matching signals from any category"
            )

        return "\n\n".join(lines)

    @staticmethod
    def _build_ideas_block(ideas: List) -> str:
        """Format ideas for the prompt."""
        lines = []
        for idea in ideas:
            lines.append(
                f"- idea_id: {idea.idea_id}\n"
                f"  idea: {idea.idea}\n"
                f"  interpretation: {idea.interpretation}\n"
                f"  abstraction: {idea.abstraction}"
            )
        return "\n".join(lines)

    # =========================================================================
    # IDEA GROUPING & BATCHING
    # =========================================================================

    def _group_ideas_by_partition(
        self,
    ) -> Dict[str, List[models.EmbeddingsSubmodel]]:
        """Group all ideas by their domain (= partition)."""
        partitions: Dict[str, List] = {}
        for resp in self._embeddings_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                ct = (idea.domain or '').strip().lower()
                if not ct:
                    continue
                if ct not in partitions:
                    partitions[ct] = []
                partitions[ct].append(idea)
        return partitions

    def _create_batches(self, ideas: list) -> list:
        """Split ideas into batches of assignment_batch_size."""
        bs = self._config.assignment_batch_size
        return [ideas[i:i + bs] for i in range(0, len(ideas), bs)]

    # =========================================================================
    # OUTPUT MODEL CONSTRUCTION
    # =========================================================================

    def _build_output_models(
        self,
        assignment_lookup: dict,
    ) -> List[models.CodeAssignedModel]:
        """Build CodeAssignedModel list preserving response structure.

        Converts each EmbeddingsSubmodel idea to CodeAssignedSubmodel,
        injecting assignment data via idea_id lookup.
        """
        output = []
        for resp in self._embeddings_models:
            new_ideas = []
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    assignment = assignment_lookup.get(idea.idea_id)
                    ct = (idea.domain or '').strip().lower()

                    # Build CodeAssignedSubmodel from EmbeddingsSubmodel
                    idea_data = idea.model_dump()
                    new_idea = models.CodeAssignedSubmodel(
                        **{k: v for k, v in idea_data.items()
                           if k in models.CodeAssignedSubmodel.model_fields},
                        assigned_category=(
                            assignment.assigned_category
                            if assignment else None
                        ),
                        category_confidence=(
                            assignment.confidence
                            if assignment else None
                        ),
                        category_rationale=(
                            assignment.rationale
                            if assignment else None
                        ),
                        partition_name=ct if ct else None,
                    )
                    new_ideas.append(new_idea)

            # Build CodeAssignedModel from EmbeddingsModel
            resp_data = resp.model_dump()
            new_resp = models.CodeAssignedModel(
                **{k: v for k, v in resp_data.items()
                   if k in models.CodeAssignedModel.model_fields
                   and k != 'response_ideas'},
                response_ideas=new_ideas,
            )
            output.append(new_resp)

        return output

    # =========================================================================
    # RATE LIMIT HELPERS
    # =========================================================================

    async def _fetch_rate_limits(self) -> RateLimits:
        """Fetch rate limits from API headers."""
        from openai import AsyncOpenAI

        if API_PROVIDER == "azure":
            from config import (
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_DEPLOYMENT_NAME,
            )
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=(
                    f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
                    f"{AZURE_OPENAI_DEPLOYMENT_NAME}/"
                ),
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self._config.assignment_model

        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        limits = extract_rate_limits_from_response(response)

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if self._config.verbose:
                print(f"  WARNING: Using fallback rate limits "
                      f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            return RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
            )
        return limits

    async def _probe_call(self, prompt: str):
        """Probe call for bootstrap measurement — returns usage dict."""
        resp = await llm_create_async(
            client=self._client,
            model=self._config.assignment_model,
            prompt=prompt,
            response_model=ProbeResponse,
            temperature=self._config.assignment_temperature,
            track_usage=False,
        )
        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        input_tokens = (
            getattr(u, "input_tokens", 0)
            or getattr(u, "prompt_tokens", 0)
        )
        output_tokens = (
            getattr(u, "output_tokens", 0)
            or getattr(u, "completion_tokens", 0)
        )
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }

    # =========================================================================
    # REPORTING
    # =========================================================================

    @staticmethod
    def _print_assignment_summary(
        output: List[models.CodeAssignedModel],
    ):
        """Print per-partition assignment summary."""
        partition_stats: Dict[str, Dict] = {}

        for resp in output:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                pt = idea.partition_name or "(unknown)"
                if pt not in partition_stats:
                    partition_stats[pt] = {
                        "total": 0,
                        "assigned": 0,
                        "confidences": [],
                        "categories": {},
                    }
                stats = partition_stats[pt]
                stats["total"] += 1
                if idea.assigned_category:
                    stats["assigned"] += 1
                    stats["confidences"].append(
                        idea.category_confidence or 0.0
                    )
                    cat = idea.assigned_category
                    stats["categories"][cat] = (
                        stats["categories"].get(cat, 0) + 1
                    )

        print(f"\n  {'─'*60}")
        print(f"  ASSIGNMENT SUMMARY")
        print(f"  {'─'*60}")

        for pt in sorted(partition_stats.keys()):
            stats = partition_stats[pt]
            avg_conf = (
                sum(stats["confidences"]) / len(stats["confidences"])
                if stats["confidences"] else 0.0
            )
            print(f"\n  Partition: {pt}")
            print(f"    Assigned: {stats['assigned']}/{stats['total']}")
            print(f"    Avg confidence: {avg_conf:.2f}")
            print(f"    Categories ({len(stats['categories'])}):")
            for cat, count in sorted(
                stats["categories"].items(),
                key=lambda x: -x[1],
            ):
                print(f"      {cat}: {count}")
