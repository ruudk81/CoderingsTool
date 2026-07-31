"""
Codebook Generator: Code generation and consolidation pipeline (P8-P9).

Pipeline (3 stages):
  EMB. Embedding + representative sample selection (per attribute per valence)
  P8.  Code Generation from Attributes (per domain) — derive codebook codes
  P9.  Codebook Consolidation (cross-domain) — merge into final MECE codebook

Accepts taxonomy results from step_4_classifier as input.
Uses SmoothRequester for rate-limited LLM dispatch.
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple


from pydantic import BaseModel, Field, create_model
import nest_asyncio

from utils.smoothRequester import SmoothRequester

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()
from utils.llm import (
    RateLimits, fetch_rate_limits, token_tracker,
)
from config import (
    FALLBACK_TPM, FALLBACK_RPM, get_reasoning_params,
)

from pipeline.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)

from models import DomainSet

from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from .prompts_codeGenerator import (
    # P8: Code Generation from Attributes
    build_code_from_attributes_prompt,
    CodeGenerationFromAttributesResult,
    # P9: Codebook Consolidation
    build_codebook_consolidation_prompt,
    CodebookConsolidationResult,
    ConsolidatedCode,
    # Enriched attributes for P8 representative samples
    EnrichedAttribute,
)
import numpy as np
from utils.clusterer import estimate_theme_count
from utils.embedder import SharedEmbedder, format_idea_text, find_representative_samples


# =============================================================================
# SHARED DATACLASSES
# =============================================================================

@dataclass
class PromptContext:
    """Shared context passed to all prompt formatting methods."""
    survey_question: str
    language: str
    dataset_context_section: str
    dimension_name: str
    dimension_description: str
    dimension_def: Optional[DimensionDefinition] = None


@dataclass
class DomainContext:
    """Partition-specific context."""
    partition_name: str
    partition_definition: str


@dataclass
class TaxonomyResult:
    """Taxonomy input from step 4 — only the fields P8/P9 consume."""
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, list]]  # domain -> {facet -> [DiscoveredAttribute]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name


@dataclass
class CodebookResult:
    """Output of codebook stages P8-P9."""
    codes: List[ConsolidatedCode]
    codebook_narrative: str


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class CodebookGenerator:
    """
    Codebook Generator: Code generation and consolidation pipeline (P8-P9).

    Pipeline (3 stages):
    EMB. REPRESENTATIVE SAMPLES:         Embed ideas, select per-attribute medoids
    P8.  CODE GENERATION:                Per domain, derive codes from enriched attributes
    P9.  CODEBOOK CONSOLIDATION:         Cross-domain, merge into MECE codebook

    All LLM calls dispatched via SmoothRequester.
    """

    def __init__(self, config: CodebookConfig, prompt_printer=None, cost_tracker=None):
        self._config = config
        self.cost_tracker = cost_tracker
        self._model_p8 = config.model_p8
        self._model_p9 = config.model_p9

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_5_code_generator", {
                "p8_code_generation": self._model_p8,
                "p9_codebook_consolidation": self._model_p9,
            })

        self._temperature_p8 = config.temperature_p8
        self._temperature_p9 = config.temperature_p9
        self._max_tokens_code_from_attributes = config.max_tokens_code_from_attributes
        self._max_tokens_codebook_consolidation = config.max_tokens_codebook_consolidation

        # Prompt capture (optional)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        # Embeddings for cache (populated during processing)
        self._idea_embeddings: Dict[str, any] = {}

        # (domain, attribute_name) -> A# id, for resolving P9 provenance to
        # stable ids at parse time (set per run via generate()).
        self._attribute_ids: Dict[Tuple[str, str], str] = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate(
        self,
        taxonomy_result: TaxonomyResult,
        partition_set: DomainSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        dimension_name: str = "",
        dimension_description: str = "",
        verbose: bool = False,
        prompt_printer=None,
        classified_ideas: Optional[List] = None,
        template_prefix: str = "",
        attribute_ids: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> CodebookResult:
        """Run codebook stages (P8-P9) from a TaxonomyResult.

        Args:
            taxonomy_result: Output from TaxonomyClassifier.process()
            partition_set: Domain partition definitions
            survey_question: The survey question being coded
            language: Language of the survey responses
            dataset_context: Optional dataset context dict
            dimension_name: Name of the dimension being analyzed
            dimension_description: Description of the dimension
            verbose: Print progress information
            prompt_printer: Optional prompt printer (overrides __init__ printer)
            classified_ideas: Taxonomy-classified ideas from step 4 (for embedding + representative samples)
            template_prefix: Template prefix from extraction metadata
            attribute_ids: (domain, attribute_name) -> A# id, for resolving P9
                provenance to stable ids at parse time (absent ids are filled
                legacy-style by ensure_codebook_ids at cache-save)
        """
        if prompt_printer is not None:
            self._prompt_printer = prompt_printer
        self._attribute_ids = attribute_ids or {}

        print(f"\n{'='*70}")
        print(f"CODEBOOK GENERATION (P8-P9)")
        print(f"{'='*70}")

        # Resolve dimension definition
        dimension_def = None
        if dimension_name:
            dimension_def = get_dimension(dimension_name)
            if dimension_def and verbose:
                print(f"  Dimension: {dimension_name}")
            elif not dimension_def and verbose:
                print(f"  WARNING: No DimensionDefinition found for '{dimension_name}'")

        dataset_context_section = self._build_dataset_context_section(dataset_context)

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            dimension_def=dimension_def,
        )

        partition_contexts = self._build_all_partition_contexts(partition_set)

        # Flatten classified ideas for embedding
        ideas_flat = []
        if classified_ideas:
            for resp_model in classified_ideas:
                if resp_model.response_ideas:
                    ideas_flat.extend(resp_model.response_ideas)

        async def _run():
            return await self._process_codebook_async(
                taxonomy_result, partition_contexts, prompt_context, verbose,
                ideas_flat=ideas_flat, template_prefix=template_prefix,
            )

        return asyncio.run(_run())

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _process_codebook_async(
        self,
        taxonomy: TaxonomyResult,
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
        ideas_flat: Optional[List] = None,
        template_prefix: str = "",
    ) -> CodebookResult:
        """Codebook stages: embedding + P8 + P9."""
        partition_assignments = taxonomy.partition_assignments
        domain_facet_attributes = taxonomy.partition_attributes
        attribute_assignments = taxonomy.attribute_assignments

        start_time = time.time()

        # Fetch rate limits once for all phases
        if verbose:
            print("  Fetching rate limits from API...")
        limits, has_headers = await fetch_rate_limits(self._model_p8)
        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if verbose:
                print(f"  WARNING: Using fallback rate limits "
                      f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            limits = RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
            )
        elif verbose:
            print(f"  Fetched from API: TPM={limits.tokens_per_minute:,}, "
                  f"RPM={limits.requests_per_minute:,}")

        # =================================================================
        # EMBEDDING + REPRESENTATIVE SAMPLES
        # =================================================================
        enriched_by_domain = {}
        attribute_valence_counts: Dict[str, Dict[str, int]] = {}
        self._idea_embeddings = {}

        if ideas_flat:
            if verbose:
                print(f"\n  Computing representative samples "
                      f"({len(ideas_flat)} ideas, code_source={self._config.code_source})...")

            representatives, group_counts, self._idea_embeddings = await self._compute_representative_samples(
                ideas_flat, attribute_assignments, verbose,
            )

            # Per-attribute valence breakdown for P9's prevalence-gated valence policy
            for (attr_name, vgroup), count in group_counts.items():
                attribute_valence_counts.setdefault(attr_name, {})[vgroup] = count

            enriched_by_domain = self._enrich_attributes_with_samples(
                domain_facet_attributes, representatives, group_counts,
            )

            if verbose:
                n_enriched = sum(
                    len(attrs)
                    for facet_map in enriched_by_domain.values()
                    for attrs in facet_map.values()
                )
                print(f"  Enriched {n_enriched} attributes with representative samples")
                print(f"  Valence example threshold: floor(log(attribute_total)) — counts always shown")
                print(f"  Cached {len(self._idea_embeddings)} idea embeddings")
        elif verbose:
            print(f"\n  No classified ideas available — skipping representative samples")

        # =================================================================
        # PHASE 8 (P8): Per-domain Code Generation
        # =================================================================
        _snap_p8 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 8: Per-domain Code Generation...")

        t_phase8 = time.time()

        # Build P8 tasks
        p8_tasks = []
        for domain_name in domain_facet_attributes:
            domain_attrs = domain_facet_attributes.get(domain_name, {})
            if not domain_attrs:
                continue

            domain_facet_ids = set(partition_assignments.get(domain_name, {}).keys())
            domain_attr_assigns = {
                iid: aname for iid, aname in attribute_assignments.items()
                if iid in domain_facet_ids
            }

            all_attr_names = [
                attr.attribute_name
                for facet_attrs in domain_attrs.values()
                for attr in facet_attrs
            ]

            # Compute theme count hint via UMAP + HDBSCAN clustering
            enriched_domain = enriched_by_domain.get(domain_name)

            # Collect domain idea embeddings
            domain_embedding_vectors = [
                self._idea_embeddings[iid]
                for iid in domain_facet_ids
                if iid in self._idea_embeddings
            ]

            # min_attr_len = smallest idea count across attributes (not valence)
            domain_attr_counts = {}
            for attr_name in domain_attr_assigns.values():
                domain_attr_counts[attr_name] = domain_attr_counts.get(attr_name, 0) + 1
            min_attr_len = min(domain_attr_counts.values()) if domain_attr_counts else 2

            if len(domain_embedding_vectors) >= 15:
                theme_hint = estimate_theme_count(
                    np.array(domain_embedding_vectors),
                    n_components=15,
                    min_cluster_size=min_attr_len,
                )
            else:
                theme_hint = None

            if verbose:
                n_attrs = len(all_attr_names)
                if theme_hint:
                    print(f"    {domain_name}: {n_attrs} attrs, min_attr_len={min_attr_len} → cluster span {theme_hint[0]}–{theme_hint[1]}")
                else:
                    print(f"    {domain_name}: {n_attrs} attrs → no cluster hint (too few embeddings or no structure)")

            p8_tasks.append({
                'domain_name': domain_name,
                'domain_facet_attributes': {domain_name: domain_attrs},
                'attribute_assignments': domain_attr_assigns,
                'domain_definition': partition_contexts[domain_name].partition_definition,
                'enriched_attributes': enriched_domain,
                'all_attr_names': all_attr_names,
                'theme_count_hint': theme_hint,
            })

        # Dispatch via SmoothRequester
        p8_requester = SmoothRequester(
            model=self._model_p8,
            phase_key="step5_p8_codebook_generation",
            num_tasks=len(p8_tasks),
            verbose=verbose,
            known_limits=limits,
            has_server_headers=has_headers,
            quiet=True,
        )
        p8_results = await p8_requester.process_all(
            p8_tasks,
            self._p8_prepare_fn(prompt_context),
            self._p8_parse_fn(),
            self._p8_fallback_fn(),
        )

        # Collect P8 results (code_domains parallel to all_codes — P9 qualifies
        # each code's source attributes with its origin domain)
        all_codes = []
        code_domains = []
        codebook_narratives = []
        for task, result in zip(p8_tasks, p8_results):
            domain_name = task['domain_name']
            if result and result.codes:
                all_codes.extend(result.codes)
                code_domains.extend([domain_name] * len(result.codes))
                codebook_narratives.append(f"[{domain_name}] {result.scratchpad}")
                if verbose:
                    print(f"    {domain_name}: {len(result.codes)} codes")

        t_phase8 = time.time() - t_phase8

        if verbose:
            print(f"\n  Phase 8 done in {t_phase8:.1f}s → {len(all_codes)} raw codes "
                  f"from {len(p8_tasks)} calls")

        # Compute code frequencies from attribute assignments
        attr_to_count: Dict[str, int] = {}
        for attr_name in attribute_assignments.values():
            attr_to_count[attr_name] = attr_to_count.get(attr_name, 0) + 1

        code_frequencies: Dict[int, int] = {}
        for idx, code in enumerate(all_codes):
            freq = sum(
                attr_to_count.get(attr, 0)
                for attr in (code.source_attributes or [])
            )
            code_frequencies[idx] = freq

        if self.cost_tracker and _snap_p8 is not None:
            self.cost_tracker.record_phase(
                "step_5_code_generator", "p8_code_generation",
                _snap_p8, token_tracker.snapshot(), self._model_p8)

        # =================================================================
        # PHASE 9 (P9): Cross-domain Codebook Consolidation
        # =================================================================
        _snap_p9 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 9: Codebook Consolidation...")

        t_phase9 = time.time()

        if len(all_codes) > 0:
            # Cross-domain attribute names, domain-qualified as "name (domain)" —
            # constrains P9 source_attributes to valid names AND eliminates the
            # collision when the same attribute name exists in several domains.
            # Parsed back to (domain, name) -> A# in _to_consolidated_code.
            all_attr_names_global = sorted({
                f"{attr.attribute_name} ({domain_name})"
                for domain_name, facet_map in domain_facet_attributes.items()
                for facet_attrs in facet_map.values()
                for attr in facet_attrs
            })

            # Qualified copies for the P9 prompt/enum; all_codes itself keeps bare
            # names (it is the fallback result when P9 fails).
            p9_raw_codes = [
                code.model_copy(update={"source_attributes": [
                    f"{a} ({domain})" for a in (code.source_attributes or [])
                ]})
                for code, domain in zip(all_codes, code_domains)
            ]

            p9_tasks = [{
                'raw_codes': p9_raw_codes,
                'code_frequencies': code_frequencies,
                'all_attr_names': all_attr_names_global,
                'attribute_valence_counts': attribute_valence_counts,
            }]

            p9_requester = SmoothRequester(
                model=self._model_p9,
                phase_key="step5_p9_consolidation",
                num_tasks=1,
                verbose=verbose,
                known_limits=limits,
                has_server_headers=has_headers,
                quiet=True,
            )
            p9_results = await p9_requester.process_all(
                p9_tasks,
                self._p9_prepare_fn(prompt_context),
                self._p9_parse_fn(),
                self._p9_fallback_fn(),
            )

            consolidation_result = p9_results[0]
            if consolidation_result and consolidation_result.codes:
                # Convert constrained dynamic codes back to ConsolidatedCode for downstream
                all_codes = [
                    self._to_consolidated_code(c) for c in consolidation_result.codes
                ]
                codebook_narratives.append(
                    f"[consolidation] {consolidation_result.scratchpad}"
                )

        codebook_narrative = "\n".join(codebook_narratives)

        t_phase9 = time.time() - t_phase9

        if verbose:
            print(f"\n  Phase 9 done in {t_phase9:.1f}s → {len(all_codes)} codes "
                  f"(after consolidation)")
            for i, code in enumerate(all_codes, 1):
                print(f"    {i}. {code.code_name}: {code.definition}")

        if self.cost_tracker and _snap_p9 is not None:
            self.cost_tracker.record_phase(
                "step_5_code_generator", "p9_codebook_consolidation",
                _snap_p9, token_tracker.snapshot(), self._model_p9)

        codebook_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Codebook (P8-P9) complete in {codebook_elapsed:.1f}s")

        return CodebookResult(
            codes=all_codes,
            codebook_narrative=codebook_narrative,
        )

    # =========================================================================
    # EMBEDDING + REPRESENTATIVE SAMPLES
    # =========================================================================

    async def _compute_representative_samples(
        self,
        ideas_flat: List,
        attribute_assignments: Dict[str, str],
        verbose: bool,
    ):
        """Embed ideas and select representative samples per attribute per valence.

        Returns:
            representatives: {(attr_name, valence_group) -> [idea, ...]} max N per group
            group_counts: {(attr_name, valence_group) -> int} total ideas per group
            all_embeddings: {idea_id -> np.ndarray} for caching
        """
        config = self._config
        n = config.max_representative_samples

        # Build lookup and group ideas by (attribute, valence_group)
        groups: Dict[tuple, List] = {}
        for idea in ideas_flat:
            attr_name = attribute_assignments.get(idea.idea_id)
            if not attr_name:
                continue
            valence = getattr(idea, "valence", "") or ""
            if valence == "+":
                valence_group = "positive"
            elif valence == "-":
                valence_group = "negative"
            else:
                valence_group = "neutral"
            key = (attr_name, valence_group)
            groups.setdefault(key, []).append(idea)

        if verbose:
            n_groups = len(groups)
            n_ideas_in_groups = sum(len(g) for g in groups.values())
            print(f"  Grouped {n_ideas_in_groups} ideas into {n_groups} (attribute, valence) groups")

        # Embed all ideas at once
        embedder = SharedEmbedder(
            model=config.embedding_model,
            batch_size=config.embedding_batch_size,
            max_concurrent=config.embedding_max_concurrent,
        )

        all_ideas_ordered = []
        all_texts = []
        for group_ideas in groups.values():
            for idea in group_ideas:
                all_ideas_ordered.append(idea)
                all_texts.append(format_idea_text(idea, config.code_source))

        if not all_texts:
            return {}, {}, {}

        all_embeddings_array = await embedder.embed_texts(all_texts)

        # Build idea_id -> embedding dict for caching
        all_embeddings = {}
        for idea, emb in zip(all_ideas_ordered, all_embeddings_array):
            all_embeddings[idea.idea_id] = emb

        # Select representative samples per group (deduplicated by instance text)
        representatives = {}
        group_counts = {}
        offset = 0
        for key, group_ideas in groups.items():
            group_size = len(group_ideas)
            group_counts[key] = group_size
            group_embeddings = all_embeddings_array[offset:offset + group_size]
            offset += group_size

            # Get all indices sorted by distance to medoid
            all_indices = find_representative_samples(group_embeddings, n=group_size)

            # Deduplicate by instance text — keep first occurrence per unique instance
            seen_instances = set()
            deduped = []
            for idx in all_indices:
                instance = (getattr(group_ideas[idx], "instance", "") or "").strip().lower()
                if instance not in seen_instances:
                    seen_instances.add(instance)
                    deduped.append(idx)
                    if len(deduped) >= n:
                        break

            representatives[key] = [group_ideas[i] for i in deduped]

        if verbose:
            total_reps = sum(len(v) for v in representatives.values())
            print(f"  Selected {total_reps} representative samples across {len(representatives)} groups")

        return representatives, group_counts, all_embeddings

    def _enrich_attributes_with_samples(
        self,
        domain_facet_attributes: Dict[str, Dict[str, list]],
        representatives: Dict[tuple, List],
        group_counts: Dict[tuple, int],
    ) -> Dict[str, Dict[str, List[EnrichedAttribute]]]:
        """Enrich attributes with representative samples per valence.

        Valence counts are always carried through unsuppressed; only the negative
        EXAMPLES are gated on floor(log(attribute_total)) — too few ideas means no
        representative verbatims, but the count itself always reaches P8.

        Returns:
            enriched: {domain -> {facet -> [EnrichedAttribute, ...]}}
        """
        enriched = {}

        for domain_name, facet_attrs in domain_facet_attributes.items():
            enriched[domain_name] = {}
            for facet_name, attrs in facet_attrs.items():
                enriched_list = []
                for attr in attrs:
                    pos_count = group_counts.get((attr.attribute_name, "positive"), 0)
                    neu_count = group_counts.get((attr.attribute_name, "neutral"), 0)
                    neg_count = group_counts.get((attr.attribute_name, "negative"), 0)
                    total = pos_count + neu_count + neg_count

                    # Attribute-level absolute minimum: floor(log(attribute_total)).
                    # No % share requirement — count visibility (must always reach
                    # P8) and example visibility (gated on having enough examples
                    # to be representative) are separate concerns.
                    min_ideas = max(2, int(math.log(max(total, 2))))
                    neg_has_examples = neg_count >= min_ideas

                    pos_samples = representatives.get((attr.attribute_name, "positive"), [])
                    neu_samples = representatives.get((attr.attribute_name, "neutral"), [])
                    neg_samples = representatives.get((attr.attribute_name, "negative"), []) if neg_has_examples else []

                    displayed_neg_count = neg_count  # true count, never suppressed to 0

                    enriched_list.append(EnrichedAttribute(
                        attribute=attr,
                        positive_samples=pos_samples,
                        neutral_samples=neu_samples,
                        negative_samples=neg_samples,
                        positive_count=pos_count,
                        neutral_count=neu_count,
                        negative_count=displayed_neg_count,
                    ))
                enriched[domain_name][facet_name] = enriched_list
        return enriched

    # =========================================================================
    # P8 SMOOTHREQUESTER CALLBACKS
    # =========================================================================

    @staticmethod
    def _build_constrained_response_model(
        attribute_names: List[str],
    ):
        """Build a CodeGenerationFromAttributesResult with source_attributes
        constrained to an enum of valid attribute names."""
        if not attribute_names:
            return CodeGenerationFromAttributesResult

        AttrLiteral = Literal[tuple(attribute_names)]

        ConstrainedCode = create_model(
            "CodeFromAttributes",
            code_name=(str, Field(..., description="Short code name (3-5 word noun phrase)")),
            definition=(str, Field(..., description="Clear definition of what this code covers (1-2 sentences)")),
            typical_indicators=(List[str], Field(..., description="Words or phrases that signal this code")),
            source_attributes=(List[AttrLiteral], Field(
                default_factory=list,
                description="Attribute names this code is derived from (must be exact names from the inventory)",
            )),
        )

        ConstrainedResult = create_model(
            "CodeGenerationFromAttributesResult",
            scratchpad=(str, CodeGenerationFromAttributesResult.model_fields["scratchpad"]),
            codes=(List[ConstrainedCode], Field(..., description="Formal codes derived from the attribute inventory")),
        )

        return ConstrainedResult

    def _p8_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P8 code generation."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_code_from_attributes_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=task['domain_name'],
                domain_definition=task['domain_definition'],
                domain_attributes=task['domain_facet_attributes'],
                attribute_assignments=task['attribute_assignments'],
                enriched_attributes=task['enriched_attributes'],
                theme_count_hint=task.get('theme_count_hint'),
            )

            response_model = self._build_constrained_response_model(task['all_attr_names'])

            # Prompt capture (once per domain)
            gate_key = f"qr_code_gen_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="code_generation_from_attributes",
                    metadata={
                        "model": self._model_p8,
                        "temperature": self._temperature_p8,
                        "max_tokens": self._max_tokens_code_from_attributes,
                        "language": prompt_context.language,
                        "n_total_attributes": len(task['all_attr_names']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': response_model,
                'temperature': self._temperature_p8,
                'max_tokens': self._max_tokens_code_from_attributes,
                'max_retries': 2,
                'extra_kwargs': get_reasoning_params(self._model_p8, phase="codegen_p8"),
            }
        return prepare_fn

    def _p8_parse_fn(self):
        """Return parse_fn closure for P8 code generation."""
        def parse_fn(task: Dict, response):
            return response
        return parse_fn

    @staticmethod
    def _p8_fallback_fn():
        """Return fallback_fn closure for P8 code generation."""
        def fallback_fn(task: Dict, reason: str):
            return CodeGenerationFromAttributesResult(codes=[], scratchpad=f"FALLBACK: {reason}")
        return fallback_fn

    # =========================================================================
    # P9 SMOOTHREQUESTER CALLBACKS
    # =========================================================================

    def _to_consolidated_code(self, code) -> ConsolidatedCode:
        """Convert a constrained P9 code back to ConsolidatedCode, splitting the
        domain-qualified enum values ("name (domain)") into bare display names
        and resolving them to A# ids via the (domain, name) map.

        rpartition takes the LAST " (" — correct even when the attribute name
        itself contains parentheses. If resolution is incomplete (no map, or an
        unresolvable pair), source_attribute_ids is left empty so the mirror
        invariant holds and ensure_codebook_ids fills it at cache-save."""
        data = code.model_dump()
        names, ids = [], []
        for qualified in data.get("source_attributes", []):
            name, sep, domain = qualified.rpartition(" (")
            if sep and domain.endswith(")"):
                domain = domain[:-1]
            else:
                name, domain = qualified, ""
            names.append(name)
            attr_id = self._attribute_ids.get((domain, name))
            if attr_id:
                ids.append(attr_id)
        data["source_attributes"] = names
        data["source_attribute_ids"] = ids if len(ids) == len(names) else []
        return ConsolidatedCode(**data)

    @staticmethod
    def _build_constrained_consolidation_model(attribute_names: List[str]):
        """Build a CodebookConsolidationResult with source_attributes constrained
        to an enum of valid (cross-domain) attribute names.

        Guarantees P9 provenance is always valid, so the post-P9 coverage check
        is sound (no invented or renamed source names)."""
        if not attribute_names:
            return CodebookConsolidationResult

        AttrLiteral = Literal[tuple(attribute_names)]

        ConstrainedCode = create_model(
            "ConsolidatedCode",
            code_name=(str, ConsolidatedCode.model_fields["code_name"]),
            definition=(str, ConsolidatedCode.model_fields["definition"]),
            diagnostic_test=(str, ConsolidatedCode.model_fields["diagnostic_test"]),
            valence=(str, ConsolidatedCode.model_fields["valence"]),
            typical_indicators=(List[str], ConsolidatedCode.model_fields["typical_indicators"]),
            source_attributes=(List[AttrLiteral], Field(
                default_factory=list,
                description="Attribute names this code is derived from (must be the exact 'name (domain)' values from the candidate code list)",
            )),
        )

        ConstrainedResult = create_model(
            "CodebookConsolidationResult",
            scratchpad=(str, CodebookConsolidationResult.model_fields["scratchpad"]),
            codes=(List[ConstrainedCode], Field(..., description="Final MECE codebook")),
        )

        return ConstrainedResult

    def _p9_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P9 codebook consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_codebook_consolidation_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                dimension_def=prompt_context.dimension_def,
                raw_codes=task['raw_codes'],
                code_frequencies=task['code_frequencies'],
                attribute_valence_counts=task.get('attribute_valence_counts'),
            )

            response_model = self._build_constrained_consolidation_model(task['all_attr_names'])

            # Prompt capture
            gate_key = "qr_codebook_consolidation"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="codebook_consolidation",
                    metadata={
                        "model": self._model_p9,
                        "temperature": self._temperature_p9,
                        "max_tokens": self._max_tokens_codebook_consolidation,
                        "language": prompt_context.language,
                        "n_raw_codes": len(task['raw_codes']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': response_model,
                'temperature': self._temperature_p9,
                'max_tokens': self._max_tokens_codebook_consolidation,
                'max_retries': 2,
                'extra_kwargs': get_reasoning_params(self._model_p9, phase="codegen_p9"),
            }
        return prepare_fn

    def _p9_parse_fn(self):
        """Return parse_fn closure for P9 codebook consolidation."""
        def parse_fn(task: Dict, response):
            return response
        return parse_fn

    @staticmethod
    def _p9_fallback_fn():
        """Return fallback_fn closure for P9 consolidation."""
        def fallback_fn(task: Dict, reason: str):
            # P9 failure: return None so raw codes pass through
            print(f"    P9 CONSOLIDATION FAILED: {reason}")
            return None
        return fallback_fn

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _build_all_partition_contexts(
        self,
        partition_set: DomainSet,
    ) -> Dict[str, DomainContext]:
        """Build DomainContext for each partition."""
        contexts = {}
        for part in partition_set.partitions:
            contexts[part.partition_name] = DomainContext(
                partition_name=part.partition_name,
                partition_definition=part.inclusion_definition,
            )
        return contexts

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
