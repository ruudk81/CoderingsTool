"""
Code Assignment per Idea.

Assigns each idea to exactly one MECE code via a single LLM call (the attribute
is pre-assigned by step 4). All partitions are processed concurrently via
SmoothRequester.

Pipeline:
  1. Group ideas by partition (domain)
  2. Consistency binding — group identical/near-identical instances; dispatch
     one representative per cluster, broadcast its code to all members
  3. Embedding pre-filter (optional) — top-N candidates per idea
  4. Build task list with scoped candidate codes
  5. SmoothRequester dispatch (rate limiting, workers, retry)
  6. Collect assignments, broadcast bound clusters, build CodeAssignedModel list

Usage:
    from .code_assignment import CodeAssigner
    from .config_codeAssigner import AssignmentConfig

    assigner = CodeAssigner(
        config=AssignmentConfig(),
        ideas_models=ideas_models,
        mece_results=mece_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
    )
    results = assigner.assign_all()
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple

import nest_asyncio

from utils.llm import token_tracker
from utils.smoothRequester import SmoothRequester
from config import get_reasoning_params, MISCELLANEOUS_CODE_LABELS

import models

from pipeline.step_6_codeAssigner.config_codeAssigner import AssignmentConfig, get_no_fit_label
from models import CodeAssignedSubmodel, CodeAssignedModel
from models import DomainSet, DomainResultModel
from .prompts_codeAssigner import (
    build_code_assignment_prompt,
    CodeAssignmentResponse,
    configure_validation_mode,
)
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from models import CodeAssignment, CodeAssignmentBatch

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()

logger = logging.getLogger(__name__)


def normalize_instance(s: str) -> str:
    """Canonical form of a verbatim instance: lowercase, strip punctuation,
    collapse whitespace. Used to group occurrences of the same word for
    consistency binding (and by view_code_divergence.py)."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# Leading articles/determiners stripped before lemmatizing (so "een fiets"
# folds onto "fiets"). Per language; unknown languages skip stripping.
_ARTICLES = {
    "nl": {"de", "het", "een", "die", "deze", "dat", "dit", "der", "den", "des"},
    "en": {"the", "a", "an"},
    "de": {"der", "die", "das", "ein", "eine", "den", "dem", "des"},
    "fr": {"le", "la", "les", "un", "une", "des", "l"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas"},
}


class CodeAssigner:
    """
    Assigns each idea to exactly one MECE category within its domain
    partition. All partitions are processed concurrently via SmoothRequester.
    """

    def __init__(
        self,
        config: AssignmentConfig,
        ideas_models: List[models.IdeasExtractedModel],
        mece_results: Dict[str, DomainResultModel],
        partition_set: DomainSet,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
        prompt_printer=None,
        codes: List[ConsolidatedCode] = None,
        attribute_assignments: Optional[Dict[str, str]] = None,
        cost_tracker=None,
    ):
        self.cost_tracker = cost_tracker
        self._config = config
        self._ideas_models = ideas_models
        self._mece_results = mece_results
        self._partition_set = partition_set
        self._extraction_metadata = extraction_metadata
        self._codes = codes or []

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_6_code_assigner", {
                "assignment": config.assignment_model,
            })

        # Embedding pre-filter results (populated in Phase 2 if enabled)
        self._idea_code_candidates: Optional[Dict[str, List[int]]] = None
        # idea_id -> (code_id, code_name). All resolution maps carry K#+name
        # TUPLES so a site still expecting a bare label fails loudly.
        self._per_task_resolutions: Dict[str, Tuple[str, str]] = {}

        # Prompt capture (optional — pass PromptPrinter to enable)
        self._prompt_printer = prompt_printer
        self._captured_assign_gates: set = set()

        # Tier-aware validation: nano → lenient field coercion; mini/default → strict
        configure_validation_mode(config.assignment_model)

        # ID-based resolution maps — populated in _assign_all_async()
        self._id_to_code: Dict[str, Tuple[str, str]] = {}
        self._no_fit_id: Optional[str] = None
        self._no_fit_label: Optional[str] = None

        # Provenance maps (attribute_id → home code idx, Overig code idx), plus
        # the name→A# resolver built from the mece structure — SAME artifact as
        # the codes' source_attribute_ids, so the id spaces always match.
        self._attr_to_code_idx: Dict[str, int] = {}
        self._attr_id_scoped: Dict[Tuple[str, str], str] = {}
        self._attr_id_by_name: Dict[str, List[str]] = {}
        self._overig_code_idx: Optional[int] = None
        # The no-fit option resolves to the Overig code when one exists, else the sentinel
        self._no_fit_resolves_to: Tuple[str, str] = ("__UNASSIGNED__", "__UNASSIGNED__")

        # Pre-assigned attributes from pipeline step 4a (idea_id -> attribute name)
        self._attribute_assignments: Dict[str, str] = attribute_assignments or {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Canonical partition key: lowercase, underscores→spaces."""
        return (key or '').strip().lower().replace('_', ' ')

    def assign_all(self) -> List[CodeAssignedModel]:
        """Sync entry point. Returns list of CodeAssignedModel."""
        _snap_before = token_tracker.snapshot() if self.cost_tracker else None

        results = asyncio.run(self._assign_all_async())

        if self.cost_tracker and _snap_before is not None:
            self.cost_tracker.record_phase(
                "step_6_code_assigner", "assignment",
                _snap_before, token_tracker.snapshot(), self._config.assignment_model)

        return results

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _assign_all_async(self) -> List[CodeAssignedModel]:
        """Main async orchestration via SmoothRequester."""
        verbose = self._config.verbose

        # ── Phase 1: Setup ──────────────────────────────────────────────────

        # 1a. Build global facet lookup from P3 facet assignments
        self._facet_lookup: Dict[str, str] = {}
        for name, mece_res in self._mece_results.items():
            if mece_res.facet_assignments:
                self._facet_lookup.update(mece_res.facet_assignments)

        if verbose:
            print(f"  Facet lookup: {len(self._facet_lookup)} entries")

        # 1b. Group all ideas by partition (domain)
        partition_ideas = self._group_ideas_by_partition()
        total_ideas = sum(len(ideas) for ideas in partition_ideas.values())

        if verbose:
            print(f"\n{'='*70}")
            print(f"CATEGORY ASSIGNMENT")
            print(f"{'='*70}")
            print(f"  Ideas: {total_ideas} across {len(partition_ideas)} partitions")
            print(f"  Codebook: global")
            print(f"  Model: {self._config.assignment_model}")

        if total_ideas == 0:
            print("  WARNING: No ideas to assign")
            return self._build_output_models({}, {})

        # 1c. Pre-build provenance maps (Overig detection) + ID maps
        self._build_provenance_maps()
        self._build_id_maps()

        # 1d. Consistency binding: group identical/near-identical instances so
        # each cluster is coded once and broadcast (same word → same code).
        rep_of: Dict[str, str] = {}
        if self._config.bind_enabled:
            partition_ideas, rep_of = await self._bind_clusters(
                partition_ideas, total_ideas, verbose
            )

        # ── Phase 2: Embedding pre-filtering ─────────────────────────────────

        if self._config.use_embedding_prefilter and self._codes:
            from .embedding_matcher import EmbeddingMatcher

            matcher = EmbeddingMatcher(
                model=self._config.embedding_model,
                batch_size=self._config.embedding_batch_size,
                max_concurrent=self._config.embedding_max_concurrent,
            )

            # Flatten all ideas in same order as Phase 4
            all_ideas_flat = []
            for pname in sorted(partition_ideas.keys()):
                all_ideas_flat.extend(partition_ideas[pname])

            idea_texts = [
                matcher.build_idea_text(idea, self._facet_lookup)
                for idea in all_ideas_flat
            ]
            code_texts = [
                matcher.build_code_text(code)
                for code in self._codes
            ]

            if verbose:
                print(f"  Embedding pre-filter: embedding {len(idea_texts)} ideas "
                      f"+ {len(code_texts)} codes...")

            idea_embeddings = await matcher.embed_texts(idea_texts)
            code_embeddings = await matcher.embed_texts(code_texts)

            top_n = self._config.embedding_top_n
            top_n_per_idea = matcher.compute_top_n(
                idea_embeddings, code_embeddings, n=top_n,
            )

            self._idea_code_candidates = {
                idea.idea_id: indices
                for idea, indices in zip(all_ideas_flat, top_n_per_idea)
            }

            if verbose:
                print(f"  Embedding pre-filter: top-{top_n} candidates per idea → "
                      f"prompt codebook scoped from {len(self._codes)} to {top_n} codes")

        # ── Phase 3: Build task list ─────────────────────────────────────────

        if not self._codes:
            if verbose:
                print("  WARNING: No codes available, skipping assignment")
            return []

        task_list = []
        seeded_count = 0
        for partition_name in sorted(partition_ideas.keys()):
            ideas = partition_ideas[partition_name]

            for idea_idx, idea in enumerate(ideas):
                # Determine codes for this idea
                if self._idea_code_candidates and idea.idea_id in self._idea_code_candidates:
                    candidate_indices = list(self._idea_code_candidates[idea.idea_id])
                    # Provenance seeding: guarantee the idea's home code is shown,
                    # so the top-N can never hide its coverage-guaranteed code.
                    # (Overig is reachable via the no-fit option, so it is not seeded.)
                    if self._config.seed_provenance_candidates:
                        attr = (self._attribute_assignments.get(idea.idea_id) or "").strip()
                        home = self._home_code_idx(partition_name, attr)
                        if home is not None and home not in candidate_indices:
                            candidate_indices.append(home)
                            seeded_count += 1
                    candidate_codes = [self._codes[idx] for idx in candidate_indices]
                else:
                    candidate_codes = self._codes

                # Build per-task ID map (scoped C1..CN → (code_id, code_name)).
                # The no-fit option is the final ID and resolves to Overig/sentinel.
                task_id_to_code = {}
                for ci, code in enumerate(candidate_codes, 1):
                    task_id_to_code[f"C{ci}"] = (code.code_id, code.code_name)
                if self._config.allow_no_fit and self._no_fit_label:
                    task_id_to_code[f"C{len(candidate_codes) + 1}"] = self._no_fit_resolves_to

                task_list.append({
                    'idea': idea,
                    'partition_name': partition_name,
                    'batch_idx': idea_idx,
                    'n_batches': len(ideas),
                    'candidate_codes': candidate_codes,
                    'task_id_to_code': task_id_to_code,
                })

        total_tasks = len(task_list)

        if verbose and self._config.seed_provenance_candidates and self._idea_code_candidates:
            print(f"  Provenance seeding: added the home code to {seeded_count}/{total_tasks} "
                  f"task candidate sets")
        if verbose:
            print(f"  No-fit option resolves to: {self._no_fit_resolves_to[1]!r} "
                  f"({self._no_fit_resolves_to[0]})")

        # ── Phase 4: SmoothRequester dispatch ────────────────────────────────

        sr = SmoothRequester(
            model=self._config.assignment_model,
            phase_key="step6_code_assignment",
            num_tasks=total_tasks,
            verbose=verbose,
        )

        sr_results = await sr.process_all(
            task_list,
            self._prepare_fn,
            self._parse_fn,
            self._fallback_fn,
        )

        # ── Phase 5: Collect assignments + ID resolution ────────────────────

        assignment_lookup = {}
        for result in sr_results:
            if result is None:
                continue
            for assignment in result.assignments:
                assignment_lookup[assignment.idea_id] = assignment

        assigned_count = len(assignment_lookup)

        # Resolve option IDs (C#) to (code_id, code_name) pairs
        # Per-task resolutions (from parse_fn) take priority over global ID map
        id_resolution: Dict[str, Tuple[str, str]] = {}
        resolve_stats = {"resolved": 0, "fallback": 0, "unresolved": 0}
        has_prefilter = bool(self._idea_code_candidates)

        for idea_id, assignment in assignment_lookup.items():
            # Try per-task resolution first (scoped IDs from embedding pre-filter)
            if idea_id in self._per_task_resolutions:
                id_resolution[idea_id] = self._per_task_resolutions[idea_id]
                resolve_stats["resolved"] += 1
                continue

            # BP1: When pre-filter is active, do NOT fall back to global
            # because scoped C1-C5 map to DIFFERENT codes than global C1-C22
            if has_prefilter:
                print(f"    WARNING: No scoped resolution for idea '{idea_id}' "
                      f"with pre-filter active — unassigned")
                resolve_stats["unresolved"] += 1
                continue

            # Global ID map (only when no pre-filter)
            raw_id = getattr(assignment, 'option_id', '') or ''
            cat_id = self._normalize_id(raw_id)
            pair = self._id_to_code.get(cat_id)
            if pair:
                id_resolution[idea_id] = pair
                resolve_stats["resolved"] += 1
            elif raw_id:
                print(f"    WARNING: Option ID '{cat_id}' not in global map for "
                      f"idea '{idea_id}' — marking __UNASSIGNED__")
                id_resolution[idea_id] = ("__UNASSIGNED__", "__UNASSIGNED__")
                resolve_stats["fallback"] += 1
            else:
                resolve_stats["unresolved"] += 1

        # Broadcast each representative's assignment to all bound cluster members
        if rep_of:
            for member_id, rep_id in rep_of.items():
                if member_id == rep_id:
                    continue
                if rep_id in assignment_lookup:
                    assignment_lookup[member_id] = assignment_lookup[rep_id]
                if rep_id in id_resolution:
                    id_resolution[member_id] = id_resolution[rep_id]
            assigned_count = len(assignment_lookup)

        if verbose:
            print(f"\n  Assigned: {assigned_count}/{total_ideas} ideas")
            print(f"\n  [ID RESOLUTION]")
            print(f"    Resolved: {resolve_stats['resolved']}")
            if resolve_stats['fallback']:
                print(f"    Fallback (invalid ID): {resolve_stats['fallback']}")
            if resolve_stats['unresolved']:
                print(f"    Unresolved (no ID): {resolve_stats['unresolved']}")

        output = self._build_output_models(assignment_lookup, id_resolution)
        if verbose:
            self._print_assignment_summary(output)
        return output

    # =========================================================================
    # SMOOTHREQUESTER CALLBACKS
    # =========================================================================

    def _prepare_fn(self, task: Dict) -> Dict:
        """Build LLM call params for a single idea assignment."""
        idea = task['idea']
        partition_name = task['partition_name']
        candidate_codes = task.get('candidate_codes', self._codes)

        prompt = self._build_assignment_prompt(idea, codes=candidate_codes)

        # Prompt capture (first task per partition)
        _assign_key = f"assign_{partition_name}"
        if (self._prompt_printer is not None
                and _assign_key not in self._captured_assign_gates):
            self._prompt_printer.capture_prompt(
                step_name="taxonomy_codes",
                utility_name="CodeAssigner",
                prompt_content=prompt,
                prompt_type="code_assignment",
                metadata={
                    "model": self._config.assignment_model,
                    "temperature": self._config.assignment_temperature,
                    "max_tokens": self._config.assignment_max_tokens,
                    "language": (
                        self._extraction_metadata.lang
                        if self._extraction_metadata else "Dutch"
                    ),
                    "partition_name": partition_name,
                    "n_codes": len(candidate_codes),
                }
            )
            self._captured_assign_gates.add(_assign_key)

        return {
            'prompt': prompt,
            'response_model': CodeAssignmentResponse,
            'temperature': self._config.assignment_temperature,
            'max_tokens': self._config.assignment_max_tokens,
            'max_retries': 5,
            'extra_kwargs': get_reasoning_params(self._config.assignment_model, phase="code_assignment"),
        }

    def _parse_fn(self, task: Dict, response) -> Optional[CodeAssignmentBatch]:
        """Parse LLM response into CodeAssignmentBatch + resolve per-task IDs."""
        idea = task['idea']
        task_id_to_code = task.get('task_id_to_code', self._id_to_code)

        # Validate response
        if response is None or not hasattr(response, 'assigned_code_id'):
            return None
        if not response.assigned_code_id:
            return None

        # Wrap into batch format (idea_id from original task, not LLM)
        wrapped = CodeAssignmentBatch(
            assignments=[CodeAssignment(
                idea_id=idea.idea_id,
                option_id=response.assigned_code_id,
                confidence=response.confidence,
                rationale=response.rationale,
            )]
        )

        # Per-task ID resolution (scoped C1-C5 from embedding pre-filter)
        raw_id = response.assigned_code_id or ''
        cat_id = self._normalize_id(raw_id)
        pair = task_id_to_code.get(cat_id)
        if pair:
            self._per_task_resolutions[idea.idea_id] = pair
        else:
            print(f"    WARNING: Option ID '{cat_id}' not in scoped "
                  f"candidates for idea '{idea.idea_id}'")

        return wrapped

    @staticmethod
    def _fallback_fn(task: Dict, reason: str) -> None:
        """Fallback for permanently failed tasks. Returns None —
        _build_output_models handles unassigned ideas with __UNASSIGNED__."""
        return None

    # =========================================================================
    # ID-BASED RESOLUTION
    # =========================================================================

    _RE_NORMALIZE_ID = re.compile(r'\s+')

    @staticmethod
    def _normalize_id(raw_id: str) -> str:
        """Normalize a raw category ID: 'c7' -> 'C7', '7' -> 'C7'."""
        cat_id = CodeAssigner._RE_NORMALIZE_ID.sub('', raw_id.strip().upper())
        if not cat_id.startswith('C') and cat_id.isdigit():
            cat_id = f"C{cat_id}"
        return cat_id

    def _build_id_maps(self) -> None:
        """Build option-ID → (code_id, code_name) maps from self._codes.

        Populates self._id_to_code, self._no_fit_id, self._no_fit_label.
        The no-fit option is the final ID and resolves to Overig/sentinel (its
        display phrase, self._no_fit_label, is shown in the prompt only).
        """
        id_to_code: Dict[str, Tuple[str, str]] = {}

        for i, code in enumerate(self._codes, 1):
            id_to_code[f"C{i}"] = (code.code_id, code.code_name)

        # Add the no-fit option as final entry (resolves to Overig/sentinel)
        if self._config.allow_no_fit:
            language = "Dutch"
            if self._extraction_metadata:
                language = getattr(self._extraction_metadata, 'lang', 'Dutch') or 'Dutch'
            self._no_fit_id = f"C{len(self._codes) + 1}"
            self._no_fit_label = get_no_fit_label(language)
            id_to_code[self._no_fit_id] = self._no_fit_resolves_to
        else:
            self._no_fit_id = None
            self._no_fit_label = None

        self._id_to_code = id_to_code

    def _build_provenance_maps(self) -> None:
        """Map each step-5 attribute id to its home code, and find the Overig code.

        Used to guarantee that an idea's coverage-guaranteed code (the code whose
        source_attribute_ids includes the idea's step-4 attribute) is always in
        the candidate set, even when the embedding pre-filter would not surface
        it. Keyed by attribute_id: same-named attributes in different domains no
        longer collide. Both the A#s and the name→A# resolver come from the mece
        cache (structure ↔ source_attribute_ids), so the id space is consistent
        even for legacy per-artifact minting.
        """
        self._attr_to_code_idx = {}
        for i, code in enumerate(self._codes):
            for attr_id in (getattr(code, 'source_attribute_ids', None) or []):
                if attr_id and attr_id not in self._attr_to_code_idx:
                    self._attr_to_code_idx[attr_id] = i

        # name→A# resolver from the mece structure (normalized domain keys)
        self._attr_id_scoped = {}
        self._attr_id_by_name = {}
        for domain, res in self._mece_results.items():
            dkey = self._normalize_key(domain)
            for attrs in res.attributes.values():
                for a in attrs:
                    name = (a.get("attribute_name") or "").strip()
                    attr_id = a.get("attribute_id")
                    if not name or not attr_id:
                        continue
                    self._attr_id_scoped.setdefault((dkey, name), attr_id)
                    ids = self._attr_id_by_name.setdefault(name, [])
                    if attr_id not in ids:
                        ids.append(attr_id)

        overig_names = {v.strip().lower() for v in MISCELLANEOUS_CODE_LABELS.values()} | {"overig"}
        self._overig_code_idx = None
        for i, code in enumerate(self._codes):
            if (code.code_name or "").strip().lower() in overig_names:
                self._overig_code_idx = i
                break

        # "No specific code fits" routes to the existing Overig code; only when
        # the codebook has no Overig does it fall back to the __UNASSIGNED__ sentinel.
        self._no_fit_resolves_to = (
            (self._codes[self._overig_code_idx].code_id,
             self._codes[self._overig_code_idx].code_name)
            if self._overig_code_idx is not None
            else ("__UNASSIGNED__", "__UNASSIGNED__")
        )

    def _home_code_idx(self, domain: str, attr_name: str) -> Optional[int]:
        """Idea's home-code index via its attribute NAME: resolve to A# against
        the mece structure ((domain, name) first, then structure-wide unique
        name), then A# → code idx. None when unresolvable — no seeding then,
        same as the pre-id behavior for unknown names."""
        if not attr_name:
            return None
        attr_id = self._attr_id_scoped.get((self._normalize_key(domain), attr_name))
        if not attr_id:
            unique = self._attr_id_by_name.get(attr_name)
            attr_id = unique[0] if unique and len(unique) == 1 else None
        return self._attr_to_code_idx.get(attr_id) if attr_id else None

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_assignment_prompt(self, idea, codes=None) -> str:
        """Build prompt for assigning a single idea to a code.

        Args:
            idea: The idea to assign.
            codes: Optional subset of codes. Defaults to self._codes (all codes).
        """
        codes = codes if codes is not None else self._codes

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

        return build_code_assignment_prompt(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            codes=codes,
            no_fit_label=self._no_fit_label,
            idea=idea,
            facet_lookup=self._facet_lookup,
        )

    # =========================================================================
    # CONSISTENCY BINDING
    # =========================================================================

    @staticmethod
    def _pick_representative(ideas: List):
        """Representative of a bound cluster: an idea with the most common exact
        instance form, breaking ties by the longest ladder (most informative)."""
        from collections import Counter
        form_counts = Counter(normalize_instance(i.instance) for i in ideas)
        top_form = form_counts.most_common(1)[0][0]
        candidates = [i for i in ideas if normalize_instance(i.instance) == top_form]
        return max(
            candidates,
            key=lambda i: len((i.interpretation or "") + (i.abstraction or "")),
        )

    def _resolve_lemma_lang(self, verbose: bool) -> Optional[str]:
        """Return a simplemma language code if lemmatization is enabled and
        supported for the dataset language, else None (falls back to exact)."""
        if not self._config.bind_lemmatize:
            return None
        lang = ""
        if self._extraction_metadata:
            lang = getattr(self._extraction_metadata, 'lang', '') or ""
        code = lang.split("-")[0].lower() if lang else ""
        if not code:
            return None
        try:
            import simplemma
            simplemma.lemmatize("test", lang=code)  # probe support
        except (ImportError, ValueError) as e:
            if verbose:
                print(f"  Binding: lemmatization unavailable for lang '{code}' "
                      f"({type(e).__name__}); using exact grouping")
            return None
        return code

    def _instance_key(self, instance: str, lemma_lang: Optional[str]) -> str:
        """Grouping key for an instance: normalized, optionally lemmatized
        (article-stripped) per the dataset language."""
        norm = normalize_instance(instance)
        if not lemma_lang or not norm:
            return norm
        import simplemma
        toks = norm.split()
        if self._config.bind_strip_articles:
            arts = _ARTICLES.get(lemma_lang, set())
            toks = [t for t in toks if t not in arts] or toks
        return " ".join(simplemma.lemmatize(t, lang=lemma_lang) for t in toks)

    async def _bind_clusters(
        self,
        partition_ideas: Dict[str, List],
        total_ideas: int,
        verbose: bool,
    ):
        """Group identical/near-identical instances into clusters.

        Returns (dispatch_partition_ideas, rep_of):
          - dispatch_partition_ideas: partition→ideas, reduced to one
            representative per bound cluster plus every singleton.
          - rep_of: idea_id → representative idea_id (identity for singletons).
        The caller dispatches only the representatives and broadcasts each
        resolved code back to its cluster members.
        """
        from collections import defaultdict

        # Flatten in the partition order used downstream
        all_ideas = []
        for pname in sorted(partition_ideas.keys()):
            all_ideas.extend(partition_ideas[pname])

        # 1. Group by normalized (and, when enabled, lemmatized) instance —
        #    the deterministic backbone. Empty/blank instances get a unique key
        #    so they are never bound.
        lemma_lang = self._resolve_lemma_lang(verbose)
        exact_groups: Dict[str, list] = defaultdict(list)
        for idea in all_ideas:
            key = self._instance_key(idea.instance, lemma_lang)
            exact_groups[key if key else f"\x00{idea.idea_id}"].append(idea)

        # 2. Optional near-duplicate merge via instance-embedding cosine
        parent = {k: k for k in exact_groups}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        word_keys = [k for k in exact_groups if not k.startswith("\x00")]
        if self._config.bind_use_embeddings and len(word_keys) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            from .embedding_matcher import EmbeddingMatcher

            matcher = EmbeddingMatcher(
                model=self._config.embedding_model,
                batch_size=self._config.embedding_batch_size,
                max_concurrent=self._config.embedding_max_concurrent,
            )
            embs = await matcher.embed_texts(word_keys)
            sim = cosine_similarity(embs)
            thr = self._config.bind_cosine_threshold
            n = len(word_keys)
            for i in range(n):
                for j in range(i + 1, n):
                    if sim[i, j] >= thr:
                        union(word_keys[i], word_keys[j])

        # 3. Form clusters, pick representatives, build the rep_of map
        clusters: Dict[str, list] = defaultdict(list)
        for k, ideas in exact_groups.items():
            clusters[find(k)].extend(ideas)

        rep_of: Dict[str, str] = {}
        dispatch_ids = set()
        n_clusters_bound = 0
        n_ideas_bound = 0
        for ideas in clusters.values():
            if len(ideas) >= self._config.bind_min_cluster_size:
                rep = self._pick_representative(ideas)
                for idea in ideas:
                    rep_of[idea.idea_id] = rep.idea_id
                dispatch_ids.add(rep.idea_id)
                n_clusters_bound += 1
                n_ideas_bound += len(ideas)
            else:
                for idea in ideas:
                    rep_of[idea.idea_id] = idea.idea_id
                    dispatch_ids.add(idea.idea_id)

        # 4. Reduce partition structure to dispatched ideas only
        dispatch_partition_ideas: Dict[str, list] = defaultdict(list)
        for pname in sorted(partition_ideas.keys()):
            for idea in partition_ideas[pname]:
                if idea.idea_id in dispatch_ids:
                    dispatch_partition_ideas[pname].append(idea)

        if verbose:
            saved = total_ideas - len(dispatch_ids)
            print(f"  Binding: {n_clusters_bound} clusters bound {n_ideas_bound} ideas "
                  f"→ {len(dispatch_ids)} dispatched ({saved} fewer LLM calls)")

        return dict(dispatch_partition_ideas), rep_of

    # =========================================================================
    # IDEA GROUPING & BATCHING
    # =========================================================================

    def _group_ideas_by_partition(
        self,
    ) -> Dict[str, List[models.IdeasExtractedSubmodel]]:
        """Group all ideas by their domain (= partition)."""
        partitions: Dict[str, List] = {}
        for resp in self._ideas_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                ct = self._normalize_key(idea.domain)
                if not ct:
                    continue
                if ct not in partitions:
                    partitions[ct] = []
                partitions[ct].append(idea)
        return partitions

    # =========================================================================
    # OUTPUT MODEL CONSTRUCTION
    # =========================================================================

    def _build_output_models(
        self,
        assignment_lookup: dict,
        id_resolution: Dict[str, Tuple[str, str]],
    ) -> List[CodeAssignedModel]:
        """Build CodeAssignedModel list preserving response structure.

        BP3: Iterates ALL original ideas (not just successful assignments).
        BP2: Creates fallback entries for unassigned ideas.
        BP4: Logs count reconciliation.
        """
        facet_lookup = self._facet_lookup
        total_ideas = 0
        unassigned_ideas = 0

        output = []
        for resp in self._ideas_models:
            new_ideas = []
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    total_ideas += 1
                    assignment = assignment_lookup.get(idea.idea_id)
                    ct = self._normalize_key(idea.domain)

                    resolved = id_resolution.get(idea.idea_id)

                    # BP2: Ideas with no resolvable assignment (failed call or an
                    # out-of-scope/invalid ID) fall back to the catch-all (Overig)
                    # when one exists, else the __UNASSIGNED__ sentinel.
                    if not resolved:
                        unassigned_ideas += 1
                        resolved = self._no_fit_resolves_to
                    resolved_code_id, resolved_label = resolved

                    idea_data = idea.model_dump()
                    explicit_fields = {
                        'assigned_code', 'assigned_code_id', 'confidence',
                        'rationale', 'assigned_attribute',
                        'partition_name', 'facet',
                    }
                    new_idea = CodeAssignedSubmodel(
                        **{k: v for k, v in idea_data.items()
                           if k in CodeAssignedSubmodel.model_fields
                           and k not in explicit_fields},
                        assigned_code=resolved_label,
                        assigned_code_id=resolved_code_id,
                        confidence=(
                            assignment.confidence
                            if assignment else 0.0
                        ),
                        rationale=(
                            assignment.rationale
                            if assignment else "No assignment from LLM"
                        ),
                        assigned_attribute=(
                            self._attribute_assignments.get(idea.idea_id)
                        ),
                        partition_name=ct if ct else None,
                        facet=facet_lookup.get(idea.idea_id, idea_data.get('facet', '')),
                    )
                    new_ideas.append(new_idea)

            resp_data = resp.model_dump()
            new_resp = CodeAssignedModel(
                **{k: v for k, v in resp_data.items()
                   if k in CodeAssignedModel.model_fields
                   and k != 'response_ideas'},
                response_ideas=new_ideas,
            )
            output.append(new_resp)

        # BP4: Count reconciliation
        if unassigned_ideas > 0:
            print(f"    {unassigned_ideas}/{total_ideas} ideas had no resolvable assignment "
                  f"(failed call / out-of-scope ID) → routed to {self._no_fit_resolves_to[1]!r} "
                  f"({unassigned_ideas/max(total_ideas,1)*100:.1f}%)")

        return output

    # =========================================================================
    # REPORTING
    # =========================================================================

    def _print_assignment_summary(
        self,
        output: List[CodeAssignedModel],
    ):
        """Print code-centric assignment summary with attribute breakdowns."""
        # Collect stats per code
        code_stats: Dict[str, Dict] = {}
        total_ideas = 0
        total_assigned = 0

        for resp in output:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                total_ideas += 1
                if not idea.assigned_code:
                    continue
                total_assigned += 1

                code = idea.assigned_code
                if code not in code_stats:
                    code_stats[code] = {
                        "count": 0,
                        "confidences": [],
                        "attributes": {},
                    }
                code_stats[code]["count"] += 1
                code_stats[code]["confidences"].append(idea.confidence or 0.0)

                attr = idea.assigned_attribute or "(no attribute)"
                code_stats[code]["attributes"][attr] = (
                    code_stats[code]["attributes"].get(attr, 0) + 1
                )

        # Build code order + valence lookup from self._codes
        code_order = []
        code_valence = {}
        for code in self._codes:
            code_order.append(code.code_name)
            code_valence[code.code_name] = getattr(code, 'valence', '') or ''

        # Add any assigned codes not in the codebook (i.e., "__UNASSIGNED__")
        for code_name in code_stats:
            if code_name not in code_valence:
                code_order.append(code_name)
                code_valence[code_name] = ''

        print(f"\n  {'─'*60}")
        print(f"  ASSIGNMENT SUMMARY ({total_assigned}/{total_ideas} ideas → "
              f"{len(code_stats)} codes)")
        print(f"  {'─'*60}")

        for i, code_name in enumerate(code_order, 1):
            if code_name not in code_stats:
                continue
            stats = code_stats[code_name]
            avg_conf = (
                sum(stats["confidences"]) / len(stats["confidences"])
                if stats["confidences"] else 0.0
            )
            valence = code_valence.get(code_name, '')
            v_tag = {"positive": "+", "negative": "-", "neutral": "~"}.get(valence, "")

            print(f"\n  [{i}] ({v_tag}) {code_name} — {stats['count']} ideas "
                  f"(conf {avg_conf:.2f})")

            for attr, count in sorted(
                stats["attributes"].items(), key=lambda x: -x[1]
            ):
                print(f"        {attr}: {count}")
