#%% 
 
 
"""
Codebook Consolidation Module (Post-Assignment)

Two-phase consolidation of fine-grained codes into a reporting-ready codebook:

Phase 1 (deterministic): Sweep smallest codes into "Other/Miscellaneous" until
    the bucket reaches 10% of deduplicated ideas.
Phase 2 (LLM per theme): Axial consolidation — merge remaining codes within each
    theme to reach target k = max(sqrt(n_ideas)/2, 10).

Output: dual codebook (original + consolidated), mapping table, methodology paragraph.

Usage (standalone):
    cd src && python -m development.step_8_codeAssigner.codebookConsolidation

Usage (from other modules):
    from development.step_8_codeAssigner.codebookConsolidation import CodebookConsolidator
    consolidator = CodebookConsolidator.from_cache(config)
    result = consolidator.consolidate()
"""

import sys
import asyncio
import math
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field

src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from development import models_exp as models
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE, MISCELLANEOUS_CODE_LABELS
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.llm import create_client, llm_create_async, token_tracker

from development.step_8_codeAssigner.prompts_exp import AXIAL_CONSOLIDATION_PROMPT

try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Unknown / catch-all labels (shared with summary.py)
_UNKNOWN_LABELS = {"Overig", "Other", "Sonstiges", "Unassigned"}

# Noise budget: fraction of total deduplicated ideas that can go to "Other"
NOISE_BUDGET_PCT = 0.10


# =============================================================================
# PYDANTIC MODELS — LLM RESPONSE
# =============================================================================

class ConsolidatedCodeMapping(BaseModel):
    """One or more original codes merged into a single consolidated code."""
    original_codes: List[str] = Field(
        description="List of original code labels being merged into this consolidated code"
    )
    consolidated_label: str = Field(
        description="New consolidated code label (semantically neutral, thematically appropriate)"
    )
    consolidated_definition: str = Field(
        description="Definition of the consolidated code (max 25 words)"
    )
    consolidation_rationale: str = Field(
        description="Brief explanation of why these codes were merged"
    )


class RetainedCode(BaseModel):
    """A code that is retained as-is (not merged)."""
    code: str = Field(description="Original code label")
    retain_rationale: str = Field(
        description="Why this code is retained (e.g. conceptually unique, critical perspective)"
    )


class ThemeConsolidationResponse(BaseModel):
    """LLM response for consolidating codes within a single theme."""
    theme: str = Field(description="Theme name")
    retained_codes: List[RetainedCode] = Field(
        description="Codes kept as standalone (dominant or conceptually unique)"
    )
    consolidated_codes: List[ConsolidatedCodeMapping] = Field(
        description="Groups of codes merged into higher-order codes"
    )
    analysis: str = Field(
        description="Brief analysis of consolidation decisions for this theme"
    )


# =============================================================================
# PYDANTIC MODELS — INTERNAL RESULTS
# =============================================================================

class NoiseRemovalResult(BaseModel):
    """Output of Phase 1: noise removal."""
    noise_codes: Dict[str, int] = Field(default_factory=dict)
    surviving_codes: Dict[str, int] = Field(default_factory=dict)
    noise_pct: float = 0.0
    total_deduped_ideas: int = 0
    total_respondents: int = 0
    other_label: str = "Other"


class ConsolidationMappingEntry(BaseModel):
    """Single row in the audit trail mapping table."""
    original_code: str
    consolidated_code: str
    theme: str
    action: str  # "retained" | "merged" | "noise_removed"


class ConsolidationResult(BaseModel):
    """Complete output of the consolidation process."""
    mapping_table: List[ConsolidationMappingEntry] = Field(default_factory=list)
    consolidated_codes: List[Dict[str, Any]] = Field(default_factory=list)
    noise_removal: Optional[NoiseRemovalResult] = None
    target_k: int = 0
    actual_k: int = 0
    methodology_paragraph: str = ""


# =============================================================================
# HELPER: DEDUPLICATION (follows summary.py pattern)
# =============================================================================

def _build_deduped_frequencies(
    results: List[models.CodeAssignedModel],
) -> tuple[Dict[str, set], Dict[str, int], int, int]:
    """Build respondent-deduplicated code frequencies.

    Returns:
        respondent_codes: dict[respondent_id, set[code]]
        code_counts: dict[code, int]  (deduplicated)
        total_deduped: total (respondent, code) pairs
        total_respondents: number of respondents
    """
    respondent_codes: Dict[str, set] = defaultdict(set)

    for result in results:
        resp_id = str(result.respondent_id)
        respondent_codes[resp_id]  # ensure key exists
        if not result.response_ideas:
            continue
        for idea in result.response_ideas:
            code = (idea.assigned_codes[0]
                    if idea.assigned_codes else "Unassigned")
            respondent_codes[resp_id].add(code)

    code_counts: Dict[str, int] = defaultdict(int)
    for codes in respondent_codes.values():
        for code in codes:
            code_counts[code] += 1

    total_deduped = sum(code_counts.values())
    total_respondents = len(respondent_codes)

    return dict(respondent_codes), dict(code_counts), total_deduped, total_respondents


def _pct(count: int, total: int) -> str:
    """Format a percentage string like '14.3%'."""
    if total == 0:
        return " 0.0%"
    return f"{count / total * 100:>4.1f}%"


def _unique_respondents_for_codes(
    codes: List[str],
    respondent_codes: Dict[str, set],
) -> int:
    """Count how many respondents have at least one of the given codes."""
    code_set = set(codes)
    return sum(1 for rc in respondent_codes.values() if rc & code_set)


# =============================================================================
# MAIN CLASS
# =============================================================================

class CodebookConsolidator:
    """Two-phase codebook consolidation: noise removal + axial merging."""

    def __init__(
        self,
        codebook: models.ThemeEnrichedCodebookModelExp,
        code_assigned_results: List[models.CodeAssignedModel],
        language: str = DEFAULT_LANGUAGE,
        model_config: Optional[ModelConfig] = None,
        verbose: bool = True,
    ):
        self.codebook = codebook
        self.results = code_assigned_results
        self.language = language
        self.verbose = verbose

        mc = model_config or ModelConfig()
        self.model = mc.get_model_for_stage("code_assignment")
        self.temperature = mc.get_temperature_for_stage("code_assignment")
        self.client = create_client(self.model, async_mode=True)

        # Build code → theme mapping and code → entry lookup
        self.code_to_theme: Dict[str, str] = {}
        self.code_to_entry: Dict[str, models.ThemeEnrichedCodebookEntryExp] = {}
        for entry in self.codebook.codes:
            if entry.code:
                self.code_to_theme[entry.code] = entry.theme or "(no theme)"
                self.code_to_entry[entry.code] = entry

        self.other_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")

        # Computed during consolidation
        self._respondent_codes: Dict[str, set] = {}
        self._code_counts: Dict[str, int] = {}
        self._total_deduped = 0
        self._total_respondents = 0

    @classmethod
    def from_cache(cls, config, **kwargs) -> "CodebookConsolidator":
        """Load codebook + assignment results from cache and construct."""
        variable_key = generate_enhanced_variable_key(
            selected_variables=[config.var_name],
            is_merged=False,
            sample_size=config.sample_size,
        )
        cache_manager = CacheManager(CacheConfig())

        codebook_list = cache_manager.load_from_cache(
            config.filename, "codebook_refinement_enriched", variable_key,
            models.ThemeEnrichedCodebookModelExp,
        )
        if not codebook_list:
            raise RuntimeError("No codebook found in cache (step 7).")
        codebook = codebook_list[0]

        results = cache_manager.load_from_cache(
            config.filename, "code_assignment_direct", variable_key,
            models.CodeAssignedModel,
        )
        if not results:
            raise RuntimeError("No code assignment results found in cache (step 8).")

        return cls(codebook=codebook, code_assigned_results=results, **kwargs)

    # =========================================================================
    # PHASE 1: NOISE REMOVAL (deterministic)
    # =========================================================================

    def phase_1_noise_removal(self) -> NoiseRemovalResult:
        """Sweep smallest codes into 'Other' until cumulative reaches NOISE_BUDGET_PCT."""

        # Build deduplicated frequencies
        self._respondent_codes, self._code_counts, self._total_deduped, self._total_respondents = (
            _build_deduped_frequencies(self.results)
        )

        noise_target = self._total_deduped * NOISE_BUDGET_PCT
        noise_codes: Dict[str, int] = {}
        surviving_codes: Dict[str, int] = {}
        cumulative = 0

        # Pre-classify: existing unknown labels always go to noise
        for code, count in self._code_counts.items():
            if code in _UNKNOWN_LABELS or code == self.other_label:
                noise_codes[code] = count
                cumulative += count

        # Sort remaining codes ascending by frequency
        remaining = sorted(
            ((code, count) for code, count in self._code_counts.items()
             if code not in noise_codes),
            key=lambda x: x[1],
        )

        # Greedy sweep: add smallest codes until budget is filled
        for code, count in remaining:
            if cumulative + count <= noise_target:
                noise_codes[code] = count
                cumulative += count
            else:
                surviving_codes[code] = count

        noise_pct = cumulative / self._total_deduped if self._total_deduped > 0 else 0.0

        result = NoiseRemovalResult(
            noise_codes=noise_codes,
            surviving_codes=surviving_codes,
            noise_pct=noise_pct,
            total_deduped_ideas=self._total_deduped,
            total_respondents=self._total_respondents,
            other_label=self.other_label,
        )

        if self.verbose:
            self._print_phase_1_report(result)

        return result

    # =========================================================================
    # PHASE 2: AXIAL CONSOLIDATION (LLM per theme)
    # =========================================================================

    @staticmethod
    def _compute_target_k(n_ideas: int) -> int:
        """Target code count: k = max(floor(sqrt(n)/2), 10)."""
        return max(int(math.sqrt(n_ideas) / 2), 10)

    @staticmethod
    def _allocate_k_per_theme(
        theme_idea_counts: Dict[str, int],
        target_k: int,
    ) -> Dict[str, int]:
        """Proportionally allocate target k across themes (min 1 each)."""
        total = sum(theme_idea_counts.values())
        if total == 0:
            return {t: 1 for t in theme_idea_counts}

        # Raw proportional allocation
        raw: Dict[str, float] = {}
        allocations: Dict[str, int] = {}
        for theme, count in theme_idea_counts.items():
            raw[theme] = (count / total) * target_k
            allocations[theme] = max(1, round(raw[theme]))

        # Adjust if rounding over-allocated
        total_allocated = sum(allocations.values())
        if total_allocated > target_k:
            # Reduce from most over-allocated themes first
            diffs = {t: allocations[t] - raw[t] for t in allocations}
            for t in sorted(diffs, key=lambda x: diffs[x], reverse=True):
                if total_allocated <= target_k:
                    break
                if allocations[t] > 1:
                    allocations[t] -= 1
                    total_allocated -= 1

        return allocations

    async def _consolidate_theme_async(
        self,
        theme: str,
        codes_in_theme: List[Dict[str, Any]],
        target_k: int,
    ) -> ThemeConsolidationResponse:
        """Send one LLM call to consolidate codes within a theme."""
        codes_formatted = self._format_codes_for_prompt(codes_in_theme)

        prompt = AXIAL_CONSOLIDATION_PROMPT.format(
            language=self.language,
            theme=theme,
            n_codes=len(codes_in_theme),
            target_k=target_k,
            codes_formatted=codes_formatted,
        )

        response = await llm_create_async(
            client=self.client,
            model=self.model,
            prompt=prompt,
            response_model=ThemeConsolidationResponse,
            temperature=self.temperature,
            max_tokens=4000,
            track_usage=True,
        )
        return response

    def _format_codes_for_prompt(self, codes: List[Dict[str, Any]]) -> str:
        """Format code details for the consolidation prompt."""
        lines = []
        for c in codes:
            entry = c.get("entry")
            freq = c.get("frequency", 0)
            pct = c.get("pct", "0.0%")
            resp_pct = c.get("resp_pct", "0.0%")

            lines.append(f'Code: "{c["code"]}"  (frequency: {freq} assignments, {pct} of ideas, {resp_pct} of respondents)')
            if entry and entry.definition:
                lines.append(f"  Definition: {entry.definition}")
            if entry and entry.boundary_test:
                lines.append(f"  Boundary test: {entry.boundary_test}")
            if entry and entry.inclusion_examples:
                examples = "; ".join(entry.inclusion_examples[:3])
                lines.append(f"  Inclusion examples: {examples}")
            if entry and entry.exclusion_examples:
                examples = "; ".join(entry.exclusion_examples[:3])
                lines.append(f"  Exclusion examples: {examples}")
            lines.append("")
        return "\n".join(lines)

    async def _phase_2_async(
        self,
        noise_result: NoiseRemovalResult,
    ) -> tuple[List[ConsolidationMappingEntry], List[Dict[str, Any]], int]:
        """Run Phase 2: compute k, allocate per theme, call LLM per theme."""

        surviving = noise_result.surviving_codes
        n_surviving_ideas = sum(surviving.values())
        target_k = self._compute_target_k(n_surviving_ideas)

        # Group surviving codes by theme
        theme_codes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        theme_idea_counts: Dict[str, int] = defaultdict(int)

        for code, count in surviving.items():
            theme = self.code_to_theme.get(code, "(no theme)")
            entry = self.code_to_entry.get(code)
            theme_codes[theme].append({
                "code": code,
                "frequency": count,
                "pct": _pct(count, noise_result.total_deduped_ideas),
                "resp_pct": _pct(count, noise_result.total_respondents),
                "entry": entry,
            })
            theme_idea_counts[theme] += count

        allocations = self._allocate_k_per_theme(theme_idea_counts, target_k)

        if self.verbose:
            print(f"\n{'=' * 78}")
            print(f"PHASE 2: AXIAL CONSOLIDATION")
            print(f"  Surviving ideas (after noise): {n_surviving_ideas}")
            print(f"  Target k: {target_k}")
            print(f"  Themes: {len(theme_codes)}")
            print(f"  Allocation: {dict(allocations)}")
            print(f"{'=' * 78}")

        # Determine which themes need LLM consolidation vs auto-retain
        mapping: List[ConsolidationMappingEntry] = []
        consolidated_codes: List[Dict[str, Any]] = []
        themes_to_consolidate: Dict[str, tuple] = {}

        for theme, codes in theme_codes.items():
            alloc = allocations.get(theme, 1)
            if len(codes) <= alloc:
                # Auto-retain: already at or below target
                for c in codes:
                    mapping.append(ConsolidationMappingEntry(
                        original_code=c["code"],
                        consolidated_code=c["code"],
                        theme=theme,
                        action="retained",
                    ))
                    consolidated_codes.append({
                        "code": c["code"],
                        "definition": c["entry"].definition if c["entry"] else "",
                        "theme": theme,
                        "frequency": c["frequency"],
                    })
                if self.verbose:
                    print(f"  [{theme}] auto-retained ({len(codes)} codes <= alloc {alloc})")
            else:
                themes_to_consolidate[theme] = (codes, alloc)

        # LLM consolidation for themes that need it
        if themes_to_consolidate:
            tasks = {}
            for theme, (codes, alloc) in themes_to_consolidate.items():
                if self.verbose:
                    print(f"  [{theme}] consolidating {len(codes)} codes -> target {alloc}...")
                tasks[theme] = self._consolidate_theme_async(theme, codes, alloc)

            results = await asyncio.gather(
                *tasks.values(), return_exceptions=True,
            )

            for theme, result in zip(tasks.keys(), results):
                codes_in_theme = themes_to_consolidate[theme][0]
                original_code_set = {c["code"] for c in codes_in_theme}
                code_freq = {c["code"]: c["frequency"] for c in codes_in_theme}

                if isinstance(result, Exception):
                    logger.error(f"LLM failed for theme '{theme}': {result}")
                    if self.verbose:
                        print(f"  [{theme}] LLM FAILED — auto-retaining all codes")
                    # Fallback: auto-retain all codes
                    for c in codes_in_theme:
                        mapping.append(ConsolidationMappingEntry(
                            original_code=c["code"],
                            consolidated_code=c["code"],
                            theme=theme,
                            action="retained",
                        ))
                        consolidated_codes.append({
                            "code": c["code"],
                            "definition": c["entry"].definition if c["entry"] else "",
                            "theme": theme,
                            "frequency": c["frequency"],
                        })
                    continue

                # Process LLM response
                seen_codes: set = set()

                # Retained codes
                for rc in result.retained_codes:
                    seen_codes.add(rc.code)
                    freq = code_freq.get(rc.code, 0)
                    entry = self.code_to_entry.get(rc.code)
                    mapping.append(ConsolidationMappingEntry(
                        original_code=rc.code,
                        consolidated_code=rc.code,
                        theme=theme,
                        action="retained",
                    ))
                    consolidated_codes.append({
                        "code": rc.code,
                        "definition": entry.definition if entry else "",
                        "theme": theme,
                        "frequency": freq,
                    })

                # Merged codes
                for mc in result.consolidated_codes:
                    merged_freq = sum(code_freq.get(oc, 0) for oc in mc.original_codes)
                    for oc in mc.original_codes:
                        seen_codes.add(oc)
                        mapping.append(ConsolidationMappingEntry(
                            original_code=oc,
                            consolidated_code=mc.consolidated_label,
                            theme=theme,
                            action="merged",
                        ))
                    consolidated_codes.append({
                        "code": mc.consolidated_label,
                        "definition": mc.consolidated_definition,
                        "theme": theme,
                        "frequency": merged_freq,
                        "merged_from": mc.original_codes,
                        "rationale": mc.consolidation_rationale,
                    })

                # Post-validation: catch any codes the LLM missed
                missing = original_code_set - seen_codes
                if missing:
                    logger.warning(f"Theme '{theme}': LLM omitted {len(missing)} codes, auto-retaining: {missing}")
                    for code in missing:
                        freq = code_freq.get(code, 0)
                        entry = self.code_to_entry.get(code)
                        mapping.append(ConsolidationMappingEntry(
                            original_code=code,
                            consolidated_code=code,
                            theme=theme,
                            action="retained",
                        ))
                        consolidated_codes.append({
                            "code": code,
                            "definition": entry.definition if entry else "",
                            "theme": theme,
                            "frequency": freq,
                        })

                if self.verbose:
                    n_retained = len(result.retained_codes) + len(missing)
                    n_merged_groups = len(result.consolidated_codes)
                    n_final = n_retained + n_merged_groups
                    print(f"  [{theme}] done: {n_retained} retained, {n_merged_groups} merged groups -> {n_final} codes")
                    if result.analysis:
                        print(f"    Analysis: {result.analysis[:120]}...")

        actual_k = len(set(m.consolidated_code for m in mapping if m.action != "noise_removed"))
        return mapping, consolidated_codes, target_k

    # =========================================================================
    # ORCHESTRATOR
    # =========================================================================

    def consolidate(self) -> ConsolidationResult:
        """Run Phase 1 + Phase 2 and return full consolidation result."""
        start_time = time.perf_counter()

        # Phase 1: noise removal
        noise_result = self.phase_1_noise_removal()

        # Add noise codes to mapping
        noise_mapping: List[ConsolidationMappingEntry] = []
        for code in noise_result.noise_codes:
            theme = self.code_to_theme.get(code, "(unknown)")
            noise_mapping.append(ConsolidationMappingEntry(
                original_code=code,
                consolidated_code=self.other_label,
                theme=theme,
                action="noise_removed",
            ))

        # Phase 2: axial consolidation
        loop = asyncio.get_event_loop()
        phase2_mapping, consolidated_codes, target_k = loop.run_until_complete(
            self._phase_2_async(noise_result)
        )

        # Combine mappings
        full_mapping = noise_mapping + phase2_mapping

        # Add the "Other" bucket as a consolidated code
        other_freq = sum(noise_result.noise_codes.values())
        if other_freq > 0:
            consolidated_codes.append({
                "code": self.other_label,
                "definition": "Miscellaneous / low-frequency codes consolidated for reporting",
                "theme": "(Other)",
                "frequency": other_freq,
                "merged_from": list(noise_result.noise_codes.keys()),
            })

        actual_k = len(set(
            c["code"] for c in consolidated_codes
        ))

        n_original = len(set(self._code_counts.keys()) - _UNKNOWN_LABELS - {self.other_label})
        n_themes = len(set(c["theme"] for c in consolidated_codes if c["theme"] != "(Other)"))

        methodology = (
            f"After initial coding, a post-hoc consolidation was performed in two phases. "
            f"Phase 1 (noise removal): codes with the lowest frequencies were swept into "
            f"'{self.other_label}' until the cumulative proportion reached "
            f"{noise_result.noise_pct * 100:.1f}% of all deduplicated code-assignments "
            f"({len(noise_result.noise_codes)} codes removed, "
            f"{sum(noise_result.noise_codes.values())} assignments). "
            f"Phase 2 (axial consolidation): within each theme, semantically similar codes "
            f"were merged into higher-order categories using LLM-assisted analysis, targeting "
            f"k={target_k} reporting codes. The original fine-grained codebook "
            f"({n_original} codes) is retained as audit trail. "
            f"Final reporting codebook: {actual_k} codes across {n_themes} themes."
        )

        elapsed = time.perf_counter() - start_time

        result = ConsolidationResult(
            mapping_table=full_mapping,
            consolidated_codes=consolidated_codes,
            noise_removal=noise_result,
            target_k=target_k,
            actual_k=actual_k,
            methodology_paragraph=methodology,
        )

        if self.verbose:
            self.print_consolidated_summary(result)
            self.print_mapping_table(result)
            self.print_methodology_paragraph(result)
            print(f"\nConsolidation completed in {elapsed:.1f}s")
            print(f"Token usage: {token_tracker.get_summary()}")

        return result

    # =========================================================================
    # DISPLAY FUNCTIONS
    # =========================================================================

    def _print_phase_1_report(self, nr: NoiseRemovalResult) -> None:
        """Print Phase 1 noise removal report."""
        print(f"\n{'=' * 78}")
        print("PHASE 1: NOISE REMOVAL")
        print(f"  Total deduplicated ideas: {nr.total_deduped_ideas}")
        print(f"  Total respondents: {nr.total_respondents}")
        print(f"  Noise budget: {NOISE_BUDGET_PCT * 100:.0f}%")
        print(f"  Codes swept to '{nr.other_label}': {len(nr.noise_codes)}")
        print(f"  Cumulative noise: {nr.noise_pct * 100:.1f}% of ideas")
        print(f"  Surviving codes: {len(nr.surviving_codes)}")
        print(f"{'=' * 78}")

        if nr.noise_codes:
            print("\n  Noise codes (swept to Other):")
            for code, count in sorted(nr.noise_codes.items(), key=lambda x: x[1]):
                pct = _pct(count, nr.total_deduped_ideas)
                print(f"    - {code:<40} n={count:>3} ({pct})")

    def print_consolidated_summary(self, result: ConsolidationResult) -> None:
        """Print consolidated summary in the same tree format as summary.py."""
        print(f"\n{'=' * 78}")
        print("CONSOLIDATED CODE SUMMARY")
        print(f"Original codes: {len(set(m.original_code for m in result.mapping_table))}  |  "
              f"Consolidated codes: {result.actual_k}  |  Target k: {result.target_k}")

        total_deduped = result.noise_removal.total_deduped_ideas if result.noise_removal else 0
        total_respondents = result.noise_removal.total_respondents if result.noise_removal else 0

        print(f"Total ideas: {total_deduped}  |  Total respondents: {total_respondents}")
        print(f"{'=' * 78}")

        # Group consolidated codes by theme
        theme_codes: Dict[str, List[Dict]] = defaultdict(list)
        for c in result.consolidated_codes:
            theme_codes[c["theme"]].append(c)

        max_code_len = max(
            (len(c["code"]) for c in result.consolidated_codes),
            default=20,
        )
        max_code_len = max(max_code_len, 20)

        for theme, codes in sorted(theme_codes.items()):
            if theme == "(Other)":
                continue  # Print Other separately at the end

            theme_freq = sum(c["frequency"] for c in codes)
            theme_resps = _unique_respondents_for_codes(
                [c["code"] for c in codes if "merged_from" not in c],
                self._respondent_codes,
            )
            # For merged codes, approximate respondent count from frequency
            theme_pct = _pct(theme_freq, total_deduped)
            theme_resp_pct = _pct(theme_resps, total_respondents) if theme_resps else ""

            print(f"\n  THEME: {theme}  [n={theme_freq} ({theme_pct})]")

            for i, c in enumerate(sorted(codes, key=lambda x: -x["frequency"])):
                is_last = (i == len(codes) - 1)
                prefix = "  " if is_last else "  "
                freq = c["frequency"]
                pct = _pct(freq, total_deduped)
                merged = c.get("merged_from")
                marker = f" [merged: {', '.join(merged)}]" if merged else ""
                print(f"  {'└─' if is_last else '├─'} {c['code']:<{max_code_len}}  "
                      f"n={freq:>3} ({pct}){marker}")

        # Other bucket
        other_codes = theme_codes.get("(Other)", [])
        if other_codes:
            c = other_codes[0]
            freq = c["frequency"]
            pct = _pct(freq, total_deduped)
            n_merged = len(c.get("merged_from", []))
            print(f"\n  {'─' * 74}")
            print(f"  {c['code']:<{max_code_len + 4}}"
                  f"n={freq:>3} ({pct})  [{n_merged} codes consolidated]")

        print(f"\n{'=' * 78}")

    def print_mapping_table(self, result: ConsolidationResult) -> None:
        """Print the audit trail mapping table."""
        print(f"\n{'=' * 78}")
        print("MAPPING TABLE (audit trail)")
        print(f"{'=' * 78}")

        # Group by theme
        by_theme: Dict[str, List[ConsolidationMappingEntry]] = defaultdict(list)
        for m in result.mapping_table:
            by_theme[m.theme].append(m)

        max_orig = max((len(m.original_code) for m in result.mapping_table), default=20)
        max_cons = max((len(m.consolidated_code) for m in result.mapping_table), default=20)

        for theme, entries in sorted(by_theme.items()):
            print(f"\n  [{theme}]")
            for m in sorted(entries, key=lambda x: (x.action, x.original_code)):
                arrow = "->" if m.action == "merged" or m.action == "noise_removed" else "=="
                action_label = {"retained": "kept", "merged": "merged", "noise_removed": "noise"}[m.action]
                print(f"    {m.original_code:<{max_orig}}  {arrow}  "
                      f"{m.consolidated_code:<{max_cons}}  ({action_label})")

    def print_methodology_paragraph(self, result: ConsolidationResult) -> None:
        """Print the methodology paragraph."""
        print(f"\n{'=' * 78}")
        print("METHODOLOGY PARAGRAPH")
        print(f"{'=' * 78}")
        print(f"\n{result.methodology_paragraph}\n")


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    @dataclass
    class _StandaloneConfig:
        filename: str = TEST_DATA.filename
        var_name: str = TEST_DATA.var_name
        sample_size: Optional[int] = TEST_DATA.sample_size

    config = _StandaloneConfig()
    print(f"Loading data for {config.filename} / {config.var_name} (sample={config.sample_size})...")

    consolidator = CodebookConsolidator.from_cache(config)
    result = consolidator.consolidate()

    # Save to cache
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size,
    )
    cache_manager = CacheManager(CacheConfig())
    cache_manager.save_metadata_to_cache(
        metadata=result,
        filename=config.filename,
        step="codebook_consolidation",
        variable_key=variable_key,
    )
    print("Result saved to cache (step: codebook_consolidation)")
