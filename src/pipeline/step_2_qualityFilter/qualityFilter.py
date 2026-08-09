"""Step 2: Quality Filter — per-response LLM grading to flag noise.

Delegates all API orchestration (rate limiting, concurrency, retry, warm-up)
to SmoothRequester. This module owns only: prompt building, response parsing,
result assembly.
"""

import asyncio
import math
import re
import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

import models
from config import DEFAULT_LANGUAGE, get_reasoning_params
from pipeline.step_2_qualityFilter.config_qualityFilter import (
    QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG,
)
from pipeline.step_2_qualityFilter.prompts_qualityFilter import (
    GRADER_INSTRUCTIONS_NANO, GRADER_INSTRUCTIONS_STRUCTURED,
    QualityFilterStructuredResponse,
)
from utils.smoothRequester import SmoothRequester
from utils.llm import token_tracker
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.promptPrinter import PromptPrinter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# CATEGORY → CODE PROJECTION
# =============================================================================
# The prompt distinguishes six categories; the Motivaction missing-value
# convention has three slots. This table is the projection between them, and it
# is deliberately many-to-one. The category itself survives on
# QualityFilteredModel.quality_filter_category, so the quality report keeps the
# detail that the .sav cannot carry.
CATEGORY_TO_CODE = {
    "1": 99999997,   # cannot give the answer      → weet niet / geen mening
    "2": 99999998,   # not applicable              → geen van dezen / geen van allen
    "3": 99999998,   # no content to report        → idem
    "4": 99999999,   # referring elsewhere         → missing
    "5": 99999999,   # no text                     → missing
    "6": 99999999,   # meaningless text            → missing
}

# Code assigned without an LLM category: empties caught before dispatch, and any
# code step 1 passed through from the source data.
CODE_NO_TEXT = 99999999


def parse_category(raw_text: str) -> Optional[int]:
    """Parse the <category> tag from raw LLM output into a category number.

    Returns None (= keep the response); unparseable output must not cost a
    response. Mapping to a code happens in one place, in parse_fn.
    """
    match = re.search(r'<category>\s*(.*?)\s*</category>', raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        return int(value) if value in CATEGORY_TO_CODE else None
    # Fallback: scan for a standalone category number near the end of the response
    match = re.search(r'\b([1-6])\b', raw_text[-50:])
    return int(match.group(1)) if match else None


# =============================================================================
# GRADER
# =============================================================================

class Grader:
    def __init__(
        self,
        responses: List[models.PreprocessedModel],
        var_lab: str,
        config: Optional[QualityFilterConfig] = None,
        verbose: bool = False,
        prompt_printer: Optional[PromptPrinter] = None,
        cost_tracker=None,
    ):
        self.responses = responses
        self.question = var_lab
        self.config = config or DEFAULT_QUALITY_FILTER_CONFIG
        self.model = self.config.model
        self._is_nano = "nano" in self.model.lower()
        self.cost_tracker = cost_tracker

        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False

        self.stats = {}
        self.failure_log = []

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_2_quality_filter", {
                "grading": self.model,
            })

    # --- Prompt building ---

    def _build_individual_prompt(self, var_lab: str, response_id: str, response_text: str) -> str:
        template = GRADER_INSTRUCTIONS_NANO if self._is_nano else GRADER_INSTRUCTIONS_STRUCTURED
        return template.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            response_text=response_text,
        )

    # --- SmoothRequester callbacks ---

    def _build_prepare_fn(self):
        grader = self

        def prepare_fn(task: Dict) -> Dict:
            prompt = grader._build_individual_prompt(
                grader.question, task['respondent_id'], task['response_text'],
            )

            if grader.prompt_printer and not grader._captured_prompt:
                grader.prompt_printer.capture_prompt(
                    step_name="quality_filter",
                    utility_name="QualityFilter",
                    prompt_content=prompt,
                    prompt_type="quality_assessment",
                    metadata={
                        "model": grader.model,
                        "var_lab": grader.question,
                        "language": DEFAULT_LANGUAGE,
                    },
                )
                grader._captured_prompt = True

            return {
                'prompt': prompt,
                'response_model': None if grader._is_nano else QualityFilterStructuredResponse,
                'temperature': grader.config.temperature,
                'max_tokens': grader.config.max_tokens,
                'max_retries': grader.config.retries,
                'extra_kwargs': get_reasoning_params(grader.model, phase="quality_filter"),
            }

        return prepare_fn

    def _build_parse_fn(self):
        grader = self

        def parse_fn(task: Dict, response) -> Optional[models.QualityFilteredModel]:
            if grader._is_nano:
                raw_text = response.output_text if hasattr(response, 'output_text') else str(response)
                category = parse_category(raw_text)
            else:
                category = response.category

            quality_code = CATEGORY_TO_CODE.get(str(category)) if category else None

            return models.QualityFilteredModel(
                respondent_id=task['respondent_id'],
                response=task['response_text'],
                quality_filter=quality_code is not None,
                quality_filter_code=quality_code,
                quality_filter_category=category,
            )

        return parse_fn

    def _build_fallback_fn(self):
        def fallback_fn(task: Dict, reason: str) -> models.QualityFilteredModel:
            return models.QualityFilteredModel(
                respondent_id=task['respondent_id'],
                response=task['response_text'],
                quality_filter=False,
                quality_filter_code=-1,
            )
        return fallback_fn

    # --- Task preparation ---

    def _prepare_tasks(self) -> List[Dict]:
        tasks = []
        for response in self.responses:
            if response.quality_filter_code is not None:
                continue

            response_text = response.response
            if isinstance(response_text, (float, int, np.floating, np.integer)):
                try:
                    if math.isnan(float(response_text)) or math.isinf(float(response_text)):
                        response_text = ''
                    else:
                        response_text = str(response_text)
                except (ValueError, TypeError):
                    response_text = str(response_text)
            elif response_text is None:
                response_text = ''

            tasks.append({
                'respondent_id': response.respondent_id,
                'response_text': response_text,
            })

        return tasks

    # --- Failure report ---

    def get_failure_report(self, total_responses: int = None) -> str:
        total = total_responses or len(self.responses)
        n_failures = len(self.failure_log)

        if n_failures == 0:
            return f"PROCESSING ERRORS: 0 of {total} responses (0%)"

        lines = [f"PROCESSING ERRORS: {n_failures} of {total} responses ({n_failures / max(total, 1) * 100:.1f}%)"]

        reason_counts = Counter()
        for f in self.failure_log:
            key = f['error_type'] if f['reason'] == 'exception' else f['reason']
            reason_counts[key] += 1
        lines.append(f"  Breakdown: {', '.join(f'{count}x {reason}' for reason, count in reason_counts.most_common())}")

        return "\n".join(lines)

    # --- Main entry point ---

    def grade(self) -> List[models.QualityFilteredModel]:
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)

        # Pre-filter empty/None responses. These never reach the LLM, so they get
        # the code directly — the same one category 5 (no text) projects onto.
        empty_values = {'none', 'nan', '<na>', 'na', ''}
        pre_filter_count = 0
        for r in self.responses:
            if r.quality_filter_code is None:
                response_text = str(r.response).strip() if r.response else ""
                if not response_text or response_text.lower() in empty_values:
                    r.quality_filter_code = CODE_NO_TEXT
                    r.quality_filter = True
                    pre_filter_count += 1
        if pre_filter_count > 0:
            print(f"Pre-filtered {pre_filter_count} empty/None responses (code {CODE_NO_TEXT})")

        # Separate items that need LLM evaluation
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        pre_filtered_items = [r for r in self.responses if r.quality_filter_code is not None]

        self.verbose_reporter.step_start("Quality Assessment")
        self.verbose_reporter.stat_line(f"Model: {self.model}")
        self.verbose_reporter.stat_line(f"Items needing LLM evaluation: {len(items_to_process)}")
        self.verbose_reporter.stat_line(f"Pre-filtered items: {len(pre_filtered_items)}")

        # Process via SmoothRequester
        llm_results_map = {}
        if items_to_process:
            tasks = self._prepare_tasks()

            requester = SmoothRequester(
                model=self.model,
                phase_key="step2_quality_filter",
                num_tasks=len(tasks),
                verbose=self.verbose_reporter.enabled,
            )

            _snap_before = token_tracker.snapshot() if self.cost_tracker else None

            llm_results = asyncio.run(
                requester.process_all(
                    tasks,
                    self._build_prepare_fn(),
                    self._build_parse_fn(),
                    self._build_fallback_fn(),
                )
            )

            if self.cost_tracker and _snap_before is not None:
                self.cost_tracker.record_phase(
                    "step_2_quality_filter", "grading",
                    _snap_before, token_tracker.snapshot(), self.model)

            self.stats = requester.stats
            self.failure_log = requester.failure_log

            # Build lookup for merging
            for result in llm_results:
                if result is not None:
                    llm_results_map[str(result.respondent_id)] = result
        else:
            self.verbose_reporter.stat_line("No items require LLM evaluation")

        # Merge results in original order
        merged_results = []
        for original in self.responses:
            if original.quality_filter_code is not None:
                merged_results.append(original)
            elif str(original.respondent_id) in llm_results_map:
                merged_results.append(llm_results_map[str(original.respondent_id)])
            else:
                original.quality_filter = False
                original.quality_filter_code = 0
                merged_results.append(original)

        # Summary statistics
        self._stats.output_count = len([r for r in merged_results if not r.quality_filter])
        self._stats.end_timing()

        total = len(merged_results)
        llm_processed = len(items_to_process)
        pre_filtered_count = len(pre_filtered_items)

        # Count codes assigned by the LLM only (graded items), so pre-filtered
        # empties are not conflated with an LLM-assigned no-text on the same code.
        llm_graded = list(llm_results_map.values())
        llm_codes = Counter(r.quality_filter_code for r in llm_graded)
        llm_categories = Counter(r.quality_filter_category for r in llm_graded)

        # Honest totals: a response is filtered iff quality_filter is True.
        total_filtered = sum(1 for r in merged_results if r.quality_filter)
        meaningful = total - total_filtered

        # Codes are the Motivaction convention; categories are what the model
        # decided. Several categories share one code, so report both — the code
        # is what ships in the .sav, the category is what it was based on.
        code_labels = {
            99999997: "weet niet / geen mening",
            99999998: "geen van dezen",
            99999999: "missing",
        }
        category_labels = {
            1: "cannot give the answer",
            2: "not applicable",
            3: "no content to report",
            4: "referring elsewhere",
            5: "no text",
            6: "meaningless text",
        }
        pre_filtered_codes = Counter(r.quality_filter_code for r in pre_filtered_items)

        def code_line(label, code, count):
            return f"    → {label:<23}{'(' + str(code) + '):':<13}{count:>5}"

        print(f"\n{'─' * 60}")
        print(f"SUMMARY ({total} total responses)")
        print(f"{'─' * 60}")
        print(f"  Pre-filtered:                           {pre_filtered_count:>5}")
        for code, count in sorted(pre_filtered_codes.items(), key=lambda kv: -kv[1]):
            print(code_line(code_labels.get(code, "unknown code"), code, count))
        print(f"      of which caught here (empty/NA):    {pre_filter_count:>5}")
        print(f"  LLM evaluated:                          {llm_processed:>5}")
        for code, label in code_labels.items():
            print(code_line(label, code, llm_codes[code]))
        print(code_line("meaningful", "null", llm_codes[None]))
        if llm_codes[-1]:
            print(code_line("errors", -1, llm_codes[-1]))
        print(f"{'─' * 60}")
        print("  Categories the LLM assigned, behind those codes:")
        for cat, label in category_labels.items():
            if llm_categories[cat]:
                print(f"      {cat}. {label:<24}→ {CATEGORY_TO_CODE[str(cat)]}  {llm_categories[cat]:>5}")
        print(f"{'─' * 60}")
        print(f"  Total filtered out:                     {total_filtered:>5}  ({total_filtered / total * 100:.1f}%)")
        print(f"  Total meaningful (passed):              {meaningful:>5}  ({meaningful / total * 100:.1f}%)")
        print(f"{'─' * 60}")

        # Filtered examples, one list per category — per category rather than per
        # code, because three categories share code 99999999 and lumping them
        # hides the rare ones. A single shared list is dominated by the largest
        # category, and that is where over-flagging hides. Hand sample_list every
        # distinct text and let it draw at random: duplicates would otherwise
        # crowd the draw, and taking the first N would sample completion order.
        for cat, label in category_labels.items():
            texts = list(dict.fromkeys(
                str(r.response).strip() for r in llm_graded if r.quality_filter_category == cat
            ))
            if texts:
                self.verbose_reporter.sample_list(
                    f"LLM-filtered — {cat}. {label}: {llm_categories[cat]} responses, {len(texts)} distinct",
                    texts,
                    max_samples=self.config.max_filter_examples,
                )

        if self.failure_log:
            print(f"\n{'=' * 70}")
            print(self.get_failure_report(total_responses=total))
            print(f"{'=' * 70}")

        return merged_results
