"""
IdeaExtractor - Taxonomy-aware idea extraction with optimal rate limiting

Extracts structured ideas from survey responses using LLM with:
- Taxonomy-aware extraction (WHAT/WHY/HOW/WHO/WHEN/WHERE dimensions)
- PID-controlled rate limiting for zero 429 errors
- Template prefix enforcement for normalized idea phrasing
- ExtractionMetadata building for downstream use

Key features:
1. Learned tiktoken→API token offset (accounts for ~300 token system overhead)
2. Real-time TPM tracking via sliding window
3. PID-style continuous throughput adjustment
4. Bootstrap measurement for accurate rate limit calibration
"""

# === MODULES ========================================================================================================
import asyncio
import random
import re
import time
import statistics
import itertools
import logging
import unicodedata
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

import nest_asyncio
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, FALLBACK_TPM, FALLBACK_RPM
from utils.llm import create_client, llm_create_async, ProbeResponse, RateLimits, extract_rate_limits_from_response

# === PROMPTS & RESPONSE MODELS ======================================================================================
# Use experimental prompts from local prompts_exp.py (not production prompts.py)
# Response models are co-located with prompts following instructor schema pattern
try:
    from .prompts_exp import (
        # Prompts
        CONSOLIDATE_SPECIFIERS_GROUP1, CONSOLIDATE_SPECIFIERS_GROUP2,
        CONTEXT_SPECIFIER_PROMPT1, CONTEXT_SPECIFIER_PROMPT2,
        CODING_DIMENSION_SCORING_PROMPT, CODING_DIMENSION_CONSOLIDATION_PROMPT,
        TAXONOMY_AWARE_SUBJECT_PROMPT, TAXONOMY_ENRICHED_EXTRACTION_PROMPT,
        # Response models
        GenericSpecifierGroup1Response,
        GenericSpecifierGroup2Response,
        CodingDimensionChunkResponse,
        CodingDimensionConsolidatedResponse,
        SubjectExtractionResponse,
        SemanticTaxonomyResponse,
        TaxonomyEnrichedIdeaResponse,
        # Model factories (axis-specific response models)
        create_subject_extraction_model,
        create_taxonomy_enriched_model,
    )
except ImportError:
    # Fallback for direct script execution
    from prompts_exp import (
        # Prompts
        CONSOLIDATE_SPECIFIERS_GROUP1, CONSOLIDATE_SPECIFIERS_GROUP2,
        CONTEXT_SPECIFIER_PROMPT1, CONTEXT_SPECIFIER_PROMPT2,
        CODING_DIMENSION_SCORING_PROMPT, CODING_DIMENSION_CONSOLIDATION_PROMPT,
        TAXONOMY_AWARE_SUBJECT_PROMPT, TAXONOMY_ENRICHED_EXTRACTION_PROMPT,
        # Response models
        GenericSpecifierGroup1Response,
        GenericSpecifierGroup2Response,
        CodingDimensionChunkResponse,
        CodingDimensionConsolidatedResponse,
        SubjectExtractionResponse,
        SemanticTaxonomyResponse,
        TaxonomyEnrichedIdeaResponse,
        # Model factories (axis-specific response models)
        create_subject_extraction_model,
        create_taxonomy_enriched_model,
    )

# === TEMPLATE LOOKUP =============================================================================================
try:
    from .template_lookup import TEMPLATE_LOOKUP
except ImportError:
    from template_lookup import TEMPLATE_LOOKUP

# === STEP-SPECIFIC CONFIG =============================================================================================
from config_ideaExtractor import (
    DEFAULT_TOKEN_HISTORY_CONFIG,
    DEFAULT_TIKTOKEN_OFFSET_CONFIG,
    DEFAULT_TIMEOUT_CONFIG,
    DEFAULT_REPORTING_CONFIG,
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_PID_CONTROLLER_CONFIG,
    DEFAULT_TPM_TRACKING_CONFIG,
    DEFAULT_THROUGHPUT_CONFIG,
    DEFAULT_SPECIFIER_CONFIG,
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cached_resources import get_openai_client, get_tiktoken_encoding

async_client = get_openai_client(OPENAI_API_KEY)


# === TEMPLATE LOOKUP HELPER =============================================================================================


def _escape_braces_for_format(s: str) -> str:
    """Escape all curly braces for use in str.format().

    Converts { to {{ and } to }} so that .format() doesn't
    try to interpret them as placeholders.
    """
    return s.replace("{", "{{").replace("}", "}}")


def _resolve_slot_type(type_name: str) -> dict:
    """Resolve a slot type name against the type_system in TEMPLATE_LOOKUP.

    If the type is an alias (e.g. 'noun_like_phrase'), expands to all member
    definitions. If it's a concrete type, returns its single definition.

    Returns:
        dict with keys: type_name, is_alias, description, short_label
    """
    ts = TEMPLATE_LOOKUP.get("type_system", {})
    definitions = ts.get("definitions", {})
    aliases = ts.get("aliases", {})

    if type_name in aliases:
        concrete_types = aliases[type_name]
        descs = [definitions[t] for t in concrete_types if t in definitions]
        return {
            "type_name": type_name,
            "is_alias": True,
            "description": " | ".join(descs),
            "short_label": ", ".join(concrete_types),
        }

    if type_name in definitions:
        return {
            "type_name": type_name,
            "is_alias": False,
            "description": definitions[type_name],
            "short_label": type_name,
        }

    return {"type_name": type_name, "is_alias": False, "description": "", "short_label": type_name}


def _resolve_schema_data(axis_data: dict) -> dict:
    """Resolve template_schema data from an axis's schema reference.

    Handles HOW's conditional dict schema (default + prescriptive variant)
    and regular string schema references on other axes.

    Args:
        axis_data: Axis definition from TEMPLATE_LOOKUP["axes"][axis]

    Returns:
        The matching template_schema dict, or {} if not found.
    """
    schema_ref = axis_data.get("schema", "")
    if isinstance(schema_ref, dict):
        schema_name = schema_ref.get("default", "")
    else:
        schema_name = schema_ref
    return TEMPLATE_LOOKUP.get("template_schemas", {}).get(schema_name, {})


def _format_lookup_for_axis(axis: str, language: str = "") -> dict:
    """Format TEMPLATE_LOOKUP data for a specific axis.

    Reads axis-specific data from TEMPLATE_LOOKUP["axes"] and resolves the
    linked template_schema to get structural_forms and notes.  Formats
    everything for injection into TAXONOMY_AWARE_SUBJECT_PROMPT and
    TAXONOMY_ENRICHED_EXTRACTION_PROMPT.

    Args:
        axis: Taxonomy axis code (WHAT, WHY, HOW, WHO, WHEN, WHERE)
        language: Response language (e.g. "Dutch") — substituted into
            slot_guidance descriptions that contain {language}.

    Returns:
        dict with keys for both prompts:
        - Subject prompt: axis_anchor, axis_dimension_description,
          axis_included_concepts, axis_excluded_concepts, axis_pattern,
          axis_slot_guidance, noun_phrase_descriptor, axis_slots,
          axis_structural_forms, axis_instruction
        - Extraction prompt: axis_anchor, axis_dimension_description,
          axis_allowed_concepts, axis_excluded_concepts, axis_disambiguation_rules,
          marker, axis_slot_guidance, axis_node_instruction,
          axis_category_instruction, axis_taxonomy_phrase_instruction, axis_focus_rules
    """
    import json

    # Fallback to WHAT if axis not found
    axis_data = TEMPLATE_LOOKUP["axes"].get(axis, TEMPLATE_LOOKUP["axes"]["WHAT"])

    # Pattern lives directly on the axis
    axis_pattern = axis_data["pattern"]

    # Resolve schema reference to get structural_forms and notes from template_schemas.
    schema_data = _resolve_schema_data(axis_data)

    # Substitute {language} in slot_guidance descriptions before formatting.
    # Some descriptions (e.g. WHAT's ANCHOR_SUBJECT) reference {language}.
    slot_guidance = {
        k: v.replace("{language}", language) if language else v
        for k, v in axis_data["slot_guidance"].items()
    }

    # Resolve type_system for each slot (position-based: anchor first, dimension second)
    schema_slots = schema_data.get("slots", {})
    schema_slot_items = list(schema_slots.items())
    slot_type_map = {"anchor": {}, "dimension": {}}

    # Enrich slot_guidance with type hints, then serialize both formats
    enriched_slot_guidance = {}
    slots_formatted_lines = []
    for idx, (slot_name, slot_desc) in enumerate(slot_guidance.items()):
        # Match by position: idx 0 = anchor, idx 1 = dimension
        if idx < len(schema_slot_items):
            _schema_slot_name, schema_slot_meta = schema_slot_items[idx]
            slot_type_name = schema_slot_meta.get("type", "")
            resolved = _resolve_slot_type(slot_type_name) if slot_type_name else {}
            role = "anchor" if idx == 0 else "dimension"
            slot_type_map[role] = resolved

            if resolved.get("is_alias"):
                type_hint = f"Form: {resolved['short_label']}."
            elif resolved.get("description"):
                type_hint = f"Form: {resolved['description']}"
            else:
                type_hint = ""

            if type_hint:
                enriched_desc = f"{slot_desc} {type_hint}"
            else:
                enriched_desc = slot_desc
            enriched_slot_guidance[slot_name] = enriched_desc
            slots_formatted_lines.append(f"- {slot_name}: {enriched_desc}")
        else:
            enriched_slot_guidance[slot_name] = slot_desc
            slots_formatted_lines.append(f"- {slot_name}: {slot_desc}")
    slots_formatted = "\n".join(slots_formatted_lines)

    # Serialize enriched slot_guidance to JSON and escape braces for .format()
    slot_guidance_json = _escape_braces_for_format(
        json.dumps(enriched_slot_guidance, indent=2)
    )

    # Structural forms from template_schema (not from axis)
    structural_forms = schema_data.get("structural_forms", [])
    forms_formatted = "\n".join(f"  {f}" for f in structural_forms)

    # Schema notes (optional additional guidance)
    schema_notes = schema_data.get("notes", [])
    notes_formatted = "\n".join(f"- {n}" for n in schema_notes)

    # Extract prompt_rules (axis-specific instructions for extraction prompt)
    prompt_rules = axis_data.get("prompt_rules", {})
    focus_rules_list = prompt_rules.get("focus_rules", [])
    focus_rules_formatted = "\n".join(f"- {r}" for r in focus_rules_list)

    # Disambiguation rules (optional, only on HOW axis)
    disambiguation_rules = axis_data.get("disambiguation_rules", [])
    disambiguation_formatted = "\n".join(f"- {r}" for r in disambiguation_rules) if disambiguation_rules else "None"

    # Concepts as comma-separated strings
    allowed_concepts_str = ", ".join(axis_data.get("allowed_concepts", []))
    excluded_concepts_str = ", ".join(axis_data.get("excluded_concepts", []))

    # Marker token: axis-specific dimension slot name (e.g., [OUTCOME_ENABLER] for HOW)
    slot_guidance_keys = list(slot_guidance.keys())
    dimension_marker = slot_guidance_keys[1] if len(slot_guidance_keys) > 1 else TEMPLATE_LOOKUP["marker"]
    marker = TEMPLATE_LOOKUP["marker"]

    # --- Dimension taxonomy data (semantic category table + priority rules) ---
    dim_tax = TEMPLATE_LOOKUP.get("dimension_taxonomy", {})
    dim_data = dim_tax.get("dimensions", {}).get(axis, {})
    axis_interpretation = dim_data.get("axis_interpretation", {})

    # Format semantic category table for prompt injection
    category_table_lines = []
    for cat, interpretation in axis_interpretation.items():
        category_table_lines.append(f"- **{cat}**: {interpretation}")
    semantic_category_table = "\n".join(category_table_lines) if category_table_lines else "No dimension-specific category table available."

    # Format priority rules (prefer dimension-specific, fall back to universal)
    decision_rules = dim_data.get("decision_reminder", dim_tax.get("priority_rules", []))
    priority_rules = "\n".join(f"{i+1}. {r}" for i, r in enumerate(decision_rules)) if decision_rules else "Apply best judgment."

    return {
        # Keys for TAXONOMY_AWARE_SUBJECT_PROMPT
        "axis_anchor": axis_data["anchor"],
        "axis_dimension_description": axis_data["dimension_description"],
        "axis_included_concepts": allowed_concepts_str,
        "axis_excluded_concepts": excluded_concepts_str,
        "axis_pattern": _escape_braces_for_format(axis_pattern),
        "axis_slot_guidance": slot_guidance_json,
        "noun_phrase_descriptor": axis_data.get("noun_phrase_descriptor", ""),
        "axis_slots": slots_formatted,
        "axis_structural_forms": _escape_braces_for_format(forms_formatted),
        "axis_instruction": axis_data.get("instruction", ""),
        "schema_notes": _escape_braces_for_format(notes_formatted),
        # Keys for TAXONOMY_ENRICHED_EXTRACTION_PROMPT
        "axis_allowed_concepts": allowed_concepts_str,
        "axis_disambiguation_rules": disambiguation_formatted,
        "marker": marker,
        "axis_node_instruction": _escape_braces_for_format(
            prompt_rules.get("node_instruction", "")
        ),
        "axis_category_instruction": _escape_braces_for_format(
            prompt_rules.get("category_instruction", "")
        ),
        "axis_taxonomy_phrase_instruction": _escape_braces_for_format(
            prompt_rules.get("taxonomy_phrase_instruction", "")
        ),
        "axis_focus_rules": _escape_braces_for_format(focus_rules_formatted),
        "axis_instance_instruction": _escape_braces_for_format(
            prompt_rules.get("instance_instruction", "")
        ),
        "axis_root_instruction": _escape_braces_for_format(
            prompt_rules.get("root_instruction", "")
        ),
        # Resolved type_system info for factory functions
        "slot_type_map": slot_type_map,
        # Axis-specific dimension marker (e.g., [OUTCOME_ENABLER] for HOW)
        "dimension_marker": dimension_marker,
        # Second slot_guidance value (dimension slot description) for inline use
        "slot_guidance_second": list(enriched_slot_guidance.values())[1] if len(enriched_slot_guidance) > 1 else "",
        # Dimension taxonomy placeholders (semantic category table + priority rules)
        "semantic_category_table": _escape_braces_for_format(semantic_category_table),
        "priority_rules": _escape_braces_for_format(priority_rules),
    }


# === CONSTANTS (from config_ideaExtractor.py) =========================================================================
# Token history windows
INPUT_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.input_history_maxlen
OUTPUT_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.output_history_maxlen
OUTPUT_RATIO_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.output_ratio_history_maxlen
DEFAULT_OUTPUT_RATIO = DEFAULT_TOKEN_HISTORY_CONFIG.default_output_ratio
ERROR_WINDOW_SIZE = DEFAULT_TOKEN_HISTORY_CONFIG.error_window_size

# Tiktoken → API token offset learning
TIKTOKEN_API_OFFSET_DEFAULT = DEFAULT_TIKTOKEN_OFFSET_CONFIG.api_offset_default
TIKTOKEN_OFFSET_HISTORY_MAXLEN = DEFAULT_TIKTOKEN_OFFSET_CONFIG.offset_history_maxlen
TIKTOKEN_OFFSET_MIN_SAMPLES = DEFAULT_TIKTOKEN_OFFSET_CONFIG.offset_min_samples

# Timeouts and latency
DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_timeout_seconds
DEFAULT_LATENCY_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_latency_seconds
MAX_TOKEN_ACQUIRE_ATTEMPTS = DEFAULT_TIMEOUT_CONFIG.max_token_acquire_attempts
BOOTSTRAP_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_CONFIG.bootstrap_timeout_seconds

# Reporting intervals
PROGRESS_REPORT_INTERVAL = DEFAULT_REPORTING_CONFIG.progress_report_interval
DIAGNOSTIC_INTERVAL = DEFAULT_REPORTING_CONFIG.diagnostic_interval
ADJUSTMENT_INTERVAL = DEFAULT_REPORTING_CONFIG.adjustment_interval

# Bootstrap settings
BOOTSTRAP_NUM_PROBES = DEFAULT_BOOTSTRAP_CONFIG.num_probes
DEFAULT_AVG_TOKENS = DEFAULT_BOOTSTRAP_CONFIG.default_avg_tokens
SAMPLE_SIZE_FOR_TOKEN_ESTIMATION = DEFAULT_BOOTSTRAP_CONFIG.sample_size_for_token_estimation

# PID-style continuous adjustment (asymmetric gains)
PID_KP_UP = DEFAULT_PID_CONTROLLER_CONFIG.kp_up
PID_KP_DOWN = DEFAULT_PID_CONTROLLER_CONFIG.kp_down
PID_KI = DEFAULT_PID_CONTROLLER_CONFIG.ki
PID_KD = DEFAULT_PID_CONTROLLER_CONFIG.kd
PID_MIN_ADJUSTMENT = DEFAULT_PID_CONTROLLER_CONFIG.min_adjustment
PID_MAX_ADJUSTMENT = DEFAULT_PID_CONTROLLER_CONFIG.max_adjustment

# Real-time TPM tracking
TPM_SLIDING_WINDOW_SECONDS = DEFAULT_TPM_TRACKING_CONFIG.sliding_window_seconds
TPM_SAMPLE_INTERVAL = DEFAULT_TPM_TRACKING_CONFIG.sample_interval
TPM_TARGET_UTILIZATION = DEFAULT_TPM_TRACKING_CONFIG.target_utilization

# Threshold-based adjustment (fallback to PID)
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = DEFAULT_THROUGHPUT_CONFIG.adjustment_min_samples
THROUGHPUT_ADJUSTMENT_THRESHOLD = DEFAULT_THROUGHPUT_CONFIG.adjustment_threshold

# Generic specifier settings
GENERIC_SPECIFIER_SAMPLE_MIN = DEFAULT_SPECIFIER_CONFIG.sample_min
GENERIC_SPECIFIER_SAMPLE_MAX = DEFAULT_SPECIFIER_CONFIG.sample_max
GENERIC_SPECIFIER_CHUNK_SIZE = DEFAULT_SPECIFIER_CONFIG.chunk_size
MAX_SPECIFIER_WORKERS = DEFAULT_SPECIFIER_CONFIG.max_workers


# === RATE LIMITING CLASSES  ========================================================================================================
class TokenBucket:
    """Simple token bucket for TPM limiting"""
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens_needed):
        """Acquire tokens, returning wait time if not available"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            # Regenerate tokens based on time elapsed
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now

            if self.available >= tokens_needed:
                self.available -= tokens_needed
                return True
            else:
                # Calculate wait time
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                return wait_seconds

    async def wait_and_acquire(self, tokens_needed):
        """Wait if necessary and acquire tokens"""
        attempts = 0
        while attempts < MAX_TOKEN_ACQUIRE_ATTEMPTS:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                # result is wait_seconds
                await asyncio.sleep(result)

        raise RuntimeError(f"Failed to acquire {tokens_needed} tokens after {MAX_TOKEN_ACQUIRE_ATTEMPTS} attempts")

    async def reconcile(self, delta_tokens: int) -> None:
        """Reconcile actual token usage against estimate."""
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


class LatencyTracker:
    """Simple EMA tracker for latencies"""
    def __init__(self, processing_config: Optional[ProcessingConfig] = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)

    def add(self, value):
        """Add a latency measurement"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def get_timeout(self, est_tokens):
        """Calculate timeout based on EMA and token count with configurable bounds"""
        config = self.processing_config
        if not self.values:
            return max(config.adaptive_timeout_min_seconds, DEFAULT_TIMEOUT_SECONDS)

        # Use P95 latency as base
        p95 = np.percentile(list(self.values), 95)
        # Simple linear scaling with token count
        token_factor = est_tokens / 1000
        timeout = p95 + (token_factor * 0.1)
        # Apply margin and configurable bounds
        return max(config.adaptive_timeout_min_seconds, min(config.adaptive_timeout_max_seconds, timeout * config.adaptive_timeout_margin))

    def get_avg_latency(self):
        """Get average latency for concurrency calculations"""
        if not self.values:
            return DEFAULT_LATENCY_SECONDS
        return self.ema if self.ema is not None else DEFAULT_LATENCY_SECONDS


# === OPTIMAL STRATEGY CLASSES ========================================================================================================

class TiktokenOffsetLearner:
    """Learns the offset between tiktoken counts and actual API token counts.

    The API always reports more tokens than tiktoken because of:
    - System messages added by the API
    - Instructor/structured output overhead
    - Message formatting tokens

    This class learns the average offset and applies it to estimates.
    """
    def __init__(self, default_offset: int = TIKTOKEN_API_OFFSET_DEFAULT):
        self.default_offset = default_offset
        self.offsets = deque(maxlen=TIKTOKEN_OFFSET_HISTORY_MAXLEN)
        self._learned_offset = None

    def record(self, tiktoken_count: int, api_count: int):
        """Record a tiktoken vs API count pair to learn the offset."""
        offset = api_count - tiktoken_count
        self.offsets.append(offset)

        # Update learned offset when we have enough samples
        if len(self.offsets) >= TIKTOKEN_OFFSET_MIN_SAMPLES:
            self._learned_offset = int(sum(self.offsets) / len(self.offsets))

    def get_offset(self) -> int:
        """Get the current offset to add to tiktoken counts."""
        if self._learned_offset is not None:
            return self._learned_offset
        return self.default_offset

    def is_learned(self) -> bool:
        """Check if we have enough samples to trust the learned offset."""
        return len(self.offsets) >= TIKTOKEN_OFFSET_MIN_SAMPLES

    def get_stats(self) -> dict:
        """Get statistics about the offset learning."""
        return {
            "samples": len(self.offsets),
            "learned_offset": self._learned_offset,
            "using_offset": self.get_offset(),
            "is_learned": self.is_learned(),
            "min_offset": min(self.offsets) if self.offsets else None,
            "max_offset": max(self.offsets) if self.offsets else None,
        }


class RealTimeTPMTracker:
    """Tracks actual TPM usage in a sliding window for real-time feedback.

    Unlike the token bucket (which tracks available capacity), this tracks
    actual consumption to provide accurate utilization metrics.
    """
    def __init__(self, window_seconds: float = TPM_SLIDING_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.samples = deque()  # (timestamp, tokens) pairs
        self.lock = asyncio.Lock()

    async def record(self, tokens: int):
        """Record token usage at current time."""
        async with self.lock:
            now = time.monotonic()
            self.samples.append((now, tokens))
            self._prune_old_samples(now)

    def _prune_old_samples(self, now: float):
        """Remove samples outside the window."""
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    async def get_current_tpm(self) -> float:
        """Get current TPM rate based on sliding window."""
        async with self.lock:
            now = time.monotonic()
            self._prune_old_samples(now)

            if not self.samples:
                return 0.0

            total_tokens = sum(t for _, t in self.samples)
            elapsed = now - self.samples[0][0] if self.samples else 1.0
            elapsed = max(elapsed, 1.0)  # Avoid division by zero

            # Extrapolate to per-minute rate
            return (total_tokens / elapsed) * 60

    async def get_utilization(self, tpm_limit: int) -> float:
        """Get current TPM utilization as a percentage."""
        current_tpm = await self.get_current_tpm()
        return (current_tpm / tpm_limit) * 100 if tpm_limit > 0 else 0.0


class PIDThroughputController:
    """PID controller for smooth, continuous throughput adjustment.

    Instead of step-based threshold adjustments, this provides gradual
    corrections that converge smoothly to optimal throughput.

    Uses ASYMMETRIC gains:
    - kp_up (0.4): Aggressive when under-utilizing (speed up faster)
    - kp_down (0.2): Gentle when over-utilizing (slow down carefully)

    The controller tracks:
    - Error: difference between target and actual TPM utilization
    - Integral: accumulated error over time (handles persistent bias)
    - Derivative: rate of change (dampens oscillations)
    """
    def __init__(
        self,
        target_utilization: float = TPM_TARGET_UTILIZATION,
        kp_up: float = PID_KP_UP,
        kp_down: float = PID_KP_DOWN,
        ki: float = PID_KI,
        kd: float = PID_KD
    ):
        self.target = target_utilization
        self.kp_up = kp_up      # Gain when under-utilizing (speed up)
        self.kp_down = kp_down  # Gain when over-utilizing (slow down)
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.adjustment_history = deque(maxlen=20)

    def compute_adjustment(self, current_utilization: float) -> float:
        """Compute throughput adjustment factor based on current utilization.

        Args:
            current_utilization: Current TPM utilization (0.0 to 1.0+)

        Returns:
            Adjustment factor to multiply current arrival rate by.
            - >1.0 means speed up (under-utilizing)
            - <1.0 means slow down (over-utilizing)
            - 1.0 means no change
        """
        now = time.monotonic()

        # Error: positive means under-utilizing, negative means over-utilizing
        error = self.target - current_utilization

        # Time delta for integral/derivative
        dt = 1.0  # Default
        if self.last_time is not None:
            dt = max(now - self.last_time, 0.1)
        self.last_time = now

        # Integral term (accumulated error)
        self.integral += error * dt
        # Clamp integral to prevent windup
        self.integral = max(-0.5, min(0.5, self.integral))

        # Derivative term (rate of change)
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        # Asymmetric proportional gain: aggressive up, gentle down
        kp = self.kp_up if error > 0 else self.kp_down

        # PID output
        output = (kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Clamp to reasonable adjustment range
        output = max(-PID_MAX_ADJUSTMENT, min(PID_MAX_ADJUSTMENT, output))

        # Convert to multiplier (1.0 + output)
        # Ignore tiny adjustments
        if abs(output) < PID_MIN_ADJUSTMENT:
            adjustment = 1.0
        else:
            adjustment = 1.0 + output

        self.adjustment_history.append({
            "time": now,
            "utilization": current_utilization,
            "error": error,
            "output": output,
            "adjustment": adjustment
        })

        return adjustment

    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def get_stats(self) -> dict:
        """Get controller statistics."""
        recent = list(self.adjustment_history)[-5:] if self.adjustment_history else []
        return {
            "target_utilization": self.target,
            "integral": self.integral,
            "last_error": self.last_error,
            "recent_adjustments": recent,
            "kp_up": self.kp_up,
            "kp_down": self.kp_down,
            "ki": self.ki,
            "kd": self.kd
        }


# === BOOTSTRAP MEASUREMENT SYSTEM ========================================================================================================
@dataclass
class ApiLimits:
    """API limits structure for bootstrap calculations"""
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits: ApiLimits, latency_seconds: float, avg_tokens: float, processing_config: Optional[ProcessingConfig] = None, cap: Optional[int] = None, min_conc: Optional[int] = None, headroom: Optional[float] = None) -> int:
    """Compute optimal concurrency using Little's Law"""
    config = processing_config or DEFAULT_PROCESSING_CONFIG
    cap = cap if cap is not None else config.concurrency_cap_default
    min_conc = min_conc if min_conc is not None else config.concurrency_min_default
    headroom = headroom if headroom is not None else config.rate_limit_headroom

    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)

    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    candidates = [rpm_throughput, tpm_throughput]
    allowed_rps = max(min(candidates), 0.0)
    target = allowed_rps * latency_seconds   # Little's Law

    return int(max(min(target, cap), min_conc))


async def bootstrap_measure_async(call_fn, n_probes: int = 3):
    """Run n_probes serial calls and return (avg_latency_s, avg_tokens). call_fn() -> usage dict."""
    latencies, tokens = [], []
    for _ in range(n_probes):
        t0 = time.perf_counter()
        usage = await call_fn()  # Let tenacity handle timeouts and retries
        t1 = time.perf_counter()
        latencies.append(max(t1 - t0, 0.001))
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tokens.append(max(pt + ct, 1))
    return sum(latencies)/len(latencies), sum(tokens)/len(tokens)


# === MAIN IDEA EXTRACTOR CLASS ========================================================================================================
class IdeaExtractor:
    def __init__(
        self,
        responses: List[models.QualityFilteredModel],
        var_lab: str,
        config: Optional[SegmentationConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose: bool = False,
        prompt_printer=None,
        verbose_reporter: Optional['VerboseReporter'] = None):

        self.responses = responses
        self.var_lab = var_lab
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.model_config.get_model_for_stage('segmentation')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.IdeasExtractedModel] = []
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        # Capture flags for each prompt type (only capture first instance)
        self._captured_context_specifier1 = False
        self._captured_context_specifier2 = False
        self._captured_taxonomy_chunk = False
        self._captured_consolidate1 = False
        self._captured_consolidate2 = False
        self._captured_taxonomy_consolidation = False
        self._captured_taxonomy_subject = False

        # Initialize tokenizer for token estimation (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Initialize OpenAI client with instructor (supports OpenAI and Azure)
        self.client = create_client(self.model, async_mode=True)

        # Rate limiting setup - use fallback values for initial setup
        self.rate_limits = RateLimits(
            tokens_per_minute=FALLBACK_TPM,
            requests_per_minute=FALLBACK_RPM,
            tokens_per_day=FALLBACK_TPM * 60 * 24
        )

        # Token bucket for TPM limiting
        self.tpm_bucket = TokenBucket(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

        # Adaptive token estimation
        self.input_token_history = deque(maxlen=INPUT_HISTORY_MAXLEN)
        self.output_token_history = deque(maxlen=OUTPUT_HISTORY_MAXLEN)
        self.output_ratio_history = deque(maxlen=OUTPUT_RATIO_HISTORY_MAXLEN)  # Track output/input ratios
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)
        self.first_prompt_tokens = None

        # Rolling average of actual total tokens
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)

        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)

        # Cache for subject extraction (with async lock to prevent race conditions)
        self._subject_cache = {}
        self._subject_lock = asyncio.Lock()

        # Generic specifiers (must be initialized before _calculate_avg_tokens)
        self.generic_specifiers = {}

        # Taxonomy axis (must be initialized before _calculate_avg_tokens)
        self.taxonomy_axis = None
        self.taxonomy_rationale = None
        self.taxonomy_axis_description = None  # Dynamic context-specific description
        self.taxonomy_sample_phrases = []  # Sample phrases from responses for dynamic examples

        # Template prefix for embedding (V3: restored for normalized clustering)
        self.template_prefix = None

        # Taxonomy actionable type (V3: chosen narrow dimension for MECE enforcement)
        self.taxonomy_actionable_type = None

        # Calculate initial average tokens estimate
        self.avg_tokens = self._calculate_avg_tokens()

        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None
        self.current_arrival_rate = None  # V3: Track current rate for PID adjustment

        # === V3 OPTIMAL STRATEGY COMPONENTS ===
        # Tiktoken→API offset learning (accounts for system overhead)
        self.tiktoken_offset_learner = TiktokenOffsetLearner()

        # Real-time TPM tracking (sliding window)
        self.tpm_tracker = RealTimeTPMTracker()

        # PID throughput controller (continuous adjustment)
        self.pid_controller = PIDThroughputController()

        # Bootstrap avg_tokens preserved for diagnostics (set after bootstrap, never updated)
        self.bootstrap_avg_tokens = None

        # V3 stats tracking
        self.v3_stats = {
            'adjustments_made': 0,
            'pid_adjustments': 0,
            'threshold_adjustments': 0,
            'max_tpm_utilization': 0.0,
            'min_tpm_utilization': 100.0,
        }

        # Stats
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0
        }

        # Failure log: tracks each permanent PROCESSING_ERROR with details
        self.failure_log = []  # List of {respondent_id, reason, error_type, response_preview}

    def _calculate_avg_tokens(self) -> int:
        """Calculate average tokens per request for rate limiting.

        V3: Uses placeholder template values for estimation.
        """
        if not self.responses:
            return DEFAULT_AVG_TOKENS

        sample_size = min(SAMPLE_SIZE_FOR_TOKEN_ESTIMATION, len(self.responses))
        sample_responses = self.responses[:sample_size]

        # Store original values to restore after estimation
        original_taxonomy_axis = self.taxonomy_axis
        original_taxonomy_axis_description = self.taxonomy_axis_description
        original_generic_specifiers = self.generic_specifiers

        # Set defaults for token estimation
        self.taxonomy_axis = "WHAT"
        self.taxonomy_axis_description = "general concepts and ideas"
        self.generic_specifiers = {}

        # Placeholder values for template estimation
        placeholder_subject = "the subject"
        placeholder_phrasing_template = "the subject is [ACTIONABLE_TAXONOMY_DIMENSION]"

        token_counts = []
        for response in sample_responses:
            prompt = self._build_taxonomy_enriched_prompt(
                response.respondent_id,
                response.response,
                placeholder_subject,
                placeholder_phrasing_template
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            completion_tokens = int(prompt_tokens * 0.25)
            token_counts.append(prompt_tokens + completion_tokens)

        # Restore original values
        self.taxonomy_axis = original_taxonomy_axis
        self.taxonomy_axis_description = original_taxonomy_axis_description
        self.generic_specifiers = original_generic_specifiers

        return int(statistics.mean(token_counts)) if token_counts else DEFAULT_AVG_TOKENS

    async def _consolidate_specifiers(self, group: int, chunk_results: List[Dict]) -> Dict[str, str]:
        """Consolidate specifier results from multiple chunks using LLM."""
        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            group_name = "Group1 (lang/perspective/intent)" if group == 1 else "Group2 (domain/topic/entity)"
            self.verbose_reporter.stat_line(f"  Consolidating {len(chunk_results)} {group_name} results via LLM...")

        formatted_results = []
        for idx, result in enumerate(chunk_results, 1):
            response_obj = result['response']
            if group == 1:
                formatted_results.append(
                    f"Chunk {idx}:\n"
                    f"  - lang: {response_obj.lang}\n"
                    f"  - perspective: {response_obj.perspective}\n"
                    f"  - intent: {response_obj.intent}"
                )
            else:
                formatted_results.append(
                    f"Chunk {idx}:\n"
                    f"  - domain: {response_obj.domain}\n"
                    f"  - topic: {response_obj.topic}\n"
                    f"  - entity: {response_obj.entity}"
                )

        chunk_results_text = "\n\n".join(formatted_results)

        if group == 1:
            prompt = CONSOLIDATE_SPECIFIERS_GROUP1.format(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text
            )
            response_model = GenericSpecifierGroup1Response
            # Capture first consolidate group 1 prompt
            if self.prompt_printer and not self._captured_consolidate1:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_consolidate_specifiers",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="consolidate_specifiers_group1",
                    metadata={"model": self.model, "survey_question": self.var_lab}
                )
                self._captured_consolidate1 = True
        else:
            prompt = CONSOLIDATE_SPECIFIERS_GROUP2.format(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text
            )
            response_model = GenericSpecifierGroup2Response
            # Capture first consolidate group 2 prompt
            if self.prompt_printer and not self._captured_consolidate2:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_consolidate_specifiers",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="consolidate_specifiers_group2",
                    metadata={"model": self.model, "survey_question": self.var_lab}
                )
                self._captured_consolidate2 = True

        async with self.semaphore:
            await self.tpm_bucket.acquire(2000)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=response_model,
                prompt=prompt,
                temperature=0.0
            )

            await self.tpm_bucket.reconcile(0)

        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            if group == 1:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: lang={response.lang}, perspective={response.perspective}, intent={response.intent}"
                )
            else:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: domain={response.domain}, topic={response.topic}, entity={response.entity}"
                )

        if group == 1:
            return {
                "lang": response.lang,
                "perspective": response.perspective,
                "intent": response.intent
            }
        else:
            return {
                "domain": response.domain,
                "topic": response.topic,
                "entity": response.entity
            }

    async def _consolidate_taxonomy(self, chunk_results: List[Dict], context_specifiers: Dict) -> CodingDimensionConsolidatedResponse:
        """Consolidate taxonomy scores from chunks to select primary + secondary axis.

        Always calls LLM consolidation to generate context-specific axis description,
        even for single chunks.

        Args:
            chunk_results: List of dicts with 'response' containing CodingDimensionChunkResponse
            context_specifiers: Dict with domain, entity, topic, perspective, intent

        Returns:
            CodingDimensionConsolidatedResponse with selected axes and context-specific description
        """
        # Format chunk results for consolidation prompt with full context
        formatted_results = []
        for idx, result in enumerate(chunk_results):
            chunk_response = result['response']

            # Format evidence snippets
            evidence_text = "\n".join([f'    - "{e}"' for e in chunk_response.evidence])

            # Build full chunk summary (without dimension scores)
            chunk_text = f"""Chunk {idx + 1}:
  Primary dimension: {chunk_response.primary_dimension}
  Evidence:
{evidence_text}
  Clarification: {chunk_response.clarification}"""

            formatted_results.append(chunk_text)

        prompt = CODING_DIMENSION_CONSOLIDATION_PROMPT.format(
            language=self.language,
            survey_question=self.var_lab,
            domain=context_specifiers.get('domain', ''),
            entity=context_specifiers.get('entity', ''),
            topic=context_specifiers.get('topic', ''),
            perspective=context_specifiers.get('perspective', ''),
            intent=context_specifiers.get('intent', ''),
            chunk_results="\n\n".join(formatted_results)
        )

        # Capture first taxonomy consolidation prompt
        if self.prompt_printer and not self._captured_taxonomy_consolidation:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction_taxonomy_consolidation",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="taxonomy_consolidation",
                metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
            )
            self._captured_taxonomy_consolidation = True

        async with self.semaphore:
            await self.tpm_bucket.acquire(2000)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=CodingDimensionConsolidatedResponse,
                prompt=prompt,
                temperature=0.0
            )

            await self.tpm_bucket.reconcile(0)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Taxonomy consolidated:")
            self.verbose_reporter.stat_line(f"    Primary: {response.primary_dimension}")
            self.verbose_reporter.stat_line(f"    Rationale: {response.primary_dimension_rationale[:100]}...")

        return response

    def _select_axis_from_single_chunk(self, chunk_response: CodingDimensionChunkResponse) -> CodingDimensionConsolidatedResponse:
        """Select primary axis from a single chunk (no consolidation needed).

        Args:
            chunk_response: CodingDimensionChunkResponse with primary_dimension

        Returns:
            CodingDimensionConsolidatedResponse with selected axes
        """
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Taxonomy (single chunk selection):")
            self.verbose_reporter.stat_line(f"    Primary: {chunk_response.primary_dimension}")

        return CodingDimensionConsolidatedResponse(
            primary_dimension=chunk_response.primary_dimension,
            primary_dimension_rationale=f"Selected from single chunk analysis: {chunk_response.primary_dimension}",
            primary_dimension_description=chunk_response.clarification
        )

    async def _extract_generic_specifiers(self) -> Tuple[Dict[str, str], CodingDimensionConsolidatedResponse]:
        """Extract context specifiers first, then taxonomy axis with context awareness.

        Two-phase extraction:
        - Phase 1: Extract context specifiers (Group 1 + Group 2) in parallel
        - Phase 2: Extract taxonomy axis scoring with context specifiers available

        Returns:
            Tuple of (context_specifiers dict, CodingDimensionConsolidatedResponse)
        """
        sample_size = min(GENERIC_SPECIFIER_SAMPLE_MAX, max(int(0.2 * len(self.responses)), GENERIC_SPECIFIER_SAMPLE_MIN))
        sample = random.sample(self.responses, min(sample_size, len(self.responses)))

        chunk_size = GENERIC_SPECIFIER_CHUNK_SIZE
        chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]
        chunk_texts = ["\n".join([f"- {r.response}" for r in chunk]) for chunk in chunks]

        self.verbose_reporter.stat_line(f"Context + Taxonomy: {len(sample)} samples, {len(chunks)} chunks")

        # === PHASE 1: Context specifiers (parallel) ===
        self.verbose_reporter.stat_line(f"  Phase 1: Extracting context specifiers...")

        context_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            # Group 1: lang/perspective/intent
            context_tasks.append({
                'task_id': f"group1_chunk{chunk_idx}",
                'group': 1,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk)
            })
            # Group 2: domain/topic/entity
            context_tasks.append({
                'task_id': f"group2_chunk{chunk_idx}",
                'group': 2,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk)
            })

        context_results = await self._process_generic_specifier_tasks(context_tasks)

        group1_results = [r for r in context_results if r['group'] == 1]
        group2_results = [r for r in context_results if r['group'] == 2]

        if self.verbose_reporter.enabled and group1_results and group2_results:
            self.verbose_reporter.stat_line(f"  Phase 1 chunk-level results:")
            for r in group1_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group1): "
                    f"lang={r['response'].lang}, perspective={r['response'].perspective}, intent={r['response'].intent}"
                )
            for r in group2_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group2): "
                    f"domain={r['response'].domain}, topic={r['response'].topic}, entity={r['response'].entity}"
                )

        # Handle missing results with fallbacks
        if not group1_results or not group2_results:
            self.verbose_reporter.stat_line(f"  Warning: Generic specifier extraction failed ({len(group1_results)} group1, {len(group2_results)} group2 results)")

            lang_code = "nl-NL" if "dutch" in self.language.lower() or "nederlands" in self.language.lower() else "en-US"

            context_specifiers = {
                "lang": lang_code,
                "perspective": "consumer",
                "intent": "evaluate",
                "domain": "general",
                "topic": "feedback",
                "entity": "unknown"
            }
            self.verbose_reporter.stat_line(f"  Using fallback context defaults: {context_specifiers}")

            # Fallback taxonomy (skip phase 2)
            taxonomy_result = CodingDimensionConsolidatedResponse(
                primary_dimension="WHAT",
                primary_dimension_rationale="Fallback default - no context specifiers available",
                primary_dimension_description=""
            )
            return context_specifiers, taxonomy_result

        # Consolidate Group 1
        if len(group1_results) == 1:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Single chunk - skipping consolidation for Group1")
            group1_consolidated = {
                "lang": group1_results[0]['response'].lang,
                "perspective": group1_results[0]['response'].perspective,
                "intent": group1_results[0]['response'].intent
            }
        else:
            group1_consolidated = await self._consolidate_specifiers(1, group1_results)

        # Consolidate Group 2
        if len(group2_results) == 1:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Single chunk - skipping consolidation for Group2")
            group2_consolidated = {
                "domain": group2_results[0]['response'].domain,
                "topic": group2_results[0]['response'].topic,
                "entity": group2_results[0]['response'].entity
            }
        else:
            group2_consolidated = await self._consolidate_specifiers(2, group2_results)

        context_specifiers = {**group1_consolidated, **group2_consolidated}
        self.verbose_reporter.stat_line(f"  Phase 1 complete. Context specifiers: {context_specifiers}")

        # === PHASE 2: Taxonomy scoring (with context awareness) ===
        self.verbose_reporter.stat_line(f"  Phase 2: Scoring taxonomy axes with context (perspective={context_specifiers.get('perspective')}, intent={context_specifiers.get('intent')})...")

        taxonomy_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            taxonomy_tasks.append({
                'task_id': f"taxonomy_chunk{chunk_idx}",
                'group': 3,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk),
                'context_specifiers': context_specifiers  # Pass context to taxonomy scoring
            })

        taxonomy_results = await self._process_generic_specifier_tasks(taxonomy_tasks)

        if self.verbose_reporter.enabled and taxonomy_results:
            self.verbose_reporter.stat_line(f"  Phase 2 chunk-level results:")
            for r in taxonomy_results:
                chunk_resp = r['response']
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Taxonomy): Dim={chunk_resp.primary_dimension}"
                )

        # Consolidate Taxonomy
        if not taxonomy_results:
            self.verbose_reporter.stat_line(f"  Warning: No taxonomy results - using fallback")
            taxonomy_consolidated = CodingDimensionConsolidatedResponse(
                primary_dimension="WHAT",
                primary_dimension_rationale="Fallback default - no taxonomy results available",
                primary_dimension_description=""
            )
        else:
            taxonomy_consolidated = await self._consolidate_taxonomy(taxonomy_results, context_specifiers)

        self.verbose_reporter.stat_line(f"  Context results: {context_specifiers}")
        self.verbose_reporter.stat_line(f"  Taxonomy: primary={taxonomy_consolidated.primary_dimension}")
        return context_specifiers, taxonomy_consolidated

    async def _process_generic_specifier_tasks(self, tasks: List[Dict]) -> List[Dict]:
        queue = asyncio.Queue()
        results = []

        for task in tasks:
            await queue.put(task)

        num_workers = min(MAX_SPECIFIER_WORKERS, len(tasks))
        for _ in range(num_workers):
            await queue.put(None)

        workers = [
            asyncio.create_task(self._generic_specifier_worker(queue, results))
            for _ in range(num_workers)
        ]

        await asyncio.gather(*workers)

        return results

    async def _generic_specifier_worker(self, queue: asyncio.Queue, results: List):
        """Worker for processing generic specifier AND taxonomy tasks."""
        while True:
            task = await queue.get()
            if task is None:
                break

            try:
                if self.semaphore is None or self.rate_limiter is None:
                    raise RuntimeError(
                        f"Rate limiters not initialized before worker started. "
                        f"semaphore={self.semaphore}, rate_limiter={self.rate_limiter}"
                    )

                async with self.semaphore:
                    await self.tpm_bucket.acquire(DEFAULT_AVG_TOKENS)
                    await self.rate_limiter.acquire()

                    if task['group'] == 1:
                        # Group 1: lang/perspective/intent
                        prompt = CONTEXT_SPECIFIER_PROMPT1.format(
                            language=self.language,
                            survey_question=self.var_lab,
                            chunk_responses=task['chunk_text'],
                            chunk_size=task['chunk_size']
                        )
                        response_model = GenericSpecifierGroup1Response
                        # Capture first context specifier 1 prompt
                        if self.prompt_printer and not self._captured_context_specifier1:
                            self.prompt_printer.capture_prompt(
                                step_name="idea_extraction_context_specifiers",
                                utility_name="IdeaExtractor",
                                prompt_content=prompt,
                                prompt_type="context_specifier_group1",
                                metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
                            )
                            self._captured_context_specifier1 = True
                    elif task['group'] == 2:
                        # Group 2: domain/topic/entity
                        prompt = CONTEXT_SPECIFIER_PROMPT2.format(
                            language=self.language,
                            survey_question=self.var_lab,
                            chunk_responses=task['chunk_text'],
                            chunk_size=task['chunk_size']
                        )
                        response_model = GenericSpecifierGroup2Response
                        # Capture first context specifier 2 prompt
                        if self.prompt_printer and not self._captured_context_specifier2:
                            self.prompt_printer.capture_prompt(
                                step_name="idea_extraction_context_specifiers",
                                utility_name="IdeaExtractor",
                                prompt_content=prompt,
                                prompt_type="context_specifier_group2",
                                metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
                            )
                            self._captured_context_specifier2 = True
                    else:  # group == 3: Taxonomy
                        # Group 3: Taxonomy axis scoring (context-aware)
                        # Extract context specifiers from task (passed from phase 1)
                        ctx = task.get('context_specifiers', {})
                        prompt = CODING_DIMENSION_SCORING_PROMPT.format(
                            language=self.language,
                            survey_question=self.var_lab,
                            chunk_responses=task['chunk_text'],
                            chunk_size=task['chunk_size'],
                            perspective=ctx.get('perspective', 'general'),
                            intent=ctx.get('intent', 'describe'),
                            domain=ctx.get('domain', ''),
                            entity=ctx.get('entity', ''),
                            topic=ctx.get('topic', '')
                        )
                        response_model = CodingDimensionChunkResponse
                        # Capture first taxonomy chunk scoring prompt
                        if self.prompt_printer and not self._captured_taxonomy_chunk:
                            self.prompt_printer.capture_prompt(
                                step_name="idea_extraction_taxonomy_chunk",
                                utility_name="IdeaExtractor",
                                prompt_content=prompt,
                                prompt_type="taxonomy_chunk_scoring",
                                metadata={
                                    "model": self.model,
                                    "survey_question": self.var_lab,
                                    "language": self.language,
                                    "perspective": ctx.get('perspective', 'general'),
                                    "intent": ctx.get('intent', 'describe')
                                }
                            )
                            self._captured_taxonomy_chunk = True

                    response = await llm_create_async(
                        client=self.client,
                        model=self.model,
                        response_model=response_model,
                        prompt=prompt,
                        temperature=0.0
                    )

                    await self.tpm_bucket.reconcile(0)

                    results.append({
                        'task_id': task['task_id'],
                        'group': task['group'],
                        'chunk_idx': task['chunk_idx'],
                        'response': response
                    })

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Generic specifier task {task['task_id']} failed: {e}", exc_info=True)
                self.verbose_reporter.stat_line(f"Generic specifier task {task['task_id']} failed: {e}")
            finally:
                queue.task_done()

    async def _extract_taxonomy_aware_subject(
        self,
        survey_question: str,
        taxonomy_axis: str,
    ) -> SubjectExtractionResponse:
        """Extract canonical subject with axis-aware template generation.

        This method generates a phrasing template that is shaped by the selected
        taxonomy axis, ensuring ideas are normalized along a consistent dimension.

        Args:
            survey_question: The original survey question
            taxonomy_axis: Primary taxonomy axis (WHAT, WHY, HOW, etc.)

        Returns:
            SubjectExtractionResponse with axis-aware canonical_phrasing
        """
        # Cache key includes taxonomy axis
        cache_key = f"{survey_question}_{taxonomy_axis}"

        # Use async lock to prevent race condition where multiple tasks
        # check cache simultaneously and all bypass it
        async with self._subject_lock:
            # Double-check cache inside lock
            if cache_key in self._subject_cache:
                return self._subject_cache[cache_key]

            try:
                # Get lookup data for this axis (template pattern, slot guidance, completion spec, verb frames)
                lookup_data = _format_lookup_for_axis(taxonomy_axis, language=self.language)

                prompt = TAXONOMY_AWARE_SUBJECT_PROMPT.format(
                    language=self.language,
                    survey_question=survey_question,
                    # Context specifiers
                    domain=self.generic_specifiers.get('domain', 'general'),
                    topic=self.generic_specifiers.get('topic', 'general'),
                    entity=self.generic_specifiers.get('entity', 'unknown'),
                    perspective=self.generic_specifiers.get('perspective', 'general'),
                    intent=self.generic_specifiers.get('intent', 'describe'),
                    # Taxonomy guidance (from lookup)
                    taxonomy_axis=taxonomy_axis,
                    axis_anchor=lookup_data["axis_anchor"],
                    axis_dimension_description=lookup_data["axis_dimension_description"],
                    axis_dimension_allowed_concepts=lookup_data["axis_included_concepts"],
                    # Axis pattern, slots, and instruction (from lookup)
                    axis_pattern=lookup_data["axis_pattern"],
                    axis_slots=lookup_data["axis_slots"],
                    dimension_marker=lookup_data["dimension_marker"],
                )

                # Create axis-specific response model with tailored Field descriptions
                axis_data = TEMPLATE_LOOKUP["axes"].get(taxonomy_axis, TEMPLATE_LOOKUP["axes"]["WHAT"])
                schema_data = _resolve_schema_data(axis_data)
                dim_marker = lookup_data["dimension_marker"]
                AxisSubjectModel = create_subject_extraction_model(
                    taxonomy_axis, axis_data, schema_data,
                    slot_type_map=lookup_data.get("slot_type_map"),
                    dimension_marker=dim_marker,
                )
                AxisSubjectModel.set_dimension_marker(dim_marker)

                response = await llm_create_async(
                    client=self.client,
                    model=self.model,
                    response_model=AxisSubjectModel,
                    prompt=prompt,
                    temperature=0.0
                )

                # Capture first taxonomy subject prompt
                if self.prompt_printer and not self._captured_taxonomy_subject:
                    self.prompt_printer.capture_prompt(
                        step_name="idea_extraction_taxonomy_subject",
                        utility_name="IdeaExtractor",
                        prompt_content=prompt,
                        prompt_type="taxonomy_aware_subject_extraction",
                        metadata={
                            "model": self.model,
                            "survey_question": survey_question,
                            "taxonomy_axis": taxonomy_axis,
                            "language": self.language
                        }
                    )
                    self._captured_taxonomy_subject = True

                self._subject_cache[cache_key] = response
                return response

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Taxonomy-aware subject extraction failed: {e}. Using fallback.", exc_info=True)
                # Generate axis-appropriate fallback template with axis-specific marker
                fb_keys = list(TEMPLATE_LOOKUP["axes"].get(taxonomy_axis, {}).get("slot_guidance", {}).keys())
                fb_marker = fb_keys[1] if len(fb_keys) > 1 else TEMPLATE_LOOKUP["marker"]
                axis_templates = {
                    "WHAT": f"the subject has {fb_marker}",
                    "WHY": f"the subject should achieve {fb_marker}",
                    "HOW": f"the subject should {fb_marker}",
                    "WHO": f"the stakeholder needs {fb_marker}",
                    "SENTIMENT": f"the subject is {fb_marker}",
                    "WHEN": f"the subject should {fb_marker}",
                    "WHERE": f"the subject at {fb_marker}",
                }
                fallback_template = axis_templates.get(taxonomy_axis, f"the subject {fb_marker}")

                SubjectExtractionResponse.set_dimension_marker(fb_marker)
                fallback = SubjectExtractionResponse(
                    canonical_term="the subject",
                    taxonomy_axis=taxonomy_axis,
                    canonical_phrasing=fallback_template,
                    taxonomy_actionable_type=taxonomy_axis.lower()
                )
                self._subject_cache[cache_key] = fallback
                return fallback

    def _build_taxonomy_enriched_prompt(
        self,
        respondent_id: str,
        response: str,
        subject: str,
        phrasing_template: str,
        taxonomy_actionable_type: str = ""
    ) -> str:
        """Build taxonomy-enriched prompt for idea extraction.

        V3: Restored template prefix for normalized idea phrasing to improve clustering.
        Ideas are expressed using the canonical phrasing template.
        Injects axis-specific instructions from template_lookup.py for
        node/category/taxonomy_phrase guidance.

        Args:
            respondent_id: Respondent identifier
            response: The response text to extract ideas from
            subject: The canonical subject/entity from survey question
            phrasing_template: Template with [ACTIONABLE_TAXONOMY_DIMENSION] placeholder
            taxonomy_actionable_type: Chosen narrow dimension type for taxonomy

        Returns:
            Formatted prompt string
        """
        # Get axis-specific data from template_lookup (includes prompt_rules)
        lookup_data = _format_lookup_for_axis(self.taxonomy_axis or "WHAT", language=self.language)

        return TAXONOMY_ENRICHED_EXTRACTION_PROMPT.format(
            var_lab=self.var_lab,
            taxonomy_axis=self.taxonomy_axis,
            taxonomy_actionable_type=taxonomy_actionable_type or "concepts",
            domain=self.generic_specifiers.get('domain', 'general'),
            topic=self.generic_specifiers.get('topic', 'feedback'),
            entity=self.generic_specifiers.get('entity', 'unknown'),
            perspective=self.generic_specifiers.get('perspective', 'general'),
            intent=self.generic_specifiers.get('intent', 'describe'),
            language=self.language,
            respondent_id=respondent_id,
            response=response,
            canonical_phrasing=phrasing_template,
            # Axis-specific placeholders from template_lookup
            axis_anchor=lookup_data["axis_anchor"],
            axis_dimension_description=lookup_data["axis_dimension_description"],
            axis_allowed_concepts=lookup_data["axis_allowed_concepts"],
            axis_excluded_concepts=lookup_data["axis_excluded_concepts"],
            axis_disambiguation_rules=lookup_data["axis_disambiguation_rules"],
            marker=lookup_data["marker"],
            axis_slot_pattern=lookup_data["axis_pattern"],
            axis_slot_guidance=lookup_data["axis_slot_guidance"],
            axis_node_instruction=lookup_data["axis_node_instruction"],
            axis_category_instruction=lookup_data["axis_category_instruction"],
            axis_taxonomy_phrase_instruction=lookup_data["axis_taxonomy_phrase_instruction"],
            axis_focus_rules=lookup_data["axis_focus_rules"],
            axis_instance_instruction=lookup_data["axis_instance_instruction"],
            axis_root_instruction=lookup_data["axis_root_instruction"],
            noun_phrase_descriptor=lookup_data["noun_phrase_descriptor"],
            slot_guidance_second=lookup_data["slot_guidance_second"],
            canonical_term=subject,
            instruction=lookup_data["axis_instruction"],
            # Dimension taxonomy placeholders
            semantic_category_table=lookup_data["semantic_category_table"],
            priority_rules=lookup_data["priority_rules"],
        )

    def estimate_tokens(self, prompt: str) -> int:
        """V3: Estimate total tokens using optimal adaptive strategy.

        V3 Improvements:
        - Applies learned tiktoken→API offset upfront
        - Reduced safety margins (offset handles the gap)
        - Faster convergence with smaller margins when learned
        """
        # Count tokens with tiktoken
        tiktoken_count = len(self.encoding.encode(prompt))

        # V3: Apply learned offset (accounts for system overhead)
        offset = self.tiktoken_offset_learner.get_offset()
        actual_input_tokens = tiktoken_count + offset

        # V3: Reduced safety margins (offset already accounts for gap)
        num_samples = len(self.estimation_errors)
        if num_samples < 5:
            safety_margin = 1.15  # V3: Reduced from 1.30 (offset handles gap)
        elif num_samples < 15:
            safety_margin = 1.10  # V3: Reduced from 1.20
        else:
            safety_margin = 1.05  # V3: Reduced from 1.15 (tight when learned)

        # Input estimation: use history average if available, blend with current
        if len(self.input_token_history) >= 5:
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            # Weighted blend: 70% history, 30% current for stability
            estimated_input = int(0.7 * avg_input + 0.3 * actual_input_tokens)
        else:
            # Early phase: use current with safety margin
            estimated_input = int(actual_input_tokens * safety_margin)

        # Always update input history (larger window now handles this better)
        self.input_token_history.append(actual_input_tokens)

        # Output estimation: use learned ratio if available
        if len(self.output_ratio_history) >= 5:
            # Use learned output/input ratio
            learned_ratio = sum(self.output_ratio_history) / len(self.output_ratio_history)
            estimated_output = int(estimated_input * learned_ratio * safety_margin)
        elif len(self.output_token_history) >= 3:
            # Use output history average
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output * safety_margin)
        else:
            # Fallback to default ratio with safety margin
            estimated_output = int(estimated_input * DEFAULT_OUTPUT_RATIO * safety_margin)

        # Cap output to max_tokens
        estimated_output = min(self.config.max_tokens, estimated_output)

        return estimated_input + estimated_output

    def get_token_estimation_stats(self) -> dict:
        """Get token estimation accuracy statistics including learned ratio."""
        if not self.estimation_errors:
            return {"status": "collecting_data", "samples": 0}

        avg_error = sum(self.estimation_errors) / len(self.estimation_errors)
        avg_input = sum(self.input_token_history) / len(self.input_token_history) if self.input_token_history else 0
        avg_output = sum(self.output_token_history) / len(self.output_token_history) if self.output_token_history else 0
        avg_actual_total = sum(self.actual_total_tokens) / len(self.actual_total_tokens) if self.actual_total_tokens else 0

        # Calculate learned output ratio
        learned_ratio = (sum(self.output_ratio_history) / len(self.output_ratio_history)) if self.output_ratio_history else DEFAULT_OUTPUT_RATIO

        return {
            "status": "learning",
            "samples": len(self.estimation_errors),
            "avg_estimation_error": avg_error,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "avg_actual_total_tokens": avg_actual_total,
            "initial_avg_tokens": self.bootstrap_avg_tokens if self.bootstrap_avg_tokens is not None else self.avg_tokens,
            "current_avg_tokens": self.avg_tokens,
            "adjustments_made": self.v3_stats['adjustments_made'],
            "threshold_adjustments": self.v3_stats['threshold_adjustments'],
            "pid_adjustments": self.v3_stats['pid_adjustments'],
            "input_samples": len(self.input_token_history),
            "output_samples": len(self.output_token_history),
            "actual_samples": len(self.actual_total_tokens),
            "learned_output_ratio": learned_ratio,
            "ratio_samples": len(self.output_ratio_history)
        }

    def get_token_bucket_status(self) -> dict:
        """Get current token bucket status"""
        available_pct = (self.tpm_bucket.available / self.tpm_bucket.tpm) * 100

        if len(self.actual_total_tokens) >= 10:
            recent_avg = sum(list(self.actual_total_tokens)[-10:]) / 10
            consumption_rate_per_sec = recent_avg / 2.0
            real_utilization_pct = (consumption_rate_per_sec / (self.tpm_bucket.tpm / 60)) * 100
        else:
            real_utilization_pct = 100 - available_pct
            consumption_rate_per_sec = 0

        return {
            "available_tokens": int(self.tpm_bucket.available),
            "capacity": self.tpm_bucket.tpm,
            "utilization_pct": real_utilization_pct,
            "low_tokens": available_pct < 10,
            "consumption_rate": consumption_rate_per_sec
        }

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            InstructorRetryException,
            asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def probe_call_no_structured(self, task_dict):
        """Probe call without structured output for bootstrap measurement.

        Uses current context values (should be called AFTER context extraction).
        This ensures bootstrap measurement uses the same prompt structure as actual processing.
        """
        # Placeholder values for probe calls (template not yet extracted)
        placeholder_subject = "the subject"
        placeholder_phrasing_template = "the subject is [ACTIONABLE_TAXONOMY_DIMENSION]"

        prompt = self._build_taxonomy_enriched_prompt(
            task_dict['respondent_id'],
            task_dict['response'],
            placeholder_subject,
            placeholder_phrasing_template
        )

        resp = await asyncio.wait_for(
            llm_create_async(
                client=self.client,
                model=self.model,
                prompt=prompt,
                response_model=ProbeResponse,
                temperature=self.config.temperature,
                track_usage=False,
            ),
            timeout=BOOTSTRAP_TIMEOUT_SECONDS
        )

        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        input_tokens = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
        output_tokens = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
        return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            InstructorRetryException,
            asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def process_task(self, task: Dict) -> models.IdeasExtractedModel:
        """Process a single idea extraction task"""
        task_start = time.perf_counter()

        try:
            # V3: Use taxonomy-aware subject extraction for template prefix
            subject_response = await self._extract_taxonomy_aware_subject(
                self.var_lab,
                self.taxonomy_axis or "WHAT",
            )
            subject = subject_response.canonical_term
            phrasing_template = subject_response.canonical_phrasing
            taxonomy_actionable_type = subject_response.taxonomy_actionable_type or "concepts"

            # Extract template prefix for consistent use (everything before the dimension marker)
            current_axis = self.taxonomy_axis or "WHAT"
            dim_lookup = _format_lookup_for_axis(current_axis, language=self.language)
            dim_marker = dim_lookup["dimension_marker"]
            template_prefix = phrasing_template.split(dim_marker)[0].strip() if dim_marker in phrasing_template else phrasing_template
            if self.template_prefix is None:
                self.template_prefix = template_prefix

            # Store actionable type for consistent use across all extractions
            if self.taxonomy_actionable_type is None:
                self.taxonomy_actionable_type = taxonomy_actionable_type

            # V3: Build prompt with subject and phrasing template
            prompt = self._build_taxonomy_enriched_prompt(
                task['respondent_id'],
                task['response'],
                subject,
                phrasing_template,
                self.taxonomy_actionable_type
            )

            if self.prompt_printer and not self._captured_prompt:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="idea_extraction_v3",
                    metadata={
                        "model": self.model,
                        "var_lab": self.var_lab,
                        "language": self.language,
                        "respondent_id": task['respondent_id'],
                        "taxonomy_axis": self.taxonomy_axis,
                        "template_prefix": template_prefix,
                        "taxonomy_axis_description": self.taxonomy_axis_description,
                        "taxonomy_actionable_type": self.taxonomy_actionable_type
                    }
                )
                self._captured_prompt = True

            est_tokens = self.estimate_tokens(prompt)

            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")

            self.stats['tasks_processed'] += 1

            timeout = self.latency_tracker.get_timeout(est_tokens)

            # Create axis-specific response model with tailored Field descriptions
            current_axis = self.taxonomy_axis or "WHAT"
            axis_data = TEMPLATE_LOOKUP["axes"].get(current_axis, TEMPLATE_LOOKUP["axes"]["WHAT"])
            schema_data = _resolve_schema_data(axis_data)
            extraction_lookup = _format_lookup_for_axis(current_axis, language=self.language)
            AxisExtractionModel = create_taxonomy_enriched_model(
                current_axis, axis_data, schema_data,
                slot_type_map=extraction_lookup.get("slot_type_map")
            )

            # Inject runtime context for validators
            AxisExtractionModel.set_template_prefix(template_prefix)
            AxisExtractionModel.set_axis_context(
                current_axis, axis_data,
                dimension_marker=extraction_lookup.get("dimension_marker", "")
            )

            async with self.semaphore:
                await self.tpm_bucket.wait_and_acquire(est_tokens)
                async with self.rate_limiter:
                    response = await asyncio.wait_for(
                        llm_create_async(
                            client=self.client,
                            model=self.model,
                            response_model=List[AxisExtractionModel],
                            prompt=prompt,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            max_retries=3
                        ),
                        timeout=timeout
                    )

                    latency = time.perf_counter() - task_start
                    self.latency_tracker.add(latency)

                    usage = getattr(response, '_raw_response', None)
                    if usage:
                        usage = getattr(usage, 'usage', None)
                    if not usage:
                        usage = getattr(response, 'usage', None)

                    if usage:
                        actual_input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
                        actual_output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (actual_input_tokens + actual_output_tokens)

                        # Always update output history (input updated in estimate_tokens)
                        self.output_token_history.append(actual_output_tokens)

                        # Track output/input ratio for learning
                        if actual_input_tokens > 0:
                            ratio = actual_output_tokens / actual_input_tokens
                            self.output_ratio_history.append(ratio)

                        self.actual_total_tokens.append(actual_total_tokens)

                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)

                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

                        # === V3: Additional tracking ===
                        # Record to TPM tracker (real-time sliding window)
                        await self.tpm_tracker.record(actual_total_tokens)

                        # Learn tiktoken→API offset (for input tokens only)
                        tiktoken_input = len(self.encoding.encode(prompt))
                        self.tiktoken_offset_learner.record(tiktoken_input, actual_input_tokens)

                    ideas = []
                    for i, idea_response in enumerate(response):
                        normalized = self._normalize_idea_text(idea_response.idea) if idea_response.idea else ""
                        if normalized and normalized not in ["", "NA", "N/A"]:
                            # Extract taxonomy fields (flat)
                            taxonomy_resp = getattr(idea_response, 'taxonomy', None)

                            # Clean idea text
                            idea_text = self._format_idea_text(normalized)
                            response_idea_id = getattr(idea_response, 'idea_id', None) or str(i+1)
                            ideas.append(models.IdeasExtractedSubmodel(
                                idea_id=f"{task['respondent_id']}_{response_idea_id}",
                                idea=idea_text,
                                instance=taxonomy_resp.instance if taxonomy_resp else "",
                                node=taxonomy_resp.node if taxonomy_resp else "",
                                semantic_category=taxonomy_resp.semantic_category if taxonomy_resp else "",
                                category_label=taxonomy_resp.category_label if taxonomy_resp else "",
                                root=taxonomy_resp.root if taxonomy_resp else "",
                            ))

                    if ideas:
                        self.stats['tasks_successful'] += 1
                        return models.IdeasExtractedModel(
                            respondent_id=task['respondent_id'],
                            response=task['response'],
                            quality_filter=task.get('quality_filter', True),
                            quality_filter_code=task.get('quality_filter_code', 0),
                            response_ideas=ideas,
                            idea_count=len(ideas),
                            template_prefix=self.template_prefix or ""  # V3: Use extracted template prefix
                        )
                    else:
                        # Empty ideas: retry up to 2 more times before falling back
                        logger.warning(f"Task {task['respondent_id']}: LLM returned empty ideas, retrying...")
                        for empty_retry in range(2):
                            retry_response = await asyncio.wait_for(
                                llm_create_async(
                                    client=self.client,
                                    model=self.model,
                                    response_model=List[AxisExtractionModel],
                                    prompt=prompt,
                                    temperature=self.config.temperature,
                                    max_tokens=self.config.max_tokens,
                                    max_retries=3
                                ),
                                timeout=timeout
                            )
                            retry_ideas = []
                            for i, idea_response in enumerate(retry_response):
                                normalized = self._normalize_idea_text(idea_response.idea) if idea_response.idea else ""
                                if normalized and normalized not in ["", "NA", "N/A"]:
                                    taxonomy_resp = getattr(idea_response, 'taxonomy', None)
                                    idea_text = self._format_idea_text(normalized)
                                    response_idea_id = getattr(idea_response, 'idea_id', None) or str(i+1)
                                    retry_ideas.append(models.IdeasExtractedSubmodel(
                                        idea_id=f"{task['respondent_id']}_{response_idea_id}",
                                        idea=idea_text,
                                        instance=taxonomy_resp.instance if taxonomy_resp else "",
                                        node=taxonomy_resp.node if taxonomy_resp else "",
                                        semantic_category=taxonomy_resp.semantic_category if taxonomy_resp else "",
                                        category_label=taxonomy_resp.category_label if taxonomy_resp else "",
                                        root=taxonomy_resp.root if taxonomy_resp else "",
                                    ))
                            if retry_ideas:
                                logger.info(f"Task {task['respondent_id']}: empty-ideas retry {empty_retry+1} succeeded ({len(retry_ideas)} ideas)")
                                self.stats['tasks_successful'] += 1
                                return models.IdeasExtractedModel(
                                    respondent_id=task['respondent_id'],
                                    response=task['response'],
                                    quality_filter=task.get('quality_filter', True),
                                    quality_filter_code=task.get('quality_filter_code', 0),
                                    response_ideas=retry_ideas,
                                    idea_count=len(retry_ideas),
                                    template_prefix=self.template_prefix or ""
                                )
                        # All retries exhausted — log and fall back
                        self.stats['tasks_failed'] += 1
                        self.failure_log.append({
                            'respondent_id': task['respondent_id'],
                            'reason': 'empty_ideas',
                            'error_type': None,
                            'response_preview': task['response'][:80]
                        })
                        logger.error(f"Task {task['respondent_id']}: empty ideas after 2 retries, creating PROCESSING_ERROR fallback")
                        return self.create_fallback_response(task, reason="empty_ideas")

        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['respondent_id']} timed out (retrying)")
            raise

        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['respondent_id']} hit rate limit")
            raise

        except InstructorRetryException as e:
            # Concise output for 429 errors wrapped in InstructorRetryException
            error_str = str(e)
            if "429" in error_str or "RateLimitReached" in error_str:
                if "token rate limit" in error_str.lower():
                    limit_type = "TPM"
                elif "call rate limit" in error_str.lower():
                    limit_type = "RPM"
                else:
                    limit_type = "rate"
                print(f"⚠️ 429 {limit_type} limit hit (task {task['respondent_id']})")
            else:
                logger.error(f"Task {task['respondent_id']} failed: {type(e).__name__}")
            raise

        except Exception as e:
            logger.error(f"Task {task['respondent_id']} failed: {type(e).__name__}: {e}")
            raise

    def create_fallback_response(self, task: Dict, reason: str = "unknown") -> models.IdeasExtractedModel:
        """Create fallback response for failed tasks"""
        return models.IdeasExtractedModel(
            respondent_id=task['respondent_id'],
            response=task['response'],
            quality_filter=task.get('quality_filter', True),
            quality_filter_code=task.get('quality_filter_code', 0),
            response_ideas=[
                models.IdeasExtractedSubmodel(
                    idea_id=f"{task['respondent_id']}_1",
                    idea=f"PROCESSING_ERROR: {reason}"
                )
            ],
            idea_count=1,
            template_prefix=self.template_prefix or ""  # V3: Use extracted template prefix
        )

    def get_failure_report(self, total_responses: int = None) -> str:
        """Return a formatted report of all PROCESSING_ERROR failures."""
        total = total_responses or self.stats.get('tasks_processed', 0)
        n_failures = len(self.failure_log)

        if n_failures == 0:
            return f"PROCESSING ERRORS: 0 of {total} responses (0%)"

        lines = [f"PROCESSING ERRORS: {n_failures} of {total} responses ({n_failures/max(total,1)*100:.1f}%)"]

        # Group by reason
        from collections import Counter
        reason_counts = Counter()
        for f in self.failure_log:
            key = f['error_type'] if f['reason'] == 'exception' else f['reason']
            reason_counts[key] += 1

        lines.append(f"  Breakdown: {', '.join(f'{count}x {reason}' for reason, count in reason_counts.most_common())}")
        lines.append("")

        for f in self.failure_log:
            reason_str = f['error_type'] if f['reason'] == 'exception' else f['reason']
            preview = f['response_preview']
            lines.append(f"  Respondent {f['respondent_id']}: {reason_str} | \"{preview}...\"")

        return "\n".join(lines)

    def _normalize_idea_text(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)
        text = text.strip()
        text = ' '.join(text.split())

        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            text = text.replace(char, '')

        return text

    def _format_idea_text(self, normalized_text: str) -> str:
        """Return clean idea text.

        Metadata (taxonomy_phrase, sentiment, sense) are stored as separate fields
        on the IdeasExtractedSubmodel.

        Args:
            normalized_text: The normalized idea text

        Returns:
            Clean idea text
        """
        return normalized_text

    def build_extraction_metadata(self, filename: str = "", var_name: str = "") -> 'models.ExtractionMetadata':
        """Build ExtractionMetadata from extracted context specifiers and taxonomy info.

        This creates a single metadata object that captures all extraction-level
        information that applies to the entire dataset (not per-idea).

        Args:
            filename: The source data filename
            var_name: The variable name being extracted

        Returns:
            ExtractionMetadata instance with all fields populated
        """
        from datetime import datetime

        return models.ExtractionMetadata(
            # File/variable info
            filename=filename,
            var_name=var_name,
            var_lab=self.var_lab,

            # Template (V3: restored for normalized clustering)
            template_prefix=self.template_prefix or "",

            # Context specifiers (6 fields)
            lang=self.generic_specifiers.get('lang', ''),
            domain=self.generic_specifiers.get('domain', ''),
            topic=self.generic_specifiers.get('topic', ''),
            perspective=self.generic_specifiers.get('perspective', ''),
            entity=self.generic_specifiers.get('entity', ''),
            intent=self.generic_specifiers.get('intent', ''),

            # Taxonomy axis info
            taxonomy_primary_axis=self.taxonomy_axis or '',
            taxonomy_secondary_axis=None,
            taxonomy_rationale=self.taxonomy_rationale or '',
            taxonomy_axis_description=self.taxonomy_axis_description or '',
            taxonomy_sample_phrases=self.taxonomy_sample_phrases or [],
            taxonomy_actionable_type=self.taxonomy_actionable_type or '',

            # Timestamp
            extraction_timestamp=datetime.now().isoformat()
        )

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from openai import AsyncOpenAI
        from config import API_PROVIDER, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

        if API_PROVIDER == "azure":
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/",
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self.model

        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )

        return extract_rate_limits_from_response(response)

    def _initialize_rate_limiters(self, avg_latency_s: float, avg_tokens: int, limits, num_tasks: int) -> int:

        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_concurrency = compute_optimal_concurrency(
            api_limits, avg_latency_s, avg_tokens,
            processing_config=self.processing_config,
            cap=self.processing_config.concurrency_cap_permissive,
            min_conc=self.processing_config.concurrency_min_permissive
        )
        max_concurrency = self.processing_config.concurrency_cap_default

        # Don't force high minimum when RPM-limited (e.g., Little's law says 1-5)
        # Use adaptive minimum: 3x calculated optimal (for burst handling), floor of 5, cap at config default
        adaptive_min = min(
            self.processing_config.concurrency_min_default,
            max(little_law_concurrency * 3, 5)
        )
        optimal = min(max_concurrency, max(little_law_concurrency, adaptive_min))

        arrival_rate = min(
            limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
            limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60)

        self.rate_limiter = AsyncLimiter(1, time_period=1.0/arrival_rate)

        self.semaphore = asyncio.Semaphore(min(num_tasks, optimal))
        self.optimal_concurrency = min(num_tasks, optimal)

        return optimal

    def _initialize_conservative_rate_limiters(self, limits: 'RateLimits', num_tasks: int = 20) -> None:
        """Initialize conservative rate limiters for context extraction phase.

        Uses very conservative settings since we don't have accurate token estimates yet.
        This is used during the context extraction phase before bootstrap measurement.
        """
        # Conservative: assume high token usage, low concurrency
        conservative_tokens = DEFAULT_AVG_TOKENS * 1.5  # 2250 tokens
        conservative_latency = 3.0  # seconds

        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        optimal = compute_optimal_concurrency(
            api_limits, conservative_latency, conservative_tokens,
            processing_config=self.processing_config,
            cap=10,  # Very conservative cap for setup phase
            min_conc=2
        )

        # Conservative arrival rate (50% of normal headroom)
        arrival_rate = min(
            limits.requests_per_minute * 0.5 / 60,
            limits.tokens_per_minute * 0.5 / conservative_tokens / 60
        )

        self.rate_limiter = AsyncLimiter(1, time_period=1.0/max(arrival_rate, 0.1))

        self.semaphore = asyncio.Semaphore(min(num_tasks, optimal))
        self.optimal_concurrency = min(num_tasks, optimal)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Conservative setup: concurrency={optimal}, tokens={conservative_tokens}")

    async def _apply_pid_adjustment(self) -> bool:
        """V3: Apply PID-style continuous throughput adjustment based on real-time TPM utilization.

        This provides smooth, gradual adjustments that converge to optimal throughput
        without the oscillations of threshold-based step changes.

        Returns True if adjustment was applied, False otherwise.
        """
        if self.current_arrival_rate is None:
            return False

        # Get real-time TPM utilization
        current_tpm = await self.tpm_tracker.get_current_tpm()
        tpm_limit = self.rate_limits.tokens_per_minute
        utilization = current_tpm / tpm_limit if tpm_limit > 0 else 0.0

        # Track min/max utilization for stats
        self.v3_stats['max_tpm_utilization'] = max(self.v3_stats['max_tpm_utilization'], utilization * 100)
        self.v3_stats['min_tpm_utilization'] = min(self.v3_stats['min_tpm_utilization'], utilization * 100)

        # Compute PID adjustment
        adjustment = self.pid_controller.compute_adjustment(utilization)

        # Skip if adjustment is negligible (1.0 = no change)
        if abs(adjustment - 1.0) < 0.01:
            return False

        # Apply adjustment to arrival rate
        old_rate = self.current_arrival_rate
        new_rate = old_rate * adjustment

        # Clamp to reasonable bounds
        rpm_max = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        new_rate = max(0.5, min(rpm_max, new_rate))

        # Only apply if change is meaningful
        if abs(new_rate - old_rate) / old_rate < 0.02:
            return False

        # Update rate limiter
        self.rate_limiter = AsyncLimiter(1, time_period=1.0/new_rate)

        self.current_arrival_rate = new_rate
        self.v3_stats['pid_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1

        return True

    def _adjust_throughput_if_needed(self) -> bool:
        """V3: Threshold-based adjustment (fallback for large corrections).

        This is kept as a fallback for when the token estimate is significantly wrong
        and a larger step correction is needed before PID can fine-tune.

        Returns True if adjustment was made, False otherwise.
        """
        # Need enough samples to make a reliable decision
        if len(self.actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False

        actual_avg = sum(self.actual_total_tokens) / len(self.actual_total_tokens)
        bootstrap_avg = self.avg_tokens

        # Calculate ratio of actual to bootstrap
        ratio = actual_avg / bootstrap_avg if bootstrap_avg > 0 else 1.0

        # V3: Only trigger threshold adjustment for significant underestimation
        # PID handles fine-tuning; this is for coarse correction
        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False

        # Calculate new arrival rate using actual tokens
        old_tpm_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / bootstrap_avg / 60
        new_tpm_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / actual_avg / 60
        rpm_throughput = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60

        new_arrival_rate = min(rpm_throughput, new_tpm_throughput)

        # Reinstall rate limiter with adjusted rate
        self.rate_limiter = AsyncLimiter(1, time_period=1.0/new_arrival_rate)

        # V3: Track current arrival rate for PID
        self.current_arrival_rate = new_arrival_rate

        # Reinitialize token bucket with fresh state
        old_bucket_available = self.tpm_bucket.available
        self.tpm_bucket = TokenBucket(int(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom))

        # Reset PID controller (we just made a step change)
        self.pid_controller.reset()

        # Update avg_tokens for future calculations
        old_avg = self.avg_tokens
        self.avg_tokens = int(actual_avg)

        self.v3_stats['threshold_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1

        # Log the adjustment
        print(f"\n⚡ THROUGHPUT ADJUSTMENT (threshold)")
        print(f"   Actual tokens ({actual_avg:.0f}) exceeded bootstrap ({bootstrap_avg:.0f}) by {(ratio-1)*100:.0f}%")
        print(f"   Arrival rate: {old_tpm_throughput:.2f}/s → {new_arrival_rate:.2f}/s")
        print(f"   avg_tokens: {old_avg} → {self.avg_tokens}")
        print(f"   Token bucket reset (was {old_bucket_available:,.0f} available)")
        print(f"   Tiktoken offset: {self.tiktoken_offset_learner.get_offset()} (learned: {self.tiktoken_offset_learner.is_learned()})")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Threshold adjustment: {old_tpm_throughput:.2f}/s → {new_arrival_rate:.2f}/s")

        return True

    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue"""
        while True:
            task = None
            try:
                task = await queue.get()
                if task is None:
                    break

                task_index, task_data = task
                result = await self.process_task(task_data)
                results[task_index] = result

            except Exception as e:
                # Extract concise error info for rate limit errors
                error_str = str(e)
                error_type = type(e).__name__
                if "429" in error_str or "RateLimitReached" in error_str:
                    # Determine if RPM or TPM limit
                    if "token rate limit" in error_str.lower():
                        limit_type = "TPM"
                    elif "call rate limit" in error_str.lower():
                        limit_type = "RPM"
                    else:
                        limit_type = "rate"
                    error_type = f"RateLimit_{limit_type}"
                    task_id = task_data.get('respondent_id', 'unknown') if task else 'unknown'
                    print(f"⚠️ 429 {limit_type} limit hit (task {task_id})")
                else:
                    # Non-rate-limit errors: show full details
                    logger.error(f"Task failed after retries: {e}")
                self.stats['tasks_failed'] += 1
                if task is not None:
                    task_index, task_data = task
                    self.failure_log.append({
                        'respondent_id': task_data.get('respondent_id', 'unknown'),
                        'reason': 'exception',
                        'error_type': error_type,
                        'response_preview': task_data.get('response', '')[:80]
                    })
                    results[task_index] = self.create_fallback_response(task_data, reason=error_type)
            finally:
                if task is not None:
                    queue.task_done()

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[models.IdeasExtractedModel]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []

        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Fetching rate limits from API...")

        limits = await self._fetch_rate_limits_from_api()

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Warning: Using fallback rate limits (TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            limits = RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
                tokens_per_day=FALLBACK_TPM * 60 * 24
            )
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Fetched from API: TPM={limits.tokens_per_minute:,}, RPM={limits.requests_per_minute:,}")

        self.rate_limits = limits

        # === PHASE 2: Initialize CONSERVATIVE rate limiters for context extraction ===
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Initializing conservative rate limiters for context extraction...")
        self._initialize_conservative_rate_limiters(limits, num_tasks=30)

        # === PHASE 3: Extract context specifiers AND taxonomy axis ===
        self.verbose_reporter.stat_line("Extracting context specifiers and taxonomy axis...")
        self.generic_specifiers, taxonomy_result = await self._extract_generic_specifiers()

        # Store taxonomy axis info for use in idea extraction
        self.taxonomy_axis = taxonomy_result.primary_dimension
        self.taxonomy_rationale = taxonomy_result.primary_dimension_rationale
        self.taxonomy_axis_description = taxonomy_result.primary_dimension_description  # Dynamic context-specific description

        # Sample phrases are no longer extracted from axis scores (removed dimension_scores)
        self.taxonomy_sample_phrases = []

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"\nTaxonomy axis selected: {self.taxonomy_axis}")
            if self.taxonomy_axis_description:
                self.verbose_reporter.stat_line(f"Description: {self.taxonomy_axis_description}")
            if self.taxonomy_actionable_type:
                self.verbose_reporter.stat_line(f"Actionable type: {self.taxonomy_actionable_type}")

        # === PHASE 4: Recalculate avg_tokens with REAL context ===
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("\nRecalculating token estimates with real context...")
        old_avg = self.avg_tokens
        self.avg_tokens = self._calculate_avg_tokens()
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Updated avg_tokens: {old_avg} → {self.avg_tokens}")

        # === PHASE 5: V3 IMPROVED Bootstrap probing with more probes for accuracy ===
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("\nOPTIMAL rate limiting with PID control")
            self.verbose_reporter.stat_line(f"Running bootstrap measurement ({BOOTSTRAP_NUM_PROBES} probe calls for accuracy)...")

        # V3: Use more samples for better accuracy
        sample_tasks = tasks[:min(BOOTSTRAP_NUM_PROBES, len(tasks))]
        if len(sample_tasks) < BOOTSTRAP_NUM_PROBES:
            sample_tasks = (sample_tasks * BOOTSTRAP_NUM_PROBES)[:BOOTSTRAP_NUM_PROBES]

        start_time = time.time()
        task_cycle = itertools.cycle(sample_tasks)

        async def probe_with_different_tasks():
            return await self.probe_call_no_structured(next(task_cycle))

        # V3: 5 probes instead of 3 for better accuracy
        avg_latency_s, avg_tokens = await bootstrap_measure_async(probe_with_different_tasks, n_probes=BOOTSTRAP_NUM_PROBES)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Probe time: {time.time() - start_time:.3f}s")
            self.verbose_reporter.stat_line(f"Bootstrap results: {avg_latency_s:.3f}s avg latency, {avg_tokens:.0f} avg tokens")

        # Seed latency tracker with bootstrap measurements
        for i in range(BOOTSTRAP_NUM_PROBES):
            self.latency_tracker.add(avg_latency_s)

        # V3: Apply tiktoken offset to bootstrap estimate
        # The bootstrap measures actual API tokens, so we use this to calibrate
        self.avg_tokens = int(avg_tokens)
        self.bootstrap_avg_tokens = self.avg_tokens

        # === PHASE 6: Initialize PRODUCTION rate limiters with measured data ===
        optimal = self._initialize_rate_limiters(avg_latency_s, avg_tokens, limits, len(tasks))

        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_concurrency = compute_optimal_concurrency(
            api_limits, avg_latency_s, avg_tokens,
            processing_config=self.processing_config,
            cap=self.processing_config.concurrency_cap_permissive,
            min_conc=self.processing_config.concurrency_min_permissive
        )

        print("\nRATE LIMITING SETUP - Bootstrap Optimized")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- Bootstrap measured avg_tokens: {self.avg_tokens}")

        rpm_throughput = limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        tpm_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        print(f"- Expected throughput: {min(rpm_throughput, tpm_throughput):.1f}/s ({bottleneck} limited)")
        print(f"- Optimal by Little's law: {little_law_concurrency}")
        adaptive_min = min(self.processing_config.concurrency_min_default, max(little_law_concurrency * 3, 5))
        print(f"- Adaptive concurrency: {optimal} (adaptive_min={adaptive_min}, max={self.processing_config.concurrency_cap_default})")

        print(f"- Processing {len(tasks):,} tasks")

        expected_throughput = min(rpm_throughput, tpm_throughput)

        # V3: Track current arrival rate for PID adjustments
        self.current_arrival_rate = expected_throughput

        # Calculate predicted utilization stats
        predicted_rpm_usage = expected_throughput * 60  # requests per minute
        predicted_tpm_usage = expected_throughput * self.avg_tokens * 60  # tokens per minute
        rpm_utilization_pct = (predicted_rpm_usage / limits.requests_per_minute) * 100
        tpm_utilization_pct = (predicted_tpm_usage / limits.tokens_per_minute) * 100

        print(f"\nPREDICTED UTILIZATION")
        print(f"- RPM: {predicted_rpm_usage:.0f}/{limits.requests_per_minute:,} ({rpm_utilization_pct:.0f}%)")
        print(f"- TPM: {predicted_tpm_usage:,.0f}/{limits.tokens_per_minute:,} ({tpm_utilization_pct:.0f}%)")
        print(f"- Bottleneck: {bottleneck} (limiting factor for throughput)")

        max_workers = self.processing_config.max_workers if hasattr(self.processing_config, 'max_workers') else 200
        # Adaptive min workers: use 2x optimal concurrency as floor (but at least 10)
        adaptive_min_workers = max(10, optimal * 2)
        num_workers = min(max_workers, max(adaptive_min_workers, int(expected_throughput * avg_latency_s * 2.0)))

        print(f"\nWorkers launched: (concurrent subroutines): {num_workers}")
        print(f"API calls in flight (concurrency ceiling/semaphore): {self.optimal_concurrency}")

        queue = asyncio.Queue()
        results = [None] * len(tasks)

        for i, task in enumerate(tasks):
            task['result_index'] = i
            task['task_index'] = i
            await queue.put((i, task))

        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results))
            workers.append(w)

        start_time = time.time()
        last_report = start_time
        last_diagnostics = start_time
        last_adjustment = start_time  # V3: Track PID adjustment timing

        while not queue.empty():
            await asyncio.sleep(1)
            now = time.time()

            if now - last_report >= PROGRESS_REPORT_INTERVAL:
                completed = self.stats['tasks_processed']
                remaining = queue.qsize()
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0

                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%), "
                      f"Rate: {rate:.1f}/s, Queue: {remaining}")
                last_report = now

            # V3: PID adjustment every ADJUSTMENT_INTERVAL (more frequent than diagnostics)
            if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                # First try threshold adjustment (for large corrections)
                if not self._adjust_throughput_if_needed():
                    # If no threshold adjustment, try PID fine-tuning
                    await self._apply_pid_adjustment()
                last_adjustment = now

            if self.verbose_reporter.enabled and now - last_diagnostics >= DIAGNOSTIC_INTERVAL:
                bucket_status = self.get_token_bucket_status()
                token_stats = self.get_token_estimation_stats()
                offset_stats = self.tiktoken_offset_learner.get_stats()

                if bucket_status['low_tokens']:
                    self.verbose_reporter.stat_line(f"Token bucket low: {bucket_status['available_tokens']:,} tokens ({bucket_status['utilization_pct']:.1f}% utilized)")

                # Token drift and threshold context diagnostics
                if token_stats['status'] == 'learning' and token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    initial_avg = token_stats['initial_avg_tokens']
                    current_avg = token_stats['current_avg_tokens']
                    difference = actual_avg - current_avg

                    learned_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(actual_avg, 1) / 60
                    current_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(current_avg, 1) / 60

                    pct_change = (difference / current_avg * 100) if current_avg > 0 else 0
                    threshold_pct = int((THROUGHPUT_ADJUSTMENT_THRESHOLD - 1) * 100)
                    threshold_note = f"below {threshold_pct}% threshold" if abs(pct_change) <= threshold_pct else f"exceeds {threshold_pct}% threshold"

                    adj_total = token_stats['adjustments_made']
                    if adj_total > 0:
                        self.verbose_reporter.stat_line(f"Token usage: Bootstrap {initial_avg:.0f}, Adjusted {current_avg:.0f}, Actual {actual_avg:.0f} "
                                                      f"({difference:+.0f} from current, {pct_change:+.1f}%) -- {threshold_note}")
                        self.verbose_reporter.stat_line(f"Throughput: pacing at {self.current_arrival_rate:.2f}/s "
                                                      f"(T:{token_stats['threshold_adjustments']}, PID:{token_stats['pid_adjustments']})")
                    else:
                        self.verbose_reporter.stat_line(f"Token usage: Initial estimate {initial_avg:.0f}, Learned average {actual_avg:.0f} "
                                                      f"({difference:+.0f} tokens, {pct_change:+.1f}%) -- {threshold_note}")
                        self.verbose_reporter.stat_line(f"Throughput: pacing at {current_throughput:.1f}/s (bootstrap), optimal {learned_throughput:.1f}/s (learned)")

                last_diagnostics = now

        await queue.join()

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - start_time
        print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
        print(f"- Successful: {self.stats['tasks_successful']}")
        print(f"- Failed: {self.stats['tasks_failed']}")
        print(f"- Rate limits: {self.stats['rate_limits']}")
        timeouts = self.stats['timeouts']
        if timeouts > 0 and self.stats['tasks_failed'] == 0:
            print(f"- Timeouts: {timeouts} (all retried successfully)")
        else:
            print(f"- Timeouts: {timeouts}")
        print(f"- Average: {elapsed/len(tasks):.2f}s/task")
        if self.v3_stats['threshold_adjustments'] > 0:
            print(f"- Throughput adjustments: {self.v3_stats['adjustments_made']} (T:{self.v3_stats['threshold_adjustments']}, PID:{self.v3_stats['pid_adjustments']})")
            print(f"  - Bootstrap avg_tokens: {self.bootstrap_avg_tokens}, Final avg_tokens: {self.avg_tokens}")

        # Print PROCESSING_ERROR failure report
        if self.failure_log:
            print(f"\n{'='*70}")
            print(self.get_failure_report(total_responses=len(tasks)))
            print(f"{'='*70}")
        else:
            print(f"\nPROCESSING ERRORS: 0 of {len(tasks)} responses (0%)")

        # V3: Print optimal strategy stats
        offset_stats = self.tiktoken_offset_learner.get_stats()
        print(f"\nOPTIMAL STRATEGY STATS:")
        print(f"- Tiktoken offset: {offset_stats['using_offset']} tokens (learned: {offset_stats['is_learned']}, samples: {offset_stats['samples']})")
        if offset_stats['min_offset'] is not None:
            print(f"  - Offset range: {offset_stats['min_offset']} to {offset_stats['max_offset']}")
        print(f"- Total adjustments: {self.v3_stats['adjustments_made']}")
        print(f"  - Threshold adjustments: {self.v3_stats['threshold_adjustments']}")
        print(f"  - PID adjustments: {self.v3_stats['pid_adjustments']}")
        if self.current_arrival_rate:
            print(f"- Final arrival rate: {self.current_arrival_rate:.2f}/s")

        if self.verbose_reporter.enabled:
            token_stats = self.get_token_estimation_stats()
            bucket_status = self.get_token_bucket_status()

            if token_stats['status'] == 'learning':
                accuracy = max(0, 100 - (token_stats['avg_estimation_error'] / max(1, token_stats['avg_input_tokens'] + token_stats['avg_output_tokens']) * 100))
                self.verbose_reporter.stat_line(f"Token estimation accuracy: {accuracy:.1f}% (avg error: {token_stats['avg_estimation_error']:.0f} tokens)")
                self.verbose_reporter.stat_line(f"Learned averages - Input: {token_stats['avg_input_tokens']:.0f}, Output: {token_stats['avg_output_tokens']:.0f}")

                if token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    initial_avg = token_stats['initial_avg_tokens']
                    current_avg = token_stats['current_avg_tokens']
                    difference = actual_avg - initial_avg

                    optimal_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(actual_avg, 1) / 60
                    initial_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(initial_avg, 1) / 60

                    pct_change = (difference / initial_avg * 100) if initial_avg > 0 else 0
                    threshold_pct = int((THROUGHPUT_ADJUSTMENT_THRESHOLD - 1) * 100)
                    residual = actual_avg - current_avg
                    residual_pct = (residual / current_avg * 100) if current_avg > 0 else 0
                    residual_note = f"below {threshold_pct}% threshold" if abs(residual_pct) <= threshold_pct else f"exceeds {threshold_pct}% threshold"

                    self.verbose_reporter.stat_line(f"Token usage summary: Bootstrap {initial_avg:.0f} -> Actual {actual_avg:.0f} "
                                                  f"({difference:+.0f} tokens, {pct_change:+.1f}%)")
                    if token_stats['adjustments_made'] > 0:
                        self.verbose_reporter.stat_line(f"Adjustments applied: {token_stats['adjustments_made']} "
                                                      f"(T:{token_stats['threshold_adjustments']}, PID:{token_stats['pid_adjustments']}) "
                                                      f"final avg_tokens: {current_avg}, residual drift {residual_pct:+.1f}% -- {residual_note}")
                    self.verbose_reporter.stat_line(f"Throughput analysis: Bootstrap {initial_throughput:.1f}/s -> Optimal {optimal_throughput:.1f}/s with perfect estimation")

        return results

    def extract(self) -> List[models.IdeasExtractedModel]:
        """Main method to extract ideas from responses using bootstrap measurement and unified processing"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)

        if not self.responses:
            self.verbose_reporter.stat_line("No responses to process")
            return []

        tasks = []
        for response in self.responses:
            tasks.append({
                'respondent_id': response.respondent_id,
                'response': response.response,
                'quality_filter': response.quality_filter,
                'quality_filter_code': response.quality_filter_code
            })

        if nest_asyncio:
            nest_asyncio.apply()
        self._results = asyncio.run(self.process_all_tasks_async(tasks))

        self._stats.output_count = len(self._results)
        self._stats.end_timing()

        unique_ideas = set()
        multi_idea_responses = 0
        total_idea_length = 0
        idea_count = 0

        response_examples = []
        for resp in self._results:
            if resp.response_ideas and len(resp.response_ideas) > 0:
                if len(resp.response_ideas) > 1:
                    multi_idea_responses += 1

                valid_ideas = []
                for idea in resp.response_ideas:
                    if idea.idea and not idea.idea.startswith("PROCESSING_ERROR") and idea.idea not in ["NA", "NOT_PROCESSED"]:
                        unique_ideas.add(idea.idea)
                        idea_words = idea.idea.split()
                        total_idea_length += len(idea_words)
                        idea_count += 1
                        # Store full idea info including taxonomy
                        valid_ideas.append({
                            'idea': idea.idea,
                            'instance': idea.instance,
                            'node': idea.node,
                            'semantic_category': idea.semantic_category,
                            'category_label': idea.category_label,
                            'root': idea.root,
                        })

                if valid_ideas and len(response_examples) < self.config.max_code_examples:
                    response_examples.append({
                        'response': resp.response,
                        'ideas': valid_ideas
                    })

        self.verbose_reporter.stat_line(f"Total responses processed: {len(self._results)}")
        self.verbose_reporter.stat_line(f"Total ideas extracted: {idea_count}")
        self.verbose_reporter.stat_line(f"Unique ideas identified: {len(unique_ideas)}")
        if multi_idea_responses > 0:
            single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1])
            self.verbose_reporter.stat_line(f"Single idea responses: {single_idea_responses} ({single_idea_responses/len(self._results)*100:.1f}%)")
            self.verbose_reporter.stat_line(f"Multiple idea responses: {multi_idea_responses} ({multi_idea_responses/len(self._results)*100:.1f}%)")

        single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1]) if multi_idea_responses > 0 else 0
        self.stats = {
            'total_responses': len(self._results),
            'total_ideas': idea_count,
            'unique_ideas': len(unique_ideas),
            'single_idea_responses': single_idea_responses,
            'multi_idea_responses': multi_idea_responses,
            'single_idea_percentage': (single_idea_responses / len(self._results) * 100) if len(self._results) > 0 and multi_idea_responses > 0 else 0,
            'multi_idea_percentage': (multi_idea_responses / len(self._results) * 100) if len(self._results) > 0 else 0
        }

        if response_examples:
            print("\n📋 Sample extracted ideas:")
            for example in response_examples:
                print(f'  • "{example["response"]}"')
                for idea_info in example['ideas']:
                    cleaned_idea = re.sub(r"\[.*?\]", "", idea_info['idea'])
                    cleaned_idea = re.sub(r"\s+", " ", cleaned_idea).strip()
                    print(f'    → "{cleaned_idea}"')
                    # Show taxonomy if available
                    tax_parts = [idea_info[f] for f in ('instance', 'node', 'semantic_category', 'category_label', 'root') if idea_info.get(f)]
                    if tax_parts:
                        print(f'      taxonomy: {" → ".join(tax_parts)}')
                if example != response_examples[-1]:
                    print()

        self.verbose_reporter.step_complete("Idea extraction completed")

        return self._results

    def summary(self) -> Dict[str, Union[int, float]]:
        """Generate summary statistics"""
        total = len(self._results)
        processed = sum(1 for r in self._results
                       if r.response_ideas and
                       not any(idea.idea.startswith("PROCESSING_ERROR") or idea.idea == "NOT_PROCESSED"
                              for idea in r.response_ideas))
        failed = total - processed

        total_ideas = sum(r.idea_count for r in self._results)
        unique_ideas = len(set(idea.idea for r in self._results
                              for idea in r.response_ideas
                              if not idea.idea.startswith("PROCESSING_ERROR") and idea.idea not in ["NA", "NOT_PROCESSED"]))

        return {
            "total_responses": total,
            "processed_responses": processed,
            "failed_responses": failed,
            "success_rate": round((processed / total) * 100, 2) if total > 0 else 0,
            "total_ideas": total_ideas,
            "unique_ideas": unique_ideas,
            "avg_ideas_per_response": round(total_ideas / total, 2) if total > 0 else 0
        }
