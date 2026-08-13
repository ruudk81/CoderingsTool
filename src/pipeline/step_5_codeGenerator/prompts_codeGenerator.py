"""
`ConsolidatedCode` — the codebook entry model.

Written by `codebook_writer.py`, cached (as dicts) in `CodingResultsCache.raw_codes`,
and reconstructed via `ConsolidatedCode(**d)` by step 6 and step 7 — those steps
import it from this module specifically, so it stays here rather than moving to
models.py or codebook_writer.py.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class ConsolidatedCode(BaseModel):
    """A codebook entry with a diagnostic test for MECE verification."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description=(
            "A short interpretive claim that reads like an analyst conclusion. "
            "Avoid vague abstract phrasing — be concrete and specific."
        )
    )
    diagnostic_test: str = Field(
        ..., description=(
            "Completes the dimension-specific diagnostic stem — "
            "must be unique per code and must not overlap with other codes."
        )
    )
    valence: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description=(
            "The code's evaluative direction. A code whose ideas span a "
            "well-represented positive AND a well-represented negative pole "
            "should not carry a single 'positive' or 'negative' label here — "
            "such a phenomenon splits into two codes, each correctly labeled "
            "for its own pole. 'neutral' is reserved for a genuinely "
            "dimensional code (no pole cleared the gate) or the Overig catch-all."
        )
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Attribute names this code is derived from (from all merged origins)"
    )
    # Stable ids (identity.py) — never part of the LLM response schema: minted at
    # cache-save (K#), or lazily at load for pre-id codebooks. source_attribute_ids
    # mirrors source_attributes as attribute ids (A#), resolved at cache-save/load.
    code_id: SkipJsonSchema[str] = ""
    source_attribute_ids: SkipJsonSchema[List[str]] = Field(default_factory=list)
