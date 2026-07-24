"""
Codebook verification layer (post-P9).

Pure Python, deterministic. Runs after P9 (and the Overig sweep) against the
consolidated codebook and the step-4 taxonomy. Does NOT touch generation logic —
it only reports, so a run can be judged against a hard definition of done.

Definition of done (PASS):
  - idea coverage = 100%        (every answer lands in a code)
  - attribute coverage = 100%   (every attribute is used by ≥1 code)
  - every code has ≥1 attribute
  - Overig ≤ 10% of ideas       (the catch-all stays small)
  - no source names invalid     (provenance is real)
  - no taxonomy-level overlap    (no two same-valence codes with an identical
    source set — compared by attribute id (A#) when present, else by name)

Also reported (warnings, do NOT block PASS):
  - under-split codes: a neutral code whose ideas have BOTH a well-represented
    positive and negative pole — it should likely have been two valence codes.
  - overlap classes: benign valence split / P9-review / taxonomy-level.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Prevalence gate (must match config_codeGenerator: min_valence_share + floor(log))
MIN_VALENCE_SHARE = 0.10


def _pole_clears(count: int, total: int) -> bool:
    """A valence pole is well-represented: ≥10% share AND more than a stray few."""
    if total <= 0:
        return False
    min_count = max(2, int(math.log(total)))
    return count >= min_count and (count / total) >= MIN_VALENCE_SHARE


# =============================================================================
# SCORECARD MODELS
# =============================================================================

class CodePair(BaseModel):
    """A pair of codes that share one or more source attributes."""
    code_a: str
    code_b: str
    shared_attributes: List[str]


class AttributeOverlap(BaseModel):
    """A single attribute that feeds more than one code."""
    attribute: str
    code_names: List[str]
    valences: List[str]
    benign_valence_split: bool  # True when the codes span different valences


class UnderSplitCode(BaseModel):
    """A neutral code whose ideas have both a well-represented + and − pole."""
    code_name: str
    positive: int
    neutral: int
    negative: int


class CodebookScorecard(BaseModel):
    """Deterministic quality scorecard for a consolidated codebook."""
    n_codes: int
    n_attributes_total: int

    # Coverage
    attribute_coverage_pct: float
    idea_coverage_pct: float
    orphan_attributes: List[str] = Field(default_factory=list)
    orphan_idea_count: int = 0
    assigned_idea_count: int = 0

    # Catch-all
    overig_code_name: Optional[str] = None
    overig_idea_share_pct: float = 0.0

    # Structural
    codes_without_attributes: List[str] = Field(default_factory=list)
    unknown_source_names: List[str] = Field(default_factory=list)

    # Overlap (Mutual Exclusivity)
    overlap_attributes: List[AttributeOverlap] = Field(default_factory=list)
    taxonomy_level_pairs: List[CodePair] = Field(default_factory=list)
    p9_review_pairs: List[CodePair] = Field(default_factory=list)

    # Valence quality (warning, does not block PASS)
    under_split_codes: List[UnderSplitCode] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Hard definition of done."""
        return (
            self.attribute_coverage_pct == 100.0
            and self.idea_coverage_pct == 100.0
            and not self.codes_without_attributes
            and not self.unknown_source_names
            and self.overig_idea_share_pct <= 10.0
            and not self.taxonomy_level_pairs
        )

    def failure_reasons(self) -> List[str]:
        reasons = []
        if self.idea_coverage_pct < 100.0:
            reasons.append(f"idea coverage {self.idea_coverage_pct}% < 100%")
        if self.attribute_coverage_pct < 100.0:
            reasons.append(f"attribute coverage {self.attribute_coverage_pct}% < 100%")
        if self.codes_without_attributes:
            reasons.append(f"{len(self.codes_without_attributes)} code(s) without attributes")
        if self.unknown_source_names:
            reasons.append(f"{len(self.unknown_source_names)} invalid source name(s)")
        if self.overig_idea_share_pct > 10.0:
            reasons.append(f"Overig {self.overig_idea_share_pct}% > 10%")
        if self.taxonomy_level_pairs:
            reasons.append(f"{len(self.taxonomy_level_pairs)} taxonomy-level overlap pair(s)")
        return reasons


# =============================================================================
# TAXONOMY EXTRACTION
# =============================================================================

def _attr(obj: Any, name: str) -> Any:
    """Read a field from a Pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def collect_taxonomy_attributes(partition_results: Dict[str, Any]) -> List[str]:
    """Ground-truth attribute names: union of attribute_name across all domains."""
    names: List[str] = []
    seen = set()
    for result in partition_results.values():
        attributes = _attr(result, "attributes") or {}
        for attr_list in attributes.values():
            for attr in attr_list:
                name = _attr(attr, "attribute_name")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def collect_idea_assignments(partition_results: Dict[str, Any]) -> Dict[str, str]:
    """Map idea_id -> attribute_name across all domains."""
    assignments: Dict[str, str] = {}
    for result in partition_results.values():
        assignments.update(_attr(result, "attribute_assignments") or {})
    return assignments


def collect_attribute_valence(partition_results: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per-attribute idea valence breakdown: attr -> {positive, neutral, negative}.

    Aggregates step-4's per-idea valence (no valence → neutral)."""
    counts: Dict[str, Dict[str, int]] = {}
    for result in partition_results.values():
        assign = _attr(result, "attribute_assignments") or {}
        valence = _attr(result, "attribute_valence") or {}
        for idea_id, attr in assign.items():
            v = valence.get(idea_id)
            bucket = "positive" if v == "+" else "negative" if v == "-" else "neutral"
            counts.setdefault(attr, {"positive": 0, "neutral": 0, "negative": 0})[bucket] += 1
    return counts


# =============================================================================
# SCORECARD BUILDER
# =============================================================================

def build_scorecard(
    codes: List[Any],
    partition_results: Dict[str, Any],
    overig_code_name: Optional[str] = None,
) -> CodebookScorecard:
    """Build a deterministic scorecard for a codebook against its taxonomy.

    Args:
        codes: List of ConsolidatedCode (or dicts) with code_name, valence, source_attributes
        partition_results: step-4 partition_results (DomainResultModel objects or dicts)
        overig_code_name: name of the catch-all code, if one was added by the Overig sweep
    """
    all_attrs = collect_taxonomy_attributes(partition_results)
    all_attrs_set = set(all_attrs)
    idea_assignments = collect_idea_assignments(partition_results)
    attr_valence = collect_attribute_valence(partition_results)

    # Legitimate source names: taxonomy attributes OR attributes ideas were assigned to.
    # (the latter lets Overig absorb step-4 dangling assignments without tripping "unknown")
    legitimate = all_attrs_set | set(idea_assignments.values())

    # attr_name -> [(code_name, valence), ...]
    attr_to_codes: Dict[str, List[tuple]] = {}
    unknown: List[str] = []
    unknown_seen = set()
    codes_without_attributes: List[str] = []
    for code in codes:
        code_name = _attr(code, "code_name") or ""
        valence = _attr(code, "valence") or ""
        sources = _attr(code, "source_attributes") or []
        if not sources:
            codes_without_attributes.append(code_name)
        for src in sources:
            attr_to_codes.setdefault(src, []).append((code_name, valence))
            if src not in legitimate and src not in unknown_seen:
                unknown_seen.add(src)
                unknown.append(src)

    covered_any = set(attr_to_codes)                          # any code source (incl. Overig)
    covered_attrs = {a for a in covered_any if a in all_attrs_set}

    # --- Coverage ---
    orphans = [a for a in all_attrs if a not in covered_attrs]
    attr_coverage = (len(covered_attrs) / len(all_attrs)) if all_attrs else 1.0

    assigned = len(idea_assignments)
    covered_ideas = sum(1 for attr in idea_assignments.values() if attr in covered_any)
    idea_coverage = (covered_ideas / assigned) if assigned else 1.0

    # --- Overig share ---
    overig_share = 0.0
    if overig_code_name:
        overig_sources = set()
        for code in codes:
            if (_attr(code, "code_name") or "") == overig_code_name:
                overig_sources = {s for s in (_attr(code, "source_attributes") or [])}
                break
        overig_ideas = sum(1 for attr in idea_assignments.values() if attr in overig_sources)
        overig_share = (overig_ideas / assigned * 100) if assigned else 0.0

    # --- Attribute-level overlap ---
    overlaps: List[AttributeOverlap] = []
    for attr, pairs in attr_to_codes.items():
        if len(pairs) < 2 or attr not in all_attrs_set:
            continue
        valences = [p[1] for p in pairs]
        overlaps.append(AttributeOverlap(
            attribute=attr,
            code_names=[p[0] for p in pairs],
            valences=valences,
            benign_valence_split=len(set(valences)) > 1,
        ))

    # --- Code-pair overlap (taxonomy-level vs P9-review) ---
    # Compare by attribute id when the code carries them (ids only cover taxonomy
    # attributes, so this matches the name path's ∩-taxonomy filter); fall back
    # to name-sets for pre-id codes. Ids keep same-named attributes in different
    # domains distinct — name-sets would falsely collide them.
    code_sources = []  # (code_name, valence, set(source ids or names ∩ taxonomy))
    for code in codes:
        ids = _attr(code, "source_attribute_ids") or []
        srcs = set(ids) if ids else {
            s for s in (_attr(code, "source_attributes") or []) if s in all_attrs_set}
        code_sources.append((_attr(code, "code_name") or "", _attr(code, "valence") or "", srcs))

    taxonomy_pairs: List[CodePair] = []
    review_pairs: List[CodePair] = []
    for (name_a, val_a, src_a), (name_b, val_b, src_b) in combinations(code_sources, 2):
        shared = src_a & src_b
        if not shared or val_a != val_b:
            continue
        if src_a == src_b:
            taxonomy_pairs.append(CodePair(code_a=name_a, code_b=name_b, shared_attributes=sorted(shared)))
        else:
            review_pairs.append(CodePair(code_a=name_a, code_b=name_b, shared_attributes=sorted(shared)))

    # --- Under-split detection (over-collapse) ---
    under_split: List[UnderSplitCode] = []
    for code in codes:
        if (_attr(code, "valence") or "") != "neutral":
            continue
        if (_attr(code, "code_name") or "") == overig_code_name:
            continue
        p = n = g = 0
        for attr in (_attr(code, "source_attributes") or []):
            c = attr_valence.get(attr, {})
            p += c.get("positive", 0)
            n += c.get("neutral", 0)
            g += c.get("negative", 0)
        total = p + n + g
        if _pole_clears(p, total) and _pole_clears(g, total):
            under_split.append(UnderSplitCode(
                code_name=_attr(code, "code_name") or "", positive=p, neutral=n, negative=g))

    return CodebookScorecard(
        n_codes=len(codes),
        n_attributes_total=len(all_attrs),
        attribute_coverage_pct=round(attr_coverage * 100, 1),
        idea_coverage_pct=round(idea_coverage * 100, 1),
        orphan_attributes=orphans,
        orphan_idea_count=assigned - covered_ideas,
        assigned_idea_count=assigned,
        overig_code_name=overig_code_name,
        overig_idea_share_pct=round(overig_share, 1),
        codes_without_attributes=codes_without_attributes,
        unknown_source_names=unknown,
        overlap_attributes=overlaps,
        taxonomy_level_pairs=taxonomy_pairs,
        p9_review_pairs=review_pairs,
        under_split_codes=under_split,
    )


# =============================================================================
# FORMATTING
# =============================================================================

def format_scorecard(sc: CodebookScorecard) -> str:
    """Render the scorecard as a human-readable console block."""
    status = "PASS" if sc.passed else "FAIL"
    lines = [
        "=" * 80,
        f"CODEBOOK SCORECARD — {status}",
        "=" * 80,
        f"  Codes:               {sc.n_codes}",
        f"  Attributes (taxon.): {sc.n_attributes_total}",
        f"  Attribute coverage:  {sc.attribute_coverage_pct}%",
        f"  Idea coverage:       {sc.idea_coverage_pct}%  "
        f"({sc.orphan_idea_count}/{sc.assigned_idea_count} uncovered)",
    ]
    if sc.overig_code_name:
        lines.append(f"  Overig share:        {sc.overig_idea_share_pct}%  (cap 10%)")

    if not sc.passed:
        lines.append(f"\n  ✗ FAIL — {'; '.join(sc.failure_reasons())}")

    if sc.orphan_attributes:
        lines.append(f"\n  ⚠ ORPHAN ATTRIBUTES — covered by no code ({len(sc.orphan_attributes)}):")
        for a in sc.orphan_attributes:
            lines.append(f"      - {a}")

    if sc.codes_without_attributes:
        lines.append(f"\n  ⚠ CODES WITHOUT ATTRIBUTES ({len(sc.codes_without_attributes)}):")
        for c in sc.codes_without_attributes:
            lines.append(f"      - {c}")

    if sc.unknown_source_names:
        lines.append(f"\n  ⚠ UNKNOWN SOURCE NAMES ({len(sc.unknown_source_names)}):")
        for a in sc.unknown_source_names:
            lines.append(f"      - {a}")

    if sc.taxonomy_level_pairs:
        lines.append(f"\n  ⚠ TAXONOMY-LEVEL OVERLAP — identical source set "
                     f"({len(sc.taxonomy_level_pairs)}) — escalate to taxonomy:")
        for p in sc.taxonomy_level_pairs:
            lines.append(f"      - \"{p.code_a}\" / \"{p.code_b}\"  ←  {', '.join(p.shared_attributes)}")

    if sc.under_split_codes:
        lines.append(f"\n  ◦ UNDER-SPLIT (warning) — neutral code with both poles well-represented "
                     f"({len(sc.under_split_codes)}) — likely should be two valence codes:")
        for u in sc.under_split_codes:
            lines.append(f"      - \"{u.code_name}\"  (+{u.positive} / ○{u.neutral} / −{u.negative})")

    if sc.p9_review_pairs:
        lines.append(f"\n  ◦ P9-REVIEW OVERLAP — same-valence partial overlap "
                     f"({len(sc.p9_review_pairs)}):")
        for p in sc.p9_review_pairs:
            lines.append(f"      - \"{p.code_a}\" / \"{p.code_b}\"  ←  {', '.join(p.shared_attributes)}")

    lines.append("=" * 80)
    return "\n".join(lines)
