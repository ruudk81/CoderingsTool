"""
Prompt builders for the Taxonomy Classifier (P1-P9).

Pipeline order — discovery, assignment, consolidation, once per level:

  P1   Axis discovery                     build_axis_discovery_prompt
  P2   Facet discovery WITH axes          build_tagged_facet_discovery_prompt
  P3   Facet discovery WITHOUT axes       build_facet_discovery_prompt
  P4   Facet assignment                   build_facet_assignment_prompt_single
  P5   Facet consolidation (in-axis)      build_in_axis_consolidation_prompt
  P6   Attribute discovery                build_attribute_discovery_prompt
  P7   Attribute assignment               build_attribute_assignment_prompt_single
  P8   Attribute consolidation (in-facet) build_in_facet_consolidation_prompt
  P9   Valence-neutral merge              build_valence_neutral_rename_prompt

P2 and P3 are the only fork: a domain with an axis system takes P2, a domain
without one takes P3 — same dispatch, different prompt. Everything after that
is a single route. Both consolidation rounds run AFTER assignment, on real
idea counts and real response texts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, create_model, model_validator
from pydantic.json_schema import SkipJsonSchema

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §0 DIMENSION CONTEXT BLOCK — shared helper for all prompts
# =============================================================================

def _norm_text(text: Optional[str]) -> str:
    """Normalise a tag value for matching. Case- and padding-insensitive
    only, mirroring `TaxonomyClassifier._norm_text` (classifier.py) — kept
    as a standalone copy here rather than an import to avoid coupling this
    prompt-builder module to the classifier."""
    return (text or "").strip().lower()


def _extract_definition(instruction: str) -> str:
    """Extract the 'Definition: ...' sentence (up to first newline) from an instruction string."""
    marker = "Definition: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    rest = instruction[idx + len(marker):]
    newline = rest.find("\n")
    if newline != -1:
        rest = rest[:newline]
    return rest.strip()


def _extract_key_idea(instruction: str) -> str:
    """Extract the 'Key idea: ...' sentence from an instruction string."""
    marker = "Key idea: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    return instruction[idx + len(marker):].strip().rstrip(".")


def _build_exclusion_block(
    items: List[Tuple[str, str]],
    tag_name: str,
) -> str:
    """Build an XML-tagged exclusion block for domains or facets.

    Args:
        items: list of (name, definition) tuples to exclude.
        tag_name: XML tag name, e.g. 'excluded_domains' or 'excluded_facets'.
    """
    if not items:
        return ""
    lines = [f"- {name} -- {definition}" for name, definition in items]
    content = "\n".join(lines)
    return (
        f"\nYou must NOT include {'facets' if tag_name == 'excluded_domains' else 'attributes'} that belong to these excluded {'domains' if tag_name == 'excluded_domains' else 'facets'}:\n"
        f"<{tag_name}>\n{content}\n</{tag_name}>\n"
    )


def _build_exclusion_block_light(
    items: List[Tuple[str, str]],
) -> str:
    """Build a short name-only exclusion list for use in scratchpad steps."""
    if not items:
        return "(none)"
    return "\n".join(f"- {name}" for name, _ in items)


def _build_exclusion__light_block(
    items: List[Tuple[str, str]],
) -> str:
    """Build an exclusion block for domains or facets without XML tags

    Args:
        items: lightweight list of (name, definition) tuples to exclude.
    """
    if not items:
        return ""
    lines = [f"- {name} — {definition}" for name, definition in items]
    content = "\n".join(lines)
    return (
        f"{content}"
    )


# =============================================================================
# §1 AXIS DISCOVERY (P1) — per-domain axis system discovery
# =============================================================================

def build_axis_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    primary_dimension: str,
    noun_phrase: str,
    domain_label: str,
    domain_definition: str,
    domain_boundary_test: str,
    sample_observations: List[str],
) -> str:
    """Discover the axes along which observations in a domain differ (P1a)."""
    observations_block = "\n".join(f"- {obs}" for obs in sample_observations)

    return f"""You are a taxonomy methodologist working on open-ended survey coding. Your task is to identify coordinate axes within a specific domain of survey responses.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question that was asked:

<survey_question>
"{survey_question}"

answers vary in terms of: {noun_phrase}
</survey_question>

You are analyzing responses within the following domain:

<domain_name>
{domain_label}
</domain_name>

<domain_definition>
{domain_definition}
</domain_definition>

<domain_boundary_test>
{domain_boundary_test}
</domain_boundary_test>

Here is a broad sample of observations from this domain:

<observations>
{observations_block}
</observations>

Your task

<task>
You are identifying coordinate axes, not categories or segments.
An axis must represent a dimension along which observations could vary within the domain, independently of other axes.
If you cannot demonstrate such independence, do not create another axis.
If the data support only one axis, return exactly one axis. Do not decompose one axis into multiple pseudo-axes.
Before returning more than one axis, verify that observations could differ on axis A while sharing the same value on axis B, and differ on axis B while sharing the same value on axis A. If not, merge or drop the axis.
</task>


PROCESS:

Use the scratchpad to:
- Examine the observations for patterns of variation
- Identify potential axes
- Test each potential axis for independence from others
- Provide concrete examples demonstrating independence (if proposing multiple axes)
- Decide on the final number of axes

Then provide your final answer with:
- A clear description of each axis
- The dimension of variation it represents
- If multiple axes: explicit demonstration of their independence using examples from the observations

For each axis you identify, describe:
- The axis name
- What dimension of variation it captures
- The range or types of values observations can take along this axis
- If proposing multiple axes: concrete examples showing how observations vary independently on each axis

Important requirements:
- All output (axis names and descriptions) must be in {language}

Provide your output as valid JSON following the response schema provided.
"""


class DiscoveredAxis(BaseModel):
    """An axis along which observations within a domain differ."""
    axis_name: str = Field(
        ..., description="Short name for the axis"
    )
    axis_description: str = Field(
        ..., description="What independent dimension of variation this axis captures (1-2 sentences)"
    )
    value_range: str = Field(
        ..., description="The range or types of values observations can take along this axis"
    )


class AxisSystemResponse(BaseModel):
    """P1a output: the axis system discovered for a single domain."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before naming the axes: "
            "(1) examine the observations for patterns of variation, "
            "(2) identify potential axes, "
            "(3) test each potential axis for independence and orthogonality from the others, "
            "(4) give concrete examples demonstrating independence when proposing "
            "more than one axis, "
            "(5) decide on the final number of axes"
        )
    )
    independence_evidence: str = Field(
        default="", description=(
            "When more than one axis is returned: concrete examples from the observations "
            "showing that observations can differ on one axis while sharing the same value "
            "on another, in both directions. Empty when a single axis is returned."
        )
    )
    axes: List[DiscoveredAxis] = Field(
        ..., description="Axes discovered for this domain"
    )


# =============================================================================
# §2 FACET DISCOVERY WITH AXES (P2) — per-domain, chunked; axis system is fixed context
# =============================================================================

def _build_axis_system_block(axis_system: AxisSystemResponse) -> str:
    """Render a validated axis system as prompt text: one numbered block per
    axis — its name, what it captures, and the values observations can take
    along it. Each block ends with a blank line, so the axes stay visually
    separated when several are shown."""
    return "".join(
        f"Axis {i}: {axis.axis_name}\n"
        f"  What it captures: {axis.axis_description}\n"
        f"  Values along this axis: {axis.value_range}\n\n"
        for i, axis in enumerate(axis_system.axes, 1)
    )


def build_tagged_facet_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    noun_phrase: str,
    domain_label: str,
    domain_definition: str,
    axis_system: AxisSystemResponse,
    chunk_observations: List[str],
) -> str:
    """Discover facets (L3) from a chunk of observations, each proposal tagged
    to exactly one (axis, segment) of the domain's fixed axis system (P1b)."""
    axis_system_block = _build_axis_system_block(axis_system)
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(chunk_observations, 1))

    return f"""You are a qualitative research analyst specializing in open-ended survey coding. Your task is to induce the minimal set of facets needed to classify all observations within a specific domain.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question being analyzed:

<survey_question>
"{survey_question}"

answers vary in terms of: {noun_phrase}
</survey_question>

Here is the domain you are working within:

<domain>
Domain: {domain_label} — {domain_definition}
</domain>

Here is the axis system that defines how responses vary within this domain:

<axis_system>
{axis_system_block}
</axis_system>

Here are the observations you need to classify:

<observations>
{observations_block}
</observations>

Your task is to induce the least number of facets needed to classify all observations within this domain.

Requirements:

1. **Facet only along the provided axes.** Do not introduce facets based on themes outside the axis system. The axis defines the dimension of variation you must capture.

2. **Use the fewest facets possible.** Only create distinct facets when observations differ in the core way specified by the axis. Do not over-differentiate.

3. **Facets must be:**
   - Mutually exclusive (each observation fits in only one facet)
   - Atomically distinct (each facet represents one clear variation type)
   - Meaningfully differentiated (facets capture real differences along the axis)
   - Orthogonal to other domains/facets (don't overlap with distinctions that belong in other domains)

4. **Handle rare or singleton patterns appropriately.** Put rare or singleton patterns into a general/residual facet unless they represent a clearly recurring and axis-relevant distinction that appears multiple times.

5. **Context-dependent responses.** If a response contains a substantive improvement suggestion plus a statement of no further advice, classify only the part relevant to this domain. Note it as context-dependent or general unless it forms a recurring pattern.

Before providing your final facet set, use the scratchpad to:
- Identify the different types of variation the observations show along each axis
- Group observations by similarity along the axis
- Consider whether apparent differences are meaningful enough to warrant separate facets
- Determine the minimal set that captures all meaningful variation

Now provide your final facet set. For each facet, include:

- **Facet name**: A clear, concise label
- **Definition**: A precise description of what this facet captures
- **Inclusion rule**: What types of responses belong in this facet
- **Exclusion rule**: What types of responses do NOT belong (if helpful for clarity)
- **Example observation numbers**: List 3-5 observation numbers that exemplify this facet

After listing all facets, provide:

- **Rationale for minimality**: Explain why this is the minimal facet set needed and why you did not split or merge facets further

Output requirements:
- All output (facet names and descriptions) must be in {language}

Provide your output as valid JSON following the response schema provided.
"""

class FacetProposal(BaseModel):
    """One facet proposed on one of the domain's axes (P1b output)."""
    facet_name: str = Field(
        ..., description="A clear, concise label for this facet"
    )
    facet_definition: str = Field(
        ..., description="A precise description of what this facet captures"
    )
    inclusion_rule: str = Field(
        ..., description="What types of responses belong in this facet"
    )
    exclusion_rule: str = Field(
        default="", description=(
            "What types of responses do NOT belong in this facet — "
            "only when it helps clarify the boundary, otherwise empty"
        )
    )
    example_observations: List[int] = Field(
        ..., description="3-5 observation numbers that exemplify this facet"
    )


def build_tagged_facet_discovery_model(axis_names: List[str]) -> type[BaseModel]:
    """Build the P2 response model for one domain.

    The domain's axes are already known from P1, so they are fixed in the
    schema itself: `axis_name` is a Literal over exactly those names. The
    model does not name an axis, it picks one of ours — and adds as many
    facets under it as that axis needs.

    Built with `create_model` because the Literal only exists at call time —
    a static annotation over a runtime value is unevaluable for type
    checkers; this way no annotation refers to it.
    """
    axis_name_literal = Literal[tuple(axis_names)]  # type: ignore[valid-type]

    axis_facets = create_model(
        "AxisFacets",
        __doc__="The facets proposed on one axis.",
        axis_name=(axis_name_literal, Field(
            ..., description="The axis these facets sit on")),
        facets=(List[FacetProposal], Field(
            ..., description="The minimal set of facets needed on this axis")),
    )

    return create_model(
        "TaggedFacetDiscoveryResponse",
        __doc__="P2 output: facets discovered in a single chunk, grouped per axis.",
        scratchpad=(str, Field(
            ..., description=(
                "Reasoning before the final facet set: group the observations by "
                "similarity along each axis, consider whether apparent differences "
                "are meaningful enough to warrant separate facets, and determine "
                "the minimal set that captures all meaningful variation"
            ))),
        axes=(List[axis_facets], Field(
            ..., description="One entry per axis, with the facets proposed on it")),
        minimality_rationale=(str, Field(
            ..., description=(
                "Why this is the minimal facet set needed, and why facets were "
                "not split or merged further"
            ))),
    )


# =============================================================================
# §3 FACET DISCOVERY WITHOUT AXES (P3) — per-domain, chunked
# =============================================================================

def build_facet_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    partition_name: str,
    partition_definition: str,
    observations: List[str],
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
    boundary_test: str = "",
    exclusions: Optional[List[str]] = None,
) -> str:
    """Discover facets (L3) from a chunk of observations within a domain."""
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(observations, 1))

    _boundary_lines = []
    if boundary_test:
        _boundary_lines.append(f"Boundary test: {boundary_test}")
    if exclusions:
        _boundary_lines.append(
            "This domain EXCLUDES (these belong to other domains): " + "; ".join(exclusions)
        )
    domain_boundary_block = ("\n" + "\n".join(_boundary_lines)) if _boundary_lines else ""

    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        facet_definition = _extract_definition(rules.facet_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_definition = "A facet identifies the analytical lens through which the domain is being examined."
        facet_key_idea = "the analytical lens applied to the subject"
        attribute_key_idea = "the specific observable property being described"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"

    excluded_block = _build_exclusion_block(
        excluded_domains or [], "excluded_domains"
    )

    excluded_block_light = _build_exclusion__light_block(
        excluded_domains or []
    )

    return f"""You are a qualitative research analyst specializing in open-ended survey coding. Your task is to induce the minimal set of facets needed to classify all observations within a specific domain.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question being analyzed:

<survey_question>
"{survey_question}"
answers vary in terms of: {noun_phrase}
</survey_question>

Here is the domain you are working within:

<taxonomy_domain>
{partition_name} — {partition_definition}{domain_boundary_block}
</taxonomy_domain>
{excluded_block}

Here are the observations you need to classify:

<observations>
{observations_block}
</observations>

Your task is to induce the least number of facets needed to classify all observations within this domain.

Requirements:

1. **Identify the coordinate axes within this domain.** Find the dimensions along which responses vary orthogonally to each other.

2. **Facet only along the axes you identified.** Do not introduce facets based on themes outside those axes. The axis defines the dimension of variation you must capture.

3. **Use the fewest facets possible.** Only create distinct facets when observations differ in the core way specified by the axis. Do not over-differentiate.

4. **Facets must be:**
   - Mutually exclusive (each observation fits in only one facet)
   - Atomically distinct (each facet represents one clear variation type)
   - Meaningfully differentiated (facets capture real differences along the axis)
   - Orthogonal to other domains/facets (don't overlap with distinctions that belong in other domains)

5. **Handle rare or singleton patterns appropriately.** Put rare or singleton patterns into a general/residual facet unless they represent a clearly recurring and axis-relevant distinction that appears multiple times.

6. **Context-dependent responses.** If a response contains a substantive improvement suggestion plus a statement of no further advice, classify only the part relevant to this domain. Note it as context-dependent or general unless it forms a recurring pattern.

Before providing your final facet set, use the scratchpad to:
- Identify the different types of variation the observations show along each axis
- Group observations by similarity along the axis
- Consider whether apparent differences are meaningful enough to warrant separate facets
- Determine the minimal set that captures all meaningful variation

Now provide your final facet set. For each facet, include:

- **Facet name**: A clear, concise label
- **Definition**: A precise description of what this facet captures
- **Inclusion rule**: What types of responses belong in this facet
- **Exclusion rule**: What types of responses do NOT belong (if helpful for clarity)
- **Example observation numbers**: List 3-5 observation numbers that exemplify this facet

After listing all facets, provide:

- **Rationale for minimality**: Explain why this is the minimal facet set needed and why you did not split or merge facets further

Output requirements:
- All output (facet names and descriptions) must be in {language}

Provide your output as valid JSON following the response schema provided.
"""

class DiscoveredFacet(BaseModel):
    """A facet (L3) discovered from observations within a domain."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)"
    )
    facet_description: str = Field(
        ..., description="What this facet captures — the specific viewpoint or aspect (1-2 sentences)"
    )
    inclusion_rule: str = Field(
        default="", description="What types of responses belong in this facet"
    )
    exclusion_rule: str = Field(
        default="", description=(
            "What types of responses do NOT belong in this facet — only when it "
            "helps clarify the boundary, otherwise empty"
        )
    )
    example_observations: List[str] = Field(
        ..., description="3-5 representative observations from the input"
    )
    boundary_test: SkipJsonSchema[str] = Field(
        default="", description="One routing sentence for the doubtful case, phrased against a named sibling facet"
    )
    axis: SkipJsonSchema[str] = Field(
        default="", description="Provenance only, written by code: the axis this facet sits on (empty when the domain has no axis system)"
    )


class FacetDiscoveryResult(BaseModel):
    """P1 output: facets discovered in observations."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before identifying facets: "
            "(1) cluster observations by shared descriptive meaning, "
            "(2) identify candidate facets and assess coherence and distinctness, "
            "(3) verify internal coherence — one clear concept per facet, "
            "(4) verify distinctness — ontologically distinct and semantically separable, "
            "(5) verify domain boundaries — exclude facets belonging to other domains, "
            "(6) prepare final output with only dominant facets that pass all checks"
        )
    )
    facets: List[DiscoveredFacet] = Field(
        ..., description="Facets identified in the observations"
    )


# =============================================================================
# §4 FACET ASSIGNMENT (P4) — one idea per task
# =============================================================================


def _build_facet_codebook_block(
    facets: List[DiscoveredFacet],
    other_label: Optional[str] = None,
    axis_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Format discovered facets as a numbered codebook for assignment.

    When any facet carries an axis tag (P1b/P2 axis-first path), items are
    grouped under one 'Axis: {name} — {description}' header per axis, axes
    in first-seen order (description omitted when not supplied — cache-loaded
    contexts have no axis-system object to pull it from). F# numbering still
    reflects each facet's position in `facets`, unchanged from the untagged
    path, so it stays consistent with the facet_id_to_name mapping built
    alongside this same list. A facet list with no axis tags renders exactly
    as before this grouping existed."""
    def _render(i: int, facet: DiscoveredFacet) -> str:
        examples = "; ".join(facet.example_observations[:3])
        return (
            f"[F{i}] {facet.facet_name}\n"
            f"    Description: {facet.facet_description}\n"
            + (f"    Belongs here: {facet.inclusion_rule}\n" if facet.inclusion_rule else "")
            + (f"    Does not belong here: {facet.exclusion_rule}\n" if facet.exclusion_rule else "")
            + (f"    Boundary: {facet.boundary_test}\n" if facet.boundary_test else "")
            + f"    Examples: {examples}"
        )

    numbered = list(enumerate(facets, 1))
    if any(facet.axis for facet in facets):
        by_axis: Dict[str, List[Tuple[int, DiscoveredFacet]]] = {}
        for i, facet in numbered:
            by_axis.setdefault(facet.axis, []).append((i, facet))
        axis_descriptions = axis_descriptions or {}
        lines = []
        for axis_name, items in by_axis.items():
            desc = axis_descriptions.get(axis_name, "")
            header = f"Axis: {axis_name} — {desc}" if desc else f"Axis: {axis_name}"
            body = "\n\n".join(_render(i, facet) for i, facet in items)
            lines.append(f"{header}\n\n{body}")
    else:
        lines = [_render(i, facet) for i, facet in numbered]

    if other_label:
        n = len(facets) + 1
        lines.append(
            f"[F{n}] {other_label}\n"
            f"    Description: Observations that do not clearly fit any of the above facets.\n"
            f"    Examples: (none)"
        )
    return "\n\n".join(lines)


def build_facet_assignment_prompt_single(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    domain_name: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    idea_label: str,
    axis_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning a single idea to a facet (L3).

    `axis_descriptions`: {axis_name: axis_description}, available only when
    the caller still holds the in-memory AxisSystemResponse for this domain
    (during a live run, via TaxonomyClassifier.axis_systems). Cache-loaded
    contexts pass None and the menu falls back to bare 'Axis: {name}'
    headers.
    """
    facet_codebook = _build_facet_codebook_block(facets, axis_descriptions=axis_descriptions)

    return f"""You are a qualitative coding assistant. Assign the survey response idea below to the facet that best captures the type of quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<domain_context>
Domain: {domain_name} -- {domain_definition}
</domain_context>

<facets>
{facet_codebook}
</facets>

<idea>
{idea_label}
</idea>

### VALENCE (evaluation relative to facet)
- "+" Positive — The attribute is described as meeting or enhancing the facet
- "-" Negative — The attribute is described as failing to meet or detracting from the facet
- "0" Neutral — The response is descriptive, ambiguous, or does not express evaluation
- Valence is not emotional sentiment, but evaluative direction relative to the facet

Assign this idea to the single best-fitting facet. Return the facet ID (e.g. "F1", "F2"), your confidence (0.0-1.0), and the valence (+, -, or 0).

Provide your response as valid JSON following the response schema provided."""


class FacetAssignmentResult(BaseModel):
    """Single idea-to-facet assignment result."""
    assigned_facet_id: str = Field(
        ..., description=(
            "The facet ID from the [F#] prefix (e.g. 'F1', 'F3'). "
            "Return ONLY the ID, not the facet name."
        )
    )
    confidence: float = Field(
        ..., description="Confidence in the assignment (0.0 to 1.0)"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="Evaluative direction relative to the facet: + positive, - negative, 0 neutral"
    )


def build_batch_facet_assignment_model(
    facet_ids: List[str],
    idea_ids: List[str],
):
    """Runtime response model for batch facet assignment.

    Literal fields make a hallucinated facet id or idea id a schema violation
    (instructor retries) instead of a content error — the same
    construction-over-instruction pattern as the tagged P2 discovery model.
    "F_NONE" is the escape hatch: no facet fits; the caller escalates that
    idea to a single full-menu call.
    """
    facet_id_literal = Literal[tuple(facet_ids + ["F_NONE"])]  # type: ignore[valid-type]
    idea_id_literal = Literal[tuple(idea_ids)]  # type: ignore[valid-type]

    item_model = create_model(
        "BatchFacetAssignmentItem",
        idea_id=(idea_id_literal, Field(
            ..., description="The [id] tag of the idea, echoed exactly")),
        assigned_facet_id=(facet_id_literal, Field(
            ..., description=(
                "The facet ID from the [F#] prefix. Return ONLY the ID. "
                "Use F_NONE when no facet fits this idea."))),
        confidence=(float, Field(
            ..., ge=0.0, le=1.0, description="Assignment confidence (0.0-1.0)")),
        valence=(Literal["+", "-", "0"], Field(
            default="0",
            description="Evaluative direction relative to the facet: + positive, - negative, 0 neutral")),
    )
    return create_model(
        "BatchFacetAssignmentResult",
        assignments=(List[item_model], Field(
            ..., description=(
                "Exactly one assignment per idea listed in the prompt, "
                "no idea skipped, no idea added"))),
    )


def build_facet_assignment_prompt_batch(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    domain_name: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    ideas: List[Tuple[str, str]],
    axis_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Batch variant of build_facet_assignment_prompt_single: one menu, a list
    of (idea_id, idea_label) pairs, one schema-validated assignment per idea.
    """
    facet_codebook = _build_facet_codebook_block(facets, axis_descriptions=axis_descriptions)
    ideas_block = "\n".join(f"[{idea_id}] {label}" for idea_id, label in ideas)

    return f"""You are a qualitative coding assistant. Assign each survey response idea below to the facet that best captures the type of quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<domain_context>
Domain: {domain_name} -- {domain_definition}
</domain_context>

<facets>
{facet_codebook}

[F_NONE] None of the facets above fits the idea.
</facets>

<ideas>
{ideas_block}
</ideas>

### VALENCE (evaluation relative to facet)
- "+" Positive — The attribute is described as meeting or enhancing the facet
- "-" Negative — The attribute is described as failing to meet or detracting from the facet
- "0" Neutral — The response is descriptive, ambiguous, or does not express evaluation
- Valence is not emotional sentiment, but evaluative direction relative to the facet

Judge every idea independently on its own text; do not let one assignment influence the next. Return exactly one item per idea, echoing that idea's [id]. Do not skip ideas; do not add ideas. If no facet fits an idea, use "F_NONE" for that idea.

Provide your output as valid JSON following the response schema provided."""


# =============================================================================
# §5 FACET CONSOLIDATION (P5) — in-axis, post-assignment
# =============================================================================

def build_in_axis_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    noun_phrase: str,
    domain_name: str,
    domain_definition: str,
    axis_name: str,
    axis_description: str,
    facets_block: str,
    neighbour_axes_block: str = "",
) -> str:
    """Consolidate the facets on ONE axis, after every idea has been assigned.

    The mirror of `build_in_facet_consolidation_prompt` one level up: where
    that one judges attributes inside a fixed facet, this judges facets inside
    a fixed axis. The axis is not part of the response schema, so a merge can
    never move a facet to another axis — when a group of ideas belongs
    elsewhere, the IDEAS move and the structure stays put.
    """
    neighbours = f"""
Here are the other axes in this domain, for reference only. They are NOT merge
candidates — they are shown so you can name a real destination when a group of
ideas belongs on another axis, and so you can write boundaries against what
actually exists next door.

<neighbour_axes>
{neighbour_axes_block}
</neighbour_axes>
""" if neighbour_axes_block else ""

    return f"""You are a qualitative research analyst specializing in open-ended survey coding. Your task is to settle the final facet set on one axis, now that every response has been assigned to a facet.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question being analyzed:

<survey_question>
"{survey_question}"

answers vary in terms of: {noun_phrase}
</survey_question>

Here is the domain and the axis you are working within:

<domain>
{domain_name} — {domain_definition}
</domain>

<axis>
{axis_name} — {axis_description}
</axis>

Here are the facets on this axis, each with the number of responses actually
assigned to it, its share of the axis, and a sample of the responses it really
holds:

<axis_facets>
{facets_block}
</axis_facets>
{neighbours}
Judge each facet on what it actually holds, not on how its label reads. The
counts and the response texts above are the evidence; the labels were written
before a single response had been assigned.

<consolidation_rules>
**1. DIMENSION FIRST.** Facets that describe different dimensions stay apart,
however similar their labels look. Orthogonality is a guardrail against merging,
never a reason to merge.

**2. PREVALENCE SETS GRANULARITY** — within one dimension only. Use the shares
shown: keep what is large, group what is thin, split what is large and diverse.

**3. LIFT, DON'T FLATTEN.** When several thin facets share a dimension, name the
concept they share. Do not dissolve them into a catch-all.

**4. PLAIN, MEANINGFUL LABELS.** A facet name states a value, not the axis it
sits on. Descriptive only — evaluation is captured per response as valence,
elsewhere.

**5. THE AXIS IS FIXED.** Every facet you return belongs to this axis. You
cannot move a facet to another axis, and you cannot add or rename axes.

**6. FOUR EXITS FOR WHAT DOES NOT FIT.** For a group of responses sitting in a
facet it does not belong to: move it to a facet that already exists (here or on
a neighbouring axis), widen the holding facet's description so it honestly
covers them, split the facet into named children, or — only when the responses
carry no substantive content at all — send them out. "Out" is not an escape
hatch for "does not fit what I chose".

**7. ONE SOURCE, ONE DESTINATION.** A source facet may be claimed by only one
returned facet, unless you route explicitly by response text.

**8. KEEP THE VALUES THAT ARE ACTUALLY THERE.** Do not collapse the axis to a
single facet because that is tidier. If the responses show four values, return
four facets.
</consolidation_rules>

Output requirements:
- All output (facet names, descriptions and rules) must be in {language}
- Copy response texts verbatim when you route them; they are matched literally

Provide your output as valid JSON following the response schema provided.
"""


class InAxisFacet(BaseModel):
    """One facet surviving consolidation on this axis."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "keep = unchanged; merge = several source facets into this one; "
            "widen = same facet, description restated to cover what it holds; "
            "split = one source facet divided into named children"
        )
    )
    facet_name: str = Field(..., description="Short descriptive name (2-5 words)")
    facet_description: str = Field(
        ..., description="What this facet captures, faithful to the responses it holds"
    )
    inclusion_rule: str = Field(
        ..., description="What types of responses belong in this facet"
    )
    exclusion_rule: str = Field(
        default="", description="What does NOT belong, when it clarifies the boundary"
    )
    example_observations: List[str] = Field(
        ..., description="2-5 responses this facet holds, verbatim"
    )
    source_facets: List[str] = Field(
        ..., description="facet_name of every source facet consumed into this one"
    )
    instance_texts: List[str] = Field(
        default_factory=list, description=(
            "Only for a split: the exact response texts routed to this child, verbatim"
        )
    )


class FacetMisfitGroup(BaseModel):
    """A group of responses sitting in a facet they do not belong to."""
    from_facet: str = Field(..., description="The facet currently holding them")
    instance_texts: List[str] = Field(
        ..., description="The exact response texts, verbatim"
    )
    verdict: Literal["move", "out"] = Field(
        ..., description="move = to a named existing facet; out = no substantive content"
    )
    target_facet: str = Field(
        default="", description="For 'move': the facet they belong to. Empty for 'out'."
    )
    reason: str = Field(..., description="One sentence on why they do not belong")


class InAxisConsolidatedResponse(BaseModel):
    """Final facet inventory for ONE axis, plus the misfits found on it."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning: (1) read each facet's contents against its label "
            "and note groups that do not belong, (2) group facets by underlying "
            "dimension, (3) set granularity by prevalence using the shares shown, "
            "(4) route each non-fitting group to one of the four exits, (5) check "
            "every label states a value rather than the axis, (6) assemble the "
            "final inventory."
        )
    )
    facets: List[InAxisFacet] = Field(
        ..., description="The complete facet set for this axis after consolidation"
    )
    misfits: List[FacetMisfitGroup] = Field(
        default_factory=list, description="Response groups that do not belong where they sit"
    )


# =============================================================================
# §6 ATTRIBUTE DISCOVERY (P6) — per facet within domain, chunked
# =============================================================================

def build_attribute_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facet_name: str,
    facet_description: str,
    observations: List[str],
    excluded_facets: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Discover concrete attributes (L4) within a facet."""
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )

    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        #attribute_guidance = rules.attribute_instruction
        attribute_definition = _extract_definition(rules.attribute_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )
        attribute_definition = attribute_guidance
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"

    excluded_block = _build_exclusion_block(
        excluded_facets or [], "excluded_facets"
    )

    excluded_block_light = _build_exclusion__light_block(
        excluded_facets or []
    )

    return f"""You are a qualitative research analyst specializing in survey response analysis. 
Your task is to identify the fewest recurring attributes that provide full coverage of a set of observations within a specific facet.

{attribute_definition} An attribute must:
- Be a descriptive, data-grounded category based on shared meaning across multiple observations
- Be non-evaluative (no judgment, sentiment, or valence)
- Stay strictly within the facet boundaries
- Be internally coherent (one clear underlying concept)
- Be externally distinctive:
  * Ontologically distinct (no overlap, no subset/superset, no reframing of same phenomenon)
  * Semantically separable (no ambiguity in coding; no "could go either way")
- Be non-redundant (adds unique conceptual value; no duplicate concepts)
- Be grounded in the data (supported by multiple observations or repeated patterns)

Here is the survey context you are working with:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Here is the taxonomy context that defines your working framework:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name} — {dimension_description}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>

And you are working within this facet:
<taxonomy_facet>
{facet_name} — {facet_description}
</taxonomy_facet>
{excluded_block}
</taxonomy_context>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>

# Instructions

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1: Cluster observations**
Group similar observations together based on shared descriptive meaning. Identify recurring patterns in what is being said within {facet_name}.

Focus on the specific quality, property, or feature being described.

**Step 2: Identify candidate attributes**
Based on these clusters, identify candidate attributes.

For each candidate attribute, assess:
- the attribute name
- the specific observable property it captures
- which observations support it
- whether it is internally coherent
- whether it is ontologically distinct from other candidate attributes

Remember: an attribute names a specific quality or trait — a concrete, observable property, not a verbatim span from the response.

**Step 3: Verify internal coherence**
Check whether each candidate attribute captures one clear underlying concept.

Reject or split candidate attributes that:
- combine multiple different kinds of phenomena
- mix descriptive content with evaluation
- are too broad to support clear coding

**Step 4: Verify distinctness**
Check each pair of candidate attributes to ensure they are:
- ontologically distinct (not overlapping in conceptual space; one is not a subset of another)
- semantically separable (someone coding a response would clearly know which attribute applies, with no "could go either way" situations)
- not two different lenses on the same phenomenon

If two attributes fail this test, consolidate them into one broader attribute or redefine the boundaries more clearly.

**Step 5: Verify facet boundaries**
Check that each retained attribute falls strictly within the included facet of {facet_name}.

Exclude attributes that belong more naturally to other facets, including:
{excluded_block_light}

**Step 6: Prepare final output**
Return only the dominant attributes that pass all checks above.

For each attribute, provide:
- a short descriptive name in {language} (2-5 words)
- a description in {language} of what the attribute captures — a concrete, observable property (1-2 sentences)
- the parent facet name: {facet_name}
- 2-3 representative observations from the input, using the exact observation text

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (attribute names, descriptions, and example observations) must be written in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Never create attributes that differ only in evaluative direction (e.g. a positive and a negative version of the same concept). Capture the concept as ONE attribute; positive/negative is recorded separately as valence. A response that is only an overall judgment with no descriptive content ("good", "fine", "not great") belongs to a single residual overall-judgment attribute, never to positive/negative variants.
- Attributes must be grounded in repeated patterns across observations
- Attributes must be internally coherent
- Attributes must be externally distinctive
- Attributes must remain strictly within the included facet
- Each attribute must capture one specific quality, not multiple
- All output must be in {language}
- Use exact observation text in the examples, not observation numbers

Use your scratchpad field for Steps 1-6 to show your analytical thinking. Then provide your final output as valid JSON."""


class DiscoveredAttribute(BaseModel):
    """A concrete attribute (L4) discovered within a facet."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures — a concrete, observable property (1-2 sentences)"
    )
    parent_facet: str = Field(
        ..., description="The facet this attribute belongs to"
    )
    example_observations: List[str] = Field(
        ..., description="2-3 representative observations from the input"
    )


class AttributeDiscoveryResult(BaseModel):
    """P5 output: attributes discovered within a facet."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before identifying attributes: "
            "(1) cluster observations by shared descriptive meaning, "
            "(2) identify candidate attributes and assess coherence and distinctness, "
            "(3) verify internal coherence — one clear concept per attribute, "
            "(4) verify distinctness — ontologically distinct and semantically separable, "
            "(5) verify facet boundaries — exclude attributes belonging to other facets, "
            "(6) prepare final output with only dominant attributes that pass all checks"
        )
    )
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="Concrete attributes identified within the facet"
    )


# =============================================================================
# §7 ATTRIBUTE ASSIGNMENT (P7) — one idea per task
# =============================================================================


def _build_attribute_codebook_block(
    attributes: List['DiscoveredAttribute'],
) -> str:
    """Format discovered attributes as a numbered list for assignment.

    The [A#] id is what `attr_id_to_name` keys the response parse on."""
    lines = []
    for i, attr in enumerate(attributes, 1):
        examples = "; ".join(attr.example_observations[:3])
        lines.append(
            f"[A{i}] {attr.attribute_name}\n"
            f"    Description: {attr.attribute_description}\n"
            f"    Examples: {examples}"
        )
    return "\n\n".join(lines)


def build_attribute_assignment_prompt_single(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    facet_name: str,
    facet_description: str,
    attributes: List['DiscoveredAttribute'],
    idea_label: str,
) -> str:
    """Build prompt for assigning a single idea to an attribute (L4) within a facet."""
    attribute_codebook = _build_attribute_codebook_block(attributes)

    return f"""You are a qualitative coding assistant. Assign the survey response idea below to the attribute that best captures the specific quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<facet_context>
Facet: {facet_name} -- {facet_description}
</facet_context>

<attributes>
{attribute_codebook}
</attributes>

<idea>
{idea_label}
</idea>

### VALENCE (evaluation relative to attribute)
- "+" Positive — The response describes a positive instance of this attribute (meeting expectations, present, sufficient)
- "-" Negative — The response describes a negative instance of this attribute (failing expectations, absent, insufficient)
- "0" Neutral — The response is descriptive, ambiguous, or does not express evaluation
- Valence is not emotional sentiment, but evaluative direction relative to the attribute

Assign this idea to the single best-fitting attribute. Return the attribute ID (e.g. "A1", "A2"), your confidence (0.0-1.0), and the valence (+, -, or 0).

Provide your response as valid JSON following the response schema provided."""


class AttributeAssignmentResult(BaseModel):
    """Single idea-to-attribute assignment result."""
    assigned_attribute_id: str = Field(
        ..., description=(
            "The attribute ID from the [A#] prefix (e.g. 'A1', 'A3'). "
            "Return ONLY the ID, not the attribute name."
        )
    )
    confidence: float = Field(
        ..., description="Confidence in the assignment (0.0 to 1.0)"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="Evaluative direction relative to the attribute: + positive, - negative, 0 neutral"
    )



# =============================================================================
# §8 ATTRIBUTE CONSOLIDATION (P8) — in-facet, post-assignment
# =============================================================================

def build_in_facet_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facet_name: str,
    facet_description: str,
    attributes_block: str,
    neighbour_block: str,
) -> str:
    """Finalise the attribute inventory of ONE facet, after every idea is assigned.

    Runs after assignment, so each attribute is shown with its real size and its
    real contents instead of the examples discovery guessed at. The facet is fixed:
    nothing in this call can move an attribute to another facet. When a group of
    ideas belongs elsewhere, the IDEAS move (`misfits`) and the structure stays put.
    """
    if dimension_def:
        rules = dimension_def.prompt_rules
        attribute_guidance = rules.attribute_instruction
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
        # What "no substantive content" means depends on the dimension's domain axis:
        # step 3 already words it per dimension for the standing drain domain.
        contentless_test = dimension_def.standing_bare.short
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property -- not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        domain_key_idea = "the subject the statement refers to"
        contentless_test = "an evaluation or filler with nothing named on the domain axis"

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to finalise the attribute inventory of ONE facet: "{facet_name}", inside domain "{domain_name}".

Every idea has already been assigned, so you see what each attribute ACTUALLY holds -- not what its label promised. Judge the contents, not the name.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Use the survey context to:

<survey_context_usage>
- Interpret the meaning of attributes relative to the survey question
- Ensure consolidated attributes are directly relevant to what is being asked
- Preserve terminology and phrasing appropriate to the survey language
- Avoid introducing attributes that are not grounded in the question intent
</survey_context_usage>

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name} — {dimension_description}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working inside this one facet:
<taxonomy_facet>
Domain: {domain_name} -- {domain_definition}
Facet:  {facet_name} -- {facet_description}
</taxonomy_facet>
</taxonomy_context>

Here are this facet's attributes, with their real size and their real contents:
<facet_attributes>
{attributes_block}
</facet_attributes>

{neighbour_block}

# Understanding Attributes

Conceptualization:
{attribute_guidance}

# Consolidation Rules

<strict_consolidation_rule>
Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping by these rules, in order.

1. DIMENSION FIRST (orthogonality — the guardrail).
   For each concept, determine WHICH underlying dimension it answers.
   - Concepts on DIFFERENT dimensions are orthogonal: NEVER merge them into one attribute (e.g. socio-economic class vs political orientation vs age are different dimensions).
   - Mutually-exclusive VALUES/POLES of the SAME dimension are also kept apart (e.g. "young" vs "old"); merging opposite poles creates an empty container.
   - Do NOT create separate attributes based only on the object discussed, when the same underlying value applies — an object is not a dimension.

2. PREVALENCE SETS GRANULARITY (within a dimension only).
   Each attribute shows its share of this facet. Judge size RELATIVE to its siblings, never against an absolute number.
   - The largest attributes keep their own identity — never dissolve a well-supported concept.
   - Attributes far below their siblings are GROUPED, but only with same-dimension neighbours, into one attribute that still names the shared value in plain language.
   - An attribute holding a large share AND visibly diverse contents is too abstract: SPLIT it (rule 6), do not widen it.
   - Variants that differ only in evaluative direction ("positive X" and "negative X") collapse to ONE attribute "X"; the direction is recorded separately as valence, not as separate attributes.
   Prevalence decides how finely to split WITHIN a dimension; it NEVER licenses merging ACROSS dimensions.

3. LIFT, DON'T FLATTEN.
   When grouping is needed, raise concepts to a shared higher-abstraction label that still carries their meaning — NOT a label that merely names the axis.
   FORBIDDEN: a container that only names the axis it sits on — the reader learns what was being measured, not what was said.
   REQUIRED: a label that states the value itself, so the reader knows what the respondents expressed.
   Test: read the label alone. If it tells you only which question was asked, it is a container; if it tells you what the answer was, it is a value.

4. PLAIN, MEANINGFUL LABELS.
   Name every surviving attribute in everyday language. Test: reading the label alone, a layperson knows which distinction is meant, given the survey question. No jargon, no nominalizations, no dimension-names.

5. THE FACET IS FIXED.
   Every attribute you return belongs to "{facet_name}". You cannot move an attribute to another facet, and you cannot create an attribute that belongs to another facet.
   If a GROUP OF IDEAS belongs elsewhere, report it under `misfits` — the ideas move, the attribute stays here.

6. FOUR EXITS FOR WHAT DOES NOT FIT.
   Read what each attribute actually contains. Where contents do not match the label, choose per group:
   - the group points at ONE existing attribute (in this facet or a neighbouring one)
       -> `misfits`, verdict "move": name the target attribute and the EXACT response texts
   - the group is one coherent concept that has no attribute yet
       -> action "split": name the child attributes and which EXACT response texts go to each
   - the group is diverse but genuinely related to this attribute
       -> action "widen": restate the description so it honestly covers what is there
   - the group carries NO SUBSTANTIVE CONTENT WHATSOEVER — filler, or {contentless_test}
       -> `misfits`, verdict "out"
   "out" is not an escape hatch for "this does not fit the attributes I chose". A text that names something real HAS substance: if it has no home yet, create one with "split". Only content-free text goes out.
   Moves and splits must be expressed as EXACT response texts copied from the contents shown above — never as counts, paraphrases or summaries. Every decision has to be checkable against the data.

7. ONE SOURCE, ONE DESTINATION — unless you route by text.
   Every attribute in the input must end up in exactly ONE returned attribute.
   If you want to divide one input attribute's contents over TWO returned attributes, that is a SPLIT: use action "split" for each part and list the exact response texts belonging to it in `instance_texts`.
   Listing the same source attribute under two returned attributes WITHOUT instance_texts is not interpretable — the ideas cannot be routed and will be left where they are.

8. KEEP THE VALUES THAT ARE ACTUALLY THERE.
   Grouping is not the same as discarding. If the contents hold two distinct values, return two attributes — merging them into one and sending the remainder "out" loses real answers.
   Collapsing a facet to a SINGLE attribute removes a whole level of the hierarchy: the facet name then says nothing the attribute does not already say. Do that only when the contents genuinely express one value.

Precedence when rules conflict: 1 (orthogonality) > 5 (facet is fixed) > 2 (prevalence grouping) > 4 (label clarity).
</strict_consolidation_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 -- Read the contents against the label**
For each attribute, compare what it HOLDS with what its name and description CLAIM. Note every group of contents that does not belong.

**Step 2 -- Identify the dimensions present**
Group the attributes by the underlying dimension each one answers. Different dimensions stay separate; never collapse across them.

**Step 3 -- Set granularity by prevalence, within a dimension**
Use the shares shown. Keep the large ones. Group the ones far below their siblings. Split the large-and-diverse ones.

**Step 4 -- Route what does not fit**
For each group from Step 1, pick one of the four exits in rule 6. When the target is in a neighbouring facet, name it exactly as it appears in the neighbour list.

**Step 5 -- Check every label is a plain, stateable value**
No dimension-name containers; each label names a value a layperson can picture.

**Step 6 -- Prepare final output**
Return the attributes that survive for THIS facet, plus every misfit group you found.

For each surviving attribute, provide:
- action: "keep", "merge", "widen" or "split"
- A short descriptive name (2-5 words)
- A description of what it captures -- a concrete, observable property (1-2 sentences)
- 2-3 representative example observations (exact text)
- source_attributes: the original attribute names that feed this one
- instance_texts: for "split" ONLY, the exact response texts routed to this child

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Attributes must be internally coherent (one clear concept each)
- Attributes must be externally distinctive (no overlap, no subset/superset)
- Every returned attribute belongs to "{facet_name}" — there is no other option
- All output must be in {language}

Use your scratchpad field for Steps 1-6 to show your analytical thinking. Then provide your final output as valid JSON."""


def build_neighbour_block(
    neighbours: List[Tuple[str, List[Tuple[str, int]]]],
) -> str:
    """Format adjacent facets as steer-clear context for in-facet consolidation.

    `neighbours`: [(facet_name, [(attribute_name, n_ideas), ...]), ...]

    Shown so the model can write its boundaries against real neighbours instead of
    abstract ones, and so it can name a target when a group of ideas belongs to one
    of them. Explicitly NOT merge candidates — without that instruction the model
    starts merging across facets, which is the failure this phase exists to prevent.
    """
    if not neighbours:
        return ""
    lines = [
        "<neighbouring_facets>",
        "These facets sit beside yours in the same domain. They are shown so you can "
        "write your boundaries against real neighbours instead of abstract ones.",
        "THEY ARE NOT MERGE CANDIDATES. You may not merge your attributes into them, "
        "and you may not restate their attributes as your own. Their only two uses:",
        "  (a) sharpen your own labels, so yours states what theirs does not;",
        "  (b) name a target when a group of ideas in YOUR facet clearly belongs to one of them.",
    ]
    for facet_name, attrs in neighbours:
        if not attrs:
            continue
        listed = ", ".join(f"{n} ({c})" for n, c in attrs)
        lines.append(f'  Facet "{facet_name}" — attributes: {listed}')
    lines.append("</neighbouring_facets>")
    return "\n".join(lines)


class InFacetAttribute(BaseModel):
    """One attribute surviving in-facet consolidation. Its facet is fixed by the task."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "What was done: 'keep' unchanged, 'merge' several sources into one, "
            "'widen' the description to cover the real contents, "
            "'split' one bucket into children (then instance_texts is required)"
        )
    )
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures (1-2 sentences)"
    )
    example_observations: List[str] = Field(
        ..., description="2-3 representative observations, exact text from the contents shown"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description=(
            "Original attribute names feeding this one (for 'keep', its own name). "
            "A source may appear under only ONE returned attribute, unless you are "
            "splitting it — then use action 'split' and fill instance_texts."
        )
    )
    instance_texts: List[str] = Field(
        default_factory=list,
        description=(
            "For action 'split' ONLY: the exact response texts routed to this child, "
            "copied verbatim from the contents shown. Required when a source attribute "
            "is divided over more than one returned attribute. Empty otherwise."
        )
    )


class MisfitGroup(BaseModel):
    """A group of ideas sitting in this facet that does not belong to the attribute holding it."""
    from_attribute: str = Field(
        ..., description="The attribute currently holding these ideas"
    )
    instance_texts: List[str] = Field(
        ..., description=(
            "The exact response texts that do not belong, copied verbatim from the "
            "contents shown. Never counts, paraphrases or summaries."
        )
    )
    verdict: Literal["move", "out"] = Field(
        ..., description=(
            "'move' when these ideas belong to a named existing attribute; "
            "'out' when they carry no substantive content at all"
        )
    )
    target_attribute: Optional[str] = Field(
        default=None,
        description=(
            "For verdict 'move': the attribute these ideas belong to, named exactly as "
            "shown in this facet or in the neighbouring facets list. Null for 'out'."
        )
    )
    reason: str = Field(
        ..., description="One sentence: why these texts do not belong where they are"
    )


class InFacetConsolidatedResponse(BaseModel):
    """Final attribute inventory for ONE facet, plus the misfits found in it."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning: (1) read each attribute's contents against its label "
            "and note groups that do not belong, (2) group attributes by underlying dimension, "
            "(3) set granularity by prevalence using the shares shown -- keep the large, group "
            "the thin, split the large-and-diverse, (4) route each non-fitting group to one of "
            "the four exits, (5) check every label states a value rather than an axis, "
            "(6) assemble the final inventory."
        )
    )
    attributes: List[InFacetAttribute] = Field(
        ..., description="The attributes surviving for this facet, all belonging to it"
    )
    misfits: List[MisfitGroup] = Field(
        default_factory=list,
        description="Groups of ideas that do not belong to the attribute holding them"
    )

    @model_validator(mode="after")
    def _routable(self):
        """Reject an inventory whose ideas cannot be routed.

        Enforced here rather than in the prompt for the same reason `parent_facet`
        was removed from the schema: a rule the model can decline to follow is not a
        rule. instructor surfaces these messages and retries, so the model gets to
        correct itself instead of silently producing an unroutable answer.
        """
        for a in self.attributes:
            if a.action == "split" and not a.instance_texts:
                raise ValueError(
                    f'attribute "{a.attribute_name}" has action "split" but no '
                    f'instance_texts. A split must list the exact response texts '
                    f'routed to each child, or the ideas cannot be divided.'
                )

        claimed_by: Dict[str, List[str]] = {}
        for a in self.attributes:
            for src in (a.source_attributes or []):
                claimed_by.setdefault(src, []).append(a.attribute_name)

        for src, claimants in claimed_by.items():
            if len(claimants) < 2:
                continue
            without_texts = [a.attribute_name for a in self.attributes
                             if src in (a.source_attributes or []) and not a.instance_texts]
            if without_texts:
                raise ValueError(
                    f'source attribute "{src}" is claimed by {len(claimants)} returned '
                    f'attributes ({", ".join(claimants)}), but {", ".join(without_texts)} '
                    f'give no instance_texts. Either let ONE attribute take "{src}", or '
                    f'make every claimant action "split" and list the exact response '
                    f'texts each one takes.'
                )
        return self


# =============================================================================
# §9 VALENCE-NEUTRAL MERGE (P9) — collapse valence-split attribute pairs
# =============================================================================

class ValenceNeutralAttribute(BaseModel):
    """One descriptive, valence-neutral attribute replacing a valence-split pair."""
    pair_id: int = Field(..., description="The id of the attribute pair this replaces")
    attribute_name: str = Field(
        ..., description="One descriptive, valence-neutral attribute name (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="A 1-2 sentence valence-neutral description"
    )


class ValenceNeutralRenameResponse(BaseModel):
    """Neutral replacements for the supplied valence-split attribute pairs."""
    attributes: List[ValenceNeutralAttribute] = Field(
        ..., description="Exactly one neutral attribute per input pair_id"
    )


def build_valence_neutral_rename_prompt(pairs: list, language: str = "Dutch") -> str:
    """Collapse valence-split attribute pairs into one descriptive, valence-neutral
    attribute each. `pairs`: list of dicts with pair_id, name_a, desc_a, name_b,
    desc_b, samples.
    """
    blocks = []
    for p in pairs:
        samples = ", ".join(f'"{s}"' for s in p.get("samples", []))
        blocks.append(
            f"[{p['pair_id']}]\n"
            f'  A: "{p["name_a"]}" — {p.get("desc_a", "")}\n'
            f'  B: "{p["name_b"]}" — {p.get("desc_b", "")}\n'
            f"  example mentions: {samples}"
        )
    pairs_block = "\n\n".join(blocks)

    return f"""You are cleaning up a taxonomy. Each numbered pair below wrongly split ONE concept by evaluative direction (valence): the two attributes mean the same thing, but one captures the positive side and the other the negative/neutral side. Valence has been baked into the attribute, which is wrong — valence is recorded separately per response.

For each pair, produce ONE descriptive, valence-neutral attribute that covers both sides:
- The name (2-5 words, in {language}) and description (1-2 sentences, in {language}) must be purely descriptive.
- Do NOT encode positive/negative/good/bad — that direction is captured separately as valence.
- Name the underlying subject the two share (e.g. a "positive impression" + "negative impression" pair becomes "overall impression").

Pairs:
{pairs_block}

Return exactly one entry per pair_id. Begin now and provide your output as valid JSON following the response schema provided."""
