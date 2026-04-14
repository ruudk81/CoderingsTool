#%%
"""
Cross-domain attribute consolidation experiment.

Finds overlapping attributes across domains using embeddings,
clusters them into small groups, and lets the LLM consolidate.

Usage:
    cd src && python -m development.cross_domain_consolidation
"""

import asyncio
import sys
from collections import defaultdict

import nest_asyncio
nest_asyncio.apply()
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_4_classifier.models_classifier import (
    TaxonomyResultsCache,
    TaxonomyClassifiedModel,
    TaxonomyClassifiedSubmodel,
)
from pipeline.step_3_ideaExtractor.models import ExtractionMetadata
from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
from utils.embedder import SharedEmbedder, format_idea_text, compute_medoid
from utils.llm import create_client, llm_create_async
from config import get_step_model, get_reasoning_params

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

# Embedding text format: "instance", "instance_interpretation", "full_abstraction_ladder"
CODE_SOURCE = "instance_interpretation"

# Cross-domain similarity threshold (0.0-1.0)
SIMILARITY_THRESHOLD = 0.6

# Sliding window for LLM-sized groups
WINDOW_SIZE = 10      # max attributes per LLM call
WINDOW_OVERLAP = 2    # overlap between adjacent windows

# LLM consolidation settings
CONSOLIDATION_MODEL = get_step_model("classifier_p7")
CONSOLIDATION_TEMPERATURE = 0.3
CONSOLIDATION_MAX_TOKENS = 16000


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AttributeEntry(NamedTuple):
    """One attribute in the global inventory."""
    domain_name: str
    facet_name: str
    attribute_name: str
    attribute_description: str
    idea_count: int


# =============================================================================
# CACHE LOADING
# =============================================================================

def load_caches(
    filename: str = FILENAME,
    variable: str = VAR_NAME,
    sample_size: int = SAMPLE_SIZE,
):
    """Load taxonomy cache, growing model, and extraction metadata."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )
    cache_manager = CacheManager()

    # 1. Taxonomy metadata (domains, facets, attributes, assignments)
    taxonomy_cache = cache_manager.load_metadata_from_cache(
        filename=filename,
        step="taxonomy",
        variable_key=variable_key,
        model_cls=TaxonomyResultsCache,
    )
    if not taxonomy_cache:
        raise FileNotFoundError(
            f"No taxonomy metadata found for variable_key '{variable_key}'.\n"
            f"Run the taxonomy pipeline (step 4) first."
        )

    # 2. Growing model (per-idea facet/attribute assignments)
    classified = cache_manager.load_from_cache(
        filename=filename,
        step="taxonomy_classified",
        variable_key=variable_key,
        model_cls=TaxonomyClassifiedModel,
    )
    if not classified:
        raise FileNotFoundError(
            f"No classified ideas found for variable_key '{variable_key}'.\n"
            f"Run the taxonomy pipeline (step 4) first."
        )

    # 3. Extraction metadata (dimension, survey context)
    extraction_meta = cache_manager.load_metadata_from_cache(
        filename=filename,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=ExtractionMetadata,
    )

    return taxonomy_cache, classified, extraction_meta


# =============================================================================
# ATTRIBUTE INVENTORY
# =============================================================================

def build_attribute_inventory(
    taxonomy_cache: TaxonomyResultsCache,
) -> List[AttributeEntry]:
    """Build a flat list of all attributes across all domains."""
    inventory = []

    for domain_name, result in taxonomy_cache.partition_results.items():
        # Count ideas per attribute from assignments
        attr_counts: Dict[str, int] = defaultdict(int)
        for attr_name in result.attribute_assignments.values():
            attr_counts[attr_name] += 1

        # Walk facet -> attributes structure
        for facet_name, attrs in result.attributes.items():
            for attr_dict in attrs:
                name = attr_dict.get("attribute_name", "?")
                desc = attr_dict.get("attribute_description", "")
                count = attr_counts.get(name, 0)

                inventory.append(AttributeEntry(
                    domain_name=domain_name,
                    facet_name=facet_name,
                    attribute_name=name,
                    attribute_description=desc,
                    idea_count=count,
                ))

    return inventory


# =============================================================================
# IDEAS PER ATTRIBUTE
# =============================================================================

def collect_ideas_per_attribute(
    classified: List[TaxonomyClassifiedModel],
) -> Dict[tuple, List[TaxonomyClassifiedSubmodel]]:
    """Group ideas by (domain, attribute) from the growing model.

    Returns dict of (domain_name, attribute_name) -> list of ideas.
    """
    groups: Dict[tuple, List[TaxonomyClassifiedSubmodel]] = defaultdict(list)

    for resp in classified:
        if not resp.response_ideas:
            continue
        for idea in resp.response_ideas:
            domain = idea.partition_name or "(unknown)"
            attr = idea.attribute or "(no attribute)"
            groups[(domain, attr)].append(idea)

    return groups


# =============================================================================
# DISPLAY
# =============================================================================

def print_inventory(
    inventory: List[AttributeEntry],
    ideas_per_attr: Dict[tuple, List[TaxonomyClassifiedSubmodel]],
    extraction_meta: ExtractionMetadata = None,
):
    """Print the global attribute inventory."""
    print("=" * 80)
    print("CROSS-DOMAIN ATTRIBUTE INVENTORY")
    print("=" * 80)

    if extraction_meta:
        print(f"  Dimension: {extraction_meta.primary_dimension}")
        if extraction_meta.var_lab:
            print(f"  Survey question: {extraction_meta.var_lab}")

    print(f"  Dataset: {FILENAME}")
    print(f"  Variable: {VAR_NAME}, Sample: {SAMPLE_SIZE}")

    # Group by domain for display
    by_domain: Dict[str, List[AttributeEntry]] = defaultdict(list)
    for entry in inventory:
        by_domain[entry.domain_name].append(entry)

    total_attributes = 0
    total_ideas = 0

    for domain_name in sorted(by_domain.keys()):
        entries = sorted(by_domain[domain_name], key=lambda e: -e.idea_count)
        domain_ideas = sum(e.idea_count for e in entries)

        print(f"\n{'─' * 80}")
        print(f"DOMAIN: {domain_name} ({len(entries)} attributes, {domain_ideas} ideas)")
        print(f"{'─' * 80}")

        for entry in entries:
            # Cross-check against growing model count
            model_count = len(ideas_per_attr.get(
                (entry.domain_name, entry.attribute_name), []
            ))
            match = "" if model_count == entry.idea_count else f" [MISMATCH: model={model_count}]"

            print(f"  [{entry.idea_count:4d}] {entry.attribute_name}")
            print(f"         facet: {entry.facet_name}")
            print(f"         {entry.attribute_description[:100]}{'...' if len(entry.attribute_description) > 100 else ''}{match}")

        total_attributes += len(entries)
        total_ideas += domain_ideas

    print(f"\n{'=' * 80}")
    print(f"TOTALS: {total_attributes} attributes, {total_ideas} ideas across {len(by_domain)} domains")
    print(f"{'=' * 80}")


# =============================================================================
# EMBEDDING & MEDOIDS
# =============================================================================

class AttributeEmbedding(NamedTuple):
    """Centroid embedding result for one attribute."""
    domain_name: str
    attribute_name: str
    centroid: np.ndarray
    medoid_text: str  # most central idea text, for display only
    idea_count: int


async def embed_and_compute_centroids(
    ideas_per_attr: Dict[tuple, List[TaxonomyClassifiedSubmodel]],
    code_source: str = CODE_SOURCE,
) -> List[AttributeEmbedding]:
    """Embed all ideas, compute centroid per attribute.

    Embeds all texts in one batched call, then slices back per attribute.
    Centroid (mean) for similarity math, medoid text for display.
    """
    # Build ordered list of (key, texts) to preserve slicing order
    attr_keys = []
    attr_texts = []
    attr_counts = []

    for (domain, attr_name), ideas in sorted(ideas_per_attr.items()):
        if attr_name == "(no attribute)":
            continue
        texts = [format_idea_text(idea, code_source) for idea in ideas]
        attr_keys.append((domain, attr_name))
        attr_texts.append(texts)
        attr_counts.append(len(texts))

    # Flatten all texts for one big embed call
    all_texts = []
    for texts in attr_texts:
        all_texts.extend(texts)

    print(f"\n  Embedding {len(all_texts)} ideas across {len(attr_keys)} attributes "
          f"(code_source={code_source})...")

    embedder = SharedEmbedder()
    all_embeddings = await embedder.embed_texts(all_texts)

    print(f"  Embedding complete: {all_embeddings.shape}")

    # Slice back into per-attribute groups, compute centroid + medoid text
    results = []
    offset = 0
    for i, (domain, attr_name) in enumerate(attr_keys):
        count = attr_counts[i]
        attr_embeddings = all_embeddings[offset:offset + count]

        centroid = attr_embeddings.mean(axis=0)
        medoid_idx = compute_medoid(attr_embeddings)
        medoid_text = attr_texts[i][medoid_idx]

        results.append(AttributeEmbedding(
            domain_name=domain,
            attribute_name=attr_name,
            centroid=centroid,
            medoid_text=medoid_text,
            idea_count=count,
        ))
        offset += count

    return results


def print_centroids(attr_embeddings: List[AttributeEmbedding]):
    """Print centroid summary per attribute."""
    print(f"\n{'=' * 80}")
    print(f"ATTRIBUTE CENTROIDS ({len(attr_embeddings)} attributes)")
    print(f"{'=' * 80}")

    by_domain: Dict[str, List[AttributeEmbedding]] = defaultdict(list)
    for a in attr_embeddings:
        by_domain[a.domain_name].append(a)

    for domain_name in sorted(by_domain.keys()):
        entries = sorted(by_domain[domain_name], key=lambda a: -a.idea_count)
        print(f"\n{'─' * 80}")
        print(f"DOMAIN: {domain_name}")
        print(f"{'─' * 80}")

        for a in entries:
            text = a.medoid_text[:100] + "..." if len(a.medoid_text) > 100 else a.medoid_text
            print(f"  [{a.idea_count:4d}] {a.attribute_name}")
            print(f"         central idea: \"{text}\"")


# =============================================================================
# CROSS-DOMAIN SIMILARITY
# =============================================================================

CandidatePair = NamedTuple("CandidatePair", [
    ("similarity", float),
    ("attr_a", AttributeEmbedding),
    ("attr_b", AttributeEmbedding),
])


def find_cross_domain_candidates(
    attr_embeddings: List[AttributeEmbedding],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[CandidatePair]:
    """Find cross-domain attribute pairs above similarity threshold.

    Computes pairwise cosine similarity between attribute centroids,
    filters to cross-domain pairs only, and returns those above threshold.
    """
    n = len(attr_embeddings)
    centroids = np.array([a.centroid for a in attr_embeddings])
    sim_matrix = cosine_similarity(centroids)

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            # Skip same-domain pairs (already handled by P7)
            if attr_embeddings[i].domain_name == attr_embeddings[j].domain_name:
                continue
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                candidates.append(CandidatePair(
                    similarity=sim,
                    attr_a=attr_embeddings[i],
                    attr_b=attr_embeddings[j],
                ))

    candidates.sort(key=lambda c: -c.similarity)
    return candidates


def print_candidates(
    candidates: List[CandidatePair],
    attr_embeddings: List[AttributeEmbedding],
    threshold: float = SIMILARITY_THRESHOLD,
):
    """Print cross-domain candidate pairs."""
    print(f"\n{'=' * 80}")
    print(f"CROSS-DOMAIN CANDIDATE PAIRS (threshold >= {threshold})")
    print(f"{'=' * 80}")

    if not candidates:
        print(f"\n  No cross-domain pairs found above {threshold}.")
        print(f"  Try lowering SIMILARITY_THRESHOLD.")
        return

    for i, c in enumerate(candidates, 1):
        print(f"\n  {i}. similarity: {c.similarity:.3f}")
        print(f"     \"{c.attr_a.attribute_name}\" ({c.attr_a.domain_name}, {c.attr_a.idea_count} ideas)")
        print(f"     \"{c.attr_b.attribute_name}\" ({c.attr_b.domain_name}, {c.attr_b.idea_count} ideas)")

    # Summary
    involved = set()
    for c in candidates:
        involved.add((c.attr_a.domain_name, c.attr_a.attribute_name))
        involved.add((c.attr_b.domain_name, c.attr_b.attribute_name))

    total = len(attr_embeddings)
    n_involved = len(involved)
    n_isolated = total - n_involved

    print(f"\n{'─' * 80}")
    print(f"SUMMARY")
    print(f"  {len(candidates)} pairs above threshold ({threshold})")
    print(f"  {n_involved}/{total} attributes have a cross-domain neighbor")
    print(f"  {n_isolated}/{total} attributes isolated (no LLM call needed)")
    print(f"{'─' * 80}")


# =============================================================================
# ATTRIBUTE ORDERING & SLIDING WINDOW (Job 4)
# =============================================================================

def compute_attribute_order(
    attr_embeddings: List[AttributeEmbedding],
) -> List[int]:
    """Order attributes so that similar ones are adjacent.

    Uses agglomerative clustering (average linkage) to produce a dendrogram,
    then extracts the leaf order — a 1D ordering where nearby attributes
    are semantically close.
    """
    centroids = np.array([a.centroid for a in attr_embeddings])

    # Cosine distance matrix → condensed form for scipy
    sim_matrix = cosine_similarity(centroids)
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)  # ensure exact zeros on diagonal
    condensed = squareform(dist_matrix)

    # Hierarchical clustering → leaf order
    Z = linkage(condensed, method="average")
    order = leaves_list(Z).tolist()

    return order


def build_sliding_windows(
    ordered_indices: List[int],
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> List[List[int]]:
    """Slide a window across ordered indices to produce overlapping groups.

    Each window has `window_size` attributes. Adjacent windows share `overlap`
    attributes. If the last window has fewer than `overlap` new attributes
    beyond the previous window, it merges into the previous window.
    """
    step = window_size - overlap
    n = len(ordered_indices)
    windows = []

    for start in range(0, n, step):
        window = ordered_indices[start:start + window_size]
        windows.append(window)
        # Stop if this window reached the end
        if start + window_size >= n:
            break

    # Merge last window if it adds fewer than `overlap` new attributes
    if len(windows) >= 2:
        prev = set(windows[-2])
        new_in_last = [i for i in windows[-1] if i not in prev]
        if len(new_in_last) <= overlap:
            # Merge: extend previous window with the new attributes
            windows[-2] = windows[-2] + new_in_last
            windows.pop()

    return windows


def print_windows(
    windows: List[List[int]],
    attr_embeddings: List[AttributeEmbedding],
):
    """Print the sliding window groups."""
    print(f"\n{'=' * 80}")
    print(f"ATTRIBUTE GROUPS FOR LLM CONSOLIDATION "
          f"(window={WINDOW_SIZE}, overlap={WINDOW_OVERLAP})")
    print(f"{'=' * 80}")

    for g, window in enumerate(windows, 1):
        attrs = [attr_embeddings[i] for i in window]
        total_ideas = sum(a.idea_count for a in attrs)
        domains = sorted(set(a.domain_name for a in attrs))

        print(f"\n{'─' * 80}")
        print(f"GROUP {g} ({len(window)} attributes, {total_ideas} ideas, "
              f"{len(domains)} domains)")
        print(f"{'─' * 80}")

        for a in sorted(attrs, key=lambda x: -x.idea_count):
            print(f"  [{a.idea_count:4d}] {a.attribute_name}")
            print(f"         domain: {a.domain_name}")

    # Summary
    all_indices = set()
    for w in windows:
        all_indices.update(w)

    print(f"\n{'─' * 80}")
    print(f"SUMMARY")
    print(f"  {len(windows)} groups, {len(all_indices)}/{len(attr_embeddings)} "
          f"attributes covered")
    sizes = [len(w) for w in windows]
    print(f"  Group sizes: {sizes}")
    print(f"{'─' * 80}")


# =============================================================================
# LLM CONSOLIDATION (Job 5)
# =============================================================================

# -- Response models --

class CrossDomainConsolidatedAttribute(BaseModel):
    """An attribute after cross-domain consolidation."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures (1-2 sentences)"
    )
    parent_domain: str = Field(
        ..., description="The domain this attribute best belongs to"
    )
    parent_facet: str = Field(
        ..., description="The facet within the domain this attribute best belongs to"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Original attribute names that were merged into this one"
    )


class CrossDomainConsolidatedResponse(BaseModel):
    """LLM response for cross-domain attribute consolidation."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before consolidating attributes: "
            "(1) identify high-prevalence anchors from idea counts, "
            "(2) map lower-prevalence attributes onto anchors, "
            "(3) apply orthogonality and disambiguation tests across domains, "
            "(4) justify any low-prevalence attributes kept separate, "
            "(5) assign each surviving attribute to the best domain and facet, "
            "(6) prepare final minimal set of consolidated attributes"
        )
    )
    attributes: List[CrossDomainConsolidatedAttribute] = Field(
        ..., description="Deduplicated attributes, each assigned to its best domain and facet"
    )


# -- Prompt builder --

def build_cross_domain_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    dimension_def,
    domain_attributes_block: str,
) -> str:
    """Build prompt for cross-domain attribute consolidation.

    Adapted from P7's build_attribute_consolidation_prompt — same consolidation
    rules, but scoped across domains instead of across facets within one domain.
    """
    if dimension_def:
        rules = dimension_def.prompt_rules
        attribute_guidance = rules.attribute_instruction
        attribute_key_idea = rules.attribute_instruction.split(".")[0] if "." in rules.attribute_instruction else rules.attribute_instruction
        facet_key_idea = rules.facet_instruction.split(".")[0] if "." in rules.facet_instruction else rules.facet_instruction
        domain_key_idea = rules.domain_instruction.split(".")[0] if "." in rules.domain_instruction else rules.domain_instruction
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property -- not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        domain_key_idea = "the subject the statement refers to"

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to deduplicate attributes across domains, producing a consolidated attribute inventory.

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
</taxonomy_context>

Here are all domains, facets, and their attributes for this group:
<domain_attributes>
{domain_attributes_block}
</domain_attributes>

# Understanding Attributes

Conceptualization:
{attribute_guidance}

# Attribute Consolidation Rules

<strict_consolidation_rule>
1. PREVALENCE WEIGHTING
Attributes MUST be primarily driven by the **number of ideas linked to attributes**.

- Attributes with HIGH idea counts MUST form the **core structure of the codebook**.
- Attributes with LOW idea counts MUST NOT become standalone attributes unless absolutely necessary.
- LOW-prevalence attributes SHOULD be:
  - merged into the closest HIGH-prevalence phenomenon, OR
  - grouped into a broader combined phenomenon.

If forced to choose between:
- conceptual nuance
- prevalence dominance

--> ALWAYS prioritize prevalence dominance.

2. MERGE BIAS
When in doubt:
- MERGE rather than split
- Especially when an attribute has relatively few ideas

Attributes with low prevalence (e.g., <10-15 ideas) should almost never result in standalone attributes.

3. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged, even if they were discovered under different domains.

4. ORTHOGONALITY (MAIN RULE)
For each pair of attributes:
"Can a single observation plausibly fall under both?"

- Yes -> merge
- Doubt -> merge
- Only if clearly no -> keep separate

5. NO HIERARCHY
Attributes must not be:
- general vs. specific
- principle vs. application
If this occurs -> merge

6. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies -> merge

7. MINIMALITY (MANDATORY)
Use the smallest number of attributes that provides full coverage.
If an attribute is not strictly necessary -> remove it

8. DOMAIN & FACET ASSIGNMENT
Assign each surviving attribute to the ONE domain and ONE facet where it fits best.
If two domains fit equally well, assign to the domain with MORE ideas for that attribute.
Do NOT restructure or rename domains or facets -- only deduplicate attributes.
</strict_consolidation_rule>

<disambiguation_test>
For any pair of attributes:
"Can a clear rule assign every observation to exactly one attribute?"
- No -> merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt -> merge attributes
</precedence_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 -- Identify High-Prevalence Anchors**
- Identify attributes with the highest number of ideas.
- Treat these as the PRIMARY building blocks of the consolidated inventory.

**Step 2 -- Map Lower-Prevalence Attributes**
- Map lower-prevalence attributes onto these high-prevalence anchors wherever possible.
- Only keep an attribute separate if it:
  - is conceptually distinct AND
  - cannot reasonably be merged.

**Step 3 -- Apply orthogonality and disambiguation tests**
For each pair of candidate attributes, apply the orthogonality test and disambiguation test. Merge if either test fails.

**Step 4 -- Justify Low-Prevalence Attributes (MANDATORY)**
- If any attribute is primarily based on low idea counts:
- Explicitly justify why it was NOT merged into a higher-prevalence phenomenon.

**Step 5 -- Assign domain and facet**
For each surviving attribute, assign it to the best-fitting domain and facet.
If equal fit across domains, choose the domain with more ideas.

**Step 6 -- Prepare final output**
Return only the minimal set of consolidated attributes that pass all checks.

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures -- a concrete, observable property (1-2 sentences)
- The parent domain and parent facet this attribute best belongs to
- source_attributes: list of original attribute names that were merged into this one

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Attributes must be grounded in repeated patterns across observations
- Attributes must be internally coherent (one clear concept each)
- Attributes must be externally distinctive (no overlap, no subset/superset)
- Each attribute must be assigned to exactly ONE parent domain and ONE parent facet (best fit)
- All output must be in {language}

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON."""


# -- Data formatting --

def format_domain_attributes_block(
    window: List[int],
    attr_embeddings: List[AttributeEmbedding],
    inventory: List[AttributeEntry],
    taxonomy_cache: TaxonomyResultsCache,
) -> str:
    """Format attributes in a window as domain → facet → attribute block."""
    # Build lookup: (domain, attribute_name) → AttributeEntry
    entry_lookup = {}
    for entry in inventory:
        entry_lookup[(entry.domain_name, entry.attribute_name)] = entry

    # Get domain definitions
    domain_defs = {}
    for part in taxonomy_cache.partition_set.partitions:
        domain_defs[part.partition_name] = part.inclusion_definition

    # Get facet descriptions
    facet_descs: Dict[str, Dict[str, str]] = {}
    for domain_name, result in taxonomy_cache.partition_results.items():
        facet_descs[domain_name] = {}
        for facet_dict in result.facets:
            facet_descs[domain_name][facet_dict["facet_name"]] = facet_dict.get(
                "facet_description", ""
            )

    # Group window attributes by domain → facet
    by_domain_facet: Dict[str, Dict[str, List[AttributeEmbedding]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx in window:
        a = attr_embeddings[idx]
        entry = entry_lookup.get((a.domain_name, a.attribute_name))
        facet = entry.facet_name if entry else "(unknown)"
        by_domain_facet[a.domain_name][facet].append(a)

    # Format
    lines = []
    for domain_name in sorted(by_domain_facet.keys()):
        domain_def = domain_defs.get(domain_name, "")
        lines.append(f'Domain: "{domain_name}" — {domain_def}')

        for facet_name in sorted(by_domain_facet[domain_name].keys()):
            facet_desc = facet_descs.get(domain_name, {}).get(facet_name, "")
            lines.append(f'  Facet: "{facet_name}" — {facet_desc}')

            for a in by_domain_facet[domain_name][facet_name]:
                entry = entry_lookup.get((a.domain_name, a.attribute_name))
                desc = entry.attribute_description if entry else ""
                lines.append(
                    f'    - "{a.attribute_name}" ({a.idea_count} ideas) — {desc}'
                )

        lines.append("")

    return "\n".join(lines)


# -- LLM calling --

async def consolidate_group(
    client,
    window: List[int],
    attr_embeddings: List[AttributeEmbedding],
    inventory: List[AttributeEntry],
    taxonomy_cache: TaxonomyResultsCache,
    extraction_meta: ExtractionMetadata,
    dimension_def,
    group_num: int,
) -> Optional[CrossDomainConsolidatedResponse]:
    """Run LLM consolidation for one group/window."""
    domain_attributes_block = format_domain_attributes_block(
        window, attr_embeddings, inventory, taxonomy_cache
    )

    # Build dataset context section
    parts = []
    for key in ["sector", "entity", "topic", "perspective", "intent"]:
        value = getattr(extraction_meta, key, "")
        if value:
            parts.append(f"{key.capitalize()}: {value}")
    dataset_context_section = (
        "<dataset_context>\n" + "\n".join(parts) + "\n</dataset_context>"
        if parts else ""
    )

    prompt = build_cross_domain_consolidation_prompt(
        survey_question=extraction_meta.var_lab or "",
        language=extraction_meta.lang or "Dutch",
        dataset_context_section=dataset_context_section,
        dimension_name=extraction_meta.primary_dimension or "",
        dimension_description=extraction_meta.primary_dimension_description or "",
        dimension_def=dimension_def,
        domain_attributes_block=domain_attributes_block,
    )

    n_attrs = len(window)
    print(f"\n  Group {group_num}: sending {n_attrs} attributes to LLM...")

    response = await llm_create_async(
        client=client,
        model=CONSOLIDATION_MODEL,
        prompt=prompt,
        response_model=CrossDomainConsolidatedResponse,
        temperature=CONSOLIDATION_TEMPERATURE,
        max_tokens=CONSOLIDATION_MAX_TOKENS,
        **get_reasoning_params(CONSOLIDATION_MODEL, phase="classifier_p7"),
    )

    if response:
        print(f"  Group {group_num}: {n_attrs} → {len(response.attributes)} attributes")
    else:
        print(f"  Group {group_num}: LLM returned empty response")

    return response


async def run_consolidation(
    windows: List[List[int]],
    attr_embeddings: List[AttributeEmbedding],
    inventory: List[AttributeEntry],
    taxonomy_cache: TaxonomyResultsCache,
    extraction_meta: ExtractionMetadata,
) -> List[Optional[CrossDomainConsolidatedResponse]]:
    """Run LLM consolidation for all groups sequentially."""
    # Resolve dimension
    dimension_def = None
    if extraction_meta.primary_dimension:
        dimension_def = get_dimension(extraction_meta.primary_dimension)

    client = create_client(model=CONSOLIDATION_MODEL, async_mode=True)

    print(f"\n{'=' * 80}")
    print(f"CROSS-DOMAIN LLM CONSOLIDATION ({len(windows)} groups)")
    print(f"  Model: {CONSOLIDATION_MODEL}")
    print(f"{'=' * 80}")

    tasks = [
        consolidate_group(
            client, window, attr_embeddings, inventory,
            taxonomy_cache, extraction_meta, dimension_def, g,
        )
        for g, window in enumerate(windows, 1)
    ]
    results = await asyncio.gather(*tasks)

    return list(results)


# -- Display --

def print_consolidation_results(
    results: List[Optional[CrossDomainConsolidatedResponse]],
    windows: List[List[int]],
    attr_embeddings: List[AttributeEmbedding],
):
    """Print LLM consolidation results per group."""
    print(f"\n{'=' * 80}")
    print(f"CONSOLIDATION RESULTS")
    print(f"{'=' * 80}")

    total_before = 0
    total_after = 0

    for g, (window, response) in enumerate(zip(windows, results), 1):
        n_before = len(window)
        total_before += n_before

        print(f"\n{'─' * 80}")
        print(f"GROUP {g}")
        print(f"{'─' * 80}")

        if not response:
            print("  (no response)")
            continue

        n_after = len(response.attributes)
        total_after += n_after
        n_merged = n_before - n_after

        print(f"  {n_before} → {n_after} attributes ({n_merged} merged)")

        for attr in response.attributes:
            merged_info = ""
            if len(attr.source_attributes) > 1:
                merged_info = f" ← merged from: {', '.join(attr.source_attributes)}"
            elif attr.source_attributes:
                src = attr.source_attributes[0]
                if src != attr.attribute_name:
                    merged_info = f" ← renamed from: {src}"

            print(f"\n  • {attr.attribute_name}")
            print(f"    domain: {attr.parent_domain} > {attr.parent_facet}")
            print(f"    {attr.attribute_description[:120]}"
                  f"{'...' if len(attr.attribute_description) > 120 else ''}")
            if merged_info:
                print(f"    {merged_info}")

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total_before} attributes in → {total_after} attributes out")
    print(f"{'=' * 80}")


# =============================================================================
# REMAPPING (Job 6)
# =============================================================================

class MergeTarget(NamedTuple):
    """Where a merged/renamed attribute ends up."""
    new_attribute_name: str
    new_domain: str
    new_facet: str
    new_description: str


def build_merge_map(
    results: List[Optional[CrossDomainConsolidatedResponse]],
    windows: List[List[int]],
    attr_embeddings: List[AttributeEmbedding],
) -> Dict[str, MergeTarget]:
    """Build merge map from all group results, resolving overlap conflicts.

    "Merge wins, first group takes precedence." Process groups in seriation
    order so the most confident merges (highest similarity) happen first.

    Returns dict: old_attribute_name → MergeTarget.
    Only contains entries where something actually changes (merge or rename).
    """
    merge_map: Dict[str, MergeTarget] = {}
    already_processed: set = set()

    for g, (window, response) in enumerate(zip(windows, results), 1):
        if not response:
            continue

        for consolidated in response.attributes:
            target = MergeTarget(
                new_attribute_name=consolidated.attribute_name,
                new_domain=consolidated.parent_domain,
                new_facet=consolidated.parent_facet,
                new_description=consolidated.attribute_description,
            )

            for source_name in consolidated.source_attributes:
                if source_name in already_processed:
                    continue
                already_processed.add(source_name)

                # Only map if something actually changes
                if source_name != consolidated.attribute_name:
                    merge_map[source_name] = target

    return merge_map


def apply_remapping_to_cache(
    taxonomy_cache: TaxonomyResultsCache,
    merge_map: Dict[str, MergeTarget],
    inventory: List[AttributeEntry],
) -> TaxonomyResultsCache:
    """Apply merge map to taxonomy cache, returning a new copy.

    Handles both within-domain and cross-domain remaps.
    """
    import copy
    new_cache = TaxonomyResultsCache.model_validate(
        copy.deepcopy(taxonomy_cache.model_dump())
    )

    # Build lookup: attribute_name → source domain (from inventory)
    attr_source_domain: Dict[str, str] = {}
    for entry in inventory:
        attr_source_domain[entry.attribute_name] = entry.domain_name

    for old_name, target in merge_map.items():
        source_domain = attr_source_domain.get(old_name)
        if not source_domain:
            continue

        source_result = new_cache.partition_results.get(source_domain)
        target_result = new_cache.partition_results.get(target.new_domain)
        if not source_result or not target_result:
            continue

        # Find idea_ids assigned to the old attribute in the source domain
        idea_ids_to_move = [
            iid for iid, aname in source_result.attribute_assignments.items()
            if aname == old_name
        ]

        if not idea_ids_to_move:
            continue

        # Remove from source domain
        for iid in idea_ids_to_move:
            del source_result.attribute_assignments[iid]
            source_result.attribute_valence.pop(iid, None)
            source_result.attribute_confidence.pop(iid, None)

        # Cross-domain: also move facet assignments
        if source_domain != target.new_domain:
            for iid in idea_ids_to_move:
                # Move facet assignment
                old_facet = source_result.facet_assignments.pop(iid, None)
                old_facet_valence = source_result.facet_valence.pop(iid, None)
                old_facet_conf = source_result.facet_confidence.pop(iid, None)

                target_result.facet_assignments[iid] = target.new_facet
                if old_facet_valence:
                    target_result.facet_valence[iid] = old_facet_valence
                if old_facet_conf:
                    target_result.facet_confidence[iid] = old_facet_conf

        # Add to target domain with new attribute name
        for iid in idea_ids_to_move:
            target_result.attribute_assignments[iid] = target.new_attribute_name

        # Remove old attribute from source domain's attributes dict
        for facet_name, attrs_list in list(source_result.attributes.items()):
            source_result.attributes[facet_name] = [
                a for a in attrs_list if a.get("attribute_name") != old_name
            ]
            # Remove empty facet entries
            if not source_result.attributes[facet_name]:
                del source_result.attributes[facet_name]

        # Ensure target attribute exists in target domain's attributes dict
        if target.new_facet not in target_result.attributes:
            target_result.attributes[target.new_facet] = []

        # Check if target attribute already exists
        existing = [
            a for a in target_result.attributes[target.new_facet]
            if a.get("attribute_name") == target.new_attribute_name
        ]
        if not existing:
            target_result.attributes[target.new_facet].append({
                "attribute_name": target.new_attribute_name,
                "attribute_description": target.new_description,
            })

    return new_cache


def apply_remapping_to_growing_model(
    classified: List[TaxonomyClassifiedModel],
    merge_map: Dict[str, MergeTarget],
) -> List[TaxonomyClassifiedModel]:
    """Apply merge map to growing model, returning a new copy."""
    import copy
    new_classified = []

    for resp in classified:
        resp_dict = copy.deepcopy(resp.model_dump())
        new_resp = TaxonomyClassifiedModel.model_validate(resp_dict)

        if new_resp.response_ideas:
            for idea in new_resp.response_ideas:
                if idea.attribute and idea.attribute in merge_map:
                    target = merge_map[idea.attribute]
                    idea.attribute = target.new_attribute_name
                    idea.facet = target.new_facet
                    idea.partition_name = target.new_domain

        new_classified.append(new_resp)

    return new_classified


def save_consolidated(
    taxonomy_cache: TaxonomyResultsCache,
    classified: List[TaxonomyClassifiedModel],
):
    """Save consolidated taxonomy to cache with xdomain step names."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME],
        is_merged=False,
        sample_size=SAMPLE_SIZE,
    )
    cache_manager = CacheManager()

    saved_meta = cache_manager.save_metadata_to_cache(
        metadata=taxonomy_cache,
        filename=FILENAME,
        step="taxonomy_xdomain",
        variable_key=variable_key,
    )
    saved_growing = cache_manager.save_to_cache(
        data=classified,
        filename=FILENAME,
        step="taxonomy_classified_xdomain",
        variable_key=variable_key,
    )

    if saved_meta and saved_growing:
        print(f"\n  Saved to cache: taxonomy_xdomain + taxonomy_classified_xdomain")
    else:
        print(f"\n  WARNING: cache save failed (meta={saved_meta}, growing={saved_growing})")


def print_remapping_report(
    merge_map: Dict[str, MergeTarget],
    taxonomy_before: TaxonomyResultsCache,
    taxonomy_after: TaxonomyResultsCache,
):
    """Print before/after remapping report."""
    print(f"\n{'=' * 80}")
    print(f"REMAPPING REPORT")
    print(f"{'=' * 80}")

    # List merges
    if merge_map:
        print(f"\n  Merges/renames ({len(merge_map)}):")
        for old_name, target in sorted(merge_map.items()):
            print(f"    \"{old_name}\" → \"{target.new_attribute_name}\" "
                  f"({target.new_domain} > {target.new_facet})")
    else:
        print(f"\n  No merges to apply.")

    # Before/after summary
    def count_taxonomy(cache: TaxonomyResultsCache):
        n_domains = len(cache.partition_results)
        n_facets = sum(len(r.facets) for r in cache.partition_results.values())
        n_attrs = sum(
            len(a) for r in cache.partition_results.values()
            for a in r.attributes.values()
        )
        n_ideas = sum(
            len(r.attribute_assignments) for r in cache.partition_results.values()
        )
        return n_domains, n_facets, n_attrs, n_ideas

    d_b, f_b, a_b, i_b = count_taxonomy(taxonomy_before)
    d_a, f_a, a_a, i_a = count_taxonomy(taxonomy_after)

    print(f"\n  {'':20s} {'BEFORE':>10s} {'AFTER':>10s}")
    print(f"  {'Domains':20s} {d_b:>10d} {d_a:>10d}")
    print(f"  {'Facets':20s} {f_b:>10d} {f_a:>10d}")
    print(f"  {'Attributes':20s} {a_b:>10d} {a_a:>10d}")
    print(f"  {'Ideas assigned':20s} {i_b:>10d} {i_a:>10d}")

    if i_b != i_a:
        print(f"\n  WARNING: idea count changed! {i_b} → {i_a} (diff: {i_a - i_b})")
    else:
        print(f"\n  Ideas preserved: {i_a}")

    # Full consolidated taxonomy view (like P7 verbose output)
    print(f"\n{'=' * 80}")
    print(f"CONSOLIDATED TAXONOMY ({d_a} domains)")
    print(f"{'=' * 80}")

    for domain_name in sorted(taxonomy_after.partition_results.keys()):
        result = taxonomy_after.partition_results[domain_name]
        n_facets = len(result.facets)
        n_attrs = sum(len(a) for a in result.attributes.values())
        n_assigned = len(result.attribute_assignments)

        # Count ideas per attribute
        attr_counts: Dict[str, int] = defaultdict(int)
        for aname in result.attribute_assignments.values():
            attr_counts[aname] += 1

        print(f"\n{'─' * 80}")
        print(f"DOMAIN: {domain_name} "
              f"({n_facets} facets, {n_attrs} attributes, {n_assigned} ideas)")
        print(f"{'─' * 80}")

        for facet_name in sorted(result.attributes.keys()):
            attrs = result.attributes[facet_name]
            print(f"  Facet: {facet_name} ({len(attrs)} attributes)")
            for attr_dict in sorted(attrs, key=lambda a: -attr_counts.get(a.get("attribute_name", ""), 0)):
                aname = attr_dict.get("attribute_name", "?")
                count = attr_counts.get(aname, 0)
                print(f"    - {aname} [{count} ideas]")

    print(f"{'=' * 80}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    taxonomy_cache, classified, extraction_meta = load_caches()

    inventory = build_attribute_inventory(taxonomy_cache)
    ideas_per_attr = collect_ideas_per_attribute(classified)

    print_inventory(inventory, ideas_per_attr, extraction_meta)

    # Job 2: Embed and compute centroids
    attr_embeddings = asyncio.run(embed_and_compute_centroids(ideas_per_attr))
    print_centroids(attr_embeddings)

    # Job 3: Cross-domain similarity
    candidates = find_cross_domain_candidates(attr_embeddings)
    print_candidates(candidates, attr_embeddings)

    # Job 4: Order attributes and build sliding window groups
    ordered = compute_attribute_order(attr_embeddings)
    windows = build_sliding_windows(ordered)
    print_windows(windows, attr_embeddings)

    # Job 5: LLM consolidation per group
    results = asyncio.run(run_consolidation(
        windows, attr_embeddings, inventory, taxonomy_cache, extraction_meta
    ))
    print_consolidation_results(results, windows, attr_embeddings)

    # Job 6: Remap and save
    merge_map = build_merge_map(results, windows, attr_embeddings)
    new_taxonomy = apply_remapping_to_cache(taxonomy_cache, merge_map, inventory)
    new_classified = apply_remapping_to_growing_model(classified, merge_map)
    print_remapping_report(merge_map, taxonomy_cache, new_taxonomy)
    save_consolidated(new_taxonomy, new_classified)


if __name__ == "__main__":
    main()

# %%
