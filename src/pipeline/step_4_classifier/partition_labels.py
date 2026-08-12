"""
Label extraction and formatting utilities for Taxonomy Classifier.

Uses step 3's taxonomy terminology (Dimension > Domain > Facet > Attribute):
  instance        — Attribute (L4): verbatim span from response
  interpretation  — Ladder rung 2: concrete meaning (survey language)
  abstraction     — Ladder rung 3: broader significance (survey language)
  facet           — Facet (L3): dimension-specific aspect
  domain          — Domain (L2): thematic domain

Handles:
- Extracting text from idea objects based on configurable label_source
- Computing composite text from stored fields
- Applying optional label_prefix (static or dynamic from idea fields)
- Formatting pre-cluster results as prompt-injectable hints

## Why the abstraction rung is not in the production label

`instance_interpretation` is what step 4 runs on, not the full three-rung
`ladder`. The third rung states an observation's broader significance, and that
is the job of the facet and the domain — so feeding it in hands the model its
own answer: on ASN Qd1 the rung "positieve algemene beoordeling van de bank"
produced a facet literally called "Algemene reputatiebeoordeling". Step 4 then
aggregates step 3's abstractions instead of inducting structure from responses.

Two measured consequences on that dataset (2026-08-12, 2199 ideas):

- Labels are 51% shorter (94 → 46 characters), and every phase pays for that
  text — discovery chunks, assignment menus, refinement contents blocks.
- Unique labels drop from 1926 to 1504. The rung was CREATING uniqueness: 422
  ideas with an identical instance and interpretation were split into separate
  reps because step 3 worded their abstraction differently. Step 3 reproduces
  themes reliably but wording only loosely, so the rung imported that
  instability straight into step 4's notion of "the same label" — and with it
  into the block-move semantics that identical text relies on.
"""

from typing import List, Optional, Tuple


# Composite format names that are computed from stored fields, not direct attributes.
COMPOSITE_FORMATS = frozenset({
    "instance_interpretation", "ladder", "idea_interpretation",
})


def _compute_instance_interpretation(idea) -> str:
    """Compute 'instance → interpretation' — the production label.

    Falls back to idea.idea when both rungs are empty.
    """
    return _join_rungs(idea, ('instance', 'interpretation'), fallback=True)


def _compute_ladder(idea) -> str:
    """Compute 'instance → interpretation → abstraction', all three rungs.

    Kept as a selectable source for diagnostics that want to read the full
    ladder; production uses `instance_interpretation` (see module docstring).
    """
    return _join_rungs(idea, ('instance', 'interpretation', 'abstraction'),
                       fallback=True)


def _compute_idea_interpretation(idea) -> str:
    """Compute 'idea → interpretation'."""
    return _join_rungs(idea, ('idea', 'interpretation'), fallback=False)


def _join_rungs(idea, fields: Tuple[str, ...], *, fallback: bool) -> str:
    """Join the non-empty fields with an arrow, in the order given."""
    parts = [(getattr(idea, field, '') or '').strip() for field in fields]
    parts = [p for p in parts if p]
    if parts:
        return " → ".join(parts)
    return (getattr(idea, 'idea', '') or '').strip() if fallback else ""


def format_label(
    idea,
    label_source: str,
    label_prefix: str = "",
) -> str:
    """Extract and format a single label from an idea object.

    Args:
        idea: Idea object with step 3 fields (idea, instance, interpretation,
              abstraction, facet, domain)
        label_source: Stored field name or composite format key.
            Stored fields: "idea", "instance", "interpretation", "abstraction", "facet", "domain"
            Computed composites:
                "instance_interpretation" — instance → interpretation  (production)
                "ladder"                  — instance → interpretation → abstraction
                "idea_interpretation"     — idea → interpretation
        label_prefix: Optional static prefix prepended to each label.

    Returns:
        Formatted label string, or empty string if all fields are empty.
    """
    if label_source == "instance_interpretation":
        raw = _compute_instance_interpretation(idea)
    elif label_source == "ladder":
        raw = _compute_ladder(idea)
    elif label_source == "idea_interpretation":
        raw = _compute_idea_interpretation(idea)
    else:
        raw = (getattr(idea, label_source, '') or '').strip()

    if not raw:
        return ""

    if label_prefix:
        return f"{label_prefix}{raw}"
    return raw


def collect_unique_labels_with_domains(
    ideas: list,
    label_source: str = "ladder",
    label_prefix: str = "",
) -> Tuple[List[str], List[Optional[str]]]:
    """Collect unique label strings and their corresponding domain (first seen).

    Returns:
        (labels, domains) — parallel lists of the same length.
        domain is None when the idea has no domain field.
    """
    seen: dict = {}  # label -> domain (first seen)
    for idea in ideas:
        label = format_label(idea, label_source, label_prefix)
        if label and label not in seen:
            domain = (getattr(idea, 'domain', '') or '') or None
            seen[label] = domain
    labels = list(seen.keys())
    domains = list(seen.values())
    return labels, domains


def collect_unique_labels(
    ideas: list,
    label_source: str = "ladder",
    label_prefix: str = "",
) -> List[str]:
    """Collect unique label strings from a list of idea objects.

    Args:
        ideas: List of idea objects (IdeasExtractedSubmodel)
        label_source: Stored field or composite format key. See format_label().
        label_prefix: Optional prefix to prepend

    Returns:
        List of unique label strings (preserving first-seen order).
    """
    seen = {}  # dict preserves insertion order
    for idea in ideas:
        label = format_label(idea, label_source, label_prefix)
        if label and label not in seen:
            seen[label] = True
    return list(seen.keys())
