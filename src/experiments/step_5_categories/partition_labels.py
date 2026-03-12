"""
Label extraction and formatting utilities for Category Discovery.

Uses step 3's abstraction ladder terminology:
  instance       — verbatim span from response
  interpretation — concrete interpretation (what it means)
  abstraction    — broader significance (why it matters)

Handles:
- Extracting text from idea objects based on configurable label_source
- Computing composite text from stored fields (ladder, idea_rungs)
- Applying optional label_prefix (static or dynamic from idea fields)
- Formatting pre-cluster results as prompt-injectable hints
"""

from typing import List, Optional, Tuple


# Composite format names that are computed from stored fields, not direct attributes.
COMPOSITE_FORMATS = frozenset({"ladder", "idea_rungs"})


def _compute_ladder(idea) -> str:
    """Compute 'instance → interpretation → abstraction'.

    Falls back to idea.idea if all ladder fields are empty.
    """
    parts = []
    for field in ('instance', 'interpretation', 'abstraction'):
        val = (getattr(idea, field, '') or '').strip()
        if val:
            parts.append(val)
    return " → ".join(parts) if parts else (getattr(idea, 'idea', '') or '').strip()


def _compute_idea_rungs(idea) -> str:
    """Compute 'idea → interpretation → abstraction'."""
    parts = []
    for field in ('idea', 'interpretation', 'abstraction'):
        val = (getattr(idea, field, '') or '').strip()
        if val:
            parts.append(val)
    return " → ".join(parts)


def format_label(idea, label_source: str, label_prefix: str = "") -> str:
    """Extract and format a single label from an idea object.

    Args:
        idea: Idea object with step 3 fields (idea, instance, interpretation,
              abstraction, domain, valence)
        label_source: Stored field name or composite format key.
            Stored fields: "idea", "instance", "interpretation", "abstraction",
                           "domain"
            Computed composites:
                "ladder"     — instance → interpretation → abstraction
                "idea_rungs" — idea → interpretation → abstraction
        label_prefix: Optional static prefix prepended to each label.

    Returns:
        Formatted label string, or empty string if all fields are empty.
    """
    if label_source == "ladder":
        raw = _compute_ladder(idea)
    elif label_source == "idea_rungs":
        raw = _compute_idea_rungs(idea)
    else:
        raw = (getattr(idea, label_source, '') or '').strip()

    if not raw:
        return ""

    if not label_prefix:
        return raw

    return f"{label_prefix}{raw}"


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
