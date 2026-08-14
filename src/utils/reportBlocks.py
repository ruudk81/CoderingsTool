"""Rendering of verbose-report blocks. Pure: records in, lines out.

Separate from `verboseReporter.py` on purpose. The reporter owns *when* something
is printed; this module owns *how* it looks. That split is what makes alignment
possible at all: a column width is a property of a whole group, so the widths
cannot be chosen until every row of that group is known. A reporter that prints
line by line — which is what `stat_line()` does — can never align anything,
however many formatting options it grows.

Being pure also makes the layout testable. There is nothing to assert about a
string the caller assembled itself; there is plenty to assert about
`render_group()` given the same rows.

## Markers stay out of the aligned columns

A marker sits on a group's title line and nowhere else. Emoji have unreliable
display width — 🕳️ carries a variation selector and renders one or two cells
depending on the terminal — so any column that contains one cannot be aligned
reliably. Keeping them on the title line means the numbers below always line up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# The block vocabulary's markers. A small fixed set where each one means
# something: decoration that carries no information is noise, only cheerier.
MARK_DROPPED = "🧹"   # removed from the data, gone
MARK_DRAIN = "🕳️"     # kept, but in a catch-all rather than a real category
MARK_PLACED = "✅"    # placed in a substantive category
MARK_WARN = "⚠️"      # needs a human eye

INDENT = "    "
EXAMPLE_SEP = " · "
MAX_EXAMPLES = 6


@dataclass
class Metric:
    """One measured row: a count, optionally of a total, optionally illustrated."""
    label: str
    value: int
    of: Optional[int] = None
    examples: List[str] = field(default_factory=list)

    @property
    def share(self) -> Optional[float]:
        if not self.of:
            return None
        return 100.0 * self.value / self.of


def _fmt_share(share: Optional[float]) -> str:
    """A percentage with one decimal, or nothing at all.

    Dutch decimal comma: these reports are read by a Dutch researcher, and a
    number that reads as English in a Dutch sentence is a small papercut on
    every line.
    """
    if share is None:
        return ""
    return f"{share:.1f}%".replace(".", ",")


def _fmt_examples(examples: Sequence[str]) -> str:
    if not examples:
        return ""
    shown = list(examples)[:MAX_EXAMPLES]
    tail = f" (+{len(examples) - len(shown)})" if len(examples) > len(shown) else ""
    return EXAMPLE_SEP.join(shown) + tail


@dataclass
class Group:
    """A titled set of rows, with an optional headline figure of its own."""
    title: str
    rows: List[Metric] = field(default_factory=list)
    marker: str = ""
    total: Optional[Metric] = None


@dataclass(frozen=True)
class Widths:
    """The column widths a block settled on. Shared by every row in it."""
    label: int
    value: int
    share: int


def measure(groups: Sequence[Group]) -> Widths:
    """One set of widths over ALL rows in a block, not per group.

    Per-group widths would align each group against itself and against nothing
    else, so the eye cannot run down a column — which is the whole reason to
    align. The cost is that one long label in one group widens every group.
    That is the right trade: a column that only sometimes lines up is worse
    than a wide one that always does.
    """
    # Totals count too: a group headline wider than any of its rows would push
    # its own numbers out of the column the rows settled on. Found by test, not
    # by eye — 10,5% on a headline against 6,3% on the rows below it.
    rows = [r for g in groups for r in g.rows]
    rows += [g.total for g in groups if g.total is not None]
    return Widths(
        label=max((len(r.label) for r in rows), default=0),
        value=max((len(f"{r.value}") for r in rows), default=0),
        share=max((len(_fmt_share(r.share)) for r in rows), default=0),
    )


def _row(prefix: str, label: str, value: int, share: Optional[float],
         widths: Widths, pad_to: int) -> str:
    """One line: label padded to a common width, then the two number columns.

    `pad_to` is measured in characters BEFORE the value column, so a group's
    headline and its rows put their numbers in the same place even though the
    headline sits one indent level higher and carries a marker.
    """
    head = f"{prefix}{label}"
    head = head + " " * max(pad_to - _width(head), 1)
    return (f"{head}{value:>{widths.value}}"
            f"  {_fmt_share(share):>{widths.share}}").rstrip()


def _width(text: str) -> int:
    """Display width, counting an emoji plus its variation selector as two.

    `len()` is wrong here: 🕳️ is two code points and renders as roughly two
    cells, so padding computed from `len()` would push that one line out of
    line with every other.
    """
    return sum(0 if ch == "️" else (2 if ord(ch) > 0x2100 else 1)
               for ch in text)


def render_group(group: Group, widths: Widths, *, indent: str = INDENT) -> List[str]:
    """One titled group, rendered against the block's shared widths."""
    pad_to = 2 * len(indent) + widths.label + 2
    marker = f"{group.marker}  " if group.marker else ""

    if group.total is None:
        lines = [f"{indent}{marker}{group.title}"]
    else:
        lines = [_row(f"{indent}{marker}", group.title, group.total.value,
                      group.total.share, widths, pad_to)]

    for row in group.rows:
        line = _row(indent * 2, row.label, row.value, row.share, widths, pad_to)
        examples = _fmt_examples(row.examples)
        if examples:
            line = f"{line}   {examples}"
        lines.append(line)
    return lines


def render_block(groups: Sequence[Group], *, indent: str = INDENT) -> List[str]:
    """Every group in one block, sharing one set of column widths."""
    widths = measure(groups)
    return [line for g in groups for line in render_group(g, widths, indent=indent)]


def render_flow(parts: Sequence[object], *, indent: str = INDENT) -> str:
    """`1236 responsen  →  2210 fragmenten  →  2203 ideeën`.

    Takes alternating count and noun. Splitting and filtering are one movement —
    splitting makes fragments, the filter drops the empty ones — so showing only
    the endpoints hides that anything happened in between.
    """
    pairs = [f"{parts[i]} {parts[i + 1]}" for i in range(0, len(parts) - 1, 2)]
    return f"{indent}{'  →  '.join(pairs)}"
