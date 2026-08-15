"""Tests for the layout of verbose report blocks.

These tests exist because the renderer is pure. There is nothing to test about a
string the caller assembled itself; given the same records, `render_block()` can
be tested completely.
"""
import pytest

from utils.reportBlocks import (
    MARK_DRAIN,
    MARK_DROPPED,
    Group,
    Metric,
    _width,
    measure,
    render_block,
    render_flow,
)


def _blok():
    return [
        Group("weggegooid", [Metric("zonder inhoud", 7, of=2210)],
              marker=MARK_DROPPED, total=Metric("weggegooid", 7, of=2210)),
        Group("vangnetten", [
            Metric("zonder onderwerp", 138, of=2203),
            Metric("onbekend met het onderwerp", 80, of=2203),
            Metric("ander onderwerp", 14, of=2203),
        ], marker=MARK_DRAIN, total=Metric("vangnetten", 232, of=2203)),
    ]


def _zichtbaar(lijn):
    """Column position as you see it, not as Python indexes it.

    An emoji is one character to `len()` and ~two cells on screen; testing on a
    string index would therefore misjudge exactly the lines carrying a marker.
    """
    return _width(lijn)


# =============================================================================
# ALIGNMENT — the reason this is a module of its own
# =============================================================================

def test_the_whole_block_shares_one_column():
    """Aligning per group aligns each group against itself and nothing else; the
    eye then cannot follow a column, and that was the whole point."""
    assert render_block(_blok()) == [
        "    \U0001f9f9  weggegooid                    7   0,3%",
        "        zonder inhoud                 7   0,3%",
        "    \U0001f573\ufe0f  vangnetten                  232  10,5%",
        "        zonder onderwerp            138   6,3%",
        "        onbekend met het onderwerp   80   3,6%",
        "        ander onderwerp              14   0,6%",
    ]


def test_every_line_puts_its_percentage_on_the_same_cell():
    """The test that counts: visible width, not string index."""
    einden = {_zichtbaar(l[:l.index("%") + 1]) for l in render_block(_blok())}
    assert len(einden) == 1, f"percentages eindigen op {einden}"


def test_the_group_header_counts_towards_the_column_width():
    """Regression: measure() looked only at rows. A header of 10.5% against rows
    of 6.3% pushed the header out of the column the rows had chosen."""
    w = measure([Group("g", [Metric("r", 1, of=1000)],
                       total=Metric("g", 500, of=1000))])
    assert w.share == len("50,0%")
    assert w.value == len("500")


# =============================================================================
# MARKERS EN BREEDTE
# =============================================================================

def test_an_emoji_counts_as_two_cells():
    """`len()` is hier fout: 🕳️ is twee codepoints en beslaat ~twee cellen, dus
    padding uit len() zou juist die regel scheeftrekken."""
    assert _width(MARK_DRAIN) == 2
    assert _width(MARK_DROPPED) == 2
    assert _width("abc") == 3


def test_a_marker_does_not_shift_the_number_column():
    """The header carries a marker and sits one level above its rows; the numbers
    must still end on the same cell. Visible width, because a string index counts
    the emoji as one character and the screen as two."""
    for lijnen in (
        render_block([Group("t", [Metric("r", 1, of=10)],
                            marker=MARK_DROPPED, total=Metric("t", 1, of=10))]),
        render_block([Group("t", [Metric("r", 1, of=10)],
                            total=Metric("t", 1, of=10))]),
    ):
        einden = {_zichtbaar(l[:l.index("%") + 1]) for l in lijnen}
        assert len(einden) == 1, f"{einden} in {lijnen}"


# =============================================================================
# GETALLEN EN VOORBEELDEN
# =============================================================================

def test_percentage_gebruikt_een_decimale_komma():
    lijnen = render_block([Group("g", [Metric("r", 1, of=3)])])
    assert "33,3%" in lijnen[1]
    assert "33.3" not in lijnen[1]


def test_without_a_denominator_no_percentage():
    lijnen = render_block([Group("g", [Metric("r", 7)])])
    assert "%" not in lijnen[1]


def test_examples_are_truncated_with_a_count():
    lijnen = render_block([Group("g", [
        Metric("r", 9, of=9, examples=[f"v{i}" for i in range(9)])])])
    assert "(+3)" in lijnen[1]
    assert "v6" not in lijnen[1]


def test_examples_separated_by_a_dot():
    lijnen = render_block([Group("g", [Metric("r", 2, of=2, examples=["a", "b"])])])
    assert "a · b" in lijnen[1]


def test_no_line_ends_on_whitespace():
    for lijn in render_block(_blok()):
        assert lijn == lijn.rstrip()


# =============================================================================
# FLOW
# =============================================================================

def test_a_flow_shows_the_intermediate_step():
    """Splitting and filtering are one movement; showing only start and end hides
    that anything happened."""
    assert render_flow([1236, "responsen", 2210, "fragmenten", 2203, "ideeën"]).strip() \
        == "1236 responsen  →  2210 fragmenten  →  2203 ideeën"
