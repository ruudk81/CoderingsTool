"""
exportNaming.py - One canonical name for everything written to exports/.

    {dataset}_{var_name}_{sample}_{doctype}.{ext}

Spaces become underscores; a sample_size of None (or the literal "full") becomes
"full". No timestamp: a rerun of the same analysis overwrites its own file. No
truncation of the dataset name either — two datasets sharing a long prefix would
otherwise overwrite each other's output.

Reading a name back needs BOTH the known dataset stems and the doctype
vocabulary, because a dataset stem (M260421_Tabellenbestand_Casus), a variable
name (Qd1_combined) and a doctype (taxonomie_fijn) can each contain underscores.
Splitting on "_" alone is ambiguous, so parse_export_filename() anchors on the
longest match at each end and takes what is left in the middle.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# The full vocabulary. export_filename() rejects anything outside it, so a typo
# becomes an error at the call site instead of a file nobody can find back.
DOCTYPES = frozenset({
    *(f"log_step{n}" for n in range(8)),        # exports/verbose_logs/
    *(f"prompts_step{n}" for n in range(1, 7)),  # exports/prompts/
    "codering",                                  # exports/coderingen/
    "codeboek",                                  # coderingen/ (.sav) + codebook/ (.xlsx, .csv)
    "gecombineerd",
    "taxonomie",
    "taxonomie_fijn",
    "taxonomie_grof",
    "taxonomie_raw",
    "kladblok",                                  # exports/codebook/ scratchpad
    "kosten",                                    # exports/costs/
})


@dataclass(frozen=True)
class ExportName:
    """A parsed export filename. `dataset` is the .sav stem with underscores."""
    dataset: str
    var_name: str
    sample: str
    doctype: str
    ext: str


def _slug(text: str) -> str:
    return text.replace(" ", "_")


def export_filename(
    filename: str,
    var_name: str,
    sample_size,
    doctype: str,
    ext: str,
) -> str:
    """Canonical name for a file in exports/.

    `filename` is the dataset's .sav name, `var_name` the bare variable (not the
    enhanced cache key), `sample_size` an int, None, or "full".
    """
    if doctype not in DOCTYPES:
        raise ValueError(f"unknown doctype {doctype!r} — add it to DOCTYPES")
    base = _slug(Path(filename).stem)
    var = _slug(var_name)
    sample = "full" if sample_size is None else str(sample_size)
    return f"{base}_{var}_{sample}_{doctype}.{ext}"


def parse_export_filename(
    name: str,
    known_stems: Iterable[str],
) -> Optional[ExportName]:
    """Read a canonical name back, or None if it is not one.

    `known_stems` are the dataset filenames (with or without .sav, spaces or
    underscores) that may appear; the longest matching one wins, so a dataset
    whose name is a prefix of another still resolves correctly.
    """
    stem, _, ext = name.rpartition(".")
    if not stem or not ext:
        return None

    # Longest doctype wins: "taxonomie_fijn" must not read as "taxonomie".
    doctype = max(
        (d for d in DOCTYPES if stem.endswith(f"_{d}")),
        key=len,
        default=None,
    )
    if doctype is None:
        return None
    head = stem[: -(len(doctype) + 1)]

    slugs = sorted({_slug(Path(s).stem) for s in known_stems}, key=len, reverse=True)
    dataset = next((s for s in slugs if head.startswith(f"{s}_")), None)
    if dataset is None:
        return None

    var_name, _, sample = head[len(dataset) + 1:].rpartition("_")
    if not var_name or not sample:
        return None
    return ExportName(dataset, var_name, sample, doctype, ext)
