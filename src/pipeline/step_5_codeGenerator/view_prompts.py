#%%
"""
View the prompts step 5 sent to the LLM: relations, umbrella merge, codebook
writing, MECE detection, MECE probing.

All five response models are built at runtime, constrained to the exact
identifiers offered in that call (attribute names, umbrella names, shape
keys, candidate names, or a pair's two code names + idea refs). Each builder
below reconstructs that same enum-constrained model from identifiers stored
in the captured prompt's own metadata, so the schema shown here is the one
instructor actually enforced — not a guess. If a captured entry predates
those metadata fields, the builder falls back to the base, unconstrained
response model instead of pretending the constrained variant is shown; the
header line then names the base class, honestly.

`mece_probe` fires once per candidate pair per round, so a captured file can
hold many entries of that type — expected, not a bug; see PROCESSING.md.

Usage:
    cd src && python -m pipeline.step_5_codeGenerator.view_prompts
"""

import sys
from pathlib import Path
from typing import NamedTuple

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_5_codeGenerator.prompts_mece import (
    OverlapDetectionResult, ProbeResult, make_overlap_model, make_probe_model,
)
from pipeline.step_5_codeGenerator.prompts_relations import RelationsResult, make_relations_model
from pipeline.step_5_codeGenerator.prompts_umbrella_merge import (
    UmbrellaMergeResult, make_umbrella_merge_model,
)
from pipeline.step_5_codeGenerator.prompts_writer import WriterResult, make_writer_model

SHOW_ALL = False


class _NamedRef(NamedTuple):
    """Stand-in for a Concept/Umbrella/CodeCandidate: the two attributes
    `_shuffled` and the `make_*_model` builders actually read."""
    attribute_id: str
    name: str


class _ShapeRef(NamedTuple):
    """Stand-in for a CodeShape: only `.key` is read by `make_writer_model`."""
    key: str


class _PairRef(NamedTuple):
    """Stand-in for a CandidatePair: only the two code names are read by
    `make_probe_model`."""
    code_a: str
    code_b: str


class _IdeaRef(NamedTuple):
    """Stand-in for a ProbeIdea: only `.idea_ref` is read by `make_probe_model`."""
    idea_ref: int


def _relations_model(metadata: dict):
    ids = metadata.get("concept_ids") or []
    names = metadata.get("concept_names") or []
    if not ids or len(ids) != len(names):
        return RelationsResult
    concepts = [_NamedRef(attribute_id=i, name=n) for i, n in zip(ids, names)]
    return make_relations_model(concepts)


def _umbrella_merge_model(metadata: dict):
    names = metadata.get("umbrella_names") or []
    if not names:
        return UmbrellaMergeResult
    umbrellas = [_NamedRef(attribute_id=n, name=n) for n in names]
    return make_umbrella_merge_model(umbrellas)


def _writer_model(metadata: dict):
    keys = metadata.get("shape_keys") or []
    if not keys:
        return WriterResult
    shapes = [_ShapeRef(key=k) for k in keys]
    return make_writer_model(shapes)


def _mece_detect_model(metadata: dict):
    names = metadata.get("candidate_names") or []
    if not names:
        return OverlapDetectionResult
    candidates = [_NamedRef(attribute_id=n, name=n) for n in names]
    return make_overlap_model(candidates)


def _mece_probe_model(metadata: dict):
    code_a, code_b = metadata.get("code_a"), metadata.get("code_b")
    idea_refs = metadata.get("idea_refs") or []
    if not code_a or not code_b or not idea_refs:
        return ProbeResult
    pair = _PairRef(code_a=code_a, code_b=code_b)
    ideas = [_IdeaRef(idea_ref=ref) for ref in idea_refs]
    return make_probe_model(pair, ideas)


PROMPT_MODELS = {
    "relations": _relations_model,
    "umbrella_merge": _umbrella_merge_model,
    "codebook_writer": _writer_model,
    "mece_detect": _mece_detect_model,
    "mece_probe": _mece_probe_model,
}


if __name__ == "__main__":
    render(step=5, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
