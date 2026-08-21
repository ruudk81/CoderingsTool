#%%
"""Bekijk de prompts van stap 5 consensus naar het LLM: consolidatie en codebook schrijven."""

import sys
from pathlib import Path
from typing import NamedTuple

src_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_5_codeGenerator.prompts_writer import WriterResult, make_writer_model
from pipeline.step_5_codeGenerator.consensus.prompts_consolidation import (
    ConsolidationResult, make_consolidation_model,
)

SHOW_ALL = False


class _CardRef(NamedTuple):
    """Stand-in voor een AttributeCard: `_shuffled` leest `.attribute_id`,
    `make_consolidation_model` leest `.tag`."""
    attribute_id: str
    name: str

    @property
    def tag(self) -> str:
        return f"[{self.attribute_id}] {self.name}"


class _ShapeRef(NamedTuple):
    """Stand-in voor een CodeShape: `make_writer_model` leest alleen `.key`."""
    key: str


def _consolidation_model(metadata: dict):
    ids = metadata.get("card_ids") or []
    names = metadata.get("card_names") or []
    if not ids or len(ids) != len(names):
        return ConsolidationResult
    return make_consolidation_model(
        [_CardRef(attribute_id=i, name=n) for i, n in zip(ids, names)])


def _writer_model(metadata: dict):
    keys = metadata.get("shape_keys") or []
    if not keys:
        return WriterResult
    return make_writer_model([_ShapeRef(key=k) for k in keys])


PROMPT_MODELS = {
    "consolidation": _consolidation_model,
    "codebook_writer": _writer_model,
}


if __name__ == "__main__":
    render(step="5c", models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
