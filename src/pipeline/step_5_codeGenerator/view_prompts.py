#%%
"""View the prompts step 5 sent to the LLM: consolidation and codebook writing.

Beide responsemodellen worden per call opgebouwd en beperkt tot precies de
identifiers die in die call zijn aangeboden — de attribuuttags bij consolidatie,
de shape-keys bij het schrijven. De builders hieronder reconstrueren datzelfde
model uit de metadata die bij de prompt is opgeslagen, zodat het schema dat je
hier ziet het schema is dat instructor daadwerkelijk heeft afgedwongen — geen
benadering. Mist een opgeslagen entry die metadata, dan valt de builder terug op
het onbeperkte basismodel in plaats van te doen alsof; de kopregel noemt dan de
basisklasse, eerlijk.

De v1-prompttypes (relations, umbrella_merge, mece_detect, mece_probe) staan
hier niet meer: die fasen draaien niet sinds de v2-promotie. Een oud
promptbestand met die types rendert nog, met "[no model mapped]" erbij.

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

from pipeline.step_5_codeGenerator.prompts_writer import WriterResult, make_writer_model
from pipeline.step_5_codeGenerator.v2.prompts_consolidation import (
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
    render(step=5, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
