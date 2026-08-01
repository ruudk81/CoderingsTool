#%%
"""
View the prompts step 3 sent to the LLM: specifiers, dimension, domains, extraction.

`idea_extraction` is the one phase whose response model does not exist until the
run: it is built from the dimension the dataset turned out to be about. Hence the
builder rather than a class — the model shown is the one that dimension produces.

Usage:
    cd src && python -m pipeline.step_3_ideaExtractor.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
    GenericSpecifierGroup1Response,
    GenericSpecifierGroup2Response,
    PrimaryDimensionChunkResponse,
    PrimaryDimensionConsolidatedResponse,
    DomainChunkResponse,
    DomainConsolidatedResponse,
    ReformulatedDomains,
    create_extraction_model,
)
from pipeline.step_3_ideaExtractor.dimension_data import get_dimension

SHOW_ALL = False


def _extraction_model(metadata: dict):
    """The extraction model for the dimension this run settled on."""
    dimension_key = metadata.get("primary_dimension", "ATTRIBUTES_ASSOCIATIONS")
    return create_extraction_model(dimension=get_dimension(dimension_key))


PROMPT_MODELS = {
    "context_specifier_group1": GenericSpecifierGroup1Response,
    "context_specifier_group2": GenericSpecifierGroup2Response,
    "consolidate_specifiers_group1": GenericSpecifierGroup1Response,
    "consolidate_specifiers_group2": GenericSpecifierGroup2Response,
    "dimension_chunk_decision_tree": PrimaryDimensionChunkResponse,
    "dimension_consolidation": PrimaryDimensionConsolidatedResponse,
    "domain_chunk": DomainChunkResponse,
    "domain_consolidation": DomainConsolidatedResponse,
    "domain_orthogonalize": ReformulatedDomains,
    "idea_extraction": _extraction_model,
}


if __name__ == "__main__":
    render(step=3, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
