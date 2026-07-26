"""Input loading for the step-5 experiment — explicit params, no module globals.

Reuses the baseline's cache conventions: corrected taxonomy preferred (via
_project_corrected), idea embeddings reused from a valid baseline mece_codes
cache when present, else computed via SharedEmbedder (same code_source).
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from models import CodingResultsCache, TaxonomyResultsCache, TaxonomyClassifiedModel
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_5_codeGenerator.run_codeGenerator import _project_corrected, load_extraction_metadata, _extract_metadata_context
from pipeline.step_5_codeGenerator.codebook_verifier import (
    collect_idea_assignments, collect_attribute_valence,
)


@dataclass
class ExperimentInputs:
    partition_results: dict
    idea_assignments: Dict[str, str]
    attr_valence: Dict[str, Dict[str, int]]
    idea_texts: Dict[str, str]
    idea_embeddings: Dict[str, List[float]]
    language: str
    variable_key: str
    survey_question: str


def _load_language_and_question(
    cm: CacheManager,
    filename: str,
    variable_key: str,
) -> Tuple[str, str]:
    """Load language and survey question from extraction metadata cache.

    Mirrors the baseline step 5 pattern: load ExtractionMetadata from cache,
    extract language (default 'Dutch') and survey_question (var_lab, default '').

    Returns: (language, survey_question)
    """
    meta = load_extraction_metadata(
        filename=filename,
        variable_key=variable_key,
        variable=None,  # not used when variable_key is provided
        sample_size=None,  # not used when variable_key is provided
    )
    survey_question, language, _, _, _ = _extract_metadata_context(meta)
    return language, survey_question


def load_inputs(filename: str, var_name: str, sample_size) -> ExperimentInputs:
    vk = generate_enhanced_variable_key(
        selected_variables=[var_name], is_merged=False, sample_size=sample_size)
    cm = CacheManager()

    tax = cm.load_metadata_from_cache(filename, "taxonomy", vk, TaxonomyResultsCache)
    if tax is None:
        raise RuntimeError("no taxonomy cache — run step 4 first")
    pr = tax.partition_results
    if cm.is_metadata_cache_valid(filename, "taxonomy_corrected", vk):
        corrected = cm.load_metadata_from_cache(
            filename, "taxonomy_corrected", vk, TaxonomyResultsCache)
        if corrected is not None:
            pr = _project_corrected(corrected).partition_results

    ideas = (cm.load_from_cache(filename, "taxonomy_classified_corrected", vk, TaxonomyClassifiedModel)
             or cm.load_from_cache(filename, "taxonomy_classified", vk, TaxonomyClassifiedModel))
    idea_texts = {}
    for m in ideas or []:
        for sub in (m.response_ideas or []):
            idea_texts[sub.idea_id] = sub.instance or sub.idea

    embeddings: Dict[str, List[float]] = {}
    if cm.is_metadata_cache_valid(filename, "mece_codes", vk):
        base = cm.load_metadata_from_cache(filename, "mece_codes", vk, CodingResultsCache)
        if base is not None and base.idea_embeddings:
            embeddings = base.idea_embeddings
    # NB: fallback-embedden (SharedEmbedder) komt pas in Task 6 (orchestrator);
    # data_io geeft {} terug als er geen hergebruik mogelijk is.

    meta = _load_language_and_question(cm, filename, vk)
    return ExperimentInputs(
        partition_results=pr,
        idea_assignments=collect_idea_assignments(pr),
        attr_valence=collect_attribute_valence(pr),
        idea_texts=idea_texts,
        idea_embeddings=embeddings,
        language=meta[0],
        variable_key=vk,
        survey_question=meta[1],
    )
