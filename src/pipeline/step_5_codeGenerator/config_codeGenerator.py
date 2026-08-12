"""
Configuration for Code Generator (P8-P9).

Pipeline: code generation from attributes → codebook consolidation.
"""

from dataclasses import dataclass
from config import get_step_model


@dataclass
class CodebookConfig:
    """Configuration for Codebook Generation (P8-P9)."""

    # LLM settings (derived from MODEL_FAMILY toggle)
    model_p8: str = get_step_model("codegen_p8")  # P8: Code Generation from Attributes
    model_p9: str = get_step_model("codegen_p9")  # P9: Codebook Consolidation
    model_relations: str = get_step_model("codegen_relations")  # relations between attributes
    model_umbrella_merge: str = get_step_model("codegen_umbrella_merge")  # consolidate umbrella names
    model_writer: str = get_step_model("codegen_writer")  # codebook writing from clusters
    model_mece_detect: str = get_step_model("codegen_mece_detect")  # MECE pass A: overlap detection
    model_mece_probe: str = get_step_model("codegen_mece_probe")  # MECE pass B: blind assignment probe
    temperature_p8: float = 0.3
    temperature_p9: float = 0.0
    temperature_relations: float = 0.0
    temperature_umbrella_merge: float = 0.0
    temperature_writer: float = 0.3
    temperature_mece_detect: float = 0.0
    temperature_mece_probe: float = 0.0

    # P8: Code Generation from Attributes (per-domain)
    max_tokens_code_from_attributes: int = 16000

    # P9: Codebook Consolidation (cross-domain review)
    max_tokens_codebook_consolidation: int = 16000

    # Relations: one cross-attribute call, output scales with attribute count
    max_tokens_relations: int = 16000

    # Umbrella merge: one cross-umbrella call, consolidates step 2's names before pooling
    max_tokens_umbrella_merge: int = 8000

    # Writer: one cross-code call, output scales with the fixed number of codes
    max_tokens_writer: int = 16000

    # MECE pass A: one cross-code call, output scales with the code count
    max_tokens_mece_detect: int = 16000
    # MECE pass B: one call per candidate pair, output scales with the probe size
    # (mece_probe_ideas_per_code per side, plus a scratchpad)
    max_tokens_mece_probe: int = 4000
    # MECE: repeat pass A + pass B until a round merges nothing, capped here —
    # merging changes the set, so a later round can surface overlaps an earlier
    # round couldn't see yet.
    mece_max_rounds: int = 3
    # MECE pass B: up to this many real ideas per code, pooled and shown blind
    # for the other side to be assigned against.
    mece_probe_ideas_per_code: int = 8
    # MECE pass B: accuracy at or below this means the pair is not reliably
    # codeable apart -> merge. Chance level on a two-way choice is 0.50.
    mece_separability_threshold: float = 0.70

    # Embedding-based representative samples
    code_source: str = "instance_interpretation"  # Text format for embedding: idea, instance, instance_interpretation, full_abstraction_ladder
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    embedding_max_concurrent: int = 5
    max_representative_samples: int = 3  # Max samples per attribute per valence group

    # Output
    verbose: bool = True

    # Prevalentiedrempel: een concept krijgt een eigen code als het door minstens
    # dit aandeel van de respondenten wordt genoemd. Let op: deze drempel en het
    # Overig-plafond van 10% bewegen tegen elkaar in — hoger hier betekent meer
    # Overig. Dit regelt zeldzaamheid, niet duplicatie — MECE-afdwinging (mece.py)
    # is het mechanisme tegen twee codes die hetzelfde concept dekken; de twee
    # mogen niet met elkaar verward worden (2026-08-12: t_keep_share stond hier
    # tijdelijk op 0.05 als poging om overlap via de drempel op te lossen — dat
    # verwijderde elke negatieve code, en loste de overlap niet op).
    t_keep_share: float = 0.01
    t_keep_min_respondents: int = 3
