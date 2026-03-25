"""
Experimental Configuration for Step 8: Code Assigner

Purpose: Experiment with configuration changes without affecting production.
"""

from dataclasses import dataclass


@dataclass
class SimilarityRoutingConfig:
    """Configuration for semantic similarity-based code routing.

    routing_mode controls how ideas are matched to codes:
    - "partition": Current behavior — route by concept_type, show only partition codes
    - "similarity": Cosine similarity selects top-K codes from entire codebook
    - "hybrid": Similarity-based, but boost same-partition codes
    """
    routing_mode: str = "similarity"
    top_k: int = 8                        # max codes to present to LLM
    min_codes: int = 5                    # minimum codes (dropoff mode)
    max_codes: int = 10                   # maximum codes (dropoff mode)
    similarity_floor: float = 0.20        # min cosine sim to include a code
    dropoff_ratio: float = 0.80           # include codes within 80% of best sim
    code_embedding_text: str = "rich"     # "simple" (code+def) or "rich" (+boundary+signals)
    idea_embedding_field: str = "ladder"  # which idea embedding to use for matching
    partition_boost: float = 0.15         # hybrid mode: additive boost for same-partition codes
