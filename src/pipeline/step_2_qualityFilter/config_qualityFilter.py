"""Step 2 quality filter configuration."""

from dataclasses import dataclass
from config import get_step_model


@dataclass
class QualityFilterConfig:
    temperature: float = 0.0
    max_tokens: int = 4000
    retries: int = 3
    max_filter_examples: int = 5
    model: str = get_step_model("quality_filter")


DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()
