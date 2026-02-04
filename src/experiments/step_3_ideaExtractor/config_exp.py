"""
Experimental Configuration for Step 3: Idea Extractor

Purpose: Experiment with configuration changes without affecting production.

To use:
1. Modify settings below as needed
2. Import from this file in run_experiment.py when USE_EXPERIMENTAL = True

Production configs are imported from config_ideaExtractor for reference.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass
from config_ideaExtractor import (
    SegmentationConfig,
    DEFAULT_SEGMENTATION_CONFIG,
    TokenHistoryConfig,
    DEFAULT_TOKEN_HISTORY_CONFIG,
    TiktokenOffsetConfig,
    DEFAULT_TIKTOKEN_OFFSET_CONFIG,
    TimeoutConfig,
    DEFAULT_TIMEOUT_CONFIG,
    ReportingConfig,
    DEFAULT_REPORTING_CONFIG,
    BootstrapConfig,
    DEFAULT_BOOTSTRAP_CONFIG,
    PIDControllerConfig,
    DEFAULT_PID_CONTROLLER_CONFIG,
    TPMTrackingConfig,
    DEFAULT_TPM_TRACKING_CONFIG,
    ThroughputConfig,
    DEFAULT_THROUGHPUT_CONFIG,
    SpecifierConfig,
    DEFAULT_SPECIFIER_CONFIG,
)


# =============================================================================
# EXPERIMENTAL OVERRIDES
# =============================================================================
# Modify these to experiment with different settings.
# Production defaults are inherited from the parent classes.

@dataclass
class SegmentationConfigExp(SegmentationConfig):
    """Experimental segmentation config - override fields as needed."""
    pass


# =============================================================================
# DEFAULT EXPERIMENTAL INSTANCES
# =============================================================================

DEFAULT_SEGMENTATION_CONFIG_EXP = SegmentationConfigExp()
