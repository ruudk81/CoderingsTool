"""
Centralized test data configuration for all pipeline steps.

Change these values once, and all run_[step].py files will use them.

Usage:
    from pipeline.test_data import TEST_DATA
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestDataConfig:
    """Configuration for the test dataset used in steps."""
    
    #filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    #id_column = "DLNMID"
    #var_name = "Q20"
    #sample_size = 500

    #filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    #id_column = "DLNMID"
    #var_name = "Q20"
    #sample_size = 500

    filename = "M000000 Associatiemonitor Merk X net databestand.sav"
    id_column = "DLNMID"
    var_name = "Qd1_combined"
    sample_size = 2000 

    #filename = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
    #id_column = "DLNMID"
    #var_name = "Q15"
    #sample_size = 2000

    # filename = "M250127 Flitspeiling NAVOtop 0meting_153832.sav"
    # id_column = "DLNMID"
    # var_name = "Q10"
    # sample_size = 50


# Singleton instance - import this in run file
TEST_DATA = TestDataConfig()
