"""
Centralized test data configuration for all development.

Change these values once, and all run_experiment.py files will use them.

Usage:
    from development.test_data import TEST_DATA
    # or when running from development folder:
    from test_data import TEST_DATA
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestDataConfig:
    """Configuration for the test dataset used in development."""
    
    #filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    #id_column = "DLNMID"
    #var_name = "Q20"
    #sample_size = 500

    #filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    #id_column = "DLNMID"
    #var_name = "Q20"
    #sample_size = 500

    #filename = "M250480 Associatiemonitor ASN Bank net databestand.sav"
    #id_column = "DLNMID"
    #var_name = "Qd1_combined"
    #sample_size = 2000 

    filename = "M250219 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
    id_column = "DLNMID"
    var_name = "Q15"
    sample_size = 2000

    # filename = "M250127 Flitspeiling NAVOtop 0meting_153832.sav"
    # id_column = "DLNMID"
    # var_name = "Q10"
    # sample_size = 50


# Singleton instance - import this in run_experiment.py
TEST_DATA = TestDataConfig()
