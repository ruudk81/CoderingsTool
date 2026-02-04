"""
Centralized test data configuration for all experiments.

Change these values once, and all run_experiment.py files will use them.

Usage:
    from experiments.test_data import TEST_DATA
    # or when running from experiments folder:
    from test_data import TEST_DATA
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestDataConfig:
    """Configuration for the test dataset used in experiments."""
    #filename: str = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    #id_column: str = "DLNMID"
    #var_name: str = "Q20"
    #sample_size: Optional[int] = 500

    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column = "DLNMID"
    var_name = "Q20"
    sample_size = 500

    #filename = "M000000 Associatiemonitor Merk X net databestand.sav"
    #id_column = "DLNMID"
    #var_name = "Qd1_combined"
    #sample_size = 2000 

    #filename = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
    #id_column = "DLNMID"
    #var_name = "Q15"
    #sample_size = 50

    # filename = "M250127 Flitspeiling NAVOtop 0meting_153832.sav"
    # id_column = "DLNMID"
    # var_name = "Q10"
    # sample_size = 50


# Singleton instance - import this in run_experiment.py
TEST_DATA = TestDataConfig()
