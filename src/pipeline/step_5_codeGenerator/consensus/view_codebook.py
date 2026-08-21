#%%
"""
Bekijk het codeboek uit de consensus-keten.

De cachesleutel `mece_codes` is gedeeld met de productiepijplijn. Dit venster
toont wat er MOMENTEEL onder die sleutel staat — dat is de laatst gedraaide
keten (productie of consensus), niet per se deze.

Voor details over de uitvoer, zie ../view_codebook.py.

Gebruik:
    cd src && python -m pipeline.step_5_codeGenerator.consensus.view_codebook
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_dir))

# Import and call the production module's main function
from pipeline.step_5_codeGenerator import view_codebook as production_view


if __name__ == "__main__":
    production_view.main()

# %%
