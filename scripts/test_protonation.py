from src.CROWN.structure_filter import filter_structures
from src.CROWN.structure_fixer import ComplexFixer
from src.CROWN.pli_filter import PLI_Filter
from src.CROWN.system_fixer import split_system
from src.CROWN.structure_refiner_test import refine_system
from src.config import DATA_DIR, SOURCE_DB_PATH

import pandas as pd
import os
from pathlib import Path
from joblib import Parallel, delayed

# Set some variables for running the scripts
NUM_CORES = 32
MAXDEV_THRESHOLD = 0.1
MAX_COUNT = 500

def main():
	"""
	Preprocessing pipeline from PLInder to semi-processed CROWN.
	This workflow comprises the following steps:

	1. Structure quality filter: select high-quality crystal structures and avoid artifacts, ions and covalent ligands
	2. Structure fixer: resolve ambiguous occupancies, missing bonds and steric clashes
	3. PLI quality filter: check quality of the PLI and remove redundant entries

	Step 4, which is structure refinement, is done separately using MOE QuickPrep.
	"""

	# Step 5: Protonate and energy-minimize
	refine_system('105m__1__1.a__1.c_1.d_1.c')

if __name__ == '__main__':
	main()
