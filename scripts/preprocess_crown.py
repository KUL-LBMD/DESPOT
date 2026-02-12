from src.CROWN.structure_filter import filter_structures
from src.CROWN.structure_fixer import ComplexFixer
from src.CROWN.pli_filter import PLI_Filter
from src.CROWN.system_fixer import split_system
from src.CROWN.structure_refiner import refine_system
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

	# Step 1: Dataframe subset from original PLInder parquet file
	initial_subset = filter_structures()
	print('Step 1 done')

	# Step 2: fix initial mmCIF structures
	#complex_fixer = ComplexFixer(initial_subset)
	#complex_fixer.wrapper(NUM_CORES)
	#print('Step 2 done')

	# Step 3: PLI filter
	pli_filter = PLI_Filter(initial_subset)
	pli_filter.wrapper(MAXDEV_THRESHOLD, MAX_COUNT)
	print('Step 3 done')

	# Step 4: make fixed systems directory to work with later
	#os.makedirs(DATA_DIR / 'CROWN' / 'systems', exist_ok = True)

	df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'pli_filter_pass.csv')
	Parallel(n_jobs = 32, verbose = 10)(delayed(split_system)(
			pdb_path = Path(f'{DATA_DIR}/CROWN/raw_pdb/{row.basename}.pdb'),
			sdf_dir = Path(f'{SOURCE_DB_PATH}/systems/{row.system_id}/ligand_files'),
			out_dir = Path(f'{DATA_DIR}/CROWN/systems/{row.basename}')
		) for row in df.itertuples())

	# Step 5: Protonate and energy-minimize
	Parallel(n_jobs = 64, verbose = 10)(delayed(refine_system)(row.basename) for row in df.itertuples())

if __name__ == '__main__':
	main()
