from src.CROWN.structure_filter import filter_structures
from src.CROWN.structure_fixer import ComplexFixer
from src.CROWN.pli_filter import PLI_Filter
import pandas as pd

# Set some variables for running the scripts
NUM_CORES = 16
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

if __name__ == '__main__':
	main()
