import pandas as pd
import os
from pathlib import Path
import multiprocessing
import time
from joblib import Parallel, delayed

# Set some variables for running the scripts
NUM_CORES = 8
MAXDEV_THRESHOLD = 0.1
MAX_COUNT = 500
REFINE_TIMEOUT = 600  # 10 minutes per complex

# Restrict sqm/antechamber to 1 thread per worker
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
os.environ["OPENMM_CPU_THREADS"] = "1"

from src.CROWN.structure_filter import filter_structures
from src.CROWN.structure_fixer import ComplexFixer
from src.CROWN.pli_filter import PLI_Filter
from src.CROWN.system_fixer import split_system
from src.CROWN.structure_refiner import refine_system
from src.config import DATA_DIR, SOURCE_DB_PATH

class _RefinementTimeout(Exception):
	pass

def _timeout_handler(signum, frame):
	raise _RefinementTimeout()

def refine_with_timeout(basename, timeout=REFINE_TIMEOUT):
	"""
	Run refine_system with a hard time limit using SIGALRM.
	Works inside joblib/loky workers on Linux because each
	worker executes tasks in its main thread.
	"""
	old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
	signal.alarm(timeout)
	try:
		result = refine_system(basename)
		signal.alarm(0)
		return result
	except _RefinementTimeout:
		print(f"[TIMEOUT] {basename} exceeded {timeout}s — skipping.")
		return None
	except Exception:
		signal.alarm(0)
		raise
	finally:
		signal.signal(signal.SIGALRM, old_handler)

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
	#initial_subset = filter_structures()
	#print('Step 1 done')

	df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'new_lines.csv')

	# Step 2: fix initial mmCIF structures
	complex_fixer = ComplexFixer(df)
	complex_fixer.wrapper(8)
	print('Step 2 done')

	# Step 3: PLI filter
	pli_filter = PLI_Filter(df)
	pli_filter.wrapper(MAXDEV_THRESHOLD, MAX_COUNT)
	print('Step 3 done')

	# Step 4: make fixed systems directory to work with later
	df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'pli_filter_pass.csv')
	os.makedirs(DATA_DIR / 'CROWN' / 'systems', exist_ok = True)
	os.makedirs(DATA_DIR / 'CROWN' / 'processed_systems', exist_ok = True)

	Parallel(n_jobs = 1, verbose = 10)(delayed(split_system)(
			pdb_path = Path(f'{DATA_DIR}/CROWN/raw_pdb/{row.basename}.pdb'),
			sdf_dir = Path(f'{SOURCE_DB_PATH}/systems/{row.system_id}/ligand_files'),
			out_dir = Path(f'{DATA_DIR}/CROWN/systems/{row.basename}')
		) for row in df.itertuples())

	# Step 5: Protonate and energy-minimize
	basename_list = os.listdir(DATA_DIR / 'CROWN' / 'processed_systems')
	subset = df[~df['basename'].isin(basename_list)].sample(frac = 1)

	tuple_list = Parallel(n_jobs = 8, verbose = 10)(delayed(refine_with_timeout)(row.basename) for row in subset.itertuples())

	rmsd_df = pd.DataFrame(tuple_list, columns = ['dirname', 'rmsd_nonmobile', 'rmsd_pocket', 'rmsd_ligand']).dropna()
	rmsd_df.to_csv('mobile_rmsd.csv', index = False)

if __name__ == '__main__':
	main()
