import pandas as pd
import os

# Set some variables for running the scripts
NUM_CORES = 64

# Restrict sqm/antechamber to 1 thread per worker
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
os.environ["OPENMM_CPU_THREADS"] = "1"

from src.CROWN.structure_fixer_full import ComplexFixer
from src.config import DATA_DIR, SOURCE_DB_PATH

df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'rsr_output.csv')
system_list = os.listdir('/media/drives/drive3/robin/plinder/2024-06/v2/systems')
subset = df[df['system_id'].isin(system_list)]
print(len(subset))

complex_fixer = ComplexFixer(subset)
complex_fixer.wrapper(NUM_CORES)
