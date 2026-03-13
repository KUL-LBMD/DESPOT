import os
import shutil
import pandas as pd

from src.config import DATA_DIR

df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'CROWN_overfit.csv')
basename_list = df['basename'].tolist()

for i, basename in enumerate(basename_list):
	print(i)
	shutil.copy(f'{DATA_DIR}/CROWN/processed_mol2/receptor/{basename}.mol2', f'{DATA_DIR}/CROWN_full/processed_mol2/receptor/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/CROWN/processed_mol2/ligand/{basename}.mol2', f'{DATA_DIR}/CROWN_full/processed_mol2/ligand/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/CROWN_min/processed_mol2/receptor/{basename}.mol2', f'{DATA_DIR}/CROWN_full_min/processed_mol2/receptor/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/CROWN_min/processed_mol2/ligand/{basename}.mol2', f'{DATA_DIR}/CROWN_full_min/processed_mol2/ligand/{basename}.mol2')

