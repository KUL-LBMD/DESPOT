import os
import pandas as pd

from src.config import DATA_DIR

df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'CROWN_full.csv')
basename_list = df['basename'].tolist()

rows_to_drop = []

for idx, basename in enumerate(basename_list):
	file_paths = [f'{DATA_DIR}/CROWN/processed_mol2/receptor/{basename}.mol2', f'{DATA_DIR}/CROWN/processed_mol2/ligand/{basename}.mol2',
		f'{DATA_DIR}/CROWN_min/processed_mol2/receptor/{basename}.mol2', f'{DATA_DIR}/CROWN_min/processed_mol2/ligand/{basename}.mol2']

	is_incomplete = any(not os.path.exists(fp) or os.path.getsize(fp) == 0 for fp in file_paths)

	if is_incomplete:
		for fp in file_paths:
			if os.path.exists(fp):
				os.remove(fp)
		rows_to_drop.append(idx)

	print(idx)

df.drop(index=rows_to_drop, inplace=True)
df.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'CROWN_full.csv')
