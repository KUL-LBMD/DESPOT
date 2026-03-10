import pandas as pd
import os

from src.config import DATA_DIR

def remove_files(basename_set, path):
	for file in os.listdir(path):
		basename = file[:-5]
		if not basename in basename_set:
			os.remove(f'{path}/{file}')

df = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'CROWN_train.csv')
basename_set = df['basename'].tolist()

remove_files(basename_set, DATA_DIR / 'CROWN' / 'processed_mol2' / 'receptor')
remove_files(basename_set, DATA_DIR / 'CROWN' / 'processed_mol2' / 'ligand')
remove_files(basename_set, DATA_DIR / 'CROWN_min' / 'processed_mol2' / 'receptor')
remove_files(basename_set, DATA_DIR / 'CROWN_min' / 'processed_mol2' / 'ligand')
