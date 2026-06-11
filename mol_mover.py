import pandas as pd
import shutil

from src.config import DATA_DIR

df = pd.read_csv('hiqbind_train.csv')
basename_list = df['basename'].tolist()

for i, basename in enumerate(basename_list):
	shutil.copy(f'{DATA_DIR}/HiQBind/processed_mol2/receptor/{basename}.mol2', f'{DATA_DIR}/HiQBind_train/processed_mol2/receptor/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/HiQBind/processed_mol2/ligand/{basename}.mol2', f'{DATA_DIR}/HiQBind_train/processed_mol2/ligand/{basename}.mol2')
	print(i)
