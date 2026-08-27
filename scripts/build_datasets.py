from src.config import DATA_DIR

import pandas as pd
from tqdm import tqdm
import os

CROWN_DIR = '/media/drives/drive3/robin/CROWN/data/mol2_files'

dir_list = [f'{DATA_DIR}/CROWN_xtal', f'{DATA_DIR}/CROWN', f'{DATA_DIR}/HiQBind_xtal', f'{DATA_DIR}/HiQBind']
for dirname in dir_list:
	os.makedirs(dirname, exist_ok = True)
	os.makedirs(f'{dirname}/processed_mol2', exist_ok = True)
	os.makedirs(f'{dirname}/processed_mol2/receptor', exist_ok = True)
	os.makedirs(f'{dirname}/processed_mol2/ligand', exist_ok = True)

df1 = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_train.parquet')
basename_list = df1['basename'].tolist()

for basename in tqdm(basename_list, desc = 'CROWN'):
	os.symlink(f'{CROWN_DIR}/{basename}/receptor.mol2', f'{DATA_DIR}/CROWN_xtal/processed_mol2/receptor/{basename}.mol2')
	os.symlink(f'{CROWN_DIR}/{basename}/receptor_minimized.mol2', f'{DATA_DIR}/CROWN/processed_mol2/receptor/{basename}.mol2')
	os.symlink(f'{CROWN_DIR}/{basename}/ligand.mol2', f'{DATA_DIR}/CROWN_xtal/processed_mol2/ligand/{basename}.mol2')
	os.symlink(f'{CROWN_DIR}/{basename}/ligand_minimized.mol2', f'{DATA_DIR}/CROWN/processed_mol2/ligand/{basename}.mol2')

df2 = pd.read_parquet(f'{DATA_DIR}/metadata/hiqbind_train.parquet')
basename_list = df2['basename'].tolist()

for basename in tqdm(basename_list, desc = 'HiQBind'):
	os.symlink(f'{CROWN_DIR}/{basename}/receptor.mol2', f'{DATA_DIR}/HiQBind_xtal/processed_mol2/receptor/{basename}.mol2')
	os.symlink(f'{CROWN_DIR}/{basename}/receptor_minimized.mol2', f'{DATA_DIR}/HiQBind/processed_mol2/receptor/{basename}.mol2')
	os.symlink(f'{CROWN_DIR}/{basename}/ligand.mol2', f'{DATA_DIR}/HiQBind_xtal/processed_mol2/ligand/{basename}.mol2')
	os.symlink(f'{CROWN_DIR}/{basename}/ligand_minimized.mol2', f'{DATA_DIR}/HiQBind/processed_mol2/ligand/{basename}.mol2')
