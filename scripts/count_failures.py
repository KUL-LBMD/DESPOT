from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter

import pandas as pd
import os

if __name__ == '__main__':
	converter = MolConverter()
	total_atoms, unk_atoms = 0, 0
	atom_list = []

	# Read in DESPOT atom types
	types_df = pd.read_csv(f'{DATA_DIR}/metadata/lig_types.csv')
	lig_types = types_df['atom_type'].tolist()

	base_dir = f'{DATA_DIR}/CASF-2016/coreset'
	for subdir in os.listdir(base_dir):
		lig_df = converter.convert_mol2(f'{base_dir}/{subdir}/{subdir}_ligand.mol2')

		print(lig_df.columns)

		in_set = lig_df[lig_df['lig_type'].isin(lig_types)]
		out_set = lig_df[~lig_df['lig_type'].isin(lig_types)]

		total_atoms += len(lig_df)
		unk_atoms += len(out_set)
		atom_list = atom_list + out_set['lig_type'].tolist()

		print(subdir)

	print(f'Unknown atoms: {unk_atoms} / {total_atoms}')
	print(atom_list)
	print(set(atom_list))
