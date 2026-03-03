from src.config import DATA_DIR, SOURCE_DB_PATH

import pandas as pd
import os

def filter_structures():
	"""
	Read PLInder metadata and select only relevant PLI systems

	Returns
	-------

	filtered_subset [pd.DataFrame]:
		- basename [str]: PLI system identifier
		- entry_resolution [float]: resolution of crystal structure
		- system_ligand_validation_average_rsr [float]: RSR of ligand
		- system_ligand_validation_average_rscc [float]: RSCC of ligand
		- system_pocket_UniProt [str]: UniProt ID of receptor
		- system_pocket_CATH [str]: CATH ID of receptor
		- ligand_unique_ccd_code [str]: CCD code of ligand
		- ligand_rdkit_canonical_smiles [str]: Canonical SMILES representation of ligand

	"""

	df1 = pd.read_parquet(SOURCE_DB_PATH / 'index' / 'annotation_table.parquet')
	df2 = pd.read_csv(DATA_DIR / 'CROWN' / 'metadata' / 'rsr_output.csv')
	df1['system_ligand_validation_average_rsr'] = df2['system_ligand_validation_average_rsr']
	df1['system_ligand_validation_average_rscc'] = df2['system_ligand_validation_average_rscc']

	df1.to_parquet(SOURCE_DB_PATH / 'index' / 'annotation_table.parquet', index = False)


if __name__ == '__main__':
	filter_structures()
