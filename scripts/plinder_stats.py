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

	df = pd.read_parquet(SOURCE_DB_PATH / 'index' / 'annotation_table.parquet')
	df['basename'] = df['system_id'].str.lower() + '_' + df['ligand_instance_chain'].str.lower()
	dir_list = os.listdir(SOURCE_DB_PATH / 'systems')

	

	subset = df[
		(df['ligand_is_ion'] == False)
		& (df['ligand_is_artifact'] == False)
		& (df['ligand_is_covalent'] == False)
		& (df['ligand_num_unresolved_heavy_atoms'] == 0)
		& (df['ligand_num_neighboring_ppi_atoms_within_8A_of_gap'] == 0) ### No missing protein atoms close to pocket
		& (df['entry_determination_method'] == 'X-RAY DIFFRACTION')
		& (df['entry_resolution'] <= 3.0)
		& ((df['system_ligand_validation_average_rsr'] < 0.3) | (df['system_ligand_validation_average_rsr'].isna()))
		& ((df['system_ligand_validation_average_rscc'] > 0.8) | (df['system_ligand_validation_average_rscc'].isna()))
		& (df['system_id'].isin(dir_list))
		]

	columns_to_select = ['basename', 'system_id', 'ligand_instance_chain', 'entry_resolution', 'system_ligand_validation_average_rsr', 'system_ligand_validation_average_rscc',
		'system_pocket_UniProt', 'system_pocket_CATH', 'ligand_unique_ccd_code', 'ligand_rdkit_canonical_smiles']
	filtered_subset = subset[columns_to_select].copy()

	# Save metadata
	filtered_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'structure_filter_pass.csv', index = False)

	return filtered_subset
