import pandas as pd
import numpy as np
import os
from biopandas.mol2 import PandasMol2
from scipy.spatial import KDTree
from joblib import Parallel, delayed

from src.config import DATA_DIR

def calc_rmsd(pos_ref, pos_target):
	"""
	Computes RMSD (in angstrom) between 2 coordinate arrays.

	Parameters
	----------

	pos_ref [N, 3]: reference coordinates
	pos_target [N, 3]: updated coordinates
	atom_indices [L]: subset of atom indices to use

	Returns
	-------

	rmsd [float]
	"""

	diff = pos_ref - pos_target
	return float(np.sqrt((diff**2).sum(axis=1).mean()))

def load_complex(mode, subdir, pocket_mask = None):

	receptor_df = PandasMol2().read_mol2(f'{DATA_DIR}/{mode}/processed_mol2/receptor/{subdir}.mol2').df
	lig_df = PandasMol2().read_mol2(f'{DATA_DIR}/{mode}/processed_mol2/ligand/{subdir}.mol2').df
	lig_df['subst_name'] = 'LIG1'
	df = pd.concat([receptor_df, lig_df], axis = 0)

	h_df = df[df['atom_type'] == 'H']
	heavy_df = df[df['atom_type'] != 'H']
	lig_df = heavy_df[heavy_df['subst_name'] == 'LIG1']
	receptor_df = heavy_df[heavy_df['subst_name'] != 'LIG1']

	h_coords = h_df[['x', 'y', 'z']].values
	lig_coords = lig_df[['x', 'y', 'z']].values
	receptor_coords = receptor_df[['x', 'y', 'z']].values

	lig_tree = KDTree(lig_coords)
	if pocket_mask is None:
		neighbors = lig_tree.query_ball_point(receptor_coords, r=6.0)
		pocket_mask = np.array([len(n) > 0 for n in neighbors])

	pocket_coords = receptor_coords[pocket_mask]
	other_coords = receptor_coords[~pocket_mask]

	return h_coords, lig_coords, pocket_coords, other_coords, pocket_mask

def process_file(subdir):
	"""
	Calculate RMSD for ligand, pocket, other and H
	"""

	if os.path.getsize(f'{DATA_DIR}/CROWN/processed_mol2/receptor/{subdir}.mol2') == 0:
		return {'basename': subdir, 'H': None, 'LIG': None, 'Pocket': None, 'Scaffold': None}

	h_og, lig_og, pocket_og, other_og, pocket_mask = load_complex('CROWN', subdir)
	h_min, lig_min, pocket_min, other_min, _ = load_complex('CROWN_min', subdir, pocket_mask)

	h_rmsd = calc_rmsd(h_og, h_min)
	lig_rmsd = calc_rmsd(lig_og, lig_min)
	pocket_rmsd = calc_rmsd(pocket_og, pocket_min)
	other_rmsd = calc_rmsd(other_og, other_min)

	return {'basename': subdir, 'H': h_rmsd, 'LIG': lig_rmsd, 'Pocket': pocket_rmsd, 'Scaffold': other_rmsd}

if __name__ == '__main__':
	crown_df = pd.read_csv(f'{DATA_DIR}/CROWN/metadata/CROWN_full.csv')
	basename_list = crown_df['basename'].tolist()
	list_of_dicts = Parallel(n_jobs = 64, verbose = 10)(delayed(process_file)(subdir) for subdir in basename_list)
	df = pd.DataFrame(list_of_dicts)
	df.to_csv('rmsd_df.csv', index = False)

