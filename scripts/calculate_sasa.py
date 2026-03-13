import numpy as np
import pandas as pd
import os
from biopandas.mol2 import PandasMol2
import freesasa
from joblib import Parallel, delayed

from src.config import DATA_DIR

NUM_CORES = 64
DATABASE = 'CROWN_min'
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'S': 1.80, 'P': 1.80, 'F': 1.47, 'CL': 1.75,
    'BR': 1.85, 'I':  1.98, 'FE': 1.80, 'ZN': 1.39,
    'MG': 1.73, 'CA': 1.97, 'MN': 1.61, 'CU': 1.40,
}

class SASA_calculator:
	def __init__(self):
		pass

	def load_mol2(self, path):
		"""
		Returns (coords, radii) from a mol2 file.
		"""

		df = PandasMol2().read_mol2(path).df
		coords = df[['x', 'y', 'z']].values

		atom_types = df['atom_type'].tolist()
		elements = [x.split('.')[0].upper() for x in atom_types]
		radii = np.array([VDW_RADII.get(x, 1.70) for x in elements])

		return coords, radii

	def process_file(self, basename):
		"""
		Calculates SASA difference between ligand in vacuum and ligand in receptor

		Parameters
		----------
		basename [str]: PLI complex identifier

		Returns
		-------
		results [Dict]:
			- basename [str]
			- sasa_free [float]
			- sasa_bound [float]
		"""

		try:

			# Load files
			lig_coords, lig_radii = self.load_mol2(DATA_DIR / DATABASE / 'processed_mol2' / 'ligand' / f'{basename}.mol2')
			receptor_coords, receptor_radii = self.load_mol2(DATA_DIR / DATABASE / 'processed_mol2' / 'receptor' / f'{basename}.mol2')
			complex_coords = np.concatenate([receptor_coords, lig_coords], axis = 0)
			complex_radii = np.concatenate([receptor_radii, lig_radii])

			# Free ligand
			sasa_free_sum = freesasa.calcCoord(lig_coords.flatten(), lig_radii).totalArea()

			# Ligand in complex
			sasa_bound = freesasa.calcCoord(complex_coords.flatten(), complex_radii)
			n_rec = len(receptor_coords)
			n_lig = len(lig_coords)
			sasa_bound_sum = sum(sasa_bound.atomArea(i) for i in range(n_rec, n_rec + n_lig))

			delta_sasa = sasa_free_sum - sasa_bound_sum
			sasa_ratio = delta_sasa / sasa_free_sum

			print(sasa_ratio)

			return {'basename': basename, 'sasa_free': sasa_free_sum, 'sasa_bound': sasa_bound_sum, 'delta_sasa': delta_sasa, 'sasa_ratio': sasa_ratio}

		except Exception as e:
			return {'basename': basename, 'sasa_free': None, 'sasa_bound': None, 'delta_sasa': None, 'sasa_ratio': None}

	def process_database(self):
		basename_list = [x[:-5] for x in os.listdir(DATA_DIR / DATABASE / 'processed_mol2' / 'receptor')]
		results = Parallel(n_jobs = NUM_CORES, verbose = 10)(delayed(self.process_file)(basename) for basename in basename_list)
		output_df = pd.DataFrame(results)
		output_df.to_csv(f'{DATABASE.lower()}_sasa.csv', index = False, float_format = '%.4f')

if __name__ == '__main__':
	sasa_calculator = SASA_calculator()
	sasa_calculator.process_database()

