from src.config import DATA_DIR
from src.dimorphite_dl import dimorphite_dl as dl

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

def protonate_ligand(sdf_path: str, ph: float = 7.4):
	"""Protonate a ligand SDF file at the given pH."""
	mol = Chem.SDMolSupplier(sdf_path, removeHs=True)[0]
	if mol is None:
		with tempfile.TemporaryDirectory() as tmp_dir:

			subprocess.run(['obabel', '-isdf', sdf_path, '-osdf', '-O', sdf_path, '-p', '7.4'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
			mol = Chem.SDMolSupplier(sdf_path, removeHs=True)[0]
			if mol is None:

				print('Loading as pdb')
				subprocess.run(['obabel', '-isdf', sdf_path, '-opdb', '-O', f'{tmp_dir}/temp.pdb', '--no-connect'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
				mol = Chem.MolFromPDBFile(f'{tmp_dir}/temp.pdb', removeHs=False, sanitize=False)
				if mol is None:
					return

				for atom in mol.GetAtoms():
					atom.SetFormalCharge(0)

				formal_charge = Chem.GetFormalCharge(mol)

				for charge in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
					try:
						rdDetermineBonds.DetermineBonds(mol, charge=charge, useVdw=True)
						break
					except ValueError:
						continue

	# Strip existing Hs, protonate at target pH
	mol_noh = Chem.RemoveAllHs(mol)
	protonated = dl.run_with_mol_list(
		[mol_noh],
		min_ph=ph, max_ph=ph,
		pka_precision=0.0,
		silent=True
	)[0]

	print(Chem.MolToSmiles(mol_noh))

	# Transfer 3D coordinates from original mol via substructure match
	protonated = AllChem.AssignBondOrdersFromTemplate(protonated, mol_noh)

	# Add explicit Hs with 3D coords
	protonated_h = Chem.AddHs(protonated, addCoords=True)
	print(Chem.MolToSmiles(protonated_h))

if __name__ == '__main__':
	for file in os.listdir(f'{DATA_DIR}/CROWN/test_sdf'):
		print(file)
		protonate_ligand(f'{DATA_DIR}/CROWN/test_sdf/{file}')
