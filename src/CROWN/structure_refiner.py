from src.config import DATA_DIR
from src.dimorphite_dl import dimorphite_dl as dl

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from scipy.spatial import KDTree
from pdbfixer import PDBFixer
from openmmforcefields.generators import SystemGenerator
from openmm.app import PDBFile, Modeller, ForceField, Simulation
from openff.toolkit import Molecule
from openff.units import unit as openff_unit
from openmm import CustomExternalForce, LangevinMiddleIntegrator, unit, Platform
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
MOBILE_RADIUS = 0.8  # Distance in nm (8 Å = 0.8 nm)
FIX_STRENGTH = 1000000.0  # kJ/mol/nm² - very high to effectively freeze atoms
TETHER_STRENGTH = 10 # kcal/(mol*A^2). Default parameter in MOE
TETHER_FLATBOTTOM = 0.25 # Within 0.25 A radius, atoms feel no tethering force
MINIMIZATION_STEPS = 5000
ENERGY_REPORT_INTERVAL = 50
TEMPERATURE = 300  # Kelvin
TIMESTEP = 0.002  # picoseconds
PH = 7.4

STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYM', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 
    'TRP', 'TYR', 'VAL', 'HIE', 'HIP', 'HID', 'HSD', 'HSE', 'HSP',
    'ACE', 'NME'
}

WATER_NAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'OPC'}
METALLOCOFACTORS_AMBER = {'HEM', 'SF4', 'MGD'}
METALLOCOFACTORS_CHARMM = {'B12', 'COB', 'CLA', 'BCL', 'CHL', 'F43'}

# ============================================================================

def resolve_duplicate_templates(ff):
    """
    Remove duplicate residue templates from an OpenMM ForceField object.
    """

    name_to_keys = defaultdict(list)
    for key, template in ff._templates.items():
        name_to_keys[template.name].append(key)

    duplicates = {n: k for n, k in name_to_keys.items() if len(k) > 1}

    for res_name, keys in duplicates.items():
        keep = keys[-1]
        for k in [k for k in keys if k != keep]:
            del ff._templates[k]

    return ff

def calc_rmsd(pos_ref, pos_target, atom_indices):
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

	if atom_indices is not None:
		pos_ref = pos_ref[atom_indices]
		pos_target = pos_target[atom_indices]

	diff = pos_ref - pos_target
	return float(np.sqrt((diff**2).sum(axis=1).mean()))

def find_cofactors(pdb_path):
	"""
	Find all metallocofactor entries in a pdb file
	"""

	amber_residues = set()
	charmm_residues = set()

	with open(pdb_path, 'r') as f:
		for line in f:
			if line.startswith(('ATOM', 'HETATM')):
				resname = line[17:20].strip()
				if resname in METALLOCOFACTORS_AMBER:
					amber_residues.add(resname)
				elif resname in METALLOCOFACTORS_CHARMM:
					charmm_residues.add(resname)

	return amber_residues, charmm_residues

def protonate_ligand(sdf_path: str, out_sdf: str, ph: float = 7.4):
	"""Protonate a ligand SDF file at the given pH."""
	mol = Chem.SDMolSupplier(sdf_path, removeHs=True)[0]
	if mol is None:
		# Try out alternate ways of loading molecule
		with tempfile.TemporaryDirectory() as tmp_dir:
			if mol is None:
				subprocess.run(['obabel', '-isdf', sdf_path, '-opdb', '-O', f'{tmp_dir}/temp.pdb'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
				mol = Chem.MolFromPDBFile(f'{tmp_dir}/temp.pdb', removeHs=False, sanitize=False)
				if mol is None:
					return

				# FIX OXYGEN FORMAL CHARGES, used to be problem in NAD+ that was deprotonated.
				for atom in mol.GetAtoms():
					if atom.GetAtomicNum() != 8:         # skip non-O
						continue
					# count DOUBLE bonds; Double-bonded O should have formal charge = 0
					n_double = sum(1 for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
					if n_double:
						atom.SetFormalCharge(0)
						continue

					# Single-bonded O attached to C should have formal charge = -1
					num_neighbors = atom.GetDegree()
					if num_neighbors == 1:
						neighbor = atom.GetNeighbors()[0]
						bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
						if (neighbor.GetAtomicNum() in (6, 15, 16) and bond.GetBondType() == Chem.BondType.SINGLE and atom.GetFormalCharge() == 0):
							atom.SetFormalCharge(-1)

				formal_charge = Chem.GetFormalCharge(mol)

				for charge in [formal_charge] + [formal_charge + d for d in (1, -1, 2, -2, 3, -3)]:
					try:
						rdDetermineBonds.DetermineBonds(mol, charge=formal_charge, covFactor=1.15, useVdw=True) #covFactor was set to 1.15 as BCP would throw errors due to carbons being too close.
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

	# Transfer 3D coordinates from original mol via substructure match
	protonated = AllChem.AssignBondOrdersFromTemplate(protonated, mol_noh)

	# Add explicit Hs with 3D coords
	protonated_h = Chem.AddHs(protonated, addCoords=True)
	formal_charge = Chem.GetFormalCharge(protonated_h)
	logger.info(f'Formal charge: {formal_charge}')

	# Compute MMFF partial charges
	mmff = AllChem.MMFFGetMoleculeProperties(protonated_h)
	charges = [mmff.GetMMFFPartialCharge(i) for i in range(protonated_h.GetNumAtoms())]
	charges_str = " ".join(f"{charge:.6f}" for charge in charges)
	protonated_h.SetProp("atom.dprop.PartialCharge", charges_str)

	writer = Chem.SDWriter(out_sdf)
	writer.write(protonated_h)
	writer.close()
	logger.info(f"Protonated ligand written to {out_sdf}")

def add_bonds(topology, positions, resname_set):
    """Add missing bonds for residues based on interatomic distances."""
    for residue in topology.residues():
        for resname in resname_set:
            if residue.name != resname:
                continue

            print(f'{resname} found')

            atoms = list(residue.atoms())
            pos = np.array([(positions[a.index].x, positions[a.index].y, positions[a.index].z) 
                        for a in atoms])

            # Collect existing bonds to avoid duplicates
            existing_bonds = set()
            for bond in topology.bonds():
                existing_bonds.add((bond[0].index, bond[1].index))
                existing_bonds.add((bond[1].index, bond[0].index))

            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):

                    if (atoms[i].index, atoms[j].index) in existing_bonds:
                        continue

                    dist = np.linalg.norm(pos[i] - pos[j])  # already in nm from OpenMM

                    ei = atoms[i].element.symbol
                    ej = atoms[j].element.symbol

                    if ei == 'H' and ej == 'H':
                        continue

                    if ei == 'H' or ej == 'H':
                        if dist < 0.12:
                            topology.addBond(atoms[i], atoms[j])

                    # Fe-N bonds are ~2.0 Å
                    metal_set = {'Fe', 'Mn', 'Mg', 'Ni', 'Zn', 'Cu'}
                    if ei in metal_set or ej in metal_set:
                        if dist < 0.25:  # 2.5 Å in nm
                            topology.addBond(atoms[i], atoms[j])
                    # C-C, C-N, C-O bonds are ~1.2-1.55 Å
                    else:
                        if dist < 0.18 and not 'H' in (ei, ej):  # 1.8 Å in nm
                            topology.addBond(atoms[i], atoms[j])

def prepare_charmm(pdb_path, resname_set):
	"""
	Prepare modeller and force field list for CHARMM
	"""

	# No capping done for CHARMM, will throw error otherwise
	Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/CROWN/custom_xml/protonation/special_residues_charmm.xml')

	fixer = PDBFixer(pdb_path)
	fixer.findMissingResidues()
	fixer.findMissingAtoms()
	fixer.addMissingAtoms()
	fixer.addMissingHydrogens(PH)

	logging.getLogger("openff").setLevel(logging.ERROR)
	modeller = Modeller(fixer.topology, fixer.positions)
	add_bonds(modeller.topology, modeller.positions, resname_set)

	modeller.addHydrogens(pH=PH)

	forcefield_list = ['charmm36_2024.xml', 'charmm36_2024/water.xml']

	return modeller, forcefield_list

def prepare_amber_special(pdb_path, resname_set):
	"""
	Prepare modeller and force field list for special AMBER residues
	"""

	Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/CROWN/custom_xml/protonation/special_residues_amber.xml')
	fixer = PDBFixer(pdb_path)

	fixer.addMissingHydrogens(PH)

	logging.getLogger("openff").setLevel(logging.ERROR)
	modeller = Modeller(fixer.topology, fixer.positions)

	modeller.addHydrogens(pH=PH)
	add_bonds(modeller.topology, modeller.positions, resname_set)
	forcefield_list = ['amber19/protein.ff19SB.xml', 'amber19/opc3.xml',
		f'{DATA_DIR}/CROWN/custom_xml/forcefield/HEM.xml', f'{DATA_DIR}/CROWN/custom_xml/forcefield/MGD.xml', f'{DATA_DIR}/CROWN/custom_xml/forcefield/SF4.xml']

	return modeller, forcefield_list

def prepare_amber_regular(pdb_path):

	fixer = PDBFixer(pdb_path)
	fixer.addMissingHydrogens(PH)

	logging.getLogger("openff").setLevel(logging.ERROR)
	modeller = Modeller(fixer.topology, fixer.positions)

	forcefield_list = ['amber19/protein.ff19SB.xml', 'amber19/opc3.xml', f'{DATA_DIR}/CROWN/custom_xml/forcefield/HEM.xml', f'{DATA_DIR}/CROWN/custom_xml/forcefield/MGD.xml', f'{DATA_DIR}/CROWN/custom_xml/forcefield/SF4.xml']

	return modeller, forcefield_list

def refine_system(input_dir):
	"""
	Structure refinement workflow:
	1. Protonate ligands and cofactors with dimorphite_dl
	2. Prepare protein-only structure with PDBFixer
	3. Combine full PLI system
	4. Run constrained energy minimization
	"""

	# ====================================================================
	# Step 1: Protonate ligands
	# ====================================================================

	handler = logging.FileHandler(DATA_DIR / 'CROWN' / 'logs' / f'{input_dir}.log')
	handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
	logger.addHandler(handler)

	try:
		with tempfile.TemporaryDirectory() as tmp_dir:
			logger.info(f"Processing {input_dir}")
			for filename in os.listdir(f'{DATA_DIR}/CROWN/systems/{input_dir}'):
				if filename.endswith('.sdf'):
					basename = filename[:-4]
					protonate_ligand(f'{DATA_DIR}/CROWN/systems/{input_dir}/{filename}', f'{tmp_dir}/{basename}_h.sdf', ph = PH)
					if not os.path.isfile(f'{tmp_dir}/{basename}_h.sdf'):
						logger.error(f"Protonation failed for {filename} — aborting refinement.")
						return

			# ====================================================================
			# Step 2: Create protein-only structure and run PDBFixer
			# ====================================================================

			# First check for weird cofactors
			pdb_path = f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_fixed.pdb'
			amber_residues, charmm_residues = find_cofactors(pdb_path)
			if amber_residues and charmm_residues:
				return
			elif charmm_residues:
				modeller, forcefield_list = prepare_charmm(pdb_path, charmm_residues)
			elif amber_residues:
				modeller, forcefield_list = prepare_amber_special(pdb_path, amber_residues)
			else:
				modeller, forcefield_list = prepare_amber_regular(pdb_path)

			# ====================================================================
                	# Step 3: Add ligands back with proper parameters
                	# ====================================================================

			# Add each ligand back to the modeller
			ligand_molecules = []
			ligand_entries = []  # (basename, Molecule, list of atom indices)

			for ligand_file in sorted(os.listdir(tmp_dir)):
				if ligand_file.endswith('_h.sdf'):
					basename = ligand_file.replace('_h.sdf', '')
					ligand_mol = Molecule.from_file(f'{tmp_dir}/{ligand_file}')
					ligand_molecules.append(ligand_mol)

					# Convert OpenFF positions to OpenMM unit system
					ligand_topology = ligand_mol.to_topology().to_openmm()
					ligand_positions = ligand_mol.conformers[0].to_openmm()

					# Record indices before adding (they'll start at current atom count)
					offset = modeller.topology.getNumAtoms()
					n_atoms = ligand_topology.getNumAtoms()
					modeller.add(ligand_topology, ligand_positions)

					if ligand_file == 'lig_fixed_1_h.sdf':
						added_atom_indices = set(range(offset, offset + n_atoms))
						for residue in modeller.topology.residues():
							if any(atom.index in added_atom_indices for atom in residue.atoms()):
								residue.name = 'LIG'

					ligand_entries.append((basename, ligand_mol, list(range(offset, offset + n_atoms))))

			# Merged set of all ligand indices (used for KDTree and restraints)
			all_atoms = list(modeller.topology.atoms())

			ligand_indices = set()
			for _, _, indices in ligand_entries:
				ligand_indices.update(i for i in indices if all_atoms[i].element.symbol != 'H')

			mobile_indices = ligand_indices.copy()
			for residue in modeller.topology.residues():
				if residue.name in {'HEM', 'MGD', 'SF4'}:
					for atom in residue.atoms():
						if atom.element.symbol != 'H':
							mobile_indices.add(atom.index)

			system_generator = SystemGenerator(
				forcefields=forcefield_list, # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
				small_molecule_forcefield='openff-2.2.0',
				molecules=ligand_molecules,
			)

			# Remove duplicate entries: TL1 / Tl, FE2 / FE
			system_generator.forcefield = resolve_duplicate_templates(system_generator.forcefield)

			system = system_generator.create_system(modeller.topology)

			# ====================================================================
			# STEP 4: Identify mobile region around ligands
			# ====================================================================

			# Use KDTree for more efficient spatial lookup
			positions = modeller.positions
			all_positions_nm = np.array([positions[i].value_in_unit(unit.nanometer) for i in range(len(positions))])
			ligand_indices_list = sorted(mobile_indices)
			ligand_positions_nm = all_positions_nm[ligand_indices_list]
			ligand_tree = KDTree(ligand_positions_nm)

			distances, _ = ligand_tree.query(all_positions_nm, k=1)  # nearest ligand atom
			mobile_mask = distances <= MOBILE_RADIUS
			mobile_atoms = {i for i in np.where(mobile_mask)[0].tolist() if all_atoms[i].element.symbol != 'H'}

			# ====================================================================
                	# STEP 5: Add restraints to both mobile and non-mobile atoms
                	# ====================================================================

			nonmobile_restraint = CustomExternalForce("k*r^2; r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)")
			nonmobile_restraint.addGlobalParameter('k', FIX_STRENGTH * unit.kilojoules_per_mole / unit.nanometer**2)
			nonmobile_restraint.addPerParticleParameter("x0")
			nonmobile_restraint.addPerParticleParameter("y0")
			nonmobile_restraint.addPerParticleParameter("z0")

			# Continuously differentiable energy term. Flat-bottom tethering with smoothstep function
			# Energy, force and second derivative at r=0.25 are 0.
			# Energy and force at 1.25 are 1, second derivative is 0.
			mobile_restraint = CustomExternalForce('w*('
				'step(r-u)*(1-step(r-(d+u)))*(a*(r-u)^5+b*(r-u)^4+c*(r-u)^3)' # [u, 1+u]
				'+step(r-(d+u))*d*(r-u)' # [1+u, +inf]
				'); '
				'r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2+eps)'
			)

			mobile_restraint.addGlobalParameter('w', TETHER_STRENGTH * unit.kilocalories_per_mole / unit.angstrom**2)
			mobile_restraint.addGlobalParameter('u', TETHER_FLATBOTTOM * unit.angstrom)
			mobile_restraint.addGlobalParameter('a', 3 * unit.angstrom**(-3))
			mobile_restraint.addGlobalParameter('b', -8 * unit.angstrom**(-2))
			mobile_restraint.addGlobalParameter('c', 6 * unit.angstrom**(-1))
			mobile_restraint.addGlobalParameter('d', 1.0 * unit.angstrom)
			mobile_restraint.addGlobalParameter('eps', 1e-16 * unit.nanometer**2) # Some noise needed in distance calculation, because r=0 in first minimization step blows up system
			mobile_restraint.addPerParticleParameter("x0")
			mobile_restraint.addPerParticleParameter("y0")
			mobile_restraint.addPerParticleParameter("z0")

			for atom in modeller.topology.atoms():
				pos = positions[atom.index].value_in_unit(unit.nanometers)

				if atom.index not in mobile_atoms:
					nonmobile_restraint.addParticle(atom.index, pos)
				else:
					mobile_restraint.addParticle(atom.index, pos)

			system.addForce(nonmobile_restraint)
			system.addForce(mobile_restraint)

			# ====================================================================
			# STEP 6: Run energy minimization
			# ====================================================================

			integrator = LangevinMiddleIntegrator(
				TEMPERATURE * unit.kelvin,
				1.0 / unit.picosecond, # friction coefficient
				TIMESTEP * unit.picoseconds
			)

			# Force OpenMM to use single-threaded CPU platform to prevent thread conflicts with multiprocessing https://github.com/openmm/openmm/issues/4424
			platform = Platform.getPlatformByName('CPU')
			properties = {'Threads': '1'}

			# Save original coordinates before energy minimization
			pdb_path = f'{DATA_DIR}/CROWN/processed_systems/{input_dir}/system_protonated.pdb'
			mol2_path = pdb_path.replace('.pdb', '.mol2')
			with open(pdb_path, 'w') as f:
				PDBFile.writeFile(pdb_modeller.topology, pdb_modeller.positions, f)
			subprocess.run(['obabel', '-ipdb', pdb_path, '-omol2', '-O', mol2_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			simulation = Simulation(modeller.topology, system, integrator, platform, properties)
			simulation.context.setPositions(modeller.positions)

			pos_before = np.array(modeller.positions.value_in_unit(unit.angstrom))

			simulation.minimizeEnergy(maxIterations = MINIMIZATION_STEPS)

			# ====================================================================
			# STEP 7: Save outputs
			# ====================================================================

			state = simulation.context.getState(getEnergy=True, getPositions=True)
			minimized_positions = state.getPositions()

			pos_after = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
			rmsd = calc_rmsd(pos_before, pos_after, sorted(mobile_atoms))
			logger.info(f'{input_dir} - RMSD {rmsd}')

			# Save minimized structure as pdb and .mol2 files
			os.makedirs(DATA_DIR / 'CROWN' / 'processed_systems' / input_dir, exist_ok = True)

			# Save minimized structure as PDB (protein/water only, no ligands)
			pdb_modeller = Modeller(modeller.topology, minimized_positions)
			pdb_path = f'{DATA_DIR}/CROWN/processed_systems/{input_dir}/system_minimized.pdb'
			mol2_path = pdb_path.replace('.pdb', '.mol2')
			with open(pdb_path, 'w') as f:
				PDBFile.writeFile(pdb_modeller.topology, pdb_modeller.positions, f)
			subprocess.run(['obabel', '-ipdb', pdb_path, '-omol2', '-O', mol2_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			# Save each ligand as a separate mol2 with minimized coordinates
			for basename, ligand_mol, atom_indices in ligand_entries:
				minimized_coords = np.array([minimized_positions[i].value_in_unit(unit.angstrom) for i in atom_indices]) * openff_unit.angstrom
				ligand_mol.conformers[0] = minimized_coords

				sdf_path = f'{tmp_dir}/{basename}_minimized.sdf'
				mol2_path = f'{DATA_DIR}/CROWN/processed_systems/{input_dir}/{basename}_minimized.mol2'

				ligand_mol.to_file(sdf_path, file_format='SDF')
				subprocess.run(['obabel', '-isdf', sdf_path, '-omol2', '-O', mol2_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			logger.removeHandler(handler)
			handler.close()

			return input_dir, rmsd

	except Exception as e:

		logger.exception(f"Refinement failed for {input_dir}")
		logger.removeHandler(handler)
		handler.close()
		return None, None
