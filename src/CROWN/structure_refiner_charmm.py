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
from openmm.app import PDBFile, Modeller, ForceField, Simulation, Topology
from openff.toolkit import Molecule
from openff.units import unit as openff_unit
from openmm import CustomExternalForce, LangevinMiddleIntegrator, unit, Platform
import logging

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
    'ACE', 'NME', 'HEM'
}

# ============================================================================

WATER_NAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'OPC'}

METAL_ATOMIC_NUMBERS = (
	set(range(3, 5))    |  # Li, Be
	set(range(11, 14))  |  # Na, Mg, Al
	set(range(19, 32))  |  # K..Ga
	set(range(37, 51))  |  # Rb..Sn
	set(range(55, 84))  |  # Cs..Bi
	set(range(87, 118))    # Fr..Og
)

def add_heme_bonds(topology, positions):
    """Add missing bonds for HEM/HEME residues based on interatomic distances."""
    for residue in topology.residues():
        if residue.name not in ('HEM', 'HEME'):
            continue
        
        atoms = list(residue.atoms())
        pos = np.array([(positions[a.index].x, positions[a.index].y, positions[a.index].z) 
                        for a in atoms])
        
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist = np.linalg.norm(pos[i] - pos[j])  # already in nm from OpenMM
                
                ei = atoms[i].element.symbol
                ej = atoms[j].element.symbol

                if ei == 'H' and ej == 'H':
                    continue

                if ei == 'H' or ej == 'H':
                    if dist < 0.12:
                        topology.addBond(atoms[i], atoms[j])
                
                # Fe-N bonds are ~2.0 Å
                if 'Fe' in (ei, ej):
                    if dist < 0.22:  # 2.2 Å in nm
                        topology.addBond(atoms[i], atoms[j])
                # C-C, C-N, C-O bonds are ~1.2-1.55 Å
                else:
                    if dist < 0.18 and not 'H' in (ei, ej):  # 1.8 Å in nm
                        topology.addBond(atoms[i], atoms[j])

def protonate_ligand(sdf_path: str, out_sdf: str, ph: float = 7.4):
	"""Protonate a ligand SDF file at the given pH."""
	mol = Chem.SDMolSupplier(sdf_path, removeHs=True)[0]
	if mol is None:
		# Try out alternate ways of loading molecule
		with tempfile.TemporaryDirectory() as tmp_dir:
			subprocess.run(['obabel', '-isdf', sdf_path, '-osdf', '-O', f'{tmp_dir}/temp.sdf'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
			mol = Chem.SDMolSupplier(f'{tmp_dir}/temp.sdf', removeHs=True)[0]
			if mol is None:
				subprocess.run(['obabel', '-isdf', sdf_path, '-omol2', '-O', f'{tmp_dir}/temp.mol2'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
				mol = Chem.MolFromMol2File(f'{tmp_dir}/temp.mol2', sanitize = True, removeHs = False)
				if mol is None:
					subprocess.run(['obabel', '-isdf', sdf_path, '-opdb', '-O', f'{tmp_dir}/temp.pdb'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
					mol = Chem.MolFromPDBFile(f'{tmp_dir}/temp.pdb', removeHs=False, sanitize=False)
					if mol is None:
						logger.error(f"Failed to load ligand from {sdf_path} using all fallback methods.")
						return False

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
					rdDetermineBonds.DetermineBonds(mol, charge=formal_charge, covFactor=1.15, useVdw=True) #covFactor was set to 1.15 as BCP would throw errors due to carbons being too close.

	for atom in mol.GetAtoms():
		if atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS:
			return True

	# Strip existing Hs, protonate at target pH
	mol_noh = Chem.RemoveAllHs(mol)
	protonated_mol = dl.run_with_mol_list(
		[mol_noh],
		min_ph=ph, max_ph=ph,
		pka_precision=0.0,
		silent=True
	)[0]

	protonated_3d = AllChem.AssignBondOrdersFromTemplate(protonated_mol, mol_noh)
	# Clear forced valence so RDKit can infer implicit Hs
	rw = Chem.RWMol(protonated_3d)
	for atom in rw.GetAtoms():
		atom.SetNoImplicit(False)
		atom.SetNumExplicitHs(0)
		atom.SetNumRadicalElectrons(0)
	Chem.SanitizeMol(rw)

	# Add explicit Hs with 3D coords
	protonated_h = Chem.AddHs(rw, addCoords=True)

	writer = Chem.SDWriter(out_sdf)
	writer.write(protonated_h)
	writer.close()
	logger.info(f"Protonated ligand written to {out_sdf}")

	return False

def add_seqres_with_caps(input_pdb: str, output_pdb: str):
	"""
	Add SEQRES records with ACE/NME caps to a PDB file.
	PDBFixer will then detect these as 'missing' and build them.
	"""

	# First, read the structure to get chain sequences
	pdb = PDBFile(input_pdb)

	# Build sequence for each chain
	chain_sequences = {}
	for chain in pdb.topology.chains():
		residues = list(chain.residues())
		protein_residues = [r for r in residues if r.name in STANDARD_AMINO_ACIDS and r.name != 'LIG']

		if protein_residues:
			# Add ACE at start, NME at end
			seq = ['ACE'] + [r.name for r in protein_residues] + ['NME']
			chain_sequences[chain.id] = seq

	# Now write modified PDB with SEQRES records
	with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
		# First write SEQRES records for each chain
		for chain_id, seq in chain_sequences.items():
			# SEQRES records: max 13 residues per line
			for i in range(0, len(seq), 13):
				chunk = seq[i:i+13]
				line_num = (i // 13) + 1
				seqres_line = f"SEQRES {line_num:>3} {chain_id} {len(seq):>4}  "
				seqres_line += " ".join(f"{res:>3}" for res in chunk)
				f_out.write(seqres_line + '\n')

		# Then copy the rest of the file (skip existing SEQRES lines)
		for line in f_in:
			if not line.startswith('SEQRES'):
				f_out.write(line)

	return chain_sequences

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

			# Identify which ligand SDFs contain metal atoms (e.g. HEM with Fe).
			# These will be kept in the receptor PDB and parameterized with the
			# protein force field instead of the small-molecule force field.
			for filename in os.listdir(f'{DATA_DIR}/CROWN/systems/{input_dir}'):
				if filename.endswith('.sdf') and not filename.endswith('_h.sdf'):
					basename = filename[:-4]
					has_metal_bool = protonate_ligand(f'{DATA_DIR}/CROWN/systems/{input_dir}/{filename}', f'{DATA_DIR}/CROWN/systems/{input_dir}/{basename}_h.sdf', ph = PH)

					if has_metal_bool:
						subprocess.run(['obabel', f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_fixed.pdb', f'{DATA_DIR}/CROWN/systems/{input_dir}/{filename}', '-O', f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_fixed.pdb', '-j'],
							stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
						logger.info(f"Metal detected in {filename} — will parameterize with receptor force field.")

					else:
						if not os.path.isfile(f'{DATA_DIR}/CROWN/systems/{input_dir}/{basename}_h.sdf'):
							logger.error(f"Protonation failed for {filename} — aborting refinement.")
							return

			# ====================================================================
			# Step 2: Create protein-only structure and run PDBFixer
			# ====================================================================

			pdb = PDBFile(f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_fixed.pdb')
			# Remove protein hydrogens using Modeller
			tmp_path = os.path.join(tmp_dir, 'receptor.pdb')
			with open(tmp_path, 'w') as tmp:
				PDBFile.writeFile(pdb.topology, pdb.positions, tmp)

			# Load with Modeller to remove protein hydrogens
			modeller = Modeller(pdb.topology, pdb.positions)
			toDelete = []
			for atom in modeller.topology.atoms():
				if atom.element.symbol == 'H' and atom.residue.name in STANDARD_AMINO_ACIDS:
					toDelete.append(atom)
			if toDelete:
				modeller.delete(toDelete)

			# Save deprotonated amino-acids PDB
			protein_only_path = f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_noH.pdb'
			with open(protein_only_path, 'w') as f:
				PDBFile.writeFile(modeller.topology, modeller.positions, f)

			# Add SEQRES with caps
			tmp_with_seqres = protein_only_path.replace('.pdb', '_seqres.pdb')
			chain_seqs = add_seqres_with_caps(protein_only_path, tmp_with_seqres)

			# Run PDBFixer on protein-only structure
			Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/CROWN/custom_xml/heme_hydrogens.xml')

			with open(f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_fixed.pdb') as f:
				for line in f:
					if 'FE' in line[12:16]:
						print(repr(line[76:80]))

			fixer = PDBFixer(f'{DATA_DIR}/CROWN/systems/{input_dir}/receptor_fixed.pdb')
			fixer.findMissingResidues()
			fixer.findMissingAtoms()
			fixer.addMissingAtoms()
			fixer.addMissingHydrogens(PH)

			# Create modeller from fixed protein
			logging.getLogger("openff").setLevel(logging.ERROR) #Otherwise a lot of partial charge assigned notifications
			modeller = Modeller(fixer.topology, fixer.positions)
			add_heme_bonds(modeller.topology, modeller.positions)

			for residue in modeller.topology.residues():
				if residue.name == 'HEM':
					residue.name = 'HEME'

			modeller.addHydrogens(pH=PH)

			# ====================================================================
                	# Step 3: Add ligands back with proper parameters
                	# ====================================================================

			# Add each ligand back to the modeller
			ligand_molecules = []
			ligand_entries = []  # (basename, Molecule, list of atom indices)

			for ligand_file in sorted(os.listdir(f'{DATA_DIR}/CROWN/systems/{input_dir}')):
				if ligand_file.endswith('_h.sdf'):
					basename = ligand_file.replace('_h.sdf', '')
					ligand_mol = Molecule.from_file(f'{DATA_DIR}/CROWN/systems/{input_dir}/{ligand_file}')
					ligand_molecules.append(ligand_mol)

					# Convert OpenFF positions to OpenMM unit system
					ligand_topology = ligand_mol.to_topology().to_openmm()
					ligand_positions = ligand_mol.conformers[0].to_openmm()

					# Record indices before adding (they'll start at current atom count)
					offset = modeller.topology.getNumAtoms()
					n_atoms = ligand_topology.getNumAtoms()
					modeller.add(ligand_topology, ligand_positions)

					ligand_entries.append((basename, ligand_mol, list(range(offset, offset + n_atoms))))

			# Merged set of all ligand indices (used for KDTree and restraints)
			all_ligand_indices = set()
			for _, _, indices in ligand_entries:
				all_ligand_indices.update(indices)

			system_generator = SystemGenerator(
				forcefields=['charmm36.xml', 'charmm36/water.xml'], # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
				small_molecule_forcefield='openff-2.2.0',
				molecules=ligand_molecules,
				cache=f'{tmp_dir}/{input_dir}_db.json'
			)

			print('Creating system')
			system = system_generator.create_system(modeller.topology)
			print('Built system')

			# ====================================================================
			# STEP 4: Identify mobile region around ligands
			# ====================================================================

			# Use KDTree for more efficient spatial lookup
			positions = modeller.positions
			all_positions_nm = np.array([positions[i].value_in_unit(unit.nanometer) for i in range(len(positions))])
			ligand_indices_list = sorted(all_ligand_indices)
			ligand_positions_nm = all_positions_nm[ligand_indices_list]
			ligand_tree = KDTree(ligand_positions_nm)

			distances, _ = ligand_tree.query(all_positions_nm, k=1)  # nearest ligand atom
			mobile_mask = distances <= MOBILE_RADIUS
			mobile_atoms = set(np.where(mobile_mask)[0].tolist()) # atom indices where distance smaller than restraint radius

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

				# Leave hydrogens and water atoms fully unrestrained
				if atom.element.symbol == 'H' or atom.residue.name in WATER_NAMES:
					continue

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

			simulation = Simulation(modeller.topology, system, integrator, platform, properties)
			simulation.context.setPositions(modeller.positions)
			simulation.minimizeEnergy(maxIterations = MINIMIZATION_STEPS)

			# ====================================================================
			# STEP 7: Save outputs
			# ====================================================================

			state = simulation.context.getState(getEnergy=True, getPositions=True)
			minimized_positions = state.getPositions()

			# Save minimized structure as pdb and .mol2 files
			os.makedirs(DATA_DIR / 'CROWN' / 'processed_systems' / input_dir, exist_ok = True)

			# Save minimized structure as PDB (protein/water only, no ligands)
			pdb_modeller = Modeller(modeller.topology, minimized_positions)
			ligand_atoms = [atom for atom in pdb_modeller.topology.atoms() if atom.index in all_ligand_indices]
			pdb_modeller.delete(ligand_atoms)
			with open(DATA_DIR / 'CROWN' / 'processed_systems' / input_dir / 'receptor_minimized.pdb', 'w') as f:
				PDBFile.writeFile(pdb_modeller.topology, pdb_modeller.positions, f)

			# Save each ligand as a separate mol2 with minimized coordinates
			for basename, ligand_mol, atom_indices in ligand_entries:
				minimized_coords = np.array([minimized_positions[i].value_in_unit(unit.angstrom) for i in atom_indices]) * openff_unit.angstrom
				ligand_mol.conformers[0] = minimized_coords

				sdf_path = f'{tmp_dir}/{basename}_minimized.sdf'
				mol2_path = f'{DATA_DIR}/CROWN/processed_systems/{input_dir}/{basename}_minimized.mol2'

				ligand_mol.to_file(sdf_path, file_format='SDF')
				subprocess.run(['obabel', '-isdf', sdf_path, '-omol2', '-O', mol2_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

	except Exception as e:
		logger.exception(f"Refinement failed for {input_dir}")

	finally:
		logger.removeHandler(handler)
		handler.close()
