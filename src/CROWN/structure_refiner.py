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

logger = logging.getLogger(__name__)

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
RESTRAINT_RADIUS = 0.6  # Distance in nm (6 Å = 0.6 nm)
RESTRAINT_STRENGTH = 1000000.0  # kJ/mol/nm² - very high to effectively freeze atoms
MINIMIZATION_STEPS = 5000
ENERGY_REPORT_INTERVAL = 50
TEMPERATURE = 300  # Kelvin
TIMESTEP = 0.002  # picoseconds
PH = 7.4

STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYM', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 
    'TRP', 'TYR', 'VAL', 'HIE', 'HIP', 'HID', 'HSD', 'HSE', 'HSP',
    'ACE', 'NME', 'UNK', 'UNL'
}

# ============================================================================

WATER_NAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'OPC'}

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
					rdDetermineBonds.DetermineBonds(mol, charge=formal_charge, covFactor=1.15, useVdw=True) #covFactor was set to 1.15 as BCP would throw errors due to carbons being too close.

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

	writer = Chem.SDWriter(out_sdf)
	writer.write(protonated_h)
	writer.close()
	logger.info(f"Protonated ligand written to {out_sdf}")

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
		protein_residues = [r for r in residues if r.name in STANDARD_AMINO_ACIDS]

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

	with tempfile.TemporaryDirectory() as tmp_dir:
		for filename in os.listdir(input_dir):
			if filename.endswith('.sdf'):
				basename = filename[:-4]
				protonate_ligand(f'{input_dir}/{filename}', f'{tmp_dir}/{basename}_h.sdf', ph = PH)
				if not os.path.isfile(f'{tmp_dir}/{basename}_h.sdf'):
					logger.error(f"Protonation failed for {filename} — aborting refinement.")
					return

		# ====================================================================
		# Step 2: Create protein-only structure and run PDBFixer
		# ====================================================================

		pdb = PDBFile(f'{input_dir}/receptor_fixed.pdb')
		# Remove protein hydrogens using Modeller
		with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
			PDBFile.writeFile(pdb.topology, pdb.positions, tmp)
			tmp_path = tmp.name

		# Load with Modeller to remove protein hydrogens
		modeller = Modeller(pdb.topology, pdb.positions)
		toDelete = []
		for atom in modeller.topology.atoms():
			if atom.element.symbol == 'H' and atom.residue.name in STANDARD_AMINO_ACIDS:
				toDelete.append(atom)
			elif atom.residue.name not in STANDARD_AMINO_ACIDS and atom.residue.name not in WATER_NAMES:
				toDelete.append(atom)
		if toDelete:
			modeller.delete(toDelete)

		# Save deprotonated amino-acids PDB
		protein_only_path = tmp_path.replace('.pdb', '_no_protein_H.pdb')
		with open(protein_only_path, 'w') as f:
			PDBFile.writeFile(modeller.topology, modeller.positions, f)

		# Add SEQRES with caps
		tmp_with_seqres = protein_only_path.replace('.pdb', '_seqres.pdb')
		chain_seqs = add_seqres_with_caps(protein_only_path, tmp_with_seqres)

		# Run PDBFixer on protein-only structure
		fixer = PDBFixer(filename=tmp_with_seqres)
		fixer.findMissingResidues()
		fixer.findMissingAtoms()
		fixer.addMissingAtoms()
		fixer.addMissingHydrogens(PH)

		# Create modeller from fixed protein
		logging.getLogger("openff").setLevel(logging.ERROR) #Otherwise a lot of partial charge assigned notifications
		modeller = Modeller(fixer.topology, fixer.positions)

		# ====================================================================
                # Step 3: Add ligands back with proper parameters
                # ====================================================================

		# Add each ligand back to the modeller
		ligand_molecules = []

		for ligand_file in os.listdir(tmp_dir):
			ligand_mol = Molecule.from_file(f'{tmp_dir}/{ligand_file}')
			ligand_molecules.append(ligand_mol)

			# Convert OpenFF positions to OpenMM unit system
			ligand_topology = ligand_mol.to_topology().to_openmm()
			ligand_positions = ligand_mol.conformers[0].to_openmm()

			# Add ligand to modeller
			modeller.add(ligand_topology, ligand_positions)

		system_generator = SystemGenerator(
			forcefields=['amber19/protein.ff19SB.xml', 'amber19/DNA.OL21.xml', 'amber19/lipid21.xml', 'amber19/opc3.xml'], # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
			small_molecule_forcefield='openff-2.2.0',
			molecules=ligand_molecules,
			cache=f'{tmp_dir}/{input_dir}_db.json')
		)

		system = system_generator.create_system(modeller.topology)

		# ====================================================================
		# STEP 4: Identify mobile region around ligands
		# ====================================================================

		# Find all ligand atoms
		ligand_atom_indices = set()
		for residue in modeller.topology.residues():
			if (residue.name not in WATER_NAMES and residue.name not in STANDARD_AMINO_ACIDS):
				for atom in residue.atoms():
					ligand_atom_indices.add(atom.index)

		positions = modeller.positions

		# Use KDTree for more efficient spatial lookup
		all_positions_nm = np.array([positions[i].value_in_unit(unit.nanometer) for i in range(len(positions))])
		ligand_indices_list = sorted(ligand_atom_indices)
		ligand_positions_nm = all_positions_nm[ligand_indices_list]
		ligand_tree = KDTree(ligand_positions_nm)

		distances, _ = ligand_tree.query(all_positions_nm, k=1)  # nearest ligand atom
		mobile_mask = distances <= RESTRAINT_RADIUS
		mobile_atoms = set(np.where(mobile_mask)[0].tolist()) # atom indices where distance smaller than restraint radius

		# ====================================================================
                # STEP 5: Add restraints to both mobile and non-mobile atoms
                # ====================================================================

		nonmobile_restraint = CustomExternalForce("k*r^2; r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)")
		nonmobile_restraint.addGlobalParameter('k', RESTRAINT_STRENGTH * unit.kilojoules_per_mole / unit.nanometer**2)
		nonmobile_restraint.addPerParticleParameter("x0")
		nonmobile_restraint.addPerParticleParameter("y0")
		nonmobile_restraint.addPerParticleParameter("z0")

		# Continuously differentiable energy term. Flat-bottom tethering with smoothstep function
		# Energy, force and second derivative at r=0.25 are 0.
		# Energy and force at 1.25 are 1, second derivative is 0.
		mobile_restraint = CustomExternalForce('w*('
			'step(r-0.25)*(1-step(r-1.25))*(3*(r-0.25)^5-8*(r-0.25)^4+6*(r-0.25)^3)' # [0.25, 1.25]
			'+step(r-1.25)*(r-0.25)' # [1.25, +inf]
			'); '
			'r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)'
		)

		mobile_restraint.addGlobalParameter('w', 10 * unit.kilocalories_per_mole / unit.angstrom**2)
		mobile_restraint.addPerParticleParameter("x0")
		mobile_restraint.addPerParticleParameter("y0")
		mobile_restraint.addPerParticleParameter("z0")

		for atom in modeller.topology.atoms():
			pos = positions[atom.index]
			if atom.index not in mobile_atoms:
				nonmobile_restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
			else:
				mobile_restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])

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

		# Save minimized structure as pdb (and .mol2?)
		with open(DATA_DIR / 'CROWN' / 'processed_pdb' / f'{input_dir}.pdb', 'w') as f:
			PDBFile.writeFile(modeller.topology, minimized_positions, f)
