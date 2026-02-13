from src.config import DATA_DIR, SOURCE_DB_PATH

from Bio.PDB import MMCIFParser, PDBIO, NeighborSearch
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio import PDB
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import tempfile
import subprocess

from joblib import Parallel, delayed

STANDARD_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

COMMON_ARTIFACTS = ['PEG', 'CRY', 'EDO', 'ACT', 'DMS', 'MES', 'GOL', 'EPE', 'BU1', 'BCN',
                    'PL9', '15P', 'P6G', 'MPD', 'PG4', 'TRS', 'PGE', '1PE', 'ACY']

VALID_BOND_ATOMS = {'C', 'N', 'O', 'S', 'P', 'B'}

METALLOCOFACTOR_LIST = ['HEM', 'SF4', 'MGD', 'B12', 'COB', 'CLA', 'BCL', 'CHL', 'F43']

@dataclass
class OccupancyInfo:
    """Store occupancy information for a residue"""
    chain: str
    res_num: str
    low_occupancy_count: int = 0
    high_occupancy_count: int = 0

@dataclass
class ResidueContact:
    """Store information about contacts between residues"""
    chain1: str
    res1: int
    chain2: str
    res2: int
    contact_count: int
    atom_pairs: List[Tuple]

def is_valid_cif(path):
    try:
        d = MMCIF2Dict(path)
        return '_atom_site.id' in d
    except Exception:
        return False

class NonUniqueStructureBuilder(PDB.StructureBuilder.StructureBuilder):
    """This makes PDB more forgiving by being able to load atoms with non-unique names within a residue"""

    @staticmethod
    def _number_to_3char_name(n):
        code = ''
        for k in range(3):
            r = n % 36
            code = chr(ord('A')+r if r < 26 else ord('0')+r-26) + code
            n = n // 36
        assert n == 0, 'number cannot fit 3 characters'
        return code

    def init_atom(self, name, coord, b_factor, occupancy, altloc, fullname, serial_number, element):
        for attempt in range(10000):
            try:
                return super().init_atom(name, coord, b_factor, occupancy, altloc, fullname, serial_number, element)
            except PDB.PDBExceptions.PDBConstructionException:
                name = name[0] + self._number_to_3char_name(attempt)

    def init_residue(self, resname, hetatm_flag, resseq, icode):
        try:
            super().init_residue(resname, hetatm_flag, resseq, icode)
        except PDB.PDBExceptions.PDBConstructionException as e:
            # Reuse existing residue instead of failing
            if resname != 'HOH':
                self.residue = self.chain[(f'{hetatm_flag}_{resname}', resseq, icode)]
            else:
                self.residue = self.chain[(hetatm_flag, resseq, icode)]

class OccupancyHandler:
    def __init__(self):
        self.occupancy_data: Dict[str, OccupancyInfo] = {}

    def analyze_occupancy(self, input_file: str) -> Dict[str, OccupancyInfo]:
        """Returns dictionary mapping residue keys to occupancy information"""

        self.occupancy_data.clear()

        with open(input_file, 'r') as infile:
            for line in infile:
                if line.startswith(('ATOM', 'HETATM')):
                    columns = line.split()
                    occupancy = float(columns[11])
                    chain = columns[4]
                    try:
                        res_num = int(columns[6])
                    except ValueError:
                        res_num = 1
                    dict_key = f'{chain}_{res_num}'

                    if occupancy != 1.0:
                        if dict_key not in self.occupancy_data:
                            self.occupancy_data[dict_key] = OccupancyInfo(
                                chain = chain, res_num = res_num
                            )

                        if occupancy < 0.5:
                            self.occupancy_data[dict_key].low_occupancy_count += 1
                        else:
                            self.occupancy_data[dict_key].high_occupancy_count += 1

        return self.occupancy_data

    def resolve_occupancy(self, line: str):
        if not line.startswith(('ATOM', 'HETATM')):
            return '1.A', 0, 1.0, line
        
        columns = line.split()
        if len(columns) < 15:
            return '1.A', 0, 1.0, line
        
        chain = columns[4]
        try:
            res_num = int(columns[6])
        except ValueError:
            res_num = 1
        dict_key = f'{chain}_{res_num}'

        occupancy = float(columns[11])
        if occupancy == 1.0:
            return chain, res_num, occupancy, line

        occ_info = self.occupancy_data[dict_key]
        low_count = occ_info.low_occupancy_count
        high_count = occ_info.high_occupancy_count

        if high_count == low_count:
            if occupancy < 0.5:
                return None, None, None, None
            
        return chain, res_num, occupancy, line
    
class ArtifactRemover:
    """Remove common artifacts and handle file formatting issues"""

    def __init__(self):
        self.artifacts = COMMON_ARTIFACTS

    def remove_artifacts_and_fix_quotes(self, input_path, output_path,
                                        occupancy_handler: OccupancyHandler):
        
        out_list = []

        with open(input_path, 'r') as infile:
            buffer = "" # Buffer for handling multi-line entries with unbalanced quotes
            atoms_read = False
            for line in infile:
                line = line.strip()
                
                if not line.startswith(('HETATM', 'ATOM')):
                    stripped_line = line.strip()
                    if buffer:
                        buffer += ' ' + stripped_line
                        if buffer.count('"') % 2 == 0:
                            line = buffer
                            buffer = ""
                        else:
                            continue
                    else:
                        if stripped_line.count('"') % 2 != 0:
                            buffer = stripped_line
                            continue
                        else:
                            line = stripped_line

                    if not atoms_read:
                        chain = '1.A'
                    else:
                        chain = '9.Z'
                    res_num = 0
                    occupancy = 1.0

                # Handle ATOM/HETATM lines
                else:
                    atoms_read = True
                    columns = line.split()
                    if len(columns) < 4:
                        continue
                    if columns[3] in self.artifacts:
                        continue

                    chain, res_num, occupancy, line = occupancy_handler.resolve_occupancy(line)
                    if line is None:
                        continue

                line = line.replace('"', '_1').replace("'", '_2')
                out_list.append((chain, res_num, occupancy, line))

        # First sort ascending on residue number, then descending on occupancy
        sorted_list = sorted(out_list, key = lambda x: (x[0], x[1], -x[2]))
        sorted_lines = [x[3] for x in sorted_list]
        with open(output_path, 'w') as outfile:
            outfile.write('\n'.join(sorted_lines))

class OverlapResolver:
    """Detect and resolve inter-residue contacts"""

    def __init__(self, 
                 distance_threshold: float = 1.8,
                 valid_atoms: Set[str] = None,
                 aa_list: List[str] = None):
        
        self.distance_threshold = distance_threshold
        self.valid_atoms = VALID_BOND_ATOMS
        self.aa_list = STANDARD_AA

    def create_chain_mapping(self, structure, ligand_id):
        """Create mapping from multi-character CIF chain IDs to single-character PDB IDs"""

        model = structure[0]
        chain_mapping = {}
        pdb_chain_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz0123456789")
        pdb_char_index = 0

        # Reserve ligand chain ID
        chain_mapping[ligand_id] = 'Z'

        # Map remaining chains
        for chain in model:
            orig_id = chain.id
            if orig_id != ligand_id:
                chain_mapping[orig_id] = pdb_chain_chars[pdb_char_index]
                pdb_char_index += 1

        return chain_mapping
    
    def rename_chains(self, structure, chain_mapping: Dict[str, str]) -> None:
        model = structure[0]
        chains_to_rename = list(model.get_chains())

        ligand_chain = None
        other_chains = []

        for chain in chains_to_rename:
            old_id = chain.id
            new_id = chain_mapping[old_id]
            model.detach_child(old_id)
            chain.id = new_id
            if new_id == 'Z':
                ligand_chain = chain
            else:
                other_chains.append(chain)

        for chain in other_chains:
            model.add(chain)

        if ligand_chain is not None:
            model.add(ligand_chain)
            return True

        else:
            return False

    def detect_contacts(self, structure) -> Tuple[Dict[str, ResidueContact], List[str], List[str], List[Tuple[str, int]]]:
        """
        Detects all contacts between residues in different chains
        
        Returns
        -------
        Tuple of:
            - Dictionary of contacts (key: sorted chain pair, value: ResidueContact)
            - List of chain pairs with 1 contact (bonds to add)
            - List of chain pairs with multiple contacts (overlaps to resolve)
            - List of intra-chain contacts
        """

        atoms = list(structure.get_atoms())
        neighbor_search = NeighborSearch(atoms)

        contact_counts = defaultdict(int)
        contact_details = defaultdict(list)
        intra_chain_contacts = set()

        for atom1 in atoms:
            # Skip invalid atom types and protein residues
            if atom1.element not in self.valid_atoms:
                continue

            res1 = atom1.get_parent()
            res1_name = res1.get_resname()
            res1_id = res1.get_id()[1]
            chain1_id = res1.get_full_id()[2]

            # Find nearby atoms
            close_atoms = neighbor_search.search(atom1.coord, self.distance_threshold)

            for atom2 in close_atoms:
                if atom1 == atom2:
                    continue

                if atom2.element not in self.valid_atoms:
                    continue

                res2 = atom2.get_parent()
                res2_name = res2.get_resname()
                res2_id = res2.get_id()[1]
                chain2_id = res2.get_full_id()[2]

                if chain1_id != chain2_id:
                    # Create sorted pair key (interactions are symmetric)
                    pair_key = tuple(sorted([chain1_id, chain2_id]))
                    atom_pair = tuple(sorted([
                        (chain1_id, res1_id, atom1.get_name()),
                        (chain2_id, res2_id, atom2.get_name())
                    ]))

                    if atom_pair not in contact_details[pair_key]:
                        contact_counts[pair_key] += 1
                        contact_details[pair_key].append(atom_pair)

            really_close_atoms = neighbor_search.search(atom1.coord, 1.0)

            # Deal with funky same-chain contacts
            for atom2 in really_close_atoms:
                if atom1 == atom2:
                    continue

                if atom2.element not in self.valid_atoms:
                    continue

                res2 = atom2.get_parent()
                res2_name = res2.get_resname()
                res2_id = res2.get_id()[1]
                chain2_id = res2.get_full_id()[2]

                if chain1_id == chain2_id:
                    atom_pair = tuple(sorted([
                        (chain1_id, res1_id, atom1.get_name()),
                        (chain1_id, res2_id, atom2.get_name())
                    ]))
                    intra_chain_contacts.add(atom_pair)

        # Separate contacts into bonds and overlaps
        bonds_to_add = []
        overlaps_to_resolve = []
        contact_info = {}

        for pair_key, count in contact_counts.items():
            chain1, chain2 = pair_key

            contact_obj = ResidueContact(
                chain1 = chain1,
                chain2 = chain2,
                res1 = None,
                res2 = None,
                contact_count = count,
                atom_pairs = contact_details[pair_key]
            )
            contact_info[pair_key] = contact_obj

            if count == 1:
                bonds_to_add.append(f'{chain1},{chain2}')
            else:
                overlaps_to_resolve.append(f'{chain1},{chain2}')

        intra_chain_contacts = list(intra_chain_contacts)
        return contact_info, bonds_to_add, overlaps_to_resolve, intra_chain_contacts
    
    def resolve_intra_chain_clashes(self, structure, intra_chain_contacts):
        """
        For intra-chain clashes, use a greedy pruning algorithm. Keep first atoms, remove any later issues

        Parameters
        ----------
        intra_chain_contacts: List of pair tuples [chain, res, atom]
        """

        model = structure[0]

        problematic_chains_list = [x[0][0] for x in intra_chain_contacts] # Extract chain_id from first tuple of each entry
        problematic_chains_list = list(set(problematic_chains_list)) # Remove duplicates

        # Construct a simple partner mapping
        partner_map = {}
        for x, y in intra_chain_contacts:
            partner_map[x] = y
            partner_map[y] = x

        greedy_atom_set = set()

        for chain_id in problematic_chains_list:
            res_num = 1
            chain = model[chain_id]
            model.detach_child(chain_id)
            new_chain = Chain(chain_id)

            for residue in chain:
                new_atoms = []
                res_id = residue.get_id()[1]
                res_name = residue.get_resname()
                
                for atom in residue.get_atoms():
                    contact_key = (chain_id, res_id, atom.get_name())
                    if contact_key in partner_map:
                        contact_value = partner_map[contact_key]
                        if not contact_value in greedy_atom_set:
                            new_atoms.append(atom)
                            greedy_atom_set.add(contact_key)
                    else:
                        new_atoms.append(atom)
                        greedy_atom_set.add(contact_key)

                if new_atoms:
                    new_residue = Residue((' ', res_num, ' '), res_name, ' ')
                    for atom in new_atoms:
                        new_residue.add(atom)

                    new_chain.add(new_residue)
                    res_num += 1

            model.add(new_chain)

    def resolve_overlaps(self, structure, overlaps: List[str]) -> None:
        """
        Remove one chain from each overlapping pair.
        Priority: keep ligand chain > keep first chain alphabetically
        """

        model = structure[0]
        for overlap_pair in overlaps:
            chain1_id, chain2_id = overlap_pair.split(',')

            if chain2_id == 'Z':
                chain_to_remove = chain1_id
            else:
                chain_to_remove = chain2_id

            if model.has_id(chain_to_remove):
                model.detach_child(chain_to_remove)

    def merge_bonded_chains(self, structure, bonds: List[str]) -> None:
        """
        Merge chains connected by single bonds into unified ligand chains.
        Ensured ligand group is placed at end of file
        """

        model = structure[0]

        # Build connected components using graph merging
        # If A and B are connected, and B and C, then final chain is A-B-C
        chain_sets = [set(bond.split(',')) for bond in bonds]
        merged = True

        while merged:
            merged = False
            new_sets = []

            while chain_sets:
                current_set = chain_sets.pop(0)

                # Check for overlaps with other sets
                for other_set in chain_sets:
                    if current_set & other_set: # Intersection found
                        current_set |= other_set # Merge sets
                        chain_sets.remove(other_set)
                        merged = True

                new_sets.append(current_set)

            chain_sets = new_sets

        # Sort chain groups: ligand groups last
        ligand_groups = []
        non_ligand_groups = []

        for chain_set in chain_sets:
            if 'Z' in chain_set:
                ligand_groups.append(list(chain_set))
            else:
                non_ligand_groups.append(list(chain_set))

        all_groups = non_ligand_groups + ligand_groups
        new_chain_names = [x[0] if not 'Z' in x else 'Z' for x in all_groups]

        # Merge connected components
        for chain_set, chain_name in zip(all_groups, new_chain_names):
            self._merge_chain_group(model, chain_set, chain_name)

    def _merge_chain_group(self, model, chain_set, chain_name):
        """Merge a set of connected chains into a single chain"""

        # Collect atoms from all chains in the group
        all_atoms = []
        residue_counts = []
        chains_to_remove = []

        for chain_id in chain_set:
            # Check if chain still exists, maybe it's been removed as overlap
            if model.has_id(chain_id):
                chain = model[chain_id]
                residue_count = 0
                for residue in chain:
                    residue_count += 1
                    for atom in residue.get_atoms():
                        all_atoms.append((atom, atom.get_parent().get_id()))
                residue_counts.append(residue_count)
                chains_to_remove.append(chain_id)

        if not all_atoms:
            return

        if chain_name != 'Z' and max(residue_counts) > 15:
            return

        for chain_id in chains_to_remove:
            model.detach_child(chain_id)
        
        # Sort atoms by residue
        all_atoms.sort(key = lambda x: (x[1][0], x[1][1]))

        # Create new merged chain
        new_chain = Chain(chain_name)

        # Create single residue for all atoms
        if chain_name == 'Z':
            res_name = 'LIG'
        else:
            res_name = 'UNK'
        new_residue = Residue((' ', 1, ' '), res_name, ' ')

        # Add atoms with unique IDs
        for i, (atom, _) in enumerate(all_atoms):
            old_id = atom.id
            new_id = f'{old_id[0]}{i+1}'
            atom.id = new_id
            atom.name = new_id
            atom.fullname = new_id
            new_residue.add(atom)

        new_chain.add(new_residue)
        model.add(new_chain)

    def rename_ligand(self, structure):
        model = structure[0]

        lig_atoms = []
        chain = model['Z']

        cofactor_bool = False

        for residue in chain:
            if residue.get_resname() in METALLOCOFACTOR_LIST:
                cofactor_bool = True
                break

            for atom in residue.get_atoms():
                lig_atoms.append(atom)

        if not cofactor_bool:

            model.detach_child('Z')

            # Create new merged chain
            new_chain = Chain('Z')
            new_residue = Residue((' ', 1, ' '), 'LIG', ' ')

            # Add atoms with unique IDs
            for i, atom in enumerate(lig_atoms):
                old_id = atom.id
                new_id = f'{old_id[0]}{i+1}'
                atom.id = new_id
                atom.name = new_id
                atom.fullname = new_id
                new_residue.add(atom)

            new_chain.add(new_residue)
            model.add(new_chain)

        # Rename all atoms
        for chain in model:
            for residue in chain:
                for i, atom in enumerate(residue):
                    if len(atom.name) > 4:
                        new_name = f'{atom.name[0]}{i:03d}'
                        atom.name = new_name
                        atom.fullname = new_name

        # Check if final ligand chain has more than 10 heavy atoms
        atom_count = sum(1 for atom in new_chain.get_atoms() if atom.element != 'H')
        return atom_count

class ComplexFixer:
    """Fix protein-ligand complexes from PLInder"""
    def __init__(self, filtered_subset):
        """
        Parameters
        ----------

        filtered_subset[pd.DataFrame]
            - basename [str]: PLI system identifier
            - system_id [str]: input filename
            - ligand_instance_chain [str]: ligand chain identifier
            - entry_resolution [float]: resolution of crystal structure
            - system_ligand_validation_average_rsr [float]: RSR of ligand
            - system_ligand_validation_average_rscc [float]: RSCC of ligand
            - system_pocket_UniProt [str]: UniProt ID of receptor
            - system_pocket_CATH [str]: CATH ID of receptor
            - ligand_unique_ccd_code [str]: CCD code of ligand
            - ligand_rdkit_canonical_smiles [str]: Canonical SMILES representation of ligand
        """

        self.occupancy_handler = OccupancyHandler()
        self.artifact_remover = ArtifactRemover()
        self.overlap_resolver = OverlapResolver()

        self.filtered_subset = filtered_subset

    def preprocess_file(self, system_id, ligand_id):

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Step 1: Analyze occupancy
            input_path = SOURCE_DB_PATH / 'systems' / system_id / 'system.cif'
            self.occupancy_handler.analyze_occupancy(input_path)

            # Step 2: Remove artefacts and fix file formatting
            basename = system_id.lower() + '_' + ligand_id.lower()
            tmp_path = f'{tmp_dir}/{basename}.cif'
            self.artifact_remover.remove_artifacts_and_fix_quotes(
                input_path, tmp_path, self.occupancy_handler
            )

            # Step 3: Load structure
            cif_bool = is_valid_cif(tmp_path)
            if not cif_bool:
                return
            parser = MMCIFParser(QUIET = True, structure_builder = NonUniqueStructureBuilder())
            structure = parser.get_structure('system', tmp_path)

            # Step 4: Chain mapping from CIF to PDB
            chain_mapping = self.overlap_resolver.create_chain_mapping(structure, ligand_id)
            lig_bool = self.overlap_resolver.rename_chains(structure, chain_mapping)
            if not lig_bool:
                return

            # Step 5: Detect contacts
            contact_info, bonds_to_add, overlaps_to_resolve, intra_chain_contacts = self.overlap_resolver.detect_contacts(structure)

            # Step 6: Resolve overlaps first
            self.overlap_resolver.resolve_intra_chain_clashes(structure, intra_chain_contacts)
            self.overlap_resolver.resolve_overlaps(structure, overlaps_to_resolve)

            # Step 7: Merge bonded chains
            self.overlap_resolver.merge_bonded_chains(structure, bonds_to_add)
            atom_count = self.overlap_resolver.rename_ligand(structure)

            if atom_count >= 10 and atom_count <= 100:
                # Step 8: save output
                io = PDBIO()
                io.set_structure(structure)

                output_path = f'{DATA_DIR}/CROWN/raw_pdb/{basename}.pdb'
                io.save(output_path)

    def wrapper(self, num_cores = 1):
        """
        Runs through all dataframe entries and converts mmCIF to semi-processed PDB

        Parameters
        ----------

        num_cores [int]: Number of CPU's for parallel processing. Default value = 1
        """

        system_id_list = self.filtered_subset['system_id'].tolist()
        ligand_id_list = self.filtered_subset['ligand_instance_chain'].tolist()

        os.makedirs(f'{DATA_DIR}/CROWN/raw_pdb', exist_ok = True)

        Parallel(n_jobs = num_cores, verbose = 10)(delayed(self.preprocess_file)(system_id, ligand_id) for system_id, ligand_id in zip(system_id_list, ligand_id_list))
