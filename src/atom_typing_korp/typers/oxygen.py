"""
Oxygen atom typing logic - OPTIMIZED VERSION.
"""

from typing import Set, Optional, Tuple

from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.data_classes import MoleculeData, NeighborInfo


class OxygenTyper(ElementTyper):
    """Atom typing logic for oxygen atoms."""
    
    def type_atom(self, idx: int, row, mol_data: MoleculeData) -> Tuple[str, Optional[str]]:
        neighbors = mol_data.get_neighbors(idx)
        biggest_ring = mol_data.ring_info.get_biggest_ring(idx)
        total_neighbors = row.total_neighbors
        heavy_neighbors = row.heavy_neighbors
        
        # Assume correct protonation states
        if total_neighbors == 1 and heavy_neighbors == 1:
            atom_type = self._type_single_bond(idx, neighbors, mol_data)
            return atom_type, 'sp2'
        
        if total_neighbors == 2:
            atom_type, hybridization = self._type_double_bond(
                idx, row, neighbors, biggest_ring, mol_data
            )
            return atom_type, hybridization
        
        return f"{row.sybyl_type}_{heavy_neighbors}", None
    
    def _type_single_bond(self, idx: int, neighbors: NeighborInfo, 
                          mol_data: MoleculeData) -> str:
        """Type oxygen with single heavy neighbor (carbonyl, etc.)."""
        neighbor_element = neighbors.elements[0]
        
        if neighbor_element != 'C':
            return f'O.2{neighbor_element.lower()}_1'  # P=O, S=O, N=O
        
        # Carbon neighbor - determine carbonyl type
        c_o_count = neighbors.num_oxygens[0]
        c_n_count = neighbors.num_nitrogens[0]
        
        if c_n_count == 1:
            return 'O.am_1'  # Amide oxygen
        
        if c_o_count == 1:
            if neighbors.num_hydrogens[0] == 1:
                return 'O.al_1'  # Aldehyde
            return 'O.ke_1'  # Ketone
        
        if c_o_count == 2:
            # Differentiate carboxylate vs ester
            c_idx = int(neighbors.indices[0])
            c_neighbors = mol_data.get_neighbors(c_idx)
            o_neighbor_sum = sum(
                n for elem, n in zip(c_neighbors.elements, c_neighbors.num_heavy_neighbors)
                if elem == 'O'
            )
            if o_neighbor_sum == 2:
                return 'O.co2_1'  # Carboxylate
            return 'O.es_1'  # Ester carbonyl
        
        return 'O.2_1'
    
    def _type_double_bond(self, idx: int, row, neighbors: NeighborInfo,
                          biggest_ring: Optional[Set[int]], 
                          mol_data: MoleculeData) -> Tuple[str, str]:
        """Type oxygen with two neighbors (ether, alcohol, etc.)."""
        num_hydrogens = row.num_hydrogens
        
        if num_hydrogens == 2:
            return 'O.h2o_0', 'sp3'  # Water
        
        if num_hydrogens == 1:
            c_valence = neighbors.valences[0]
            if c_valence == 4:
                return 'O.3oh_1', 'sp3'  # Aliphatic alcohol
            if c_valence == 3:
                return 'O.ph_1', 'sp2'  # Phenol
        
        if num_hydrogens == 0:
            atom_type, hybridization = self._type_ether_like(
                idx, neighbors, biggest_ring, mol_data
            )
            return atom_type, hybridization
        
        return 'O.3_2', 'sp3'
    
    def _type_ether_like(self, idx: int, neighbors: NeighborInfo,
                         biggest_ring: Optional[Set[int]], 
                         mol_data: MoleculeData) -> Tuple[str, str]:
        """Type ether-like oxygen (no hydrogens, two heavy neighbors)."""
        elements_set = set(neighbors.elements)
        
        # Phosphate/sulfate ester
        if 'P' in elements_set:
            return 'O.3p_2', 'sp3'
        if 'S' in elements_set:
            return 'O.3s_2', 'sp3'
        
        neighbor_valence = sum(neighbors.valences)
        
        # Both neighbors sp3
        if neighbor_valence == 8:
            if biggest_ring is None:
                return 'O.3et_2', 'sp3'  # Dialkyl ether
            return 'O.3ret_2', 'sp3'  # Ring ether (glycoside)
        
        # One sp3, one sp2 with oxygen
        if neighbor_valence == 7 and 2 in neighbors.num_oxygens:
            return 'O.3es_2', 'sp2'  # Ester linking oxygen
        
        # Anhydride
        if neighbor_valence == 6 and sum(neighbors.num_oxygens) == 4:
            return 'O.3anh_2', 'sp2'
        
        # Ring systems
        if biggest_ring is not None and neighbor_valence == 6:
            return 'O.ar_2', 'sp2'
        
        # Check for aromatic ether (anisole-type)
        for list_idx, c_idx in enumerate(neighbors.indices):
            if neighbors.valences[list_idx] == 3:
                ring = mol_data.ring_info.get_biggest_ring(c_idx)
                if ring is not None and len(ring) in (5, 6):
                    return 'O.3eta_2', 'sp2'

        if 'N' in elements_set:
            return 'O.3n_2', 'sp3'
        
        return 'O.3_2', 'sp3'
