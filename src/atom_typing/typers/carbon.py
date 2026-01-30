"""
Carbon atom typing logic.
"""

from typing import Set, Optional
from collections import Counter
import pandas as pd

from src.atom_typing.typers.base import ElementTyper, RingPositionCalculator
from src.atom_typing.data_classes import MoleculeData, NeighborInfo

class CarbonTyper(ElementTyper):
    """Atom typing logic for carbon atoms."""
    
    def type_atom(self, idx, row, mol_data: MoleculeData) -> str:
        neighbors = mol_data.get_neighbors(idx)
        biggest_ring = mol_data.ring_info.get_biggest_ring(idx)
        total_neighbors = row.total_neighbors
        heavy_neighbors = row.heavy_neighbors
        
        # SP hybridization
        if total_neighbors == 2:
            return self._type_sp(neighbors, heavy_neighbors), 'sp'
        
        # SP2 hybridization
        if total_neighbors == 3:
            return self._type_sp2(idx, neighbors, biggest_ring, heavy_neighbors, mol_data), 'sp2'
        
        # SP3 hybridization
        if total_neighbors == 4:
            return self._type_sp3(neighbors, biggest_ring, heavy_neighbors), 'sp3'
        
        return f"{row.sybyl_type}_{heavy_neighbors}", None
    
    def _type_sp(self, neighbors: NeighborInfo, heavy_neighbors: int) -> str:
        """Type sp-hybridized carbon (linear geometry)."""
        if sum(neighbors.num_heavy_neighbors) > 1:
            return f'C.1_{heavy_neighbors}'
        return 'C.co2_2'
    
    def _type_sp2(self, idx: int, neighbors: NeighborInfo, biggest_ring: Optional[Set[int]], 
                  heavy_neighbors: int, mol_data: MoleculeData) -> str:
        """Type sp2-hybridized carbon (trigonal planar)."""
        elements_set = set(neighbors.elements)
        
        if biggest_ring is None:
            return self._type_sp2_acyclic(neighbors, elements_set, heavy_neighbors)
        return self._type_sp2_ring(idx, neighbors, biggest_ring, heavy_neighbors, mol_data)
    
    def _type_sp2_acyclic(self, neighbors: NeighborInfo, elements_set: Set[str], 
                          heavy_neighbors: int) -> str:
        """Type acyclic sp2 carbon."""
        if elements_set == {'C'}:
            return f'C.2c_{heavy_neighbors}'  # Alkene
        if elements_set == {'C', 'N'}:
            return f'C.2n_{heavy_neighbors}'  # Imine
        if elements_set == {'C', 'S'}:
            return f'C.2s_{heavy_neighbors}'  # Thioketone
        if neighbors.elements == ['N', 'N', 'N']:
            return 'C.guh_3'  # Guanidinium
        if 'N' in elements_set and 'O' in elements_set:
            return f'C.am_{heavy_neighbors}'  # Amide
        if 'O' in elements_set:
            return self._type_carbonyl(neighbors, heavy_neighbors)
        return f'C.2_{heavy_neighbors}'
    
    def _type_carbonyl(self, neighbors: NeighborInfo, heavy_neighbors: int) -> str:
        """Type carbonyl carbon (aldehydes, ketones, carboxylates, esters)."""
        o_count = Counter(neighbors.elements)['O']
        if o_count == 1:
            return f'C.o_{heavy_neighbors}'
        
        # Differentiate carboxylate vs ester
        o_neighbor_sum = sum(
            n for elem, n in zip(neighbors.elements, neighbors.num_heavy_neighbors)
            if elem == 'O'
        )
        if o_neighbor_sum == 2:
            return f'C.co2_{heavy_neighbors}'  # Carboxylate
        return f'C.es_{heavy_neighbors}'  # Ester
    
    def _type_sp2_ring(self, idx: int, neighbors: NeighborInfo, ring: Set[int], 
                       heavy_neighbors: int, mol_data: MoleculeData) -> str:
        """Type ring sp2 carbon (aromatic)."""
        ring_elements = mol_data.df.loc[list(ring), 'element'].tolist()
        ring_valences = mol_data.df.loc[list(ring), 'total_neighbors'].tolist()
        ring_length = len(ring)

        # Check for aromaticity
        if all(x < 4 for x in ring_valences):    
            if ring_length == 6:
                if set(ring_elements) == {'C'}:
                    # Pure benzene ring
                    if set(neighbors.elements) == {'C'}:
                        return f'C.ar6_{heavy_neighbors}'
                    return f'C.ar6x_{heavy_neighbors}'  # Substituted
            
                # Heteroaromatic - determine position relative to heteroatom
                hetero_indices = [i for i in ring if mol_data.df.loc[i, 'element'] != 'C']
                ring_graph = mol_data.ring_info.get_ring_graph(ring)
                position = RingPositionCalculator.get_position(idx, ring, hetero_indices, ring_graph)
                return f'C.ar6{position[0]}_{heavy_neighbors}'
            
            else:
                return f'C.ar{ring_length}_{heavy_neighbors}'
            
        else:
            if set(neighbors.elements) == {'C'}:
                return f'C.2r{ring_length}_{heavy_neighbors}'
            else:
                return f'C.2r{ring_length}x_{heavy_neighbors}'
            
    def _type_sp3(self, neighbors: NeighborInfo, biggest_ring: Optional[Set[int]], 
                  heavy_neighbors: int) -> str:
        """Type sp3-hybridized carbon (tetrahedral)."""
        elements_set = set(neighbors.elements)
        
        # Ring structures
        if biggest_ring is not None:
            ring_length = len(biggest_ring)
            if elements_set == {'C'}:
                return f'C.3r{ring_length}_{heavy_neighbors}'
            return f'C.3r{ring_length}x_{heavy_neighbors}'
        
        if elements_set == {'C'}:
            return f'C.3c_{heavy_neighbors}'  # Alkane
        if 'N' in elements_set and not 'O' in elements_set:
            return f'C.3n_{heavy_neighbors}'  # Amine-adjacent
        if 'O' in elements_set:
            o_idx = neighbors.elements.index('O')
            if neighbors.num_hydrogens[o_idx] == 0:
                return f'C.et_{heavy_neighbors}'  # Ether
            return f'C.oh_{heavy_neighbors}'  # Alcohol
        if 'S' in elements_set:
            return f'C.3s_{heavy_neighbors}'

        if 'P' in elements_set:
            return f'C.3p_{heavy_neighbors}'

        for element in ['F', 'Cl', 'I', 'Br']:
            if element in elements_set:
                return f'C.3hal_{heavy_neighbors}'
        
        return f'C.3_{heavy_neighbors}'
