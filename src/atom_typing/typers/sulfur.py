"""
Sulfur atom typing logic.
"""

from typing import Set, Optional
import pandas as pd

from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.data_classes import MoleculeData, NeighborInfo

class SulfurTyper(ElementTyper):
    """Atom typing logic for sulfur atoms."""
    
    def type_atom(self, idx, row, mol_data: MoleculeData) -> str:
        neighbors = mol_data.get_neighbors(idx)
        biggest_ring = mol_data.ring_info.get_biggest_ring(idx)
        total_neighbors = row.total_neighbors
        heavy_neighbors = row.heavy_neighbors
        
        # Thione (C=S)
        if total_neighbors == 1:
            atom_type = f"S.2{neighbors.elements[0].lower()}_1"
            if neighbors.elements[0] == 'P':
                hybridization = 'sp3'
            else:
                hybridization = 'sp2'
            return f"S.2{neighbors.elements[0].lower()}_1", hybridization
        
        # Thiol
        if total_neighbors == 2 and heavy_neighbors == 1:
            return f"S.{neighbors.elements[0].lower()}sh_1", 'sp3'
        
        # Divalent sulfur (thioether, disulfide, thiophene)
        if total_neighbors == 2 and heavy_neighbors == 2:
            atom_type, hybridization = self._type_divalent(neighbors, biggest_ring)
            return atom_type, hybridization
        
        # Sulfoxide
        if heavy_neighbors == 3:
            return f"S.o{row.num_oxygens}_3", 'sp3'
        
        # Sulfone/sulfonamide
        if heavy_neighbors == 4:
            return f"S.o{row.num_oxygens}n{row.num_nitrogens}_4", 'sp3'
        
        return f"{row.sybyl_type}_{heavy_neighbors}", 'sp3'
    
    def _type_divalent(self, neighbors: NeighborInfo, 
                       biggest_ring: Optional[Set[int]]) -> str:
        """Type divalent sulfur (two heavy neighbors, no hydrogens)."""
        # Ring sulfur
        if biggest_ring is not None:
            if any(x == 4 for x in neighbors.valences):
                return 'S.r_2', 'sp3'  # Saturated ring (thiane)
            return 'S.ar_2', 'sp2'  # Aromatic (thiophene)
        
        # Acyclic
        if 'O' in neighbors.elements:
            return 'S.o_2', 'sp3'  # Sulfoxide-type
        if 'S' in neighbors.elements:
            return 'S.s_2', 'sp3'  # Disulfide
        return 'S.3_2', 'sp3'  # Thioether
