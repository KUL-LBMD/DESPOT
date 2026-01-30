"""
Nitrogen atom typing logic.
"""

from typing import Set, Optional
import numpy as np
import pandas as pd

from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.data_classes import MoleculeData, NeighborInfo

class NitrogenTyper(ElementTyper):
    """Atom typing logic for nitrogen atoms."""
    
    def type_atom(self, idx, row, mol_data: MoleculeData) -> str:
        neighbors = mol_data.get_neighbors(idx)
        biggest_ring = mol_data.ring_info.get_biggest_ring(idx)
        heavy_neighbors = row.heavy_neighbors
        total_neighbors = row.total_neighbors
        
        # Special cases first
        if heavy_neighbors == 0:
            return 'N.3_0', 'sp3'
        
        if total_neighbors == 1 and neighbors.elements[0] == 'C':
            return 'N.1_1', 'sp'  # Nitrile
        
        if total_neighbors == 4 and set(neighbors.elements) == {'C'}:
            return f'N.4_{heavy_neighbors}', 'sp3'  # Quaternary amine
        
        # Check functional groups
        atom_type, hybridization = self._check_functional_groups(idx, row, neighbors, biggest_ring, mol_data)
        if atom_type:
            return atom_type, hybridization
        
        return f"{row.sybyl_type}_{heavy_neighbors}", None
    
    def _check_functional_groups(self, idx, row, neighbors: NeighborInfo,
                                  biggest_ring: Optional[Set[int]], 
                                  mol_data: MoleculeData) -> Optional[str]:
        """Check for various nitrogen functional groups."""
        heavy_neighbors = row.heavy_neighbors
        
        # Guanidinium
        if any(x == 3 for x in neighbors.num_nitrogens):
            return f'N.gu_{heavy_neighbors}', 'sp2'
        
        # Amidine (not in ring)
        if any(x == 2 and y == 3 for x, y in zip(neighbors.num_nitrogens, neighbors.valences)):
            if biggest_ring is None:
                return f'N.mih_{heavy_neighbors}', 'sp2'
        
        # Sulfonamide
        if any(x == 'S' and y == 2 for x, y in zip(neighbors.elements, neighbors.num_oxygens)):
            return f'N.sa2_{heavy_neighbors}', 'sp2'

        if any(x == 'S' and y == 3 for x, y in zip(neighbors.elements, neighbors.num_oxygens)):
            return f'N.sa3_{heavy_neighbors}', 'sp2'

        # N-P coupling
        if row.total_neighbors == 3 and 'P' in neighbors.elements:
            return f'N.3p_{heavy_neighbors}', 'sp3'
        
        # Amides and imides
        amide_count = sum(
            1 for x, y, z in zip(neighbors.elements, neighbors.num_oxygens, neighbors.valences)
            if x == 'C' and y == 1 and z == 3
        )
        if amide_count == 1:
            return f'N.am_{heavy_neighbors}', 'sp2'
        if amide_count == 2:
            return f'N.im_{heavy_neighbors}', 'sp2'
        
        # N-O bonds
        if row.num_oxygens == 2:
            return f'N.o2_{heavy_neighbors}', 'sp2'  # Nitro
        if row.num_oxygens == 1:
            atom_type, hybridization = self._type_n_oxide(row, neighbors, heavy_neighbors)
            if np.any(mol_data.adj_matrix_heavy_bonds[idx, :] == 2):
                hybridization = 'sp2'
            return atom_type, hybridization
        
        # Ring systems
        if biggest_ring is not None:
            atom_type, hybridization = self._type_ring_nitrogen(idx, row, neighbors, biggest_ring, mol_data)
            return atom_type, hybridization
        
        # N-N coupling
        if row.num_nitrogens > 0:
            if np.any(mol_data.adj_matrix_heavy_bonds[idx, :] == 2):
                return f'N.2n_{heavy_neighbors}', 'sp2'  # Azo
            return f'N.3n_{heavy_neighbors}', 'sp3'  # Hydrazine
        
        # Carbon-only neighbors
        if set(neighbors.elements) == {'C'}:
            atom_type, hybridization = self._type_amine(row, idx, neighbors, mol_data)
            return atom_type, hybridization
        
        return f'{row.sybyl_type}_{heavy_neighbors}', 'sp3'
    
    def _type_n_oxide(self, row, neighbors: NeighborInfo, heavy_neighbors: int) -> str:
        """Type N-O containing nitrogen."""
        # Check for hydroxamic acid
        if any(x == 'C' and y == 1 and z == 3 
               for x, y, z in zip(neighbors.elements, neighbors.num_oxygens, neighbors.valences)):
            return f'N.ohac_{heavy_neighbors}', 'sp2'
        
        # What is the hybridization of N?
        if row.total_neighbors == 4:
            hybridization = 'sp3'
        else:
            hybridization = 'sp2'

        return f'N.oh_{heavy_neighbors}', hybridization  # Nitroso or N-oxide
    
    def _type_ring_nitrogen(self, idx, row, neighbors: NeighborInfo,
                            ring: Set[int], mol_data: MoleculeData) -> str:
        """Type ring nitrogen (pyridine, pyrrole, etc.)."""
        # Try to get valence column, fall back to total_neighbors
        ring_valences = mol_data.df.loc[list(ring), 'total_neighbors'].tolist()
        
        ring_length = len(ring)
        heavy_neighbors = row.heavy_neighbors
        
        # Aromatic ring (no sp3 atoms)
        if all(x < 4 for x in ring_valences):
            non_c = next((x for x in neighbors.elements if x != 'C'), None)
            suffix = 'p' if row.total_neighbors != 2 else ''  # Protonated
            
            if non_c:
                return f'N.ar{ring_length}{non_c.lower()}{suffix}_{heavy_neighbors}', 'sp2'
            return f'N.ar{ring_length}{suffix}_{heavy_neighbors}', 'sp2'
        
        # Non-aromatic ring
        if np.any(mol_data.adj_matrix_heavy_bonds[idx, :] == 2):
            hybridization = 'sp2'
        else:
            hybridization = 'sp3'
        return f'N.r{ring_length}_{heavy_neighbors}', hybridization
    
    def _type_amine(self, row, idx: int, neighbors: NeighborInfo, mol_data: MoleculeData) -> str:
        """Type amine nitrogen with carbon neighbors only."""
        heavy_neighbors = row.heavy_neighbors
        
        # Check for aromatic neighbor (aniline-type)
        for list_idx, c_idx in enumerate(neighbors.indices):
            if neighbors.valences[list_idx] == 3:
                ring = mol_data.ring_info.atom_to_rings.get(c_idx, None)
                if ring is not None and len(ring) in [5, 6]:
                    return f'N.aa_{heavy_neighbors}', 'sp2'  # Aromatic amine
        
        # Check hybridization by bond order
        if np.any(mol_data.adj_matrix_heavy_bonds[idx, :] == 2):
            return f'N.2c_{heavy_neighbors}', 'sp2'  # Imine
        return f'N.3c_{heavy_neighbors}', 'sp3'  # Aliphatic amine
