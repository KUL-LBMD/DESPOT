"""
Molecule data processing and preparation - OPTIMIZED VERSION.
Key optimizations:
- Fully vectorized neighbor counting
- Efficient boolean masking
- Reduced DataFrame copies
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.atom_typing.data_classes import MoleculeData, RingInfo
from src.atom_typing.io import ELEMENT_DICT


class MoleculeProcessor:
    """Prepares molecule data for atom typing."""
    
    def __init__(self, element_dict: Optional[Dict[str, str]] = None):
        self.element_dict = element_dict or ELEMENT_DICT
    
    def process(self, df: pd.DataFrame, adj_matrix: np.ndarray) -> Optional[MoleculeData]:
        """
        Process raw molecule data into MoleculeData for typing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw atom data from MOL2 file
        adj_matrix : np.ndarray
            Adjacency matrix with bond orders
        
        Returns
        -------
        MoleculeData
            Processed molecule data ready for atom typing
        """
        # Get sybyl types as numpy array for fast operations
        sybyl_types = df['sybyl_type'].values
        
        # Compute neighbor counts (vectorized)
        adj_binary = (adj_matrix > 0).astype(np.int8)  # Use int8 to save memory
        h_mask = (sybyl_types == 'H').astype(np.int8)
        heavy_mask_full = (h_mask == 0)
        
        # Vectorized neighbor counting
        num_heavy_neighbors = adj_binary @ (1 - h_mask)
        num_hydrogens = adj_binary @ h_mask
        
        # Build element mapping array (vectorized lookup)
        elements = np.array([self.element_dict.get(s, s) for s in sybyl_types])
        
        # Create processed DataFrame (single copy)
        df = df.copy()
        df['heavy_neighbors'] = num_heavy_neighbors
        df['num_hydrogens'] = num_hydrogens
        df['total_neighbors'] = num_heavy_neighbors + num_hydrogens
        df['element'] = elements
        df['resname'] = df['subst_name'].str[:3]
        
        # Filter to heavy atoms only using boolean indexing
        heavy_indices = np.where(heavy_mask_full)[0]
        adj_heavy = adj_binary[np.ix_(heavy_indices, heavy_indices)]
        adj_heavy_bonds = adj_matrix[np.ix_(heavy_indices, heavy_indices)]
        subset = df.iloc[heavy_indices].reset_index(drop=True)

        # Check against covalent bonds (vectorized)
        resnames = subset['resname'].values
        lig_mask = (resnames == 'LIG')
        
        if lig_mask.any() and (~lig_mask).any():
            # Check for covalent bonds between LIG and non-LIG
            has_covalent_bond = np.any(adj_heavy[lig_mask][:, ~lig_mask])
            if has_covalent_bond:
                return None
        
        # Compute oxygen/nitrogen neighbor counts (vectorized)
        elements_heavy = subset['element'].values
        oxygen_mask = (elements_heavy == 'O').astype(np.int8)
        nitrogen_mask = (elements_heavy == 'N').astype(np.int8)
        subset['num_oxygens'] = adj_heavy @ oxygen_mask
        subset['num_nitrogens'] = adj_heavy @ nitrogen_mask
        
        # Build ring info with intra-residue filtering
        adj_intrares = self._get_intraresidue_adjacency(subset, adj_heavy)
        ring_info = RingInfo.from_adjacency_matrix(adj_intrares)
        
        return MoleculeData(
            df=subset,
            adj_matrix_heavy=adj_heavy,
            adj_matrix_heavy_bonds=adj_heavy_bonds,
            ring_info=ring_info
        )
    
    def _get_intraresidue_adjacency(self, df: pd.DataFrame, 
                                     adj_heavy: np.ndarray) -> np.ndarray:
        """Get adjacency matrix filtered to intra-residue bonds only."""
        res_nums = df['subst_id'].values
        # Vectorized comparison using broadcasting
        res_mask = (res_nums[:, np.newaxis] == res_nums[np.newaxis, :])
        return adj_heavy * res_mask.astype(np.int8)
