"""
Molecule data processing and preparation.
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
    
    def process(self, df: pd.DataFrame, adj_matrix: np.ndarray) -> MoleculeData:
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
        # Compute neighbor counts
        adj_binary = (adj_matrix > 0).astype(int)
        h_mask = (df['sybyl_type'] == 'H').values.astype(int)
        num_heavy_neighbors = np.sum(adj_binary * (1 - h_mask[np.newaxis, :]), axis=1)
        num_hydrogens = np.sum(adj_binary * h_mask[np.newaxis, :], axis=1)
        
        df = df.copy()
        df['heavy_neighbors'] = num_heavy_neighbors
        df['num_hydrogens'] = num_hydrogens
        df['total_neighbors'] = num_heavy_neighbors + num_hydrogens
        df['element'] = df['sybyl_type'].apply(lambda x: self.element_dict.get(x, x))
        df['resname'] = df['subst_name'].str[:3]
        
        # Filter to heavy atoms only
        heavy_mask = (h_mask == 0)
        adj_heavy = adj_binary[np.ix_(heavy_mask, heavy_mask)]
        adj_heavy_bonds = adj_matrix[np.ix_(heavy_mask, heavy_mask)]
        subset = df[heavy_mask].reset_index(drop=True).copy()

        # Check against covalent bonds
        lig_mask = (subset['resname'] == 'LIG').values.astype(bool)
        has_covalent_bond = np.any(adj_heavy[lig_mask][:, ~lig_mask])
        if has_covalent_bond:
            return None
        
        # Compute oxygen/nitrogen neighbor counts
        self._add_heteroatom_counts(subset, adj_heavy)
        
        # Build ring info with intra-residue filtering
        adj_intrares = self._get_intraresidue_adjacency(subset, adj_heavy)
        ring_info = RingInfo.from_adjacency_matrix(adj_intrares)
        
        return MoleculeData(
            df=subset,
            adj_matrix_heavy=adj_heavy,
            adj_matrix_heavy_bonds=adj_heavy_bonds,
            ring_info=ring_info
        )
    
    def _add_heteroatom_counts(self, df: pd.DataFrame, adj_heavy: np.ndarray) -> None:
        """Add oxygen and nitrogen neighbor counts to DataFrame (in-place)."""
        oxygen_mask = (df['element'] == 'O').values.astype(int)
        nitrogen_mask = (df['element'] == 'N').values.astype(int)
        df['num_oxygens'] = np.sum(adj_heavy * oxygen_mask[np.newaxis, :], axis=1)
        df['num_nitrogens'] = np.sum(adj_heavy * nitrogen_mask[np.newaxis, :], axis=1)
    
    def _get_intraresidue_adjacency(self, df: pd.DataFrame, 
                                     adj_heavy: np.ndarray) -> np.ndarray:
        """Get adjacency matrix filtered to intra-residue bonds only."""
        res_nums = df['subst_id'].values
        res_mask = (res_nums[:, np.newaxis] == res_nums[np.newaxis, :]).astype(int)
        return adj_heavy * res_mask
