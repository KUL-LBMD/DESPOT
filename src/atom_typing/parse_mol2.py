"""
Main atom typing orchestration and file conversion - OPTIMIZED VERSION.
Key optimizations:
- SINGLE PASS: Combines atom typing and vector assignment
- Pre-computed numpy arrays for all lookups
- Minimal function call overhead in hot loop
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.atom_typing.data_classes import MoleculeData, NeighborInfo
from src.atom_typing.io import MOL2Reader
from src.atom_typing.processing import MoleculeProcessor
from src.atom_typing.geometry import unit_vector, find_plane_normal
from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.typers.carbon import CarbonTyper
from src.atom_typing.typers.oxygen import OxygenTyper
from src.atom_typing.typers.nitrogen import NitrogenTyper
from src.atom_typing.typers.phosphorus import PhosphorusTyper
from src.atom_typing.typers.sulfur import SulfurTyper


class AtomTyperWithVectors:
    """
    Combined atom typing and vector assignment in a single pass.
    
    This eliminates the need to loop through atoms twice and keeps
    neighbor data hot in cache.
    """
    
    def __init__(self):
        self.element_typers: Dict[str, ElementTyper] = {
            'C': CarbonTyper(),
            'N': NitrogenTyper(),
            'O': OxygenTyper(),
            'P': PhosphorusTyper(),
            'S': SulfurTyper(),
        }
        self._halogen_types = frozenset(['F', 'I', 'Br', 'Cl'])
    
    def process_atoms(self, mol_data: MoleculeData) -> Tuple[
        List[str], List[Optional[str]], np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Assign atom types AND local reference frames in a single pass.
        
        Parameters
        ----------
        mol_data : MoleculeData
            Processed molecule data
        
        Returns
        -------
        atom_types : List[str]
        hybridizations : List[Optional[str]]
        v1_arr, v2_arr, v3_arr : np.ndarray (N, 3) each
        """
        df = mol_data.df
        n_atoms = len(df)
        
        # Pre-extract all arrays for fast access
        elements = mol_data._elements
        sybyl_types = df['sybyl_type'].values
        coords_arr = mol_data._coords
        total_neighbors_arr = mol_data._total_neighbors
        heavy_neighbors_arr = mol_data._heavy_neighbors
        num_hydrogens_arr = mol_data._num_hydrogens
        num_oxygens_arr = mol_data._num_oxygens
        num_nitrogens_arr = mol_data._num_nitrogens
        
        # Pre-allocate outputs
        atom_types = [''] * n_atoms
        hybridizations: List[Optional[str]] = [None] * n_atoms
        v1_arr = np.full((n_atoms, 3), np.nan)
        v2_arr = np.full((n_atoms, 3), np.nan)
        v3_arr = np.full((n_atoms, 3), np.nan)
        
        # Lightweight row proxy
        class RowProxy:
            __slots__ = ['Index', 'sybyl_type', 'total_neighbors', 'heavy_neighbors',
                        'element', 'num_hydrogens', 'num_oxygens', 'num_nitrogens']
        
        row = RowProxy()
        
        # Single pass through all atoms
        for idx in range(n_atoms):
            # Build row proxy
            row.Index = idx
            row.sybyl_type = sybyl_types[idx]
            row.total_neighbors = total_neighbors_arr[idx]
            row.heavy_neighbors = heavy_neighbors_arr[idx]
            row.element = elements[idx]
            row.num_hydrogens = num_hydrogens_arr[idx]
            row.num_oxygens = num_oxygens_arr[idx]
            row.num_nitrogens = num_nitrogens_arr[idx]
            
            # Get neighbors ONCE - used for both typing and vectors
            neighbors = mol_data.get_neighbors(idx)
            
            # === ATOM TYPING ===
            element = elements[idx]
            typer = self.element_typers.get(element)
            
            if typer:
                atom_type, hybridization = typer.type_atom(idx, row, mol_data)
            else:
                if row.sybyl_type in self._halogen_types:
                    atom_type = f"{row.sybyl_type}_{row.heavy_neighbors}"
                    hybridization = 'sp3'
                else:
                    atom_type = row.sybyl_type
                    hybridization = None
            
            atom_types[idx] = atom_type
            hybridizations[idx] = hybridization
            
            # === VECTOR ASSIGNMENT ===
            # Use the same neighbors and hybridization we just computed
            heavy_neighbors = row.heavy_neighbors
            coords = coords_arr[idx]
            
            if heavy_neighbors == 1 and hybridization in ('sp', 'sp3'):
                # Axial
                neighbor_coords = neighbors.coords[0]
                v1_arr[idx] = unit_vector(coords - neighbor_coords)

            elif heavy_neighbors == 1 and hybridization == 'sp2':
                neighbor_coords = neighbors.coords[0]
                neighbor_idx = neighbors.indices[0]
                
                next_neighbor_data = mol_data.get_neighbors(neighbor_idx)
                next_neighbor_coords = next_neighbor_data.coords
                
                n_next = len(next_neighbor_coords)
                if n_next in (2, 3):
                    v1 = find_plane_normal(neighbor_coords, next_neighbor_coords[:2])
                    v2 = unit_vector(coords - neighbor_coords)
                    v3 = np.cross(v1, v2)
                    v1_arr[idx] = v1
                    v2_arr[idx] = v2
                    v3_arr[idx] = v3
                else:
                    v1_arr[idx] = unit_vector(coords - neighbor_coords)

            elif heavy_neighbors == 2 and hybridization != 'sp':
                neighbor_coords = neighbors.coords
                mean_coords = np.mean(neighbor_coords, axis=0)
                v1 = find_plane_normal(coords, neighbor_coords)
                v2 = unit_vector(coords - mean_coords)
                v3 = np.cross(v1, v2)
                v1_arr[idx] = v1
                v2_arr[idx] = v2
                v3_arr[idx] = v3

            elif heavy_neighbors == 3:
                neighbor_coords = neighbors.coords
                if hybridization == 'sp3':
                    mean_coords = np.mean(neighbor_coords, axis=0)
                    v1_arr[idx] = unit_vector(coords - mean_coords)
                elif hybridization == 'sp2':
                    v1_arr[idx] = find_plane_normal(coords, neighbor_coords[:2])
        
        return atom_types, hybridizations, v1_arr, v2_arr, v3_arr

class MolConverter:
    """
    High-level converter for MOL2 files to typed CSV.
    
    Uses combined single-pass processing for optimal performance.
    """
    
    def __init__(self):
        self.processor = MoleculeProcessor()
        self.typer_with_vectors = AtomTyperWithVectors()
    
    def convert_mol2(self, file_path: str, verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Convert a single MOL2 file to typed DataFrame.
        
        Parameters
        ----------
        file_path : str
            Path to the MOL2 file
        verbose : bool
            Whether to print timing information
        
        Returns
        -------
        pd.DataFrame or None
            Processed DataFrame with atom types, or None if parsing failed
        """
        # Read file
        reader = MOL2Reader(file_path)
        df, adj_matrix = reader.read()

        # Control: a residue should not have more than 100 heavy atoms
        subset = df[df['sybyl_type'] != 'H']
        if subset['subst_id'].value_counts().gt(100).any():
            return None
            
        # Process molecule
        mol_data = self.processor.process(df, adj_matrix)        
        if mol_data is None:
            return None
        
        # === SINGLE PASS: Type atoms AND assign vectors ===
        atom_types, hybridizations, v1_arr, v2_arr, v3_arr = \
            self.typer_with_vectors.process_atoms(mol_data)
        
        mol_data.df['atom_type'] = atom_types
        mol_data.df['hybridization'] = hybridizations
        mol_data.df[['v1_x', 'v1_y', 'v1_z']] = v1_arr
        mol_data.df[['v2_x', 'v2_y', 'v2_z']] = v2_arr
        mol_data.df[['v3_x', 'v3_y', 'v3_z']] = v3_arr

        # Select output columns
        cols = [
            'atom_id', 'atom_name', 'atom_type', 'sybyl_type', 'subst_id', 
            'subst_name', 'charge', 'hybridization', 'heavy_neighbors', 
            'num_hydrogens', 'x', 'y', 'z', 
            'v1_x', 'v1_y', 'v1_z', 'v2_x', 'v2_y', 'v2_z', 'v3_x', 'v3_y', 'v3_z'
        ]
        df_out = mol_data.df[cols]

        return df_out
