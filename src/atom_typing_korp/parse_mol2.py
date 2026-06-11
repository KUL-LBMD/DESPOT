"""
Main atom typing orchestration and file conversion - OPTIMIZED VERSION.
Key optimizations:
- SINGLE PASS: Combines atom typing and vector assignment
- Pre-computed numpy arrays for all lookups
- Minimal function call overhead in hot loop
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from biopandas.mol2 import PandasMol2

from src.atom_typing_korp.data_classes import MoleculeData
from src.atom_typing_korp.io import MOL2Reader
from src.atom_typing_korp.processing import MoleculeProcessor
from src.atom_typing_korp.geometry import unit_vector
from src.atom_typing_korp.typers.base import ElementTyper
from src.atom_typing_korp.typers.carbon import CarbonTyper
from src.atom_typing_korp.typers.oxygen import OxygenTyper
from src.atom_typing_korp.typers.nitrogen import NitrogenTyper
from src.atom_typing_korp.typers.phosphorus import PhosphorusTyper
from src.atom_typing_korp.typers.sulfur import SulfurTyper

class LigTyper:
    """
    Combined atom typing and vector assignment in a single pass.
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
    
    def process_atoms(self, mol_data: MoleculeData) -> List[str]:
        """
        Assign atom types AND local reference frames in a single pass.
        
        Parameters
        ----------
        mol_data : MoleculeData
            Processed molecule data
        
        Returns
        -------
        atom_types : List[str]
        """

        df = mol_data.df
        n_atoms = len(df)
        
        # Pre-extract all arrays for fast access
        elements = mol_data._elements
        sybyl_types = df['sybyl_type'].values
        total_neighbors_arr = mol_data._total_neighbors
        heavy_neighbors_arr = mol_data._heavy_neighbors
        num_hydrogens_arr = mol_data._num_hydrogens
        num_oxygens_arr = mol_data._num_oxygens
        num_nitrogens_arr = mol_data._num_nitrogens
        
        # Pre-allocate outputs
        atom_types = [''] * n_atoms
        
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
            
            
            # === ATOM TYPING ===
            element = elements[idx]
            typer = self.element_typers.get(element)
            
            if typer:
                atom_type, _ = typer.type_atom(idx, row, mol_data)
            else:
                if row.sybyl_type in self._halogen_types:
                    atom_type = f"{row.sybyl_type}_{row.heavy_neighbors}"
                else:
                    atom_type = row.sybyl_type
            
            atom_types[idx] = atom_type
        
        return atom_types
    
class ProtTyper:
    """
    Vector assignment for protein CA atoms
    """

    def __init__(self):

        self.standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
        
    def process_atoms(self, df: pd.DataFrame) -> Tuple[
        List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray
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

        df['res_name'] = df['subst_name'].str[:3]
        subset = df[(df['res_name'].isin(self.standard_aa)) & (df['atom_name'].isin({'N', 'CA', 'C'}))].copy()

        res_types, ca_list, v1_list, v2_list, v3_list = [], [], [], [], []

        for subst_name, group in subset.groupby('subst_name', sort = False):

            try:

                n_coord = group.loc[group['atom_name'] == 'N', ['x', 'y', 'z']].to_numpy()[0]
                ca_coord = group.loc[group['atom_name'] == 'CA', ['x', 'y', 'z']].to_numpy()[0]
                c_coord = group.loc[group['atom_name'] == 'C', ['x', 'y', 'z']].to_numpy()[0]

                v1 = unit_vector(n_coord + c_coord - 2*ca_coord)
                v2 = unit_vector(np.cross(v1, (n_coord - ca_coord)))
                v3 = unit_vector(np.cross(v2, v1))

                res_types.append(subst_name[:3])
                ca_list.append(ca_coord)
                v1_list.append(v1)
                v2_list.append(v2)
                v3_list.append(v3)

            except IndexError:
                pass

        coords_arr = np.array(ca_list)
        v1_arr = np.array(v1_list)
        v2_arr = np.array(v2_list)
        v3_arr = np.array(v3_list)

        return res_types, coords_arr, v1_arr, v2_arr, v3_arr

class MolConverter:
    """
    High-level converter for MOL2 files to typed CSV.
    
    Uses combined single-pass processing for optimal performance.
    """
    
    def __init__(self):
        self.processor = MoleculeProcessor()
        self.protein_typer = ProtTyper()
        self.ligand_typer = LigTyper()

    def convert_ligand(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Convert a single MOL2 file to typed DataFrame.
        
        Parameters
        ----------
        file_path : str
            Path to the MOL2 file
        
        Returns
        -------
        pd.DataFrame or None
            Processed DataFrame with atom types, or None if parsing failed
        """


        # Read file
        reader = MOL2Reader(file_path)
        df, adj_matrix = reader.read()

        # Process molecule
        mol_data = self.processor.process(df, adj_matrix)        
        if mol_data is None:
            return None
        
        # === SINGLE PASS: Type atoms ===
        atom_types = self.ligand_typer.process_atoms(mol_data)
        mol_data.df['lig_type'] = atom_types
        mol_data.df['prot_type'] = mol_data.df['lig_type']

        # Select output columns
        cols = [
            'atom_id', 'element', 'atom_name', 'prot_type', 'lig_type', 'sybyl_type', 'subst_id', 
            'subst_name', 'charge', 'heavy_neighbors', 
            'num_hydrogens', 'x', 'y', 'z', 
        ]

        df_out = mol_data.df[cols]

        return df_out
    
    def convert_receptor(self, file_path: str) -> Optional[pd.DataFrame]:
        
        mol2_df = PandasMol2().read_mol2(file_path).df
        res_types, coords_arr, v1_arr, v2_arr, v3_arr = self.protein_typer.process_atoms(mol2_df)

        new_df = pd.DataFrame()
        new_df['res_type'] = res_types
        new_df[['x', 'y', 'z']] = coords_arr
        new_df[['v1_x', 'v1_y', 'v1_z']] = v1_arr
        new_df[['v2_x', 'v2_y', 'v2_z']] = v2_arr
        new_df[['v3_x', 'v3_y', 'v3_z']] = v3_arr

        return new_df
