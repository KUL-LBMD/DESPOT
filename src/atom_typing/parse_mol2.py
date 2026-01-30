"""
Main atom typing orchestration and file conversion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import time
import os

from src.atom_typing.data_classes import MoleculeData
from src.atom_typing.io import MOL2Reader
from src.atom_typing.processing import MoleculeProcessor
from src.atom_typing.geometry import VectorBuilder
from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.typers.carbon import CarbonTyper
from src.atom_typing.typers.oxygen import OxygenTyper
from src.atom_typing.typers.nitrogen import NitrogenTyper
from src.atom_typing.typers.phosphorus import PhosphorusTyper
from src.atom_typing.typers.sulfur import SulfurTyper

class AtomTyper:
    """Main class that orchestrates atom typing using element-specific typers."""
    
    def __init__(self):
        """
        Initialize the atom typer.
        
        Parameters
        ----------
        custom_typers : Dict[str, ElementTyper], optional
            Custom typers to override defaults. Keys are element symbols.
        """
        self.element_typers: Dict[str, ElementTyper] = {
            'C': CarbonTyper(),
            'N': NitrogenTyper(),
            'O': OxygenTyper(),
            'P': PhosphorusTyper(),
            'S': SulfurTyper(),
        }
    
    def type_atoms(self, mol_data: MoleculeData) -> List[str]:
        """
        Assign atom types to all heavy atoms.
        
        Parameters
        ----------
        mol_data : MoleculeData
            Processed molecule data
        
        Returns
        -------
        List[str]
            Atom type for each heavy atom
        """
        atom_types = []
        hybridizations = []
        df = mol_data.df
        
        for row in df.itertuples():
            idx = row.Index
            element = row.element
            
            typer = self.element_typers.get(element)
            if typer:
                atom_type, hybridization = typer.type_atom(idx, row, mol_data)
            else:
                # Fallback for unknown elements
                if row.sybyl_type in ['F', 'I', 'Br', 'Cl']:
                    atom_type, hybridization = f"{row.sybyl_type}_{row.heavy_neighbors}", 'sp3'
                else:
                    atom_type, hybridization = row.sybyl_type, None
            
            atom_types.append(atom_type)
            hybridizations.append(hybridization)
        
        return atom_types, hybridizations

class MolConverter:
    """
    High-level converter for MOL2 files to typed CSV.
    
    Maintains API compatibility with original implementation.
    """
    
    def __init__(self):
        """
        Initialize the converter.
        """
        self.processor = MoleculeProcessor()
        self.typer = AtomTyper()
        self.vector_builder = VectorBuilder()
    
    def convert_mol2(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Convert a single MOL2 file to typed CSV.
        
        Parameters
        ----------
        file_path : str
            Filename (relative to input_path) of the MOL2 file
        
        Returns
        -------
        pd.DataFrame or None
            Processed DataFrame with atom types, or None if parsing failed
        """

        # Read file
        start = time.time()
        reader = MOL2Reader(file_path)
        df, adj_matrix = reader.read()
        end = time.time()
        print(f'Mol2 reading: {end - start}')

        # Control: a residue should not have more than 100 heavy atoms
        subset = df[df['sybyl_type'] != 'H']
        if subset['subst_id'].value_counts().gt(100).any():
            return None
            
        # Process molecule
        start = time.time()
        mol_data = self.processor.process(df, adj_matrix)
        end = time.time()
        print(f'MolData creation: {end - start}')
        if mol_data is None:
            return None
            
        # Type atoms
        start = time.time()
        atom_types, hybridizations = self.typer.type_atoms(mol_data)
        mol_data.df['atom_type'] = atom_types
        mol_data.df['hybridization'] = hybridizations
        end = time.time()
        print(f'Atom typing: {end - start}')

        # Define local reference frames
        start = time.time()
        v1_arr, v2_arr, v3_arr = self.vector_builder.assign_vectors(mol_data)
        mol_data.df[['v1_x', 'v1_y', 'v1_z']] = v1_arr
        mol_data.df[['v2_x', 'v2_y', 'v2_z']] = v2_arr
        mol_data.df[['v3_x', 'v3_y', 'v3_z']] = v3_arr
        end = time.time()
        print(f'Vector assignment: {end - start}')

        # Save output
        cols = ['atom_id', 'atom_name', 'atom_type', 'sybyl_type', 'subst_id', 'subst_name', 'charge', 'hybridization', 'heavy_neighbors', 'num_hydrogens', 'x', 'y', 'z', 'v1_x', 'v1_y', 'v1_z', 'v2_x', 'v2_y', 'v2_z', 'v3_x', 'v3_y', 'v3_z']
        df_out = mol_data.df[cols]

        return df_out
