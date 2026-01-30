"""
File I/O for molecular file formats.
"""

from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2

# Standard SYBYL type to element mapping
ELEMENT_DICT: Dict[str, str] = {
    'C.1': 'C', 'C.2': 'C', 'C.3': 'C', 'C.ar': 'C', 'C.cat': 'C',
    'N.1': 'N', 'N.2': 'N', 'N.3': 'N', 'N.4': 'N', 'N.am': 'N', 'N.ar': 'N', 'N.pl3': 'N',
    'O.2': 'O', 'O.3': 'O', 'O.co2': 'O',
    'P.3': 'P',
    'S.2': 'S', 'S.3': 'S', 'S.o': 'S', 'S.o2': 'S'
}

class MOL2Reader:
    """Handles reading and parsing MOL2 files."""
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
    
    def read(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Read MOL2 file and return DataFrame and adjacency matrix.
        
        Returns
        -------
        df : pd.DataFrame
            Atom data with 'sybyl_type' column
        adj_matrix : np.ndarray
            Adjacency matrix with bond orders
        """
        df = PandasMol2().read_mol2(str(self.file_path)).df
        df.rename(columns={'atom_type': 'sybyl_type'}, inplace=True)
        
        adj_matrix = self._parse_bonds(len(df))
        
        return df, adj_matrix
    
    def _parse_bonds(self, num_atoms: int) -> np.ndarray:
        """Parse bond block from MOL2 file."""
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
        
        in_bond_block = False
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('@<TRIPOS>BOND'):
                    in_bond_block = True
                    continue
                
                if line.startswith('@<TRIPOS>SUBSTRUCTURE'):
                    break
                
                if in_bond_block:
                    parts = line.split()
                    try:
                        node_1 = int(parts[1]) - 1
                        node_2 = int(parts[2]) - 1
                    except (ValueError, IndexError):
                        continue
                    
                    bond_order = parts[3]
                    if bond_order in ['ar', 'am']:
                        bond_order = 2
                    else:
                        bond_order = int(bond_order)
                    
                    adj_matrix[node_1, node_2] = bond_order
                    adj_matrix[node_2, node_1] = bond_order
        
        return adj_matrix
