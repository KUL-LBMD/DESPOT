"""
File I/O for molecular file formats - OPTIMIZED VERSION.
Key optimizations:
- Uses numpy for bond matrix construction
- Pre-compiled bond order mapping
"""

from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
import io

# Standard SYBYL type to element mapping
ELEMENT_DICT: Dict[str, str] = {
    'C.1': 'C', 'C.2': 'C', 'C.3': 'C', 'C.ar': 'C', 'C.cat': 'C',
    'N.1': 'N', 'N.2': 'N', 'N.3': 'N', 'N.4': 'N', 'N.am': 'N', 'N.ar': 'N', 'N.pl3': 'N',
    'O.2': 'O', 'O.3': 'O', 'O.co2': 'O',
    'P.3': 'P',
    'S.2': 'S', 'S.3': 'S', 'S.o': 'S', 'S.o2': 'S'
}

# Pre-defined bond order mapping for faster lookup
BOND_ORDER_MAP: Dict[str, int] = {
    'ar': 2, 'am': 2, '1': 1, '2': 2, '3': 3
}

def _fix_mol2(path):
    with open(path) as f:
        lines = f.readlines()

    in_atom = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('@<TRIPOS>'):
            in_atom = stripped == '@<TRIPOS>ATOM'
            continue
        if in_atom and stripped:
            parts = line.split()
            if len(parts) == 8:                       # atom_name missing
                # parts = [id, x, y, z, type, subst_id, subst_name, charge]
                parts.insert(1, parts[4])             # use atom_type as a stand-in name
                lines[i] = '  '.join(parts) + '\n'
    return io.StringIO(''.join(lines))

def read_mol2_safe(path):
    pmol = PandasMol2()
    pmol.read_mol2_from_list(
        mol2_lines=_fix_mol2(path).readlines(),
        mol2_code=path,
    )
    return pmol


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
        try:
            df = PandasMol2().read_mol2(str(self.file_path)).df
        except Exception as e:
            df = read_mol2_safe(str(self.file_path)).df

        df.rename(columns={'atom_type': 'sybyl_type'}, inplace=True)
        
        adj_matrix = self._parse_bonds(len(df))
        
        return df, adj_matrix
    
    def _parse_bonds(self, num_atoms: int) -> np.ndarray:
        """Parse bond block from MOL2 file.
        
        OPTIMIZED: Pre-allocates arrays, uses faster parsing.
        """
        # Pre-allocate with int8 for memory efficiency
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int8)
        
        # Collect bonds first, then fill matrix (allows batch operations)
        bonds = []
        
        in_bond_block = False
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.startswith('@<TRIPOS>BOND'):
                    in_bond_block = True
                    continue
                
                if line.startswith('@<TRIPOS>SUBSTRUCTURE'):
                    break
                
                if in_bond_block:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    
                    try:
                        node_1 = int(parts[1]) - 1
                        node_2 = int(parts[2]) - 1
                    except ValueError:
                        continue
                    
                    bond_type = parts[3]
                    bond_order = BOND_ORDER_MAP.get(bond_type)
                    if bond_order is None:
                        try:
                            bond_order = int(bond_type)
                        except ValueError:
                            bond_order = 1
                    
                    bonds.append((node_1, node_2, bond_order))
        
        # Fill matrix from collected bonds
        for n1, n2, order in bonds:
            adj_matrix[n1, n2] = order
            adj_matrix[n2, n1] = order
        
        return adj_matrix
