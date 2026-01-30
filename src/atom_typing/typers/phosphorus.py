"""
Phosphorus atom typing logic.
"""

import pandas as pd

from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.data_classes import MoleculeData, NeighborInfo

class PhosphorusTyper(ElementTyper):
    """Atom typing logic for phosphorus atoms."""
    
    def type_atom(self, idx, row, mol_data: MoleculeData) -> str:
        """
        Type phosphorus based on oxygen neighbor count.
        
        Covers phosphates, phosphonates, phosphines, etc.
        """
        return f"P.o{row.num_oxygens}_{row.heavy_neighbors}", 'sp3'
