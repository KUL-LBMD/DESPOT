"""
Phosphorus atom typing logic - OPTIMIZED VERSION.
"""

from typing import Optional, Tuple

from src.atom_typing.typers.base import ElementTyper
from src.atom_typing.data_classes import MoleculeData


class PhosphorusTyper(ElementTyper):
    """Atom typing logic for phosphorus atoms."""
    
    def type_atom(self, idx: int, row, mol_data: MoleculeData) -> Tuple[str, Optional[str]]:
        """
        Type phosphorus based on oxygen neighbor count.
        
        Covers phosphates, phosphonates, phosphines, etc.
        """
        return f"P.o{row.num_oxygens}_{row.heavy_neighbors}", 'sp3'
