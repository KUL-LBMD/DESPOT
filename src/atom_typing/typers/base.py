"""
Element-specific atom typers using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd
import networkx as nx

from src.atom_typing.data_classes import MoleculeData, NeighborInfo


class ElementTyper(ABC):
    """Base class for element-specific atom typing."""
    
    @abstractmethod
    def type_atom(self, idx: int, row, mol_data: MoleculeData) -> Tuple[str, Optional[str]]:
        """
        Determine the atom type for a given atom.
        
        Parameters
        ----------
        idx : int
            Atom index in the molecule DataFrame
        row : object
            Row data for this atom (can be namedtuple or proxy object)
        mol_data : MoleculeData
            Full molecule data container
        
        Returns
        -------
        Tuple[str, Optional[str]]
            Atom type string and hybridization
        """
        pass


class RingPositionCalculator:
    """Efficiently calculate ring positions (ortho/meta/para) using pre-built graphs."""
    
    @staticmethod
    def get_position(atom_idx: int, ring: Set[int], heteroatom_indices: List[int], 
                     ring_graph: nx.Graph) -> str:
        """
        Determine if atom is ortho, meta, or para to nearest heteroatom.
        
        Parameters
        ----------
        atom_idx : int
            Index of the atom to classify
        ring : Set[int]
            Set of atom indices in the ring
        heteroatom_indices : List[int]
            Indices of heteroatoms in the ring
        ring_graph : nx.Graph
            Pre-built subgraph of the ring
        
        Returns
        -------
        str
            'ortho', 'meta', or 'para'
        """
        min_dist = float('inf')
        
        for het_idx in heteroatom_indices:
            try:
                dist = nx.shortest_path_length(ring_graph, atom_idx, het_idx)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
        
        if min_dist == 1:
            return 'ortho'
        elif min_dist == 2:
            return 'meta'
        return 'para'
