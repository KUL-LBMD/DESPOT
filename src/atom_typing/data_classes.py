"""
Data structures for atom typing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import numpy as np
import networkx as nx
import pandas as pd


@dataclass
class NeighborInfo:
    """Structured neighbor information for an atom."""
    elements: List[str]
    coords: List[List[float]]
    valences: List[int]
    num_hydrogens: List[int]
    num_heavy_neighbors: List[int]
    num_oxygens: List[int]
    num_nitrogens: List[int]
    indices: np.ndarray

@dataclass
class RingInfo:
    """Pre-computed ring information for fast lookup."""
    sssr: List[Set[int]]
    atom_to_rings: Dict[int, List[Set[int]]]
    ring_graphs: Dict[int, nx.Graph]
    
    @classmethod
    def from_adjacency_matrix(cls, adj_matrix: np.ndarray) -> 'RingInfo':
        """Build ring info with pre-computed lookups."""
        G = nx.from_numpy_array(adj_matrix)
        cycles = nx.minimum_cycle_basis(G)
        sssr = [set(c) for c in sorted(cycles, key=len, reverse=True) if len(c) <= 6]
        
        atom_to_rings: Dict[int, List[Set[int]]] = {}
        ring_graphs: Dict[int, nx.Graph] = {}
        
        for ring in sssr:
            ring_key = hash(frozenset(ring))
            
            # Build subgraph once per ring
            subgraph = nx.Graph()
            ring_list = list(ring)
            for i in ring_list:
                for j in ring_list:
                    if adj_matrix[i, j] > 0:
                        subgraph.add_edge(i, j)
            ring_graphs[ring_key] = subgraph
            
            # Map atoms to their rings
            for atom_idx in ring:
                if atom_idx not in atom_to_rings:
                    atom_to_rings[atom_idx] = ring
        
        return cls(sssr=sssr, atom_to_rings=atom_to_rings, ring_graphs=ring_graphs)
    
    def get_biggest_ring(self, atom_idx: int) -> Optional[Set[int]]:
        """Get the largest ring containing this atom (O(1) lookup)."""
        ring = self.atom_to_rings.get(atom_idx)
        return ring if ring else None
    
    def get_ring_graph(self, ring: Set[int]) -> nx.Graph:
        """Get pre-built subgraph for a ring."""
        return self.ring_graphs[hash(frozenset(ring))]

@dataclass  
class MoleculeData:
    """Container for all molecule data needed for atom typing."""
    df: pd.DataFrame
    adj_matrix_heavy: np.ndarray
    adj_matrix_heavy_bonds: np.ndarray
    ring_info: RingInfo
    neighbor_cache: Dict[int, NeighborInfo] = field(default_factory=dict)
    
    def get_neighbors(self, idx: int) -> NeighborInfo:
        """Get neighbor info with caching."""
        if idx not in self.neighbor_cache:
            self.neighbor_cache[idx] = self._compute_neighbors(idx)
        return self.neighbor_cache[idx]
    
    def _compute_neighbors(self, idx: int) -> NeighborInfo:
        """Compute neighbor information for an atom."""
        neighbors = np.where(self.adj_matrix_heavy[idx] > 0)[0]
        df = self.df
        
        return NeighborInfo(
            elements=df.loc[neighbors, 'element'].tolist(),
            coords=df.loc[neighbors, ['x', 'y', 'z']].values.tolist(),
            valences=df.loc[neighbors, 'total_neighbors'].values.tolist(),
            num_hydrogens=df.loc[neighbors, 'num_hydrogens'].values.tolist(),
            num_heavy_neighbors=df.loc[neighbors, 'heavy_neighbors'].values.tolist(),
            num_oxygens=df.loc[neighbors, 'num_oxygens'].values.tolist(),
            num_nitrogens=df.loc[neighbors, 'num_nitrogens'].values.tolist(),
            indices=neighbors
        )
