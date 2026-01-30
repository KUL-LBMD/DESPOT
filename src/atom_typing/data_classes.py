"""
Data structures for atom typing - OPTIMIZED VERSION.
Key optimizations:
- Pre-computed numpy arrays for all properties
- Vectorized neighbor lookups
- Efficient ring graph caching
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import networkx as nx
import pandas as pd


@dataclass
class NeighborInfo:
    """Structured neighbor information for an atom."""
    elements: List[str]
    coords: np.ndarray  # Changed to numpy array for faster access
    valences: np.ndarray
    num_hydrogens: np.ndarray
    num_heavy_neighbors: np.ndarray
    num_oxygens: np.ndarray
    num_nitrogens: np.ndarray
    indices: np.ndarray


@dataclass
class RingInfo:
    """Pre-computed ring information for fast lookup."""
    sssr: List[Set[int]]
    atom_to_rings: Dict[int, Set[int]]  # Maps to single largest ring
    ring_graphs: Dict[int, nx.Graph]
    
    @classmethod
    def from_adjacency_matrix(cls, adj_matrix: np.ndarray) -> 'RingInfo':
        """Build ring info with pre-computed lookups."""
        G = nx.from_numpy_array(adj_matrix)
        cycles = nx.minimum_cycle_basis(G)
        # Sort by length descending so we process largest rings first
        sssr = [set(c) for c in sorted(cycles, key=len, reverse=True) if len(c) <= 6]
        
        atom_to_rings: Dict[int, Set[int]] = {}
        ring_graphs: Dict[int, nx.Graph] = {}
        
        for ring in sssr:
            ring_key = hash(frozenset(ring))
            
            # Build subgraph using numpy indexing (faster)
            ring_list = np.array(list(ring))
            submatrix = adj_matrix[np.ix_(ring_list, ring_list)]
            subgraph = nx.from_numpy_array(submatrix)
            # Relabel nodes to original indices
            mapping = {i: ring_list[i] for i in range(len(ring_list))}
            subgraph = nx.relabel_nodes(subgraph, mapping)
            ring_graphs[ring_key] = subgraph
            
            # Map atoms to their largest ring (first one processed due to sorting)
            for atom_idx in ring:
                if atom_idx not in atom_to_rings:
                    atom_to_rings[atom_idx] = ring
        
        return cls(sssr=sssr, atom_to_rings=atom_to_rings, ring_graphs=ring_graphs)
    
    def get_biggest_ring(self, atom_idx: int) -> Optional[Set[int]]:
        """Get the largest ring containing this atom (O(1) lookup)."""
        return self.atom_to_rings.get(atom_idx)
    
    def get_ring_graph(self, ring: Set[int]) -> nx.Graph:
        """Get pre-built subgraph for a ring."""
        return self.ring_graphs[hash(frozenset(ring))]


@dataclass  
class MoleculeData:
    """Container for all molecule data needed for atom typing.
    
    OPTIMIZED: Pre-computes numpy arrays for fast loop access.
    """
    df: pd.DataFrame
    adj_matrix_heavy: np.ndarray
    adj_matrix_heavy_bonds: np.ndarray
    ring_info: RingInfo
    neighbor_cache: Dict[int, NeighborInfo] = field(default_factory=dict)
    
    # Pre-computed arrays (set in __post_init__)
    _elements: np.ndarray = field(default=None, repr=False)
    _coords: np.ndarray = field(default=None, repr=False)
    _total_neighbors: np.ndarray = field(default=None, repr=False)
    _num_hydrogens: np.ndarray = field(default=None, repr=False)
    _heavy_neighbors: np.ndarray = field(default=None, repr=False)
    _num_oxygens: np.ndarray = field(default=None, repr=False)
    _num_nitrogens: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """Pre-compute numpy arrays for fast access."""
        self._elements = self.df['element'].values
        self._coords = self.df[['x', 'y', 'z']].values
        self._total_neighbors = self.df['total_neighbors'].values
        self._num_hydrogens = self.df['num_hydrogens'].values
        self._heavy_neighbors = self.df['heavy_neighbors'].values
        self._num_oxygens = self.df['num_oxygens'].values
        self._num_nitrogens = self.df['num_nitrogens'].values
    
    def get_neighbors(self, idx: int) -> NeighborInfo:
        """Get neighbor info with caching."""
        if idx not in self.neighbor_cache:
            self.neighbor_cache[idx] = self._compute_neighbors(idx)
        return self.neighbor_cache[idx]
    
    def _compute_neighbors(self, idx: int) -> NeighborInfo:
        """Compute neighbor information using pre-computed arrays (FAST)."""
        neighbors = np.where(self.adj_matrix_heavy[idx] > 0)[0]
        
        return NeighborInfo(
            elements=self._elements[neighbors].tolist(),
            coords=self._coords[neighbors],  # Keep as numpy array
            valences=self._total_neighbors[neighbors],
            num_hydrogens=self._num_hydrogens[neighbors],
            num_heavy_neighbors=self._heavy_neighbors[neighbors],
            num_oxygens=self._num_oxygens[neighbors],
            num_nitrogens=self._num_nitrogens[neighbors],
            indices=neighbors
        )
    
    def precompute_all_neighbors(self) -> None:
        """Pre-compute neighbors for all atoms at once."""
        n_atoms = len(self.df)
        for idx in range(n_atoms):
            if idx not in self.neighbor_cache:
                self.neighbor_cache[idx] = self._compute_neighbors(idx)
