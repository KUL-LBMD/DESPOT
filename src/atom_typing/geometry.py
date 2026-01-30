"""
Geometry utilities for molecular calculations - OPTIMIZED VERSION.

Note: The VectorBuilder class is kept for backward compatibility.
For best performance, use AtomTyperWithVectors from parse_mol2.py
which combines typing and vector assignment in a single pass.
"""

import numpy as np
from src.atom_typing.data_classes import MoleculeData


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def find_plane_normal(atom_coords: np.ndarray, neighbor_coords: np.ndarray) -> np.ndarray:
    """
    Find the normal to a plane defined by an atom and two neighbors.
    
    Parameters
    ----------
    atom_coords : np.ndarray
        Coordinates of the central atom (3,)
    neighbor_coords : np.ndarray
        Coordinates of two neighbors (2, 3)
    
    Returns
    -------
    np.ndarray
        Unit normal vector to the plane (3,)
    """
    v1 = unit_vector(atom_coords - neighbor_coords[0])
    v2 = unit_vector(atom_coords - neighbor_coords[1])
    return unit_vector(np.cross(v1, v2))
