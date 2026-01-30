"""
Geometry utilities for molecular calculations.
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
    v1 = unit_vector(atom_coords - neighbor_coords[0, :])
    v2 = unit_vector(atom_coords - neighbor_coords[1, :])
    return unit_vector(np.cross(v1, v2))

class VectorBuilder:
    """
    Assigns local refence frames to atoms: isotropic, axial or anisotropic
    """

    def __init__(self):
        pass

    def assign_vectors(self, mol_data: MoleculeData):
        """
        Assign local reference frames to all atoms.
        
        Parameters
        ----------
        mol_data : MoleculeData
            Processed molecule data
        
        Returns
        -------
        theta_arr [np.array(N, 3)]
        phi_arr [np.array(N, 3)]
        """

        df = mol_data.df
        
        num_atoms = len(df)
        v1_arr = np.full((num_atoms, 3), np.nan)
        v2_arr = np.full((num_atoms, 3), np.nan)
        v3_arr = np.full((num_atoms, 3), np.nan)

        hybridization_list = df['hybridization'].values
        heavy_neighbors_list = df['heavy_neighbors'].values
        coords_list = df[['x', 'y', 'z']].values

        for idx in range(num_atoms):
            hybridization = hybridization_list[idx]
            heavy_neighbors = heavy_neighbors_list[idx]
            coords = coords_list[idx]
            neighbors = mol_data._compute_neighbors(idx)

            if heavy_neighbors == 1 and hybridization in ['sp', 'sp3']:
                # Axial
                neighbor_coords = np.array(neighbors.coords[0])
                v1 = unit_vector(coords - neighbor_coords)
                v1_arr[idx, :] = v1

            elif heavy_neighbors == 1 and hybridization == 'sp2':
                neighbor_coords = np.array(neighbors.coords[0])
                neighbor_idx = neighbors.indices[0]
                next_neighbor_data = mol_data._compute_neighbors(neighbor_idx)
                next_neighbor_coords = np.array(next_neighbor_data.coords) # [L, 3]
                if next_neighbor_coords.shape[0] in [2,3]:

                    v1 = find_plane_normal(neighbor_coords, next_neighbor_coords)
                    v2 = unit_vector(coords - neighbor_coords)
                    v3 = np.cross(v1, v2)

                    v1_arr[idx, :] = v1
                    v2_arr[idx, :] = v2
                    v3_arr[idx, :] = v3

                elif next_neighbor_coords.shape[0] == 4:
                    v1 = unit_vector(coords - neighbor_coords)
                    v1_arr[idx, :] = v1

            elif heavy_neighbors == 2:
                if hybridization != 'sp':
                    neighbor_coords = np.array(neighbors.coords) # [2, 3]
                    mean_coords = np.mean(neighbor_coords, axis = 0) # [3]
                    v1 = find_plane_normal(coords, neighbor_coords)
                    v2 = unit_vector(coords - mean_coords)
                    v3 = np.cross(v1, v2)

                    v1_arr[idx, :] = v1
                    v2_arr[idx, :] = v2
                    v3_arr[idx, :] = v3

            elif heavy_neighbors == 3:
                neighbor_coords = np.array(neighbors.coords)
                if hybridization == 'sp3':
                    mean_coords = np.mean(neighbor_coords, axis = 0) # [3]
                    v1 = unit_vector(coords - mean_coords)
                elif hybridization == 'sp2':
                    v1 = find_plane_normal(coords, neighbor_coords)

                v1_arr[idx, :] = v1

        return v1_arr, v2_arr, v3_arr
