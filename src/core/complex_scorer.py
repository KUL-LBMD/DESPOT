"""
Numba-accelerated DESPOT Scorer for maximum performance.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from numba import njit
from numba.typed import Dict
from numba import types

from src.config import DATA_DIR

# ============================================================================
# Numba-compiled interpolation kernels
# ============================================================================

@njit(fastmath=True, cache=True)
def interp_1d(scores, r_grid, n_r):
    """
    Linear interpolation on 1D grid.

    Parameters
    ----------
    scores: array of shape [r], 1D grid
    r_grid [float]: position on grid
    n_r [int]: marks outer boundary
    """

    if r_grid < 0 or r_grid >= n_r - 1:
        return 0.0
    
    i0 = int(r_grid)
    i1 = min(i0 + 1, n_r - 1)
    t = r_grid - i0
    
    return scores[i0] * (1 - t) + scores[i1] * t


@njit(fastmath=True, cache=True)
def interp_2d(scores, r_grid, theta_grid, n_r, n_theta):
    """
    Bilinear interpolation on 2D grid.

    Parameters
    ----------
    scores: array of shape [r, theta], 2D grid
    r_grid, theta_grid [float]: position on grid
    n_r, n_theta [int]: marks outer boundary
    """

    if r_grid < 0 or r_grid >= n_r - 1:
        return 0.0
    if theta_grid < 0 or theta_grid >= n_theta - 1:
        return 0.0
    
    i0 = int(r_grid)
    i1 = min(i0 + 1, n_r - 1)
    j0 = int(theta_grid)
    j1 = min(j0 + 1, n_theta - 1)
    
    tr = r_grid - i0
    tt = theta_grid - j0
    
    c00 = scores[i0, j0]
    c01 = scores[i0, j1]
    c10 = scores[i1, j0]
    c11 = scores[i1, j1]
    
    return (c00 * (1 - tr) * (1 - tt) +
            c10 * tr * (1 - tt) +
            c01 * (1 - tr) * tt +
            c11 * tr * tt)


@njit(fastmath=True, cache=True)
def interp_3d(scores, r_grid, theta_grid, phi_grid, n_r, n_theta, n_phi):
    """
    Trilinear interpolation on 3D grid.

    Parameters 
    ----------
    scores: array of shape [r, theta, phi], 3D grid
    r_grid, theta_grid, phi_grid [float]: position on grid
    n_r, n_theta, n_phi [int]: marks outer boundary
    """

    if r_grid < 0 or r_grid >= n_r - 1:
        return 0.0
    if theta_grid < 0 or theta_grid >= n_theta - 1:
        return 0.0
    if phi_grid < 0 or phi_grid >= n_phi - 1:
        return 0.0
    
    i0 = int(r_grid)
    i1 = min(i0 + 1, n_r - 1)
    j0 = int(theta_grid)
    j1 = min(j0 + 1, n_theta - 1)
    k0 = int(phi_grid)
    k1 = min(k0 + 1, n_phi - 1)
    
    tr = r_grid - i0
    tt = theta_grid - j0
    tp = phi_grid - k0
    
    c000 = scores[i0, j0, k0]
    c001 = scores[i0, j0, k1]
    c010 = scores[i0, j1, k0]
    c011 = scores[i0, j1, k1]
    c100 = scores[i1, j0, k0]
    c101 = scores[i1, j0, k1]
    c110 = scores[i1, j1, k0]
    c111 = scores[i1, j1, k1]
    
    return (c000 * (1 - tr) * (1 - tt) * (1 - tp) +
            c100 * tr * (1 - tt) * (1 - tp) +
            c010 * (1 - tr) * tt * (1 - tp) +
            c110 * tr * tt * (1 - tp) +
            c001 * (1 - tr) * (1 - tt) * tp +
            c101 * tr * (1 - tt) * tp +
            c011 * (1 - tr) * tt * tp +
            c111 * tr * tt * tp)


# ============================================================================
# Core scoring kernels
# ============================================================================

@njit(fastmath=True, cache=True)
def score_1d_kernel(
    prot_indices,      # int32[:] - indices of 1D protein atoms
    prot_type_indices, # int32[:] - type index for each protein atom
    prot_coords,       # float32[:, 3]
    lig_coords,        # float32[:, 3]
    lig_type_indices,  # int32[:] - type index for each ligand atom
    scores_1d,         # float32[n_prot_types, n_lig_types, n_r]
    r_min, r_step,
    n_r,
    b_factors          # float32[:] - output, modified in place
):
    """Score all 1D interactions using numba."""
    n_prot = len(prot_indices)
    n_lig = len(lig_coords)
    
    # Process each protein atom in parallel
    for p_idx in range(n_prot):
        i = prot_indices[p_idx]
        p_type_idx = prot_type_indices[p_idx]
        px, py, pz = prot_coords[i, 0], prot_coords[i, 1], prot_coords[i, 2]
        
        for j in range(n_lig):
            l_type_idx = lig_type_indices[j]
            if l_type_idx < 0:
                continue
            
            # Compute distance
            dx = lig_coords[j, 0] - px
            dy = lig_coords[j, 1] - py
            dz = lig_coords[j, 2] - pz
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Convert to grid coordinates
            r_grid = (dist - r_min) / r_step
            
            # Interpolate and accumulate (atomic add for thread safety)
            score = interp_1d(scores_1d[p_type_idx, l_type_idx, :], r_grid, n_r)
            # Note: In practice, for thread safety we'd need atomic operations
            # For now, assuming serial accumulation or single-threaded execution
            b_factors[j] += score


@njit(fastmath=True, cache=True)
def score_2d_kernel(
    prot_indices,
    prot_type_indices,
    prot_coords,
    v1_arr,
    lig_coords,
    lig_type_indices,
    scores_2d,
    r_min, r_step, angular_step,
    n_r, n_theta,
    b_factors
):
    """Score all 2D interactions."""
    n_prot = len(prot_indices)
    n_lig = len(lig_coords)
    
    for p_idx in range(n_prot):
        i = prot_indices[p_idx]
        p_type_idx = prot_type_indices[p_idx]
        px, py, pz = prot_coords[i, 0], prot_coords[i, 1], prot_coords[i, 2]
        v1x, v1y, v1z = v1_arr[i, 0], v1_arr[i, 1], v1_arr[i, 2]
        
        for j in range(n_lig):
            l_type_idx = lig_type_indices[j]
            if l_type_idx < 0:
                continue
            
            # Compute interaction vector
            dx = lig_coords[j, 0] - px
            dy = lig_coords[j, 1] - py
            dz = lig_coords[j, 2] - pz
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < 1e-10:
                continue
            
            # Compute theta
            cos_theta = (dx*v1x + dy*v1y + dz*v1z) / dist
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = np.arccos(cos_theta)
            
            # Convert to grid coordinates
            r_grid = (dist - r_min) / r_step
            theta_grid = theta / angular_step
            
            score = interp_2d(scores_2d[p_type_idx, l_type_idx, :, :], 
                             r_grid, theta_grid, n_r, n_theta)
            b_factors[j] += score


@njit(fastmath=True, cache=True)
def score_3d_kernel(
    prot_indices,
    prot_type_indices,
    prot_coords,
    v1_arr, v2_arr, v3_arr,
    lig_coords,
    lig_type_indices,
    scores_3d,
    r_min, r_step, angular_step,
    n_r, n_theta, n_phi,
    b_factors
):
    """Score all 3D interactions."""
    n_prot = len(prot_indices)
    n_lig = len(lig_coords)
    
    for p_idx in range(n_prot):
        i = prot_indices[p_idx]
        p_type_idx = prot_type_indices[p_idx]
        px, py, pz = prot_coords[i, 0], prot_coords[i, 1], prot_coords[i, 2]
        v1x, v1y, v1z = v1_arr[i, 0], v1_arr[i, 1], v1_arr[i, 2]
        v2x, v2y, v2z = v2_arr[i, 0], v2_arr[i, 1], v2_arr[i, 2]
        v3x, v3y, v3z = v3_arr[i, 0], v3_arr[i, 1], v3_arr[i, 2]
        
        for j in range(n_lig):
            l_type_idx = lig_type_indices[j]
            if l_type_idx < 0:
                continue
            
            # Compute interaction vector
            dx = lig_coords[j, 0] - px
            dy = lig_coords[j, 1] - py
            dz = lig_coords[j, 2] - pz
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < 1e-10:
                continue
            
            # Compute theta (with abs for symmetry)
            cos_theta = abs(dx*v1x + dy*v1y + dz*v1z) / dist
            cos_theta = min(1.0, cos_theta)
            theta = np.arccos(cos_theta)
            
            # Compute phi
            proj_v2 = dx*v2x + dy*v2y + dz*v2z
            proj_v3 = abs(dx*v3x + dy*v3y + dz*v3z)
            phi = np.arctan2(proj_v3, proj_v2)
            
            # Convert to grid coordinates
            r_grid = (dist - r_min) / r_step
            theta_grid = theta / angular_step
            phi_grid = phi / angular_step
            
            score = interp_3d(scores_3d[p_type_idx, l_type_idx, :, :, :],
                             r_grid, theta_grid, phi_grid,
                             n_r, n_theta, n_phi)
            b_factors[j] += score

# ============================================================================
# Main scorer class
# ============================================================================

class DESPOT_Scorer:
    """
    Numba-accelerated DESPOT scorer.
    
    First call will be slower due to JIT compilation.
    Subsequent calls will be much faster.
    """

    def __init__(self):
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')
        colname1, colname2 = 'total_occurrence', 'total_occurrence'

        self.types_list_1d = counts_df.loc[
            (counts_df['local_reference_frame'] == 'Isotropic') & (counts_df[colname1] > 1000),
            'atom_type'
        ].dropna().unique().tolist()

        self.types_list_2d = counts_df.loc[
            (counts_df['local_reference_frame'] == 'Axial') & (counts_df[colname1] > 1000),
            'atom_type'
        ].dropna().unique().tolist()

        self.types_list_3d = counts_df.loc[
            (counts_df['local_reference_frame'] == 'Anisotropic') & (counts_df[colname1] > 1000),
            'atom_type'
        ].dropna().unique().tolist()

        self.ligand_types_list = counts_df.loc[
            (counts_df[colname2] > 500),
            'atom_type'
        ].dropna().unique().tolist()

        # Type mappings
        self.prot_type_to_idx_1d = {t: i for i, t in enumerate(self.types_list_1d)}
        self.prot_type_to_idx_2d = {t: i for i, t in enumerate(self.types_list_2d)}
        self.prot_type_to_idx_3d = {t: i for i, t in enumerate(self.types_list_3d)}
        self.ligand_type_to_idx = {t: i for i, t in enumerate(self.ligand_types_list)}

        self.types_set_1d = set(self.types_list_1d)
        self.types_set_2d = set(self.types_list_2d)
        self.types_set_3d = set(self.types_list_3d)

        # Load scores as contiguous float32 arrays
        loaded = np.load(DATA_DIR / 'potentials' / 'despot_scores.npz')
        self.scores_1d = np.ascontiguousarray(loaded['scores_1d'].astype(np.float32))
        self.scores_2d = np.ascontiguousarray(loaded['scores_2d'].astype(np.float32))
        self.scores_3d = np.ascontiguousarray(loaded['scores_3d'].astype(np.float32))

        # Grid parameters
        self.r_min = np.float32(1.0)
        self.r_step = np.float32(0.1)
        self.angular_step = np.float32(np.deg2rad(3.0))
        
        self.n_r = self.scores_1d.shape[2]
        self.n_theta_2d = self.scores_2d.shape[3]
        self.n_theta_3d = self.scores_3d.shape[3]
        self.n_phi = self.scores_3d.shape[4]

        # Warm up JIT compilation
        self._warmup()

    def _warmup(self):
        """Force JIT compilation with dummy data."""
        dummy_coords = np.zeros((1, 3), dtype=np.float32)
        dummy_idx = np.zeros(1, dtype=np.int32)
        dummy_b = np.zeros(1, dtype=np.float32)
        
        # These calls will trigger compilation
        try:
            score_1d_kernel(
                dummy_idx, dummy_idx, dummy_coords, dummy_coords, dummy_idx,
                self.scores_1d, self.r_min, self.r_step, self.n_r, dummy_b
            )
            score_2d_kernel(
                dummy_idx, dummy_idx, dummy_coords, dummy_coords, dummy_coords, dummy_idx,
                self.scores_2d, self.r_min, self.r_step, self.angular_step,
                self.n_r, self.n_theta_2d, dummy_b
            )
            score_3d_kernel(
                dummy_idx, dummy_idx, dummy_coords, dummy_coords, dummy_coords, dummy_coords,
                dummy_coords, dummy_idx, self.scores_3d,
                self.r_min, self.r_step, self.angular_step,
                self.n_r, self.n_theta_3d, self.n_phi, dummy_b
            )
        except:
            pass  # Ignore errors during warmup

    def score_complex(self, prot_df, lig_df):
        """Score protein-ligand complex."""
        # Extract arrays
        prot_types = prot_df['atom_type'].values
        prot_coords = np.ascontiguousarray(prot_df[['x', 'y', 'z']].values.astype(np.float32))
        v1_arr = np.ascontiguousarray(prot_df[['v1_x', 'v1_y', 'v1_z']].values.astype(np.float32))
        v2_arr = np.ascontiguousarray(prot_df[['v2_x', 'v2_y', 'v2_z']].values.astype(np.float32))
        v3_arr = np.ascontiguousarray(prot_df[['v3_x', 'v3_y', 'v3_z']].values.astype(np.float32))

        lig_types = lig_df['atom_type'].values
        lig_coords = np.ascontiguousarray(lig_df[['x', 'y', 'z']].values.astype(np.float32))
        n_lig = len(lig_coords)

        b_factors = np.zeros(n_lig, dtype=np.float32)

        # Map ligand types to indices
        lig_type_indices = np.array([
            self.ligand_type_to_idx.get(t, -1) for t in lig_types
        ], dtype=np.int32)

        if not (lig_type_indices >= 0).any():
            return b_factors

        # Find nearby protein atoms
        prot_tree = KDTree(prot_coords)
        neighbors = prot_tree.query_ball_point(lig_coords, r=6.0)
        flat_neighbors = [idx for sublist in neighbors for idx in sublist]

        if not flat_neighbors:
            return b_factors

        prot_indices = np.unique(flat_neighbors).astype(np.int32)

        # Separate by type
        prot_1d_list = []
        prot_1d_type_idx = []
        prot_2d_list = []
        prot_2d_type_idx = []
        prot_3d_list = []
        prot_3d_type_idx = []

        for i in prot_indices:
            p_type = prot_types[i]
            if p_type in self.types_set_1d:
                prot_1d_list.append(i)
                prot_1d_type_idx.append(self.prot_type_to_idx_1d[p_type])
            elif p_type in self.types_set_2d and not np.isnan(v1_arr[i, 0]):
                prot_2d_list.append(i)
                prot_2d_type_idx.append(self.prot_type_to_idx_2d[p_type])
            elif p_type in self.types_set_3d and not np.isnan(v2_arr[i, 0]):
                prot_3d_list.append(i)
                prot_3d_type_idx.append(self.prot_type_to_idx_3d[p_type])

        # Score 1D interactions
        if prot_1d_list:
            prot_1d_arr = np.array(prot_1d_list, dtype=np.int32)
            prot_1d_type_arr = np.array(prot_1d_type_idx, dtype=np.int32)
            score_1d_kernel(
                prot_1d_arr, prot_1d_type_arr, prot_coords, lig_coords, lig_type_indices,
                self.scores_1d, self.r_min, self.r_step, self.n_r, b_factors
            )

        # Score 2D interactions
        if prot_2d_list:
            prot_2d_arr = np.array(prot_2d_list, dtype=np.int32)
            prot_2d_type_arr = np.array(prot_2d_type_idx, dtype=np.int32)
            score_2d_kernel(
                prot_2d_arr, prot_2d_type_arr, prot_coords, v1_arr,
                lig_coords, lig_type_indices,
                self.scores_2d, self.r_min, self.r_step, self.angular_step,
                self.n_r, self.n_theta_2d, b_factors
            )

        # Score 3D interactions
        if prot_3d_list:
            prot_3d_arr = np.array(prot_3d_list, dtype=np.int32)
            prot_3d_type_arr = np.array(prot_3d_type_idx, dtype=np.int32)
            score_3d_kernel(
                prot_3d_arr, prot_3d_type_arr, prot_coords, v1_arr, v2_arr, v3_arr,
                lig_coords, lig_type_indices,
                self.scores_3d, self.r_min, self.r_step, self.angular_step,
                self.n_r, self.n_theta_3d, self.n_phi, b_factors
            )

        return b_factors


# ============================================================================
# Isotropic-only scorer
# ============================================================================

@njit(fastmath=True, cache=True)
def score_iso_kernel(
    prot_indices,
    prot_type_indices,
    prot_coords,
    lig_coords,
    lig_type_indices,
    scores_1d,
    r_min, r_step, n_r,
    b_factors
):
    """Fast isotropic scoring kernel."""
    n_prot = len(prot_indices)
    n_lig = len(lig_coords)
    
    for p_idx in range(n_prot):
        i = prot_indices[p_idx]
        p_type_idx = prot_type_indices[p_idx]
        px, py, pz = prot_coords[i, 0], prot_coords[i, 1], prot_coords[i, 2]
        
        for j in range(n_lig):
            l_type_idx = lig_type_indices[j]
            if l_type_idx < 0:
                continue
            
            dx = lig_coords[j, 0] - px
            dy = lig_coords[j, 1] - py
            dz = lig_coords[j, 2] - pz
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            r_grid = (dist - r_min) / r_step
            score = interp_1d(scores_1d[p_type_idx, l_type_idx, :], r_grid, n_r)
            b_factors[j] += score

class DESPOT_Isotropic_Scorer:
    """Numba-accelerated isotropic-only scorer."""

    def __init__(self, mode):
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')
        colname1, colname2 = 'total_occurrence', 'total_occurrence'

        if mode == 'mif':
            loaded = np.load(DATA_DIR / 'potentials' / 'despot_iso_scores.npz')
        else:
            loaded = np.load(DATA_DIR / 'potentials' / 'despot_ds_scores.npz')

        self.types_list_1d = counts_df.loc[
            (counts_df['local_reference_frame'] == 'Isotropic') & (counts_df[colname1] > 1000),
            'atom_type'
        ].dropna().unique().tolist()

        self.types_list_2d = counts_df.loc[
            (counts_df['local_reference_frame'] == 'Axial') & (counts_df[colname1] > 1000),
            'atom_type'
        ].dropna().unique().tolist()

        self.types_list_3d = counts_df.loc[
            (counts_df['local_reference_frame'] == 'Anisotropic') & (counts_df[colname1] > 1000),
            'atom_type'
        ].dropna().unique().tolist()

        self.ligand_types_list = counts_df.loc[
            (counts_df[colname2] > 500),
            'atom_type'
        ].dropna().unique().tolist()

        self.prot_types_list = self.types_list_1d + self.types_list_2d + self.types_list_3d
        self.prot_type_to_idx = {t: i for i, t in enumerate(self.prot_types_list)}
        self.ligand_type_to_idx = {t: i for i, t in enumerate(self.ligand_types_list)}
        self.prot_types_set = set(self.prot_types_list)

        self.scores_1d = np.ascontiguousarray(loaded['scores_1d'].astype(np.float32))
        
        self.r_min = np.float32(1.0)
        self.r_step = np.float32(0.1)
        self.n_r = self.scores_1d.shape[2]

    def score_complex(self, prot_df, lig_df):
        prot_types = prot_df['atom_type'].values
        prot_coords = np.ascontiguousarray(prot_df[['x', 'y', 'z']].values.astype(np.float32))

        lig_types = lig_df['atom_type'].values
        lig_coords = np.ascontiguousarray(lig_df[['x', 'y', 'z']].values.astype(np.float32))
        n_lig = len(lig_coords)

        b_factors = np.zeros(n_lig, dtype=np.float32)

        lig_type_indices = np.array([
            self.ligand_type_to_idx.get(t, -1) for t in lig_types
        ], dtype=np.int32)

        if not (lig_type_indices >= 0).any():
            return b_factors

        prot_tree = KDTree(prot_coords)
        neighbors = prot_tree.query_ball_point(lig_coords, r=6.0)
        flat_neighbors = [idx for sublist in neighbors for idx in sublist]

        if not flat_neighbors:
            return b_factors

        prot_indices_all = np.unique(flat_neighbors).astype(np.int32)

        # Filter and map types
        prot_list = []
        prot_type_idx_list = []
        for i in prot_indices_all:
            p_type = prot_types[i]
            if p_type in self.prot_types_set:
                prot_list.append(i)
                prot_type_idx_list.append(self.prot_type_to_idx[p_type])

        if not prot_list:
            return b_factors

        prot_indices = np.array(prot_list, dtype=np.int32)
        prot_type_indices = np.array(prot_type_idx_list, dtype=np.int32)

        score_iso_kernel(
            prot_indices, prot_type_indices, prot_coords,
            lig_coords, lig_type_indices,
            self.scores_1d, self.r_min, self.r_step, self.n_r,
            b_factors
        )

        return b_factors
