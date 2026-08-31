"""
Numba-accelerated DESPOT Scorer for maximum performance.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from numba import njit

from src.config import DATA_DIR

@njit(fastmath=True, cache=True)
def interp_3d(scores, r_grid, theta_grid, phi_grid, n_r, n_theta, n_phi):
    """
    Trilinear interpolation on a 3D grid where stored values represent
    bin CENTERS, with phi treated as a periodic axis.

    The caller passes raw "bin-edge" grid coordinates:
        r_grid     = (dist  - r_min) / r_step
        theta_grid = theta  / angular_step
        phi_grid   = phi    / angular_step       (phi in [0, 2*pi))

    This function applies the -0.5 shift internally so that integer
    grid values land on bin centers.

    Returns 0.0 if r or theta fall outside the trained range.
    Phi wraps periodically.
    """

    # --- shift to bin-center coordinates ---
    r_c     = r_grid     - 0.5
    theta_c = theta_grid - 0.5
    phi_c   = phi_grid   - 0.5

    # --- hard cutoff on r and theta (these axes are not periodic) ---
    # Valid bin-center range is [-0.5, n - 0.5). Outside that, return 0.
    if r_c < -0.5 or r_c >= n_r - 0.5:
        return 0.0
    if theta_c < -0.5 or theta_c >= n_theta - 0.5:
        return 0.0

    # --- clamp r and theta to interpolatable range [0, n-1] ---
    # The thin edge regions [-0.5, 0) and (n-1, n-0.5) get nearest-neighbor
    # extrapolation, which is fine for half-bin slivers.
    if r_c < 0.0:
        r_c = 0.0
    elif r_c > n_r - 1:
        r_c = n_r - 1
    if theta_c < 0.0:
        theta_c = 0.0
    elif theta_c > n_theta - 1:
        theta_c = n_theta - 1

    i0 = int(r_c)
    i1 = i0 + 1 if i0 < n_r - 1 else i0
    j0 = int(theta_c)
    j1 = j0 + 1 if j0 < n_theta - 1 else j0

    tr = r_c - i0
    tt = theta_c - j0

    # --- phi: periodic wrap ---
    # Shift into the strictly-positive range so int() == floor().
    phi_pos = phi_c + n_phi
    k0_unwrapped = int(phi_pos)
    tp = phi_pos - k0_unwrapped
    k0 = k0_unwrapped % n_phi
    k1 = (k0 + 1) % n_phi

    # --- trilinear blend ---
    c000 = scores[i0, j0, k0]
    c001 = scores[i0, j0, k1]
    c010 = scores[i0, j1, k0]
    c011 = scores[i0, j1, k1]
    c100 = scores[i1, j0, k0]
    c101 = scores[i1, j0, k1]
    c110 = scores[i1, j1, k0]
    c111 = scores[i1, j1, k1]

    return (c000 * (1 - tr) * (1 - tt) * (1 - tp) +
            c100 * tr       * (1 - tt) * (1 - tp) +
            c010 * (1 - tr) * tt       * (1 - tp) +
            c110 * tr       * tt       * (1 - tp) +
            c001 * (1 - tr) * (1 - tt) * tp       +
            c101 * tr       * (1 - tt) * tp       +
            c011 * (1 - tr) * tt       * tp       +
            c111 * tr       * tt       * tp)


@njit(fastmath=True, cache=True)
def score_3d_kernel(
    prot_indices,
    prot_type_indices,
    prot_coords,
    v1_arr, v2_arr, v3_arr,
    lig_coords,
    lig_type_indices,
    scores,
    r_min, r_step, angular_step,
    n_r, n_theta, n_phi,
    b_factors,
):
    """
    Accumulate KORP scores into b_factors (one entry per ligand atom).

    Note: the trained potential uses theta in [0, pi] without symmetry folding,
    so we keep the full polar angle here too.
    """
    n_prot = len(prot_indices)
    n_lig = len(lig_coords)
    pi = np.pi
    two_pi = 2.0 * pi

    for p_idx in range(n_prot):
        i = prot_indices[p_idx]
        p_type_idx = prot_type_indices[p_idx]

        px = prot_coords[i, 0]
        py = prot_coords[i, 1]
        pz = prot_coords[i, 2]
        v1x = v1_arr[i, 0]; v1y = v1_arr[i, 1]; v1z = v1_arr[i, 2]
        v2x = v2_arr[i, 0]; v2y = v2_arr[i, 1]; v2z = v2_arr[i, 2]
        v3x = v3_arr[i, 0]; v3y = v3_arr[i, 1]; v3z = v3_arr[i, 2]

        for j in range(n_lig):
            l_type_idx = lig_type_indices[j]
            if l_type_idx < 0:
                continue

            dx = lig_coords[j, 0] - px
            dy = lig_coords[j, 1] - py
            dz = lig_coords[j, 2] - pz
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < 1e-20:
                continue

            dist = np.sqrt(dist_sq)
            inv_dist = 1.0 / dist

            # --- polar angle theta in [0, pi] ---
            # FIX: parenthesize the whole dot product before dividing.
            cos_theta = (dx * v1x + dy * v1y + dz * v1z) * inv_dist
            # Guard against tiny FP excursions outside [-1, 1].
            if cos_theta > 1.0:
                cos_theta = 1.0
            elif cos_theta < -1.0:
                cos_theta = -1.0
            theta = np.arccos(cos_theta)

            # --- azimuthal angle phi in [0, 2*pi) ---
            proj_v2 = dx * v2x + dy * v2y + dz * v2z
            proj_v3 = dx * v3x + dy * v3y + dz * v3z
            phi = np.arctan2(proj_v3, proj_v2) + pi
            # arctan2 returns (-pi, pi], so phi here lives in (0, 2*pi].
            # Wrap the upper edge back to 0 so the interpolator's modular
            # math sees a clean half-open range.
            if phi >= two_pi:
                phi -= two_pi

            # --- raw "bin-edge" grid coordinates; interp_3d shifts to centers ---
            r_grid = (dist - r_min) / r_step
            theta_grid = theta / angular_step
            phi_grid = phi / angular_step

            score = interp_3d(
                scores[p_type_idx, l_type_idx, :, :, :],
                r_grid, theta_grid, phi_grid,
                n_r, n_theta, n_phi,
            )
            b_factors[j] += score

# ============================================================================
# Main scorer class
# ============================================================================

class KORP_Scorer:
    """
    Numba-accelerated KORP scorer.
    
    First call will be slower due to JIT compilation.
    Subsequent calls will be much faster.
    """

    def __init__(self, mode = 'korp', database = 'CROWN'):

        self.mode = mode
        self.database = database

        self.prot_types_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        
        lig_df = pd.read_csv(DATA_DIR / 'metadata' / 'lig_types.csv')
        self.lig_types_list = lig_df['atom_type'].tolist()

        self.prot_to_idx = {t: i for i, t in enumerate(self.prot_types_list)}
        self.lig_to_idx = {t: i for i, t in enumerate(self.lig_types_list)}

        loaded = np.load(DATA_DIR / 'potentials' / f'{self.mode.lower()}_scores_{self.database.lower()}.npz')
        self.scores = np.ascontiguousarray(loaded['scores'].astype(np.float32))

        # Grid parameters
        self.r_min = np.float32(2.0)
        self.r_step = np.float32(0.2)
        self.angular_step = np.float32(np.deg2rad(6.0))
        
        self.n_r = self.scores.shape[2]
        self.n_theta = self.scores.shape[3]
        self.n_phi = self.scores.shape[4]

        # Warm up JIT compilation
        self._warmup()

    def _warmup(self):
        """Force JIT compilation with dummy data."""
        dummy_coords = np.zeros((1, 3), dtype=np.float32)
        dummy_idx = np.zeros(1, dtype=np.int32)
        dummy_b = np.zeros(1, dtype=np.float32)
        
        # These calls will trigger compilation
        try:
            score_3d_kernel(
                dummy_idx, dummy_idx, dummy_coords, dummy_coords, dummy_coords, dummy_coords,
                dummy_coords, dummy_idx, self.scores,
                self.r_min, self.r_step, self.angular_step,
                self.n_r, self.n_theta, self.n_phi, dummy_b
            )
        except:
            pass  # Ignore errors during warmup

    def score_complex(self, prot_df, lig_df):
        """Score protein-ligand complex."""

        # Extract arrays
        prot_df.dropna(axis = 0, how = 'any', inplace = True)
        prot_types = prot_df['res_type'].values
        prot_coords = np.ascontiguousarray(prot_df[['x', 'y', 'z']].values.astype(np.float32))
        v1_arr = np.ascontiguousarray(prot_df[['v1_x', 'v1_y', 'v1_z']].values.astype(np.float32))
        v2_arr = np.ascontiguousarray(prot_df[['v2_x', 'v2_y', 'v2_z']].values.astype(np.float32))
        v3_arr = np.ascontiguousarray(prot_df[['v3_x', 'v3_y', 'v3_z']].values.astype(np.float32))

        lig_types = lig_df['lig_type'].values
        lig_coords = np.ascontiguousarray(lig_df[['x', 'y', 'z']].values.astype(np.float32))
        n_lig = len(lig_coords)

        b_factors = np.zeros(n_lig, dtype=np.float32)

        # Map ligand types to indices
        lig_type_indices = np.array([
            self.lig_to_idx.get(t, -1) for t in lig_types
        ], dtype=np.int32)

        print(lig_types)
        if not (lig_type_indices >= 0).any():
            print('No known lig types')
            return b_factors

        # Find nearby protein atoms
        prot_tree = KDTree(prot_coords)
        neighbors = prot_tree.query_ball_point(lig_coords, r=11.0)
        flat_neighbors = [idx for sublist in neighbors for idx in sublist]

        if not flat_neighbors:
            return b_factors

        prot_indices = np.unique(flat_neighbors).astype(np.int32)
        prot_type_indices = np.array([self.prot_to_idx[prot_types[i]] for i in prot_indices], dtype = np.int32)
        score_3d_kernel(prot_indices, prot_type_indices, prot_coords, v1_arr, v2_arr, v3_arr,
                        lig_coords, lig_type_indices, self.scores, self.r_min, self.r_step, self.angular_step,
                        self.n_r, self.n_theta, self.n_phi, b_factors)

        return b_factors
