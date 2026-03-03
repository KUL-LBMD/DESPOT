import numpy as np
import pyshtools
from scipy.ndimage import gaussian_filter
from einops import rearrange
import pandas as pd

from src.config import DATA_DIR

class DESPOT_Builder:
    """
    Class for building anisotropic statistical potentials
    """

    def __init__(self, database):

        self.database = database

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / f'atom_type_counts_{database.lower()}.csv')

        self.types_list_1d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Isotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Axial') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Anisotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            counts_df.loc[
                (counts_df['total_occurrence'] > 500),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        # Load raw counts
        loaded = np.load(DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz')
        self.counts_1d = loaded['arr_1d']
        self.counts_2d = loaded['arr_2d']
        self.counts_3d = loaded['arr_3d']

        self.r_bins = np.arange(1.0, 6.1, 0.1)
        self.theta_bins_2d = np.deg2rad(np.arange(0, 183.0, 3.0))
        self.theta_bins_3d = np.deg2rad(np.arange(0, 93.0, 3.0))
        self.phi_bins = np.deg2rad(np.arange(0, 183.0, 3.0))

        self.sigma_r = 1
        self.sigma_angle = 1

    def blur_counts(self):
        """
        Applies volume normalization and Gaussian smoothing on raw counts
        """

        n_lat, n_lon = 60, 60

        # 1D case
        volume_corrections_1d = np.zeros((self.counts_1d.shape[2]))
        for i in range(volume_corrections_1d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            volume_corrections_1d[i] = 4 * np.pi * r_factor

        normalized_counts_1d = self.counts_1d / volume_corrections_1d[np.newaxis, np.newaxis, :]

        # 2D case
        volume_corrections_2d = np.zeros((self.counts_2d.shape[2], self.counts_2d.shape[3]))
        for i in range(volume_corrections_2d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            for j in range(volume_corrections_2d.shape[1]):
                theta_i, theta_e = self.theta_bins_2d[j], self.theta_bins_2d[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                volume_corrections_2d[i,j] = 2 * np.pi * theta_factor * r_factor

        # Radial smoothing first (Gaussian along r axis)
        normalized_counts_2d = self.counts_2d / volume_corrections_2d[np.newaxis, np.newaxis, :, :]

        # 3D case
        volume_corrections_3d = np.zeros((self.counts_3d.shape[2], self.counts_3d.shape[3], self.counts_3d.shape[4]))
        for i in range(volume_corrections_3d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)

            for j in range(volume_corrections_3d.shape[1]):
                theta_i, theta_e = self.theta_bins_3d[j], self.theta_bins_3d[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                for k in range(volume_corrections_3d.shape[2]):
                    phi_i, phi_e = self.phi_bins[k], self.phi_bins[k+1]
                    phi_factor = phi_e - phi_i

                    # Multiply by factor 4: with 2 unsigned axes, 4 voxels are always equivalent
                    volume_corrections_3d[i,j,k] = 4 * phi_factor * theta_factor * r_factor 

        normalized_counts_3d = self.counts_3d / volume_corrections_3d[np.newaxis, np.newaxis, :, :, :]
        rho_3d_full = np.concatenate([normalized_counts_3d, normalized_counts_3d[:, :, :, ::-1, :]], axis=3) / 2 # Equally divide density over 2 halfs (60, 60)
        del normalized_counts_3d

        self.n_p_1d = self.counts_1d.shape[0]
        self.n_p_2d = self.counts_2d.shape[0]
        self.n_p_3d = self.counts_3d.shape[0]

        del self.counts_1d, self.counts_2d, self.counts_3d

        ### Extend all normalized counts to full sphere and divide accordingly ###
        print('Building rho_1d')
        rho_1d_full = (
            normalized_counts_1d[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, n_lat, n_lon)) / (n_lat * n_lon)
        )

        print('Building rho_2d')
        rho_2d_full = (
            normalized_counts_2d[:, :, :, :, np.newaxis] * np.ones((1, 1, 1, 1, n_lon)) / n_lon
        )

        print('Building rho_3d')

        del normalized_counts_1d, normalized_counts_2d

        print(rho_1d_full.shape)
        print(rho_2d_full.shape)
        print(rho_3d_full.shape)

        # Concatenate all protein types along axis 0
        rho = np.concatenate([rho_1d_full, rho_2d_full, rho_3d_full], axis=0)
        del rho_1d_full, rho_2d_full, rho_3d_full

        print(rho.shape)

        # Radial Gaussian smoothing
        self.rho = gaussian_filter(rho, sigma = [0, 0, self.sigma_r, self.sigma_angle, self.sigma_angle])

    def counts_to_prob(self):
        """P(l | p, r) = n(p,l,r) / sum_l{n(p,l,r)}"""

        print('Calculating P(l | p,r,theta,phi)')

        self.rho = np.clip(self.rho, a_min = 0, a_max = None) # Ensure non-negative values
        lig_sum = np.sum(self.rho, axis = 1) # [p,r, theta, phi]
        xi = np.max(lig_sum, axis = (1,2,3), keepdims = True)
        decoy_vals = xi - lig_sum
        self.rho = np.concatenate((self.rho, decoy_vals[:, np.newaxis, :, :, :]), axis = 1) # [p, l+1, r, theta, phi]
        self.rho = np.clip(self.rho, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob = self.rho / np.sum(self.rho) # L1-normalization to get P(p, l, r, theta, phi)
        del self.rho
        self.cond_prob = self.prob / np.sum(self.prob, axis = 1, keepdims = True) # P(l | p,r,theta,phi)

    def ref_probs(self):
        """
        P(l) = sum_p{P(p) * mean_r[P(l | p, r)]}
        """

        print('Calculating P(l)')
        self.ref_prob = np.sum(self.prob, axis = (0, 2, 3, 4), keepdims = True) # [l+1]

    def inverse_boltzmann(self):
        """
        score[p,l,r] = ln[P(l | p,r) / P(l)]
        """

        print('Running inverse Boltzmann')
        eps = 1e-12 # Lower bound, prevent 0 probabilities

        scores = self.cond_prob / self.ref_prob
        scores = np.clip(scores, eps, None)
        scores = np.clip(-1 * np.log10(scores), a_min = -5, a_max = 5)

        # Split back by symmetry class for save compatibility
        i1 = self.n_p_1d
        i2 = i1 + self.n_p_2d
        self.scores_1d = scores[:i1, :-1, :, 0, 0] # Don't take decoy atom type
        self.scores_2d = scores[i1:i2, :-1, :, :, 0]
        self.scores_3d = scores[i2:, :-1, :, :30, :]

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores_1d, scores_2d = self.scores_2d, scores_3d = self.scores_3d)

class DESPOT_SH_Builder:
    """
    Class for building anisotropic statistical potentials
    """

    def __init__(self, database):

        self.database = database

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / f'atom_type_counts_{database.lower()}.csv')

        self.types_list_1d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Isotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Axial') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Anisotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            counts_df.loc[
                (counts_df['total_occurrence'] > 500),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        # Load raw counts
        loaded = np.load(DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz')
        self.counts_1d = loaded['arr_1d']
        self.counts_2d = loaded['arr_2d']
        self.counts_3d = loaded['arr_3d']

        self.r_bins = np.arange(1.0, 6.1, 0.1)
        self.theta_bins_2d = np.deg2rad(np.arange(0, 183.0, 3.0))
        self.theta_bins_3d = np.deg2rad(np.arange(0, 93.0, 3.0))
        self.phi_bins = np.deg2rad(np.arange(0, 183.0, 3.0))

        self.sigma_r = 1
        self.sigma_angle = 1

    def _sh_heat_filter(self, lmax, sigma_rad):
        """
        Spherical heat kernel filter coefficients.
        Analog of Gaussian smoothing on S²: each degree-l coefficient is
        multiplied by exp(-l(l+1) * sigma² / 2).

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        sigma_rad : float
            Smoothing width in radians. Comparable to the Gaussian sigma
            projected onto the sphere.

        Returns
        -------
        taper : np.ndarray of shape (lmax+1,)
        """
        ls = np.arange(lmax + 1)
        return np.exp(-ls * (ls + 1) * sigma_rad ** 2 / 2.0)

    def _sh_smooth_3d(self, angular_slice, sigma_rad):
        """
        Smooth a 2D function on the quarter-sphere θ∈[0,90°), ϕ∈[0,180°)
        using spherical harmonics with even-symmetry extension.

        Symmetry: f(180°−θ, ϕ) = f(θ, ϕ)  and  f(θ, 360°−ϕ) = f(θ, ϕ)

        After mirroring → (60, 120) DH2 grid, lmax = 29.

        Parameters
        ----------
        angular_slice : ndarray, shape (n_theta, n_phi) — (30, 60)
        sigma_rad     : float

        Returns
        -------
        smoothed : ndarray, shape (n_theta, n_phi) — (30, 60)
        """

        n_lat, n_lon = angular_slice.shape
        full_sphere = np.concatenate([angular_slice, angular_slice[:, ::-1]], axis=1)  # (60, 120)

        # --- SH expand, filter, reconstruct -----------------------------------
        sh_grid = pyshtools.SHGrid.from_array(full_sphere, grid='DH', copy=True)
        coeffs = sh_grid.expand()

        # Apply heat kernel + cutoff
        taper = self._sh_heat_filter(coeffs.lmax, sigma_rad)

        for l in range(coeffs.lmax + 1):
            coeffs.coeffs[:, l, :l+1] *= taper[l]

        smoothed_full = coeffs.expand(grid='DH2', extend = False, lmax_calc = coeffs.lmax).to_array()

        return smoothed_full[:, :n_lon]

    def blur_counts(self):
        """
        Applies volume normalization and Gaussian smoothing on raw counts
        """

        sigma_angle_rad = np.deg2rad(self.sigma_angle * 3.0)  # convert bin-units → radians
        n_lat, n_lon = 60, 60

        # 1D case
        volume_corrections_1d = np.zeros((self.counts_1d.shape[2]))
        for i in range(volume_corrections_1d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            volume_corrections_1d[i] = 4 * np.pi * r_factor

        normalized_counts_1d = self.counts_1d / volume_corrections_1d[np.newaxis, np.newaxis, :]

        # 2D case
        volume_corrections_2d = np.zeros((self.counts_2d.shape[2], self.counts_2d.shape[3]))
        for i in range(volume_corrections_2d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            for j in range(volume_corrections_2d.shape[1]):
                theta_i, theta_e = self.theta_bins_2d[j], self.theta_bins_2d[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                volume_corrections_2d[i,j] = 2 * np.pi * theta_factor * r_factor

        # Radial smoothing first (Gaussian along r axis)
        normalized_counts_2d = self.counts_2d / volume_corrections_2d[np.newaxis, np.newaxis, :, :]

        # 3D case
        volume_corrections_3d = np.zeros((self.counts_3d.shape[2], self.counts_3d.shape[3], self.counts_3d.shape[4]))
        for i in range(volume_corrections_3d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)

            for j in range(volume_corrections_3d.shape[1]):
                theta_i, theta_e = self.theta_bins_3d[j], self.theta_bins_3d[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                for k in range(volume_corrections_3d.shape[2]):
                    phi_i, phi_e = self.phi_bins[k], self.phi_bins[k+1]
                    phi_factor = phi_e - phi_i

                    # Multiply by factor 4: with 2 unsigned axes, 4 voxels are always equivalent
                    volume_corrections_3d[i,j,k] = 4 * phi_factor * theta_factor * r_factor 

        normalized_counts_3d = self.counts_3d / volume_corrections_3d[np.newaxis, np.newaxis, :, :, :]
        rho_3d_full = np.concatenate([normalized_counts_3d, normalized_counts_3d[:, :, :, ::-1, :]], axis=3) / 2 # Equally divide density over 2 halfs (60, 60)
        del normalized_counts_3d

        self.n_p_1d = self.counts_1d.shape[0]
        self.n_p_2d = self.counts_2d.shape[0]
        self.n_p_3d = self.counts_3d.shape[0]

        del self.counts_1d, self.counts_2d, self.counts_3d

        ### Extend all normalized counts to full sphere and divide accordingly ###
        print('Building rho_1d')
        rho_1d_full = (
            normalized_counts_1d[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, n_lat, n_lon)) / (n_lat * n_lon)
        )

        print('Building rho_2d')
        rho_2d_full = (
            normalized_counts_2d[:, :, :, :, np.newaxis] * np.ones((1, 1, 1, 1, n_lon)) / n_lon
        )

        print('Building rho_3d')

        del normalized_counts_1d, normalized_counts_2d

        print(rho_1d_full.shape)
        print(rho_2d_full.shape)
        print(rho_3d_full.shape)

        # Concatenate all protein types along axis 0
        rho = np.concatenate([rho_1d_full, rho_2d_full, rho_3d_full], axis=0)
        del rho_1d_full, rho_2d_full, rho_3d_full

        print(rho.shape)

        # Radial Gaussian smoothing
        rho = gaussian_filter(rho, sigma = [0, 0, self.sigma_r, 0, 0])
        # Spherical harmonics angular smoothing
        n_p, n_l, n_r, n_theta, n_phi = rho.shape
        for ip in range(n_p):
            print(f'{ip} / {n_p}')
            for il in range(n_l):
                for ir in range(n_r):
                    slice_data = rho[ip, il, ir, :, :]
                    # Apply smoothing in log-space to avoid Gibbs ringing
                    log_slice = np.log(slice_data + 1e-12)
                    log_smoothed = self._sh_smooth_3d(log_slice, sigma_angle_rad)
                    smoothed = np.exp(log_smoothed)
                    rho[ip, il, ir, :, :] = smoothed

        self.rho = rho

    def counts_to_prob(self):
        """P(l | p, r) = n(p,l,r) / sum_l{n(p,l,r)}"""

        print('Calculating P(l | p,r,theta,phi)')

        self.rho = np.clip(self.rho, a_min = 0, a_max = None) # Ensure non-negative values
        lig_sum = np.sum(self.rho, axis = 1) # [p,r, theta, phi]
        xi = np.max(lig_sum, axis = (1,2,3), keepdims = True)
        decoy_vals = xi - lig_sum
        self.rho = np.concatenate((self.rho, decoy_vals[:, np.newaxis, :, :, :]), axis = 1) # [p, l+1, r, theta, phi]
        self.rho = np.clip(self.rho, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob = self.rho / np.sum(self.rho) # L1-normalization to get P(p, l, r, theta, phi)
        del self.rho
        self.cond_prob = self.prob / np.sum(self.prob, axis = 1, keepdims = True) # P(l | p,r,theta,phi)

    def ref_probs(self):
        """
        P(l) = sum_p{P(p) * mean_r[P(l | p, r)]}
        """

        print('Calculating P(l)')
        self.ref_prob = np.sum(self.prob, axis = (0, 2, 3, 4), keepdims = True) # [l+1]

    def inverse_boltzmann(self):
        """
        score[p,l,r] = ln[P(l | p,r) / P(l)]
        """

        print('Running inverse Boltzmann')
        eps = 1e-12 # Lower bound, prevent 0 probabilities

        scores = self.cond_prob / self.ref_prob
        scores = np.clip(scores, eps, None)
        scores = np.clip(-1 * np.log10(scores), a_min = -5, a_max = 5)

        # Split back by symmetry class for save compatibility
        i1 = self.n_p_1d
        i2 = i1 + self.n_p_2d
        self.scores_1d = scores[:i1, :-1, :, 0, 0] # Don't take decoy atom type
        self.scores_2d = scores[i1:i2, :-1, :, :, 0]
        self.scores_3d = scores[i2:, :-1, :, :30, :]

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_sh_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores_1d, scores_2d = self.scores_2d, scores_3d = self.scores_3d)

class DESPOT_Iso_Builder:
    """
    Class for building anisotropic statistical potentials
    """

    def __init__(self, database):

        self.database = database
        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / f'atom_type_counts_{database.lower()}.csv')

        self.types_list_1d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Isotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Axial') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Anisotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            counts_df.loc[
                (counts_df['total_occurrence'] > 500),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.prot_types_list = self.types_list_1d + self.types_list_2d + self.types_list_3d

        # Load raw counts
        loaded = np.load(DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz')
        counts_1d = loaded['arr_1d']
        counts_2d = loaded['arr_2d']
        counts_3d = loaded['arr_3d']

        # Combine all counts into [p,l,r] array
        total_counts_2d = np.sum(counts_2d, axis = 3)
        total_counts_3d = np.sum(counts_3d, axis = (3,4))
        self.counts = np.concatenate((counts_1d, total_counts_2d, total_counts_3d))

        self.r_bins = np.arange(1.0, 6.1, 0.1)
        self.sigma_r = 1

    def blur_counts(self):
        """
        Applies volume normalization and Gaussian smoothing on raw counts
        """

        # 1D case
        volume_corrections_1d = np.zeros((self.counts.shape[2]))
        for i in range(volume_corrections_1d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            volume_corrections_1d[i] = 4 * np.pi * r_factor

        normalized_counts_1d = self.counts / volume_corrections_1d[np.newaxis, np.newaxis, :]
        self.rho = gaussian_filter(normalized_counts_1d, sigma = [0, 0, self.sigma_r])

    def counts_to_prob(self):
        """P(l | p, r) = n(p,l,r) / sum_l{n(p,l,r)}"""

        print('Calculating P(l | p,r,theta,phi)')

        lig_sum = np.sum(self.rho, axis = 1) # [p,r]
        xi = np.max(lig_sum, axis = 1, keepdims = True)
        decoy_vals = xi - lig_sum
        self.rho = np.concatenate((self.rho, decoy_vals[:, np.newaxis, :]), axis = 1) # [p, l+1, r]
        self.rho = np.clip(self.rho, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob = self.rho / np.sum(self.rho) # L1-normalization to get P(p, l, r)
        self.cond_prob = self.prob / np.sum(self.prob, axis = 1, keepdims = True) # P(l | p,r)

    def ref_probs(self):
        """
        P(l) = sum_p{P(p) * mean_r[P(l | p, r)]}
        """

        print('Calculating P(l)')
        self.ref_prob = np.sum(self.prob, axis = (0, 2), keepdims = True) # [l+1]

    def inverse_boltzmann(self):
        """
        score[p,l,r] = ln[P(l | p,r) / P(l)]
        """

        print('Running inverse Boltzmann')
        eps = 1e-12 # Lower bound, prevent 0 probabilities

        scores = self.cond_prob / self.ref_prob
        scores = np.clip(scores, eps, None)
        scores = np.clip(-1 * np.log10(scores), a_min = -5, a_max = 5)

        self.scores_1d = scores[:, :-1, :] # Don't take decoy atom type

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_iso_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores_1d)

class DESPOT_DS_Builder:
    """
    Class for building isotropic statistical potentials
    """

    def __init__(self, database):

        self.database = database

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / f'atom_type_counts_{database.lower()}.csv')

        self.types_list_1d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Isotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Axial') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Anisotropic') &
                (counts_df['total_occurrence'] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            counts_df.loc[
                (counts_df['total_occurrence'] > 500),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.prot_types_list = self.types_list_1d + self.types_list_2d + self.types_list_3d

        # Load raw counts
        loaded = np.load(DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz')
        counts_1d = loaded['arr_1d']
        counts_2d = loaded['arr_2d']
        counts_3d = loaded['arr_3d']

        # Combine all counts into [p,l,r] array
        total_counts_2d = np.sum(counts_2d, axis = 3)
        total_counts_3d = np.sum(counts_3d, axis = (3,4))
        self.counts = np.concatenate((counts_1d, total_counts_2d, total_counts_3d))

        self.r_bins = np.arange(1.0, 6.1, 0.1)
        self.sigma_r = 1

    def blur_counts(self):
        """
        Applies volume normalization and Gaussian smoothing on raw counts
        """

        THRESHOLD = 500
        self.zero_combos = []
        for i in range(self.counts.shape[0]):
            for j in range(self.counts.shape[1]):
                value = self.counts[i,j,:].sum()
                if value < THRESHOLD:
                    self.counts[i,j,:] = 0
                    self.zero_combos.append((i,j))

        # 1D case
        volume_corrections_1d = np.zeros((self.counts.shape[2]))
        for i in range(volume_corrections_1d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            volume_corrections_1d[i] = 4 * np.pi * r_factor

        normalized_counts_1d = self.counts / volume_corrections_1d[np.newaxis, np.newaxis, :]
        self.rho = gaussian_filter(normalized_counts_1d, sigma = [0, 0, self.sigma_r])

    def counts_to_prob(self):
        """P(r | p, l) = n(p,l,r) / sum_r{n(p,l,r)}"""

        self.prob = self.rho / np.sum(self.rho) # Joint probability distribution [p,l,r]
        self.cond_prob = self.prob / np.sum(self.prob, axis = 2, keepdims = True) # P(r | p, l)

    def ref_probs(self):
        self.ref_prob = np.sum(self.prob, axis = (0,1), keepdims = True) # [r]

    def inverse_boltzmann(self):
        """score[p,l,r] = ln[P(r | p,l) / P(r)]"""

        print('Running inverse Boltzmann')
        eps = 1e-12 # Lower bound, prevent 0 probabilities

        scores = self.cond_prob / self.ref_prob
        scores = np.clip(scores, eps, None)
        scores = np.clip(-1 * np.log10(scores), a_min = -5, a_max = 5)

        for i,j in self.zero_combos:
            scores[i,j,:] = 0.0

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_ds_scores_{self.database.lower()}.npz', 
            scores_1d = scores)
