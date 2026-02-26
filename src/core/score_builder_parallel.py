import numpy as np
import pyshtools
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
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
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')

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

    def _precompute_sh_kernels(self, n_lat, sigma_rad):
        """
        Precompute per-azimuthal-order kernel matrices for SH smoothing
        on a Driscoll–Healy grid of shape (n_lat, 2*n_lat).

        The smoothing operation becomes:
            1. rfft along longitude  →  (n_lat, n_lat+1) complex
            2. for each order m: multiply by K_m along latitude axis
            3. irfft back            →  (n_lat, 2*n_lat)

        Parameters
        ----------
        n_lat : int
            Number of latitude rows (must be even).  lmax = n_lat // 2 - 1.
        sigma_rad : float
            Heat-kernel width in radians.

        Returns
        -------
        kernels : list of ndarray, length lmax+1
            K_m of shape (n_lat, n_lat) for m = 0 … lmax.
            Orders m > lmax are zeroed during application.
        """
        lmax = n_lat // 2 - 1
        taper = self._sh_heat_filter(lmax, sigma_rad)

        # DH colatitudes: θ_i = i * π / n_lat
        thetas = np.arange(n_lat) * np.pi / n_lat
        cos_thetas = np.cos(thetas)

        # DH quadrature weights (length n_lat)
        weights = pyshtools.utils.DHaj(n_lat)[:n_lat]

        # Precompute all 4π-normalized associated Legendre values
        # PlmBar returns a 1D array indexed by l*(l+1)//2 + m
        plm_all = [pyshtools.legendre.PlmBar(lmax, ct) for ct in cos_thetas]

        def _plm_idx(l, m):
            return l * (l + 1) // 2 + m

        kernels = []
        for m in range(lmax + 1):
            n_l = lmax + 1 - m                               # number of degrees at this order
            P_m = np.zeros((n_lat, n_l))                     # (n_lat, n_l)
            tap_m = np.array([taper[l] for l in range(m, lmax + 1)])  # (n_l,)

            for i in range(n_lat):
                for idx, l in enumerate(range(m, lmax + 1)):
                    P_m[i, idx] = plm_all[i][_plm_idx(l, m)]

            # K_m = P_m @ diag(tap_m) @ P_m^T @ diag(w)
            #     = (P_m * tap_m) @ P_m^T * w          (broadcasting)
            K_m = (P_m * tap_m[np.newaxis, :]) @ P_m.T * weights[np.newaxis, :]
            kernels.append(K_m)

        return kernels

    def _apply_sh_smooth_batch(self, data, kernels):
        """
        Vectorized SH smoothing for full-sphere DH grid data.

        Parameters
        ----------
        data : ndarray, shape (..., n_lat, n_lon) — (…, 60, 120)
        kernels : list from _precompute_sh_kernels

        Returns
        -------
        smoothed : ndarray, same shape as data
        """
        batch_shape = data.shape[:-2]
        n_lat, n_lon = data.shape[-2:]              # (60, 120)
        lmax = len(kernels) - 1                     # 29

        flat = data.reshape(-1, n_lat, n_lon)       # (N, 60, 120)

        # Forward FFT along longitude → (N, 60, 61) complex
        F = np.fft.rfft(flat, axis=2)

        # Per-order batched matmul (30 iterations, each pure BLAS)
        F_smooth = np.zeros_like(F)
        for m in range(lmax + 1):
            F_smooth[:, :, m] = F[:, :, m] @ kernels[m].T

        # Inverse FFT → (N, 60, 120)
        smoothed = np.fft.irfft(F_smooth, n=n_lon, axis=2)
        return smoothed.reshape(*batch_shape, n_lat, n_lon)

    def blur_counts(self):
        """
        Applies volume normalization and Gaussian smoothing on raw counts
        """

        sigma_angle_rad = np.deg2rad(self.sigma_angle * 3.0)  # convert bin-units → radians
        n_lat, n_lon = 60, 120
        sh_kernels = self._precompute_sh_kernels(n_lat, sigma_angle_rad)

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

        ### Extend all normalized counts to full sphere and divide accordingly ###
        rho_1d_full = (
            normalized_counts_1d[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, n_lat, n_lon)) / (n_lat * n_lon)
        )

        rho_2d_full = (
            normalized_counts_2d[:, :, :, :, np.newaxis] * np.ones((1, 1, 1, 1, n_lon)) / n_lon
        )

        c3 = normalized_counts_3d
        theta_full = np.concatenate([c3, c3[:, :, :, ::-1, :]], axis=3)
        rho_3d_full = np.concatenate([theta_full, theta_full[:, :, :, :, ::-1]], axis=4) / 4

        # Concatenate all protein types along axis 0
        self.n_p_1d = self.counts_1d.shape[0]
        self.n_p_2d = self.counts_2d.shape[0]
        self.n_p_3d = self.counts_3d.shape[0]
        rho = np.concatenate([rho_1d_full, rho_2d_full, rho_3d_full], axis=0)

        # Radial Gaussian smoothing
        rho = gaussian_filter(rho, sigma = [0, 0, self.sigma_r, 0, 0])
        # Spherical harmonics angular smoothing
        self.rho = self._apply_sh_smooth_batch(rho, sh_kernels)

    def counts_to_prob(self):
        """P(l | p, r) = n(p,l,r) / sum_l{n(p,l,r)}"""

        xi = np.sum(self.rho, axis = 1) # [p,r, theta, phi]
        decoy_vals = np.max(xi, axis = (1,2,3), keepdims = True) - xi
        density = np.concatenate((self.rho, decoy_vals[:, np.newaxis, :, :, :]), axis = 1) # [p, l+1, r, theta, phi]
        density = np.clip(density, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob = density / np.sum(density) # L1-normalization to get P(p, l, r, theta, phi)

    def ref_probs(self):
        """
        P(l) = sum_p{P(p) * mean_r[P(l | p, r)]}
        """

        self.ref = np.sum(self.prob, axis = (0, 2, 3, 4), keepdims = True) # [l+1]

    def inverse_boltzmann(self):
        """
        score[p,l,r] = ln[P(l | p,r) / P(l)]
        """

        eps = 1e-12 # Lower bound, prevent 0 probabilities

        init_scores = self.prob / self.ref
        init_scores = np.clip(init_scores, eps, None)
        temp_scores = np.clip(-1 * np.log10(init_scores), a_min = -5, a_max = 5)
        scores = temp_scores[:, :-1, :, :, :] # Don't take decoy atom type

        # Split back by symmetry class for save compatibility
        i1 = self.n_p_1d
        i2 = i1 + self.n_p_2d
        self.scores_1d = scores[:i1, :, :, 0, 0]
        self.scores_2d = scores[i1:i2, :, :, :, 0]
        self.scores_3d = scores[i2:, :, :, :30, :60] # Only take one sphere quadrant

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_scores_{self.databse.lower()}.npz', 
            scores_1d = self.scores_1d, scores_2d = self.scores_2d, scores_3d = self.scores_3d)
        
class DESPOT_Iso_Builder:
    """
    Class for building anisotropic statistical potentials
    """

    def __init__(self, database):

        self.database = database
        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')

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
        self.rho_1d = gaussian_filter(normalized_counts_1d, sigma = [0, 0, self.sigma_r])

    def counts_to_prob(self):
        """P(l | p, r) = n(p,l,r) / sum_l{n(p,l,r)}"""

        # 1D case
        xi = np.sum(self.rho_1d, axis = 1) # [p,r]
        decoy_vals = np.max(xi, axis = 1, keepdims = True) - xi
        density_1d = np.concatenate((self.rho_1d, decoy_vals[:, np.newaxis, :]), axis = 1) # [p, l+1, r]
        density_1d = np.clip(density_1d, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob_1d = density_1d / np.sum(density_1d, axis = 1, keepdims = True) # L1-normalization to get sum_l{P(l | p,r)} = 1

    def ref_probs(self):
        """
        P(l) = sum_p{P(p) * mean_r[P(l | p, r)]}
        """

        # Get marginal probabilities P(p)
        wp_1d = np.sum(self.rho_1d, axis = (1, 2))
        wp_total = wp_1d / np.sum(wp_1d) # L1-normalization for valid probability distribution

        # Get mean P(l | p)
        mean_total = np.mean(self.prob_1d, axis = 2)

        self.ref = wp_total @ mean_total

    def inverse_boltzmann(self):
        """
        score[p,l,r] = ln[P(l | p,r) / P(l)]
        """

        eps = 1e-12 # Lower bound, prevent 0 probabilities

        # 1D case
        init_scores = self.prob_1d / self.ref[np.newaxis, :, np.newaxis]
        init_scores = np.clip(init_scores, eps, None)
        temp_scores = np.clip(-1 * np.log10(init_scores), a_min = -5, a_max = 5)
        self.scores_1d = temp_scores[:, :-1, :] # Don't take decoy atom type

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_iso_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores_1d)
        
class DESPOT_DS_Builder:
    """
    Class for building isotropic statistical potentials
    """

    def __init__(self, database):

        self.database = database

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')

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
        self.rho_1d = gaussian_filter(normalized_counts_1d, sigma = [0, 0, self.sigma_r])

    def counts_to_prob(self):
        """P(r | p, l) = n(p,l,r) / sum_r{n(p,l,r)}"""

        denom_arr = np.sum(self.rho_1d, axis = 2, keepdims = True) + 1e-12 # [p,l,r]
        self.prob = self.rho_1d / denom_arr

        for i,j in self.zero_combos:
            self.prob[i,j,:] = 0

    def cluster_probs(self):
        """Applies complete linkage clustering to probability functions"""

        # Step 1: compute distance matrix
        flat_probs = rearrange(self.prob, 'p l r -> (p l) r')
        dist_matrix = squareform(cdist(flat_probs, flat_probs, metric = 'sqeuclidean'))

        # Complete linkage clustering
        dendrogram = linkage(dist_matrix, method = 'complete')
        clusters = fcluster(dendrogram, t = 0.0025, criterion = 'distance') - 1
        num_clusters = np.max(clusters) + 1

        # Cluster PDFs
        self.clustered_probs = np.zeros((num_clusters, len(self.r_bins)-1))
        for cluster_idx in range(num_clusters):
            mask = clusters == cluster_idx
            cluster_probs = flat_probs[mask]
            self.clustered_probs[cluster_idx,:] = np.mean(cluster_probs, axis = 0)

        # Map original p_l pairs to cluster
        keys = [f'{p}_{l}' for p in self.prot_types_list for l in self.ligand_types_list]
        self.map_dict = {k:v for k,v in zip(keys, clusters)}

    def ref_probs(self):
        """P(r) = mean_{p,l}[P(r | p, l)]"""

        n_c, n_r = self.clustered_probs.shape
        num_pot = 0
        ref_sum = np.zeros(n_r)

        self.zero_cluster = None

        for c in range(n_c):
            if not self.clustered_probs[c].sum() == 0:
                ref_sum += self.clustered_probs[c]
                num_pot += 1
            else:
                self.zero_cluster = c

        self.ref = ref_sum / num_pot

    def inverse_boltzmann(self):
        """score[p,l,r] = ln[P(r | p,l) / P(r)]"""

        eps = 1e-12 # Lower bound, prevent 0 probabilities

        # 1D case
        init_scores = self.clustered_probs / self.ref[np.newaxis, :] # [c, r]
        init_scores = np.clip(init_scores, eps, None)
        temp_scores = np.clip(-1 * np.log10(init_scores), a_min = -5, a_max = 5)

        if self.zero_cluster is not None:
            temp_scores[self.zero_cluster] = 0.0

        # Map scores back to original p_l indices
        n_p, n_l, n_r = self.counts.shape
        self.scores = np.zeros_like(self.counts)

        for i in range(n_p):
            for j in range(n_l):
                key = f'{self.prot_types_list[i]}_{self.ligand_types_list[j]}'
                cluster_idx = self.map_dict[key]
                self.scores[i,j,:] = temp_scores[cluster_idx]

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_ds_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores)
