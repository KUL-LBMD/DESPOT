import numpy as np
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

        # Set types lists for ligand atoms and protein atoms
        self.database = database
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / f'atom_type_counts_{self.database.lower()}.csv')

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

        # 1D case
        volume_corrections_1d = np.zeros((self.counts_1d.shape[2]))
        for i in range(volume_corrections_1d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            volume_corrections_1d[i] = 4 * np.pi * r_factor

        normalized_counts_1d = self.counts_1d / volume_corrections_1d[np.newaxis, np.newaxis, :]
        self.rho_1d = gaussian_filter(normalized_counts_1d, sigma = [0, 0, self.sigma_r])

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

        normalized_counts_2d = self.counts_2d / volume_corrections_2d[np.newaxis, np.newaxis, :, :]
        self.rho_2d = gaussian_filter(normalized_counts_2d, sigma = [0, 0, self.sigma_r, self.sigma_angle])

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
        self.rho_3d = gaussian_filter(normalized_counts_3d, sigma = [0, 0, self.sigma_r, self.sigma_angle, self.sigma_angle])

    def counts_to_prob(self):
        """P(l | p, r) = n(p,l,r) / sum_l{n(p,l,r)}"""

        # 1D case
        xi = np.sum(self.rho_1d, axis = 1) # [p,r]
        decoy_vals = np.max(xi, axis = 1, keepdims = True) - xi
        density_1d = np.concatenate((self.rho_1d, decoy_vals[:, np.newaxis, :]), axis = 1) # [p, l+1, r]
        density_1d = np.clip(density_1d, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob_1d = density_1d / np.sum(density_1d, axis = 1, keepdims = True) # L1-normalization to get sum_l{P(l | p,r)} = 1

        # 2D case
        xi = np.sum(self.rho_2d, axis = 1) # [p,r, theta]
        decoy_vals = np.max(xi, axis = (1,2), keepdims = True) - xi
        density_2d = np.concatenate((self.rho_2d, decoy_vals[:, np.newaxis, :, :]), axis = 1) # [p, l+1, r, theta]
        density_2d = np.clip(density_2d, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob_2d = density_2d / np.sum(density_2d, axis = 1, keepdims = True) # L1-normalization to get sum_l{P(l | p,r)} = 1

        # 3D case
        xi = np.sum(self.rho_3d, axis = 1) # [p,r, theta, phi]
        decoy_vals = np.max(xi, axis = (1,2,3), keepdims = True) - xi
        density_3d = np.concatenate((self.rho_3d, decoy_vals[:, np.newaxis, :, :, :]), axis = 1) # [p, l+1, r, theta, phi]
        density_3d = np.clip(density_3d, a_min = 0, a_max = None) # Ensure non-negative values
        self.prob_3d = density_3d / np.sum(density_3d, axis = 1, keepdims = True) # L1-normalization to get sum_l{P(l | p,r)} = 1

    def ref_probs(self):
        """
        P(l) = sum_p{P(p) * mean_r[P(l | p, r)]}
        """

        # Get marginal probabilities P(p)
        wp_1d = np.sum(self.rho_1d, axis = (1, 2))
        wp_2d = np.sum(self.rho_2d, axis = (1, 2, 3))
        wp_3d = np.sum(self.rho_3d, axis = (1, 2, 3, 4))
        wp_total = np.concatenate((wp_1d, wp_2d, wp_3d), axis = 0)
        wp_total = wp_total / np.sum(wp_total) # L1-normalization for valid probability distribution

        # Get mean P(l | p)
        mean_1d = np.mean(self.prob_1d, axis = 2)
        mean_2d = np.mean(self.prob_2d, axis = (2,3))
        mean_3d = np.mean(self.prob_3d, axis = (2,3,4))
        mean_total = np.concatenate((mean_1d, mean_2d, mean_3d), axis = 0)

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

        # 2D case
        init_scores = self.prob_2d / self.ref[np.newaxis, :, np.newaxis, np.newaxis]
        init_scores = np.clip(init_scores, eps, None)
        temp_scores = np.clip(-1 * np.log10(init_scores), a_min = -5, a_max = 5)
        self.scores_2d = temp_scores[:, :-1, :, :] # Don't take decoy atom type

        # 3D case
        init_scores = self.prob_3d / self.ref[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        init_scores = np.clip(init_scores, eps, None)
        temp_scores = np.clip(-1 * np.log10(init_scores), a_min = -5, a_max = 5)
        self.scores_3d = temp_scores[:, :-1, :, :, :] # Don't take decoy atom type

        np.savez_compressed(DATA_DIR / 'potentials' / f'despot_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores_1d, scores_2d = self.scores_2d, scores_3d = self.scores_3d)
        
class DESPOT_Iso_Builder:
    """
    Class for building anisotropic statistical potentials
    """

    def __init__(self, database):

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / f'atom_type_counts_{database.lower()}.csv')
        self.database = database

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
