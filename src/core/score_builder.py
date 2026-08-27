import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from einops import rearrange
import pandas as pd
import pyshtools as pysh

from src.config import DATA_DIR

class DESPOT_Builder:
    """
    Class for building anisotropic statistical potentials
    """

    def __init__(self, database):
        """
        Parameters
        ----------
        database : str
            Name of the count database to load.
        """

        # Set types lists for ligand atoms and protein atoms
        self.database = database

        prot_counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'prot_types.csv')
        lig_counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'lig_types.csv')

        self.types_list_1d = (
            prot_counts_df.loc[
                (prot_counts_df['local_reference_frame'] == 'Isotropic'),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            prot_counts_df.loc[
                (prot_counts_df['local_reference_frame'] == 'Axial'),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            prot_counts_df.loc[
                (prot_counts_df['local_reference_frame'] == 'Anisotropic'),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            lig_counts_df['atom_type']
            .dropna()
            .unique()
            .tolist()
        )

        self.r_bins = np.arange(1.0, 6.1, 0.1)
        self.theta_bins = np.deg2rad(np.arange(0, 183.0, 3.0))
        self.phi_bins = np.deg2rad(np.arange(0, 363.0, 3.0))

        self.sigma_r = 1
        self.sigma_angle = 0.05
        self.n_lat = 60
        self.n_lon = 120

    def blur_counts(self):
        """
        Applies volume normalization and Gaussian smoothing on raw counts
        """

        # Load raw counts
        loaded = np.load(DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz')
        counts_1d = loaded['arr_1d'].astype(np.float32)
        counts_2d = loaded['arr_2d'].astype(np.float32)
        counts_3d = loaded['arr_3d'].astype(np.float32)

        ### Step 1: volume normalization ###

        # 1D case
        volume_corrections_1d = np.zeros((counts_1d.shape[2]), dtype = np.float32)
        for i in range(volume_corrections_1d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            volume_corrections_1d[i] = 4 * np.pi * r_factor

        counts_1d = counts_1d / volume_corrections_1d[np.newaxis, np.newaxis, :]

        # 2D case
        volume_corrections_2d = np.zeros((counts_2d.shape[2], counts_2d.shape[3]), dtype = np.float32)
        for i in range(volume_corrections_2d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)
            for j in range(volume_corrections_2d.shape[1]):
                theta_i, theta_e = self.theta_bins[j], self.theta_bins[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                volume_corrections_2d[i,j] = 2 * np.pi * theta_factor * r_factor

        counts_2d = counts_2d / volume_corrections_2d[np.newaxis, np.newaxis, :, :]

        # 3D case
        volume_corrections_3d = np.zeros((counts_3d.shape[2], counts_3d.shape[3], counts_3d.shape[4]), dtype = np.float32)
        for i in range(volume_corrections_3d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)

            for j in range(volume_corrections_3d.shape[1]):
                theta_i, theta_e = self.theta_bins[j], self.theta_bins[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                for k in range(volume_corrections_3d.shape[2]):
                    phi_i, phi_e = self.phi_bins[k], self.phi_bins[k+1]
                    phi_factor = phi_e - phi_i

                    # Multiply by factor 4: with 2 unsigned axes, 4 voxels are always equivalent
                    volume_corrections_3d[i,j,k] = 4 * phi_factor * theta_factor * r_factor 

        counts_3d = counts_3d / volume_corrections_3d[np.newaxis, np.newaxis, :, :, :]

        del volume_corrections_1d, volume_corrections_2d, volume_corrections_3d

        ### Step 2: Map everything onto full sphere ###
        counts_1d = (counts_1d[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, self.n_lat, self.n_lon)))
        counts_2d = (counts_2d[:, :, :, :, np.newaxis] * np.ones((1, 1, 1, 1, self.n_lon)))
        counts_3d = np.concatenate([counts_3d, counts_3d[:, :, :, ::-1, :]], axis = 3)
        counts_3d = np.concatenate([counts_3d[:, :, :, :, ::-1], counts_3d], axis = 4)

        rho = np.concatenate([counts_1d, counts_2d, counts_3d], axis = 0)
        print(rho.shape)

        ### Step 3: SH smoothing + radial smoothing ###
        rho = gaussian_filter(rho, sigma = [0, 0, self.sigma_r, 0, 0])
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                print(f'{i} / {rho.shape[0]} - {j} / {rho.shape[1]}')
                for k in range(rho.shape[2]):
                    X = rho[i,j,k,:,:]
                    Xgrid = pysh.SHGrid.from_array(X)
                    Xcoeff = Xgrid.expand()
                    l = np.arange(Xcoeff.lmax + 1)
                    gauss = np.exp(-0.5  * l * (l + 1) * self.sigma_angle**2)
                    hann = 0.5 * (1 + np.cos(np.pi * l / l[-1]))      # 0 at lmax
                    lanczos = np.sinc(l / (l[-1] + 1))
                    lowpass = gauss * hann
                    filtered_coeffs = Xcoeff.copy()
                    for l_idx in range(Xcoeff.lmax + 1):
                        filtered_coeffs.coeffs[:, l_idx, :l_idx+1] *= lowpass[l_idx]
                    smoothed_grid = filtered_coeffs.expand(extend = False)
                    Xsmooth = smoothed_grid.to_array().clip(min = 0)
                    rho[i,j,k,:,:] = Xsmooth

        return rho
    
    def counts_to_prob(self, rho):
        """P(l | p, r, theta, phi) = n(p,l,r, theta, phi) / sum_l{n(p,l,r, theta, phi)}"""

        volume_corrections_3d = np.zeros((rho.shape[2], rho.shape[3], rho.shape[4]), dtype = np.float32)
        for i in range(volume_corrections_3d.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)

            for j in range(volume_corrections_3d.shape[1]):
                theta_i, theta_e = self.theta_bins[j], self.theta_bins[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                for k in range(volume_corrections_3d.shape[2]):
                    phi_i, phi_e = self.phi_bins[k], self.phi_bins[k+1]
                    phi_factor = phi_e - phi_i

                    volume_corrections_3d[i,j,k] = phi_factor * theta_factor * r_factor

        self.volume_corrections_3d = volume_corrections_3d / np.sum(volume_corrections_3d)

        lig_sum = np.sum(rho, axis = 1) # [p,r, theta, phi]
        xi = np.max(lig_sum, axis = (1,2,3), keepdims = True)
        decoy_vals = xi - lig_sum # Add void count to density, such that total density is equal across all spherical voxels
        rho = np.concatenate((rho, decoy_vals[:, np.newaxis, :, :, :]), axis = 1) # [p, l+1, r, theta, phi]

        cond_prob = rho / np.sum(rho, axis = 1, keepdims = True) # P(l | p,r,theta,phi)
        return cond_prob
    
    def ref_probs(self, cond_prob):
        """
        Compute the reference distribution P(l) according to self.ref_mode.

          - 'marginal': P(l) = sum_{p, r, theta, phi} P(p, l, r, theta, phi)
          - 'uniform':  P(l) = sum_p P(p) * mean_{r, theta, phi}[P(l | p, r, theta, phi)]

        Both branches return ref_prob with shape (1, L+1, 1, 1, 1) so it
        broadcasts against cond_prob in inverse_boltzmann the same way.
        """

        ref_prob = np.sum(cond_prob * self.volume_corrections_3d[np.newaxis, np.newaxis, :, :, :], axis = (2,3,4)) + 1e-12 # [p,l]
        return ref_prob[:, :, np.newaxis, np.newaxis, np.newaxis]

    def inverse_boltzmann(self, cond_prob, ref_prob):
        """
        score[p,l,r,theta,phi] = ln[P(l | p,r,theta,phi) / P(l)]
        """

        eps = 1e-12 # Lower bound, prevent 0 probabilities
        scores = cond_prob / ref_prob
        scores = np.clip(scores, eps, None)
        scores = np.clip(-1 * np.log10(scores), a_min = -5, a_max = 5)

        # Split back by symmetry class
        i1 = len(self.types_list_1d)
        i2 = i1 + len(self.types_list_2d)
        scores_1d = scores[:i1, :-1, :, :, :].mean(axis=(-1, -2))
        scores_2d = scores[i1:i2, :-1, :, :, :].mean(axis=-1)
        scores_3d = scores[i2:, :-1, :, :(self.n_lat // 2), (self.n_lon // 2):]

        # Include ref_mode in the filename so benchmarking runs do not overwrite each other.
        out_path = (
            DATA_DIR / 'potentials'
            / f'despot_scores_{self.database.lower()}.npz'
        )
        np.savez_compressed(out_path,
            scores_1d = scores_1d, scores_2d = scores_2d, scores_3d = scores_3d)
        print(f'Saved scores to {out_path}')

class DESPOT_DS_Builder:
    """
    Class for building isotropic statistical potentials
    """

    def __init__(self, database):

        # Set types lists for ligand atoms and protein atoms
        self.database = database
        prot_counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'prot_types.csv')
        lig_counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'lig_types.csv')

        self.types_list_1d = (
            prot_counts_df.loc[
                (prot_counts_df['local_reference_frame'] == 'Isotropic'),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            prot_counts_df.loc[
                (prot_counts_df['local_reference_frame'] == 'Axial'),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            prot_counts_df.loc[
                (prot_counts_df['local_reference_frame'] == 'Anisotropic'),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            lig_counts_df['atom_type']
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

        denom_arr = np.sum(self.rho, axis = 2, keepdims = True) + 1e-12 # [p,l,r]
        self.prob = self.rho / denom_arr

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

        print('Running inverse Boltzmann')
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

        np.savez_compressed(DATA_DIR / 'potentials' / f'drugscore_scores_{self.database.lower()}.npz', 
            scores_1d = self.scores)
