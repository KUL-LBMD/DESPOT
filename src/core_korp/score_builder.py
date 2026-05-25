import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import pyshtools as pysh

from src.config import DATA_DIR

class KORP_Builder:
    """
    Bootleg KORP: anisotropic knowledge-based potential
    """

    def __init__(self, database):

        # Set types lists for ligand atoms and protein atoms
        self.database = database

        self.prot_types_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        
        lig_df = pd.read_csv(DATA_DIR / 'metadata' / 'lig_types.csv')
        self.lig_types_list = lig_df['atom_type'].tolist()

        self.r_bins = np.arange(2.0, 11.1, 0.1)
        self.theta_bins = np.deg2rad(np.arange(0, 183.0, 3.0))
        self.phi_bins = np.deg2rad(np.arange(0, 363.0, 3.0))

        self.sigma_r = 2
        self.sigma_angle = 0.1
        self.eps = 1e-16

    def blur_counts(self):

        # Load raw counts
        loaded = np.load(DATA_DIR / 'potentials' / f'korp_counts_{self.database.lower()}.npz')
        counts = loaded['counts'].astype(np.float32)

        ### Step 1: volume normalization ###
        volume_corrections = np.zeros((counts.shape[2], counts.shape[3], counts.shape[4]), dtype = np.float32)
        for i in range(volume_corrections.shape[0]):
            r_i, r_e = self.r_bins[i], self.r_bins[i+1]
            r_mid = (r_i + r_e) / 2
            r_factor = r_mid**2 * (r_e - r_i)

            for j in range(volume_corrections.shape[1]):
                theta_i, theta_e = self.theta_bins[j], self.theta_bins[j+1]
                theta_mid = (theta_i + theta_e) / 2
                theta_factor = np.sin(theta_mid) * (theta_e - theta_i)

                for k in range(volume_corrections.shape[2]):
                    phi_i, phi_e = self.phi_bins[k], self.phi_bins[k+1]
                    phi_factor = phi_e - phi_i

                    # Multiply by factor 4: with 2 unsigned axes, 4 voxels are always equivalent
                    volume_corrections[i,j,k] = phi_factor * theta_factor * r_factor

        rho = counts / volume_corrections[np.newaxis, np.newaxis, :, :, :]

        ### Step 2: SH smoothing + radial smoothing ###
        rho = gaussian_filter(rho, sigma = [0, 0, self.sigma_r, 0, 0])
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                print(f'{i} / {rho.shape[0]} - {j} / {rho.shape[1]}')
                for k in range(rho.shape[2]):
                    X = rho[i,j,k,:,:]
                    Xgrid = pysh.SHGrid.from_array(X)
                    Xcoeff = Xgrid.expand()
                    l = np.arange(Xcoeff.lmax + 1)
                    lowpass_filter = np.exp(-0.5  * l * (l + 1) * self.sigma_angle**2)
                    filtered_coeffs = Xcoeff.copy()
                    for l_idx in range(Xcoeff.lmax + 1):
                        filtered_coeffs.coeffs[:, l_idx, :l_idx+1] *= lowpass_filter[l_idx]
                    smoothed_grid = filtered_coeffs.expand(extend = False)
                    Xsmooth = smoothed_grid.to_array().clip(min=0)
                    rho[i,j,k,:,:] = Xsmooth

        return rho
    
    def counts_to_prob(self, rho):
        """
        P(r, theta, phi | p, l) = n(p,l,r,theta,phi) / sum_{r,theta,phi}[n(p,l,r,theta,phi)]
        """

        cond_prob = rho / np.sum(rho, axis = (2, 3, 4), keepdims = True)
        return cond_prob
    
    def ref_probs(self, rho):
        """
        P(r, theta, phi)
        """

        temp_prob = np.sum(rho, axis = (0,1))
        ref_prob = temp_prob / np.sum(temp_prob) # L1-normalization
        return ref_prob
    
    def inverse_boltzmann(self, cond_prob, ref_prob):
        """
        score[p,l,r,theta,phi] = log10[P(r,theta,phi | p,l) / P(r,theta,phi)]

        - cond_prob [p,l,r,theta,phi]
        - ref_prob [r,theta,phi]
        """

        scores = (cond_prob + self.eps) / (ref_prob[np.newaxis, np.newaxis, :, :, :] + self.eps)
        scores =  np.clip(-1 * np.log10(scores), a_min = -5, a_max = 5)

        # KORP: normalize per (p,l,r) shell?
        normed_scores = scores - np.mean(scores, axis = (3,4), keepdims = True)

        out_path = (
            DATA_DIR / 'potentials'
            / f'korp_scores_{self.database.lower()}.npz'
        )
        np.savez_compressed(out_path,
            scores = scores, normed_scores = normed_scores)