import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial import KDTree

from src.config import DATA_DIR

class DESPOT_Scorer:
    """
    Class for scoring protein-ligand complexes
    """

    def __init__(self):

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')
        colname1, colname2 = 'total_occurrence', 'total_occurrence'

        self.types_list_1d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Isotropic') &
                (counts_df[colname1] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Axial') &
                (counts_df[colname1] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Anisotropic') &
                (counts_df[colname1] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            counts_df.loc[
                (counts_df[colname2] > 500),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_type_to_idx = {t: i for i, t in enumerate(self.ligand_types_list)}

        # Load score matrices
        loaded = np.load(DATA_DIR / 'potentials' / 'despot_scores.npz')
        scores_1d = loaded['scores_1d']
        scores_2d = loaded['scores_2d']
        scores_3d = loaded['scores_3d']

        r_bins = np.arange(1.0, 6.0, 0.1)
        theta_bins_2d = np.deg2rad(np.arange(0, 180.0, 3.0))
        theta_bins_3d = np.deg2rad(np.arange(0, 90.0, 3.0))
        phi_bins = np.deg2rad(np.arange(0, 180.0, 3.0))

        ### Initialize interpolators ###
        self.interpolator_dict_1d = {}
        self.interpolator_dict_2d = {}
        self.interpolator_dict_3d = {}

        for i, p_type in enumerate(self.types_list_1d):
            arr_dict = {}
            for j, l_type in enumerate(self.ligand_types_list):
                interpolator = interp1d(r_bins, scores_1d[i,j,:], bounds_error = False, fill_value = 0)
                arr_dict[l_type] = interpolator
            self.interpolator_dict_1d[p_type] = arr_dict

        for i, p_type in enumerate(self.types_list_2d):
            arr_dict = {}
            for j, l_type in enumerate(self.ligand_types_list):
                interpolator = RegularGridInterpolator((r_bins, theta_bins_2d), scores_2d[i,j,:,:], bounds_error = False, fill_value = 0)
                arr_dict[l_type] = interpolator
            self.interpolator_dict_2d[p_type] = arr_dict

        for i, p_type in enumerate(self.types_list_3d):
            arr_dict = {}
            for j, l_type in enumerate(self.ligand_types_list):
                interpolator = RegularGridInterpolator((r_bins, theta_bins_3d, phi_bins), scores_3d[i,j,:,:,:], bounds_error = False, fill_value = 0)
                arr_dict[l_type] = interpolator
            self.interpolator_dict_3d[p_type] = arr_dict

    def score_complex(self, prot_df, lig_df):
        """
        Updates counts arrays with contacts from one protein-ligand pair.

        Parameters
        ----------
        prot_df [pd.DataFrame]:
            - atom_type (str)
            - [x, y, z] (float)
            - [v1_{x,y,z}, v2_{x,y,z}, v3_{x,y,z}] (float)

        lig_df [pd.DataFrame]:
            - atom_type (str)
            - [x, y, z] (float)

        Returns
        -------

        b_factors [List]: score for each heavy ligand atom
        """

        # Extract data as numpy arrays to avoid repeated DataFrame access
        prot_types = prot_df['atom_type'].values
        prot_coords = prot_df[['x', 'y', 'z']].values
        v1_arr = prot_df[['v1_x', 'v1_y', 'v1_z']].values
        v2_arr = prot_df[['v2_x', 'v2_y', 'v2_z']].values
        v3_arr = prot_df[['v3_x', 'v3_y', 'v3_z']].values

        lig_types = lig_df['atom_type'].values
        lig_coords = lig_df[['x', 'y', 'z']].values

        # Initialize score array
        # Group ligand atoms by type
        b_factors = np.zeros(len(lig_coords))
        lig_type_groups = {}
        for l_type in self.ligand_types_list:
            mask = lig_types == l_type
            if mask.any():
                lig_type_groups[l_type] = np.where(mask)[0]

        # Obtain list of pocket atoms, within 6A of interface
        prot_tree = KDTree(prot_coords)
        neighbors = prot_tree.query_ball_point(lig_coords, r = 6.0)
        flat_neighbors = [idx for sublist in neighbors for idx in sublist]
        if not flat_neighbors:
            return b_factors
        prot_indices = np.unique(flat_neighbors).astype(int)

        for i in prot_indices:

            p_coord = prot_coords[i]
            p_type = prot_types[i]
            v1 = v1_arr[i]
            v2 = v2_arr[i]
            v3 = v3_arr[i]

            int_vectors = lig_coords - p_coord
            distances = np.linalg.norm(int_vectors, axis = 1)

            # 1D case
            if p_type in self.types_list_1d:
                for l_type, lig_indices in lig_type_groups.items():
                    interpolator = self.interpolator_dict_1d[p_type][l_type]
                    b_factors[lig_indices] += interpolator(distances[lig_indices])

            # 2D case
            elif p_type in self.types_list_2d and not np.isnan(v1[0]):
                cos_theta = (int_vectors @ v1) / distances
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                points = np.column_stack((distances, theta))

                for l_type, lig_indices in lig_type_groups.items():           
                    interpolator = self.interpolator_dict_2d[p_type][l_type]
                    b_factors[lig_indices] += interpolator(points[lig_indices])

            # 3D case
            elif p_type in self.types_list_3d and not np.isnan(v2[0]):
                cos_theta = np.abs((int_vectors @ v1) / distances)
                theta = np.arccos(np.clip(cos_theta, 0.0, 1.0))
                phi = np.atan2(np.abs(int_vectors @ v3), int_vectors @ v2)
                points = np.column_stack((distances, theta, phi))

                for l_type, lig_indices in lig_type_groups.items():           
                    interpolator = self.interpolator_dict_3d[p_type][l_type]
                    b_factors[lig_indices] += interpolator(points[lig_indices])

        return b_factors
    
class DESPOT_Isotropic_Scorer:
    """
    Class for scoring protein-ligand complexes
    """

    def __init__(self, mode):

        # Set types lists for ligand atoms and protein atoms
        counts_df = pd.read_csv(DATA_DIR / 'metadata' / 'atom_type_counts.csv')
        colname1, colname2 = 'total_occurrence', 'total_occurrence'

        # Load score matrices
        if mode == 'mif':
            loaded = np.load(DATA_DIR / 'potentials' / 'despot_iso_scores.npz')
        else:
            loaded = np.load(DATA_DIR / 'potentials' / 'despot_ds_scores.npz')

        self.types_list_1d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Isotropic') &
                (counts_df[colname1] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_2d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Axial') &
                (counts_df[colname1] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.types_list_3d = (
            counts_df.loc[
                (counts_df['local_reference_frame'] == 'Anisotropic') &
                (counts_df[colname1] > 1000),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.ligand_types_list = (
            counts_df.loc[
                (counts_df[colname2] > 500),
                'atom_type'
            ]
            .dropna()
            .unique()
            .tolist()
        )

        self.prot_types_list = self.types_list_1d + self.types_list_2d + self.types_list_3d
        self.ligand_type_to_idx = {t: i for i, t in enumerate(self.ligand_types_list)}

        scores_1d = loaded['scores_1d']
        r_bins = np.arange(1.0, 6.0, 0.1)

        ### Initialize interpolators ###
        self.interpolator_dict = {}

        for i, p_type in enumerate(self.prot_types_list):
            arr_dict = {}
            for j, l_type in enumerate(self.ligand_types_list):
                interpolator = interp1d(r_bins, scores_1d[i,j,:], bounds_error = False, fill_value = 0)
                arr_dict[l_type] = interpolator
            self.interpolator_dict[p_type] = arr_dict

    def score_complex(self, prot_df, lig_df):
        """
        Updates counts arrays with contacts from one protein-ligand pair.

        Parameters
        ----------
        prot_df [pd.DataFrame]:
            - atom_type (str)
            - [x, y, z] (float)
            - [v1_{x,y,z}, v2_{x,y,z}, v3_{x,y,z}] (float)

        lig_df [pd.DataFrame]:
            - atom_type (str)
            - [x, y, z] (float)

        Returns
        -------

        b_factors [List]: score for each heavy ligand atom
        """

        # Extract data as numpy arrays to avoid repeated DataFrame access
        prot_types = prot_df['atom_type'].values
        prot_coords = prot_df[['x', 'y', 'z']].values

        lig_types = lig_df['atom_type'].values
        lig_coords = lig_df[['x', 'y', 'z']].values

        # Initialize score array
        # Group ligand atoms by type
        b_factors = np.zeros(len(lig_coords))
        lig_type_groups = {}
        for l_type in self.ligand_types_list:
            mask = lig_types == l_type
            if mask.any():
                lig_type_groups[l_type] = np.where(mask)[0]

        # Obtain list of pocket atoms, within 6A of interface
        prot_tree = KDTree(prot_coords)
        neighbors = prot_tree.query_ball_point(lig_coords, r = 6.0)
        flat_neighbors = [idx for sublist in neighbors for idx in sublist]
        if not flat_neighbors:
            return b_factors
        prot_indices = np.unique(flat_neighbors).astype(int)

        for i in prot_indices:
            p_coord = prot_coords[i]
            p_type = prot_types[i]

            int_vectors = lig_coords - p_coord
            distances = np.linalg.norm(int_vectors, axis = 1)

            # 1D case
            if p_type in self.prot_types_list:
                for l_type, lig_indices in lig_type_groups.items():
                    interpolator = self.interpolator_dict[p_type][l_type]
                    b_factors[lig_indices] += interpolator(distances[lig_indices])

        return b_factors
