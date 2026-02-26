import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter

def _convert_file(filename, database):
    """Convert a single protein-ligand pair. Runs in worker process."""
    converter = MolConverter()

    prot_df = converter.convert_mol2(
        DATA_DIR / database / 'processed_mol2' / 'receptor' / filename
    )
    lig_df = converter.convert_mol2(
        DATA_DIR / database / 'processed_mol2' / 'ligand' / filename
    )

    return filename, prot_df, lig_df

class DESPOT_Counter:
    """
    Discretize protein-ligand interactions in geometric bins.
    """

    def __init__(self, database):

        self.database = database

        self.converter = MolConverter()
        self.file_list = os.listdir(DATA_DIR / self.database / 'processed_mol2' / 'receptor')

        self.r_bins = np.arange(1.0, 6.0, 0.1)
        self.theta_bins_2d = np.arange(0, 180.0, 3.0)
        self.theta_bins_3d = np.arange(0, 90.0, 3.0)
        self.phi_bins = np.arange(0, 180.0, 3.0)

        # Set types lists for ligand atoms and protein atoms

        counts_df = pd.read_csv(DATA_DIR / 'metadata' /f'atom_type_counts_{database.lower()}.csv')

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

        # Create fast lookup dictionaries
        self.type_1d_to_idx = {t: i for i, t in enumerate(self.types_list_1d)}
        self.type_2d_to_idx = {t: i for i, t in enumerate(self.types_list_2d)}
        self.type_3d_to_idx = {t: i for i, t in enumerate(self.types_list_3d)}
        self.ligand_type_to_idx = {t: i for i, t in enumerate(self.ligand_types_list)}

        # Initialize empty counts arrays
        self.bin_arr_1d = np.zeros((len(self.types_list_1d), len(self.ligand_types_list), len(self.r_bins)))
        self.bin_arr_2d = np.zeros((len(self.types_list_2d), len(self.ligand_types_list), len(self.r_bins), len(self.theta_bins_2d)))
        self.bin_arr_3d = np.zeros((len(self.types_list_3d), len(self.ligand_types_list), len(self.r_bins), len(self.theta_bins_3d), len(self.phi_bins)))

    def _process_interaction_pair(self, prot_df, lig_df):
        """
        Updates counts arrays with contacts from one protein-ligand pair.

        Parameters
        ----------
        prot_df [pd.DataFrame]:
            - atom_type (str)
            - hybridization (str)
            - heavy_neighbors (int)
            - [x, y, z] (float)
            - [v1_{x,y,z}, v2_{x,y,z}, v3_{x,y,z}] (float)

        lig_df [pd.DataFrame]:
            - atom_type (str)
            - [x, y, z] (float)
        """

        # Extract data as numpy arrays to avoid repeated DataFrame access
        prot_types = prot_df['atom_type'].values
        prot_hybridizations = prot_df['hybridization'].values
        prot_neighbors = prot_df['heavy_neighbors'].values
        prot_coords = prot_df[['x', 'y', 'z']].values
        v1_arr = prot_df[['v1_x', 'v1_y', 'v1_z']].values
        v2_arr = prot_df[['v2_x', 'v2_y', 'v2_z']].values
        v3_arr = prot_df[['v3_x', 'v3_y', 'v3_z']].values

        lig_subset = lig_df[lig_df['atom_type'].isin(self.ligand_types_list)]
        lig_types = lig_subset['atom_type'].values
        lig_coords = lig_subset[['x', 'y', 'z']].values

        # Build KDTree
        prot_tree = KDTree(prot_coords)
        nearby_indices = prot_tree.query_ball_point(lig_coords, r = 6)

        # Process interactions
        for i in range(lig_coords.shape[0]):
            l_coord = lig_coords[i]
            l_type = lig_types[i]
            l_idx = self.ligand_type_to_idx[l_type]

            for j in nearby_indices[i]:
                p_coord = prot_coords[j]
                p_type = prot_types[j]
                v1 = v1_arr[j]
                v2 = v2_arr[j]
                v3 = v3_arr[j]

                # Calculate interaction vector and discretize distance
                int_vector = l_coord - p_coord
                dist = np.clip(np.linalg.norm(int_vector), 0.01, 5.99)
                r_idx = int(dist * 10) - 10

                # Handle 1D types (r only)
                if p_type in self.types_list_1d:
                    p_idx = self.type_1d_to_idx[p_type]
                    self.bin_arr_1d[p_idx, l_idx, r_idx] += 1

                # Handle 2D types (axially symmetric)
                elif p_type in self.types_list_2d:
                    p_idx = self.type_2d_to_idx[p_type]

                    if not np.isnan(v1[0]): 
                        cos_theta = np.dot(int_vector, v1) / np.linalg.norm(int_vector)
                        theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        theta = np.clip(theta, 0.01, 179.99)
                        theta_idx = int(theta / 3)

                        # Handle planar symmetry
                        if prot_hybridizations[j] == 'sp2' and prot_neighbors[j] == 3:
                            new_theta_idx = int((180 - theta) / 3)
                            self.bin_arr_2d[p_idx, l_idx, r_idx, theta_idx] += 0.5
                            self.bin_arr_2d[p_idx, l_idx, r_idx, new_theta_idx] += 0.5

                        else:
                            self.bin_arr_2d[p_idx, l_idx, r_idx, theta_idx] += 1.0

                # Handle 3D types (fully anisotropic)
                elif p_type in self.types_list_3d:
                    p_idx = self.type_3d_to_idx[p_type]

                    if not np.isnan(v2[0]):
                        cos_theta = np.abs(np.dot(int_vector, v1) / np.linalg.norm(int_vector))
                        theta = np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0)))
                        theta = np.clip(theta, 0.01, 89.99)
                        theta_idx = int(theta / 3)

                        phi = np.degrees(np.atan2(np.abs(np.dot(int_vector, v3)), np.dot(int_vector, v2)))
                        phi = np.clip(phi, 0.01, 179.99)
                        phi_idx = int(phi / 3)

                        self.bin_arr_3d[p_idx, l_idx, r_idx, theta_idx, phi_idx] += 1

    def find_interactions(self):

        num_files = len(self.file_list)

        for i, file in enumerate(self.file_list):
            filename, prot_df, lig_df = _convert_file(file, self.database)

            self._process_interaction_pair(prot_df, lig_df)
            self._process_interaction_pair(lig_df, prot_df)

            print(f'{i} / {num_files} done')

        np.savez_compressed(
            DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz',
            arr_1d=self.bin_arr_1d,
            arr_2d=self.bin_arr_2d,
            arr_3d=self.bin_arr_3d
        )

    def find_interactions_parallel(self, n_workers=4, max_queued=8):
        """
        Loop over all files with parallel MOL2 conversion feeding a processing queue.
        """
        num_files = len(self.file_list)
        
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            pending = set()
            file_iter = iter(self.file_list)
            
            for f in itertools.islice(file_iter, max_queued):
                pending.add(executor.submit(_convert_file, f, self.database))
            
            with tqdm(total=num_files, desc="Processing structures", unit="file") as pbar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    
                    for future in done:
                        try:
                            filename, prot_df, lig_df = future.result()
                            
                            if prot_df is not None and lig_df is not None:
                                self._process_interaction_pair(prot_df, lig_df)
                                self._process_interaction_pair(lig_df, prot_df)
                                pbar.set_postfix_str(filename[:20])
                            else:
                                pbar.write(f"{filename} (skipped)")
                                
                        except Exception as e:
                            pass
                        
                        pbar.update(1)
                        
                        try:
                            next_file = next(file_iter)
                            pending.add(executor.submit(_convert_file, next_file, self.database))
                        except StopIteration:
                            pass

        np.savez_compressed(
            DATA_DIR / 'potentials' / f'despot_counts_{self.database.lower()}.npz',
            arr_1d=self.bin_arr_1d,
            arr_2d=self.bin_arr_2d,
            arr_3d=self.bin_arr_3d
        )
