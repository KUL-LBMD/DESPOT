from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter
from src.core.complex_scorer import DESPOT_Scorer, DESPOT_Isotropic_Scorer

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import math

def run_scoring(database):
	scorer1 = DESPOT_Scorer(mode = 'gaussian', database = database) # DESPOT
	scorer2 = DESPOT_Isotropic_Scorer(mode = 'mif', database = database) # DESPOT-Iso
	converter = MolConverter()

	# 1.1-1.2: Scoring + ranking power
	print('Starting scoring benchmark')

	score_list_of_dicts = []
	subdirs = os.listdir(DATA_DIR / 'CASF-2016' / 'coreset')

	for subdir in tqdm(subdirs, desc = 'Scoring'):

		prot_df = converter.convert_mol2(f'{DATA_DIR}/CASF-2016/coreset/{subdir}/{subdir}_protein.mol2')
		lig_df = converter.convert_mol2(f'{DATA_DIR}/CASF-2016/coreset/{subdir}/{subdir}_ligand.mol2')

		score1 = np.sum(scorer1.score_complex(prot_df, lig_df))
		score2 = np.sum(scorer2.score_complex(prot_df, lig_df))

		score_dict = {'pdb_id': subdir, 'score1': score1, 'score2': score2}
		score_list_of_dicts.append(score_dict)

	score_df = pd.DataFrame(score_list_of_dicts)

	# Add logKa information
	aff_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/power_scoring/CoreSet.dat', sep=r"\s+")[['pdb_id', 'logKa']]
	score_df = pd.merge(score_df, aff_df, on = 'pdb_id')

	df1 = score_df[['pdb_id', 'logKa', 'score1']].copy().rename(columns = {'score1': 'score'})
	df2 = score_df[['pdb_id', 'logKa', 'score2']].copy().rename(columns = {'score2': 'score'})

	df1.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/despot_{database.lower()}_scorepower.csv', index = False, float_format = '%.4f')
	df2.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/despot_iso_{database.lower()}_scorepower.csv', index = False, float_format = '%.4f')

def run_docking(database):
	scorer1 = DESPOT_Scorer(mode = 'gaussian', database = database) # DESPOT
	scorer2 = DESPOT_Isotropic_Scorer(mode = 'mif', database = database) # DESPOT-Iso
	converter = MolConverter()

	print('Starting docking benchmark')

	df_list = []
	subdirs = os.listdir(DATA_DIR / 'CASF-2016' / 'coreset')

	for subdir in tqdm(subdirs, desc = 'Docking'):

		code_list = []
		concat_dfs = []
		prot_df = converter.convert_mol2(f'{DATA_DIR}/CASF-2016/coreset/{subdir}/{subdir}_protein.mol2')
		file_list = sorted([x for x in os.listdir(f'{DATA_DIR}/CASF-2016/decoys_docking/single_mols') if x[:4] == subdir])

		for i, filename in enumerate(file_list):
			temp_df = converter.convert_mol2(f'{DATA_DIR}/CASF-2016/decoys_docking/single_mols/{filename}')
			temp_df['label_num'] = i
			concat_dfs.append(temp_df)
			code_list.append(os.path.splitext(filename)[0])

		# Score all ligands simultaneously
		lig_df = pd.concat(concat_dfs, axis = 0)
		scores1_init = scorer1.score_complex(prot_df, lig_df)
		scores2_init = scorer2.score_complex(prot_df, lig_df)

		# Now trace back scorer per ligand
		labels = lig_df['label_num'].values.astype(np.int64)
		score1 = list(np.bincount(labels, weights = scores1_init))
		score2 = list(np.bincount(labels, weights = scores2_init))

		temp_df = pd.DataFrame(
			{'code': code_list,
			 'score1': score1,
			 'score2': score2
			})

		rmsd_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/decoys_docking/{subdir}_rmsd.csv')
		subdir_df = pd.merge(temp_df, rmsd_df, on = 'code')
		df_list.append(subdir_df)

	dock_df = pd.concat(df_list, axis = 0)
	dock_df['pdb_id'] = dock_df['code'].str.split('_').str[0]
	dock_df['pose_id'] = dock_df['code'].str.split('_').str[1]

	df1 = dock_df[['pdb_id', 'pose_id', 'rmsd', 'score1']].copy().rename(columns = {'score1': 'score'})
	df2 = dock_df[['pdb_id', 'pose_id', 'rmsd', 'score2']].copy().rename(columns = {'score2': 'score'})

	df1.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/despot_{database.lower()}_dockingpower.csv', index = False, float_format = '%.4f')
	df2.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/despot_iso_{database.lower()}_dockingpower.csv', index = False, float_format = '%.4f')

def run_screening(n_jobs=-1, database = 'CROWN'):
    print('Starting screening benchmark')
    molecule_list = [f'{subdir}_{i+1}' for subdir in sorted(os.listdir(DATA_DIR / 'CASF-2016' / 'coreset')) for i in range(100)]
    target_list = sorted(os.listdir(DATA_DIR / 'CASF-2016' / 'decoys_screening'))

    chunk_size = 2000

    def process_target(subdir, database):
        """Process a single target protein against all molecules."""
        scorer1 = DESPOT_Scorer(mode = 'gaussian', database = database)
        scorer2 = DESPOT_Isotropic_Scorer(mode='mif', database = database)
        converter = MolConverter()

        prot_df = converter.convert_mol2(f'{DATA_DIR}/CASF-2016/coreset/{subdir}/{subdir}_protein.mol2')
        num_chunks = math.ceil(len(molecule_list) / chunk_size)

        target_scores = np.zeros((2, len(molecule_list)), dtype=np.float32)

        for j in range(num_chunks):
            subset_list = molecule_list[(j*chunk_size):((j+1)*chunk_size)]
            concat_dfs = []

            for k, basename in enumerate(subset_list):
                temp_df = converter.convert_mol2(f'{DATA_DIR}/CASF-2016/decoys_screening/{subdir}/single_mols/{basename}.mol2')
                temp_df['label_num'] = k
                concat_dfs.append(temp_df)

            lig_df = pd.concat(concat_dfs, axis=0)
            scores1_init = scorer1.score_complex(prot_df, lig_df)
            scores2_init = scorer2.score_complex(prot_df, lig_df)

            if scores1_init is not None:
                labels = lig_df['label_num'].values.astype(np.int64)
                score1 = np.bincount(labels, weights=scores1_init)
                score2 = np.bincount(labels, weights=scores2_init)
            else:
                score1 = np.full(len(subset_list), np.nan)
                score2 = np.full(len(subset_list), np.nan)

            target_scores[0, (j*chunk_size):((j+1)*chunk_size)] = score1
            target_scores[1, (j*chunk_size):((j+1)*chunk_size)] = score2

        return target_scores

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_target)(subdir, database) for subdir in tqdm(target_list, desc='Screening')
    )

    # Stack results into final array
    score_arr = np.stack(results, axis=1)  # Shape: (4, n_targets, n_molecules)

    df1 = pd.DataFrame(score_arr[0,:,:], index=target_list, columns=molecule_list)
    df2 = pd.DataFrame(score_arr[1,:,:], index=target_list, columns=molecule_list)

    file_path = f"{DATA_DIR}/CASF-2016/power_screening/TargetInfo.dat"
    target_dict = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        T = parts[0]
        Ls = parts[1:]
        target_dict[T] = Ls

    df_list = [df1, df2]
    name_list = ['despot', 'despot_iso']

    for df, name in zip(df_list, name_list):
        df_long = (
            df
            .reset_index()                     # turn index into a column
            .rename(columns={"index": "pdb_id"})
            .melt(
                id_vars="pdb_id",
                var_name="ligand_id",
                value_name="score"
            )
        )

        df_long['ligand_id'] = df_long['ligand_id'].str.split('_').str[0]
        df_long['is_binder'] = df_long.apply(
            lambda r: int(r['ligand_id'] in target_dict.get(r['pdb_id'], [])),
            axis=1
        )

        df_long.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/{name}_{database.lower()}_screeningpower.csv', float_format='%.4f')
