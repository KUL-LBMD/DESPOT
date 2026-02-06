from src.config import DATA_DIR

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def get_scoring_values(name_list):
    """
    Returns
    -------
    df: 285 rows for 285 targets. S different scores + logKa
    """
    df = pd.read_csv(f'{DATA_DIR}/CASF-2016/power_scoring/CoreSet.dat', sep=r'\s+')

    for name in name_list:
        new_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/{name}_scorepower.csv')
        new_df.rename(columns={'score': f'{name}_score'}, inplace=True)
        new_df[f'{name}_score'] = -1 * new_df[f'{name}_score']
        df = pd.merge(df, new_df, on= ['pdb_id', 'logKa'])

    return df

def get_ranking_values(name_list):
    """
    Returns
    -------
    spearman_arr [S, 57]: 57 columns for 57 targets. 
    Spearman rank correlations per target for S different scores.
    Row order matches name_list order.
    """
    S = len(name_list)
    df = pd.read_csv(f'{DATA_DIR}/CASF-2016/power_scoring/CoreSet.dat', sep=r'\s+')

    # Get number of unique targets
    for name in name_list:
        new_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/{name}_scorepower.csv')
        new_df.rename(columns={'score': f'{name}_score'}, inplace=True)
        new_df[f'{name}_score'] = -1 * new_df[f'{name}_score']
        df = pd.merge(df, new_df, on= ['pdb_id', 'logKa'])

    unique_targets = df['target'].unique()
    n_targets = len(unique_targets)
    spearman_arr = np.zeros((S, n_targets))

    # Process Spearman correlation per group
    for i, group in enumerate(unique_targets):
        subset = df[df['target'] == group]
        for j, name in enumerate(name_list):
            spearman_arr[j, i] = spearmanr(subset['logKa'], subset[f'{name}_score'])[0]

    return spearman_arr

def get_docking_values(name_list):
    """
    Returns
    -------
    top_arr [S, 3]: Top-1, -2 and -3 recovery of true crystal pose.
    Row order matches name_list order.
    """
    S = len(name_list)
    top_arr = np.zeros((S, 3))
    
    # FIX: Properly merge all CSVs on 'code' column to ensure alignment
    # Instead of assuming all CSVs have rows in the same order
    merged_df = None
    
    for name in name_list:
        df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/{name}_dockingpower.csv')
        df.rename(columns={'score': f'{name}_score'}, inplace=True)
        df[f'{name}_score'] = -1 * df[f'{name}_score']
        df['pose_id'] = df['pose_id'].astype(str)

        if merged_df is None:
            # First CSV: keep code and rmsd as reference
            merged_df = df.copy()
        else:
            # Subsequent CSVs: merge on 'code' to ensure proper alignment
            merged_df = pd.merge(
                merged_df, 
                df[['pdb_id', 'pose_id', f'{name}_score']], 
                on=['pdb_id', 'pose_id'],
                how='inner'
            )
    
    # Build col_list in the same order as name_list for consistent indexing
    score_cols = [f'{name}_score' for name in name_list]
    
    unique_targets = merged_df['pdb_id'].unique()
    n_targets = len(unique_targets)

    for group in unique_targets:
        subset = merged_df[merged_df['pdb_id'] == group].copy()
        
        for i, score_col in enumerate(score_cols):
            # Sort by score (descending) and find rank of poses with RMSD < 2.0
            subset_sorted = subset.sort_values(by=[score_col], ascending=False, ignore_index=True)
            idx_list = subset_sorted.index[subset_sorted['rmsd'] < 2.0].tolist()
            
            if idx_list:
                real_idx = idx_list[0]
            else:
                real_idx = 4  # Not found in top positions

            if real_idx < 3:
                top_arr[i, 2] += 1 / n_targets
            if real_idx < 2:
                top_arr[i, 1] += 1 / n_targets
            if real_idx < 1:
                top_arr[i, 0] += 1 / n_targets

    # Other task: get spearman thresholds
    threshold_list = list(range(2, 11))
    spearman_arr = np.zeros((S, 285, 9))

    for i, name in enumerate(name_list):
        for j, group in enumerate(merged_df['pdb_id'].unique()):
            subset = merged_df[merged_df['pdb_id'] == group].copy()
            for k, threshold in enumerate(threshold_list):
                subsubset = subset[subset['rmsd'] < threshold].copy()
                if len(subsubset) > 0:
                    value = spearmanr(subsubset['rmsd'], subsubset[f'{name}_score'])[0]
                    if not np.isnan(value):
                        spearman_arr[i,j,k] = value

    spearman_thresholds = -1 * np.mean(spearman_arr, axis = 1)

    return top_arr, spearman_thresholds

def get_screening_values(name_list):
    """
    Returns
    -------
    df: DataFrame with screening scores
    forward_top_arr [S, 3]: Forward success rates at 1%, 5%, 10%
    reverse_top_arr [S, 3]: Reverse success rates at 2%, 5%, 10%
    Row order matches name_list order.
    """
    S = len(name_list)
    forward_top_arr = np.zeros((S, 3))
    reverse_top_arr = np.zeros((S, 3))
    
    # FIX: Properly merge all CSVs on protein-ligand pairs
    merged_df = None
    
    for name in name_list:
        flat_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/{name}_screeningpower.csv')
        flat_df.rename(columns = {'score': f'{name}_score'}, inplace = True)
        flat_df[f'{name}_score'] = -1 * flat_df[f'{name}_score']

        # 🔧 FIX: keep only the *most positive* score per (pdb_id, ligand_id)
        flat_df = (
            flat_df
            .groupby(['pdb_id', 'ligand_id', 'is_binder'], as_index=False)[f'{name}_score']
            .max()
        )
        
        if merged_df is None:
            merged_df = flat_df.copy()
        else:
            merged_df = pd.merge(
                merged_df,
                flat_df[['pdb_id', 'ligand_id', f'{name}_score']],
                on=['pdb_id', 'ligand_id'],
                how='inner'
            )
    
    # Build score column list in same order as name_list
    score_cols = [f'{name}_score' for name in name_list]

    # 4.1: Forward success rate
    unique_proteins = merged_df['pdb_id'].unique()
    n_proteins = len(unique_proteins)

    for group in unique_proteins:
        subset = merged_df[merged_df['pdb_id'] == group].copy()
        for i, score_col in enumerate(score_cols):
            subset_sorted = subset.copy()
            subset_sorted.sort_values(by=[score_col], ascending=False, inplace=True, ignore_index=True)
            # Take best-scoring pose per protein-ligand pair
            subset_sorted.drop_duplicates(subset=['pdb_id', 'ligand_id'], keep='first', inplace=True, ignore_index=True)
            # Find highest position of 1 positive
            real_idx = subset_sorted['is_binder'].idxmax()
            real_pos = (real_idx + 1) / len(subset_sorted)

            if real_pos < 0.05:
                forward_top_arr[i, 2] += 1 / n_proteins
            if real_pos < 0.02:
                forward_top_arr[i, 1] += 1 / n_proteins
            if real_pos < 0.01:
                forward_top_arr[i, 0] += 1 / n_proteins

    ### 4.2: Reverse success rate ###
    unique_ligands = merged_df['ligand_id'].unique()
    n_ligands = len(unique_ligands)

    for group in unique_ligands:
        subset = merged_df[merged_df['ligand_id'] == group].copy()
        for i, score_col in enumerate(score_cols):
            subset_sorted = subset.copy()
            subset_sorted.sort_values(by=[score_col], ascending=False, inplace=True, ignore_index=True)
            subset_sorted.drop_duplicates(subset=['pdb_id', 'ligand_id'], keep='first', inplace=True, ignore_index=True)
            real_idx = subset_sorted['is_binder'].idxmax()
            real_pos = (real_idx + 1) / len(subset_sorted)

            if real_pos < 0.1:
                reverse_top_arr[i, 2] += 1 / n_ligands
            if real_pos < 0.05:
                reverse_top_arr[i, 1] += 1 / n_ligands
            if real_pos < 0.02:
                reverse_top_arr[i, 0] += 1 / n_ligands

    return merged_df, forward_top_arr, reverse_top_arr
    
def get_enrichment_factors(df, name_list):
    """
    Returns
    -------
    ef_arr [S, 57, 3]: enrichment factors for 57 targets at 3 different levels.
    Row order matches name_list order.
    """
    
    threshold_list = [0.01, 0.02, 0.05]
    unique_proteins = df['pdb_id'].unique()
    num_groups = len(unique_proteins)
    S = len(name_list)
    ef_arr = np.zeros((S, num_groups, len(threshold_list)))

    score_cols = [f'{name}_score' for name in name_list]

    for j, group in enumerate(unique_proteins):
        subset = df[df['pdb_id'] == group].copy()
        for i, name in enumerate(score_cols):
            subset_sorted = subset.copy()
            subset_sorted.sort_values(by=[name], ascending=False, inplace=True, ignore_index=True)
            subset_sorted.drop_duplicates(subset=['pdb_id', 'ligand_id'], keep='first', inplace=True, ignore_index=True)

            for k, threshold in enumerate(threshold_list):
                total_actives = subset_sorted['is_binder'].sum()
                total_entries = len(subset_sorted)
                subset_entries = int(threshold * total_entries)
                subsubset = subset_sorted.head(subset_entries).copy()
                subset_actives = subsubset['is_binder'].sum()

                ef = (subset_actives / total_actives) / (subset_entries / total_entries)
                ef_arr[i, j, k] = ef

    return ef_arr
