from src.config import DATA_DIR

import os
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_file(file_path):
    """
    Parse GNINA results from raw .txt file

    Returns
    -------

    results [Dict[str, Tuple[float, float, float]]]:
        -> lig_name: (cnn_score, cnn_affinity, cnn_vs)
    """

    results = {}
    cnn_score = np.nan
    cnn_affinity = np.nan

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('CNNscore'):
                cnn_score = -1 * float(line.split(':')[1].strip())

            if line.startswith('CNNaffinity'):
                cnn_affinity = -1 * float(line.split(':')[1].strip())

            if line.startswith('##') and cnn_score != np.nan:
                lig_name = line.split(' ')[1].strip()
                cnn_vs = -1 * np.abs(cnn_score * cnn_affinity)
                results[lig_name] = [cnn_score, cnn_affinity, cnn_vs]
                cnn_score, cnn_affinity = np.nan, np.nan

    return results

def main():

    # Step 1: scoring power
    # score_dict = {}
    subdir_list = os.listdir(f'{DATA_DIR}/CASF-2016/coreset')
    # for subdir in subdir_list:
    #     new_results = parse_file(f'{DATA_DIR}/CASF-2016/coreset/{subdir}/gnina_results.txt')
    #     old_key = list(new_results.keys())[0]
    #     score_dict[subdir] = new_results[old_key]

    # complexes = list(score_dict.keys())
    # cnn_scores = [x[0] for _, x in score_dict.items()]
    # cnn_affs = [x[1] for _, x in score_dict.items()]
    # cnn_vs = [x[2] for _, x in score_dict.items()]

    # score_df = pd.DataFrame({'pdb_id': complexes, 'cnn_score': cnn_scores, 'cnn_aff': cnn_affs, 'cnn_vs': cnn_vs})
    # baseline_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/deltavina_scorepower.csv')
    # merged_df = baseline_df.merge(score_df, how = 'inner', on = ['pdb_id'])

    # df1 = merged_df[['pdb_id', 'cnn_score', 'logKa']].rename(columns = {'cnn_score': 'score'})
    # df2 = merged_df[['pdb_id', 'cnn_aff', 'logKa']].rename(columns = {'cnn_aff': 'score'})
    # df3 = merged_df[['pdb_id', 'cnn_vs', 'logKa']].rename(columns = {'cnn_vs': 'score'})

    # df1.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnscore_scorepower.csv', index = False, float_format = '%.4f')
    # df2.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnaff_scorepower.csv', index = False, float_format = '%.4f')
    # df3.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnvs_scorepower.csv', index = False, float_format = '%.4f')

    # Step 2: docking power
    # score_dict = {}
    # for subdir in subdir_list:
    #     new_results = parse_file(f'{DATA_DIR}/CASF-2016/decoys_docking/gnina_results/{subdir}.txt')
    #     score_dict.update(new_results)

    # pdb_ids = [x.strip('_')[0] for x in score_dict.keys()]
    # pose_ids = [x.strip('_')[1] for x in score_dict.keys()]
    # cnn_scores = [x[0] for _, x in score_dict.items()]
    # cnn_affs = [x[1] for _, x in score_dict.items()]
    # cnn_vs = [x[2] for _, x in score_dict.items()]

    # score_df = pd.DataFrame({'pdb_id': pdb_ids, 'pose_id': pose_ids,
    #                          'cnn_score': cnn_scores, 'cnn_aff': cnn_affs, 'cnn_vs': cnn_vs})
    # baseline_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/deltavina_dockingpower.csv')
    # merged_df = baseline_df.merge(score_df, how = 'inner', on = ['pdb_id', 'pose_id'])

    # df1 = merged_df[['pdb_id', 'pose_id', 'rmsd', 'cnn_score']].rename(columns = {'cnn_score': 'score'})
    # df2 = merged_df[['pdb_id', 'pose_id', 'rmsd', 'cnn_aff']].rename(columns = {'cnn_aff': 'score'})
    # df3 = merged_df[['pdb_id', 'pose_id', 'rmsd', 'cnn_vs']].rename(columns = {'cnn_vs': 'score'})

    # df1.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnscore_dockingpower.csv', index = False, float_format = '%.4f')
    # df2.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnaff_dockingpower.csv', index = False, float_format = '%.4f')
    # df3.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnvs_dockingpower.csv', index = False, float_format = '%.4f')

    # Step 3: screening power
    score_dict = defaultdict(lambda: [float('inf'), float('inf'), float('inf')])
    for subdir in os.listdir(f'{DATA_DIR}/CASF-2016/decoys_screening'):
        new_results = parse_file(f'{DATA_DIR}/CASF-2016/decoys_screening/{subdir}/gnina_results.txt')

        for k, v in new_results.items():
            new_key = f'{subdir}_{k.split('_')[0]}'
            for i, value in enumerate(v):
                if value < score_dict[new_key][i]:
                    score_dict[new_key][i] = value

    pdb_ids = [x.split('_')[0] for x in score_dict.keys()]
    ligand_ids = [x.split('_')[1] for x in score_dict.keys()]
    cnn_scores = [x[0] for _, x in score_dict.items()]
    cnn_affs = [x[1] for _, x in score_dict.items()]
    cnn_vs = [x[2] for _, x in score_dict.items()]

    score_df = pd.DataFrame({'pdb_id': pdb_ids, 'ligand_id': ligand_ids,
                             'cnn_score': cnn_scores, 'cnn_aff': cnn_affs, 'cnn_vs': cnn_vs})
    baseline_df = pd.read_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/deltavina_screeningpower.csv')
    merged_df = baseline_df.merge(score_df, how = 'inner', on = ['pdb_id', 'ligand_id'])

    print(score_df)
    print(baseline_df)

    df1 = merged_df[['pdb_id', 'ligand_id', 'is_binder', 'cnn_score']].rename(columns = {'cnn_score': 'score'})
    df2 = merged_df[['pdb_id', 'ligand_id', 'is_binder', 'cnn_aff']].rename(columns = {'cnn_aff': 'score'})
    df3 = merged_df[['pdb_id', 'ligand_id', 'is_binder', 'cnn_vs']].rename(columns = {'cnn_vs': 'score'})

    df1.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnscore_screeningpower.csv', index = False, float_format = '%.4f')
    df2.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnaff_screeningpower.csv', index = False, float_format = '%.4f')
    df3.to_csv(f'{DATA_DIR}/CASF-2016/benchmark_results/gnina_cnnvs_screeningpower.csv', index = False, float_format = '%.4f')

if __name__ == '__main__':
    main()
