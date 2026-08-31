from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter
from src.atom_typing_korp.parse_mol2 import KorpMolConverter
from src.core.complex_scorer import DESPOT_Scorer, DESPOT_Isotropic_Scorer
from src.core_korp.complex_scorer import KORP_Scorer
from src.utils import split_mol2, write_pdbs

import numpy as np
import argparse
import pandas as pd
import os
import tempfile
from tqdm import tqdm

# This variable controls number of ligands to score in parallel.
# Feel free to adjust according to your memory constraints.

CHUNK_SIZE = 2000

def parse_arguments():
	"""Parse command line arguments for complex scoring."""

	parser = argparse.ArgumentParser(
		description = 'Use DESPOT to score protein-ligand interactions.',
		formatter_class = argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('-p', '--protein', help = 'Path (relative or full) to receptor .mol2 file. Example: 1cel_receptor.mol2')

	parser.add_argument('-l', '--ligand', help = 'Path (relative or full) to ligand (multi-)mol2 file. Example: 1cel_ligand.mol2')

	parser.add_argument('-o', '--outpath', help = 'Path (relative or full) to output csv file. Example: test_out.csv')

	parser.add_argument('--bfac', action = 'store_true',
		help = 'Make subdirectory that stores separate PDB file of each ligand pose, with atom-wise score stored as b-factor')

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()

	# Initialize scorer and converter
	converter = MolConverter()
	korp_converter = KorpMolConverter()

	scorer = DESPOT_Scorer(mode = 'despot', database = 'CROWN')
	iso_scorer = DESPOT_Isotropic_Scorer(mode = 'drugscore', database = 'CROWN')
	screen_scorer = KORP_Scorer(mode = 'korp', database = 'CROWN')

	# Initialize empty score list
	score1_list = []
	score2_list = []
	score3_list = []

	# Initialize directory for bfactors?
	if args.bfac:
		bfac_dir = args.outpath[:-4]
		os.makedirs(bfac_dir, exist_ok = True)

	# Split multi-mol2 file
	with tempfile.TemporaryDirectory() as tmp_dir:
		split_mol2(args.ligand, tmp_dir)

		# Convert protein and ligands
		prot_df = converter.convert_mol2(args.protein)
		prot_df_korp = korp_converter.convert_receptor(args.protein)

		file_list = os.listdir(tmp_dir)
		num_chunks = len(file_list) // CHUNK_SIZE + 1

		for i in tqdm(range(num_chunks), desc = 'Scoring complexes'):
			subset_list = file_list[(i*CHUNK_SIZE):((i+1)*CHUNK_SIZE)]
			concat_dfs = []
			concat_dfs_korp = []

			for j, filename in enumerate(subset_list):
				print(f'{tmp_dir}/{filename}')

				temp_df = converter.convert_mol2(f'{tmp_dir}/{filename}')
				temp_df['label_num'] = j
				concat_dfs.append(temp_df)

				temp_df = korp_converter.convert_ligand(f'{tmp_dir}/{filename}')
				temp_df['label_num'] = j
				concat_dfs_korp.append(temp_df)

			# Score ligands simultaneously
			lig_df = pd.concat(concat_dfs, axis = 0)
			lig_df_korp = pd.concat(concat_dfs_korp, axis = 0)

			scores_init_despot = scorer.score_complex(prot_df, lig_df)
			labels = lig_df['label_num'].values.astype(np.int64)
			scores = np.bincount(labels, weights = scores_init_despot)
			score1_list.extend(list(scores))

			scores_init = iso_scorer.score_complex(prot_df, lig_df)
			labels = lig_df['label_num'].values.astype(np.int64)
			scores = np.bincount(labels, weights = scores_init)
			score2_list.extend(list(scores))

			scores_init = screen_scorer.score_complex(prot_df_korp, lig_df_korp)
			labels = lig_df_korp['label_num'].values.astype(np.int64)
			scores = np.bincount(labels, weights = scores_init)
			score3_list.extend(list(scores))

			# Store b-factors?
			if args.bfac:
				lig_df['bfac'] = scores_init_despot
				basename_list = [x[:-5] for x in subset_list]
				write_pdbs(lig_df, basename_list, bfac_dir)

	# Store final output csv

	combo_list = [x+y for x,y in zip(score1_list, score3_list)]

	basename_list = [x[:-5] for x in file_list]
	df = pd.DataFrame(
		{'ligand': basename_list,
		 'despot_score': score1_list,
		 'despot_iso_score': score2_list,
		 'despot_screen_score': score3_list,
		 'despot_combo_score': combo_list
		})

	df.to_csv(args.outpath, index = False, float_format = '%.6f')
