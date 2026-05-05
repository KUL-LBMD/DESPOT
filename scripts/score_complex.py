from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter
from src.core.complex_scorer import DESPOT_Scorer, DESPOT_Isotropic_Scorer
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

	parser.add_argument('-m', '--mode', type = str, choices = ['full', 'ds'], default = 'full',
		help = 'Which DESPOT mode to use for inference')

	parser.add_argument('--bfac', action = 'store_true',
		help = 'Make subdirectory that stores separate PDB file of each ligand pose, with atom-wise score stored as b-factor')

	parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_Xtal', 'CROWN_leaky'], default = 'CROWN_train',
		help = 'Data source to use')

	return parser.parse_args()

def make_pymol_session(receptor_path, bfac_dir, session_path):
    """Write a PyMOL .pml script that builds the visualization and saves a .pse.
    Run headless:  pymol -cq <script>.pml
    Or in the GUI: @<script>.pml
    """
    ligand_files = sorted(f for f in os.listdir(bfac_dir) if f.endswith('.pdb'))
    if not ligand_files:
        print(f'No ligand PDBs found in {bfac_dir}; skipping PyMOL script.')
        return

    script_path  = session_path[:-4] + '.pml'
    receptor_abs = os.path.abspath(receptor_path)
    bfac_abs     = os.path.abspath(bfac_dir)
    session_abs  = os.path.abspath(session_path)

    lines = [f'load {receptor_abs}, receptor']
    for pdb in ligand_files:
        obj = os.path.splitext(pdb)[0]
        lines.append(f'load {os.path.join(bfac_abs, pdb)}, {obj}')
        lines.append(f'group ligands, {obj}, add')

    lines += [
        'hide everything',
        'show sticks, ligands',
        'spectrum b, blue_white_red, ligands, minimum=-20, maximum=20',
        'show lines, receptor within 6 of ligands',
        'zoom ligands, buffer=4',
        f'save {session_abs}',
    ]

    with open(script_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote PyMOL script to {script_path}')
    print(f'  Headless: pymol -cq {script_path}   ->  {session_abs}')
    print(f'  In GUI:   @{script_path}')

if __name__ == '__main__':
	args = parse_arguments()
	DATABASE = args.database

	# Initialize scorer and converter
	converter = MolConverter()

	if args.mode == 'full':
		scorer = DESPOT_Scorer(mode = 'gaussian', database = DATABASE)
	else:
		scorer = DESPOT_Isotropic_Scorer(mode = 'drugscore', database = DATABASE)

	# Initialize empty score list
	score_list = []

	# Initialize directory for bfactors?
	if args.bfac:
		bfac_dir = args.outpath[:-4]
		os.makedirs(bfac_dir, exist_ok = True)

	# Split multi-mol2 file
	with tempfile.TemporaryDirectory() as tmp_dir:
		split_mol2(args.ligand, tmp_dir)

		# Convert protein and ligands
		prot_df = converter.convert_mol2(args.protein)

		file_list = os.listdir(tmp_dir)
		num_chunks = len(file_list) // CHUNK_SIZE + 1

		for i in tqdm(range(num_chunks), desc = 'Scoring complexes'):
			subset_list = file_list[(i*CHUNK_SIZE):((i+1)*CHUNK_SIZE)]
			concat_dfs = []

			for j, filename in enumerate(subset_list):
				print(f'{tmp_dir}/{filename}')
				temp_df = converter.convert_mol2(f'{tmp_dir}/{filename}')
				temp_df['label_num'] = j
				concat_dfs.append(temp_df)

			# Score ligands simultaneously
			lig_df = pd.concat(concat_dfs, axis = 0)
			scores_init = scorer.score_complex(prot_df, lig_df)
			labels = lig_df['label_num'].values.astype(np.int64)
			scores = np.bincount(labels, weights = scores_init)
			score_list.extend(list(scores))

			# Store b-factors?
			if args.bfac:
				lig_df['bfac'] = scores_init
				basename_list = [x[:-5] for x in subset_list]
				write_pdbs(lig_df, basename_list, bfac_dir)

	# Store final output csv
	basename_list = [x[:-5] for x in file_list]
	df = pd.DataFrame(
		{'ligand': basename_list,
		 'score': score_list
		})

	df.to_csv(args.outpath, index = False, float_format = '%.6f')

	# PyMOL session for visual inspection of the per-atom DESPOT scores
	if args.bfac:
		session_path = args.outpath[:-4] + '.pse'
		make_pymol_session(args.protein, bfac_dir, session_path)
