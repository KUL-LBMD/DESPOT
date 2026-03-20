import pandas as pd
import numpy as np
import argparse
from scipy.spatial import ConvexHull

from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter
from src.core.complex_scorer import DESPOT_Scorer
from src.utils import write_pdbs

BOX_LENGTH = 24.0
RESOLUTION = 0.5
NUM_VOXELS = int(BOX_LENGTH / RESOLUTION)

def parse_arguments():
    """
    Parse command-line arguments for voxel creation
    """

    parser = argparse.ArgumentParser(
        description = 'Use DESPOT to create a DESPOT MIF channel',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--protein', help = 'Path (relative or full) to receptor .mol2 file. Example: 1cel_receptor.mol2')
    parser.add_argument('--pocket', help = 'FPocket pqr file used for centroid estimation')
    parser.add_argument('--channel', help = 'Ligand channel used for voxel scoring')
    parser.add_argument('--output', help = 'Path to output pdb file')
    parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_Xtal', 'CROWN_leaky'], default = 'CROWN_train',
		help = 'Data source to use')
    
    return parser.parse_args()

def parse_pdb(file_path):
	"""
	Parameters
	----------

	file_path [str]: path to pdb file

	Returns
	-------

	coords [N, 3]: coordinates of atoms in file
	"""

	coords_list = []

	with open(file_path, 'r') as infile:
		for line in infile:
			if line.startswith(('ATOM', 'HETATM')):
				x = float(line[30:38])
				y = float(line[38:46])
				z = float(line[46:54])
				coords_list.append([x, y, z])

	return np.array(coords_list)

def get_hull_centroid(fpocket_path):
	"""
	Parameters
	----------

	fpocket_path [str]: Path to FPocket .pqr file

	Returns
	-------

	centroid [3,]
	"""

	# Parse PDB
	coords = parse_pdb(fpocket_path)

	if coords.shape[0] > 3:
		hull = ConvexHull(coords)
		A = hull.points[hull.simplices[:, 0], :]
		B = hull.points[hull.simplices[:, 1], :]
		C = hull.points[hull.simplices[:, 2], :]
		N = np.cross(B-A, C-A)

		# get consistent outer orientation (compensate for the lack of ordering in scipy's facetes), assume a convex hull
		M = np.mean(hull.points[hull.vertices, :], axis=0)
		sign = np.sign(np.sum((A - M) * N, axis=1, keepdims=True))
		N = N * sign

		vol = np.sum(N*A)/6
		centroid = 1/(2*vol)*(1/24 * np.sum(N*((A+B)**2 + (B+C)**2 + (C+A)**2), axis=0))

	else:
		centroid = np.mean(coords, axis = 0)

	return centroid

def build_pocket_df(centroid, lig_channel):
    """
    Creates 'fake' dataframe for scoring voxel channel

    Parameters
    ----------
    centroid [np.array(3,)]
    lig_channel [str]
	
    Returns
    -------
    lig_df [pd.DataFrame]
    """
	
    # place the center point at one of the two middle voxels (not centered, but center will not be quantized)
    start = centroid - RESOLUTION * (NUM_VOXELS // 2)
    end = centroid + RESOLUTION * (NUM_VOXELS //2 - 1)
    gridx, gridy, gridz = np.meshgrid(
        np.linspace(start[0], end[0], NUM_VOXELS),
		np.linspace(start[1], end[1], NUM_VOXELS),
		np.linspace(start[2], end[2], NUM_VOXELS),
		indexing='ij')
    coords = np.stack([gridx, gridy, gridz], axis=-1).reshape(-1, 3)
	
    # Now create fake dataframe
    n = len(coords)
    lig_df = pd.DataFrame({
        'atom_id': 'X', 'element': 'X', 'atom_name': 'X',
        'atom_type': lig_channel, 'sybyl_type': lig_channel,
        'subst_id': 'X', 'subst_name': 'LIG1', 'num_hydrogens': 0,
        'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2],
        'v1_x': 0, 'v1_y': 0, 'v1_z': 0,
        'v2_x': 0, 'v2_y': 0, 'v2_z': 0,
        'v3_x': 0, 'v3_y': 0, 'v3_z': 0,
    }, index=range(n))
	
    return lig_df

if __name__ == '__main__':
    args = parse_arguments()
    converter = MolConverter()
    scorer = DESPOT_Scorer(mode = 'gaussian', database = args.database)
	
    prot_df = converter.convert_mol2(args.protein)
    centroid = get_hull_centroid(args.pocket)
    lig_df = build_pocket_df(centroid, args.channel)
	
    scores = scorer.score_complex(prot_df, lig_df)
    lig_df['bfac'] = scores
    lig_df['label_num'] = 0
    write_pdbs(lig_df, [args.output[:-4]], './')
	
