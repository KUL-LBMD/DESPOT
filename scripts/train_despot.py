from src.config import DATA_DIR
from src.core.score_builder import DESPOT_Builder, DESPOT_DS_Builder, DFIRE_Builder
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_xtal', 'CROWN_leaky', 'PDBBind', 'HiQBind'], default = 'CROWN_train', help = 'Data source to use')
	args = parser.parse_args()

	DATABASE = args.database

	# Build scores
	print('Building DESPOT')
	builder = DESPOT_Builder(database = DATABASE)
	rho = builder.blur_counts()
	prob, cond_prob = builder.counts_to_prob(rho)
	ref_prob = builder.ref_probs(prob, cond_prob)
	builder.inverse_boltzmann(cond_prob, ref_prob)

	print('Building DESPOT-DS')
	builder = DESPOT_DS_Builder(DATABASE)
	builder.blur_counts()
	builder.counts_to_prob()
	builder.cluster_probs()
	builder.ref_probs()
	builder.inverse_boltzmann()

	print('Building DFIRE')
	builder = DFIRE_Builder(DATABASE)
	rho = builder.counts_to_rho()
	cond_prob = builder.get_cond_prob(rho)
	ref_prob = builder.get_ref_prob(rho)
	del rho
	builder.inverse_boltzmann(cond_prob, ref_prob)
