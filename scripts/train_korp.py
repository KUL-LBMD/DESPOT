from src.config import DATA_DIR
from src.core_korp.score_builder import KORP_Builder
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_xtal', 'CROWN_leaky', 'HiQBind', 'HiQBind_train'], default = 'CROWN_train', help = 'Data source to use')
	args = parser.parse_args()

	DATABASE = args.database

	# Build scores
	print('Building KORP')
	builder = KORP_Builder(DATABASE)
	rho = builder.blur_counts()
	cond_prob = builder.counts_to_prob(rho)
	ref_prob = builder.ref_probs(rho)
	builder.inverse_boltzmann(cond_prob, ref_prob)
