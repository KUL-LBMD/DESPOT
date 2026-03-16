from src.config import DATA_DIR
from src.core.interaction_counter import DESPOT_Counter
from src.core.score_builder import DESPOT_Builder, DESPOT_DS_Builder
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_Xtal', 'CROWN_leaky'], default = 'CROWN_train', help = 'Data source to use')
	args = parser.parse_args()

	DATABASE = args.database

	# Build scores
	print('Building DESPOT')
	builder = DESPOT_Builder(DATABASE)
	builder.blur_counts()
	builder.counts_to_prob()
	builder.ref_probs()
	builder.inverse_boltzmann()

	print('Building DESPOT-DS')
	builder = DESPOT_DS_Builder(DATABASE)
	builder.blur_counts()
	builder.counts_to_prob()
	builder.cluster_probs()
	builder.ref_probs()
	builder.inverse_boltzmann()
