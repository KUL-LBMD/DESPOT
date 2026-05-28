from src.config import DATA_DIR
from src.core.score_builder import DESPOT_Builder, DESPOT_DS_Builder, DFIRE_Builder, DFIRE_Isotropic_Builder
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_Xtal', 'CROWN_leaky', 'PDBBind', 'HiQBind'], default = 'CROWN_train', help = 'Data source to use')
	args = parser.parse_args()

	DATABASE = args.database

	# Build scores
	print('Building DESPOT')

#	for smooth_mode in ['SH']:

#		print(smooth_mode)
#		builder = DESPOT_Builder(database = DATABASE, ref_mode = 'uniform', smooth_mode = smooth_mode)
#		rho = builder.blur_counts()
#		prob, cond_prob = builder.counts_to_prob(rho)
#		del rho

#		for ref_mode in ['uniform']:
#			print(f'\n=== ref_mode = {ref_mode!r} ===')
#			builder.ref_mode = ref_mode
#			print('Computing reference probability')
#			ref_prob = builder.ref_probs(prob, cond_prob)
#			print('Running inverse Boltzmann')
#			builder.inverse_boltzmann(cond_prob, ref_prob)

	print('Building DESPOT-DS')
	builder = DESPOT_DS_Builder(DATABASE)
	builder.blur_counts()
	builder.counts_to_prob()
	builder.cluster_probs()
	builder.ref_probs()
	builder.inverse_boltzmann()

	print('Building DFIRE')
#	builder = DFIRE_Builder(DATABASE)
#	rho = builder.blur_counts()
#	rho_ref = builder.ref_probs(rho)
#	builder.inverse_boltzmann(rho, rho_ref)

	print('Building DFIRE-Iso')
	builder = DFIRE_Isotropic_Builder(DATABASE)
	rho = builder.blur_counts()
	rho_ref = builder.ref_probs(rho)
	builder.inverse_boltzmann(rho, rho_ref)
