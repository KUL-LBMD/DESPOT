from src.config import DATA_DIR
from src.core.interaction_counter import DESPOT_Counter
from src.core.score_builder import DESPOT_Builder, DESPOT_DS_Builder
from src.core_korp.interaction_counter import KORP_Counter
from src.core_korp.score_builder import KORP_Builder

import os
os.makedirs(f'{DATA_DIR}/potentials', exist_ok = True)

if __name__ == '__main__':
	# Train models on both datasets
	for database in ['CROWN', 'CROWN_xtal', 'HiQBind', 'HiQBind_xtal']:

		### Step 1: bin interactions ###
		# DESPOT
		counter = DESPOT_Counter(database)
		counter.find_interactions_parallel()

		# KORP
		counter = KORP_Counter(database)
		counter.find_interactions_parallel()

		### Step 2: build potentials ###
		print('Building DESPOT')
		builder = DESPOT_Builder(database = database)
		rho = builder.blur_counts()
		cond_prob = builder.counts_to_prob(rho)
		ref_prob = builder.ref_probs(cond_prob)
		builder.inverse_boltzmann(cond_prob, ref_prob)

		print('Building DESPOT-DS')
		builder = DESPOT_DS_Builder(database)
		builder.blur_counts()
		builder.counts_to_prob()
		builder.cluster_probs()
		builder.ref_probs()
		builder.inverse_boltzmann()

		print('Building KORP')
		builder = KORP_Builder(database)
		rho = builder.blur_counts()
		cond_prob = builder.counts_to_prob(rho)
		ref_prob = builder.ref_probs(rho)
		builder.inverse_boltzmann(cond_prob, ref_prob)
