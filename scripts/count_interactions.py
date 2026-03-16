from src.config import DATA_DIR
from src.core.interaction_counter import DESPOT_Counter

import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_Xtal', 'CROWN_leaky'], default = 'CROWN_train', help = 'Data source to use')
	args = parser.parse_args()

	DATABASE = args.database

	# Count interactions
	counter = DESPOT_Counter(DATABASE)
	counter.find_interactions_parallel()
