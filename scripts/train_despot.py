from src.config import DATA_DIR
from src.core.interaction_counter import DESPOT_Counter
from src.core.score_builder import DESPOT_Builder, DESPOT_Iso_Builder

#DATABASE = 'HiQBind'
DATABASE = 'CROWN_druglike_min'

if __name__ == '__main__':

	# Build scores
	print('Building DESPOT')
	builder = DESPOT_Builder(DATABASE)
	builder.blur_counts()
	builder.counts_to_prob()
	builder.ref_probs()
	builder.inverse_boltzmann()

	print('Building DESPOT-Iso')
	builder = DESPOT_Iso_Builder(DATABASE)
	builder.blur_counts()
	builder.counts_to_prob()
	builder.ref_probs()
	builder.inverse_boltzmann()
