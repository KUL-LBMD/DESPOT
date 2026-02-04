from src.config import DATA_DIR
from src.core.interaction_counter import DESPOT_Counter
from src.core.score_builder import DESPOT_Builder, DESPOT_Iso_Builder, DESPOT_DS_Builder

if __name__ == '__main__':

	# Count interactions
	counter = DESPOT_Counter()
	counter.find_interactions(n_workers = 32, max_queued = 128)

	# Build scores
	builder = DESPOT_Builder()
	builder.blur_counts()
	builder.counts_to_prob()
	builder.ref_probs()
	builder.inverse_boltzmann()

	builder = DESPOT_Iso_Builder()
	builder.blur_counts()
	builder.counts_to_prob()
	builder.ref_probs()
	builder.inverse_boltzmann()

	builder = DESPOT_DS_Builder()
	builder.blur_counts()
	builder.counts_to_prob()
	builder.cluster_probs()
	builder.ref_probs()
	builder.inverse_boltzmann()
