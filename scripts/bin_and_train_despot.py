from src.config import DATA_DIR
from src.core.interaction_counter import DESPOT_Counter

DATABASE = 'HiQBind'
#DATABASE = 'CROWN'

if __name__ == '__main__':

	# Count interactions
	counter = DESPOT_Counter(DATABASE)
	counter.find_interactions_parallel()
