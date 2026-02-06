"""
This script runs the 3 DESPOT variants on CASF-2016, and computes and plots the statistics
(with comparison against some competitors)
"""

from src.config import DATA_DIR
from src.casf.run_despot import run_scoring, run_docking, run_screening
from src.casf.metrics import get_scoring_values, get_ranking_values, get_docking_values, get_screening_values, get_enrichment_factors
from src.casf.plot import generate_combined_figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

NAME_LIST = ['despot', 'despot_iso', 'despot_ds', 
             'dsx', 'asp', 'autodockvina', 'drugscore_csd',
             'drugscore2018', 'glide', 'gold', 'pmf', 'chemscore', 'chemplp', 'gbvi_wsa', 'deltavina']

NAME_LIST_CLEAN = ['DESPOT', 'DESPOT-Iso', 'DESPOT-DS', 
                   'DrugScoreX', 'ASP', 'AutoDockVina', 'DrugScoreCSD', 
                   'DrugScore2018', 'GlideScore-SP', 'GoldScore', 'PMF04', 'ChemScore', 'ChemPLP', 'GBVI-WSA-dG', 'ΔVinaRF20']

if __name__ == '__main__':

	### Step 1: run DESPOT on all CASF entries and store data ###
	run_scoring()
	#run_docking()
	run_screening(n_jobs = 8)

	### Step 2: Get benchmark metrics ###
	score_df = get_scoring_values(NAME_LIST)
	rank_spearman_arr = get_ranking_values(NAME_LIST)
	dock_top_arr, dock_spearman_thresholds = get_docking_values(NAME_LIST)
	screen_df, forward_top_arr, reverse_top_arr = get_screening_values(NAME_LIST)
	ef_arr = get_enrichment_factors(screen_df, NAME_LIST)

	### Step 3: Compute and plot statistics ###
	generate_combined_figure('casf_combined.pdf', NAME_LIST, NAME_LIST_CLEAN, score_df, rank_spearman_arr, dock_top_arr, dock_spearman_thresholds,
		forward_top_arr, reverse_top_arr, ef_arr)
