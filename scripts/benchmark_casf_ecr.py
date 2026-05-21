"""
This script runs the 3 DESPOT variants on CASF-2016, and computes and plots the statistics
(with comparison against some competitors).

ERC (Exponential Rank Consensus) combinations are computed for DESPOT paired
with DeltaVina, GlideScore, ChemPLP, and AutoDockVina.
"""

from src.config import DATA_DIR
from src.casf.run_despot import run_scoring, run_docking, run_screening
from src.casf.metrics import (
    get_scoring_values, get_ranking_values, get_docking_values,
    get_screening_values, get_enrichment_factors,
)

from src.casf.plot_ecr import generate_erc_figure

import numpy as np
import pandas as pd
import os
import argparse

# ============================================================================
# Name lists
# ============================================================================

NAME_LIST = [
    'despot_crown_train',
    'autodockvina',
    'drugscore2018', 'glide', 'gold', 'chemscore',
    'chemplp', 'deltavina',
]

NAME_LIST_CLEAN = [
    'DESPOT-train',
    'AutoDockVina',
    'DrugScore2018', 'GlideScore-SP', 'GoldScore',
    'ChemScore', 'ChemPLP', 'ΔVinaRF20',
]

NAME_MAP = {k: v for k, v in zip(NAME_LIST, NAME_LIST_CLEAN)}

CATEGORY_COLORS = {
    'empirical': '#E63946',
    'physical': '#457B9D',
    'kbp': '#2A9D8F',
}

SCORE_CATEGORY = {
    'DrugScoreX': 'kbp',
    'ASP': 'kbp',
    'AutoDockVina': 'empirical',
    'DrugScoreCSD': 'kbp',
    'DrugScore2018': 'kbp',
    'GlideScore-SP': 'empirical',
    'GoldScore': 'physical',
    'PMF04': 'kbp',
    'ChemScore': 'empirical',
    'ChemPLP': 'empirical',
    'GBVI-WSA-dG': 'physical',
    'ΔVinaRF20': 'empirical'
}

# ============================================================================
# ERC configuration
# ============================================================================

#ERC_PARTNERS = ['dsx', 'asp', 'drugscore_csd', 'drugscore2018', 'gold', 'pmf', 'chemscore', 'gbvi_wsa', 'deltavina', 'glide', 'chemplp', 'autodockvina']

ERC_COMBOS = [
    ['despot_crown_train', 'deltavina'],
    ['despot_crown_train', 'chemplp'],
    ['despot_crown_train', 'glide'],
    ['despot_crown_train', 'deltavina', 'chemplp'],
    ['despot_crown_train', 'deltavina', 'chemplp', 'glide'],
    ['deltavina', 'chemplp', 'glide']
]

ERC_CONFIG = {
    'partner_combos': ERC_COMBOS,
    'sigma_frac': 0.05,
    # drop 'base' — add_erc_columns derives it from partners[0]
}

ERC_NAMES_CLEAN = [
    'DESPOT-train + ΔVinaRF20 (ERC)',
    'DESPOT-train + ChemPLP (ERC)',
    'DESPOT-train + GlideScore-SP (ERC)',
    'DESPOT-train + ΔVinaRF20 + ChemPLP (ERC)',
    'DESPOT-train + ΔVinaRF20 + ChemPLP + GlideScore-SP (ERC)',
    'ΔVinaRF20 + ChemPLP + GlideScore-SP (ERC)'
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True, choices=['CROWN_train', 'CROWN_Xtal', 'CROWN_leaky'], default = 'CROWN_train', help = 'Data source to use')
    args = parser.parse_args()

    DATABASE = args.database

    ### Step 1: run DESPOT on all CASF entries and store data ###
    #run_scoring(DATABASE)
    #run_docking(DATABASE)
    #run_screening(n_jobs=8, database = DATABASE)

    ### Step 2: Get benchmark metrics (with ERC for docking & screening) ###

    dock_top_arr, dock_spearman_thresholds, dock_names_ext = get_docking_values(NAME_LIST, erc_config=ERC_CONFIG)
    screen_df, forward_top_arr, reverse_top_arr, screen_names_ext = get_screening_values(NAME_LIST, erc_config=ERC_CONFIG)
    ef_arr = get_enrichment_factors(screen_df, screen_names_ext)

    ### Step 3: Compute and plot statistics ###

    # Scoring & ranking use the original name lists (no ERC).
    # Docking, screening & enrichment use the extended lists.
    generate_erc_figure(
        'casf_erc.pdf',
        dock_name_list=dock_names_ext,
        dock_name_list_clean=NAME_LIST_CLEAN + ERC_NAMES_CLEAN,
        dock_top_arr=dock_top_arr,
        dock_spearman_thresholds=dock_spearman_thresholds,
        forward_top_arr=forward_top_arr,
        reverse_top_arr=reverse_top_arr,
        ef_arr=ef_arr,
    )
